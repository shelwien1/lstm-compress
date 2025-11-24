
#include "coro3b.hpp"
#include "ppmd1.hpp"
#include <thread>
#include <atomic>
#include <time.h>
#include <stdarg.h>
#include <string.h>

uint pmd_args1[] = { 6, 358, 1 };

uint BytesLoaded;
void* fload( const char* fname ) {
  FILE* temp = fopen(fname,"rb");
  if (temp==0) return 0;
  unsigned int len = flen(temp);
  BytesLoaded = len;
  char* buf = new char[len];
  fread( buf, len, 1, temp );
  fclose( temp );
  return buf;
}

struct idx {
  uint beg;
  uint end;
};

enum {
  inpbufsize = 1<<16,
  outbufsize = 1<<16,
  max_threads = 256,
  ringsize = 1<<16  // 64k elements for ring buffer
};

// Global arrays for thread-specific Models
Model<0>* thread_models[max_threads];

// Single global output buffer (shared by all threads, only size matters)
ALIGN(4096) byte outbuf[outbufsize];

// Compute compressed size for a single block or pair of blocks
uint compute_size_blocks( Model<0>& C, byte* outbuf, idx* I, uint N, byte* f, uint* block_indices, uint num_blocks ) {
  uint i0, csize = -1;
  C.Init(pmd_args1[0]/*ppmd_order*/,pmd_args1[1]/*ppmd_memory*/,pmd_args1[2]/*ppmd_restore*/);
  C.coro_init();
  C.addout( outbuf, outbufsize );
  for( i0=0; ; ) {
    uint l, r = C.coro_call(&C);
    if( r==1 ) {
      if( i0 < num_blocks ) {
        uint i = block_indices[i0];
        byte* x = &f[I[i].beg];
        byte* y = &f[I[i].end];
        if( i < N-1 ) y++; // include \x01
        C.addinp( x, y-x );
        i0++;
      } else {
        C.addinp( 0, 0 ); C.f_quit=1;
      }
    } else {
      l = C.getoutsize();
      csize += l;
      C.addout( outbuf, outbufsize );
      if( r!=2 ) break;
    }
  }
  C.Quit();
  return csize;
}

// Structure to hold results for ordered output
struct Result {
  qword index;  // For single blocks, this is i; for pairs, this is a linear index
  uint i, j;   // Block indices (j is -1 for single blocks)
  uint csize;
  bool ready;

  Result() : index(0), i(0), j(-1), csize(0), ready(false) {}
};

// Static ring buffer for thread-safe result passing
struct RingBuffer {
  Result buffer[ringsize];
  volatile uint head;  // Write position
  volatile uint tail;  // Read position
  volatile bool finished;

  RingBuffer() : head(0), tail(0), finished(false) {}

  // Push result to ring buffer (called by worker threads)
  void push(const Result& r) {
    uint next_head = (head + 1) & (ringsize - 1);
    // Spin-wait if buffer is full
    while (next_head == tail) {
      // Buffer full, wait for consumer
      #if defined(__x86_64__) || defined(_M_X64)
      __builtin_ia32_pause();
      #endif
    }
    buffer[head] = r;
    head = next_head;
  }

  // Pop result from ring buffer (called by main thread)
  bool pop(Result& r) {
    if (tail == head) {
      // Buffer empty
      return false;
    }
    r = buffer[tail];
    tail = (tail + 1) & (ringsize - 1);
    return true;
  }

  void set_finished() {
    finished = true;
  }

  bool is_finished() const {
    return finished;
  }
};

// Static ring buffers for each phase
static RingBuffer results_individual;
static RingBuffer results_pairs;

// Static pending results buffer (for out-of-order results)
static Result pending_buffer[4096];
static uint pending_count = 0;

// Double-buffered output (128k total = 64k * 2)
enum { output_bufsize = 64*1024 };
static char output_buffer[output_bufsize * 2];
static uint output_pos = 0;
static FILE* output_file = 0;

// Flush output buffer
void flush_output() {
  if (output_file && output_pos > 0) {
    fwrite(output_buffer, 1, output_pos, output_file);
    output_pos = 0;
  }
}

// Write formatted string to output buffer with double buffering
void buffered_printf(const char* fmt, ...) {
  char temp[256];
  va_list args;
  va_start(args, fmt);
  int len = vsprintf(temp, fmt, args);
  va_end(args);

  // Check if we need to flush first half
  if (output_pos >= output_bufsize && output_pos < output_bufsize * 2) {
    // Write first half
    if (output_file) {
      fwrite(output_buffer, 1, output_bufsize, output_file);
    }
    // Move overflow from second half to first half
    uint overflow = output_pos - output_bufsize;
    if (overflow > 0) {
      memcpy(output_buffer, output_buffer + output_bufsize, overflow);
    }
    output_pos = overflow;
  } else if (output_pos >= output_bufsize * 2) {
    // Buffer completely full, flush everything
    flush_output();
  }

  // Append to buffer
  memcpy(output_buffer + output_pos, temp, len);
  output_pos += len;
}

// Worker thread for individual blocks
void worker_individual(int thread_id, int num_threads, idx* I, uint N, byte* f,
                       RingBuffer& rq, std::atomic<uint>& progress) {
  // Thread x processes blocks k*N+x
  for( uint i = thread_id; i < N; i += num_threads ) {
    uint block_idx = i;
    uint csize = compute_size_blocks( *thread_models[thread_id], outbuf, I, N, f, &block_idx, 1 );

    Result res;
    res.index = i;
    res.i = i;
    res.j = -1;
    res.csize = csize;
    res.ready = true;
    rq.push(res);

    progress.fetch_add(1);
  }
}

// Worker thread for pair blocks
void worker_pairs(int thread_id, int num_threads, idx* I, uint N, byte* f,
                  RingBuffer& rq, std::atomic<qword>& progress) {
  // Thread x processes pairs with linear index k*num_threads+x
  qword pair_idx = 0;
  for( uint i = 0; i < N; i++ ) {
    for( uint j = i+1; j < N; j++ ) {
      if( (pair_idx % num_threads) == thread_id ) {
        uint block_indices[2] = { i, j };
        uint csize = compute_size_blocks( *thread_models[thread_id], outbuf, I, N, f, block_indices, 2 );

        Result res;
        res.index = pair_idx;
        res.i = i;
        res.j = j;
        res.csize = csize;
        res.ready = true;
        rq.push(res);

        progress.fetch_add(1);
      }
      pair_idx++;
    }
  }
}

int main( int argc, char** argv ) {

  // Parse command-line argument for thread count (default 1)
  int num_threads = 1;
  if( argc > 1 ) {
    num_threads = atoi(argv[1]);
    if( num_threads < 1 ) {
      printf( "Error: Thread count must be at least 1\n" );
      return 1;
    }
  }
  printf( "Using %d thread(s)\n", num_threads );

  printf( "Loading enwik_art_idx... " );
  idx* I = (idx*)fload( "enwik_art_idx" ); if( I==0 )
  I = (idx*)fload( "./enwik_art_idx" ); if( I==0 ) return 1;
  printf( "\b\b Done.\n" );
  uint N = BytesLoaded/sizeof(idx);

  printf( "Loading enwik_text2_drt... " );
  byte* f = (byte*)fload( "./enwik_text2_drt" ); if( f==0 ) return 1;
  printf( "\b\b Done.\n" );

  printf( "Total blocks: %u\n", N );

  // Allocate Model<0> instances for each thread in global array
  for( int t = 0; t < num_threads; t++ ) {
    thread_models[t] = new Model<0>;
    uint r = thread_models[t]->StartSubAllocator( pmd_args1[1] );
    if( r!=1 ) {
      printf( "Error: Cannot allocate ppmd memory for thread %d\n", t );
      return 1;
    }
  }

  // Part 1: Compute individual compressed sizes
  printf( "Computing individual block sizes...\n" );
  output_file = fopen( "compressed_sizes.txt", "wb" );
  if( !output_file ) {
    printf( "Error: Cannot open compressed_sizes.txt for writing\n" );
    return 1;
  }
  output_pos = 0;

  {
    results_individual.head = 0;
    results_individual.tail = 0;
    results_individual.finished = false;
    std::atomic<uint> progress(0);
    std::thread* threads[max_threads];

    // Launch worker threads
    for( int t = 0; t < num_threads; t++ ) {
      threads[t] = new std::thread(worker_individual, t, num_threads, I, N, f,
                                    std::ref(results_individual), std::ref(progress));
    }

    // Main thread: collect and print results in order
    pending_count = 0;
    qword next_index = 0;
    time_t start_time = time(0);
    time_t last_update = start_time;

    while( next_index < N ) {
      // Try to find next result in pending buffer
      bool found = false;
      for( uint k = 0; k < pending_count; k++ ) {
        if( pending_buffer[k].index == next_index ) {
          buffered_printf( "%06i - %i\n", pending_buffer[k].i, pending_buffer[k].csize );
          // Remove from pending by moving last element to this position
          pending_buffer[k] = pending_buffer[pending_count - 1];
          pending_count--;
          next_index++;
          found = true;
          break;
        }
      }

      if( !found ) {
        // Wait for more results
        Result res;
        if( results_individual.pop(res) ) {
          if( res.index == next_index ) {
            buffered_printf( "%06i - %i\n", res.i, res.csize );
            next_index++;
          } else {
            if( pending_count < 4096 ) {
              pending_buffer[pending_count++] = res;
            }
          }
        }
      }

      // Update progress every second
      time_t now = time(0);
      if( now > last_update ) {
        last_update = now;
        double percent = (100.0 * next_index) / N;
        double elapsed = difftime(now, start_time);
        double eta = (elapsed * N / next_index) - elapsed;
        if( next_index > 0 ) {
          printf( "Processed %llu / %u (%.1f%%) - ETA: %.0fs    \r", next_index, N, percent, eta );
          fflush(stdout);
        }
      }
    }

    // Wait for all threads to finish
    for( int t = 0; t < num_threads; t++ ) {
      threads[t]->join();
      delete threads[t];
    }

    flush_output();
    printf( "Processed %u / %u individual blocks - Done!                    \n", N, N );
  }
  fclose( output_file );
  output_file = 0;

#if 1
  // Part 2: Compute pair compressed sizes
  printf( "Computing pair block sizes...\n" );
  output_file = fopen( "pair_compressed_sizes.txt", "wb" );
  if( !output_file ) {
    printf( "Error: Cannot open pair_compressed_sizes.txt for writing\n" );
    return 1;
  }
  output_pos = 0;

  qword expected_pairs = ((qword)N * (N-1)) / 2;

  {
    results_pairs.head = 0;
    results_pairs.tail = 0;
    results_pairs.finished = false;
    std::atomic<qword> progress(0);
    std::thread* threads[max_threads];

    // Launch worker threads
    for( int t = 0; t < num_threads; t++ ) {
      threads[t] = new std::thread(worker_pairs, t, num_threads, I, N, f,
                                    std::ref(results_pairs), std::ref(progress));
    }

    // Main thread: collect and print results in order
    pending_count = 0;
    qword next_index = 0;
    time_t start_time = time(0);
    time_t last_update = start_time;

    while( next_index < expected_pairs ) {
      // Try to find next result in pending buffer
      bool found = false;
      for( uint k = 0; k < pending_count; k++ ) {
        if( pending_buffer[k].index == next_index ) {
          buffered_printf( "%06i_%06i - %i\n", pending_buffer[k].i, pending_buffer[k].j, pending_buffer[k].csize );
          // Remove from pending by moving last element to this position
          pending_buffer[k] = pending_buffer[pending_count - 1];
          pending_count--;
          next_index++;
          found = true;
          break;
        }
      }

      if( !found ) {
        // Wait for more results
        Result res;
        if( results_pairs.pop(res) ) {
          if( res.index == next_index ) {
            buffered_printf( "%06i_%06i - %i\n", res.i, res.j, res.csize );
            next_index++;
          } else {
            if( pending_count < 4096 ) {
              pending_buffer[pending_count++] = res;
            }
          }
        }
      }

      // Update progress every second
      time_t now = time(0);
      if( now > last_update ) {
        last_update = now;
        double percent = (100.0 * next_index) / expected_pairs;
        double elapsed = difftime(now, start_time);
        double eta = (elapsed * expected_pairs / next_index) - elapsed;
        if( next_index > 0 ) {
          printf( "Processed %llu / %llu (%.1f%%) - ETA: %.0fs    \r", next_index, expected_pairs, percent, eta );
          fflush(stdout);
        }
      }
    }

    // Wait for all threads to finish
    for( int t = 0; t < num_threads; t++ ) {
      threads[t]->join();
      delete threads[t];
    }

    flush_output();
    printf( "Processed %llu / %llu pairs - Done!                    \n", expected_pairs, expected_pairs );
  }
  fclose( output_file );
  output_file = 0;

  printf( "All processing complete!\n" );
#endif

  // Cleanup
  for( int t = 0; t < num_threads; t++ ) {
    delete thread_models[t];
  }

  return 0;
}
