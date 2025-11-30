
#include "coro3b.hpp"
#include "ppmd1.hpp"
#include <thread>
#include <time.h>
#include <stdarg.h>
#include <string.h>

#include "bufprint.inc"

// Forward declaration
void SLEEP_MS(int ms);

uint pmd_args1[] = { 16, 358, 1 };

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
  max_threads = 64,
  ringsize = 1<<20  // 64k elements for ring buffer
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
  uint i, j;   // Block indices (j is -1 for single blocks)
  uint csize;

  Result() : i(0), j(-1), csize(0) {}
};

// Ring buffer for thread-safe result passing (per-thread)
struct RingBuffer {
  Result buffer[ringsize];
  volatile qword head;  // Write position (only incremented, wrap on access)
  volatile qword tail;  // Read position (only incremented, wrap on access)

  RingBuffer() : head(0), tail(0) {}

  // Push result to ring buffer (called by worker thread)
  void push(const Result& r) {
    // Reserve a slot
    qword my_head = head++;

    // Spin-wait if buffer is full
    while (my_head - tail >= ringsize) {
      // Buffer full, wait for consumer
      SLEEP_MS(1);
    }

    // Write to reserved slot (apply mask only when accessing array)
    buffer[my_head & (ringsize - 1)] = r;
  }

  // Pop result from ring buffer (called by main thread)
  bool pop(Result& r) {
    qword current_tail = tail;
    qword current_head = head;

    if (current_tail >= current_head) {
      // Buffer empty
      return false;
    }

    // Read from buffer (apply mask only when accessing array)
    r = buffer[current_tail & (ringsize - 1)];

    // Advance tail (only increment, no masking)
    tail = current_tail + 1;
    return true;
  }
};

// Per-thread ring buffers - each thread writes to its own buffer
static RingBuffer thread_buffers[max_threads];

// Worker thread for individual blocks
void worker_individual(int thread_id, int num_threads, idx* I, uint N, byte* f) {
  // Thread x processes blocks k*num_threads+x
  RingBuffer& rb = thread_buffers[thread_id];
  for( uint i = thread_id; i < N; i += num_threads ) {
    uint block_idx = i;
    uint csize = compute_size_blocks( *thread_models[thread_id], outbuf, I, N, f, &block_idx, 1 );

    Result res;
    res.i = i;
    res.j = -1;
    res.csize = csize;
    rb.push(res);
  }
}

// Worker thread for pair blocks
void worker_pairs(int thread_id, int num_threads, idx* I, uint N, byte* f) {
  // Thread x processes pairs with linear index k*num_threads+x
  RingBuffer& rb = thread_buffers[thread_id];
  qword pair_idx = 0;
  for( uint i = 0; i < N; i++ ) {
    for( uint j = 0; j < N; j++ ) {
      if( (pair_idx % num_threads) == thread_id ) {
        uint block_indices[2] = { i, j };
        uint csize = 0x7FFFFFFF;

        if( i!=j ) 
        csize = compute_size_blocks( *thread_models[thread_id], outbuf, I, N, f, block_indices, 2 );

        Result res;
        res.i = i;
        res.j = j;
        res.csize = csize;
        rb.push(res);
      }
      pair_idx++;
    }
  }

  Result res;
  res.i = 0xFFFFFFFF;
  res.j = 0xFFFFFFFF;
  res.csize = 0xFFFFFFFF;
  rb.push(res);

  fprintf(stderr,"!!done %i!!\n", thread_id ); fflush(stderr);
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
  uint N = BytesLoaded/sizeof(idx); // N items to sort

  printf( "Loading enwik_text2_drt... " );
  byte* f = (byte*)fload( "./enwik_text2_drt" ); if( f==0 ) return 1;
  printf( "\b\b Done.\n" );

  printf( "Total blocks: %u\n", N );

  uint* psize = new uint[N]; if( psize==0 ) return 1;

  // Allocate Model<0> instances for each thread in global array
  for( int t = 0; t < num_threads; t++ ) {
    thread_models[t] = new Model<0>;
    uint r = thread_models[t]->StartSubAllocator( pmd_args1[1] );
    if( r!=1 ) {
      printf( "Error: Cannot allocate ppmd memory for thread %d\n", t );
      return 1;
    }
    // Initialize ring buffer
    thread_buffers[t].head = 0;
    thread_buffers[t].tail = 0;
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
    std::thread* threads[max_threads];

    // Launch worker threads
    for( int t = 0; t < num_threads; t++ ) {
      threads[t] = new std::thread(worker_individual, t, num_threads, I, N, f);
    }

    // Main thread: collect and print results in order
    time_t start_time = time(0);
    time_t last_update = start_time;

    for( uint idx = 0; idx < N; idx++ ) {
      // Determine which thread should produce this result
      int thread_id = idx % num_threads;
      RingBuffer& rb = thread_buffers[thread_id];

      // Wait for result to be available
      Result res;
      while( !rb.pop(res) ) {
        SLEEP_MS(1);
      }

      // Print result
      BPRINTF( "%06i - %i\n", res.i, res.csize );
      psize[res.i] = res.csize;

      // Update progress every second
      time_t now = time(0);
      if( now > last_update ) {
        last_update = now;
        double percent = (100.0 * (idx + 1)) / N;
        double elapsed = difftime(now, start_time);
        double eta = (elapsed * N / (idx + 1)) - elapsed;
        printf( "Processed %u / %u (%.1f%%) - ETA: %.0fs    \r", idx + 1, N, percent, eta );
        fflush(stdout);
      }
    }

    // Wait for all threads to finish
    for( int t = 0; t < num_threads; t++ ) {
      threads[t]->join();
      delete threads[t];
    }

    flush_output();
    printf( "Processed %u / %u individual blocks - Done!                    \n", N, N ); fflush(stdout);
  }

  fclose( output_file );
  output_file = 0;

#if 1
  // Part 2: Compute pair compressed sizes
  printf( "Computing pair block sizes...\n" ); fflush(stdout);
  output_file = fopen( "pair_compressed_sizes.txt", "wb" );
  if( !output_file ) {
    printf( "Error: Cannot open pair_compressed_sizes.txt for writing\n" );
    return 1;
  }

  output_pos = sprintf(output_buffer,
"NAME : CompressedSizes\n"
"TYPE : ATSP\n"
"COMMENT : Pairwise compression sizes from compressed_sizes.txt and pair_compressed_sizes.txt\n"
"DIMENSION : %i\n"
"EDGE_WEIGHT_TYPE : EXPLICIT\n"
"EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
"EDGE_WEIGHT_SECTION\n", N );

  //qword expected_pairs = ((qword)N * (N-1)) / 2;
  qword expected_pairs = qword(N) * N;

  {
    std::thread* threads[max_threads];
    uint pquit[max_threads];
    uint n_pquit = 0;

    // Reset ring buffers
    for( int t = 0; t < num_threads; t++ ) {
      pquit[t] = 0;
      thread_buffers[t].head = 0;
      thread_buffers[t].tail = 0;
    }

    // Launch worker threads
    for( int t = 0; t < num_threads; t++ ) {
      threads[t] = new std::thread(worker_pairs, t, num_threads, I, N, f);
    }

    // Main thread: collect and print results in order
    time_t start_time = time(0);
    time_t last_update = start_time;
    Result res; 
    int thread_id;

    for( qword idx = 0; n_pquit<num_threads; idx++ ) {
      // Determine which thread should produce this result
      thread_id = idx % num_threads;
      RingBuffer& rb = thread_buffers[thread_id];

      // Wait for result to be available
      res.i=uint(-1); res.j=uint(-1); res.csize=uint(-1);

      if( pquit[thread_id]==0 ) {
        while( !rb.pop(res) ) {
          SLEEP_MS(1);
        }
      } else continue;

      if( int(res.i)==-1 ) { pquit[thread_id]=1; ++n_pquit; continue; }

      // Print result
      res.csize -= psize[res.i] + psize[res.j];

      if( res.j>=N-1 ) {
        BPRINTF( "%i\n", int(res.csize) );
      } else {
        BPRINTF( "%i ", int(res.csize) );
      }

      // Update progress every second
      time_t now = time(0);
      if( now > last_update ) {
        last_update = now;
        double percent = (100.0 * (idx + 1)) / expected_pairs;
        double elapsed = difftime(now, start_time);
        double eta = (elapsed * expected_pairs / (idx + 1)) - elapsed;
        printf( "Processed %llu / %llu (%.1f%%) - ETA: %.0fs    \r", idx + 1, expected_pairs, percent, eta );
        fflush(stdout);
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

  fprintf(output_file, "EOF\n"); // .tsp

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

// Platform-specific sleep implementation
#ifdef _WIN32
#include <windows.h>
void SLEEP_MS(int ms) {
  Sleep(ms);
}
#else
#include <unistd.h>
void SLEEP_MS(int ms) {
  usleep(ms * 1000);
}
#endif
