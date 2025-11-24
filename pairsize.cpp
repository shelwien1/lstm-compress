
#include "coro3b.hpp"
#include "ppmd1.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <atomic>

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

//ALIGN(4096) pmd_codec C;
ALIGN(4096) Model<0> C;

enum { inpbufsize = 1<<16, outbufsize = 1<<16 };

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

// Thread-safe result queue
class ResultQueue {
  std::queue<Result> q;
  std::mutex mtx;
  std::condition_variable cv;
  bool finished;

public:
  ResultQueue() : finished(false) {}

  void push(const Result& r) {
    std::lock_guard<std::mutex> lock(mtx);
    q.push(r);
    cv.notify_one();
  }

  bool pop(Result& r) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this]{ return !q.empty() || finished; });
    if (q.empty()) return false;
    r = q.front();
    q.pop();
    return true;
  }

  void set_finished() {
    std::lock_guard<std::mutex> lock(mtx);
    finished = true;
    cv.notify_all();
  }
};

// Worker thread for individual blocks
void worker_individual(int thread_id, int num_threads, idx* I, uint N, byte* f,
                       Model<0>* C_local, byte* outbuf_local,
                       ResultQueue& rq, std::atomic<uint>& progress) {
  // Thread x processes blocks k*N+x
  for( uint i = thread_id; i < N; i += num_threads ) {
    uint block_idx = i;
    uint csize = compute_size_blocks( *C_local, outbuf_local, I, N, f, &block_idx, 1 );

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
                  Model<0>* C_local, byte* outbuf_local,
                  ResultQueue& rq, std::atomic<qword>& progress) {
  // Thread x processes pairs with linear index k*num_threads+x
  qword pair_idx = 0;
  for( uint i = 0; i < N; i++ ) {
    for( uint j = i+1; j < N; j++ ) {
      if( (pair_idx % num_threads) == thread_id ) {
        uint block_indices[2] = { i, j };
        uint csize = compute_size_blocks( *C_local, outbuf_local, I, N, f, block_indices, 2 );

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
  idx* I;

  uint r = C.StartSubAllocator( pmd_args1[1] );
  if( r!=1 ) {
    printf( "Error: Cannot allocate ppmd memory\n" );
    return 1;
  }
  //if( !C.StartSubAllocator( _MMAX ) ) return 1;

  I = (idx*)fload( "enwik_art_idx" ); if( I==0 )
  I = (idx*)fload( "./enwik_art_idx" ); if( I==0 ) return 1;
  printf( "\b\b Done.\n" );
  uint N = BytesLoaded/sizeof(idx);

  printf( "Loading enwik_text2_drt... " );
  byte* f = (byte*)fload( "./enwik_text2_drt" ); if( f==0 ) return 1;
  printf( "\b\b Done.\n" );

  printf( "Total blocks: %u\n", N );

  // Allocate Model<0> instances and output buffers for each thread
  Model<0>* thread_models = new Model<0>[num_threads];
  byte** thread_outbufs = new byte*[num_threads];

  for( int t = 0; t < num_threads; t++ ) {
    thread_outbufs[t] = new (std::align_val_t(4096)) byte[outbufsize];
    uint r = thread_models[t].StartSubAllocator( pmd_args1[1] );
    if( r!=1 ) {
      printf( "Error: Cannot allocate ppmd memory for thread %d\n", t );
      return 1;
    }
  }

  // Part 1: Compute individual compressed sizes
  printf( "Computing individual block sizes...\n" );
  FILE* out1 = fopen( "compressed_sizes.txt", "wb" );
  if( !out1 ) {
    printf( "Error: Cannot open compressed_sizes.txt for writing\n" );
    return 1;
  }

  {
    ResultQueue rq;
    std::atomic<uint> progress(0);
    std::vector<std::thread> threads;

    // Launch worker threads
    for( int t = 0; t < num_threads; t++ ) {
      threads.emplace_back(worker_individual, t, num_threads, I, N, f,
                           &thread_models[t], thread_outbufs[t],
                           std::ref(rq), std::ref(progress));
    }

    // Main thread: collect and print results in order
    std::vector<Result> pending_results;
    qword next_index = 0;

    while( next_index < N ) {
      // Try to find next result in pending buffer
      bool found = false;
      for( size_t k = 0; k < pending_results.size(); k++ ) {
        if( pending_results[k].index == next_index ) {
          fprintf( out1, "%06i - %i\n", pending_results[k].i, pending_results[k].csize );
          pending_results.erase(pending_results.begin() + k);
          next_index++;
          found = true;
          break;
        }
      }

      if( !found ) {
        // Wait for more results
        Result res;
        if( rq.pop(res) ) {
          if( res.index == next_index ) {
            fprintf( out1, "%06i - %i\n", res.i, res.csize );
            next_index++;
          } else {
            pending_results.push_back(res);
          }
        }
      }

      if( next_index % 100 == 0 ) {
        printf( "Processed %llu / %u individual blocks\r", next_index, N );
        fflush(stdout);
      }
    }

    // Wait for all threads to finish
    for( auto& t : threads ) {
      t.join();
    }

    printf( "Processed %u / %u individual blocks - Done!\n", N, N );
  }
  fclose( out1 );

#if 1
  // Part 2: Compute pair compressed sizes
  printf( "Computing pair block sizes...\n" );
  FILE* out2 = fopen( "pair_compressed_sizes.txt", "wb" );
  if( !out2 ) {
    printf( "Error: Cannot open pair_compressed_sizes.txt for writing\n" );
    return 1;
  }

  qword expected_pairs = ((qword)N * (N-1)) / 2;

  {
    ResultQueue rq;
    std::atomic<qword> progress(0);
    std::vector<std::thread> threads;

    // Launch worker threads
    for( int t = 0; t < num_threads; t++ ) {
      threads.emplace_back(worker_pairs, t, num_threads, I, N, f,
                           &thread_models[t], thread_outbufs[t],
                           std::ref(rq), std::ref(progress));
    }

    // Main thread: collect and print results in order
    std::vector<Result> pending_results;
    qword next_index = 0;

    while( next_index < expected_pairs ) {
      // Try to find next result in pending buffer
      bool found = false;
      for( size_t k = 0; k < pending_results.size(); k++ ) {
        if( pending_results[k].index == next_index ) {
          fprintf( out2, "%06i_%06i - %i\n", pending_results[k].i, pending_results[k].j, pending_results[k].csize );
          pending_results.erase(pending_results.begin() + k);
          next_index++;
          found = true;
          break;
        }
      }

      if( !found ) {
        // Wait for more results
        Result res;
        if( rq.pop(res) ) {
          if( res.index == next_index ) {
            fprintf( out2, "%06i_%06i - %i\n", res.i, res.j, res.csize );
            next_index++;
          } else {
            pending_results.push_back(res);
          }
        }
      }

      if( next_index % 1000 == 0 ) {
        printf( "Processed %llu / %llu pairs\r", next_index, expected_pairs );
        fflush(stdout);
      }
    }

    // Wait for all threads to finish
    for( auto& t : threads ) {
      t.join();
    }

    printf( "Processed %llu / %llu pairs - Done!\n", expected_pairs, expected_pairs );
  }
  fclose( out2 );

  printf( "All processing complete!\n" );
#endif

  // Cleanup
  for( int t = 0; t < num_threads; t++ ) {
    delete[] thread_outbufs[t];
  }
  delete[] thread_outbufs;
  delete[] thread_models;

  return 0;
}
