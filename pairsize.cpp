
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

// Helper function to collect all leaf indices from a tree node
void collect_tree_indices( int* tree, uint node_idx, uint N, uint* indices, uint& count ) {
  if( node_idx < N ) {
    // Leaf node - add to indices
    indices[count++] = node_idx;
  } else {
    // Internal node - recurse on children
    int left = tree[node_idx*2+0];
    int right = tree[node_idx*2+1];

    if( left >= 0 ) {
      collect_tree_indices( tree, left, N, indices, count );
    }
    if( right >= 0 ) {
      collect_tree_indices( tree, right, N, indices, count );
    }
  }
}

// Worker thread for merged node pairs
// Computes pair-gain for pairs involving new_idx
void worker_merged(int thread_id, int num_threads, idx* I, uint N, byte* f,
                   int* tree, int* imap, uint imapsize, uint new_idx,
                   int* pairgain, uint* psize, uint* thread_indices) {

  RingBuffer& rb = thread_buffers[thread_id];
  uint* my_indices = &thread_indices[thread_id * N];

  // Process pairs where one element is new_idx
  uint pair_count = 0;
  for( uint i = 0; i < imapsize; i++ ) {
    for( uint j = 0; j < imapsize; j++ ) {
      // Skip if both are same or neither is new_idx
      if( i == j ) continue;

      uint ii = imap[i];
      uint jj = imap[j];

      // Only process pairs involving new_idx
      if( ii != new_idx && jj != new_idx ) continue;

      if( (pair_count % num_threads) == thread_id ) {
        // Collect all indices from tree node ii
        uint count_i = 0;
        collect_tree_indices( tree, ii, N, my_indices, count_i );

        // Collect all indices from tree node jj
        uint count_j = 0;
        collect_tree_indices( tree, jj, N, &my_indices[count_i], count_j );

        // Build block_indices array
        uint total_blocks = count_i + count_j;

        // Compute compressed size
        uint csize = compute_size_blocks( *thread_models[thread_id], outbuf, I, N, f, my_indices, total_blocks );

        // Compute pair-gain
        int gain = int(csize) - int(psize[ii]) - int(psize[jj]);

        Result res;
        res.i = i;  // Index in imap
        res.j = j;
        res.csize = gain;
        rb.push(res);
      }
      pair_count++;
    }
  }

  // Signal completion
  Result res;
  res.i = 0xFFFFFFFF;
  res.j = 0xFFFFFFFF;
  res.csize = 0xFFFFFFFF;
  rb.push(res);
}

// Find the minimum pairgain (most negative value) in the pairgain array
// Returns the x,y indices via reference parameters
void find_min_pairgain( int* pairgain, uint N, uint& min_x, uint& min_y, int& min_gain ) {
  min_gain = 0x7FFFFFFF; // Start with max int
  min_x = 0;
  min_y = 0;

  for( uint i = 0; i < N; i++ ) {
    for( uint j = 0; j < N; j++ ) {
      if( i != j ) {
        int gain = pairgain[i*N+j];
        if( gain < min_gain && gain != 0 ) {  // Skip zero entries (cleared or unused)
          min_gain = gain;
          min_x = i;
          min_y = j;
        }
      }
    }
  }
}

// Traverse tree depth-first and collect item indices
void traverse_tree( int* tree, uint node_idx, uint N, uint* output, uint& output_idx ) {
  if( node_idx < N ) {
    // Leaf node - store item index
    output[output_idx++] = node_idx;
  } else {
    // Internal node - recurse on children
    int left = tree[node_idx*2+0];
    int right = tree[node_idx*2+1];

    if( left >= 0 ) {
      traverse_tree( tree, left, N, output, output_idx );
    }
    if( right >= 0 ) {
      traverse_tree( tree, right, N, output, output_idx );
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
  uint N = BytesLoaded/sizeof(idx); // N items to sort

  printf( "Loading enwik_text2_drt... " );
  byte* f = (byte*)fload( "./enwik_text2_drt" ); if( f==0 ) return 1;
  printf( "\b\b Done.\n" );

  printf( "Total blocks: %u\n", N );

  uint* psize = new uint[2*N]; if( psize==0 ) return 1;

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

  // Allocate pairgain array for storing pair compression gains
  int* pairgain = new int[N*N];
  if( pairgain==0 ) {
    printf( "Error: Cannot allocate pairgain array\n" );
    return 1;
  }
  // Initialize with zeros
  for( uint i = 0; i < N*N; i++ ) {
    pairgain[i] = 0;
  }

  // Allocate tree array for merging pairs
  int* tree = new int[4*N];
  if( tree==0 ) {
    printf( "Error: Cannot allocate tree array\n" );
    return 1;
  }

  // Allocate imap array for index mapping
  int* imap = new int[2*N];
  if( imap==0 ) {
    printf( "Error: Cannot allocate imap array\n" );
    return 1;
  }

  // Initialize tree and imap
  uint tree_top, imapsize;
  for( uint i = 0; i < N; i++ ) {
    tree[i*2+0] = i;
    tree[i*2+1] = -1;
    imap[i] = i;
  }
  tree_top = N;
  imapsize = N;

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

      // Store in pairgain array
      pairgain[res.i*N+res.j] = res.csize;

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

  printf( "Pair processing complete. Starting optimization...\n" );

  // Allocate thread_indices array for worker threads
  uint* thread_indices = new uint[num_threads * N];
  if( thread_indices == 0 ) {
    printf( "Error: Cannot allocate thread_indices array\n" );
    return 1;
  }

  // Main optimization loop - merge pairs until only one item remains
  while( imapsize > 1 ) {
    // Find minimum pairgain pair
    uint min_x, min_y;
    int min_gain;
    find_min_pairgain( pairgain, N, min_x, min_y, min_gain );

    // x and y are already actual tree indices from pairgain array
    uint x = min_x;
    uint y = min_y;

    printf( "Merging pair (%u,%u) with gain %d, imapsize=%u\n", x, y, min_gain, imapsize );

    // Add to tree
    tree[tree_top*2+0] = x;
    tree[tree_top*2+1] = y;
    uint new_idx = tree_top++;

    // Set psize for merged node
    psize[new_idx] = min_gain + psize[x] + psize[y];

    // Update imap: replace x with new_idx, remove y
    uint j = 0;
    for( uint i = 0; i < imapsize; i++ ) {
      uint c = imap[i];
      if( c == x ) c = new_idx;
      if( c != y ) imap[j++] = c;
    }
    imapsize = j;
    imap[new_idx] = x;

    // Clear old pairgain entries for x and y
    for( uint i = 0; i < N; i++ ) {
      pairgain[x*N+i] = 0;
      pairgain[i*N+x] = 0;
      pairgain[y*N+i] = 0;
      pairgain[i*N+y] = 0;
    }

    // If only one item left, we're done
    if( imapsize <= 1 ) break;

    // Compute new pairgain values for pairs involving new_idx
    printf( "Computing pairgain for merged node %u...\n", new_idx );

    // Reset ring buffers
    for( int t = 0; t < num_threads; t++ ) {
      thread_buffers[t].head = 0;
      thread_buffers[t].tail = 0;
    }

    // Launch worker threads
    std::thread* threads[max_threads];
    uint pquit[max_threads];
    uint n_pquit = 0;
    for( int t = 0; t < num_threads; t++ ) {
      pquit[t] = 0;
      threads[t] = new std::thread(worker_merged, t, num_threads, I, N, f,
                                    tree, imap, imapsize, new_idx,
                                    pairgain, psize, thread_indices);
    }

    // Collect results
    qword idx = 0;
    while( n_pquit < num_threads ) {
      int thread_id = idx % num_threads;
      RingBuffer& rb = thread_buffers[thread_id];

      Result res;
      res.i = uint(-1);
      res.j = uint(-1);
      res.csize = uint(-1);

      if( pquit[thread_id] == 0 ) {
        while( !rb.pop(res) ) {
          SLEEP_MS(1);
        }
      } else {
        idx++;
        continue;
      }

      if( int(res.i) == -1 ) {
        pquit[thread_id] = 1;
        ++n_pquit;
        idx++;
        continue;
      }

      // Store result in pairgain
      uint ii = imap[res.i];
      uint jj = imap[res.j];
      pairgain[ii*N+jj] = res.csize;

      idx++;
    }

    // Wait for threads to finish
    for( int t = 0; t < num_threads; t++ ) {
      threads[t]->join();
      delete threads[t];
    }

    printf( "Merged node %u complete.\n", new_idx );
  }

  printf( "Optimization complete! Final tree root: %u\n", imap[0] );

  // Traverse tree and output final order
  printf( "Writing order.txt...\n" );
  FILE* order_file = fopen( "order.txt", "w" );
  if( !order_file ) {
    printf( "Error: Cannot open order.txt for writing\n" );
    return 1;
  }

  uint* order = new uint[N];
  uint order_idx = 0;
  traverse_tree( tree, imap[0], N, order, order_idx );

  for( uint i = 0; i < order_idx; i++ ) {
    fprintf( order_file, "%u\n", order[i] );
  }

  fclose( order_file );
  printf( "order.txt written with %u items.\n", order_idx );

  delete[] order;
  delete[] thread_indices;
  delete[] pairgain;
  delete[] tree;
  delete[] imap;

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
