#include <time.h>
#include <windows.h>

#include "coro3b.hpp"
#include "ppmd1.hpp"

//#include "timer.inc"


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
ALIGN(4096) byte inpbuf[inpbufsize];
ALIGN(4096) byte outbuf[outbufsize];

// Compute compressed size for a single block or pair of blocks
uint compute_size_blocks( idx* I, uint N, byte* f, uint* block_indices, uint num_blocks ) {
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

int main( int argc, char** argv ) {

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

  // Part 1: Compute individual compressed sizes
  printf( "Computing individual block sizes...\n" );
  FILE* out1 = fopen( "compressed_sizes.txt", "wb" );
  if( !out1 ) {
    printf( "Error: Cannot open compressed_sizes.txt for writing\n" );
    return 1;
  }

  for( uint i = 0; i < N; i++ ) {
    uint block_idx = i;
    uint csize = compute_size_blocks( I, N, f, &block_idx, 1 );
    fprintf( out1, "%06i - %i\n", i, csize );
    if( i % 100 == 0 ) {
      printf( "Processed %u / %u individual blocks\r", i, N );
      fflush(stdout);
    }
  }
  fclose( out1 );
  printf( "Processed %u / %u individual blocks - Done!\n", N, N );

#if 0
  // Part 2: Compute pair compressed sizes
  printf( "Computing pair block sizes...\n" );
  FILE* out2 = fopen( "pair_compressed_sizes.txt", "wb" );
  if( !out2 ) {
    printf( "Error: Cannot open pair_compressed_sizes.txt for writing\n" );
    return 1;
  }

  uint total_pairs = 0;
  uint expected_pairs = (N * (N-1)) / 2;

  for( uint i = 0; i < N; i++ ) {
    for( uint j = i+1; j < N; j++ ) {
      uint block_indices[2] = { i, j };
      uint csize = compute_size_blocks( I, N, f, block_indices, 2 );
      fprintf( out2, "%06i_%06i - %i\n", i, j, csize );
      total_pairs++;
      if( total_pairs % 1000 == 0 ) {
        printf( "Processed %u / %u pairs\r", total_pairs, expected_pairs );
        fflush(stdout);
      }
    }
  }
  fclose( out2 );
  printf( "Processed %u / %u pairs - Done!\n", total_pairs, expected_pairs );

  printf( "All processing complete!\n" );
#endif

  return 0;
}
