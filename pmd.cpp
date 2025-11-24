#include <time.h>
#include <windows.h>

#include "coro3b.hpp"

#include "ppmd1.hpp"

//--- #include "timer.inc"

//extern "C" __declspec(dllimport) void __stdcall Sleep( uint );

//extern "C" __declspec(dllimport) uint __stdcall GetTickCount( void );

#if 1

#include <sys/types.h>

#include <sys/time.h>

#define GetTickCount GetTickCount1

int GetTickCount(void) {
  timeval t;
  gettimeofday( &t, 0 );
  return t.tv_sec*1000 + t.tv_usec/1000;
}

#else

//#ifndef _INC_WINDOWS
#ifndef _WINDOWS_
extern "C" __declspec(dllimport) unsigned __stdcall GetTickCount( void );
#endif
//#endif

#endif

uint starttick,lasttick,curtick,fintick;

#define StartTimer() (starttick=lasttick=GetTickCount())
//#define StartTimer(xxx) (starttick=GetTickCount(),lasttick=starttick+xxx)

#define CheckTimer(xxx) (curtick=GetTickCount(),(lasttick<=curtick)?lasttick=curtick+(xxx),1:0)

#define BreakTimer() (fintick=GetTickCount()-starttick)

#define PrintTimer() (printf("%i.%03is",fintick/1000,fintick%1000))



//uint pmd_args1[] = { 12, 1000, 1 };
//uint pmd_args1[] = { 7, 506, 1 };
uint pmd_args1[] = { 6, 358, 1 };

const uint N_threads = 3;

uint BytesLoaded;
void* fload( char* fname ) {
  FILE* temp = fopen(fname,"rb");
  if (temp==0) return 0;
  unsigned int len = flen(temp);
  BytesLoaded = len;
  char* buf = new char[len];
  fread( buf, len, 1, temp );
  fclose( temp );
  return buf;
}

void fsave( char* fname, void* buf, uint len ) {
  FILE* g = fopen( fname, "wb" );
  if( g ) {
    fwrite( buf, 1,len, g );
    fclose(g);
  }
}

struct idx {
  uint beg;
  uint end;
};



//--- #include "tester.inc"

struct Tester : thread<Tester> {

  ALIGN(4096) pmd_codec C;

  enum { inpbufsize = 1<<16, outbufsize = 1<<16 };
  ALIGN(4096) byte inpbuf[inpbufsize];
  ALIGN(4096) byte outbuf[outbufsize];

  idx* I;
  uint N;
  byte* f;
  
  volatile uint csize; // compressed size, thread result
  volatile uint p1,p2; // swap points
  volatile uint flag;  // processing flag

  ALIGN(4096) struct {} _align;

  void init( idx* _I, uint _N, byte* f_ ) {
    I=_I; N=_N; f=f_; p1=0; p2=0; csize=-1; flag=0;
  }

  void thread( void ) {
    flag = 1; csize = 0;
    csize = compute_size(I,N,f,p1,p2);
    flag = 0;
  }

  NOINLINE
  uint compute_size( idx* I, uint N, byte* f, uint p1, uint p2 ) {
    uint i,i0,csize=-1;
    if( !C.Init(0,pmd_args1) ) {
      C.addout( outbuf, outbufsize );
      for( i0=0;; ) {
        uint l,r = C.coro_call(&C); //-V678
        if( r==1 ) {
          if( i0<N ) {
            i=i0; if( (i==p1) || (i==p2) ) i^=p1^p2;
            byte* x = &f[I[i].beg];
            byte* y = &f[I[i].end]; 
            if( i<N-1 ) y++; // include \x01
            C.addinp( x, y-x );
            i0++;
          } else {
            C.addinp( 0, 0 ); C.f_quit=1;
          }
        } else {
          l = C.getoutsize(); //C.outptr-C.outbeg; //C.getoutlen();
          csize +=l;
          C.addout( outbuf, outbufsize );
          if( r!=2 ) break;
        }
      }
      C.Quit();
    }
    return csize;
  }

};

ALIGN(4096) Tester C[N_threads];

uint basesize = 0;

void SwapIdx( idx* I, uint i, uint j ) {
  idx t = I[i];
  I[i]=I[j];
  I[j]=t;
}

int main( int argc, char** argv ) {

  srand(time(0));

  printf( "Loading enwik_art_idx... " );
  idx* I; 

  I = (idx*)fload( "enwik_art_idx" ); if( I==0 )
  I = (idx*)fload( "./enwik_art_idx" ); if( I==0 ) return 1;
  printf( "\b\b Done.\n" );
  uint N = BytesLoaded/sizeof(idx);

  printf( "Loading enwik_text2_drt... " );
  byte* f = (byte*)fload( "./enwik_text2_drt" ); if( f==0 ) return 1;
  printf( "\b\b Done.\n" );

  uint csize,i,j,k,u;

  for( i=0; i<N_threads; i++ ) C[i].init( I,N,f );

  C[0].start(); C[0].quit();
  basesize = C[0].csize;
  printf( "basesize=%i\n", basesize );

  while(1) {

    for( k=0; k<N_threads; k++ ) if( C[k].flag==0 ) {

      // stop everything if there's a better result
      csize = C[k].csize;
      if( csize<basesize ) {
        for( u=0; u<N_threads; u++ ) if( C[u].flag ) {
          C[u].quit();
          if( C[u].csize<csize ) csize=C[u].csize,k=u;
        }
        printf( "[%i] {%i,%i} %i %i\n", k, C[k].p1,C[k].p2, basesize, csize-basesize );
        basesize = csize;
        SwapIdx(I,C[k].p1,C[k].p2);
        // save the improved permutation
        fsave( "enwik_art_idx", I, sizeof(idx)*N );
      }

      while(1) {
        i = rand()%N;
        j = rand()%N;
        if( i!=j ) break;
      }

      printf( "[%i] swap %i,%i\n", k, i,j );
  //    csize = C.compute_size(I,N,f, i,j);

      C[k].p1=i; C[k].p2=j;
      C[k].start();
    }

    if( k>=N_threads ) thread_wait();
  }

  return 0;
}

