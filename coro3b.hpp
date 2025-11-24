
#define INC_FLEN
//--- #include "common.inc"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <memory.h>
#undef EOF

#pragma pack(1)

typedef unsigned short word;
typedef unsigned int   uint;
typedef unsigned char  byte;
typedef unsigned long long qword;
typedef signed long long sqword;


template <class T> void bzero( T &_p ) { int i; byte* p = (byte*)&_p; for( i=0; i<sizeof(_p); i++ ) p[i]=0; }
//template <class T, int N> void bzero( T (&p)[N] ) { int i; for( i=0; i<N; i++ ) p[i]=0; }

template <class T, int N> void bzero( T (&p)[N] ) { 
  char* q = (char*)&p[0]; uint n=N*sizeof(T);
  uint i; for( i=0; i<n; i++ ) q[i]=0; 
}

template <class T> void bzero( T* p, int N ) { int i; for( i=0; i<N; i++ ) p[i]=0; }

template <class T, int N, int M> void bzero( T (&p)[N][M] ) { int i; for( i=0; i<N*M; i++ ) p[0][i]=0; }

template <typename T1, typename T2> T1 Min( T1 t1, T2 t2 ) { return t1<t2?t1:t2; }
template <typename T1, typename T2> T1 Max( T1 t1, T2 t2 ) { return t1>t2?t1:t2; }

#define macro_Min( x, y ) (((x)<(y)) ? (x) : (y))
#define macro_Max( x, y ) (((x)>(y)) ? (x) : (y))

template <class T,int N> constexpr int DIM( T (&wr)[N] ) { return sizeof(wr)/sizeof(wr[0]); };
#define AlignUp(x,r) ((x)+((r)-1))/(r)*(r)
template<byte a,byte b,byte c,byte d> struct wc { 
  static const unsigned int n=(d<<24)+(c<<16)+(b<<8)+a; 
  static const unsigned int x=(a<<24)+(b<<16)+(c<<8)+d;
};

#ifdef __GNUC__
 #define INLINE   __attribute__((always_inline)) inline
 #define NOINLINE __attribute__((noinline))
 #define ALIGN(n) __attribute__((aligned(n)))
// #define __assume_aligned(x,y) x=(byte*)__builtin_assume_aligned((void*)x,y)
 #define __assume_aligned(x,y) (x=decltype(x)(__builtin_assume_aligned((void*)x,y)))
 #define restrict __restrict
#else
 #define INLINE   __forceinline
 #define NOINLINE __declspec(noinline)
 #define ALIGN(n) __declspec(align(n))
#endif

#define if_e0(x) if(__builtin_expect((x),0))
#define if_e1(x) if(__builtin_expect((x),1))
#define for_e0(x,y,z) for( (x); __builtin_expect((y),0); (z) )
#define for_e1(x,y,z) for( (x); __builtin_expect((y),1); (z) )
#define while_e0(x) while(__builtin_expect((x),0))
#define while_e1(x) while(__builtin_expect((x),1))

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
 #define __builtin_expect(x,y) (x)
// #define __assume_aligned(x,y) 
 #define __assume_aligned(x,y) __assume( (((byte*)x)-((byte*)0))%(y)==0 )
 #define restrict __restrict
 //#include "intrin.h"
 #ifndef COMMON_SKIP_BSF
 extern "C" {
 byte __cdecl _BitScanForward( uint* _Index, uint _Mask);
 byte __cdecl _BitScanForward64( uint* _Index, qword _Mask );
 }
 #endif
#endif

#if !defined(_MSC_VER) && !defined(__INTEL_COMPILER)
 #define __assume(x) (x)
#endif

#ifdef INC_FLEN
static uint flen( FILE* f ) {
  fseek( f, 0, SEEK_END );
  uint len = ftell(f);
  fseek( f, 0, SEEK_SET );
  return len;
}
#endif

#ifdef INC_LOG2I
static uint log2i( uint x ) {
#if ((defined __GNUC__) || (defined __INTEL_COMPILER))
 #ifdef __GNUC__
  return 31-__builtin_clz(x);
 #else
  return _bit_scan_reverse(x);
 #endif
#else
  uint i; 
  for( i=0; i<32; i++,x>>=1 ) if( x==0 ) break;
  return i-1;
#endif
}
#endif

#if defined(__x86_64) || defined(_M_X64)
 #define X64
 #define X64flag 1
#else
 #undef X64
 #define X64flag 0
#endif

#if 0
unsigned totalmem = 0;

void* __cdecl operator new( size_t N ) {
  void* p = malloc(N); // (void*)VirtualAlloc( 0, N, 0x1000/*MEM_COMMIT*/, 0x04/*PAGE_READWRITE*/ );
  totalmem += N;
  if( (p==0) || (N>1000000) ) printf( "alloc %i @ %X (%u)\n", N, p, totalmem );
  return p;
}

void __cdecl operator delete( void* p ) {
//  printf( "delete %X\n", p );
//  VirtualFree(p);
  free(p);
  return;
}
#endif

//--- #include "thread.inc"

template <class child> 
struct thread {

  HANDLE th;

  uint start( void ) {
    th = CreateThread( 0, 0, &thread_w, this, 0, 0 );
    return th!=0;
  }

  void quit( void ) {
    WaitForSingleObject( th, INFINITE );
    CloseHandle( th );
  }

  // wrappers to redirect static calls into object method calls
  static DWORD WINAPI thread_w( LPVOID lpParameter ) { ((child*)lpParameter)->thread(); return 0; }
};

void thread_wait( void ) { 
  Sleep(10); 
} 

//--- #include "coro3b.inc"

//#define CORO_NOASM 1

#if defined(_MSC_VER) || defined(__clang__)
#pragma runtime_checks( "scu", off )
#pragma check_stack(off)
#pragma strict_gs_check(off)
#endif

#if ((defined __GNUC__) || (defined __INTEL_COMPILER) || (defined __clang__)) && (!defined CORO_NOASM)
  #ifdef X64
//---     #include "coro3_setjmp_x64.h"

struct my_jmpbuf {
  qword rip,rsp;
};

#define ASM __asm__ volatile

INLINE
static int my_setjmp( my_jmpbuf* regs ) {
  qword r;

  ASM ("\
   movq %%rsp,8(%1); \
   call 1f; \
1: popq 0(%1); \
  " : "=a"(r) : "b"(regs),"a"(0) : "%rcx","%rdx","%rsi","%rdi","%rbp","%r8","%r9","%r10","%r11","%r12","%r13","%r14","%r15",
"ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15"
  );

  return r;
}

INLINE
static void my_jmp( my_jmpbuf* regs, int ) {
  ASM ("\
  xchg %0,%%rsp; \
  jmp *%1; \
  " :  : "d"(regs->rsp),"b"( ((byte*)regs->rip)+2 ),"a"(1) : 
  );
}

typedef my_jmpbuf m_jmp_buf[1];
#define jmp_buf m_jmp_buf
#define longjmp my_jmp
#define setjmp  my_setjmp

//    #include "coro3_setjmp_x64b.h"
  #else
//---     #include "coro3_setjmp_x32.h"

struct my_jmpbuf {
  uint eip,esp;
};

#define ASM __asm__ volatile

INLINE
static int my_setjmp( my_jmpbuf* regs ) {
  int r;

  ASM ("\
   movl %%esp,4(%1); \
   call 1f; \
1: popl 0(%1); \
  " : "=a"(r) : "b"(regs),"a"(0) : "%ecx","%edx","%esi","%edi","%ebp"
  );

  return r;
}

INLINE
static void my_jmp( my_jmpbuf* regs, int ) {
  ASM ("\
  xchg %0,%%esp; \
  jmp *%1; \
  " :  : "d"(regs->esp),"b"( ((byte*)regs->eip)+2 ),"a"(1) : 
  );
}

typedef my_jmpbuf m_jmp_buf[1];
#define jmp_buf m_jmp_buf
#define longjmp my_jmp
#define setjmp  my_setjmp

//    #include "coro3_setjmp_x32b.h"
  #endif
#else 
  #ifndef CORO_NOASM
  #define CORO_NOASM 1
  #endif
  #include <setjmp.h>
#endif

struct Coroutine;
static void yield( void* p, int value );

//--- #include "coro3_pin.inc"

//--- #include "coro3_pin_0.inc"

struct coro3_pin_0 {
  byte* ptr;
  byte* beg;
  byte* end;
  uint  f_EOF;
  word  base_offs;
  word  r_code;

  uint getinplen()  { return end-ptr; } //-V110
  uint getinpleft() { return end-ptr; } //-V110
  uint getinpsize() { return ptr-beg; } //-V110

  uint getoutlen()  { return end-ptr; } //-V110
  uint getoutleft() { return end-ptr; } //-V110
  uint getoutsize() { return ptr-beg; } //-V110

  void addinp( byte* inp,uint inplen ) { addbuf(inp,inplen); }
  void addout( byte* out,uint outlen ) { addbuf(out,outlen); }

  void addbuf( byte* buf,uint len ) {
    beg = ptr = buf;
    end = &buf[len];
  }

};

struct coro3_pin: coro3_pin_0 {
  typedef Coroutine wrap;

  void pin_init( wrap* that, uint _r_code ) {
    ptr=beg=end=0; f_EOF=0;
    base_offs = ((char*)this) - ((char*)that);
    r_code = _r_code;
  }

  void yield_r( void ) {
    wrap& W = *(wrap*)(((char*)this) - base_offs );
    yield( (void*)&W, r_code );
  }

  inline uint f_quit( void );

#define coro3_pin_DEFINE_f_quit                     \
  inline uint coro3_pin::f_quit( void ) {           \
    wrap& W = *(wrap*)(((char*)this) - base_offs ); \
    return f_EOF | W.f_quit;                        \
  }                                                 

  void chkinp( void ) { if_e0( ptr>=end ) yield_r(); }

  void chkout( uint d=0 ) { if_e0( ptr>=end-d ) yield_r(); }

  byte get0( void ) { return *ptr++; }
  void put0( uint c ) { *ptr++ = c; }

  uint get( void ) { 
    m0:
    if_e0( ptr>=end ) {
      if_e0( f_quit() ) return uint(-1);
      yield_r();
      goto m0;
    }
    return *ptr++; 
  }

  void put( uint c ) { 
    *ptr++ = c; chkout(); 
  }

};

static uint coro_call0( Coroutine* that );
static void call_do_process0( Coroutine* that );

enum CAPI { 
  INIT=0,  // have to run Init()
  QUIT=1,  // all done, have to run Quit()
  FLUSH=2,
  PROG=3,  // processing possible
  DONE=4,  // Quit() complete
  ALLOC=8880,
  FREE=8881,
};

struct Coroutine {

  union {
    struct {
    byte* inpptr;
    byte* inpbeg;
    byte* inpend;
    uint  inp_f_EOF;
    uint  inp_pad_x64_;

    byte* outptr;
    byte* outbeg;
    byte* outend;
    uint  out_f_EOF;
    uint  out_pad_x64_;
    };
    coro3_pin pin[4];
  };

  volatile uint  state;
  volatile uint  f_quit;
           uint  f_init; // state of main routine

  ALIGN(32) jmp_buf PointA;
  ALIGN(32) jmp_buf PointB;

  ALIGN(8)
  volatile char* stkptrH;
  volatile char* stkptrL; // remembered _sp value for this instance

  typedef void (Coroutine::*t_do_process)( void );
  t_do_process p_do_process;

  enum{ STKPAD=4096*4+24 }; // coroutine stack size
  enum{ STKPAD0=1<<16 }; // stack padding from frontend to coroutine

  ALIGN(8) byte stk[STKPAD];

  void coro_init( void ) {
    f_init = CAPI::INIT;
    f_quit = 0;
    state = 0; 
    for( uint i=0; i<DIM(pin); i++ ) pin[i].pin_init( this, 1+i );
  }

  template <typename T> 
  INLINE
  uint coro_call( T* that ) {
    p_do_process = (t_do_process)&T::do_process;
    return coro_call0(that);
  }

//---------------------

  void chkinp( void ) { pin[0].chkinp(); }
  void chkout( uint d=0 ) { pin[1].chkout(d); }
  uint get( void ) { return pin[0].get(); }
  void put( uint c ) { pin[1].put(c); }

  byte get0( void ) { return pin[0].get0(); }
  void put0( uint c ) { pin[1].put0(c); }

  uint getinplen() { return pin[0].getinplen(); } //-V110
  uint getoutlen() { return pin[1].getoutlen(); } //-V110
  uint getinpleft() { return pin[0].getinpleft(); } //-V524 //-V110
  uint getoutleft() { return pin[1].getoutleft(); } //-V524 //-V110
  uint getinpsize() { return pin[0].getinpsize(); } //-V110
  uint getoutsize() { return pin[1].getoutsize(); } //-V110

  void addinp( byte* inp,uint inplen ) { pin[0].addinp(inp,inplen); }
  void addout( byte* out,uint outlen ) { pin[1].addout(out,outlen); }

};


NOINLINE
static void yield( void* p, int value ) { 
  Coroutine& q = *(Coroutine*)p;
  char curtmp; q.stkptrL=(&curtmp)-16;
  if( setjmp(q.PointB)==0 ) { 
    q.state=value; 
    memcpy( q.stk, (char*)q.stkptrL, q.stkptrH-q.stkptrL );
    longjmp(q.PointA,1); 
    __assume(0);
  }
}

NOINLINE
static uint coro_call0( Coroutine* that ) {
  if_e1( setjmp(that->PointA)==0 ) {
    if_e1( that->state ) { // calls usually take this path, since other runs only on init
      memcpy( (char*)that->stkptrL, that->stk, that->stkptrH-that->stkptrL );
      longjmp(that->PointB,1); 
      __assume(0);
    }
    call_do_process0(that);
    __assume(0);
  }
  return that->state;
}


NOINLINE
static void call_do_process0( Coroutine* that ) {
  // call_do_process0 needs to be an actual separate function to allocate stack pad in its frame
  byte stktmp[Coroutine::STKPAD0]; 
  that->stkptrH = ((char*)stktmp);

  // do_process also needs a separate stack frame, to avoid merging stktmp into it, but ptr call is ok
  (that->*(that->p_do_process))();

  // do_process ends with yield(0) (can't normally return to changed frontend stack)
  // so tell compiler that this point can't be reached
  __assume(0);
}


coro3_pin_DEFINE_f_quit
#undef coro3_pin_DEFINE_f_quit

//#include "coro3_init.inc"

