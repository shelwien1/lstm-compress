
struct my_jmpbuf {
  qword rip,rsp;
};

#define ASM __asm__ volatile

INLINE
static int my_setjmp( my_jmpbuf* regs ) {
  qword r;

  ASM ("\
   movq %%rsp,8(%1); \n\
   call 1f; \n\
1: popq 0(%1); \n\
  " : "=a"(r) : "b"(regs),"a"(0) : "%rcx","%rdx","%rsi","%rdi","%rbp","%r8","%r9","%r10","%r11","%r12","%r13","%r14","%r15",
"ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15"
  );

  return r;
}

INLINE
static void my_jmp( my_jmpbuf* regs, int ) {
  ASM ("\
  xchg %0,%%rsp; \n\
  jmp *%1; \n\
  " :  : "d"(regs->rsp),"b"( ((byte*)regs->rip)+2 ),"a"(1) : 
  );
}

typedef my_jmpbuf m_jmp_buf[1];
#define jmp_buf m_jmp_buf
#define longjmp my_jmp
#define setjmp  my_setjmp

