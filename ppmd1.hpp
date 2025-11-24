//--- #include "libpmd.inc"

//namespace {

//--- #include "model_def.inc"

//#pragma pack(1)

const int ORealMAX=256;

//--- #include "sh_v1x.inc"

template< int ProcMode >
struct Rangecoder_SH1x : Coroutine {

  enum {
    SCALElog = 15,
    SCALE    = 1<<SCALElog
  };

  enum {
    NUM   = 4,
    sTOP  = 0x01000000U,
    gTOP  = 0x00010000U,
    Thres = 0xFF000000U,
    Threg = 0x00FF0000U
  };

//  int   ProcMode; // 0=encode, 1=decode;
  union {
    struct {
      uint  low;  
      uint  Carry;
    };
    qword lowc;
    uint  code; 
  };
  uint  FFNum;
  uint  Cache;
  uint  range;

  void rc_Process( uint cumFreq, uint freq, uint totFreq ) {
    uint tmp = cumFreq*range;
    if( ProcMode ) code-=tmp; else lowc+=tmp;
    range *= freq;
    Renorm();
  }


  void rc_Arrange( uint totFreq ) {
    range /= totFreq;
  }

  uint rc_GetFreq( uint totFreq ) {
    return code/range;
  }

  void Renorm( void ) {
    if( ProcMode ) {
 //     while( range<sTOP ) range<<=8, (code<<=8)+=get();
      if( range<gTOP ) range<<=16, (code<<=16)+=(get()<<8)+get(); 
      else if( range<sTOP ) range<<=8, (code<<=8)+=get();
    } else {
 //     while( range<sTOP ) range<<=8, ShiftLow();
      if( range<gTOP ) range<<=16, ShiftLow2(); 
      else if( range<sTOP ) range<<=8, ShiftLow();
    }
  }

//  NOINLINE
  void rc_BProcess( uint freq, int& bit ) { 

 //   uint rnew = (qword(range)*(freq<<(32-SCALElog)))>>32;
 //   uint rnew = ( range>=16*sTOP ) ? freq*(range>>SCALElog) : freq*(range>>(SCALElog-4))>>4;
    uint rnew = (range>>SCALElog)*freq;

 //   if( ProcMode ) bit = 1 + uint((qword(code)-rnew)>>32);
    if( ProcMode ) bit = (code>=rnew);

    range = ((range-rnew-rnew)&(-bit)) + rnew;
    rnew &= -bit;

    if( ProcMode ) code -= rnew; else lowc += rnew;

    Renorm();
  }

  void ShiftLow( void ) {
    if( low<Thres || Carry ) {
      put( Cache+Carry );
      for (;FFNum != 0;FFNum--) put( Carry-1 ); // (Carry-1)&255;
      Cache = low>>24;
      Carry = 0;
    } else FFNum++;
    low<<=8;
  }

  void ShiftLow2( void ) {
    if( low<Thres || Carry ) {
      put( Cache+Carry );
      for (;FFNum != 0;FFNum--) put( Carry-1 ); // (Carry-1)&255;
      Cache = low>>24;
      Carry = 0;
    } else FFNum++;
    low &= sTOP-1;
    if( low<Threg ) {
      put( Cache );
      for(; FFNum!=0; FFNum-- ) put( 0xFF ); // (Carry-1)&255;
      Cache = low>>16;
    } else FFNum++;
    low<<=16;
  }

  void rcInit( void ) { 
    range = 0xFFFFFFFF;
    low   = 0;
    FFNum = 0;
    Carry = 0;    
    Cache = 0;
  }
  
  void rc_Init( void ) {
    rcInit();
    if( ProcMode==1 ) {
      for(int _=0; _<NUM+1; _++) (code<<=8)+=get(); 
    }
  }

  void rc_Quit( void ) {
    if( ProcMode==0 ) {
      for(int _=0; _<NUM+1; _++) ShiftLow(); 
    }
  }

};


static signed char EscCoef[12] = { 16, -10, 1, 51, 14, 89, 23, 35, 64, 26, -42, 43  };

// Tabulated escapes for exponential symbol distribution
static const byte ExpEscape[16]={ 51,43,18,12,11,9,8,7,6,5,4,3,3,2,2,2 };


template< int ProcMode >
struct Model: Rangecoder_SH1x<ProcMode> {

  typedef Rangecoder_SH1x<ProcMode> Base;

  using Base::rc_BProcess;
  using Base::rc_Arrange;
  using Base::rc_GetFreq;
  using Base::rc_Process;
  using Base::rc_Init;
  using Base::rc_Quit;
  using Base::get;
  using Base::put;
  using Base::f_quit;

  typedef byte* pbyte;
  byte* HeapStart;
  uint   Ptr2Indx( void* p ) { return pbyte(p)-HeapStart; }
  void*  Indx2Ptr(uint indx) { return indx + HeapStart; }

  enum{ 
    UNIT_SIZE=12, 
    N1=4, N2=4, N3=4, N4=(128+3-1*N1-2*N2-3*N3)/4,
    N_INDEXES=N1+N2+N3+N4 
  };

//---   #include "alloc_node.inc"

struct _MEM_BLK { 
  uint Stamp;
  uint NextIndx;
  uint NU; 
};


struct BLK_NODE {
  uint Stamp;
  uint NextIndx;
  int avail() const { return (NextIndx!=0); }
};


BLK_NODE* getNext( BLK_NODE* This ) { 
  return (BLK_NODE*)Indx2Ptr(This->NextIndx); 
}

void setNext( BLK_NODE* This, BLK_NODE* p ) { 
  This->NextIndx = Ptr2Indx(p); 
}

void link( BLK_NODE* This, BLK_NODE* p ) { 
  p->NextIndx = This->NextIndx; 
  setNext( This, p ); 
}

void unlink( BLK_NODE* This ) { 
  This->NextIndx = getNext(This)->NextIndx; 
}

void* remove( BLK_NODE* This ) {
  BLK_NODE* p = getNext(This); 
  unlink(This);
  This->Stamp--;
  return p;
}

void insert( BLK_NODE* This, void* pv, int NU ) {
  BLK_NODE* p = (BLK_NODE*)pv;
  link(This,p);
  p->Stamp = ~uint(0);
  ((_MEM_BLK&)*p).NU = NU;
  This->Stamp++;
}


struct MEM_BLK : public BLK_NODE {
  uint NU; 
};

typedef BLK_NODE* pBLK_NODE;

typedef MEM_BLK* pMEM_BLK;

BLK_NODE BList[N_INDEXES+1];

  uint  GlueCount;
  uint  GlueCount1;
  uint  SubAllocatorSize;
  byte* pText;
  byte* UnitsStart;
  byte* LoUnit;
  byte* HiUnit;
  byte* AuxUnit;

//---   #include "alloc_units.inc"

uint U2B( uint NU ) { 
  return 8*NU+4*NU; 
}

int StartSubAllocator( uint SASize ) {
  uint t = SASize << 20U;
  HeapStart = new byte[t];
//  HeapStart = mAlloc<byte>(t);
//  HeapStart = (byte*)VirtualAlloc( 0, t, MEM_COMMIT, PAGE_READWRITE );
  if( HeapStart==NULL ) return 0;
  SubAllocatorSize = t;
  return 1;
}

void InitSubAllocator() {
  memset( BList, 0, sizeof(BList) );
  HiUnit = (pText=HeapStart) + SubAllocatorSize;
  uint Diff = U2B(SubAllocatorSize/8/UNIT_SIZE*7);
  LoUnit=UnitsStart = HiUnit-Diff;
  GlueCount=GlueCount1=0;
}

uint GetUsedMemory() {
  int i;
  uint RetVal = SubAllocatorSize - (HiUnit-LoUnit) - (UnitsStart-pText);
  for( i=0; i<N_INDEXES; i++ )
    RetVal -= U2B( Indx2Units[i]*BList[i].Stamp );
  return RetVal;
}

void StopSubAllocator() {
  if( SubAllocatorSize ) { SubAllocatorSize=0; delete[] HeapStart; }
//  if( SubAllocatorSize ) SubAllocatorSize=0, VirtualFree(HeapStart, 0, MEM_RELEASE);
}

//----------------------------------------

void GlueFreeBlocks() {
  uint i, k, sz;
  MEM_BLK s0;
  pMEM_BLK p, p0=&s0, p1;

  if( LoUnit!=HiUnit ) LoUnit[0]=0;

  for( p0->NextIndx=0,i=0; i<=N_INDEXES; i++ ) {
     while( BList[i].avail() ) {
       p = (MEM_BLK*)remove(&BList[i]);
       if( p->NU ) {
         while( p1 = p + p->NU, p1->Stamp==~uint(0) ) {
           p->NU += p1->NU;
           p1->NU = 0;
         }
         link(p0,p); p0=p;
       }
     }
  }

  while( s0.avail() ) {
    p = (MEM_BLK*)remove(&s0); 
    sz= p->NU;
    if( sz ) {
      for(; sz>128; sz-=128, p+=128 ) insert(&BList[N_INDEXES-1],p,128);
      i = Units2Indx[sz-1];
      if( Indx2Units[i] != sz ) {
        k = sz - Indx2Units[--i];
        insert( &BList[k-1], p+(sz-k) , k );
      }
      insert( &BList[i], p, Indx2Units[i] );
    }
  }

  GlueCount = 1 << (13+GlueCount1++);
}

void SplitBlock( void* pv, uint OldIndx, uint NewIndx ) {
  uint i, k, UDiff=Indx2Units[OldIndx]-Indx2Units[NewIndx];
  byte* p = ((byte*)pv)+U2B(Indx2Units[NewIndx]);
  i = Units2Indx[UDiff-1];
  if( Indx2Units[i]!=UDiff ) {
    k=Indx2Units[--i];
    insert(&BList[i],p,k);
    p += U2B(k);
    UDiff -= k;
  }
  insert( &BList[Units2Indx[UDiff-1]], p, UDiff );
}

void* AllocUnitsRare( uint indx ) {
  uint i = indx;
  do {
    if( ++i == N_INDEXES ) {
      if( !GlueCount-- ) {
        GlueFreeBlocks();
        if( BList[i=indx].avail() ) return remove(&BList[i]);
      } else {
        i = U2B(Indx2Units[indx]);
        return (UnitsStart-pText>i) ? UnitsStart-=i : NULL;
      }
    }
  } while( !BList[i].avail() );

  void* RetVal=remove(&BList[i]);
  SplitBlock( RetVal, i, indx );

  return RetVal;
}

void* AllocUnits( uint NU ) {
  uint indx = Units2Indx[NU-1];
  if( BList[indx].avail() ) return remove(&BList[indx]);
  void* RetVal=LoUnit; 
  LoUnit += U2B(Indx2Units[indx]);
  if( LoUnit<=HiUnit ) return RetVal;
  LoUnit -= U2B(Indx2Units[indx]);
  return AllocUnitsRare(indx);
}

void* AllocContext() {
  if( HiUnit!=LoUnit ) return HiUnit-=UNIT_SIZE;
  return BList->avail() ? remove(BList) : AllocUnitsRare(0);
}

void FreeUnits( void* ptr, uint NU ) {
  uint indx = Units2Indx[NU-1];
  insert( &BList[indx], ptr, Indx2Units[indx] );
}

void FreeUnit( void* ptr ) {
  int i = (byte*)ptr > UnitsStart+128*1024 ? 0 : N_INDEXES;
  insert( &BList[i], ptr, 1 );
}

//----------------------------------------

void UnitsCpy( void* Dest, void* Src, uint NU ) {
  memcpy( Dest, Src, 12*NU );
}

void* ExpandUnits( void* OldPtr, uint OldNU ) {
  uint i0 = Units2Indx[OldNU-1];
  uint i1 = Units2Indx[OldNU-1+1];
  if( i0==i1 ) return OldPtr;
  void* ptr = AllocUnits(OldNU+1);
  if( ptr ) { 
    UnitsCpy( ptr, OldPtr, OldNU ); 
    insert( &BList[i0], OldPtr, OldNU );
  }
  return ptr;
}

void* ShrinkUnits( void* OldPtr, uint OldNU, uint NewNU ) {
  uint i0 = Units2Indx[OldNU-1];
  uint i1 = Units2Indx[NewNU-1];
  if( i0==i1 ) return OldPtr;
  if( BList[i1].avail() ) {
    void* ptr = remove(&BList[i1]);
    UnitsCpy( ptr, OldPtr, NewNU );
    insert( &BList[i0], OldPtr, Indx2Units[i0] );
    return ptr;
  } else { 
    SplitBlock(OldPtr,i0,i1);
    return OldPtr; 
  }
}

void* MoveUnitsUp( void* OldPtr, uint NU ) {
  uint indx = Units2Indx[NU-1];
  PrefetchData(OldPtr);
  if( (byte*)OldPtr > UnitsStart+128*1024 ||
      (BLK_NODE*)OldPtr > getNext(&BList[indx]) ) return OldPtr;

  void* ptr = remove(&BList[indx]);
  UnitsCpy( ptr, OldPtr, NU );

  insert( &BList[N_INDEXES], OldPtr, Indx2Units[indx] );

  return ptr;
}

void PrepareTextArea() {
  AuxUnit = (byte*)AllocContext();
  if( !AuxUnit ) {
    AuxUnit = UnitsStart;
  } else {
    if( AuxUnit==UnitsStart) AuxUnit = (UnitsStart+=UNIT_SIZE);
  }
}

void ExpandTextArea() {
  BLK_NODE* p;
  uint Count[N_INDEXES], i=0;
  memset( Count, 0, sizeof(Count) );

  if( AuxUnit!=UnitsStart ) {
    if( *(uint*)AuxUnit != ~uint(0) ) 
      UnitsStart += UNIT_SIZE;
    else
      insert( BList, AuxUnit, 1 );
  }

  while( (p=(BLK_NODE*)UnitsStart)->Stamp == ~uint(0) ) {
    MEM_BLK* pm = (MEM_BLK*)p;
    UnitsStart = (byte*)(pm + pm->NU);
    Count[Units2Indx[pm->NU-1]]++;
    i++;
    pm->Stamp = 0;
  }

  if( i ) {

    for( p=BList+N_INDEXES; p->NextIndx; p=getNext(p) ) {
      while( p->NextIndx && !getNext(p)->Stamp ) {
        Count[Units2Indx[((MEM_BLK*)getNext(p))->NU-1]]--;
        unlink(p);
        BList[N_INDEXES].Stamp--;
      }
      if( !p->NextIndx ) break;
    }

    for( i=0; i<N_INDEXES; i++ ) {
      for( p=BList+i; Count[i]!=0; p=getNext(p) ) {
        while( !getNext(p)->Stamp ) {
          unlink(p); BList[i].Stamp--;
          if ( !--Count[i] ) break;
        }
      }
    }

  }

}


//---   #include "ppmd_init.inc"

static const int MAX_O=ORealMAX;  // maximum allowed model order

//enum { FALSE=0,TRUE=1 };

template <class T> 
  T CLAMP( const T& X, const T& LoX, const T& HiX ) { return (X >= LoX)?((X <= HiX)?(X):(HiX)):(LoX); }

template <class T>
  void SWAP( T& t1, T& t2 ) { T tmp=t1; t1=t2; t2=tmp; }

void PrefetchData(void* Addr) {
  byte Prefetchbyte = *(volatile byte*)Addr;
}

enum { 
  UP_FREQ     = 5
};

byte Indx2Units[N_INDEXES];
byte Units2Indx[128]; // constants

byte NS2BSIndx[256];
byte QTable[260]; 


// constants initialization
void PPMD_STARTUP( void ) {
  int i, k, m, Step;

  for( i=0,k=1; i<N1         ;i++,k+=1 ) Indx2Units[i]=k;
  for( k++;     i<N1+N2      ;i++,k+=2 ) Indx2Units[i]=k;
  for( k++;     i<N1+N2+N3   ;i++,k+=3 ) Indx2Units[i]=k;
  for( k++;     i<N1+N2+N3+N4;i++,k+=4 ) Indx2Units[i]=k;

  for( k=0,i=0; k<128; k++ ) {
    i += Indx2Units[i]<k+1;
    Units2Indx[k]=i;
  }

  NS2BSIndx[0] = 2*0; //-V525
  NS2BSIndx[1] = 2*1;
  NS2BSIndx[2] = 2*1;
  memset(NS2BSIndx+3,  2*2, 26);             
  memset(NS2BSIndx+29, 2*3, 256-29);

  for( i=0; i<UP_FREQ; i++ ) QTable[i]=i;

  for( m=i=UP_FREQ, k=Step=1; i<260; i++ ) {
    QTable[i] = m;
    if( !--k ) k = ++Step, m++;
  }
}

//---   #include "mod_context.inc"

enum {
  MAX_FREQ    = 124,
  O_BOUND     = 9 
};

struct PPM_CONTEXT;

struct STATE {
  byte Symbol;
  byte Freq;
  uint iSuccessor;
};

PPM_CONTEXT* getSucc( STATE* This ) { 
  return (PPM_CONTEXT*)Indx2Ptr( This->iSuccessor ); 
}


void SWAP( STATE& s1, STATE& s2 ) {
  word t1       = (word&)s1;
  uint t2       = s1.iSuccessor;
  (word&)s1     = (word&)s2;
  s1.iSuccessor = s2.iSuccessor;
  (word&)s2     = t1;                        
  s2.iSuccessor = t2;
}

struct PPM_CONTEXT {

  byte NumStats;
  byte Flags;
  word SummFreq;
  uint iStats;
  uint iSuffix;

  STATE& oneState() const { return (STATE&) SummFreq; }
};

STATE* getStats( PPM_CONTEXT* This ) { return (STATE*)Indx2Ptr(This->iStats); }

PPM_CONTEXT* suff( PPM_CONTEXT* This ) { return (PPM_CONTEXT*)Indx2Ptr(This->iSuffix); }


  int _MaxOrder, _CutOff, _MMAX;
  uint _filesize;
  int OrderFall;

  STATE* FoundState; // found next state transition
  PPM_CONTEXT* MaxContext;

  uint EscCount;
  uint CharMask[256];

  int  BSumm;
  int  RunLength;
  int  InitRL;

//---   #include "mod_see.inc"

enum { 
  INT_BITS    = 7, 
  PERIOD_BITS = 7, 
  TOT_BITS    = INT_BITS + PERIOD_BITS,
  INTERVAL    = 1 << INT_BITS, 
  BIN_SCALE   = 1 << TOT_BITS, 
  ROUND       = 16
};

// SEE-contexts for PPM-contexts with masked symbols
struct SEE2_CONTEXT { 
  word Summ;
  byte Shift;
  byte Count;

  void init( uint InitVal ) { 
    Shift = PERIOD_BITS-4;
    Summ  = InitVal << Shift; 
    Count = 7; 
  }

  uint getMean() {
    return Summ >> Shift;
  }

  void update() { 
    if( --Count==0 ) setShift_rare(); 
  }

  void setShift_rare() {
    uint i = Summ >> Shift;
    i = PERIOD_BITS - (i>40) - (i>280) - (i>1020);
    if( i<Shift ) { Summ >>= 1; Shift--; } else
    if( i>Shift ) { Summ <<= 1; Shift++; }
    Count = 5 << Shift;
  }
};


  int  NumMasked;

//---   #include "mod_rescale.inc"

STATE* rescale( PPM_CONTEXT& q, int OrderFall, STATE* FoundState ) {
  STATE tmp; STATE* p; STATE* p1;

  q.Flags &= 0x14;

  // move the current node to rank0
  p1 = getStats(&q);
  tmp = FoundState[0];
  for( p=FoundState; p!=p1; p-- ) p[0]=p[-1];
  p1[0] = tmp;

  int of = (OrderFall != 0);
  int a, i;
  int f0 = p->Freq;
  int sf = q.SummFreq;
  int EscFreq = sf-f0;
  q.SummFreq = p->Freq = (f0+of)>>1;

  // sort symbols by freqs
  for( i=0; i<q.NumStats; i++ ) {
    p++;
    a = p->Freq;
    EscFreq  -= a;
    a = (a+of)>>1;
    p->Freq = a; 
    q.SummFreq += a;
    if( a ) q.Flags |= 0x08*(p->Symbol>=0x40);
    if( a > p[-1].Freq ) {
      tmp = p[0];
      for( p1=p; tmp.Freq>p1[-1].Freq; p1-- ) p1[0]=p1[-1];
      p1[0] = tmp;
    }
  }

  // remove the zero freq nodes
  if( p->Freq==0 ) {
    for( i=0; p->Freq==0; i++,p-- );
    EscFreq += i;
    a = (q.NumStats+2) >> 1;
    if( (q.NumStats-=i)==0 ) {
      tmp = getStats(&q)[0];
      tmp.Freq = Min( MAX_FREQ/3, (2*tmp.Freq+EscFreq-1)/EscFreq );
      q.Flags &= 0x18;
      FreeUnits( getStats(&q), a );
      q.oneState() = tmp;
      FoundState = &q.oneState();
      return FoundState;
    }
    q.iStats = Ptr2Indx( ShrinkUnits(getStats(&q),a,(q.NumStats+2)>>1) );
  }

  // some weird magic
  q.SummFreq += (EscFreq+1) >> 1;
  if( OrderFall || (q.Flags & 0x04)==0 ) {
    a = (sf-=EscFreq) - f0;
    a = CLAMP( uint( ( f0*q.SummFreq - sf*getStats(&q)->Freq + a-1 ) / a ), 2U, MAX_FREQ/2U-18U );
  } else {
    a = 2;
  }

  (FoundState=getStats(&q))->Freq += a;
  q.SummFreq += a;
  q.Flags |= 0x04;

  return FoundState;
}
//---   #include "mod_cutoff.inc"

void AuxCutOff( STATE* p, int Order, int MaxOrder ) {
  if( Order<MaxOrder ) {
    PrefetchData( getSucc(p) );
    p->iSuccessor = cutOff( getSucc(p)[0], Order+1,MaxOrder);
  } else {
    p->iSuccessor=0;
  }
}

uint cutOff( PPM_CONTEXT& q, int Order, int MaxOrder ) {
  int i, tmp, EscFreq, Scale;
  STATE* p;
  STATE* p0;

  // for binary context, just cut off the successors
  if( q.NumStats==0 ) {

    int flag = 1;
    p = &q.oneState();
    if( (byte*)getSucc(p) >= UnitsStart ) {
      AuxCutOff( p, Order, MaxOrder );
      if( p->iSuccessor || Order<O_BOUND ) flag=0;
    }
    if( flag ) {
      FreeUnit( &q );
      return 0;
    }

  } else {

    tmp = (q.NumStats+2)>>1;
    p0 = (STATE*)MoveUnitsUp(getStats(&q),tmp);
    q.iStats = Ptr2Indx(p0);

    // cut the branches with links to text
    for( i=q.NumStats, p=&p0[i]; p>=p0; p-- ) {
      if( (byte*)getSucc(p) < UnitsStart ) {
        p[0].iSuccessor=0;
        SWAP( p[0], p0[i--] );
      } else AuxCutOff( p, Order, MaxOrder );
    }

    // if something was cut
    if( i!=q.NumStats && Order>0 ) {
      q.NumStats = i;
      p = p0;
      if( i<0 ) { 
        FreeUnits( p, tmp );
        FreeUnit( &q );
        return 0;
      } 
      if( i==0 ) {
        q.Flags = (q.Flags & 0x10) + 0x08*(p[0].Symbol>=0x40);
        p[0].Freq = 1+(2*(p[0].Freq-1))/(q.SummFreq-p[0].Freq);
        q.oneState() = p[0];
        FreeUnits( p, tmp );
      } else {
        p = (STATE*)ShrinkUnits( p0, tmp, (i+2)>>1 );
        q.iStats = Ptr2Indx(p);
        Scale = (q.SummFreq>16*i); // av.freq > 16
        q.Flags = (q.Flags & (0x10+0x04*Scale));
        if( Scale ) {
          EscFreq = q.SummFreq;
          q.SummFreq = 0;
          for( i=0; i<=q.NumStats; i++ ) {
            EscFreq  -= p[i].Freq;
            p[i].Freq = (p[i].Freq+1)>>1;
            q.SummFreq += p[i].Freq;
            q.Flags |= 0x08*(p[i].Symbol>=0x40);
          };
          EscFreq = (EscFreq+1)>>1;
          q.SummFreq += EscFreq;
        } else {
          for( i=0; i<=q.NumStats; i++ ) q.Flags |= 0x08*(p[i].Symbol>=0x40);
        }
      }
    }

  }

  if( (byte*)&q==UnitsStart ) {
    // if this is a root, copy it
    UnitsCpy( AuxUnit, &q, 1 );
    return Ptr2Indx(AuxUnit);
  } else {
    // if suffix is root, switch the pointer
    if( (byte*)suff(&q)==UnitsStart ) q.iSuffix=Ptr2Indx(AuxUnit);
  }

  return Ptr2Indx(&q);
}

//---   #include "ppmd_flush.inc"

NOINLINE
void StartModelRare( void ) {
  int i, k, s;
  byte i2f[25];

  memset( CharMask, 0, sizeof(CharMask) );
  EscCount=1;

  // we are in solid mode
  if( _MaxOrder<2 ) {
    OrderFall = _MaxOrder;
    for( PPM_CONTEXT* pc=MaxContext; pc->iSuffix!=0; pc=suff(pc) ) OrderFall--;
    return;
  }

  OrderFall = _MaxOrder;

  InitSubAllocator();

  InitRL = -( (_MaxOrder<13) ? _MaxOrder : 13 );
  RunLength = InitRL;

  // alloc and init order0 context
  MaxContext = (PPM_CONTEXT*)AllocContext();
  MaxContext->NumStats = 255;
  MaxContext->SummFreq = 255+2;
  MaxContext->iStats   = Ptr2Indx(AllocUnits(256/2));
  MaxContext->Flags    = 0;
  MaxContext->iSuffix  = 0;
  PrevSuccess          = 0;

  for( i=0; i<256; i++ ) {
    getStats(MaxContext)[i].Symbol     = i; 
    getStats(MaxContext)[i].Freq       = 1;
    getStats(MaxContext)[i].iSuccessor = 0;
  }

  // _InitSEE
  if( 1 ) {
    // a freq for quant?
    for( k=i=0; i<25; i2f[i++]=k+1 ) while( QTable[k]==i ) k++;

    // bin SEE init
    for( k=0; k<64; k++ ) {
      for( s=i=0; i<6; i++ ) s += EscCoef[2*i+((k>>i)&1)];
      s = 128*CLAMP( s, 32, 256-32 );
      for( i=0; i<25; i++ ) BinSumm[i][k] = BIN_SCALE - s/i2f[i];
    }

    // masked SEE init
    for( i=0; i<23; i++ ) for( k=0; k<32; k++ ) SEE2Cont[i][k].init(8*i+5);
    DummySEE2Cont.init(0);
  }
}


// model flush
NOINLINE
void RestoreModelRare( void ) {
  STATE* p; 
  pText = HeapStart;
  PPM_CONTEXT* pc = saved_pc;

  // from maxorder down, while there 2 symbols and 2nd symbol has a text pointer
  for(;; MaxContext=suff(MaxContext) ) {
   if( (MaxContext->NumStats==1) && (MaxContext!=pc) ) {
     p = getStats(MaxContext);
     if( (byte*)(getSucc(p+1))>=UnitsStart ) break;
   } else break;
    // turn a context with 2 symbols into a context with 1 symbol
    MaxContext->Flags = (MaxContext->Flags & 0x10) + 0x08*(p->Symbol>=0x40);
    p[0].Freq = (p[0].Freq+1) >> 1;
    MaxContext->oneState() = p[0];
    MaxContext->NumStats=0;
    FreeUnits( p, 1 );
  }

  // go all the way down
  while( MaxContext->iSuffix ) MaxContext=suff(MaxContext);

  AuxUnit = UnitsStart;

  ExpandTextArea();

  // free up 25% of memory
  do {
    PrepareTextArea();
    cutOff( MaxContext[0], 0, _MaxOrder ); // MaxContext is a tree root here, order0
    ExpandTextArea();
  } while( GetUsedMemory()>3*(SubAllocatorSize>>2) );

  GlueCount = GlueCount1 = 0;
  OrderFall = _MaxOrder;
}


//---   #include "ppmd_update.inc"

PPM_CONTEXT* saved_pc;

PPM_CONTEXT* UpdateModel( PPM_CONTEXT* MinContext ) {
  byte Flag, sym, FSymbol;
  uint ns1, ns, cf, sf, s0, FFreq;
  uint iSuccessor, iFSuccessor;
  PPM_CONTEXT* pc;
  STATE* p = NULL;

  FSymbol     = FoundState->Symbol;
  FFreq       = FoundState->Freq;
  iFSuccessor = FoundState->iSuccessor;

  // partial update for the suffix context
  if( MinContext->iSuffix ) {
    pc = suff(MinContext);
    // is it binary?
    if( pc[0].NumStats ) {
      p = getStats(pc);
      if( p[0].Symbol!=FSymbol ) {
        for( p++; p[0].Symbol!=FSymbol; p++ );
        if( p[0].Freq >= p[-1].Freq ) SWAP( p[0], p[-1] ), p--;
      }
      if( p[0].Freq<MAX_FREQ-3 ) {
        cf = 2 + (FFreq<28);
        p[0].Freq += cf;
        pc[0].SummFreq += cf;
      }
    } else { 
      p = &(pc[0].oneState());
      p[0].Freq += (p[0].Freq<14); 
    }
  }

  // try increasing the order
  if( !OrderFall && iFSuccessor ) {
    FoundState->iSuccessor = CreateSuccessors( 1, p, MinContext );
    if( !FoundState->iSuccessor ) { saved_pc=pc; return 0; };
    MaxContext = getSucc(FoundState);
    return MaxContext;
  }

  *pText++ = FSymbol;
  iSuccessor = Ptr2Indx(pText);
  if( pText>=UnitsStart ) { saved_pc=pc; return 0; };

  if( iFSuccessor ) {
    if( (byte*)Indx2Ptr(iFSuccessor) < UnitsStart )
     iFSuccessor = CreateSuccessors( 0, p, MinContext );
    else
     PrefetchData( Indx2Ptr(iFSuccessor) );
  } else
    iFSuccessor = ReduceOrder( p, MinContext );

  if( !iFSuccessor ) { saved_pc=pc; return 0; };

  if( !--OrderFall ) {
    iSuccessor = iFSuccessor;
    pText -= (MaxContext!=MinContext);
  }

  s0 = MinContext->SummFreq - FFreq;
  ns = MinContext->NumStats;
  Flag = 0x08*(FSymbol>=0x40);
  for( pc=MaxContext; pc!=MinContext; pc=suff(pc) ) {
    ns1 = pc[0].NumStats;
    // non-binary context?
    if( ns1 ) {
      // realloc table with alphabet size is odd
      if( ns1&1 ) {
        p = (STATE*)ExpandUnits( getStats(pc),(ns1+1)>>1 );
        if( !p ) { saved_pc=pc; return 0; };
        pc[0].iStats = Ptr2Indx(p);
      }
      // increase escape freq (more for larger alphabet)
      pc[0].SummFreq += QTable[ns+4] >> 3;
    } else {
      // escaped binary context
      p = (STATE*)AllocUnits(1);
      if( !p ) { saved_pc=pc; return 0; };
      p[0] = pc[0].oneState();
      pc[0].iStats = Ptr2Indx(p);
      p[0].Freq = (p[0].Freq<=MAX_FREQ/3) ? (2*p[0].Freq-1) : (MAX_FREQ-15);
      // update escape
      pc[0].SummFreq = p[0].Freq + (ns>1) + ExpEscape[QTable[BSumm>>8]]; //-V602
    }

    // inheritance
    cf = (FFreq-1)*(5 + pc[0].SummFreq); 
    sf = s0 + pc[0].SummFreq;
    // this is a weighted rescaling of symbol's freq into a new context (cf/sf)
    if( cf<=3*sf ) {
      // if the new freq is too small the we increase the escape freq too
      cf = 1 + (2*cf>sf) + (2*cf>3*sf);
      pc[0].SummFreq += 4;
    } else {
      cf = 5 + (cf>5*sf) + (cf>6*sf) + (cf>8*sf) + (cf>10*sf) + (cf>12*sf);
      pc[0].SummFreq += cf;
    }

    p = getStats(pc) + (++pc[0].NumStats);  
    p[0].iSuccessor = iSuccessor;
    p[0].Symbol = FSymbol;
    p[0].Freq   = cf;
    pc[0].Flags |= Flag; // flag if last added symbol was >=0x40
  }

  MaxContext = (PPM_CONTEXT*)Indx2Ptr(iFSuccessor);
  return MaxContext;
}


uint CreateSuccessors( uint Skip, STATE* p, PPM_CONTEXT* pc ) {
  byte tmp;
  uint cf, s0;
  STATE*  ps[MAX_O];
  STATE** pps=ps;

  byte sym = FoundState->Symbol;
  uint iUpBranch = FoundState->iSuccessor;

  if( !Skip ) {
    *pps++ = FoundState;
    if( !pc[0].iSuffix ) goto NO_LOOP;
  }

  if( p ) { pc = suff(pc); goto LOOP_ENTRY; }

  do {
    pc = suff(pc);

    // increment current symbol's freq in lower order contexts
    // more partial updates?
    if( pc[0].NumStats ) {
      // find sym node
      for( p=getStats(pc); p[0].Symbol!=sym; p++ );
      // increment freq if limit allows
      tmp = 2*(p[0].Freq<MAX_FREQ-1);
      p[0].Freq += tmp;
      pc[0].SummFreq += tmp;
    } else {
      // binary context
      p = &(pc[0].oneState());
      p[0].Freq += (!suff(pc)->NumStats & (p[0].Freq<16));
    }

LOOP_ENTRY:
    if( p[0].iSuccessor!=iUpBranch ) {
      pc = getSucc(p);
      break;
    }
    *pps++ = p;
  } while ( pc[0].iSuffix );

NO_LOOP:
  if( pps==ps ) return Ptr2Indx(pc);

  // fetch a following symbol from the text buffer
  PPM_CONTEXT ct;
  ct.NumStats = 0;
  ct.Flags = 0x10*(sym>=0x40);
  sym = *(byte*)Indx2Ptr(iUpBranch);
  ct.oneState().iSuccessor = Ptr2Indx((byte*)Indx2Ptr(iUpBranch)+1);
  ct.oneState().Symbol = sym;
  ct.Flags |= 0x08*(sym>=0x40);

  // pc is MinContext, the context used for encoding
  if( pc[0].NumStats ) {
    for( p=getStats(pc); p[0].Symbol!=sym; p++ );
    cf = p[0].Freq - 1;
    s0 = pc[0].SummFreq - pc[0].NumStats - cf;
    cf = 1 + ((2*cf<s0) ? (12*cf>s0) : 2+cf/s0);
    ct.oneState().Freq = Min<uint>( 7, cf );
  } else {
    ct.oneState().Freq = pc[0].oneState().Freq;
  }

  // attach the new node to all orders
  do {
    PPM_CONTEXT* pc1 = (PPM_CONTEXT*)AllocContext();
    if( !pc1 ) return 0;
    ((uint*)pc1)[0] = ((uint*)&ct)[0];
    ((uint*)pc1)[1] = ((uint*)&ct)[1];
    pc1->iSuffix = Ptr2Indx(pc);
    pc = pc1; pps--;
    pps[0][0].iSuccessor = Ptr2Indx(pc);
  } while( pps!=ps );

  return Ptr2Indx(pc);
}


uint ReduceOrder( STATE* p, PPM_CONTEXT* pc ) {
  byte tmp;
  STATE* p1;
  PPM_CONTEXT* pc1=pc;
  FoundState->iSuccessor = Ptr2Indx(pText);
  byte sym = FoundState->Symbol;
  uint iUpBranch = FoundState->iSuccessor;
  OrderFall++;

  if( p ) { pc=suff(pc); goto LOOP_ENTRY; }

  while(1) {
    if( !pc->iSuffix ) return Ptr2Indx(pc);
    pc = suff(pc);

    if( pc->NumStats ) {
      for( p=getStats(pc); p[0].Symbol!=sym; p++ );
      tmp = 2*(p->Freq<MAX_FREQ-3);
      p->Freq += tmp;
      pc->SummFreq += tmp;
    } else { 
      p = &(pc->oneState());
      p->Freq += (p->Freq<11);
    }

LOOP_ENTRY:
    if( p->iSuccessor ) break;
    p->iSuccessor = iUpBranch;
    OrderFall++;
  }

  if( p->iSuccessor<=iUpBranch ) {
    p1 = FoundState;
    FoundState = p;
    p->iSuccessor = CreateSuccessors(0,0,pc);
    FoundState = p1;
  }

  if( OrderFall==1 && pc1==MaxContext ) {
    FoundState->iSuccessor = p->iSuccessor;
    pText--;
  }

  return p->iSuccessor;
}

//---   #include "ppmd_proc0.inc"

int  PrevSuccess;
word BinSumm[25][64]; // binary SEE-contexts

void processBinSymbol( PPM_CONTEXT& q, int symbol ) {
  STATE& rs = q.oneState();
  int   i  = NS2BSIndx[suff(&q)->NumStats] + PrevSuccess + q.Flags + ((RunLength>>26) & 0x20);
  word& bs = BinSumm[QTable[rs.Freq-1]][i];
  BSumm    = bs;
  bs      -= (BSumm+64) >> PERIOD_BITS;

  int flag = ProcMode ? 0 : rs.Symbol!=symbol;
  rc_BProcess( BSumm+BSumm, flag );

  if( flag ) {
    CharMask[rs.Symbol] = EscCount;
    NumMasked = 0;
    PrevSuccess = 0;
    FoundState = 0;
  } else {
    bs += INTERVAL;
    rs.Freq += (rs.Freq<196);
    RunLength++;
    PrevSuccess = 1;
    FoundState = &rs;
  }
}

//---   #include "ppmd_proc1.inc"

// encode in unmasked (maxorder) context
void processSymbol1( PPM_CONTEXT& q, int symbol ) {
  STATE* p = getStats(&q);

  int cnum  = q.NumStats;
  int i     = p[0].Symbol;
  int low   = 0;
  int freq  = p[0].Freq;
  int total = q.SummFreq;
  int flag;
//  int mode;
  int count;

  rc_Arrange( total );
  if( ProcMode ) {
    count = rc_GetFreq( total );
    flag  = count<freq;
  } else {
    flag  = i==symbol;
  }

  if( flag ) {

//    mode = 2;
    PrevSuccess = 0;//(2*freq>1*total);
    p[0].Freq  += 4;
    q.SummFreq += 4;

  } else {

    PrevSuccess = 0;

    for( low=freq,i=1; i<=cnum; i++ ) {
      freq = p[i].Freq;
      flag = ProcMode ? low+freq>count : p[i].Symbol==symbol;
      if( flag ) break;
      low += freq;
    }

//    mode = 2+1+flag;

    if( flag ) {
      p[i].Freq  += 4;
      q.SummFreq += 4;
      if( p[i].Freq > p[i-1].Freq ) SWAP( p[i], p[i-1] ), i--;
      p = &p[i];
    } else {
      if( q.iSuffix ) PrefetchData( suff(&q) );
      freq = total-low;
      NumMasked = cnum;
      for( i=0; i<=cnum; i++ ) CharMask[p[i].Symbol]=EscCount;
      p = NULL;
    }
  }

  rc_Process( low, freq, total );

  FoundState = p;
  if( p && (p[0].Freq>MAX_FREQ) ) FoundState=rescale(q,OrderFall,FoundState);
}

//---   #include "ppmd_proc2.inc"

SEE2_CONTEXT SEE2Cont[23][32];
SEE2_CONTEXT DummySEE2Cont;

// encode in masked context
void processSymbol2( PPM_CONTEXT& q, int symbol ) {
  byte px[256];
  STATE* p = getStats(&q);

  int c;
  int count;
  int low;
  int see_freq;
  int freq;
  int cnum = q.NumStats;

  SEE2_CONTEXT* psee2c;
  if( cnum != 0xFF ) {
    psee2c = SEE2Cont[ QTable[cnum+3]-4 ];
    psee2c+= (q.SummFreq > 10*(cnum+1));
    psee2c+= 2*(2*cnum < suff(&q)->NumStats+NumMasked) + q.Flags;
    see_freq = psee2c->getMean()+1;
//    if( see_freq==0 ) psee2c->Summ+=1, see_freq=1;
  } else { 
    psee2c = &DummySEE2Cont;
    see_freq = 1; 
  }

  int flag=0,pj,pl;

  int i,j;
  for( i=0,j=0,low=0; i<=cnum; i++ ) {
    c = p[i].Symbol; 
    if( CharMask[c]!=EscCount ) {
      CharMask[c]=EscCount;
      low += p[i].Freq;
      if( ProcMode ) 
        px[j++] = i;
      else
        if( c==symbol ) flag=1,j=i,pl=low;
    }
  }

  int Total = see_freq + low;

  rc_Arrange( Total );
  if( ProcMode ) {
    count = rc_GetFreq( Total );
    flag = count<low;
  }

  if( flag ) {
    if( ProcMode ) {
      for( low=0, i=0; (low+=p[j=px[i]].Freq)<=count; i++ );
    } else {
      low = pl;
    }
    p+=j;

    freq = p[0].Freq;

    if( see_freq>2 ) psee2c->Summ -= see_freq;
    psee2c->update();

    FoundState = p;
    p[0].Freq  += 4;
    q.SummFreq += 4; if( p[0].Freq > MAX_FREQ ) FoundState=rescale(q,OrderFall,FoundState);
    RunLength = InitRL;
    EscCount++;

  } else {

    low = Total;
    freq = see_freq;

    NumMasked  = cnum;
    psee2c->Summ += Total-see_freq;

  }

  rc_Process( low-freq, freq, Total );

}


  uint Init( uint MaxOrder, uint MMAX, uint CutOff/*, uint filesize*/ ) {
    _MaxOrder = MaxOrder;
    _CutOff = CutOff;
    _MMAX = MMAX;
    //_filesize = filesize;

    PPMD_STARTUP();

    //f_quit=0; coro_init();

    if( !StartSubAllocator( _MMAX ) ) return 1;

    StartModelRare();

//printf( "f_DEC=%i ord=%i mem=%i cutoff=%i\n", ProcMode, _MaxOrder, _MMAX, _CutOff );

    return 0;
  }

  void Quit( void ) {
    StopSubAllocator();
  }

  void do_process( void ) {
    uint c;

    rc_Init();

    while( f_quit==0 ) {
      c = 0;

      if( ProcMode==0 ) { c = get(); if( f_quit ) break; }

      c = ProcessByte( c ); 

      if( ProcMode==1 ) { if( c!=-1 ) put(c); else f_quit=1; }
    }

    if( ProcMode==0 ) {
      ProcessByte( -1 );
      rc_Quit(); 
    }

    yield(this,0);
  }

//---   #include "ppmd_byte.inc"

uint ProcessByte( uint c ) {

  PPM_CONTEXT* MinContext = MaxContext;
  if( MinContext->NumStats ) {
    processSymbol1(   MinContext[0], c );
  } else {
    processBinSymbol( MinContext[0], c );
  }

  while( !FoundState ) {
    do {
      if( !MinContext->iSuffix ) { return -1; };
      OrderFall++;
      MinContext = suff(MinContext);
    } while( MinContext->NumStats==NumMasked );
    processSymbol2( MinContext[0], c );
  }

  if( ProcMode ) c = FoundState->Symbol;

  PPM_CONTEXT* p;
  if( (OrderFall!=0) || ((byte*)getSucc(FoundState)<UnitsStart) ) {
    p = UpdateModel( MinContext );
    if( p ) MaxContext = p;
  } else {
    p = MaxContext = getSucc(FoundState);
  }

  if( p==0 ) {
    if( _CutOff ) {
      RestoreModelRare();
    } else {
      StartModelRare();
    }
  }

  return c;
}

};

//#pragma pack()

//typedef Model<0> Model0;
//typedef Model<1> Model1;

struct pmd_codec : Coroutine {

  struct DecWrap : Model<1> {};
  struct EncWrap : Model<0> {};

  #define Max(x,y) ((x)>(y)?(x):(y))
  enum {
    MSize0 = sizeof(EncWrap),
    MSize1 = sizeof(DecWrap),
    MSize2 = Max( MSize0, MSize1 ) - sizeof(Coroutine)
  };
  #undef Max

  ALIGN(32)
  byte pad[ MSize2 ];

  uint f_DEC;

  uint ppmd_order;
  uint ppmd_memory;
  uint ppmd_restore; // flag
  uint ppmd_filesize;

//  uint Init( int MaxOrder, int CutOff, int MMAX ) {

  uint Init( uint _mode, uint* _args, uint f_noflush=1 ) { 
    DecWrap& M1 = *(DecWrap*)this;
    EncWrap& M0 = *(EncWrap*)this;
    uint r=0;
    if( f_noflush ) coro_init(), f_quit=0;
    f_DEC = _mode;
    ppmd_order = _args[0];
    ppmd_memory = _args[1];
    ppmd_restore = _args[2];
    //ppmd_filesize = _args[3];

    r = f_DEC ? M1.Init(ppmd_order,ppmd_memory,ppmd_restore)
              : M0.Init(ppmd_order,ppmd_memory,ppmd_restore);
    return r;
  }

  void Flush( uint _mode, uint* _args ) {
    Quit();
    Init(_mode,_args,0);
  }

  void Quit( void ) {
    DecWrap& M1 = *(DecWrap*)this;
    EncWrap& M0 = *(EncWrap*)this;
    f_DEC ? M1.Quit() : M0.Quit();
  }

  uint GetUsedMemory() {
    DecWrap& M1 = *(DecWrap*)this;
    EncWrap& M0 = *(EncWrap*)this;
    return f_DEC ? M1.GetUsedMemory() : M0.GetUsedMemory();
  }

  NOINLINE
  void do_process( void ) {

    DecWrap& M1 = *(DecWrap*)this;
    EncWrap& M0 = *(EncWrap*)this;

    f_DEC ? M1.do_process() : M0.do_process();

    yield(this,0);
  }

};


//#include "pmd_api_.inc"

//}



