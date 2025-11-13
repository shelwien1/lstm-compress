
#include "mim-include/mimalloc-new-delete.h"

// C library headers
#include <stdlib.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <math.h>

// C++ library headers
#include <algorithm>
#include <memory>
#include <numeric>
#include <valarray>
#include <vector>

#define INC_FLEN
#include "common.inc"

#include "sh_v2f.inc"

#include "lstm-model.inc"
#include "sigmoid.hpp"
#include "neuron-layer.hpp"
#include "lstm-layer.hpp"
#include "lstm.hpp"
#include "byte-model.hpp"
#include "ppmd-model.hpp"
#include "model.hpp"

//#include <optional>

Rangecoder rc;

static const uint CNUM = 256;

char cmap[CNUM];

//std::optional<PPMD::PPMD> byte_model_;
unsigned int bit_context_ = 1;

int main( int argc, char** argv ) {

  if( argc<3 ) return 1;

  uint f_DEC = (argv[1][0]=='d');
  FILE* f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  FILE* g = fopen(argv[3],"wb"); if( g==0 ) return 3;

  uint i,j,c,pc=10,code,low,total=0,freq[CNUM],f_len,f_pos;
  for( i=0; i<CNUM; i++ ) total+=(freq[i]=1);

  for( i=0; i<CNUM; i++ ) cmap[i]=0;

  if( f_DEC==0 ) {
    f_len = flen(f);
    fwrite( &f_len, 1,sizeof(f_len), g );

    for( f_pos=0; f_pos<f_len; f_pos++ ) cmap[getc(f)]=1;

    fseek( f, 0, SEEK_SET );

    rc.StartEncode(g);

  } else {
    f_len = 0;
    fread( &f_len, 1,sizeof(f_len), f );
    rc.StartDecode(f);
  }

  for( i=0,total=0; i<CNUM; i++ ) total+=( cmap[i]=rc.rc_BProcess(SCALE/2,cmap[i]) );

auto byte_model_ = new PPMD::PPMD(12, 1000, cmap);

byte_model_->Byte_Model::ByteUpdate();

  srand(0xDEADBEEF);
  //ByteModel* PM = new ByteModel( cmap, new Lstm(0, total, 90, 3, 10, 0.05, 2) );
  //ByteModel* PM = new ByteModel( cmap, new Lstm(total, total, 90, 3, 10, 0.05, 2) );
  Model* PM = new Model( cmap, new Lstm(128, total, 90, 3, 10, 0.05, 2) );
  //ByteModel* PM = new ByteModel( cmap, new Lstm(128, total, total, 3, 10, 0.05, 2) );
//  ByteModel* PM = new ByteModel( cmap, new Lstm(128, total, 128, 3, 10, 0.05, 2) );
//      vocab_size, new Lstm(vocab_size, vocab_size, 200, 1, 128, 0.03, 10));

  for( f_pos=0; f_pos<f_len; f_pos++ ) {

//const std::valarray<float>& q = byte_model_->BytePredict();

    for( i=0,total=0; i<CNUM; i++ ) {
      freq[i] = PM->probs_[i]*SCALE;
//      freq[i] = q[i]*SCALE;
      freq[i] += ((freq[i]==0) & cmap[i]);
      total += freq[i];
    }

    if( f_DEC==0 ) {
      c = getc(f);
      for( i=0,low=0; i<c; i++ ) low+=freq[i];
      rc.rc_Process(low,freq[c],total);
    } else {
      code = rc.rc_GetFreq(total);
      for( c=0,low=0; low+freq[c]<=code; c++ ) low+=freq[c];
      rc.rc_Process(low,freq[c],total);
    }

    if( f_DEC==1 ) putc(c,g);

bit_context_=c;
byte_model_->ByteUpdate(bit_context_);

const std::valarray<float>& p = byte_model_->BytePredict();
PM->lstm_->SetInput(p);

    PM->Update( c );
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}
