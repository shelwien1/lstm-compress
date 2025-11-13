
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

void print_usage(const char* program_name) {
  fprintf(stderr, "LSTM Compressor - Neural network based file compression\n\n");
  fprintf(stderr, "Usage: %s <mode> <input> <output> [options]\n\n", program_name);
  fprintf(stderr, "Required arguments:\n");
  fprintf(stderr, "  <mode>    'e' for encode/compress, 'd' for decode/decompress\n");
  fprintf(stderr, "  <input>   Input file path\n");
  fprintf(stderr, "  <output>  Output file path\n\n");
  fprintf(stderr, "Optional parameters (in order):\n");
  fprintf(stderr, "  [ppmd_order]          PPMD model order (default: 12)\n");
  fprintf(stderr, "  [ppmd_memory]         PPMD memory in MB (default: 1000)\n");
  fprintf(stderr, "  [lstm_input_size]     LSTM input layer size (default: 128)\n");
  fprintf(stderr, "  [lstm_num_cells]      LSTM number of cells (default: 90)\n");
  fprintf(stderr, "  [lstm_num_layers]     LSTM number of layers (default: 3)\n");
  fprintf(stderr, "  [lstm_horizon]        LSTM horizon (default: 10)\n");
  fprintf(stderr, "  [lstm_learning_rate]  LSTM learning rate (default: 0.05)\n");
  fprintf(stderr, "  [lstm_gradient_clip]  LSTM gradient clip (default: 2.0)\n\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  %s e input.txt output.compressed\n", program_name);
  fprintf(stderr, "  %s d output.compressed restored.txt\n", program_name);
  fprintf(stderr, "  %s e input.txt output.compressed 10 800 100 80 3 10 0.05 2.0\n", program_name);
}

int main( int argc, char** argv ) {

  if( argc<4 || (argc>=2 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) ) {
    print_usage(argv[0]);
    return (argc>=2 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) ? 0 : 1;
  }

  uint f_DEC = (argv[1][0]=='d');
  FILE* f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  FILE* g = fopen(argv[3],"wb"); if( g==0 ) return 3;

  // Parse optional parameters with defaults
  int ppmd_order = (argc > 4) ? atoi(argv[4]) : 12;
  int ppmd_memory = (argc > 5) ? atoi(argv[5]) : 1000;
  int lstm_input_size = (argc > 6) ? atoi(argv[6]) : 128;
  int lstm_num_cells = (argc > 7) ? atoi(argv[7]) : 90;
  int lstm_num_layers = (argc > 8) ? atoi(argv[8]) : 3;
  int lstm_horizon = (argc > 9) ? atoi(argv[9]) : 10;
  float lstm_learning_rate = (argc > 10) ? (float)atof(argv[10]) : 0.05f;
  float lstm_gradient_clip = (argc > 11) ? (float)atof(argv[11]) : 2.0f;

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

auto byte_model_ = new PPMD::PPMD(ppmd_order, ppmd_memory, cmap);

byte_model_->Byte_Model::ByteUpdate();

  srand(0xDEADBEEF);
  //ByteModel* PM = new ByteModel( cmap, new Lstm(0, total, 90, 3, 10, 0.05, 2) );
  //ByteModel* PM = new ByteModel( cmap, new Lstm(total, total, 90, 3, 10, 0.05, 2) );
  Model* PM = new Model( cmap, new Lstm(lstm_input_size, total, lstm_num_cells, lstm_num_layers, lstm_horizon, lstm_learning_rate, lstm_gradient_clip) );
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
