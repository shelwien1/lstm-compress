
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
  printf(
"LSTM Compressor - Neural network based file compression\n"
"\n"
"Usage: %s <mode> <input> <output> [options]\n"
"\n"
"Required arguments:\n"
"  <mode>    'e' for encode/compress, 'd' for decode/decompress\n"
"  <input>   Input file path\n"
"  <output>  Output file path\n"
"\n"
"Optional parameters:\n"
"  Can be specified by name (name=value) or positionally (in order shown):\n"
"\n"
"  ppmd_order=<n>           PPMD model order (default: 12)\n"
"  ppmd_memory=<n>          PPMD memory in MB (default: 1000)\n"
"  lstm_input_size=<n>      LSTM input layer size (default: 128)\n"
"  lstm_num_cells=<n>       LSTM number of cells (default: 90)\n"
"  lstm_num_layers=<n>      LSTM number of layers (default: 3)\n"
"  lstm_horizon=<n>         LSTM horizon (default: 10)\n"
"  lstm_learning_rate=<f>   LSTM learning rate (default: 0.05)\n"
"  lstm_gradient_clip=<f>   LSTM gradient clip (default: 2.0)\n"
"\n"
"Examples:\n"
"  %s e input.txt output.compressed\n"
"  %s d output.compressed restored.txt\n"
"  %s e input.txt output.compressed ppmd_order=9 lstm_num_layers=1\n"
"  %s e input.txt output.compressed 10 800 100 80 3 10 0.05 2.0\n",
  program_name, program_name, program_name, program_name, program_name);
}

int main( int argc, char** argv ) {

  if( argc<4 || (argc>=2 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) ) {
    print_usage(argv[0]);
    return (argc>=2 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) ? 0 : 1;
  }

  uint f_DEC = (argv[1][0]=='d');
  FILE* f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  FILE* g = fopen(argv[3],"wb"); if( g==0 ) return 3;

  // Initialize parameters with defaults
  int ppmd_order = 12;
  int ppmd_memory = 1000;
  int lstm_input_size = 128;
  int lstm_num_cells = 90;
  int lstm_num_layers = 3;
  int lstm_horizon = 10;
  float lstm_learning_rate = 0.05f;
  float lstm_gradient_clip = 2.0f;

  // Parse optional parameters
  int positional_index = 0;
  for (int i = 4; i < argc; i++) {
    char* arg = argv[i];
    char* equals = strchr(arg, '=');

    if (equals != NULL) {
      // Named parameter: parse key=value
      *equals = '\0';  // Split string at '='
      char* key = arg;
      char* value = equals + 1;

      if (strcmp(key, "ppmd_order") == 0) {
        ppmd_order = atoi(value);
        positional_index = 1;
      } else if (strcmp(key, "ppmd_memory") == 0) {
        ppmd_memory = atoi(value);
        positional_index = 2;
      } else if (strcmp(key, "lstm_input_size") == 0) {
        lstm_input_size = atoi(value);
        positional_index = 3;
      } else if (strcmp(key, "lstm_num_cells") == 0) {
        lstm_num_cells = atoi(value);
        positional_index = 4;
      } else if (strcmp(key, "lstm_num_layers") == 0) {
        lstm_num_layers = atoi(value);
        positional_index = 5;
      } else if (strcmp(key, "lstm_horizon") == 0) {
        lstm_horizon = atoi(value);
        positional_index = 6;
      } else if (strcmp(key, "lstm_learning_rate") == 0) {
        lstm_learning_rate = (float)atof(value);
        positional_index = 7;
      } else if (strcmp(key, "lstm_gradient_clip") == 0) {
        lstm_gradient_clip = (float)atof(value);
        positional_index = 8;
      } else {
        fprintf(stderr, "Unknown parameter: %s\n", key);
        print_usage(argv[0]);
        return 1;
      }

      *equals = '=';  // Restore original string
    } else {
      // Positional parameter
      switch (positional_index) {
        case 0: ppmd_order = atoi(arg); break;
        case 1: ppmd_memory = atoi(arg); break;
        case 2: lstm_input_size = atoi(arg); break;
        case 3: lstm_num_cells = atoi(arg); break;
        case 4: lstm_num_layers = atoi(arg); break;
        case 5: lstm_horizon = atoi(arg); break;
        case 6: lstm_learning_rate = (float)atof(arg); break;
        case 7: lstm_gradient_clip = (float)atof(arg); break;
      }
      positional_index++;
    }
  }

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
