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

//#include "common.inc"
typedef unsigned short word;
typedef unsigned int   uint;
typedef unsigned char  byte;
typedef unsigned long long qword;

#ifdef __GNUC__
 #define INLINE   __attribute__((always_inline)) 
 #define NOINLINE __attribute__((noinline))
 #define ALIGN(n) __attribute__((aligned(n)))
 #define restrict __restrict
#else
 #define INLINE   __forceinline
 #define NOINLINE __declspec(noinline)
 #define ALIGN(n) __declspec(align(n))
#endif

#define AlignUp(x,r) ((x)+((r)-1))/(r)*(r)

uint flen( FILE* f ) {
  fseek( f, 0, SEEK_END );
  uint len = ftell(f);
  fseek( f, 0, SEEK_SET );
  return len;
}

#include "sh_v2f.inc"
#include "ppmd.hpp"

//--- #include "neuron-layer.hpp"

struct NeuronLayer {
  std::valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::valarray<std::valarray<float>> weights_, state_, update_, m_, v_,
      transpose_, norm_;
  uint num_cells_, input_size_, transpose_size_;
  uint horizon_;
  uint input_array_size_;

  void Init(uint input_size, uint num_cells, uint horizon, uint offset, uint input_array_size) {
    num_cells_ = num_cells;
    input_size_ = input_size;
    horizon_ = horizon;
    transpose_size_ = input_size - offset;
    input_array_size_ = input_array_size;
    error_.resize(num_cells);
    ivar_.resize(horizon);
    gamma_ = std::valarray<float>(1.0, num_cells);
    gamma_u_.resize(num_cells);
    gamma_m_.resize(num_cells);
    gamma_v_.resize(num_cells);
    beta_.resize(num_cells);
    beta_u_.resize(num_cells);
    beta_m_.resize(num_cells);
    beta_v_.resize(num_cells);
    weights_.resize(num_cells);
    for (uint i = 0; i < num_cells; ++i) {
      weights_[i].resize(input_size);
    }
    state_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      state_[i].resize(num_cells);
    }
    update_.resize(num_cells);
    for (uint i = 0; i < num_cells; ++i) {
      update_[i].resize(input_size);
    }
    m_.resize(num_cells);
    for (uint i = 0; i < num_cells; ++i) {
      m_[i].resize(input_size);
    }
    v_.resize(num_cells);
    for (uint i = 0; i < num_cells; ++i) {
      v_[i].resize(input_size);
    }
    transpose_.resize(input_size - offset);
    for (uint i = 0; i < input_size - offset; ++i) {
      transpose_[i].resize(num_cells);
    }
    norm_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      norm_[i].resize(num_cells);
    }
  }

  void Adam(std::valarray<float>* g, std::valarray<float>* m, std::valarray<float>* v,
            std::valarray<float>* w, float learning_rate, qword t, uint update_limit) {
    const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
    float alpha;
    if (t < update_limit) {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    } else {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * update_limit + 1.0f);
    }
    (*m) *= beta1;
    (*m) += (1.0f - beta1) * (*g);
    (*v) *= beta2;
    (*v) += (1.0f - beta2) * (*g) * (*g);
    if (t < update_limit) {
      (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))) /
          (sqrt((*v) / (float)(1.0f - pow(beta2, t)) + eps)));
    } else {
      (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, update_limit))) /
          (sqrt((*v) / (float)(1.0f - pow(beta2, update_limit)) + eps)));
    }
  }

  void ForwardPass(const std::valarray<float>& input, uint input_symbol,
                   uint num_cells, uint output_size, uint epoch) {
    for (uint i = 0; i < num_cells; ++i) {
      float f = weights_[i][input_symbol];
      for (uint j = 0; j < input_array_size_; ++j) {
        f += input[j] * weights_[i][output_size + j];
      }
      norm_[epoch][i] = f;
    }
    ivar_[epoch] = 1.0f / sqrt(((norm_[epoch] * norm_[epoch]).sum() / num_cells) + 1e-5f);
    norm_[epoch] *= ivar_[epoch];
    state_[epoch] = norm_[epoch] * gamma_ + beta_;
  }

  void BackwardPass(const std::valarray<float>& input,
                    uint epoch,
                    uint layer,
                    uint input_symbol,
                    std::valarray<float>* hidden_error,
                    uint num_cells,
                    uint horizon,
                    uint output_size,
                    uint input_size,
                    std::valarray<float>& stored_error,
                    float learning_rate,
                    qword update_steps,
                    uint update_limit) {
    if (epoch == horizon - 1) {
      gamma_u_ = 0;
      beta_u_ = 0;
      for (uint i = 0; i < num_cells; ++i) {
        update_[i] = 0;
        uint offset = output_size + input_size;
        for (uint j = 0; j < transpose_size_; ++j) {
          transpose_[j][i] = weights_[i][j + offset];
        }
      }
    }
    beta_u_ += error_;
    gamma_u_ += error_ * norm_[epoch];
    error_ *= gamma_ * ivar_[epoch];
    error_ -= ((error_ * norm_[epoch]).sum() / num_cells) * norm_[epoch];
    if (layer > 0) {
      for (uint i = 0; i < num_cells; ++i) {
        float f = 0;
        for (uint j = 0; j < num_cells; ++j) {
          f += error_[j] * transpose_[num_cells + i][j];
        }
        (*hidden_error)[i] += f;
      }
    }
    if (epoch > 0) {
      for (uint i = 0; i < num_cells; ++i) {
        float f = 0;
        for (uint j = 0; j < num_cells; ++j) {
          f += error_[j] * transpose_[i][j];
        }
        stored_error[i] += f;
      }
    }
    for (uint i = 0; i < num_cells; ++i) {
      for (uint j = 0; j < input_array_size_; ++j) {
        update_[i][output_size + j] += error_[i] * input[j];
      }
      update_[i][input_symbol] += error_[i];
    }
    if (epoch == 0) {
      for (uint i = 0; i < num_cells; ++i) {
        Adam(&update_[i], &m_[i], &v_[i], &weights_[i], learning_rate, update_steps, update_limit);
      }
      Adam(&gamma_u_, &gamma_m_, &gamma_v_, &gamma_, learning_rate, update_steps, update_limit);
      Adam(&beta_u_, &beta_m_, &beta_v_, &beta_, learning_rate, update_steps, update_limit);
    }
  }
};
//--- #include "lstm-layer.hpp"

struct LstmLayer {
  std::valarray<float> state_, state_error_, stored_error_;
  std::valarray<std::valarray<float>> tanh_state_, input_gate_state_, last_state_;
  float gradient_clip_, learning_rate_;
  uint num_cells_, epoch_, horizon_, input_size_, output_size_;
  qword update_steps_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;
  uint UPDATE_LIMIT;

  void Init(uint input_size, uint auxiliary_input_size,
            uint output_size, uint num_cells, uint horizon,
            float gradient_clip, float learning_rate, int update_limit) {
    state_.resize(num_cells);
    state_error_.resize(num_cells);
    stored_error_.resize(num_cells);
    tanh_state_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      tanh_state_[i].resize(num_cells);
    }
    input_gate_state_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      input_gate_state_[i].resize(num_cells);
    }
    last_state_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      last_state_[i].resize(num_cells);
    }
    gradient_clip_ = gradient_clip;
    learning_rate_ = learning_rate;
    num_cells_ = num_cells;
    epoch_ = 0;
    horizon_ = horizon;
    input_size_ = auxiliary_input_size;
    output_size_ = output_size;
    UPDATE_LIMIT = update_limit;
    update_steps_ = 0;

    forget_gate_.Init(input_size, num_cells, horizon, output_size_ + input_size_, input_size - output_size_);
    input_node_.Init(input_size, num_cells, horizon, output_size_ + input_size_, input_size - output_size_);
    output_gate_.Init(input_size, num_cells, horizon, output_size_ + input_size_, input_size - output_size_);

    float val = sqrt(6.0f / float(input_size_ + output_size_));
    float low = -val;
    float range = 2 * val;
    for (uint i = 0; i < num_cells_; ++i) {
      for (uint j = 0; j < forget_gate_.input_size_; ++j) {
        forget_gate_.weights_[i][j] = low + Rand() * range;
        input_node_.weights_[i][j] = low + Rand() * range;
        output_gate_.weights_[i][j] = low + Rand() * range;
      }
      forget_gate_.weights_[i][forget_gate_.input_size_ - 1] = 1;
    }
  }

  void ForwardPass(const std::valarray<float>& input, uint input_symbol, std::valarray<float>* hidden, uint hidden_start ) {
    last_state_[epoch_] = state_;
    ForwardPass(forget_gate_, input, input_symbol);
    ForwardPass(input_node_, input, input_symbol);
    ForwardPass(output_gate_, input, input_symbol);
    for (uint i = 0; i < num_cells_; ++i) {
      forget_gate_.state_[epoch_][i] = Logistic(forget_gate_.state_[epoch_][i]);
      input_node_.state_[epoch_][i] = tanh(input_node_.state_[epoch_][i]);
      output_gate_.state_[epoch_][i] = Logistic(output_gate_.state_[epoch_][i]);
    }
    input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
    state_ *= forget_gate_.state_[epoch_];
    state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
    tanh_state_[epoch_] = tanh(state_);
    for (uint i = 0; i < num_cells_; ++i) {
      (*hidden)[hidden_start + i] = output_gate_.state_[epoch_][i] * tanh_state_[epoch_][i];
    }
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
  }

  void BackwardPass(const std::valarray<float>& input, uint epoch, uint layer, uint input_symbol, std::valarray<float>* hidden_error) {
    if (epoch == (int)horizon_ - 1) {
      stored_error_ = *hidden_error;
      state_error_ = 0;
    } else {
      stored_error_ += *hidden_error;
    }

    output_gate_.error_ = tanh_state_[epoch] * stored_error_ *
        output_gate_.state_[epoch] * (1.0f - output_gate_.state_[epoch]);
    state_error_ += stored_error_ * output_gate_.state_[epoch] * (1.0f -
        (tanh_state_[epoch] * tanh_state_[epoch]));
    input_node_.error_ = state_error_ * input_gate_state_[epoch] * (1.0f -
        (input_node_.state_[epoch] * input_node_.state_[epoch]));
    forget_gate_.error_ = (last_state_[epoch] - input_node_.state_[epoch]) *
        state_error_ * forget_gate_.state_[epoch] * input_gate_state_[epoch];

    *hidden_error = 0;
    if (epoch > 0) {
      state_error_ *= forget_gate_.state_[epoch];
      stored_error_ = 0;
    } else {
      if (update_steps_ < UPDATE_LIMIT) ++update_steps_;
    }

    BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
    BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
    BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

    ClipGradients(&state_error_);
    ClipGradients(&stored_error_);
    ClipGradients(hidden_error);
  }

  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  static inline float Logistic(float p) {
    return 1.0f / (1.0f + exp(-p));
  }

  std::vector<std::valarray<std::valarray<float>>*> Weights() {
    std::vector<std::valarray<std::valarray<float>>*> weights;
    weights.push_back(&forget_gate_.weights_);
    weights.push_back(&input_node_.weights_);
    weights.push_back(&output_gate_.weights_);
    return weights;
  }

  void ClipGradients(std::valarray<float>* arr) {
    for (uint i = 0; i < arr->size(); ++i) {
      if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
      else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
    }
  }

  void ForwardPass(NeuronLayer& neurons, const std::valarray<float>& input, uint input_symbol) {
    neurons.ForwardPass(input, input_symbol, num_cells_, output_size_, epoch_);
  }

  void BackwardPass(NeuronLayer& neurons, const std::valarray<float>&input,
      uint epoch, uint layer, uint input_symbol,
      std::valarray<float>* hidden_error) {
    neurons.BackwardPass(input, epoch, layer, input_symbol, hidden_error,
                         num_cells_, horizon_, output_size_, input_size_,
                         stored_error_, learning_rate_, update_steps_,
                         UPDATE_LIMIT);
  }
};
//--- #include "lstm.hpp"

struct Lstm {
  std::vector<LstmLayer> layers_;
  std::vector<uint8_t> input_history_;
  std::valarray<float> hidden_, hidden_error_;
  std::valarray<std::valarray<std::valarray<float>>> layer_input_, output_layer_;
  std::valarray<std::valarray<float>> output_;
  float learning_rate_;
  uint num_cells_, epoch_, horizon_, input_size_, output_size_;
  uint last_input_ = -1;
  uint num_layers_, hidden_size_, hidden_error_size_;
  uint layer_input_size_0_, layer_input_size_rest_;

  NOINLINE
  void Init(uint input_size, uint output_size, uint num_cells, uint num_layers,
            uint horizon, float learning_rate, float gradient_clip, uint update_limit) {
    num_layers_ = num_layers;
    hidden_size_ = num_cells * num_layers + 1;
    hidden_error_size_ = num_cells;
    layer_input_size_0_ = 1 + num_cells + input_size;
    layer_input_size_rest_ = input_size + 1 + num_cells * 2;
    input_history_.resize(horizon);
    hidden_.resize(num_cells * num_layers + 1);
    hidden_error_.resize(num_cells);
    layer_input_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      layer_input_[i].resize(num_layers);
      for (uint j = 0; j < num_layers; ++j) {
        layer_input_[i][j].resize(input_size + 1 + num_cells * 2);
      }
    }
    output_layer_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      output_layer_[i].resize(output_size);
      for (uint j = 0; j < output_size; ++j) {
        output_layer_[i][j].resize(num_cells * num_layers + 1);
      }
    }
    output_.resize(horizon);
    for (uint i = 0; i < horizon; ++i) {
      output_[i] = std::valarray<float>(1.0 / output_size, output_size);
    }
    learning_rate_ = learning_rate;
    num_cells_ = num_cells;
    epoch_ = 0;
    horizon_ = horizon;
    input_size_ = input_size;
    output_size_ = output_size;
    last_input_ = -1;

    hidden_[hidden_size_ - 1] = 1;
    for (uint epoch = 0; epoch < horizon; ++epoch) {
      layer_input_[epoch][0].resize(layer_input_size_0_);
      layer_input_[epoch][0][layer_input_size_0_ - 1] = 1;
      for (uint i = 1; i < num_layers; ++i) {
        layer_input_[epoch][i][layer_input_size_rest_ - 1] = 1;
      }
    }
    layers_.resize(num_layers);
    layers_[0].Init(layer_input_size_0_ + output_size, input_size_, output_size_,
                    num_cells, horizon, gradient_clip, learning_rate, update_limit);
    for (uint i = 1; i < num_layers; ++i) {
      layers_[i].Init(layer_input_size_rest_ + output_size, input_size_, output_size_,
                      num_cells, horizon, gradient_clip, learning_rate, update_limit);
    }
  }

  NOINLINE
  void SetInput(const std::valarray<float>& input) {
    for (uint i = 0; i < num_layers_; ++i) {
      for (uint j = 0; j < input_size_; ++j) {
        layer_input_[epoch_][i][j] = input[j];
      }
    }
  }

  NOINLINE
  std::valarray<float>& Perceive(uint input) {
    uint last_epoch = epoch_ - 1;
    if (last_epoch == -1) last_epoch = horizon_ - 1;
    uint old_input = input_history_[last_epoch];
    input_history_[last_epoch] = input;
    if (epoch_ == 0) {
      for (uint epoch = horizon_ - 1; epoch!=-1; --epoch) {
        for (uint layer = num_layers_ - 1; layer!=-1; --layer) {
          uint offset = layer * num_cells_;
          for (uint i = 0; i < output_size_; ++i) {
            float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
            for (uint j = 0; j < hidden_error_size_; ++j) {
              hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
            }
          }
          uint prev_epoch = epoch - 1;
          if (prev_epoch == -1) prev_epoch = horizon_ - 1;
          uint input_symbol = input_history_[prev_epoch];
          if (epoch == 0) input_symbol = old_input;
          layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer, input_symbol, &hidden_error_);
        }
      }
    }

    for (uint i = 0; i < output_size_; ++i) {
      float error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
      output_layer_[epoch_][i] = output_layer_[last_epoch][i];
      output_layer_[epoch_][i] -= learning_rate_ * error * hidden_;
    }
    return Predict(input);
  }

  NOINLINE
  std::valarray<float>& Predict(uint input) {
    for (uint i = 0; i < num_layers_; ++i) {
      for (uint j = 0; j < num_cells_; ++j) {
        layer_input_[epoch_][i][input_size_ + j] = hidden_[i * num_cells_ + j];
      }
      layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
          num_cells_);
      if (i < num_layers_ - 1) {
        for (uint j = 0; j < num_cells_; ++j) {
          layer_input_[epoch_][i + 1][num_cells_ + input_size_ + j] = hidden_[i * num_cells_ + j];
        }
      }
    }
    for (uint i = 0; i < output_size_; ++i) {
      float sum = 0;
      for (uint j = 0; j < hidden_size_; ++j) {
        sum += hidden_[j] * output_layer_[epoch_][i][j];
      }
      output_[epoch_][i] = exp(sum);
    }
    output_[epoch_] /= output_[epoch_].sum();
    uint epoch = epoch_;
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
    last_input_ = input;
    return output_[epoch];
  }
};
//--- #include "unified-model.hpp"

template<typename LstmType>
struct UnifiedModel {
  ppmd_Model ppmd_model_;
  LstmType* lstm_;
  char* vocab_;
  byte byte_map_[256];
  float ppmd_probs_[256];
  float packedprobs[256];
  float lstm_probs_[256];

  NOINLINE
  void Init(int order, int memory, char* vocab, LstmType* lstm) {
    uint i, offset;
    vocab_ = vocab;
    lstm_ = lstm;

    // Initialize PPMD
    ppmd_model_.Init(order, memory, 1, 0);

    // Initialize byte mapping and probabilities
    offset = 0;
    for (i = 0; i < 256; i++) {
      byte_map_[i] = offset;
      if (vocab_[i]) {
        ++offset;
        lstm_probs_[i] = ppmd_probs_[i] = 1.0 / 256;
      } else {
        lstm_probs_[i] = ppmd_probs_[i] = 0;
      }
    }
    Renorm_probs(ppmd_probs_);
  }

  void Renorm_probs( float* ppmd_probs_ ) {
    uint i;
    float sum = 0;
    for (i = 0; i < 256; ++i) sum += ppmd_probs_[i];
    for (i = 0; i < 256; ++i) ppmd_probs_[i] /= sum;
  }

  NOINLINE
  void Make_Packed( float* ppmd_probs_ ) {
    uint i,j;
    for( i=0,j=0; i<256; i++ ) if( vocab_[i] ) packedprobs[j++]=ppmd_probs_[i];
  }

  NOINLINE
  void UpdatePPMD(uint byte) {
    uint i;
    ppmd_model_.ppmd_UpdateByte(byte & 0xFF);
    ppmd_model_.ppmd_PrepareByte();
    for (i = 0; i < 256; ++i) {
      if (vocab_[i]) {
        ppmd_probs_[i] = ppmd_model_.sqp[i];
        if (ppmd_probs_[i] < 1) ppmd_probs_[i] = 1;
      } else {
        ppmd_probs_[i] = 0;
      }
    }
    Renorm_probs(ppmd_probs_);
  }

  NOINLINE
  void UpdateLSTM(int sym) {
    const std::valarray<float>& output = lstm_->Perceive(byte_map_[sym]);
    uint i, offset;
    offset = 0;
    for (i = 0; i < 256; i++) {
      if (vocab_[i]) {
        lstm_probs_[i] = output[offset];
        offset++;
      } else {
        lstm_probs_[i] = 0;
      }
    }
    //Renorm_probs(lstm_probs_); // already done in Predict
  }

};

static const uint CNUM = 256;

char cmap[CNUM];

// Fixed template parameters (previously command-line configurable)
constexpr int PPMD_ORDER = 9;
constexpr int PPMD_MEMORY = 1000;
constexpr uint LSTM_INPUT_SIZE = 128;
constexpr uint LSTM_NUM_CELLS = 90;
constexpr uint LSTM_NUM_LAYERS = 2;
constexpr uint LSTM_HORIZON = 73;
constexpr uint LSTM_LEARNING_RATE_X100000 = 7200;  // 0.072 * 100000
constexpr uint LSTM_GRADIENT_CLIP_X10 = 20;         // 2.0 * 10
constexpr uint UPDATE_LIMIT_DEFAULT = 3000;

Lstm* lstm = nullptr;

ALIGN(64) Rangecoder rc;

UnifiedModel<Lstm> M;


int main( int argc, char** argv ) {
  uint f_DEC, i, j, c, pc, code, low, total,n_chars, freq[CNUM], f_len, f_pos;
  FILE* f;
  FILE* g;

  printf( "sizeof(UnifiedModel)=%i\n", int(sizeof(M)));

  if( argc < 4 ) {
    printf(
      "LSTM Compressor - Neural network based file compression\n"
      "\n"
      "Usage: %s <mode> <input> <output>\n"
      "\n"
      "Arguments:\n"
      "  <mode>    'c' for compress, 'd' for decompress\n"
      "  <input>   Input file path\n"
      "  <output>  Output file path\n"
      "\n"
      "Fixed parameters:\n"
      "  ppmd_order=%d ppmd_memory=%d lstm_input_size=%u\n"
      "  lstm_num_cells=%u lstm_num_layers=%u lstm_horizon=%u\n"
      "  lstm_learning_rate=%.5f lstm_gradient_clip=%.1f update_limit=%u\n",
      argv[0],
      PPMD_ORDER, PPMD_MEMORY, LSTM_INPUT_SIZE,
      LSTM_NUM_CELLS, LSTM_NUM_LAYERS, LSTM_HORIZON,
      LSTM_LEARNING_RATE_X100000 / 100000.0f,
      LSTM_GRADIENT_CLIP_X10 / 10.0f, UPDATE_LIMIT_DEFAULT
    );
    return 1;
  }

  f_DEC = (argv[1][0]=='d');
  f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  g = fopen(argv[3],"wb"); if( g==0 ) return 3;

  pc = 10;
  n_chars = 0;
  for( i=0; i<CNUM; i++ ) n_chars+=(freq[i]=1);

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

  for( i=0,n_chars=0; i<CNUM; i++ ) n_chars+=( cmap[i]=rc.rc_BProcess(SCALE/2,cmap[i]) );

  srand(0xDEADBEEF);
  lstm = new Lstm();
  lstm->Init(LSTM_INPUT_SIZE, n_chars, LSTM_NUM_CELLS, LSTM_NUM_LAYERS,
             LSTM_HORIZON, LSTM_LEARNING_RATE_X100000 / 100000.0f,
             LSTM_GRADIENT_CLIP_X10 / 10.0f, UPDATE_LIMIT_DEFAULT);
  M.Init(PPMD_ORDER, PPMD_MEMORY, cmap, lstm);

  for( f_pos=0; f_pos<f_len; f_pos++ ) {

    // Mix PPMD and LSTM predictions (adaptive weight)
    float mix_weight = 0.35f /*+ 0.15f * (float)Min<int>(f_pos/3,f_len) / (float)f_len*/;  // 0.4 to 0.7
    for( i=0,total=0; i<CNUM; i++ ) {
      freq[i] = ((1.0f - mix_weight) * M.lstm_probs_[i] + mix_weight * M.ppmd_probs_[i]) * SCALE;
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

    M.UpdatePPMD(c);
    //M.Make_Packed(M.ppmd_probs_); std::valarray<float> packed_input(M.packedprobs, n_chars);
    std::valarray<float> packed_input(M.ppmd_probs_, LSTM_INPUT_SIZE);
    M.lstm_->SetInput(packed_input);
    M.UpdateLSTM(c);

/*if( ftell(rc.f)>(1<<20) ) break;*/
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}
