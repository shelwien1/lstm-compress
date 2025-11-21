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

template<uint NUM_CELLS, uint HORIZON, uint LEARNING_RATE_X100000, uint UPDATE_LIMIT>
struct NeuronLayer {
  std::vector<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::vector<std::vector<float>> weights_, state_, update_, m_, v_,
      transpose_, norm_;
  uint input_size_, transpose_size_;
  uint input_array_size_;

  void Init(uint input_size, uint offset, uint input_array_size) {
    input_size_ = input_size;
    transpose_size_ = input_size - offset;
    input_array_size_ = input_array_size;
    error_.resize(NUM_CELLS);
    ivar_.resize(HORIZON);
    gamma_.resize(NUM_CELLS);
    for (uint i = 0; i < NUM_CELLS; ++i) gamma_[i] = 1.0;
    gamma_u_.resize(NUM_CELLS);
    gamma_m_.resize(NUM_CELLS);
    gamma_v_.resize(NUM_CELLS);
    beta_.resize(NUM_CELLS);
    beta_u_.resize(NUM_CELLS);
    beta_m_.resize(NUM_CELLS);
    beta_v_.resize(NUM_CELLS);
    weights_.resize(NUM_CELLS);
    for (uint i = 0; i < NUM_CELLS; ++i) {
      weights_[i].resize(input_size);
    }
    state_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      state_[i].resize(NUM_CELLS);
    }
    update_.resize(NUM_CELLS);
    for (uint i = 0; i < NUM_CELLS; ++i) {
      update_[i].resize(input_size);
    }
    m_.resize(NUM_CELLS);
    for (uint i = 0; i < NUM_CELLS; ++i) {
      m_[i].resize(input_size);
    }
    v_.resize(NUM_CELLS);
    for (uint i = 0; i < NUM_CELLS; ++i) {
      v_[i].resize(input_size);
    }
    transpose_.resize(input_size - offset);
    for (uint i = 0; i < input_size - offset; ++i) {
      transpose_[i].resize(NUM_CELLS);
    }
    norm_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      norm_[i].resize(NUM_CELLS);
    }
  }

  void Adam(std::vector<float>* g, std::vector<float>* m, std::vector<float>* v,
            std::vector<float>* w, qword t) {
    constexpr float learning_rate = LEARNING_RATE_X100000 / 100000.0f;
    const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
    float alpha;
    if (t < UPDATE_LIMIT) {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    } else {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
    }
    for (uint i = 0; i < g->size(); ++i) {
      (*m)[i] *= beta1;
      (*m)[i] += (1.0f - beta1) * (*g)[i];
      (*v)[i] *= beta2;
      (*v)[i] += (1.0f - beta2) * (*g)[i] * (*g)[i];
      if (t < UPDATE_LIMIT) {
        (*w)[i] -= alpha * (((*m)[i] / (float)(1.0f - pow(beta1, t))) /
            (sqrt((*v)[i] / (float)(1.0f - pow(beta2, t)) + eps)));
      } else {
        (*w)[i] -= alpha * (((*m)[i] / (float)(1.0f - pow(beta1, UPDATE_LIMIT))) /
            (sqrt((*v)[i] / (float)(1.0f - pow(beta2, UPDATE_LIMIT)) + eps)));
      }
    }
  }

  void ForwardPass(const std::vector<float>& input, uint input_symbol,
                   uint output_size, uint epoch) {
    for (uint i = 0; i < NUM_CELLS; ++i) {
      float f = weights_[i][input_symbol];
      for (uint j = 0; j < input_array_size_; ++j) {
        f += input[j] * weights_[i][output_size + j];
      }
      norm_[epoch][i] = f;
    }
    float sum_sq = 0.0f;
    for (uint i = 0; i < NUM_CELLS; ++i) {
      sum_sq += norm_[epoch][i] * norm_[epoch][i];
    }
    ivar_[epoch] = 1.0f / sqrt((sum_sq / NUM_CELLS) + 1e-5f);
    for (uint i = 0; i < NUM_CELLS; ++i) {
      norm_[epoch][i] *= ivar_[epoch];
      state_[epoch][i] = norm_[epoch][i] * gamma_[i] + beta_[i];
    }
  }

  void BackwardPass(const std::vector<float>& input,
                    uint epoch,
                    uint layer,
                    uint input_symbol,
                    std::vector<float>* hidden_error,
                    uint output_size,
                    uint input_size,
                    std::vector<float>& stored_error,
                    qword update_steps) {
    if (epoch == HORIZON - 1) {
      for (uint i = 0; i < NUM_CELLS; ++i) {
        gamma_u_[i] = 0;
        beta_u_[i] = 0;
        for (uint j = 0; j < update_[i].size(); ++j) {
          update_[i][j] = 0;
        }
        uint offset = output_size + input_size;
        for (uint j = 0; j < transpose_size_; ++j) {
          transpose_[j][i] = weights_[i][j + offset];
        }
      }
    }
    for (uint i = 0; i < NUM_CELLS; ++i) {
      beta_u_[i] += error_[i];
      gamma_u_[i] += error_[i] * norm_[epoch][i];
      error_[i] *= gamma_[i] * ivar_[epoch];
    }
    float sum_err_norm = 0.0f;
    for (uint i = 0; i < NUM_CELLS; ++i) {
      sum_err_norm += error_[i] * norm_[epoch][i];
    }
    for (uint i = 0; i < NUM_CELLS; ++i) {
      error_[i] -= (sum_err_norm / NUM_CELLS) * norm_[epoch][i];
    }
    if (layer > 0) {
      for (uint i = 0; i < NUM_CELLS; ++i) {
        float f = 0;
        for (uint j = 0; j < NUM_CELLS; ++j) {
          f += error_[j] * transpose_[NUM_CELLS + i][j];
        }
        (*hidden_error)[i] += f;
      }
    }
    if (epoch > 0) {
      for (uint i = 0; i < NUM_CELLS; ++i) {
        float f = 0;
        for (uint j = 0; j < NUM_CELLS; ++j) {
          f += error_[j] * transpose_[i][j];
        }
        stored_error[i] += f;
      }
    }
    for (uint i = 0; i < NUM_CELLS; ++i) {
      for (uint j = 0; j < input_array_size_; ++j) {
        update_[i][output_size + j] += error_[i] * input[j];
      }
      update_[i][input_symbol] += error_[i];
    }
    if (epoch == 0) {
      for (uint i = 0; i < NUM_CELLS; ++i) {
        Adam(&update_[i], &m_[i], &v_[i], &weights_[i], update_steps);
      }
      Adam(&gamma_u_, &gamma_m_, &gamma_v_, &gamma_, update_steps);
      Adam(&beta_u_, &beta_m_, &beta_v_, &beta_, update_steps);
    }
  }
};
//--- #include "lstm-layer.hpp"

template<uint NUM_CELLS, uint HORIZON, uint LEARNING_RATE_X100000, uint GRADIENT_CLIP_X10, uint UPDATE_LIMIT>
struct LstmLayer {
  std::vector<float> state_, state_error_, stored_error_;
  std::vector<std::vector<float>> tanh_state_, input_gate_state_, last_state_;
  uint epoch_, input_size_, output_size_;
  qword update_steps_ = 0;
  NeuronLayer<NUM_CELLS, HORIZON, LEARNING_RATE_X100000, UPDATE_LIMIT> forget_gate_, input_node_, output_gate_;

  void Init(uint input_size, uint auxiliary_input_size, uint output_size) {
    state_.resize(NUM_CELLS);
    state_error_.resize(NUM_CELLS);
    stored_error_.resize(NUM_CELLS);
    tanh_state_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      tanh_state_[i].resize(NUM_CELLS);
    }
    input_gate_state_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      input_gate_state_[i].resize(NUM_CELLS);
    }
    last_state_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      last_state_[i].resize(NUM_CELLS);
    }
    epoch_ = 0;
    input_size_ = auxiliary_input_size;
    output_size_ = output_size;
    update_steps_ = 0;

    forget_gate_.Init(input_size, output_size_ + input_size_, input_size - output_size);
    input_node_.Init(input_size, output_size_ + input_size_, input_size - output_size);
    output_gate_.Init(input_size, output_size_ + input_size_, input_size - output_size);

    float val = sqrt(6.0f / float(input_size_ + output_size_));
    float low = -val;
    float range = 2 * val;
    for (uint i = 0; i < NUM_CELLS; ++i) {
      for (uint j = 0; j < forget_gate_.input_size_; ++j) {
        forget_gate_.weights_[i][j] = low + Rand() * range;
        input_node_.weights_[i][j] = low + Rand() * range;
        output_gate_.weights_[i][j] = low + Rand() * range;
      }
      forget_gate_.weights_[i][forget_gate_.input_size_ - 1] = 1;
    }
  }

  void ForwardPass(const std::vector<float>& input, uint input_symbol, std::vector<float>* hidden, uint hidden_start ) {
    last_state_[epoch_] = state_;
    ForwardPass(forget_gate_, input, input_symbol);
    ForwardPass(input_node_, input, input_symbol);
    ForwardPass(output_gate_, input, input_symbol);
    for (uint i = 0; i < NUM_CELLS; ++i) {
      forget_gate_.state_[epoch_][i] = Logistic(forget_gate_.state_[epoch_][i]);
      input_node_.state_[epoch_][i] = tanh(input_node_.state_[epoch_][i]);
      output_gate_.state_[epoch_][i] = Logistic(output_gate_.state_[epoch_][i]);
    }
    for (uint i = 0; i < NUM_CELLS; ++i) {
      input_gate_state_[epoch_][i] = 1.0f - forget_gate_.state_[epoch_][i];
      state_[i] *= forget_gate_.state_[epoch_][i];
      state_[i] += input_node_.state_[epoch_][i] * input_gate_state_[epoch_][i];
      tanh_state_[epoch_][i] = tanh(state_[i]);
      (*hidden)[hidden_start + i] = output_gate_.state_[epoch_][i] * tanh_state_[epoch_][i];
    }
    ++epoch_;
    if (epoch_ == HORIZON) epoch_ = 0;
  }

  void BackwardPass(const std::vector<float>& input, uint epoch, uint layer, uint input_symbol, std::vector<float>* hidden_error) {
    if (epoch == (int)HORIZON - 1) {
      stored_error_ = *hidden_error;
      for (uint i = 0; i < NUM_CELLS; ++i) {
        state_error_[i] = 0;
      }
    } else {
      for (uint i = 0; i < NUM_CELLS; ++i) {
        stored_error_[i] += (*hidden_error)[i];
      }
    }

    for (uint i = 0; i < NUM_CELLS; ++i) {
      output_gate_.error_[i] = tanh_state_[epoch][i] * stored_error_[i] *
          output_gate_.state_[epoch][i] * (1.0f - output_gate_.state_[epoch][i]);
      state_error_[i] += stored_error_[i] * output_gate_.state_[epoch][i] * (1.0f -
          (tanh_state_[epoch][i] * tanh_state_[epoch][i]));
      input_node_.error_[i] = state_error_[i] * input_gate_state_[epoch][i] * (1.0f -
          (input_node_.state_[epoch][i] * input_node_.state_[epoch][i]));
      forget_gate_.error_[i] = (last_state_[epoch][i] - input_node_.state_[epoch][i]) *
          state_error_[i] * forget_gate_.state_[epoch][i] * input_gate_state_[epoch][i];
    }

    for (uint i = 0; i < hidden_error->size(); ++i) {
      (*hidden_error)[i] = 0;
    }
    if (epoch > 0) {
      for (uint i = 0; i < NUM_CELLS; ++i) {
        state_error_[i] *= forget_gate_.state_[epoch][i];
        stored_error_[i] = 0;
      }
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

  std::vector<std::vector<std::vector<float>>*> Weights() {
    std::vector<std::vector<std::vector<float>>*> weights;
    weights.push_back(&forget_gate_.weights_);
    weights.push_back(&input_node_.weights_);
    weights.push_back(&output_gate_.weights_);
    return weights;
  }

  void ClipGradients(std::vector<float>* arr) {
    constexpr float gradient_clip = GRADIENT_CLIP_X10 / 10.0f;
    for (uint i = 0; i < arr->size(); ++i) {
      if ((*arr)[i] < -gradient_clip) (*arr)[i] = -gradient_clip;
      else if ((*arr)[i] > gradient_clip) (*arr)[i] = gradient_clip;
    }
  }

  void ForwardPass(NeuronLayer<NUM_CELLS, HORIZON, LEARNING_RATE_X100000, UPDATE_LIMIT>& neurons,
                   const std::vector<float>& input, uint input_symbol) {
    neurons.ForwardPass(input, input_symbol, output_size_, epoch_);
  }

  void BackwardPass(NeuronLayer<NUM_CELLS, HORIZON, LEARNING_RATE_X100000, UPDATE_LIMIT>& neurons,
                    const std::vector<float>&input,
                    uint epoch, uint layer, uint input_symbol,
                    std::vector<float>* hidden_error) {
    neurons.BackwardPass(input, epoch, layer, input_symbol, hidden_error,
                         output_size_, input_size_,
                         stored_error_, update_steps_);
  }
};
//--- #include "lstm.hpp"

template<uint NUM_CELLS, uint NUM_LAYERS, uint HORIZON, uint LEARNING_RATE_X100000, uint GRADIENT_CLIP_X10, uint UPDATE_LIMIT>
struct Lstm {
  std::vector<LstmLayer<NUM_CELLS, HORIZON, LEARNING_RATE_X100000, GRADIENT_CLIP_X10, UPDATE_LIMIT>> layers_;
  std::vector<uint8_t> input_history_;
  std::vector<float> hidden_, hidden_error_;
  std::vector<std::vector<std::vector<float>>> layer_input_, output_layer_;
  std::vector<std::vector<float>> output_;
  uint epoch_, input_size_, output_size_;
  uint last_input_ = -1;
  uint hidden_size_, hidden_error_size_;
  uint layer_input_size_0_, layer_input_size_rest_;

  NOINLINE
  void Init(uint input_size, uint output_size) {
    hidden_size_ = NUM_CELLS * NUM_LAYERS + 1;
    hidden_error_size_ = NUM_CELLS;
    layer_input_size_0_ = 1 + NUM_CELLS + input_size;
    layer_input_size_rest_ = input_size + 1 + NUM_CELLS * 2;
    input_history_.resize(HORIZON);
    hidden_.resize(NUM_CELLS * NUM_LAYERS + 1);
    hidden_error_.resize(NUM_CELLS);
    layer_input_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      layer_input_[i].resize(NUM_LAYERS);
      for (uint j = 0; j < NUM_LAYERS; ++j) {
        layer_input_[i][j].resize(input_size + 1 + NUM_CELLS * 2);
      }
    }
    output_layer_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      output_layer_[i].resize(output_size);
      for (uint j = 0; j < output_size; ++j) {
        output_layer_[i][j].resize(NUM_CELLS * NUM_LAYERS + 1);
      }
    }
    output_.resize(HORIZON);
    for (uint i = 0; i < HORIZON; ++i) {
      output_[i].resize(output_size);
      for (uint j = 0; j < output_size; ++j) {
        output_[i][j] = 1.0f / output_size;
      }
    }
    epoch_ = 0;
    input_size_ = input_size;
    output_size_ = output_size;
    last_input_ = -1;

    hidden_[hidden_size_ - 1] = 1;
    for (uint epoch = 0; epoch < HORIZON; ++epoch) {
      layer_input_[epoch][0].resize(layer_input_size_0_);
      layer_input_[epoch][0][layer_input_size_0_ - 1] = 1;
      for (uint i = 1; i < NUM_LAYERS; ++i) {
        layer_input_[epoch][i][layer_input_size_rest_ - 1] = 1;
      }
    }
    layers_.resize(NUM_LAYERS);
    layers_[0].Init(layer_input_size_0_ + output_size, input_size, output_size);
    for (uint i = 1; i < NUM_LAYERS; ++i) {
      layers_[i].Init(layer_input_size_rest_ + output_size, input_size, output_size);
    }
  }

  NOINLINE
  void SetInput(const std::vector<float>& input) {
    for (uint i = 0; i < NUM_LAYERS; ++i) {
      for (uint j = 0; j < input_size_; ++j) {
        layer_input_[epoch_][i][j] = input[j];
      }
    }
  }

  NOINLINE
  std::vector<float>& Perceive(uint input) {
    constexpr float learning_rate = LEARNING_RATE_X100000 / 100000.0f;
    uint last_epoch = epoch_ - 1;
    if (last_epoch == -1) last_epoch = HORIZON - 1;
    uint old_input = input_history_[last_epoch];
    input_history_[last_epoch] = input;
    if (epoch_ == 0) {
      for (uint epoch = HORIZON - 1; epoch!=-1; --epoch) {
        for (uint layer = NUM_LAYERS - 1; layer!=-1; --layer) {
          uint offset = layer * NUM_CELLS;
          for (uint i = 0; i < output_size_; ++i) {
            float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
            for (uint j = 0; j < hidden_error_size_; ++j) {
              hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
            }
          }
          uint prev_epoch = epoch - 1;
          if (prev_epoch == -1) prev_epoch = HORIZON - 1;
          uint input_symbol = input_history_[prev_epoch];
          if (epoch == 0) input_symbol = old_input;
          layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer, input_symbol, &hidden_error_);
        }
      }
    }

    for (uint i = 0; i < output_size_; ++i) {
      float error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
      output_layer_[epoch_][i] = output_layer_[last_epoch][i];
      for (uint j = 0; j < hidden_size_; ++j) {
        output_layer_[epoch_][i][j] -= learning_rate * error * hidden_[j];
      }
    }
    return Predict(input);
  }

  NOINLINE
  std::vector<float>& Predict(uint input) {
    for (uint i = 0; i < NUM_LAYERS; ++i) {
      for (uint j = 0; j < NUM_CELLS; ++j) {
        layer_input_[epoch_][i][input_size_ + j] = hidden_[i * NUM_CELLS + j];
      }
      layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
          NUM_CELLS);
      if (i < NUM_LAYERS - 1) {
        for (uint j = 0; j < NUM_CELLS; ++j) {
          layer_input_[epoch_][i + 1][NUM_CELLS + input_size_ + j] = hidden_[i * NUM_CELLS + j];
        }
      }
    }
    float sum_exp = 0.0f;
    for (uint i = 0; i < output_size_; ++i) {
      float sum = 0;
      for (uint j = 0; j < hidden_size_; ++j) {
        sum += hidden_[j] * output_layer_[epoch_][i][j];
      }
      output_[epoch_][i] = exp(sum);
      sum_exp += output_[epoch_][i];
    }
    for (uint i = 0; i < output_size_; ++i) {
      output_[epoch_][i] /= sum_exp;
    }
    uint epoch = epoch_;
    ++epoch_;
    if (epoch_ == HORIZON) epoch_ = 0;
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
    const std::vector<float>& output = lstm_->Perceive(byte_map_[sym]);
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

using LstmInstance = Lstm<LSTM_NUM_CELLS, LSTM_NUM_LAYERS, LSTM_HORIZON,
                           LSTM_LEARNING_RATE_X100000, LSTM_GRADIENT_CLIP_X10, UPDATE_LIMIT_DEFAULT>;

LstmInstance* lstm = nullptr;

ALIGN(64) Rangecoder rc;

UnifiedModel<LstmInstance> M;


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
  lstm = new LstmInstance();
  lstm->Init(LSTM_INPUT_SIZE, n_chars);
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
    //M.Make_Packed(M.ppmd_probs_); std::vector<float> packed_input(M.packedprobs, M.packedprobs + n_chars);
    std::vector<float> packed_input(M.ppmd_probs_, M.ppmd_probs_ + LSTM_INPUT_SIZE);
    M.lstm_->SetInput(packed_input);
    M.UpdateLSTM(c);

/*if( ftell(rc.f)>(1<<20) ) break;*/
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}
