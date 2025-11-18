


// C library headers
#include <stdlib.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <math.h>

// C++ library headers
// #include <vector>  // No longer needed - replaced with C arrays

//#define INC_FLEN
//#include "common.inc"
typedef unsigned short word;
typedef unsigned int   uint;
typedef unsigned char  byte;
typedef unsigned long long qword;
typedef signed long long sqword;

#ifdef __GNUC__
 #define INLINE   __attribute__((always_inline)) 
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

uint flen( FILE* f ) {
  fseek( f, 0, SEEK_END );
  uint len = ftell(f);
  fseek( f, 0, SEEK_SET );
  return len;
}

#include "sh_v2f.inc"
#include "ppmd.hpp"

//--- #include "sigmoid.hpp"

class Sigmoid {
 public:
  void Init(int logit_size) {
    int i;
    logit_size_ = logit_size;
    logit_table_ = new float[logit_size_];
    for (i = 0; i < logit_size_; ++i) {
      logit_table_[i] = SlowLogit((i + 0.5f) / logit_size_);
    }
  }

  void Quit() {
    delete[] logit_table_;
  }

  float Logit(float p) const {
    int index = p * logit_size_;
    if (index >= logit_size_) index = logit_size_ - 1;
    else if (index < 0) index = 0;
    return logit_table_[index];
  }

  static float Logistic(float p) {
    return 1 / (1 + exp(-p));
  }

  static float FastLogistic(float p) {
    return (0.5f * (p / (1.0f + fabsf(p)) + 1.0f));
  }

 private:
  float SlowLogit(float p) {
    return log(p / (1 - p));
  }

  int logit_size_;
  float* logit_table_;
};
//--- #include "neuron-layer.hpp"

struct NeuronLayer {
  void Init(uint input_size, uint num_cells, int horizon,
    int offset) {
    uint i;
    int j;
    num_cells_ = num_cells;
    horizon_ = horizon;
    input_size_ = input_size;
    transpose_size_ = input_size - offset;

    error_ = new float[num_cells]();
    ivar_ = new float[horizon]();
    gamma_ = new float[num_cells]();
    for (i = 0; i < num_cells; ++i) gamma_[i] = 1.0;
    gamma_u_ = new float[num_cells]();
    gamma_m_ = new float[num_cells]();
    gamma_v_ = new float[num_cells]();
    beta_ = new float[num_cells]();
    beta_u_ = new float[num_cells]();
    beta_m_ = new float[num_cells]();
    beta_v_ = new float[num_cells]();

    weights_ = new float*[num_cells];
    for (i = 0; i < num_cells; ++i) {
      weights_[i] = new float[input_size]();  // () initializes to zero
    }
    state_ = new float*[horizon];
    for (j = 0; j < horizon; ++j) {
      state_[j] = new float[num_cells]();
    }
    update_ = new float*[num_cells];
    for (i = 0; i < num_cells; ++i) {
      update_[i] = new float[input_size]();
    }
    m_ = new float*[num_cells];
    for (i = 0; i < num_cells; ++i) {
      m_[i] = new float[input_size]();
    }
    v_ = new float*[num_cells];
    for (i = 0; i < num_cells; ++i) {
      v_[i] = new float[input_size]();
    }
    transpose_ = new float*[transpose_size_];
    for (i = 0; i < transpose_size_; ++i) {
      transpose_[i] = new float[num_cells]();
    }
    norm_ = new float*[horizon];
    for (j = 0; j < horizon; ++j) {
      norm_[j] = new float[num_cells]();
    }
  }

  void Quit() {
    uint i;
    delete[] error_;
    delete[] ivar_;
    delete[] gamma_;
    delete[] gamma_u_;
    delete[] gamma_m_;
    delete[] gamma_v_;
    delete[] beta_;
    delete[] beta_u_;
    delete[] beta_m_;
    delete[] beta_v_;

    for (i = 0; i < num_cells_; ++i) delete[] weights_[i];
    delete[] weights_;
    for (i = 0; i < horizon_; ++i) delete[] state_[i];
    delete[] state_;
    for (i = 0; i < num_cells_; ++i) delete[] update_[i];
    delete[] update_;
    for (i = 0; i < num_cells_; ++i) delete[] m_[i];
    delete[] m_;
    for (i = 0; i < num_cells_; ++i) delete[] v_[i];
    delete[] v_;
    for (i = 0; i < transpose_size_; ++i) delete[] transpose_[i];
    delete[] transpose_;
    for (i = 0; i < horizon_; ++i) delete[] norm_[i];
    delete[] norm_;
  }

  uint num_cells_, horizon_, input_size_, transpose_size_;
  float* error_;
  float* ivar_;
  float* gamma_;
  float* gamma_u_;
  float* gamma_m_;
  float* gamma_v_;
  float* beta_;
  float* beta_u_;
  float* beta_m_;
  float* beta_v_;
  float** weights_;
  float** state_;
  float** update_;
  float** m_;
  float** v_;
  float** transpose_;
  float** norm_;
};
//--- #include "lstm-layer.hpp"

template<uint NUM_CELLS, uint HORIZON,
         uint GRADIENT_CLIP_X10, uint LEARNING_RATE_X100000,
         uint UPDATE_LIMIT>
class LstmLayer {
 public:
  static constexpr float gradient_clip_ = GRADIENT_CLIP_X10 / 10.0f;
  static constexpr float learning_rate_ = LEARNING_RATE_X100000 / 100000.0f;

  void Init(uint input_size, uint auxiliary_input_size,
      uint output_size) {
    uint i, j, h;
    float val, low, range;
    num_cells_ = NUM_CELLS;
    epoch_ = 0;
    horizon_ = HORIZON;
    input_size_ = auxiliary_input_size;
    output_size_ = output_size;
    forget_gate_.Init(input_size, NUM_CELLS, HORIZON, output_size_ + input_size_);
    input_node_.Init(input_size, NUM_CELLS, HORIZON, output_size_ + input_size_);
    output_gate_.Init(input_size, NUM_CELLS, HORIZON, output_size_ + input_size_);
    for (i = 0; i < NUM_CELLS; ++i) {
      state_[i] = 0;
      state_error_[i] = 0;
      stored_error_[i] = 0;
    }
    for (h = 0; h < HORIZON; ++h) {
      for (i = 0; i < NUM_CELLS; ++i) {
        tanh_state_[h][i] = 0;
        input_gate_state_[h][i] = 0;
        last_state_[h][i] = 0;
      }
    }
    val = sqrt(6.0f / float(input_size_ + output_size_));
    low = -val;
    range = 2 * val;
    for (i = 0; i < num_cells_; ++i) {
      for (j = 0; j < forget_gate_.input_size_; ++j) {
        forget_gate_.weights_[i][j] = low + Rand() * range;
        input_node_.weights_[i][j] = low + Rand() * range;
        output_gate_.weights_[i][j] = low + Rand() * range;
      }
      forget_gate_.weights_[i][forget_gate_.input_size_ - 1] = 1;
    }
  }

  void Quit() {
    forget_gate_.Quit();
    input_node_.Quit();
    output_gate_.Quit();
  }

  void ForwardPass(const float* input, int input_symbol,
      float* hidden, int hidden_start) {
    uint i;
    // last_state_[epoch_] = state_;
    for (i = 0; i < num_cells_; ++i) {
      last_state_[epoch_][i] = state_[i];
    }
    ForwardPass(forget_gate_, input, input_symbol);
    ForwardPass(input_node_, input, input_symbol);
    ForwardPass(output_gate_, input, input_symbol);
    for (i = 0; i < num_cells_; ++i) {
      forget_gate_.state_[epoch_][i] = Sigmoid::Logistic(
          forget_gate_.state_[epoch_][i]);
      input_node_.state_[epoch_][i] = tanh(input_node_.state_[epoch_][i]);
      output_gate_.state_[epoch_][i] = Sigmoid::Logistic(
          output_gate_.state_[epoch_][i]);
    }
    // input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
    for (i = 0; i < num_cells_; ++i) {
      input_gate_state_[epoch_][i] = 1.0f - forget_gate_.state_[epoch_][i];
    }
    // state_ *= forget_gate_.state_[epoch_];
    for (i = 0; i < num_cells_; ++i) {
      state_[i] *= forget_gate_.state_[epoch_][i];
    }
    // state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
    for (i = 0; i < num_cells_; ++i) {
      state_[i] += input_node_.state_[epoch_][i] * input_gate_state_[epoch_][i];
    }
    // tanh_state_[epoch_] = tanh(state_);
    for (i = 0; i < num_cells_; ++i) {
      tanh_state_[epoch_][i] = tanh(state_[i]);
    }
    // (*hidden)[slice] = output_gate_.state_[epoch_] * tanh_state_[epoch_];
    for (i = 0; i < num_cells_; ++i) {
      hidden[hidden_start + i] = output_gate_.state_[epoch_][i] * tanh_state_[epoch_][i];
    }
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
  }

  void BackwardPass(const float* input, int epoch,
      int layer, int input_symbol, float* hidden_error) {
    uint i;
    if (epoch == (int)horizon_ - 1) {
      // stored_error_ = *hidden_error;
      for (i = 0; i < num_cells_; ++i) {
        stored_error_[i] = hidden_error[i];
      }
      // state_error_ = 0;
      for (i = 0; i < num_cells_; ++i) {
        state_error_[i] = 0;
      }
    } else {
      // stored_error_ += *hidden_error;
      for (i = 0; i < num_cells_; ++i) {
        stored_error_[i] += hidden_error[i];
      }
    }

    // output_gate_.error_ = tanh_state_[epoch] * stored_error_ * output_gate_.state_[epoch] * (1.0f - output_gate_.state_[epoch]);
    for (i = 0; i < num_cells_; ++i) {
      output_gate_.error_[i] = tanh_state_[epoch][i] * stored_error_[i] *
          output_gate_.state_[epoch][i] * (1.0f - output_gate_.state_[epoch][i]);
    }
    // state_error_ += stored_error_ * output_gate_.state_[epoch] * (1.0f - (tanh_state_[epoch] * tanh_state_[epoch]));
    for (i = 0; i < num_cells_; ++i) {
      state_error_[i] += stored_error_[i] * output_gate_.state_[epoch][i] * (1.0f -
          (tanh_state_[epoch][i] * tanh_state_[epoch][i]));
    }
    // input_node_.error_ = state_error_ * input_gate_state_[epoch] * (1.0f - (input_node_.state_[epoch] * input_node_.state_[epoch]));
    for (i = 0; i < num_cells_; ++i) {
      input_node_.error_[i] = state_error_[i] * input_gate_state_[epoch][i] * (1.0f -
          (input_node_.state_[epoch][i] * input_node_.state_[epoch][i]));
    }
    // forget_gate_.error_ = (last_state_[epoch] - input_node_.state_[epoch]) * state_error_ * forget_gate_.state_[epoch] * input_gate_state_[epoch];
    for (i = 0; i < num_cells_; ++i) {
      forget_gate_.error_[i] = (last_state_[epoch][i] - input_node_.state_[epoch][i]) *
          state_error_[i] * forget_gate_.state_[epoch][i] * input_gate_state_[epoch][i];
    }

    // *hidden_error = 0;
    for (i = 0; i < num_cells_; ++i) {
      hidden_error[i] = 0;
    }
    if (epoch > 0) {
      // state_error_ *= forget_gate_.state_[epoch];
      for (i = 0; i < num_cells_; ++i) {
        state_error_[i] *= forget_gate_.state_[epoch][i];
      }
      // stored_error_ = 0;
      for (i = 0; i < num_cells_; ++i) {
        stored_error_[i] = 0;
      }
    } else {
      if (update_steps_ < UPDATE_LIMIT) {
        ++update_steps_;
      }
    }

    BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
    BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
    BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

    ClipGradients(state_error_);
    ClipGradients(stored_error_);
    ClipGradients(hidden_error);
  }

  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

 private:
  float state_[NUM_CELLS];
  float state_error_[NUM_CELLS];
  float stored_error_[NUM_CELLS];
  float tanh_state_[HORIZON][NUM_CELLS];
  float input_gate_state_[HORIZON][NUM_CELLS];
  float last_state_[HORIZON][NUM_CELLS];
  uint num_cells_, epoch_, horizon_, input_size_, output_size_;
  qword update_steps_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;

// ============================================================================
// Adam optimizer (helper function template)
// ============================================================================

  template<uint UPD_LIMIT>
  static void Adam(float* g, float* m, float* v, float* w, uint size, float learning_rate, float t) {
    const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
    float alpha;
    uint i;
    if (t < UPD_LIMIT) {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    } else {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * UPD_LIMIT + 1.0f);
    }
    // m *= beta1;
    for (i = 0; i < size; ++i) {
      m[i] *= beta1;
    }
    // m += (1.0f - beta1) * g;
    for (i = 0; i < size; ++i) {
      m[i] += (1.0f - beta1) * g[i];
    }
    // v *= beta2;
    for (i = 0; i < size; ++i) {
      v[i] *= beta2;
    }
    // v += (1.0f - beta2) * g * g;
    for (i = 0; i < size; ++i) {
      v[i] += (1.0f - beta2) * g[i] * g[i];
    }
    if (t < UPD_LIMIT) {
      // w -= alpha * ((m / (float)(1.0f - pow(beta1, t))) / (sqrt(v / (float)(1.0f - pow(beta2, t)) + eps)));
      for (i = 0; i < size; ++i) {
        w[i] -= alpha * ((m[i] / (float)(1.0f - pow(beta1, t))) /
            (sqrt(v[i] / (float)(1.0f - pow(beta2, t)) + eps)));
      }
    } else {
      // w -= alpha * ((m / (float)(1.0f - pow(beta1, UPD_LIMIT))) / (sqrt(v / (float)(1.0f - pow(beta2, UPD_LIMIT)) + eps)));
      for (i = 0; i < size; ++i) {
        w[i] -= alpha * ((m[i] / (float)(1.0f - pow(beta1, UPD_LIMIT))) /
            (sqrt(v[i] / (float)(1.0f - pow(beta2, UPD_LIMIT)) + eps)));
      }
    }
  }

  void ClipGradients(float* arr) {
    uint i;
    for (i = 0; i < num_cells_; ++i) {
      if (arr[i] < -gradient_clip_) arr[i] = -gradient_clip_;
      else if (arr[i] > gradient_clip_) arr[i] = gradient_clip_;
    }
  }

  void ForwardPass(NeuronLayer& neurons, const float* input,
      int input_symbol) {
    uint i, j;
    float f, sum;
    for (i = 0; i < num_cells_; ++i) {
      f = neurons.weights_[i][input_symbol];
      for (j = 0; j < input_size_; ++j) {
        f += input[j] * neurons.weights_[i][output_size_ + j];
      }
      neurons.norm_[epoch_][i] = f;
    }
    // neurons.ivar_[epoch_] = 1.0f / sqrt(((neurons.norm_[epoch_] * neurons.norm_[epoch_]).sum() / num_cells_) + 1e-5f);
    sum = 0;
    for (i = 0; i < num_cells_; ++i) {
      sum += neurons.norm_[epoch_][i] * neurons.norm_[epoch_][i];
    }
    neurons.ivar_[epoch_] = 1.0f / sqrt((sum / num_cells_) + 1e-5f);
    // neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
    for (i = 0; i < num_cells_; ++i) {
      neurons.norm_[epoch_][i] *= neurons.ivar_[epoch_];
    }
    // neurons.state_[epoch_] = neurons.norm_[epoch_] * neurons.gamma_ + neurons.beta_;
    for (i = 0; i < num_cells_; ++i) {
      neurons.state_[epoch_][i] = neurons.norm_[epoch_][i] * neurons.gamma_[i] +
          neurons.beta_[i];
    }
  }

  void BackwardPass(NeuronLayer& neurons, const float* input,
      int epoch, int layer, int input_symbol,
      float* hidden_error) {
    uint i, j;
    int offset;
    float sum, f;
    if (epoch == (int)horizon_ - 1) {
      // neurons.gamma_u_ = 0;
      for (i = 0; i < neurons.num_cells_; ++i) {
        neurons.gamma_u_[i] = 0;
      }
      // neurons.beta_u_ = 0;
      for (i = 0; i < neurons.num_cells_; ++i) {
        neurons.beta_u_[i] = 0;
      }
      for (i = 0; i < num_cells_; ++i) {
        // neurons.update_[i] = 0;
        for (j = 0; j < neurons.input_size_; ++j) {
          neurons.update_[i][j] = 0;
        }
        offset = output_size_ + input_size_;
        for (j = 0; j < neurons.transpose_size_; ++j) {
          neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
        }
      }
    }
    // neurons.beta_u_ += neurons.error_;
    for (i = 0; i < num_cells_; ++i) {
      neurons.beta_u_[i] += neurons.error_[i];
    }
    // neurons.gamma_u_ += neurons.error_ * neurons.norm_[epoch];
    for (i = 0; i < num_cells_; ++i) {
      neurons.gamma_u_[i] += neurons.error_[i] * neurons.norm_[epoch][i];
    }
    // neurons.error_ *= neurons.gamma_ * neurons.ivar_[epoch];
    for (i = 0; i < num_cells_; ++i) {
      neurons.error_[i] *= neurons.gamma_[i] * neurons.ivar_[epoch];
    }
    // neurons.error_ -= ((neurons.error_ * neurons.norm_[epoch]).sum() / num_cells_) * neurons.norm_[epoch];
    sum = 0;
    for (i = 0; i < num_cells_; ++i) {
      sum += neurons.error_[i] * neurons.norm_[epoch][i];
    }
    for (i = 0; i < num_cells_; ++i) {
      neurons.error_[i] -= (sum / num_cells_) * neurons.norm_[epoch][i];
    }
    if (layer > 0) {
      for (i = 0; i < num_cells_; ++i) {
        f = 0;
        for (j = 0; j < num_cells_; ++j) {
          f += neurons.error_[j] * neurons.transpose_[num_cells_ + i][j];
        }
        hidden_error[i] += f;
      }
    }
    if (epoch > 0) {
      for (i = 0; i < num_cells_; ++i) {
        f = 0;
        for (j = 0; j < num_cells_; ++j) {
          f += neurons.error_[j] * neurons.transpose_[i][j];
        }
        stored_error_[i] += f;
      }
    }
    // neurons.update_[i][slice] += neurons.error_[i] * input;
    for (i = 0; i < num_cells_; ++i) {
      for (j = 0; j < input_size_; ++j) {
        neurons.update_[i][output_size_ + j] += neurons.error_[i] * input[j];
      }
      neurons.update_[i][input_symbol] += neurons.error_[i];
    }
    if (epoch == 0) {
      for (i = 0; i < num_cells_; ++i) {
        Adam<UPDATE_LIMIT>(neurons.update_[i], neurons.m_[i], neurons.v_[i],
            neurons.weights_[i], neurons.input_size_, learning_rate_, update_steps_);
      }
      Adam<UPDATE_LIMIT>(neurons.gamma_u_, neurons.gamma_m_, neurons.gamma_v_,
          neurons.gamma_, neurons.num_cells_, learning_rate_, update_steps_);
      Adam<UPDATE_LIMIT>(neurons.beta_u_, neurons.beta_m_, neurons.beta_v_,
          neurons.beta_, neurons.num_cells_, learning_rate_, update_steps_);
    }
  }
};
//--- #include "lstm.hpp"

template<uint INPUT_SIZE, uint NUM_CELLS, uint NUM_LAYERS,
         uint HORIZON, uint GRADIENT_CLIP_X10,
         uint LEARNING_RATE_X100000, uint UPDATE_LIMIT>
class Lstm {
 public:
  using LstmLayerType = LstmLayer<NUM_CELLS, HORIZON, GRADIENT_CLIP_X10,
                                   LEARNING_RATE_X100000, UPDATE_LIMIT>;
  static constexpr float learning_rate_ = LEARNING_RATE_X100000 / 100000.0f;

  NOINLINE
  void Init(uint output_size) {
    int h, epoch;
    uint l, i, layer0_size, input_size_for_layer;
    num_cells_ = NUM_CELLS;
    epoch_ = 0;
    horizon_ = HORIZON;
    input_size_ = INPUT_SIZE;
    output_size_ = output_size;
    layer_input_size_ = INPUT_SIZE + 1 + NUM_CELLS * 2;
    output_layer_size_ = NUM_CELLS * NUM_LAYERS + 1;

    layer_input_ = new float**[HORIZON];
    for (h = 0; h < HORIZON; ++h) {
      layer_input_[h] = new float*[NUM_LAYERS];
      for (l = 0; l < NUM_LAYERS; ++l) {
        layer_input_[h][l] = new float[layer_input_size_]();  // () initializes to zero
      }
    }
    output_layer_ = new float**[HORIZON];
    for (h = 0; h < HORIZON; ++h) {
      output_layer_[h] = new float*[output_size];
      for (i = 0; i < output_size; ++i) {
        output_layer_[h][i] = new float[output_layer_size_]();
      }
    }
    output_ = new float*[HORIZON];
    for (h = 0; h < HORIZON; ++h) {
      output_[h] = new float[output_size];
      for (i = 0; i < output_size; ++i) {
        output_[h][i] = 1.0 / output_size;
      }
    }
    for (i = 0; i < NUM_CELLS * NUM_LAYERS + 1; ++i) {
      hidden_[i] = 0;
    }
    hidden_[NUM_CELLS * NUM_LAYERS] = 1;
    for (i = 0; i < NUM_CELLS; ++i) {
      hidden_error_[i] = 0;
    }
    for (epoch = 0; epoch < HORIZON; ++epoch) {
      input_history_[epoch] = 0;
      // Note: layer 0 uses smaller size but we allocated max size for all
      for (i = 0; i < NUM_LAYERS; ++i) {
        layer_input_[epoch][i][layer_input_size_ - 1] = 1;
      }
    }
    // layer_input_[0][0] size is (1 + NUM_CELLS + INPUT_SIZE) = (INPUT_SIZE + 1 + NUM_CELLS)
    layer0_size = 1 + NUM_CELLS + INPUT_SIZE;
    for (i = 0; i < NUM_LAYERS; ++i) {
      input_size_for_layer = (i == 0) ? layer0_size : layer_input_size_;
      layers_[i].Init(input_size_for_layer + output_size, INPUT_SIZE, output_size);
    }
  }

  void Quit() {
    uint i, l;
    int h;
    // Clean up layers first
    for (i = 0; i < NUM_LAYERS; ++i) {
      layers_[i].Quit();
    }

    for (h = 0; h < HORIZON; ++h) {
      for (l = 0; l < NUM_LAYERS; ++l) {
        delete[] layer_input_[h][l];
      }
      delete[] layer_input_[h];
    }
    delete[] layer_input_;

    for (h = 0; h < HORIZON; ++h) {
      for (i = 0; i < output_size_; ++i) {
        delete[] output_layer_[h][i];
      }
      delete[] output_layer_[h];
    }
    delete[] output_layer_;

    for (h = 0; h < HORIZON; ++h) {
      delete[] output_[h];
    }
    delete[] output_;
  }

  NOINLINE
  void SetInput(const float* input) {
    uint i, j;
    for (i = 0; i < NUM_LAYERS; ++i) {
      for (j = 0; j < input_size_; ++j) {
        layer_input_[epoch_][i][j] = input[j];
      }
    }
  }

  NOINLINE
  float* Perceive(uint input) {
    int last_epoch, old_input, epoch, layer, offset, prev_epoch, input_symbol;
    uint i, j;
    float error;
    last_epoch = epoch_ - 1;
    if (last_epoch == -1) last_epoch = horizon_ - 1;
    old_input = input_history_[last_epoch];
    input_history_[last_epoch] = input;
    if (epoch_ == 0) {
      for (epoch = horizon_ - 1; epoch >= 0; --epoch) {
        for (layer = NUM_LAYERS - 1; layer >= 0; --layer) {
          offset = layer * num_cells_;
          for (i = 0; i < output_size_; ++i) {
            error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
            for (j = 0; j < NUM_CELLS; ++j) {
              hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
            }
          }
          prev_epoch = epoch - 1;
          if (prev_epoch == -1) prev_epoch = horizon_ - 1;
          input_symbol = input_history_[prev_epoch];
          if (epoch == 0) input_symbol = old_input;
          layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer,
              input_symbol, hidden_error_);
        }
      }
    }

    for (i = 0; i < output_size_; ++i) {
      error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
      // output_layer_[epoch_][i] = output_layer_[last_epoch][i];
      for (j = 0; j < output_layer_size_; ++j) {
        output_layer_[epoch_][i][j] = output_layer_[last_epoch][i][j];
      }
      // output_layer_[epoch_][i] -= learning_rate_ * error * hidden_;
      for (j = 0; j < output_layer_size_; ++j) {
        output_layer_[epoch_][i][j] -= learning_rate_ * error * hidden_[j];
      }
    }
    return Predict(input);
  }

  NOINLINE
  float* Predict(uint input) {
    uint i, j, hidden_offset, dest_offset;
    int epoch;
    float sum;
    for (i = 0; i < NUM_LAYERS; ++i) {
      hidden_offset = i * num_cells_;
      for (j = 0; j < num_cells_; ++j) {
        layer_input_[epoch_][i][input_size_ + j] = hidden_[hidden_offset + j];
      }
      layers_[i].ForwardPass(layer_input_[epoch_][i], input, hidden_, i *
          num_cells_);
      if (i < NUM_LAYERS - 1) {
        dest_offset = num_cells_ + input_size_;
        for (j = 0; j < num_cells_; ++j) {
          layer_input_[epoch_][i + 1][dest_offset + j] = hidden_[hidden_offset + j];
        }
      }
    }
    for (i = 0; i < output_size_; ++i) {
      sum = 0;
      for (j = 0; j < NUM_CELLS * NUM_LAYERS + 1; ++j) {
        sum += hidden_[j] * output_layer_[epoch_][i][j];
      }
      output_[epoch_][i] = exp(sum);
    }
    // output_[epoch_] /= output_[epoch_].sum();
    sum = 0;
    for (i = 0; i < output_size_; ++i) {
      sum += output_[epoch_][i];
    }
    for (i = 0; i < output_size_; ++i) {
      output_[epoch_][i] /= sum;
    }
    epoch = epoch_;
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
    last_input_ = input;
    return output_[epoch];
  }

 private:
  LstmLayerType layers_[NUM_LAYERS];
  uint8_t input_history_[HORIZON];
  float hidden_[NUM_CELLS * NUM_LAYERS + 1];
  float hidden_error_[NUM_CELLS];
  float*** layer_input_;
  float*** output_layer_;
  float** output_;
  uint num_cells_, epoch_, horizon_, input_size_, output_size_;
  uint layer_input_size_, output_layer_size_;
  int last_input_ = -1;
};
//--- #include "byte-model.hpp"

class Byte_Model {
 public:
  virtual void Quit() {}

  void Init(char* vocab) {
    int i;
    ex = 0;
    top_ = 255;
    mid_ = 0;
    bot_ = 0;
    vocab_ = vocab;
    outputs_[0] = 0.5;
    for (i = 0; i < 256; ++i) {
      probs_[i] = 1.0 / 256;
    }
  }

  const float* Predict() const {return outputs_;}
  unsigned int NumOutputs() {return 1;}

  float* Predict() {
    int mid, i;
    float num, denom, max_prob_val;
    mid = bot_ + ((top_ - bot_) / 2);
    num = 0.0f;
    for (i = mid + 1; i <= top_; ++i) {
      num += probs_[i];
    }
    denom = num;
    for (i = bot_; i <= mid; ++i) {
      denom += probs_[i];
    }
    ex = bot_;
    max_prob_val = probs_[bot_];
    for (i = bot_ + 1; i <= top_; i++) {
      if (probs_[i] > max_prob_val) {
        max_prob_val = probs_[i];
        ex = i;
      }
    }
    if (denom == 0) outputs_[0] = 0.5;
    else outputs_[0] = num / denom;
    return outputs_;
  }

  void Perceive(int bit) {
    mid_ = bot_ + ((top_ - bot_) / 2);
    if (bit) {
      bot_ = mid_ + 1;
    } else {
      top_ = mid_;
    }
  }

  const float* BytePredict() {
    return probs_;
  }

  void ByteUpdate() {
    int i;
    top_ = 255;
    bot_ = 0;
    for (i = 0; i < 256; ++i) {
      if (!vocab_[i]) probs_[i] = 0;
    }
  }

  int ex;

 protected:
  mutable float outputs_[1];
  int top_, mid_, bot_;
  char* vocab_;
  float probs_[256];
};

//--- #include "ppmd-model.hpp"

class PPMD : public Byte_Model {
 public:

  NOINLINE
  void Init(int order, int memory, char* vocab) {
    Byte_Model::Init(vocab);
    ppmd_model_ = new ppmd_Model();
    ppmd_model_->Init(order,memory,1,0);
  }

  void Quit() {
    delete ppmd_model_;
  }

  NOINLINE
  void ByteUpdate(unsigned int byte) {
    int i;
    float sum;
    ppmd_model_->ppmd_UpdateByte( byte&0xFF );
    ppmd_model_->ppmd_PrepareByte();
    for (i = 0; i < 256; ++i) {
      probs_[i] = ppmd_model_->sqp[i];
      if (probs_[i] < 1) probs_[i] = 1;
    }
    Byte_Model::ByteUpdate();
    // probs_ /= probs_.sum();
    sum = 0;
    for (i = 0; i < 256; ++i) {
      sum += probs_[i];
    }
    for (i = 0; i < 256; ++i) {
      probs_[i] /= sum;
    }
  }

 private:
  ppmd_Model* ppmd_model_;
};

//--- #include "model.hpp"

template<typename LstmType>
struct Model {
  int byte_map_[256];
  float probs_[256];
  LstmType* lstm_;
  char* vocab_;

  void Init( char* vocab, LstmType* lstm ) {
    int i, offset;
    vocab_ = vocab;
    lstm_ = lstm;
    offset = 0;
    for( i = 0; i < 256; i++ ) {
      byte_map_[i] = offset;
      if (vocab_[i]) ++offset;
      probs_[i]=1.0/256;
    }
  }

  void Update( int sym ) {
    const float* output;
    int i, offset;
    output = lstm_->Perceive( byte_map_[sym] );
    offset = 0;
    for( i = 0; i < 256; i++ ) {
      if( vocab_[i] ) {
        probs_[i] = output[offset];
        offset++;
      } else {
        probs_[i] = 0;
      }
    }
  }

};

//#include <optional>

Rangecoder rc;

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
constexpr uint UPDATE_LIMIT = 3000;

using LstmType = Lstm<LSTM_INPUT_SIZE, LSTM_NUM_CELLS, LSTM_NUM_LAYERS,
                      LSTM_HORIZON, LSTM_GRADIENT_CLIP_X10,
                      LSTM_LEARNING_RATE_X100000, UPDATE_LIMIT>;

int main( int argc, char** argv ) {
  uint f_DEC, i, j, c, pc, code, low, total, freq[CNUM], f_len, f_pos;
  FILE* f;
  FILE* g;
  PPMD* byte_model_;
  LstmType* lstm;
  Model<LstmType>* PM;
  const float* p;

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
      LSTM_GRADIENT_CLIP_X10 / 10.0f, UPDATE_LIMIT
    );
    return 1;
  }

  f_DEC = (argv[1][0]=='d');
  f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  g = fopen(argv[3],"wb"); if( g==0 ) return 3;

  pc = 10;
  total = 0;
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

  byte_model_ = new PPMD();
  byte_model_->Init(PPMD_ORDER, PPMD_MEMORY, cmap);

  byte_model_->Byte_Model::ByteUpdate();

  srand(0xDEADBEEF);
  lstm = new LstmType();
  lstm->Init(total);
  PM = new Model<LstmType>();
  PM->Init(cmap, lstm);

  for( f_pos=0; f_pos<f_len; f_pos++ ) {

//const std::vector<float>& q = byte_model_->BytePredict();

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

byte_model_->ByteUpdate(c);

p = byte_model_->BytePredict();
PM->lstm_->SetInput(p);

    PM->Update( c );

//if( ftell(rc.f)>(1<<20) ) break;
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  // Cleanup
  delete PM;
  lstm->Quit();
  delete lstm;
  byte_model_->Quit();
  delete byte_model_;

  return 0;
}
