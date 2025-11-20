// C library headers
#include <stdlib.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <math.h>

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

static constexpr uint ROW_a=4;

//--- #include "neuron-layer.hpp"

template<uint INPUT_SIZE, uint NUM_CELLS, uint HORIZON, uint TRANSPOSE_SIZE>
struct NeuronLayer {
  //static constexpr uint MAX_INPUT_SIZE = 1024;  // Maximum input_size for weights
  static constexpr uint MAX_OUTPUT_SIZE = 256 +0;
  static constexpr uint MAX_INPUT_SIZE = INPUT_SIZE + 1 + NUM_CELLS*2 + MAX_OUTPUT_SIZE;
  static constexpr uint MAX_INPUT_SIZE_a = AlignUp(MAX_INPUT_SIZE ,ROW_a);
  static constexpr uint NUM_CELLS_a = AlignUp(NUM_CELLS,ROW_a);
  static constexpr uint HORIZON_a = AlignUp(HORIZON,ROW_a);

  uint input_size_;
  float error_[NUM_CELLS_a];
  float ivar_[HORIZON_a];
  float gamma_[NUM_CELLS_a];
  float gamma_u_[NUM_CELLS_a];
  float gamma_m_[NUM_CELLS_a];
  float gamma_v_[NUM_CELLS_a];
  float beta_[NUM_CELLS_a];
  float beta_u_[NUM_CELLS_a];
  float beta_m_[NUM_CELLS_a];
  float beta_v_[NUM_CELLS_a];
  float weights_[NUM_CELLS][MAX_INPUT_SIZE_a];
  float update_[NUM_CELLS][MAX_INPUT_SIZE_a];
  float m_[NUM_CELLS][MAX_INPUT_SIZE_a];
  float v_[NUM_CELLS][MAX_INPUT_SIZE_a];
  float norm_[HORIZON * NUM_CELLS_a];
  float transpose_[TRANSPOSE_SIZE * NUM_CELLS_a];
  float state_[HORIZON * NUM_CELLS_a];

  void Init(uint input_size) {
    uint i;
    int j;
//printf( "NeuronLayer @ %I64X\n", this );
    input_size_ = input_size;
    for (i = 0; i < NUM_CELLS; ++i) error_[i] = 0;
    for (i = 0; i < HORIZON; ++i) ivar_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) gamma_[i] = 1.0;
    for (i = 0; i < NUM_CELLS; ++i) gamma_u_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) gamma_m_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) gamma_v_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) beta_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) beta_u_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) beta_m_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) beta_v_[i] = 0;
    for (i = 0; i < NUM_CELLS; ++i) {
      for (j = 0; j < input_size; ++j) {
        weights_[i][j] = 0;
        update_[i][j] = 0;
        m_[i][j] = 0;
        v_[i][j] = 0;
      }
    }
    for (i = 0; i < HORIZON * NUM_CELLS; ++i) state_[i] = 0;
    for (i = 0; i < TRANSPOSE_SIZE * NUM_CELLS; ++i) transpose_[i] = 0;
    for (i = 0; i < HORIZON * NUM_CELLS; ++i) norm_[i] = 0;
  }

};
//--- #include "lstm-layer.hpp"

template<uint INPUT_SIZE, uint NUM_CELLS, uint HORIZON,uint GRADIENT_CLIP_X10, uint LEARNING_RATE_X100000,uint UPDATE_LIMIT>
struct LstmLayer {
  using t_NeuronLayer = NeuronLayer<INPUT_SIZE, NUM_CELLS, HORIZON, 1+NUM_CELLS*2>;
  static constexpr float gradient_clip_ = GRADIENT_CLIP_X10 / 10.0f;
  static constexpr float learning_rate_ = LEARNING_RATE_X100000 / 100000.0f;
  static constexpr uint NUM_CELLS_a = AlignUp(NUM_CELLS,ROW_a);
  static constexpr uint HORIZON_a = AlignUp(HORIZON,ROW_a);

  float state_[NUM_CELLS_a];
  float state_error_[NUM_CELLS_a];
  float stored_error_[NUM_CELLS_a];
  float tanh_state_[HORIZON][NUM_CELLS_a];
  float input_gate_state_[HORIZON][NUM_CELLS_a];
  float last_state_[HORIZON][NUM_CELLS_a];
  uint num_cells_, epoch_, horizon_, output_size_;
  qword update_steps_;
  t_NeuronLayer forget_gate_;
  t_NeuronLayer input_node_;
  t_NeuronLayer output_gate_;

  void Init(uint input_size, uint output_size) {
    uint i, h, j;
    float val, low, range;
//printf( "LstmLayer @ %I64X\n", this );
    update_steps_ = 0;
    num_cells_ = NUM_CELLS;
    epoch_ = 0;
    horizon_ = HORIZON;
    output_size_ = output_size;
    forget_gate_.Init(input_size);
    input_node_.Init(input_size);
    output_gate_.Init(input_size);
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
    val = sqrt(6.0f / float(INPUT_SIZE + output_size_));
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

  void ForwardPass(const float* input, uint input_size, int input_symbol, float* hidden, int hidden_start) {
    uint i;
    for (i = 0; i < num_cells_; ++i) last_state_[epoch_][i] = state_[i];
    ForwardPass(forget_gate_, input, input_size, input_symbol);
    ForwardPass(input_node_, input, input_size, input_symbol);
    ForwardPass(output_gate_, input, input_size, input_symbol);
    for (i = 0; i < num_cells_; ++i) {
      forget_gate_.state_[epoch_ * num_cells_ + i] = Logistic(forget_gate_.state_[epoch_ * num_cells_ + i]);
      input_node_.state_[epoch_ * num_cells_ + i] = tanh(input_node_.state_[epoch_ * num_cells_ + i]);
      output_gate_.state_[epoch_ * num_cells_ + i] = Logistic(output_gate_.state_[epoch_ * num_cells_ + i]);
    }
    for (i = 0; i < num_cells_; ++i) {
      input_gate_state_[epoch_][i] = 1.0f - forget_gate_.state_[epoch_ * num_cells_ + i];
    }
    for (i = 0; i < num_cells_; ++i) state_[i] *= forget_gate_.state_[epoch_ * num_cells_ + i];
    for (i = 0; i < num_cells_; ++i) state_[i] += input_node_.state_[epoch_ * num_cells_ + i] * input_gate_state_[epoch_][i];
    for (i = 0; i < num_cells_; ++i) tanh_state_[epoch_][i] = tanh(state_[i]);
    for (i = 0; i < num_cells_; ++i) hidden[hidden_start + i] = output_gate_.state_[epoch_ * num_cells_ + i] * tanh_state_[epoch_][i];
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
  }

  void BackwardPass(const float* input, uint input_size, int epoch, int layer, int input_symbol, float* hidden_error) {
    uint i;
    if (epoch == (int)horizon_ - 1) {
      for (i = 0; i < num_cells_; ++i) stored_error_[i] = hidden_error[i];
      for (i = 0; i < num_cells_; ++i) state_error_[i] = 0;
    } else {
      for (i = 0; i < num_cells_; ++i) stored_error_[i] += hidden_error[i];
    }

    for (i = 0; i < num_cells_; ++i) {
      output_gate_.error_[i] = tanh_state_[epoch][i] * stored_error_[i] * output_gate_.state_[epoch * num_cells_ + i] * (1.0f - output_gate_.state_[epoch * num_cells_ + i]);
    }
    for (i = 0; i < num_cells_; ++i) {
      state_error_[i] += stored_error_[i] * output_gate_.state_[epoch * num_cells_ + i] * (1.0f - (tanh_state_[epoch][i] * tanh_state_[epoch][i]));
    }
    for (i = 0; i < num_cells_; ++i) {
      input_node_.error_[i] = state_error_[i] * input_gate_state_[epoch][i] * (1.0f - (input_node_.state_[epoch * num_cells_ + i] * input_node_.state_[epoch * num_cells_ + i]));
    }
    for (i = 0; i < num_cells_; ++i) {
      forget_gate_.error_[i] = (last_state_[epoch][i] - input_node_.state_[epoch * num_cells_ + i]) * state_error_[i] * forget_gate_.state_[epoch * num_cells_ + i] * input_gate_state_[epoch][i];
    }

    for (i = 0; i < num_cells_; ++i) hidden_error[i] = 0;
    if (epoch > 0) {
      for (i = 0; i < num_cells_; ++i) state_error_[i] *= forget_gate_.state_[epoch * num_cells_ + i];
      for (i = 0; i < num_cells_; ++i) stored_error_[i] = 0;
    } else {
      if( update_steps_<UPDATE_LIMIT ) ++update_steps_;
    }

    BackwardPass(forget_gate_, input, input_size, epoch, layer, input_symbol, hidden_error);
    BackwardPass(input_node_, input, input_size, epoch, layer, input_symbol, hidden_error);
    BackwardPass(output_gate_, input, input_size, epoch, layer, input_symbol, hidden_error);

    ClipGradients(state_error_);
    ClipGradients(stored_error_);
    ClipGradients(hidden_error);
  }

  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  static inline float Logistic(float p) {
    return 1.0f / (1.0f + exp(-p));
  }

  static void Adam(float* g, float* m, float* v, float* w, uint size, float learning_rate, float t) {
    const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
    float alpha;
    uint i;
    if (t < UPDATE_LIMIT) {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    } else {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
    }
    for (i = 0; i < size; ++i) m[i] *= beta1;
    for (i = 0; i < size; ++i) m[i] += (1.0f - beta1) * g[i];
    for (i = 0; i < size; ++i) v[i] *= beta2;
    for (i = 0; i < size; ++i) v[i] += (1.0f - beta2) * g[i] * g[i];
    if( t<UPDATE_LIMIT ) {
      for (i = 0; i < size; ++i)
        w[i] -= alpha * ((m[i] / (float)(1.0f - pow(beta1, t))) / (sqrt(v[i] / (float)(1.0f - pow(beta2, t)) + eps)));
    } else {
      for (i = 0; i < size; ++i)
        w[i] -= alpha * ((m[i] / (float)(1.0f - pow(beta1, UPDATE_LIMIT))) / (sqrt(v[i] / (float)(1.0f - pow(beta2, UPDATE_LIMIT)) + eps)));
    }
  }

  void ClipGradients(float* arr) {
    uint i;
    for (i = 0; i < num_cells_; ++i) {
      if (arr[i] < -gradient_clip_) arr[i] = -gradient_clip_;
      else if (arr[i] > gradient_clip_) arr[i] = gradient_clip_;
    }
  }

  void ForwardPass(t_NeuronLayer& neurons, const float* input, uint input_size, int input_symbol) {
    uint i, j;
    float f, sum;
    for (i = 0; i < num_cells_; ++i) {
      f = neurons.weights_[i][input_symbol];
      for (j = 0; j < input_size; ++j) f += input[j] * neurons.weights_[i][output_size_ + j];
      neurons.norm_[epoch_ * NUM_CELLS + i] = f;
    }
    sum = 0;
    for (i = 0; i < num_cells_; ++i) sum += neurons.norm_[epoch_ * NUM_CELLS + i] * neurons.norm_[epoch_ * NUM_CELLS + i];
    neurons.ivar_[epoch_] = 1.0f / sqrt((sum / num_cells_) + 1e-5f);
    for (i = 0; i < num_cells_; ++i) neurons.norm_[epoch_ * NUM_CELLS + i] *= neurons.ivar_[epoch_];
    for (i = 0; i < num_cells_; ++i) {
      neurons.state_[epoch_ * NUM_CELLS + i] = neurons.norm_[epoch_ * NUM_CELLS + i] * neurons.gamma_[i] + neurons.beta_[i];
    }
  }

  void BackwardPass(t_NeuronLayer& neurons, const float* input, uint input_size, int epoch, int layer, int input_symbol, float* hidden_error) {
    uint i, j;
    int offset;
    float sum, f;
    if( epoch==(int)horizon_-1 ) {
      for (i = 0; i < NUM_CELLS; ++i) neurons.gamma_u_[i] = 0;
      for (i = 0; i < NUM_CELLS; ++i) neurons.beta_u_[i] = 0;
      for (i = 0; i < num_cells_; ++i) {
        for (j = 0; j < neurons.input_size_; ++j) neurons.update_[i][j] = 0;
        offset = output_size_ + INPUT_SIZE;
        for (j = 0; j < 1 + NUM_CELLS * 2; ++j) {
          neurons.transpose_[j * NUM_CELLS + i] = neurons.weights_[i][j + offset];
        }
      }
    }
    for (i = 0; i < num_cells_; ++i) neurons.beta_u_[i] += neurons.error_[i];
    for (i = 0; i < num_cells_; ++i) neurons.gamma_u_[i] += neurons.error_[i] * neurons.norm_[epoch * NUM_CELLS + i];
    for (i = 0; i < num_cells_; ++i) neurons.error_[i] *= neurons.gamma_[i] * neurons.ivar_[epoch];
    sum = 0;
    for (i = 0; i < num_cells_; ++i) sum += neurons.error_[i] * neurons.norm_[epoch * NUM_CELLS + i];
    for (i = 0; i < num_cells_; ++i) neurons.error_[i] -= (sum / num_cells_) * neurons.norm_[epoch * NUM_CELLS + i];
    if( layer>0 ) {
      for (i = 0; i < num_cells_; ++i) {
        f = 0;
        for (j = 0; j < num_cells_; ++j) f += neurons.error_[j] * neurons.transpose_[(num_cells_ + i) * NUM_CELLS + j];
        hidden_error[i] += f;
      }
    }
    if( epoch > 0 ) {
      for (i = 0; i < num_cells_; ++i) {
        f = 0;
        for (j = 0; j < num_cells_; ++j) f += neurons.error_[j] * neurons.transpose_[i * NUM_CELLS + j];
        stored_error_[i] += f;
      }
    }
    for (i = 0; i < num_cells_; ++i) {
      for (j = 0; j < input_size; ++j) neurons.update_[i][output_size_ + j] += neurons.error_[i] * input[j];
      neurons.update_[i][input_symbol] += neurons.error_[i];
    }
    if (epoch == 0) {
      for (i = 0; i < num_cells_; ++i) {
        Adam(neurons.update_[i], neurons.m_[i], neurons.v_[i], neurons.weights_[i], neurons.input_size_, learning_rate_, update_steps_);
      }
      Adam(neurons.gamma_u_, neurons.gamma_m_, neurons.gamma_v_, neurons.gamma_, NUM_CELLS, learning_rate_, update_steps_);
      Adam(neurons.beta_u_, neurons.beta_m_, neurons.beta_v_, neurons.beta_, NUM_CELLS, learning_rate_, update_steps_);
    }
  }
};
//--- #include "lstm.hpp"

template<uint INPUT_SIZE, uint NUM_CELLS, uint NUM_LAYERS,uint HORIZON, uint GRADIENT_CLIP_X10,uint LEARNING_RATE_X100000, uint UPDATE_LIMIT>
struct Lstm {
  using LstmLayerType = LstmLayer<INPUT_SIZE, NUM_CELLS, HORIZON, GRADIENT_CLIP_X10,LEARNING_RATE_X100000, UPDATE_LIMIT>;
  static constexpr float learning_rate_ = LEARNING_RATE_X100000 / 100000.0f;
  static constexpr uint MAX_OUTPUT_SIZE = 256 +0;
  static constexpr uint MAX_LAYER_INPUT_SIZE = INPUT_SIZE + 1 + NUM_CELLS * 2;
  static constexpr uint MAX_LAYER_INPUT_SIZE_a = AlignUp( INPUT_SIZE + 1 + NUM_CELLS * 2, ROW_a);
  static constexpr uint NUM_CELLS_a = AlignUp(NUM_CELLS,ROW_a);
  static constexpr uint HORIZON_a = AlignUp(HORIZON,ROW_a);
  static constexpr uint NCmNLp1_a = AlignUp(NUM_CELLS * NUM_LAYERS + 1,ROW_a);

  LstmLayerType layers_[NUM_LAYERS];
  uint8_t input_history_[HORIZON_a];
  float hidden_[NCmNLp1_a];
  float hidden_error_[NUM_CELLS_a];
  float layer_input_[HORIZON][NUM_LAYERS][MAX_LAYER_INPUT_SIZE_a];
  float output_layer_[HORIZON][MAX_OUTPUT_SIZE][NCmNLp1_a];
  float output_[HORIZON][MAX_OUTPUT_SIZE];
  uint num_cells_, epoch_, horizon_, output_size_;
  int last_input_;


  NOINLINE
  void Init(uint output_size) {
    int h, epoch;
    uint i, j, l;
//printf( "Lstm @ %I64X\n", this );
    last_input_ = -1;
    // Initialize layer_input_ arrays to 0
    for (h = 0; h < HORIZON; ++h) {
      for (l = 0; l < NUM_LAYERS; ++l) {
        for (j = 0; j < MAX_LAYER_INPUT_SIZE; ++j) {
          layer_input_[h][l][j] = 0;
        }
      }
    }
    // Initialize output_layer_ arrays to 0
    for (h = 0; h < HORIZON; ++h) {
      for (i = 0; i < MAX_OUTPUT_SIZE; ++i) {
        for (j = 0; j < NUM_CELLS * NUM_LAYERS + 1; ++j) {
          output_layer_[h][i][j] = 0;
        }
      }
    }
    // Initialize output_ array
    for (h = 0; h < HORIZON; ++h) {
      for (i = 0; i < MAX_OUTPUT_SIZE; ++i) {
        output_[h][i] = (i < output_size) ? (1.0f / output_size) : 0.0f;
      }
    }
    num_cells_ = NUM_CELLS;
    epoch_ = 0;
    horizon_ = HORIZON;
    output_size_ = output_size;
    for (i = 0; i < NUM_CELLS * NUM_LAYERS + 1; ++i) hidden_[i] = 0;
    hidden_[NUM_CELLS * NUM_LAYERS] = 1;
    for (i = 0; i < NUM_CELLS; ++i) hidden_error_[i] = 0;
    for (epoch = 0; epoch < HORIZON; ++epoch) {
      input_history_[epoch] = 0;
      // Set the last element to 1 for each layer
      // Layer 0 uses size (1 + NUM_CELLS + INPUT_SIZE), last element at index (NUM_CELLS + INPUT_SIZE)
      layer_input_[epoch][0][NUM_CELLS + INPUT_SIZE] = 1;
      // Other layers use size MAX_LAYER_INPUT_SIZE, last element at index (MAX_LAYER_INPUT_SIZE - 1)
      for (i = 1; i < NUM_LAYERS; ++i) {
        layer_input_[epoch][i][MAX_LAYER_INPUT_SIZE - 1] = 1;
      }
    }
    // Initialize layers with proper input sizes
    // Layer 0: (1 + NUM_CELLS + INPUT_SIZE) + output_size
    // Other layers: MAX_LAYER_INPUT_SIZE + output_size
    layers_[0].Init((1 + NUM_CELLS + INPUT_SIZE) + output_size, output_size);
    for (i = 1; i < NUM_LAYERS; ++i) {
      layers_[i].Init(MAX_LAYER_INPUT_SIZE + output_size, output_size);
    }

  }

  NOINLINE
  void SetInput(const float* input) {
    uint i, j;
    for (i = 0; i < NUM_LAYERS; ++i) {
      for (j = 0; j < INPUT_SIZE; ++j) layer_input_[epoch_][i][j] = input[j];
    }
  }

  NOINLINE
  const float* Perceive(uint input) {
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
            for (j = 0; j < NUM_CELLS; ++j) hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
          }
          prev_epoch = epoch - 1;
          if (prev_epoch == -1) prev_epoch = horizon_ - 1;
          input_symbol = input_history_[prev_epoch];
          if (epoch == 0) input_symbol = old_input;
          uint layer_input_size = (layer == 0) ? (1 + NUM_CELLS + INPUT_SIZE) : MAX_LAYER_INPUT_SIZE;
          layers_[layer].BackwardPass(layer_input_[epoch][layer], layer_input_size, epoch, layer, input_symbol, hidden_error_);
        }
      }
    }

    for (i = 0; i < output_size_; ++i) {
      error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
      for (j = 0; j < NUM_CELLS * NUM_LAYERS + 1; ++j) {
        output_layer_[epoch_][i][j] = output_layer_[last_epoch][i][j];
      }
      for (j = 0; j < NUM_CELLS * NUM_LAYERS + 1; ++j) {
        output_layer_[epoch_][i][j] -= learning_rate_ * error * hidden_[j];
      }
    }
    return Predict(input);
  }

  NOINLINE
  const float* Predict(uint input) {
    uint i, j, hidden_offset, dest_offset;
    float sum;
    int epoch;
    for (i = 0; i < NUM_LAYERS; ++i) {
      hidden_offset = i * num_cells_;
      for (j = 0; j < num_cells_; ++j) {
        layer_input_[epoch_][i][INPUT_SIZE + j] = hidden_[hidden_offset + j];
      }
      uint layer_input_size = (i == 0) ? (1 + NUM_CELLS + INPUT_SIZE) : MAX_LAYER_INPUT_SIZE;
      layers_[i].ForwardPass(layer_input_[epoch_][i], layer_input_size, input, hidden_, i * num_cells_);
      if (i < NUM_LAYERS - 1) {
        dest_offset = num_cells_ + INPUT_SIZE;
        for (j = 0; j < num_cells_; ++j) {
          layer_input_[epoch_][i + 1][dest_offset + j] = hidden_[hidden_offset + j];
        }
      }
    }
    for (i = 0; i < output_size_; ++i) {
      sum = 0;
      for (j = 0; j < NUM_CELLS * NUM_LAYERS + 1; ++j) sum += hidden_[j] * output_layer_[epoch_][i][j];
      output_[epoch_][i] = exp(sum);
    }
    sum = 0;
    for (i = 0; i < output_size_; ++i) sum += output_[epoch_][i];
    for (i = 0; i < output_size_; ++i) output_[epoch_][i] /= sum;
    epoch = epoch_;
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
    last_input_ = input;
    return output_[epoch];
  }

};
//--- #include "byte-model.hpp"

struct Byte_Model {

  void Init(char* vocab) {
    int i;
//printf( "Byte_Model @ %I64X\n", this );
    vocab_ = vocab;
    for (i = 0; i < 256; ++i) {
      probs_[i] = 1.0 / 256;
    }
  }

  const float* BytePredict() {
    return probs_;
  }

  void ByteUpdate() {
    int i;
    for (i = 0; i < 256; ++i) {
      if (!vocab_[i]) probs_[i] = 0;
    }
  }

  char* vocab_;
  float probs_[256];
};

//--- #include "ppmd-model.hpp"

struct PPMD : Byte_Model {
  ppmd_Model ppmd_model_;

  NOINLINE
  void Init(int order, int memory, char* vocab) {
//printf( "PPMD @ %I64X\n", this );
    Byte_Model::Init(vocab);
    ppmd_model_.Init(order,memory,1,0);
  }

  NOINLINE
  void ByteUpdate(uint byte) {
    int i;
    float sum;
    ppmd_model_.ppmd_UpdateByte( byte&0xFF );
    ppmd_model_.ppmd_PrepareByte();
    for (i = 0; i < 256; ++i) {
      probs_[i] = ppmd_model_.sqp[i];
      if (probs_[i] < 1) probs_[i] = 1;
    }
    Byte_Model::ByteUpdate();
    /* probs_ /= probs_.sum(); */
    sum = 0;
    for (i = 0; i < 256; ++i) sum += probs_[i];
    for (i = 0; i < 256; ++i) probs_[i] /= sum;
  }

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
//printf( "Model @ %I64X\n", this );
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
    const float* output = lstm_->Perceive( byte_map_[sym] );
    int i, offset;
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

using LstmType = Lstm<LSTM_INPUT_SIZE, LSTM_NUM_CELLS, LSTM_NUM_LAYERS,LSTM_HORIZON, LSTM_GRADIENT_CLIP_X10,LSTM_LEARNING_RATE_X100000, UPDATE_LIMIT>;

//ALIGN(4096) char lstm_place[sizeof(LstmType)+4096];
//LstmType& lstm = *(LstmType*)(lstm_place+8*3);
ALIGN(16) LstmType lstm;

ALIGN(64) Rangecoder rc;

ALIGN(64) PPMD byte_model_;

ALIGN(64) Model<LstmType> M;


int main( int argc, char** argv ) {
  uint f_DEC, i, j, c, pc, code, low, total, freq[CNUM], f_len, f_pos;
  FILE* f;
  FILE* g;
  const float* p;

  printf( "sizeof(lstm)=%i; sizeof(PPMD)=%i; sizeof(Model)=%i\n", int(sizeof(lstm)), int(sizeof(byte_model_)), int(sizeof(M)));

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

  //byte_model_ = new PPMD();
  byte_model_.Init(PPMD_ORDER, PPMD_MEMORY, cmap);

  byte_model_.Byte_Model::ByteUpdate();

  srand(0xDEADBEEF);
  //lstm = new LstmType();
  lstm.Init(total);
  //PM = new Model<LstmType>();
  M.Init(cmap, &lstm);

  for( f_pos=0; f_pos<f_len; f_pos++ ) {

    for( i=0,total=0; i<CNUM; i++ ) {
      freq[i] = M.probs_[i]*SCALE;
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

    byte_model_.ByteUpdate(c);
    p = byte_model_.BytePredict();
    M.lstm_->SetInput(p);
    M.Update( c );

/*if( ftell(rc.f)>(1<<20) ) break;*/
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}
