


// C library headers
#include <stdlib.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <math.h>

// C++ library headers
#include <vector>

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
  Sigmoid(int logit_size) : logit_size_(logit_size),
      logit_table_(logit_size, 0) {
    for (int i = 0; i < logit_size_; ++i) {
      logit_table_[i] = SlowLogit((i + 0.5f) / logit_size_);
    }
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
    return (0.5f * (p / (1.0f + abs(p)) + 1.0f));
  }

 private:
  float SlowLogit(float p) {
    return log(p / (1 - p));
  }

  int logit_size_;
  std::vector<float> logit_table_;
};
//--- #include "neuron-layer.hpp"

struct NeuronLayer {
  NeuronLayer(unsigned int input_size, unsigned int num_cells, int horizon,
    int offset) : error_(num_cells), ivar_(horizon), gamma_(num_cells, 1.0),
    gamma_u_(num_cells), gamma_m_(num_cells), gamma_v_(num_cells),
    beta_(num_cells), beta_u_(num_cells), beta_m_(num_cells),
    beta_v_(num_cells), weights_(num_cells, std::vector<float>(input_size)),
    state_(horizon, std::vector<float>(num_cells)),
    update_(num_cells, std::vector<float>(input_size)),
    m_(num_cells, std::vector<float>(input_size)),
    v_(num_cells, std::vector<float>(input_size)),
    transpose_(input_size - offset, std::vector<float>(num_cells)),
    norm_(horizon, std::vector<float>(num_cells)) {}

  std::vector<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::vector<std::vector<float>> weights_, state_, update_, m_, v_,
      transpose_, norm_;
};
//--- #include "lstm-layer.hpp"

template<unsigned int NUM_CELLS, unsigned int HORIZON,
         unsigned int GRADIENT_CLIP_X10, unsigned int LEARNING_RATE_X100000,
         unsigned int UPDATE_LIMIT>
class LstmLayer {
 public:
  static constexpr float gradient_clip_ = GRADIENT_CLIP_X10 / 10.0f;
  static constexpr float learning_rate_ = LEARNING_RATE_X100000 / 100000.0f;

  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size) :
      state_(NUM_CELLS), state_error_(NUM_CELLS), stored_error_(NUM_CELLS),
      tanh_state_(HORIZON, std::vector<float>(NUM_CELLS)),
      input_gate_state_(HORIZON, std::vector<float>(NUM_CELLS)),
      last_state_(HORIZON, std::vector<float>(NUM_CELLS)),
      num_cells_(NUM_CELLS), epoch_(0), horizon_(HORIZON),
      input_size_(auxiliary_input_size), output_size_(output_size),
      forget_gate_(input_size, NUM_CELLS, HORIZON, output_size_ + input_size_),
      input_node_(input_size, NUM_CELLS, HORIZON, output_size_ + input_size_),
      output_gate_(input_size, NUM_CELLS, HORIZON, output_size_ + input_size_)
  {
    float val = sqrt(6.0f / float(input_size_ + output_size_));
    float low = -val;
    float range = 2 * val;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      for (unsigned int j = 0; j < forget_gate_.weights_[i].size(); ++j) {
        forget_gate_.weights_[i][j] = low + Rand() * range;
        input_node_.weights_[i][j] = low + Rand() * range;
        output_gate_.weights_[i][j] = low + Rand() * range;
      }
      forget_gate_.weights_[i][forget_gate_.weights_[i].size() - 1] = 1;
    }
  }

  void ForwardPass(const std::vector<float>& input, int input_symbol,
      std::vector<float>* hidden, int hidden_start) {
    // last_state_[epoch_] = state_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      last_state_[epoch_][i] = state_[i];
    }
    ForwardPass(forget_gate_, input, input_symbol);
    ForwardPass(input_node_, input, input_symbol);
    ForwardPass(output_gate_, input, input_symbol);
    for (unsigned int i = 0; i < num_cells_; ++i) {
      forget_gate_.state_[epoch_][i] = Sigmoid::Logistic(
          forget_gate_.state_[epoch_][i]);
      input_node_.state_[epoch_][i] = tanh(input_node_.state_[epoch_][i]);
      output_gate_.state_[epoch_][i] = Sigmoid::Logistic(
          output_gate_.state_[epoch_][i]);
    }
    // input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      input_gate_state_[epoch_][i] = 1.0f - forget_gate_.state_[epoch_][i];
    }
    // state_ *= forget_gate_.state_[epoch_];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      state_[i] *= forget_gate_.state_[epoch_][i];
    }
    // state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      state_[i] += input_node_.state_[epoch_][i] * input_gate_state_[epoch_][i];
    }
    // tanh_state_[epoch_] = tanh(state_);
    for (unsigned int i = 0; i < num_cells_; ++i) {
      tanh_state_[epoch_][i] = tanh(state_[i]);
    }
    // (*hidden)[slice] = output_gate_.state_[epoch_] * tanh_state_[epoch_];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      (*hidden)[hidden_start + i] = output_gate_.state_[epoch_][i] * tanh_state_[epoch_][i];
    }
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
  }

  void BackwardPass(const std::vector<float>& input, int epoch,
      int layer, int input_symbol, std::vector<float>* hidden_error) {
    if (epoch == (int)horizon_ - 1) {
      // stored_error_ = *hidden_error;
      for (unsigned int i = 0; i < num_cells_; ++i) {
        stored_error_[i] = (*hidden_error)[i];
      }
      // state_error_ = 0;
      for (unsigned int i = 0; i < num_cells_; ++i) {
        state_error_[i] = 0;
      }
    } else {
      // stored_error_ += *hidden_error;
      for (unsigned int i = 0; i < num_cells_; ++i) {
        stored_error_[i] += (*hidden_error)[i];
      }
    }

    // output_gate_.error_ = tanh_state_[epoch] * stored_error_ * output_gate_.state_[epoch] * (1.0f - output_gate_.state_[epoch]);
    for (unsigned int i = 0; i < num_cells_; ++i) {
      output_gate_.error_[i] = tanh_state_[epoch][i] * stored_error_[i] *
          output_gate_.state_[epoch][i] * (1.0f - output_gate_.state_[epoch][i]);
    }
    // state_error_ += stored_error_ * output_gate_.state_[epoch] * (1.0f - (tanh_state_[epoch] * tanh_state_[epoch]));
    for (unsigned int i = 0; i < num_cells_; ++i) {
      state_error_[i] += stored_error_[i] * output_gate_.state_[epoch][i] * (1.0f -
          (tanh_state_[epoch][i] * tanh_state_[epoch][i]));
    }
    // input_node_.error_ = state_error_ * input_gate_state_[epoch] * (1.0f - (input_node_.state_[epoch] * input_node_.state_[epoch]));
    for (unsigned int i = 0; i < num_cells_; ++i) {
      input_node_.error_[i] = state_error_[i] * input_gate_state_[epoch][i] * (1.0f -
          (input_node_.state_[epoch][i] * input_node_.state_[epoch][i]));
    }
    // forget_gate_.error_ = (last_state_[epoch] - input_node_.state_[epoch]) * state_error_ * forget_gate_.state_[epoch] * input_gate_state_[epoch];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      forget_gate_.error_[i] = (last_state_[epoch][i] - input_node_.state_[epoch][i]) *
          state_error_[i] * forget_gate_.state_[epoch][i] * input_gate_state_[epoch][i];
    }

    // *hidden_error = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      (*hidden_error)[i] = 0;
    }
    if (epoch > 0) {
      // state_error_ *= forget_gate_.state_[epoch];
      for (unsigned int i = 0; i < num_cells_; ++i) {
        state_error_[i] *= forget_gate_.state_[epoch][i];
      }
      // stored_error_ = 0;
      for (unsigned int i = 0; i < num_cells_; ++i) {
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

    ClipGradients(&state_error_);
    ClipGradients(&stored_error_);
    ClipGradients(hidden_error);
  }

  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  std::vector<std::vector<std::vector<float>>*> Weights() {
    std::vector<std::vector<std::vector<float>>*> weights;
    weights.push_back(&forget_gate_.weights_);
    weights.push_back(&input_node_.weights_);
    weights.push_back(&output_gate_.weights_);
    return weights;
  }

 private:
  std::vector<float> state_, state_error_, stored_error_;
  std::vector<std::vector<float>> tanh_state_, input_gate_state_, last_state_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;

// ============================================================================
// Adam optimizer (helper function template)
// ============================================================================

  template<unsigned int UPD_LIMIT>
  static void Adam(std::vector<float>* g, std::vector<float>* m, std::vector<float>* v, std::vector<float>* w, float learning_rate, float t) {
    const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
    float alpha;
    if (t < UPD_LIMIT) {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    } else {
      alpha = learning_rate * 0.1f / sqrt(5e-5f * UPD_LIMIT + 1.0f);
    }
    // (*m) *= beta1;
    for (unsigned int i = 0; i < m->size(); ++i) {
      (*m)[i] *= beta1;
    }
    // (*m) += (1.0f - beta1) * (*g);
    for (unsigned int i = 0; i < m->size(); ++i) {
      (*m)[i] += (1.0f - beta1) * (*g)[i];
    }
    // (*v) *= beta2;
    for (unsigned int i = 0; i < v->size(); ++i) {
      (*v)[i] *= beta2;
    }
    // (*v) += (1.0f - beta2) * (*g) * (*g);
    for (unsigned int i = 0; i < v->size(); ++i) {
      (*v)[i] += (1.0f - beta2) * (*g)[i] * (*g)[i];
    }
    if (t < UPD_LIMIT) {
      // (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))) / (sqrt((*v) / (float)(1.0f - pow(beta2, t)) + eps)));
      for (unsigned int i = 0; i < w->size(); ++i) {
        (*w)[i] -= alpha * (((*m)[i] / (float)(1.0f - pow(beta1, t))) /
            (sqrt((*v)[i] / (float)(1.0f - pow(beta2, t)) + eps)));
      }
    } else {
      // (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, UPD_LIMIT))) / (sqrt((*v) / (float)(1.0f - pow(beta2, UPD_LIMIT)) + eps)));
      for (unsigned int i = 0; i < w->size(); ++i) {
        (*w)[i] -= alpha * (((*m)[i] / (float)(1.0f - pow(beta1, UPD_LIMIT))) /
            (sqrt((*v)[i] / (float)(1.0f - pow(beta2, UPD_LIMIT)) + eps)));
      }
    }
  }

  void ClipGradients(std::vector<float>* arr) {
    for (unsigned int i = 0; i < arr->size(); ++i) {
      if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
      else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
    }
  }

  void ForwardPass(NeuronLayer& neurons, const std::vector<float>& input,
      int input_symbol) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = neurons.weights_[i][input_symbol];
      for (unsigned int j = 0; j < input.size(); ++j) {
        f += input[j] * neurons.weights_[i][output_size_ + j];
      }
      neurons.norm_[epoch_][i] = f;
    }
    // neurons.ivar_[epoch_] = 1.0f / sqrt(((neurons.norm_[epoch_] * neurons.norm_[epoch_]).sum() / num_cells_) + 1e-5f);
    float sum = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      sum += neurons.norm_[epoch_][i] * neurons.norm_[epoch_][i];
    }
    neurons.ivar_[epoch_] = 1.0f / sqrt((sum / num_cells_) + 1e-5f);
    // neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.norm_[epoch_][i] *= neurons.ivar_[epoch_];
    }
    // neurons.state_[epoch_] = neurons.norm_[epoch_] * neurons.gamma_ + neurons.beta_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.state_[epoch_][i] = neurons.norm_[epoch_][i] * neurons.gamma_[i] +
          neurons.beta_[i];
    }
  }

  void BackwardPass(NeuronLayer& neurons, const std::vector<float>&input,
      int epoch, int layer, int input_symbol,
      std::vector<float>* hidden_error) {
    if (epoch == (int)horizon_ - 1) {
      // neurons.gamma_u_ = 0;
      for (unsigned int i = 0; i < neurons.gamma_u_.size(); ++i) {
        neurons.gamma_u_[i] = 0;
      }
      // neurons.beta_u_ = 0;
      for (unsigned int i = 0; i < neurons.beta_u_.size(); ++i) {
        neurons.beta_u_[i] = 0;
      }
      for (unsigned int i = 0; i < num_cells_; ++i) {
        // neurons.update_[i] = 0;
        for (unsigned int j = 0; j < neurons.update_[i].size(); ++j) {
          neurons.update_[i][j] = 0;
        }
        int offset = output_size_ + input_size_;
        for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
          neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
        }
      }
    }
    // neurons.beta_u_ += neurons.error_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.beta_u_[i] += neurons.error_[i];
    }
    // neurons.gamma_u_ += neurons.error_ * neurons.norm_[epoch];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.gamma_u_[i] += neurons.error_[i] * neurons.norm_[epoch][i];
    }
    // neurons.error_ *= neurons.gamma_ * neurons.ivar_[epoch];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.error_[i] *= neurons.gamma_[i] * neurons.ivar_[epoch];
    }
    // neurons.error_ -= ((neurons.error_ * neurons.norm_[epoch]).sum() / num_cells_) * neurons.norm_[epoch];
    float sum = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      sum += neurons.error_[i] * neurons.norm_[epoch][i];
    }
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.error_[i] -= (sum / num_cells_) * neurons.norm_[epoch][i];
    }
    if (layer > 0) {
      for (unsigned int i = 0; i < num_cells_; ++i) {
        float f = 0;
        for (unsigned int j = 0; j < num_cells_; ++j) {
          f += neurons.error_[j] * neurons.transpose_[num_cells_ + i][j];
        }
        (*hidden_error)[i] += f;
      }
    }
    if (epoch > 0) {
      for (unsigned int i = 0; i < num_cells_; ++i) {
        float f = 0;
        for (unsigned int j = 0; j < num_cells_; ++j) {
          f += neurons.error_[j] * neurons.transpose_[i][j];
        }
        stored_error_[i] += f;
      }
    }
    // neurons.update_[i][slice] += neurons.error_[i] * input;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      for (unsigned int j = 0; j < input.size(); ++j) {
        neurons.update_[i][output_size_ + j] += neurons.error_[i] * input[j];
      }
      neurons.update_[i][input_symbol] += neurons.error_[i];
    }
    if (epoch == 0) {
      for (unsigned int i = 0; i < num_cells_; ++i) {
        Adam<UPDATE_LIMIT>(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i],
            &neurons.weights_[i], learning_rate_, update_steps_);
      }
      Adam<UPDATE_LIMIT>(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
          &neurons.gamma_, learning_rate_, update_steps_);
      Adam<UPDATE_LIMIT>(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
          &neurons.beta_, learning_rate_, update_steps_);
    }
  }
};
//--- #include "lstm.hpp"

template<unsigned int INPUT_SIZE, unsigned int NUM_CELLS, unsigned int NUM_LAYERS,
         unsigned int HORIZON, unsigned int GRADIENT_CLIP_X10,
         unsigned int LEARNING_RATE_X100000, unsigned int UPDATE_LIMIT>
class Lstm {
 public:
  using LstmLayerType = LstmLayer<NUM_CELLS, HORIZON, GRADIENT_CLIP_X10,
                                   LEARNING_RATE_X100000, UPDATE_LIMIT>;
  static constexpr float learning_rate_ = LEARNING_RATE_X100000 / 100000.0f;

  NOINLINE
  Lstm(unsigned int output_size) : input_history_(HORIZON),
      hidden_(NUM_CELLS * NUM_LAYERS + 1), hidden_error_(NUM_CELLS),
      layer_input_(HORIZON, std::vector<std::vector<float>>(NUM_LAYERS,
      std::vector<float>(INPUT_SIZE + 1 + NUM_CELLS * 2))),
      output_layer_(HORIZON, std::vector<std::vector<float>>(output_size,
      std::vector<float>(NUM_CELLS * NUM_LAYERS + 1))),
      output_(HORIZON, std::vector<float>(output_size, 1.0 / output_size)),
      num_cells_(NUM_CELLS), epoch_(0),
      horizon_(HORIZON), input_size_(INPUT_SIZE), output_size_(output_size) {
    hidden_[hidden_.size() - 1] = 1;
    for (int epoch = 0; epoch < HORIZON; ++epoch) {
      layer_input_[epoch][0].resize(1 + NUM_CELLS + INPUT_SIZE);
      for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
        layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
      }
    }
    for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
      layers_.emplace_back(layer_input_[0][i].size() + output_size, INPUT_SIZE, output_size);
    }
  }

  ~Lstm() {}

  NOINLINE
  void SetInput(const std::vector<float>& input) {
    for (unsigned int i = 0; i < layers_.size(); ++i) {
      for (unsigned int j = 0; j < input_size_; ++j) {
        layer_input_[epoch_][i][j] = input[j];
      }
    }
  }

  NOINLINE
  std::vector<float>& Perceive(unsigned int input) {
    int last_epoch = epoch_ - 1;
    if (last_epoch == -1) last_epoch = horizon_ - 1;
    int old_input = input_history_[last_epoch];
    input_history_[last_epoch] = input;
    if (epoch_ == 0) {
      for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
        for (int layer = layers_.size() - 1; layer >= 0; --layer) {
          int offset = layer * num_cells_;
          for (unsigned int i = 0; i < output_size_; ++i) {
            float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
            for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
              hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
            }
          }
          int prev_epoch = epoch - 1;
          if (prev_epoch == -1) prev_epoch = horizon_ - 1;
          int input_symbol = input_history_[prev_epoch];
          if (epoch == 0) input_symbol = old_input;
          layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer,
              input_symbol, &hidden_error_);
        }
      }
    }

    for (unsigned int i = 0; i < output_size_; ++i) {
      float error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
      // output_layer_[epoch_][i] = output_layer_[last_epoch][i];
      for (unsigned int j = 0; j < output_layer_[epoch_][i].size(); ++j) {
        output_layer_[epoch_][i][j] = output_layer_[last_epoch][i][j];
      }
      // output_layer_[epoch_][i] -= learning_rate_ * error * hidden_;
      for (unsigned int j = 0; j < output_layer_[epoch_][i].size(); ++j) {
        output_layer_[epoch_][i][j] -= learning_rate_ * error * hidden_[j];
      }
    }
    return Predict(input);
  }

  NOINLINE
  std::vector<float>& Predict(unsigned int input) {
    for (unsigned int i = 0; i < layers_.size(); ++i) {
      unsigned int hidden_offset = i * num_cells_;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        layer_input_[epoch_][i][input_size_ + j] = hidden_[hidden_offset + j];
      }
      layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
          num_cells_);
      if (i < layers_.size() - 1) {
        unsigned int dest_offset = num_cells_ + input_size_;
        for (unsigned int j = 0; j < num_cells_; ++j) {
          layer_input_[epoch_][i + 1][dest_offset + j] = hidden_[hidden_offset + j];
        }
      }
    }
    for (unsigned int i = 0; i < output_size_; ++i) {
      float sum = 0;
      for (unsigned int j = 0; j < hidden_.size(); ++j) {
        sum += hidden_[j] * output_layer_[epoch_][i][j];
      }
      output_[epoch_][i] = exp(sum);
    }
    // output_[epoch_] /= output_[epoch_].sum();
    float sum = 0;
    for (unsigned int i = 0; i < output_size_; ++i) {
      sum += output_[epoch_][i];
    }
    for (unsigned int i = 0; i < output_size_; ++i) {
      output_[epoch_][i] /= sum;
    }
    int epoch = epoch_;
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
    last_input_ = input;
    return output_[epoch];
  }

 private:
  std::vector<LstmLayerType> layers_;
  std::vector<uint8_t> input_history_;
  std::vector<float> hidden_, hidden_error_;
  std::vector<std::vector<std::vector<float>>> layer_input_,
      output_layer_;
  std::vector<std::vector<float>> output_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  int last_input_ = -1;
};
//--- #include "byte-model.hpp"

class Byte_Model {
 public:
  virtual ~Byte_Model() {}

  Byte_Model(char* vocab) : outputs_(1, 0.5), ex(0), top_(255), mid_(0),
      bot_(0), vocab_(vocab), probs_(256, 1.0 / 256) {}

  const std::vector<float>& Predict() const {return outputs_;}
  unsigned int NumOutputs() {return outputs_.size();}

  std::vector<float>& Predict() {
    auto mid = bot_ + ((top_ - bot_) / 2);
    float num = 0.0f;
    for (int i = mid + 1; i <= top_; ++i) {
      num += probs_[i];
    }
    float denom = num;
    for (int i = bot_; i <= mid; ++i) {
      denom += probs_[i];
    }
    ex = bot_;
    float max_prob_val = probs_[bot_];
    for (int i = bot_ + 1; i <= top_; i++) {
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

  const std::vector<float>& BytePredict() {
    return probs_;
  }

  void ByteUpdate() {
    top_ = 255;
    bot_ = 0;
    for (int i = 0; i < 256; ++i) {
      if (!vocab_[i]) probs_[i] = 0;
    }
  }

  int ex;

 protected:
  mutable std::vector<float> outputs_;
  int top_, mid_, bot_;
  char* vocab_;
  std::vector<float> probs_;
};

//--- #include "ppmd-model.hpp"

class PPMD : public Byte_Model {
 public:

  NOINLINE
  PPMD(int order, int memory, char* vocab) : Byte_Model(vocab) {
    ppmd_model_ = new ppmd_Model();
    ppmd_model_->Init(order,memory,1,0);
  }

  ~PPMD() {
    delete ppmd_model_;
  }

  NOINLINE
  void ByteUpdate(unsigned int byte) {
    ppmd_model_->ppmd_UpdateByte( byte&0xFF );
    ppmd_model_->ppmd_PrepareByte();
    for (int i = 0; i < 256; ++i) {
      probs_[i] = ppmd_model_->sqp[i];
      if (probs_[i] < 1) probs_[i] = 1;
    }
    Byte_Model::ByteUpdate();
    // probs_ /= probs_.sum();
    float sum = 0;
    for (int i = 0; i < 256; ++i) {
      sum += probs_[i];
    }
    for (int i = 0; i < 256; ++i) {
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

  Model( char* vocab, LstmType* lstm ) {
    vocab_ = vocab;
    lstm_ = lstm;
    int i, offset = 0;
    for( i = 0; i < 256; i++ ) {
      byte_map_[i] = offset;
      if (vocab_[i]) ++offset;
      probs_[i]=1.0/256;
    }
  }

  void Update( int sym ) {
    const auto& output = lstm_->Perceive( byte_map_[sym] );
    int i, offset = 0;
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
constexpr unsigned int LSTM_INPUT_SIZE = 128;
constexpr unsigned int LSTM_NUM_CELLS = 90;
constexpr unsigned int LSTM_NUM_LAYERS = 2;
constexpr unsigned int LSTM_HORIZON = 73;
constexpr unsigned int LSTM_LEARNING_RATE_X100000 = 7200;  // 0.072 * 100000
constexpr unsigned int LSTM_GRADIENT_CLIP_X10 = 20;         // 2.0 * 10
constexpr unsigned int UPDATE_LIMIT = 3000;

using LstmType = Lstm<LSTM_INPUT_SIZE, LSTM_NUM_CELLS, LSTM_NUM_LAYERS,
                      LSTM_HORIZON, LSTM_GRADIENT_CLIP_X10,
                      LSTM_LEARNING_RATE_X100000, UPDATE_LIMIT>;

int main( int argc, char** argv ) {

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

  auto byte_model_ = new PPMD(PPMD_ORDER, PPMD_MEMORY, cmap);

  byte_model_->Byte_Model::ByteUpdate();

  srand(0xDEADBEEF);
  Model<LstmType>* PM = new Model<LstmType>( cmap, new LstmType(total) );

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

const std::vector<float>& p = byte_model_->BytePredict();
PM->lstm_->SetInput(p);

    PM->Update( c );

//if( ftell(rc.f)>(1<<20) ) break;
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}
