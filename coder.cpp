


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
    int offset) : error_(num_cells), ivar_(horizon), gamma_(1.0, num_cells),
    gamma_u_(num_cells), gamma_m_(num_cells), gamma_v_(num_cells),
    beta_(num_cells), beta_u_(num_cells), beta_m_(num_cells),
    beta_v_(num_cells), weights_(std::valarray<float>(input_size), num_cells),
    state_(std::valarray<float>(num_cells), horizon),
    update_(std::valarray<float>(input_size), num_cells),
    m_(std::valarray<float>(input_size), num_cells),
    v_(std::valarray<float>(input_size), num_cells),
    transpose_(std::valarray<float>(num_cells), input_size - offset),
    norm_(std::valarray<float>(num_cells), horizon) {}

  std::valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::valarray<std::valarray<float>> weights_, state_, update_, m_, v_,
      transpose_, norm_;
};
//--- #include "lstm-layer.hpp"

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate, int update_limit) :
      state_(num_cells), state_error_(num_cells), stored_error_(num_cells),
      tanh_state_(std::valarray<float>(num_cells), horizon),
      input_gate_state_(std::valarray<float>(num_cells), horizon),
      last_state_(std::valarray<float>(num_cells), horizon),
      gradient_clip_(gradient_clip), learning_rate_(learning_rate),
      num_cells_(num_cells), epoch_(0), horizon_(horizon),
      input_size_(auxiliary_input_size), output_size_(output_size),
      forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
      input_node_(input_size, num_cells, horizon, output_size_ + input_size_),
      output_gate_(input_size, num_cells, horizon, output_size_ + input_size_) 
  {
    UPDATE_LIMIT = update_limit;
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

  void ForwardPass(const std::valarray<float>& input, int input_symbol,
      std::valarray<float>* hidden, int hidden_start) {
    last_state_[epoch_] = state_;
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
    input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
    state_ *= forget_gate_.state_[epoch_];
    state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
    tanh_state_[epoch_] = tanh(state_);
    std::slice slice = std::slice(hidden_start, num_cells_, 1);
    (*hidden)[slice] = output_gate_.state_[epoch_] * tanh_state_[epoch_];
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
  }

  void BackwardPass(const std::valarray<float>& input, int epoch,
      int layer, int input_symbol, std::valarray<float>* hidden_error) {
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

  std::vector<std::valarray<std::valarray<float>>*> Weights() {
    std::vector<std::valarray<std::valarray<float>>*> weights;
    weights.push_back(&forget_gate_.weights_);
    weights.push_back(&input_node_.weights_);
    weights.push_back(&output_gate_.weights_);
    return weights;
  }

 private:
  std::valarray<float> state_, state_error_, stored_error_;
  std::valarray<std::valarray<float>> tanh_state_, input_gate_state_, last_state_;
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;

//--- #include "adam.hpp"

uint UPDATE_LIMIT;

// ============================================================================
// Adam optimizer (helper function)
// ============================================================================

void Adam(std::valarray<float>* g, std::valarray<float>* m, std::valarray<float>* v, std::valarray<float>* w, float learning_rate, float t) {
  const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
  float alpha;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
  }
  (*m) *= beta1;
  (*m) += (1.0f - beta1) * (*g);
  (*v) *= beta2;
  (*v) += (1.0f - beta2) * (*g) * (*g);
  if (t < UPDATE_LIMIT) {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))) /
        (sqrt((*v) / (float)(1.0f - pow(beta2, t)) + eps)));
  } else {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, UPDATE_LIMIT))) /
        (sqrt((*v) / (float)(1.0f - pow(beta2, UPDATE_LIMIT)) + eps)));
  }
}

  void ClipGradients(std::valarray<float>* arr) {
    for (unsigned int i = 0; i < arr->size(); ++i) {
      if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
      else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
    }
  }

  void ForwardPass(NeuronLayer& neurons, const std::valarray<float>& input,
      int input_symbol) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = neurons.weights_[i][input_symbol];
      for (unsigned int j = 0; j < input.size(); ++j) {
        f += input[j] * neurons.weights_[i][output_size_ + j];
      }
      neurons.norm_[epoch_][i] = f;
    }
    neurons.ivar_[epoch_] = 1.0f / sqrt(((neurons.norm_[epoch_] *
        neurons.norm_[epoch_]).sum() / num_cells_) + 1e-5f);
    neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
    neurons.state_[epoch_] = neurons.norm_[epoch_] * neurons.gamma_ +
        neurons.beta_;
  }

  void BackwardPass(NeuronLayer& neurons, const std::valarray<float>&input,
      int epoch, int layer, int input_symbol,
      std::valarray<float>* hidden_error) {
    if (epoch == (int)horizon_ - 1) {
      neurons.gamma_u_ = 0;
      neurons.beta_u_ = 0;
      for (unsigned int i = 0; i < num_cells_; ++i) {
        neurons.update_[i] = 0;
        int offset = output_size_ + input_size_;
        for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
          neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
        }
      }
    }
    neurons.beta_u_ += neurons.error_;
    neurons.gamma_u_ += neurons.error_ * neurons.norm_[epoch];
    neurons.error_ *= neurons.gamma_ * neurons.ivar_[epoch];
    neurons.error_ -= ((neurons.error_ * neurons.norm_[epoch]).sum() /
        num_cells_) * neurons.norm_[epoch];
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
    std::slice slice = std::slice(output_size_, input.size(), 1);
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.update_[i][slice] += neurons.error_[i] * input;
      neurons.update_[i][input_symbol] += neurons.error_[i];
    }
    if (epoch == 0) {
      for (unsigned int i = 0; i < num_cells_; ++i) {
        Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i],
            &neurons.weights_[i], learning_rate_, update_steps_);
      }
      Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
          &neurons.gamma_, learning_rate_, update_steps_);
      Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
          &neurons.beta_, learning_rate_, update_steps_);
    }
  }
};
//--- #include "lstm.hpp"

class Lstm {
 public:
  NOINLINE
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip, int update_limit) : input_history_(horizon),
      hidden_(num_cells * num_layers + 1), hidden_error_(num_cells),
      layer_input_(std::valarray<std::valarray<float>>(std::valarray<float>
      (input_size + 1 + num_cells * 2), num_layers), horizon),
      output_layer_(std::valarray<std::valarray<float>>(std::valarray<float>
     (num_cells * num_layers + 1), output_size), horizon),
      output_(std::valarray<float>(1.0 / output_size, output_size), horizon),
      learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
      horizon_(horizon), input_size_(input_size), output_size_(output_size) {
    hidden_[hidden_.size() - 1] = 1;
    for (int epoch = 0; epoch < horizon; ++epoch) {
      layer_input_[epoch][0].resize(1 + num_cells + input_size);
      for (unsigned int i = 0; i < num_layers; ++i) {
        layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
      }
    }
    for (unsigned int i = 0; i < num_layers; ++i) {
      layers_.emplace_back(layer_input_[0][i].size() + output_size, input_size_, output_size_,num_cells, horizon, gradient_clip, learning_rate,update_limit);
    }
  }

  ~Lstm() {}

  NOINLINE
  void SetInput(const std::valarray<float>& input) {
    for (unsigned int i = 0; i < layers_.size(); ++i) {
      std::copy(begin(input), begin(input) + input_size_, begin(layer_input_[epoch_][i]));
    }
  }

  NOINLINE
  std::valarray<float>& Perceive(unsigned int input) {
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
      output_layer_[epoch_][i] = output_layer_[last_epoch][i];
      output_layer_[epoch_][i] -= learning_rate_ * error * hidden_;
    }
    return Predict(input);
  }

  NOINLINE
  std::valarray<float>& Predict(unsigned int input) {
    for (unsigned int i = 0; i < layers_.size(); ++i) {
      auto start = begin(hidden_) + i * num_cells_;
      std::copy(start, start + num_cells_, begin(layer_input_[epoch_][i]) +
          input_size_);
      layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
          num_cells_);
      if (i < layers_.size() - 1) {
        auto start2 = begin(layer_input_[epoch_][i + 1]) + num_cells_ +
            input_size_;
        std::copy(start, start + num_cells_, start2);
      }
    }
    for (unsigned int i = 0; i < output_size_; ++i) {
      float sum = 0;
      for (unsigned int j = 0; j < hidden_.size(); ++j) {
        sum += hidden_[j] * output_layer_[epoch_][i][j];
      }
      output_[epoch_][i] = exp(sum);
    }
    output_[epoch_] /= output_[epoch_].sum();
    int epoch = epoch_;
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
    last_input_ = input;
    return output_[epoch];
  }

 private:
  std::vector<LstmLayer> layers_;
  std::vector<uint8_t> input_history_;
  std::valarray<float> hidden_, hidden_error_;
  std::valarray<std::valarray<std::valarray<float>>> layer_input_,
      output_layer_;
  std::valarray<std::valarray<float>> output_;
  float learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  int last_input_ = -1;
};
//--- #include "byte-model.hpp"

class Byte_Model {
 public:
  virtual ~Byte_Model() {}

  Byte_Model(char* vocab) : outputs_(0.5, 1), ex(0), top_(255), mid_(0),
      bot_(0), vocab_(vocab), probs_(1.0 / 256, 256) {}

  const std::valarray<float>& Predict() const {return outputs_;}
  unsigned int NumOutputs() {return outputs_.size();}

  std::valarray<float>& Predict() {
    auto mid = bot_ + ((top_ - bot_) / 2);
    float num = std::accumulate(&probs_[mid + 1], &probs_[top_ + 1], 0.0f);
    float denom = std::accumulate(&probs_[bot_], &probs_[mid + 1], num);
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

  const std::valarray<float>& BytePredict() {
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
  mutable std::valarray<float> outputs_;
  int top_, mid_, bot_;
  char* vocab_;
  std::valarray<float> probs_;
};

//--- #include "ppmd-model.hpp"

class PPMD : public Byte_Model {
 public:

  NOINLINE
  PPMD(int order, int memory, char* vocab) : Byte_Model(vocab) {
    ppmd_model_.reset(new ppmd_Model());
    ppmd_model_->Init(order,memory,1,0);
  }

  ~PPMD() {
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
    probs_ /= probs_.sum();
  }

 private:
  std::unique_ptr<ppmd_Model> ppmd_model_;
};

//--- #include "model.hpp"

struct Model {
  int byte_map_[256];
  float probs_[256];
  Lstm* lstm_;
  char* vocab_;

  Model( char* vocab, Lstm* lstm ) {
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

int main( int argc, char** argv ) {

  // Initialize parameters with defaults
  int ppmd_order = 9;
  int ppmd_memory = 1000;
  int lstm_input_size = 128;
  int lstm_num_cells = 90;
  int lstm_num_layers = 2;
  int lstm_horizon = 73;
  float lstm_learning_rate = 7200;
  float lstm_gradient_clip = 2.0f;
  int update_limit = 3000;

auto print_usage = [&](const char* program_name) {
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
"  ppmd_order=<n>           PPMD model order (default: %i)\n"
"  ppmd_memory=<n>          PPMD memory in MB (default: %i)\n"
"  lstm_input_size=<n>      LSTM input layer size (default: %i)\n"
"  lstm_num_cells=<n>       LSTM number of cells (default: %i)\n"
"  lstm_num_layers=<n>      LSTM number of layers (default: %i)\n"
"  lstm_horizon=<n>         LSTM horizon (default: %i)\n"
"  lstm_learning_rate=<f>   LSTM learning rate (default: %.1f)\n"
"  lstm_gradient_clip=<f>   LSTM gradient clip (default: %.1f)\n"
"  update_limit=<n>         Update limit for Adam optimizer (default: %i)\n"
"\n"
"Examples:\n"
"  %s e input.txt output.compressed\n"
"  %s d output.compressed restored.txt\n"
"  %s e input.txt output.compressed ppmd_order=9 lstm_num_layers=1\n"
"  %s e input.txt output.compressed 10 800 100 80 %i %i %.1f %.1f %i\n",
  program_name, 
  ppmd_order, ppmd_memory, lstm_input_size, lstm_num_cells, lstm_num_layers, lstm_horizon, lstm_learning_rate, lstm_gradient_clip, update_limit,
  program_name, program_name, program_name, program_name,
  lstm_num_layers, lstm_horizon, lstm_learning_rate, lstm_gradient_clip, update_limit
  );
};

  if( argc<4 || (argc>=2 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) ) {
    print_usage(argv[0]);
    return (argc>=2 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) ? 0 : 1;
  }

  uint f_DEC = (argv[1][0]=='d');
  FILE* f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  FILE* g = fopen(argv[3],"wb"); if( g==0 ) return 3;

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
      } else if (strcmp(key, "update_limit") == 0) {
        update_limit = atoi(value);
        positional_index = 9;
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
        case 8: update_limit = atoi(arg); break;
      }
      positional_index++;
    }
  }

  // Print parsed parameters
  printf("Parameters: ppmd_order=%d ppmd_memory=%d lstm_input_size=%d lstm_num_cells=%d lstm_num_layers=%d lstm_horizon=%d lstm_learning_rate=%.3f lstm_gradient_clip=%.3f update_limit=%d\n",
         ppmd_order, ppmd_memory, lstm_input_size, lstm_num_cells, lstm_num_layers, lstm_horizon, lstm_learning_rate, lstm_gradient_clip, update_limit);

  lstm_learning_rate /= 100000;

  // Set global UPDATE_LIMIT from command line parameter
  //UPDATE_LIMIT = update_limit;

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

auto byte_model_ = new PPMD(ppmd_order, ppmd_memory, cmap);

byte_model_->Byte_Model::ByteUpdate();

  srand(0xDEADBEEF);
  //ByteModel* PM = new ByteModel( cmap, new Lstm(0, total, 90, 3, 10, 0.05, 2) );
  //ByteModel* PM = new ByteModel( cmap, new Lstm(total, total, 90, 3, 10, 0.05, 2) );
  Model* PM = new Model( cmap, new Lstm(lstm_input_size, total, lstm_num_cells, lstm_num_layers, lstm_horizon, lstm_learning_rate, lstm_gradient_clip, update_limit) );
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

byte_model_->ByteUpdate(c);

const std::valarray<float>& p = byte_model_->BytePredict();
PM->lstm_->SetInput(p);

    PM->Update( c );

//if( ftell(rc.f)>(1<<20) ) break;
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}
