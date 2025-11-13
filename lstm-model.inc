
#define byte byte1
#undef byte

using namespace std;

#define UPDATE_LIMIT 3000
#define FAST_TANH tanh
#define FAST_TANH_VEC tanh

// ============================================================================
// BaseModel class
// ============================================================================

class BaseModel {
 public:
  BaseModel() : outputs_(0.5, 1) {}
  BaseModel(int size) : outputs_(0.5, size) {}
  ~BaseModel() {}
  const std::valarray<float>& Predict() const {return outputs_;}
  unsigned int NumOutputs() {return outputs_.size();}
  void Perceive(int bit) {}
  void ByteUpdate() {}

 protected:
  mutable std::valarray<float> outputs_;
};

// ============================================================================
// Sigmoid class
// ============================================================================

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

// ============================================================================
// Adam optimizer (helper function)
// ============================================================================

namespace {
inline void Adam(std::valarray<float>* g, std::valarray<float>* m,
    std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
    float t) {
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
}

// ============================================================================
// NeuronLayer struct
// ============================================================================

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

// ============================================================================
// LstmLayer class
// ============================================================================

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate) :
      state_(num_cells), state_error_(num_cells), stored_error_(num_cells),
      tanh_state_(std::valarray<float>(num_cells), horizon),
      input_gate_state_(std::valarray<float>(num_cells), horizon),
      last_state_(std::valarray<float>(num_cells), horizon),
      gradient_clip_(gradient_clip), learning_rate_(learning_rate),
      num_cells_(num_cells), epoch_(0), horizon_(horizon),
      input_size_(auxiliary_input_size), output_size_(output_size),
      forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
      input_node_(input_size, num_cells, horizon, output_size_ + input_size_),
      output_gate_(input_size, num_cells, horizon, output_size_ + input_size_) {
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
      input_node_.state_[epoch_][i] = FAST_TANH(input_node_.state_[epoch_][i]);
      output_gate_.state_[epoch_][i] = Sigmoid::Logistic(
          output_gate_.state_[epoch_][i]);
    }
    input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
    state_ *= forget_gate_.state_[epoch_];
    state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
    tanh_state_[epoch_] = FAST_TANH_VEC(state_);
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
  std::valarray<std::valarray<float>> tanh_state_, input_gate_state_,
      last_state_;
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;

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

// ============================================================================
// Lstm class
// ============================================================================

class Lstm {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip) : input_history_(horizon),
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
      layers_.emplace_back(layer_input_[0][i].size() + output_size, input_size_, output_size_,
          num_cells, horizon, gradient_clip, learning_rate);
    }
  }

  ~Lstm() {}

  void SetInput(const std::valarray<float>& input) {
    for (unsigned int i = 0; i < layers_.size(); ++i) {
      std::copy(begin(input), begin(input) + input_size_, begin(layer_input_[epoch_][i]));
    }
  }

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

// ============================================================================
// Byte_Model class
// ============================================================================

class Byte_Model : public BaseModel {
 public:
  virtual ~Byte_Model() {}

  Byte_Model(const std::vector<bool>& vocab) : ex(0), top_(255), mid_(0),
      bot_(0), vocab_(vocab), probs_(1.0 / 256, 256) {}

  const std::valarray<float>& BytePredict() {
    return probs_;
  }

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

  void ByteUpdate() {
    top_ = 255;
    bot_ = 0;
    for (int i = 0; i < 256; ++i) {
      if (!vocab_[i]) probs_[i] = 0;
    }
  }

  int ex;

 protected:
  int top_, mid_, bot_;
  const std::vector<bool>& vocab_;
  std::valarray<float> probs_;
};

// ============================================================================
// PPMD class
// ============================================================================

// byte-model.h is now in lstm-model.inc, which is included before this file

#include "ppmd.hpp"

namespace PPMD {

class PPMD : public Byte_Model {
 public:
  PPMD(int order, int memory, const std::vector<bool>& vocab) : Byte_Model(vocab) {
    ppmd_model_.reset(new ppmd_Model());
    ppmd_model_->Init(order,memory,1,0);
  }

  ~PPMD() {
  }

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

} // namespace PPMD

// ============================================================================
// Model struct
// ============================================================================

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

