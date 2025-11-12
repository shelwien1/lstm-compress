
#define UPDATE_LIMIT 3000

struct NeuronLayer {
  NeuronLayer(uint input_size, uint num_cells, int horizon, int offset) : 
    error_(num_cells), ivar_(horizon), gamma_(1.0, num_cells),
    gamma_u_(num_cells), gamma_m_(num_cells), gamma_v_(num_cells),
    beta_(num_cells), beta_u_(num_cells), beta_m_(num_cells),
    beta_v_(num_cells), weights_(valarray<float>(input_size), num_cells),
    state_(valarray<float>(num_cells), horizon),
    update_(valarray<float>(input_size), num_cells),
    m_(valarray<float>(input_size), num_cells),
    v_(valarray<float>(input_size), num_cells),
    transpose_(valarray<float>(num_cells), input_size - offset),
    norm_(valarray<float>(num_cells), horizon) {}

  valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_, beta_, beta_u_, beta_m_, beta_v_;
  valarray<valarray<float>> weights_, state_, update_, m_, v_, transpose_, norm_;
};

static inline void Adam(valarray<float>* g, valarray<float>* m, valarray<float>* v, valarray<float>* w, 
    float learning_rate, float t) {
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

struct LstmLayer {
  uint num_cells_, epoch_, horizon_, inp_size_, out_size_;
  float learning_rate_, gradient_clip_;
  qword update_steps_;

  valarray<float> state_, state_err_, stored_err_;
  valarray<valarray<float>> tanh_state_, inp_gate_state_, last_state_;
  NeuronLayer fgt_gate_, inp_node_, out_gate_;

  static inline float Rand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  LstmLayer(uint inp_size, uint auxiliary_inp_size, uint out_size, uint num_cells, int horizon,
      float learning_rate, float gradient_clip) : 
    state_(num_cells), state_err_(num_cells), stored_err_(num_cells),
    tanh_state_(valarray<float>(num_cells), horizon),
    inp_gate_state_(valarray<float>(num_cells), horizon),
    last_state_(valarray<float>(num_cells), horizon),
    learning_rate_(learning_rate), gradient_clip_(gradient_clip),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    inp_size_(auxiliary_inp_size), out_size_(out_size), update_steps_(0),
    fgt_gate_(inp_size, num_cells, horizon, out_size_ + inp_size_),
    inp_node_(inp_size, num_cells, horizon, out_size_ + inp_size_),
    out_gate_(inp_size, num_cells, horizon, out_size_ + inp_size_)
  {
    float val = sqrt(6.0f / float(inp_size_ + out_size_));
    float low = -val;
    float range = 2 * val;
    for (uint i = 0; i < num_cells_; ++i) {
      for (uint j = 0; j < fgt_gate_.weights_[i].size(); ++j) {
        fgt_gate_.weights_[i][j] = low + Rand() * range;
        inp_node_.weights_[i][j] = low + Rand() * range;
        out_gate_.weights_[i][j] = low + Rand() * range;
      }
      fgt_gate_.weights_[i][fgt_gate_.weights_[i].size() - 1] = 1;
    }
  }

  void ForwardPass(const valarray<float>& inp, int inp_symbol, valarray<float>* hidden, int hidden_start) {
    last_state_[epoch_] = state_;
    ForwardPassNeurons(fgt_gate_, inp, inp_symbol);
    ForwardPassNeurons(inp_node_, inp, inp_symbol);
    ForwardPassNeurons(out_gate_, inp, inp_symbol);
    for (uint i = 0; i < num_cells_; ++i) {
      fgt_gate_.state_[epoch_][i] = Sigmoid::Logistic(fgt_gate_.state_[epoch_][i]);
      inp_node_.state_[epoch_][i] = tanh(inp_node_.state_[epoch_][i]);
      out_gate_.state_[epoch_][i] = Sigmoid::Logistic(out_gate_.state_[epoch_][i]);
    }
    inp_gate_state_[epoch_] = 1.0f - fgt_gate_.state_[epoch_];
    state_ *= fgt_gate_.state_[epoch_];
    state_ += inp_node_.state_[epoch_] * inp_gate_state_[epoch_];
    tanh_state_[epoch_] = tanh(state_);
    std::slice slice = std::slice(hidden_start, num_cells_, 1);
    (*hidden)[slice] = out_gate_.state_[epoch_] * tanh_state_[epoch_];
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
  }

  void ForwardPassNeurons(NeuronLayer& neurons, const valarray<float>& inp, int inp_symbol) {
    for (uint i = 0; i < num_cells_; ++i) {
      float f = neurons.weights_[i][inp_symbol];
      for (uint j = 0; j < inp.size(); ++j) {
        f += inp[j] * neurons.weights_[i][out_size_ + j];
      }
      neurons.norm_[epoch_][i] = f;
    }
    neurons.ivar_[epoch_] = 1.0f / sqrt(((neurons.norm_[epoch_] * neurons.norm_[epoch_]).sum() / 
      num_cells_) + 1e-5f);
    neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
    neurons.state_[epoch_] = neurons.norm_[epoch_] * neurons.gamma_ + neurons.beta_;
  }

  void ClipGradients(valarray<float>* arr) {
    for (uint i = 0; i < arr->size(); ++i) {
      if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
      else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
    }
  }

  void BackwardPass(const valarray<float>&inp, int epoch, int layer, int inp_symbol, 
      valarray<float>* hidden_err) {
    if (epoch == (int)horizon_ - 1) {
      stored_err_ = *hidden_err;
      state_err_ = 0;
    } else {
      stored_err_ += *hidden_err;
    }

    out_gate_.error_ = tanh_state_[epoch] * stored_err_ * out_gate_.state_[epoch] * 
      (1.0f - out_gate_.state_[epoch]);
    state_err_ += stored_err_ * out_gate_.state_[epoch] * (1.0f - (tanh_state_[epoch] * tanh_state_[epoch]));
    inp_node_.error_ = state_err_ * inp_gate_state_[epoch] * (1.0f - (inp_node_.state_[epoch] * 
      inp_node_.state_[epoch]));
    fgt_gate_.error_ = (last_state_[epoch] - inp_node_.state_[epoch]) * state_err_ * 
      fgt_gate_.state_[epoch] * inp_gate_state_[epoch];

    *hidden_err = 0;
    if (epoch > 0) {
      state_err_ *= fgt_gate_.state_[epoch];
      stored_err_ = 0;
    } else {
      if (update_steps_ < UPDATE_LIMIT) {
        ++update_steps_;
      }
    }

    BackwardPassNeurons(fgt_gate_, inp, epoch, layer, inp_symbol, hidden_err);
    BackwardPassNeurons(inp_node_, inp, epoch, layer, inp_symbol, hidden_err);
    BackwardPassNeurons(out_gate_, inp, epoch, layer, inp_symbol, hidden_err);

    ClipGradients(&state_err_);
    ClipGradients(&stored_err_);
    ClipGradients(hidden_err);
  }

  void BackwardPassNeurons(NeuronLayer& neurons, const valarray<float>&inp, int epoch, int layer, 
      int inp_symbol, valarray<float>* hidden_err) {
    if (epoch == (int)horizon_ - 1) {
      neurons.gamma_u_ = 0;
      neurons.beta_u_ = 0;
      for (uint i = 0; i < num_cells_; ++i) {
        neurons.update_[i] = 0;
        int offset = out_size_ + inp_size_;
        for (uint j = 0; j < neurons.transpose_.size(); ++j) {
          neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
        }
      }
    }
    neurons.beta_u_ += neurons.error_;
    neurons.gamma_u_ += neurons.error_ * neurons.norm_[epoch];
    neurons.error_ *= neurons.gamma_ * neurons.ivar_[epoch];
    neurons.error_ -= ((neurons.error_ * neurons.norm_[epoch]).sum() / num_cells_) * neurons.norm_[epoch];
    if (layer > 0) {
      for (uint i = 0; i < num_cells_; ++i) {
        float f = 0;
        for (uint j = 0; j < num_cells_; ++j) {
          f += neurons.error_[j] * neurons.transpose_[num_cells_ + i][j];
        }
        (*hidden_err)[i] += f;
      }
    }
    if (epoch > 0) {
      for (uint i = 0; i < num_cells_; ++i) {
        float f = 0;
        for (uint j = 0; j < num_cells_; ++j) {
          f += neurons.error_[j] * neurons.transpose_[i][j];
        }
        stored_err_[i] += f;
      }
    }
    std::slice slice = std::slice(out_size_, inp.size(), 1);
    for (uint i = 0; i < num_cells_; ++i) {
      neurons.update_[i][slice] += neurons.error_[i] * inp;
      neurons.update_[i][inp_symbol] += neurons.error_[i];
    }
    if (epoch == 0) {
      for (uint i = 0; i < num_cells_; ++i) {
        Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i], &neurons.weights_[i], learning_rate_, 
          update_steps_);
      }
      Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_, &neurons.gamma_, learning_rate_, 
        update_steps_);
      Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_, &neurons.beta_, learning_rate_, 
        update_steps_);
    }
  }
};
