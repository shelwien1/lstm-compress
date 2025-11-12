
struct Lstm {
  float learning_rate_;
  uint num_cells_;
  uint epoch_;
  uint horizon_;
  uint inp_size_;
  uint out_size_;
  int last_inp_;

  vector<unique_ptr<LstmLayer>> layers_;
  vector<byte> inp_history_;
  valarray<float> hidden_;
  valarray<float> hidden_err_;
  valarray<valarray<valarray<float>>> layer_inp_;
  valarray<valarray<valarray<float>>> out_layer_;
  valarray<valarray<float>> out_;

  Lstm(uint inp_size, uint out_size, uint num_cells, uint num_layers, int horizon, float learning_rate, 
      float gradient_clip) : 
    inp_history_(horizon),
    hidden_(num_cells * num_layers + 1), hidden_err_(num_cells),
    layer_inp_(valarray<valarray<float>>(valarray<float>(inp_size + 1 + num_cells * 2), num_layers), horizon),
    out_layer_(valarray<valarray<float>>(valarray<float>(num_cells * num_layers + 1), out_size), horizon),
    out_(valarray<float>(1.0 / out_size, out_size), horizon),
    learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
    horizon_(horizon), inp_size_(inp_size), out_size_(out_size), last_inp_(-1)
  {
    hidden_[hidden_.size() - 1] = 1;
    for (int epoch = 0; epoch < horizon; ++epoch) {
      layer_inp_[epoch][0].resize(1 + num_cells + inp_size);
      for (uint i = 0; i < num_layers; ++i) {
        layer_inp_[epoch][i][layer_inp_[epoch][i].size()-1] = 1;
      }
    }
    for (uint i = 0; i < num_layers; ++i) {
      layers_.push_back(unique_ptr<LstmLayer>(new LstmLayer(
        layer_inp_[0][i].size() + out_size, inp_size_, out_size_,
        num_cells, horizon, learning_rate, gradient_clip)));
    }
  }

  void Setinp(int index, float val) {
    for (uint i = 0; i < layers_.size(); ++i) layer_inp_[epoch_][i][index] = val;
  }

  valarray<float>& Perceive(uint inp) {
    int last_epoch = epoch_ - 1;
    if (last_epoch == -1) last_epoch = horizon_ - 1;
    int old_inp = inp_history_[last_epoch];
    inp_history_[last_epoch] = inp;
    if (epoch_ == 0) {
      for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
        for (int layer = layers_.size() - 1; layer >= 0; --layer) {
          int offset = layer * num_cells_;
          for (uint i = 0; i < out_size_; ++i) {
            float err = (i == inp_history_[epoch]) ? (out_[epoch][i] - 1) : out_[epoch][i];
            for (uint j = 0; j < hidden_err_.size(); ++j) {
              hidden_err_[j] += out_layer_[epoch][i][j + offset] * err;
            }
          }
          int prev_epoch = epoch - 1;
          if (prev_epoch == -1) prev_epoch = horizon_ - 1;
          int inp_symbol = inp_history_[prev_epoch];
          if (epoch == 0) inp_symbol = old_inp;
          layers_[layer]->BackwardPass(layer_inp_[epoch][layer], epoch, layer, inp_symbol, &hidden_err_);
        }
      }
    }

    for (uint i = 0; i < out_size_; ++i) {
      float err = (i == inp) ? (out_[last_epoch][i] - 1) : out_[last_epoch][i];
      out_layer_[epoch_][i] = out_layer_[last_epoch][i];
      out_layer_[epoch_][i] -= learning_rate_ * err * hidden_;
    }
    return Predict(inp);
  }

  valarray<float>& Predict(uint inp) {
    for (uint i = 0; i < layers_.size(); ++i) {
      auto start = begin(hidden_) + i * num_cells_;
      copy(start, start + num_cells_, begin(layer_inp_[epoch_][i]) + inp_size_);
      layers_[i]->ForwardPass(layer_inp_[epoch_][i], inp, &hidden_, i * num_cells_);
      if (i < layers_.size() - 1) {
        auto start2 = begin(layer_inp_[epoch_][i + 1]) + num_cells_ + inp_size_;
        copy(start, start + num_cells_, start2);
      }
    }
    for (uint i = 0; i < out_size_; ++i) {
      float sum = 0;
      for (uint j = 0; j < hidden_.size(); ++j) {
        sum += hidden_[j] * out_layer_[epoch_][i][j];
      }
      out_[epoch_][i] = exp(sum);
    }
    out_[epoch_] /= out_[epoch_].sum();
    int epoch = epoch_;
    ++epoch_;
    if (epoch_ == horizon_) epoch_ = 0;
    last_inp_ = inp;
    return out_[epoch];
  }
};
