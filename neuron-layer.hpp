
#pragma once

#include <valarray>

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

  void ForwardPass(const std::valarray<float>& input, unsigned int input_symbol,
                   unsigned int num_cells, unsigned int output_size, unsigned int epoch) {
    for (unsigned int i = 0; i < num_cells; ++i) {
      float f = weights_[i][input_symbol];
      for (unsigned int j = 0; j < input.size(); ++j) {
        f += input[j] * weights_[i][output_size + j];
      }
      norm_[epoch][i] = f;
    }
    ivar_[epoch] = 1.0f / sqrt(((norm_[epoch] * norm_[epoch]).sum() / num_cells) + 1e-5f);
    norm_[epoch] *= ivar_[epoch];
    state_[epoch] = norm_[epoch] * gamma_ + beta_;
  }

  template<typename AdamFunc>
  void BackwardPass(const std::valarray<float>& input,
                    unsigned int epoch,
                    unsigned int layer,
                    unsigned int input_symbol,
                    std::valarray<float>* hidden_error,
                    unsigned int num_cells,
                    unsigned int horizon,
                    unsigned int output_size,
                    unsigned int input_size,
                    std::valarray<float>& stored_error,
                    float learning_rate,
                    unsigned long long update_steps,
                    AdamFunc adam_func) {
    if (epoch == horizon - 1) {
      gamma_u_ = 0;
      beta_u_ = 0;
      for (unsigned int i = 0; i < num_cells; ++i) {
        update_[i] = 0;
        unsigned int offset = output_size + input_size;
        for (unsigned int j = 0; j < transpose_.size(); ++j) {
          transpose_[j][i] = weights_[i][j + offset];
        }
      }
    }
    beta_u_ += error_;
    gamma_u_ += error_ * norm_[epoch];
    error_ *= gamma_ * ivar_[epoch];
    error_ -= ((error_ * norm_[epoch]).sum() / num_cells) * norm_[epoch];
    if (layer > 0) {
      for (unsigned int i = 0; i < num_cells; ++i) {
        float f = 0;
        for (unsigned int j = 0; j < num_cells; ++j) {
          f += error_[j] * transpose_[num_cells + i][j];
        }
        (*hidden_error)[i] += f;
      }
    }
    if (epoch > 0) {
      for (unsigned int i = 0; i < num_cells; ++i) {
        float f = 0;
        for (unsigned int j = 0; j < num_cells; ++j) {
          f += error_[j] * transpose_[i][j];
        }
        stored_error[i] += f;
      }
    }
    std::slice slice = std::slice(output_size, input.size(), 1);
    for (unsigned int i = 0; i < num_cells; ++i) {
      update_[i][slice] += error_[i] * input;
      update_[i][input_symbol] += error_[i];
    }
    if (epoch == 0) {
      for (unsigned int i = 0; i < num_cells; ++i) {
        adam_func(&update_[i], &m_[i], &v_[i], &weights_[i], learning_rate, update_steps);
      }
      adam_func(&gamma_u_, &gamma_m_, &gamma_v_, &gamma_, learning_rate, update_steps);
      adam_func(&beta_u_, &beta_m_, &beta_v_, &beta_, learning_rate, update_steps);
    }
  }
};
