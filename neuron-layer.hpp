
#pragma once

#include <valarray>

struct NeuronLayer {
  std::valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::valarray<std::valarray<float>> weights_, state_, update_, m_, v_,
      transpose_, norm_;
  unsigned int num_cells_, input_size_, transpose_size_;
  int horizon_;
  unsigned int input_array_size_;

  void Init(unsigned int input_size, unsigned int num_cells, int horizon,
            int offset, unsigned int input_array_size) {
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
    for (unsigned int i = 0; i < num_cells; ++i) {
      weights_[i].resize(input_size);
    }
    state_.resize(horizon);
    for (int i = 0; i < horizon; ++i) {
      state_[i].resize(num_cells);
    }
    update_.resize(num_cells);
    for (unsigned int i = 0; i < num_cells; ++i) {
      update_[i].resize(input_size);
    }
    m_.resize(num_cells);
    for (unsigned int i = 0; i < num_cells; ++i) {
      m_[i].resize(input_size);
    }
    v_.resize(num_cells);
    for (unsigned int i = 0; i < num_cells; ++i) {
      v_[i].resize(input_size);
    }
    transpose_.resize(input_size - offset);
    for (int i = 0; i < input_size - offset; ++i) {
      transpose_[i].resize(num_cells);
    }
    norm_.resize(horizon);
    for (int i = 0; i < horizon; ++i) {
      norm_[i].resize(num_cells);
    }
  }

  void Adam(std::valarray<float>* g, std::valarray<float>* m, std::valarray<float>* v,
            std::valarray<float>* w, float learning_rate, unsigned long long t, unsigned int update_limit) {
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

  void ForwardPass(const std::valarray<float>& input, unsigned int input_symbol,
                   unsigned int num_cells, unsigned int output_size, unsigned int epoch) {
    for (unsigned int i = 0; i < num_cells; ++i) {
      float f = weights_[i][input_symbol];
      for (unsigned int j = 0; j < input_array_size_; ++j) {
        f += input[j] * weights_[i][output_size + j];
      }
      norm_[epoch][i] = f;
    }
    ivar_[epoch] = 1.0f / sqrt(((norm_[epoch] * norm_[epoch]).sum() / num_cells) + 1e-5f);
    norm_[epoch] *= ivar_[epoch];
    state_[epoch] = norm_[epoch] * gamma_ + beta_;
  }

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
                    unsigned int update_limit) {
    if (epoch == horizon - 1) {
      gamma_u_ = 0;
      beta_u_ = 0;
      for (unsigned int i = 0; i < num_cells; ++i) {
        update_[i] = 0;
        unsigned int offset = output_size + input_size;
        for (unsigned int j = 0; j < transpose_size_; ++j) {
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
    std::slice slice = std::slice(output_size, input_array_size_, 1);
    for (unsigned int i = 0; i < num_cells; ++i) {
      update_[i][slice] += error_[i] * input;
      update_[i][input_symbol] += error_[i];
    }
    if (epoch == 0) {
      for (unsigned int i = 0; i < num_cells; ++i) {
        Adam(&update_[i], &m_[i], &v_[i], &weights_[i], learning_rate, update_steps, update_limit);
      }
      Adam(&gamma_u_, &gamma_m_, &gamma_v_, &gamma_, learning_rate, update_steps, update_limit);
      Adam(&beta_u_, &beta_m_, &beta_v_, &beta_, learning_rate, update_steps, update_limit);
    }
  }
};
