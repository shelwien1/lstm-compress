
#pragma once

#include <vector>
#include <cmath>

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
