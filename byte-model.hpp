
#pragma once

#include <vector>
#include <valarray>
#include <numeric>

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
