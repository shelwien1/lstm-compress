#ifndef PPMD_H
#define PPMD_H

// byte-model.h is now in lstm-model.inc, which is included before this file
#include <memory>
#include <valarray>
#include <vector>

class Byte_Model;

namespace PPMD {

struct ppmd_Model;
extern unsigned long long counter_;

class PPMD : public Byte_Model {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab) : Byte_Model(vocab), byte_(bit_context) {
    ppmd_model_.reset(new ppmd_Model());
    ppmd_model_->Init(order,memory,1,0);
  }

  ~PPMD() {
  }

  void ByteUpdate() {
    ++counter_;
    ppmd_model_->ppmd_UpdateByte( byte_&0xFF );
    ppmd_model_->ppmd_PrepareByte();
    for (int i = 0; i < 256; ++i) {
      probs_[i] = ppmd_model_->sqp[i];
      if (probs_[i] < 1) probs_[i] = 1;
    }
    Byte_Model::ByteUpdate();
    probs_ /= probs_.sum();
  }

 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::valarray<int> byte_map_;
};

} // namespace PPMD

#endif

