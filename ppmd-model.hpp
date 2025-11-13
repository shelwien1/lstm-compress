
#pragma once

#include <memory>
#include <vector>
#include <valarray>
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
