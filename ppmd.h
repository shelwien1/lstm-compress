#ifndef PPMD_H
#define PPMD_H

// byte-model.h is now in lstm-model.inc, which is included before this file
#include <memory>
#include <valarray>
#include <vector>

class Byte_Model;

namespace PPMD {

struct ppmd_Model;

class PPMD : public Byte_Model {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab);
  ~PPMD();
  void ByteUpdate();
 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::valarray<int> byte_map_;
};

} // namespace PPMD

#endif

