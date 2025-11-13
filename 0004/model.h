#ifndef MODEL_H
#define MODEL_H

#include <valarray>

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

#endif

