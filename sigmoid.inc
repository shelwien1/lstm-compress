
#if 0
struct Sigmoid {

  int logit_size_;
  std::vector<float> logit_table_;

  Sigmoid( int logit_size ) : 
    logit_size_(logit_size),
    logit_table_(logit_size, 0) 
  {
    for( int i = 0; i < logit_size_; ++i ) logit_table_[i] = SlowLogit((i + 0.5) / logit_size_);
  }

  float Logit( float p ) const {
    return logit_table_[p * logit_size_];
  }

  static float Logistic( float p ) {
    return 1 / (1 + exp(-p));
  }

  float SlowLogit( float p ) const {
    return log(p / (1 - p));
  }

};
#endif

namespace Sigmoid {

  float Logistic( float p ) {
    return 1 / (1 + exp(-p));
  }

}
