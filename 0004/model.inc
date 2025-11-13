
struct Model {
  int byte_map_[256];
  float probs_[256];
  Lstm* lstm_;
  char* vocab_;

  Model( char* vocab, Lstm* lstm ) {
    vocab_ = vocab; 
    lstm_ = lstm;
    int i, offset = 0;
    for( i = 0; i < 256; i++ ) {
      byte_map_[i] = offset;
      if (vocab_[i]) ++offset;
      probs_[i]=1.0/256;
    }
  }

  void Update( int sym ) {
    const auto& output = lstm_->Perceive( byte_map_[sym] );
    int i, offset = 0;
    for( i = 0; i < 256; i++ ) {
      if( vocab_[i] ) {
        probs_[i] = output[offset];
        offset++;
      } else {
        probs_[i] = 0;
      }
    }
  }

};


