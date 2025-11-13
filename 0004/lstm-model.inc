
#define byte byte1
#include <math.h>

#include <cstdint>
#include <vector>
#include <valarray>
#include <numeric>   // inner_product
#include <memory>    // unique_ptr
#undef byte

using namespace std;

//#include "sigmoid.inc"
//#include "lstm-layer.inc"
#include "sigmoid.h"
#include "sigmoid.hpp"

#define UPDATE_LIMIT 3000
#include "lstm.h"

#include "ppmd.h"
#include "byte-model.hpp"
#include "ppmd.hpp"

#include "model.inc"

