# KANNCOMPR.CPP - Code Structure Analysis

## Overview
This file (2518 lines) contains a hybrid C/C++ neural network compression library. It's partially refactored to use modern C++ classes and templates while maintaining backward compatibility with C-style function pointers.

---

## 1. MAIN STRUCTURES/CLASSES DEFINED

### A. Core Neural Network Structures

#### **struct kad_node_t** (lines 52-127)
- **Purpose**: Represents a computational graph node (KAD = Kann Automatic Differentiation)
- **Key Members**:
  - `n_d`: Number of dimensions
  - `flag`: Node flags (VAR, CONST, POOL, etc.)
  - `op`: Operation type (ID into kad_op_list)
  - `n_child`: Number of child nodes
  - `d[4]`: Dimension array
  - `float *x`: Forward value buffer
  - `float *g`: Gradient buffer
  - `kad_node_t **child`: Child node pointers
  - `kad_node_t *pre`: Previous node (for RNN recurrence)

- **Instance Methods**:
  - `bool is_var()`, `is_const()`, `is_feed()`, `is_pivot()`, `is_switch()`
  - `void eval_enable()`, `eval_disable()`
  - `int length()` - compute tensor size

- **Static Factory Methods**:
  - `new_core()`, `vleaf()`, `const_node()`, `feed()`, `var()`
  - `finalize_node()`, `op1_core()`, `op2_core()`
  - `pooling_general()`, `concat_array()`, `switch_node()`, `dup1()`

- **Static Binary Operations**: `add()`, `sub()`, `mul()`, `cmul()`, `matmul()`, `ce_multi()`, `ce_bin()`, `mse()`
- **Static Unary Operations**: `log()`, `exp()`, `sin()`, `square()`, `sigm()`, `tanh_op()`, `relu()`, `softmax()`, `stdnorm()`

#### **struct kann_t** (lines 248-260)
- **Purpose**: Neural network container wrapping compiled KAD graph
- **Key Members**:
  - `int n`: Number of nodes in graph
  - `kad_node_t **v`: Array of nodes
  - `float *x`: Combined variable buffer
  - `float *g`: Combined gradient buffer
  - `float *c`: Combined constant buffer
  - `void *mt`: Metadata pointer

- **Instance Methods**:
  - `int size_var()`, `size_const()`
  - `void set_batch_size(int B)`

#### **template<typename T> struct kvec_t** (lines 863-889)
- **Purpose**: Dynamic vector container (C++ replacement for macro-based kvec_t)
- **Methods**: `push()`, `pop()`, `release()`, destructor
- **Used for**: Graph compilation, node lists

#### **typedef struct kanncompr_t** (lines 1898-1929)
- **Purpose**: Main compression state object containing model and training state
- **Key Members**:
  - `kann_t *ann`: Trained neural network
  - `kann_t *ua`: Unrolled network for sequence processing
  - `int n_layers, n_neurons, ulen`: Architecture parameters
  - `float **x, **y`: Training data arrays
  - `float *m, *v`: Adam optimizer momentum/variance
  - `word mini_batch_size, mini_batch_step`: Training parameters
  - `byte *symb_list`: Symbol frequency list
  - `float alpha1, beta1, beta2, eps`: Adam parameters

#### **typedef struct stats_t** (lines 1931-1938)
- **Purpose**: Statistics tracking for compression progress
- **Members**: counters for original data, compression ratio display

#### **struct Rangecoder** (lines 1788-1892)
- **Purpose**: Arithmetic coding implementation
- **Members**:
  - `uint low, Carry`: Encoding state
  - `uint code`: Decoding state
  - `qword range`: Frequency range
  - `FILE *f`: File pointer
  - `uint f_DEC`: Encode/decode flag

- **Methods**: `StartEncode()`, `StartDecode()`, `rc_Process()`, `ShiftCode()`, `ShiftLow()`, `rc_Renorm()`, `rc_Quit()`

---

### B. Helper Classes (All Static Methods)

#### **class KadGraph** (lines 132-151)
Graph compilation and manipulation - wraps C-style graph algorithms
- **Static Methods**:
  - `compile_array()`, `compile()` - Build computation graph
  - `delete_graph()` - Free graph memory
  - `eval_at()`, `eval_marked()` - Forward pass execution
  - `grad()` - Backward pass (gradient computation)
  - `sync_dim()`, `allocate_internal()` - Memory management
  - `mark_back()`, `propagate_marks()` - Gradient flow marking
  - `unroll()` - Unroll RNN for fixed sequence length
  - `n_pivots()` - Count RNN pivot nodes
  - `size_var()`, `size_const()` - Get parameter counts
  - `ext_collate()`, `ext_sync()` - External memory management
  - `copy_dim1()` - Copy dimension info

#### **class KadRng** (lines 204-215)
Random number generation (xoroshiro128+ algorithm)
- **Static Methods**: `create()`, `seed()`, `rand()`, `drand()`, `drand_normal()`
- **Private**: `splitmix64()`, `xoroshiro128plus_next()`, `xoroshiro128plus_jump()`
- **Wraps**: Global `kad_rng_t` struct and `kad_rng_dat` variable

#### **class KadMath** (lines 218-226)
Vector and matrix operations (SIMD optimized)
- **Static Methods**:
  - `saxpy()` - Single precision a*x + y (SIMD)
  - `sdot()` - Dot product with SIMD
  - `vec_mul_sum()` - Element-wise multiply and sum
  - `sgemm_simple()` - General matrix multiply

---

## 2. GLOBAL FUNCTIONS (Struct Pointer Arguments Pattern)

### A. KANN Network Management Functions
```cpp
kann_t *kann_new(kad_node_t *cost, int n_rest, ...);
kann_t *kann_unroll(kann_t *a, ...);
kann_t *kann_unroll_array(kann_t *a, int *len);
void kann_delete(kann_t *a);
void kann_delete_unrolled(kann_t *a);
```

### B. KANN Feed/Forward Functions
```cpp
int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x);
float kann_cost(kann_t *a, int cost_label, int cal_grad);
int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label);
const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label);
const float *kann_apply1(kann_t *a, float *x);
```

### C. KANN RNN Functions
```cpp
void kann_rnn_start(kann_t *a);      // Initialize RNN state
void kann_rnn_end(kann_t *a);        // Finalize RNN state
```

### D. KANN Training Functions
```cpp
void kann_switch(kann_t *a, int is_train);  // Switch train/inference mode
float kann_grad_clip(float thres, int n, float *g);
void kann_switch_core(kann_t *a, int is_train);  // Implementation
float kann_cost_core(kann_t *a, int cost_label, int cal_grad);  // Implementation
```

### E. Layer Construction Functions
```cpp
kad_node_t *kann_layer_input(int n1);
kad_node_t *kann_layer_dense(kad_node_t *in, int n1);
kad_node_t *kann_layer_layernorm(kad_node_t *in);
kad_node_t *kann_layer_dropout(kad_node_t *t, float r);
```

### F. KANNCOMPR Specific Functions (Domain-specific)
```cpp
void ann_init(kanncompr_t *options);              // Initialize model & buffers
void ann_predict(kanncompr_t *options, ...);      // Single symbol prediction
void ann_train(kanncompr_t *options);             // Training step
void ann_end(kanncompr_t options);                // Cleanup (pass by value)
kann_t *ann_structure(kanncompr_t options);       // Build network topology
```

### G. File I/O Helper Functions (Pattern: many small I/O functions)
```cpp
int fput_ui08(FILE *file, byte value);
int fget_ui08(FILE *file, byte *value);
int fput_ui16(FILE *file, word value);
int fget_ui16(FILE *file, word *value);
int fput_ui32(FILE *file, uint value);
int fget_ui32(FILE *file, uint *value);
int fput_fl(FILE *file, float value);
int fget_fl(FILE *file, float *value);
// ... plus int32 and int variants
```

### H. Utility/Display Functions
```cpp
void display_stats(stats_t *stats, qword comprpos);
float PERC(float V, float T);     // Percentage computation
float BPB(float V, float T);      // Bits per byte computation
void Adam(...);                    // Optimizer (free function, not class)
```

---

## 3. STATIC METHODS WITH CLASS POINTERS AS ARGUMENTS

These are methods that take arrays or pointers to instances as arguments:

### KadGraph Static Methods (operating on node arrays):
```cpp
// Takes array of nodes or function pointers to nodes
static kad_node_t** compile_array(int *n_node, int n_roots, kad_node_t **roots);
static void delete_graph(int n, kad_node_t **a);                    // array of n nodes
static void eval_marked(int n, kad_node_t **a);                     // array of n nodes
static int sync_dim(int n, kad_node_t **v, int batch_size);         // array of n nodes
static void grad(int n, kad_node_t **a, int from);                  // array of n nodes
static void mark_back(int n, kad_node_t **v);                       // array of n nodes
static void allocate_internal(int n, kad_node_t **v);               // array of n nodes
static void propagate_marks(int n, kad_node_t **a);                 // array of n nodes
static void ext_collate(int n, kad_node_t **a, float **_x, ...);   // array of n nodes
static void ext_sync(int n, kad_node_t **a, float *x, ...);        // array of n nodes
```

### KadRng Static Methods (operating on opaque void* instances):
```cpp
static void seed(void *d, uint64_t seed);        // operates on RNG instance
static double drand(void *d);                     // operates on RNG instance
static double drand_normal(void *d);              // operates on RNG instance
```

### KadMath Static Methods (vector/matrix operations):
```cpp
static void saxpy(int n, float a, const float *x, float *y);  // modify y in place
static float sdot(int n, const float *x, const float *y);
static void sgemm_simple(..., const float *A, const float *B, float *C);  // modify C
```

---

## 4. PATTERNS SUGGESTING OBJECTS WITHOUT CLASSES

### A. Free Functions Operating on Opaque Pointers
```cpp
// C-style object management - no encapsulation
void kann_delete(kann_t *a);              // Manual destruction
void ann_init(kanncompr_t *options);      // Manual initialization
void ann_train(kanncompr_t *options);     // Manual operations
```

### B. Manual Memory Management Pattern
```cpp
kanncompr_t options;  // Stack allocation
options.x = (float**)malloc(options.ulen * sizeof(float*));
options.y = (float**)malloc(options.ulen * sizeof(float*));
// ... manual member access and cleanup
free(options.x); free(options.y);
```

### C. Helper Functions Without Classes
```cpp
// Utility functions that don't belong to any class
void Adam(const int n_var, const float alpha, ...);      // Optimizer algorithm
void display_stats(stats_t *stats, qword comprpos);      // Statistics display
float PERC(float V, float T);                            // Utility computation
float BPB(float V, float T);                             // Utility computation
```

### D. Array-of-Pointers Pattern (Collection without Container Class)
```cpp
// No collection class - raw arrays passed around
kad_node_t **v;  // Array of node pointers
int n;           // Count stored separately
// Functions operate: kad_delete(n, v), kad_eval_at(n, v, from), etc.
```

### E. typedef structs with No Methods (Pure Data)
```cpp
typedef struct kanncompr_s {
  // All public members, no methods
  int n_char_in, n_char_out;
  int n_layers, n_neurons;
  float **x, **y;
  // ...
} kanncompr_t;  // Used as POD (Plain Old Data)
```

### F. Function Pointer Arrays (Strategy Pattern Without Classes)
```cpp
typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[KAD_MAX_OP];  // Array of operation implementations
// Operations selected by: kad_op_list[node->op](node, mode)
```

### G. Nested Helper Functions Within Functions
```cpp
void push_nodes(nodes_t *w, kad_node_t *p) { /* ... */ }
void kad_unroll_helper(int n_v, kad_node_t **v, ...);  // Helper, not a method
```

---

## 5. HYBRID C/C++ PATTERNS IN THE CODE

### A. Backward Compatibility Wrappers
```cpp
// Modern C++ class with static methods
class KadGraph {
public:
  static void delete_graph(int n, kad_node_t **a);
};

// Legacy C-style wrapper for compatibility
void kad_delete(int n, kad_node_t **a) { 
  KadGraph::delete_graph(n, a); 
}
```

### B. Modern C++ Features Currently Used
- `constexpr` for compile-time constants (replacing macros)
- `using` type aliases (replacing typedef)
- Template structures: `kvec_t<T>`
- Static classes for namespacing
- Instance methods in structs
- Default member initializers: `uint8_t n_d{0}`

### C. Legacy C Patterns Still Present
- Manual memory allocation/deallocation
- va_list for variadic arguments
- void pointers for generic data
- Preprocessor macros (some converted, some remain)
- C-style casts
- Function callbacks via function pointers

---

## 6. KEY ARCHITECTURAL OBSERVATIONS

### Computation Graph Design
- **Lazy Evaluation**: Nodes computed on-demand via `kad_eval_at()`
- **Automatic Differentiation**: `kad_grad()` computes all gradients
- **Dynamic Shapes**: Dimensions computed during sync phase
- **Memory Pooling**: Variables share buffers via `ext_collate()/ext_sync()`

### RNN Support
- **Pivot Nodes**: Mark unrolled timesteps via `is_pivot()`
- **State Threading**: `pre` pointer chains state through time
- **Unrolling**: Fixed-length unrolling via `KadGraph::unroll()`

### Training Pipeline
1. `ann_init()` - Allocate buffers, unroll network
2. `ann_train()` - Forward pass, backward pass, gradient clipping
3. `Adam()` - Update weights with Adam optimizer
4. `ann_end()` - Cleanup

### Compression-Specific (kanncompr_t)
- **Symbol Prediction**: Single-symbol RNN codec
- **Adaptive Training**: Online learning during compression
- **Arithmetic Coding**: Via Rangecoder struct with embedded methods
- **Vocabulary Management**: Maintains symbol frequency lists

---

## 7. REFACTORING OPPORTUNITIES

### High Priority
1. **Create AnnNetwork class** wrapping `kann_t` + operations
2. **Create CompressionModel class** wrapping `kanncompr_t` + ann_init/train/predict
3. **Create NodeFactory class** for `kad_node_t` creation
4. **Create NodeArray template** replacing `(int n, kad_node_t **v)` patterns

### Medium Priority
5. **Create MatrixOperations class** for SIMD operations (consolidate KadMath)
6. **Create RandomGenerator class** replacing void* pointers
7. **Create FileSerializer class** for I/O functions
8. **Create OptimizerState class** for Adam parameters

### Low Priority (Backward Compatibility)
9. Keep C-style legacy functions as thin wrappers
10. Support both old and new APIs during transition

