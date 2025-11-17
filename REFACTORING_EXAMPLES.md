# Refactoring Examples - Current vs. Modern C++ Patterns

## Current Code Patterns in kanncompr.cpp

### Pattern 1: Manual Lifecycle Management (Current)

**Problem: Free functions operating on struct pointers**

```cpp
// CURRENT - Line 2326-2360 (main function excerpt)
int main(int argc, char** argv) {
  kanncompr_t options;
  
  // Manual initialization of struct members
  options.ann = NULL;
  options.n_char_in = 256;
  options.n_char_out = 256;
  options.seed = 1;
  // ... manually set 20+ struct members ...
  
  // Create network topology
  options.ann = ann_structure(options);
  
  // Initialize buffers (separate function call)
  ann_init(&options);
  
  // Manual training loop
  for (...) {
    ann_predict(&options, freq, &total);
    // ... encoding logic ...
    ann_train(&options);
  }
  
  // Manual cleanup
  ann_end(options);  // Note: pass by value!
  fclose(filein);
  fclose(fileout);
  return 0;
}
```

**Why it's problematic:**
- Four separate functions to manage one logical object
- Easy to forget `ann_init()` or `ann_end()`
- No RAII guarantees
- Pass-by-value in `ann_end()` is unusual
- Raw pointer management (`options.x`, `options.y`, etc.)

---

### Pattern 2: Static Classes as Namespaces (Current)

**Current approach: Class with all static methods**

```cpp
// Lines 132-151
class KadGraph {
public:
  static kad_node_t** compile_array(int *n_node, int n_roots, kad_node_t **roots);
  static void delete_graph(int n, kad_node_t **a);
  static const float* eval_at(int n, kad_node_t **a, int from);
  static void grad(int n, kad_node_t **a, int from);
  // ... 20+ more static methods ...
};

// Usage:
int n_nodes;
kad_node_t** nodes = KadGraph::compile_array(&n_nodes, 1, &root);
KadGraph::eval_at(n_nodes, nodes, output_idx);
KadGraph::grad(n_nodes, nodes, output_idx);
KadGraph::delete_graph(n_nodes, nodes);
```

**Why it's problematic:**
- Not really a class - it's a namespace simulation
- No encapsulation of related state
- `(int n, kad_node_t **v)` repeated 10+ times
- Hard to add new graph types without modifying KadGraph
- No opportunity for polymorphism

---

### Pattern 3: Array-of-Pointers Without Container (Current)

**Problem: Unencapsulated dynamic arrays**

```cpp
// Lines 1145-1179 (kad_unroll function)
kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len) {
  int i, j, n_pivots = 0;
  kad_node_t **t;
  nodes_t w = {0, 0, 0};  // Manual struct initialization
  
  t = (kad_node_t**)calloc(n_v, sizeof(kad_node_t*));
  // ... complex graph manipulation ...
  push_nodes(&w, t[i]);
  // ...
  
  *new_n = w.n;  // Out parameter!
  return w.v;    // Manual memory management
}

// Called as:
int new_n;
if (n_pivots) {
  int k, *i_pivots;
  i_pivots = (int*)calloc(n_pivots, sizeof(int));  // Manual alloc
  for (i = k = 0; i < n_v; ++i)
    if (v[i]->is_pivot()) i_pivots[k++] = i;
  // ...
  free(i_pivots);  // Manual dealloc
}
```

**Why it's problematic:**
- Multiple levels of manual memory management
- Out parameters for return values
- `nodes_t` struct is just a container, should be a class
- Easy to leak memory if exceptions occur
- Array size stored separately from array

---

### Pattern 4: Opaque Pointers (Current)

**Current approach in KadRng class**

```cpp
// Lines 204-215 (KadRng declaration)
class KadRng {
public:
  static void* create();                    // Returns void*!
  static void seed(void *d, uint64_t seed);
  static double drand(void *d);
  static double drand_normal(void *d);
};

// Usage:
void *rng = KadRng::create();
KadRng::seed(rng, 12345);
double val = KadRng::drand(rng);
// Caller doesn't know about kad_rng_t structure

// Legacy wrapper:
void *kad_rng(void) { return KadRng::create(); }
void kad_srand(void *d, uint64_t seed) { KadRng::seed(d, seed); }
```

**Why it's problematic:**
- No type safety
- Hidden internal structure (kad_rng_t)
- Easy to accidentally pass wrong pointer type
- No opportunity for inheritance
- API doesn't express intent clearly

---

### Pattern 5: Manual Optimizer State (Current)

**Problem: Optimizer parameters scattered**

```cpp
// Lines 2151-2175 (ann_init)
void ann_init(kanncompr_t *options) {
  options->x = (float**)malloc(...);
  options->y = (float**)malloc(...);
  options->xp = (float*)calloc(...);
  options->n_var = kann_size_var(options->ann);
  options->ua = kann_unroll(options->ann, options->ulen);
  
  if(options->beta1 != 0.0 || options->beta1t != 0.0)
    options->m = (float*)calloc(options->n_var, sizeof(float));
  options->v = (float*)calloc(options->n_var, sizeof(float));
}

// Lines 2196-2228 (ann_train)
void ann_train(kanncompr_t *options) {
  // ... many operations on scattered state ...
  Adam(options->n_var, options->alpha1, options->beta1,
       options->beta1t, options->beta2, options->beta2t,
       options->eps, options->ua->g, options->ua->x,
       options->m, options->v);
  
  options->alpha1 = options->alpha1 > options->alpha2 ?
    options->alpha1 * options->alpha1d : options->alpha2;
  options->beta1t *= options->beta1;
  options->beta2t *= options->beta2;
}

// Lines 2230-2244 (ann_end)
void ann_end(kanncompr_t options) {  // Pass by value!
  kann_rnn_end(options.ann);
  kann_switch(options.ua, 0);
  kann_delete_unrolled(options.ua);
  kann_delete(options.ann);
  if(options.m != NULL) free(options.m);
  free(options.v);
  free(options.xp);
  for(int k = 0; k < options.ulen; k++) {
    free(options.y[k]);
    free(options.x[k]);
  }
  free(options.y);
  free(options.x);
}
```

**Why it's problematic:**
- Huge struct with many responsibilities
- Optimizer state spread across multiple members
- Three separate functions to manage one object
- `ann_end` takes pass-by-value (unusual and potentially expensive)
- No object lifetime guarantees
- Difficult to extend with new optimizer algorithms

---

## Proposed Modern C++ Patterns

### Pattern 1: RAII Resource Management (Proposed)

```cpp
class CompressionCodec {
private:
  std::unique_ptr<kann_t> trained_network_;
  std::unique_ptr<kann_t> unrolled_network_;
  std::unique_ptr<float[]> input_buffer_;
  std::unique_ptr<float[]> output_buffer_;
  std::unique_ptr<float[]> xp_buffer_;
  std::vector<float*> x_sequences_;
  std::vector<float*> y_sequences_;
  
  AdamOptimizer optimizer_;
  
public:
  CompressionCodec(const CompressionConfig& config) {
    // Constructor handles all initialization
    initializeNetwork(config);
    allocateBuffers(config);
    unrollNetwork(config);
  }
  
  // Automatic cleanup on destruction
  ~CompressionCodec() = default;  // unique_ptr handles cleanup
  
  uint8_t predictSymbol(std::vector<uint>& freq, uint& total) {
    // Single method with clear interface
  }
  
  void trainOnSymbol(uint8_t symbol) {
    // Clear, encapsulated training
  }
};

// Usage:
{
  CompressionCodec codec(config);
  codec.trainOnSymbol(symbol);
  codec.predictSymbol(freq, total);
} // Automatic cleanup via RAII
```

**Benefits:**
- Automatic resource management
- Single object manages all lifetime
- Type-safe
- Exception-safe
- Clear initialization and cleanup

---

### Pattern 2: Proper Container Classes (Proposed)

```cpp
class NodeGraph {
private:
  std::vector<kad_node_t*> nodes_;
  std::unique_ptr<float[]> var_buffer_;
  std::unique_ptr<float[]> grad_buffer_;
  std::unique_ptr<float[]> const_buffer_;
  
public:
  void compile(int n_roots, kad_node_t** roots) {
    KadGraph::compile_array(&nodes_.size(), n_roots, roots);
  }
  
  void evalutate(int output_idx) {
    KadGraph::eval_at(nodes_.size(), nodes_.data(), output_idx);
  }
  
  void computeGradients(int output_idx) {
    KadGraph::grad(nodes_.size(), nodes_.data(), output_idx);
  }
  
  size_t nodeCount() const { return nodes_.size(); }
  
  ~NodeGraph() {
    KadGraph::delete_graph(nodes_.size(), nodes_.data());
  }
};

// Usage:
{
  NodeGraph graph;
  graph.compile(1, &root);
  graph.evalutate(output_idx);
  graph.computeGradients(output_idx);
} // Automatic cleanup
```

**Benefits:**
- Encapsulates (int n, T** array) pattern
- Memory managed automatically
- Size always consistent
- Clear API
- No out-parameters

---

### Pattern 3: Typed Pointer Wrapper (Proposed)

```cpp
class RandomNumberGenerator {
private:
  std::unique_ptr<kad_rng_t> impl_;
  
public:
  RandomNumberGenerator(uint64_t seed = 0) 
    : impl_(std::make_unique<kad_rng_t>()) {
    KadRng::seed(impl_.get(), seed);
  }
  
  uint64_t nextInt() {
    return KadRng::rand(impl_.get());
  }
  
  double nextDouble() {
    return KadRng::drand(impl_.get());
  }
  
  double nextGaussian() {
    return KadRng::drand_normal(impl_.get());
  }
};

// Usage:
RandomNumberGenerator rng(seed);
double value = rng.nextGaussian();  // Type-safe, clear intent
```

**Benefits:**
- Type safety
- No opaque pointers
- Automatic lifetime management
- Clear interface
- Can be extended with new methods

---

### Pattern 4: Optimizer as Separate Class (Proposed)

```cpp
class AdamOptimizer {
private:
  float alpha1_, alpha2_, alpha1d_;
  float beta1_, beta1t_, beta2_, beta2t_;
  float eps_;
  std::unique_ptr<float[]> m_;  // First moment
  std::unique_ptr<float[]> v_;  // Second moment
  
public:
  AdamOptimizer(float alpha1, float alpha2, float beta1, 
                float beta2, float eps)
    : alpha1_(alpha1), alpha2_(alpha2), beta1_(beta1),
      beta2_(beta2), eps_(eps), alpha1d_(1.0f) {}
  
  void initialize(int n_vars) {
    m_ = std::make_unique<float[]>(n_vars);
    v_ = std::make_unique<float[]>(n_vars);
    std::fill(m_.get(), m_.get() + n_vars, 0.0f);
    std::fill(v_.get(), v_.get() + n_vars, 0.0f);
  }
  
  void step(int n_vars, float* gradients, float* weights) {
    // Encapsulated Adam algorithm
    for (int i = 0; i < n_vars; ++i) {
      m_[i] = (1.0f - beta1_) * gradients[i] + beta1_ * m_[i];
      v_[i] = (1.0f - beta2_) * gradients[i] * gradients[i] 
              + beta2_ * v_[i];
      weights[i] -= alpha1_ * (m_[i] / (1.0f - beta1t_)) 
                    / (std::sqrt(v_[i] / (1.0f - beta2t_)) + eps_);
    }
    
    // Update schedule
    alpha1_ = std::max(alpha1_ * alpha1d_, alpha2_);
    beta1t_ *= beta1_;
    beta2t_ *= beta2_;
  }
};

// Usage:
AdamOptimizer optimizer(0.001, 0.00001, 0.9, 0.999, 1e-8);
optimizer.initialize(n_vars);
optimizer.step(n_vars, gradients, weights);
```

**Benefits:**
- Encapsulation of optimizer state
- Single responsibility principle
- Reusable across different networks
- Easy to implement other optimizers
- Type-safe parameter handling

---

## Migration Path

### Phase 1: Add Modern Wrappers (Non-Breaking)

Create new classes alongside existing C functions:
```cpp
// NEW CLASS
class NeuralNetwork {
  std::unique_ptr<kann_t> impl_;
public:
  float computeCost(int label, bool grad) {
    return kann_cost(impl_.get(), label, grad);
  }
};

// KEEP OLD FUNCTION
void kann_delete(kann_t *a) { /* ... */ }
```

### Phase 2: Deprecate Old API

Mark functions as deprecated:
```cpp
[[deprecated("Use NeuralNetwork class instead")]]
void kann_delete(kann_t *a);
```

### Phase 3: Full Replacement

Once all call sites migrated:
```cpp
// Remove old function, API is now:
// NeuralNetwork::~NeuralNetwork() handles cleanup
```

---

## Key Refactoring Targets (Priority Order)

1. **CompressionCodec class** - wraps kanncompr_t + ann_init/train/predict/end
2. **NeuralNetwork class** - wraps kann_t + lifecycle
3. **NodeGraph class** - wraps (int n, kad_node_t** v) arrays
4. **AdamOptimizer class** - wraps optimizer state
5. **RandomNumberGenerator class** - wraps opaque void* RNG
6. **FileSerializer class** - wraps fput/fget functions

