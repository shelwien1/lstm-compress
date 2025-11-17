
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

// Modern C++ type aliases instead of typedefs
using word = uint16_t;
using uint = uint32_t;
using byte = uint8_t;
using qword = uint64_t;
using sqword = int64_t;

// Version constants - replacing macros with constexpr
constexpr const char* KANN_VERSION = "r549";

// KANN flags - replacing macros with constexpr
constexpr uint32_t KANN_F_IN = 0x1;
constexpr uint32_t KANN_F_OUT = 0x2;
constexpr uint32_t KANN_F_TRUTH = 0x4;
constexpr uint32_t KANN_F_COST = 0x8;

// KANN cost types - replacing macros with constexpr
constexpr int KANN_C_CEB = 1;
constexpr int KANN_C_CEM = 2;
constexpr int KANN_C_CEB_NEG = 3;
constexpr int KANN_C_MSE = 4;

// KANN RNN flags - replacing macros with constexpr
constexpr uint32_t KANN_RNN_VAR_H0 = 0x1;
constexpr uint32_t KANN_RNN_NORM = 0x2;

// KAD version - replacing macro with constexpr
constexpr const char* KAD_VERSION = "r544";

// KAD dimensions and operations - replacing macros with constexpr
constexpr int KAD_MAX_DIM = 4;
constexpr int KAD_MAX_OP = 64;

// KAD flags - replacing macros with constexpr
constexpr uint8_t KAD_VAR = 0x1;
constexpr uint8_t KAD_CONST = 0x2;
constexpr uint8_t KAD_POOL = 0x4;
constexpr uint8_t KAD_SHARE_RNG = 0x10;

// kad_node_t struct with C++ methods
struct kad_node_t {
  uint8_t     n_d{0};
  uint8_t     flag{0};
  uint16_t    op{0};
  int32_t     n_child{0};
  int32_t     tmp{0};
  int32_t     ptr_size{0};
  int32_t     d[KAD_MAX_DIM]{};
  int32_t     ext_label{0};
  uint32_t    ext_flag{0};
  float      *x{nullptr};
  float      *g{nullptr};
  void       *ptr{nullptr};
  void       *gtmp{nullptr};
  struct kad_node_t **child{nullptr};
  struct kad_node_t  *pre{nullptr};

  // Member methods replacing external functions
  bool is_back() const { return flag & KAD_VAR; }
  bool is_ext() const { return n_child == 0; }
  bool is_var() const { return is_ext() && is_back(); }
  bool is_const() const { return is_ext() && (flag & KAD_CONST); }
  bool is_feed() const { return is_ext() && !is_back() && !(flag & KAD_CONST); }
  bool is_pivot() const { return n_child == 1 && (flag & KAD_POOL); }
  bool is_switch() const { return op == 12 && !(flag & KAD_POOL); }
  bool use_rng() const { return op == 15 || op == 24; }

  void eval_enable() { tmp = 1; }
  void eval_disable() { tmp = -1; }

  // Calculate tensor length
  int length() const {
    int n = 1;
    for (int i = 0; i < n_d; ++i) n *= d[i];
    return n;
  }

  // Static factory methods for node creation
  static kad_node_t* new_core(int n_d, int op, int n_child);
  static kad_node_t* vleaf(uint8_t flag, float *x, float *g, int n_d, va_list ap);
  static kad_node_t* const_node(float *x, int n_d, ...);
  static kad_node_t* feed(int n_d, ...);
  static kad_node_t* var(float *x, float *g, int n_d, ...);
  static kad_node_t* finalize_node(kad_node_t *s);
  static kad_node_t* op2_core(int op, kad_node_t *x, kad_node_t *y);
  static kad_node_t* op1_core(int op, kad_node_t *x);
  static kad_node_t* pooling_general(int op, int n, kad_node_t **x);
  static kad_node_t* concat_array(int axis, int n, kad_node_t **p);
  static kad_node_t* concat(int axis, int n, ...);
  static kad_node_t* switch_node(int n, kad_node_t **p);
  static kad_node_t* dup1(const kad_node_t *p);

  // Binary operations
  static kad_node_t* add(kad_node_t* x, kad_node_t* y);
  static kad_node_t* sub(kad_node_t* x, kad_node_t* y);
  static kad_node_t* mul(kad_node_t* x, kad_node_t* y);
  static kad_node_t* cmul(kad_node_t* x, kad_node_t* y);
  static kad_node_t* matmul(kad_node_t* x, kad_node_t* y);
  static kad_node_t* ce_multi(kad_node_t* x, kad_node_t* y);
  static kad_node_t* ce_bin(kad_node_t* x, kad_node_t* y);
  static kad_node_t* ce_bin_neg(kad_node_t* x, kad_node_t* y);
  static kad_node_t* mse(kad_node_t* x, kad_node_t* y);

  // Unary operations
  static kad_node_t* log(kad_node_t* x);
  static kad_node_t* exp(kad_node_t* x);
  static kad_node_t* sin(kad_node_t* x);
  static kad_node_t* square(kad_node_t* x);
  static kad_node_t* sigm(kad_node_t* x);
  static kad_node_t* tanh_op(kad_node_t* x);
  static kad_node_t* relu(kad_node_t* x);
  static kad_node_t* one_minus(kad_node_t* x);
  static kad_node_t* softmax(kad_node_t* x);
  static kad_node_t* stdnorm(kad_node_t* x);
  static kad_node_t* avg(int n, kad_node_t **x);
};

using kad_node_p = kad_node_t*;

// KadGraph class for graph compilation and manipulation
class KadGraph {
public:
  static kad_node_t** compile_array(int *n_node, int n_roots, kad_node_t **roots);
  static kad_node_t** compile(int *n_node, int n_roots, ...);
  static void delete_graph(int n, kad_node_t **a);
  static const float* eval_at(int n, kad_node_t **a, int from);
  static void eval_marked(int n, kad_node_t **a);
  static int sync_dim(int n, kad_node_t **v, int batch_size);
  static void grad(int n, kad_node_t **a, int from);
  static kad_node_t** unroll(int n_v, kad_node_t **v, int *new_n, int *len);
  static int n_pivots(int n_v, kad_node_t **v);
  static int size_var(int n, kad_node_t *const* v);
  static int size_const(int n, kad_node_t *const* v);
  static void mark_back(int n, kad_node_t **v);
  static void allocate_internal(int n, kad_node_t **v);
  static void propagate_marks(int n, kad_node_t **a);
  static void ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c);
  static void ext_sync(int n, kad_node_t **a, float *x, float *g, float *c);
  static void copy_dim1(kad_node_t *dst, const kad_node_t *src);
};

// Legacy function declarations for backward compatibility
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots);
kad_node_t **kad_compile(int *n_node, int n_roots, ...);
void kad_delete(int n, kad_node_t **a);
const float *kad_eval_at(int n, kad_node_t **a, int from);
void kad_eval_marked(int n, kad_node_t **a);
int kad_sync_dim(int n, kad_node_t **v, int batch_size);
void kad_grad(int n, kad_node_t **a, int from);
kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len);
int kad_n_pivots(int n_v, kad_node_t **v);

kad_node_t *kad_var(float *x, float *g, int n_d, ...); 
kad_node_t *kad_feed(int n_d, ...);                    

kad_node_t *kad_add(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y); 

kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);       

kad_node_t *kad_ce_multi(kad_node_t *x, kad_node_t *y);   

// KAD padding modes - replacing macros with constexpr
constexpr int KAD_PAD_NONE = 0;
constexpr int KAD_PAD_SAME = -2;   

kad_node_t *kad_sigm(kad_node_t *x);   
kad_node_t *kad_tanh(kad_node_t *x);   
kad_node_t *kad_softmax(kad_node_t *x);
kad_node_t *kad_1minus(kad_node_t *x); 
kad_node_t *kad_sin(kad_node_t *x);    

kad_node_t *kad_stdnorm(kad_node_t *x); 

kad_node_t *kad_avg(int n, kad_node_t **x);   

kad_node_t *kad_concat(int axis, int n, ...);                       
kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p);      
kad_node_t *kad_switch(int n, kad_node_t **p);                      

int kad_size_var(int n, kad_node_t *const* v);
int kad_size_const(int n, kad_node_t *const* v);

// Forward declare kad_rng_t
typedef struct {
  uint64_t s[2];
  double n_gset;
  int n_iset;
  volatile int lock;
} kad_rng_t;

// RandomNumberGenerator - Modern C++ class replacing opaque void* RNG
class RandomNumberGenerator {
private:
  kad_rng_t rng_;

  static uint64_t splitmix64(uint64_t x);
  static uint64_t xoroshiro128plus_next(kad_rng_t *r);
  static void xoroshiro128plus_jump(kad_rng_t *r);

public:
  RandomNumberGenerator(uint64_t seed = 0);
  void seed(uint64_t seed);
  uint64_t nextInt();
  double nextDouble();
  double nextGaussian();

  // For legacy compatibility
  kad_rng_t* raw() { return &rng_; }
};

// KadRng class for legacy compatibility with static methods
class KadRng {
public:
  static void* create();
  static void seed(void *d, uint64_t seed);
  static uint64_t rand(void *d);
  static double drand(void *d);
  static double drand_normal(void *d);

  // Made public for RandomNumberGenerator to use
  static uint64_t splitmix64(uint64_t x);
  static uint64_t xoroshiro128plus_next(kad_rng_t *r);
  static void xoroshiro128plus_jump(kad_rng_t *r);
};

// KadMath class for mathematical utilities
class KadMath {
public:
  static void saxpy(int n, float a, const float *x, float *y);
  static float sdot(int n, const float *x, const float *y);
  static void vec_mul_sum(int n, float *a, const float *b, const float *c);
  static void sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C);
private:
  static void saxpy_inlined(int n, float a, const float *x, float *y);
};

// Legacy function declarations for backward compatibility
void *kad_rng(void);
void kad_srand(void *d, uint64_t seed);
uint64_t kad_rand(void *d);
double kad_drand(void *d);
double kad_drand_normal(void *d);
void kad_saxpy(int n, float a, const float *x, float *y);

void kad_trap_fe(void); 

// KAD operation modes - replacing macros with constexpr
constexpr int KAD_ALLOC = 1;
constexpr int KAD_FORWARD = 2;
constexpr int KAD_BACKWARD = 3;
constexpr int KAD_SYNC_DIM = 4;

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[KAD_MAX_OP];

// kann_t struct with C++ methods
struct kann_t {
  int n{0};
  kad_node_t **v{nullptr};
  float *x{nullptr};
  float *g{nullptr};
  float *c{nullptr};
  void *mt{nullptr};

  // Helper methods
  int size_var() const { return kad_size_var(n, v); }
  int size_const() const { return kad_size_const(n, v); }
  void set_batch_size(int B) { kad_sync_dim(n, v, B); }
};

extern int kann_verbose;

// functions replacing kann_ macros
int kann_size_var(const kann_t* a) { return kad_size_var(a->n, a->v); }
int kann_size_const(const kann_t* a) { return kad_size_const(a->n, a->v); }
void kann_srand(uint64_t seed) { kad_srand(nullptr, seed); }
double kann_drand() { return kad_drand(nullptr); }
void kann_set_batch_size(kann_t* ann, int B) { kad_sync_dim(ann->n, ann->v, B); }

kann_t *kann_new(kad_node_t *cost, int n_rest, ...);

kann_t *kann_unroll(kann_t *a, ...);

kann_t *kann_unroll_array(kann_t *a, int *len);
void kann_delete(kann_t *a);          
void kann_delete_unrolled(kann_t *a); 

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x);

float kann_cost(kann_t *a, int cost_label, int cal_grad);

int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label);

void kann_rnn_start(kann_t *a);

void kann_rnn_end(kann_t *a);

void kann_switch(kann_t *a, int is_train);

float kann_grad_clip(float thres, int n, float *g);

kad_node_t *kann_layer_input(int n1);
kad_node_t *kann_layer_dense(kad_node_t *in, int n1);
kad_node_t *kann_layer_layernorm(kad_node_t *in);
kad_node_t *kann_layer_dropout(kad_node_t *t, float r);

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...); 
kad_node_t *kann_new_scalar(uint8_t flag, float x);
kad_node_t *kann_new_weight(int n_row, int n_col);
kad_node_t *kann_new_bias(int n);
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col);
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len);

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...);
kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1);
kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in);

const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label);
const float *kann_apply1(kann_t *a, float *x);

void KadGraph::ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c)
{
  int i, j, k, l, n_var;
  float *x, *g, *c;
  n_var = size_var(n, a);
  x = *_x = (float*)realloc(*_x, n_var * sizeof(float));
  g = *_g = (float*)realloc(*_g, n_var * sizeof(float));
  c = *_c = (float*)realloc(*_c, size_const(n, a) * sizeof(float));
  memset(g, 0, n_var * sizeof(float));
  for (i = j = k = 0; i < n; ++i) {
    kad_node_t *v = a[i];
    if (v->is_var()) {
      l = v->length();
      memcpy(&x[j], v->x, l * sizeof(float));
      free(v->x);
      v->x = &x[j];
      v->g = &g[j];
      j += l;
    } else if (v->is_const()) {
      l = v->length();
      memcpy(&c[k], v->x, l * sizeof(float));
      free(v->x);
      v->x = &c[k];
      k += l;
    }
  }
}

void KadGraph::ext_sync(int n, kad_node_t **a, float *x, float *g, float *c)
{
  int i, j, k;
  for (i = j = k = 0; i < n; ++i) {
    kad_node_t *v = a[i];
    if (v->is_var()) {
      v->x = &x[j];
      v->g = &g[j];
      j += v->length();
    } else if (v->is_const()) {
      v->x = &c[k];
      k += v->length();
    }
  }
}

// Legacy compatibility functions
void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c) { KadGraph::ext_collate(n, a, _x, _g, _c); }
void kad_ext_sync(int n, kad_node_t **a, float *x, float *g, float *c) { KadGraph::ext_sync(n, a, x, g, c); }

kann_t *kann_new(kad_node_t *cost, int n_rest, ...)
{
  kann_t *a;
  int i, n_roots = 1 + n_rest, has_pivot = 0, has_recur = 0;
  kad_node_t **roots;
  va_list ap;

  if (cost->n_d != 0) return 0;

  va_start(ap, n_rest);
  roots = (kad_node_t**)malloc((n_roots + 1) * sizeof(kad_node_t*));
  for (i = 0; i < n_rest; ++i)
    roots[i] = va_arg(ap, kad_node_t*);
  roots[i++] = cost;
  va_end(ap);

  cost->ext_flag |= KANN_F_COST;
  a = (kann_t*)calloc(1, sizeof(kann_t));
  a->v = kad_compile_array(&a->n, n_roots, roots);

  for (i = 0; i < a->n; ++i) {
    if (a->v[i]->pre) has_recur = 1;
    if (a->v[i]->is_pivot()) has_pivot = 1;
  }
  if (has_recur && !has_pivot) { 
    cost->ext_flag &= ~KANN_F_COST;
    roots[n_roots-1] = cost = kad_avg(1, &cost), cost->ext_flag |= KANN_F_COST;
    free(a->v);
    a->v = kad_compile_array(&a->n, n_roots, roots);
  }
  kad_ext_collate(a->n, a->v, &a->x, &a->g, &a->c);
  free(roots);
  return a;
}

kann_t *kann_unroll_array(kann_t *a, int *len)
{
  kann_t *b;
  b = (kann_t*)calloc(1, sizeof(kann_t));
  b->x = a->x, b->g = a->g, b->c = a->c; 
  b->v = kad_unroll(a->n, a->v, &b->n, len);
  return b;
}

kann_t *kann_unroll(kann_t *a, ...)
{
  kann_t *b;
  va_list ap;
  int i, n_pivots, *len;
  n_pivots = kad_n_pivots(a->n, a->v);
  len = (int*)calloc(n_pivots, sizeof(int));
  va_start(ap, a);
  for (i = 0; i < n_pivots; ++i) len[i] = va_arg(ap, int);
  va_end(ap);
  b = kann_unroll_array(a, len);
  free(len);
  return b;
}

void kann_delete_unrolled(kann_t *a)
{
  
  if (a && a->v) kad_delete(a->n, a->v);
  free(a);
}

void kann_delete(kann_t *a)
{
  if (a == 0) return;
  free(a->x); free(a->g); free(a->c);
  kann_delete_unrolled(a);
}

void kann_switch_core(kann_t *a, int is_train)
{
  int i;
  for (i = 0; i < a->n; ++i)
    if (a->v[i]->op == 12 && a->v[i]->n_child == 2)
      *(int32_t*)a->v[i]->ptr = !!is_train;
}

// functions replacing chk_ macros
bool chk_flg(uint32_t flag, uint32_t mask) { return mask == 0 || (flag & mask); }
bool chk_lbl(int32_t label, int32_t query) { return query == 0 || label == query; }

int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
  int i, k, r = -1;
  for (i = k = 0; i < a->n; ++i)
    if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
      ++k, r = i;
  return k == 1? r : k == 0? -1 : -2;
}

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x)
{
  int i, k;
  if (x == 0) return 0;
  for (i = k = 0; i < a->n; ++i)
    if (a->v[i]->is_feed() && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
      a->v[i]->x = x[k++];
  return k;
}

float kann_cost_core(kann_t *a, int cost_label, int cal_grad)
{
  int i_cost;
  float cost;
  i_cost = kann_find(a, KANN_F_COST, cost_label);
  assert(i_cost >= 0);
  cost = *kad_eval_at(a->n, a->v, i_cost);
  if (cal_grad) kad_grad(a->n, a->v, i_cost);
  return cost;
}

void kann_rnn_start(kann_t *a)
{
  int i;
  kann_set_batch_size(a, 1);
  for (i = 0; i < a->n; ++i) {
    kad_node_t *p = a->v[i];
    if (p->pre) { 
      kad_node_t *q = p->pre;
      if (q->x) memcpy(p->x, q->x, p->length() * sizeof(float));
      else memset(p->x, 0, p->length() * sizeof(float));
      if (q->n_child > 0) free(q->x);
      q->x = p->x;
    }
  }
}

void kann_rnn_end(kann_t *a)
{
  int i;
  kad_ext_sync(a->n, a->v, a->x, a->g, a->c);
  for (i = 0; i < a->n; ++i)
    if (a->v[i]->pre && a->v[i]->pre->n_child > 0)
      a->v[i]->pre->x = (float*)calloc(a->v[i]->pre->length(), sizeof(float));
}

float kann_cost(kann_t *a, int cost_label, int cal_grad) { return kann_cost_core(a, cost_label, cal_grad); }
void kann_switch(kann_t *ann, int is_train) { return kann_switch_core(ann, is_train); }

// KANN magic constant - replacing macro with constexpr
constexpr const char* KANN_MAGIC = "KAN\1";

kad_node_t *kann_new_leaf_array(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, int32_t d[KAD_MAX_DIM])
{
  int i, len, off = offset && par? *offset : -1;
  kad_node_t *p;

  if (off >= 0 && par[off]) return par[(*offset)++];
  p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
  p->n_d = n_d, p->flag = flag;
  memcpy(p->d, d, n_d * sizeof(int32_t));
  len = p->length();
  p->x = (float*)calloc(len, sizeof(float));
  if (p->n_d <= 1) {
    for (i = 0; i < len; ++i)
      p->x[i] = x0_01;
  } else {
    double sdev_inv;
    sdev_inv = 1.0 / sqrt((double)len / p->d[0]);
    for (i = 0; i < len; ++i)
      p->x[i] = (float)(kad_drand_normal(0) * sdev_inv);
  }
  if (off >= 0) par[off] = p, ++(*offset);
  return p;
}

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...)
{
  int32_t i, d[KAD_MAX_DIM];
  va_list ap;
  va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
  return kann_new_leaf_array(offset, par, flag, x0_01, n_d, d);
}

kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1)
{
  int n0;
  kad_node_t *w, *b;
  n0 = in->n_d >= 2? in->length() / in->d[0] : in->length();
  w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
  b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
  return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in)
{
  int n0;
  kad_node_t *alpha, *beta;
  n0 = in->n_d >= 2? in->length() / in->d[0] : in->length();
  alpha = kann_new_leaf2(offset, par, KAD_VAR, 1.0f, 1, n0);
  beta  = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n0);
  return kad_add(kad_mul(kad_stdnorm(in), alpha), beta);
}

kad_node_t *cmul_norm2(int *offset, kad_node_t **par, kad_node_t *x, kad_node_t *w, int use_norm)
{
  return use_norm? kann_layer_layernorm2(offset, par, kad_cmul(x, w)) : kad_cmul(x, w);
}

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...)
{
  int32_t i, d[KAD_MAX_DIM];
  va_list ap;
  va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
  return kann_new_leaf_array(0, 0, flag, x0_01, n_d, d);
}

kad_node_t *kann_new_scalar(uint8_t flag, float x) { return kann_new_leaf(flag, x, 0); }
kad_node_t *kann_new_weight(int n_row, int n_col) { return kann_new_leaf(KAD_VAR, 0.0f, 2, n_row, n_col); }
kad_node_t *kann_new_vec(int n, float x) { return kann_new_leaf(KAD_VAR, x, 1, n); }
kad_node_t *kann_new_bias(int n) { return kann_new_vec(n, 0.0f); }
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col) { return kann_new_leaf(KAD_VAR, 0.0f, 4, n_out, n_in, k_row, k_col); }
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len) { return kann_new_leaf(KAD_VAR, 0.0f, 3, n_out, n_in, kernel_len); }

kad_node_t *kann_layer_input(int n1)
{
  kad_node_t *t;
  t = kad_feed(2, 1, n1), t->ext_flag |= KANN_F_IN;
  return t;
}

kad_node_t *kann_layer_dense(kad_node_t *in, int n1) { return kann_layer_dense2(0, 0, in, n1); }
kad_node_t *kann_layer_layernorm(kad_node_t *in) { return kann_layer_layernorm2(0, 0, in); }
kad_node_t *kann_layer_dropout(kad_node_t *t, float r) { return t; } 

kad_node_t *kann_cmul_norm(kad_node_t *x, kad_node_t *w)
{
  return kann_layer_layernorm(kad_cmul(x, w));
}

float kann_grad_clip(float thres, int n, float *g)
{
  int i;
  double s2 = 0.0;
  for (i = 0; i < n; ++i)
    s2 += g[i] * g[i];
  s2 = sqrt(s2);
  if (s2 > thres)
    for (i = 0, s2 = 1.0 / s2; i < n; ++i)
      g[i] *= (float)s2;
  return (float)s2 / thres;
}

const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label)
{
  int i_out;
  i_out = kann_find(a, ext_flag, ext_label);
  if (i_out < 0) return 0;
  kann_set_batch_size(a, 1);
  kann_feed_bind(a, KANN_F_IN, 0, &x);
  kad_eval_at(a->n, a->v, i_out);
  return a->v[i_out]->x;
}

const float *kann_apply1(kann_t *a, float *x)
{
  return kann_apply1_to(a, x, KANN_F_OUT, 0);
}

kad_node_t *kad_node_t::new_core(int n_d, int op, int n_child)
{
  kad_node_t *s;
  if (n_d >= KAD_MAX_DIM) return 0;
  s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
  s->n_d = n_d, s->op = op, s->n_child = n_child;
  if (s->n_child) s->child = (kad_node_t**)calloc(s->n_child, sizeof(kad_node_t*));
  return s;
}

kad_node_t *kad_node_t::vleaf(uint8_t flag, float *x, float *g, int n_d, va_list ap)
{
  int i;
  kad_node_t *p;
  if (n_d > KAD_MAX_DIM) return 0;
  p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
  p->n_d = n_d;
  for (i = 0; i < n_d; ++i)
    p->d[i] = va_arg(ap, int32_t);
  p->x = x, p->g = g, p->flag = flag;
  return p;
}

kad_node_t *kad_node_t::const_node(float *x, int n_d, ...)
{
  kad_node_t *p;
  va_list ap;
  va_start(ap, n_d); p = vleaf(KAD_CONST, x, 0, n_d, ap); va_end(ap);
  return p;
}

kad_node_t *kad_node_t::feed(int n_d, ...)
{
  kad_node_t *p;
  va_list ap;
  va_start(ap, n_d); p = vleaf(0, 0, 0, n_d, ap); va_end(ap);
  return p;
}

kad_node_t *kad_node_t::var(float *x, float *g, int n_d, ...)
{
  kad_node_t *p;
  va_list ap;
  va_start(ap, n_d); p = vleaf(KAD_VAR, x, g, n_d, ap); va_end(ap);
  return p;
}

kad_node_t *kad_node_t::finalize_node(kad_node_t *s)
{
  int i;
  if (kad_op_list[s->op](s, KAD_SYNC_DIM) < 0) {
    if (s->ptr) free(s->ptr);
    free(s->child); free(s);
    return 0;
  }
  for (i = 0; i < s->n_child; ++i)
    if (s->child[i]->is_back())
      break;
  if (i < s->n_child) s->flag |= KAD_VAR;
  return s;
}

kad_node_t *kad_node_t::op2_core(int op, kad_node_t *x, kad_node_t *y)
{
  kad_node_t *s;
  s = new_core(0, op, 2);
  s->child[0] = x, s->child[1] = y;
  return finalize_node(s);
}

kad_node_t *kad_node_t::op1_core(int op, kad_node_t *x)
{
  kad_node_t *s;
  s = new_core(0, op, 1);
  s->child[0] = x;
  return finalize_node(s);
}

// Binary operations - static methods of kad_node_t
kad_node_t* kad_node_t::add(kad_node_t* x, kad_node_t* y) { return op2_core(1, x, y); }
kad_node_t* kad_node_t::sub(kad_node_t* x, kad_node_t* y) { return op2_core(23, x, y); }
kad_node_t* kad_node_t::mul(kad_node_t* x, kad_node_t* y) { return op2_core(2, x, y); }
kad_node_t* kad_node_t::cmul(kad_node_t* x, kad_node_t* y) { return op2_core(3, x, y); }
kad_node_t* kad_node_t::matmul(kad_node_t* x, kad_node_t* y) { return op2_core(9, x, y); }
kad_node_t* kad_node_t::ce_multi(kad_node_t* x, kad_node_t* y) { return op2_core(13, x, y); }
kad_node_t* kad_node_t::ce_bin(kad_node_t* x, kad_node_t* y) { return op2_core(22, x, y); }
kad_node_t* kad_node_t::ce_bin_neg(kad_node_t* x, kad_node_t* y) { return op2_core(4, x, y); }
kad_node_t* kad_node_t::mse(kad_node_t* x, kad_node_t* y) { return op2_core(29, x, y); }

// Unary operations - static methods of kad_node_t
kad_node_t* kad_node_t::log(kad_node_t* x) { return op1_core(27, x); }
kad_node_t* kad_node_t::exp(kad_node_t* x) { return op1_core(33, x); }
kad_node_t* kad_node_t::sin(kad_node_t* x) { return op1_core(34, x); }
kad_node_t* kad_node_t::square(kad_node_t* x) { return op1_core(5, x); }
kad_node_t* kad_node_t::sigm(kad_node_t* x) { return op1_core(6, x); }
kad_node_t* kad_node_t::tanh_op(kad_node_t* x) { return op1_core(7, x); }
kad_node_t* kad_node_t::relu(kad_node_t* x) { return op1_core(8, x); }
kad_node_t* kad_node_t::one_minus(kad_node_t* x) { return op1_core(11, x); }
kad_node_t* kad_node_t::softmax(kad_node_t* x) { return op1_core(14, x); }
kad_node_t* kad_node_t::stdnorm(kad_node_t* x) { return op1_core(32, x); }

// Legacy function declarations for backward compatibility
kad_node_t* kad_add(kad_node_t* x, kad_node_t* y) { return kad_node_t::add(x, y); }
kad_node_t* kad_sub(kad_node_t* x, kad_node_t* y) { return kad_node_t::sub(x, y); }
kad_node_t* kad_mul(kad_node_t* x, kad_node_t* y) { return kad_node_t::mul(x, y); }
kad_node_t* kad_cmul(kad_node_t* x, kad_node_t* y) { return kad_node_t::cmul(x, y); }
kad_node_t* kad_matmul(kad_node_t* x, kad_node_t* y) { return kad_node_t::matmul(x, y); }
kad_node_t* kad_ce_multi(kad_node_t* x, kad_node_t* y) { return kad_node_t::ce_multi(x, y); }
kad_node_t* kad_ce_bin(kad_node_t* x, kad_node_t* y) { return kad_node_t::ce_bin(x, y); }
kad_node_t* kad_ce_bin_neg(kad_node_t* x, kad_node_t* y) { return kad_node_t::ce_bin_neg(x, y); }
kad_node_t* kad_mse(kad_node_t* x, kad_node_t* y) { return kad_node_t::mse(x, y); }
kad_node_t* kad_log(kad_node_t* x) { return kad_node_t::log(x); }
kad_node_t* kad_exp(kad_node_t* x) { return kad_node_t::exp(x); }
kad_node_t* kad_sin(kad_node_t* x) { return kad_node_t::sin(x); }
kad_node_t* kad_square(kad_node_t* x) { return kad_node_t::square(x); }
kad_node_t* kad_sigm(kad_node_t* x) { return kad_node_t::sigm(x); }
kad_node_t* kad_tanh(kad_node_t* x) { return kad_node_t::tanh_op(x); }
kad_node_t* kad_relu(kad_node_t* x) { return kad_node_t::relu(x); }
kad_node_t* kad_1minus(kad_node_t* x) { return kad_node_t::one_minus(x); }
kad_node_t* kad_softmax(kad_node_t* x) { return kad_node_t::softmax(x); }
kad_node_t* kad_stdnorm(kad_node_t* x) { return kad_node_t::stdnorm(x); }
kad_node_t* kad_const(float *x, int n_d, ...) { va_list ap; va_start(ap, n_d); kad_node_t* p = kad_node_t::vleaf(KAD_CONST, x, 0, n_d, ap); va_end(ap); return p; }
kad_node_t* kad_feed(int n_d, ...) { va_list ap; va_start(ap, n_d); kad_node_t* p = kad_node_t::vleaf(0, 0, 0, n_d, ap); va_end(ap); return p; }
kad_node_t* kad_var(float *x, float *g, int n_d, ...) { va_list ap; va_start(ap, n_d); kad_node_t* p = kad_node_t::vleaf(KAD_VAR, x, g, n_d, ap); va_end(ap); return p; }

typedef struct {
  int kernel_size, stride, pad[2];
} conv_conf_t;

kad_node_t *kad_node_t::pooling_general(int op, int n, kad_node_t **x)
{
  int i;
  kad_node_t *s;
  s = new_core(0, op, n);
  s->flag |= KAD_POOL;
  for (i = 0; i < n; ++i)
    s->child[i] = x[i];
  return finalize_node(s);
}

kad_node_t *kad_node_t::avg(int n, kad_node_t **x)   { return pooling_general(10, n, x); }

kad_node_t *kad_node_t::concat_array(int axis, int n, kad_node_t **p)
{
  kad_node_t *s;
  int32_t i, *aux;
  aux = (int32_t*)malloc(4);
  aux[0] = axis;
  s = new_core(0, 31, n);
  for (i = 0; i < n; ++i)
    s->child[i] = p[i];
  s->ptr = aux, s->ptr_size = 4;
  return finalize_node(s);
}

kad_node_t *kad_node_t::concat(int axis, int n, ...)
{
  int i;
  kad_node_t **p, *s;
  va_list ap;
  p = (kad_node_t**)malloc(n * sizeof(kad_node_t*));
  va_start(ap, n);
  for (i = 0; i < n; ++i) p[i] = va_arg(ap, kad_node_p);
  va_end(ap);
  s = concat_array(axis, n, p);
  free(p);
  return s;
}

kad_node_t *kad_node_t::switch_node(int n, kad_node_t **p)
{
  kad_node_t *s;
  int32_t i, *aux;
  aux = (int32_t*)calloc(1, 4);
  s = new_core(0, 12, n);
  for (i = 0; i < n; ++i)
    s->child[i] = p[i];
  s->ptr = aux, s->ptr_size = 4;
  return finalize_node(s);
}

// Legacy compatibility functions for kad_node_t operations
kad_node_t *kad_pooling_general(int op, int n, kad_node_t **x) { return kad_node_t::pooling_general(op, n, x); }
kad_node_t *kad_avg(int n, kad_node_t **x) { return kad_node_t::avg(n, x); }
kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p) { return kad_node_t::concat_array(axis, n, p); }
kad_node_t *kad_concat(int axis, int n, ...) { va_list ap; va_start(ap, n); kad_node_t **p = (kad_node_t**)malloc(n * sizeof(kad_node_t*)); for (int i = 0; i < n; ++i) p[i] = va_arg(ap, kad_node_p); va_end(ap); kad_node_t* s = kad_node_t::concat_array(axis, n, p); free(p); return s; }
kad_node_t *kad_switch(int n, kad_node_t **p) { return kad_node_t::switch_node(n, p); }

void KadGraph::mark_back(int n, kad_node_t **v)
{
  int i, j;
  for (i = 0; i < n; ++i) {
    if (v[i]->n_child == 0) continue;
    for (j = 0; j < v[i]->n_child; ++j)
      if (v[i]->child[j]->is_back())
        break;
    if (j < v[i]->n_child) v[i]->flag |= KAD_VAR;
    else v[i]->flag &= ~KAD_VAR;
  }
}

void KadGraph::allocate_internal(int n, kad_node_t **v)
{
  int i;
  mark_back(n, v);
  for (i = 0; i < n; ++i) {
    kad_node_t *p = v[i];
    if (p->n_child == 0) continue;
    p->x = (float*)realloc(p->x, p->length() * sizeof(float));
    if (p->is_back()) {
      p->g = (float*)realloc(p->g, p->length() * sizeof(float));
      kad_op_list[p->op](p, KAD_ALLOC);
    }
  }
}

int KadGraph::sync_dim(int n, kad_node_t **v, int batch_size)
{
  int i, req_alloc = 0, req_sync = 0, old_size = 0;
  for (i = 0; i < n; ++i) {
    if (v[i]->is_feed()) {
      old_size = v[i]->d[0];
      if (batch_size > 0 && v[i]->d[0] != batch_size)
        v[i]->d[0] = batch_size, req_sync = 1;
    } else if (v[i]->n_child > 0 && req_sync)
      kad_op_list[v[i]->op](v[i], KAD_SYNC_DIM);
  }
  if (old_size < batch_size) req_alloc = 1;
  for (i = 0; i < n; ++i)
    if (v[i]->n_child > 0 && v[i]->x == 0) req_alloc = 1;
  if (req_alloc) allocate_internal(n, v);
  return batch_size > 0? batch_size : old_size;
}

// Legacy compatibility functions
void kad_mark_back(int n, kad_node_t **v) { KadGraph::mark_back(n, v); }
void kad_allocate_internal(int n, kad_node_t **v) { KadGraph::allocate_internal(n, v); }
int kad_sync_dim(int n, kad_node_t **v, int batch_size) { return KadGraph::sync_dim(n, v, batch_size); }

// Template vector structure replacing kvec_t macro
template<typename T>
struct kvec_t {
  size_t n{0};
  size_t m{0};
  T* a{nullptr};

  T pop() { return a[--n]; }

  void push(const T& x) {
    if (n == m) {
      m = m ? m << 1 : 2;
      a = static_cast<T*>(realloc(a, sizeof(T) * m));
    }
    a[n++] = x;
  }

  // Release ownership of the array (for when returning it)
  T* release() {
    T* result = a;
    a = nullptr;
    n = 0;
    m = 0;
    return result;
  }

  ~kvec_t() { free(a); }
};

kad_node_t **KadGraph::compile_array(int *n_node, int n_roots, kad_node_t **roots)
{
  int i;
  kvec_t<kad_node_p> stack = {}, a = {};

  for (i = 0; i < n_roots; ++i) {
    roots[i]->tmp = 1;
    stack.push(roots[i]);
  }
  while (stack.n) {
    kad_node_t *p = stack.pop();
    for (i = 0; i < p->n_child; ++i) {
      kad_node_t *q = p->child[i];
      if (q->tmp == 0) stack.push(q);
      q->tmp += 1<<1;
    }
  }

  for (i = 0; i < n_roots; ++i)
    if (roots[i]->tmp>>1 == 0)
      stack.push(roots[i]);
  while (stack.n) {
    kad_node_t *p = stack.pop();
    a.push(p);
    for (i = 0; i < p->n_child; ++i) {
      p->child[i]->tmp -= 1<<1;
      if (p->child[i]->tmp>>1 == 0)
        stack.push(p->child[i]);
    }
  }
  // stack is automatically freed by destructor
  for (i = 0; i < (int)a.n; ++i) {
    assert(a.a[i]->tmp>>1 == 0);
    a.a[i]->tmp = 0;
  }

  for (i = 0; i < (int)a.n>>1; ++i) {
    kad_node_p t;
    t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
  }
  allocate_internal(a.n, a.a);

  *n_node = a.n;
  // Release ownership before returning to prevent destructor from freeing
  return a.release();
}

kad_node_t **KadGraph::compile(int *n_node, int n_roots, ...)
{
  int i;
  kad_node_t **roots, **ret;
  va_list ap;

  roots = (kad_node_t**)malloc(n_roots * sizeof(kad_node_t*));
  va_start(ap, n_roots);
  for (i = 0; i < n_roots; ++i) roots[i] = va_arg(ap, kad_node_p);
  va_end(ap);
  ret = compile_array(n_node, n_roots, roots);
  free(roots);
  return ret;
}

void KadGraph::delete_graph(int n, kad_node_t **a)
{
  int i;
  for (i = 0; i < n; ++i) {
    kad_node_t *p = a[i];
    if (p->n_child) {
      free(p->x); free(p->g);
    }
    free(p->child); free(p->ptr); free(p->gtmp); free(p);
  }
  free(a);
}

int KadGraph::size_var(int n, kad_node_t *const* v)
{
  int c, i;
  for (i = c = 0; i < n; ++i)
    if (v[i]->is_var())
      c += v[i]->length();
  return c;
}

int KadGraph::size_const(int n, kad_node_t *const* v)
{
  int c, i;
  for (i = c = 0; i < n; ++i)
    if (v[i]->is_const())
      c += v[i]->length();
  return c;
}

void KadGraph::propagate_marks(int n, kad_node_t **a)
{
  int i, j;
  for (i = n - 1; i >= 0; --i) {
    kad_node_t *p = a[i];
    if (p->tmp > 0) {
      if (p->is_switch()) {
        int32_t *aux = (int32_t*)p->ptr;
        if (p->child[*aux]->tmp == 0)
          p->child[*aux]->tmp = 1;
      } else {
        for (j = 0; j < p->n_child; ++j)
          if (p->child[j]->tmp == 0)
            p->child[j]->tmp = 1;
      }
    }
  }
}

// Legacy compatibility functions
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots) { return KadGraph::compile_array(n_node, n_roots, roots); }
kad_node_t **kad_compile(int *n_node, int n_roots, ...) { va_list ap; va_start(ap, n_roots); kad_node_t **roots = (kad_node_t**)malloc(n_roots * sizeof(kad_node_t*)); for (int i = 0; i < n_roots; ++i) roots[i] = va_arg(ap, kad_node_p); va_end(ap); kad_node_t** ret = KadGraph::compile_array(n_node, n_roots, roots); free(roots); return ret; }
void kad_delete(int n, kad_node_t **a) { KadGraph::delete_graph(n, a); }
int kad_size_var(int n, kad_node_t *const* v) { return KadGraph::size_var(n, v); }
int kad_size_const(int n, kad_node_t *const* v) { return KadGraph::size_const(n, v); }
void kad_propagate_marks(int n, kad_node_t **a) { KadGraph::propagate_marks(n, a); }

void KadGraph::eval_marked(int n, kad_node_t **a)
{
  int i;
  propagate_marks(n, a);
  for (i = 0; i < n; ++i)
    if (a[i]->n_child && a[i]->tmp > 0)
      kad_op_list[a[i]->op](a[i], KAD_FORWARD);
  for (i = 0; i < n; ++i) a[i]->tmp = 0;
}

const float *KadGraph::eval_at(int n, kad_node_t **a, int from)
{
  int i;
  if (from < 0 || from >= n) from = n - 1;
  for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
  eval_marked(n, a);
  return a[from]->x;
}

void KadGraph::grad(int n, kad_node_t **a, int from)
{
  int i;
  if (from < 0 || from >= n) from = n - 1;
  assert(a[from]->n_d == 0);
  for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
  propagate_marks(n, a);
  for (i = 0; i <= from; ++i)
    if (a[i]->g && a[i]->tmp > 0)
      memset(a[i]->g, 0, a[i]->length() * sizeof(float));
  for (i = from, a[i]->g[0] = 1.0f; i >= 0; --i)
    if (a[i]->n_child && a[i]->tmp > 0)
      kad_op_list[a[i]->op](a[i], KAD_BACKWARD);
  for (i = 0; i <= from; ++i) a[i]->tmp = 0;
}

// Legacy compatibility functions
void kad_eval_marked(int n, kad_node_t **a) { KadGraph::eval_marked(n, a); }
const float *kad_eval_at(int n, kad_node_t **a, int from) { return KadGraph::eval_at(n, a, from); }
void kad_grad(int n, kad_node_t **a, int from) { KadGraph::grad(n, a, from); }

kad_node_t *kad_node_t::dup1(const kad_node_t *p)
{
  kad_node_t *q;
  q = (kad_node_t*)malloc(sizeof(kad_node_t));
  memcpy(q, p, sizeof(kad_node_t));
  q->pre = 0, q->tmp = 0, q->gtmp = 0;
  if (p->ptr && p->ptr_size > 0) {
    if (p->use_rng() && !(p->flag & KAD_SHARE_RNG) && p->ptr_size == sizeof(kad_rng_t)) {
      q->ptr = kad_rng();
    } else {
      q->ptr = malloc(p->ptr_size);
      memcpy(q->ptr, p->ptr, p->ptr_size);
    }
  }
  if (q->n_child) {
    q->x = q->g = 0;
    q->child = (kad_node_t**)calloc(q->n_child, sizeof(kad_node_t*));
  }
  return q;
}

// Legacy compatibility function
kad_node_t *kad_dup1(const kad_node_t *p) { return kad_node_t::dup1(p); }

typedef struct {
  int32_t n, m;
  kad_node_t **v;
} nodes_t;

void push_nodes(nodes_t *w, kad_node_t *p)
{
  if (w->n == w->m) {
    w->m = w->m? w->m<<1 : 16;
    w->v = (kad_node_t**)realloc(w->v, w->m * sizeof(kad_node_t*));
  }
  w->v[w->n++] = p;
}

void kad_unroll_helper(int n_v, kad_node_t **v, int i_pivot, kad_node_t **t, int len, nodes_t *w)
{
  int i, j, l;
  uint8_t *flag;
  kad_node_t **aux;

  assert(v[i_pivot]->is_pivot() && t[i_pivot] == 0);
  t[i_pivot] = kad_dup1(v[i_pivot]);
  t[i_pivot]->n_child = len;
  t[i_pivot]->child = (kad_node_t**)realloc(t[i_pivot]->child, len * sizeof(kad_node_t*));

  flag = (uint8_t*)calloc(n_v, 1);
  for (i = i_pivot, flag[i] = 16; i >= 0; --i) {
    if (i < i_pivot && v[i]->is_pivot()) continue; 
    if (flag[i]&16) 
      for (j = 0; j < v[i]->n_child; ++j)
        flag[v[i]->child[j]->tmp] = 16;
  }
  for (i = 0; i < i_pivot; ++i) {
    if (!(flag[i]&16)) continue;
    if (v[i]->is_var() || v[i]->is_const() || v[i]->is_pivot()) flag[i] |= 1; 
    if (v[i]->pre) flag[v[i]->pre->tmp] |= 2;
  }
  flag[v[i_pivot]->child[0]->tmp] |= 4;
  aux = (kad_node_t**)calloc(n_v, sizeof(kad_node_t*));
  for (l = 0; l < len; ++l) {
    for (i = 0; i < i_pivot; ++i) {
      if (!(flag[i]&16) || ((flag[i]&3) && t[i])) continue;
      t[i] = kad_dup1(v[i]);
      if (v[i]->n_child)
        for (j = 0; j < v[i]->n_child; ++j)
          t[i]->child[j] = t[v[i]->child[j]->tmp];
      if (flag[i]&4) t[i_pivot]->child[l] = t[i];
      if (l == 0 && (flag[i]&2)) aux[i] = t[i];
      if (v[i]->pre) {
        t[v[i]->pre->tmp] = t[i];
        if (l == len - 1) t[i]->pre = aux[v[i]->pre->tmp]; 
      }
      push_nodes(w, t[i]);
    }
  }
  push_nodes(w, t[i_pivot]);
  free(aux); free(flag);
}

int KadGraph::n_pivots(int n_v, kad_node_t **v)
{
  int i, n_pivots = 0;
  for (i = 0; i < n_v; ++i)
    if (v[i]->is_pivot()) ++n_pivots;
  return n_pivots;
}

// Legacy compatibility function
int kad_n_pivots(int n_v, kad_node_t **v) { return KadGraph::n_pivots(n_v, v); }

kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len)
{
  int i, j, n_pivots = 0;
  kad_node_t **t;
  nodes_t w = {0,0,0};

  t = (kad_node_t**)calloc(n_v, sizeof(kad_node_t*));
  n_pivots = kad_n_pivots(n_v, v);
  for (i = 0; i < n_v; ++i) v[i]->tmp = i;
  if (n_pivots) {
    int k, *i_pivots;
    i_pivots = (int*)calloc(n_pivots, sizeof(int));
    for (i = k = 0; i < n_v; ++i) 
      if (v[i]->is_pivot()) i_pivots[k++] = i;
    for (i = 0; i < n_pivots; ++i) 
      kad_unroll_helper(n_v, v, i_pivots[i], t, len[i], &w);
    free(i_pivots);
  }
  for (i = 0; i < n_v; ++i) { 
    if (t[i]) continue;
    t[i] = kad_dup1(v[i]);
    if (v[i]->n_child)
      for (j = 0; j < v[i]->n_child; ++j)
        t[i]->child[j] = t[v[i]->child[j]->tmp];
    push_nodes(&w, t[i]);
  }
  free(t);
  for (i = 0; i < n_v; ++i) v[i]->tmp = 0;
  for (i = 0; i < w.n; ++i) 
    if (w.v[i]->n_child > 0)
      kad_op_list[w.v[i]->op](w.v[i], KAD_SYNC_DIM);
  kad_allocate_internal(w.n, w.v);
  *new_n = w.n;
  return w.v;
}

float KadMath::sdot(int n, const float *x, const float *y)
{
  int i, n8 = n>>3<<3;
  __m128 vs1, vs2;
  float s, t[4];
  vs1 = _mm_setzero_ps();
  vs2 = _mm_setzero_ps();
  for (i = 0; i < n8; i += 8) {
    __m128 vx1, vx2, vy1, vy2;
    vx1 = _mm_loadu_ps(&x[i]);
    vx2 = _mm_loadu_ps(&x[i+4]);
    vy1 = _mm_loadu_ps(&y[i]);
    vy2 = _mm_loadu_ps(&y[i+4]);
    vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
    vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
  }
  for (s = 0.; i < n; ++i) s += x[i] * y[i];
  _mm_storeu_ps(t, vs1);
  s += t[0] + t[1] + t[2] + t[3];
  _mm_storeu_ps(t, vs2);
  s += t[0] + t[1] + t[2] + t[3];
  return s;
}

void KadMath::saxpy_inlined(int n, float a, const float *x, float *y)
{
  int i, n8 = n>>3<<3;
  __m128 va;
  va = _mm_set1_ps(a);
  for (i = 0; i < n8; i += 8) {
    __m128 vx1, vx2, vy1, vy2, vt1, vt2;
    vx1 = _mm_loadu_ps(&x[i]);
    vx2 = _mm_loadu_ps(&x[i+4]);
    vy1 = _mm_loadu_ps(&y[i]);
    vy2 = _mm_loadu_ps(&y[i+4]);
    vt1 = _mm_add_ps(_mm_mul_ps(va, vx1), vy1);
    vt2 = _mm_add_ps(_mm_mul_ps(va, vx2), vy2);
    _mm_storeu_ps(&y[i], vt1);
    _mm_storeu_ps(&y[i+4], vt2);
  }
  for (; i < n; ++i) y[i] += a * x[i];
}

void KadMath::vec_mul_sum(int n, float *a, const float *b, const float *c)
{
  int i;
  for (i = 0; i < n; ++i) a[i] += b[i] * c[i];
}

void KadMath::saxpy(int n, float a, const float *x, float *y) { saxpy_inlined(n, a, x, y); }

void KadMath::sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C)
{
  static const int x = 16;
  int i, j, k;
  if (!trans_A && trans_B) {
    for (i = 0; i < M; i += x)
      for (j = 0; j < N; j += x) {
        int ii, ie = M < i + x? M : i + x;
        int jj, je = N < j + x? N : j + x;
        for (ii = i; ii < ie; ++ii) {
          const float *aii = A + ii * K, *bjj;
          float *cii = C + ii * N;
          for (jj = j, bjj = B + j * K; jj < je; ++jj, bjj += K)
            cii[jj] += sdot(K, aii, bjj);
        }
      }
  } else if (!trans_A && !trans_B) {
    for (i = 0; i < M; ++i)
      for (k = 0; k < K; ++k)
        saxpy_inlined(N, A[i*K+k], &B[k*N], &C[i*N]);
  } else if (trans_A && !trans_B) {
    for (k = 0; k < K; ++k)
      for (i = 0; i < M; ++i)
        saxpy_inlined(N, A[k*M+i], &B[k*N], &C[i*N]);
  } else abort();
}

// Legacy compatibility functions for public KadMath methods
float kad_sdot(int n, const float *x, const float *y) { return KadMath::sdot(n, x, y); }
void kad_vec_mul_sum(int n, float *a, const float *b, const float *c) { KadMath::vec_mul_sum(n, a, b, c); }
void kad_saxpy(int n, float a, const float *x, float *y) { KadMath::saxpy(n, a, x, y); }
void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C) { KadMath::sgemm_simple(trans_A, trans_B, M, N, K, A, B, C); }

kad_rng_t kad_rng_dat = { {0x50f5647d2380309dULL, 0x91ffa96fc4c62cceULL}, 0.0, 0, 0 };

uint64_t KadRng::splitmix64(uint64_t x)
{
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

uint64_t KadRng::xoroshiro128plus_next(kad_rng_t *r)
{
  const uint64_t s0 = r->s[0];
  uint64_t s1 = r->s[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  r->s[0] = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
  r->s[1] = s0 << 36 | s0 >> 28;
  return result;
}

void KadRng::xoroshiro128plus_jump(kad_rng_t *r)
{
  static const uint64_t JUMP[] = { 0xbeac0467eba5facbULL, 0xd86b048b86aa9922ULL };
  uint64_t s0 = 0, s1 = 0;
  int i, b;
  for (i = 0; i < 2; ++i)
    for (b = 0; b < 64; b++) {
      if (JUMP[i] & 1ULL << b)
        s0 ^= r->s[0], s1 ^= r->s[1];
      xoroshiro128plus_next(r);
    }
  r->s[0] = s0, r->s[1] = s1;
}

void KadRng::seed(void *d, uint64_t seed)
{
  kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
  r->n_gset = 0.0, r->n_iset = 0;
  r->s[0] = splitmix64(seed);
  r->s[1] = splitmix64(r->s[0]);
}

void *KadRng::create()
{
  kad_rng_t *r;
  r = (kad_rng_t*)calloc(1, sizeof(kad_rng_t));
  xoroshiro128plus_jump(&kad_rng_dat);
  r->s[0] = kad_rng_dat.s[0], r->s[1] = kad_rng_dat.s[1];
  return r;
}

uint64_t KadRng::rand(void *d) { return xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat); }

double KadRng::drand(void *d)
{
  union { uint64_t i; double d; } u;
  u.i = 0x3FFULL << 52 | xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat) >> 12;
  return u.d - 1.0;
}

double KadRng::drand_normal(void *d)
{
  kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
  if (r->n_iset == 0) {
    double fac, rsq, v1, v2;
    do {
      v1 = 2.0 * drand(d) - 1.0;
      v2 = 2.0 * drand(d) - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    r->n_gset = v1 * fac;
    r->n_iset = 1;
    return v2 * fac;
  } else {
    r->n_iset = 0;
    return r->n_gset;
  }
}

// Legacy compatibility functions for public KadRng methods
void *kad_rng(void) { return KadRng::create(); }
void kad_srand(void *d, uint64_t seed) { KadRng::seed(d, seed); }
uint64_t kad_rand(void *d) { return KadRng::rand(d); }
double kad_drand(void *d) { return KadRng::drand(d); }
double kad_drand_normal(void *d) { return KadRng::drand_normal(d); }

// RandomNumberGenerator class implementation - Modern C++ wrapper
RandomNumberGenerator::RandomNumberGenerator(uint64_t seed) {
  rng_.n_gset = 0.0;
  rng_.n_iset = 0;
  rng_.lock = 0;
  if (seed == 0) {
    KadRng::xoroshiro128plus_jump(&kad_rng_dat);
    rng_.s[0] = kad_rng_dat.s[0];
    rng_.s[1] = kad_rng_dat.s[1];
  } else {
    rng_.s[0] = KadRng::splitmix64(seed);
    rng_.s[1] = KadRng::splitmix64(rng_.s[0]);
  }
}

void RandomNumberGenerator::seed(uint64_t seed) {
  rng_.n_gset = 0.0;
  rng_.n_iset = 0;
  rng_.s[0] = KadRng::splitmix64(seed);
  rng_.s[1] = KadRng::splitmix64(rng_.s[0]);
}

uint64_t RandomNumberGenerator::nextInt() {
  return KadRng::xoroshiro128plus_next(&rng_);
}

double RandomNumberGenerator::nextDouble() {
  union { uint64_t i; double d; } u;
  u.i = 0x3FFULL << 52 | KadRng::xoroshiro128plus_next(&rng_) >> 12;
  return u.d - 1.0;
}

double RandomNumberGenerator::nextGaussian() {
  if (rng_.n_iset == 0) {
    double fac, rsq, v1, v2;
    do {
      v1 = 2.0 * nextDouble() - 1.0;
      v2 = 2.0 * nextDouble() - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    rng_.n_gset = v1 * fac;
    rng_.n_iset = 1;
    return v2 * fac;
  } else {
    rng_.n_iset = 0;
    return rng_.n_gset;
  }
}

void KadGraph::copy_dim1(kad_node_t *dst, const kad_node_t *src)
{
  dst->n_d = src->n_d;
  if (src->n_d) memcpy(dst->d, src->d, src->n_d * sizeof(int));
}

// Legacy compatibility function
void kad_copy_dim1(kad_node_t *dst, const kad_node_t *src) { KadGraph::copy_dim1(dst, src); }

int kad_op_add(kad_node_t *p, int action)
{
  int i, n0, n1;
  kad_node_t *q[2];

  q[0] = p->child[0], n0 = q[0]->length();
  q[1] = p->child[1], n1 = q[1]->length();
  if (action == KAD_SYNC_DIM) {
    if (n0 % n1 != 0) return -1;
    kad_copy_dim1(p, q[0]);
  } else if (action == KAD_FORWARD) {
    assert(n0 >= n1);
    memcpy(p->x, q[0]->x, n0 * sizeof(float));
    for (i = 0; i < n0; i += n1)
      KadMath::saxpy(n1, 1.0f, q[1]->x, p->x + i);
  } else if (action == KAD_BACKWARD) {
    if (q[0]->is_back()) KadMath::saxpy(n0, 1.0f, p->g, q[0]->g);
    if (q[1]->is_back())
      for (i = 0; i < n0; i += n1)
        KadMath::saxpy(n1, 1.0f, p->g + i, q[1]->g);
  }
  return 0;
}

int kad_op_mul(kad_node_t *p, int action)
{
  int i, n0, n1;
  kad_node_t *q[2];

  q[0] = p->child[0], n0 = q[0]->length();
  q[1] = p->child[1], n1 = q[1]->length();
  if (action == KAD_SYNC_DIM) {
    if (n0 % n1 != 0) return -1;
    kad_copy_dim1(p, q[0]);
  } else if (action == KAD_FORWARD) {
    assert(n0 >= n1);
    memset(p->x, 0, n0 * sizeof(float));
    if (q[0]->x != 0 && q[1]->x != 0)
      for (i = 0; i < n0; i += n1) 
        kad_vec_mul_sum(n1, p->x + i, q[0]->x + i, q[1]->x);
  } else if (action == KAD_BACKWARD) {
    if (q[0]->is_back() && q[1]->x)
      for (i = 0; i < n0; i += n1)
        kad_vec_mul_sum(n1, q[0]->g + i, p->g + i, q[1]->x);
    if (q[1]->is_back() && q[0]->x)
      for (i = 0; i < n0; i += n1)
        kad_vec_mul_sum(n1, q[1]->g, p->g + i, q[0]->x + i);
  }
  return 0;
}

int kad_op_cmul(kad_node_t *p, int action)
{
  int i, n_a_row, n_b_row, n_col, n_a_col = 1, n_b_col = 1;
  kad_node_t *q[2];

  q[0] = p->child[0], q[1] = p->child[1];
  n_col = q[0]->d[q[0]->n_d - 1] > q[1]->d[q[1]->n_d - 1]? q[0]->d[q[0]->n_d - 1] : q[1]->d[q[1]->n_d - 1];
  for (i = q[0]->n_d - 1; i >= 0; --i) if (n_a_col < n_col) n_a_col *= q[0]->d[i];
  for (i = q[1]->n_d - 1; i >= 0; --i) if (n_b_col < n_col) n_b_col *= q[1]->d[i];
  n_a_row = q[0]->length() / n_a_col, n_b_row = q[1]->length() / n_b_col;
  if (action == KAD_SYNC_DIM) {
    if (n_a_col != n_b_col) return -1;
    p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_row;
  } else if (action == KAD_FORWARD) {
    memset(p->x, 0, n_a_row * n_b_row * sizeof(float));
    if (q[0]->x && q[1]->x)
      KadMath::sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x, q[1]->x, p->x); 
  } else if (action == KAD_BACKWARD) {
    if (q[0]->is_back() && q[1]->x)
      KadMath::sgemm_simple(0, 0, n_a_row, n_col, n_b_row, p->g, q[1]->x, q[0]->g); 
    if (q[1]->is_back() && q[0]->x)
      KadMath::sgemm_simple(1, 0, n_b_row, n_col, n_a_row, p->g, q[0]->x, q[1]->g); 
  }
  return 0;
}

int kad_op_1minus(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = q->length();
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i) p->x[i] = 1.0f - q->x[i];
  } else if (action == KAD_BACKWARD && q->is_back()) {
    KadMath::saxpy(n, -1.0f, p->g, q->g);
  }
  return 0;
}

int kad_op_concat(kad_node_t *p, int action)
{
  kad_node_t *q = p->child[0];
  int32_t *aux;
  int i, j, k, axis, d0, d1;

  assert(p->ptr);
  aux = (int32_t*)p->ptr, axis = aux[0];
  for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
  for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
  if (action == KAD_SYNC_DIM) {
    for (i = 1; i < p->n_child; ++i) {
      if (p->child[i]->n_d != q->n_d) return -1;
      for (j = 0; j < q->n_d; ++j)
        if (j != axis && q->d[j] != p->child[i]->d[j]) return -1;
    }
    kad_copy_dim1(p, q);
    for (i = 1; i < p->n_child; ++i)
      p->d[axis] += p->child[i]->d[axis];
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < d0; ++i)
      for (j = k = 0; j < p->n_child; ++j) {
        q = p->child[j];
        memcpy(&p->x[(i * p->d[axis] + k) * d1], &q->x[i * q->d[axis] * d1], q->d[axis] * d1 * sizeof(float));
        k += q->d[axis];
      }
  } else if (action == KAD_BACKWARD) {
    for (i = 0; i < d0; ++i)
      for (j = k = 0; j < p->n_child; ++j) {
        q = p->child[j];
        if (!q->is_back()) continue;
        KadMath::saxpy(q->d[axis] * d1, 1.0f, &p->g[(i * p->d[axis] + k) * d1], &q->g[i * q->d[axis] * d1]);
        k += q->d[axis];
      }
  }
  return 0;
}

int kad_op_ce_multi(kad_node_t *p, int action)
{
  static const float tiny = 1e-9f;
  kad_node_t *y1 = p->child[0]; 
  kad_node_t *y0 = p->child[1]; 
  kad_node_t *c = 0;
  int i, j, n1, d0;

  n1 = y0->d[y0->n_d - 1];
  d0 = y0->length() / n1;
  if (p->n_child == 3) {
    c = p->child[2];
    assert(c->n_d == 1 && c->d[0] == n1);
  }
  if (action == KAD_SYNC_DIM) {
    if (y0->length() != y1->length() || y0->d[y0->n_d - 1] != y1->d[y1->n_d - 1]) return -1;
    p->n_d = 0;
  } else if (action == KAD_FORWARD) {
    double cost = 0.0;
    if (c == 0) {
      for (j = 0; j < d0; ++j) {
        float *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
        for (i = 0; i < n1; ++i)
          if (x0[i] > 0.0f)
            cost += x0[i] * log(x0[i] / (x1[i] > tiny? x1[i] : tiny));
      }
    } else {
      for (j = 0; j < d0; ++j) {
        float *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
        for (i = 0; i < n1; ++i)
          if (x0[i] > 0.0f)
            cost += c->x[i] * x0[i] * log(x0[i] / (x1[i] > tiny? x1[i] : tiny));
      }
    }
    p->x[0] = (float)(cost / d0);
  } else if (action == KAD_BACKWARD && y1->is_back()) {
    float t = p->g[0] / d0;
    if (c == 0) {
      for (j = 0; j < d0; ++j) {
        float *g = &y1->g[j * n1], *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
        for (i = 0; i < n1; ++i)
          g[i] -= t * x0[i] / (x1[i] > tiny? x1[i] : tiny);
      }
    } else {
      for (j = 0; j < d0; ++j) {
        float *g = &y1->g[j * n1], *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
        for (i = 0; i < n1; ++i)
          g[i] -= t * c->x[i] * x0[i] / (x1[i] > tiny? x1[i] : tiny);
      }
    }
  }
  return 0;
}

int kad_op_stdnorm(kad_node_t *p, int action)
{
  int i, j, n, m;
  kad_node_t *q = p->child[0];
  assert(q->n_d > 0);
  n = q->d[q->n_d - 1];
  m = q->length() / n;
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_ALLOC) {
    p->gtmp = realloc(p->gtmp, m * sizeof(float));
  } else if (action == KAD_FORWARD) {
    float *si = (float*)p->gtmp;
    for (j = 0; j < m; ++j) {
      float *px = &p->x[j * n], *qx = &q->x[j * n];
      float avg, std_inv;
      double s;
      for (i = 0, s = 0.0; i < n; ++i) s += qx[i];
      avg = (float)(s / n);
      for (i = 0; i < n; ++i) px[i] = qx[i] - avg;
      for (i = 0, s = 0.0; i < n; ++i) s += px[i] * px[i];
      std_inv = s == 0.0? 1.0f : (float)(1.0 / sqrt(s / n));
      for (i = 0; i < n; ++i) px[i] *= std_inv;
      si[j] = std_inv;
    }
  } else if (action == KAD_BACKWARD && q->is_back()) {
    float *si = (float*)p->gtmp;
    for (j = 0; j < m; ++j) {
      float *pg = &p->g[j * n], *qg = &q->g[j * n], *px = &p->x[j * n], std_inv = si[j];
      double s, t;
      for (i = 0, s = t = 0.0; i < n; ++i)
        s += pg[i], t += px[i] * pg[i];
      s /= n, t /= n;
      for (i = 0; i < n; ++i)
        qg[i] += std_inv * (pg[i] - s - px[i] * t);
    }
  }
  return 0;
}

int kad_op_sigm(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = q->length();
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i)
      p->x[i] = 1.0f / (1.0f + expf(-q->x[i]));
  } else if (action == KAD_BACKWARD && q->is_back()) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * (p->x[i] * (1.0f - p->x[i]));
  }
  return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = q->length();
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i) {
      if (q->x[i] < -20.0f) p->x[i] = -1.0f;
      else {
        float y;
        y = expf(-2.0f * q->x[i]);
        p->x[i] = (1.0f - y) / (1.0f + y);
      }
    }
  } else if (action == KAD_BACKWARD && q->is_back()) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * (1.0f - p->x[i] * p->x[i]);
  }
  return 0;
}

int kad_op_softmax(kad_node_t *p, int action)
{
  int i, j, n1, d0;
  kad_node_t *q = p->child[0];

  n1 = q->d[q->n_d - 1];
  d0 = q->length() / n1;
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (j = 0; j < d0; ++j) {
      float s, max, *x = &q->x[j * n1], *y = &p->x[j * n1];
      for (i = 0, max = -FLT_MAX; i < n1; ++i)
        max = max > x[i]? max : x[i];
      for (i = 0, s = 0.0f; i < n1; ++i) {
        y[i] = expf(x[i] - max);
        s += y[i];
      }
      for (i = 0, s = 1.0f / s; i < n1; ++i) y[i] *= s;
    }
  } else if (action == KAD_BACKWARD && q->is_back()) {
    for (j = 0; j < d0; ++j) {
      float s, *g = &p->g[j * n1], *y = &p->x[j * n1], *h = &q->g[j * n1];
      for (i = 0, s = 0.0f; i < n1; ++i)
        s += g[i] * y[i];
      for (i = 0; i < n1; ++i)
        h[i] += y[i] * (g[i] - s);
    }
  }
  return 0;
}

int kad_op_avg(kad_node_t *p, int action)
{
  int i, n;
  float tmp;
  kad_node_t *q;

  assert(p->n_child > 0);
  tmp = 1.0f / p->n_child;
  q = p->child[0];
  n = q->length();
  if (action == KAD_SYNC_DIM) {
    for (i = 1; i < p->n_child; ++i)
      if (p->child[i]->length() != n) return -1;
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    memcpy(p->x, q->x, n * sizeof(float));
    for (i = 1; i < p->n_child; ++i)
      KadMath::saxpy(n, 1.0f, p->child[i]->x, p->x);
    for (i = 0; i < n; ++i) p->x[i] *= tmp;
  } else if (action == KAD_BACKWARD) {
    for (i = 0; i < p->n_child; ++i)
      if (p->child[i]->is_back())
        KadMath::saxpy(n, tmp, p->g, p->child[i]->g);
  }
  return 0;
}

// Inline function replacing conv_out_size macro
inline int conv_out_size(int in_size, const conv_conf_t *aux) {
  return ((in_size) - aux->kernel_size + aux->pad[0] + aux->pad[1]) / aux->stride + 1;
}

// Inline function replacing process_row_for macro
inline void process_row_for(const float *_xx, const float *_ww, float *_yy, int _wn, int _pn, int _stride, int _pad, float *_t) {
  int j, l;
  if (_stride > 1) {
    for (l = 0; l < _wn; ++l) {
      const float *xl = &_xx[l - _pad];
      for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl;
      KadMath::saxpy(_pn, _ww[l], _t, _yy);
    }
  } else for (l = 0; l < _wn; ++l) KadMath::saxpy(_pn, _ww[l], &_xx[l - _pad], _yy);
}

// Inline function replacing process_row_back_x macro
inline void process_row_back_x(float *_xx, const float *_ww, const float *_yy, int _wn, int _pn, int _stride, int _pad, float *_t) {
  int j, l;
  if (_stride > 1) {
    for (l = 0; l < _wn; ++l) {
      float *xl = &_xx[l - _pad];
      memset(_t, 0, _pn * sizeof(float));
      KadMath::saxpy(_pn, _ww[l], _yy, _t);
      for (j = 0; j < _pn; ++j, xl += _stride) *xl += _t[j];
    }
  } else for (l = 0; l < _wn; ++l) KadMath::saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]);
}

// Inline function replacing process_row_back_w macro
inline void process_row_back_w(const float *_xx, float *_ww, const float *_yy, int _wn, int _pn, int _stride, int _pad, float *_t) {
  int j, l;
  if (_stride > 1) {
    for (l = 0; l < _wn; ++l) {
      const float *xl = &_xx[l - _pad];
      for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl;
      _ww[l] += KadMath::sdot(_pn, _yy, _t);
    }
  } else for (l = 0; l < _wn; ++l) _ww[l] += KadMath::sdot(_pn, _yy, &_xx[l - _pad]);
}

kad_op_f kad_op_list[KAD_MAX_OP] = {
  0,
  kad_op_add,        
  kad_op_mul,        
  kad_op_cmul,       
  0, 
  0,     
  kad_op_sigm,       
  kad_op_tanh,       
  0,       
  0,     
  kad_op_avg,        
  kad_op_1minus,     
  0,     
  kad_op_ce_multi,   
  kad_op_softmax,    
  0,    
  0,     
  0,      
  0,     
  0,      
  0,      
  0,        
  0,     
  0,        
  0,  
  0,     
  0,    
  0,        
  0,      
  0,        
  0,    
  kad_op_concat,     
  kad_op_stdnorm,    
  0,        
  0,        
  0,      
  0     
};

void kad_trap_fe(void)
{
#ifdef __SSE__
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
}

void kad_add_delta(int n, kad_node_t **a, float c, float *delta)
{
  int i, k;
  for (i = k = 0; i < n; ++i)
    if (a[i]->is_var()) {
      KadMath::saxpy(a[i]->length(), c, &delta[k], a[i]->x);
      k += a[i]->length();
    }
}

// Range coder constants
const int SCALElog = 15;
const int SCALE    = 1<<SCALElog;

// RangeCoder class - Modern C++ arithmetic coder with proper encapsulation
class RangeCoder {
private:
  static const uint NUM   = 4;
  static const uint sTOP  = 0x01000000U;
  static const uint gTOP  = 0x00010000U;
  static const uint Thres = 0xFF000000U;

  uint f_DEC_{0};
  FILE* f_{nullptr};
  qword counter_{0};

  union {
    struct {
      uint  low_;
      uint  Carry_;
    };
    qword lowc_;
  };
  uint  code_{0};
  uint  FFNum_{0};
  uint  Cache_{0};
  qword range_{0};

  byte get() { counter_++; return byte(getc(f_)); }
  void put(byte c) { counter_++; putc(c, f_); }

  uint muldivR(uint a, uint b) { return (qword(a)*b)/range_; }
  uint mulRdiv(uint a, uint c) { return (qword(a)*range_)/c; }

  void rc_Renorm() {
    while(range_ < sTOP) ShiftStuff();
  }

  void ShiftStuff() {
    range_ = (range_<<8)+0x00;
    if(f_DEC_ == 0) ShiftLow(); else ShiftCode();
  }

  void ShiftCode() {
    code_ = (code_<<8)+0x00;
    uint c = get();
    FFNum_ += (c==-1);
    code_ += byte(c);
  }

  void ShiftLow() {
    if((low_ < Thres) || Carry_) {
      if(Cache_ != uint(-1)) put(Cache_ + Carry_);
      for(; FFNum_ != 0; FFNum_--) put(Carry_ - 1);
      Cache_ = low_ >> 24;
      Carry_ = 0;
    } else FFNum_++;
    low_ <<= 8;
  }

  void rc_Init() {
    low_   = 0;
    FFNum_ = 0;
    Carry_ = 0;
    Cache_ = uint(-1);
    if(f_DEC_ == 1) {
      for(int _ = 0; _ < NUM; _++) ShiftCode();
    }
    range_ = 1ULL << 32;
    counter_ = 0;
  }

  void rc_Quit() {
    if(f_DEC_ == 0) {
      uint i, n = NUM;

      qword llow = low_;
      qword high = llow + range_;

      if((llow |       0xFF) < high) low_ |=       0xFF, n--;
      if((llow |     0xFFFF) < high) low_ |=     0xFFFF, n--;
      if((llow |   0xFFFFFF) < high) low_ |=   0xFFFFFF, n--;
      if((llow | 0xFFFFFFFF) < high) low_ |= 0xFFFFFFFF, n--;

      if((Cache_ != uint(-1)) && ((n != 0) || (Cache_ + Carry_ != 0xFF))) put(Cache_ + Carry_);
      if((n == 0) && (Carry_ == 0)) FFNum_ = 0;
      for(i = 0; i < FFNum_; i++) put(0xFF + Carry_);
      for(i = 0; i < n; i++) put(low_ >> 24), low_ <<= 8;
    }
  }

public:
  RangeCoder() = default;

  void StartEncode(FILE *F) { f_ = F; f_DEC_ = 0; rc_Init(); }
  void FinishEncode() { rc_Quit(); }
  void StartDecode(FILE *F) { f_ = F; f_DEC_ = 1; rc_Init(); }
  qword Counter() const { return counter_; }

  void rc_Process(uint cumFreq, uint freq, uint totFreq) {
    uint tmp  = range_ - mulRdiv(totFreq - cumFreq, totFreq);
    uint rnew = range_ - mulRdiv(totFreq - (cumFreq + freq), totFreq);

    if(f_DEC_) code_ -= tmp; else lowc_ += tmp;
    range_ = rnew - tmp;

    rc_Renorm();
  }

  uint rc_GetFreq(uint totFreq) {
    return muldivR(code_, totFreq);
  }
};

// Legacy struct name for backward compatibility
struct Rangecoder : public RangeCoder {
  // Inherit everything from RangeCoder
};

Rangecoder rc;

// Version constants - replacing macros with constexpr
constexpr uint8_t VERSION_ID = 0x04;
constexpr const char* VERSION_DATE = "2024/07/26";

typedef struct kanncompr_s {

  kann_t *ann;

  int n_char_in, n_char_out;
  int rnn_type;
  int n_layers, n_neurons, ulen;
  int n_layers_embed_hidden, n_layers_embed_output;
  float dropout;
  float temper;
  int var_h0, norm;

  byte bias_init_type[4];
  float bias_init_from_to[4][4];

  uint seed;
  float grad_clip;

  kann_t *ua;
  int n_var;
  float **x, **y, *xp;
  float *m, *v;
  word mini_batch_freq, mini_batch_freq_cnt, mini_batch_size, mini_batch_step;
  byte *symb_list;
  int symb_list_ndx, symb_list_size;

  float alpha1, alpha2, alpha1d;
  float beta1, beta1t, beta2, beta2t;
  float eps;

  byte vocab_type, *vocab_symb;
} kanncompr_t;

// AdamOptimizer class - Modern C++ encapsulation of optimizer state
class AdamOptimizer {
private:
  float alpha1_, alpha2_, alpha1d_;
  float beta1_, beta1t_, beta2_, beta2t_;
  float eps_;
  float *m_{nullptr};  // First moment
  float *v_{nullptr};  // Second moment
  int n_vars_{0};
  bool owns_memory_{false};

public:
  // Constructor with parameters
  AdamOptimizer(float alpha1, float alpha2, float alpha1d,
                float beta1, float beta1t, float beta2, float beta2t, float eps)
    : alpha1_(alpha1), alpha2_(alpha2), alpha1d_(alpha1d),
      beta1_(beta1), beta1t_(beta1t), beta2_(beta2), beta2t_(beta2t), eps_(eps) {}

  // Initialize moment buffers (takes ownership)
  void initialize(int n_vars, float *m, float *v, bool own = false) {
    n_vars_ = n_vars;
    m_ = m;
    v_ = v;
    owns_memory_ = own;
  }

  // Allocate and initialize moment buffers
  void allocate(int n_vars, bool allocate_m = true) {
    n_vars_ = n_vars;
    if (allocate_m && (beta1_ != 0.0 || beta1t_ != 0.0)) {
      m_ = (float*)calloc(n_vars, sizeof(float));
    } else {
      m_ = nullptr;
    }
    v_ = (float*)calloc(n_vars, sizeof(float));
    owns_memory_ = true;
  }

  // Perform optimization step
  void step(float *gradients, float *weights) {
    const float weight_decay = 0.0f;
    const int decoupled_weight_decay = 1;

    if (weight_decay != 0.0f) {
      if (decoupled_weight_decay) {
        for (int i = 0; i < n_vars_; i++)
          weights[i] -= alpha1_ * weight_decay * weights[i];
      } else {
        for (int i = 0; i < n_vars_; i++)
          gradients[i] += weight_decay * weights[i];
      }
    }

    if (m_ != nullptr) {
      for (int i = 0; i < n_vars_; i++) {
        m_[i] = (1.0f - beta1_) * gradients[i] + beta1_ * m_[i];
        v_[i] = (1.0f - beta2_) * gradients[i] * gradients[i] + beta2_ * v_[i];
        weights[i] -= alpha1_ * (m_[i] / (1.0f - beta1t_)) / sqrtf(v_[i] / (1.0f - beta2t_) + eps_);
      }
    } else {
      for (int i = 0; i < n_vars_; i++) {
        v_[i] = (1.0f - beta2_) * gradients[i] * gradients[i] + beta2_ * v_[i];
        weights[i] -= alpha1_ * gradients[i] / sqrtf(v_[i] / (1.0f - beta2t_) + eps_);
      }
    }

    // Update learning rate schedule
    alpha1_ = alpha1_ > alpha2_ ? alpha1_ * alpha1d_ : alpha2_;
    beta1t_ *= beta1_;
    beta2t_ *= beta2_;
  }

  // Getters for updated parameters
  float alpha1() const { return alpha1_; }
  float beta1t() const { return beta1t_; }
  float beta2t() const { return beta2t_; }

  // Setters for updating parameters
  void setAlpha1(float alpha1) { alpha1_ = alpha1; }
  void setBeta1t(float beta1t) { beta1t_ = beta1t; }
  void setBeta2t(float beta2t) { beta2t_ = beta2t; }

  // Get moment pointers
  float* m() { return m_; }
  float* v() { return v_; }

  // Cleanup
  void cleanup() {
    if (owns_memory_) {
      if (m_ != nullptr) free(m_);
      free(v_);
      owns_memory_ = false;
    }
    m_ = nullptr;
    v_ = nullptr;
    n_vars_ = 0;
  }

  ~AdamOptimizer() {
    cleanup();
  }
};

// CompressionCodec class - Modern C++ wrapper for compression state with RAII
class CompressionCodec {
private:
  kann_t *ann_{nullptr};
  kann_t *ua_{nullptr};
  int n_char_in_, n_char_out_;
  int rnn_type_;
  int n_layers_, n_neurons_, ulen_;
  int n_layers_embed_hidden_, n_layers_embed_output_;
  float dropout_;
  float temper_;
  int var_h0_, norm_;
  byte bias_init_type_[4];
  float bias_init_from_to_[4][4];
  uint seed_;
  float grad_clip_;
  int n_var_{0};
  float **x_{nullptr};
  float **y_{nullptr};
  float *xp_{nullptr};
  word mini_batch_freq_, mini_batch_freq_cnt_, mini_batch_size_, mini_batch_step_;
  byte *symb_list_{nullptr};
  int symb_list_ndx_{0}, symb_list_size_{0};
  byte vocab_type_{0};
  byte *vocab_symb_{nullptr};

  AdamOptimizer optimizer_;

public:
  // Constructor with configuration
  CompressionCodec(int n_char_in, int n_char_out, int rnn_type,
                   int n_layers, int n_neurons, int ulen,
                   int n_layers_embed_hidden, int n_layers_embed_output,
                   float dropout, float temper, int var_h0, int norm,
                   const byte bias_init_type[4], const float bias_init_from_to[4][4],
                   uint seed, float grad_clip,
                   word mini_batch_freq, word mini_batch_size, word mini_batch_step,
                   float alpha1, float alpha2, float alpha1d,
                   float beta1, float beta1t, float beta2, float beta2t, float eps)
    : n_char_in_(n_char_in), n_char_out_(n_char_out), rnn_type_(rnn_type),
      n_layers_(n_layers), n_neurons_(n_neurons), ulen_(ulen),
      n_layers_embed_hidden_(n_layers_embed_hidden), n_layers_embed_output_(n_layers_embed_output),
      dropout_(dropout), temper_(temper), var_h0_(var_h0), norm_(norm),
      seed_(seed), grad_clip_(grad_clip),
      mini_batch_freq_(mini_batch_freq), mini_batch_freq_cnt_(0),
      mini_batch_size_(mini_batch_size), mini_batch_step_(mini_batch_step),
      optimizer_(alpha1, alpha2, alpha1d, beta1, beta1t, beta2, beta2t, eps)
  {
    memcpy(bias_init_type_, bias_init_type, 4 * sizeof(byte));
    memcpy(bias_init_from_to_, bias_init_from_to, 4 * 4 * sizeof(float));
    symb_list_size_ = ulen_ + mini_batch_step_ * (mini_batch_size_ - 1) + 1;
  }

  // Methods that replace global functions
  void buildNetwork();  // Replaces ann_structure
  void initialize();    // Replaces ann_init
  void predict(unsigned *freq, unsigned *total);  // Replaces ann_predict
  void train();         // Replaces ann_train
  void cleanup();       // Replaces ann_end

  // Accessors for symbol list manipulation
  void incrementSymbolIndex() {
    symb_list_ndx_ = (symb_list_ndx_ + 1) % symb_list_size_;
  }
  void setCurrentSymbol(byte symbol) {
    symb_list_[symb_list_ndx_] = symbol;
  }
  int getSymbolIndex() const { return symb_list_ndx_; }

  // Mini-batch frequency counter
  bool shouldTrain() {
    if (++mini_batch_freq_cnt_ == mini_batch_freq_) {
      mini_batch_freq_cnt_ = 0;
      return true;
    }
    return false;
  }

  // Allocate symbol list
  void allocateSymbolList() {
    symb_list_ = (byte*)calloc(symb_list_size_, sizeof(byte));
  }

  // Free symbol list
  void freeSymbolList() {
    free(symb_list_);
    symb_list_ = nullptr;
  }

  ~CompressionCodec() {
    cleanup();
    if (symb_list_) freeSymbolList();
  }
};

typedef struct stats_s {
  uint orig_current;
  uint orig_total;
  uint next_display;
  uint next_display_step;
  qword rc_previous;
  uint orig_previous;
} stats_t;

// Template function replacing sizeof_array macro
template<typename T, size_t N>
constexpr size_t sizeof_array(const T (&)[N]) { return N; }

// Legacy Adam function for backward compatibility
void Adam(const int n_var, const float alpha, const float beta1, const float beta1t, const float beta2, const float beta2t, const float eps, float *g, float *t, float *m, float *v) {
  const float weight_decay = 0.0f; 
  const int decoupled_weight_decay = 1;

  if( weight_decay!=0.0f ) {
    if( decoupled_weight_decay ) {
      for(int i = 0; i < n_var; i++) t[i] -= alpha * weight_decay * t[i];
    } else {
      for(int i = 0; i < n_var; i++) g[i] += weight_decay * t[i];
    }
  }

  if( m!=NULL ) {
    // does it ever get in here?
    for(int i = 0; i < n_var; i++) {
      m[i] = (1.0f - beta1) * g[i] + beta1 * m[i];

      v[i] = (1.0f - beta2) * g[i] * g[i] + beta2 * v[i];

      t[i] -= alpha * (m[i] / (1.0f - beta1t)) / sqrtf(v[i] / (1.0f - beta2t) + eps); 

    }
  } else {
    for(int i = 0; i < n_var; i++) {
      v[i] = (1.0f - beta2) * g[i] * g[i] + beta2 * v[i];

      t[i] -= alpha * g[i] / sqrtf(v[i] / (1.0f - beta2t) + eps); 
    }
  }
}

// KANNCOMPR RNN flags - replacing macros with constexpr
constexpr uint32_t KANNCOMPR_RNN_VAR_H0 = 0x0001;
constexpr uint32_t KANNCOMPR_RNN_NORM = 0x0002;
constexpr uint32_t KANNCOMPR_GRU_MINIMAL_GATED_UNIT = 0x0004;
constexpr uint32_t KANNCOMPR_LSTM_INPUT_FORGET_GATE_COUPLED = 0x0008;
constexpr uint32_t KANNCOMPR_LSTM_MV_VARIANT1 = 0x1000;
constexpr uint32_t KANNCOMPR_LSTM_MV_VARIANT2 = 0x2000;
constexpr uint32_t KANNCOMPR_LSTM_MV_VARIANT = KANNCOMPR_LSTM_MV_VARIANT1 | KANNCOMPR_LSTM_MV_VARIANT2;

kad_node_t *kanncompr_new_vector(const int n, const int x_init_type, const float x_init_from_to[2]) {
  int i, offset = 0;
  double sdev_inv, x_init_from, x_init_to, x_init_len;
  kad_node_p par[1];
  kad_node_t *leaf, *p;

  x_init_from = x_init_from_to[0];
  x_init_to   = x_init_from_to[1];
  x_init_len  = x_init_to - x_init_from;
  par[0] = 0;

  leaf = kann_new_leaf2(&offset, par, KAD_VAR, x_init_from, 1, n);

  p = par[0];
  if(x_init_type == 4) {
    sdev_inv    = 1.0 / sqrt((double)n / p->d[0]);
    x_init_len *= sdev_inv;
  }

  for(i = 0; i < n; ++i) {
    switch(x_init_type) {
    case 1 : { p->x[i] = x_init_from;                                             break; }
    case 2 : { p->x[i] = x_init_from + x_init_len * i / (n - 1);                  break; }
    case 3 : { p->x[i] = (float)(kad_drand(0) * x_init_len + x_init_from);        break; }
    case 4 :
    case 5 : { p->x[i] = (float)(kad_drand_normal(0) * x_init_len + x_init_from); break; }
    default: {                                                                    break; }
    }
  }

  return leaf;
}

kad_node_t *kanncompr_new_bias(const int n, const int x_init_type, const float x_init_from_to[2]) {
  return kanncompr_new_vector(n, x_init_type, x_init_from_to);
}

kad_node_t *kanncompr_layer_lstm(kad_node_t *in, int n1, uint rnn_flag, byte bias_init_type[4], float bias_init_from_to[4][2]) {
  int n0;
  kad_node_t *i, *f, *o, *g, *w, *u, *b, *h0, *c0, *c, *out;
  kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANNCOMPR_RNN_NORM)? kann_cmul_norm : kad_cmul;
  n0 = in->n_d >= 2? in->length() / in->d[0] : in->length();
  h0 = (rnn_flag & KANNCOMPR_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  h0->x = (float*)calloc(n1, sizeof(float));
  c0 = (rnn_flag & KANNCOMPR_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  c0->x = (float*)calloc(n1, sizeof(float));

  if (!(rnn_flag & KANNCOMPR_LSTM_INPUT_FORGET_GATE_COUPLED)) {
    w = kann_new_weight(n1, n0);
    u = kann_new_weight(n1, n1);
    
    b = kanncompr_new_bias(n1, bias_init_type[0], bias_init_from_to[0]);
    i = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  }
  
  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  
  b = kanncompr_new_bias(n1, bias_init_type[1], bias_init_from_to[1]); 
  f = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));

  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  
  b = kanncompr_new_bias(n1, bias_init_type[2], bias_init_from_to[2]);
  o = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  
  if (!(rnn_flag & KANNCOMPR_LSTM_MV_VARIANT1)) {
    w = kann_new_weight(n1, n0);
    u = kann_new_weight(n1, n1);
    
    b = kanncompr_new_bias(n1, bias_init_type[3], bias_init_from_to[3]);
    g = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  } else {
    w = kann_new_weight(n1, n0);
    
    b = kanncompr_new_bias(n1, bias_init_type[3], bias_init_from_to[3]);
    g = kad_tanh(kad_add(cmul(in, w), b));
  }
  if (!(rnn_flag & KANNCOMPR_LSTM_INPUT_FORGET_GATE_COUPLED)) {
    
    c = kad_add(kad_mul(f, c0), kad_mul(g, i)); 
  }
  else {
    
    c = kad_add(kad_mul(f, c0), kad_mul(g, kad_1minus(f))); 
  }
  
  if (!(rnn_flag & KANNCOMPR_LSTM_MV_VARIANT2)) {
    c->pre = c0;
    if (rnn_flag & KANNCOMPR_RNN_NORM) c = kann_layer_layernorm(c); 

    out = kad_mul(kad_tanh(c), o);
  }
  else {
    c = kad_tanh(c);
    c->pre = c0;
    out = kad_mul(c, o);
  }
  out->pre = h0;

  return out;
}

kad_node_t *kanncompr_layer_cost(kad_node_t *t, int n_out, float temper) {
  kad_node_t *temp = 0, *cost = 0, *truth = 0;

  t = kann_layer_dense(t, n_out);

  if (temper > 0.0) {
    temp = kann_new_scalar(KAD_CONST, 1.0f / temper);
    t = kad_mul(t, temp); 
  }
  else if (temper < 0.0) {
    temp = kann_new_vec(n_out, 1.0f / -temper);
    t = kad_mul(t, temp); 
  } 

  t = kad_softmax(t);

  truth = kad_feed(2, 1, n_out), truth->ext_flag |= KANN_F_TRUTH;

  cost = kad_ce_multi(t, truth);
  t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;

  return cost;
}

kann_t *ann_structure(kanncompr_t options) {
  int i;
  unsigned int j, k;
  int n_layers_1 = options.n_layers - 1 + (options.n_layers == 1);
  float bias_init_from_to[4][2];
  uint rnn_flag = (options.var_h0 ? KANNCOMPR_RNN_VAR_H0 : 0) | (options.norm ? KANNCOMPR_RNN_NORM : 0);
  kad_node_t **tlist = (kad_node_t**)malloc(options.n_layers * sizeof(kad_node_t*));

  kad_node_t *t = kann_layer_input(options.n_char_in);

  for(i = 0; i < options.n_layers; ++i){
    if(i >= 2 && options.n_layers_embed_hidden >= 2)
      t = i <= options.n_layers_embed_hidden ? kad_concat_array(1, i, &tlist[0]) : kad_concat_array(1, options.n_layers_embed_hidden, &tlist[i - options.n_layers_embed_hidden]);

    for(j = 0; j < sizeof_array(bias_init_from_to); ++j)
      for(k = 0; k < sizeof_array(bias_init_from_to[0]); ++k)
        bias_init_from_to[j][k] = ((n_layers_1 - i) * options.bias_init_from_to[j][k] + i * options.bias_init_from_to[j][k + 2]) / n_layers_1;

    switch(options.rnn_type) {
    case 51: 
      t = kanncompr_layer_lstm(t, options.n_neurons, rnn_flag | KANNCOMPR_LSTM_MV_VARIANT | KANNCOMPR_LSTM_INPUT_FORGET_GATE_COUPLED, options.bias_init_type, bias_init_from_to);
      break;
    default:;
      break;
    }

    if(options.dropout > 0.0) t = kann_layer_dropout(t, options.dropout);

    tlist[i] = t;
  }

  if(i >= 2 && options.n_layers_embed_output >= 2)
    t = i <= options.n_layers_embed_output ? kad_concat_array(1, i, &tlist[0]) : kad_concat_array(1, options.n_layers_embed_output, &tlist[i - options.n_layers_embed_output]);

  free(tlist);

  return kann_new(kanncompr_layer_cost(t, options.n_char_out, options.temper), 0);
}

void ann_init(kanncompr_t *options) {
  options->x = (float**)malloc(options->ulen * sizeof(float*));
  options->y = (float**)malloc(options->ulen * sizeof(float*));
  for(int k = 0; k < options->ulen; k++) {
    options->x[k] = (float*)calloc(options->mini_batch_size * options->n_char_in , sizeof(float));
    options->y[k] = (float*)calloc(options->mini_batch_size * options->n_char_out, sizeof(float));
  }

  options->xp = (float*)calloc(options->n_char_in, sizeof(float));

  options->n_var = kann_size_var(options->ann); 

  options->ua = kann_unroll(options->ann, options->ulen); 
  kann_set_batch_size(options->ua, options->mini_batch_size);

  if(options->beta1 != 0.0 || options->beta1t != 0.0)
    options->m = (float*)calloc(options->n_var, sizeof(float)); 
  options->v = (float*)calloc(options->n_var, sizeof(float)); 

  kann_feed_bind(options->ua, KANN_F_IN,    0, options->x); 
  kann_feed_bind(options->ua, KANN_F_TRUTH, 0, options->y); 
  kann_switch(options->ua, 1);

  kann_rnn_start(options->ann);
}

void ann_predict(kanncompr_t *options, unsigned *freq, unsigned *total) {
  unsigned i;
  float sum = 0.0;

  options->xp[options->symb_list[options->symb_list_ndx]] = 1.0;
  const float *yp = kann_apply1(options->ann, options->xp);
  options->xp[options->symb_list[options->symb_list_ndx]] = 0.0;

  for(i = 0; i < (unsigned)options->n_char_out; i++) sum += yp[i]; 
  for(i = 0; i < (unsigned)options->n_char_out; i++) {
    freq[i] = yp[i] * (SCALE - options->n_char_out - 1) / sum;
    
    freq[i] += freq[i] == 0;
  }
  *total = 1;
  for(i = 0; i < (unsigned)options->n_char_out; i++) *total += freq[i];
  if(*total >= SCALE) exit(6);
}

void ann_train(kanncompr_t *options) {
  int ul_x, ul_y = (options->symb_list_ndx + 1) % options->symb_list_size;
  for(int ul = 0; ul < options->ulen; ul++) {
    memset(options->x[ul], 0, options->mini_batch_size * options->n_char_in  * sizeof(float));
    memset(options->y[ul], 0, options->mini_batch_size * options->n_char_out * sizeof(float));

    ul_x = ul_y;
    ul_y = (ul_y + 1) % options->symb_list_size;
    for(int mbs = 0, mbs_x = ul_x, mbs_y = ul_y; mbs < options->mini_batch_size; mbs++, mbs_x = (mbs_x + options->mini_batch_step) % options->symb_list_size, mbs_y = (mbs_y + options->mini_batch_step) % options->symb_list_size) {
      options->x[ul][mbs * options->n_char_in  + options->symb_list[mbs_x]] = 1.0;
      options->y[ul][mbs * options->n_char_out + options->symb_list[mbs_y]] = 1.0;
    }
  }

  kann_cost(options->ua, 0, 1);

  if(options->grad_clip > 0.0f) {
    kann_grad_clip(options->grad_clip, options->n_var, options->ua->g);
    
  } else if(options->grad_clip < 0.0f) {
    for(int i = 0; i < options->n_var; ++i)
      if(options->ua->g[i] < options->grad_clip)
        options->ua->g[i] = options->grad_clip;
      else if(options->ua->g[i] > -options->grad_clip)
        options->ua->g[i] = -options->grad_clip;
  } 

  Adam(options->n_var, options->alpha1, options->beta1, options->beta1t, options->beta2, options->beta2t, options->eps, options->ua->g, options->ua->x, options->m, options->v);

  options->alpha1  = options->alpha1 > options->alpha2 ? options->alpha1 * options->alpha1d : options->alpha2;
  options->beta1t *= options->beta1;
  options->beta2t *= options->beta2;
}

void ann_end(kanncompr_t options) {
  kann_rnn_end(options.ann);

  kann_switch(options.ua, 0);

  kann_delete_unrolled(options.ua);

  kann_delete(options.ann);

  if(options.m != NULL) free(options.m);
  free(options.v);
  free(options.xp);
  for(int k = 0; k < options.ulen; k++) { free(options.y[k]); free(options.x[k]); }
  free(options.y); free(options.x);
}

// CompressionCodec method implementations

void CompressionCodec::buildNetwork() {
  int i;
  unsigned int j, k;
  int n_layers_1 = n_layers_ - 1 + (n_layers_ == 1);
  float bias_init_from_to[4][2];
  uint rnn_flag = (var_h0_ ? KANNCOMPR_RNN_VAR_H0 : 0) | (norm_ ? KANNCOMPR_RNN_NORM : 0);
  kad_node_t **tlist = (kad_node_t**)malloc(n_layers_ * sizeof(kad_node_t*));

  kad_node_t *t = kann_layer_input(n_char_in_);

  for(i = 0; i < n_layers_; ++i){
    if(i >= 2 && n_layers_embed_hidden_ >= 2)
      t = i <= n_layers_embed_hidden_ ? kad_concat_array(1, i, &tlist[0]) : kad_concat_array(1, n_layers_embed_hidden_, &tlist[i - n_layers_embed_hidden_]);

    for(j = 0; j < sizeof_array(bias_init_from_to); ++j)
      for(k = 0; k < sizeof_array(bias_init_from_to[0]); ++k)
        bias_init_from_to[j][k] = ((n_layers_1 - i) * bias_init_from_to_[j][k] + i * bias_init_from_to_[j][k + 2]) / n_layers_1;

    switch(rnn_type_) {
    case 51:
      t = kanncompr_layer_lstm(t, n_neurons_, rnn_flag | KANNCOMPR_LSTM_MV_VARIANT | KANNCOMPR_LSTM_INPUT_FORGET_GATE_COUPLED, bias_init_type_, bias_init_from_to);
      break;
    default:;
      break;
    }

    if(dropout_ > 0.0) t = kann_layer_dropout(t, dropout_);

    tlist[i] = t;
  }

  if(i >= 2 && n_layers_embed_output_ >= 2)
    t = i <= n_layers_embed_output_ ? kad_concat_array(1, i, &tlist[0]) : kad_concat_array(1, n_layers_embed_output_, &tlist[i - n_layers_embed_output_]);

  free(tlist);

  ann_ = kann_new(kanncompr_layer_cost(t, n_char_out_, temper_), 0);
}

void CompressionCodec::initialize() {
  x_ = (float**)malloc(ulen_ * sizeof(float*));
  y_ = (float**)malloc(ulen_ * sizeof(float*));
  for(int k = 0; k < ulen_; k++) {
    x_[k] = (float*)calloc(mini_batch_size_ * n_char_in_ , sizeof(float));
    y_[k] = (float*)calloc(mini_batch_size_ * n_char_out_, sizeof(float));
  }

  xp_ = (float*)calloc(n_char_in_, sizeof(float));

  n_var_ = kann_size_var(ann_);

  ua_ = kann_unroll(ann_, ulen_);
  kann_set_batch_size(ua_, mini_batch_size_);

  // Initialize optimizer
  optimizer_.allocate(n_var_, optimizer_.alpha1() != 0.0 || optimizer_.beta1t() != 0.0);

  kann_feed_bind(ua_, KANN_F_IN,    0, x_);
  kann_feed_bind(ua_, KANN_F_TRUTH, 0, y_);
  kann_switch(ua_, 1);

  kann_rnn_start(ann_);
}

void CompressionCodec::predict(unsigned *freq, unsigned *total) {
  unsigned i;
  float sum = 0.0;

  xp_[symb_list_[symb_list_ndx_]] = 1.0;
  const float *yp = kann_apply1(ann_, xp_);
  xp_[symb_list_[symb_list_ndx_]] = 0.0;

  for(i = 0; i < (unsigned)n_char_out_; i++) sum += yp[i];
  for(i = 0; i < (unsigned)n_char_out_; i++) {
    freq[i] = yp[i] * (SCALE - n_char_out_ - 1) / sum;
    freq[i] += freq[i] == 0;
  }
  *total = 1;
  for(i = 0; i < (unsigned)n_char_out_; i++) *total += freq[i];
  if(*total >= SCALE) exit(6);
}

void CompressionCodec::train() {
  int ul_x, ul_y = (symb_list_ndx_ + 1) % symb_list_size_;
  for(int ul = 0; ul < ulen_; ul++) {
    memset(x_[ul], 0, mini_batch_size_ * n_char_in_  * sizeof(float));
    memset(y_[ul], 0, mini_batch_size_ * n_char_out_ * sizeof(float));

    ul_x = ul_y;
    ul_y = (ul_y + 1) % symb_list_size_;
    for(int mbs = 0, mbs_x = ul_x, mbs_y = ul_y; mbs < mini_batch_size_; mbs++, mbs_x = (mbs_x + mini_batch_step_) % symb_list_size_, mbs_y = (mbs_y + mini_batch_step_) % symb_list_size_) {
      x_[ul][mbs * n_char_in_  + symb_list_[mbs_x]] = 1.0;
      y_[ul][mbs * n_char_out_ + symb_list_[mbs_y]] = 1.0;
    }
  }

  kann_cost(ua_, 0, 1);

  if(grad_clip_ > 0.0f) {
    kann_grad_clip(grad_clip_, n_var_, ua_->g);
  } else if(grad_clip_ < 0.0f) {
    for(int i = 0; i < n_var_; ++i)
      if(ua_->g[i] < grad_clip_)
        ua_->g[i] = grad_clip_;
      else if(ua_->g[i] > -grad_clip_)
        ua_->g[i] = -grad_clip_;
  }

  optimizer_.step(ua_->g, ua_->x);
}

void CompressionCodec::cleanup() {
  if (ann_) {
    kann_rnn_end(ann_);
    kann_switch(ua_, 0);
    kann_delete_unrolled(ua_);
    kann_delete(ann_);
    ann_ = nullptr;
    ua_ = nullptr;
  }

  optimizer_.cleanup();

  if (xp_) {
    free(xp_);
    xp_ = nullptr;
  }

  if (x_) {
    for(int k = 0; k < ulen_; k++) {
      if (y_) free(y_[k]);
      free(x_[k]);
    }
    free(x_);
    x_ = nullptr;
  }

  if (y_) {
    free(y_);
    y_ = nullptr;
  }
}

// FileSerializer class - Modern C++ encapsulation of file I/O operations
class FileSerializer {
public:
  // Write operations
  static int writeU8(FILE *file, byte value) {
    return fputc(value, file) == EOF;
  }

  static int writeU16(FILE *file, word value) {
    return writeU8(file, (value >> 0) & 0xff) || writeU8(file, (value >> 8) & 0xff);
  }

  static int writeU32(FILE *file, uint value) {
    return writeU16(file, (value >> 0) & 0xffff) || writeU16(file, (value >> 16) & 0xffff);
  }

  static int writeS32(FILE *file, long value) {
    return writeU16(file, (value >> 0) & 0xffff) || writeU16(file, (value >> 16) & 0xffff);
  }

  static int writeFloat(FILE *file, float value) {
    uint v;
    memcpy(&v, &value, sizeof(float));
    return writeU32(file, v);
  }

  // Read operations
  static int readU8(FILE *file, byte *value) {
    int v = fgetc(file);
    if(v == EOF) return 1;
    *value = v;
    return 0;
  }

  static int readU16(FILE *file, word *value) {
    byte v1, v2;
    if(readU8(file, &v1) || readU8(file, &v2)) return 1;
    *value = (word)v1 << 0 | (word)v2 << 8;
    return 0;
  }

  static int readU32(FILE *file, uint *value) {
    word v1, v2;
    if(readU16(file, &v1) || readU16(file, &v2)) return 1;
    *value = (uint)v1 << 0 | (uint)v2 << 16;
    return 0;
  }

  static int readS32(FILE *file, long *value) {
    word v1, v2;
    if(readU16(file, &v1) || readU16(file, &v2)) return 1;
    *value = (long)v1 << 0 | (long)v2 << 16;
    return 0;
  }

  static int readFloat(FILE *file, float *value) {
    uint v;
    if(readU32(file, &v)) return 1;
    memcpy(value, &v, sizeof(float));
    return 0;
  }
};

// Legacy functions for backward compatibility
int fput_ui08(FILE *file, byte value) {
  return FileSerializer::writeU8(file, value);
}

int fget_ui08(FILE *file, byte *value) {
  return FileSerializer::readU8(file, value);
}

int fput_ui16(FILE *file, word value) {
  return FileSerializer::writeU16(file, value);
}

int fget_ui16(FILE *file, word *value) {
  return FileSerializer::readU16(file, value);
}

int fput_ui32(FILE *file, uint value) {
  return FileSerializer::writeU32(file, value);
}

int fget_ui32(FILE *file, uint *value) {
  return FileSerializer::readU32(file, value);
}

int fput_si32(FILE *file, long value) {
  return FileSerializer::writeS32(file, value);
}

int fget_si32(FILE *file, long *value) {
  return FileSerializer::readS32(file, value);
}

int fput_si00(FILE *file, int value) {
  return fput_ui16(file, (value >> 0) & 0xffff) || fput_ui16(file, (value >> 16) & 0xffff);
}

int fget_si00(FILE *file, int *value) {
  word v1, v2;
  if(fget_ui16(file, &v1) || fget_ui16(file, &v2)) return 1;
  *value = (int)v1 << 0 | (int)v2 << 16;
  return 0;
}

int fput_fl(FILE *file, float value) {
  return fput_ui32(file, *((uint *) &value));
}

int fget_fl(FILE *file, float *value) {
  uint v;
  if(fget_ui32(file, &v)) return 1;
  *value = *((float *) &v);
  return 0;
}

char const *vers[] = { "0.1", "1.0", "2.0", "3.0", "4.0" };

// functions replacing PERC and BPB macros (moved before use)
float PERC(float V, float T) { return 100.0f * V / T; }
float BPB(float V, float T) { return 8.0f * V / T; }

void display_stats(stats_t *stats, qword comprpos) {
  if(stats->orig_current >= 1 && stats->orig_total >= 1 && comprpos != stats->rc_previous && stats->orig_current != stats->orig_previous)
    printf("%7.3f%% %7.3f%%/%6.3f %7.3f%%/%6.3f\r", PERC(stats->orig_current, stats->orig_total), PERC(comprpos, stats->orig_current), BPB(comprpos, stats->orig_current), PERC(comprpos - stats->rc_previous, stats->orig_current - stats->orig_previous), BPB(comprpos - stats->rc_previous, stats->orig_current - stats->orig_previous));
  stats->rc_previous   = comprpos;
  stats->orig_previous = stats->orig_current;
}

int main(int argc, char** argv) {
  kanncompr_t options;

  if(argc != 4 || (argv[1][0] != 'c' && argv[1][0] != 'd')) {
    printf("kanncompr %s - Mauro Vezzosi - %s\n", vers[VERSION_ID], VERSION_DATE);
    printf("https://encode.su/threads/4149-kanncompr\n");
    printf("Compression  : kanncompr c original_file compressed_file\n");
    printf("Decompression: kanncompr d compressed_file decompressed_file\n");
    exit(1);
  }

  options.ann                     = NULL;
  options.n_char_in               = 256;
  options.n_char_out              = 256;
  options.seed                    = 1;
  options.rnn_type                = 51;
  options.n_layers                = 3;
  options.n_neurons               = 256;
  options.ulen                    = 20;
  options.n_layers_embed_hidden   = 2;
  options.n_layers_embed_output   = 3;
  options.grad_clip               = 4.0f;
  options.dropout                 = 0.0;
  options.temper                  = 0.0;
  options.var_h0                  = 1;
  options.norm                    = 1;
  
  options.bias_init_type[0]       = 0;
  options.bias_init_from_to[0][0] = 0.0;
  options.bias_init_from_to[0][1] = 0.0;
  options.bias_init_from_to[0][2] = 0.0;
  options.bias_init_from_to[0][3] = 0.0;
  
  options.bias_init_type[1]       = 0;
  options.bias_init_from_to[1][0] = 1.0;
  options.bias_init_from_to[1][1] = 1.0;
  options.bias_init_from_to[1][2] = 1.0;
  options.bias_init_from_to[1][3] = 1.0;
  
  options.bias_init_type[2]       = 0;
  options.bias_init_from_to[2][0] = 0.0;
  options.bias_init_from_to[2][1] = 0.0;
  options.bias_init_from_to[2][2] = 0.0;
  options.bias_init_from_to[2][3] = 0.0;
  
  options.bias_init_type[3]       = 0;
  options.bias_init_from_to[3][0] = 0.0;
  options.bias_init_from_to[3][1] = 0.0;
  options.bias_init_from_to[3][2] = 0.0;
  options.bias_init_from_to[3][3] = 0.0;
  options.alpha1                  = 0.003;
  options.alpha2                  = 0.001;
  options.alpha1d                 = 0.9999;
  options.beta1                   = 0.0;
  options.beta1t                  = 0.0;
  options.beta2                   = 0.9999;
  options.beta2t                  = 0.9999;
  options.eps                     = 1e-10;
  options.ua                      = NULL;
  options.n_var                   = 0;
  options.x                       = NULL;
  options.y                       = NULL;
  options.xp                      = NULL;
  options.m                       = NULL;
  options.v                       = NULL;
  options.mini_batch_freq         = 20;
  options.mini_batch_freq_cnt     = 0;
  options.mini_batch_size         = 2;
  options.mini_batch_step         = 8;
  options.symb_list               = NULL;
  options.symb_list_ndx           = 0;
  options.symb_list_size          = options.ulen + options.mini_batch_step * (options.mini_batch_size - 1) + 1;
  options.vocab_type              = 0;

  FILE *filein = fopen(argv[2], "rb");  if( filein==NULL ) exit(2);
  FILE *fileout = fopen(argv[3], "wb"); if( fileout==NULL) exit(3);

  long fileoriglen = 0;

  if(argv[1][0] == 'c') {
    fseek(filein, 0, SEEK_END);
    fileoriglen = ftell(filein);
    fseek(filein, 0, SEEK_SET);

  } else if(argv[1][0] == 'd') {
  }

  kann_srand(options.seed);
  kad_trap_fe();

  options.ann = ann_structure(options);

  ann_init(&options);

  unsigned int i, code, low, total = 0, *freq;
  int c;
  freq = (unsigned int*)calloc(1 + options.n_char_out, sizeof(unsigned int));
  for(i = 0; i <= (unsigned int)options.n_char_out; i++) total += freq[i] = 1;
  options.symb_list_size = options.ulen + options.mini_batch_step * (options.mini_batch_size - 1) + 1;
  options.symb_list = (byte*)calloc(options.symb_list_size, sizeof(byte));

  uint fileposstep = 16 * 1024;
  
  stats_t stats;

  stats.orig_current      = 0;
  stats.orig_total        = 0;
  stats.next_display      = fileposstep;
  stats.next_display_step = fileposstep;
  stats.rc_previous       = 0;
  stats.orig_previous     = 0;
  printf("Original Compr. %%    BPS Block  %%    BPS\n");

  if(argv[1][0] == 'c') {
    rc.StartEncode(fileout);

    stats.orig_total = fileoriglen;
    display_stats(&stats, rc.Counter());

    for(long fileorigpos = 0; fileorigpos < fileoriglen; fileorigpos++) {
      ann_predict(&options, freq, &total);
      options.symb_list_ndx = (options.symb_list_ndx + 1) % options.symb_list_size;
      c = fgetc(filein);
      options.symb_list[options.symb_list_ndx] = c;
      stats.orig_current++;
      for(i = 0, low = 0; i < (unsigned)c; i++) low += freq[i];
      rc.rc_Process(low, freq[c], total);

      if(++options.mini_batch_freq_cnt == options.mini_batch_freq) {
        ann_train(&options);
        options.mini_batch_freq_cnt = 0;
      }

      if(stats.orig_current >= stats.next_display) {
        display_stats(&stats, rc.Counter());
        stats.next_display += stats.next_display_step;
      }
    }

    ann_predict(&options, freq, &total);
    c = options.n_char_out;
    for(i = 0, low = 0; i < (unsigned)c; i++) low += freq[i];
    rc.rc_Process(low, freq[c], total);

    rc.FinishEncode();

    display_stats(&stats, rc.Counter());
    printf("\n");
  } else if(argv[1][0] == 'd') {
    rc.StartDecode(filein);

    stats.orig_total = fileoriglen;
    display_stats(&stats, rc.Counter());

    c = options.n_char_out; 

    while(1) {
      ann_predict(&options, freq, &total);
      code = rc.rc_GetFreq(total);
      for(c = 0, low = 0; low + freq[c] <= code; c++) low += freq[c];
      rc.rc_Process(low, freq[c], total);

      if(c == options.n_char_out) break;

      stats.orig_current++;
      fputc(c, fileout);
      options.symb_list_ndx = (options.symb_list_ndx + 1) % options.symb_list_size;
      options.symb_list[options.symb_list_ndx] = c;

      if(++options.mini_batch_freq_cnt == options.mini_batch_freq) {
        ann_train(&options);
        options.mini_batch_freq_cnt = 0;
      }

      if(stats.orig_current >= stats.next_display) {
        display_stats(&stats, rc.Counter());
        stats.next_display += stats.next_display_step;
      }
    }

    display_stats(&stats, rc.Counter());
    printf("\n");
  }

  free(options.symb_list);
  free(freq);

  ann_end(options);

  fclose(filein);
  fclose(fileout);

  return 0;
}
