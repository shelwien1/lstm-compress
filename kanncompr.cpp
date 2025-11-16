
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

//#include "trace.inc"

typedef unsigned short word;
typedef unsigned int   uint;
typedef unsigned char  byte;
typedef unsigned long long qword;
typedef signed long long sqword;

#ifndef KANN_H
#define KANN_H

#define KANN_VERSION "r549"

#define KANN_F_IN       0x1   
#define KANN_F_OUT      0x2   
#define KANN_F_TRUTH    0x4   
#define KANN_F_COST     0x8   

#define KANN_C_CEB      1   
#define KANN_C_CEM      2   
#define KANN_C_CEB_NEG  3   
#define KANN_C_MSE      4   

#define KANN_RNN_VAR_H0 0x1 
#define KANN_RNN_NORM   0x2 

#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#define KAD_VERSION "r544"


#ifdef __STRICT_ANSI__
#define inline
#endif

#define KAD_MAX_DIM 4     
#define KAD_MAX_OP  64    

#define KAD_VAR        0x1
#define KAD_CONST      0x2
#define KAD_POOL       0x4
#define KAD_SHARE_RNG  0x10 

#define kad_is_back(p)  ((p)->flag & KAD_VAR)
#define kad_is_ext(p)   ((p)->n_child == 0)
#define kad_is_var(p)   (kad_is_ext(p) && kad_is_back(p))
#define kad_is_const(p) (kad_is_ext(p) && ((p)->flag & KAD_CONST))
#define kad_is_feed(p)  (kad_is_ext(p) && !kad_is_back(p) && !((p)->flag & KAD_CONST))
#define kad_is_pivot(p) ((p)->n_child == 1 && ((p)->flag & KAD_POOL))
#define kad_is_switch(p) ((p)->op == 12 && !((p)->flag & KAD_POOL))
#define kad_use_rng(p)  ((p)->op == 15 || (p)->op == 24)

#define kad_eval_enable(p) ((p)->tmp = 1)
#define kad_eval_disable(p) ((p)->tmp = -1)

typedef struct kad_node_t {
  uint8_t     n_d;            
  uint8_t     flag;           
  uint16_t    op;             
  int32_t     n_child;        
  int32_t     tmp;            
  int32_t     ptr_size;       
  int32_t     d[KAD_MAX_DIM]; 
  int32_t     ext_label;      
  uint32_t    ext_flag;       
  float      *x;              
  float      *g;              
  void       *ptr;            
  void       *gtmp;           
  struct kad_node_t **child;  
  struct kad_node_t  *pre;    
} kad_node_t, *kad_node_p;

#ifdef __cplusplus
extern "C" {
#endif

kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots);

kad_node_t **kad_compile(int *n_node, int n_roots, ...); 
void kad_delete(int n, kad_node_t **a); 

const float *kad_eval_at(int n, kad_node_t **a, int from);

void kad_eval_marked(int n, kad_node_t **a);
int kad_sync_dim(int n, kad_node_t **v, int batch_size);

void kad_grad(int n, kad_node_t **a, int from);

kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len);
int kad_n_pivots(int n_v, kad_node_t **v);

kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size);

kad_node_t *kad_var(float *x, float *g, int n_d, ...); 
kad_node_t *kad_const(float *x, int n_d, ...);         
kad_node_t *kad_feed(int n_d, ...);                    

kad_node_t *kad_add(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_sub(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y); 

kad_node_t *kad_matmul(kad_node_t *x, kad_node_t *y);     
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);       

kad_node_t *kad_mse(kad_node_t *x, kad_node_t *y);        
kad_node_t *kad_ce_multi(kad_node_t *x, kad_node_t *y);   
kad_node_t *kad_ce_bin(kad_node_t *x, kad_node_t *y);     
kad_node_t *kad_ce_bin_neg(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight);

#define KAD_PAD_NONE  0      
#define KAD_PAD_SAME  (-2)   

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int r_stride, int c_stride, int r_pad, int c_pad);             
kad_node_t *kad_max2d(kad_node_t *x, int kernel_h, int kernel_w, int r_stride, int c_stride, int r_pad, int c_pad); 
kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int pad);  
kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int pad); 
kad_node_t *kad_avg1d(kad_node_t *x, int kernel_size, int stride, int pad); 

kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *r);                      
kad_node_t *kad_sample_normal(kad_node_t *x);                               

kad_node_t *kad_square(kad_node_t *x); 
kad_node_t *kad_sigm(kad_node_t *x);   
kad_node_t *kad_tanh(kad_node_t *x);   
kad_node_t *kad_relu(kad_node_t *x);   
kad_node_t *kad_softmax(kad_node_t *x);
kad_node_t *kad_1minus(kad_node_t *x); 
kad_node_t *kad_exp(kad_node_t *x);    
kad_node_t *kad_log(kad_node_t *x);    
kad_node_t *kad_sin(kad_node_t *x);    

kad_node_t *kad_stdnorm(kad_node_t *x); 

kad_node_t *kad_avg(int n, kad_node_t **x);   
kad_node_t *kad_max(int n, kad_node_t **x);   
kad_node_t *kad_stack(int n, kad_node_t **x); 
kad_node_t *kad_select(int n, kad_node_t **x, int which); 

kad_node_t *kad_reduce_sum(kad_node_t *x, int axis);  
kad_node_t *kad_reduce_mean(kad_node_t *x, int axis); 

kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end); 
kad_node_t *kad_concat(int axis, int n, ...);                       
kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p);      
kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d);            
kad_node_t *kad_reverse(kad_node_t *x, int axis);
kad_node_t *kad_switch(int n, kad_node_t **p);                      

int kad_size_var(int n, kad_node_t *const* v);   
int kad_size_const(int n, kad_node_t *const* v); 

int kad_save(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_load(FILE *fp, int *_n_node);

void *kad_rng(void);
void kad_srand(void *d, uint64_t seed);
uint64_t kad_rand(void *d);
double kad_drand(void *d);
double kad_drand_normal(void *d);
void kad_saxpy(int n, float a, const float *x, float *y);

void kad_trap_fe(void); 
void kad_print_graph(FILE *fp, int n, kad_node_t **v);
void kad_check_grad(int n, kad_node_t **a, int from);

#ifdef __cplusplus
}
#endif

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_DIM   4

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[KAD_MAX_OP];
extern char *kad_op_name[KAD_MAX_OP];

int kad_len(const kad_node_t *p) 
{
  int n = 1, i;
  for (i = 0; i < p->n_d; ++i) n *= p->d[i];
  return n;
}

#endif

typedef struct {
  int n;            
  kad_node_t **v;   
  float *x, *g, *c; 
  void *mt;         
} kann_t;

extern int kann_verbose;

#define kann_size_var(a) kad_size_var((a)->n, (a)->v)
#define kann_size_const(a) kad_size_const((a)->n, (a)->v)
#define kann_dim_in(a) kann_feed_dim((a), KANN_F_IN, 0)
#define kann_dim_out(a) kann_feed_dim((a), KANN_F_TRUTH, 0)
#define kann_srand(seed) kad_srand(0, (seed))
#define kann_drand() kad_drand(0)
#define kann_set_batch_size(ann, B) kad_sync_dim((ann)->n, (ann)->v, (B))

#ifdef __cplusplus
extern "C" {
#endif

kann_t *kann_new(kad_node_t *cost, int n_rest, ...);

kann_t *kann_unroll(kann_t *a, ...);

kann_t *kann_unroll_array(kann_t *a, int *len);
kann_t *kann_clone(kann_t *a, int batch_size);
void kann_delete(kann_t *a);          
void kann_delete_unrolled(kann_t *a); 

void kann_mt(kann_t *ann, int n_threads, int max_batch_size);

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x);

float kann_cost(kann_t *a, int cost_label, int cal_grad);

int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label);
int kann_eval_out(kann_t *a);
int kann_class_error(const kann_t *ann, int *base);

int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label);

int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label);

void kann_rnn_start(kann_t *a);

void kann_rnn_end(kann_t *a);

void kann_switch(kann_t *a, int is_train);

void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r);

void kann_shuffle(int n, int *s);
float kann_grad_clip(float thres, int n, float *g);

kad_node_t *kann_layer_input(int n1);
kad_node_t *kann_layer_dense(kad_node_t *in, int n1);
kad_node_t *kann_layer_dropout(kad_node_t *t, float r);
kad_node_t *kann_layer_layernorm(kad_node_t *in);
kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, int rnn_flag);
kad_node_t *kann_layer_lstm(kad_node_t *in, int n1, int rnn_flag);
kad_node_t *kann_layer_gru(kad_node_t *in, int n1, int rnn_flag);
kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c);
kad_node_t *kann_layer_conv1d(kad_node_t *in, int n_flt, int k_size, int stride, int pad);
kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type);

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...); 
kad_node_t *kann_new_scalar(uint8_t flag, float x);
kad_node_t *kann_new_weight(int n_row, int n_col);
kad_node_t *kann_new_bias(int n);
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col);
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len);

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...);
kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1);
kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r);
kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in);
kad_node_t *kann_layer_rnn2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag);
kad_node_t *kann_layer_gru2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag);

int kann_train_fnn1(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y);
float kann_cost_fnn1(kann_t *a, int n, float **x, float **y);
const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label);
const float *kann_apply1(kann_t *a, float *x);

void kann_save_fp(FILE *fp, kann_t *ann);
void kann_save(const char *fn, kann_t *ann);
kann_t *kann_load_fp(FILE *fp);
kann_t *kann_load(const char *fn);

#ifdef __cplusplus
}
#endif

#endif

int kann_verbose = 3;

void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c)
{
  int i, j, k, l, n_var;
  float *x, *g, *c;
  n_var = kad_size_var(n, a);
  x = *_x = (float*)realloc(*_x, n_var * sizeof(float));
  g = *_g = (float*)realloc(*_g, n_var * sizeof(float));
  c = *_c = (float*)realloc(*_c, kad_size_const(n, a) * sizeof(float));
  memset(g, 0, n_var * sizeof(float));
  for (i = j = k = 0; i < n; ++i) {
    kad_node_t *v = a[i];
    if (kad_is_var(v)) {
      l = kad_len(v);
      memcpy(&x[j], v->x, l * sizeof(float));
      free(v->x);
      v->x = &x[j];
      v->g = &g[j];
      j += l;
    } else if (kad_is_const(v)) {
      l = kad_len(v);
      memcpy(&c[k], v->x, l * sizeof(float));
      free(v->x);
      v->x = &c[k];
      k += l;
    }
  }
}

void kad_ext_sync(int n, kad_node_t **a, float *x, float *g, float *c)
{
  int i, j, k;
  for (i = j = k = 0; i < n; ++i) {
    kad_node_t *v = a[i];
    if (kad_is_var(v)) {
      v->x = &x[j];
      v->g = &g[j];
      j += kad_len(v);
    } else if (kad_is_const(v)) {
      v->x = &c[k];
      k += kad_len(v);
    }
  }
}

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
    if (kad_is_pivot(a->v[i])) has_pivot = 1;
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

kann_t *kann_clone(kann_t *a, int batch_size)
{
  kann_t *b;
  b = (kann_t*)calloc(1, sizeof(kann_t));
  b->n = a->n;
  b->v = kad_clone(a->n, a->v, batch_size);
  kad_ext_collate(b->n, b->v, &b->x, &b->g, &b->c);
  return b;
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
  if (a && a->mt) kann_mt(a, 0, 0);
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

#define chk_flg(flag, mask) ((mask) == 0 || ((flag) & (mask)))
#define chk_lbl(label, query) ((query) == 0 || (label) == (query))

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
    if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
      a->v[i]->x = x[k++];
  return k;
}

int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
  int i, k, n = 0;
  for (i = k = 0; i < a->n; ++i)
    if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
      ++k, n = a->v[i]->n_d > 1? kad_len(a->v[i]) / a->v[i]->d[0] : a->v[i]->n_d == 1? a->v[i]->d[0] : 1;
  return k == 1? n : k == 0? -1 : -2;
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

int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label)
{
  int i, k;
  for (i = k = 0; i < a->n; ++i)
    if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
      ++k, a->v[i]->tmp = 1;
  kad_eval_marked(a->n, a->v);
  return k;
}

void kann_rnn_start(kann_t *a)
{
  int i;
  kann_set_batch_size(a, 1);
  for (i = 0; i < a->n; ++i) {
    kad_node_t *p = a->v[i];
    if (p->pre) { 
      kad_node_t *q = p->pre;
      if (q->x) memcpy(p->x, q->x, kad_len(p) * sizeof(float));
      else memset(p->x, 0, kad_len(p) * sizeof(float));
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
      a->v[i]->pre->x = (float*)calloc(kad_len(a->v[i]->pre), sizeof(float));
}

int kann_class_error_core(const kann_t *ann, int *base)
{
  int i, j, k, m, n, off, n_err = 0;
  for (i = 0, *base = 0; i < ann->n; ++i) {
    kad_node_t *p = ann->v[i];
    if (((p->op == 13 && (p->n_child == 2 || p->n_child == 3)) || (p->op == 22 && p->n_child == 2)) && p->n_d == 0) { 
      kad_node_t *x = p->child[0], *t = p->child[1];
      n = t->d[t->n_d - 1], m = kad_len(t) / n;
      for (j = off = 0; j < m; ++j, off += n) {
        float t_sum = 0.0f, t_min = 1.0f, t_max = 0.0f, x_max = 0.0f, x_min = 1.0f;
        int x_max_k = -1, t_max_k = -1;
        for (k = 0; k < n; ++k) {
          float xk = x->x[off+k], tk = t->x[off+k];
          t_sum += tk;
          t_min = t_min < tk? t_min : tk;
          x_min = x_min < xk? x_min : xk;
          if (t_max < tk) t_max = tk, t_max_k = k;
          if (x_max < xk) x_max = xk, x_max_k = k;
        }
        if (t_sum - 1.0f == 0 && t_min >= 0.0f && x_min >= 0.0f && x_max <= 1.0f) {
          ++(*base);
          n_err += (x_max_k != t_max_k);
        }
      }
    }
  }
  return n_err;
}

void kann_mt(kann_t *ann, int n_threads, int max_batch_size) {}
float kann_cost(kann_t *a, int cost_label, int cal_grad) { return kann_cost_core(a, cost_label, cal_grad); }
int kann_eval_out(kann_t *a) { return kann_eval(a, KANN_F_OUT, 0); }
int kann_class_error(const kann_t *a, int *base) { return kann_class_error_core(a, base); }
void kann_switch(kann_t *ann, int is_train) { return kann_switch_core(ann, is_train); }

#define KANN_MAGIC "KAN\1"

void kann_save_fp(FILE *fp, kann_t *ann)
{
  kann_set_batch_size(ann, 1);
  fwrite(KANN_MAGIC, 1, 4, fp);
  kad_save(fp, ann->n, ann->v);
  fwrite(ann->x, sizeof(float), kann_size_var(ann), fp);
  fwrite(ann->c, sizeof(float), kann_size_const(ann), fp);
}

void kann_save(const char *fn, kann_t *ann)
{
  FILE *fp;
  fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
  kann_save_fp(fp, ann);
  fclose(fp);
}

kann_t *kann_load_fp(FILE *fp)
{
  char magic[4];
  kann_t *ann;
  int n_var, n_const;

  fread(magic, 1, 4, fp);
  if (strncmp(magic, KANN_MAGIC, 4) != 0) {
    fclose(fp);
    return 0;
  }
  ann = (kann_t*)calloc(1, sizeof(kann_t));
  ann->v = kad_load(fp, &ann->n);
  n_var = kad_size_var(ann->n, ann->v);
  n_const = kad_size_const(ann->n, ann->v);
  ann->x = (float*)malloc(n_var * sizeof(float));
  ann->g = (float*)calloc(n_var, sizeof(float));
  ann->c = (float*)malloc(n_const * sizeof(float));
  fread(ann->x, sizeof(float), n_var, fp);
  fread(ann->c, sizeof(float), n_const, fp);
  kad_ext_sync(ann->n, ann->v, ann->x, ann->g, ann->c);
  return ann;
}

kann_t *kann_load(const char *fn)
{
  FILE *fp;
  kann_t *ann;
  fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
  ann = kann_load_fp(fp);
  fclose(fp);
  return ann;
}

kad_node_t *kann_new_leaf_array(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, int32_t d[KAD_MAX_DIM])
{
  int i, len, off = offset && par? *offset : -1;
  kad_node_t *p;

  if (off >= 0 && par[off]) return par[(*offset)++];
  p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
  p->n_d = n_d, p->flag = flag;
  memcpy(p->d, d, n_d * sizeof(int32_t));
  len = kad_len(p);
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
  n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
  w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
  b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
  return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r)
{
  kad_node_t *x[2], *cr;
  cr = kann_new_leaf2(offset, par, KAD_CONST, r, 0);
  x[0] = t, x[1] = kad_dropout(t, cr);
  return kad_switch(2, x);
}

kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in)
{
  int n0;
  kad_node_t *alpha, *beta;
  n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
  alpha = kann_new_leaf2(offset, par, KAD_VAR, 1.0f, 1, n0);
  beta  = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n0);
  return kad_add(kad_mul(kad_stdnorm(in), alpha), beta);
}

kad_node_t *cmul_norm2(int *offset, kad_node_t **par, kad_node_t *x, kad_node_t *w, int use_norm)
{
  return use_norm? kann_layer_layernorm2(offset, par, kad_cmul(x, w)) : kad_cmul(x, w);
}

kad_node_t *kann_layer_rnn2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag)
{
  int n0, n1 = h0->d[h0->n_d-1], use_norm = !!(rnn_flag & KANN_RNN_NORM);
  kad_node_t *t, *w, *u, *b, *out;

  u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
  b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
  t = cmul_norm2(offset, par, h0, u, use_norm);
  if (in) {
    n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
    t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
  }
  out = kad_tanh(kad_add(t, b));
  out->pre = h0;
  return out;
}

kad_node_t *kann_layer_gru2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag)
{
  int n0 = 0, n1 = h0->d[h0->n_d-1], use_norm = !!(rnn_flag & KANN_RNN_NORM);
  kad_node_t *t, *r, *z, *w, *u, *b, *s, *out;

  if (in) n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
  
  u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
  b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
  t = cmul_norm2(offset, par, h0, u, use_norm);
  if (in) {
    w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
    t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
  }
  z = kad_sigm(kad_add(t, b));
  
  u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
  b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
  t = cmul_norm2(offset, par, h0, u, use_norm);
  if (in) {
    w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
    t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
  }
  r = kad_sigm(kad_add(t, b));
  
  u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
  b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
  t = cmul_norm2(offset, par, kad_mul(r, h0), u, use_norm);
  if (in) {
    w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
    t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
  }
  s = kad_tanh(kad_add(t, b));
  
  out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
  out->pre = h0;
  return out;
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
kad_node_t *kann_layer_dropout(kad_node_t *t, float r) { return kann_layer_dropout2(0, 0, t, r); }
kad_node_t *kann_layer_layernorm(kad_node_t *in) { return kann_layer_layernorm2(0, 0, in); }

kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, int rnn_flag)
{
  kad_node_t *h0;
  h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  h0->x = (float*)calloc(n1, sizeof(float));
  return kann_layer_rnn2(0, 0, in, h0, rnn_flag);
}

kad_node_t *kann_layer_gru(kad_node_t *in, int n1, int rnn_flag)
{
  kad_node_t *h0;
  h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  h0->x = (float*)calloc(n1, sizeof(float));
  return kann_layer_gru2(0, 0, in, h0, rnn_flag);
}

kad_node_t *kann_cmul_norm(kad_node_t *x, kad_node_t *w)
{
  return kann_layer_layernorm(kad_cmul(x, w));
}

kad_node_t *kann_layer_lstm(kad_node_t *in, int n1, int rnn_flag)
{
  int n0;
  kad_node_t *i, *f, *o, *g, *w, *u, *b, *h0, *c0, *c, *out;
  kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANN_RNN_NORM)? kann_cmul_norm : kad_cmul;

  n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
  h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  h0->x = (float*)calloc(n1, sizeof(float));
  c0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  c0->x = (float*)calloc(n1, sizeof(float));

  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  b = kann_new_bias(n1);
  i = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  
  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  b = kann_new_vec(n1, 1.0f); 
  f = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  
  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  b = kann_new_bias(n1);
  o = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  
  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  b = kann_new_bias(n1);
  g = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  
  c = kad_add(kad_mul(f, c0), kad_mul(g, i)); 
  c->pre = c0;
  
  if (rnn_flag & KANN_RNN_NORM) c = kann_layer_layernorm(c); 
  out = kad_mul(kad_tanh(c), o);
  out->pre = h0;
  return out;
}

kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c)
{
  kad_node_t *w;
  w = kann_new_weight_conv2d(n_flt, in->d[1], k_rows, k_cols);
  return kad_conv2d(in, w, stride_r, stride_c, pad_r, pad_c);
}

kad_node_t *kann_layer_conv1d(kad_node_t *in, int n_flt, int k_size, int stride, int pad)
{
  kad_node_t *w;
  w = kann_new_weight_conv1d(n_flt, in->d[1], k_size);
  return kad_conv1d(in, w, stride, pad);
}

kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type)
{
  kad_node_t *cost = 0, *truth = 0;
  assert(cost_type == KANN_C_CEB || cost_type == KANN_C_CEM || cost_type == KANN_C_CEB_NEG || cost_type == KANN_C_MSE);
  t = kann_layer_dense(t, n_out);
  truth = kad_feed(2, 1, n_out), truth->ext_flag |= KANN_F_TRUTH;
  if (cost_type == KANN_C_MSE) {
    cost = kad_mse(t, truth);
  } else if (cost_type == KANN_C_CEB) {
    t = kad_sigm(t);
    cost = kad_ce_bin(t, truth);
  } else if (cost_type == KANN_C_CEB_NEG) {
    t = kad_tanh(t);
    cost = kad_ce_bin_neg(t, truth);
  } else if (cost_type == KANN_C_CEM) {
    t = kad_softmax(t);
    cost = kad_ce_multi(t, truth);
  }
  t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;
  return cost;
}

void kann_shuffle(int n, int *s)
{
  int i, j, t;
  for (i = 0; i < n; ++i) s[i] = i;
  for (i = n; i > 0; --i) {
    j = (int)(i * kad_drand(0));
    t = s[j], s[j] = s[i-1], s[i-1] = t;
  }
}


void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
  int i, n4 = n>>2<<2;
  __m128 vh, vg, vr, vt, vd, vd1, tmp, vtiny;
  vh = _mm_set1_ps(h0);
  vd = _mm_set1_ps(decay);
  vd1 = _mm_set1_ps(1.0f - decay);
  vtiny = _mm_set1_ps(1e-6f);
  for (i = 0; i < n4; i += 4) {
    vt = _mm_loadu_ps(&t[i]);
    vr = _mm_loadu_ps(&r[i]);
    vg = _mm_loadu_ps(&g[i]);
    if (h) vh = _mm_loadu_ps(&h[i]);
    vr = _mm_add_ps(_mm_mul_ps(vd1, _mm_mul_ps(vg, vg)), _mm_mul_ps(vd, vr));
    _mm_storeu_ps(&r[i], vr);
    tmp = _mm_sub_ps(vt, _mm_mul_ps(_mm_mul_ps(vh, _mm_rsqrt_ps(_mm_add_ps(vtiny, vr))), vg));
    _mm_storeu_ps(&t[i], tmp);
  }
  for (; i < n; ++i) {
    r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
    t[i] -= (h? h[i] : h0) / sqrtf(1e-6f + r[i]) * g[i];
  }
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

int kann_train_fnn1(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y)
{
  int i, j, *shuf, n_train, n_val, n_in, n_out, n_var, n_const, drop_streak = 0, min_set = 0;
  float **x, **y, *x1, *y1, *r, min_val_cost = FLT_MAX, *min_x, *min_c;

  n_in = kann_dim_in(ann);
  n_out = kann_dim_out(ann);
  if (n_in < 0 || n_out < 0) return -1;
  n_var = kann_size_var(ann);
  n_const = kann_size_const(ann);
  r = (float*)calloc(n_var, sizeof(float));
  shuf = (int*)malloc(n * sizeof(int));
  x = (float**)malloc(n * sizeof(float*));
  y = (float**)malloc(n * sizeof(float*));
  kann_shuffle(n, shuf);
  for (j = 0; j < n; ++j)
    x[j] = _x[shuf[j]], y[j] = _y[shuf[j]];
  n_val = (int)(n * frac_val);
  n_train = n - n_val;
  min_x = (float*)malloc(n_var * sizeof(float));
  min_c = (float*)malloc(n_const * sizeof(float));

  x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
  y1 = (float*)malloc(n_out * mini_size * sizeof(float));
  kann_feed_bind(ann, KANN_F_IN,    0, &x1);
  kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);

  for (i = 0; i < max_epoch; ++i) {
    int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
    double train_cost = 0.0, val_cost = 0.0;
    kann_shuffle(n_train, shuf);
    kann_switch(ann, 1);
    while (n_proc < n_train) {
      int b, c, ms = n_train - n_proc < mini_size? n_train - n_proc : mini_size;
      for (b = 0; b < ms; ++b) {
        memcpy(&x1[b*n_in],  x[shuf[n_proc+b]], n_in  * sizeof(float));
        memcpy(&y1[b*n_out], y[shuf[n_proc+b]], n_out * sizeof(float));
      }
      kann_set_batch_size(ann, ms);
      train_cost += kann_cost(ann, 0, 1) * ms;
      c = kann_class_error(ann, &b);
      n_train_err += c, n_train_base += b;
      kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
      n_proc += ms;
    }
    train_cost /= n_train;
    kann_switch(ann, 0);
    n_proc = 0;
    while (n_proc < n_val) {
      int b, c, ms = n_val - n_proc < mini_size? n_val - n_proc : mini_size;
      for (b = 0; b < ms; ++b) {
        memcpy(&x1[b*n_in],  x[n_train+n_proc+b], n_in  * sizeof(float));
        memcpy(&y1[b*n_out], y[n_train+n_proc+b], n_out * sizeof(float));
      }
      kann_set_batch_size(ann, ms);
      val_cost += kann_cost(ann, 0, 0) * ms;
      c = kann_class_error(ann, &b);
      n_val_err += c, n_val_base += b;
      n_proc += ms;
    }
    if (n_val > 0) val_cost /= n_val;
    if (kann_verbose >= 3) {
      fprintf(stderr, "epoch: %d; training cost: %g", i+1, train_cost);
      if (n_train_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_train_err / n_train);
      if (n_val > 0) {
        fprintf(stderr, "; validation cost: %g", val_cost);
        if (n_val_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_val_err / n_val);
      }
      fputc('\n', stderr);
    }
    if (i >= max_drop_streak && n_val > 0) {
      if (val_cost < min_val_cost) {
        min_set = 1;
        memcpy(min_x, ann->x, n_var * sizeof(float));
        memcpy(min_c, ann->c, n_const * sizeof(float));
        drop_streak = 0;
        min_val_cost = (float)val_cost;
      } else if (++drop_streak >= max_drop_streak)
        break;
    }
  }
  if (min_set) {
    memcpy(ann->x, min_x, n_var * sizeof(float));
    memcpy(ann->c, min_c, n_const * sizeof(float));
  }

  free(min_c); free(min_x); free(y1); free(x1); free(y); free(x); free(shuf); free(r);
  return i;
}

float kann_cost_fnn1(kann_t *ann, int n, float **x, float **y)
{
  int n_in, n_out, n_proc = 0, mini_size = 64 < n? 64 : n;
  float *x1, *y1;
  double cost = 0.0;

  n_in = kann_dim_in(ann);
  n_out = kann_dim_out(ann);
  if (n <= 0 || n_in < 0 || n_out < 0) return 0.0;

  x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
  y1 = (float*)malloc(n_out * mini_size * sizeof(float));
  kann_feed_bind(ann, KANN_F_IN,    0, &x1);
  kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);
  kann_switch(ann, 0);
  while (n_proc < n) {
    int b, ms = n - n_proc < mini_size? n - n_proc : mini_size;
    for (b = 0; b < ms; ++b) {
      memcpy(&x1[b*n_in],  x[n_proc+b], n_in  * sizeof(float));
      memcpy(&y1[b*n_out], y[n_proc+b], n_out * sizeof(float));
    }
    kann_set_batch_size(ann, ms);
    cost += kann_cost(ann, 0, 0) * ms;
    n_proc += ms;
  }
  free(y1); free(x1);
  return (float)(cost / n);
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


#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#define KAD_VERSION "r544"


#ifdef __STRICT_ANSI__
#define inline
#endif

#define KAD_MAX_DIM 4     
#define KAD_MAX_OP  64    

#define KAD_VAR        0x1
#define KAD_CONST      0x2
#define KAD_POOL       0x4
#define KAD_SHARE_RNG  0x10 

#define kad_is_back(p)  ((p)->flag & KAD_VAR)
#define kad_is_ext(p)   ((p)->n_child == 0)
#define kad_is_var(p)   (kad_is_ext(p) && kad_is_back(p))
#define kad_is_const(p) (kad_is_ext(p) && ((p)->flag & KAD_CONST))
#define kad_is_feed(p)  (kad_is_ext(p) && !kad_is_back(p) && !((p)->flag & KAD_CONST))
#define kad_is_pivot(p) ((p)->n_child == 1 && ((p)->flag & KAD_POOL))
#define kad_is_switch(p) ((p)->op == 12 && !((p)->flag & KAD_POOL))
#define kad_use_rng(p)  ((p)->op == 15 || (p)->op == 24)

#define kad_eval_enable(p) ((p)->tmp = 1)
#define kad_eval_disable(p) ((p)->tmp = -1)

typedef struct kad_node_t {
  uint8_t     n_d;            
  uint8_t     flag;           
  uint16_t    op;             
  int32_t     n_child;        
  int32_t     tmp;            
  int32_t     ptr_size;       
  int32_t     d[KAD_MAX_DIM]; 
  int32_t     ext_label;      
  uint32_t    ext_flag;       
  float      *x;              
  float      *g;              
  void       *ptr;            
  void       *gtmp;           
  struct kad_node_t **child;  
  struct kad_node_t  *pre;    
} kad_node_t, *kad_node_p;

#ifdef __cplusplus
extern "C" {
#endif

kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots);

kad_node_t **kad_compile(int *n_node, int n_roots, ...); 
void kad_delete(int n, kad_node_t **a); 

const float *kad_eval_at(int n, kad_node_t **a, int from);

void kad_eval_marked(int n, kad_node_t **a);
int kad_sync_dim(int n, kad_node_t **v, int batch_size);

void kad_grad(int n, kad_node_t **a, int from);

kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len);
int kad_n_pivots(int n_v, kad_node_t **v);

kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size);

kad_node_t *kad_var(float *x, float *g, int n_d, ...); 
kad_node_t *kad_const(float *x, int n_d, ...);         
kad_node_t *kad_feed(int n_d, ...);                    

kad_node_t *kad_add(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_sub(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y); 

kad_node_t *kad_matmul(kad_node_t *x, kad_node_t *y);     
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);       

kad_node_t *kad_mse(kad_node_t *x, kad_node_t *y);        
kad_node_t *kad_ce_multi(kad_node_t *x, kad_node_t *y);   
kad_node_t *kad_ce_bin(kad_node_t *x, kad_node_t *y);     
kad_node_t *kad_ce_bin_neg(kad_node_t *x, kad_node_t *y); 
kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight);

#define KAD_PAD_NONE  0      
#define KAD_PAD_SAME  (-2)   

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int r_stride, int c_stride, int r_pad, int c_pad);             
kad_node_t *kad_max2d(kad_node_t *x, int kernel_h, int kernel_w, int r_stride, int c_stride, int r_pad, int c_pad); 
kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int pad);  
kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int pad); 
kad_node_t *kad_avg1d(kad_node_t *x, int kernel_size, int stride, int pad); 

kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *r);                      
kad_node_t *kad_sample_normal(kad_node_t *x);                               

kad_node_t *kad_square(kad_node_t *x); 
kad_node_t *kad_sigm(kad_node_t *x);   
kad_node_t *kad_tanh(kad_node_t *x);   
kad_node_t *kad_relu(kad_node_t *x);   
kad_node_t *kad_softmax(kad_node_t *x);
kad_node_t *kad_1minus(kad_node_t *x); 
kad_node_t *kad_exp(kad_node_t *x);    
kad_node_t *kad_log(kad_node_t *x);    
kad_node_t *kad_sin(kad_node_t *x);    

kad_node_t *kad_stdnorm(kad_node_t *x); 

kad_node_t *kad_avg(int n, kad_node_t **x);   
kad_node_t *kad_max(int n, kad_node_t **x);   
kad_node_t *kad_stack(int n, kad_node_t **x); 
kad_node_t *kad_select(int n, kad_node_t **x, int which); 

kad_node_t *kad_reduce_sum(kad_node_t *x, int axis);  
kad_node_t *kad_reduce_mean(kad_node_t *x, int axis); 

kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end); 
kad_node_t *kad_concat(int axis, int n, ...);                       
kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p);      
kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d);            
kad_node_t *kad_reverse(kad_node_t *x, int axis);
kad_node_t *kad_switch(int n, kad_node_t **p);                      

int kad_size_var(int n, kad_node_t *const* v);   
int kad_size_const(int n, kad_node_t *const* v); 

int kad_save(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_load(FILE *fp, int *_n_node);

void *kad_rng(void);
void kad_srand(void *d, uint64_t seed);
uint64_t kad_rand(void *d);
double kad_drand(void *d);
double kad_drand_normal(void *d);
void kad_saxpy(int n, float a, const float *x, float *y);

void kad_trap_fe(void); 
void kad_print_graph(FILE *fp, int n, kad_node_t **v);
void kad_check_grad(int n, kad_node_t **a, int from);

#ifdef __cplusplus
}
#endif

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_DIM   4

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[KAD_MAX_OP];
extern char *kad_op_name[KAD_MAX_OP];

int kad_len(const kad_node_t *p) 
{
  int n = 1, i;
  for (i = 0; i < p->n_d; ++i) n *= p->d[i];
  return n;
}

#endif

typedef struct {
  uint64_t s[2];
  double n_gset;
  int n_iset;
  volatile int lock;
} kad_rng_t;

kad_node_t *kad_new_core(int n_d, int op, int n_child)
{
  kad_node_t *s;
  if (n_d >= KAD_MAX_DIM) return 0;
  s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
  s->n_d = n_d, s->op = op, s->n_child = n_child;
  if (s->n_child) s->child = (kad_node_t**)calloc(s->n_child, sizeof(kad_node_t*));
  return s;
}

kad_node_t *kad_vleaf(uint8_t flag, float *x, float *g, int n_d, va_list ap)
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

kad_node_t *kad_const(float *x, int n_d, ...)
{
  kad_node_t *p;
  va_list ap;
  va_start(ap, n_d); p = kad_vleaf(KAD_CONST, x, 0, n_d, ap); va_end(ap);
  return p;
}

kad_node_t *kad_feed(int n_d, ...)
{
  kad_node_t *p;
  va_list ap;
  va_start(ap, n_d); p = kad_vleaf(0, 0, 0, n_d, ap); va_end(ap);
  return p;
}

kad_node_t *kad_var(float *x, float *g, int n_d, ...)
{
  kad_node_t *p;
  va_list ap;
  va_start(ap, n_d); p = kad_vleaf(KAD_VAR, x, g, n_d, ap); va_end(ap);
  return p;
}

kad_node_t *kad_finalize_node(kad_node_t *s) 
{
  int i;
  if (kad_op_list[s->op](s, KAD_SYNC_DIM) < 0) { 
    if (s->ptr) free(s->ptr);
    free(s->child); free(s);
    return 0;
  }
  for (i = 0; i < s->n_child; ++i)
    if (kad_is_back(s->child[i]))
      break;
  if (i < s->n_child) s->flag |= KAD_VAR;
  return s;
}

kad_node_t *kad_op2_core(int op, kad_node_t *x, kad_node_t *y)
{
  kad_node_t *s;
  s = kad_new_core(0, op, 2);
  s->child[0] = x, s->child[1] = y;
  return kad_finalize_node(s);
}

kad_node_t *kad_op1_core(int op, kad_node_t *x)
{
  kad_node_t *s;
  s = kad_new_core(0, op, 1);
  s->child[0] = x;
  return kad_finalize_node(s);
}

#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
KAD_FUNC_OP2(kad_sub, 23)
KAD_FUNC_OP2(kad_mul, 2)
KAD_FUNC_OP2(kad_cmul, 3)
KAD_FUNC_OP2(kad_matmul, 9)
KAD_FUNC_OP2(kad_ce_multi, 13)
KAD_FUNC_OP2(kad_ce_bin, 22)
KAD_FUNC_OP2(kad_ce_bin_neg, 4)
KAD_FUNC_OP2(kad_mse, 29)

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }

KAD_FUNC_OP1(kad_log, 27)
KAD_FUNC_OP1(kad_exp, 33)
KAD_FUNC_OP1(kad_sin, 34)
KAD_FUNC_OP1(kad_square, 5)
KAD_FUNC_OP1(kad_sigm, 6)
KAD_FUNC_OP1(kad_tanh, 7)
KAD_FUNC_OP1(kad_relu, 8)
KAD_FUNC_OP1(kad_1minus, 11)
KAD_FUNC_OP1(kad_softmax, 14)
KAD_FUNC_OP1(kad_stdnorm, 32)

kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight)
{
  kad_node_t *s;
  s = kad_new_core(0, 13, 3);
  s->child[0] = pred, s->child[1] = truth, s->child[2] = weight;
  return kad_finalize_node(s);
}

int conv_find_par(int in_size, int kernel_size, int stride, int pad0, int *new_pad0, int *new_pad1)
{
  int out_size, pad_both;
  
  if (pad0 == KAD_PAD_SAME && stride == 1) out_size = in_size;
  else out_size = (in_size - kernel_size + (pad0 > 0? pad0 : 0) + stride - 1) / stride + 1;
  pad_both = (out_size - 1) * stride + kernel_size - in_size;
  *new_pad0 = pad_both / 2;
  *new_pad1 = pad_both - *new_pad0;
  return out_size;
}

typedef struct {
  int kernel_size, stride, pad[2];
} conv_conf_t;

conv_conf_t *conv2d_gen_aux(int in_row, int in_col, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
  conv_conf_t *cnn;
  cnn = (conv_conf_t*)calloc(2, sizeof(conv_conf_t));
  cnn[0].kernel_size = kernel_r, cnn[0].stride = stride_r;
  cnn[1].kernel_size = kernel_c, cnn[1].stride = stride_c;
  conv_find_par(in_row, kernel_r, stride_r, top_pad,  &cnn[0].pad[0], &cnn[0].pad[1]);
  conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn[1].pad[0], &cnn[1].pad[1]);
  return cnn;
}

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int stride_r, int stride_c, int top_pad, int left_pad)
{
  kad_node_t *s;
  if (x->n_d != 4 || w->n_d != 4) return 0;
  s = kad_new_core(0, 16, 2);
  s->child[0] = x, s->child[1] = w;
  s->ptr = conv2d_gen_aux(x->d[2], x->d[3], w->d[2], w->d[3], stride_r, stride_c, top_pad, left_pad);
  s->ptr_size = sizeof(conv_conf_t) * 2;
  return kad_finalize_node(s);
}

kad_node_t *kad_max2d(kad_node_t *x, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
  kad_node_t *s;
  if (x->n_d != 4) return 0;
  s = kad_new_core(0, 17, 1);
  s->child[0] = x;
  s->ptr = conv2d_gen_aux(x->d[2], x->d[3], kernel_r, kernel_c, stride_r, stride_c, top_pad, left_pad);
  s->ptr_size = sizeof(conv_conf_t) * 2;
  return kad_finalize_node(s);
}

conv_conf_t *conv1d_gen_aux(int in_col, int kernel_c, int stride_c, int left_pad)
{
  conv_conf_t *cnn;
  cnn = (conv_conf_t*)calloc(1, sizeof(conv_conf_t));
  cnn->kernel_size = kernel_c, cnn->stride = stride_c;
  conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn->pad[0], &cnn->pad[1]);
  return cnn;
}

kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int left_pad)
{
  kad_node_t *s;
  if (x->n_d != 3 || w->n_d != 3) return 0;
  s = kad_new_core(0, 18, 2);
  s->child[0] = x, s->child[1] = w;
  s->ptr = conv1d_gen_aux(x->d[2], w->d[2], stride, left_pad);
  s->ptr_size = sizeof(conv_conf_t);
  return kad_finalize_node(s);
}

kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int left_pad)
{
  kad_node_t *s;
  if (x->n_d != 3) return 0;
  s = kad_new_core(0, 19, 1);
  s->child[0] = x;
  s->ptr = conv1d_gen_aux(x->d[2], kernel_size, stride, left_pad);
  s->ptr_size = sizeof(conv_conf_t);
  return kad_finalize_node(s);
}

kad_node_t *kad_avg1d(kad_node_t *x, int kernel_size, int stride, int left_pad)
{
  kad_node_t *s;
  if (x->n_d != 3) return 0;
  s = kad_new_core(0, 28, 1);
  s->child[0] = x;
  s->ptr = conv1d_gen_aux(x->d[2], kernel_size, stride, left_pad);
  s->ptr_size = sizeof(conv_conf_t);
  return kad_finalize_node(s);
}

kad_node_t *kad_pooling_general(int op, int n, kad_node_t **x)
{
  int i;
  kad_node_t *s;
  s = kad_new_core(0, op, n);
  s->flag |= KAD_POOL;
  for (i = 0; i < n; ++i)
    s->child[i] = x[i];
  return kad_finalize_node(s);
}

kad_node_t *kad_avg(int n, kad_node_t **x)   { return kad_pooling_general(10, n, x); }
kad_node_t *kad_max(int n, kad_node_t **x)   { return kad_pooling_general(21, n, x); }
kad_node_t *kad_stack(int n, kad_node_t **x) { return kad_pooling_general(35, n, x); }

kad_node_t *kad_select(int n, kad_node_t **x, int which)
{
  kad_node_t *s;
  int32_t i, *aux;
  aux = (int32_t*)calloc(1, 4);
  *aux = which;
  s = kad_new_core(0, 12, n);
  for (i = 0; i < n; ++i) s->child[i] = x[i];
  s->flag |= KAD_POOL, s->ptr = aux, s->ptr_size = 4;
  return kad_finalize_node(s);
}

kad_node_t *kad_reduce_general(int op, kad_node_t *x, int axis)
{
  kad_node_t *s;
  int32_t *aux;
  aux = (int32_t*)malloc(4);
  aux[0] = axis;
  s = kad_new_core(0, op, 1);
  s->child[0] = x;
  s->ptr = aux, s->ptr_size = 4;
  return kad_finalize_node(s);
}

kad_node_t *kad_reduce_sum(kad_node_t *x, int axis)  { return kad_reduce_general(25, x, axis); }
kad_node_t *kad_reduce_mean(kad_node_t *x, int axis) { return kad_reduce_general(26, x, axis); }

kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *y)
{
  kad_node_t *z;
  z = kad_op2_core(15, x, y);
  z->ptr = kad_rng(), z->ptr_size = sizeof(kad_rng_t);
  return z;
}

kad_node_t *kad_sample_normal(kad_node_t *x)
{
  kad_node_t *z;
  z = kad_op1_core(24, x);
  z->ptr = kad_rng(), z->ptr_size = sizeof(kad_rng_t);
  return z;
}

kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end)
{
  kad_node_t *s;
  int32_t *aux;
  if (end < start || start < 0) return 0;
  aux = (int32_t*)malloc(3 * 4);
  aux[0] = axis, aux[1] = start, aux[2] = end;
  s = kad_new_core(0, 20, 1);
  s->child[0] = x;
  s->ptr = aux, s->ptr_size = 3 * 4;
  return kad_finalize_node(s);
}

kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p)
{
  kad_node_t *s;
  int32_t i, *aux;
  aux = (int32_t*)malloc(4);
  aux[0] = axis;
  s = kad_new_core(0, 31, n);
  for (i = 0; i < n; ++i)
    s->child[i] = p[i];
  s->ptr = aux, s->ptr_size = 4;
  return kad_finalize_node(s);
}

kad_node_t *kad_concat(int axis, int n, ...)
{
  int i;
  kad_node_t **p, *s;
  va_list ap;
  p = (kad_node_t**)malloc(n * sizeof(kad_node_t*));
  va_start(ap, n);
  for (i = 0; i < n; ++i) p[i] = va_arg(ap, kad_node_p);
  va_end(ap);
  s = kad_concat_array(axis, n, p);
  free(p);
  return s;
}

kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d)
{
  kad_node_t *s;
  int32_t i, *aux = 0;
  if (n_d > 0) {
    aux = (int32_t*)malloc(n_d * 4);
    for (i = 0; i < n_d; ++i) aux[i] = d? d[i] : -1;
  }
  s = kad_new_core(0, 30, 1);
  s->child[0] = x, s->ptr = aux, s->ptr_size = n_d * 4;
  return kad_finalize_node(s);
}

kad_node_t *kad_reverse(kad_node_t *x, int axis)
{
  kad_node_t *s;
  int32_t *aux;
  aux = (int32_t*)malloc(4);
  *aux = axis;
  s = kad_new_core(0, 36, 1);
  s->child[0] = x, s->ptr = aux, s->ptr_size = 4;
  return kad_finalize_node(s);
}

kad_node_t *kad_switch(int n, kad_node_t **p)
{
  kad_node_t *s;
  int32_t i, *aux;
  aux = (int32_t*)calloc(1, 4);
  s = kad_new_core(0, 12, n);
  for (i = 0; i < n; ++i)
    s->child[i] = p[i];
  s->ptr = aux, s->ptr_size = 4;
  return kad_finalize_node(s);
}

void kad_mark_back(int n, kad_node_t **v)
{
  int i, j;
  for (i = 0; i < n; ++i) {
    if (v[i]->n_child == 0) continue;
    for (j = 0; j < v[i]->n_child; ++j)
      if (kad_is_back(v[i]->child[j]))
        break;
    if (j < v[i]->n_child) v[i]->flag |= KAD_VAR;
    else v[i]->flag &= ~KAD_VAR;
  }
}

void kad_allocate_internal(int n, kad_node_t **v)
{
  int i;
  kad_mark_back(n, v);
  for (i = 0; i < n; ++i) {
    kad_node_t *p = v[i];
    if (p->n_child == 0) continue;
    p->x = (float*)realloc(p->x, kad_len(p) * sizeof(float));
    if (kad_is_back(p)) {
      p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
      kad_op_list[p->op](p, KAD_ALLOC);
    }
  }
}

int kad_sync_dim(int n, kad_node_t **v, int batch_size)
{
  int i, req_alloc = 0, req_sync = 0, old_size = 0;
  for (i = 0; i < n; ++i) {
    if (kad_is_feed(v[i])) {
      old_size = v[i]->d[0]; 
      if (batch_size > 0 && v[i]->d[0] != batch_size)
        v[i]->d[0] = batch_size, req_sync = 1;
    } else if (v[i]->n_child > 0 && req_sync)
      kad_op_list[v[i]->op](v[i], KAD_SYNC_DIM);
  }
  if (old_size < batch_size) req_alloc = 1;
  for (i = 0; i < n; ++i)
    if (v[i]->n_child > 0 && v[i]->x == 0) req_alloc = 1;
  if (req_alloc) kad_allocate_internal(n, v);
  return batch_size > 0? batch_size : old_size;
}

#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])

#define kv_push(type, v, x) do { \
    if ((v).n == (v).m) { \
      (v).m = (v).m? (v).m<<1 : 2; \
      (v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
    } \
    (v).a[(v).n++] = (x); \
  } while (0)

kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots)
{
  int i;
  kvec_t(kad_node_p) stack = {0,0,0}, a = {0,0,0};

  for (i = 0; i < n_roots; ++i) {
    roots[i]->tmp = 1; 
    kv_push(kad_node_p, stack, roots[i]);
  }
  while (stack.n) {
    kad_node_t *p = kv_pop(stack);
    for (i = 0; i < p->n_child; ++i) {
      kad_node_t *q = p->child[i];
      if (q->tmp == 0) kv_push(kad_node_p, stack, q);
      q->tmp += 1<<1;
    }
  }

  for (i = 0; i < n_roots; ++i)
    if (roots[i]->tmp>>1 == 0) 
      kv_push(kad_node_p, stack, roots[i]);
  while (stack.n) {
    kad_node_t *p = kv_pop(stack);
    kv_push(kad_node_p, a, p);
    for (i = 0; i < p->n_child; ++i) {
      p->child[i]->tmp -= 1<<1;
      if (p->child[i]->tmp>>1 == 0)
        kv_push(kad_node_p, stack, p->child[i]);
    }
  }
  free(stack.a);
  for (i = 0; i < (int)a.n; ++i) { 
    assert(a.a[i]->tmp>>1 == 0);
    a.a[i]->tmp = 0;
  }

  for (i = 0; i < (int)a.n>>1; ++i) { 
    kad_node_p t;
    t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
  }
  kad_allocate_internal(a.n, a.a);

  *n_node = a.n;
  return a.a;
}

kad_node_t **kad_compile(int *n_node, int n_roots, ...)
{
  int i;
  kad_node_t **roots, **ret;
  va_list ap;

  roots = (kad_node_t**)malloc(n_roots * sizeof(kad_node_t*));
  va_start(ap, n_roots);
  for (i = 0; i < n_roots; ++i) roots[i] = va_arg(ap, kad_node_p);
  va_end(ap);
  ret = kad_compile_array(n_node, n_roots, roots);
  free(roots);
  return ret;
}

void kad_delete(int n, kad_node_t **a)
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

int kad_size_var(int n, kad_node_t *const* v)
{
  int c, i;
  for (i = c = 0; i < n; ++i)
    if (kad_is_var(v[i]))
      c += kad_len(v[i]);
  return c;
}

int kad_size_const(int n, kad_node_t *const* v)
{
  int c, i;
  for (i = c = 0; i < n; ++i)
    if (kad_is_const(v[i]))
      c += kad_len(v[i]);
  return c;
}

void kad_propagate_marks(int n, kad_node_t **a)
{
  int i, j;
  for (i = n - 1; i >= 0; --i) {
    kad_node_t *p = a[i];
    if (p->tmp > 0) {
      if (kad_is_switch(p)) {
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

void kad_eval_marked(int n, kad_node_t **a)
{
  int i;
  kad_propagate_marks(n, a);
  for (i = 0; i < n; ++i)
    if (a[i]->n_child && a[i]->tmp > 0)
      kad_op_list[a[i]->op](a[i], KAD_FORWARD);
  for (i = 0; i < n; ++i) a[i]->tmp = 0;
}

const float *kad_eval_at(int n, kad_node_t **a, int from)
{
  int i;
  if (from < 0 || from >= n) from = n - 1;
  for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
  kad_eval_marked(n, a);
  return a[from]->x;
}

void kad_grad(int n, kad_node_t **a, int from)
{
  int i;
  if (from < 0 || from >= n) from = n - 1;
  assert(a[from]->n_d == 0);
  for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
  kad_propagate_marks(n, a);
  for (i = 0; i <= from; ++i) 
    if (a[i]->g && a[i]->tmp > 0)
      memset(a[i]->g, 0, kad_len(a[i]) * sizeof(float));
  for (i = from, a[i]->g[0] = 1.0f; i >= 0; --i) 
    if (a[i]->n_child && a[i]->tmp > 0)
      kad_op_list[a[i]->op](a[i], KAD_BACKWARD);
  for (i = 0; i <= from; ++i) a[i]->tmp = 0;
}

void kad_save1(FILE *fp, const kad_node_t *p)
{
  fwrite(&p->ext_label, 4, 1, fp);
  fwrite(&p->ext_flag, 4, 1, fp);
  fwrite(&p->flag, 1, 1, fp);
  fwrite(&p->n_child, 4, 1, fp);
  if (p->n_child) {
    int32_t j, pre = p->pre? p->pre->tmp : -1;
    fwrite(&p->op, 2, 1, fp);
    for (j = 0; j < p->n_child; ++j)
      fwrite(&p->child[j]->tmp, 4, 1, fp);
    fwrite(&pre, 4, 1, fp);
    fwrite(&p->ptr_size, 4, 1, fp);
    if (p->ptr_size > 0 && p->ptr)
      fwrite(p->ptr, p->ptr_size, 1, fp);
  } else {
    fwrite(&p->n_d, 1, 1, fp);
    if (p->n_d) fwrite(p->d, 4, p->n_d, fp);
  }
}

kad_node_t *kad_load1(FILE *fp, kad_node_t **node)
{
  kad_node_t *p;
  p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
  fread(&p->ext_label, 4, 1, fp);
  fread(&p->ext_flag, 4, 1, fp);
  fread(&p->flag, 1, 1, fp);
  fread(&p->n_child, 4, 1, fp);
  if (p->n_child) {
    int32_t j, k;
    p->child = (kad_node_t**)calloc(p->n_child, sizeof(kad_node_t*));
    fread(&p->op, 2, 1, fp);
    for (j = 0; j < p->n_child; ++j) {
      fread(&k, 4, 1, fp);
      p->child[j] = node? node[k] : 0;
    }
    fread(&k, 4, 1, fp);
    if (k >= 0) p->pre = node[k];
    fread(&p->ptr_size, 4, 1, fp);
    if (p->ptr_size > 0) {
      p->ptr = malloc(p->ptr_size);
      fread(p->ptr, p->ptr_size, 1, fp);
    }
  } else {
    fread(&p->n_d, 1, 1, fp);
    if (p->n_d) fread(p->d, 4, p->n_d, fp);
  }
  return p;
}

int kad_save(FILE *fp, int n_node, kad_node_t **node)
{
  int32_t i, k = n_node;
  fwrite(&k, 4, 1, fp);
  for (i = 0; i < n_node; ++i) node[i]->tmp = i;
  for (i = 0; i < n_node; ++i) kad_save1(fp, node[i]);
  for (i = 0; i < n_node; ++i) node[i]->tmp = 0;
  return 0;
}

kad_node_t **kad_load(FILE *fp, int *_n_node)
{
  int32_t i, n_node;
  kad_node_t **node;
  fread(&n_node, 4, 1, fp);
  node = (kad_node_t**)malloc(n_node * sizeof(kad_node_t*));
  for (i = 0; i < n_node; ++i) {
    kad_node_t *p;
    p = node[i] = kad_load1(fp, node);
    if (p->n_child) {
      kad_op_list[p->op](p, KAD_ALLOC);
      kad_op_list[p->op](p, KAD_SYNC_DIM);
    }
  }
  *_n_node = n_node;
  kad_mark_back(n_node, node);
  return node;
}

kad_node_t *kad_dup1(const kad_node_t *p)
{
  kad_node_t *q;
  q = (kad_node_t*)malloc(sizeof(kad_node_t));
  memcpy(q, p, sizeof(kad_node_t));
  q->pre = 0, q->tmp = 0, q->gtmp = 0;
  if (p->ptr && p->ptr_size > 0) {
    if (kad_use_rng(p) && !(p->flag & KAD_SHARE_RNG) && p->ptr_size == sizeof(kad_rng_t)) {
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

kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size)
{
  int i, j;
  kad_node_t **u;
  u = (kad_node_t**)calloc(n, sizeof(kad_node_t*));
  for (i = 0; i < n; ++i) v[i]->tmp = i;
  for (i = 0; i < n; ++i) {
    kad_node_t *p = v[i], *q;
    q = u[i] = kad_dup1(p);
    if (p->pre) q->pre = u[p->pre->tmp];
    if (p->n_child) {
      for (j = 0; j < p->n_child; ++j)
        q->child[j] = u[p->child[j]->tmp];
    } else if (!kad_is_feed(p)) {
      q->x = (float*)malloc(kad_len(p) * sizeof(float));
      memcpy(q->x, p->x, kad_len(p) * sizeof(float));
      q->g = 0;
    }
  }
  for (i = 0; i < n; ++i) v[i]->tmp = 0;
  kad_sync_dim(n, u, batch_size); 
  return u;
}

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

  assert(kad_is_pivot(v[i_pivot]) && t[i_pivot] == 0);
  t[i_pivot] = kad_dup1(v[i_pivot]);
  t[i_pivot]->n_child = len;
  t[i_pivot]->child = (kad_node_t**)realloc(t[i_pivot]->child, len * sizeof(kad_node_t*));

  flag = (uint8_t*)calloc(n_v, 1);
  for (i = i_pivot, flag[i] = 16; i >= 0; --i) {
    if (i < i_pivot && kad_is_pivot(v[i])) continue; 
    if (flag[i]&16) 
      for (j = 0; j < v[i]->n_child; ++j)
        flag[v[i]->child[j]->tmp] = 16;
  }
  for (i = 0; i < i_pivot; ++i) {
    if (!(flag[i]&16)) continue;
    if (kad_is_var(v[i]) || kad_is_const(v[i]) || kad_is_pivot(v[i])) flag[i] |= 1; 
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

int kad_n_pivots(int n_v, kad_node_t **v)
{
  int i, n_pivots = 0;
  for (i = 0; i < n_v; ++i)
    if (kad_is_pivot(v[i])) ++n_pivots;
  return n_pivots;
}

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
      if (kad_is_pivot(v[i])) i_pivots[k++] = i;
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

float kad_sdot(int n, const float *x, const float *y) 
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
void kad_saxpy_inlined(int n, float a, const float *x, float *y) 
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

void kad_vec_mul_sum(int n, float *a, const float *b, const float *c)
{
  int i;
  for (i = 0; i < n; ++i) a[i] += b[i] * c[i];
}

void kad_saxpy(int n, float a, const float *x, float *y) { kad_saxpy_inlined(n, a, x, y); }

void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C) 
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
            cii[jj] += kad_sdot(K, aii, bjj);
        }
      }
  } else if (!trans_A && !trans_B) {
    for (i = 0; i < M; ++i)
      for (k = 0; k < K; ++k)
        kad_saxpy_inlined(N, A[i*K+k], &B[k*N], &C[i*N]);
  } else if (trans_A && !trans_B) {
    for (k = 0; k < K; ++k)
      for (i = 0; i < M; ++i)
        kad_saxpy_inlined(N, A[k*M+i], &B[k*N], &C[i*N]);
  } else abort(); 
}

kad_rng_t kad_rng_dat = { {0x50f5647d2380309dULL, 0x91ffa96fc4c62cceULL}, 0.0, 0, 0 };

uint64_t kad_splitmix64(uint64_t x)
{
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

uint64_t kad_xoroshiro128plus_next(kad_rng_t *r)
{
  const uint64_t s0 = r->s[0];
  uint64_t s1 = r->s[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  r->s[0] = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
  r->s[1] = s0 << 36 | s0 >> 28;
  return result;
}

void kad_xoroshiro128plus_jump(kad_rng_t *r)
{
  static const uint64_t JUMP[] = { 0xbeac0467eba5facbULL, 0xd86b048b86aa9922ULL };
  uint64_t s0 = 0, s1 = 0;
  int i, b;
  for (i = 0; i < 2; ++i)
    for (b = 0; b < 64; b++) {
      if (JUMP[i] & 1ULL << b)
        s0 ^= r->s[0], s1 ^= r->s[1];
      kad_xoroshiro128plus_next(r);
    }
  r->s[0] = s0, r->s[1] = s1;
}

void kad_srand(void *d, uint64_t seed)
{
  kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
  r->n_gset = 0.0, r->n_iset = 0;
  r->s[0] = kad_splitmix64(seed);
  r->s[1] = kad_splitmix64(r->s[0]);
}

void *kad_rng(void)
{
  kad_rng_t *r;
  r = (kad_rng_t*)calloc(1, sizeof(kad_rng_t));
  kad_xoroshiro128plus_jump(&kad_rng_dat);
  r->s[0] = kad_rng_dat.s[0], r->s[1] = kad_rng_dat.s[1];
  return r;
}

uint64_t kad_rand(void *d) { return kad_xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat); }

double kad_drand(void *d)
{
  union { uint64_t i; double d; } u;
  u.i = 0x3FFULL << 52 | kad_xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat) >> 12;
  return u.d - 1.0;
}

double kad_drand_normal(void *d)
{
  kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
  if (r->n_iset == 0) {
    double fac, rsq, v1, v2;
    do {
      v1 = 2.0 * kad_drand(d) - 1.0;
      v2 = 2.0 * kad_drand(d) - 1.0;
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

void kad_copy_dim1(kad_node_t *dst, const kad_node_t *src) 
{
  dst->n_d = src->n_d;
  if (src->n_d) memcpy(dst->d, src->d, src->n_d * sizeof(int));
}

int kad_op_add(kad_node_t *p, int action)
{
  int i, n0, n1;
  kad_node_t *q[2];

  q[0] = p->child[0], n0 = kad_len(q[0]);
  q[1] = p->child[1], n1 = kad_len(q[1]);
  if (action == KAD_SYNC_DIM) {
    if (n0 % n1 != 0) return -1;
    kad_copy_dim1(p, q[0]);
  } else if (action == KAD_FORWARD) {
    assert(n0 >= n1);
    memcpy(p->x, q[0]->x, n0 * sizeof(float));
    for (i = 0; i < n0; i += n1)
      kad_saxpy(n1, 1.0f, q[1]->x, p->x + i);
  } else if (action == KAD_BACKWARD) {
    if (kad_is_back(q[0])) kad_saxpy(n0, 1.0f, p->g, q[0]->g);
    if (kad_is_back(q[1]))
      for (i = 0; i < n0; i += n1)
        kad_saxpy(n1, 1.0f, p->g + i, q[1]->g);
  }
  return 0;
}

int kad_op_sub(kad_node_t *p, int action)
{
  int i, n0, n1;
  kad_node_t *q[2];

  q[0] = p->child[0], n0 = kad_len(q[0]);
  q[1] = p->child[1], n1 = kad_len(q[1]);
  if (action == KAD_SYNC_DIM) {
    if (n0 % n1 != 0) return -1;
    kad_copy_dim1(p, q[0]);
  } else if (action == KAD_FORWARD) {
    assert(n0 >= n1);
    memcpy(p->x, q[0]->x, n0 * sizeof(float));
    for (i = 0; i < n0; i += n1)
      kad_saxpy(n1, -1.0f, q[1]->x, p->x + i);
  } else if (action == KAD_BACKWARD) {
    if (kad_is_back(q[0])) kad_saxpy(n0, 1.0f, p->g, q[0]->g);
    if (kad_is_back(q[1]))
      for (i = 0; i < n0; i += n1)
        kad_saxpy(n1, -1.0f, p->g + i, q[1]->g);
  }
  return 0;
}

int kad_op_mul(kad_node_t *p, int action)
{
  int i, n0, n1;
  kad_node_t *q[2];

  q[0] = p->child[0], n0 = kad_len(q[0]);
  q[1] = p->child[1], n1 = kad_len(q[1]);
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
    if (kad_is_back(q[0]) && q[1]->x)
      for (i = 0; i < n0; i += n1)
        kad_vec_mul_sum(n1, q[0]->g + i, p->g + i, q[1]->x);
    if (kad_is_back(q[1]) && q[0]->x)
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
  n_a_row = kad_len(q[0]) / n_a_col, n_b_row = kad_len(q[1]) / n_b_col;
  if (action == KAD_SYNC_DIM) {
    if (n_a_col != n_b_col) return -1;
    p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_row;
  } else if (action == KAD_FORWARD) {
    memset(p->x, 0, n_a_row * n_b_row * sizeof(float));
    if (q[0]->x && q[1]->x)
      kad_sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x, q[1]->x, p->x); 
  } else if (action == KAD_BACKWARD) {
    if (kad_is_back(q[0]) && q[1]->x)
      kad_sgemm_simple(0, 0, n_a_row, n_col, n_b_row, p->g, q[1]->x, q[0]->g); 
    if (kad_is_back(q[1]) && q[0]->x)
      kad_sgemm_simple(1, 0, n_b_row, n_col, n_a_row, p->g, q[0]->x, q[1]->g); 
  }
  return 0;
}

int kad_op_matmul(kad_node_t *p, int action) 
{
  int n_a_row, n_b_row, n_a_col, n_b_col;
  kad_node_t *q[2];

  q[0] = p->child[0];
  q[1] = p->child[1];
  n_a_row = q[0]->n_d == 1? 1 : q[0]->d[0];
  n_b_row = q[1]->n_d == 1? 1 : q[1]->d[0];
  n_a_col = kad_len(q[0]) / n_a_row;
  n_b_col = kad_len(q[1]) / n_b_row;
  if (action == KAD_SYNC_DIM) {
    if (n_a_col != n_b_row) return -1;
    p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_col;
  } else if (action == KAD_FORWARD) {
    memset(p->x, 0, n_a_row * n_b_col * sizeof(float));
    if (q[0]->x && q[1]->x)
      kad_sgemm_simple(0, 0, n_a_row, n_b_col, n_a_col, q[0]->x, q[1]->x, p->x); 
  } else if (action == KAD_BACKWARD) {
    if (kad_is_back(q[0]) && q[1]->x)
      kad_sgemm_simple(0, 1, n_a_row, n_a_col, n_b_col, p->g, q[1]->x, q[0]->g); 
    if (kad_is_back(q[1]) && q[0]->x)
      kad_sgemm_simple(1, 0, n_b_row, n_b_col, n_a_row, q[0]->x, p->g, q[1]->g); 
  }
  return 0;
}

int kad_op_square(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i)
      p->x[i] = q->x[i] * q->x[i];
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * (q->x[i] + q->x[i]);
  }
  return 0;
}

int kad_op_1minus(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i) p->x[i] = 1.0f - q->x[i];
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    kad_saxpy(n, -1.0f, p->g, q->g);
  }
  return 0;
}

int kad_op_exp(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i) p->x[i] = expf(q->x[i]);
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * p->x[i];
  }
  return 0;
}

int kad_op_log(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i) p->x[i] = logf(q->x[i]);
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] / q->x[i];
  }
  return 0;
}

int kad_op_reduce_sum(kad_node_t *p, int action)
{
  kad_node_t *q = p->child[0];
  int i, j, k, axis, d0, d1;

  assert(p->ptr);
  axis = *(int32_t*)p->ptr;
  if (axis < 0 || axis >= q->n_d) return -1;
  for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
  for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
  if (action == KAD_SYNC_DIM) {
    p->n_d = q->n_d - 1;
    for (i = j = 0; i < q->n_d; ++i)
      if (i != axis) p->d[j++] = q->d[i];
  } else if (action == KAD_FORWARD) {
    memset(p->x, 0, kad_len(p) * sizeof(float));
    for (i = 0; i < d0; ++i)
      for (j = 0; j < q->d[axis]; ++j)
        for (k = 0; k < d1; ++k)
          p->x[i * d1 + k] += q->x[(i * q->d[axis] + j) * d1 + k];
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < d0; ++i)
      for (j = 0; j < q->d[axis]; ++j)
        for (k = 0; k < d1; ++k)
          q->g[(i * q->d[axis] + j) * d1 + k] += p->g[i * d1 + k];
  }
  return 0;
}

int kad_op_reduce_mean(kad_node_t *p, int action)
{
  kad_node_t *q = p->child[0];
  int i, j, k, axis, d0, d1;

  assert(p->ptr);
  axis = *(int32_t*)p->ptr;
  if (axis < 0 || axis >= q->n_d) return -1;
  for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
  for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
  if (action == KAD_SYNC_DIM) {
    p->n_d = q->n_d - 1;
    for (i = j = 0; i < q->n_d; ++i)
      if (i != axis) p->d[j++] = q->d[i];
  } else if (action == KAD_FORWARD) {
    float t = 1.0f / q->d[axis];
    memset(p->x, 0, kad_len(p) * sizeof(float));
    for (i = 0; i < d0; ++i)
      for (j = 0; j < q->d[axis]; ++j)
        for (k = 0; k < d1; ++k)
          p->x[i * d1 + k] += t * q->x[(i * q->d[axis] + j) * d1 + k];
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    float t = 1.0f / q->d[axis];
    for (i = 0; i < d0; ++i)
      for (j = 0; j < q->d[axis]; ++j)
        for (k = 0; k < d1; ++k)
          q->g[(i * q->d[axis] + j) * d1 + k] += t * p->g[i * d1 + k];
  }
  return 0;
}

int kad_op_dropout(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  assert(p->child[1]->n_d == 0);
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_ALLOC) {
    if (kad_is_back(p->child[0]))
      p->gtmp = realloc(p->gtmp, n);
  } else if (action == KAD_FORWARD) {
    float r = kad_is_const(q) || kad_is_var(q)? 0.0f : *p->child[1]->x, z = 1.0f / (1.0f - r);
    uint8_t *flag = (uint8_t*)p->gtmp;
    for (i = 0; i < n; ++i) {
      int kept = (kad_drand(p->ptr) >= r);
      p->x[i] = kept? q->x[i] * z : 0.0f;
      if (flag) flag[i] = kept;
    }
  } else if (action == KAD_BACKWARD && kad_is_back(p->child[0])) {
    float r = kad_is_const(q) || kad_is_var(q)? 0.0f : *p->child[1]->x, z = 1.0f / (1.0f - r);
    uint8_t *flag = (uint8_t*)p->gtmp;
    for (i = 0; i < n; ++i)
      if (flag[i]) q->g[i] += z * p->g[i];
  }
  return 0;
}

int kad_op_sample_normal(kad_node_t *p, int action) 
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_ALLOC) {
    if (kad_is_back(p->child[0]))
      p->gtmp = realloc(p->gtmp, n * sizeof(float));
  } else if (action == KAD_FORWARD) {
    float *r = (float*)p->gtmp;
    for (i = 0; i < n; ++i) {
      float z;
      z = (float)kad_drand_normal(p->ptr);
      p->x[i] = q->x[i] * z;
      if (r) r[i] = z;
    }
  } else if (action == KAD_BACKWARD && kad_is_back(p->child[0])) {
    float *r = (float*)p->gtmp;
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * r[i];
  }
  return 0;
}

int kad_op_slice(kad_node_t *p, int action)
{
  kad_node_t *q = p->child[0];
  int32_t *aux, *range;
  int i, axis, d0, d1;

  assert(p->ptr);
  aux = (int32_t*)p->ptr, axis = aux[0], range = aux + 1;
  if (axis < 0 || axis >= q->n_d) return -1;
  for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
  for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
  if (action == KAD_SYNC_DIM) {
    if (range[0] >= range[1] || range[0] < 0 || range[1] > q->d[axis]) return -1;
    kad_copy_dim1(p, q);
    p->d[axis] = range[1] - range[0];
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < d0; ++i)
      memcpy(&p->x[i * p->d[axis] * d1], &q->x[(i * q->d[axis] + range[0]) * d1], (range[1] - range[0]) * d1 * sizeof(float));
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < d0; ++i)
      kad_saxpy((range[1] - range[0]) * d1, 1.0f, &p->g[i * p->d[axis] * d1], &q->g[(i * q->d[axis] + range[0]) * d1]);
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
        if (!kad_is_back(q)) continue;
        kad_saxpy(q->d[axis] * d1, 1.0f, &p->g[(i * p->d[axis] + k) * d1], &q->g[i * q->d[axis] * d1]);
        k += q->d[axis];
      }
  }
  return 0;
}

int kad_op_reshape(kad_node_t *p, int action)
{
  kad_node_t *q = p->child[0];

  if (action == KAD_SYNC_DIM) {
    if (p->ptr) {
      int32_t *aux = (int32_t*)p->ptr;
      int i, len = 1, n_missing = 0;
      p->n_d = p->ptr_size / 4;
      for (i = 0; i < p->n_d; ++i) p->d[i] = aux[i];
      for (i = 0; i < p->n_d; ++i)
        if (p->d[i] <= 0) ++n_missing;
        else len *= p->d[i];
      if (n_missing == 0 && len != kad_len(q)) return -1;
      if (n_missing > 1) { 
        for (i = 0; i < p->n_d; ++i)
          if (p->d[i] <= 0 && i < q->n_d) {
            p->d[i] = q->d[i], len *= p->d[i];
            if (--n_missing == 1) break;
          }
        if (n_missing > 1) return -1;
      }
      if (n_missing == 1) { 
        if (kad_len(q) % len != 0) return -1;
        for (i = 0; i < p->n_d; ++i)
          if (p->d[i] <= 0) p->d[i] = kad_len(q) / len;
      }
    } else kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    memcpy(p->x, q->x, kad_len(p) * sizeof(float));
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    kad_saxpy(kad_len(p), 1.0f, p->g, q->g);
  }
  return 0;
}

int kad_op_reverse(kad_node_t *p, int action)
{
  kad_node_t *q = p->child[0];
  int axis, i, j, n, d0, d1;

  axis = p->ptr? *(int32_t*)p->ptr : 0;
  if (axis < 0) axis += q->n_d;
  assert(axis >= 0 && axis < q->n_d);
  for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
  n = q->d[axis];
  for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < d0; ++i)
      for (j = 0; j < n; ++j)
        memcpy(&p->x[(i * n + n - 1 - j) * d1], &q->x[(i * n + j) * d1], d1 * sizeof(float));
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < d0; ++i)
      for (j = 0; j < n; ++j)
        kad_saxpy(d1, 1.0f, &p->g[(i * n + n - 1 - j) * d1], &q->g[(i * n + j) * d1]);
  }
  return 0;
}

int kad_op_mse(kad_node_t *p, int action)
{
  kad_node_t *y1 = p->child[0]; 
  kad_node_t *y0 = p->child[1]; 
  int i, n;

  n = kad_len(y0);
  if (action == KAD_SYNC_DIM) {
    if (n != kad_len(y1)) return -1;
    p->n_d = 0;
  } else if (action == KAD_FORWARD) {
    double cost = 0.0;
    for (i = 0; i < n; ++i)
      cost += (y1->x[i] - y0->x[i]) * (y1->x[i] - y0->x[i]);
    p->x[0] = (float)(cost / n);
  } else if (action == KAD_BACKWARD && kad_is_back(y1)) {
    float t = 2.0f * p->g[0] / n;
    for (i = 0; i < n; ++i)
      y1->g[i] += t * (y1->x[i] - y0->x[i]);
  }
  return 0;
}

int kad_op_ce_bin(kad_node_t *p, int action)
{
  static const float tiny = 1e-9f;
  kad_node_t *y1 = p->child[0]; 
  kad_node_t *y0 = p->child[1]; 
  int i, n;

  n = kad_len(y0);
  if (action == KAD_SYNC_DIM) {
    if (n != kad_len(y1)) return -1;
    p->n_d = 0;
  } else if (action == KAD_FORWARD) {
    double cost = 0.0;
    for (i = 0; i < n; ++i) {
      if (y0->x[i] > 0.0f)
        cost += y0->x[i] * log(y0->x[i] / (y1->x[i] > tiny? y1->x[i] : tiny));
      if (1.0f - y0->x[i] > 0.0f)
        cost += (1.0f - y0->x[i]) * log((1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny));
    }
    p->x[0] = (float)(cost / n);
  } else if (action == KAD_BACKWARD && kad_is_back(y1)) {
    float t = p->g[0] / n;
    for (i = 0; i < n; ++i) {
      if (y0->x[i] > 0.0f)
        y1->g[i] -= t * y0->x[i] / (y1->x[i] > tiny? y1->x[i] : tiny);
      if (1.0f - y0->x[i] > 0.0f)
        y1->g[i] += t * (1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny);
    }
  }
  return 0;
}

int kad_op_ce_bin_neg(kad_node_t *p, int action)
{
  static const float tiny = 1e-9f;
  kad_node_t *y1 = p->child[0]; 
  kad_node_t *y0 = p->child[1]; 
  int i, n;

  n = kad_len(y0);
  if (action == KAD_SYNC_DIM) {
    if (n != kad_len(y1)) return -1;
    p->n_d = 0;
  } else if (action == KAD_FORWARD) {
    double cost = 0.0;
    for (i = 0; i < n; ++i) {
      if (1.0f + y0->x[i] > 0.0f)
        cost += .5f * (1.0f + y0->x[i]) * log((1.0f + y0->x[i]) / (1.0f + y1->x[i] > tiny? 1.0f + y1->x[i] : tiny));
      if (1.0f - y0->x[i] > 0.0f)
        cost += .5f * (1.0f - y0->x[i]) * log((1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny));
    }
    p->x[0] = (float)(cost / n);
  } else if (action == KAD_BACKWARD && kad_is_back(y1)) {
    float t = p->g[0] / n;
    for (i = 0; i < n; ++i) {
      if (1.0f + y0->x[i] > 0.0f)
        y1->g[i] -= .5f * t * (1.0f + y0->x[i]) / (1.0f + y1->x[i] > tiny? 1.0f + y1->x[i] : tiny);
      if (1.0f - y0->x[i] > 0.0f)
        y1->g[i] += .5f * t * (1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny);
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
  d0 = kad_len(y0) / n1;
  if (p->n_child == 3) {
    c = p->child[2];
    assert(c->n_d == 1 && c->d[0] == n1);
  }
  if (action == KAD_SYNC_DIM) {
    if (kad_len(y0) != kad_len(y1) || y0->d[y0->n_d - 1] != y1->d[y1->n_d - 1]) return -1;
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
  } else if (action == KAD_BACKWARD && kad_is_back(y1)) {
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
  m = kad_len(q) / n;
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
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
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
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i)
      p->x[i] = 1.0f / (1.0f + expf(-q->x[i]));
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * (p->x[i] * (1.0f - p->x[i]));
  }
  return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
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
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * (1.0f - p->x[i] * p->x[i]);
  }
  return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i)
      p->x[i] = q->x[i] > 0.0f? q->x[i] : 0.0f;
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      if (q->x[i] > 0.0f)
        q->g[i] += p->g[i];
  }
  return 0;
}

int kad_op_sin(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    for (i = 0; i < n; ++i) p->x[i] = sinf(q->x[i]);
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    for (i = 0; i < n; ++i)
      q->g[i] += p->g[i] * cosf(q->x[i]);
  }
  return 0;
}

int kad_op_softmax(kad_node_t *p, int action)
{
  int i, j, n1, d0;
  kad_node_t *q = p->child[0];

  n1 = q->d[q->n_d - 1];
  d0 = kad_len(q) / n1;
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
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
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
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    for (i = 1; i < p->n_child; ++i)
      if (kad_len(p->child[i]) != n) return -1;
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    memcpy(p->x, q->x, n * sizeof(float));
    for (i = 1; i < p->n_child; ++i)
      kad_saxpy(n, 1.0f, p->child[i]->x, p->x);
    for (i = 0; i < n; ++i) p->x[i] *= tmp;
  } else if (action == KAD_BACKWARD) {
    for (i = 0; i < p->n_child; ++i)
      if (kad_is_back(p->child[i]))
        kad_saxpy(n, tmp, p->g, p->child[i]->g);
  }
  return 0;
}

int kad_op_max(kad_node_t *p, int action)
{
  int i, n;
  kad_node_t *q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    int *max_j;
    for (i = 1; i < p->n_child; ++i)
      if (kad_len(p->child[i]) != n) return -1;
    kad_copy_dim1(p, q);
    max_j = (int*)calloc(n, sizeof(int));
    p->gtmp = max_j;
  } else if (action == KAD_FORWARD) {
    int j, *max_j = (int*)p->gtmp;
    memset(max_j, 0, n * sizeof(int));
    memcpy(p->x, q->x, n * sizeof(float));
    for (j = 1; j < p->n_child; ++j)
      for (i = 0, q = p->child[j]; i < n; ++i)
        if (q->x[i] > p->x[i]) p->x[i] = q->x[i], max_j[i] = j;
  } else if (action == KAD_BACKWARD) {
    int *max_j = (int*)p->gtmp;
    for (i = 0; i < n; ++i)
      p->child[max_j[i]]->g[i] += p->g[i];
  }
  return 0;
}

int kad_op_stack(kad_node_t *p, int action) 
{
  int i, n, axis = 0;
  kad_node_t *q;

  assert(p->n_child > 0);
  q = p->child[0];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    for (i = 1; i < p->n_child; ++i)
      if (kad_len(p->child[i]) != n) return -1;
    p->n_d = q->n_d + 1;
    for (i = 0; i < axis; ++i) p->d[i] = q->d[i];
    p->d[axis] = p->n_child;
    for (; i < q->n_d; ++i) p->d[i+1] = q->d[i];
  } else if (action == KAD_FORWARD) { 
    for (i = 0; i < p->n_child; ++i)
      memcpy(&p->x[i * n], p->child[i]->x, n * sizeof(float));
  } else if (action == KAD_BACKWARD) {
    for (i = 0; i < p->n_child; ++i)
      if (kad_is_back(p->child[i]))
        kad_saxpy(n, 1.0f, &p->g[i * n], p->child[i]->g);
  }
  return 0;
}

int kad_op_select(kad_node_t *p, int action)
{
  kad_node_t *q;
  int i, n, which;

  which = *(int32_t*)p->ptr;
  if (which < 0) which += p->n_child;
  assert(which >= 0 && which < p->n_child);
  q = p->child[which];
  n = kad_len(q);
  if (action == KAD_SYNC_DIM) {
    for (i = 0; i < p->n_child; ++i)
      if (p->child[i]->n_d != q->n_d || kad_len(p->child[i]) != n)
        break;
    if (i < p->n_child) return -1;
    kad_copy_dim1(p, q);
  } else if (action == KAD_FORWARD) {
    memcpy(p->x, q->x, n * sizeof(float));
  } else if (action == KAD_BACKWARD && kad_is_back(q)) {
    kad_saxpy(n, 1.0f, p->g, q->g);
  }
  return 0;
}

void conv_rot180(int d0, int d1, float *x) 
{
  int i, j;
  for (i = 0; i < d0; ++i) {
    float tmp, *xi = &x[i * d1];
    for (j = 0; j < d1>>1; ++j)
      tmp = xi[j], xi[j] = xi[d1-1-j], xi[d1-1-j] = tmp; 
  }
}

void conv2d_move_1to3(int d[4], const float *x, float *y) 
{
  int i, j, k, l;
  for (i = 0; i < d[0]; ++i)
    for (j = 0; j < d[1]; ++j)
      for (k = 0; k < d[2]; ++k) {
        int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
        for (l = 0; l < d[3]; ++l)
          y[(ik + l) * d[1] + j] = x[ijk + l];
      }
}

void conv2d_add_3to1(int d[4], const float *y, float *x) 
{
  int i, j, k, l;
  for (i = 0; i < d[0]; ++i)
    for (j = 0; j < d[1]; ++j)
      for (k = 0; k < d[2]; ++k) {
        int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
        for (l = 0; l < d[3]; ++l)
          x[ijk + l] += y[(ik + l) * d[1] + j];
      }
}

#define conv_out_size(in_size, aux) (((in_size) - (aux)->kernel_size + (aux)->pad[0] + (aux)->pad[1]) / (aux)->stride + 1)

#define process_row_for(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
  int j, l; \
  if (_stride > 1) { \
    for (l = 0; l < _wn; ++l) { \
      const float *xl = &_xx[l - _pad]; \
      for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
      kad_saxpy(_pn, _ww[l], _t, _yy); \
    } \
  } else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], &_xx[l - _pad], _yy); \
} while (0)

#define process_row_back_x(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
  int j, l; \
  if (_stride > 1) { \
    for (l = 0; l < _wn; ++l) { \
      float *xl = &_xx[l - _pad]; \
      memset(_t, 0, _pn * sizeof(float)); \
      kad_saxpy(_pn, _ww[l], _yy, _t); \
      for (j = 0; j < _pn; ++j, xl += _stride) *xl += _t[j]; \
    } \
  } else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]); \
} while (0)

#define process_row_back_w(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
  int j, l; \
  if (_stride > 1) { \
    for (l = 0; l < _wn; ++l) { \
      const float *xl = &_xx[l - _pad]; \
      for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
      _ww[l] += kad_sdot(_pn, _yy, _t); \
    } \
  } else for (l = 0; l < _wn; ++l) _ww[l] += kad_sdot(_pn, _yy, &_xx[l - _pad]); \
} while (0)

int kad_op_conv2d(kad_node_t *p, int action) 
{
#define conv2d_loop1(_x, _w, _y, _tmp, _row_func) do {  \
    int n, c1, c0, i, k, ii; \
    for (n = 0; n < q->d[0]; ++n)  \
      for (c1 = 0; c1 < w->d[0]; ++c1)  \
        for (c0 = 0; c0 < w->d[1]; ++c0)  \
          for (k = 0; k < w->d[2]; ++k) {  \
            float *_ww = &(_w)[((c1 * w->d[1] + c0) * w->d[2] + k) * w->d[3]]; \
            for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) {  \
              float *_xx = &(_x)[((n * q->d[1] + c0) * q->d[2] + ii) * q->d[3]]; \
              float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i)  * p->d[3]]; \
              if (x_padded) { \
                memcpy(x_padded + aux[1].pad[0], _xx, q->d[3] * sizeof(float)); \
                _xx = x_padded + aux[1].pad[0]; \
              } \
              _row_func(_xx, _ww, _yy, w->d[3], p->d[3], aux[1].stride, aux[1].pad[0], (_tmp)); \
            }  \
          }  \
  } while (0)

#define conv2d_loop2(_x, _w, _y, _code) do {  \
    int n, c1, i, j, k, ii, j_skip = aux[1].stride * q->d[1], m = w->d[3] * w->d[1]; \
    for (n = 0; n < q->d[0]; ++n)  \
      for (c1 = 0; c1 < w->d[0]; ++c1)  \
        for (k = 0; k < w->d[2]; ++k) {  \
          float *_ww = &(_w)[(c1 * w->d[2] + k) * m]; \
          for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) {  \
            float *_xx = &(_x)[(n * q->d[2] + ii) * q->d[3] * q->d[1]]; \
            float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i) * p->d[3]]; \
            if (x_padded) { \
              memcpy(x_padded + aux[1].pad[0] * q->d[1], _xx, q->d[3] * q->d[1] * sizeof(float)); \
              _xx = x_padded; \
            } \
            for (j = 0; j < p->d[3]; ++j, _xx += j_skip, ++_yy) _code;  \
          }  \
        }  \
  } while (0)

  conv_conf_t *aux = (conv_conf_t*)p->ptr;
  kad_node_t *q = p->child[0], *w = p->child[1];
  float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
  int algo_switch = 0;

  if (action == KAD_FORWARD || action == KAD_BACKWARD) { 
    if (w->d[3] * w->d[1] < 16) {
      t = (float*)malloc(p->d[3] * sizeof(float));
      x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc(q->d[3] + aux[1].pad[0] + aux[1].pad[1], sizeof(float)) : 0;
    } else {
      q1 = (float*)malloc(kad_len(q) * sizeof(float));
      w1 = (float*)malloc(kad_len(w) * sizeof(float));
      x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc((q->d[3] + aux[1].pad[0] + aux[1].pad[1]) * q->d[1], sizeof(float)) : 0;
      algo_switch = 1;
    }
  }
  if (action == KAD_SYNC_DIM) {
    if (q->n_d != 4 || w->n_d != 4) return -1;
    if (q->d[1] != w->d[1]) return -1; 
    p->n_d = 4;
    p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
  } else if (action == KAD_FORWARD) {
    conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
    memset(p->x, 0, kad_len(p) * sizeof(float));
    if (!algo_switch) { 
      conv2d_loop1(q->x, w->x, p->x, t, process_row_for);
    } else { 
      conv2d_move_1to3(q->d, q->x, q1);
      conv2d_move_1to3(w->d, w->x, w1);
      conv2d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
    }
    conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
  } else if (action == KAD_BACKWARD) {
    if (kad_is_back(p->child[0])) { 
      conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
      if (!algo_switch) {
        conv2d_loop1(q->g, w->x, p->g, t, process_row_back_x);
      } else {
        memset(q1, 0, kad_len(q) * sizeof(float));
        conv2d_move_1to3(w->d, w->x, w1);
        conv2d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
        conv2d_add_3to1(q->d, q1, q->g);
      }
      conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
    }
    if (kad_is_back(p->child[1])) { 
      conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g);
      if (!algo_switch) {
        conv2d_loop1(q->x, w->g, p->g, t, process_row_back_w);
      } else {
        conv2d_move_1to3(q->d, q->x, q1);
        memset(w1, 0, kad_len(w) * sizeof(float));
        conv2d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
        conv2d_add_3to1(w->d, w1, w->g);
      }
      conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g);
    }
  }
  free(t); free(q1); free(w1); free(x_padded);
  return 0;
}

int kad_op_max2d(kad_node_t *p, int action)
{
  conv_conf_t *aux = (conv_conf_t*)p->ptr;
  kad_node_t *q = p->child[0];
  if (action == KAD_SYNC_DIM) {
    if (q->n_d != 4) return -1;
    p->n_d = 4;
    p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
  } else if (action == KAD_ALLOC) {
    p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
  } else if (action == KAD_FORWARD) {
    int rest = 1, len, t, i;
    int *f = (int*)p->gtmp;
    len = kad_len(p);
    for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
    for (i = 0; i < p->n_d - 2; ++i) rest *= p->d[i];
    for (t = 0; t < rest; ++t) {
      int i, j, k, l, p_row = p->d[p->n_d - 2], p_col = p->d[p->n_d - 1];
      for (i = 0; i < p_row; ++i) {
        int u = (t * p_row + i) * p_col;
        for (k = 0; k < aux[0].kernel_size; ++k) {
          int v, v0, v_end, ii = i * aux[0].stride + k - aux[0].pad[0];
          if (ii < 0 || ii >= q->d[p->n_d - 2]) continue;
          v0 = (t * q->d[p->n_d - 2] + ii) * q->d[p->n_d - 1];
          v_end = v0 + q->d[p->n_d - 1];
          for (l = 0; l < aux[1].kernel_size; ++l)
            for (j = 0, v = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0); j < p_col && v < v_end; ++j, v += aux[1].stride)
              if (p->x[u + j] < q->x[v])
                p->x[u + j] = q->x[v], f[u + j] = v;
        } 
      } 
    }
  } else if (action == KAD_BACKWARD) {
    int i, len, *f = (int*)p->gtmp;
    len = kad_len(p);
    for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
  }
  return 0;
}

void conv1d_move_1to2(int d[3], const float *x, float *y)
{
  int i, j, k;
  for (k = 0; k < d[0]; ++k)
    for (j = 0; j < d[1]; ++j)
      for (i = 0; i < d[2]; ++i)
        y[(k * d[2] + i) * d[1] + j] = x[(k * d[1] + j) * d[2] + i];
}

void conv1d_add_2to1(int d[3], const float *y, float *x)
{
  int i, j, k;
  for (k = 0; k < d[0]; ++k)
    for (j = 0; j < d[1]; ++j)
      for (i = 0; i < d[2]; ++i)
        x[(k * d[1] + j) * d[2] + i] += y[(k * d[2] + i) * d[1] + j];
}

int kad_op_conv1d(kad_node_t *p, int action) 
{
#define conv1d_loop1(_x, _w, _y, _tmp, _row_func) do {  \
    int n, c1, c0; \
    for (n = 0; n < q->d[0]; ++n)  \
      for (c1 = 0; c1 < w->d[0]; ++c1)  \
        for (c0 = 0; c0 < w->d[1]; ++c0) {  \
          float *_ww = &(_w)[(c1 * w->d[1] + c0) * w->d[2]]; \
          float *_xx = &(_x)[(n  * q->d[1] + c0) * q->d[2]]; \
          float *_yy = &(_y)[(n  * p->d[1] + c1) * p->d[2]]; \
          if (x_padded) { \
            memcpy(x_padded + aux->pad[0], _xx, q->d[2] * sizeof(float)); \
            _xx = x_padded + aux->pad[0]; \
          } \
          _row_func(_xx, _ww, _yy, w->d[2], p->d[2], aux->stride, aux->pad[0], (_tmp)); \
        }  \
  } while (0)

#define conv1d_loop2(_x, _w, _y, _code) do {  \
    int n, c1, j, j_skip = aux->stride * q->d[1], m = w->d[2] * w->d[1]; \
    for (n = 0; n < q->d[0]; ++n)  \
      for (c1 = 0; c1 < w->d[0]; ++c1) {  \
        float *_ww = &(_w)[c1 * m]; \
        float *_xx = &(_x)[n * q->d[1] * q->d[2]]; \
        float *_yy = &(_y)[(n * p->d[1] + c1) * p->d[2]]; \
        if (x_padded) { \
          memcpy(x_padded + aux->pad[0] * q->d[1], _xx, q->d[2] * q->d[1] * sizeof(float)); \
          _xx = x_padded; \
        } \
        for (j = 0; j < p->d[2]; ++j, _xx += j_skip, ++_yy) _code; \
      }  \
  } while (0)

  conv_conf_t *aux = (conv_conf_t*)p->ptr;
  kad_node_t *q = p->child[0], *w = p->child[1];
  float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
  int algo_switch = 0;

  if (action == KAD_FORWARD || action == KAD_BACKWARD) { 
    if (w->d[2] * w->d[1] < 32) {
      t = (float*)malloc(p->d[2] * sizeof(float));
      x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc(q->d[2] + aux->pad[0] + aux->pad[1], sizeof(float)) : 0;
    } else {
      q1 = (float*)malloc(kad_len(q) * sizeof(float));
      w1 = (float*)malloc(kad_len(w) * sizeof(float));
      x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc((q->d[2] + aux->pad[0] + aux->pad[1]) * q->d[1], sizeof(float)) : 0;
      algo_switch = 1;
    }
  }
  if (action == KAD_SYNC_DIM) {
    if (q->n_d != 3 || w->n_d != 3) return -1;
    if (q->d[1] != w->d[1]) return -1; 
    p->n_d = 3;
    p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], aux);
  } else if (action == KAD_FORWARD) {
    conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
    memset(p->x, 0, kad_len(p) * sizeof(float));
    if (!algo_switch) { 
      conv1d_loop1(q->x, w->x, p->x, t, process_row_for);
    } else { 
      conv1d_move_1to2(q->d, q->x, q1);
      conv1d_move_1to2(w->d, w->x, w1);
      conv1d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
    }
    conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
  } else if (action == KAD_BACKWARD) {
    if (kad_is_back(p->child[0])) { 
      conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
      if (!algo_switch) {
        conv1d_loop1(q->g, w->x, p->g, t, process_row_back_x);
      } else {
        memset(q1, 0, kad_len(q) * sizeof(float));
        conv1d_move_1to2(w->d, w->x, w1);
        conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
        conv1d_add_2to1(q->d, q1, q->g);
      }
      conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
    }
    if (kad_is_back(p->child[1])) { 
      conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
      if (!algo_switch) {
        conv1d_loop1(q->x, w->g, p->g, t, process_row_back_w);
      } else {
        conv1d_move_1to2(q->d, q->x, q1);
        memset(w1, 0, kad_len(w) * sizeof(float));
        conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
        conv1d_add_2to1(w->d, w1, w->g);
      }
      conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
    }
  }
  free(t); free(q1); free(w1); free(x_padded);
  return 0;
}

int kad_op_max1d(kad_node_t *p, int action)
{
  conv_conf_t *aux = (conv_conf_t*)p->ptr;
  kad_node_t *q = p->child[0];
  if (action == KAD_SYNC_DIM) {
    if (q->n_d != 3) return -1;
    p->n_d = 3;
    p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
  } else if (action == KAD_ALLOC) {
    p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
  } else if (action == KAD_FORWARD) {
    int rest = 1, len, t, i;
    int *f = (int*)p->gtmp;
    len = kad_len(p);
    for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
    for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
    for (t = 0; t < rest; ++t) {
      int j, l, p_width = p->d[p->n_d - 1];
      int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
      for (l = 0; l < aux->kernel_size; ++l)
        for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
          if (p->x[u + j] < q->x[v])
            p->x[u + j] = q->x[v], f[u + j] = v;
    }
  } else if (action == KAD_BACKWARD) {
    int i, len, *f = (int*)p->gtmp;
    len = kad_len(p);
    for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
  }
  return 0;
}

int kad_op_avg1d(kad_node_t *p, int action)
{
  conv_conf_t *aux = (conv_conf_t*)p->ptr;
  kad_node_t *q = p->child[0];
  if (action == KAD_SYNC_DIM) {
    if (q->n_d != 3) return -1;
    p->n_d = 3;
    p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
  } else if (action == KAD_ALLOC) {
    p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
  } else if (action == KAD_FORWARD) {
    int rest = 1, len, t, i;
    int *f = (int*)p->gtmp;
    len = kad_len(p);
    for (i = 0; i < len; ++i) p->x[i] = 0.0f, f[i] = 0;
    for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
    for (t = 0; t < rest; ++t) {
      int j, l, p_width = p->d[p->n_d - 1];
      int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
      for (l = 0; l < aux->kernel_size; ++l)
        for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
          p->x[u + j] += q->x[v], ++f[u + j];
    }
    for (i = 0; i < len; ++i) p->x[i] /= f[i];
  } else if (action == KAD_BACKWARD) {
    int rest = 1, t, i;
    int *f = (int*)p->gtmp;
    for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
    for (t = 0; t < rest; ++t) {
      int j, l, p_width = p->d[p->n_d - 1];
      int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
      for (l = 0; l < aux->kernel_size; ++l)
        for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
          q->g[v] += p->g[u + j] / f[u + j];
    }
  }
  return 0;
}

kad_op_f kad_op_list[KAD_MAX_OP] = {
  0,
  kad_op_add,        
  kad_op_mul,        
  kad_op_cmul,       
  kad_op_ce_bin_neg, 
  kad_op_square,     
  kad_op_sigm,       
  kad_op_tanh,       
  kad_op_relu,       
  kad_op_matmul,     
  kad_op_avg,        
  kad_op_1minus,     
  kad_op_select,     
  kad_op_ce_multi,   
  kad_op_softmax,    
  kad_op_dropout,    
  kad_op_conv2d,     
  kad_op_max2d,      
  kad_op_conv1d,     
  kad_op_max1d,      
  kad_op_slice,      
  kad_op_max,        
  kad_op_ce_bin,     
  kad_op_sub,        
  kad_op_sample_normal,  
  kad_op_reduce_sum,     
  kad_op_reduce_mean,    
  kad_op_log,        
  kad_op_avg1d,      
  kad_op_mse,        
  kad_op_reshape,    
  kad_op_concat,     
  kad_op_stdnorm,    
  kad_op_exp,        
  kad_op_sin,        
  kad_op_stack,      
  kad_op_reverse     
};

char *kad_op_name[KAD_MAX_OP] = {
  0, "add", "mul", "cmul", "ce_bin_neg", "square", "sigm", "tanh", "relu", "matmul", "avg", "1minus", "select", "ce_multi", "softmax",
  "dropout", "conv2d", "max2d", "conv1d", "max1d", "slice", "max", "ce_bin", "sub", "sample_normal", "reduce_sum", "reduce_mean", "log",
  "avg1d", "mse", "reshape", "concat", "stdnorm", "exp", "sin", "stack", "reverse"
};

void kad_trap_fe(void)
{
#ifdef __SSE__
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
}

void kad_print_graph(FILE *fp, int n, kad_node_t **v)
{
  int i, j;
  for (i = 0; i < n; ++i) v[i]->tmp = i;
  for (i = 0; i < n; ++i) {
    kad_node_t *p = v[i];
    fprintf(fp, "%d\t%x:%x\t%d\t", i, p->flag, p->ext_flag, p->ext_label);
    if (p->pre) fprintf(fp, "%d\t", p->pre->tmp);
    else fprintf(fp, ".\t");
    fputs("[", fp);
    for (j = 0; j < p->n_d; ++j) {
      if (j) fputc(',', fp);
      fprintf(fp, "%d", p->d[j]);
    }
    fprintf(fp, "]\t");
    if (p->n_child) {
      fprintf(fp, "%s(", kad_op_name[p->op]);
      for (j = 0; j < p->n_child; ++j) {
        if (j) fputc(',', fp);
        fprintf(fp, "$%d", p->child[j]->tmp);
      }
      fprintf(fp, ")");
    } else fprintf(fp, "%s", kad_is_feed(p)? "feed" : kad_is_var(p)? "var" : kad_is_const(p)? "const" : "N/A");
    fputc('\n', fp);
  }
  for (i = 0; i < n; ++i) v[i]->tmp = 0;
}

void kad_add_delta(int n, kad_node_t **a, float c, float *delta)
{
  int i, k;
  for (i = k = 0; i < n; ++i)
    if (kad_is_var(a[i])) {
      kad_saxpy(kad_len(a[i]), c, &delta[k], a[i]->x);
      k += kad_len(a[i]);
    }
}

void kad_check_grad(int n, kad_node_t **a, int from)
{
  const float eps = 1e-5f, rel = 1e-7f / eps;
  int i, k, n_var;
  float *g0, *delta, f0, f_minus, f_plus, s0, s1, rel_err, p_m_err;
  n_var = kad_size_var(n, a);
  g0 = (float*)calloc(n_var, sizeof(float));
  f0 = *kad_eval_at(n, a, from);
  kad_grad(n, a, from);
  for (i = k = 0; i < n; ++i)
    if (kad_is_var(a[i])) {
      memcpy(&g0[k], a[i]->g, kad_len(a[i]) * sizeof(float));
      k += kad_len(a[i]);
    }
  delta = (float*)calloc(n_var, sizeof(float));
  for (k = 0; k < n_var; ++k) delta[k] = (float)kad_drand(0) * eps;
  kad_add_delta(n, a, 1.0f, delta);
  f_plus = *kad_eval_at(n, a, from);
  kad_add_delta(n, a, -2.0f, delta);
  f_minus = *kad_eval_at(n, a, from);
  kad_add_delta(n, a, 1.0f, delta);
  s0 = kad_sdot(n_var, g0, delta);
  s1 = .5f * (f_plus - f_minus);
  fprintf(stderr, "Gradient check -- %g <=> %g @ %g -- ", s0/eps, s1/eps, f0);
  if (fabs(s1) >= rel * eps) {
    rel_err = fabsf(fabsf(s0) - fabsf(s1)) / (fabsf(s0) + fabsf(s1));
    p_m_err = fabsf(f_plus + f_minus - 2.0f * f0) / fabsf(f_plus - f_minus);
    fprintf(stderr, "rel_err:%g p_m_err:%g -- ", rel_err, p_m_err);
    if (rel_err >= rel && rel_err > p_m_err) fprintf(stderr, "failed\n");
    else fprintf(stderr, "passed\n");
  } else fprintf(stderr, "skipped\n");
  free(delta); free(g0);
}


typedef unsigned int   uint;
typedef unsigned char  byte;
typedef unsigned long long qword;

const int SCALElog = 15;
const int SCALE    = 1<<SCALElog;

struct Rangecoder {
  uint f_DEC; 
  FILE* f;
  qword counter;
  byte get() { counter++; return byte(getc(f)); }
  void put( byte c ) { counter++; putc(c,f); }
  void StartEncode( FILE *F ) { f=F; f_DEC=0; rc_Init(); }
  void FinishEncode( void ) { rc_Quit(); }
  void StartDecode( FILE *F ) { f=F; f_DEC=1; rc_Init(); }
  qword Counter() { return counter; }

  static const uint NUM   = 4;
  static const uint sTOP  = 0x01000000U;
  static const uint gTOP  = 0x00010000U;
  static const uint Thres = 0xFF000000U;

  union {
    struct {
      uint  low;
      uint  Carry;
    };
    qword lowc;
  };
  uint  code;
  uint  FFNum;
  uint  Cache;
  qword range;

  uint muldivR( uint a, uint b ) { return (qword(a)*b)/range; }

  uint mulRdiv( uint a, uint c ) { return (qword(a)*range)/c; }

  void rc_Renorm( void ) {
    while( range<sTOP ) ShiftStuff();
  }

  void rc_Process( uint cumFreq, uint freq, uint totFreq ) {

    uint tmp  = range-mulRdiv( totFreq-cumFreq, totFreq );
    uint rnew = range-mulRdiv( totFreq-(cumFreq+freq), totFreq );

    if( f_DEC ) code-=tmp; else lowc+=tmp;
    range = rnew - tmp;

    rc_Renorm();
  }

  uint rc_GetFreq( uint totFreq ) {
    return muldivR( code, totFreq );
  }

  void ShiftStuff( void ) {
    range = (range<<8)+0x00;
    if( f_DEC==0 ) ShiftLow(); else ShiftCode();
  }

  void ShiftCode( void ) {
    code  = (code<<8)+0x00;
    uint c = get();
    FFNum += (c==-1);
    code += byte(c);
  }

  void ShiftLow( void ) {
    if( (low<Thres) || Carry ) {
      if( Cache!=-1 ) put( Cache+Carry );
      for(;FFNum!=0;FFNum--) put( Carry-1 ); 
      Cache = low>>24;
      Carry = 0;
    } else FFNum++;
    low<<=8;
  }

  void rc_Init( void ) {
    low   = 0;
    FFNum = 0;
    Carry = 0;
    Cache = -1;
    if( f_DEC==1 ) {
      for(int _=0; _<NUM; _++) ShiftCode();
    }
    range = 1ULL<<32;
    counter = 0;
  }

  void rc_Quit( void ) {
    if( f_DEC==0 ) {
      uint i, n = NUM;

      qword llow=low;
      qword high=llow+range;

      if( (llow|      0xFF) < high ) low|=      0xFF,n--;
      if( (llow|    0xFFFF) < high ) low|=    0xFFFF,n--;
      if( (llow|  0xFFFFFF) < high ) low|=  0xFFFFFF,n--;
      if( (llow|0xFFFFFFFF) < high ) low|=0xFFFFFFFF,n--;

      if( (Cache!=-1) && ((n!=0) || (Cache+Carry!=0xFF) ) ) put( Cache+Carry );
      if( (n==0) && (Carry==0) ) FFNum=0; 
      for( i=0; i<FFNum; i++ ) put( 0xFF+Carry );
      for( i=0; i<n; i++ ) put( low>>24 ), low<<=8;
    }
  }

} rc;

#define VERSION_ID                      (0x04)
#define VERSION_DATE                    "2024/07/26"

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

typedef struct stats_s {
  uint orig_current;
  uint orig_total;
  uint next_display;
  uint next_display_step;
  qword rc_previous;
  uint orig_previous;
} stats_t;

#define sizeof_array(A)                 (sizeof(A) / sizeof(A[0]))

void Adam(const int n_var, const float alpha, const float beta1, const float beta1t, const float beta2, const float beta2t, const float eps, float *g, float *t, float *m, float *v) {
#if 1

  const float weight_decay = 0.0f; 
  const int decoupled_weight_decay = 1;
  if (weight_decay != 0.0f) {
    if (decoupled_weight_decay) {
      for(int i = 0; i < n_var; i++)
        t[i] -= alpha * weight_decay * t[i];
    } else {
      for(int i = 0; i < n_var; i++)
        g[i] += weight_decay * t[i];
    }
  }

  if(m != NULL) {
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
#elif 0
  if(m != NULL) {
    for(int i = 0; i < n_var; i++) m[i] *= beta1;
    for(int i = 0; i < n_var; i++) m[i] += (1.0f - beta1) * g[i];
    for(int i = 0; i < n_var; i++) v[i] *= beta2;
    for(int i = 0; i < n_var; i++) v[i] += (1.0f - beta2) * g[i] * g[i];
    for(int i = 0; i < n_var; i++) t[i] -= alpha * (m[i] / (1.0f - beta1t)) / sqrtf(v[i] / (1.0f - beta2t) + eps); 
  } else {
    for(int i = 0; i < n_var; i++) v[i] *= beta2;
    for(int i = 0; i < n_var; i++) v[i] += (1.0f - beta2) * g[i] * g[i];
    for(int i = 0; i < n_var; i++) t[i] -= alpha * g[i] / sqrtf(v[i] / (1.0f - beta2t) + eps); 
  }
#endif
}

#define KANNCOMPR_RNN_VAR_H0                        (0x0001)
#define KANNCOMPR_RNN_NORM                          (0x0002)
#define KANNCOMPR_GRU_MINIMAL_GATED_UNIT            (0x0004)
#define KANNCOMPR_LSTM_INPUT_FORGET_GATE_COUPLED    (0x0008)
#define KANNCOMPR_LSTM_MV_VARIANT1                  (0x1000)
#define KANNCOMPR_LSTM_MV_VARIANT2                  (0x2000)
#define KANNCOMPR_LSTM_MV_VARIANT                   (KANNCOMPR_LSTM_MV_VARIANT1 | KANNCOMPR_LSTM_MV_VARIANT2)

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

kad_node_t *kanncompr_layer_gru(kad_node_t *in, int n1, uint rnn_flag) {
  int n0;
  kad_node_t *t, *r, *z, *w, *u, *b, *s, *h0, *out;
  kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANNCOMPR_RNN_NORM)? kann_cmul_norm : kad_cmul;

  n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
  h0 = (rnn_flag & KANNCOMPR_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  h0->x = (float*)calloc(n1, sizeof(float));

  u = kann_new_leaf(KAD_VAR, 0.0f, 2, n1, n1);
  b = kann_new_leaf(KAD_VAR, 0.0f, 1, n1);
  t = cmul(h0, u);
  w = kann_new_leaf(KAD_VAR, 0.0f, 2, n1, n0);
  t = kad_add(cmul(in, w), t);
  z = kad_sigm(kad_add(t, b));
  if (!(rnn_flag & KANNCOMPR_GRU_MINIMAL_GATED_UNIT)) {
    
    u = kann_new_leaf(KAD_VAR, 0.0f, 2, n1, n1);
    b = kann_new_leaf(KAD_VAR, 0.0f, 1, n1);
    t = cmul(h0, u);
    w = kann_new_leaf(KAD_VAR, 0.0f, 2, n1, n0);
    t = kad_add(cmul(in, w), t);
    r = kad_sigm(kad_add(t, b));
  }
  
  u = kann_new_leaf(KAD_VAR, 0.0f, 2, n1, n1);
  b = kann_new_leaf(KAD_VAR, 0.0f, 1, n1);
  t = cmul(kad_mul(!(rnn_flag & KANNCOMPR_GRU_MINIMAL_GATED_UNIT) ? r : z, h0), u);
  w = kann_new_leaf(KAD_VAR, 0.0f, 2, n1, n0);
  t = kad_add(cmul(in, w), t);
  s = kad_tanh(kad_add(t, b));
  
  out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
  out->pre = h0;
  return out;
}

kad_node_t *kanncompr_layer_lstm(kad_node_t *in, int n1, uint rnn_flag, byte bias_init_type[4], float bias_init_from_to[4][2]) {
  int n0;
  kad_node_t *i, *f, *o, *g, *w, *u, *b, *h0, *c0, *c, *out;
  kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANNCOMPR_RNN_NORM)? kann_cmul_norm : kad_cmul;
  n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
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

kad_node_t *kanncompr_layer_YamRNN(kad_node_t *in, int n1, int variant, uint rnn_flag) {
  int n0;
  kad_node_t *v1, *v2, *w, *u, *b, *h0, *out;
  kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANNCOMPR_RNN_NORM)? kann_cmul_norm : kad_cmul;

  n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
  h0 = (rnn_flag & KANNCOMPR_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
  h0->x = (float*)calloc(n1, sizeof(float));

  w = kann_new_weight(n1, n0);
  u = kann_new_weight(n1, n1);
  b = kann_new_bias(n1);
  v1 = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
  
  w = kann_new_weight(n1, n0);
  b = kann_new_vec(n1, 1.0f);
  v2 = kad_tanh(kad_add(cmul(in, w), b));

  switch(variant) {
  case 1: { 
    if (rnn_flag & KANNCOMPR_RNN_NORM) v2 = kann_layer_layernorm(v2);
    w = kann_new_weight(n1, n0);
    b = kann_new_vec(n1, 1.0f);
    u = kann_new_weight(n1, n1);
    kad_node_t *x = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
    out = kad_sub(kad_square(x), kad_mul(kad_add(kad_square(v2), h0), v1)); 
    break;
  }
  case 2: { 
    if (rnn_flag & KANNCOMPR_RNN_NORM) v2 = kann_layer_layernorm(v2);
    w = kann_new_weight(n1, n0);
    b = kann_new_vec(n1, 1.0f);
    u = kann_new_weight(n1, n1);
    kad_node_t *x = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
    out = kad_tanh(kad_sub(x, kad_mul(kad_add(v2, h0), v1)));
    break;
  }
  case 3: { 
    if (rnn_flag & KANNCOMPR_RNN_NORM) v2 = kann_layer_layernorm(v2);
    kad_node_t *c0 = (rnn_flag & KANNCOMPR_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
    c0->x = (float*)calloc(n1, sizeof(float));
    w = kann_new_weight(n1, n0);
    b = kann_new_vec(n1, 1.0f);
    u = kann_new_weight(n1, n1);
    kad_node_t *c = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(c0, u)), b));
    c->pre = c0;
    if (rnn_flag & KANNCOMPR_RNN_NORM) c = kann_layer_layernorm(c);
    out = kad_tanh(kad_sub(c, kad_mul(kad_add(c, v2), v1)));
    break;
  }
  case 0: 
  default: {
    
    out = kad_sub(kad_square(kad_1minus(v2)), kad_mul(kad_add(kad_square(v2), h0), v1));
    break;
  }
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

int fput_ui08(FILE *file, byte value) {
  return fputc(value, file) == EOF;
}

int fget_ui08(FILE *file, byte *value) {
  int v = fgetc(file);
  if(v == EOF) return 1;
  *value = v;
  return 0;
}

int fput_ui16(FILE *file, word value) {
  return fput_ui08(file, (value >> 0) & 0xff) || fput_ui08(file, (value >> 8) & 0xff);
}

int fget_ui16(FILE *file, word *value) {
  byte v1, v2;
  if(fget_ui08(file, &v1) || fget_ui08(file, &v2)) return 1;
  *value = (word)v1 << 0 | (word)v2 << 8;
  return 0;
}

int fput_ui32(FILE *file, uint value) {
  return fput_ui16(file, (value >> 0) & 0xffff) || fput_ui16(file, (value >> 16) & 0xffff);
}

int fget_ui32(FILE *file, uint *value) {
  word v1, v2;
  if(fget_ui16(file, &v1) || fget_ui16(file, &v2)) return 1;
  *value = (uint)v1 << 0 | (uint)v2 << 16;
  return 0;
}

int fput_si32(FILE *file, long value) {
  return fput_ui16(file, (value >> 0) & 0xffff) || fput_ui16(file, (value >> 16) & 0xffff);
}

int fget_si32(FILE *file, long *value) {
  word v1, v2;
  if(fget_ui16(file, &v1) || fget_ui16(file, &v2)) return 1;
  *value = (long)v1 << 0 | (long)v2 << 16;
  return 0;
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

void display_stats(stats_t *stats, qword comprpos) {
  if(stats->orig_current >= 1 && stats->orig_total >= 1 && comprpos != stats->rc_previous && stats->orig_current != stats->orig_previous)
#define PERC(V, T)                      (100.0 * (float)(V) / (float)(T))
#define BPB(V, T)                       (  8.0 * (float)(V) / (float)(T))
    printf("%7.3f%% %7.3f%%/%6.3f %7.3f%%/%6.3f\r", PERC(stats->orig_current, stats->orig_total), PERC(comprpos, stats->orig_current), BPB(comprpos, stats->orig_current), PERC(comprpos - stats->rc_previous, stats->orig_current - stats->orig_previous), BPB(comprpos - stats->rc_previous, stats->orig_current - stats->orig_previous));
    
#undef BPB
#undef PERC
  stats->rc_previous   = comprpos;
  stats->orig_previous = stats->orig_current;
}

int main(int argc, char** argv) {
  kanncompr_t options;

  if(argc != 4 || (argv[1][0] != 'c' && argv[1][0] != 'd')) {
    printf("kanncompr %s - Mauro Vezzosi - " VERSION_DATE "\n", vers[VERSION_ID]);
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

  FILE *filein = fopen(argv[2], "rb");
  if(filein == NULL) exit(2);
  FILE *fileout = fopen(argv[3], "wb");
  if(fileout == NULL) exit(3);

  long fileoriglen = 0;

  if(argv[1][0] == 'c') {
    fseek(filein, 0, SEEK_END);
    fileoriglen = ftell(filein);
    fseek(filein, 0, SEEK_SET);

    fputc('K', fileout);
    fputc('C', fileout);
    fputc(VERSION_ID, fileout);

    if(fput_si00(fileout, options.rnn_type             ) ||
       fput_si00(fileout, options.n_layers             ) ||
       fput_si00(fileout, options.n_layers_embed_hidden) ||
       fput_si00(fileout, options.n_layers_embed_output) ||
       fput_si00(fileout, options.n_neurons            ) ||
       fput_si00(fileout, options.ulen                 ) ||
       fput_si00(fileout, options.norm                 ) ||
       fput_si00(fileout, options.var_h0               ) ||
       fput_fl  (fileout, options.grad_clip            ) ||
       fput_fl  (fileout, options.dropout              ) ||
       fput_fl  (fileout, options.temper               ) ||
       fput_ui32(fileout, options.seed                 ) ||
       fput_fl  (fileout, options.alpha1               ) ||
       fput_fl  (fileout, options.alpha2               ) ||
       fput_fl  (fileout, options.alpha1d              ) ||
       fput_fl  (fileout, options.beta1                ) ||
       fput_fl  (fileout, options.beta1t               ) ||
       fput_fl  (fileout, options.beta2                ) ||
       fput_fl  (fileout, options.beta2t               ) ||
       fput_fl  (fileout, options.eps                  ) ||
       fput_ui16(fileout, options.mini_batch_freq      ) ||
       fput_ui16(fileout, options.mini_batch_size      ) ||
       fput_ui16(fileout, options.mini_batch_step      ) ||
       fput_si32(fileout, fileoriglen                  ) ||
       fput_ui08(fileout, options.vocab_type           )
      )
      exit(5);
  } else if(argv[1][0] == 'd') {
    if(fgetc(filein) != 'K' || fgetc(filein) != 'C' || fgetc(filein) != VERSION_ID) exit(4);

    if(fget_si00(filein, &options.rnn_type             ) ||
       fget_si00(filein, &options.n_layers             ) ||
       fget_si00(filein, &options.n_layers_embed_hidden) ||
       fget_si00(filein, &options.n_layers_embed_output) ||
       fget_si00(filein, &options.n_neurons            ) ||
       fget_si00(filein, &options.ulen                 ) ||
       fget_si00(filein, &options.norm                 ) ||
       fget_si00(filein, &options.var_h0               ) ||
       fget_fl  (filein, &options.grad_clip            ) ||
       fget_fl  (filein, &options.dropout              ) ||
       fget_fl  (filein, &options.temper               ) ||
       fget_ui32(filein, &options.seed                 ) ||
       fget_fl  (filein, &options.alpha1               ) ||
       fget_fl  (filein, &options.alpha2               ) ||
       fget_fl  (filein, &options.alpha1d              ) ||
       fget_fl  (filein, &options.beta1                ) ||
       fget_fl  (filein, &options.beta1t               ) ||
       fget_fl  (filein, &options.beta2                ) ||
       fget_fl  (filein, &options.beta2t               ) ||
       fget_fl  (filein, &options.eps                  ) ||
       fget_ui16(filein, &options.mini_batch_freq      ) ||
       fget_ui16(filein, &options.mini_batch_size      ) ||
       fget_ui16(filein, &options.mini_batch_step      ) ||
       fget_si32(filein, &fileoriglen                  ) ||
       fget_ui08(filein, &options.vocab_type           )
      )
      exit(5);
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