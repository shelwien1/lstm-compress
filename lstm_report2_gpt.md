
# Compression Improvement Methods for the LSTM+PPMD Coder  
**(No change to number of LSTM layers)**  
**Generated:** 2025-11-20

## Overview

I read the uploaded `coder.cpp` (CPU-only hybrid LSTM + PPMD online-learning compressor) and the experiment report `lstm_report1.md`. Below is a focused, practical list of ways to improve compression (and runtime/robustness) **without increasing the number of LSTM layers**. For each idea I include motivation, estimated cost/complexity, and a concrete action you can try in `coder.cpp`.

---

## 1) Better numerical stability & softmax correctness (low effort, high reliability)

**Why:** The current softmax computation does `output_[epoch_][i] = exp(sum); ... normalize`. This is numerically unstable and can produce poor gradients and overflow for large sums.

**Actionable fixes**
- Use `max-sum` trick:
```cpp
// compute sums
float maxv = -INFINITY;
for (i=0; i<output_size_; ++i) maxv = max(maxv, sum_i[i]);
for (i=0; i<output_size_; ++i) output_[epoch_][i] = exp(sum_i[i] - maxv);
float Z = 0;
for (i=0; i<output_size_; ++i) Z += output_[epoch_][i];
for (i=0; i<output_size_; ++i) output_[epoch_][i] /= Z;
```
- Use `log-sum-exp` when computing losses / updates.
**Expected impact:** Removes numeric noise, slightly better predictions and more stable online updates.

---

## 2) Improved initialization and weight regularization (low effort, medium impact)

**Why:** Initialization affects early online learning heavily for small files.

**Actionable fixes**
- Replace `Rand()` uniform init with Xavier/Glorot or orthogonal init for recurrent weights:
```cpp
float val = sqrt(6.0f / float(INPUT_SIZE + output_size_));
weights_[i][j] = low + Rand()*range; // current
// suggest using Glorot or: orthogonal for recurrent matrices
```
- Add light L2 weight decay during Adam updates (small λ like 1e-6–1e-4).
**Expected impact:** Faster and steadier convergence, fewer noisy early updates.

---

## 3) Smarter optimizer schedule & warmup (medium effort, high ROI)

**Why:** Online learning on small files is sensitive to the initial learning steps.

**Actionable fixes**
- Add a warmup period where learning rate grows from small → target over first ~100–1000 updates.
- Use cosine annealing or linear decay after warmup.
- Make Adam hyperparameters configurable and consider switching to Adafactor / Adamp (memory-friendly) if needed.
**Concrete:** change Adam alpha schedule and add warmup multiplier dependent on `update_steps_`.
**Expected impact:** Avoids catastrophic early updates; consistent with your successful LR tuning experiments.

---

## 4) Per-parameter / per-layer adaptive learning rates (medium effort, medium-high impact)

**Why:** Different parameter groups (gate biases, recurrent weights, output layer) should have different learning speeds.

**Actionable fixes**
- Use separate learning rates for:
  - output_layer (often benefits from smaller LR)
  - recurrent weights
  - normalization gains (`gamma_`) and biases (`beta_`) (often smaller LR)
- Implement simple scaling factors applied inside `Adam()` call.
**Expected impact:** Better fine-tuning of output probabilities and reduced oscillation.

---

## 5) Gradient handling improvements (low effort)

**Why:** Clipping scheme matters. Current code clips per-element to `gradient_clip_`. Norm clipping or RMS-based clipping can be better.

**Actionable fixes**
- Replace element-wise clipping by global-norm clipping:
```cpp
float norm = sqrt(sum_i(arr[i]*arr[i]));
if (norm > clip) { float s = clip / norm; for(i) arr[i]*=s; }
```
- Alternatively use scaled clipping for gates vs cell states.
**Expected impact:** Better gradient flow, fewer truncated updates.

---

## 6) More robust output modeling (medium effort, high impact)

**Why:** Output softmax inputs come from `output_layer_` which is updated online per-step. Better output parametrizations can improve probability mass allocation.

**Actionable fixes**
- **Bias correction:** add small bias terms for frequent bytes.
- **Temperature scaling:** allow a small temperature parameter `τ` to be learned or tuned; using `exp(sum/τ)` can make distributions sharper/softer.
- **Adaptive softmax / hierarchical softmax:** if `total` active symbols vary widely, a hierarchical scheme reduces variance of gradient updates for rare symbols (more complex).
**Expected impact:** Lower cross-entropy per byte → direct compression gains.

---

## 7) Better blending of PPMD and LSTM predictions (high impact)

**Why:** PPMD provides strong local context predictions; LSTM captures meta-patterns. Currently the code uses LSTM output (M.probs_) as final prediction — but blending adaptively often performs better.

**Actionable fixes**
- Compute a weighted mixture between `ppmd` probabilities and LSTM output:
  - static weight `α` or adaptive weight based on recent local accuracy.
- Implementation idea: maintain short rolling count of recent correct predictions for each model and compute `α = sigmoid(k*(ppmd_score - lstm_score))`.
- Simpler: use a logistic mixer (single-layer) that takes features (ppmd top-prob, entropy of ppmd, entropy of LSTM, recent error) and outputs mixing weights.
**Expected impact:** Usually substantial improvement because PPMD and LSTM have complementary strengths.

---

## 8) Feature engineering & input augmentation (high impact)

**Why:** The LSTM sees only raw `probs` from PPMD. Adding richer inputs can help it learn better corrections.

**Actionable fixes**
- Add scalar features to `SetInput()` such as:
  - current PPMD entropy (−Σ p log p)
  - top-k PPMD probability and index
  - run-lengths / repetition counters
  - byte type flags (printable ASCII, whitespace, zero byte, high bit set)
  - position-in-file normalized (pos / file_len) or local byte offset mod N
- Concatenate these scalars (normalized) to the existing PPMD-prob vector before feeding the LSTM.
**Expected impact:** Higher-quality conditional predictions → fewer bits.

---

## 9) Preprocessing transforms (very high impact on many file types)

**Why:** Transforming input to a more predictable domain is often the single best lever for compression.

**Actionable fixes**
- Apply **BWT + MTF + RLE** for text-like files prior to modeling; feed transformed stream to the same pipeline.
- For executables/binaries use delta filters, x86-specific filters, deduplication windows.
- Detect file type heuristically and apply a small pipeline of filters (text vs binary vs image-like).
**Expected impact:** Huge for files with localized structure—may beat any model tuning.

---

## 10) Model sparsity, pruning & low-rank factorization (medium effort, medium impact)

**Why:** Large dense matrices may overfit and waste capacity and runtime.

**Actionable fixes**
- Train with L1 or structured-sparsity regularizer, then prune small weights and fine-tune online.
- Use low-rank factorization: represent weights as `U*V` with smaller rank `r` (reduces params and improves generalization).
**Expected impact:** Reduced overfitting, smaller memory footprint, sometimes better generalization.

---

## 11) Caching & computation optimizations (low effort, engineering gains)

**Why:** Expensive ops (`exp`, loops over 256 symbols) are repeated each byte.

**Actionable fixes**
- Cache `exp(sum)` partial results or reuse last-hidden->output matrix-vector products when hidden is unchanged.
- Precompute `hidden_[j] * output_layer_[i][j]` inner-products with SIMD; reuse across repeated inputs.
- Use approximations for `exp()` (fast math) or `fast_softmax` while validating impact on accuracy.
**Expected impact:** Lower CPU time, allowing more complex dynamics or larger horizons.

---

## 12) Training data strategies (medium effort, high impact)

**Why:** Single-file online training is limited; transfer learning or meta-training helps.

**Actionable fixes**
- Pretrain LSTM on a corpus (similar files) offline, save weights, then fine-tune online per-file.
- Use meta-learning (MAML-like) to find initialization that adapts quickly to small files.
- Keep a small on-disk cache of previously seen files with learned priors.
**Expected impact:** Often large: pretraining drastically reduces online updates needed and improves final compression.

---

## 13) Horizon & BPTT tuning (low effort)

**Why:** The truncated BPTT horizon determines how many steps of feedback occur.

**Actionable fixes**
- Tune `LSTM_HORIZON` (73 currently): test smaller/larger horizons; sometimes shorter horizon reduces noisy gradients while larger may capture longer dependencies.
- Consider variable horizon depending on file type.
**Expected impact:** Moderate; depends on file statistics.

---

## 14) Output-layer improvements: sampled / importance updates (advanced)

**Why:** When output vocabulary (active symbols) is large, full softmax updates create noisy gradients.

**Actionable fixes**
- Use sampled-softmax or importance sampling for large `total` to reduce variance.
- Use adaptive softmax or class-based softmax for bytes grouped by frequency.
**Expected impact:** Reduces variance of updates and can improve rare-symbol modeling.

---

## 15) Evaluation, automated search & reproducibility (essential)

**Actionable steps**
- Automate ablation experiments (single-variable changes).
- Track per-byte bits, moving average cross-entropy, and per-symbol perplexity.
- Use Bayesian or evolutionary hyperparameter search across LR, beta1, clip, horizon, temperature, mixing weight.
**Expected impact:** More efficient discovery of robust improvements.

---

## Prioritized short checklist (what to try first)

1. Fix softmax numerics (max-subtract) and stable log-sum-exp. (very low effort)  
2. Add warmup and small LR schedule changes (low effort; high payoff).  
3. Implement adaptive mixing of PPMD and LSTM predictions. (medium effort; high payoff)  
4. Add small feature scalars (entropy, top-k). (medium effort)  
5. Experiment with preprocessing transforms (BWT+MTF) on text-like files. (variable effort; potentially large payoff)  
6. Consider pretraining on a corpus if you have relevant data. (higher cost; big payoff)

---

## Concrete code pointers in `coder.cpp`

- Improve softmax in `Lstm::Predict()` where `output_[epoch_][i] = exp(sum);` and normalization follows.  
- Make Adam `beta1`, `beta2`, `learning_rate` external and configurable at runtime for fast experiments (they're currently compile-time constants).  
- Add small `ppmd_entropy` calculation after `ppmd_PrepareByte()` and pass it into `SetInput()`.  
- Replace element-wise clipping `ClipGradients()` with optional norm clipping.  
- Implement mixing inside the main loop where `M.Update(c)` and frequency `freq[i] = M.probs_[i]*SCALE;` — compute `ppmd_probs` and `lstm_probs` then `mixed = α*ppmd + (1-α)*lstm`.

---

## Final notes

- Many of the improvements above are orthogonal — combine the low-effort stability & optimizer fixes first, then move to mixing, features and transforms.  
- For small files (like `build.sh`) gentle optimization of learning dynamics and mixing PPMD/LSTM tends to yield the best returns. For larger files, pretraining and architectural changes (other than layer-count) become more important.  
- If you want, I can produce a patch for `coder.cpp` implementing the softmax fix, warmup schedule and a simple static PPMD/LSTM mixture as a first pull request.

---

**End of document**
