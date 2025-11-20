Nice — this is a neat online LSTM+PPMD compressor. Below I'll walk through the *math* the code implements, map the equations to the code's variables, and explain the optimization objective and training loop (truncated BPTT + Adam). I'll keep the mapping compact so you can jump back into the source quickly. (All references below come from your uploaded `coder.cpp`.) 

---

# 1) What the model receives and returns

* **Input at time t**: a real vector $x_t\in\mathbb{R}^{d}$ equal to the **PPMD byte-probability distribution** (the code calls `SetInput(p)`, where `p` is the PPMD probs). In the code `d = LSTM_INPUT_SIZE` (128). 
* **Symbol to predict**: the actual next byte $y_t$ (or its index in the vocabulary). The model is trained online to predict the next byte.
* **Output**: a probability vector $\hat{p}_t\in\Delta^{K-1}$ (softmax over `output_size_`, K = number of allowed symbols). This is returned by `Perceive()`/`Predict()` and used by the range coder. 

---

# 2) The LSTM cell equations (as implemented)

The LSTM variant implemented is slightly nonstandard — the input gate is *tied* to the forget gate ($i_t = 1 - f_t$). The code uses three neuron layers per LSTM layer: forget gate, input-node (candidate), and output gate. The layer also applies *layer normalization* to the pre-activations.

Notation:

* $x_t$ — continuous input vector (PPMD probs).
* $h_{t-1}$ — previous hidden vector (concatenated hidden across layers); in code `hidden_`.
* For a given LSTM cell (index $j$) at time t let:

  * pre-activation for gate $u\in\{f, g, o\}$: $a^{(u)}_{t,j}$.
  * normalized pre-activation after layer-norm: $\tilde a^{(u)}_{t,j}$.
  * gate activations: forget $f_{t,j}=\sigma(\tilde a^{(f)}_{t,j})$, candidate $g_{t,j}=\tanh(\tilde a^{(g)}_{t,j})$, output $o_{t,j}=\sigma(\tilde a^{(o)}_{t,j})$.
  * cell state: $s_{t,j}$.
  * hidden output: $h_{t,j} = o_{t,j}\cdot\tanh(s_{t,j})$.

**Pre-activation (linear part)**
The neuron-layer computes, for each cell $j$:
$$
a^{(u)}_{t,j} = b^{(u)}_{j}(x_{\text{symbol}}) + \sum_{k} W^{(u)}_{j,k} \, v_{t,k}
$$
where:

* $b^{(u)}_{j}(x_{symbol})$ corresponds to `neurons.weights_[i][input_symbol]` (an embedding-like indexed bias for the discrete symbol index passed into `ForwardPass`), and
* the sum corresponds to `for (j) f += input[j] * neurons.weights_[i][output_size_ + j]`. Here `input` is the continuous vector (which contains both PPMD input and previous hidden concatenated depending on layer). See code in `ForwardPass(t_NeuronLayer& neurons, const float* input, ...)`. 

**Layer normalization**
The code computes:

1. mean-square: $m_t = \frac{1}{N}\sum_j (a^{(u)}_{t,j})^2$
2. inverse-variance approx: $\mathrm{ivar}_t = \frac{1}{\sqrt{m_t + \varepsilon}}$ where $\varepsilon = 1e{-}5$.
3. normalized vector: $\mathrm{norm}_{t,j} = a^{(u)}_{t,j}\cdot \mathrm{ivar}_t$.
4. affine transform: $\tilde a^{(u)}_{t,j} = \gamma^{(u)}_j\cdot \mathrm{norm}_{t,j} + \beta^{(u)}_j$.

This is implemented by computing `neurons.norm_[epoch_*NUM_CELLS + i]`, `neurons.ivar_[epoch_]`, then `neurons.state_ = neurons.norm_ * gamma + beta`. (So they use a variant of layer norm: normalization by mean square, not mean & variance separately; but functionally similar.) 

**Nonlinearities and cell update**
After normalization:
$$
f_{t,j} = \sigma(\tilde a^{(f)}_{t,j}),\quad
g_{t,j} = \tanh(\tilde a^{(g)}_{t,j}),\quad
o_{t,j} = \sigma(\tilde a^{(o)}_{t,j})
$$
Then the implementation sets:
$$
i_{t,j} = 1 - f_{t,j} \quad(\text{input gate tied to forget})
$$
Cell-state update:
$$
s_{t,j} = f_{t,j} \odot s_{t-1,j} + i_{t,j} \odot g_{t,j}
$$
Hidden:
$$
h_{t,j} = o_{t,j} \odot \tanh(s_{t,j})
$$
All of the above lines correspond to code in `LstmLayer::ForwardPass` (the logistic, tanh, setting `input_gate_state_`, updating `state_`, computing `tanh_state_`, and output `hidden`). 

---

# 3) The output layer and final probability

After all LSTM layers produce hidden units, the model computes logits for each possible next byte class $k$ by a linear mapping from the concatenated hidden vector (plus bias):
$$
z_{t,k} = w_k^\top \, \tilde h_t \quad\text{(where }\tilde h_t = [h_t^{(1)},\ldots,h_t^{(L)}, 1]\text{)}
$$
and then
$$
\hat{p}_{t,k} = \frac{\exp(z_{t,k})}{\sum_{k'} \exp(z_{t,k'})}
$$
Code: `Predict()` computes `sum = Σ hidden_[j]*output_layer_[epoch_][i][j]`, then `output_[epoch_][i] = exp(sum)` and normalizes. `output_layer_` stores the output weights (and there is one copy per epoch index). 

---

# 4) Loss / optimization objective

The optimization objective is **negative log-likelihood (cross-entropy)** for the next-byte prediction. The gradient used in backprop is exactly the softmax cross-entropy gradient:
$$
\frac{\partial \mathcal{L}}{\partial z_{t,k}} = \hat p_{t,k} - \mathbf{1}_{\{k=y_t\}}
$$
This appears in code as:

```cpp
error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
```

— i.e. $p - y$. That is the gradient of cross-entropy w.r.t logits/softmax outputs. 

Because compression uses log-probabilities, minimizing cross-entropy directly means minimizing the expected code length (in nats; dividing by ln2 gives bits). So the LSTM is directly trained to produce better predictive probabilities and thus reduce compressed file size.

---

# 5) Training protocol (online, truncated BPTT)

* **Online learning** while compressing/decompressing: after each symbol is observed/processed the model updates itself (teacher-forcing using the real next byte). 
* **Truncated BPTT horizon**: the LSTM stores `HORIZON` timesteps and performs BPTT across that window. Template param `LSTM_HORIZON` is 73 in the compile-time constants. When `epoch_ == 0` the code runs a backward pass across the horizon: it loops epochs `horizon-1 .. 0` calling `BackwardPass` on each layer and accumulates gradients. See `Perceive()` where the BPTT (if `epoch_ == 0`) is triggered. 
* **Parameter updates**:

  * For the **neuron-layer parameters** (weights, gamma, beta), the code accumulates gradients during BPTT in `neurons.update_`, `neurons.gamma_u_`, `neurons.beta_u_`, and then calls `Adam()` (per-parameter Adam) in `NeuronLayer::BackwardPass` when `epoch == 0` to update `weights_`, `gamma_`, `beta_`. Adam here uses unusual hyperparams: `beta1=0.025`, `beta2=0.9999`, eps = 1e-6; learning rate is `learning_rate_ * 0.1 / sqrt(5e-5*t + 1)` with bias correction. `t` is `update_steps_`. 
  * For the **output-layer weights** (`output_layer_`), the code applies a simple gradient descent step every step: `output_layer_[epoch_][i][j] -= learning_rate_ * error * hidden_[j]` (so output weights are updated online, not via Adam). 
* **Gradient clipping**: gradients on cell-level arrays are clipped to `± gradient_clip_` before being used: `ClipGradients`. Template param `LSTM_GRADIENT_CLIP_X10` is used to set this threshold. 
* **Parameter sharing / coupling**: input gate computed from forget gate ($i_t=1-f_t$) reduces parameters and couples their behavior. That's a deliberate architectural choice in the code. 

---

# 6) Where discrete symbol information enters (the `input_symbol` trick)

* Each neuron-layer's `ForwardPass` uses `neurons.weights_[i][input_symbol]` — this is effectively an embedding/bias lookup for the previous discrete symbol. So the linear pre-activation is a sum of:

  * a learned coefficient tied to the previous discrete byte (via `input_symbol` index), plus
  * a dot-product over the continuous input vector (which contains PPMD probs and previous hidden values as appropriate).
* This is a compact way to combine discrete-symbol information (one-hot-like) with continuous vector inputs. 

---

# 7) Practical consequences for compression

* The LSTM learns (online) conditional distributions $P(y_t \mid x_{t}, x_{t-1}, \ldots)$ where $x_t$ is the PPMD-provided vector plus hidden states — so it's modeling residual structure that PPMD doesn't capture (meta-patterns across the PPMD posterior sequence).
* Loss minimized is cross-entropy, i.e. **expected code length**. Better predictions → smaller range-coded output.
* Because training is online and truncated-BPTT, it adapts to the particular file being compressed during the run (this is typical in compression schemes that train models during coding).

---

# 8) Summary mapping (equations → code identifiers)

* Input vector $x_t$ ← `SetInput(p)` → `layer_input_[epoch_][...][0..INPUT_SIZE-1]`. 
* Pre-activation $a$ ← `f = neurons.weights_[i][input_symbol]; for j: f += input[j] * neurons.weights_[i][output_size_ + j];` (in `ForwardPass`). 
* Layer-normalization: compute `neurons.ivar_[epoch_] = 1.0f / sqrt((sum/num_cells)+1e-5f)` then `neurons.state_ = neurons.norm_*gamma + beta`. 
* Gate nonlinearity: logistic/tanh calls in `LstmLayer::ForwardPass`. 
* Cell update: `state_ = forget*state_ + input_node*input_gate` (input_gate = `1-forget`). 
* Hidden: `hidden[...] = output_gate * tanh_state_`. 
* Output softmax: `output_[i] = exp(sum); normalize`. 
* Loss gradient: `error = p - y` used for backprop. 
* Optimizers: Adam for neuron weights (`Adam()`), plain SGD step for output weights. Gradient clipping applied. 

---

# 9) Short note about hyperparameters

* Truncated BPTT length `HORIZON = 73`.
* Learning rate template `LSTM_LEARNING_RATE_X100000` gives `learning_rate_ = 0.072` here.
* Adam uses `beta1=0.025` (very small momentum), `beta2=0.9999`. Update limit `UPDATE_LIMIT=3000` caps some schedules. All in the compile-time constants in `coder.cpp`. 

---

## Final concise answer

**What does this LSTM do?**
It takes PPMD's continuous byte-probability vector as input, runs it (together with previous hidden states and a discrete-symbol embedding) through stacked LSTM layers that use layer-normalized pre-activations, produces logits which are softmaxed into a predicted distribution $\hat p_t$, and hands $\hat p_t$ to the range coder.

**What is its optimization metric?**
Cross-entropy (negative log-likelihood) between predicted distribution and the true next byte. Minimizing that directly minimizes expected code length, so the LSTM's training objective is exactly aligned with compression performance.

If you want, I can:

* Derive the exact backprop derivatives step-by-step for this implementation (showing how `error` flows into `neurons.update_`), or
* Produce a compact math-only description (no code) of the model and optimizer, or
* Visualize the dataflow (inputs → gates → cell → output) as a diagram.

Which would be most helpful next?
