# LSTM Mathematics in Compression Implementation

## Architecture Overview

The LSTM acts as a **probability distribution mixer/refiner** that combines PPMD's predictions with learned patterns to produce better byte predictions for compression.

## The Math Flow

### 1. Input Stage (line 700: `SetInput`)

```cpp
M.lstm_->SetInput(p);  // p is PPMD's probability distribution
```

The LSTM receives PPMD's 256-element probability distribution as input features. This becomes part of the input vector along with:
- The previous predicted byte (embedded)
- The LSTM's hidden state from previous timestep
- The output from previous prediction

### 2. LSTM Forward Pass (lines 157-177)

The classic LSTM equations are implemented:

**Forget Gate:**
$f_t = σ(W_f · [h_{t-1}, x_t] + b_f)$
Controls what to forget from previous cell state (line 164)

**Input Gate:**
$i_t = 1 - f_t$    (line 169)
$g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)$    (line 165)


**Cell State Update:**
$C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t$    (lines 171-172)

**Output Gate:**
$o_t = σ(W_o · [h_{t-1}, x_t] + b_o)$    (line 166)
$h_t = o_t ⊙ tanh(C_t)$    (line 174)

Where:
- σ is the logistic sigmoid function: `1 / (1 + exp(-x))`
- ⊙ denotes element-wise multiplication
- $W_f$, $W_g$, $W_o$ are weight matrices
- $h_{t-1}$ is the previous hidden state
- $x_t$ is the current input (PPMD probabilities + context)

### 3. Output Layer (lines 466-473: `Perceive`)

The hidden states go through a **softmax layer** to produce probabilities:

```cpp
for (i = 0; i < output_size_; ++i) {
  sum = 0;
  for (j = 0; j < NUM_CELLS * NUM_LAYERS + 1; ++j) 
    sum += hidden_[j] * output_layer_[epoch_][i][j];
  output_[epoch_][i] = exp(sum);
}
// Normalize to probabilities
sum = 0;
for (i = 0; i < output_size_; ++i) sum += output_[i];
for (i = 0; i < output_size_; ++i) 
  output_[i] /= sum;
```

This produces: **P(byte_i | history, PPMD_probs)**

Mathematically:
$logit_i = Σ_j (h_j · W_{out}[i,j])$
$P(byte_i) = exp(logit_i) / Σ_k exp(logit_k)$

### 4. Optimization Metric (lines 424-465)

The loss function is **cross-entropy** between the predicted distribution and the actual byte:

```cpp
// Line 427-438: Compute cross-entropy loss
for (i = 0; i < output_size_; ++i) {
  if (input == (int)i) {
    error = 1.0f / output_[epoch][i];  // Target is 1.0 for correct byte
  } else {
    error = 0;  // Target is 0 for wrong bytes
  }
  output_layer_error_[epoch][i] = error;
}
```

**Loss Function:** $L = -log(P(correct_byte))$

This is equivalent to minimizing: $L = -Σ_i y_i · log(p_i)$
where $y_i$ is the one-hot encoded target (1 for correct byte, 0 otherwise).

**Gradient Computation:**

For the correct byte (lines 444-445):
```cpp
error = (1.0f - output_[i]) * output_[i];
```
This gives: $∂L/∂logit_{correct} = p_{correct} - 1$

For incorrect bytes (lines 446-447):
```cpp
error = -output_[j] * output_[i];
```
This gives: $∂L/∂logit_{wrong} = p_{wrong}$

### 5. Backpropagation Through Time (lines 179-216)

Gradients flow backward through the LSTM gates using the chain rule:

Output Gate Error (line 189): $∂L/∂o_t = tanh(C_t) · ∂L/∂h_t · o_t(1 - o_t)$

Cell State Error (line 192): $∂L/∂C_t = ∂L/∂h_t · o_t · (1 - tanh²(C_t))$

Input Node Error (line 195): $∂L/∂g_t = ∂L/∂C_t · i_t · (1 - g_t²)$

Forget Gate Error (line 198): $∂L/∂f_t = (C_{t-1} - g_t) · ∂L/∂C_t · f_t · i_t$

Gradient Propagation to Previous Timestep (line 203): $∂L/∂C_{t-1} = ∂L/∂C_t · f_t$

This implements **truncated backpropagation through time (BPTT)** with horizon length of 73 timesteps.

### 6. Weight Updates (lines 226-246: Adam Optimizer)

Uses the **Adam optimizer** with adaptive learning rates:

$m_t = β₁ · m_{t-1} + (1 - β₁) · g_t$
$v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²$

$m̂_t = m_t / (1 - β₁^t)$
$v̂_t = v_t / (1 - β₂^t)$

$W_t = W_{t-1} - α · m̂_t / (√v̂_t + ε)$

**Parameters:**
- β₁ = 0.025 (first moment decay, unusually low)
- β₂ = 0.9999 (second moment decay)
- ε = 1e-6 (numerical stability)
- Base learning rate = 0.072

**Learning Rate Decay (lines 230-234):**
```cpp
if (t < UPDATE_LIMIT) {
  alpha = learning_rate * 0.1 / sqrt(5e-5 * t + 1.0);
} else {
  alpha = learning_rate * 0.1 / sqrt(5e-5 * UPDATE_LIMIT + 1.0);
}
```

This gives a decreasing learning rate schedule that stabilizes after UPDATE_LIMIT steps (3000).

### 7. Gradient Clipping (lines 388-393)

To prevent exploding gradients:

```cpp
void ClipGradients(float* error) {
  for (i = 0; i < num_cells_; ++i) {
    if (error[i] < -gradient_clip_) error[i] = -gradient_clip_;
    if (error[i] > gradient_clip_) error[i] = gradient_clip_;
  }
}
```

With `gradient_clip_ = 2.0`, this constrains gradient magnitudes to [-2.0, 2.0].

## What the LSTM Actually Does

The LSTM learns to:

### 1. Pattern Recognition
Identify byte sequences PPMD misses, particularly:
- Long-range dependencies beyond PPMD's context order
- Non-Markovian patterns that depend on complex history
- Structural patterns (e.g., file format headers, repeated sections)

### 2. Distribution Mixing
Weight PPMD's predictions based on learned context:
- When PPMD is confident and historically correct → trust it more
- When PPMD is uncertain → rely more on learned patterns  
- When context suggests atypical patterns → override PPMD's prediction

### 3. Adaptive Modeling
The network learns which contexts benefit from which information source, effectively creating a meta-model that decides when to trust statistical predictions vs. neural patterns.

## Compression Benefit

By producing $P_{LSTM}(byte | PPMD_{distribution}, history)$ instead of just $P_{PPMD}(byte)$, the range coder receives a more accurate probability distribution.

**Information-Theoretic View:**

The expected code length for a byte is: $E[L] = -Σ_i P_{true}(i) · log₂(P_{model}(i))$

The LSTM minimizes cross-entropy: $H(P_{true}, P_{model}) = -Σ_i P_{true}(i) · log(P_{model}(i))$

Minimizing cross-entropy is equivalent to minimizing expected code length in bits (with natural log converted to log₂).

**Practical Benefits:**
- Fewer bits for predictable bytes (high P_model for correct byte)
- Better modeling of data patterns PPMD struggles with
- Adaptive mixing reduces worst-case performance degradation

## Key Design Choices

### Model Architecture
- **Input size:** 128 (reduced from 256-byte alphabet)
- **Hidden cells:** 90 per layer
- **Layers:** 2
- **Horizon:** 73 timesteps for BPTT

### Training Strategy
- **Online learning:** Updates after every byte
- **Warm-up period:** First 3000 bytes get special treatment (UPDATE_LIMIT)
- **No mini-batching:** Sequential, single-sample updates
- **Stateful:** LSTM state persists across the entire file

### Optimization Details
- Unusually low β₁ = 0.025 (typical is 0.9) for faster adaptation
- High β₂ = 0.9999 for stable second moment estimates
- Aggressive learning rate decay for stability after warm-up

## The Complete Pipeline

```
Input Byte → PPMD → P_PPMD(byte_next)
                        ↓
              LSTM.SetInput(P_PPMD)
                        ↓
              LSTM.Forward() → hidden_state
                        ↓
              Softmax(hidden_state) → P_LSTM(byte_next)
                        ↓
              RangeCoder.Encode(actual_byte, P_LSTM)
                        ↓
              Compute Loss: -log(P_LSTM(actual_byte))
                        ↓
              LSTM.Backward() → weight updates
                        ↓
              LSTM.Update() → Adam step
```

This creates a **neural adaptive arithmetic coder** where the LSTM learns to predict better probability distributions for the range coder, resulting in improved compression ratios compared to PPMD alone.
