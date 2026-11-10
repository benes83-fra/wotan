# The `tensor` package: a user guide

`wotan/tensor` is Wotan's automatic-differentiation engine. A `Tensor` wraps a `linalg` matrix. Operations on tensors record a computation graph, and `tensor_backward` walks that graph in reverse to compute gradients. The package provides:
- element-wise arithmetic and activations;
- matrix products, reshaping, slicing and concatenation;
- losses, including finance-specific ones (negative Sharpe ratio, prospect-theory utility);
- layer, RMS and batch normalisation, and dropout;
- 1-D and 2-D convolution and pooling;
- RNN, GRU and LSTM cells, embeddings, attention (plain, masked, tiled "flash", causal) and a selective state-space scan;
- graph scopes: one arena per graph, so a training loop allocates no new memory after its first iteration.

Everything works on `f64`. The `nn` package builds its layers, optimisers and models on top of this one. This guide covers the tensor level: what you need to write your own models, losses or layers.

The guide assumes you know basic Odin and the `Matrix` type from the [linalg user guide](linalg_guide.md).

Every example was compiled and run against the current code, and the output shown is the real output. The numbers agree with PyTorch 2.1.0 in double precision. This includes gradients and a 500-step training run, which matches to all 10 printed digits. The examples free everything they allocate: a debug build with a tracking allocator reports no leaks.

## Contents

1. [Setup](#1-setup)
2. [The Tensor type](#2-the-tensor-type)
3. [A first gradient](#3-a-first-gradient)
4. [Element-wise operations](#4-element-wise-operations)
5. [Matrix products and shape operations](#5-matrix-products-and-shape-operations)
6. [Reductions and losses](#6-reductions-and-losses)
7. [Normalisation and dropout](#7-normalisation-and-dropout)
8. [Convolution and pooling](#8-convolution-and-pooling)
9. [Sequence models: recurrent cells, embeddings, attention](#9-sequence-models-recurrent-cells-embeddings-attention)
10. [The backward pass in detail](#10-the-backward-pass-in-detail)
11. [Memory management and graph scopes](#11-memory-management-and-graph-scopes)
12. [A complete example](#12-a-complete-example)
13. [Quick reference](#13-quick-reference)

## 1. Setup

Import the package under a short name. This guide uses `t`, as the tests and the `nn` package do. You will usually need `linalg` as well, to build the matrices that tensors wrap:

```odin
import l "wotan/linalg"
import t "wotan/tensor"
import "core:fmt"
```

Inside the Wotan repository the path is relative, for example `import t "../wotan/tensor"` from `tests/`.

Most examples use this small helper to make a tensor from literal values:

```odin
from_values :: proc(vals: []f64, rows, cols: int, requires_grad := false) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols)
	copy(m.data, vals)
	return t.tensor_new(m, requires_grad)
}
```

## 2. The Tensor type

A tensor is a heap-allocated struct, always handled through a pointer `^Tensor`. These are the fields you use:

| Field | Meaning |
|---|---|
| `data` | the values: an `l.Matrix(f64)`, row-major |
| `grad` | the gradient, same size as `data`. It is allocated for trainable leaves at creation, and for every other node during `tensor_backward`. |
| `requires_grad` | whether gradients flow to this tensor |
| `shape` | the logical shape `[4]int`, see below |
| `op`, `inputs` | the operation that produced the tensor and its inputs. A leaf has `op == .None`. |
| `allocator` | the allocator that owns this tensor's memory |
| `owned_by_graph` | lets `tensor_free_graph` free a leaf too (section 11) |

`tensor_new(matrix, requires_grad = false, allocator = context.allocator)` wraps a matrix you allocated. The tensor takes ownership of the matrix, and `tensor_free` frees both. With `requires_grad = true`, a zeroed gradient buffer is allocated at the same time. Tensors you create this way are **leaves**: parameters, inputs and targets.

```odin
x := from_values({1, 2, 3, 4, 5, 6}, 2, 3)
defer t.tensor_free(x)
fmt.println(x.data.rows, x.data.cols, x.shape, x.requires_grad, x.op)
```
```
2 3 [2, 3, 1, 1] false None
```

### Logical shape and storage

The data is always one row-major matrix. Higher-dimensional tensors keep their dimensions in `shape`, and most of them are stored as a single row:

| What | `shape` | `data` |
|---|---|---|
| matrix | `[rows, cols, 1, 1]` | `rows × cols` |
| image batch | `[N, C, H, W]` | `1 × N·C·H·W`, element (n, c, h, w) at `((n·C + c)·H + h)·W + w` |
| sequence batch | `[batch, seq_len, features, 1]` | `1 × batch·seq_len·features` |

`tensor_new_4d(N, C, H, W, requires_grad, allocator)` makes a zero image tensor. For a sequence, make the flat matrix and set `shape` yourself:

```odin
img := t.tensor_new_4d(2, 3, 4, 4) // N, C, H, W
defer t.tensor_free(img)
fmt.println(img.data.rows, img.data.cols, img.shape)

seq := from_values({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 12)
defer t.tensor_free(seq)
seq.shape = {2, 3, 2, 1} // batch 2, 3 time steps, 2 features
```
```
1 96 [2, 3, 4, 4]
```

Every op checks that the shapes it reads describe their data. It panics with a clear message if `shape` and `data` disagree, so a wrong shape cannot silently corrupt a result. There is **no broadcasting**: `tensor_add`, `tensor_sub`, `tensor_mul` and `tensor_div` need two tensors of the same size. The broadcasting cases have their own ops: `tensor_add_bias` adds a row vector to every row, and `tensor_gate_mul` scales every row by its own number.

## 3. A first gradient

Ops return new tensors and record how they were made. `tensor_backward(loss)` then fills `grad` for every tensor in the graph that requires gradients:

```odin
x := from_values({1, 2, 3}, 1, 3, true)
w := from_values({0.5, -1, 2}, 1, 3, true)
defer {t.tensor_free(x); t.tensor_free(w)}

loss := t.tensor_sum(t.tensor_tanh(t.tensor_mul(x, w))) // Σ tanh(x·w)
fmt.printf("loss %.6f op %v\n", loss.data.data[0], loss.op)
t.tensor_backward(loss)
fmt.printf("dloss/dx %.6f %.6f %.6f\n", x.grad.data[0], x.grad.data[1], x.grad.data[2])
fmt.printf("dloss/dw %.6f %.6f %.6f\n", w.grad.data[0], w.grad.data[1], w.grad.data[2])
t.tensor_free_graph(loss) // frees the op nodes, keeps x and w
```
```
loss 0.498077 op Sum
dloss/dx 0.393224 -0.070651 0.000049
dloss/dw 0.786448 0.141302 0.000074
```

Three rules, all as in PyTorch:

- **An op output requires gradients if any of its inputs does.** Ops whose inputs are all constants record no graph.
- **Gradients accumulate.** Each backward pass adds to `grad`. Clear the gradients with `tensor_zero_grad` before the next step:

  ```odin
  // a second forward and backward pass without zeroing
  fmt.printf("after 2nd backward dloss/dx[0] %.6f\n", x.grad.data[0])
  t.tensor_zero_grad(x); t.tensor_zero_grad(w)
  ```
  ```
  after 2nd backward dloss/dx[0] 0.786448
  ```

- **The graph is freed by you.** Use `tensor_free_graph(root)` without a scope, or `graph_end` with one (section 11).

## 4. Element-wise operations

All of these return a tensor of the input's shape and are differentiable:

| Op | Value | Notes |
|---|---|---|
| `tensor_add(a, b)`, `tensor_sub`, `tensor_mul`, `tensor_div` | a + b, a − b, a·b, a / b | same size, no broadcasting |
| `tensor_neg(a)`, `tensor_scale(a, s)` | −a, s·a | |
| `tensor_exp`, `tensor_log`, `tensor_sqrt` | | `log` and `sqrt` need positive inputs for a finite gradient |
| `tensor_lgamma(a)` | log Γ(a) | gradient is the digamma function, also exported as `digamma(x)` |
| `tensor_relu`, `tensor_leaky_relu(a, alpha = 0.01)` | | |
| `tensor_sigmoid`, `tensor_tanh`, `tensor_softplus` | | `softplus` is log(1 + eˣ), computed stably |
| `tensor_gelu(a)` | x·Φ(x) | exact (erf-based) GELU, as `torch.nn.functional.gelu` |
| `tensor_norm_cdf(a)` | Φ(a) | standard normal CDF, accurate in the tails |
| `tensor_clamp(a, lo, hi)` | | gradient 0 outside [lo, hi] |
| `tensor_prospect_value(x, alpha = 0.88, beta = 0.88, lambda = 2.25)` | xᵅ for x ≥ 0, −λ(−x)ᵝ for x < 0 | Tversky–Kahneman (1992) value function; subgradient 0 at x = 0 |
| `tensor_prob_weighting(p, gamma = 0.61)` | pᵞ / (pᵞ + (1−p)ᵞ)^(1/γ) | probability weighting; p is clamped to [1e-7, 1 − 1e-7] |
| `tensor_add_bias(a, bias)` | a + bias in every row | `bias` is `[1, D]`; `a` is `[N, D]` or a flattened sequence of D-wide rows |
| `tensor_gate_mul(x, gate)` | x[n, d]·gate[n] | per-row gate, as in mixture-of-experts routing |

## 5. Matrix products and shape operations

`tensor_matmul(a, b)` is the ordinary product of an `[N, in]` and an `[in, out]` matrix. With `tensor_add_bias` it gives a linear layer:

```odin
X := from_values({1, 2, 3, 4, 5, 6}, 2, 3)
W := from_values({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 3, 2, true)
b := from_values({1, -1}, 1, 2, true)
y := t.tensor_add_bias(t.tensor_matmul(X, W), b)
fmt.println("linear", y.data.data, y.shape)
```
```
linear [3.2, 1.7999999999999998, 5.9, 5.3999999999999995] [2, 2, 1, 1]
```

`tensor_matmul` also accepts a flattened tensor as its left operand: a `1 × N·in` row whose length is a multiple of `b.rows`. It multiplies each `in`-wide row by `b` and keeps the result flattened. The feature dimension of `shape` is replaced by `out`, so a sequence `[batch, seq, in, 1]` becomes `[batch, seq, out, 1]`. This is how the `nn` layers apply one weight matrix to every time step:

```odin
S := from_values({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 12)
S.shape = {2, 2, 3, 1} // batch 2, seq 2, 3 features
z := t.tensor_add_bias(t.tensor_matmul(S, W), b)
fmt.println("seq linear", z.data.rows, z.data.cols, z.shape)
```
```
seq linear 1 8 [2, 2, 2, 1]
```

The matrix products use the multi-threaded SIMD kernels of `linalg`. Their results do not depend on the number of threads.

### Shape operations

All of these copy the data and are differentiable:

```odin
A := from_values({1, 2, 3, 4, 5, 6}, 2, 3, true)
tr := t.tensor_transpose(A)             // [3, 2]
rs := t.tensor_reshape(A, {3, 2, 1, 1}) // new shape, same flat order
sc := t.tensor_slice_cols(A, 1, 2)      // columns 1 ..< 3
cc := t.tensor_concat({A, A}, 1)        // along dim 1
cr := t.tensor_concat({A, A}, 0)        // along dim 0
```
```
transpose [1, 4, 2, 5, 3, 6] [3, 2, 1, 1]
reshape 1 6 [3, 2, 1, 1]
slice_cols [2, 3, 5, 6]
concat dim1 [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6] [2, 6, 1, 1]
concat dim0 [4, 3, 1, 1]
```

- `tensor_reshape` stores its result as a single row (`1 × total`), like every non-matrix tensor. Only `shape` changes.
- `tensor_concat` works along any of the four logical dimensions, and every other dimension must agree.
- `tensor_flatten(x)` turns `[N, C, H, W]` into an `[N, C·H·W]` matrix with N real rows, ready for a dense layer.
- Two specialised permutations exist:
  - `tensor_permute_mha(x, batch, seq_len, num_heads, head_dim)` regroups `[batch, seq, heads·head_dim]` into `[batch·heads, seq, head_dim]` for multi-head attention. `tensor_permute_mha_inverse` undoes it.
  - `tensor_permute_lob(x, batch, c, t, l, allocator)` turns `[B, C, T, L]` into `[B, T, C·L]` (used by DeepLOB).
- `tensor_slice_gate(gates, e)` is `tensor_slice_cols(gates, e, 1)`.

## 6. Reductions and losses

| Op | Output | Value |
|---|---|---|
| `tensor_sum(a)`, `tensor_mean(a)` | scalar | over all elements |
| `tensor_sum_dim1(a)` | `[N, 1]` | row sums of an `[N, C]` tensor |
| `tensor_logsumexp_dim1(q, num_actions)` | `[N, 1]` | stable log Σ exp over each row |
| `tensor_softmax(logits)` | same shape | softmax of each row of `data` |
| `tensor_mse_loss(pred, target)` | scalar | mean (pred − target)²; gradients flow to both |
| `tensor_cross_entropy_loss(logits, targets)` | scalar | mean −log softmax(logits)[target]; `targets` is `[]int` |
| `tensor_bce_loss(p, y)`, `tensor_binary_cross_entropy(p, y)` | scalar | mean binary cross-entropy of probabilities p (not logits), p clamped to [1e-7, 1 − 1e-7] |
| `tensor_kl_divergence(mu, log_var)` | scalar | −½ · mean(1 + log σ² − μ² − σ²), the VAE term against N(0, 1) |
| `tensor_entropy(p)` | scalar | −Σ p ln p over the entries above 1e-10 |
| `tensor_sharpe_loss(r, risk_free_rate = 0)` | scalar | −(mean(r) − r_f) / √(var(r) + 1e-8), population variance |
| `tensor_prospect_utility(r, alpha, beta, lambda)` | scalar | −mean of the prospect value of r |

```odin
logits := from_values({0.5, -1.0, 2.0, 0.1, 0.3, -0.7}, 2, 3, true)
ce := t.tensor_cross_entropy_loss(logits, {2, 0})
t.tensor_backward(ce)
```
```
softmax row 0: 0.1753 0.0391 0.7856
cross entropy 0.611832  grad row 0: 0.087645 0.019556 -0.107201
logsumexp 2.241311 1.082352 [2, 1, 1, 1]
sum_dim1 [1.5, -0.29999999999999993]
bce 0.279777
mse 0.166667
sharpe loss -0.348450
prospect utility loss -0.000018
```

Check these points before you use a loss:

- **Softmax and cross-entropy use different rows.** `tensor_softmax` normalises each row of the `data` matrix. A flattened `1 × N·C` sequence is a single row, so its softmax runs over all N·C values. Apply it to an `[N, C]` matrix, for example the output of a 2-D `tensor_matmul`. `tensor_cross_entropy_loss` instead also accepts the flattened form: given a `1 × N·C` row and N targets, it splits the row into N rows of C.
- **The BCE losses take probabilities.** Apply `tensor_sigmoid` to logits first. The two BCE ops compute the same value; both are kept for existing callers.
- **The KL term is a mean over elements, not a sum.** Multiply by the number of latent dimensions if you follow a paper that sums.

## 7. Normalisation and dropout

```odin
x := from_values({0.5, -1.0, 2.0, 0.1, 0.3, -0.7}, 2, 3, true)
g := from_values({1, 1, 1}, 1, 3, true); b := from_values({0, 0, 0}, 1, 3, true)
ln := t.tensor_layer_norm(x, g, b)  // eps = 1e-5
rn := t.tensor_rms_norm(x, g)       // eps = 1e-6
```
```
layer_norm row 0: 0.000000 -1.224741 1.224741
rms_norm row 0: 0.377964 -0.755929 1.511857
```

- **`tensor_layer_norm(x, gamma, beta, eps = 1e-5)`** normalises every D-wide row, where D is the width of `gamma` and `beta` (`[1, D]`). `x` is `[N, D]` or a flattened sequence.
- **`tensor_rms_norm(x, gamma, eps = 1e-6)`** works on the same rows: x / √(mean x² + ε) · γ, with no centring and no shift.
- **`tensor_normalize_time(x, eps = 1e-8)`** z-scores a `[N, C, T, L]` tensor along T, separately for each (n, c, l).
- **`tensor_batch_norm_2d(x, weight, bias, running_mean, running_var, training, momentum = 0.1, eps = 1e-5)`** normalises each channel of an `[N, C, H, W]` tensor. It follows `torch.nn.BatchNorm2d`:
  - In training it uses the batch statistics. It updates the running mean and the running variance in place, and the running variance uses the unbiased estimate.
  - In evaluation it uses the running statistics.

```odin
bx := t.tensor_new_4d(2, 2, 1, 2); copy(bx.data.data, []f64{1, 2, 3, 4, 5, 6, 7, 9})
gamma := from_values({1, 2}, 1, 2, true); beta := from_values({0, 1}, 1, 2, true)
rm := from_values({0, 0}, 1, 2); rv := from_values({1, 1}, 1, 2)
bn := t.tensor_batch_norm_2d(bx, gamma, beta, rm, rv, training = true)
```
```
batch_norm channel 0: -1.212677 -0.727606 0.727606 1.212677
running mean 0.3500 0.5750  running var 1.466667 1.658333
```

`tensor_dropout(x, p, training)` is inverted dropout. In training it zeroes each element with probability p and scales the others by 1/(1 − p). Otherwise, or when p ≤ 0, it **returns `x` itself** rather than a copy, so do not free the result separately from `x`.

```
dropout: every value is 0 or 2x the input: true  eval returns the input: true
```

## 8. Convolution and pooling

`tensor_conv2d(input, weight, bias = nil, stride = 1, pad = 0)` takes the PyTorch layouts. `input` is `[N, C_in, H, W]`, `weight` is `[C_out, C_in, kH, kW]` and `bias` is `[1, C_out]` or nil. The output is `[N, C_out, H_out, W_out]` with H_out = (H + 2·pad − kH) / stride + 1. It runs as im2col followed by a SIMD matrix product.

```odin
img := t.tensor_new_4d(1, 1, 4, 4, requires_grad = true)
for i in 0 ..< 16 {img.data.data[i] = f64(i) * 0.1 - 0.5}
k := t.tensor_new_4d(2, 1, 3, 3, requires_grad = true) // C_out, C_in, kH, kW
for i in 0 ..< 18 {k.data.data[i] = 0.1 * f64(i % 7) - 0.2}
kb := from_values({0.05, -0.05}, 1, 2, true)

cv := t.tensor_conv2d(img, k, kb, stride = 2, pad = 1)
mp := t.tensor_max_pool2d(cv, 2, 2, 1) // kH, kW, stride
loss := t.tensor_sum(mp)
t.tensor_backward(loss)
```
```
conv2d [1, 2, 2, 2] [-0.15000000000000002, -0.15000000000000002, 0.01999999999999995, 0.4000000000000001, -0.17000000000000004, -0.22999999999999998, 0.06000000000000005, 0.22000000000000014]
max_pool2d [1, 2, 1, 1] [0.4000000000000001, 0.22000000000000014]
dloss/dbias 1.0 1.0
```

`tensor_max_pool2d(x, kH, kW, stride)` and `tensor_avg_pool2d` use no padding.

`tensor_conv1d(x, weight, bias, dilation = 1, pad_left = 0, pad_right = 0)` is a 1-D convolution of a **channels-last** sequence `[batch, T, C_in]`, as the sequence layers use them. The weight is in PyTorch layout `[C_out, C_in, K]` and the bias is `[1, C_out]` or nil. The output is `[batch, T_out, C_out]` with T_out = T + pad_left + pad_right − dilation·(K − 1). For a causal convolution, set `pad_left = dilation·(K − 1)` and leave `pad_right = 0`:

```odin
x1 := from_values({0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8}, 1, 8); x1.shape = {1, 4, 2, 1}
w1 := from_values({-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 1, 12); w1.shape = {3, 2, 2, 1}
b1 := from_values({0.1, 0, -0.1}, 1, 3)
c1 := t.tensor_conv1d(x1, w1, b1, dilation = 2, pad_left = 2) // causal
```
```
conv1d [1, 4, 3, 1] t=0: 0.02 0.04 0.06  t=3: -0.31 0.23 0.77
```

## 9. Sequence models: recurrent cells, embeddings, attention

### RNN, GRU and LSTM

`tensor_rnn`, `tensor_gru` and `tensor_lstm` run a cell over a whole sequence `x` of shape `[batch, seq_len, input]`. They return every hidden state as `[batch, seq_len, H, 1]`.

The weights are stored **transposed** compared with PyTorch, and there is **one** bias. G is the number of gates: 1 for the RNN, 3 for the GRU and 4 for the LSTM.

| Argument | Shape | PyTorch equivalent |
|---|---|---|
| `W_ih` | `[input, G·H]` | `weight_ih_l0ᵀ` |
| `W_hh` | `[H, G·H]` | `weight_hh_l0ᵀ` |
| `bias` | `[1, G·H]` | `bias_ih_l0 + bias_hh_l0` (see the GRU note) |
| `h_0`, `c_0` | `[batch, H]` | `h_0[0]`, `c_0[0]` |

The cells are defined as follows:
- **RNN:** h = tanh(x W_ih + h W_hh + b).
- **LSTM:** gates are in the order i, f, g, o, as in PyTorch. c = f·c + i·g and h = o·tanh(c).
- **GRU:** gates are in the order r, z, n, and the new state is h = (1 − z)·n + z·h. The single bias sits outside the reset gate in n = tanh(x W_in + b_n + r·(h W_hn)). This equals PyTorch's GRU when `bias_hh_l0`'s n part is zero.

```odin
B, S, I, H := 1, 2, 2, 2
x := from_values({0.5, -0.2, 0.1, 0.4}, 1, B * S * I); x.shape = {B, S, I, 1}
h0 := from_values({0, 0}, B, H); c0 := from_values({0, 0}, B, H)
wih := l.matrix_new(f64, I, 4 * H); whh := l.matrix_new(f64, H, 4 * H); bm := l.matrix_new(f64, 1, 4 * H)
for i in 0 ..< 16 {wih.data[i] = 0.05 * f64(i) - 0.3; whh.data[i] = 0.3 - 0.04 * f64(i)}
for i in 0 ..< 8 {bm.data[i] = 0.01 * f64(i)}
W_ih := t.tensor_new(wih, true); W_hh := t.tensor_new(whh, true); bias := t.tensor_new(bm, true)
out := t.tensor_lstm(x, h0, c0, W_ih, W_hh, bias)
```
```
lstm [1, 2, 2, 1]: -0.0158246750 -0.0104533186 0.0320958293 0.0465395806
```

`torch.nn.LSTM` with these weights gives the same four numbers.

The input projection x·W_ih is computed for all time steps in one threaded product before the time loop. With gradients, the forward pass keeps every state and gate for the backward pass, so memory grows linearly with `seq_len`.

### Embeddings

`tensor_embedding(ids, weight)` looks up rows of `weight` (`[vocab, dim]`). The token ids are stored as `f64` in a tensor shaped `[batch, seq_len, 1, 1]`. The output is `[batch, seq_len, dim, 1]`. Gradients flow only to `weight`, and an id outside the vocabulary panics.

```odin
ids := from_values({2, 0, 1, 2}, 1, 4); ids.shape = {2, 2, 1, 1}
E := from_values({0, 0, 1, 1, 2, 2}, 3, 2, true)
emb := t.tensor_embedding(ids, E)
```
```
embedding [2, 2, 2, 1] [2, 2, 0, 0, 1, 1, 2, 2]
```

### Attention

All attention ops take Q `[batch, seq_q, d_k]`, K `[batch, seq_k, d_k]` and V `[batch, seq_k, d_v]`, and return `[batch, seq_q, d_v]` = softmax(QKᵀ/√d_k + mask)·V. For several heads, permute to `[batch·heads, seq, head_dim]` first with `tensor_permute_mha`.

| Op | Mask | Memory |
|---|---|---|
| `tensor_scaled_dot_product_attention(Q, K, V)` | none | one seq_q × seq_k block per sequence |
| `tensor_masked_scaled_dot_product_attention(Q, K, V, mask)` | additive: `[seq_q·seq_k]` shared, or `[batch·seq_q·seq_k]`; `nil` for none | as above |
| `tensor_flash_attention(Q, K, V, causal = false)` | optional causal | tiled with online softmax: O(seq) extra memory |

In an additive mask, a hidden position gets a large negative number such as −1e9; the `nn` layers use −1e9. The backward passes recompute the attention weights instead of storing them.

```odin
q := from_values({1, 0, 0, 1, 1, 1}, 1, 6); q.shape = {1, 3, 2, 1}
att := t.tensor_scaled_dot_product_attention(q, q, q)
fa := t.tensor_flash_attention(q, q, q, causal = true)
causal := []f64{0, -1e9, -1e9, 0, 0, -1e9, 0, 0, 0}
ma := t.tensor_masked_scaled_dot_product_attention(q, q, q, causal)
```
```
sdpa row 0: 0.802224 0.598888
causal row 1 flash: 0.330238 0.669762  masked: 0.330238 0.669762
```

These are the values of `torch.nn.functional.scaled_dot_product_attention`, with and without `is_causal`.

### Selective state-space scan

`tensor_ssm(x, h_0, A, B, C, Delta, D)` is the Mamba-style selective scan. Its arguments are:

| Argument | Shape |
|---|---|
| `x` | `[batch, seq, d_model]` |
| `h_0` | `[batch, d_model, d_state]` |
| `A` | `[d_model, d_state]`, normally negative |
| `B`, `C` | `[batch, seq, d_state]` |
| `Delta` | `[batch, seq, d_model]`, positive |
| `D` | `[1, d_model]` |

For each step it computes h ← exp(Δ·A)·h + Δ·B·x and y = C·h + D·x. Batch rows are scanned in parallel threads, and the results are identical to a single-threaded run. All seven inputs are differentiable.

## 10. The backward pass in detail

- **`tensor_backward(root)`:**
  - A scalar root (a loss) is seeded with dL/dL = 1.
  - A non-scalar root is seeded with ones, unless you have already written a non-zero gradient into `root.grad`.
- **`tensor_backward_with_grad(root, grad_output)`** seeds a non-scalar root with an explicit upstream gradient. It computes a vector-Jacobian product, like PyTorch's `y.backward(v)`:

  ```odin
  x := from_values({1, 2, 3}, 1, 3, true)
  y := t.tensor_scale(t.tensor_exp(x), 2) // y = 2·eˣ
  t.tensor_backward_with_grad(y, {1, 0, -1})
  ```
  ```
  vector-Jacobian product: 5.436564 0.000000 -40.171074
  ```

- **Op outputs get their gradient buffers lazily**, during `tensor_backward`. To read or seed an op output's gradient before that, call `tensor_ensure_grad(t)`.
- **`tensor_detach(t)`** returns a copy of `t` without history. It is a new leaf with `requires_grad = false` and the same allocator as `t`, and it is yours to free. Use it to treat a value as a constant:

  ```odin
  d := t.tensor_detach(x)
  defer t.tensor_free(d)
  z := t.tensor_sum(t.tensor_mul(x, d)) // d(Σ x·d)/dx = d
  ```
  ```
  d(sum x*detach(x))/dx = [1, 2, 3]
  ```

- **Non-differentiable helpers:**
  - `tensor_top_k_mask(probs, k)` returns an `[N, E]` tensor with a 1 at the k largest entries of each row. It is a constant marked `owned_by_graph`.
  - `tensor_clip_weights(t, lo, hi)` clamps a tensor's values in place without recording anything, for example for WGAN weight clipping.
- **Graph walks use an explicit stack.** Unrolled recurrent nets with graphs thousands of nodes deep do not overflow the call stack.
- **Debugging aids:** `tensor_validate(t, "where")` and `tensor_validate_graph(root)` report empty data and trainable leaves without a gradient buffer.

The package's own gradient checks compare every op's backward pass with central finite differences: [`tests/gradcheck_ops_test.odin`](../tests/gradcheck_ops_test.odin), [`gradcheck_layers_test.odin`](../tests/gradcheck_layers_test.odin) and [`gradcheck_test.odin`](../tests/gradcheck_test.odin).

## 11. Memory management and graph scopes

Every op allocates its output. You free the graph in one of two ways.

### Without a scope: `tensor_free_graph`

`tensor_free_graph(root)` frees `root` and every op node it can reach through `inputs`. It also frees leaves marked `owned_by_graph`, which is the way to attach constants made inside a loss function. It **keeps** the other leaves: your parameters, inputs and targets. Free those with `tensor_free` when you are done.

```odin
loss := t.tensor_mse_loss(model(X), Y)
t.tensor_backward(loss)
t.tensor_free_graph(loss)
```

Four pitfalls:

- **Constant-only op chains are not linked.** An op records its inputs only when its output requires gradients. If no input of a chain requires gradients, `tensor_free_graph` finds nothing to free and the intermediate outputs leak. Free each of them with `tensor_free`, or run the computation in a no-grad scope (below).
- **Do not free a shared node twice.** Calling `tensor_free_graph` on two roots that share op nodes frees the shared nodes twice. Combine the losses into one root with `tensor_add` first, or use a scope.
- **`tensor_dropout` in evaluation mode returns its input**, not a copy (section 7).
- **`tensor_free(t)` frees exactly one tensor:** its data, its gradient and the struct.

### Graph scopes

A scope gives each graph one arena. While a scope is active on the current thread, every op output and every backward scratch buffer is allocated in it. `graph_end` releases all of it at once, and the arena keeps its memory for the next graph, so a training loop allocates new pages only in its first iteration. The training loop looks like this:

```odin
scope: t.Graph_Scope
defer t.graph_scope_destroy(&scope)
for _ in 0 ..< 100 {
	t.tensor_zero_grad(W)
	t.graph_begin(&scope)
	loss := t.tensor_mse_loss(t.tensor_tanh(t.tensor_matmul(X, W)), Y)
	t.tensor_backward(loss)
	last = loss.data.data[0]        // copy out what must outlive the graph
	used, _ = t.graph_scope_bytes(&scope)
	t.graph_end(&scope)             // every node of this graph is gone
	for i in 0 ..< 6 {W.data.data[i] -= 0.05 * W.grad.data[i]} // W and W.grad live on
}
```
```
loss after 100 steps 0.380933, one graph used 2176 bytes, 64 MB reserved
```

The rules:

- **Leaves keep their own allocator.** Tensors made with `tensor_new`, the parameters, inputs and targets, are not moved into the scope; only op outputs are. The same holds for `tensor_detach` of a leaf.
- **To put a constant in the scope**, pass `t.graph_allocator(&scope)` to `matrix_new` and `tensor_new`.
- **Nothing made in the scope survives `graph_end`.** Do not use or free those nodes afterwards. Copy out the numbers you need first. Inside the scope, `tensor_free` and `tensor_free_graph` on its nodes are allowed and cost nothing.
- **Scopes nest.** `graph_end` releases only the innermost scope.
- **A scope belongs to one thread.** The active scope is thread-local.
- **Arena memory comes in 64 MB chunks** (`GRAPH_CHUNK_SIZE`), and `graph_scope_bytes(&scope)` reports the bytes used and reserved. `graph_scope_destroy` returns the memory to the operating system.

**No-grad scopes** are for inference. With `graph_begin(&scope, no_grad = true)`, op outputs neither require gradients nor keep their inputs, so no graph is recorded and `graph_end` frees everything:

```odin
t.graph_begin(&scope, no_grad = true)
p := t.tensor_tanh(t.tensor_matmul(X, W)) // W requires grad, p does not
fmt.printf("no_grad: requires_grad %v, op %v, inputs %d\n", p.requires_grad, p.op, len(p.inputs))
t.graph_end(&scope)
```
```
no_grad: requires_grad false, op None, inputs 0
```

`grad_enabled()` tells you whether ops currently record a graph. Gradient recording follows the innermost scope, so a scope begun with `no_grad = false` inside a no-grad scope turns recording back on.

### Counters

`global_tensors_created` and `global_tensors_freed` count tensors. `graph_end` counts the nodes it releases as freed, so in a program that frees everything the two numbers are equal. All the examples in this guide together create and free 3408 tensors.

### Fixed while writing this guide

- **`tensor_top_k_mask` chose the wrong entries.** It started from columns 0 ..< k and could select one column twice. For the rows [5, 1, 3] and [0.2, 0.9, 0.4] with k = 2, it returned {0} and {0, 1}; `torch.topk` gives {0, 2} and {1, 2}. It also ignored the active graph scope, so its mask leaked on the heap after `graph_end`. Both are fixed, and k outside 1 ..= E now fails an assertion.
- **`tensor_masked_scaled_dot_product_attention` crashed on a `nil` mask.** It indexed the empty mask; the backward pass already handled `nil`. A `nil` mask now means no mask.
- **Some ops recorded no graph when only certain inputs were trainable.** This happened in `tensor_batch_norm_2d` and `tensor_conv2d` when only the bias required gradients, and in `tensor_ssm` when only `h_0` did. The backward passes already computed those gradients; they were just never reached.

[`tests/tensor_fixes_test.odin`](../tests/tensor_fixes_test.odin) guards against all of these.

## 12. A complete example

This program trains a two-layer perceptron, y = tanh(x W₁ + b₁) W₂ + b₂, to learn f(a, b) = a·b + a/2 from 16 points. It uses plain gradient descent with every step in a graph scope, then predicts in a no-grad scope:

```odin
package main

import "base:runtime"
import "core:fmt"
import "core:mem"
import l "wotan/linalg"
import t "wotan/tensor"

main :: proc() {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, runtime.default_allocator())
	context.allocator = mem.tracking_allocator(&track)

	N :: 16
	X := l.matrix_new(f64, N, 2)
	Y := l.matrix_new(f64, N, 1)
	for i in 0 ..< N {
		a, b := f64(i) / f64(N - 1) * 2 - 1, f64((i * 7) % N) / f64(N - 1) * 2 - 1
		X.data[i * 2], X.data[i * 2 + 1] = a, b
		Y.data[i] = a * b + 0.5 * a
	}
	x := t.tensor_new(X)
	y := t.tensor_new(Y)

	// Parameters: deterministic initial values, zero biases
	W1 := l.matrix_new(f64, 2, 8); b1 := l.matrix_new(f64, 1, 8)
	W2 := l.matrix_new(f64, 8, 1); b2 := l.matrix_new(f64, 1, 1)
	for i in 0 ..< 16 {W1.data[i] = 0.3 * f64((i * 5) % 7 - 3) / 3}
	for i in 0 ..< 8 {W2.data[i] = 0.2 * f64((i * 3) % 5 - 2)}
	params := [4]^t.Tensor {
		t.tensor_new(W1, true), t.tensor_new(b1, true),
		t.tensor_new(W2, true), t.tensor_new(b2, true),
	}

	scope: t.Graph_Scope
	lr :: 0.2
	for step in 0 ..= 500 {
		for p in params {t.tensor_zero_grad(p)}
		t.graph_begin(&scope)
		h := t.tensor_tanh(t.tensor_add_bias(t.tensor_matmul(x, params[0]), params[1]))
		pred := t.tensor_add_bias(t.tensor_matmul(h, params[2]), params[3])
		loss := t.tensor_mse_loss(pred, y)
		t.tensor_backward(loss)
		if step % 100 == 0 {fmt.printf("step %3d  loss %.10f\n", step, loss.data.data[0])}
		t.graph_end(&scope)
		for p in params {
			for i in 0 ..< len(p.data.data) {p.data.data[i] -= lr * p.grad.data[i]}
		}
	}

	// Inference: nothing is recorded, graph_end frees it all
	q := l.matrix_new(f64, 1, 2)
	q.data[0], q.data[1] = 0.5, -0.5
	qt := t.tensor_new(q)
	t.graph_begin(&scope, no_grad = true)
	h := t.tensor_tanh(t.tensor_add_bias(t.tensor_matmul(qt, params[0]), params[1]))
	out := t.tensor_add_bias(t.tensor_matmul(h, params[2]), params[3])
	fmt.printf("prediction at (0.5, -0.5): %.10f (target %.2f)\n", out.data.data[0], 0.5 * -0.5 + 0.25)
	t.graph_end(&scope)

	t.graph_scope_destroy(&scope)
	t.tensor_free(qt)
	for p in params {t.tensor_free(p)}
	t.tensor_free(x)
	t.tensor_free(y)
	fmt.printf("live allocations: %d\n", len(track.allocation_map))
}
```

Output:

```
step   0  loss 0.1792402504
step 100  loss 0.1190616253
step 200  loss 0.0625315064
step 300  loss 0.0359512992
step 400  loss 0.0160555535
step 500  loss 0.0040087520
prediction at (0.5, -0.5): -0.0399466587 (target 0.00)
live allocations: 0
```

The same network in PyTorch 2.1.0 (float64, the same initial weights, plain SGD) prints the same losses and the same prediction, to all ten digits. For real models, the `nn` package wraps these steps in layers and the Adam and SGD optimisers.

## 13. Quick reference

| Task | Procedure |
|---|---|
| New tensor / free | `tensor_new(matrix, requires_grad)`, `tensor_new_4d(N, C, H, W)`, `tensor_free` |
| Gradients | `tensor_backward(root)`, `tensor_backward_with_grad(root, g)`, `tensor_zero_grad`, `tensor_ensure_grad`, `tensor_detach` |
| Free a graph | `tensor_free_graph(root)`; `owned_by_graph = true` for constants |
| Graph scopes | `graph_begin(&s, no_grad)`, `graph_end`, `graph_scope_destroy`, `graph_allocator`, `graph_scope_bytes`, `grad_enabled` |
| Arithmetic | `tensor_add`, `_sub`, `_mul`, `_div`, `_neg`, `_scale`, `_add_bias`, `_gate_mul` |
| Element-wise math | `tensor_exp`, `_log`, `_sqrt`, `_lgamma`, `_clamp`, `_norm_cdf` |
| Activations | `tensor_relu`, `_leaky_relu`, `_sigmoid`, `_tanh`, `_softplus`, `_gelu`, `_softmax` |
| Products and shapes | `tensor_matmul`, `_transpose`, `_reshape`, `_flatten`, `_concat`, `_slice_cols`, `_slice_gate`, `_permute_mha`, `_permute_mha_inverse`, `_permute_lob` |
| Reductions | `tensor_sum`, `_mean`, `_sum_dim1`, `_logsumexp_dim1` |
| Losses | `tensor_mse_loss`, `_cross_entropy_loss`, `_bce_loss`, `_binary_cross_entropy`, `_kl_divergence`, `_entropy`, `_sharpe_loss`, `_prospect_utility` |
| Behavioural finance | `tensor_prospect_value`, `_prob_weighting`, `_prospect_utility` |
| Normalisation | `tensor_layer_norm`, `_rms_norm`, `_batch_norm_2d`, `_normalize_time`, `_dropout` |
| Convolution | `tensor_conv2d`, `_conv1d`, `_max_pool2d`, `_avg_pool2d` |
| Sequences | `tensor_rnn`, `_gru`, `_lstm`, `_embedding`, `_ssm` |
| Attention | `tensor_scaled_dot_product_attention`, `_masked_scaled_dot_product_attention`, `_flash_attention` |
| Constants and checks | `tensor_top_k_mask`, `tensor_clip_weights`, `tensor_validate`, `tensor_validate_graph` |
