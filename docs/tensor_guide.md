# The `tensor` package: a user guide

`wotan/tensor` is Wotan's automatic-differentiation engine. A `Tensor` wraps a `linalg` matrix. Operations on tensors record a computation graph, and `tensor_backward` walks that graph in reverse to compute gradients. The package provides:
- element-wise arithmetic and activations;
- matrix products, reshaping, slicing and concatenation;
- losses, including finance-specific ones (negative Sharpe ratio, prospect-theory utility);
- layer, RMS and batch normalisation, and dropout;
- 1-D and 2-D convolution and pooling;
- RNN, GRU and LSTM cells, embeddings, attention (plain, masked, tiled "flash", causal) and a selective state-space scan;
- graph scopes: one arena per graph, so a training loop allocates no new memory after its first iteration.

Everything works on `f64`. An opt-in GPU backend runs the same operations on a graphics card; see the [gpu user guide](gpu_guide.md). The `nn` package builds its layers, optimisers and models on top of this one (see the [nn user guide](nn_guide.md)). This guide covers the tensor level: what you need to write your own models, losses or layers.

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

  `tensor_drop_grad(t)` is the alternative that PyTorch's `zero_grad()` uses by default (`grad = None`): the tensor then has no gradient (`tensor_has_grad(t)` is false) until the next backward pass reaches it. The buffer is kept and zero-filled by that pass, so dropping allocates nothing. The nn optimisers' `zero_grad` procedures drop.

- **The graph is freed by you.** Use `tensor_free_graph(root)` without a scope, or `graph_end` with one (section 11).

## 4. Element-wise operations

All of these return a tensor of the input's shape and are differentiable:

| Op | Value | Notes |
|---|---|---|
| `tensor_add(a, b)`, `tensor_sub`, `tensor_mul`, `tensor_div` | a + b, a − b, a·b, a / b | same size, no broadcasting |
| `tensor_neg(a)`, `tensor_scale(a, s)` | −a, s·a | |
| `tensor_exp`, `tensor_log`, `tensor_sqrt` | | `log` and `sqrt` need positive inputs for a finite gradient |
| `tensor_lgamma(a)` | log Γ(a) | gradient is the digamma function, also exported as `digamma(x)` |
| `tensor_abs(a)` | \|a\| | gradient sign(a), 0 at a = 0 (as torch) |
| `tensor_sin(a)`, `tensor_cos(a)` | sin a, cos a | |
| `tensor_sign(a)` | −1, 0 or 1 | gradient 0 everywhere (as `torch.sign`) |
| `tensor_relu`, `tensor_leaky_relu(a, alpha = 0.01)` | | |
| `tensor_sigmoid`, `tensor_tanh`, `tensor_softplus` | | `softplus` is log(1 + eˣ), computed stably |
| `tensor_gelu(a)` | x·Φ(x) | exact (erf-based) GELU, as `torch.nn.functional.gelu` |
| `tensor_gelu_tanh(a)` | ½x(1 + tanh(√(2/π)(x + 0.044715x³))) | tanh approximation, as `gelu(approximate="tanh")` and GPT-2's `gelu_new` |
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

`tensor_linear(x, W, b)` computes the same in one op. Its values and gradients are exactly those of the two ops above, but the graph records one node instead of two, and on the GPU it runs as one kernel. The `nn` layers use it. With `b = nil` it is `tensor_matmul`.

```odin
y2 := t.tensor_linear(X, W, b)
fmt.println("tensor_linear", y2.data.data, y2.shape)
```
```
tensor_linear [3.2, 1.7999999999999998, 5.9, 5.3999999999999995] [2, 2, 1, 1]
```

`tensor_matmul` (and so `tensor_linear`) also accepts a flattened tensor as its left operand: a `1 × N·in` row whose length is a multiple of `b.rows`. It multiplies each `in`-wide row by `b` and keeps the result flattened. The feature dimension of `shape` is replaced by `out`, so a sequence `[batch, seq, in, 1]` becomes `[batch, seq, out, 1]`. This is how the `nn` layers apply one weight matrix to every time step:

```odin
S := from_values({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 12)
S.shape = {2, 2, 3, 1} // batch 2, seq 2, 3 features
z := t.tensor_add_bias(t.tensor_matmul(S, W), b)
fmt.println("seq linear", z.data.rows, z.data.cols, z.shape)
```
```
seq linear 1 8 [2, 2, 2, 1]
```

The matrix products use the multi-threaded SIMD kernels of `linalg`. Large element-wise ops and the costly ones (`tensor_norm_cdf`, and so GELU, and `tensor_gelu_tanh`) are split across threads as well. Attention runs its sequences (and heads) on parallel threads. None of this changes results: they are bit-identical to a single-threaded run.

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

### Time steps of sequences

These ops select, cut and pad the time steps of channels-last sequences `[batch, T, C]` (a signal `[batch, T]` counts as C = 1). They are all differentiable:

- `tensor_index_select_time(x, indices, per_sequence = false)` gathers the steps `indices` (any order, repeats allowed) into `[batch, len(indices), C]`. An index of −1 gives a step of zeros. With `per_sequence = true`, `indices` holds one row of T_out indices per sequence. The gradient of a step is the sum over all the output steps that read it.
- `tensor_slice_time(x, start, length)` keeps the steps `start ..< start + length`, like `torch.narrow`.
- `tensor_pad_time(x, left, right, mode = .Zero)` adds `left` steps before and `right` after. The `Time_Pad` modes match `torch.nn.functional.pad`:
  - `.Zero` adds zeros;
  - `.Reflect` mirrors the sequence without repeating the edge step, so `left` and `right` must be below T;
  - `.Replicate` repeats the edge step.
- `tensor_length_regulate(x, durations, max_len = -1)` repeats every step by its duration (T durations per sequence, zero drops the step), like FastSpeech's length regulator or `torch.repeat_interleave` per sequence. The sequences are padded with zero steps to the longest one, or cut or padded to `max_len`. It also returns each sequence's length, ready for `sequence_mask`. `length_regulator_indices` returns the indices it uses.
- To join sequences along time, use `tensor_concat({a, b}, 1)`.

```odin
// one sequence of 4 steps with 2 channels: step s is (s + 1, 10·(s + 1))
x := from_values({1, 10, 2, 20, 3, 30, 4, 40}, 1, 8, true); x.shape = {1, 4, 2, 1}
g := t.tensor_index_select_time(x, {3, 3, -1, 0})
s := t.tensor_slice_time(x, 1, 2)
p := t.tensor_pad_time(x, 2, 1, .Reflect)
r, lengths := t.tensor_length_regulate(x, {2, 0, 1, 3})
t.tensor_backward(t.tensor_sum(r))
```
```
index_select [1, 4, 2, 1] [4, 40, 4, 40, 0, 0, 1, 10]
slice [1, 2, 2, 1] [2, 20, 3, 30]
pad reflect [1, 7, 2, 1] [3, 30, 2, 20, 1, 10, 2, 20, 3, 30, 4, 40, 3, 30]
length_regulate [1, 6, 2, 1] [1, 10, 1, 10, 3, 30, 4, 40, 4, 40, 4, 40] lengths [6]
d sum / dx [2, 2, 0, 0, 1, 1, 3, 3]
```

The gradient of each step is its duration, because every copy of it adds 1 to the sum.

## 6. Reductions and losses

| Op | Output | Value |
|---|---|---|
| `tensor_sum(a)`, `tensor_mean(a)` | scalar | over all elements |
| `tensor_sum_dim1(a)` | `[N, 1]` | row sums of an `[N, C]` tensor |
| `tensor_logsumexp_dim1(q, num_actions)` | `[N, 1]` | stable log Σ exp over each row |
| `tensor_softmax(logits)` | same shape | softmax over the last dimension of `shape`, as `torch.softmax(x, dim = -1)` |
| `tensor_log_softmax(logits)` | same shape | log softmax over the same dimension, computed stably as x − max − log Σ exp(x − max), as `torch.log_softmax(x, dim = -1)` |
| `tensor_mse_loss(pred, target)` | scalar | mean (pred − target)²; gradients flow to both |
| `tensor_l1_loss(pred, target)` | scalar | mean \|pred − target\|, as `torch.nn.functional.l1_loss`; gradients flow to both, 0 where pred = target |
| `tensor_masked_sum(x, mask)` | scalar | Σ x[i]·mask[i / inner], inner = numel(x) / len(mask): each mask value weights a contiguous block of x |
| `tensor_masked_mean(x, mask)` | scalar | `tensor_masked_sum` / (inner · Σ mask): the mean over the valid elements of a padded batch |
| `tensor_cross_entropy_loss(logits, targets)` | scalar | mean −log softmax(logits)[target]; `targets` is `[]int` |
| `tensor_bce_loss(p, y)`, `tensor_binary_cross_entropy(p, y)` | scalar | mean binary cross-entropy of probabilities p (not logits), p clamped to [1e-7, 1 − 1e-7] |
| `tensor_kl_divergence(mu, log_var)` | scalar | −½ · Σ (1 + log σ² − μ² − σ²) over the latent dimension, averaged over the batch: the VAE term against N(0, 1) |
| `tensor_entropy(p)` | scalar | −Σ p ln p over the entries above 1e-10 |
| `tensor_sharpe_loss(r, risk_free_rate = 0)` | scalar | −(mean(r) − r_f) / √(var(r) + 1e-8), population variance |
| `tensor_prospect_utility(r, alpha, beta, lambda)` | scalar | −mean of the prospect value of r |
| `tensor_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = .Mean, zero_infinity = false)` | scalar, or `[batch]` with `.None` | connectionist temporal classification loss, as `torch.nn.functional.ctc_loss` (see below) |

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

- **Softmax works on the last dimension of `shape`, not on the rows of `data`.** Trailing 1s in `shape` are unused dimensions, so the last dimension is C of `[N, C, 1, 1]`, F of a sequence `[B, S, F, 1]` and W of `[N, C, H, W]`. A flattened `1 × N·C` tensor with shape `[N, C, 1, 1]` therefore gets N softmaxes of C values each, like an `[N, C]` matrix. A sequence with a single feature, `[B, S, 1, 1]`, reads as `[B, S]` and is normalised over S. `tensor_cross_entropy_loss` also accepts the flattened form: given a `1 × N·C` row and N targets, it splits the row into N rows of C.
- **The BCE losses take probabilities.** Apply `tensor_sigmoid` to logits first. The two BCE ops compute the same value; both are kept for existing callers.
- **Padded batches need a masked loss.** Sequences of different lengths are padded to a common length T, and the padded steps must not count. `sequence_mask(lengths, T)` builds the `[B, T]` 0/1 mask (`[]f64`, 1 for t < lengths[b]). With a channels-last `[B, T, C]` tensor each mask value then covers the C values of one step, so `tensor_masked_mean(x, mask)` is `(x * mask.unsqueeze(-1)).sum() / (mask.sum() * C)` in torch. The mask is a constant and its length must divide the number of elements; `tensor_masked_mean` panics if the mask sums to 0. A masked L1 loss is `tensor_masked_mean(tensor_abs(tensor_sub(pred, target)), mask)`:

  ```odin
  // two sequences padded to T = 3, C = 2 features; the second has 2 valid steps
  pred := from_values({1, 2, 3, 4, 5, 6, 1, 1, 2, 2, 9, 9}, 1, 12, true)
  pred.shape = {2, 3, 2, 1}
  target := from_values({1, 1, 3, 3, 5, 5, 0, 0, 2, 2, 0, 0}, 1, 12)
  target.shape = {2, 3, 2, 1}
  mask := t.sequence_mask({3, 2}, 3)
  loss := t.tensor_masked_mean(t.tensor_abs(t.tensor_sub(pred, target)), mask)
  t.tensor_backward(loss)
  ```
  ```
  masked L1 0.500 (padding ignored), grad of the padded step: [0, 0]
  ```

- **The CTC loss** scores label sequences that are shorter than the input and not aligned with it. Examples are the characters of a recording, or the text tokens a spectrogram must pass through in order. The loss of a sample is −log of the summed probability of all its alignments: every step emits a class, and the sequence collapses to the labels after merging repeated classes and dropping blanks.
  - `log_probs` is `[batch, T, C]`, channels-last, usually `tensor_log_softmax` of the logits. torch's layout is `[T, batch, C]`.
  - `targets` holds the labels of all samples one after the other, `target_lengths[b]` of them for sample b, as torch's concatenated form. Labels are in [0, C) and never `blank`.
  - Sample b uses only its first `input_lengths[b]` steps.
  - `reduction = .Mean` divides each sample's loss by its target length (at least 1) and averages over the batch, as torch does. `.Sum` adds the losses, and `.None` returns them as `[batch]`.
  - A sample whose labels cannot fit into its steps has loss +inf and a NaN gradient. Each label needs one step, plus one blank between repeated labels. With `zero_infinity = true`, its loss and gradient are 0.
  - The gradient with respect to `log_probs` is the exact derivative, minus the posterior of each class at each step. torch's `ctc_loss` returns exp(log_probs) minus that posterior instead. The two give the same logit gradient after `tensor_log_softmax`, which is how they should be used.
  - With equal logits over 3 classes, each alignment of 3 steps has probability (1/3)³. The target `[1]` has 6 alignments (blank 0): `0 0 1`, `0 1 1`, `1 1 1`, `1 0 0`, `1 1 0` and `0 1 0`:

  ```odin
  logits := t.tensor_new_4d(1, 3, 3, 1, requires_grad = true) // zeros: 1 sample, 3 steps, 3 classes
  loss := t.tensor_ctc_loss(t.tensor_log_softmax(logits), {1}, {3}, {1}, reduction = .Sum)
  t.tensor_backward(loss)
  ```
  ```
  loss 1.504077  (-ln(6/27) = 1.504077)
  d loss / d logits, step 0: -0.1667 -0.1667 0.3333
  ```

  At step 0, half of the alignments emit the blank and half emit label 1, so the logit gradient is softmax − posterior = (1/3 − 1/2, 1/3 − 1/2, 1/3).

- **The KL term sums over the latent dimension and averages over the batch**, the usual VAE convention (`-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()) / batch_size`). The latent dimension is the last dimension of `mu.shape`. For `mu` of shape `[B, L]` the sum is divided by B; for a sequence `[B, S, L, 1]` it is divided by B·S. σ² = exp(log σ²) is clamped at `KL_EXP_CLAMP` (1e10). Where the clamp is active that term is constant, so its gradient is 0, as with `torch.clamp`.

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

`tensor_dropout(x, p, training)` is inverted dropout. In training it zeroes each element with probability p and scales the others by 1/(1 − p). Otherwise, or when p ≤ 0, it returns a new tensor equal to `x` that passes gradients through unchanged. Either way the result is a new tensor, never `x` itself. If `x` requires gradients, the result is an op node that `tensor_free_graph` frees. If `x` is data without gradients, the result is a constant marked `owned_by_graph`: a graph that uses it frees it, so dropout on input data does not leak in a loop without a scope. A result that no graph uses is yours to free with `tensor_free`.

```
dropout: every value is 0 or 2x the input: true  eval returns an equal new tensor: true
```

## 8. Convolution and pooling

`tensor_conv2d(input, weight, bias = nil, stride = 1, pad = 0)` takes the PyTorch layouts. `input` is `[N, C_in, H, W]`, `weight` is `[C_out, C_in, kH, kW]` and `bias` is `[1, C_out]` or nil. The output is `[N, C_out, H_out, W_out]` with H_out = (H + 2·pad − kH) / stride + 1. It runs as im2col followed by a SIMD matrix product.

The kernel can be rectangular: kH and kW come from the weight's shape. `stride` and `pad` apply to both axes unless `stride_w` and `pad_w` set the width separately, which is PyTorch's `stride = (stride, stride_w)` and `padding = (pad, pad_w)`. For example, a `(5, 1)` kernel moving 3 rows at a time down a `[N, C, rows, columns]` input, with 2 rows of padding and none on the columns, is `t.tensor_conv2d(x, w, b, stride = 3, pad = 2, stride_w = 1, pad_w = 0)` with `w` of shape `[C_out, C_in, 5, 1]`.

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

`tensor_conv1d(x, weight, bias, dilation = 1, pad_left = 0, pad_right = 0, stride = 1, groups = 1)` is a 1-D convolution of a **channels-last** sequence `[batch, T, C_in]`, as the sequence layers use them. The weight is in PyTorch layout `[C_out, C_in / groups, K]` and the bias is `[1, C_out]` or nil. The output is `[batch, T_out, C_out]` with T_out = ⌊(T + pad_left + pad_right − dilation·(K − 1) − 1) / stride⌋ + 1, as `torch.nn.functional.conv1d` on the input padded with `pad_left` and `pad_right` zeros. For a causal convolution, set `pad_left = dilation·(K − 1)` and leave `pad_right = 0`:

```odin
x1 := from_values({0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8}, 1, 8); x1.shape = {1, 4, 2, 1}
w1 := from_values({-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 1, 12); w1.shape = {3, 2, 2, 1}
b1 := from_values({0.1, 0, -0.1}, 1, 3)
c1 := t.tensor_conv1d(x1, w1, b1, dilation = 2, pad_left = 2) // causal
```
```
conv1d [1, 4, 3, 1] t=0: 0.02 0.04 0.06  t=3: -0.31 0.23 0.77
```

`stride` moves the window by that many steps, so a stride of s keeps about every s-th output. `groups` splits the input and output channels into that many groups, and each group of outputs sees only its own group of inputs. Both channel counts must be divisible by `groups`. With `groups = C_in` the convolution is **depthwise**: each channel has its own kernel (with C_out = m·C_in, m kernels per channel). Here channel 0 gets a moving sum and channel 1 the identity, at stride 2:

```odin
x2 := from_values({1, 10, 2, 20, 3, 30, 4, 40, 5, 50}, 1, 10); x2.shape = {1, 5, 2, 1}
w2 := from_values({1, 1, 1, 0, 1, 0}, 1, 6); w2.shape = {2, 1, 3, 1}
c2 := t.tensor_conv1d(x2, w2, nil, pad_left = 1, pad_right = 1, stride = 2, groups = 2)
```
```
conv1d depthwise, stride 2 [1, 3, 2, 1]: [3, 10, 9, 30, 9, 50]
```

A strided convolution with a fixed basis also computes framed transforms. With one input channel, `stride = hop` and a weight whose rows are the window times cos and −sin of the DFT frequencies, the output holds the real and imaginary parts of a short-time Fourier transform, one frame per step (the test `conv1d_reference_test` checks it against `torch.stft`).

On the CPU, `tensor_conv1d` with `groups = 1` copies the input windows into a buffer of `[batch·T_out, K·C_in]` values (im2col) and multiplies it with the weight in one SIMD matrix product. The backward pass needs two more products. The buffer exists only during the call. Grouped convolutions run as direct loops, because each group's product would be tiny.

`tensor_overlap_add(frames, hop, window = nil, center = false, length = -1, start = -1)` goes the other way: it adds frames `[batch, F, N]` into signals `[batch, L]`, with frame f starting at sample f·hop. The full signal has (F − 1)·hop + N samples. Without a window the overlapping samples are simply summed. With a window w (N values), each frame is multiplied by w and every sample is divided by the sum of w² over the frames that cover it. This is the synthesis step of an inverse STFT: if the frames are windowed slices of a signal, the result is that signal again. The window and hop must leave no gaps; the summed w² must be positive at every output sample (Hann with hop ≤ N / 2 works). `center = true` drops the first and last N / 2 samples, as `torch.istft(center = True)`. `start` drops any other number of samples at each end: Vocos' "same" padding is `start = (N − hop) / 2`. `length` sets the number of output samples (samples past the end of the full signal are zero):

```odin
fr := from_values({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, 1, 12); fr.shape = {1, 3, 4, 1}
sum := t.tensor_overlap_add(fr, 2) // 3 frames of 4 samples, hop 2

// frames f = win · x[2f ..< 2f + 4] of the signal x = 1, 2, ..., 8
win := []f64{0, 0.5, 1, 0.5} // periodic Hann, N = 4
wf := from_values({0, 1, 3, 2, 0, 2, 5, 3, 0, 3, 7, 4}, 1, 12); wf.shape = {1, 3, 4, 1}
y := t.tensor_overlap_add(wf, 2, win, center = true)
```
```
overlap-add [1, 8, 1, 1] [1, 1, 3, 3, 5, 5, 3, 3]
windowed, centred [1, 4, 1, 1] [3, 4, 5, 6]
```

For a full inverse STFT, turn each spectrum frame (real parts, then imaginary parts) into time samples with `tensor_matmul` by a fixed inverse real-DFT basis, reshape the product to `[batch, F, N]` and overlap-add it with the window. The test `overlap_add_reference_test` checks this against `torch.istft`, including the gradient with respect to the spectrum.

## 9. Sequence models: recurrent cells, embeddings, attention

### RNN, GRU and LSTM

`tensor_rnn`, `tensor_gru` and `tensor_lstm` run a cell over a whole sequence `x` of shape `[batch, seq_len, input]`. They return every hidden state as `[batch, seq_len, H, 1]`.

The weights are stored **transposed** compared with PyTorch, and the input and hidden biases are combined into **one** bias (the GRU keeps one part apart, see below). G is the number of gates: 1 for the RNN, 3 for the GRU and 4 for the LSTM. The GRU is called as `tensor_gru(x, h_0, W_ih, W_hh, bias, bias_hn)`.

| Argument | Shape | PyTorch equivalent |
|---|---|---|
| `W_ih` | `[input, G·H]` | `weight_ih_l0ᵀ` |
| `W_hh` | `[H, G·H]` | `weight_hh_l0ᵀ` |
| `bias` | `[1, G·H]` | `bias_ih_l0 + bias_hh_l0`; for the GRU's n gate `bias_ih_l0` only |
| `bias_hn` (GRU) | `[1, H]` | the n part of `bias_hh_l0`; `nil` means 0 |
| `h_0`, `c_0` | `[batch, H]` | `h_0[0]`, `c_0[0]` |

The cells are defined as follows:
- **RNN:** h = tanh(x W_ih + h W_hh + b).
- **LSTM:** gates are in the order i, f, g, o, as in PyTorch. c = f·c + i·g and h = o·tanh(c).
- **GRU:** gates are in the order r, z, n, as in PyTorch, and the new state is h = (1 − z)·n + z·h, with n = tanh(x W_in + b_n + r·(h W_hn + b_hn)). `bias_hn` sits inside the reset gate.

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

All attention ops take Q `[batch, seq_q, d_k]`, K `[batch, seq_k, d_k]` and V `[batch, seq_k, d_v]`, and return `[batch, seq_q, d_v]` = softmax(QKᵀ/√d_k + mask)·V. For several heads, use `tensor_multi_head_attention` (below), or permute to `[batch·heads, seq, head_dim]` first with `tensor_permute_mha`.

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

**Several heads.** `tensor_multi_head_attention(q, k, v, batch, seq_q, seq_k, num_heads, head_dim, mask = nil, flash = false, causal = false)` is the attention step of multi-head attention. It takes the projected queries `[batch, seq_q, num_heads·head_dim]` and the projected keys and values `[batch, seq_k, num_heads·head_dim]`, where head h is the columns h·head_dim up to (h+1)·head_dim. Each head attends on its own, and the result `[batch, seq_q, num_heads·head_dim]` puts the heads side by side again. It stands for three steps:
1. `tensor_permute_mha` on q, k and v;
2. the attention op: masked attention when a mask is given (`[seq_q·seq_k]` shared, or `[batch·num_heads·seq_q·seq_k]`), flash attention when `flash` is set (optionally `causal`), plain attention otherwise;
3. `tensor_permute_mha_inverse`.

On the CPU it records exactly these ops. On the GPU it records a single node whose kernels read and write the `[batch, seq, num_heads·head_dim]` layout directly, which saves the permutes. `causal` needs `flash = true`; with the standard kernel, pass the causal mask as `mask`. The `nn` attention layers use this op.

```odin
// one sequence of 2 positions, 2 heads of width 2: [1, 2, 2·2]
x := from_values({1, 0, 0, 1, 1, 1, 0.5, -1}, 1, 8, true)
x.shape = {1, 2, 4, 1}
y := t.tensor_multi_head_attention(x, x, x, 1, 2, 2, 2, 2)
yc := t.tensor_multi_head_attention(x, x, x, 1, 2, 2, 2, 2, flash = true, causal = true)
fmt.println("shape", y.shape)
fmt.printfln("position 1: %.6f", t.tensor_values(y)[4:8])
fmt.printfln("causal position 0: %.6f", t.tensor_values(yc)[0:4])
s := t.tensor_sum(y)
t.tensor_backward(s)
fmt.printfln("dx position 0: %.6f", t.tensor_grad_values(x)[0:4])
```
```
shape [1, 2, 4, 1]
position 1: [1.000000, 0.669762, 0.415378, -0.661511]
causal position 0: [1.000000, 0.000000, 0.000000, 1.000000]
dx position 0: [0.497063, 0.850617, 0.964806, 1.325142]
```

torch gives the same values and gradients for `scaled_dot_product_attention` on the heads split as `[batch, heads, seq, head_dim]`. With the causal mask, position 0 sees only itself, so each head returns its own value row.

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
- **`tensor_detach(t, allocator = context.allocator)`** returns a copy of `t` without history: a new leaf with `requires_grad = false` and the shape of `t`. Inside a graph scope it is allocated in the scope like an op output, and `graph_end` frees it. Outside a scope it uses `allocator`, and it is yours to free. Use it to treat a value as a constant:

  ```odin
  d := t.tensor_detach(x)
  defer t.tensor_free(d)
  z := t.tensor_sum(t.tensor_mul(x, d)) // d(Σ x·d)/dx = d
  ```
  ```
  d(sum x*detach(x))/dx = [1, 2, 3]
  ```

- **Non-differentiable helpers:**
  - `tensor_top_k_mask(probs, k)` returns an `[N, E]` tensor with a 1 at the k largest entries of each row, the entries `torch.topk` picks; k must lie in 1 ..= E. It is a constant marked `owned_by_graph`.
  - `tensor_clip_weights(t, lo, hi)` clamps a tensor's values in place without recording anything, for example for WGAN weight clipping.
- **Graph walks use an explicit stack.** Unrolled recurrent nets with graphs thousands of nodes deep do not overflow the call stack.
- **Debugging aids:** `tensor_validate(t, "where")` and `tensor_validate_graph(root)` report empty data and trainable leaves without a gradient buffer.

The package's own gradient checks compare every op's backward pass with central finite differences: [`tests/gradcheck_ops_test.odin`](../tests/gradcheck_ops_test.odin), [`gradcheck_layers_test.odin`](../tests/gradcheck_layers_test.odin) and [`gradcheck_test.odin`](../tests/gradcheck_test.odin).

## 11. Memory management and graph scopes

Every op allocates its output. You free the graph in one of two ways.

### Without a scope: `tensor_free_graph`

`tensor_free_graph(root)` frees `root` and every op node it can reach through `inputs`. It also frees leaves marked `owned_by_graph`, which is the way to attach constants made inside a loss function. It **keeps** the other leaves: your parameters, inputs and targets. Free those with `tensor_free` when you are done.

It takes any number of roots. Roots whose graphs share nodes must be freed **in one call**, such as the policy and value losses of an actor-critic model that share a network body: `tensor_free_graph(policy_loss, value_loss)` frees every shared node once. Two separate calls would free the shared nodes twice.

```odin
loss := t.tensor_mse_loss(model(X), Y)
t.tensor_backward(loss)
t.tensor_free_graph(loss)
```

Three pitfalls:

- **Constant-only op chains are not linked.** An op records its inputs only when its output requires gradients. If no input of a chain requires gradients, `tensor_free_graph` finds nothing to free and the intermediate outputs leak. Run such code in a graph scope (below), or record it: between `constant_record_begin()` and `constant_record_end(results...)` every op output without gradient is noted. At the end, the ones that an op with gradient reads are handed to that graph (`owned_by_graph`), and the rest are freed, except the results you pass to `constant_record_end`. Every nn layer and model forward records itself, and `nn.forward_record_begin` / `nn.forward_record_end` record a chain of layer calls (they do nothing inside a scope).

  ```odin
  W := from_values({0.5, -0.3, 0.2, 0.8}, 2, 2) // frozen: no gradient
  V := from_values({0.1, 0.4}, 2, 1, requires_grad = true)
  start := live()
  t.constant_record_begin()
  h := t.tensor_tanh(t.tensor_matmul(X, W)) // a constant-only chain
  loss := t.tensor_mse_loss(t.tensor_matmul(h, V), Y)
  t.constant_record_end(loss)
  t.tensor_backward(loss)
  fmt.printf("dloss/dV = %.6f %.6f
", V.grad.data[0], V.grad.data[1])
  t.tensor_free_graph(loss)
  fmt.println("tensors left behind:", live() - start)
  ```
  ```
  dloss/dV = 0.062205 -0.022677
  tensors left behind: 0
  ```

  Here X = [[1, 2], [3, 4]], Y = [1, 0], and `live()` is `t.global_tensors_created - t.global_tensors_freed`. The matrix product X·W is freed by `constant_record_end`; `h` is read by the second product, so `tensor_free_graph(loss)` frees it. PyTorch gives the same gradient. Record only code whose intermediates stay inside: a tensor kept elsewhere (a cache, a struct field) must be passed to `constant_record_end`.
- **Free graphs that share nodes together**, in one `tensor_free_graph` call (above) or in one scope.
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

- **Leaves keep their own allocator.** Tensors made with `tensor_new`, the parameters, inputs and targets, are not moved into the scope; only op outputs and copies made with `tensor_detach` are. To keep a detached copy after `graph_end`, make it before `graph_begin`.
- **To put a constant in the scope**, pass `t.graph_allocator(&scope)` to `matrix_new` and `tensor_new`.
- **Nothing made in the scope survives `graph_end`.** Do not use or free those nodes afterwards. Copy out the numbers you need first. Inside the scope, `tensor_free` and `tensor_free_graph` on its nodes are allowed and cost nothing.
- **Scopes nest.** `graph_end` releases only the innermost scope.
- **A scope belongs to one thread.** The active scope is thread-local.
- **Arena memory comes in 64 MB chunks** (`GRAPH_CHUNK_SIZE`), or one chunk of its own for a larger allocation. When an allocation doesn't fit the current chunk, the arena continues in the smallest unused chunk that fits. So graphs that allocate in different patterns, such as a GAN's discriminator and generator updates, share a fixed set of chunks.
- **`graph_scope_bytes(&scope)`** reports the bytes in use (while the scope is active) and the bytes reserved. `graph_scope_destroy` returns the memory to the operating system.

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

`global_tensors_created` and `global_tensors_freed` count tensors. `graph_end` counts the nodes it releases as freed, so in a program that frees everything the two numbers are equal. All the examples in this guide together create and free 3409 tensors.

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
| Gradients | `tensor_backward(root)`, `tensor_backward_with_grad(root, g)`, `tensor_zero_grad`, `tensor_drop_grad`, `tensor_has_grad`, `tensor_ensure_grad`, `tensor_detach` |
| Free a graph | `tensor_free_graph(root)`, `tensor_free_graph(root_a, root_b)` for shared graphs; `owned_by_graph = true` for constants |
| Graph scopes | `graph_begin(&s, no_grad)`, `graph_end`, `graph_scope_destroy`, `graph_allocator`, `graph_scope_bytes`, `grad_enabled` |
| Constant-only chains | `constant_record_begin()`, `constant_record_end(results...)` |
| Arithmetic | `tensor_add`, `_sub`, `_mul`, `_div`, `_neg`, `_scale`, `_add_bias`, `_gate_mul` |
| Element-wise math | `tensor_exp`, `_log`, `_sqrt`, `_lgamma`, `_abs`, `_sin`, `_cos`, `_sign`, `_clamp`, `_norm_cdf` |
| Activations | `tensor_relu`, `_leaky_relu`, `_sigmoid`, `_tanh`, `_softplus`, `_gelu`, `_gelu_tanh`, `_softmax`, `_log_softmax` |
| Products and shapes | `tensor_matmul`, `_transpose`, `_reshape`, `_as_matrix`, `_flatten`, `_concat`, `_slice_cols`, `_slice_gate`, `_permute_mha`, `_permute_mha_inverse`, `_permute_lob` |
| Time steps | `tensor_index_select_time`, `_slice_time`, `_pad_time` (`Time_Pad`), `_length_regulate`; `length_regulator_indices` |
| Reductions | `tensor_sum`, `_mean`, `_masked_sum`, `_masked_mean`, `_sum_dim1`, `_logsumexp_dim1`; `sequence_mask` |
| Losses | `tensor_mse_loss`, `_l1_loss`, `_cross_entropy_loss`, `_bce_loss`, `_binary_cross_entropy`, `_kl_divergence`, `_entropy`, `_sharpe_loss`, `_prospect_utility`, `_ctc_loss` (`Loss_Reduction`) |
| Behavioural finance | `tensor_prospect_value`, `_prob_weighting`, `_prospect_utility` |
| Normalisation | `tensor_layer_norm`, `_rms_norm`, `_batch_norm_2d`, `_normalize_time`, `_dropout` |
| Convolution | `tensor_conv2d`, `_conv1d`, `_overlap_add`, `_max_pool2d`, `_avg_pool2d` |
| Sequences | `tensor_rnn`, `_gru`, `_lstm`, `_embedding`, `_ssm` |
| Attention | `tensor_scaled_dot_product_attention`, `_masked_scaled_dot_product_attention`, `_flash_attention`, `_multi_head_attention` |
| Constants and checks | `tensor_top_k_mask`, `tensor_clip_weights`, `tensor_validate`, `tensor_validate_graph` |
| Size and GPU residency | `tensor_numel`; for tensors on a GPU: `tensor_values`, `tensor_grad_values`, `tensor_item`, `tensor_sync_host`, `tensor_mark_host_dirty`, `tensor_is_on_device` (see the [gpu guide](gpu_guide.md)) |
| Buffers written in place (KV caches) | `tensor_zeros_beside(like, shape, allocator)`: zeros where `like` lives (device or host), kept after `graph_end`; `tensor_write_steps(dst, src, start)`: writes `src` `[B, n, W]` into steps `start ..< start + n` of `dst` `[B, S, W]`, without gradient |
