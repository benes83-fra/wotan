# The `nn` package: a user guide

`wotan/nn` builds neural networks on top of the [`tensor`](tensor_guide.md) autograd engine. It provides:
- layers: linear, 1-D and 2-D convolution, pooling, batch / layer / RMS normalisation, dropout, RNN / GRU / LSTM, embeddings, positional encoding, multi-head self- and cross-attention, transformer encoder and decoder blocks, TCN blocks, graph attention, Mamba and mixture of experts;
- containers: an `MLP` and a `Sequential` model that holds any mix of those layers;
- probabilistic losses: Gaussian and Student-t negative log-likelihood, and the pinball (quantile) loss;
- the SGD and Adam / AdamW optimisers, with per-parameter learning rates, schedules and gradient clipping;
- complete models: GPT (with mixture of experts, RMSNorm, tied embeddings and a KV cache for generation), BERT, a GAN and a VAE;
- image augmentation;
- saving and loading of models and training checkpoints, partial loading, freezing layers and replacing heads for transfer learning;
- in the subpackage `wotan/nn/tokenizers`: WordPiece (BERT) and byte-level BPE (GPT-2) tokenizers, BPE vocabulary training, and a time-series tokenizer (section 13).

The guide assumes you know the [tensor user guide](tensor_guide.md), in particular how gradients are computed and how graphs are freed (its section 11).

Every example was compiled and run against the current code, and the output shown is the real output. The layers, models, losses and optimisers agree with PyTorch 2.1.0 in double precision to within 2e-15: `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, `nn.RNN` and `nn.LSTM`, `F.conv1d`, `F.gaussian_nll_loss`, `torch.distributions.StudentT`, `torch.optim.SGD` and `AdamW`, and hand-written PyTorch versions of GPT, BERT, the TCN block, mixture of experts and Mamba. The complete example in section 12 trains a small GPT for 150 steps; replaying the same steps in PyTorch from the same initial weights gives the same losses to all 10 printed digits and the same generated tokens. The tokenizers agree with Hugging Face `transformers` 4.24 (section 13). A debug build with a tracking allocator reports no leaks, except for the one example in section 11 that shows a leak on purpose.

## Contents

1. [Setup](#1-setup)
2. [Layers](#2-layers)
3. [MLP](#3-mlp)
4. [Sequential models](#4-sequential-models)
5. [Losses](#5-losses)
6. [Optimisers and training loops](#6-optimisers-and-training-loops)
7. [GPT](#7-gpt)
8. [BERT](#8-bert)
9. [GAN, VAE and image augmentation](#9-gan-vae-and-image-augmentation)
10. [Saving and loading](#10-saving-and-loading)
11. [Transfer learning and memory](#11-transfer-learning-and-memory)
12. [A complete example](#12-a-complete-example)
13. [Tokenizers](#13-tokenizers)
14. [Quick reference](#14-quick-reference)

## 1. Setup

```odin
import l "wotan/linalg"
import nn "wotan/nn"
import optim "wotan/optimize" // learning-rate schedules
import t "wotan/tensor"
import "core:fmt"
import "core:math/rand"
```

Inside the Wotan repository the paths are relative, for example `import nn "../wotan/nn"` from `tests/`.

The examples use three small helpers:

```odin
from_values :: proc(vals: []f64, rows, cols: int, requires_grad := false) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols)
	copy(m.data, vals)
	return t.tensor_new(m, requires_grad)
}

// a [batch, steps, features] sequence batch, filled with a smooth pattern
sequence :: proc(batch, steps, features: int) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, batch * steps * features)
	for &v, i in m.data {v = 0.5 * math.sin(0.3 * f64(i))}
	x := t.tensor_new(m)
	x.shape = {batch, steps, features, 1}
	return x
}

// token ids [batch, steps] for embeddings, GPT and BERT
token_ids :: proc(ids: []f64, batch, steps: int) -> ^t.Tensor {
	x := from_values(ids, 1, batch * steps)
	x.shape = {batch, steps, 1, 1}
	return x
}
```

Layers initialise their weights with `core:math/rand`. The examples call `rand.reset(seed)` first, so their output is reproducible.

## 2. Layers

A layer is a plain struct that owns its parameter tensors. Each layer type has three procedures: `<layer>_new` creates the layer, `<layer>_forward` applies it, and `<layer>_free` frees its tensors. Parameters are leaves with `requires_grad = true`. A forward pass returns an op output, which is part of a graph (see section 11).

```odin
rand.reset(1)
lin := nn.linear_layer_new(3, 2) // weights [3, 2] Xavier-uniform, bias [1, 2] zero
defer nn.linear_layer_free(&lin)
fmt.println(lin.weights.shape, lin.bias.shape)

x := from_values({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, 4, 3)
defer t.tensor_free(x)
y := nn.linear_forward(&lin, x) // x @ W + b
defer t.tensor_free_graph(y)
fmt.println(y.shape)

s := sequence(2, 5, 3) // [batch 2, 5 steps, 3 features]
defer t.tensor_free(s)
ys := nn.linear_forward(&lin, s) // applied to every step
defer t.tensor_free_graph(ys)
fmt.println(ys.shape)
```
```
[3, 2, 1, 1] [1, 2, 1, 1]
[4, 2, 1, 1]
[2, 5, 2, 1]
```

A linear layer accepts a matrix `[N, in]` or a flattened sequence `[batch, steps, in]`, and works on the last dimension.

### The layers

| Layer | `_new(...)` | Forward input → output |
|---|---|---|
| `LinearLayer` | `linear_layer_new(in, out)` | `[N, in]` or `[B, S, in]` → same with `out` |
| `Conv2dLayer` | `conv2d_layer_new(c_in, c_out, k, stride = 1, padding = 0, use_bias = true)` | `[N, C_in, H, W]` → `[N, C_out, H', W']` |
| `MaxPool2dLayer`, `AvgPool2dLayer` | `maxpool2d_layer_new(k, stride = k)`, `avgpool2d_layer_new(k, stride = k)` | `[N, C, H, W]` → pooled |
| `BatchNorm2dLayer` | `batch_norm_2d_layer_new(features, eps = 1e-5, momentum = 0.1)` | `[N, C, H, W]` → same |
| `DropoutLayer` | `dropout_layer_new(p = 0.5)` | same shape |
| `Activation` | an enum value: `.ReLU`, `.Sigmoid`, `.Tanh`, `.LeakyReLU` (slope 0.01), `.Gelu`, `.Softmax`, `.Softplus`, `.None` | same shape |
| `FlattenLayer` | `nn.FlattenLayer{}` | `[N, C, H, W]` → `[N, C·H·W]` |
| `RNNLayer`, `GRULayer`, `LSTMLayer` | `rnn_layer_new(in, hidden)` (likewise `gru_`, `lstm_`) | `[B, S, in]` → `[B, S, hidden]`, every step |
| `EmbeddingLayer` | `embedding_layer_new(count, dim)` | ids `[B, S]` → `[B, S, dim]` |
| `PositionalEncoding` | `positional_encoding_new(max_len, dim)` | `[B, S, dim]` → plus the sinusoidal encoding |
| `MultiHeadAttentionLayer` | `multi_head_attention_layer_new(d, heads, use_flash = false, is_causal = false)` | `[B, S, d]` → `[B, S, d]` |
| `CrossAttentionLayer` | `cross_attention_layer_new(d, heads, use_flash = false)` | queries `[B, S_q, d]`, keys and values `[B, S_k, d]` → `[B, S_q, d]` |
| `LayerNormLayer` | `layer_norm_layer_new(d, eps = 1e-5)`, `rms_norm_layer_new(d, eps = 1e-5)` | same shape, over the last dimension |
| `FFNLayer` | `ffn_layer_new(d, d_ff)` | Linear → GELU (exact) → Linear, same shape |
| `TransformerEncoderBlock`, `TransformerEncoder` | `transformer_encoder_block_new(d, heads, d_ff)`, `transformer_encoder_new(layers, d, heads, d_ff)` | `[B, S, d]` → same |
| `TransformerDecoderBlock` | `transformer_decoder_block_new(d, heads, d_ff, use_flash = false)` | `(x [B, S, d], encoder output [B, S_k, d], causal mask)` → `[B, S, d]` |
| `Conv1dLayer` | `conv1d_layer_new(c_in, c_out, k, dilation = 1, padding = 0, causal = false, use_bias = true)` | channels-last `[B, T, C_in]` → `[B, T', C_out]` |
| `TCNBlock` | `tcn_block_new(c_in, c_out, k, dilation, dropout = 0)` | `[B, T, C_in]` → `[B, T, C_out]` |
| `GATLayer` | `gat_layer_new(in, hidden, heads, adjacency)` | `[B, nodes, in]` → `[B, nodes, hidden]` |
| `MambaLayer` | `mamba_layer_new(d, d_state = 16)` | `(x [B, S, d], h_0 [B, d, d_state])` → `[B, S, d]` |
| `MoELayer` | `moe_layer_new(d, d_ff, experts, top_k = 2)` | `[B, S, d]` → same |

Notes on individual layers:
- **Average pooling, batch norm and dropout** have no `_forward` procedure of their own (max pooling has `maxpool2d_layer_forward`). They run inside a `Sequential` model (section 4), which passes its training flag to them. Outside a model, use the tensor ops `t.tensor_avg_pool2d`, `t.tensor_batch_norm_2d` and `t.tensor_dropout` directly.
- **Recurrent layers** take an initial state: `rnn_layer_forward(&l, x, h_0)`, `gru_layer_forward(&l, x, h_0)`, `lstm_layer_forward(&l, x, h_0, c_0)`, each state with `batch · hidden` values. In a `Sequential` model they start from zeros. To use only the last step, take it with `nn.sequence_step(h, S - 1, allocator)`, which returns `[B, 1, hidden]`.
- **Attention.** `multi_head_attention_layer_forward(&mha, x, attention_mask = nil)` is self-attention. `attention_mask` is a `[B, S]` tensor that is 1 at real tokens and 0 at padding: padded keys receive no attention (PyTorch's `key_padding_mask`). With `is_causal = true`, every position sees only itself and earlier positions. `masked_multi_head_attention_layer_forward(&mha, x, mask)` adds an explicit `[S, S]` mask, for example from `create_causal_mask(S, allocator)`. `use_flash = true` selects the tiled FlashAttention kernel. It gives the same result, but takes no explicit masks: with a padding mask or an explicit mask the standard kernel is used.
- **The transformer blocks** are post-LayerNorm, as in the original transformer, BERT and PyTorch's default: `x = norm(x + attention(x))`, then `x = norm(x + ffn(x))`. GPT's blocks (section 7) are pre-LayerNorm.
- **The decoder block** applies causal self-attention, then cross-attention to the encoder output, then the feed-forward network, each followed by a residual connection and a LayerNorm.
- **Conv1d** works on channels-last sequences, so its input is a normal `[B, T, C]` sequence batch. `causal = true` pads `dilation · (k − 1)` zeros on the left only, so the output has `T` steps and step `t` depends only on inputs up to `t`. A **TCN block** (Bai, Kolter and Koltun, 2018) has two such convolutions with ReLU and dropout, and a residual connection through a 1 × 1 convolution when the widths differ. The paper's weight normalisation is not used.
- **GAT** is multi-head attention restricted to the edges of a graph: the scores get `log A[i, j]` added, so node `i` attends only to the nodes `j` with `A[i, j] > 0`, weighted by `A[i, j]`. Every node needs at least one edge, such as a self-loop. The layer stores the adjacency and frees it in `gat_layer_free`.
- **Mamba** here is a simplified selective state-space layer: linear projections give x, B, C and Δ = softplus(·). The parameter is θ_A, with A = −exp(θ_A) so that A is always negative. The scan is `t.tensor_ssm`, followed by an output projection. There is no convolution and no gating branch.
- **Mixture of experts.** The router picks the `top_k` experts of every token. Their gates are a softmax over those `top_k` router logits, and the output is the gate-weighted sum of the experts' FFNs. This equals the top-k routing of Hugging Face's Mixtral (top-k of the router softmax, renormalised). Every forward pass stores the load-balancing loss E · Σ_e f_e · P_e in `layer.aux_loss_val`, computed as Hugging Face's `load_balancing_loss_func`. Here f_e is the fraction of tokens routed to expert e, so the f_e sum to `top_k` and the loss is `top_k` when the load is balanced. With `moe_layer_forward(&moe, x, allocator, aux_loss = true)`, the same loss is also returned in the graph as `layer.aux_loss`: add it to the training loss so that it trains the router.

### Layout compared with PyTorch

To load PyTorch weights, or to compare results with PyTorch, map the parameters as follows:

| wotan | PyTorch |
|---|---|
| `LinearLayer.weights` `[in, out]`, `bias` `[1, out]` | `nn.Linear.weight` transposed, `bias` |
| attention `q_proj`, `k_proj`, `v_proj`, `out_proj` (linear layers) | `in_proj_weight = cat(Wq.T, Wk.T, Wv.T)`, `in_proj_bias = cat(bq, bk, bv)`, `out_proj` |
| `Conv2dLayer.weight` `[C_out, C_in, k, k]`, `Conv1dLayer.weight` `[C_out, C_in, k]` | the same layout. For Conv1d, transpose the input from `[B, T, C]` to PyTorch's `[B, C, T]`. |
| RNN / GRU / LSTM `w_ih` `[in, G·H]`, `w_hh` `[H, G·H]`, one `bias` | `weight_ih_l0.T`, `weight_hh_l0.T`, `bias = bias_ih + bias_hh`. The gate order is the same (LSTM i, f, g, o; GRU r, z, n). For the GRU this matches PyTorch only when the n-gate part of `bias_hh` is 0, because PyTorch adds that bias inside the reset gate. |
| `EmbeddingLayer.weight` `[count, dim]` | the same |
| `LayerNormLayer.gamma`, `beta` `[1, d]` | `weight`, `bias` |

Initialisation:
- Linear, convolution and recurrent weights are Xavier-uniform. The convolutions use fan-in `C_in · k²` and fan-out `C_out · k²`.
- Biases start at zero.
- Embeddings are uniform in ±1/√dim.
- Norm scales start at 1 and their shifts at 0. Batch norm's running variance starts at 1, as in PyTorch.

### Example: padding in attention

A padding token gets no attention, so changing it leaves every other position unchanged:

```odin
rand.reset(2)
mha := nn.multi_head_attention_layer_new(4, 2) // d_model 4, 2 heads
defer nn.multi_head_attention_layer_free(&mha)

x := sequence(2, 3, 4)
defer t.tensor_free(x)
keep := token_ids({1, 1, 0, 1, 1, 1}, 2, 3) // the last token of sequence 0 is padding
defer t.tensor_free(keep)

a := nn.multi_head_attention_layer_forward(&mha, x, keep)
defer t.tensor_free_graph(a)
x.data.data[2 * 4] += 5 // change the padding token
b := nn.multi_head_attention_layer_forward(&mha, x, keep)
defer t.tensor_free_graph(b)
// diff(a, b, from, to): the largest |a - b| over data[from:to]
fmt.printf("steps 0-1 of sequence 0 change by %g\n", diff(a, b, 0, 8))
fmt.printf("step 2 (the padding) changes by %.3f\n", diff(a, b, 8, 12))
```
```
steps 0-1 of sequence 0 change by 0
step 2 (the padding) changes by 0.045
```

The padding position itself still produces an output (it attends to the real tokens). Ignore it in the loss.

## 3. MLP

`mlp_new(sizes, activation = .ReLU)` builds a stack of linear layers. The activation follows every layer except the last. `mlp_forward(&net, x, drop_prob = 0, training = true)` applies dropout after each hidden activation, but only when `training` is true and `drop_prob > 0`. `mlp_add_to_opt` registers all weights and biases with an SGD or Adam optimiser.

```odin
rand.reset(3)
net := nn.mlp_new({2, 8, 1}, .Tanh) // 2 -> 8 (tanh) -> 1 (no activation)
defer nn.mlp_free(&net)
opt := nn.adam_new(0.05)
defer nn.adam_free(&opt)
nn.mlp_add_to_opt(&net, &opt)

x := from_values({0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
y := from_values({0, 1, 1, 0}, 4, 1)
defer {t.tensor_free(x); t.tensor_free(y)}

for step in 0 ..< 300 {
	nn.adam_zero_grad(&opt)
	pred := nn.mlp_forward(&net, x, training = false)
	loss := t.tensor_mse_loss(pred, y)
	t.tensor_backward(loss)
	if step % 100 == 0 || step == 299 {fmt.printf("step %3d  loss %.6f\n", step, loss.data.data[0])}
	t.tensor_free_graph(loss)
	nn.adam_step(&opt)
}
pred := nn.mlp_forward(&net, x, training = false)
defer t.tensor_free_graph(pred)
fmt.printf("XOR: %.3f %.3f %.3f %.3f\n", pred.data.data[0], pred.data.data[1], pred.data.data[2], pred.data.data[3])
```
```
step   0  loss 0.893085
step 100  loss 0.000181
step 200  loss 0.000000
step 299  loss 0.000000
XOR: -0.000 1.000 1.000 -0.000
```

## 4. Sequential models

A `Sequential` model holds a list of layers of any of the types in section 2. The `Layer` type is a union of all of them.

```odin
rand.reset(4)
model := nn.sequential_new()
defer nn.sequential_free(model)
nn.sequential_add(
	model,
	nn.conv2d_layer_new(1, 4, 3, padding = 1), // [N, 1, 8, 8] -> [N, 4, 8, 8]
	nn.batch_norm_2d_layer_new(4),
	nn.Activation.ReLU,
	nn.maxpool2d_layer_new(2), // -> [N, 4, 4, 4]
	nn.FlattenLayer{}, // -> [N, 64]
	nn.dropout_layer_new(0.25),
	nn.linear_layer_new(64, 3), // -> [N, 3]
)
params := nn.sequential_parameters(model)
defer delete(params)
n := 0
for p in params {n += len(p.data.data)}
fmt.printf("%d layers, %d parameter tensors, %d values\n", len(model.layers), len(params), n)

images := t.tensor_new_4d(2, 1, 8, 8)
defer t.tensor_free(images)
for &v, i in images.data.data {v = f64(i % 9) / 9}

out := nn.sequential_forward(model, images) // training mode: dropout on, batch statistics
defer t.tensor_free_graph(out)
fmt.println(out.shape)
bn := model.layers[1].(nn.BatchNorm2dLayer)
fmt.printf("running mean after one training batch: %.4f\n", bn.running_mean.data.data[0])

nn.sequential_eval(model) // dropout off, running statistics
e1 := nn.sequential_forward(model, images)
e2 := nn.sequential_forward(model, images)
defer {t.tensor_free_graph(e1); t.tensor_free_graph(e2)}
fmt.println("eval outputs repeat:", e1.data.data[0] == e2.data.data[0])
```
```
7 layers, 6 parameter tensors, 243 values
[2, 3, 1, 1]
running mean after one training batch: 0.0063
eval outputs repeat: true
```

Things to know about `Sequential`:
- **The training flag.** A new model is in training mode. `sequential_train` and `sequential_eval` switch the mode, which affects dropout, batch norm (batch statistics and running-average updates, or the running statistics) and the dropout of TCN blocks. `load_checkpoint` returns a model in evaluation mode.
- **Initial states.** Recurrent and Mamba layers start from a zero state. GAT layers use the adjacency they store. Mixture-of-experts layers run without the auxiliary loss (read `aux_loss_val` from the layer if you need it).
- **Parameters.** `sequential_parameters(model, allocator, all = false)` returns the trainable tensors layer by layer, in a fixed order. Checkpoints store the optimiser state in this order. With `all = true` it includes frozen tensors too. `sequential_add_to_opt(model, &opt)` registers the trainable tensors with SGD or Adam.
- **Freeing.** `sequential_free` frees every layer and the model itself. A layer you added belongs to the model from then on, so don't free it separately.

### Sequence backends

`sequential_add_seq_block(model, backend, in_size, hidden, d_state = 16)` appends a sequence block whose output width is `hidden`, whatever the backend:

| `SeqBackend` | Layers appended |
|---|---|
| `.LSTM` (default), `.GRU` | one recurrent layer `in_size → hidden` |
| `.Mamba` | a linear layer `in_size → hidden`, then a Mamba layer (Mamba keeps its width) |
| `.TCN` | three causal TCN blocks with kernel 3 and dilations 1, 2, 4: a receptive field of 29 steps |

```odin
rand.reset(5)
x := sequence(2, 10, 3) // [batch 2, 10 steps, 3 features]
defer t.tensor_free(x)
for backend in nn.SeqBackend {
	model := nn.sequential_new()
	defer nn.sequential_free(model)
	nn.sequential_add_seq_block(model, backend, in_size = 3, hidden = 8)
	h := nn.sequential_forward(model, x) // [2, 10, 8]
	last := nn.sequence_step(h, 9, context.allocator) // [2, 1, 8]
	defer t.tensor_free_graph(last)
	fmt.printf("%-5v %d layer(s)  out %v  last step %v\n", backend, len(model.layers), h.shape, last.shape)
}
```
```
LSTM  1 layer(s)  out [2, 10, 8, 1]  last step [2, 1, 8, 1]
GRU   1 layer(s)  out [2, 10, 8, 1]  last step [2, 1, 8, 1]
Mamba 2 layer(s)  out [2, 10, 8, 1]  last step [2, 1, 8, 1]
TCN   3 layer(s)  out [2, 10, 8, 1]  last step [2, 1, 8, 1]
```

So a forecasting model can switch its sequence engine with one argument, and the layers after the block stay the same.

## 5. Losses

The tensor package has the basic losses: `t.tensor_mse_loss`, `t.tensor_cross_entropy_loss`, `t.tensor_bce_loss`, `t.tensor_kl_divergence` and others. `nn` adds three probabilistic ones:

| Procedure | Loss |
|---|---|
| `gaussian_nll_loss(mu, var, target, eps = 1e-6)` | mean of ½ · (log v + (target − μ)² / v), with v = max(var, eps). This is PyTorch's `gaussian_nll_loss` with `full = false`. Unlike PyTorch, the clamp at `eps` also stops the gradient of `var`. |
| `student_t_nll_loss(mu, sigma, nu, target)` | −log of the location-scale Student-t density, averaged, with all constants. Equals `-torch.distributions.StudentT(nu, mu, sigma).log_prob(target).mean()`. ν can be learned. |
| `quantile_loss(pred, target, quantiles)` | the pinball loss: the mean of q·u for u ≥ 0 and (q − 1)·u for u < 0, with u = target − prediction. `pred` is `[N, Q]`, with column j predicting the quantile at level `quantiles[j]`. For H targets per row (for example forecast horizons), `pred` is `[N, H·Q]` with column h·Q + j for `target[n·H + h]`. |

All inputs have the same shape. Any of them can be plain data, such as a fixed variance or a fixed ν; gradients reach the inputs that require them.

```odin
mu := from_values({0}, 1, 1, true)
var := from_values({1}, 1, 1)
y := from_values({1}, 1, 1)
defer {t.tensor_free(mu); t.tensor_free(var); t.tensor_free(y)}
g := nn.gaussian_nll_loss(mu, var, y)
t.tensor_backward(g)
fmt.printf("Gaussian NLL %.4f  d/dmu %.4f\n", g.data.data[0], mu.grad.data[0])
t.tensor_free_graph(g)

sigma := from_values({1}, 1, 1)
nu := from_values({3}, 1, 1)
defer {t.tensor_free(sigma); t.tensor_free(nu)}
st := nn.student_t_nll_loss(mu, sigma, nu, y)
defer t.tensor_free_graph(st)
fmt.printf("Student-t NLL %.4f\n", st.data.data[0])

// two rows, quantiles 0.1 and 0.9 as columns
pred := from_values({0.0, 1.0, -1.0, 2.0}, 2, 2, true)
defer t.tensor_free(pred)
q := nn.quantile_loss(pred, {0.5, 0.5}, {0.1, 0.9})
defer t.tensor_free_graph(q)
fmt.printf("pinball loss %.4f\n", q.data.data[0])
```
```
Gaussian NLL 0.5000  d/dmu -1.0000
Student-t NLL 1.5763
pinball loss 0.1000
```

Two helpers prepare model outputs for losses:
- **`as_matrix(x, rows, cols)`** turns a flattened `[B, S, V]` output into a true `[B·S, V]` matrix, and gradients flow back to `x`. Row-wise ops such as `t.tensor_softmax` need it, because they work on data rows. `t.tensor_cross_entropy_loss` recognises flattened logits by itself, so for cross-entropy `as_matrix` is optional.
- **`sequence_step(x, step, allocator)`** picks one position of every sequence (for example the last step, or BERT's `[CLS]`) as `[B, 1, width]`.

## 6. Optimisers and training loops

| | SGD | Adam |
|---|---|---|
| Create | `sgd_new(lr, allocator, momentum = 0, weight_decay = 0, nesterov = false)` | `adam_new(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, allocator, weight_decay = 0)` |
| Register | `sgd_add_param(&opt, p)` | `adam_add_param(&opt, p)`, `adam_add_param_with_lr(&opt, p, lr)` |
| Step | `sgd_step(&opt)` | `adam_step(&opt)`, `adam_step_scheduled(&opt, schedule, step)` |
| Clear gradients | `sgd_zero_grad(&opt)` | `adam_zero_grad(&opt)` |
| Free | `sgd_free(&opt)` | `adam_free(&opt)` |

The details:
- **SGD** is `torch.optim.SGD` with dampening 0, and its weight decay is added to the gradient.
- **Adam** with `weight_decay > 0` is AdamW: the decay is decoupled from the gradient. Each parameter counts its own steps for the bias correction, as in PyTorch, so a parameter registered later starts its own count.
- **Frozen parameters** are skipped by both optimisers, as are parameters without a gradient buffer.
- **Zeroed, not dropped.** `adam_zero_grad` and `sgd_zero_grad` set the gradients to zero; they do not drop them, as PyTorch's `zero_grad()` does by default. Trainable parameters get their gradient buffer when they are created, so the optimisers update every trainable parameter in every step, with a zero gradient if the parameter was not used in that step.
- **Gradient clipping.** `clip_grad_norm(&opt, max_norm)` (or `clip_grad_norm(params, max_norm)`) scales all gradients together when their global norm exceeds `max_norm`, as `torch.nn.utils.clip_grad_norm_` does, and returns the norm before clipping. Call it between `tensor_backward` and the step.
- **Registration helpers.** The `*_add_to_opt` / `*_add_to_optimizer` procedures of the models register their trainable parameters.

**Learning-rate schedules** come from the `optimize` package. Each is a closed form of the step number, so training can resume at any step:

```odin
s := optim.lr_cosine(0.01, total_steps = 100, final_lr = 0.001, warmup_steps = 10)
for step in ([]int{0, 5, 10, 55, 99, 150}) {
	fmt.printf("step %3d  lr %.5f\n", step, optim.lr_schedule_at(s, step))
}
```
```
step   0  lr 0.00100
step   5  lr 0.00550
step  10  lr 0.01000
step  55  lr 0.00550
step  99  lr 0.00100
step 150  lr 0.00100
```

The warmup starts at `warmup_start = 0.1` × the base rate. The other schedules are `lr_constant`, `lr_step_decay` and `lr_linear_decay`. Set `opt.learning_rate = optim.lr_schedule_at(s, step)` before each step. If some parameters have their own rate (`adam_add_param_with_lr`), use `adam_step_scheduled(&opt, s, step)` instead. It schedules every parameter as its own PyTorch param group: a cosine takes each rate down to the same `final_lr`, as `CosineAnnealingLR`'s `eta_min` does.

**The recommended training loop** runs each step in a graph scope (see the tensor guide, section 11). The scope frees the whole graph at `graph_end`, and from the second step on it allocates no new memory:

```odin
scope: t.Graph_Scope
defer t.graph_scope_destroy(&scope)
for step in 0 ..< steps {
	t.graph_begin(&scope)
	nn.adam_zero_grad(&opt)
	loss := model_loss(...)
	t.tensor_backward(loss)
	value := loss.data.data[0] // copy out what must outlive the graph
	t.graph_end(&scope)

	nn.clip_grad_norm(&opt, 1.0)
	opt.learning_rate = optim.lr_schedule_at(schedule, step)
	nn.adam_step(&opt) // parameters and their gradients live on
}
```

Section 12 is a complete program built on this loop.

## 7. GPT

`gpt_model_new` builds a decoder-only transformer. It has token embeddings and learned position embeddings, and `num_layers` pre-LayerNorm blocks, each causal self-attention followed by an FFN or a mixture of experts. A final norm and an output projection follow. The options are:

| Parameter | Default | Meaning |
|---|---|---|
| `vocab_size`, `d_model`, `num_heads`, `d_ff`, `num_layers`, `max_seq_len` | | the architecture. `max_seq_len` is the number of position embeddings. |
| `use_flash` | false | FlashAttention kernel |
| `use_moe`, `num_experts`, `top_k` | false, 8, 2 | mixture-of-experts blocks |
| `dropout` | 0.1 | on the attention and FFN outputs of every block, only while `training` |
| `rms_norm` | false | RMSNorm instead of LayerNorm, in the blocks and the final norm |
| `tie_embeddings` | false | the output projection reuses the token embedding: logits = x · Eᵀ + b |

`gpt_model_forward(&model, ids, mask, training, aux_loss = false)` maps token ids `[B, S]` to logits `[B, S, vocab_size]`. The blocks are causal by themselves, so `mask` is usually `nil`. An explicit `[S, S]` mask is added to the causal one, except with `use_flash`, where it is ignored.

```odin
rand.reset(6)
model := nn.gpt_model_new(
	vocab_size = 10,
	d_model = 16,
	num_heads = 2,
	d_ff = 32,
	num_layers = 2,
	max_seq_len = 8,
)
defer nn.gpt_model_free(&model)
params := nn.gpt_model_parameters(&model)
defer delete(params)
fmt.printf("%d parameter tensors\n", len(params))

ids := token_ids({1, 2, 3, 4, 5, 9, 8, 7, 6, 5}, 2, 5)
defer t.tensor_free(ids)
logits := nn.gpt_model_forward(&model, ids, nil, training = true) // [2, 5, 10]
fmt.println(logits.shape)
targets := []int{2, 3, 4, 5, 6, 8, 7, 6, 5, 4} // the next token of every position
loss := t.tensor_cross_entropy_loss(nn.as_matrix(logits, 10, 10), targets)
t.tensor_backward(loss)
fmt.printf("loss %.4f (ln 10 = %.4f)\n", loss.data.data[0], math.ln(f64(10)))
t.tensor_free_graph(loss)

moe := nn.gpt_model_new(10, 16, 2, 32, 2, 8, use_moe = true, num_experts = 4, top_k = 2)
defer nn.gpt_model_free(&moe)
logits = nn.gpt_model_forward(&moe, ids, nil, training = true, aux_loss = true)
loss = t.tensor_add(
	t.tensor_cross_entropy_loss(nn.as_matrix(logits, 10, 10), targets),
	t.tensor_scale(moe.aux_loss, 0.01),
)
t.tensor_backward(loss)
fmt.printf("load-balancing loss %.4f (summed over 2 blocks, 2 per block when balanced)\n", moe.total_aux_loss)
t.tensor_free_graph(loss)
```
```
38 parameter tensors
[2, 5, 10, 1]
loss 2.7211 (ln 10 = 2.3026)
load-balancing loss 4.5684 (summed over 2 blocks, 2 per block when balanced)
```

After every forward pass of a model with MoE blocks, `model.total_aux_loss` holds the summed load-balancing loss. With `aux_loss = true`, `model.aux_loss` is the same loss as a graph node; otherwise it is `nil`.

**Parameter order.** `gpt_model_parameters(&model, allocator, all = false)` returns the parameters in this order:
1. the token and position embeddings;
2. for each block: `ln1`, the attention (q, k, v and out projections, each weights then bias), `ln2`, then the FFN (`fc1`, `fc2`) or the MoE (router, then each expert);
3. the final norm and the output projection.

With tied embeddings the output projection contributes only its bias. `gpt_model_add_to_optimizer(&model, &opt)` registers the trainable parameters with Adam.

### Generation: KV cache and sampling

Generating token by token with `gpt_model_forward` recomputes the whole prefix for every new token. A KV cache keeps every block's keys and values, so each call computes only the new positions:
- `gpt_kv_cache_new(&model, batch)` creates the cache.
- `gpt_model_forward_cached(&model, &cache, ids)` takes the new tokens `[B, n]` and returns their logits `[B, n, vocab]`. The result equals the last `n` positions of a full forward pass. It must run in a no-grad scope; it asserts this.
- Positions are absolute, so a sequence holds at most `max_seq_len` tokens. When `cache.length` reaches it, call `gpt_kv_cache_reset(&cache)` and prefill the cache again with the most recent tokens.
- Free the cache with `gpt_kv_cache_free`.

Section 12 generates with the cache.

The sampling procedures take one row of logits:

| Procedure | Picks |
|---|---|
| `gpt_sample_temperature(logits, temperature, recent, penalty)` | from softmax(logits / temperature). A temperature of 0 or below is greedy (argmax). |
| `gpt_sample_top_k(logits, k, temperature, recent, penalty)` | from the k most likely tokens |
| `gpt_sample_top_p(logits, p, temperature, recent, penalty)` | from the smallest set of most likely tokens whose probability reaches p (nucleus sampling) |
| `gpt_softmax_sample(logits)` | from softmax(logits) |

A repetition penalty above 1 divides the positive logits of the tokens in `recent` by `penalty` and multiplies their negative logits by it, once per distinct token (as in CTRL). The logits you pass in are not changed.

```odin
rand.reset(7)
logits := []f64{2.0, 1.5, 0.3, -1.0}
fmt.println("greedy:", nn.gpt_sample_temperature(logits, 0, nil, 1))
fmt.println("greedy, token 0 seen, penalty 1.5:", nn.gpt_sample_temperature(logits, 0, {0}, 1.5))
counts: [4]int
for _ in 0 ..< 10000 {counts[nn.gpt_sample_top_k(logits, 2, 1.0, nil, 1)] += 1}
fmt.println("top-k 2, 10000 draws:", counts)
counts = {}
for _ in 0 ..< 10000 {counts[nn.gpt_sample_top_p(logits, 0.9, 1.0, nil, 1)] += 1}
fmt.println("top-p 0.9, 10000 draws:", counts)
```
```
greedy: 0
greedy, token 0 seen, penalty 1.5: 1
top-k 2, 10000 draws: [6249, 3751, 0, 0]
top-p 0.9, 10000 draws: [5609, 3348, 1043, 0]
```

The expected shares are 0.62 / 0.38 for top-k, and 0.56 / 0.34 / 0.10 for top-p: the first three tokens carry 97 % of the probability, the first two only 87 %.

## 8. BERT

`bert_model_new(vocab, d_model, heads, d_ff, layers, max_seq_len, use_flash = false, dropout = 0.1)` builds a BERT encoder:
- token, position and segment embeddings, followed by a LayerNorm;
- `layers` post-LayerNorm encoder blocks;
- three heads: the masked-language-model head over all positions, a pooler (tanh) on the `[CLS]` position 0, and the next-sentence head on the pooler output.

`bert_model_forward(&model, ids, segments, training, skip_mlm = false)` returns `(mlm_logits [B, S, vocab], nsp_logits [B, 1, classes])`:
- **Padding.** Token id 0 is padding: padded keys receive no attention.
- **Segments.** `segments` holds 0 for sentence A and 1 for sentence B.
- **`skip_mlm = true`** skips the vocabulary projection and returns `nil` for the MLM logits. Use it when you only classify.

```odin
rand.reset(8)
model := nn.bert_model_new(12, 16, 2, 32, 2, 6)
defer nn.bert_model_free(&model)

// sequence 1 ends in two padding tokens (id 0)
ids := token_ids({2, 5, 7, 3, 2, 6, 0, 0}, 2, 4)
segments := token_ids({0, 0, 1, 1, 0, 0, 0, 0}, 2, 4)
defer {t.tensor_free(ids); t.tensor_free(segments)}

scope: t.Graph_Scope // inference: no graph, everything freed by graph_end
defer t.graph_scope_destroy(&scope)
t.graph_begin(&scope, no_grad = true)
mlm, nsp := nn.bert_model_forward(&model, ids, segments, training = false)
fmt.println(mlm.shape, nsp.shape)
t.graph_end(&scope)

nn.bert_replace_nsp_head(&model, 3, context.allocator) // a 3-class classifier on [CLS]
t.graph_begin(&scope, no_grad = true)
no_mlm, cls := nn.bert_model_forward(&model, ids, segments, training = false, skip_mlm = true)
fmt.println(no_mlm == nil, cls.shape)
t.graph_end(&scope)
```
```
[2, 4, 12, 1] [2, 1, 2, 1]
true [2, 1, 3, 1]
```

The two outputs share most of their graph. Free them with one `graph_end`, as here, or add both losses into one root before `tensor_free_graph`. Calling `tensor_free_graph` on each output separately would free the shared nodes twice.

To fine-tune for classification, replace the next-sentence head with `bert_replace_nsp_head(&model, classes, allocator)`, run with `skip_mlm = true` and train on `t.tensor_cross_entropy_loss(cls, labels)`. `bert_replace_mlm_head` changes the vocabulary. `bert_model_parameters` and `bert_model_add_to_optimizer` work as for GPT. The parameter order is:
1. the three embeddings and the embedding norm;
2. each block: `ln1`, attention, `ln2`, FFN;
3. the pooler, the MLM head and the next-sentence head.

**Converted Hugging Face checkpoints.** `scripts/convert_bert.py <model> <output>` converts a Hugging Face BERT model (for example `ProsusAI/finbert`) to a wotan file, and `load_bert_model` reads it. Those checkpoints were trained with LayerNorm epsilon 1e-12, while `bert_model_new` uses 1e-5, which moves their logits by about 5e-4. Call `bert_set_layer_norm_eps(model, 1e-12)` after loading. `scripts/convert_gpt2.py` converts DistilGPT-2 for `load_gpt_model`. GPT-2 was trained with the tanh approximation of GELU, while wotan's FFN uses the exact GELU, so the converted model's logits are close to Hugging Face's but not equal. `tests/sentiment_reference_test.odin` and `tests/gpt_completion_test.odin` show the full workflow, with the tokenizers.

## 9. GAN, VAE and image augmentation

**GAN.** `generator_new(noise_dim, hidden, data_dim)` and `discriminator_new(data_dim, hidden)` are three-layer MLPs:
- The generator uses ReLU and ends in tanh, so its outputs are in [−1, 1].
- The discriminator uses ReLU and ends in a sigmoid, giving a probability.
- `discriminator_forward_no_sigmoid` returns the raw score of a WGAN critic.
- `generator_add_to_optimizer` and `discriminator_add_to_optimizer` register the parameters, usually with two separate Adam optimisers.

**VAE.**
- `vae_encoder_forward(&enc, x)` returns `(mu, log_var)`, both clamped to [−5, 5].
- `reparameterize(mu, log_var, allocator)` samples z = μ + exp(½ log σ²) · ε, and gradients reach both μ and log σ².
- `vae_decoder_forward(&dec, z)` ends in tanh.
- Combine the reconstruction loss with `t.tensor_kl_divergence(mu, log_var)`. That KL term is a mean over the elements; see the tensor guide.

```odin
rand.reset(9)
enc := nn.vae_encoder_new(input_dim = 6, hidden_dim = 8, latent_dim = 2)
dec := nn.vae_decoder_new(latent_dim = 2, hidden_dim = 8, input_dim = 6)
defer {nn.vae_encoder_free(&enc); nn.vae_decoder_free(&dec)}
opt := nn.adam_new(0.01)
defer nn.adam_free(&opt)
nn.vae_encoder_add_to_optimizer(&enc, &opt)
nn.vae_decoder_add_to_optimizer(&dec, &opt)

x := from_values({0.5, -0.5, 0.2, 0.1, -0.3, 0.8, -0.2, 0.4, 0.6, -0.7, 0.0, 0.3}, 2, 6)
defer t.tensor_free(x)
for step in 0 ..< 200 {
	nn.adam_zero_grad(&opt)
	mu, log_var := nn.vae_encoder_forward(&enc, x)
	z := nn.reparameterize(mu, log_var, context.allocator)
	recon := nn.vae_decoder_forward(&dec, z)
	loss := t.tensor_add(t.tensor_mse_loss(recon, x), t.tensor_scale(t.tensor_kl_divergence(mu, log_var), 0.01))
	t.tensor_backward(loss)
	if step % 100 == 0 || step == 199 {fmt.printf("step %3d  loss %.4f\n", step, loss.data.data[0])}
	t.tensor_free_graph(loss) // one root: the shared encoder nodes are freed once
	nn.adam_step(&opt)
}
```
```
step   0  loss 0.2105
step 100  loss 0.0175
step 199  loss 0.0131
```

**Image augmentation.** `augment_batch(images, config, allocator)` applies one random affine transformation per image of an `[N, C, H, W]` batch, the same for all of its channels:
- a rotation by an angle uniform in ±`rotation_range` degrees;
- a shift by integers uniform in ±`translation_range` pixels;
- a scaling by a factor uniform in 1 ± `scale_range`.

Every output pixel samples the input bilinearly, with zeros outside the image. With `apply = false` it returns the input batch itself, so free the result only when `apply` is true. `affine_transform_image(input, output, width, height, angle, tx, ty, scale)` transforms one image. Positive angles turn counter-clockwise, so the result equals torchvision's `affine(img, -angle, [tx, ty], scale, 0)`; torchvision's `affine` turns clockwise.

```odin
rand.reset(10)
images := t.tensor_new_4d(4, 1, 8, 8)
defer t.tensor_free(images)
images.data.data[3 * 8 + 3] = 1 // one bright pixel in image 0
cfg := nn.AugmentationConfig {
	rotation_range    = 15,
	translation_range = 2,
	scale_range       = 0.1,
	apply             = true,
}
aug := nn.augment_batch(images, cfg, context.allocator)
defer t.tensor_free(aug)
fmt.println(aug.shape, aug != images)
cfg.apply = false
fmt.println("apply = false returns the input:", nn.augment_batch(images, cfg, context.allocator) == images)
```
```
[4, 1, 8, 8] true
apply = false returns the input: true
```

## 10. Saving and loading

| Saves | Loads |
|---|---|
| `save_checkpoint(model, &opt, path, epoch)`: a `Sequential` model with its Adam state | `load_checkpoint(path) -> (model, opt, epoch, ok)` |
| `save_gpt_model(&model, path)` | `load_gpt_model(path, use_flash = false, allocator) -> (^GPTModel, ok)` |
| `save_bert_model(&model, path)` | `load_bert_model(path, allocator, use_flash = false) -> (^BERTModel, ok)` |
| `save_generator`, `save_discriminator`, `save_vae_encoder`, `save_vae_decoder` | `load_generator(path)` and the others, each returning `(^Model, ok)` |

Note that `load_gpt_model` and `load_bert_model` take `use_flash` and the allocator in opposite orders. Pass them by name.

The file format is little-endian binary with a magic header and a type id, followed by each layer's configuration and tensors:
- **What is stored.** A model stores its full configuration, including the attention flags, MoE, RMSNorm and tied-embedding settings and the dropout rate, so a loaded model is built exactly as it was saved.
- **Old files.** Files from older versions of the format still load.
- **Changed heads.** A stored tensor of another size replaces the layer's tensor. A model saved after `bert_replace_nsp_head`, for example, loads with its new head.
- **Robustness.** Reading checks every access. A truncated, corrupt or foreign file returns `ok = false` with a message, and leaks nothing. A layer header with an invalid configuration, such as 0 attention heads, counts as corrupt in both `load_checkpoint` and `sequential_load_partial`.
- **Random numbers.** Loading uses a private random generator for the layer initialisation, so it leaves your random sequence unchanged.

```odin
rand.reset(11)
model := small_classifier(2) // Linear(4, 8), ReLU, Linear(8, 8), ReLU, Linear(8, classes)
defer nn.sequential_free(model)
opt := nn.adam_new(0.01)
defer nn.adam_free(&opt)
nn.sequential_add_to_opt(model, &opt)
// ... training ...
ok := nn.save_checkpoint(model, &opt, "classifier.ckpt", epoch = 12)
fmt.println("saved:", ok)

loaded, loaded_opt, epoch, ok2 := nn.load_checkpoint("classifier.ckpt")
defer {nn.sequential_free(loaded); nn.adam_free(loaded_opt); free(loaded_opt)}
fmt.println(ok2, epoch, len(loaded.layers), loaded.training, len(loaded_opt.parameters))
nn.sequential_train(loaded) // to continue training

_, _, _, ok3 := nn.load_checkpoint("does_not_exist.ckpt")
fmt.println("missing file:", ok3)
```
```
saved: true
true 12 5 false 6
Failed to read does_not_exist.ckpt: Not_Exist
missing file: false
```

Here `small_classifier(classes)` is a helper that returns a `Sequential` model with the layers named in its comment.

What `load_checkpoint` restores and what you must free:
- **Optimizer state.** It restores the Adam hyperparameters, the moments and each parameter's step count. Training that continues after a load is bit-for-bit identical to training without the interruption.
- **Freeing.** The model and the optimiser are allocated: free them with `sequential_free(model)`, then `adam_free(opt)` and `free(opt)`. The same applies to the other `load_*` procedures: call `gpt_model_free(model)` (or the model's own free procedure) and then `free(model)`.
- **Frozen layers are not stored.** After loading, every parameter is trainable again. If the model was saved with frozen layers, the stored moments no longer match its parameters, and `load_checkpoint` resets them with the message "optimizer state does not match the model; moments reset". Freeze the layers again after loading.

## 11. Transfer learning and memory

### Reusing a trained model

The building blocks:
- **Partial loading.** `sequential_load_partial(model, path)` loads the layers of a checkpoint whose type and configuration match the model's layer at the same position, and skips the others, such as a new output head.
- **Replacing heads.** `sequential_replace_last_layer(model, layer)`, `gpt_replace_output_head(&model, vocab, allocator)`, `bert_replace_nsp_head` and `bert_replace_mlm_head` replace a head with a freshly initialised one. Replacing GPT's head unties tied embeddings.
- **Freezing.** A frozen parameter has `requires_grad = false`: it gets no gradient, and the optimisers leave it unchanged.
  - Sequential models: `sequential_freeze_all`, `sequential_unfreeze_all` and `sequential_freeze_range(model, start, end)`, which freezes layers `[start, end)`.
  - GPT: `gpt_freeze_all`, `gpt_unfreeze_all` and `gpt_freeze_blocks(&model, start, end)`.
  - BERT: `bert_freeze_all`, `bert_unfreeze_all` and `bert_freeze_blocks`.
  - A single tensor: `tensor_set_requires_grad(p, on)`.
- **Order matters.** Freeze before you register the parameters: the `*_add_to_opt` helpers register only trainable ones.

```odin
rand.reset(12)
// a 3-class model that reuses the body of the 2-class one saved above
model := small_classifier(3)
defer nn.sequential_free(model)
loaded, skipped, ok := nn.sequential_load_partial(model, "classifier.ckpt")
fmt.println("loaded", loaded, "skipped", skipped, ok)

nn.sequential_freeze_range(model, 0, 3) // first linear layer, ReLU, second linear layer
opt := nn.adam_new(0.01)
defer nn.adam_free(&opt)
nn.sequential_add_to_opt(model, &opt) // only the head
fmt.println("trainable tensors:", len(opt.parameters))

x := from_values({0.1, 0.2, 0.3, 0.4, -0.5, 0.1, 0.0, 0.9}, 2, 4)
defer t.tensor_free(x)
scope: t.Graph_Scope
defer t.graph_scope_destroy(&scope)
w_before := model.layers[0].(nn.LinearLayer).weights.data.data[0]
head_before := model.layers[4].(nn.LinearLayer).weights.data.data[0]
for _ in 0 ..< 5 {
	t.graph_begin(&scope)
	nn.adam_zero_grad(&opt)
	loss := t.tensor_cross_entropy_loss(nn.sequential_forward(model, x), {0, 2})
	t.tensor_backward(loss)
	t.graph_end(&scope)
	nn.adam_step(&opt)
}
fmt.println(
	"frozen weight unchanged:",
	model.layers[0].(nn.LinearLayer).weights.data.data[0] == w_before,
	" head changed:",
	model.layers[4].(nn.LinearLayer).weights.data.data[0] != head_before,
)
```
```
loaded 4 skipped 1 true
trainable tensors: 2
frozen weight unchanged: true  head changed: true
```

The loaded layers are the first two linear layers and the two ReLUs. A layer without tensors matches when its configuration matches.

### Memory rules

1. **Layers and models own their parameters.** Free each one once with its `_free` procedure. `sequential_free` frees all of a model's layers.
2. **Models returned by the `load_*` procedures are pointers.** Free the model first, then the pointer: `gpt_model_free(m)` followed by `free(m)`, for example.
3. **A forward pass returns a node of a computation graph.** Free it in one of two ways:
   - Call `tensor_free_graph(root)` once on the single root of each graph.
   - Run the step in a graph scope. That is the recommended way, and the only practical one for transformers.
4. **Outputs that share a graph must be freed together.** BERT's two outputs, or a VAE encoder's `mu` and `log_var`, share most of their graph. Combine them into one loss before `tensor_free_graph`, or use a scope.
5. **Use a no-grad scope for inference** (`graph_begin(&scope, no_grad = true)`). It records no graph, and `graph_end` frees every output. The KV-cached GPT forward requires one. Copy out what you need, such as logits or the chosen token, before `graph_end`.
6. **Constant inputs are handled for you.** Forward passes create constant inputs: position ids, zero states, masks and the positional encoding. Inside a scope they live in the scope's arena. Without a scope they belong to the graph and are freed with it.

**Caveat: frozen leading layers without a scope.** An op output needs a gradient only if one of its inputs does. When the first layers of a model are frozen and the input is data, those layers' outputs need no gradient, and the tensor package does not link them into the graph. Section 11 of the tensor guide calls these constant-only chains. `tensor_free_graph` therefore leaves these outputs behind, one set per step. A graph scope frees them:

```odin
rand.reset(13)
model := small_classifier(2)
defer nn.sequential_free(model)
nn.sequential_freeze_range(model, 0, 2)
x := from_values({0.1, 0.2, 0.3, 0.4}, 1, 4)
defer t.tensor_free(x)
live :: proc() -> int {return t.global_tensors_created - t.global_tensors_freed}

start := live()
for _ in 0 ..< 3 {
	loss := t.tensor_cross_entropy_loss(nn.sequential_forward(model, x), {1})
	t.tensor_backward(loss)
	t.tensor_free_graph(loss)
}
fmt.println("tensor_free_graph: tensors left behind:", live() - start)

scope: t.Graph_Scope
defer t.graph_scope_destroy(&scope)
start = live()
for _ in 0 ..< 3 {
	t.graph_begin(&scope)
	loss := t.tensor_cross_entropy_loss(nn.sequential_forward(model, x), {1})
	t.tensor_backward(loss)
	t.graph_end(&scope)
}
fmt.println("graph scope: tensors left behind:", live() - start)
```
```
tensor_free_graph: tensors left behind: 9
graph scope: tensors left behind: 0
```

The 9 tensors are the matrix product, the bias addition and the ReLU of the frozen first layer, in each of the 3 steps. This example deliberately leaks those 9 tensors. Layers frozen later in the model (for example GPT blocks after a trainable embedding) are not affected, because their inputs need gradients.

## 12. A complete example

This program trains a small GPT to count modulo 5:
- the training loop uses a graph scope, Adam with a warmup-cosine schedule, and gradient clipping;
- the program then saves and reloads the model;
- finally it generates greedily with the KV cache, in a no-grad scope.

```odin
package main

import "core:fmt"
import "core:math/rand"
import l "wotan/linalg"
import nn "wotan/nn"
import optim "wotan/optimize"
import t "wotan/tensor"

VOCAB :: 5 // tokens 0..4
STEPS :: 7 // input positions per training sequence

main :: proc() {
	rand.reset(42)
	model := nn.gpt_model_new(
		VOCAB,
		d_model = 16,
		num_heads = 2,
		d_ff = 32,
		num_layers = 2,
		max_seq_len = 8,
		dropout = 0,
	)
	defer nn.gpt_model_free(&model)

	// Counting modulo 5 from every start: the model reads tokens 0..6 of each
	// sequence and predicts tokens 1..7
	ids_data := l.matrix_new(f64, 1, VOCAB * STEPS)
	targets := make([]int, VOCAB * STEPS)
	defer delete(targets)
	for s in 0 ..< VOCAB {
		for j in 0 ..< STEPS {
			ids_data.data[s * STEPS + j] = f64((s + j) % VOCAB)
			targets[s * STEPS + j] = (s + j + 1) % VOCAB
		}
	}
	ids := t.tensor_new(ids_data)
	ids.shape = {VOCAB, STEPS, 1, 1} // [batch, seq_len]
	defer t.tensor_free(ids)

	opt := nn.adam_new()
	defer nn.adam_free(&opt)
	nn.gpt_model_add_to_optimizer(&model, &opt)
	schedule := optim.lr_cosine(0.01, total_steps = 150, final_lr = 0.0005, warmup_steps = 10)

	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for step in 0 ..< 150 {
		t.graph_begin(&scope)
		nn.adam_zero_grad(&opt)
		logits := nn.gpt_model_forward(&model, ids, nil, training = true) // [5, 7, 5]
		loss := t.tensor_cross_entropy_loss(nn.as_matrix(logits, VOCAB * STEPS, VOCAB), targets)
		t.tensor_backward(loss)
		value := loss.data.data[0] // copy out before graph_end
		t.graph_end(&scope)

		nn.clip_grad_norm(&opt, 1.0)
		opt.learning_rate = optim.lr_schedule_at(schedule, step)
		nn.adam_step(&opt)
		if step % 30 == 0 || step == 149 {fmt.printf("step %3d  loss %.10f\n", step, value)}
	}

	// Save, load, and generate greedily with the KV cache
	if !nn.save_gpt_model(&model, "counter.gpt") {return}
	loaded, ok := nn.load_gpt_model("counter.gpt")
	if !ok {return}
	defer {nn.gpt_model_free(loaded); free(loaded)}

	cache := nn.gpt_kv_cache_new(loaded, batch = 1)
	defer nn.gpt_kv_cache_free(&cache)
	next := t.tensor_new(l.matrix_new(f64, 1, 1))
	next.shape = {1, 1, 1, 1}
	defer t.tensor_free(next)

	tokens := make([dynamic]int)
	defer delete(tokens)
	append(&tokens, 3) // the prompt
	next.data.data[0] = 3
	for len(tokens) < 8 {
		t.graph_begin(&scope, no_grad = true)
		logits := nn.gpt_model_forward_cached(loaded, &cache, next) // [1, 1, 5]
		token := nn.gpt_sample_temperature(logits.data.data, 0, nil, 1) // temperature 0: greedy
		t.graph_end(&scope)
		append(&tokens, token)
		next.data.data[0] = f64(token)
	}
	fmt.println("generated:", tokens[:])
}
```
```
step   0  loss 1.9516468937
step  30  loss 0.0118833859
step  60  loss 0.0016907916
step  90  loss 0.0011693092
step 120  loss 0.0010303451
step 149  loss 0.0009919999
generated: [3, 4, 0, 1, 2, 3, 4, 0]
```

The program stops at 8 tokens because the model has 8 position embeddings (`max_seq_len = 8`), and the training sequences used all of them.

To check it, the initial weights were copied into a hand-written PyTorch GPT (pre-LayerNorm, `F.scaled_dot_product_attention` with `is_causal`, exact GELU), and the same 150 steps were run with `torch.optim.Adam`, `clip_grad_norm_` and the same learning rates. PyTorch printed the same six losses to all 10 digits and generated the same tokens. Under a tracking allocator, the program ends with 0 live allocations.

## 13. Tokenizers

The package `wotan/nn/tokenizers` turns text and time series into token ids for the models above. Import it as `import tk "wotan/nn/tokenizers"`. It has three tokenizers:

| Tokenizer | Works like | Files |
|---|---|---|
| `WordPieceTokenizer` | Hugging Face's `BertTokenizer` (BERT, FinBERT) | `vocab.txt` |
| `BPETokenizer` | Hugging Face's `GPT2Tokenizer`: byte-level BPE | `vocab.json`, `merges.txt` |
| `TimeSeriesTokenizer` | Chronos' `MeanScaleUniformBins` (Ansari et al., 2024) | none |

**Validation.** The WordPiece and BPE tokenizers were compared with Hugging Face `transformers` 4.24, using the real GPT-2 and bert-base-uncased vocabularies (BERT both cased and uncased). The test used 35 texts covering:
- contractions, runs of spaces, tabs and line breaks;
- accents, the Greek final sigma, the Turkish İ, CJK, Korean, Arabic and emoji;
- combining marks, zero-width, format, private-use and unassigned characters;
- special tokens, a 120-character word, URLs and punctuation.

The ids, the decoded text, single and paired inputs with truncation and padding, and padded batches all equal the slow and the fast Hugging Face tokenizers. There is one exception: an unassigned code point is dropped by `BertTokenizer` and by wotan, but becomes `[UNK]` in `BertTokenizerFast`. The time-series tokens and decoded values equal Chronos' formulas, re-implemented from its source. BPE training is checked against Hugging Face's `BpeTrainer` by `tests/bpe_train_test.odin`.

The examples below use the vocabularies that the tests download to `model_checkpoints/` (see the [test catalog](test_catalog.md)).

### WordPiece (BERT)

```odin
wp, ok := tk.wordpiece_tokenizer_new("model_checkpoints/bert_vocab.txt", max_len = 12)
if !ok {return}
defer tk.wordpiece_tokenizer_free(&wp)

ids := tk.wordpiece_encode(&wp, "Naïve café owners don't tokenize!")
defer delete(ids)
fmt.println(ids)
for id in ids {fmt.printf("%s ", wp.ids_to_tokens[id])}
fmt.println()
text := tk.wordpiece_decode(&wp, ids)
defer delete(text)
fmt.println(text)

input, mask := tk.tokenize(&wp, "The market rallied.") // [CLS] ... [SEP], padded to max_len
defer {delete(input); delete(mask)}
fmt.println(input)
fmt.println(mask)

pair, pair_mask, segments := tk.tokenize_pair(&wp, "Is it up?", "Yes, it rallied strongly today.")
defer {delete(pair); delete(pair_mask); delete(segments)}
fmt.println(pair)
fmt.println(segments)

batch_ids, batch_segments, batch_mask := tk.tokenize_batch_to_tensors(&wp, {"Rates rise.", "Stocks fell sharply on Monday."})
defer {t.tensor_free(batch_ids); t.tensor_free(batch_segments); t.tensor_free(batch_mask)}
fmt.println(batch_ids.shape, batch_mask.data.data[:8])
```
```
[15743, 7668, 5608, 2123, 1005, 1056, 19204, 4697, 999]
naive cafe owners don ' t token ##ize !
naive cafe owners don't tokenize!
[101, 1996, 3006, 24356, 1012, 102, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
[101, 2003, 2009, 2039, 1029, 102, 2748, 1010, 2009, 24356, 6118, 102]
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
[2, 8, 1, 1] [1, 1, 1, 1, 1, 0, 0, 0]
```

How WordPiece encodes a text:
1. **Cleaning.** The text is split at the special tokens `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` and `[MASK]`, which are never broken up. Control, format, private-use and unassigned characters are removed. Whitespace becomes plain spaces, and CJK ideographs get spaces around them.
2. **Casing.** With `lowercase = true` (the default, for uncased models), each word is lowercased and its accents are stripped. Pass `lowercase = false` for cased models such as bert-base-cased.
3. **Splitting.** Punctuation is split off.
4. **Matching.** Each piece is matched greedily, longest first, against the vocabulary, with continuations written as `##…`. A piece that cannot be matched completely, or that is longer than 100 characters, becomes `[UNK]`.

`wordpiece_decode(&wp, ids, allocator, skip_special_tokens = false)` joins the tokens as Hugging Face's `decode` does: it attaches the `##` continuations and removes the spaces before `. , ! ?` and in English contractions.

The model-input helpers:
- `tokenize(&wp, text)` returns `max_len` ids, `[CLS] text [SEP]`, truncated or padded with `[PAD]` (id 0), together with the attention mask.
- `tokenize_pair(&wp, a, b)` returns `[CLS] a [SEP] b [SEP]` with segment ids 0 and 1. When the pair is too long, it removes the last token of the longer text until both fit, as Hugging Face's `longest_first` truncation does. In the example the second text lost "today .".
- `tokenize_batch_to_tensors(&wp, texts, allocator, pairs = nil, pad_to_longest = true)` returns `[batch, L]` tensors of ids, segment ids and attention mask. `L` is the longest input, or `max_len` with `pad_to_longest = false`.
- `tokenize_to_tensors` and `tokenize_pair_to_tensors` return a single input as `[1, max_len]` tensors.

The ids and segment ids feed `bert_model_forward` directly. BERT treats id 0 as padding, which is `[PAD]` in BERT vocabularies.

### Byte-level BPE (GPT-2)

```odin
bpe, ok := tk.bpe_tokenizer_new("model_checkpoints/vocab.json", "model_checkpoints/merges.txt", max_len = 16)
if !ok {return}
defer tk.bpe_tokenizer_free(&bpe)

ids := tk.bpe_encode(&bpe, "Hello world, it's 2026.<|endoftext|>")
defer delete(ids)
fmt.println(ids)
for id in ids {fmt.printf("%q ", bpe.ids_to_tokens[id])}
fmt.println()
text := tk.bpe_decode(&bpe, ids)
defer delete(text)
fmt.println(text)

batch, mask := tk.bpe_tokenize_batch_to_tensors(&bpe, {"Short.", "A somewhat longer text."})
defer {t.tensor_free(batch); t.tensor_free(mask)}
fmt.println(batch.shape, batch.data.data[:6], mask.data.data[:6])
```
```
[15496, 995, 11, 340, 338, 1160, 2075, 13, 50256]
"Hello" "Ġworld" "," "Ġit" "'s" "Ġ20" "26" "." "<|endoftext|>"
Hello world, it's 2026.<|endoftext|>
[2, 5, 1, 1] [16438, 13, 0, 0, 0, 32] [1, 1, 0, 0, 0, 1]
```

How BPE encodes a text:
1. **Special tokens.** Tokens of the form `<|…|>` are never split. So are `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` and `[MASK]` when the vocabulary has them.
2. **Pre-tokenizing.** The rest is cut into pieces with GPT-2's pattern: contractions, runs of letters, digits or other characters, and whitespace. A space stays with the word after it, which is the `Ġ` in the tokens above.
3. **Merging.** The UTF-8 bytes of each piece are merged by the priorities in `merges.txt`.

`bpe_decode` maps the tokens back to their bytes. Encoding and then decoding gives back the original text byte for byte. A decoded prefix may end in an incomplete UTF-8 sequence if the ids cut a character in two.

Padding: GPT-2 has no `[PAD]` token, so `bpe_tokenize_batch_to_tensors` pads with id 0, which is the token "!". The attention mask is what tells padding apart.

Watch out for `bpe_tokenize(&bpe, text)`: it adds `[CLS]` / `[SEP]` by default (`add_special_tokens = true`, meant for BERT-style BPE vocabularies). With the GPT-2 vocabulary, which has neither token, those ids are 0 as well. Pass `add_special_tokens = false` for GPT-2. `bpe_tokenize_to_tensors` and the batch procedure already default to no special tokens.

### Training a BPE vocabulary

`bpe_train(texts, vocab_size, special_tokens = {"<|endoftext|>"}, min_frequency = 0)` learns a byte-level BPE vocabulary, as Hugging Face's `BpeTrainer` does with the byte-level pre-tokenizer:
1. The vocabulary starts with the special tokens, then the 256 byte characters.
2. It then repeatedly merges the most frequent pair of adjacent symbols (on a tie, the pair of smaller ids) until it has `vocab_size` tokens.
3. It stops early when no pair occurs at least `min_frequency` times.

Unlike `BpeTrainer`, the special tokens are cut out of the texts first, so `<|endoftext|>` between documents adds no merges. `bpe_save` writes `vocab.json` and `merges.txt` in Hugging Face's format.

```odin
corpus := []string {
	"the cat sat on the mat",
	"the dog sat on the log",
	"the cat and the dog",
}
trained := tk.bpe_train(corpus, vocab_size = 270)
defer tk.bpe_tokenizer_free(&trained)
fmt.println("vocabulary:", len(trained.vocab), "merges:", len(trained.merges))
ids := tk.bpe_encode(&trained, "the cat sat")
defer delete(ids)
for id in ids {fmt.printf("%q ", trained.ids_to_tokens[id])}
fmt.println()

ok := tk.bpe_save(&trained, "toy_vocab.json", "toy_merges.txt")
again, ok2 := tk.bpe_tokenizer_new("toy_vocab.json", "toy_merges.txt", max_len = 32)
defer tk.bpe_tokenizer_free(&again)
ids2 := tk.bpe_encode(&again, "the cat sat")
defer delete(ids2)
fmt.println(ok, ok2, ids2)
```
```
vocabulary: 270 merges: 13
"the" "Ġcat" "Ġsat"
true true [258, 266, 269]
```

The 270 tokens are 1 special token, 256 byte characters and 13 merges. Hugging Face's `GPT2Tokenizer` can read the saved files too.

### Time series (Chronos)

The time-series tokenizer turns a series into tokens in two steps, as Chronos does:
1. **Scaling.** The series is divided by its scale s, the mean absolute value of its context. Missing values (NaN) are left out, and s = 1 if the mean is not positive.
2. **Binning.** Each scaled value becomes the token of its bin. The bin centres are spaced evenly on `[low, high]`, and values outside that range fall into the outer bins.

A token is `n_special + 1 + bin`, and decoding gives `centre[bin] · s`.

The defaults are Chronos' (4096 tokens, centres on [−15, 15], a context of 512 values) but without an end-of-sequence token, which suits a decoder-only model such as GPT. `ml_finance/token_forecast.odin` trains a GPT on these tokens and forecasts by sampling paths.

```odin
ts := tk.ts_tokenizer_new() // 4096 tokens, bin centres on [-15, 15], context 512
defer tk.ts_tokenizer_free(&ts)

prices := []f64{101.2, 102.0, math.nan_f64(), 100.5, 99.8, 103.1}
ids, mask, scale := tk.ts_encode_context(&ts, prices)
defer {delete(ids); delete(mask)}
fmt.printf("scale %.4f\n", scale)
fmt.println(ids)
fmt.println(mask)
values := tk.ts_decode(&ts, ids, scale)
defer delete(values)
fmt.printf("%.3f\n", values)
```
```
scale 101.3200
[2185, 2186, 0, 2184, 2183, 2188]
[1, 1, 0, 1, 1, 1]
[101.023, 101.766, -1519.800, 100.280, 99.537, 103.251]
```

The details:
- **Missing values.** A missing value becomes the padding token 0, with mask 0. Decoding a special token gives the lowest bin, as in Chronos: that is the −1519.8 above. Use the mask to ignore it.
- **Rounding.** The decoded values are the bin centres. With the defaults a bin is 30 / 4092 ≈ 0.73 % of the scale wide, so a value comes back within half of that.
- **Context length.** `ts_encode_context` keeps the last `context_length` values.
- **Other procedures.** `ts_encode(&ts, values, scale)` encodes values at a given scale, for example future values with the context's scale, as training targets. `ts_encode_batch_to_tensors(&ts, series)` encodes series of different lengths as one left-padded batch and returns each series' scale.

## 14. Quick reference

| Area | Procedures |
|---|---|
| Layers | `linear_layer_new`, `conv2d_layer_new`, `maxpool2d_layer_new`, `avgpool2d_layer_new`, `batch_norm_2d_layer_new`, `dropout_layer_new`, `rnn_layer_new`, `gru_layer_new`, `lstm_layer_new`, `embedding_layer_new`, `positional_encoding_new`, `multi_head_attention_layer_new`, `cross_attention_layer_new`, `layer_norm_layer_new`, `rms_norm_layer_new`, `ffn_layer_new`, `transformer_encoder_block_new`, `transformer_encoder_new`, `transformer_decoder_block_new`, `conv1d_layer_new`, `tcn_block_new`, `gat_layer_new`, `mamba_layer_new`, `moe_layer_new`; each with `_forward` (except average pooling, batch norm and dropout) and, where they hold tensors, `_free` |
| Activations | `Activation` enum, `apply_activation(x, act)` |
| Helpers | `as_matrix`, `sequence_step`, `create_causal_mask`, `xavier_uniform`, `init_uniform`, `scale_linear_weights` |
| Containers | `mlp_new`, `mlp_forward`, `mlp_add_to_opt`, `mlp_free`; `sequential_new`, `sequential_add`, `sequential_forward`, `sequential_train`, `sequential_eval`, `sequential_parameters`, `layer_parameters`, `sequential_add_to_opt`, `sequential_add_seq_block`, `sequential_add_tcn`, `sequential_replace_last_layer`, `sequential_free`, `layer_free` |
| Losses | `gaussian_nll_loss`, `student_t_nll_loss`, `quantile_loss` |
| Optimisers | `sgd_new`, `sgd_add_param`, `sgd_step`, `sgd_zero_grad`, `sgd_free`; `adam_new`, `adam_add_param`, `adam_add_param_with_lr`, `adam_step`, `adam_step_scheduled`, `adam_zero_grad`, `adam_free`; `clip_grad_norm` |
| GPT | `gpt_model_new`, `gpt_model_forward`, `gpt_model_parameters`, `gpt_block_parameters`, `gpt_model_add_to_optimizer`, `gpt_model_free`, `gpt_kv_cache_new`, `gpt_model_forward_cached`, `gpt_kv_cache_reset`, `gpt_kv_cache_free`, `gpt_sample_temperature`, `gpt_sample_top_k`, `gpt_sample_top_p`, `gpt_softmax_sample`, `gpt_replace_output_head` |
| BERT | `bert_model_new`, `bert_model_forward`, `bert_model_parameters`, `bert_block_parameters`, `bert_model_add_to_optimizer`, `bert_set_layer_norm_eps`, `bert_replace_nsp_head`, `bert_replace_mlm_head`, `bert_model_free` |
| GAN / VAE | `generator_new`, `generator_forward`, `discriminator_new`, `discriminator_forward`, `discriminator_forward_no_sigmoid`, `*_add_to_optimizer`, `*_free`; `vae_encoder_new`, `vae_encoder_forward`, `vae_decoder_new`, `vae_decoder_forward`, `reparameterize` |
| Augmentation | `AugmentationConfig`, `augment_batch`, `affine_transform_image` |
| Files | `save_checkpoint`, `load_checkpoint`, `sequential_load_partial`, `save_gpt_model`, `load_gpt_model`, `save_bert_model`, `load_bert_model`, `save_generator`, `load_generator`, `save_discriminator`, `load_discriminator`, `save_vae_encoder`, `load_vae_encoder`, `save_vae_decoder`, `load_vae_decoder` |
| Transfer | `sequential_freeze_all`, `sequential_unfreeze_all`, `sequential_freeze_range`, `gpt_freeze_all`, `gpt_unfreeze_all`, `gpt_freeze_blocks`, `bert_freeze_all`, `bert_unfreeze_all`, `bert_freeze_blocks`, `tensor_set_requires_grad` |
| Tokenizers (`tk`) | `wordpiece_tokenizer_new`, `wordpiece_encode`, `wordpiece_decode`, `tokenize`, `tokenize_to_tensors`, `tokenize_pair`, `tokenize_pair_to_tensors`, `tokenize_batch_to_tensors`, `wordpiece_tokenizer_free`; `bpe_tokenizer_new`, `bpe_encode`, `bpe_decode`, `bpe_tokenize`, `bpe_tokenize_to_tensors`, `bpe_tokenize_batch_to_tensors`, `bpe_train`, `bpe_save`, `bpe_tokenizer_free`; `ts_tokenizer_new`, `ts_scale`, `ts_token`, `ts_encode`, `ts_encode_context`, `ts_decode`, `ts_encode_batch_to_tensors`, `ts_tokenizer_free` |
