# The `gpu` package: a user guide

`wotan/gpu` runs the `tensor` autograd engine and the `nn` optimizers on a GPU. It uses WebGPU compute shaders through wgpu-native, so the same code runs on Vulkan, DirectX 12 and Metal: Windows, Linux and macOS.

The GPU is opt-in. A program that never calls `gpu.init` runs entirely on the CPU, exactly as described in the [tensor guide](tensor_guide.md) and the [nn guide](nn_guide.md). A program that does call it keeps the same tensor and `nn` calls. It moves its parameters and data to the device and reads results through a few sync procedures.

The guide assumes you know the [tensor guide](tensor_guide.md), in particular graph scopes, and the training loop from the [nn guide](nn_guide.md).

Every example was compiled and run against the current code on an NVIDIA RTX 3060 Laptop GPU (Vulkan), and the output shown is the real output.

## Contents

1. [Setup](#1-setup)
2. [Training on the GPU](#2-training-on-the-gpu)
3. [Where tensors live](#3-where-tensors-live)
4. [Ops with and without GPU kernels](#4-ops-with-and-without-gpu-kernels)
5. [Transformers on the GPU](#5-transformers-on-the-gpu)
6. [Convolutional networks on the GPU](#6-convolutional-networks-on-the-gpu)
7. [Finance models on the GPU](#7-finance-models-on-the-gpu)
8. [Optimizers, clipping and checkpoints](#8-optimizers-clipping-and-checkpoints)
9. [Memory](#9-memory)
10. [Precision](#10-precision)
11. [Performance](#11-performance)
12. [Choosing the adapter](#12-choosing-the-adapter)
13. [Validation](#13-validation)
14. [Quick reference](#14-quick-reference)

## 1. Setup

```odin
import gpu "wotan/gpu"
import nn "wotan/nn"
import t "wotan/tensor"
```

Only programs that import `wotan/gpu` need the wgpu-native library; the rest of Wotan never links it. Odin's `vendor:wgpu` package expects the library in its `lib` folder:

- **Windows:** the Odin distribution ships it (`vendor/wgpu/lib/wgpu-windows-x86_64-msvc-release`).
- **Linux and macOS:** download the release named in `vendor/wgpu/doc.odin` from the [wgpu-native releases](https://github.com/gfx-rs/wgpu-native/releases) and unzip it into `vendor/wgpu/lib` under the name it was released with.

A missing library stops the build with a message saying where it should be.

`gpu.init` opens the GPU. It returns `false` when there is no usable adapter; `gpu.init_error()` says why, and the program can carry on on the CPU:

```odin
if !gpu.init() {
	fmt.println("no GPU:", gpu.init_error())
	// everything still works, on the CPU
}
defer gpu.shutdown()
```

`gpu.shutdown` waits for the GPU and releases it. Tensors that are still on the device lose their device data. Move what you keep back first with `gpu.to_host`.

## 2. Training on the GPU

This trains a small MLP on 256 points of `y = sin(x0 − x2)`. Compared with a CPU training loop, the only new lines are `gpu.init`, the three `gpu.to_device` calls, and `t.tensor_item` to read the loss:

```odin
main :: proc() {
	if !gpu.init() {
		fmt.println("no GPU:", gpu.init_error())
		return
	}
	defer gpu.shutdown()
	info := gpu.device_info()
	fmt.printfln("GPU: %s (%v)", info.name, info.backend)

	// y = sin(x0 - x2) for 256 random points in [-1, 1]^4
	rand.reset(42)
	xm := l.matrix_new(f64, 256, 4)
	ym := l.matrix_new(f64, 256, 1)
	for i in 0 ..< 256 {
		for j in 0 ..< 4 {xm.data[i * 4 + j] = 2 * rand.float64() - 1}
		ym.data[i] = math.sin(xm.data[i * 4] - xm.data[i * 4 + 2])
	}
	x := t.tensor_new(xm)
	y := t.tensor_new(ym)
	defer {t.tensor_free(x); t.tensor_free(y)}

	model := nn.sequential_new()
	defer nn.sequential_free(model)
	nn.sequential_add(model, nn.linear_layer_new(4, 64), nn.Activation.Tanh, nn.linear_layer_new(64, 1))
	params := nn.sequential_parameters(model)
	defer delete(params)
	opt := nn.adam_new(0.01)
	defer nn.adam_free(&opt)
	for p in params {nn.adam_add_param(&opt, p)}

	// the only GPU-specific lines: move parameters and data to the device
	gpu.to_device(params[:])
	gpu.to_device(x)
	gpu.to_device(y)

	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for step in 0 ..< 200 {
		t.graph_begin(&scope)
		loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
		t.tensor_backward(loss)
		if step % 50 == 0 || step == 199 {fmt.printfln("step %3d  loss %.5f", step, t.tensor_item(loss))}
		t.graph_end(&scope)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
	}

	st := gpu.stats()
	fallbacks := 0
	for n in st.forward_fallbacks {fallbacks += n}
	for n in st.backward_fallbacks {fallbacks += n}
	fmt.printfln("kernel launches %d, submissions %d, downloads %d, CPU fallbacks %d", st.dispatches, st.flushes, st.downloads, fallbacks)
}
```

Output:

```
GPU: NVIDIA GeForce RTX 3060 Laptop GPU (Vulkan)
step   0  loss 0.44436
step  50  loss 0.01022
step 100  loss 0.00908
step 150  loss 0.00818
step 199  loss 0.00695
kernel launches 4000, submissions 208, downloads 5, CPU fallbacks 0
```

The forward pass, the backward pass and the Adam step all ran on the GPU: 20 kernel launches per step and no CPU fallback. Ops do not wait for each other. They are recorded and submitted together, and the program waits for the GPU only when it reads a value: here, the 5 printed losses.

## 3. Where tensors live

A tensor is on the CPU until `gpu.to_device` moves it, or until a GPU op produces it. On the device, its data and its gradient live in GPU memory, and `t.data.data` / `t.grad.data` are **nil**. Reading them directly fails at once instead of returning stale numbers. `rows`, `cols` and `shape` stay valid.

Read device values through the sync procedures:

| Procedure | Returns |
|---|---|
| `t.tensor_values(x)` | the data as `[]f64`, downloaded first if needed |
| `t.tensor_grad_values(x)` | the gradient (nil if there is none) |
| `t.tensor_item(x)` | the value of a one-element tensor, e.g. a loss |
| `t.tensor_sync_host(x)` | nothing; makes `x.data.data` and `x.grad.data` current |
| `t.tensor_is_on_device(x)` | whether the tensor has device storage |
| `t.tensor_numel(x)` | the element count (`len(x.data.data)` is 0 on the device) |

After a sync, the host copy stays valid until a GPU op writes the tensor again. To change values in place, sync, write, then call `t.tensor_mark_host_dirty(x)`: the next GPU op uploads the new values.

```odin
m := l.matrix_new(f64, 2, 3)
for &v, i in m.data {v = f64(i)}
x := t.tensor_new(m, requires_grad = true)
gpu.to_device(x)
fmt.println("on device:", t.tensor_is_on_device(x), " x.data.data == nil:", x.data.data == nil)
fmt.println("values:", t.tensor_values(x)) // downloads, the host copy is valid again

// edit the synced host copy in place, then tell the backend
t.tensor_values(x)[0] = 10
t.tensor_mark_host_dirty(x)

// an op without a GPU kernel runs on the CPU automatically; switching the
// softmax kernel off shows it
gpu.set_kernel_enabled(.Softmax, false)
probs := t.tensor_softmax(x)
loss := t.tensor_sum(t.tensor_mul(probs, probs))
t.tensor_backward(loss)
gpu.set_kernel_enabled(.Softmax, true)
fmt.printfln("loss %.6f", t.tensor_item(loss))
fmt.printfln("grad %.6f", t.tensor_grad_values(x))
st := gpu.stats()
fmt.println("softmax forward on the CPU:", st.forward_fallbacks[.Softmax], " backward:", st.backward_fallbacks[.Softmax])

t.tensor_free_graph(loss)
gpu.to_host(x) // back to the CPU: data and gradient downloaded
fmt.println("back on the host:", x.data.data, " grad:", len(x.grad.data), "values")
t.tensor_free(x)
```

Output:

```
on device: true  x.data.data == nil: true
values: [0, 1, 2, 3, 4, 5]
loss 1.509626
grad [0.000916, -0.000246, -0.000670, -0.075718, -0.130105, 0.205823]
softmax forward on the CPU: 1  backward: 1
back on the host: [10, 1, 2, 3, 4, 5]  grad: 6 values
```

**Placement rules:**
- An op runs on the GPU when at least one input is on the device and the op has a GPU kernel. Its output is then on the device.
- A host tensor used by a GPU op is uploaded automatically: constants, masks, targets. Its host data stays valid. `gpu.stats().auto_uploads` counts these uploads. Inputs that are reused every step should be moved once with `to_device`.
- Ops whose inputs are all on the host run on the CPU, exactly as without a GPU.
- `t.tensor_detach` of a device tensor returns a device tensor.
- `gpu.to_host(x)` downloads the data and gradient and frees the device buffers, so `x` is an ordinary CPU tensor again.

`gpu.init({strict = true})` makes every automatic upload panic instead. This helps find a forgotten `to_device`.

## 4. Ops with and without GPU kernels

These ops have GPU kernels for the forward and the backward pass:

| Group | Ops |
|---|---|
| element-wise | `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_neg`, `tensor_scale`, `tensor_exp`, `tensor_log`, `tensor_sqrt`, `tensor_clamp`, `tensor_norm_cdf`, `tensor_gate_mul` |
| activations | `tensor_relu`, `tensor_leaky_relu`, `tensor_sigmoid`, `tensor_tanh`, `tensor_softplus`, `tensor_gelu` (through `tensor_norm_cdf`), `tensor_gelu_tanh`, `tensor_softmax` |
| linear | `tensor_matmul` (both the 2-D and the flattened `[1, N·in]` form), `tensor_add_bias` (both forms) |
| reductions and losses | `tensor_sum`, `tensor_mean`, `tensor_sum_dim1`, `tensor_logsumexp_dim1`, `tensor_mse_loss`, `tensor_cross_entropy_loss`, `tensor_bce_loss`, `tensor_binary_cross_entropy` |
| normalisation and dropout | `tensor_layer_norm`, `tensor_rms_norm`, `tensor_batch_norm_2d`, `tensor_dropout` |
| convolution and pooling | `tensor_conv2d`, `tensor_conv1d`, `tensor_max_pool2d`, `tensor_avg_pool2d`, `tensor_flatten` |
| finance and VAE | `tensor_sharpe_loss`, `tensor_prospect_value`, `tensor_prob_weighting`, `tensor_prospect_utility`, `tensor_normalize_time`, `tensor_permute_lob`, `tensor_lgamma`, `tensor_entropy`, `tensor_kl_divergence`, `nn.reparameterize`, `tensor_top_k_mask` (no gradient) |
| sequences | `tensor_embedding`, `tensor_permute_mha`, `tensor_permute_mha_inverse`, `tensor_scaled_dot_product_attention`, `tensor_masked_scaled_dot_product_attention`, `tensor_flash_attention` |
| shape | `tensor_reshape`, `tensor_as_matrix`, `tensor_transpose`, `tensor_slice_cols`, `tensor_concat` |

With these, MLPs, CNNs, TCNs, GPT (with flash attention, RMSNorm, tied embeddings or mixture-of-experts layers), BERT, VAEs and the `ml_finance` models train without any CPU fallback. Only the recurrent cells (`tensor_rnn`, `tensor_gru`, `tensor_lstm` and Mamba's `tensor_ssm`) run on the fallback: they step through time one step after another, which a GPU does not speed up at these sizes.

`gpu.has_kernel(op)` answers the same question in code.

Every other op still accepts device tensors. It downloads its inputs, runs the CPU code, and returns a host tensor, which the next GPU op uploads. The backward pass does the same for nodes without a GPU kernel. Results are correct either way; the fallback only costs transfers. `gpu.stats()` counts fallbacks per op (`forward_fallbacks[op]`, `backward_fallbacks[op]`), which shows what a model would gain from more kernels.

## 5. Transformers on the GPU

GPT, BERT and the transformer layers of `nn` run on the GPU the same way: move the parameters, keep the calls. This trains a GPT with flash attention and dropout to continue sequences that count up modulo 16:

```odin
VOCAB :: 16
SEQ :: 16
BATCH :: 32

main :: proc() {
	if !gpu.init() {return}
	defer gpu.shutdown()

	// a GPT with 2 blocks; the task: continue a sequence that counts up mod 16
	rand.reset(1)
	model := nn.gpt_model_new(VOCAB, 64, 4, 128, 2, SEQ, use_flash = true, dropout = 0.1)
	defer nn.gpt_model_free(&model)
	params := nn.gpt_model_parameters(&model)
	defer delete(params)
	opt := nn.adam_new(0.003)
	defer nn.adam_free(&opt)
	for p in params {nn.adam_add_param(&opt, p)}
	gpu.to_device(params[:])

	mask := nn.create_causal_mask(SEQ, context.allocator)
	defer delete(mask)
	targets := make([]int, BATCH * SEQ)
	defer delete(targets)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for step in 0 ..< 300 {
		ids_m := l.matrix_new(f64, 1, BATCH * SEQ)
		for b in 0 ..< BATCH {
			start := rand.int_max(VOCAB)
			for s in 0 ..< SEQ {
				ids_m.data[b * SEQ + s] = f64((start + s) % VOCAB)
				targets[b * SEQ + s] = (start + s + 1) % VOCAB
			}
		}
		ids := t.tensor_new(ids_m)
		ids.shape = {BATCH, SEQ, 1, 1}

		t.graph_begin(&scope)
		logits := nn.gpt_model_forward(&model, ids, mask, true)
		loss := t.tensor_cross_entropy_loss(logits, targets)
		t.tensor_backward(loss)
		if step % 100 == 0 || step == 299 {fmt.printfln("step %3d  loss %.4f", step, t.tensor_item(loss))}
		t.graph_end(&scope)
		nn.clip_grad_norm(&opt, 1.0)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
		t.tensor_free(ids)
	}

	st := gpu.stats()
	fallbacks := 0
	for n in st.forward_fallbacks {fallbacks += n}
	for n in st.backward_fallbacks {fallbacks += n}
	fmt.printfln("automatic uploads %d (token and position ids), CPU fallbacks %d", st.auto_uploads, fallbacks)
}
```

Output:

```
step   0  loss 3.6016
step 100  loss 0.0008
step 200  loss 0.0004
step 299  loss 0.0003
automatic uploads 600 (token and position ids), CPU fallbacks 0
```

The token ids are host tensors, made fresh every step; they are uploaded automatically, together with the position ids GPT builds in its forward pass. Everything else ran on the GPU.

How the transformer ops behave on the GPU:
- **Attention:** plain, masked and flash attention compute the same attention as on the CPU. The scores, the masked softmax and the weighted sum are batched over all sequences and heads. Flash attention's causal mask is applied inside the softmax kernel.
- **Dropout:** masks are drawn on the CPU from `core:math/rand`, exactly as the CPU op draws them, and uploaded. With the same seed, a GPU run uses the same masks as a CPU run.
- **Embeddings:** token ids that are still on the host are checked against the vocabulary size, as on the CPU. Ids that only exist on the device are not checked; an id outside the vocabulary gives a zero row and no gradient.
- **Host steps inside the models:** a few model steps need values on the host and download them. The KV-cache of `gpt_model_forward_cached` lives on the host. Mixture-of-experts layers choose their top-k experts on the host. BERT builds its padding mask from host token ids. Each costs one download per forward pass.

## 6. Convolutional networks on the GPU

Convolutions, pooling, batch norm and flatten have GPU kernels, so CNNs built from `nn` layers train entirely on the GPU. This trains a small CNN to tell three noisy line patterns apart (rows, columns, diagonals) and then evaluates it on a fresh batch:

```odin
N :: 48
SIZE :: 16

// N noisy 16x16 images of three patterns: rows, columns, diagonals
images :: proc() -> (x: ^t.Tensor, labels: []int) {
	m := l.matrix_new(f64, 1, N * SIZE * SIZE)
	labels = make([]int, N)
	for n in 0 ..< N {
		labels[n] = rand.int_max(3)
		shift := rand.int_max(4)
		for i in 0 ..< SIZE {
			for j in 0 ..< SIZE {
				line := [3]int{i, j, (i + j)}[labels[n]]
				m.data[(n * SIZE + i) * SIZE + j] = (line + shift) % 4 == 0 ? 1 : 0
				m.data[(n * SIZE + i) * SIZE + j] += 1.5 * rand.float64()
			}
		}
	}
	x = t.tensor_new(m)
	x.shape = {N, 1, SIZE, SIZE}
	return
}

main :: proc() {
	if !gpu.init() {return}
	defer gpu.shutdown()

	rand.reset(7)
	model := nn.sequential_new()
	defer nn.sequential_free(model)
	nn.sequential_add(
		model,
		nn.conv2d_layer_new(1, 8, 3, padding = 1, use_bias = false), // batch norm follows
		nn.batch_norm_2d_layer_new(8),
		nn.Activation.ReLU,
		nn.maxpool2d_layer_new(2), // 8x8
		nn.conv2d_layer_new(8, 16, 3, padding = 1),
		nn.Activation.ReLU,
		nn.avgpool2d_layer_new(2), // 4x4
		nn.FlattenLayer{},
		nn.linear_layer_new(16 * 4 * 4, 3),
	)
	params := nn.sequential_parameters(model)
	defer delete(params)
	opt := nn.adam_new(0.01)
	defer nn.adam_free(&opt)
	for p in params {nn.adam_add_param(&opt, p)}
	gpu.to_device(params[:])

	x, labels := images()
	defer {t.tensor_free(x); delete(labels)}
	gpu.to_device(x)

	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for step in 0 ..< 60 {
		t.graph_begin(&scope)
		loss := t.tensor_cross_entropy_loss(nn.sequential_forward(model, x), labels)
		t.tensor_backward(loss)
		if step % 20 == 0 || step == 59 {fmt.printfln("step %2d  loss %.4f", step, t.tensor_item(loss))}
		t.graph_end(&scope)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
	}

	// evaluation: batch norm uses its running statistics
	nn.sequential_eval(model)
	test_x, test_labels := images()
	defer {t.tensor_free(test_x); delete(test_labels)}
	t.graph_begin(&scope)
	logits := t.tensor_values(nn.sequential_forward(model, test_x))
	correct := 0
	for n in 0 ..< N {
		best := 0
		for c in 1 ..< 3 {if logits[n * 3 + c] > logits[n * 3 + best] {best = c}}
		if best == test_labels[n] {correct += 1}
	}
	t.graph_end(&scope)
	fmt.printfln("test accuracy %d / %d", correct, N)

	bn := model.layers[1].(nn.BatchNorm2dLayer)
	fmt.printfln("running mean of channel 0: %.4f (on the device: %v)", t.tensor_values(bn.running_mean)[0], t.tensor_is_on_device(bn.running_mean))
	st := gpu.stats()
	fallbacks := 0
	for n in st.forward_fallbacks {fallbacks += n}
	for n in st.backward_fallbacks {fallbacks += n}
	fmt.printfln("automatic uploads %d, CPU fallbacks %d", st.auto_uploads, fallbacks)
}
```

Output:

```
step  0  loss 1.0757
step 20  loss 0.0713
step 40  loss 0.0007
step 59  loss 0.0001
test accuracy 48 / 48
running mean of channel 0: 0.0914 (on the device: true)
automatic uploads 3, CPU fallbacks 0
```

The three automatic uploads are the batch norm's running mean and variance, which `sequential_parameters` does not return because they are not trained, and the test batch.

How these ops behave on the GPU:
- **Convolutions:** `tensor_conv2d` and `tensor_conv1d` copy the input patches into a column buffer (im2col) and multiply it with the weight in one matrix product. The backward pass builds the column buffer again instead of keeping it. The buffer holds `C_in · kH · kW` values per output position of the whole batch; like any tensor, it must fit in one storage buffer (see [Memory](#9-memory)).
- **TCNs:** `nn.tcn_block_new` and `nn.sequential_add_tcn` use `tensor_conv1d`, so dilated causal convolutions, their dropout and the residual connection all run on the GPU.
- **Pooling:** max pooling keeps no argmax indices on the device. Its backward pass finds the largest value of each window again, taking the first one in row-major order when values tie, as on the CPU.
- **Batch norm:** the running mean and variance move to the device the first time a GPU batch norm uses them, and training-mode steps update them there. Read them with `t.tensor_values`. `nn.save_checkpoint` and the other save procedures sync them like parameters.
- **A bias in front of batch norm** has a gradient of exactly zero, because batch norm subtracts every per-channel constant. In `f32`, Adam turns the rounding residue into steps of the learning rate (see [Precision](#10-precision)), so such a bias and the running mean that tracks it drift on the GPU without changing the output. The example therefore builds that convolution with `use_bias = false`, the usual choice on any device.
- **DeepLOB** (`ml_finance`) runs entirely on the GPU with the TCN sequence layer. With an LSTM, a GRU or Mamba, that sequence layer uses the fallback.

## 7. Finance models on the GPU

The models of `ml_finance` keep their training calls. Move the parameters, and the model's own losses, transforms and training steps run on the GPU. This trains a deep hedger for an at-the-money call with the CVaR of the hedged loss as the objective:

```odin
PATHS :: 512
DATES :: 10

main :: proc() {
	if !gpu.init() {return}
	defer gpu.shutdown()

	// GBM paths of (spot, time to maturity) and the payoff of an at-the-money call
	rand.reset(3)
	pm := l.matrix_new(f64, 1, PATHS * DATES * 2)
	payoff := l.matrix_new(f64, PATHS, 1)
	for b in 0 ..< PATHS {
		s := 1.0
		for k in 0 ..< DATES {
			pm.data[(b * DATES + k) * 2] = s
			pm.data[(b * DATES + k) * 2 + 1] = 1 - f64(k) / f64(DATES)
			s *= math.exp(-0.02 / f64(DATES) + 0.2 * math.sqrt(1 / f64(DATES)) * rand.float64_normal(0, 1))
		}
		payoff.data[b] = max(pm.data[(b * DATES + DATES - 1) * 2] - 1, 0)
	}
	paths := t.tensor_new(pm)
	paths.shape = {PATHS, DATES, 2, 1}
	payoffs := t.tensor_new(payoff)
	defer {t.tensor_free(paths); t.tensor_free(payoffs)}

	hedger := ml.deep_hedger_new(
		{state_size = 2, hidden_size = 32, num_layers = 2, risk_measure = .CVaR, cvar_alpha = 0.1, transaction_cost = 0.001, num_assets = 1},
	)
	defer ml.deep_hedger_free(hedger)
	params := nn.sequential_parameters(hedger.network)
	defer delete(params)
	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	for p in params {nn.adam_add_param(&opt, p)}
	gpu.to_device(params[:]) // the hedging network trains on the GPU

	for step in 0 ..< 300 {
		loss := ml.deep_hedger_train_step(hedger, paths, payoffs, &opt)
		if step % 100 == 0 || step == 299 {fmt.printfln("step %3d  CVaR(90%%) of the hedged loss %.4f", step, loss)}
	}
	st := gpu.stats()
	fallbacks := 0
	for n in st.forward_fallbacks {fallbacks += n}
	for n in st.backward_fallbacks {fallbacks += n}
	fmt.printfln("CPU fallbacks %d, downloads %d", fallbacks, st.downloads)
}
```

Output:

```
step   0  CVaR(90%) of the hedged loss 0.2147
step 100  CVaR(90%) of the hedged loss 0.1278
step 200  CVaR(90%) of the hedged loss 0.1214
step 299  CVaR(90%) of the hedged loss 0.1189
CPU fallbacks 0, downloads 600
```

The paths stay on the host: the hedger reads them there to build each date's network input, and the GPU ops upload those inputs. The two downloads per step are the P&L, whose worst outcomes the CVaR weights are chosen from, and the returned loss.

How the finance and VAE ops behave on the GPU:
- **Element-wise transforms** (`tensor_prospect_value`, `tensor_prob_weighting`, `tensor_lgamma`) and the scalar losses (`tensor_sharpe_loss`, `tensor_prospect_utility`, `tensor_entropy`, `tensor_kl_divergence`) are computed on the GPU in `f32`. `tensor_lgamma` uses a Stirling series with a reflection below 0.5, and its gradient, the digamma function, uses the same recurrence and series as the CPU.
- **`nn.reparameterize`** draws its noise on the CPU from `core:math/rand`, exactly as the CPU version does, and uploads it. With the same seed, a GPU run samples the same `z` as a CPU run.
- **`tensor_top_k_mask`** builds its mask on the GPU when the probabilities are there. The mask is a constant without a gradient, as on the CPU.

Some `ml_finance` steps work on host values and download what they need. Each is one transfer per call:

| Model | Host step |
|---|---|
| deep hedging | CVaR tail weights from the P&L |
| conditional autoencoder | the L1 subgradient of the weights (when `l1 > 0`); the early-stopping snapshot of the best weights |
| CQL | the soft update of the target network (`cql_soft_update`) |
| PPO | sampling the action from the policy's probabilities |
| token forecasting, sentiment | reading the logits to sample or to report probabilities |
| GAT training in `metrics` | the early-stopping snapshot |

Inputs that a model reads on the host, such as the deep hedger's paths or DeepLOB's normalisation statistics, can stay on the host. If they are on the device, the model downloads them once.

## 8. Optimizers, clipping and checkpoints

`nn.adam_step`, `nn.sgd_step` and `nn.clip_grad_norm` work on device parameters without any change to the calls:
- **Adam and SGD:** each parameter is updated by one fused kernel, and the optimizer state (Adam's moments, SGD's velocity) stays on the device.
- **Gradient clipping:** the gradient norm is computed on the GPU and read back once per call.

`nn.save_checkpoint` and the other `nn` save procedures read device parameters and device optimizer state through the sync procedures. A checkpoint saved from a GPU model loads on the CPU, and the reverse. After loading into a model whose parameters are already on the device, the next GPU op uploads the loaded values.

Optimizer state follows the parameter. If a parameter returns to the host with `gpu.to_host`, the next `adam_step` copies its moments back from the device and continues on the CPU.

## 9. Memory

Device buffers come from a pool and go back to it:
- `t.tensor_free` and `t.tensor_free_graph` return the buffers of the tensors they free.
- `t.graph_end` returns the buffers of every tensor created in the scope.

With a graph scope, a training loop allocates GPU memory only in its first steps; `gpu.stats().pool_bytes` then stays flat (the test suite checks this). Parameters, their gradients and the optimizer state are created outside the scope and survive `graph_end`, as on the CPU.

A single tensor must fit in one storage buffer of the device (`maxStorageBufferBindingSize`, 2 GB on the GPU used here). A larger one panics with the size in the message.

## 10. Precision

The GPU computes in `f32`. The host API stays `f64`: values are converted to `f32` on upload and back to `f64` on download. Results therefore agree with the CPU engine to `f32` precision, not bit for bit:
- **single ops:** relative differences of about 1e-7;
- **matrix products and long reductions:** up to about 1e-6;
- **40 Adam steps of an MLP:** about 1e-6 (see [Validation](#13-validation)).

Where results need `f64` accuracy, keep those tensors on the CPU. A program can train one model on the GPU and price with another on the CPU.

**Adam and parameters with a zero gradient.** Some parameters have a gradient that is exactly zero in exact arithmetic. Two examples: the bias of an attention layer's key projection (adding a constant to every key shifts all scores of a softmax row equally), and the bias of a convolution or linear layer directly followed by batch norm (batch norm subtracts every per-channel constant). What the computation actually produces is a rounding residue: about 1e-17 on the CPU, which Adam's epsilon (1e-8) absorbs, but about 1e-8 in `f32`, which Adam normalises into full learning-rate steps. On the GPU such parameters therefore drift slowly. The model's output does not change, because they have no effect on it (PyTorch in `f32` behaves the same way).

## 11. Performance

Mean time of one Adam training step (forward, backward, update) of an MLP with ReLU, measured by the test suite on the RTX 3060 Laptop GPU and its host CPU:

| Layer sizes | Batch | CPU | GPU | Speed-up |
|---|---|---|---|---|
| 16, 64, 64, 1 | 64 | 1.47 ms | 0.64 ms | 2.3× |
| 64, 256, 256, 1 | 256 | 21.07 ms | 1.31 ms | 16.1× |
| 256, 1024, 1024, 1 | 512 | 296.89 ms | 38.89 ms | 7.6× |
| 512, 2048, 2048, 1 | 1024 | 1807.41 ms | 75.79 ms | 23.8× |

Timings on a laptop vary between runs with clock speeds and power state.

**What makes it fast:**
- **Keep tensors on the device.** Every read of a device value (`tensor_item`, `tensor_values`) waits for the GPU to finish. Read the loss every few steps, not every step, when speed matters.
- **Move reused inputs once.** `to_device` on data used every step avoids a re-upload per step.
- **Check the fallbacks.** An op on the fallback costs a download and an upload. `gpu.stats()` shows which ops they are.

## 12. Choosing the adapter

`gpu.init` takes `gpu.Options`:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `.Auto` | `.Vulkan`, `.DX12`, `.Metal` or `.OpenGL` to force one; `.Auto` prefers Vulkan or Metal, then DX12 |
| `prefer` | `.Discrete` | `.Integrated` picks an integrated GPU over a discrete one |
| `allow_software` | `false` | also accept a software adapter (Microsoft Basic Render Driver, lavapipe): slow, but runs without a GPU |
| `strict` | `false` | panic on automatic uploads (see [section 3](#3-where-tensors-live)) |

`gpu.device_info()` returns the chosen adapter's name, vendor, backend and type.

## 13. Validation

The GPU suite is a separate program, so that the main test suite builds without the wgpu library:

```
odin run gpu_tests -collection:root=. -debug
```

It compares every GPU result with the CPU engine, which is itself validated against PyTorch:
- **every op with a kernel:** forward and backward, with a random upstream gradient, including masked and causal attention, repeated embedding ids, wide softmax rows, convolutions with stride, padding and dilation, overlapping pooling windows, and the finance and VAE ops (lgamma also below 0.5, a Sharpe ratio over more returns than one workgroup, top-k masks, seeded reparameterization). Most ops agree to 1e-7 or better and matrix products to 7e-7. The largest difference, 3.5e-6, is tanh of inputs scaled by 30, where the f32 rounding of the scaled input dominates.
- **the same ops with their kernel disabled,** through the CPU fallback, and an op that has no kernel (`tensor_rnn`).
- **training:** 40 Adam steps of MLPs with five activations, SGD with Nesterov momentum and gradient clipping, and a model with every kernel disabled. Losses and final parameters agree with the CPU to 4e-6 or better.
- **convolutional models:** 10 steps of a CNN (convolutions with stride and padding, batch norm, max and average pooling, flatten, dropout) and of a TCN (dilated causal convolutions, dropout, residual connections). Losses and parameters, including the batch norm's running statistics, agree with the CPU to 2e-6; neither model uses a CPU fallback. Separate checks cover the running-statistics update and evaluation mode, and a backward pass on the CPU after a forward pass on the GPU (the CPU then finds the max-pool argmax and fetches the batch-norm statistics and the Sharpe ratio's mean and standard deviation from the device).
- **`ml_finance` models:** a deep hedger with CVaR, N-BEATS, PatchTST, a neural SDF, a conditional autoencoder with L1 penalty and early stopping, CQL with soft target updates, DeepLOB with the TCN sequence layer, and a VAE, each trained a few steps on both devices from the same seed. Losses and parameters agree with the CPU to 2e-6 or better (PatchTST: except its attention key biases, see [Precision](#10-precision)), with no CPU fallback.
- **transformers:** 12 steps of four small GPTs (layer norm and GELU; flash attention; RMSNorm, tied embeddings and tanh GELU; mixture of experts) and a BERT with padding masks, all with dropout on and the same masks on both devices. Losses agree with the CPU to 1e-6 and parameters to 2e-4 or better, except the attention key biases (see [Precision](#10-precision)). None of these models uses a CPU fallback.
- **residency:** host copies, sync procedures, automatic uploads, `to_host`, detach.
- **memory:** a scoped training loop keeps the pool size flat, and a debug build reports no leaks.
- **checkpoints:** saved from the GPU, loaded on the CPU, parameters and Adam moments identical.

Without a usable adapter the suite prints `SKIP` and exits with code 0.

## 14. Quick reference

| Procedure | Purpose |
|---|---|
| `gpu.init(opts := Options{}) -> bool` | open the GPU; false if there is none (`gpu.init_error()`) |
| `gpu.shutdown()` | release the GPU |
| `gpu.is_active()` | whether the GPU is open |
| `gpu.device_info() -> Device_Info` | adapter name, vendor, backend, type |
| `gpu.to_device(x)` / `gpu.to_device(xs)` | move tensors to the GPU |
| `gpu.to_host(x)` / `gpu.to_host(xs)` | move tensors back to the CPU |
| `gpu.synchronize()` | wait until the GPU has finished all recorded work |
| `gpu.stats() -> Stats` / `gpu.reset_stats()` | launches, submissions, transfers, pool size, automatic uploads, fallbacks per op |
| `gpu.has_kernel(op)` | whether an op has a GPU kernel |
| `gpu.set_kernel_enabled(op, enabled)` | run an op through the CPU fallback (testing, comparisons) |
| `t.tensor_values(x)`, `t.tensor_grad_values(x)`, `t.tensor_item(x)` | read device data on the host |
| `t.tensor_sync_host(x)` | make `x.data.data` and `x.grad.data` current |
| `t.tensor_mark_host_dirty(x)`, `t.tensor_mark_grad_host_dirty(x)` | after writing a synced host copy in place |
| `t.tensor_is_on_device(x)`, `t.tensor_numel(x)` | residency and element count |
