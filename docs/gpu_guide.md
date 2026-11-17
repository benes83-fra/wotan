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
5. [Optimizers, clipping and checkpoints](#5-optimizers-clipping-and-checkpoints)
6. [Memory](#6-memory)
7. [Precision](#7-precision)
8. [Performance](#8-performance)
9. [Choosing the adapter](#9-choosing-the-adapter)
10. [Validation](#10-validation)
11. [Quick reference](#11-quick-reference)

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

// an op without a GPU kernel (softmax) runs on the CPU automatically
probs := t.tensor_softmax(x)
loss := t.tensor_sum(t.tensor_mul(probs, probs))
t.tensor_backward(loss)
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
| element-wise | `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_neg`, `tensor_scale`, `tensor_exp`, `tensor_log`, `tensor_sqrt`, `tensor_clamp` |
| activations | `tensor_relu`, `tensor_leaky_relu`, `tensor_sigmoid`, `tensor_tanh`, `tensor_softplus` |
| linear | `tensor_matmul` (both the 2-D and the flattened `[1, N·in]` form), `tensor_add_bias` (both forms) |
| reductions and losses | `tensor_sum`, `tensor_mean`, `tensor_sum_dim1`, `tensor_mse_loss` |
| shape | `tensor_reshape`, `tensor_as_matrix`, `tensor_transpose`, `tensor_slice_cols`, `tensor_concat` |

`gpu.has_kernel(op)` answers the same question in code.

Every other op still accepts device tensors. It downloads its inputs, runs the CPU code, and returns a host tensor, which the next GPU op uploads. The backward pass does the same for nodes without a GPU kernel. Results are correct either way; the fallback only costs transfers. `gpu.stats()` counts fallbacks per op (`forward_fallbacks[op]`, `backward_fallbacks[op]`), which shows what a model would gain from more kernels.

## 5. Optimizers, clipping and checkpoints

`nn.adam_step`, `nn.sgd_step` and `nn.clip_grad_norm` work on device parameters without any change to the calls:
- **Adam and SGD:** each parameter is updated by one fused kernel, and the optimizer state (Adam's moments, SGD's velocity) stays on the device.
- **Gradient clipping:** the gradient norm is computed on the GPU and read back once per call.

`nn.save_checkpoint` and the other `nn` save procedures read device parameters and device optimizer state through the sync procedures. A checkpoint saved from a GPU model loads on the CPU, and the reverse. After loading into a model whose parameters are already on the device, the next GPU op uploads the loaded values.

Optimizer state follows the parameter. If a parameter returns to the host with `gpu.to_host`, the next `adam_step` copies its moments back from the device and continues on the CPU.

## 6. Memory

Device buffers come from a pool and go back to it:
- `t.tensor_free` and `t.tensor_free_graph` return the buffers of the tensors they free.
- `t.graph_end` returns the buffers of every tensor created in the scope.

With a graph scope, a training loop allocates GPU memory only in its first steps; `gpu.stats().pool_bytes` then stays flat (the test suite checks this). Parameters, their gradients and the optimizer state are created outside the scope and survive `graph_end`, as on the CPU.

A single tensor must fit in one storage buffer of the device (`maxStorageBufferBindingSize`, 2 GB on the GPU used here). A larger one panics with the size in the message.

## 7. Precision

The GPU computes in `f32`. The host API stays `f64`: values are converted to `f32` on upload and back to `f64` on download. Results therefore agree with the CPU engine to `f32` precision, not bit for bit:
- **single ops:** relative differences of about 1e-7;
- **matrix products and long reductions:** up to about 1e-6;
- **40 Adam steps of an MLP:** about 1e-6 (see [Validation](#10-validation)).

Where results need `f64` accuracy, keep those tensors on the CPU. A program can train one model on the GPU and price with another on the CPU.

Dropout masks are drawn on the CPU from `core:math/rand`, as without a GPU.

## 8. Performance

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

## 9. Choosing the adapter

`gpu.init` takes `gpu.Options`:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `.Auto` | `.Vulkan`, `.DX12`, `.Metal` or `.OpenGL` to force one; `.Auto` prefers Vulkan or Metal, then DX12 |
| `prefer` | `.Discrete` | `.Integrated` picks an integrated GPU over a discrete one |
| `allow_software` | `false` | also accept a software adapter (Microsoft Basic Render Driver, lavapipe): slow, but runs without a GPU |
| `strict` | `false` | panic on automatic uploads (see [section 3](#3-where-tensors-live)) |

`gpu.device_info()` returns the chosen adapter's name, vendor, backend and type.

## 10. Validation

The GPU suite is a separate program, so that the main test suite builds without the wgpu library:

```
odin run gpu_tests -collection:root=. -debug
```

It compares every GPU result with the CPU engine, which is itself validated against PyTorch:
- **every op with a kernel:** forward and backward, with a random upstream gradient. Most ops agree to 1e-7 or better and matrix products to 7e-7. The largest difference, 3.5e-6, is tanh of inputs scaled by 30, where the f32 rounding of the scaled input dominates.
- **the same ops with their kernel disabled,** through the CPU fallback, and two ops that have no kernel (softmax, layer norm).
- **training:** 40 Adam steps of MLPs with five activations, SGD with Nesterov momentum and gradient clipping, and a model with every kernel disabled. Losses and final parameters agree with the CPU to 4e-6 or better.
- **residency:** host copies, sync procedures, automatic uploads, `to_host`, detach.
- **memory:** a scoped training loop keeps the pool size flat, and a debug build reports no leaks.
- **checkpoints:** saved from the GPU, loaded on the CPU, parameters and Adam moments identical.

Without a usable adapter the suite prints `SKIP` and exits with code 0.

## 11. Quick reference

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
