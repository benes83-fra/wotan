package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"

// One op under test: input shapes, how to build the op from the inputs, and
// the tolerance (relative to 1 + |reference|)
Op_Case :: struct {
	name:     string,
	op:       t.Op, // the op whose kernel the fallback run disables
	shapes:   [][4]int, // logical input shapes (stored as [rows, cols] = shape[0], rest)
	storage:  [][2]int, // storage [rows, cols] when it differs from the shape (flattened)
	positive: bool, // inputs in (0.5, 2) (log, sqrt, div)
	no_grad:  []int, // inputs that need no gradient (all others do)
	tol:      f64,
	run:      proc(xs: []^t.Tensor) -> ^t.Tensor,
}

Run_Result :: struct {
	out:   []f64,
	grads: [][]f64,
}

// _run_case builds the inputs from `values`, runs the op forward and backward
// with the upstream gradient `upstream`, and returns copies of the output and
// the input gradients. on_gpu: inputs moved to the device first.
// cpu_backward: the forward runs on the device, the op's backward on the CPU.
_run_case :: proc(
	c: Op_Case,
	values: [][]f64,
	upstream: []f64,
	on_gpu: bool,
	cpu_backward := false, // backward with the op's kernel disabled
) -> (
	r: Run_Result,
	out_numel: int,
) {
	xs := make([]^t.Tensor, len(c.shapes), context.temp_allocator)
	for shape, i in c.shapes {
		rows, cols := shape[0], shape[1] * shape[2] * shape[3]
		if len(c.storage) > i && c.storage[i] != {} {rows, cols = c.storage[i][0], c.storage[i][1]}
		m := l.matrix_new(f64, rows, cols)
		copy(m.data, values[i])
		needs_grad := true
		for k in c.no_grad {if k == i {needs_grad = false}}
		xs[i] = t.tensor_new(m, needs_grad)
		xs[i].shape = shape
		if on_gpu {gpu.to_device(xs[i])}
	}
	out := c.run(xs)
	out_numel = t.tensor_numel(out)
	if upstream != nil {
		if cpu_backward {gpu.set_kernel_enabled(c.op, false)}
		t.tensor_backward_with_grad(out, upstream)
		if cpu_backward {gpu.set_kernel_enabled(c.op, true)}
		r.out = _clone(t.tensor_values(out))
		r.grads = make([][]f64, len(xs), context.temp_allocator)
		for x, i in xs {r.grads[i] = _clone(t.tensor_grad_values(x))}
	}
	t.tensor_free_graph(out)
	for x in xs {t.tensor_free(x)}
	return
}

_clone :: proc(v: []f64) -> []f64 {
	out := make([]f64, len(v), context.temp_allocator)
	copy(out, v)
	return out
}

// max over i of |a - b| / (1 + |b|)
_max_rel :: proc(a, b: []f64) -> f64 {
	if len(a) != len(b) {return math.INF_F64}
	worst := 0.0
	for v, i in a {
		d := abs(v - b[i]) / (1 + abs(b[i]))
		if math.is_nan(d) {return math.INF_F64}
		worst = max(worst, d)
	}
	return worst
}

// The cases, as a package-level table (their slices must outlive a procedure)
op_cases := [?]Op_Case {
	{
		name = "add",
		op = .Add,
		shapes = {{7, 33, 1, 1}, {7, 33, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_add(x[0], x[1])},
	},
	{
		name = "add (same input twice)",
		op = .Add,
		shapes = {{5, 9, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_add(x[0], x[0])},
	},
	{
		name = "sub",
		op = .Sub,
		shapes = {{7, 33, 1, 1}, {7, 33, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sub(x[0], x[1])},
	},
	{
		name = "mul",
		op = .Mul,
		shapes = {{300, 41, 1, 1}, {300, 41, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_mul(x[0], x[1])},
	},
	{
		name = "mul (same input twice)",
		op = .Mul,
		shapes = {{5, 9, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_mul(x[0], x[0])},
	},
	{
		name = "sub (only b needs a gradient)",
		op = .Sub,
		shapes = {{7, 33, 1, 1}, {7, 33, 1, 1}},
		no_grad = {0},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sub(x[0], x[1])},
	},
	{
		name = "mul (only a needs a gradient)",
		op = .Mul,
		shapes = {{7, 33, 1, 1}, {7, 33, 1, 1}},
		no_grad = {1},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_mul(x[0], x[1])},
	},
	{
		name = "div",
		op = .Div,
		shapes = {{7, 33, 1, 1}, {7, 33, 1, 1}},
		positive = true,
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_div(x[0], x[1])},
	},
	{
		name = "neg",
		op = .Neg,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_neg(x[0])},
	},
	{
		name = "scale",
		op = .Scale,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_scale(x[0], -2.5)},
	},
	{
		name = "relu",
		op = .Relu,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_relu(x[0])},
	},
	{
		name = "leaky_relu",
		op = .LeakyReLU,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_leaky_relu(x[0], 0.2)},
	},
	{
		name = "sigmoid",
		op = .Sigmoid,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sigmoid(
				t.tensor_scale(x[0], 30),
			)},
	},
	{
		name = "tanh",
		op = .Tanh,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_tanh(t.tensor_scale(x[0], 30))},
	},
	{
		name = "exp",
		op = .Exp,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_exp(x[0])},
	},
	{
		name = "log",
		op = .Log,
		shapes = {{4, 50, 1, 1}},
		positive = true,
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_log(x[0])},
	},
	{
		name = "sqrt",
		op = .Sqrt,
		shapes = {{4, 50, 1, 1}},
		positive = true,
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sqrt(x[0])},
	},
	{
		name = "clamp",
		op = .Clamp,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_clamp(x[0], -0.3, 0.5)},
	},
	{
		name = "softplus",
		op = .Softplus,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_softplus(
				t.tensor_scale(x[0], 50),
			)},
	},
	{
		name = "add_bias",
		op = .AddBias,
		shapes = {{37, 19, 1, 1}, {1, 19, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_add_bias(x[0], x[1])},
	},
	{
		name = "add_bias (flattened [B, S, D])",
		op = .AddBias,
		shapes = {{3, 5, 7, 1}, {1, 7, 1, 1}},
		storage = {{1, 105}, {}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_add_bias(x[0], x[1])},
	},
	{
		name = "matmul",
		op = .MatMul,
		shapes = {{70, 45, 1, 1}, {45, 33, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_matmul(x[0], x[1])},
	},
	{
		name = "matmul (flattened [B, S, in])",
		op = .MatMul,
		shapes = {{4, 6, 20, 1}, {20, 9, 1, 1}},
		storage = {{1, 480}, {}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_matmul(x[0], x[1])},
	},
	{
		name = "linear",
		op = .Linear,
		shapes = {{70, 45, 1, 1}, {45, 33, 1, 1}, {1, 33, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_linear(x[0], x[1], x[2])},
	},
	{
		name = "linear (fixed weights, bias trained)",
		op = .Linear,
		shapes = {{70, 45, 1, 1}, {45, 33, 1, 1}, {1, 33, 1, 1}},
		no_grad = {1},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_linear(x[0], x[1], x[2])},
	},
	{
		name = "linear (fixed bias)",
		op = .Linear,
		shapes = {{70, 45, 1, 1}, {45, 33, 1, 1}, {1, 33, 1, 1}},
		no_grad = {2},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_linear(x[0], x[1], x[2])},
	},
	{
		name = "linear 300x20 @ 20x150 (bias gradient over 300 rows)",
		op = .Linear,
		shapes = {{300, 20, 1, 1}, {20, 150, 1, 1}, {1, 150, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_linear(x[0], x[1], x[2])},
	},
	{
		name = "linear (flattened [B, S, in])",
		op = .Linear,
		shapes = {{3, 5, 12, 1}, {12, 7, 1, 1}, {1, 7, 1, 1}},
		storage = {{1, 180}, {}, {}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_linear(x[0], x[1], x[2])},
	},
	{
		name = "linear 130x90 @ 90x75 (register-tiled kernel, edge tiles)",
		op = .Linear,
		shapes = {{130, 90, 1, 1}, {90, 75, 1, 1}, {1, 75, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_linear(x[0], x[1], x[2])},
	},
	{
		name = "one input, three uses (x + tanh(x) * x)",
		op = .Add,
		shapes = {{6, 40, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_add(x[0], t.tensor_mul(t.tensor_tanh(x[0]), x[0]))
		},
	},
	{
		name = "sum",
		op = .Sum,
		shapes = {{130, 77, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sum(x[0])},
	},
	{
		name = "mean",
		op = .Mean,
		shapes = {{130, 77, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_mean(x[0])},
	},
	{
		name = "sum_dim1",
		op = .SumDim1,
		shapes = {{31, 17, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sum_dim1(x[0])},
	},
	{
		name = "mse_loss",
		op = .MSELoss,
		shapes = {{64, 3, 1, 1}, {64, 3, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_mse_loss(x[0], x[1])},
	},
	{
		name = "reshape",
		op = .Reshape,
		shapes = {{6, 10, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_reshape(x[0], {2, 3, 10, 1})},
	},
	{
		name = "as_matrix",
		op = .Reshape,
		shapes = {{2, 3, 10, 1}},
		storage = {{1, 60}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_as_matrix(x[0], 6, 10)},
	},
	{
		name = "transpose",
		op = .Transpose,
		shapes = {{13, 29, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_transpose(x[0])},
	},
	{
		name = "slice_cols",
		op = .SliceCols,
		shapes = {{11, 20, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_slice_cols(x[0], 5, 9)},
	},
	{
		name = "concat dim 1",
		op = .Concat,
		shapes = {{6, 4, 1, 1}, {6, 7, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_concat(x, 1)},
	},
	{
		name = "concat dim 0",
		op = .Concat,
		shapes = {{3, 5, 1, 1}, {4, 5, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_concat(x, 0)},
	},
	{
		name = "chain (matmul, bias, tanh, mse)",
		op = .MatMul,
		shapes = {{16, 8, 1, 1}, {8, 5, 1, 1}, {1, 5, 1, 1}, {16, 5, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			h := t.tensor_tanh(t.tensor_add_bias(t.tensor_matmul(x[0], x[1]), x[2]))
			return t.tensor_mse_loss(h, x[3])
		},
	},
	// ---- phase 2 ----
	{
		name = "gelu_tanh",
		op = .GeluTanh,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gelu_tanh(
				t.tensor_scale(x[0], 3),
			)},
	},
	{
		name = "norm_cdf (tails to ±6)",
		op = .NormCDF,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_norm_cdf(
				t.tensor_scale(x[0], 6),
			)},
	},
	{
		name = "gelu (x · norm_cdf(x))",
		op = .Gelu,
		shapes = {{4, 50, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gelu(t.tensor_scale(x[0], 3))},
	},
	{
		name = "softmax [N, C]",
		op = .Softmax,
		shapes = {{8, 12, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_softmax(t.tensor_scale(x[0], 4))},
	},
	{
		name = "softmax [B, S, F] (wide rows)",
		op = .Softmax,
		shapes = {{2, 3, 300, 1}},
		storage = {{1, 1800}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_softmax(t.tensor_scale(x[0], 4))},
	},
	{
		name = "logsumexp_dim1",
		op = .LogSumExpDim1,
		shapes = {{6, 9, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_logsumexp_dim1(
				t.tensor_scale(x[0], 3),
				9,
			)},
	},
	{
		name = "cross_entropy",
		op = .CrossEntropy,
		shapes = {{10, 7, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_cross_entropy_loss(
				t.tensor_scale(x[0], 3),
				CE_TARGETS[:],
			)},
	},
	{
		name = "cross_entropy (flattened [1, N·C])",
		op = .CrossEntropy,
		shapes = {{1, 70, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_cross_entropy_loss(
				x[0],
				CE_TARGETS[:],
			)},
	},
	{
		name = "bce_loss",
		op = .BCELoss,
		shapes = {{9, 4, 1, 1}, {9, 4, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_bce_loss(
				t.tensor_sigmoid(t.tensor_scale(x[0], 3)),
				t.tensor_sigmoid(x[1]),
			)
		},
	},
	{
		name = "binary_cross_entropy",
		op = .BinaryCrossEntropy,
		shapes = {{9, 4, 1, 1}, {9, 4, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_binary_cross_entropy(
				t.tensor_sigmoid(t.tensor_scale(x[0], 3)),
				t.tensor_sigmoid(x[1]),
			)
		},
	},
	{
		name = "layer_norm",
		op = .LayerNorm,
		shapes = {{6, 16, 1, 1}, {1, 16, 1, 1}, {1, 16, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_layer_norm(x[0], x[1], x[2])},
	},
	{
		name = "layer_norm (flattened, eps 1e-3)",
		op = .LayerNorm,
		shapes = {{1, 3, 16, 1}, {1, 16, 1, 1}, {1, 16, 1, 1}},
		storage = {{1, 48}, {}, {}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_layer_norm(
				x[0],
				x[1],
				x[2],
				1e-3,
			)},
	},
	{
		name = "rms_norm",
		op = .RMSNorm,
		shapes = {{5, 12, 1, 1}, {1, 12, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_rms_norm(x[0], x[1])},
	},
	{
		name = "dropout (training)",
		op = .Dropout,
		shapes = {{7, 30, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			rand.reset(77) // the same mask on the CPU and the GPU
			return t.tensor_dropout(x[0], 0.3, true)
		},
	},
	{
		name = "dropout (evaluation)",
		op = .Dropout,
		shapes = {{7, 30, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_dropout(x[0], 0.3, false)},
	},
	{
		name = "embedding (repeated ids)",
		op = .Embedding,
		shapes = {{11, 6, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			m := l.matrix_new(f64, 2, 5)
			copy(m.data, EMB_IDS[:])
			ids := t.tensor_new(m)
			ids.owned_by_graph = true // freed with the graph
			return t.tensor_embedding(ids, x[0])
		},
	},
	{
		name = "permute_mha",
		op = .PermuteMHA,
		shapes = {{2, 3, 8, 1}},
		storage = {{1, 48}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_permute_mha(x[0], 2, 3, 2, 4)},
	},
	{
		name = "permute_mha_inverse",
		op = .PermuteMHAInverse,
		shapes = {{4, 3, 4, 1}},
		storage = {{1, 48}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_permute_mha_inverse(
				x[0],
				2,
				3,
				2,
				4,
			)},
	},
	{
		name = "attention",
		op = .ScaledDotProductAttention,
		shapes = {{2, 5, 4, 1}, {2, 6, 4, 1}, {2, 6, 3, 1}},
		storage = {{1, 40}, {1, 48}, {1, 36}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_scaled_dot_product_attention(
				x[0],
				x[1],
				x[2],
			)},
	},
	{
		name = "masked attention (shared mask)",
		op = .MaskedScaledDotProductAttention,
		shapes = {{2, 5, 4, 1}, {2, 6, 4, 1}, {2, 6, 3, 1}},
		storage = {{1, 40}, {1, 48}, {1, 36}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_masked_scaled_dot_product_attention(x[0], x[1], x[2], _shared_mask())
		},
	},
	{
		name = "masked attention (per-sequence mask)",
		op = .MaskedScaledDotProductAttention,
		shapes = {{2, 5, 4, 1}, {2, 6, 4, 1}, {2, 6, 3, 1}},
		storage = {{1, 40}, {1, 48}, {1, 36}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_masked_scaled_dot_product_attention(x[0], x[1], x[2], _batch_mask())
		},
	},
	{
		name = "flash attention (causal)",
		op = .FlashAttention,
		shapes = {{3, 7, 4, 1}, {3, 7, 4, 1}, {3, 7, 5, 1}},
		storage = {{1, 84}, {1, 84}, {1, 105}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_flash_attention(
				x[0],
				x[1],
				x[2],
				true,
			)},
	},
	{
		name = "flash attention (seq_k 70 > tile)",
		op = .FlashAttention,
		shapes = {{2, 5, 8, 1}, {2, 70, 8, 1}, {2, 70, 3, 1}},
		storage = {{1, 80}, {1, 1120}, {1, 420}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_flash_attention(
				x[0],
				x[1],
				x[2],
				false,
			)},
	},
	{
		// q [2, 5, 2·4], k and v [2, 6, 2·4]: 2 heads of width 4
		name = "multi-head attention",
		op = .MultiHeadAttention,
		shapes = {{2, 5, 8, 1}, {2, 6, 8, 1}, {2, 6, 8, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_multi_head_attention(x[0], x[1], x[2], 2, 5, 6, 2, 4)
		},
	},
	{
		name = "multi-head attention (shared mask)",
		op = .MultiHeadAttention,
		shapes = {{2, 5, 8, 1}, {2, 6, 8, 1}, {2, 6, 8, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_multi_head_attention(
				x[0],
				x[1],
				x[2],
				2,
				5,
				6,
				2,
				4,
				mask = _shared_mask(),
			)
		},
	},
	{
		name = "multi-head attention (per-head mask)",
		op = .MultiHeadAttention,
		shapes = {{2, 5, 8, 1}, {2, 6, 8, 1}, {2, 6, 8, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_multi_head_attention(
				x[0],
				x[1],
				x[2],
				2,
				5,
				6,
				2,
				4,
				mask = _head_mask(),
			)
		},
	},
	{
		// 3 heads of width 2, flattened inputs
		name = "multi-head attention (flash, causal)",
		op = .MultiHeadAttention,
		shapes = {{3, 7, 6, 1}, {3, 7, 6, 1}, {3, 7, 6, 1}},
		storage = {{1, 126}, {1, 126}, {1, 126}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_multi_head_attention(
				x[0],
				x[1],
				x[2],
				3,
				7,
				7,
				3,
				2,
				flash = true,
				causal = true,
			)
		},
	},
	{
		// every product at least 32 x 32: the register-tiled kernel, with edge tiles
		name = "multi-head attention 40x33, head_dim 32 (register-tiled kernel)",
		op = .MultiHeadAttention,
		shapes = {{2, 40, 64, 1}, {2, 33, 64, 1}, {2, 33, 64, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_multi_head_attention(x[0], x[1], x[2], 2, 40, 33, 2, 32)
		},
	},
	{
		name = "chain (layer_norm, matmul, gelu_tanh, softmax, cross-entropy)",
		op = .LayerNorm,
		shapes = {{10, 8, 1, 1}, {1, 8, 1, 1}, {1, 8, 1, 1}, {8, 7, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			h := t.tensor_gelu_tanh(t.tensor_matmul(t.tensor_layer_norm(x[0], x[1], x[2]), x[3]))
			return t.tensor_cross_entropy_loss(h, CE_TARGETS[:])
		},
	},
	{
		name = "gate_mul",
		op = .GateMul,
		shapes = {{9, 6, 1, 1}, {9, 1, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gate_mul(x[0], x[1])},
	},
	// ---- phase 3 ----
	{
		name = "conv2d 3x3, padding 1, bias",
		op = .Conv2d,
		shapes = {{2, 3, 7, 6}, {4, 3, 3, 3}, {1, 4, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_conv2d(x[0], x[1], x[2], 1, 1)},
	},
	{
		name = "conv2d 2x3, stride 2, no bias",
		op = .Conv2d,
		shapes = {{3, 2, 9, 8}, {5, 2, 2, 3}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_conv2d(x[0], x[1], nil, 2, 0)},
	},
	{
		name = "conv2d 5x5, padding 2, stride 2",
		op = .Conv2d,
		shapes = {{2, 4, 6, 5}, {3, 4, 5, 5}, {1, 3, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_conv2d(x[0], x[1], x[2], 2, 2)},
	},
	{
		name = "conv1d causal, dilation 2, bias",
		op = .Conv1d,
		shapes = {{3, 10, 4, 1}, {5, 4, 3, 1}, {1, 5, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_conv1d(
				x[0],
				x[1],
				x[2],
				2,
				4,
				0,
			)},
	},
	{
		name = "conv1d padding 1 / 2, no bias",
		op = .Conv1d,
		shapes = {{2, 7, 3, 1}, {6, 3, 4, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_conv1d(x[0], x[1], nil, 1, 1, 2)},
	},
	{
		name = "conv1d 1x1",
		op = .Conv1d,
		shapes = {{2, 5, 3, 1}, {4, 3, 1, 1}, {1, 4, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_conv1d(x[0], x[1], x[2])},
	},
	{
		name = "max_pool2d 2x2, stride 2",
		op = .MaxPool2d,
		shapes = {{2, 3, 6, 8}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_max_pool2d(x[0], 2, 2, 2)},
	},
	{
		name = "max_pool2d 3x3, stride 1 (overlapping)",
		op = .MaxPool2d,
		shapes = {{2, 2, 7, 7}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_max_pool2d(x[0], 3, 3, 1)},
	},
	{
		name = "max_pool2d 3x2, stride 2 (uneven)",
		op = .MaxPool2d,
		shapes = {{1, 3, 9, 7}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_max_pool2d(x[0], 3, 2, 2)},
	},
	{
		name = "avg_pool2d 2x2, stride 2",
		op = .AvgPool2d,
		shapes = {{2, 3, 6, 8}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_avg_pool2d(x[0], 2, 2, 2)},
	},
	{
		name = "avg_pool2d 3x3, stride 2 (overlapping)",
		op = .AvgPool2d,
		shapes = {{2, 2, 9, 9}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_avg_pool2d(x[0], 3, 3, 2)},
	},
	{
		name = "batch_norm_2d (training)",
		op = .BatchNorm2d,
		// x, gamma, beta, running mean, running variance (positive inputs)
		shapes = {{4, 3, 5, 6}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
		positive = true,
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_batch_norm_2d(x[0], x[1], x[2], x[3], x[4], true)
		},
	},
	{
		name = "batch_norm_2d (evaluation)",
		op = .BatchNorm2d,
		shapes = {{4, 3, 5, 6}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
		positive = true,
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_batch_norm_2d(x[0], x[1], x[2], x[3], x[4], false)
		},
	},
	{
		name = "flatten",
		op = .Flatten,
		shapes = {{3, 2, 4, 5}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_flatten(t.tensor_scale(x[0], 2))},
	},
	{
		name = "chain (conv2d, batch_norm, relu, max_pool, flatten, matmul)",
		op = .Conv2d,
		shapes = {
			{4, 2, 6, 6},
			{3, 2, 3, 3},
			{1, 3, 1, 1},
			{1, 3, 1, 1},
			{1, 3, 1, 1},
			{1, 3, 1, 1},
			{1, 3, 1, 1},
			{27, 5, 1, 1},
		},
		positive = true,
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			h := t.tensor_conv2d(x[0], x[1], x[2], 1, 1)
			h = t.tensor_relu(t.tensor_batch_norm_2d(h, x[3], x[4], x[5], x[6], true))
			h = t.tensor_flatten(t.tensor_max_pool2d(h, 2, 2, 2))
			return t.tensor_matmul(h, x[7])
		},
	},
	// ---- phase 4 ----
	{
		name = "lgamma (0.5 .. 2)",
		op = .Lgamma,
		shapes = {{8, 12, 1, 1}},
		positive = true,
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_lgamma(x[0])},
	},
	{
		name = "lgamma (1 .. 11, and 0.05 .. 0.45 by reflection)",
		op = .Lgamma,
		shapes = {{8, 12, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			// 5·x + 6 for x in [-1, 1] covers 1 .. 11; 0.2·x + 0.25 covers 0.05 .. 0.45
			hi := t.tensor_lgamma(t.tensor_add(t.tensor_scale(x[0], 5), _const_like(x[0], 6)))
			lo := t.tensor_lgamma(t.tensor_add(t.tensor_scale(x[0], 0.2), _const_like(x[0], 0.25)))
			return t.tensor_add(hi, lo)
		},
	},
	{
		name = "prospect_value",
		op = .ProspectValue,
		shapes = {{6, 10, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_prospect_value(
				x[0],
				0.88,
				0.8,
				2.25,
			)},
	},
	{
		name = "prob_weighting",
		op = .ProbWeighting,
		shapes = {{6, 10, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_prob_weighting(t.tensor_sigmoid(t.tensor_scale(x[0], 3)), 0.69)
		},
	},
	{
		name = "kl_divergence",
		op = .KLDivergence,
		shapes = {{6, 4, 1, 1}, {6, 4, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_kl_divergence(
				x[0],
				t.tensor_scale(x[1], 2),
			)},
	},
	{
		name = "entropy (of a softmax)",
		op = .Entropy,
		shapes = {{5, 8, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_entropy(
				t.tensor_softmax(t.tensor_scale(x[0], 3)),
			)},
	},
	{
		name = "sharpe_loss",
		op = .SharpeLoss,
		shapes = {{40, 1, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_sharpe_loss(
				t.tensor_add(t.tensor_scale(x[0], 0.1), _const_like(x[0], 0.02)),
				0.01,
			)
		},
	},
	{
		name = "sharpe_loss (600 returns, > one workgroup)",
		op = .SharpeLoss,
		shapes = {{600, 1, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_sharpe_loss(t.tensor_relu(x[0]))},
	},
	{
		name = "prospect_utility",
		op = .ProspectUtility,
		shapes = {{30, 1, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_prospect_utility(
				t.tensor_scale(x[0], 0.1),
			)},
	},
	{
		name = "normalize_time [N, C, T, L]",
		op = .NormalizeTime,
		shapes = {{2, 3, 7, 4}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_normalize_time(x[0])},
	},
	{
		name = "permute_lob",
		op = .PermuteLOB,
		shapes = {{2, 3, 5, 4}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_permute_lob(
				x[0],
				2,
				3,
				5,
				4,
				context.allocator,
			)},
	},
	{
		name = "top_k_mask (probabilities · mask)",
		op = .TopKMask,
		shapes = {{7, 6, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			probs := t.tensor_softmax(t.tensor_scale(x[0], 3))
			return t.tensor_mul(probs, t.tensor_top_k_mask(probs, 2))
		},
	},
	{
		name = "reparameterize (seeded noise)",
		op = .Reparameterize,
		shapes = {{6, 5, 1, 1}, {6, 5, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			rand.reset(55) // the same noise on the CPU and the GPU
			return nn.reparameterize(x[0], x[1], context.allocator)
		},
	},
	// ---- phase 5: products large enough for the register-tiled kernel ----
	{
		name = "matmul 70x90 @ 90x65 (edge tiles)",
		op = .MatMul,
		shapes = {{70, 90, 1, 1}, {90, 65, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_matmul(x[0], x[1])},
	},
	{
		name = "matmul 200x40 @ 40x130, then a second product",
		op = .MatMul,
		shapes = {{200, 40, 1, 1}, {40, 130, 1, 1}, {130, 33, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_matmul(
				t.tensor_tanh(t.tensor_matmul(x[0], x[1])),
				x[2],
			)},
	},
	{
		name = "attention 3 x (40 queries, 72 keys, d 36)",
		op = .ScaledDotProductAttention,
		shapes = {{3, 40, 36, 1}, {3, 72, 36, 1}, {3, 72, 34, 1}},
		storage = {{1, 4320}, {1, 7776}, {1, 7344}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_scaled_dot_product_attention(
				x[0],
				x[1],
				x[2],
			)},
	},
	// x [B, S, in], h_0, c_0 [B, H], W_ih [in, 4H], W_hh [H, 4H], bias [1, 4H]
	{
		name = "lstm 3 sequences x 5 steps, hidden 6",
		op = .LSTM,
		shapes = {
			{3, 5, 4, 1},
			{3, 6, 1, 1},
			{3, 6, 1, 1},
			{4, 24, 1, 1},
			{6, 24, 1, 1},
			{1, 24, 1, 1},
		},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_lstm(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				x[5],
			)},
	},
	{
		name = "lstm, fixed initial state",
		op = .LSTM,
		shapes = {
			{2, 4, 3, 1},
			{2, 5, 1, 1},
			{2, 5, 1, 1},
			{3, 20, 1, 1},
			{5, 20, 1, 1},
			{1, 20, 1, 1},
		},
		no_grad = {1, 2},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_lstm(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				x[5],
			)},
	},
	{
		name = "lstm hidden 300 (more units than threads)",
		op = .LSTM,
		shapes = {
			{2, 3, 5, 1},
			{2, 300, 1, 1},
			{2, 300, 1, 1},
			{5, 1200, 1, 1},
			{300, 1200, 1, 1},
			{1, 1200, 1, 1},
		},
		tol = 1e-5,
		// W_hh scaled to the size layer initialisation gives (about 1/√H): with
		// entries in (-1, 1) the gate inputs reach ±10 and f32 rounding of the
		// 300-term sums alone exceeds the tolerance
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_lstm(
				x[0],
				x[1],
				x[2],
				x[3],
				t.tensor_scale(x[4], 0.05),
				x[5],
			)},
	},
	{
		name = "lstm 4 sequences x 40 steps",
		op = .LSTM,
		shapes = {
			{4, 40, 3, 1},
			{4, 8, 1, 1},
			{4, 8, 1, 1},
			{3, 32, 1, 1},
			{8, 32, 1, 1},
			{1, 32, 1, 1},
		},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_lstm(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				x[5],
			)},
	},
	// x [B, S, in], h_0 [B, H], W_ih [in, 3H], W_hh [H, 3H], bias [1, 3H], bias_hn [1, H]
	{
		name = "gru 3 sequences x 5 steps, hidden 6",
		op = .GRU,
		shapes = {
			{3, 5, 4, 1},
			{3, 6, 1, 1},
			{4, 18, 1, 1},
			{6, 18, 1, 1},
			{1, 18, 1, 1},
			{1, 6, 1, 1},
		},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gru(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				x[5],
			)},
	},
	{
		name = "gru without bias_hn, fixed initial state",
		op = .GRU,
		shapes = {{2, 4, 3, 1}, {2, 5, 1, 1}, {3, 15, 1, 1}, {5, 15, 1, 1}, {1, 15, 1, 1}},
		no_grad = {1},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gru(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				nil,
			)},
	},
	{
		name = "gru hidden 300 (more units than threads)",
		op = .GRU,
		shapes = {
			{2, 3, 5, 1},
			{2, 300, 1, 1},
			{5, 900, 1, 1},
			{300, 900, 1, 1},
			{1, 900, 1, 1},
			{1, 300, 1, 1},
		},
		tol = 1e-5,
		// W_hh at initialisation scale (see the LSTM case)
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gru(
				x[0],
				x[1],
				x[2],
				t.tensor_scale(x[3], 0.05),
				x[4],
				x[5],
			)},
	},
	{
		name = "gru 4 sequences x 40 steps",
		op = .GRU,
		shapes = {
			{4, 40, 3, 1},
			{4, 8, 1, 1},
			{3, 24, 1, 1},
			{8, 24, 1, 1},
			{1, 24, 1, 1},
			{1, 8, 1, 1},
		},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_gru(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				x[5],
			)},
	},
	// x [B, S, in], h_0 [B, H] (here stored [1, B·H]), W_ih [in, H], W_hh [H, H], bias [1, H]
	{
		name = "rnn 2 sequences x 4 steps, hidden 5",
		op = .RNN,
		shapes = {{2, 4, 3, 1}, {1, 10, 1, 1}, {3, 5, 1, 1}, {5, 5, 1, 1}, {1, 5, 1, 1}},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_rnn(
				x[0],
				x[1],
				x[2],
				x[3],
				x[4],
			)},
	},
	{
		name = "rnn hidden 300 (more units than threads)",
		op = .RNN,
		shapes = {{2, 3, 5, 1}, {2, 300, 1, 1}, {5, 300, 1, 1}, {300, 300, 1, 1}, {1, 300, 1, 1}},
		no_grad = {1},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_rnn(
				x[0],
				x[1],
				x[2],
				t.tensor_scale(x[3], 0.05),
				x[4],
			)},
	},
	{
		name = "rnn 4 sequences x 40 steps",
		op = .RNN,
		shapes = {{4, 40, 3, 1}, {4, 8, 1, 1}, {3, 8, 1, 1}, {8, 8, 1, 1}, {1, 8, 1, 1}},
		tol = 1e-5,
		// W_hh scaled so that 40 steps of tanh do not saturate
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_rnn(
				x[0],
				x[1],
				x[2],
				t.tensor_scale(x[3], 0.3),
				x[4],
			)},
	},
}

// _const_like: a constant tensor shaped like x, freed with the graph
_const_like :: proc(x: ^t.Tensor, v: f64) -> ^t.Tensor {
	m := l.matrix_new(f64, x.data.rows, x.data.cols)
	for &e in m.data {e = v}
	c := t.tensor_new(m)
	c.shape = x.shape
	c.owned_by_graph = true
	return c
}

CE_TARGETS := [10]int{3, 0, 6, 6, 1, 2, 5, 4, 0, 3}
EMB_IDS := [10]f64{4, 1, 4, 10, 0, 4, 7, 1, 1, 9}

// additive masks for [5, 6] score blocks: shared, and one block per sequence
_shared_mask :: proc() -> []f64 {
	m := make([]f64, 30, context.temp_allocator)
	for i in 0 ..< 5 {for j in 0 ..< 6 {if (i + 2 * j) % 4 == 0 {m[i * 6 + j] = -1e9} else {m[i * 6 + j] = 0.1 * f64(j - i)}}}
	return m
}

// one [5, 6] block per sequence and head (2 sequences, 2 heads)
_head_mask :: proc() -> []f64 {
	m := make([]f64, 120, context.temp_allocator)
	for blk in 0 ..< 4 {for i in 0 ..< 5 {for j in 0 ..< 6 {if j > i + blk % 3 {m[(blk * 5 + i) * 6 + j] = -1e9} else {m[(blk * 5 + i) * 6 + j] = 0.05 * f64(blk)}}}}
	return m
}

_batch_mask :: proc() -> []f64 {
	m := make([]f64, 60, context.temp_allocator)
	for b in 0 ..< 2 {for i in 0 ..< 5 {for j in 0 ..< 6 {if j > i + b {m[(b * 5 + i) * 6 + j] = -1e9}}}}
	return m
}

// _compare_case: CPU vs GPU (kernel enabled), and CPU vs GPU with the op's
// kernel disabled (the CPU fallback)
_compare_case :: proc(
	c: Op_Case,
	fallback: bool,
	cpu_backward := false,
) -> (
	worst: f64,
	fallbacks: int,
) {
	values := make([][]f64, len(c.shapes), context.temp_allocator)
	for shape, i in c.shapes {
		n := shape[0] * shape[1] * shape[2] * shape[3]
		values[i] = make([]f64, n, context.temp_allocator)
		for &v in values[i] {
			v = 0.5 + 1.5 * rand.float64() if c.positive else 2 * rand.float64() - 1
			v = f64(f32(v)) // the device stores f32: compare on the same inputs
		}
	}
	_, n_out := _run_case(c, values, nil, false)
	upstream := make([]f64, n_out, context.temp_allocator)
	for &v in upstream {v = f64(f32(2 * rand.float64() - 1))}

	ref, _ := _run_case(c, values, upstream, false)
	if fallback {gpu.set_kernel_enabled(c.op, false)}
	gpu.reset_stats()
	got, _ := _run_case(c, values, upstream, true, cpu_backward)
	st := gpu.stats()
	if fallback {gpu.set_kernel_enabled(c.op, true)}
	for n in st.forward_fallbacks {fallbacks += n}
	for n in st.backward_fallbacks {fallbacks += n}
	worst = _max_rel(got.out, ref.out)
	for g, i in got.grads {worst = max(worst, _max_rel(g, ref.grads[i]))}
	return
}

gpu_op_parity_test :: proc() {
	fmt.println(
		"gpu_op_parity_test: every GPU kernel against the CPU engine (forward and backward)",
	)
	rand.reset(1)
	for c in op_cases {
		worst, fallbacks := _compare_case(c, false)
		check(worst <= c.tol, "%s: max relative error %.2e > %.0e", c.name, worst, c.tol)
		check(fallbacks == 0, "%s: %d CPU fallbacks with the kernel enabled", c.name, fallbacks)
		fmt.printfln("    %-34s max rel err %.2e", c.name, worst)
	}
}

gpu_fallback_op_test :: proc() {
	fmt.println(
		"gpu_fallback_op_test: the same ops with their kernel disabled (CPU fallback on device tensors)",
	)
	rand.reset(2)
	for c in op_cases {
		worst, fallbacks := _compare_case(c, true)
		check(
			worst <= c.tol,
			"%s (fallback): max relative error %.2e > %.0e",
			c.name,
			worst,
			c.tol,
		)
		// without their kernel, multi-head attention and GELU run as the unfused
		// ops, which have kernels: there is no CPU fallback to count
		if c.op != .MultiHeadAttention && c.op != .Gelu {
			check(fallbacks > 0, "%s (fallback): the fallback did not run", c.name)
		}
	}
	// an op without a GPU kernel at all (Mamba's selective scan stays on the
	// CPU), on device inputs
	ssm := Op_Case {
		name = "ssm (no kernel)",
		op = .SSM,
		// x [B, S, d_model], h_0 [B, d_model·d_state], A [d_model, d_state],
		// B, C [B, S, d_state], Delta [B, S, d_model] (positive), D [1, d_model]
		shapes = {
			{2, 4, 3, 1},
			{2, 6, 1, 1},
			{3, 2, 1, 1},
			{2, 4, 2, 1},
			{2, 4, 2, 1},
			{2, 4, 3, 1},
			{1, 3, 1, 1},
		},
		tol = 1e-5,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {
			return t.tensor_ssm(x[0], x[1], x[2], x[3], x[4], t.tensor_softplus(x[5]), x[6])
		},
	}
	for c in ([]Op_Case{ssm}) {
		worst, fallbacks := _compare_case(c, false)
		check(worst <= c.tol, "%s: max relative error %.2e > %.0e", c.name, worst, c.tol)
		check(fallbacks > 0, "%s: no fallback counted", c.name)
		fmt.printfln("    %-34s max rel err %.2e (%d fallbacks)", c.name, worst, fallbacks)
	}
}

// every matmul-backed case (products, attention, convolutions) with each matmul
// kernel forced: the register-tiled kernel also on products smaller than a tile
gpu_matmul_kernel_test :: proc() {
	fmt.println("gpu_matmul_kernel_test: matmul-backed ops with each matmul kernel forced")
	defer gpu.set_matmul_kernel(.Auto)
	rand.reset(4)
	for kernel in ([]gpu.Matmul_Kernel{.Tiled, .Simple}) {
		gpu.set_matmul_kernel(kernel)
		worst := 0.0
		count := 0
		for c in op_cases {
			#partial switch c.op {
			case .MatMul,
			     .Linear,
			     .ScaledDotProductAttention,
			     .MaskedScaledDotProductAttention,
			     .FlashAttention,
			     .MultiHeadAttention,
			     .Conv2d,
			     .Conv1d:
			case:
				continue
			}
			w, _ := _compare_case(c, false)
			check(
				w <= c.tol,
				"%s (%v kernel): max relative error %.2e > %.0e",
				c.name,
				kernel,
				w,
				c.tol,
			)
			worst = max(worst, w)
			count += 1
		}
		fmt.printfln("    %-7v kernel: %d cases, max rel err %.2e", kernel, count, worst)
	}
}

// Linear layers on one input recorded one after the other (attention's query,
// key and value projections): what grouping must handle
Group_Case :: struct {
	name:                   string,
	rows, in_f:             int,
	outs:                   []int, // output width of each layer
	x_grad, w_grad, b_grad: bool, // which of x, the weights, the biases are trained
	shared:                 int, // this layer reuses layer 0's weights and bias (-1: none)
	saved:                  int, // launches grouping saves (forward and backward)
}

group_cases := [?]Group_Case {
	{"3 layers, all trained", 40, 24, {20, 20, 20}, true, true, true, -1, 8},
	{"3 layers, register-tiled sizes", 70, 48, {40, 40, 40}, true, true, true, -1, 8},
	{"fixed weights", 40, 24, {20, 20, 20}, true, false, true, -1, 6},
	{"input without gradient, fixed biases", 40, 24, {20, 20, 20}, false, true, false, -1, 4},
	{"only the biases trained", 40, 24, {20, 20, 20}, false, false, true, -1, 4},
	{"4 layers (a group of 3, then 1)", 40, 24, {20, 20, 20, 20}, true, true, true, -1, 8},
	{"2 layers", 40, 24, {20, 20}, true, true, true, -1, 4},
	{"layer 1 shares layer 0's weights", 40, 24, {20, 20, 20}, true, true, true, 1, 5},
	{"widths 20, 20, 12", 40, 24, {20, 20, 12}, true, true, true, -1, 4},
}

// _group_run: the layers' outputs and the gradients of x, the weights and the
// biases (loss Σ concat(outputs) ⊙ c), and the launches on the GPU
_group_run :: proc(c: Group_Case, on_gpu, grouped: bool) -> (vals: [dynamic]f64, launches: int) {
	gpu.set_linear_grouping(grouped)
	rand.reset(31)
	rnd :: proc(rows, cols: int, rg: bool) -> ^t.Tensor {
		m := l.matrix_new(f64, rows, cols)
		for &v in m.data {v = f64(f32(2 * rand.float64() - 1))}
		return t.tensor_new(m, rg)
	}
	vals = make([dynamic]f64, context.temp_allocator)
	x := rnd(c.rows, c.in_f, c.x_grad)
	ws := make([dynamic]^t.Tensor, context.temp_allocator)
	bs := make([dynamic]^t.Tensor, context.temp_allocator)
	leaves := make([dynamic]^t.Tensor, context.temp_allocator)
	append(&leaves, x)
	width := 0
	for o, i in c.outs {
		if i == c.shared {
			append(&ws, ws[0])
			append(&bs, bs[0])
		} else {
			append(&ws, rnd(c.in_f, o, c.w_grad))
			append(&bs, rnd(1, o, c.b_grad))
			append(&leaves, ws[i], bs[i])
		}
		width += o
	}
	// stored as the concatenation is: [1, rows · width]
	cm := rnd(1, c.rows * width, false)
	cm.shape = {c.rows, width, 1, 1}
	if on_gpu {
		gpu.to_device(leaves[:])
		gpu.to_device(cm)
	}
	gpu.reset_stats()
	outs := make([dynamic]^t.Tensor, context.temp_allocator)
	for _, i in c.outs {append(&outs, t.tensor_linear(x, ws[i], bs[i]))}
	loss := t.tensor_sum(t.tensor_mul(t.tensor_concat(outs[:], 1), cm))
	t.tensor_backward(loss)
	for o in outs {append(&vals, ..t.tensor_values(o))}
	for p in leaves {if p.requires_grad {append(&vals, ..t.tensor_grad_values(p))}}
	launches = gpu.stats().dispatches
	t.tensor_free_graph(loss)
	for p in leaves {t.tensor_free(p)}
	t.tensor_free(cm)
	return
}

gpu_linear_group_test :: proc() {
	fmt.println(
		"gpu_linear_group_test: linear layers on one input, grouped vs one launch per layer vs the CPU",
	)
	defer gpu.set_linear_grouping(true)
	defer gpu.set_matmul_kernel(.Auto)
	for kernel in ([]gpu.Matmul_Kernel{.Auto, .Simple, .Tiled}) {
		gpu.set_matmul_kernel(kernel)
		for c in group_cases {
			cpu, _ := _group_run(c, false, false)
			single, n_single := _group_run(c, true, false)
			grouped, n_grouped := _group_run(c, true, true)
			identical := len(grouped) == len(single)
			for v, i in grouped {if identical && v != single[i] {identical = false}}
			err := _max_rel(grouped[:], cpu[:])
			if kernel == .Auto {
				fmt.printfln(
					"    %-38s %3d -> %3d launches, max rel err %.2e, same values as ungrouped: %v",
					c.name,
					n_single,
					n_grouped,
					err,
					identical,
				)
			}
			check(
				identical,
				"%s (%v kernel): grouped values differ from ungrouped",
				c.name,
				kernel,
			)
			check(err < 1e-5, "%s (%v kernel): max rel err %.2e to the CPU", c.name, kernel, err)
			check(
				n_single - n_grouped == c.saved,
				"%s (%v kernel): grouping saved %d launches, want %d",
				c.name,
				kernel,
				n_single - n_grouped,
				c.saved,
			)
		}
	}
}

// f32 matmul time of the two kernels (printed, not checked: timings vary)
gpu_matmul_benchmark :: proc() {
	fmt.println("gpu_matmul_benchmark: C = A @ B, 16x16 kernel vs register-tiled kernel")
	defer gpu.set_matmul_kernel(.Auto)
	for n in ([]int{256, 512, 1024, 2048}) {
		gpu.set_matmul_kernel(.Simple)
		a := gpu.benchmark_matmul(n, n, n)
		gpu.set_matmul_kernel(.Tiled)
		b := gpu.benchmark_matmul(n, n, n)
		gf := 2 * f64(n) * f64(n) * f64(n) / 1e6
		fmt.printfln(
			"    %4d x %4d x %4d: 16x16 %8.3f ms (%5.0f GFLOP/s)   tiled %8.3f ms (%5.0f GFLOP/s)   %.1fx",
			n,
			n,
			n,
			a,
			gf / a,
			b,
			gf / b,
			a / b,
		)
	}
}
