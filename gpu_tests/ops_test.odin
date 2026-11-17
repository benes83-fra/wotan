package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
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
_run_case :: proc(
	c: Op_Case,
	values: [][]f64,
	upstream: []f64,
	on_gpu: bool,
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
		xs[i] = t.tensor_new(m, true)
		xs[i].shape = shape
		if on_gpu {gpu.to_device(xs[i])}
	}
	out := c.run(xs)
	out_numel = t.tensor_numel(out)
	if upstream != nil {
		t.tensor_backward_with_grad(out, upstream)
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
}

// _compare_case: CPU vs GPU (kernel enabled), and CPU vs GPU with the op's
// kernel disabled (the CPU fallback)
_compare_case :: proc(c: Op_Case, fallback: bool) -> (worst: f64, fallbacks: int) {
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
	got, _ := _run_case(c, values, upstream, true)
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
		check(fallbacks > 0, "%s (fallback): the fallback did not run", c.name)
	}
	// ops without a GPU kernel at all, on device inputs
	softmax := Op_Case {
		name = "softmax (no kernel)",
		op = .Softmax,
		shapes = {{8, 12, 1, 1}},
		tol = 1e-6,
		run = proc(x: []^t.Tensor) -> ^t.Tensor {return t.tensor_softmax(x[0])},
	}
	layer_norm := Op_Case {
		name = "layer_norm (no kernel)",
		op = .LayerNorm,
		shapes = {{6, 16, 1, 1}, {1, 16, 1, 1}, {1, 16, 1, 1}},
		tol = 1e-5,
		run = proc(
			x: []^t.Tensor,
		) -> ^t.Tensor {return t.tensor_layer_norm(t.tensor_relu(x[0]), x[1], x[2])},
	}
	for c in ([]Op_Case{softmax, layer_norm}) {
		worst, fallbacks := _compare_case(c, false)
		check(worst <= c.tol, "%s: max relative error %.2e > %.0e", c.name, worst, c.tol)
		check(fallbacks > 0, "%s: no fallback counted", c.name)
		fmt.printfln("    %-34s max rel err %.2e (%d fallbacks)", c.name, worst, fallbacks)
	}
}
