package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// Numerical gradient checks for the multi-input layer ops (MatMul, attention,
// recurrent cells, Conv2d). Every input is checked, and the loss is a fixed
// random weighting of the output so no gradient vanishes by symmetry.

GCL_MAX_INPUTS :: 7

gcl_in: [GCL_MAX_INPUTS]^t.Tensor
gcl_mask: []f64
gcl_conv_stride, gcl_conv_pad: int

// Deterministic pseudo-random values in [-1, 1)
gcl_fill :: proc(dst: []f64, seed: u64) {
	s := seed * 6364136223846793005 + 1442695040888963407
	for i in 0 ..< len(dst) {
		s = s * 6364136223846793005 + 1442695040888963407
		dst[i] = f64(s >> 11) / f64(u64(1) << 53) * 2.0 - 1.0
	}
}

gcl_rand :: proc(
	rows, cols: int,
	shape: [4]int,
	seed: u64,
	rg: bool,
	allocator: mem.Allocator,
	scale: f64 = 1.0,
) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols, allocator)
	gcl_fill(m.data, seed)
	for i in 0 ..< len(m.data) {m.data[i] *= scale}
	x := t.tensor_new(m, rg, allocator)
	x.shape = shape
	return x
}

// loss = sum(out * W) with W a fixed random tensor shaped like out
gcl_weighted_sum :: proc(out: ^t.Tensor) -> ^t.Tensor {
	W := gcl_rand(out.data.rows, out.data.cols, out.shape, 977, false, context.allocator)
	W.owned_by_graph = true
	return t.tensor_sum(t.tensor_mul(out, W))
}

gcl_check :: proc(
	name: string,
	n_inputs: int,
	f: proc() -> ^t.Tensor,
	allocator: mem.Allocator,
	h: f64 = 1e-6,
) {
	for k in 0 ..< n_inputs {t.tensor_zero_grad(gcl_in[k])}
	loss := f()
	t.tensor_backward(loss, allocator)
	t.tensor_free_graph(loss, allocator)

	max_err := 0.0
	worst_input := -1
	for k in 0 ..< n_inputs {
		x := gcl_in[k]
		for i in 0 ..< len(x.data.data) {
			orig := x.data.data[i]
			x.data.data[i] = orig + h
			lp := f(); fp := lp.data.data[0]; t.tensor_free_graph(lp, allocator)
			x.data.data[i] = orig - h
			lm := f(); fm := lm.data.data[0]; t.tensor_free_graph(lm, allocator)
			x.data.data[i] = orig

			num := (fp - fm) / (2 * h)
			an := x.grad.data[i]
			err := abs(num - an) / max(abs(num), abs(an), 1e-6)
			if err > max_err {
				max_err = err
				worst_input = k
			}
		}
	}

	gc_total += 1
	if max_err < GC_TOLERANCE {
		gc_passed += 1
		fmt.printf("OK   %-40s max_rel_err=%.2e\n", name, max_err)
	} else {
		fmt.printf("FAIL %-40s max_rel_err=%.2e (input %d)\n", name, max_err, worst_input)
	}
}

gcl_free_inputs :: proc(n: int) {
	for k in 0 ..< n {
		t.tensor_free(gcl_in[k])
		gcl_in[k] = nil
	}
}

gradcheck_layers_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Autograd Layer Gradient Checks ===")
	context.allocator = allocator
	gc_passed, gc_total = 0, 0

	B, S, I, H := 2, 3, 2, 3

	// 1. Standard 2D matmul, gradient wrt both operands
	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 1, true, allocator)
	gcl_in[1] = gcl_rand(4, 2, {4, 2, 1, 1}, 2, true, allocator)
	gcl_check("matmul 2D [3,4]@[4,2]", 2, proc() -> ^t.Tensor {
		return gcl_weighted_sum(t.tensor_matmul(gcl_in[0], gcl_in[1]))
	}, allocator)
	gcl_free_inputs(2)

	// 2. Flattened [1, N*D] matmul, gradient wrt input and weight
	gcl_in[0] = gcl_rand(1, 3 * 4, {1, 3, 4, 1}, 3, true, allocator)
	gcl_in[1] = gcl_rand(4, 5, {4, 5, 1, 1}, 4, true, allocator)
	gcl_check("matmul flattened [1,3*4]@[4,5]", 2, proc() -> ^t.Tensor {
		return gcl_weighted_sum(t.tensor_matmul(gcl_in[0], gcl_in[1]))
	}, allocator)
	gcl_free_inputs(2)

	// 3. Scaled dot-product attention (seq_q != seq_k, d_k != d_v)
	gcl_in[0] = gcl_rand(1, 2 * 3 * 4, {2, 3, 4, 1}, 5, true, allocator) // Q [B, Sq, dk]
	gcl_in[1] = gcl_rand(1, 2 * 4 * 4, {2, 4, 4, 1}, 6, true, allocator) // K [B, Sk, dk]
	gcl_in[2] = gcl_rand(1, 2 * 4 * 3, {2, 4, 3, 1}, 7, true, allocator) // V [B, Sk, dv]
	gcl_check("scaled_dot_product_attention", 3, proc() -> ^t.Tensor {
		return gcl_weighted_sum(
			t.tensor_scaled_dot_product_attention(gcl_in[0], gcl_in[1], gcl_in[2], context.allocator),
		)
	}, allocator)
	gcl_free_inputs(3)

	// 4. Masked (causal) attention
	gcl_mask = make([]f64, 3 * 3, allocator)
	defer delete(gcl_mask, allocator)
	for i in 0 ..< 3 {
		for j in 0 ..< 3 {
			if j > i {gcl_mask[i * 3 + j] = -1e9}
		}
	}
	gcl_in[0] = gcl_rand(1, 2 * 3 * 4, {2, 3, 4, 1}, 8, true, allocator)
	gcl_in[1] = gcl_rand(1, 2 * 3 * 4, {2, 3, 4, 1}, 9, true, allocator)
	gcl_in[2] = gcl_rand(1, 2 * 3 * 2, {2, 3, 2, 1}, 10, true, allocator)
	gcl_check("masked attention (causal)", 3, proc() -> ^t.Tensor {
		return gcl_weighted_sum(
			t.tensor_masked_scaled_dot_product_attention(
				gcl_in[0],
				gcl_in[1],
				gcl_in[2],
				gcl_mask,
				context.allocator,
			),
		)
	}, allocator)
	gcl_free_inputs(3)

	// 5-7. Recurrent cells: x [B,S,I], h_0 [B,H], w_ih [I,kH], w_hh [H,kH], bias [1,kH]
	gcl_in[0] = gcl_rand(1, B * S * I, {B, S, I, 1}, 11, true, allocator)
	gcl_in[1] = gcl_rand(B, H, {B, H, 1, 1}, 12, true, allocator, 0.5)
	gcl_in[2] = gcl_rand(I, H, {I, H, 1, 1}, 13, true, allocator, 0.5)
	gcl_in[3] = gcl_rand(H, H, {H, H, 1, 1}, 14, true, allocator, 0.5)
	gcl_in[4] = gcl_rand(1, H, {1, H, 1, 1}, 15, true, allocator, 0.5)
	gcl_check("rnn", 5, proc() -> ^t.Tensor {
		return gcl_weighted_sum(
			t.tensor_rnn(gcl_in[0], gcl_in[1], gcl_in[2], gcl_in[3], gcl_in[4], context.allocator),
		)
	}, allocator)
	gcl_free_inputs(5)

	gcl_in[0] = gcl_rand(1, B * S * I, {B, S, I, 1}, 16, true, allocator)
	gcl_in[1] = gcl_rand(B, H, {B, H, 1, 1}, 17, true, allocator, 0.5)
	gcl_in[2] = gcl_rand(I, 3 * H, {I, 3 * H, 1, 1}, 18, true, allocator, 0.5)
	gcl_in[3] = gcl_rand(H, 3 * H, {H, 3 * H, 1, 1}, 19, true, allocator, 0.5)
	gcl_in[4] = gcl_rand(1, 3 * H, {1, 3 * H, 1, 1}, 20, true, allocator, 0.5)
	gcl_check("gru", 5, proc() -> ^t.Tensor {
		return gcl_weighted_sum(
			t.tensor_gru(gcl_in[0], gcl_in[1], gcl_in[2], gcl_in[3], gcl_in[4], context.allocator),
		)
	}, allocator)
	gcl_free_inputs(5)

	gcl_in[0] = gcl_rand(1, B * S * I, {B, S, I, 1}, 21, true, allocator)
	gcl_in[1] = gcl_rand(B, H, {B, H, 1, 1}, 22, true, allocator, 0.5)
	gcl_in[2] = gcl_rand(B, H, {B, H, 1, 1}, 23, true, allocator, 0.5)
	gcl_in[3] = gcl_rand(I, 4 * H, {I, 4 * H, 1, 1}, 24, true, allocator, 0.5)
	gcl_in[4] = gcl_rand(H, 4 * H, {H, 4 * H, 1, 1}, 25, true, allocator, 0.5)
	gcl_in[5] = gcl_rand(1, 4 * H, {1, 4 * H, 1, 1}, 26, true, allocator, 0.5)
	gcl_check("lstm", 6, proc() -> ^t.Tensor {
		return gcl_weighted_sum(
			t.tensor_lstm(
				gcl_in[0],
				gcl_in[1],
				gcl_in[2],
				gcl_in[3],
				gcl_in[4],
				gcl_in[5],
				context.allocator,
			),
		)
	}, allocator)
	gcl_free_inputs(6)

	// 8. Conv2d with bias: input [N,C,H,W], weight [C_out,C_in,kH,kW]
	conv_cases := [][2]int{{1, 1}, {2, 0}} // {stride, pad}
	for cc in conv_cases {
		gcl_conv_stride, gcl_conv_pad = cc[0], cc[1]
		gcl_in[0] = gcl_rand(1, 2 * 2 * 5 * 5, {2, 2, 5, 5}, 27, true, allocator)
		// Same layout as nn.conv2d_layer_new: data [C_out, C_in*kH*kW], shape [C_out, C_in, kH, kW]
		gcl_in[1] = gcl_rand(3, 2 * 3 * 3, {3, 2, 3, 3}, 28, true, allocator, 0.5)
		gcl_in[2] = gcl_rand(1, 3, {3, 1, 1, 1}, 29, true, allocator)
		name := fmt.tprintf("conv2d stride=%d pad=%d", cc[0], cc[1])
		gcl_check(name, 3, proc() -> ^t.Tensor {
			return gcl_weighted_sum(
				t.tensor_conv2d(
					gcl_in[0],
					gcl_in[1],
					gcl_in[2],
					gcl_conv_stride,
					gcl_conv_pad,
					context.allocator,
				),
			)
		}, allocator)
		gcl_free_inputs(3)
	}

	fmt.printf("Layer gradcheck summary: %d/%d passed\n", gc_passed, gc_total)
}
