package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"

// Regression tests for wotan/tensor issues found while writing the tensor guide:
// tensor_top_k_mask chose the wrong entries (it seeded the selection with
// columns 0..k-1 and could pick one twice) and ignored the active graph scope;
// tensor_masked_scaled_dot_product_attention indexed a nil mask; batch norm and
// conv2d recorded no graph when only the bias required gradients, nor did the
// SSM when only h_0 did. Reference values: torch 2.1.0 (torch.topk).

@(private = "file")
tf_close :: proc(a, b: []f64, tol: f64) -> bool {
	if len(a) != len(b) do return false
	for i in 0 ..< len(a) {if math.abs(a[i] - b[i]) > tol do return false}
	return true
}

@(private = "file")
tf_mat :: proc(vals: []f64, rows, cols: int, rg: bool, allocator: mem.Allocator) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols, allocator)
	copy(m.data, vals)
	return t.tensor_new(m, rg, allocator)
}

// Inputs that stay constant in the "only one input trainable" gradient checks
@(private = "file")
tf_const: [6]^t.Tensor

tensor_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== tensor regression tests ===")
	context.allocator = allocator
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	{ 	// top-k: torch.topk([[5, 1, 3], [0.2, 0.9, 0.4]], 2) -> [[0, 2], [1, 2]]
		p := tf_mat({5, 1, 3, 0.2, 0.9, 0.4}, 2, 3, false, allocator)
		m2 := t.tensor_top_k_mask(p, 2)
		m1 := t.tensor_top_k_mask(p, 1)
		m3 := t.tensor_top_k_mask(p, 3)
		good :=
			tf_close(m2.data.data, {1, 0, 1, 0, 1, 1}, 0) &&
			tf_close(m1.data.data, {1, 0, 0, 0, 1, 0}, 0) &&
			tf_close(m3.data.data, {1, 1, 1, 1, 1, 1}, 0) &&
			m2.shape == {2, 3, 1, 1}
		report("tensor_top_k_mask k = 1, 2, 3: torch.topk indices", good, &passed, &total)
		t.tensor_free(m1)
		t.tensor_free(m2)
		t.tensor_free(m3)
		t.tensor_free(p)
	}
	{ 	// top-k inside a graph scope: the mask lives in the scope, not on the heap
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		a := mem.tracking_allocator(&track)
		p := tf_mat({5, 1, 3, 0.2, 0.9, 0.4}, 2, 3, false, a)
		before := len(track.allocation_map)
		scope: t.Graph_Scope
		t.graph_begin(&scope)
		m := t.tensor_top_k_mask(t.tensor_softmax(p), 2, a)
		good := tf_close(m.data.data, {1, 0, 1, 0, 1, 1}, 0)
		t.graph_end(&scope)
		t.graph_scope_destroy(&scope)
		good = good && len(track.allocation_map) == before
		t.tensor_free(p)
		report(
			"tensor_top_k_mask in a graph scope: no heap allocation",
			good && len(track.allocation_map) == 0,
			&passed,
			&total,
		)
	}
	{ 	// masked attention with a nil mask == unmasked attention, forward and backward
		q := tf_mat({0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.7, -0.1}, 1, 8, true, allocator)
		k := tf_mat({0.1, 0.4, -0.3, 0.2, 0.6, -0.5, 0.2, 0.3}, 1, 8, true, allocator)
		v := tf_mat({1, 2, -1, 0.5, 0.25, -2, 3, 1}, 1, 8, true, allocator)
		for x in ([]^t.Tensor{q, k, v}) {x.shape = {2, 2, 2, 1}} 	// [batch, seq, d]
		grads :: proc(q, k, v: ^t.Tensor, masked: bool) -> (out: [8]f64, g: [24]f64) {
			for x in ([]^t.Tensor{q, k, v}) {t.tensor_zero_grad(x)}
			o :=
				t.tensor_masked_scaled_dot_product_attention(q, k, v, nil) if masked else t.tensor_scaled_dot_product_attention(q, k, v)
			copy(out[:], o.data.data)
			w := tf_mat({1, -1, 2, 0.5, -0.5, 1, 0.25, 3}, 1, 8, false, context.allocator)
			w.shape = o.shape
			w.owned_by_graph = true
			loss := t.tensor_sum(t.tensor_mul(o, w))
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
			copy(g[0:8], q.grad.data)
			copy(g[8:16], k.grad.data)
			copy(g[16:24], v.grad.data)
			return
		}
		o1, g1 := grads(q, k, v, false)
		o2, g2 := grads(q, k, v, true)
		report(
			"masked attention, nil mask: equals unmasked (output, dQ, dK, dV)",
			tf_close(o1[:], o2[:], 1e-14) && tf_close(g1[:], g2[:], 1e-14),
			&passed,
			&total,
		)
		t.tensor_free(q)
		t.tensor_free(k)
		t.tensor_free(v)
	}

	// Only one input trainable: the op must still record the graph, and the
	// gradient must match central differences
	gc_p0, gc_t0 := gc_passed, gc_total
	{ 	// batch norm, bias only: x [2, 2, 1, 2]
		gcl_in[0] = gcl_rand(1, 2, {1, 2, 1, 1}, 301, true, allocator) // bias
		tf_const[0] = gcl_rand(1, 8, {2, 2, 1, 2}, 302, false, allocator) // x
		tf_const[1] = gcl_rand(1, 2, {1, 2, 1, 1}, 303, false, allocator) // gamma
		tf_const[2] = tf_mat({0, 0}, 1, 2, false, allocator) // running mean
		tf_const[3] = tf_mat({1, 1}, 1, 2, false, allocator) // running var
		gcl_check("batch_norm_2d, only the bias trainable", 1, proc() -> ^t.Tensor {
				return gcl_weighted_sum(
					t.tensor_batch_norm_2d(
						tf_const[0],
						tf_const[1],
						gcl_in[0],
						tf_const[2],
						tf_const[3],
						false,
					),
				)
			}, allocator)
		gcl_free_inputs(1)
		for i in 0 ..< 4 {t.tensor_free(tf_const[i])}
	}
	{ 	// conv2d, bias only: x [1, 1, 4, 4], weight [2, 1, 3, 3], stride 2, pad 1
		gcl_in[0] = gcl_rand(1, 2, {1, 2, 1, 1}, 311, true, allocator) // bias
		tf_const[0] = gcl_rand(1, 16, {1, 1, 4, 4}, 312, false, allocator)
		tf_const[1] = gcl_rand(1, 18, {2, 1, 3, 3}, 313, false, allocator)
		gcl_check("conv2d, only the bias trainable", 1, proc() -> ^t.Tensor {
				return gcl_weighted_sum(t.tensor_conv2d(tf_const[0], tf_const[1], gcl_in[0], 2, 1))
			}, allocator)
		gcl_free_inputs(1)
		for i in 0 ..< 2 {t.tensor_free(tf_const[i])}
	}
	{ 	// SSM, h_0 only (shapes as in gradcheck_ops_test)
		Bn, S, dm, ds := 2, 3, 2, 3
		gcl_in[0] = gcl_rand(1, Bn * dm * ds, {Bn, dm, ds, 1}, 321, true, allocator, 0.5) // h_0
		tf_const[0] = gcl_rand(1, Bn * S * dm, {Bn, S, dm, 1}, 322, false, allocator) // x
		tf_const[1] = gcl_rand(dm, ds, {dm, ds, 1, 1}, 323, false, allocator) // A
		for i in 0 ..< len(tf_const[1].data.data) {tf_const[1].data.data[i] = -0.2 - 0.6 * abs(tf_const[1].data.data[i])}
		tf_const[2] = gcl_rand(1, Bn * S * ds, {Bn, S, ds, 1}, 324, false, allocator) // B
		tf_const[3] = gcl_rand(1, Bn * S * ds, {Bn, S, ds, 1}, 325, false, allocator) // C
		tf_const[4] = gcl_rand(1, Bn * S * dm, {Bn, S, dm, 1}, 326, false, allocator) // Delta
		for i in 0 ..< len(tf_const[4].data.data) {tf_const[4].data.data[i] = 0.1 + 0.5 * abs(tf_const[4].data.data[i])}
		tf_const[5] = gcl_rand(1, dm, {1, dm, 1, 1}, 327, false, allocator) // D
		gcl_check("ssm, only h_0 trainable", 1, proc() -> ^t.Tensor {
				return gcl_weighted_sum(
					t.tensor_ssm(
						tf_const[0],
						gcl_in[0],
						tf_const[1],
						tf_const[2],
						tf_const[3],
						tf_const[4],
						tf_const[5],
					),
				)
			}, allocator)
		gcl_free_inputs(1)
		for i in 0 ..< 6 {t.tensor_free(tf_const[i])}
	}
	gc_new_passed, gc_new_total := gc_passed - gc_p0, gc_total - gc_t0
	passed += gc_new_passed
	total += gc_new_total

	fmt.printf("tensor regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
