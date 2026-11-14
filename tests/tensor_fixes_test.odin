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
// SSM when only h_0 did. Softmax over the logical last dim, the KL term (sum
// over latent dims, mean over the batch, clamp gradient), dropout in evaluation
// mode, freeing graphs with shared nodes in one call, tensor_detach in a scope.
// Reference values: torch 2.1.0.

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

	// Softmax over the logical last dim (torch.softmax(x, dim = -1), torch 2.1.0):
	// forward and d/dx of sum(softmax(x) * w), whatever the storage
	SM_X := [12]f64{0.5, -1.0, 2.0, 0.1, 0.3, -0.4, 1.2, 0.7, -0.6, 0.9, 0.0, 0.25}
	SM_W := [12]f64{1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5, 1.5, -3.0, 0.75, 1.0}
	SM2_OUT := [6]f64 {
		0.1752903921400367,
		0.039112573270687456,
		0.7855970345892759,
		0.3536151147265019,
		0.4319064764540744,
		0.2144784088194238,
	}
	SM2_DX := [6]f64 {
		0.08942198104727624,
		-0.09738497886019377,
		0.007962997812917543,
		0.8194823525275616,
		-0.7267079001755491,
		-0.09277445235201262,
	}
	SM3_OUT := [12]f64 {
		0.8175744761936437,
		0.18242552380635632,
		0.8698915256370021,
		0.13010847436299786,
		0.6681877721681662,
		0.33181222783183395,
		0.6224593312018546,
		0.37754066879814546,
		0.18242552380635632,
		0.8175744761936437,
		0.4378234991142019,
		0.5621765008857981,
	}
	SM3_DX := [12]f64 {
		0.44743935621099856,
		-0.4474393562109985,
		-0.2829506481548274,
		0.28295064815482746,
		-0.27714109161638634,
		0.27714109161638634,
		0.5875092805039862,
		-0.5875092805039863,
		0.6711590343164978,
		-0.6711590343164977,
		-0.061533520684399545,
		0.061533520684399635,
	}
	SM4_OUT := [12]f64 {
		0.1752903921400367,
		0.039112573270687456,
		0.7855970345892759,
		0.3536151147265019,
		0.4319064764540744,
		0.2144784088194238,
		0.5643883753749914,
		0.34231885365033493,
		0.09329277097467355,
		0.5185066867524072,
		0.2108090872061516,
		0.2706842260414412,
	}
	SM4_DX := [12]f64 {
		0.08942198104727624,
		-0.09738497886019377,
		0.007962997812917543,
		0.8194823525275616,
		-0.7267079001755491,
		-0.09277445235201262,
		0.5093286319021687,
		-0.5468737030076819,
		0.03754507110551342,
		-0.9713035298456529,
		0.3956315313888785,
		0.5756719984567743,
	}
	softmax_case :: proc(
		n, rows, cols: int,
		shape: [4]int,
		want_out, want_dx, x_vals, w_vals: []f64,
	) -> bool {
		x := tf_mat(x_vals[:n], rows, cols, true, context.allocator)
		x.shape = shape
		w := tf_mat(w_vals[:n], rows, cols, false, context.allocator)
		w.shape = shape
		w.owned_by_graph = true
		y := t.tensor_softmax(x)
		ok := tf_close(y.data.data, want_out, 1e-15) && y.shape == shape
		loss := t.tensor_sum(t.tensor_mul(y, w))
		t.tensor_backward(loss)
		ok = ok && tf_close(x.grad.data, want_dx, 1e-15)
		t.tensor_free_graph(loss)
		t.tensor_free(x)
		return ok
	}
	report(
		"softmax [2, 3] matrix: torch forward and backward",
		softmax_case(6, 2, 3, {2, 3, 1, 1}, SM2_OUT[:], SM2_DX[:], SM_X[:], SM_W[:]),
		&passed,
		&total,
	)
	report(
		"softmax [2, 3] stored as 1 x 6: per logical row (torch)",
		softmax_case(6, 1, 6, {2, 3, 1, 1}, SM2_OUT[:], SM2_DX[:], SM_X[:], SM_W[:]),
		&passed,
		&total,
	)
	report(
		"softmax [2, 3, 2, 1] sequence: over the features (torch)",
		softmax_case(12, 1, 12, {2, 3, 2, 1}, SM3_OUT[:], SM3_DX[:], SM_X[:], SM_W[:]),
		&passed,
		&total,
	)
	report(
		"softmax [1, 2, 2, 3] image: over W (torch)",
		softmax_case(12, 1, 12, {1, 2, 2, 3}, SM4_OUT[:], SM4_DX[:], SM_X[:], SM_W[:]),
		&passed,
		&total,
	)

	{ 	// KL: sum over the latent dim, mean over the batch; exp(log_var) clamped at
		// 1e10 with zero gradient past the clamp (torch.clamp), torch 2.1.0
		mu := tf_mat({0.3, -0.5, 1.2, 0.0, 0.8, -0.2}, 2, 3, true, allocator)
		lv := tf_mat({0.1, -0.7, 0.4, 30.0, -1.5, 0.2}, 2, 3, true, allocator)
		kl := t.tensor_kl_divergence(mu, lv)
		value := kl.data.data[0]
		t.tensor_backward(kl)
		good :=
			math.abs(value - 2499999993.1245284) < 1e-6 &&
			tf_close(mu.grad.data, {0.15, -0.25, 0.6, 0.0, 0.4, -0.1}, 1e-15) &&
			tf_close(
				lv.grad.data,
				{
					0.026292729518911928,
					-0.12585367405214762,
					0.12295617441031759,
					-0.25,
					-0.19421745996289255,
					0.055350689540042464,
				},
				1e-15,
			)
		t.tensor_free_graph(kl)
		t.tensor_free(mu)
		t.tensor_free(lv)
		report(
			"kl_divergence [2, 3], one log_var past the clamp: torch value and grads",
			good,
			&passed,
			&total,
		)

		// [2, 2, 3, 1] sequence: 4 latent vectors of size 3
		mu3 := tf_mat(
			{0.3, -0.5, 1.2, 0.0, 0.8, -0.2, 0.1, 0.2, 0.3, -0.4, 0.5, -0.6},
			1,
			12,
			true,
			allocator,
		)
		lv3 := tf_mat(
			{0.1, -0.7, 0.4, 0.3, -1.5, 0.2, 0.0, 0.5, -0.5, 1.0, -0.2, 0.6},
			1,
			12,
			true,
			allocator,
		)
		mu3.shape = {2, 2, 3, 1}
		lv3.shape = {2, 2, 3, 1}
		kl3 := t.tensor_kl_divergence(mu3, lv3)
		value3 := kl3.data.data[0]
		t.tensor_backward(kl3)
		good3 :=
			math.abs(value3 - 0.7090444947166534) < 1e-15 &&
			tf_close(
				mu3.grad.data,
				{0.075, -0.125, 0.3, 0.0, 0.2, -0.05, 0.025, 0.05, 0.075, -0.1, 0.125, -0.15},
				1e-15,
			) &&
			tf_close(
				lv3.grad.data,
				{
					0.013146364759455964,
					-0.06292683702607381,
					0.06147808720515879,
					0.0437323509470004,
					-0.09710872998144628,
					0.027675344770021232,
					0.0,
					0.08109015883751602,
					-0.04918366753592082,
					0.21478522855738064,
					-0.022658655865252272,
					0.10276485004881361,
				},
				1e-15,
			)
		t.tensor_free_graph(kl3)
		t.tensor_free(mu3)
		t.tensor_free(lv3)
		report(
			"kl_divergence [2, 2, 3, 1]: mean over the 4 latent vectors (torch)",
			good3,
			&passed,
			&total,
		)
	}

	{ 	// Dropout in evaluation mode / p = 0: a distinct tensor, identity in autograd;
		// input and output are freed independently (tracking allocator: no double free)
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		track.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
		a := mem.tracking_allocator(&track)
		x := tf_mat({1, -2, 3, 0.5}, 2, 2, true, a)
		w := tf_mat({0.5, 2, -1, 4}, 2, 2, false, a)
		w.owned_by_graph = true
		y := t.tensor_dropout(x, 0.5, false)
		good := y != x && tf_close(y.data.data, x.data.data, 0) && y.shape == x.shape
		loss := t.tensor_sum(t.tensor_mul(y, w))
		t.tensor_backward(loss)
		good = good && tf_close(x.grad.data, {0.5, 2, -1, 4}, 0)
		t.tensor_free_graph(loss) // frees y and w, keeps x
		c := tf_mat({1, 2}, 1, 2, false, a)
		z := t.tensor_dropout(c, 0, true) // p = 0, constant input
		good = good && z != c && tf_close(z.data.data, c.data.data, 0)
		t.tensor_free(z)
		// dropout on data (constant input) feeding a trainable op, no scope:
		// tensor_free_graph frees the dropout output too, in both modes
		v := tf_mat({0.5, 1.5}, 2, 1, true, a)
		for training in ([]bool{false, true}) {
			for _ in 0 ..< 3 {
				loss2 := t.tensor_sum(t.tensor_matmul(t.tensor_dropout(c, 0.5, training), v))
				t.tensor_backward(loss2)
				t.tensor_free_graph(loss2)
			}
		}
		t.tensor_free(v)
		t.tensor_free(c)
		t.tensor_free(x)
		report(
			"dropout eval / p = 0: new tensor, identity gradient; constant input freed with the graph",
			good && len(track.allocation_map) == 0 && len(track.bad_free_array) == 0,
			&passed,
			&total,
		)
	}

	{ 	// tensor_free_graph with several roots sharing nodes frees each node once
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		track.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
		a := mem.tracking_allocator(&track)
		context.allocator = a
		x := tf_mat({0.1, 0.2, 0.3, 0.4}, 2, 2, false, a)
		w := tf_mat({1, -1, 0.5, 2}, 2, 2, true, a)
		h := t.tensor_tanh(t.tensor_matmul(x, w)) // shared by both losses
		policy := t.tensor_sum(h)
		value := t.tensor_sum(t.tensor_mul(h, h))
		t.tensor_backward(policy)
		t.tensor_backward(value)
		t.tensor_free_graph(policy, value)
		t.tensor_free_graph() // no roots: nothing to do
		t.tensor_free(x)
		t.tensor_free(w)
		report(
			"tensor_free_graph(policy, value) on a shared graph: no double free or leak",
			len(track.allocation_map) == 0 && len(track.bad_free_array) == 0,
			&passed,
			&total,
		)
	}

	{ 	// tensor_detach of a heap parameter inside a graph scope lives in the scope
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		a := mem.tracking_allocator(&track)
		p := tf_mat({1, 2, 3, 4, 5, 6}, 1, 6, true, a)
		p.shape = {2, 3, 1, 1}
		before := len(track.allocation_map)
		scope: t.Graph_Scope
		t.graph_begin(&scope)
		d := t.tensor_detach(p, a)
		in_scope := len(track.allocation_map) == before && tf_close(d.data.data, p.data.data, 0)
		t.graph_end(&scope)
		t.graph_scope_destroy(&scope)
		d2 := t.tensor_detach(p, a) // no scope: in the given allocator, caller frees
		outside :=
			len(track.allocation_map) > before &&
			!d2.requires_grad &&
			d2.op == .None &&
			d2.shape == p.shape &&
			d2.allocator.data == a.data
		t.tensor_free(d2)
		t.tensor_free(p)
		report(
			"tensor_detach: graph scope arena inside a scope, given allocator outside",
			in_scope && outside && len(track.allocation_map) == 0,
			&passed,
			&total,
		)
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
