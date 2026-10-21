package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// Numerical gradient checks for the autograd engine.
// Each check compares the analytic gradient from tensor_backward against a
// central finite difference of a scalar loss f(x).

GC_TOLERANCE :: 1e-4

gc_passed, gc_total: int
gc_W1, gc_W2, gc_SW: ^t.Tensor

gc_mk :: proc(vals: []f64, rows, cols: int, rg: bool, allocator: mem.Allocator) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols, allocator)
	copy(m.data, vals)
	return t.tensor_new(m, rg, allocator)
}

gc_check :: proc(
	name: string,
	x: ^t.Tensor,
	f: proc(x: ^t.Tensor) -> ^t.Tensor,
	allocator: mem.Allocator,
	h: f64 = 1e-6,
) {
	t.tensor_zero_grad(x)
	loss := f(x)
	t.tensor_backward(loss, allocator)
	analytic := make([]f64, len(x.grad.data), allocator)
	defer delete(analytic, allocator)
	copy(analytic, x.grad.data)
	t.tensor_free_graph(loss, allocator)

	max_err := 0.0
	for i in 0 ..< len(x.data.data) {
		orig := x.data.data[i]
		x.data.data[i] = orig + h
		lp := f(x); fp := lp.data.data[0]; t.tensor_free_graph(lp, allocator)
		x.data.data[i] = orig - h
		lm := f(x); fm := lm.data.data[0]; t.tensor_free_graph(lm, allocator)
		x.data.data[i] = orig

		num := (fp - fm) / (2 * h)
		// Relative error; the 1e-6 floor keeps finite-difference noise on
		// near-zero gradients from failing, while still catching tiny wrong ones.
		err := abs(num - analytic[i]) / max(abs(num), abs(analytic[i]), 1e-6)
		if err > max_err || err != err {max_err = err} 	// err != err: NaN must fail, not be skipped
		if i < 3 {fmt.printf("    [%d] analytic=% .6e numeric=% .6e\n", i, analytic[i], num)}
	}

	gc_total += 1
	status := "FAIL"
	if max_err < GC_TOLERANCE {
		status = "OK  "
		gc_passed += 1
	}
	fmt.printf("%s %-40s max_rel_err=%.2e\n\n", status, name, max_err)
}

gradcheck_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Autograd Numerical Gradient Checks ===")
	// Closures cannot capture, so tensors built inside the loss procs use context.allocator
	context.allocator = allocator
	gc_passed, gc_total = 0, 0

	gc_W1 = gc_mk([]f64{0.1, -0.2, 0.3, 0.4, 0.5, -0.6}, 2, 3, false, allocator)
	gc_W2 = gc_mk([]f64{0.7, 0.1, -0.3, 0.2, 0.9, -0.5}, 2, 3, false, allocator)
	gc_SW = gc_mk([]f64{0.3, -1.2, 0.8, 2.0, 0.5, -0.7}, 2, 3, false, allocator)
	defer {
		t.tensor_free(gc_W1)
		t.tensor_free(gc_W2)
		t.tensor_free(gc_SW)
	}

	// 1. Flattened [1, N*D] input feeding TWO matmuls (like Q/K/V projections)
	x := gc_mk([]f64{1, 2, 3, 4}, 1, 4, true, allocator) // N=2, D=2
	defer t.tensor_free(x)
	gc_check("matmul flattened, fan-out (QKV pattern)", x, proc(x: ^t.Tensor) -> ^t.Tensor {
		a := t.tensor_matmul(x, gc_W1)
		b := t.tensor_matmul(x, gc_W2)
		return t.tensor_sum(t.tensor_add(a, b))
	}, allocator)

	// 1b. Flattened matmul + residual add
	gc_check("matmul flattened + residual add", x, proc(x: ^t.Tensor) -> ^t.Tensor {
		W := gc_mk([]f64{0.1, -0.2, 0.3, 0.4}, 2, 2, false, context.allocator)
		W.owned_by_graph = true
		a := t.tensor_matmul(x, W)
		return t.tensor_sum(t.tensor_mul(t.tensor_add(a, x), t.tensor_add(a, x)))
	}, allocator)

	// 2. KL divergence wrt log_var
	lv := gc_mk([]f64{0.5, -0.3, 1.2}, 1, 3, true, allocator)
	defer t.tensor_free(lv)
	gc_check("kl_divergence d/dlog_var", lv, proc(lv: ^t.Tensor) -> ^t.Tensor {
		mu := gc_mk([]f64{0.1, 0.2, 0.3}, 1, 3, false, context.allocator)
		mu.owned_by_graph = true
		return t.tensor_kl_divergence(mu, lv)
	}, allocator)

	// 3. Div with constant numerator
	d := gc_mk([]f64{1.5, 2.0, 3.0}, 1, 3, true, allocator)
	defer t.tensor_free(d)
	gc_check("div(const, x)", d, proc(x: ^t.Tensor) -> ^t.Tensor {
		one := gc_mk([]f64{1, 1, 1}, 1, 3, false, context.allocator)
		one.owned_by_graph = true
		return t.tensor_sum(t.tensor_div(one, x))
	}, allocator)

	// 4. Scale (large and small scalars)
	s := gc_mk([]f64{1, 2, 3}, 1, 3, true, allocator)
	defer t.tensor_free(s)
	gc_check("scale(x, 0.33)", s, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_scale(x, 0.33))
	}, allocator)
	gc_check("scale(x, 1/(252*10000))", s, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_scale(t.tensor_sum(x), 1.0 / (252.0 * 10000.0))
	}, allocator)

	// 5. LeakyReLU alpha != 0.01
	lr := gc_mk([]f64{-1, 2, -3}, 1, 3, true, allocator)
	defer t.tensor_free(lr)
	gc_check("leaky_relu alpha=0.2", lr, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_leaky_relu(x, 0.2))
	}, allocator)

	// 6. Sharpe loss with realistic daily returns
	r := gc_mk([]f64{0.0012, -0.0007, 0.00031, 0.0009, -0.00045}, 1, 5, true, allocator)
	defer t.tensor_free(r)
	gc_check("sharpe_loss daily returns", r, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sharpe_loss(x, 0.0000159, context.allocator)
	}, allocator)

	// 7. Flatten with fan-out (flatten output + original 4D input both used)
	f := gc_mk([]f64{1, 2, 3, 4}, 1, 4, true, allocator)
	defer t.tensor_free(f)
	f.shape = [4]int{1, 1, 2, 2}
	gc_check("flatten fan-out", f, proc(x: ^t.Tensor) -> ^t.Tensor {
		fl := t.tensor_flatten(x)
		y := t.tensor_mul(fl, fl)
		return t.tensor_add(t.tensor_sum(y), t.tensor_sum(t.tensor_mul(x, x)))
	}, allocator)

	// 8. Softmax on a [2,3] batch: each row must be normalized independently
	lg := gc_mk([]f64{1, 2, 3, 1, 1, 1}, 2, 3, false, allocator)
	defer t.tensor_free(lg)
	p := t.tensor_softmax(lg, allocator)
	defer t.tensor_free(p)
	row0 := p.data.data[0] + p.data.data[1] + p.data.data[2]
	row1 := p.data.data[3] + p.data.data[4] + p.data.data[5]
	gc_total += 1
	if abs(row0 - 1.0) < 1e-12 && abs(row1 - 1.0) < 1e-12 && abs(p.data.data[3] - 1.0 / 3.0) < 1e-12 {
		gc_passed += 1
		fmt.printf("OK   %-40s rows=%.4f %.4f\n\n", "softmax [2,3] per-row normalization", row0, row1)
	} else {
		fmt.printf("FAIL %-40s rows=%.4f %.4f\n\n", "softmax [2,3] per-row normalization", row0, row1)
	}

	// 8b. Softmax backward on a batch (weighted sum, otherwise the gradient is 0)
	sm := gc_mk([]f64{0.5, -1.0, 2.0, 0.1, 0.3, -0.4}, 2, 3, true, allocator)
	defer t.tensor_free(sm)
	gc_check("softmax [2,3] backward", sm, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_mul(t.tensor_softmax(x), gc_SW))
	}, allocator)

	// 8c. Single-row softmax (PPO policy head path)
	sm1 := gc_mk([]f64{0.2, 1.5, -0.3, 0.9}, 1, 4, true, allocator)
	defer t.tensor_free(sm1)
	gc_check("softmax [1,4] backward", sm1, proc(x: ^t.Tensor) -> ^t.Tensor {
		W := gc_mk([]f64{1.0, -2.0, 0.5, 3.0}, 1, 4, false, context.allocator)
		W.owned_by_graph = true
		return t.tensor_sum(t.tensor_mul(t.tensor_softmax(x), W))
	}, allocator)

	// 9. Clamp with a tiny bound (old int*1e6 encoding rounded lo=-2.5e-7 to 0).
	// x[0] = -1e-7 is inside [lo, hi], so its gradient must be 1.
	cl := gc_mk([]f64{-1e-7, 0.3, 2.0}, 1, 3, true, allocator)
	defer t.tensor_free(cl)
	gc_check("clamp(x, -2.5e-7, 1.5)", cl, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_clamp(x, -2.5e-7, 1.5))
	}, allocator, 1e-9)

	// 10. Prospect theory ops with non-round parameters
	pv := gc_mk([]f64{0.02, -0.015, 0.3, -0.4}, 1, 4, true, allocator)
	defer t.tensor_free(pv)
	gc_check("prospect_value(a=0.8812345)", pv, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_prospect_value(x, 0.8812345, 0.8712345, 2.2512345, context.allocator))
	}, allocator)
	gc_check("prospect_utility(a=0.8812345)", pv, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_prospect_utility(x, 0.8812345, 0.8712345, 2.2512345, context.allocator)
	}, allocator)

	pw := gc_mk([]f64{0.1, 0.35, 0.6, 0.9}, 1, 4, true, allocator)
	defer t.tensor_free(pw)
	gc_check("prob_weighting(gamma=0.6123457)", pw, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_prob_weighting(x, 0.6123457, context.allocator))
	}, allocator)

	// 11. Time-dimension z-score (eps passed through f64_metadata)
	nt := gc_mk([]f64{0.5, -1.0, 2.0, 0.3, -0.7, 1.1, 0.0, 0.9}, 1, 8, true, allocator)
	defer t.tensor_free(nt)
	nt.shape = [4]int{1, 1, 4, 2} // [N, C, T, L]
	gc_check("normalize_time [1,1,4,2]", nt, proc(x: ^t.Tensor) -> ^t.Tensor {
		W := gc_mk([]f64{0.3, -1.2, 0.8, 2.0, 0.5, -0.7, 1.4, -0.2}, 1, 8, false, context.allocator)
		W.owned_by_graph = true
		return t.tensor_sum(t.tensor_mul(t.tensor_normalize_time(x, 1e-8, context.allocator), W))
	}, allocator)

	fmt.printf("Gradcheck summary: %d/%d passed\n", gc_passed, gc_total)
	record_failures(gc_total - gc_passed)
}
