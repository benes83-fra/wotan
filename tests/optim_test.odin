package tests

import l "../wotan/linalg"
import optim "../wotan/optimize"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Test Helper: Quadratic Loss (f(w) = 0.5 * ||w - target||^2)
// ============================================================================

_quadratic_loss :: proc(w, target: []f64) -> f64 {
	n := len(w)
	if n != len(target) {panic("quadratic_loss: length mismatch")}

	loss := 0.0
	for i in 0 ..< n {
		diff := w[i] - target[i]
		loss += 0.5 * diff * diff
	}
	return loss
}

_quadratic_gradient :: proc(w, target, grad: []f64) {
	n := len(w)
	if n != len(target) || n != len(grad) {panic("quadratic_gradient: length mismatch")}
	for i in 0 ..< n {
		grad[i] = w[i] - target[i]
	}
}

_quadratic_loss_mat :: proc(W, target: ^l.Matrix(f64)) -> f64 {
	if W.rows != target.rows || W.cols != target.cols {
		panic("quadratic_loss_mat: dimension mismatch")
	}
	n := W.rows * W.cols

	loss := 0.0
	for i in 0 ..< n {
		diff := W.data[i] - target.data[i]
		loss += 0.5 * diff * diff
	}
	return loss
}

_quadratic_gradient_mat :: proc(W, target, grad: ^l.Matrix(f64)) {
	if W.rows != target.rows || W.cols != target.cols {
		panic("quadratic_gradient_mat: rows mismatch")
	}
	if W.rows != grad.rows || W.cols != grad.cols {
		panic("quadratic_gradient_mat: cols mismatch")
	}
	n := W.rows * W.cols
	for i in 0 ..< n {
		grad.data[i] = W.data[i] - target.data[i]
	}
}

// ============================================================================
// Test: SGD Slice API
// ============================================================================

test_sgd_slice :: proc(allocator: mem.Allocator) -> bool {
	fmt.println("Testing SGD (slice API)...")

	n := 10
	target := make([]f64, n, allocator)
	defer delete(target, allocator)
	for i in 0 ..< n {target[i] = f64(i) * 0.1}

	w := make([]f64, n, allocator)
	defer delete(w, allocator)
	for i in 0 ..< n {w[i] = 0.0}

	grad := make([]f64, n, allocator)
	defer delete(grad, allocator)

	// Test: Basic convergence without momentum
	opt := optim.optimizer_sgd_init(n, 0.1, 0.0, 0.0, allocator)
	defer optim.optimizer_sgd_free(&opt)

	initial_loss := _quadratic_loss(w, target)
	for iter in 0 ..< 100 {
		_quadratic_gradient(w, target, grad)
		optim.optimizer_sgd_step(&opt, w, grad)
	}
	final_loss := _quadratic_loss(w, target)

	fmt.printf("  SGD: loss %.4f -> %.4f\n", initial_loss, final_loss)
	if final_loss >= initial_loss * 0.01 {
		fmt.println("  ❌ SGD failed to converge")
		return false
	}

	fmt.println("  ✅ SGD slice API passed")
	return true
}

// ============================================================================
// Test: SGD Matrix API
// ============================================================================

test_sgd_matrix :: proc(allocator: mem.Allocator) -> bool {
	fmt.println("Testing SGD (matrix API)...")

	batch := 4
	n := 10
	target := l.matrix_new(f64, batch, n, allocator)
	defer l.matrix_free(&target)
	for i in 0 ..< batch {
		for j in 0 ..< n {
			target.data[i * n + j] = f64(j) * 0.1
		}
	}

	W := l.matrix_new(f64, batch, n, allocator)
	defer l.matrix_free(&W)
	for i in 0 ..< batch * n {W.data[i] = 0.0}

	grad := l.matrix_new(f64, batch, n, allocator)
	defer l.matrix_free(&grad)

	opt := optim.optimizer_sgd_init_mat(batch, n, 0.1, 0.0, 0.0, allocator)
	defer optim.optimizer_sgd_free(&opt)

	initial_loss := _quadratic_loss_mat(&W, &target)
	for iter in 0 ..< 100 {
		_quadratic_gradient_mat(&W, &target, &grad)
		optim.optimizer_sgd_step_mat(&opt, &W, &grad)
	}
	final_loss := _quadratic_loss_mat(&W, &target)

	fmt.printf("  SGD matrix: loss %.4f -> %.4f\n", initial_loss, final_loss)
	if final_loss >= initial_loss * 0.01 {
		fmt.println("  ❌ SGD matrix failed to converge")
		return false
	}

	fmt.println("  ✅ SGD matrix API passed")
	return true
}

// ============================================================================
// Test: Adam Slice API
// ============================================================================

test_adam_slice :: proc(allocator: mem.Allocator) -> bool {
	fmt.println("Testing Adam (slice API)...")

	n := 10
	target := make([]f64, n, allocator)
	defer delete(target, allocator)
	for i in 0 ..< n {target[i] = f64(i) * 0.1}

	w := make([]f64, n, allocator)
	defer delete(w, allocator)
	for i in 0 ..< n {w[i] = 0.0}

	grad := make([]f64, n, allocator)
	defer delete(grad, allocator)

	opt := optim.optimizer_adam_init(n, 0.01, 0.9, 0.999, 1e-8, 0.0, allocator)
	defer optim.optimizer_adam_free(&opt)

	initial_loss := _quadratic_loss(w, target)
	for iter in 0 ..< 200 {
		_quadratic_gradient(w, target, grad)
		optim.optimizer_adam_step(&opt, w, grad)
	}
	final_loss := _quadratic_loss(w, target)

	fmt.printf("  Adam: loss %.4f -> %.4f\n", initial_loss, final_loss)
	if final_loss >= initial_loss * 0.01 {
		fmt.println("  ❌ Adam failed to converge")
		return false
	}

	fmt.println("  ✅ Adam slice API passed")
	return true
}

// ============================================================================
// Test: RMSProp Slice API
// ============================================================================

test_rmsprop_slice :: proc(allocator: mem.Allocator) -> bool {
	fmt.println("Testing RMSProp (slice API)...")

	n := 10
	target := make([]f64, n, allocator)
	defer delete(target, allocator)
	for i in 0 ..< n {target[i] = f64(i) * 0.1}

	w := make([]f64, n, allocator)
	defer delete(w, allocator)
	for i in 0 ..< n {w[i] = 0.0}

	grad := make([]f64, n, allocator)
	defer delete(grad, allocator)

	opt := optim.optimizer_rmsprop_init(n, 0.01, 0.99, 1e-8, 0.0, allocator)
	defer optim.optimizer_rmsprop_free(&opt)

	initial_loss := _quadratic_loss(w, target)
	for iter in 0 ..< 200 {
		_quadratic_gradient(w, target, grad)
		optim.optimizer_rmsprop_step(&opt, w, grad)
	}
	final_loss := _quadratic_loss(w, target)

	fmt.printf("  RMSProp: loss %.4f -> %.4f\n", initial_loss, final_loss)
	if final_loss >= initial_loss * 0.01 {
		fmt.println("  ❌ RMSProp failed to converge")
		return false
	}

	fmt.println("  ✅ RMSProp slice API passed")
	return true
}

// ============================================================================
// Test: Learning Rate Schedulers
// ============================================================================

test_schedulers :: proc() -> bool {
	fmt.println("Testing learning rate schedulers...")

	// Linear decay (stays at 0 after total_steps)
	lin := optim.lr_linear_decay(0.1, 100)
	lr_lin := optim.lr_schedule_at(lin, 100)
	if math.abs(lr_lin) > 1e-6 || optim.lr_schedule_at(lin, 150) != lr_lin {
		fmt.printf("  ❌ Linear decay failed: got %.6f, expected ~0.0\n", lr_lin)
		return false
	}
	fmt.printf("  Linear decay: 0.1 -> %.6f ✅\n", lr_lin)

	// Cosine annealing
	cos := optim.lr_cosine(0.1, 100, 0.001)
	lr_cos := optim.lr_schedule_at(cos, 100)
	if math.abs(lr_cos - 0.001) > 1e-4 {
		fmt.printf("  ❌ Cosine annealing failed: got %.6f, expected ~0.001\n", lr_cos)
		return false
	}
	fmt.printf("  Cosine annealing: 0.1 -> %.6f ✅\n", lr_cos)

	fmt.println("  ✅ Scheduler tests passed")
	return true
}


// ============================================================================
// Test: L-BFGS (vs the textbook two-loop recursion and Newton's method)
// ============================================================================

// log Σ exp(A x + b) + ½‖x‖² with its gradient Aᵀ softmax(A x + b) + x
@(private = "file")
_LSE_A := [6][5]f64 {
	{1.0, -2.0, 0.5, 0.0, 1.5},
	{0.3, 0.8, -1.0, 2.0, 0.0},
	{-1.5, 0.2, 0.7, -0.4, 1.0},
	{2.0, 1.0, 0.0, -1.0, -0.5},
	{0.0, -0.6, 1.2, 0.9, -2.0},
	{0.5, 0.5, 0.5, 0.5, 0.5},
}
@(private = "file")
_LSE_B := [6]f64{0.1, -0.4, 0.3, 0.0, 0.2, -0.1}

@(private = "file")
_lse_fg :: proc(x, grad: []f64, _: rawptr) -> f64 {
	z: [6]f64
	top := math.inf_f64(-1)
	for i in 0 ..< 6 {
		z[i] = _LSE_B[i]
		for j in 0 ..< 5 {z[i] += _LSE_A[i][j] * x[j]}
		top = max(top, z[i])
	}
	sum := 0.0
	for i in 0 ..< 6 {sum += math.exp(z[i] - top)}
	f := top + math.ln(sum)
	for j in 0 ..< 5 {
		grad[j] = x[j]
		f += 0.5 * x[j] * x[j]
	}
	for i in 0 ..< 6 {
		p := math.exp(z[i] - top) / sum
		for j in 0 ..< 5 {grad[j] += p * _LSE_A[i][j]}
	}
	return f
}

@(private = "file")
_rosen_fg :: proc(x, grad: []f64, _: rawptr) -> f64 {
	n := len(x)
	f := 0.0
	for &g in grad {g = 0}
	for i in 0 ..< n - 1 {
		a := x[i + 1] - x[i] * x[i]
		b := 1 - x[i]
		f += 100 * a * a + b * b
		grad[i] += -400 * x[i] * a - 2 * b
		grad[i + 1] += 200 * a
	}
	return f
}

@(private = "file")
_rosen_f :: proc(x: []f64, _: rawptr) -> f64 {
	g := make([]f64, len(x), context.temp_allocator)
	return _rosen_fg(x, g, nil)
}

// returns the number of failed checks
test_lbfgs :: proc(allocator: mem.Allocator) -> int {
	fmt.println("Testing L-BFGS...")
	failed := 0
	check :: proc(failed: ^int, ok: bool, name: string) {
		if !ok {failed^ += 1}
		fmt.printf("  %s %s\n", "✅" if ok else "❌", name)
	}

	// 1. the direction after three steps on f = ½ Σ a_i x_i², a = (1, 3, 10, 30),
	// against the textbook two-loop recursion (NumPy)
	{
		a := [4]f64{1, 3, 10, 30}
		xs := [4][4]f64 {
			{1, 1, 1, 1},
			{0.9, 0.7, 0.2, -0.5},
			{0.5, 0.4, -0.1, 0.2},
			{0.3, 0.1, 0.05, -0.02},
		}
		o := optim.optimizer_lbfgs_init(4, 1.0, 10, 1e-5, allocator)
		defer optim.optimizer_lbfgs_free(&o)
		for x in xs {
			w, g: [4]f64
			for i in 0 ..< 4 {w[i] = x[i]; g[i] = a[i] * x[i]}
			optim.optimizer_lbfgs_step(&o, w[:], g[:])
		}
		want := [4]f64{-0.04909212, -0.05212452, -0.03265854, 0.0098095}
		err := 0.0
		for i in 0 ..< 4 {err = max(err, math.abs(o.d[i] - want[i]))}
		check(
			&failed,
			err < 1e-8,
			fmt.tprintf("two-loop direction = textbook recursion (max diff %.1e)", err),
		)
	}

	// 2. lbfgs_minimize on log-sum-exp + ½‖x‖² vs Newton's method to machine
	// precision (SciPy L-BFGS-B at gtol 1e-12 stops 5e-10 away)
	{
		x := make([]f64, 5, allocator)
		defer delete(x, allocator)
		r := optim.lbfgs_minimize(_lse_fg, x, nil, gtol = 1e-8, allocator = allocator)
		want := [5]f64 {
			-0.18605639446886502,
			-0.01953895245579049,
			-0.38769143650013266,
			-0.18659590866948247,
			-0.10971886851472654,
		}
		err := 0.0
		for i in 0 ..< 5 {err = max(err, math.abs(x[i] - want[i]))}
		check(
			&failed,
			r.converged && err < 1e-8 && math.abs(r.f - 1.6983083475520402) < 1e-12,
			fmt.tprintf(
				"lbfgs_minimize, log-sum-exp: minimiser = Newton (max diff %.1e, %d iterations)",
				err,
				r.n_iter,
			),
		)
	}

	// 3. Rosenbrock in 2 and 10 dimensions from (-1.2, 1, -1.2, ...): minimum at 1
	for n in ([]int{2, 10}) {
		x := make([]f64, n, allocator)
		defer delete(x, allocator)
		for i in 0 ..< n {x[i] = -1.2 if i % 2 == 0 else 1.0}
		r := optim.lbfgs_minimize(
			_rosen_fg,
			x,
			nil,
			max_iter = 5000,
			gtol = 1e-10,
			allocator = allocator,
		)
		err := 0.0
		for v in x {err = max(err, math.abs(v - 1))}
		check(
			&failed,
			r.converged && err < 1e-8,
			fmt.tprintf(
				"lbfgs_minimize, Rosenbrock n=%d: max |x - 1| %.1e (%d iterations, %d evaluations)",
				n,
				err,
				r.n_iter,
				r.n_eval,
			),
		)
	}

	// 4. the step interface with a line search, Rosenbrock n=2
	{
		o := optim.optimizer_lbfgs_init(2, 1.0, 10, 1e-5, allocator)
		defer optim.optimizer_lbfgs_free(&o)
		w := [2]f64{-1.2, 1}
		g: [2]f64
		steps := 0
		for it in 0 ..< 1000 {
			f := _rosen_fg(w[:], g[:], nil)
			if max(math.abs(g[0]), math.abs(g[1])) < 1e-8 do break
			optim.optimizer_lbfgs_step(&o, w[:], g[:], f, _rosen_f, nil)
			steps = it + 1
			if o.line_search_failed do break
		}
		err := max(math.abs(w[0] - 1), math.abs(w[1] - 1))
		check(
			&failed,
			err < 1e-6,
			fmt.tprintf(
				"optimizer_lbfgs_step, Rosenbrock n=2: max |x - 1| %.1e after %d steps",
				err,
				steps,
			),
		)
	}

	return failed
}

// ============================================================================
// Main Test Runner
// ============================================================================

// Every failed test counts as one failed check (test_lbfgs: one per failed
// check), recorded for exit_on_failures
run_optimizer_tests :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Running Optimizer Unit Tests ===\n")

	failed := 0
	for ok in ([]bool {
			test_sgd_slice(allocator),
			test_sgd_matrix(allocator),
			test_adam_slice(allocator),
			test_rmsprop_slice(allocator),
			test_schedulers(),
		}) {
		if !ok do failed += 1
	}
	failed += test_lbfgs(allocator)
	record_failures(failed)

	fmt.println("\n=== Optimizer Tests Complete ===")
	if failed == 0 {
		fmt.println("✅ All tests passed!")
	} else {
		fmt.printfln("❌ %d check(s) failed", failed)
	}
	fmt.println()
}
