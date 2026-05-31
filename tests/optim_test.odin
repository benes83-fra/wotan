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

	// Linear decay
	sched_lin := optim.scheduler_linear_decay_init(0.1, 100)
	for i in 0 ..< 100 {
		optim.scheduler_linear_decay_step(&sched_lin)
	}
	if math.abs(sched_lin.current_lr) > 1e-6 {
		fmt.printf("  ❌ Linear decay failed: got %.6f, expected ~0.0\n", sched_lin.current_lr)
		return false
	}
	fmt.printf("  Linear decay: 0.1 -> %.6f ✅\n", sched_lin.current_lr)

	// Cosine annealing
	sched_cos := optim.scheduler_cosine_annealing_init(0.1, 0.001, 100)
	for i in 0 ..< 100 {
		optim.scheduler_cosine_annealing_step(&sched_cos)
	}
	if math.abs(sched_cos.current_lr - 0.001) > 1e-4 {
		fmt.printf(
			"  ❌ Cosine annealing failed: got %.6f, expected ~0.001\n",
			sched_cos.current_lr,
		)
		return false
	}
	fmt.printf("  Cosine annealing: 0.1 -> %.6f ✅\n", sched_cos.current_lr)

	fmt.println("  ✅ Scheduler tests passed")
	return true
}

// ============================================================================
// Main Test Runner
// ============================================================================

run_optimizer_tests :: proc(allocator: mem.Allocator) -> bool {
	fmt.println("\n=== Running Optimizer Unit Tests ===\n")

	all_passed := true

	all_passed = test_sgd_slice(allocator) && all_passed
	all_passed = test_sgd_matrix(allocator) && all_passed
	all_passed = test_adam_slice(allocator) && all_passed
	all_passed = test_rmsprop_slice(allocator) && all_passed
	all_passed = test_schedulers() && all_passed

	fmt.println("\n=== Optimizer Tests Complete ===")
	if all_passed {
		fmt.println("✅ All tests passed!")
	} else {
		fmt.println("❌ Some tests failed")
	}
	fmt.println()

	return all_passed
}
