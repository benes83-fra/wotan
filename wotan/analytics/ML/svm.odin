package ML

import l "../../linalg"
import optim "../../optimize"
import "core:fmt"
import "core:math"
import "core:mem"
// ============================================================================
// Linear SVM Structures (Primal Formulation)
// ============================================================================

LinearSVM :: struct {
	weights:         []f64, // w: feature weights
	bias:            f64, // b: intercept
	support_vectors: []int, // Indices of support vectors (for dual form later)
	n_iter:          int,
	converged:       bool,
	allocator:       mem.Allocator,
}

SVMParams :: struct {
	C:             f64, // Regularization strength (1/C is regularization)
	max_iter:      int,
	tol:           f64, // Convergence tolerance
	learning_rate: f64, // For gradient descent
	fit_intercept: bool,
}

// ============================================================================
// Public API: Fit Linear SVM
// ============================================================================
svm_fit_linear :: proc(
	X: ^l.Matrix(f64),
	y: []f64, // Labels: -1 or +1
	params: SVMParams,
	allocator: mem.Allocator = context.allocator,
) -> LinearSVM {
	n_samples := X.rows
	n_features := X.cols

	// Initialize weights
	w := make([]f64, n_features, allocator)
	b := 0.0

	// Setup optimizer & scheduler
	opt := optim.optimizer_sgd_init(
		n_features,
		params.learning_rate,
		momentum = 0.0,
		weight_decay = 1.0,
		allocator = allocator,
	)
	defer optim.optimizer_sgd_free(&opt)

	sched := optim.scheduler_linear_decay_init(params.learning_rate, params.max_iter)

	converged := false
	n_iter := 0

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1
		sched.current_lr = sched.initial_lr * f64(1.0 - f64(iter) / f64(params.max_iter))
		opt.learning_rate = sched.current_lr

		for i in 0 ..< n_samples {
			// Compute score = w·x + b
			score := b
			for f in 0 ..< n_features {
				score += w[f] * X.data[i * n_features + f]
			}

			// Hinge loss gradient update
			if y[i] * score < 1.0 {
				// Margin violation: grad_w = w - C*y*x
				for f in 0 ..< n_features {
					w_grad_val := w[f] - params.C * y[i] * X.data[i * n_features + f]
					// ✅ FIX: Create a 1-element stack array, then slice it
					w_grad_arr := [1]f64{w_grad_val}
					optim.optimizer_sgd_step(&opt, w[f:1], w_grad_arr[:])
				}
				b += opt.learning_rate * params.C * y[i]
			} else {
				// No violation: only weight decay (grad_w = w)
				for f in 0 ..< n_features {
					w_grad_arr := [1]f64{w[f]}
					optim.optimizer_sgd_step(&opt, w[f:1], w_grad_arr[:])
				}
			}
		}
	}

	return LinearSVM {
		weights = w,
		bias = b,
		support_vectors = make([]int, 0, allocator),
		n_iter = n_iter,
		converged = converged,
		allocator = allocator,
	}
}
// ============================================================================
// Public API: Predict with Linear SVM
// ============================================================================

svm_predict_linear :: proc(
	model: ^LinearSVM,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	n_features := len(model.weights)

	preds := make([]f64, n, allocator)

	// Use SIMD dot product for fast prediction
	for i in 0 ..< n {
		x_row := X.data[i * n_features:i * n_features + n_features]
		score := model.bias + l.dot_simd(model.weights, x_row)
		preds[i] = score
	}

	return preds
}

svm_free :: proc(model: ^LinearSVM) {
	if len(model.weights) > 0 {
		delete(model.weights, model.allocator)
	}
	if len(model.support_vectors) > 0 {
		delete(model.support_vectors, model.allocator)
	}
}

// ============================================================================
// Future: Kernel SVM Placeholder (To Be Extended)
// ============================================================================

// Kernel function type
SVMKernel :: proc(x1, x2: []f64, gamma: f64, degree: int, coef0: f64) -> f64

// RBF kernel: K(x, x') = exp(-gamma * ||x - x'||^2)
_kernel_rbf :: proc(x1, x2: []f64, gamma: f64, degree: int, coef0: f64) -> f64 {
	n := len(x1)
	// Use your SIMD distance function for ||x1 - x2||^2
	diff := make([]f64, n, context.temp_allocator)
	defer delete(diff, context.temp_allocator)

	l.vec_sub_simd(x1, x2, diff)
	dist_sq := l.dot_simd(diff, diff)

	return math.exp(-gamma * dist_sq)
}

// Linear kernel: K(x, x') = x · x'
_kernel_linear :: proc(x1, x2: []f64, gamma: f64, degree: int, coef0: f64) -> f64 {
	return l.dot_simd(x1, x2)
}

// Polynomial kernel: K(x, x') = (gamma * x·x' + coef0)^degree
_kernel_poly :: proc(x1, x2: []f64, gamma: f64, degree: int, coef0: f64) -> f64 {
	dot := l.dot_simd(x1, x2)
	return math.pow(gamma * dot + coef0, f64(degree))
}
