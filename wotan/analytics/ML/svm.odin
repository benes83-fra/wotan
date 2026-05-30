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

	// Allocate combined params: [w (n_features), b (1)]
	params_vec := make([]f64, n_features + 1, allocator)
	defer delete(params_vec, allocator) // Free at end

	// Initialize weights to zero, bias to zero
	for i in 0 ..< n_features + 1 {params_vec[i] = 0.0}

	// Setup optimizer for combined params
	opt := optim.optimizer_sgd_init(
		n_features + 1,
		params.learning_rate,
		momentum = 0.0,
		weight_decay = 0.0, // We handle regularization manually
		allocator = allocator,
	)
	defer optim.optimizer_sgd_free(&opt)

	converged := false
	n_iter := 0
	initial_lr := params.learning_rate

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1
		// Linear decay with minimum LR to avoid zeroing out
		lr := math.max(initial_lr * (1.0 - f64(iter) / f64(params.max_iter)), initial_lr * 0.01)
		opt.learning_rate = lr

		for i in 0 ..< n_samples {
			// Compute score = w·x + b
			score := params_vec[n_features] // bias is last element
			for f in 0 ..< n_features {
				score += params_vec[f] * X.data[i * n_features + f]
			}

			// Hinge loss gradient
			// Hinge loss gradient
			if y[i] * score < 1.0 {
				// Margin violation: grad_w = w - C*y*x, grad_b = -C*y
				for f in 0 ..< n_features {
					grad_val := params_vec[f] - params.C * y[i] * X.data[i * n_features + f]
					grad_arr := [1]f64{grad_val}
					// ✅ FIX: Correct slice syntax [start : start+1] for 1-element slice
					optim.optimizer_sgd_step(&opt, params_vec[f:f + 1], grad_arr[:])
				}
				// Update bias (last element at index n_features)
				grad_b := -params.C * y[i]
				grad_arr := [1]f64{grad_b}
				// ✅ FIX: Bias slice is [n_features : n_features+1]
				optim.optimizer_sgd_step(&opt, params_vec[n_features:n_features + 1], grad_arr[:])
			} else {
				// No violation: grad_w = w (L2 reg), grad_b = 0
				for f in 0 ..< n_features {
					grad_arr := [1]f64{params_vec[f]}
					optim.optimizer_sgd_step(&opt, params_vec[f:f + 1], grad_arr[:])
				}
				// Bias gradient = 0 (no update needed)
			}
		}
	}

	// Extract final weights and bias
	w := make([]f64, n_features, allocator)
	for f in 0 ..< n_features {w[f] = params_vec[f]}
	b := params_vec[n_features]

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
