package ML

import l "../../linalg"
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
	y: []f64, // Labels: -1 or +1 for classification
	params: SVMParams,
	allocator: mem.Allocator = context.allocator,
) -> LinearSVM {
	n_samples := X.rows
	n_features := X.cols

	// Initialize weights to zero
	w := make([]f64, n_features, allocator)
	b := 0.0

	converged := false
	n_iter := 0

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1
		max_update := 0.0

		// Gradient descent on primal objective:
		// L = 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i*(w·x_i + b)))
		for i in 0 ..< n_samples {
			// Compute prediction
			score := b
			for f in 0 ..< n_features {
				score += w[f] * X.data[i * n_features + f]
			}

			// Hinge loss gradient
			if y[i] * score < 1.0 {
				// Misclassified or margin violation: update
				grad_b := -params.C * y[i]
				b_update := params.learning_rate * grad_b
				b += b_update

				for f in 0 ..< n_features {
					grad_w := w[f] - params.C * y[i] * X.data[i * n_features + f]
					w_update := params.learning_rate * grad_w
					w[f] += w_update

					update_mag := math.abs(w_update)
					if update_mag > max_update {max_update = update_mag}
				}
			}
		}

		// Check convergence
		if max_update < params.tol {
			converged = true
			break
		}
	}

	return LinearSVM {
		weights         = w,
		bias            = b,
		support_vectors = make([]int, 0, allocator), // Empty for primal form
		n_iter          = n_iter,
		converged       = converged,
		allocator       = allocator,
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

	for i in 0 ..< n {
		score := model.bias
		for f in 0 ..< n_features {
			score += model.weights[f] * X.data[i * n_features + f]
		}
		// Return decision function value (sign gives class)
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
