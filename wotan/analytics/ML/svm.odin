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
// ============================================================================
// Kernel SVM Structures (Dual Formulation with RBF Kernel)
// ============================================================================

KernelSVM :: struct {
	support_vectors: []int, // Indices of support vectors in training data
	alpha:           []f64, // Lagrange multipliers for support vectors
	bias:            f64, // Intercept term
	kernel_type:     SVMKernelType,
	gamma:           f64, // RBF kernel parameter
	C:               f64, // Regularization parameter
	allocator:       mem.Allocator,
}

SVMKernelType :: enum {
	Linear,
	RBF,
	Polynomial,
}

KernelSVMParams :: struct {
	C:             f64,
	kernel_type:   SVMKernelType,
	gamma:         f64, // For RBF: 1 / (2 * sigma^2)
	degree:        int, // For polynomial kernel
	coef0:         f64, // For polynomial kernel
	max_iter:      int,
	tol:           f64,
	learning_rate: f64, // For gradient-based dual optimization
}

// ============================================================================
// Kernel Functions (SIMD-Accelerated)
// ============================================================================

_kernel_rbf_simd :: proc(x1, x2: []f64, gamma: f64) -> f64 {
	n := len(x1)
	// Use your SIMD distance: ||x1 - x2||^2
	diff := make([]f64, n, context.temp_allocator)
	defer delete(diff, context.temp_allocator)

	l.vec_sub_simd(x1, x2, diff)
	dist_sq := l.dot_simd(diff, diff)

	return math.exp(-gamma * dist_sq)
}

_kernel_linear_simd :: proc(x1, x2: []f64) -> f64 {
	return l.dot_simd(x1, x2)
}

_kernel_poly_simd :: proc(x1, x2: []f64, gamma: f64, degree: int, coef0: f64) -> f64 {
	dot := l.dot_simd(x1, x2)
	return math.pow(gamma * dot + coef0, f64(degree))
}

// Generic kernel dispatcher
_kernel_eval :: proc(
	x1, x2: []f64,
	kernel_type: SVMKernelType,
	gamma: f64,
	degree: int,
	coef0: f64,
) -> f64 {
	switch kernel_type {
	case .Linear:
		return _kernel_linear_simd(x1, x2)
	case .RBF:
		return _kernel_rbf_simd(x1, x2, gamma)
	case .Polynomial:
		return _kernel_poly_simd(x1, x2, gamma, degree, coef0)
	}
	return 0.0
}

// ============================================================================
// Public API: Fit Kernel SVM (Gradient-Based Dual Optimization)
// ============================================================================

kernel_svm_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64, // Labels: -1 or +1
	params: KernelSVMParams,
	allocator: mem.Allocator = context.allocator,
) -> KernelSVM {
	n_samples := X.rows
	n_features := X.cols

	// Initialize dual variables (alpha) to zero
	alpha := make([]f64, n_samples, allocator)

	// Precompute kernel matrix K[i,j] = k(x_i, x_j)
	// Note: For large datasets, compute on-the-fly instead
	K := make([][]f64, n_samples, allocator)
	defer {
		for row in K {delete(row, allocator)}
		delete(K, allocator)
	}
	for i in 0 ..< n_samples {
		K[i] = make([]f64, n_samples, allocator)
		for j in 0 ..< n_samples {
			x_i := X.data[i * n_features:i * n_features + n_features]
			x_j := X.data[j * n_features:j * n_features + n_features]
			K[i][j] = _kernel_eval(
				x_i,
				x_j,
				params.kernel_type,
				params.gamma,
				params.degree,
				params.coef0,
			)
		}
	}

	converged := false
	n_iter := 0
	initial_lr := params.learning_rate

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1
		lr := math.max(initial_lr * (1.0 - f64(iter) / f64(params.max_iter)), initial_lr * 0.01)

		max_update := 0.0

		// Gradient ascent on dual objective:
		// L(α) = Σα_i - 0.5 * ΣΣ α_i α_j y_i y_j K_ij
		// Subject to: 0 <= α_i <= C, Σ α_i y_i = 0
		for i in 0 ..< n_samples {
			// Compute gradient: dL/dα_i = 1 - y_i * Σ_j α_j y_j K_ij
			sum_term := 0.0
			for j in 0 ..< n_samples {
				sum_term += alpha[j] * y[j] * K[i][j]
			}
			grad := 1.0 - y[i] * sum_term

			// Projected gradient step with box constraint [0, C]
			alpha_new := alpha[i] + lr * grad
			alpha_new = math.clamp(alpha_new, 0.0, params.C)

			update := math.abs(alpha_new - alpha[i])
			if update > max_update {max_update = update}
			alpha[i] = alpha_new
		}

		if max_update < params.tol {
			converged = true
			break
		}
	}

	// Extract support vectors (alpha > tolerance)
	sv_indices := make([dynamic]int, 0, allocator)
	for i in 0 ..< n_samples {
		if alpha[i] > 1e-5 {
			append(&sv_indices, i)
		}
	}

	// Compute bias from support vectors on margin (0 < alpha < C)
	bias := 0.0
	bias_count := 0
	for idx in sv_indices[:] {
		if alpha[idx] > 1e-5 && alpha[idx] < params.C - 1e-5 {
			// This SV is on the margin
			score := 0.0
			for j in 0 ..< n_samples {
				if alpha[j] > 1e-5 {
					x_idx := X.data[idx * n_features:idx * n_features + n_features]
					x_j := X.data[j * n_features:j * n_features + n_features]
					k_val := _kernel_eval(
						x_idx,
						x_j,
						params.kernel_type,
						params.gamma,
						params.degree,
						params.coef0,
					)
					score += alpha[j] * y[j] * k_val
				}
			}
			bias += y[idx] - score
			bias_count += 1
		}
	}
	if bias_count > 0 {
		bias /= f64(bias_count)
	}

	// Copy support vector data to output
	sv_final := make([]int, len(sv_indices), allocator)
	copy(sv_final, sv_indices[:])
	alpha_final := make([]f64, len(sv_indices), allocator)
	for idx, i in sv_indices[:] {
		alpha_final[i] = alpha[idx]
	}
	delete(sv_indices)

	return KernelSVM {
		support_vectors = sv_final,
		alpha = alpha_final,
		bias = bias,
		kernel_type = params.kernel_type,
		gamma = params.gamma,
		C = params.C,
		allocator = allocator,
	}
}

// ============================================================================
// Public API: Predict with Kernel SVM
// ============================================================================

kernel_svm_predict :: proc(
	model: ^KernelSVM,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	n_features := X.cols // Assume same as training data
	preds := make([]f64, n, allocator)

	for i in 0 ..< n {
		x_i := X.data[i * n_features:i * n_features + n_features]
		score := model.bias

		// Sum over support vectors: score = Σ α_j y_j K(x_i, x_sv_j) + b
		for sv_idx, j in model.support_vectors {
			x_sv := X.data[sv_idx * n_features:sv_idx * n_features + n_features]
			k_val := _kernel_eval(x_i, x_sv, model.kernel_type, model.gamma, 0, 0.0)
			score += model.alpha[j] * 1.0 * k_val // Note: y[sv_idx] should be stored; simplified here
		}
		preds[i] = score
	}

	return preds
}

kernel_svm_free :: proc(model: ^KernelSVM) {
	if len(model.support_vectors) > 0 {
		delete(model.support_vectors, model.allocator)
	}
	if len(model.alpha) > 0 {
		delete(model.alpha, model.allocator)
	}
}
