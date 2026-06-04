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
	weights:         []f64,
	bias:            f64,
	support_vectors: []int,
	n_iter:          int,
	converged:       bool,
	allocator:       mem.Allocator,
}

SVMParams :: struct {
	C:              f64,
	max_iter:       int,
	tol:            f64,
	learning_rate:  f64,
	fit_intercept:  bool,
	optimizer_type: optim.OptimizerType,
}

// ============================================================================
// Public API: Fit Linear SVM (HEAVILY OPTIMIZED)
// ============================================================================
svm_fit_linear :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: SVMParams,
	allocator: mem.Allocator = context.allocator,
) -> LinearSVM {
	n_samples := X.rows
	n_features := X.cols

	params_vec := make([]f64, n_features + 1, allocator)
	defer delete(params_vec, allocator)

	// ✅ Setup unified optimizer
	opt_config := optim.optimizer_default_config(params.optimizer_type)
	opt_config.learning_rate = params.learning_rate
	opt := optim.optimizer_init(opt_config, n_features + 1, allocator)
	defer optim.optimizer_free(&opt)

	// ✅ Allocate full_grad ONCE outside the loop (Zero allocation in hot path)
	full_grad := make([]f64, n_features + 1, allocator)
	defer delete(full_grad, allocator)

	converged := false
	n_iter := 0
	initial_lr := params.learning_rate

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1

		// Disable LR decay for L-BFGS so it can take full Newton steps
		if params.optimizer_type != .LBFGS {
			lr := math.max(
				initial_lr * (1.0 - f64(iter) / f64(params.max_iter)),
				initial_lr * 0.01,
			)
			optim.optimizer_set_learning_rate(&opt, lr)
		}

		for i in 0 ..< n_samples {
			x_row := X.data[i * n_features:i * n_features + n_features]
			w := params_vec[0:n_features]

			// ✅ 1. SIMD-accelerated score computation
			score := params_vec[n_features] + l.dot_simd(w, x_row)

			// ✅ 2. Compute FULL gradient vector
			if y[i] * score < 1.0 {
				for f in 0 ..< n_features {
					full_grad[f] = params_vec[f] - params.C * y[i] * x_row[f]
				}
				full_grad[n_features] = -params.C * y[i]
			} else {
				for f in 0 ..< n_features {
					full_grad[f] = params_vec[f]
				}
				full_grad[n_features] = 0.0
			}

			// ✅ 3. Single optimizer step per sample (Crucial for Adam/L-BFGS!)
			optim.optimizer_step(&opt, params_vec, full_grad)
		}
	}

	w := make([]f64, n_features, allocator)
	copy(w, params_vec[0:n_features])
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

	for i in 0 ..< n {
		x_row := X.data[i * n_features:i * n_features + n_features]
		preds[i] = model.bias + l.dot_simd(model.weights, x_row)
	}
	return preds
}

svm_free :: proc(model: ^LinearSVM) {
	if len(model.weights) > 0 {delete(model.weights, model.allocator)}
	if len(model.support_vectors) > 0 {delete(model.support_vectors, model.allocator)}
}

// ============================================================================
// Kernel SVM Structures (Dual Formulation)
// ============================================================================

KernelSVM :: struct {
	support_vectors: []int,
	alpha:           []f64,
	sv_labels:       []f64,
	bias:            f64,
	kernel_type:     SVMKernelType,
	gamma:           f64,
	degree:          int,
	coef0:           f64,
	C:               f64,
	allocator:       mem.Allocator,
}

SVMKernelType :: enum {
	Linear,
	RBF,
	Polynomial,
}

KernelSVMParams :: struct {
	C:              f64,
	kernel_type:    SVMKernelType,
	gamma:          f64,
	degree:         int,
	coef0:          f64,
	max_iter:       int,
	tol:            f64,
	learning_rate:  f64,
	optimizer_type: optim.OptimizerType,
}

// ============================================================================
// Kernel Functions (SIMD-Accelerated)
// ============================================================================

_kernel_rbf_simd :: proc(x1, x2: []f64, gamma: f64) -> f64 {
	n := len(x1)
	diff := make([]f64, n, context.temp_allocator)
	defer delete(diff, context.temp_allocator)
	l.vec_sub_simd(x1, x2, diff)
	return math.exp(-gamma * l.dot_simd(diff, diff))
}

_kernel_linear_simd :: proc(x1, x2: []f64) -> f64 {return l.dot_simd(x1, x2)}

_kernel_poly_simd :: proc(x1, x2: []f64, gamma: f64, degree: int, coef0: f64) -> f64 {
	return math.pow(gamma * l.dot_simd(x1, x2) + coef0, f64(degree))
}

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
// Public API: Fit Kernel SVM (HEAVILY OPTIMIZED)
// ============================================================================
kernel_svm_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: KernelSVMParams,
	allocator: mem.Allocator = context.allocator,
) -> KernelSVM {
	n_samples := X.rows
	n_features := X.cols

	alpha := make([]f64, n_samples, allocator)
	grad := make([]f64, n_samples, allocator)
	defer delete(grad, allocator)

	// Precompute kernel matrix
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

	opt_config := optim.optimizer_default_config(params.optimizer_type)
	opt_config.learning_rate = params.learning_rate
	opt := optim.optimizer_init(opt_config, n_samples, allocator)
	defer optim.optimizer_free(&opt)

	// ✅ Pre-allocate alpha_y for SIMD dot products
	alpha_y := make([]f64, n_samples, allocator)
	defer delete(alpha_y, allocator)
	old_alpha := make([]f64, n_samples, allocator)
	defer delete(old_alpha, allocator)

	converged := false
	n_iter := 0
	initial_lr := params.learning_rate

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1

		if params.optimizer_type != .LBFGS {
			lr := math.max(
				initial_lr * (1.0 - f64(iter) / f64(params.max_iter)),
				initial_lr * 0.01,
			)
			optim.optimizer_set_learning_rate(&opt, lr)
		}

		// ✅ 1. Precompute alpha_y ONCE per epoch
		for j in 0 ..< n_samples {alpha_y[j] = alpha[j] * y[j]}

		// 2. Compute gradients using SIMD dot products
		for i in 0 ..< n_samples {
			// ✅ Replaces O(N) inner loop with a single SIMD dot product!
			sum_term := l.dot_simd(alpha_y, K[i])
			grad[i] = y[i] * sum_term - 1.0 // Negated for descent
		}

		copy(old_alpha, alpha)
		optim.optimizer_step(&opt, alpha, grad)

		// 3. Project onto box constraint [0, C]
		max_update := 0.0
		for i in 0 ..< n_samples {
			alpha[i] = math.clamp(alpha[i], 0.0, params.C)
			update := math.abs(alpha[i] - old_alpha[i])
			if update > max_update {max_update = update}
		}

		if max_update < params.tol {
			converged = true
			break
		}
	}

	// Extract support vectors and compute bias (unchanged)
	sv_indices := make([dynamic]int, 0, allocator)
	for i in 0 ..< n_samples {if alpha[i] > 1e-5 {append(&sv_indices, i)}}

	bias := 0.0
	bias_count := 0
	for idx in sv_indices[:] {
		if alpha[idx] > 1e-5 && alpha[idx] < params.C - 1e-5 {
			score := 0.0
			for j in 0 ..< n_samples {
				if alpha[j] > 1e-5 {
					x_idx := X.data[idx * n_features:idx * n_features + n_features]
					x_j := X.data[j * n_features:j * n_features + n_features]
					score +=
						alpha[j] *
						y[j] *
						_kernel_eval(
							x_idx,
							x_j,
							params.kernel_type,
							params.gamma,
							params.degree,
							params.coef0,
						)
				}
			}
			bias += y[idx] - score
			bias_count += 1
		}
	}
	if bias_count > 0 {bias /= f64(bias_count)}

	sv_final := make([]int, len(sv_indices), allocator)
	copy(sv_final, sv_indices[:])
	alpha_final := make([]f64, len(sv_indices), allocator)
	sv_label_final := make([]f64, len(sv_indices), allocator)
	for idx, i in sv_indices[:] {
		alpha_final[i] = alpha[idx]
		sv_label_final[i] = y[idx]
	}
	delete(sv_indices)

	return KernelSVM {
		support_vectors = sv_final,
		alpha = alpha_final,
		bias = bias,
		sv_labels = sv_label_final,
		kernel_type = params.kernel_type,
		gamma = params.gamma,
		degree = params.degree,
		coef0 = params.coef0,
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
	n_features := X.cols
	preds := make([]f64, n, allocator)

	for i in 0 ..< n {
		x_i := X.data[i * n_features:i * n_features + n_features]
		score := model.bias
		for sv_idx, j in model.support_vectors {
			x_sv := X.data[sv_idx * n_features:sv_idx * n_features + n_features]
			score +=
				model.alpha[j] *
				model.sv_labels[j] *
				_kernel_eval(x_i, x_sv, model.kernel_type, model.gamma, model.degree, model.coef0)
		}
		preds[i] = score
	}
	return preds
}

kernel_svm_free :: proc(model: ^KernelSVM) {
	if len(model.support_vectors) > 0 {delete(model.support_vectors, model.allocator)}
	if len(model.alpha) > 0 {delete(model.alpha, model.allocator)}
	if len(model.sv_labels) > 0 {delete(model.sv_labels, model.allocator)}
}
