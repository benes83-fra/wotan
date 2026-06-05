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
// Internal: Context & Objective for Linear SVM L-BFGS Line Search
// ============================================================================

_LinearSVM_Context :: struct {
	X:          ^l.Matrix(f64),
	y:          []f64,
	C:          f64,
	n_samples:  int,
	n_features: int,
}

// The objective function: 0.5 * ||w||^2 + C * sum(max(0, 1 - y * (w.x + b)))
// We divide by N to match the averaged gradient!
_linear_svm_loss :: proc(params_vec: []f64, user_data: rawptr) -> f64 {
	ctx := cast(^_LinearSVM_Context)(user_data)
	w := params_vec[0:ctx.n_features]
	b := params_vec[ctx.n_features]

	// Regularization term: 0.5 * ||w||^2
	reg := 0.0
	for f in 0 ..< ctx.n_features {
		reg += w[f] * w[f]
	}
	reg *= 0.5

	// Hinge loss term: C * sum(max(0, 1 - y * (w.x + b)))
	hinge := 0.0
	for i in 0 ..< ctx.n_samples {
		x_row := ctx.X.data[i * ctx.n_features:i * ctx.n_features + ctx.n_features]
		score := b + l.dot_simd(w, x_row)
		margin := ctx.y[i] * score
		if margin < 1.0 {
			hinge += 1.0 - margin
		}
	}

	// Average the loss to perfectly match the averaged gradient!
	return (reg + ctx.C * hinge) / f64(ctx.n_samples)
}
// ============================================================================
// Public API: Fit Linear SVM (Full-Batch for L-BFGS compatibility)
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

	// Setup unified optimizer
	opt_config := optim.optimizer_default_config(params.optimizer_type)
	opt_config.learning_rate = params.learning_rate
	opt := optim.optimizer_init(opt_config, n_features + 1, allocator)
	defer optim.optimizer_free(&opt)

	// Allocate full_grad ONCE outside the loop
	full_grad := make([]f64, n_features + 1, allocator)
	defer delete(full_grad, allocator)
	ctx := _LinearSVM_Context {
		X          = X,
		y          = y,
		C          = params.C,
		n_samples  = n_samples,
		n_features = n_features,
	}
	converged := false
	n_iter := 0
	initial_lr := params.learning_rate

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1

		// Disable LR decay for L-BFGS
		if params.optimizer_type != .LBFGS {
			lr := math.max(
				initial_lr * (1.0 - f64(iter) / f64(params.max_iter)),
				initial_lr * 0.01,
			)
			optim.optimizer_set_learning_rate(&opt, lr)
		}

		// ✅ 1. Zero out the full batch gradient
		for f in 0 ..< n_features + 1 {
			full_grad[f] = 0.0
		}

		// ✅ 2. Accumulate gradients over ALL samples
		for i in 0 ..< n_samples {
			x_row := X.data[i * n_features:i * n_features + n_features]
			w := params_vec[0:n_features]

			score := params_vec[n_features] + l.dot_simd(w, x_row)

			if y[i] * score < 1.0 {
				for f in 0 ..< n_features {
					full_grad[f] += params_vec[f] - params.C * y[i] * x_row[f]
				}
				full_grad[n_features] += -params.C * y[i]
			} else {
				for f in 0 ..< n_features {
					full_grad[f] += params_vec[f]
				}
			}
		}

		// ✅ 3. Average the gradient (CRITICAL for L-BFGS stability!)
		inv_n := 1.0 / f64(n_samples)
		for f in 0 ..< n_features + 1 {
			full_grad[f] *= inv_n
		}

		loss := _linear_svm_loss(params_vec, &ctx)

		// ✅ 5. Single optimizer step per epoch WITH line search!
		optim.optimizer_step(&opt, params_vec, full_grad, loss, _linear_svm_loss, &ctx)
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
	sv_data:         l.Matrix(f64),
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
// Internal: Context & Objective for Kernel SVM L-BFGS Line Search
// ============================================================================

_KernelSVM_Context :: struct {
	y:         []f64,
	K:         [][]f64,
	n_samples: int,
	Q_alpha:   []f64, // Precomputed Q * alpha for the current epoch
}

// The dual objective to minimize: 0.5 * alpha^T Q alpha - sum(alpha)
_kernel_svm_loss :: proc(alpha: []f64, user_data: rawptr) -> f64 {
	ctx := cast(^_KernelSVM_Context)(user_data)
	n := ctx.n_samples

	// Quadratic term: 0.5 * sum(alpha_i * Q_alpha_i)
	quad_term := 0.0
	for i in 0 ..< n {
		quad_term += alpha[i] * ctx.Q_alpha[i]
	}
	quad_term *= 0.5

	// Linear term: -sum(alpha)
	lin_term := 0.0
	for i in 0 ..< n {
		lin_term -= alpha[i]
	}

	return quad_term + lin_term
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
	// ✅ Allocate Q_alpha and Context for L-BFGS line search
	Q_alpha := make([]f64, n_samples, allocator)

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
	defer delete(Q_alpha, allocator)

	ctx := _KernelSVM_Context {
		y         = y,
		K         = K,
		n_samples = n_samples,
		Q_alpha   = Q_alpha,
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


		// ✅ 1. Precompute alpha_y and Q_alpha ONCE per epoch
		for j in 0 ..< n_samples {alpha_y[j] = alpha[j] * y[j]}

		// 2. Compute gradients and Q_alpha using SIMD dot products
		for i in 0 ..< n_samples {
			sum_term := l.dot_simd(alpha_y, K[i])
			Q_alpha[i] = y[i] * sum_term // ✅ Store for loss function
			grad[i] = Q_alpha[i] - 1.0 // Negated for descent
		}

		copy(old_alpha, alpha)

		// ✅ 3. Compute loss and pass to optimizer for line search!
		loss := _kernel_svm_loss(alpha, &ctx)
		optim.optimizer_step(&opt, alpha, grad, loss, _kernel_svm_loss, &ctx)

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

	// Copy support vector DATA to output
	sv_final := make([]int, len(sv_indices), allocator)
	copy(sv_final, sv_indices[:])

	// ✅ Allocate and copy the actual feature data for the support vectors
	n_sv := len(sv_indices)
	sv_data := l.matrix_new(f64, n_sv, n_features, allocator)
	for idx, i in sv_indices[:] {
		for j in 0 ..< n_features {
			sv_data.data[i * n_features + j] = X.data[idx * n_features + j]
		}
	}

	alpha_final := make([]f64, len(sv_indices), allocator)
	sv_label_final := make([]f64, len(sv_indices), allocator)
	for idx, i in sv_indices[:] {
		alpha_final[i] = alpha[idx]
		sv_label_final[i] = y[idx]
	}
	delete(sv_indices)

	return KernelSVM {
		support_vectors = sv_final,
		sv_data         = sv_data, // ✅ Assign the new field
		alpha           = alpha_final,
		bias            = bias,
		sv_labels       = sv_label_final,
		kernel_type     = params.kernel_type,
		gamma           = params.gamma,
		degree          = params.degree,
		coef0           = params.coef0,
		C               = params.C,
		allocator       = allocator,
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

		// ✅ Iterate over the stored support vector data
		for j in 0 ..< len(model.support_vectors) {
			x_sv := model.sv_data.data[j * n_features:j * n_features + n_features]
			k_val := _kernel_eval(
				x_i,
				x_sv,
				model.kernel_type,
				model.gamma,
				model.degree,
				model.coef0,
			)
			score += model.alpha[j] * model.sv_labels[j] * k_val
		}
		preds[i] = score
	}
	return preds
}

kernel_svm_free :: proc(model: ^KernelSVM) {
	if len(model.support_vectors) > 0 {delete(model.support_vectors, model.allocator)}
	if model.sv_data.data != nil {l.matrix_free(&model.sv_data)} 	// ✅ Free the matrix
	if len(model.alpha) > 0 {delete(model.alpha, model.allocator)}
	if len(model.sv_labels) > 0 {delete(model.sv_labels, model.allocator)}
}
