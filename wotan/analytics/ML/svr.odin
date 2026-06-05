package ML

import l "../../linalg"
import optim "../../optimize"
import "core:math"
import "core:mem"

// ============================================================================
// Support Vector Regression (SVR) Structures
// ============================================================================

SupportVectorRegression :: struct {
	support_vectors: []int,
	sv_data:         l.Matrix(f64), // ✅ NEW
	alpha_diff:      []f64,
	bias:            f64,
	kernel_type:     SVMKernelType,
	gamma:           f64,
	degree:          int,
	coef0:           f64,
	C:               f64,
	epsilon:         f64,
	converged:       bool,
	n_iter:          int,
	allocator:       mem.Allocator,
}
SVRParams :: struct {
	C:              f64,
	epsilon:        f64,
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
// Internal: Context & Objective for L-BFGS Line Search
// ============================================================================

_SVR_Context :: struct {
	y:         []f64,
	K:         [][]f64,
	epsilon:   f64,
	n_samples: int,
	// ✅ NEW: Precomputed K * beta to make loss O(N) instead of O(N^2)
}

// ✅ OPTIMIZED: The dual objective function.
// By using the precomputed K_beta, this is now O(N) instead of O(N^2).
// This makes L-BFGS line searches incredibly fast.
// The dual objective function to minimize.
// ✅ FIX: Computes K_beta dynamically for the trial 'v' passed by L-BFGS!
_svr_loss :: proc(v: []f64, user_data: rawptr) -> f64 {
	ctx := cast(^_SVR_Context)(user_data)
	n := ctx.n_samples

	// Quadratic term: 0.5 * beta^T K beta
	quad_term := 0.0
	for i in 0 ..< n {
		beta_i := v[i] - v[i + n]
		if beta_i == 0.0 {continue}

		// Compute K_beta[i] dynamically for this specific trial v
		sum := 0.0
		for j in 0 ..< n {
			beta_j := v[j] - v[j + n]
			sum += beta_j * ctx.K[i][j]
		}
		quad_term += beta_i * sum
	}
	quad_term *= 0.5

	// Linear term: epsilon * sum(alpha + alpha*) - sum(y * beta)
	lin_term := 0.0
	for i in 0 ..< n {
		lin_term += ctx.epsilon * (v[i] + v[i + n])
		lin_term -= ctx.y[i] * (v[i] - v[i + n])
	}

	return quad_term + lin_term
}
// ============================================================================
// Public API: Fit Support Vector Regression
// ============================================================================
svr_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: SVRParams,
	allocator: mem.Allocator = context.allocator,
) -> SupportVectorRegression {
	n_samples := X.rows
	n_features := X.cols

	v := make([]f64, 2 * n_samples, allocator)
	grad := make([]f64, 2 * n_samples, allocator)
	defer delete(v, allocator)
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
	opt := optim.optimizer_init(opt_config, 2 * n_samples, allocator)
	defer optim.optimizer_free(&opt)

	// ✅ Allocate K_beta and add it to the context
	K_beta := make([]f64, n_samples, allocator)
	defer delete(K_beta, allocator)

	ctx := _SVR_Context {
		y         = y,
		K         = K,
		epsilon   = params.epsilon,
		n_samples = n_samples,
	}

	beta := make([]f64, n_samples, allocator)
	defer delete(beta, allocator)
	old_v := make([]f64, 2 * n_samples, allocator)
	defer delete(old_v, allocator)

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

		// Compute beta = alpha - alpha_star
		for i in 0 ..< n_samples {
			beta[i] = v[i] - v[i + n_samples]
		}

		// ✅ Compute K_beta ONCE per epoch using SIMD
		for i in 0 ..< n_samples {
			sum_term := l.dot_simd(beta, K[i])
			grad[i] = sum_term - y[i] + params.epsilon
			grad[i + n_samples] = -sum_term + y[i] + params.epsilon
		}

		copy(old_v, v)
		loss := _svr_loss(v, &ctx)
		optim.optimizer_step(&opt, v, grad, loss, _svr_loss, &ctx)

		// Project onto box constraint [0, C]
		max_update := 0.0
		for i in 0 ..< 2 * n_samples {
			v[i] = math.clamp(v[i], 0.0, params.C)
			update := math.abs(v[i] - old_v[i])
			if update > max_update {max_update = update}
		}

		if max_update < params.tol {
			converged = true
			break
		}
	}

	// ✅ CRITICAL BUG FIX: Recompute beta from the final clamped v!
	// If we don't do this, the bias calculation will use the unclamped beta.
	for i in 0 ..< n_samples {
		beta[i] = v[i] - v[i + n_samples]
	}

	// Extract support vectors
	sv_indices := make([dynamic]int, 0, allocator)
	alpha_diff_final := make([dynamic]f64, 0, allocator)

	for i in 0 ..< n_samples {
		alpha := v[i]
		alpha_star := v[i + n_samples]
		if alpha > 1e-5 || alpha_star > 1e-5 {
			append(&sv_indices, i)
			append(&alpha_diff_final, alpha - alpha_star)
		}
	}

	// Compute bias b
	bias := 0.0
	bias_count := 0

	for idx in sv_indices[:] {
		alpha := v[idx]
		alpha_star := v[idx + n_samples]

		// ✅ Use SIMD dot product for bias calculation
		sum_term := l.dot_simd(beta, K[idx])

		if alpha > 1e-5 && alpha < params.C - 1e-5 {
			bias += y[idx] - sum_term - params.epsilon
			bias_count += 1
		} else if alpha_star > 1e-5 && alpha_star < params.C - 1e-5 {
			bias += y[idx] - sum_term + params.epsilon
			bias_count += 1
		}
	}

	if bias_count > 0 {
		bias /= f64(bias_count)
	} else {
		// Fallback if no free support vectors
		for idx in sv_indices[:] {
			sum_term := l.dot_simd(beta, K[idx])
			bias += y[idx] - sum_term
			bias_count += 1
		}
		if bias_count > 0 {bias /= f64(bias_count)}
	}

	// Copy to final slices
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

	alpha_diff_slice := make([]f64, len(alpha_diff_final), allocator)
	copy(alpha_diff_slice, alpha_diff_final[:])

	delete(sv_indices)
	delete(alpha_diff_final)

	return SupportVectorRegression {
		support_vectors = sv_final,
		sv_data         = sv_data, // ✅ Assign the new field
		alpha_diff      = alpha_diff_slice,
		bias            = bias,
		kernel_type     = params.kernel_type,
		gamma           = params.gamma,
		degree          = params.degree,
		coef0           = params.coef0,
		C               = params.C,
		epsilon         = params.epsilon,
		converged       = converged,
		n_iter          = n_iter,
		allocator       = allocator,
	}
}

// ============================================================================
// Public API: Predict with SVR
// ============================================================================
svr_predict :: proc(
	model: ^SupportVectorRegression,
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
			score += model.alpha_diff[j] * k_val
		}
		preds[i] = score
	}

	return preds
}

// ============================================================================
// Public API: Free Resources
// ============================================================================
svr_free :: proc(model: ^SupportVectorRegression) {
	if len(model.support_vectors) > 0 {delete(model.support_vectors, model.allocator)}
	if model.sv_data.data != nil {l.matrix_free(&model.sv_data)} 	// ✅ Free the matrix
	if len(model.alpha_diff) > 0 {delete(model.alpha_diff, model.allocator)}
}
