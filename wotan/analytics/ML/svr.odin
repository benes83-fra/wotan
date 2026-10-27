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
// Public API: Fit Support Vector Regression (ε-SVR dual, solved exactly by SMO)
//   min ½ (α-α*)ᵀK(α-α*) + ε Σ(α+α*) - Σ y_i (α_i-α*_i)
//   s.t. 0 ≤ α, α* ≤ C,  Σ(α_i - α*_i) = 0
//   f(x) = Σ (α_i-α*_i) K(x_i, x) + b,  b = -ρ  (= sklearn SVR)
// gamma ≤ 0 means sklearn's "scale"; tol: KKT tolerance (≤ 0: 1e-3);
// max_iter: passes over the data (≤ 0: no limit). learning_rate /
// optimizer_type are not used by the exact solver.
// sample_weight (optional): α_i, α*_i ≤ C · sample_weight[i]; samples with
// weight 0 are removed before training (sklearn SVR).
// ============================================================================
svr_fit :: proc(
	X_in: ^l.Matrix(f64),
	y_in: []f64,
	params: SVRParams,
	allocator: mem.Allocator = context.allocator,
	sample_weight: []f64 = nil,
) -> SupportVectorRegression {
	if len(y_in) != X_in.rows do panic("svr_fit: y length mismatch")
	if X_in.rows == 0 do panic("svr_fit: no samples")
	sub := _svm_weighted_subset(X_in, y_in, sample_weight, .None, params.C, 2)
	defer _svm_subset_free(&sub)
	X, y := &sub.X, sub.y
	n_samples := X.rows
	n_features := X.cols

	gamma := params.gamma if params.gamma > 0 else _svm_gamma_scale(X)
	k := _smo_kernel_init(X, params.kernel_type, gamma, params.degree, params.coef0, context.temp_allocator)
	defer _smo_kernel_free(&k)

	// 2n variables: α (sign +1, p = ε - y) then α* (sign -1, p = ε + y)
	L := 2 * n_samples
	v := make([]f64, L, context.temp_allocator)
	defer delete(v, context.temp_allocator)
	p := make([]f64, L, context.temp_allocator)
	defer delete(p, context.temp_allocator)
	sgn := make([]f64, L, context.temp_allocator)
	defer delete(sgn, context.temp_allocator)
	for i in 0 ..< n_samples {
		p[i], sgn[i] = params.epsilon - y[i], 1.0
		p[i + n_samples], sgn[i + n_samples] = params.epsilon + y[i], -1.0
	}

	rho, converged, n_iter := _smo_solve(&k, p, sgn, sub.Cs, params.tol, params.max_iter * L, true, v)

	// Support vectors: α_i - α*_i ≠ 0
	sv_indices := make([dynamic]int, 0, allocator)
	for i in 0 ..< n_samples {
		if v[i] - v[i + n_samples] != 0 {append(&sv_indices, i)}
	}
	n_sv := len(sv_indices)
	sv_final := make([]int, n_sv, allocator)
	for idx, i in sv_indices {sv_final[i] = sub.rows[idx]} // index into the caller's X
	sv_data := l.matrix_new(f64, n_sv, n_features, allocator)
	alpha_diff_slice := make([]f64, n_sv, allocator)
	for idx, i in sv_indices[:] {
		for j in 0 ..< n_features {
			sv_data.data[i * n_features + j] = X.data[idx * n_features + j]
		}
		alpha_diff_slice[i] = v[idx] - v[idx + n_samples]
	}
	delete(sv_indices)

	return SupportVectorRegression {
		support_vectors = sv_final,
		sv_data         = sv_data,
		alpha_diff      = alpha_diff_slice,
		bias            = -rho,
		kernel_type     = params.kernel_type,
		gamma           = gamma,
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
