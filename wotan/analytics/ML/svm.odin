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
	class_weight:   ClassWeight, // C of class c is C · class_weight[c] (sklearn class_weight)
}
// ============================================================================
// Public API: Fit Linear SVM
// Primal: min ½||w||² + C Σ max(0, 1 - y_i (w·x_i + b)), labels y ∈ {-1, +1};
// b is not penalised (= sklearn SVC(kernel="linear", C)). Solved exactly in the
// dual by SMO (fit_intercept) or dual coordinate descent (no intercept, b = 0).
// tol: KKT tolerance (≤ 0: 1e-3, sklearn's default); max_iter: passes over the
// data (≤ 0: no limit), i.e. at most max_iter·n SMO updates.
// learning_rate / optimizer_type are not used by the exact solver.
// sample_weight (optional): C_i = C · sample_weight[i] · class_weight[y_i], as
// LIBSVM / sklearn; samples with weight 0 are removed before training.
// ============================================================================
svm_fit_linear :: proc(
	X_in: ^l.Matrix(f64),
	y_in: []f64,
	params: SVMParams,
	allocator: mem.Allocator = context.allocator,
	sample_weight: []f64 = nil,
) -> LinearSVM {
	_svm_check_labels(y_in, X_in.rows, "svm_fit_linear")
	sub := _svm_weighted_subset(X_in, y_in, sample_weight, params.class_weight, params.C, 1)
	defer _svm_subset_free(&sub)
	X, y := &sub.X, sub.y
	n_samples := X.rows
	n_features := X.cols

	k := _smo_kernel_init(X, .Linear, 0, 0, 0, context.temp_allocator)
	defer _smo_kernel_free(&k)
	alpha := make([]f64, n_samples, context.temp_allocator)
	defer delete(alpha, context.temp_allocator)
	p := make([]f64, n_samples, context.temp_allocator)
	defer delete(p, context.temp_allocator)
	for i in 0 ..< n_samples {p[i] = -1.0}

	rho, converged, n_iter := _smo_solve(
		&k,
		p,
		y,
		sub.Cs,
		params.tol,
		params.max_iter * max(n_samples, 1),
		params.fit_intercept,
		alpha,
	)

	// w = Σ α_i y_i x_i, b = -ρ
	w := make([]f64, n_features, allocator)
	sv := make([dynamic]int, 0, allocator)
	for i in 0 ..< n_samples {
		if alpha[i] <= 0 do continue
		append(&sv, sub.rows[i]) // index into the caller's X
		for f in 0 ..< n_features {
			w[f] += alpha[i] * y[i] * X.data[i * n_features + f]
		}
	}
	sv_final := make([]int, len(sv), allocator)
	copy(sv_final, sv[:])
	delete(sv)

	return LinearSVM {
		weights = w,
		bias = -rho,
		support_vectors = sv_final,
		n_iter = n_iter,
		converged = converged,
		allocator = allocator,
	}
}

// Rows with non-zero effective weight and their bounds C_i = C · weight
// (repeated `copies` times: ε-SVR has two dual variables per sample).
// Without weights it is a plain view of X / y with C_i = C.
_SVMSubset :: struct {
	X:     l.Matrix(f64),
	y:     []f64,
	Cs:    []f64,
	rows:  []int, // original row of every kept sample
	owned: bool,
}

_svm_weighted_subset :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	sample_weight: []f64,
	class_weight: ClassWeight,
	C: f64,
	copies: int,
) -> _SVMSubset {
	n := X.rows
	w := _effective_weights(y, sample_weight, class_weight, "svm", context.temp_allocator)
	defer delete(w, context.temp_allocator)
	s: _SVMSubset
	kept := 0
	for i in 0 ..< n {if w == nil || w[i] > 0 do kept += 1}
	s.rows = make([]int, kept, context.temp_allocator)
	s.Cs = make([]f64, kept * copies, context.temp_allocator)
	j := 0
	for i in 0 ..< n {
		if w != nil && w[i] <= 0 do continue
		s.rows[j] = i
		wi := 1.0
		if w != nil do wi = w[i]
		for c in 0 ..< copies {s.Cs[c * kept + j] = C * wi}
		j += 1
	}
	if kept == n {
		s.X, s.y = X^, y
	} else {
		s.owned = true
		s.X = subset_matrix(X, s.rows, context.temp_allocator)
		s.y = subset_slice(y, s.rows, context.temp_allocator)
	}
	return s
}

_svm_subset_free :: proc(s: ^_SVMSubset) {
	if s.owned {
		l.matrix_free(&s.X)
		delete(s.y, context.temp_allocator)
	}
	delete(s.rows, context.temp_allocator)
	delete(s.Cs, context.temp_allocator)
}

_svm_check_labels :: proc(y: []f64, n: int, name: string) {
	if len(y) != n do panic("svm: y length mismatch")
	if n == 0 do panic("svm: no samples")
	for v in y {
		if v != 1.0 && v != -1.0 do panic("svm: labels must be -1 or +1")
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
	if X.cols != n_features do panic("svm_predict_linear: X has the wrong number of columns")
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
	class_weight:   ClassWeight, // C of class c is C · class_weight[c] (sklearn class_weight)
}

// ============================================================================
// Kernel Functions: K(x,z) = x·z | exp(-γ||x-z||²) | (γ x·z + coef0)^degree
// (sklearn's definitions; gamma ≤ 0 in the fit params means sklearn's "scale")
// ============================================================================

_kernel_rbf_simd :: proc(x1, x2: []f64, gamma: f64) -> f64 {
	d2 := 0.0
	for i in 0 ..< len(x1) {
		d := x1[i] - x2[i]
		d2 += d * d
	}
	return math.exp(-gamma * d2)
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
// Public API: Fit Kernel SVM (C-SVC dual, solved exactly by SMO)
//   min ½ αᵀQα - Σα, 0 ≤ α ≤ C, yᵀα = 0, Q_ij = y_i y_j K(x_i, x_j)
//   decision(x) = Σ α_i y_i K(x_i, x) + b,  b = -ρ  (= sklearn SVC)
// tol: KKT tolerance (≤ 0: 1e-3); max_iter: passes over the data (≤ 0: no
// limit). learning_rate / optimizer_type are not used by the exact solver.
// sample_weight (optional): C_i = C · sample_weight[i] · class_weight[y_i];
// samples with weight 0 are removed before training (gamma="scale" is then
// computed on the remaining rows, as sklearn).
// ============================================================================
kernel_svm_fit :: proc(
	X_in: ^l.Matrix(f64),
	y_in: []f64,
	params: KernelSVMParams,
	allocator: mem.Allocator = context.allocator,
	sample_weight: []f64 = nil,
) -> KernelSVM {
	_svm_check_labels(y_in, X_in.rows, "kernel_svm_fit")
	sub := _svm_weighted_subset(X_in, y_in, sample_weight, params.class_weight, params.C, 1)
	defer _svm_subset_free(&sub)
	X, y := &sub.X, sub.y
	n_samples := X.rows
	n_features := X.cols

	gamma := params.gamma if params.gamma > 0 else _svm_gamma_scale(X)
	k := _smo_kernel_init(X, params.kernel_type, gamma, params.degree, params.coef0, context.temp_allocator)
	defer _smo_kernel_free(&k)
	alpha := make([]f64, n_samples, context.temp_allocator)
	defer delete(alpha, context.temp_allocator)
	p := make([]f64, n_samples, context.temp_allocator)
	defer delete(p, context.temp_allocator)
	for i in 0 ..< n_samples {p[i] = -1.0}

	rho, _, _ := _smo_solve(&k, p, y, sub.Cs, params.tol, params.max_iter * n_samples, true, alpha)

	// Support vectors: α > 0
	sv_indices := make([dynamic]int, 0, allocator)
	for i in 0 ..< n_samples {if alpha[i] > 0 {append(&sv_indices, i)}}
	n_sv := len(sv_indices)

	sv_final := make([]int, n_sv, allocator)
	for idx, i in sv_indices {sv_final[i] = sub.rows[idx]} // index into the caller's X
	sv_data := l.matrix_new(f64, n_sv, n_features, allocator)
	alpha_final := make([]f64, n_sv, allocator)
	sv_label_final := make([]f64, n_sv, allocator)
	for idx, i in sv_indices[:] {
		for j in 0 ..< n_features {
			sv_data.data[i * n_features + j] = X.data[idx * n_features + j]
		}
		alpha_final[i] = alpha[idx]
		sv_label_final[i] = y[idx]
	}
	delete(sv_indices)

	return KernelSVM {
		support_vectors = sv_final,
		sv_data         = sv_data,
		alpha           = alpha_final,
		bias            = -rho,
		sv_labels       = sv_label_final,
		kernel_type     = params.kernel_type,
		gamma           = gamma,
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
