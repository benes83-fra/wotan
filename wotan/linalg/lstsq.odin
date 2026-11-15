package wotan_linalg

import "core:math"
import "core:mem"

// ============================================================================
// Least Squares Method Selection
// ============================================================================
LstsqMethod :: enum {
	Auto,
	QR,
	Cholesky,
	SVD,
}

// ============================================================================
// Least Squares Result
// ============================================================================
// condition_estimate is cond₂(X) = σ_max / σ_min for every method: exact from
// the singular values for .SVD (over the rank part: the singular values above
// the rank tolerance), and estimated from the triangular factor for .QR (R) and
// .Cholesky (L, with XᵀX = LLᵀ), a lower bound usually correct to several
// digits. +Inf for X = 0; 0 for an empty X.
LstsqResult :: struct {
	beta:               []f64,
	residuals:          []f64,
	rank:               int,
	condition_estimate: f64, // cond₂(X), see above
	method_used:        LstsqMethod,
	success:            bool,
}

// ============================================================================
// Main Least Squares Solver
// ============================================================================
lstsq :: proc(
	X: ^Matrix(f64),
	y: []f64,
	method: LstsqMethod = .Auto,
	allocator: mem.Allocator = context.allocator,
) -> LstsqResult {
	m, n := X.rows, X.cols

	// Edge cases
	if m == 0 || n == 0 || len(y) == 0 {
		result := LstsqResult{}
		result.beta = make([]f64, n, allocator)
		result.residuals = make([]f64, m, allocator)
		result.rank = 0
		result.condition_estimate = 0.0
		result.method_used = method
		result.success = true
		return result
	}
	if len(y) != m {
		panic("lstsq: dimension mismatch")
	}

	// Auto-select method
	selected := method
	if method == .Auto {
		selected = _lstsq_auto_select(X)
	}

	// Dispatch
	if selected == .QR {
		return _lstsq_qr(X, y, allocator)
	} else if selected == .Cholesky {
		return _lstsq_cholesky(X, y, allocator)
	} else if selected == .SVD {
		return _lstsq_svd(X, y, allocator)
	}
	return _lstsq_qr(X, y, allocator)
}

// ============================================================================
// Auto-select helper
// ============================================================================
_lstsq_auto_select :: proc(X: ^Matrix(f64)) -> LstsqMethod {
	m, n := X.rows, X.cols
	if m < 50 && n < 20 {
		return .QR
	}
	if m >= 5 * n {
		return .QR
	}

	col_norms := make([]f64, n, context.temp_allocator)
	for j in 0 ..< n {
		norm := 0.0
		for i in 0 ..< m {
			v := X.data[i * X.cols + j]
			norm += v * v
		}
		col_norms[j] = math.sqrt(norm)
	}

	max_n := 0.0
	min_n := f64(1e308)
	for j in 0 ..< n {
		cn := col_norms[j]
		if cn > max_n {
			max_n = cn
		}
		if cn > 0.0 && cn < min_n {
			min_n = cn
		}
	}

	if min_n < f64(1e308) && max_n / min_n > f64(1e6) {
		return .SVD
	}
	return .Cholesky
}

// ============================================================================
// QR-based least squares
// ============================================================================
_lstsq_qr :: proc(X: ^Matrix(f64), y: []f64, allocator: mem.Allocator) -> LstsqResult {
	// Underdetermined systems (fewer rows than columns) have no unique solution:
	// return the minimum-norm one from the SVD, as numpy.linalg.lstsq does
	if X.rows < X.cols {return _lstsq_svd(X, y, allocator)}
	res := _lstsq_householder(X, y, allocator)
	// Rank-deficient or numerically singular X (cond(R) ≥ 1e12): the triangular
	// solve divides by a ~0 pivot. Fall back to the SVD minimum-norm solution,
	// as the Cholesky path does
	if !res.success {
		lstsq_result_free(&res, allocator)
		return _lstsq_svd(X, y, allocator)
	}
	return res
}

// Householder QR least squares for m ≥ n: the reflections are applied to a copy
// of X and to y as they are formed, so Q (m×m) is never built. O(m·n²) time and
// O(m·n) memory, against O(m²·n) and O(m²) for forming Q, which made tall
// problems (thousands of rows) impractical.
@(private = "file")
_lstsq_householder :: proc(X: ^Matrix(f64), y: []f64, allocator: mem.Allocator) -> LstsqResult {
	m, n := X.rows, X.cols
	R := make([]f64, m * n, context.temp_allocator) // row-major copy, becomes R
	defer delete(R, context.temp_allocator)
	for i in 0 ..< m {copy(R[i * n:][:n], X.data[i * X.cols:][:n])}
	qty := make([]f64, m, context.temp_allocator) // becomes Qᵀy
	defer delete(qty, context.temp_allocator)
	copy(qty, y)
	v := make([]f64, m, context.temp_allocator)
	defer delete(v, context.temp_allocator)

	for k in 0 ..< n {
		norm := 0.0
		for i in k ..< m {norm += R[i * n + k] * R[i * n + k]}
		norm = math.sqrt(norm)
		if norm == 0 {continue}
		alpha := -norm if R[k * n + k] >= 0 else norm
		// v = x − alpha·e_1, H = I − 2vvᵀ/(vᵀv)
		vv := 0.0
		for i in k ..< m {
			v[i] = R[i * n + k]
			if i == k {v[i] -= alpha}
			vv += v[i] * v[i]
		}
		if vv == 0 {continue}
		for j in k ..< n {
			s := 0.0
			for i in k ..< m {s += v[i] * R[i * n + j]}
			s *= 2.0 / vv
			for i in k ..< m {R[i * n + j] -= s * v[i]}
		}
		s := 0.0
		for i in k ..< m {s += v[i] * qty[i]}
		s *= 2.0 / vv
		for i in k ..< m {qty[i] -= s * v[i]}
	}

	// Back substitution R[:n, :n]·beta = (Qᵀy)[:n]
	beta := make([]f64, n, allocator)
	for i := n - 1; i >= 0; i -= 1 {
		s := qty[i]
		for j in i + 1 ..< n {s -= R[i * n + j] * beta[j]}
		beta[i] = s / R[i * n + i]
	}

	residuals := make([]f64, m, allocator)
	for i in 0 ..< m {
		pred := 0.0
		for j in 0 ..< n {pred += X.data[i * X.cols + j] * beta[j]}
		residuals[i] = y[i] - pred
	}

	rank := 0
	eps := 1e-12 * abs(R[0])
	for j in 0 ..< n {if abs(R[j * n + j]) > eps {rank += 1}}
	cond := _cond2_upper(R, n, n) // cond₂(R) = cond₂(X)

	return LstsqResult {
		beta               = beta,
		residuals          = residuals,
		rank               = rank,
		condition_estimate = cond,
		method_used        = .QR,
		success            = cond < 1e12, // larger: R is numerically singular
	}
}

// ============================================================================
// Cholesky-based least squares
// NOTE: cholesky_decompose modifies matrix in-place (returns void)
// ============================================================================
_lstsq_cholesky :: proc(X: ^Matrix(f64), y: []f64, allocator: mem.Allocator) -> LstsqResult {
	m, n := X.rows, X.cols

	// Compute AtA = XᵀX (n×n symmetric)
	AtA := xtx_simd(X, allocator)
	defer matrix_free(&AtA)

	// Compute Xᵀy (n×1)
	Xty := make([]f64, n, allocator)
	for j in 0 ..< n {
		sum := 0.0
		for i in 0 ..< m {
			sum += X.data[i * X.cols + j] * y[i]
		}
		Xty[j] = sum
	}

	// Cholesky: AtA = L Lᵀ (in-place, lower triangle). Rank-deficient X makes
	// XᵀX singular: fall back to the SVD (minimum-norm) solution
	if !cholesky_try(&AtA) {
		delete(Xty, allocator)
		return _lstsq_svd(X, y, allocator)
	}

	// L z = Xᵀy, then Lᵀ beta = z
	z := forward_substitute_simd(&AtA, Xty, context.temp_allocator)
	beta := backward_substitute_simd(&AtA, z, allocator)
	delete(z, context.temp_allocator)
	delete(Xty, allocator)

	// Residuals
	residuals := make([]f64, m, allocator)
	for i in 0 ..< m {
		pred := 0.0
		for j in 0 ..< n {
			pred += X.data[i * X.cols + j] * beta[j]
		}
		residuals[i] = y[i] - pred
	}

	// cond₂(X) = cond₂(Lᵀ), Lᵀ upper triangular (L is the lower triangle of AtA)
	Lt := make([]f64, n * n, context.temp_allocator)
	defer delete(Lt, context.temp_allocator)
	for i in 0 ..< n {for j in 0 ..= i {Lt[j * n + i] = AtA.data[i * AtA.cols + j]}}
	cond := _cond2_upper(Lt, n, n)
	// Numerically singular normal equations (cond(XᵀX) = cond(X)² ≥ 1e12): the
	// SVD solution is the reliable one
	if !(cond < f64(1e6)) {
		delete(beta, allocator)
		delete(residuals, allocator)
		return _lstsq_svd(X, y, allocator)
	}

	result := LstsqResult{}
	result.beta = beta
	result.residuals = residuals
	result.rank = n
	result.condition_estimate = cond
	result.method_used = .Cholesky
	result.success = true
	return result
}

// ============================================================================
// SVD-based least squares
// ============================================================================
_lstsq_svd :: proc(X: ^Matrix(f64), y: []f64, allocator: mem.Allocator) -> LstsqResult {
	m, n := X.rows, X.cols

	U, S, V := svd(X, .Jacobi, allocator)
	defer matrix_free(&U)
	defer matrix_free(&V)
	defer delete(S, allocator)

	S_max := 0.0
	for j in 0 ..< n {
		if S[j] > S_max {
			S_max = S[j]
		}
	}
	tol := f64(max(m, n)) * f64(1e-12) * S_max
	rank := 0
	for j in 0 ..< n {
		if S[j] > tol {
			rank += 1
		}
	}

	c := make([]f64, n, allocator)
	for j in 0 ..< n {
		sum := 0.0
		for i in 0 ..< m {
			sum += U.data[i * U.cols + j] * y[i]
		}
		c[j] = sum
	}
	defer delete(c, allocator)

	d := make([]f64, n, allocator)
	for j in 0 ..< n {
		if S[j] > tol {
			d[j] = c[j] / S[j]
		} else {
			d[j] = 0.0
		}
	}
	defer delete(d, allocator)

	beta := matvec_dyn_simd(&V, d, allocator)

	residuals := make([]f64, m, allocator)
	for i in 0 ..< m {
		pred := 0.0
		for j in 0 ..< n {
			pred += X.data[i * X.cols + j] * beta[j]
		}
		residuals[i] = y[i] - pred
	}

	cond := math.inf_f64(1)
	if rank > 0 {cond = S[0] / S[rank - 1]}

	result := LstsqResult{}
	result.beta = beta
	result.residuals = residuals
	result.rank = rank
	result.condition_estimate = cond
	result.method_used = .SVD
	result.success = rank > 0 && cond < f64(1e12)
	return result
}

// ============================================================================
// Condition number of a triangular factor
// ============================================================================
// cond₂(R) = σ_max / σ_min of an n×n upper triangular R (row-major, row stride
// `stride`), estimated by power iteration on RᵀR (σ_max) and inverse iteration
// through the triangular solves (σ_min). O(n²) per step. Both iterations
// approach their singular value from inside, so the estimate is a lower bound
// on cond₂(R); it is +Inf when a diagonal entry is zero. For R from X = QR (or
// R = Lᵀ from XᵀX = LLᵀ), cond₂(R) = cond₂(X).
@(private = "file")
_cond2_upper :: proc(R: []f64, n, stride: int) -> f64 {
	if n == 0 {return 0}
	for i in 0 ..< n {if R[i * stride + i] == 0 {return math.inf_f64(1)}}
	scratch := context.temp_allocator
	v := make([]f64, n, scratch)
	w := make([]f64, n, scratch)
	defer {delete(w, scratch); delete(v, scratch)}

	start :: proc(v: []f64) { 	// fixed start vector with no special structure
		s := 0.0
		for i in 0 ..< len(v) {v[i] = 1 + 0.5 * math.sin(f64(i + 1)); s += v[i] * v[i]}
		s = math.sqrt(s)
		for i in 0 ..< len(v) {v[i] /= s}
	}
	normalize :: proc(v: []f64) -> f64 {
		s := math.sqrt(dot_simd(v, v))
		if s > 0 && !math.is_inf(s) {for i in 0 ..< len(v) {v[i] /= s}}
		return s
	}
	MAX_ITS :: 50
	RTOL :: 1e-6

	// σ_max²: largest eigenvalue of RᵀR
	start(v)
	lam_max := 0.0
	for _ in 0 ..< MAX_ITS {
		for i in 0 ..< n { 	// w = R·v
			s := 0.0
			for j in i ..< n {s += R[i * stride + j] * v[j]}
			w[i] = s
		}
		for j in 0 ..< n {v[j] = 0}
		for i in 0 ..< n { 	// v = Rᵀ·w
			wi := w[i]
			for j in i ..< n {v[j] += R[i * stride + j] * wi}
		}
		lam := normalize(v)
		done := abs(lam - lam_max) <= RTOL * lam
		lam_max = lam
		if done {break}
	}

	// 1/σ_min²: largest eigenvalue of (RᵀR)⁻¹ = R⁻¹R⁻ᵀ
	start(v)
	mu_max := 0.0
	for _ in 0 ..< MAX_ITS {
		for i in 0 ..< n { 	// Rᵀ·w = v (forward substitution)
			s := v[i]
			for k in 0 ..< i {s -= R[k * stride + i] * w[k]}
			w[i] = s / R[i * stride + i]
		}
		for i := n - 1; i >= 0; i -= 1 { 	// R·v = w (back substitution)
			s := w[i]
			for j in i + 1 ..< n {s -= R[i * stride + j] * v[j]}
			v[i] = s / R[i * stride + i]
		}
		mu := normalize(v)
		if math.is_inf(mu) || math.is_nan(mu) {return math.inf_f64(1)}
		done := abs(mu - mu_max) <= RTOL * mu
		mu_max = mu
		if done {break}
	}
	return math.sqrt(lam_max * mu_max)
}

// ============================================================================
// Free LstsqResult
// ============================================================================
lstsq_result_free :: proc(res: ^LstsqResult, allocator: mem.Allocator) {
	if len(res.beta) > 0 {
		mem.free(transmute(rawptr)&res.beta[0], allocator)
	}
	if len(res.residuals) > 0 {
		mem.free(transmute(rawptr)&res.residuals[0], allocator)
	}
	res.beta = nil
	res.residuals = nil
}
