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
LstsqResult :: struct {
	beta:               []f64,
	residuals:          []f64,
	rank:               int,
	condition_estimate: f64,
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
	m, n := X.rows, X.cols

	Q, R := qr_decompose(X, .Blocked, allocator)
	defer matrix_free(&Q)
	defer matrix_free(&R)

	Qty := make([]f64, n, allocator)
	for j in 0 ..< n {
		sum := 0.0
		for i in 0 ..< m {
			sum += Q.data[i * Q.cols + j] * y[i]
		}
		Qty[j] = sum
	}

	beta := upper_tri_solve(&R, Qty, allocator)

	residuals := make([]f64, m, allocator)
	for i in 0 ..< m {
		pred := 0.0
		for j in 0 ..< n {
			pred += X.data[i * X.cols + j] * beta[j]
		}
		residuals[i] = y[i] - pred
	}

	rank := 0
	eps := f64(1e-12) * math.abs(R.data[0])
	for j in 0 ..< n {
		if math.abs(R.data[j * R.cols + j]) > eps {
			rank += 1
		}
	}

	max_d := 0.0
	min_d := f64(1e308)
	for j in 0 ..< n {
		d := math.abs(R.data[j * R.cols + j])
		if d > max_d {
			max_d = d
		}
		if d > 0.0 && d < min_d {
			min_d = d
		}
	}
	cond := f64(1e308)
	if min_d < f64(1e308) {
		cond = max_d / min_d
	}

	result := LstsqResult{}
	result.beta = beta
	result.residuals = residuals
	result.rank = rank
	result.condition_estimate = cond
	result.method_used = .QR
	result.success = cond < f64(1e12)
	return result
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

	// Cholesky: AtA = L Lᵀ (in-place modification of AtA)
	// First check positive definiteness heuristically
	is_spd := true
	for j in 0 ..< n {
		if AtA.data[j * AtA.cols + j] <= 0.0 {
			is_spd = false
			break
		}
	}
	if !is_spd {
		return _lstsq_qr(X, y, allocator) // fallback
	}

	// Decompose in-place: AtA now contains L (lower triangle)
	cholesky_decompose(&AtA)

	// Forward substitution: L z = Xty
	z := forward_subst_unit_lower_simd(&AtA, Xty, allocator)

	// Back substitution: Lᵀ beta = z
	beta := back_subst_upper_simd(&AtA, z, allocator)

	// Residuals
	residuals := make([]f64, m, allocator)
	for i in 0 ..< m {
		pred := 0.0
		for j in 0 ..< n {
			pred += X.data[i * X.cols + j] * beta[j]
		}
		residuals[i] = y[i] - pred
	}

	// Condition estimate from L diagonal
	max_d := 0.0
	min_d := f64(1e308)
	for j in 0 ..< n {
		d := AtA.data[j * AtA.cols + j] // L[j,j] stored on diagonal
		if d > max_d {
			max_d = d
		}
		if d > 0.0 && d < min_d {
			min_d = d
		}
	}
	cond := f64(1e308)
	if min_d < f64(1e308) {
		ratio := max_d / min_d
		cond = ratio * ratio
	}

	result := LstsqResult{}
	result.beta = beta
	result.residuals = residuals
	result.rank = n
	result.condition_estimate = cond
	result.method_used = .Cholesky
	result.success = cond < f64(1e12)
	return result
}

// ============================================================================
// SVD-based least squares
// ============================================================================
_lstsq_svd :: proc(X: ^Matrix(f64), y: []f64, allocator: mem.Allocator) -> LstsqResult {
	m, n := X.rows, X.cols

	U, S, V := svd(X, .GolubReinsch, allocator)
	defer matrix_free(&U)
	defer matrix_free(&V)
	defer mem.free(transmute(rawptr)&S[0], allocator)

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

	d := make([]f64, n, allocator)
	for j in 0 ..< n {
		if S[j] > tol {
			d[j] = c[j] / S[j]
		} else {
			d[j] = 0.0
		}
	}

	beta := matvec_dyn_simd(&V, d, allocator)

	residuals := make([]f64, m, allocator)
	for i in 0 ..< m {
		pred := 0.0
		for j in 0 ..< n {
			pred += X.data[i * X.cols + j] * beta[j]
		}
		residuals[i] = y[i] - pred
	}

	cond := f64(1e308)
	if rank > 0 {
		max_s := S[0]
		min_s := S[rank - 1]
		if min_s > 0.0 {
			cond = max_s / min_s
		}
	}

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
