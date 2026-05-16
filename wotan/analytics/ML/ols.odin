package ML

import w "../../core"
import l "../../linalg"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:simd"


OLSResult :: struct {
	beta:      []f64,
	vcov:      l.Matrix(f64),
	stderr:    []f64,
	tvalues:   []f64,
	sigma2:    f64,
	residuals: []f64,
	fitted:    []f64,
	r2:        f64,
	r2_adj:    f64,
	fstat:     f64,
	ci_low:    []f64,
	ci_high:   []f64,
}
OLSMethod :: enum {
	Cholesky,
	QR,
}

// X: n x p (rows = observations, cols = regressors)
ols_xtx :: proc(X: ^l.Matrix(f64), allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	return l.xtx(X, allocator)
}

ols_xty :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	return l.xty(X, y, allocator)
}

ols_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	return ols_fit_full(X, y, method, allocator)
}


ols_from_df :: proc(
	df: ^w.DataFrame,
	y_name: string,
	x_names: []string,
	add_intercept: bool = true,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	// Build X
	X := l.matrix_from_df(df, x_names, allocator)
	defer l.matrix_free(&X)

	if add_intercept {
		// extend X with leading column of ones
		n := X.rows
		p := X.cols
		X2 := l.matrix_new(f64, n, p + 1, allocator)
		for i := 0; i < n; i += 1 {
			X2.data[i * X2.cols + 0] = 1.0
			for j := 0; j < p; j += 1 {
				X2.data[i * X2.cols + (j + 1)] = X.data[i * X.cols + j]
			}
		}
		l.matrix_free(&X)
		X = X2
	}

	y := l.vector_from_df(df, y_name, allocator)

	return ols_fit(&X, y, method, allocator)
}; ols_fit_full :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	if X.rows != len(y) do panic("ols_fit: dimension mismatch")

	n := X.rows
	p := X.cols

	// 1. XtX and Xty (still needed for vcov etc.)
	XtX := l.xtx(X, allocator)
	Xty := l.xty(X, y, allocator)

	// 2. β: choose solver
	beta: []f64
	switch method {
	case .Cholesky:
		beta = l.solve_spd_cholesky(&XtX, Xty, allocator)

	case .QR:
		// QR decomposition of X
		Q, R := l.qr_decompose(X, allocator)
		// Q is m x m, R is m x p (we use leading p x p of R and first p columns of Q)

		// Compute b = Q₁ᵀ y, where Q₁ = first p columns of Q
		b := make([]f64, p, allocator)
		for j := 0; j < p; j += 1 {
			sum := 0.0
			for i := 0; i < n; i += 1 {
				sum += Q.data[i * Q.cols + j] * y[i]
			}
			b[j] = sum
		}

		// Solve R₁ β = b (upper triangular, leading p x p block)
		beta = l.upper_tri_solve(&R, b, allocator)
	}

	// 3. fitted
	fitted := l.matvec_dyn_simd(X, beta, allocator)

	// 4. residuals
	residuals := make([]f64, n, allocator)
	for i := 0; i < n; i += 1 {
		residuals[i] = y[i] - fitted[i]
	}

	// 5. σ²
	rss := l.dot_simd(residuals, residuals)
	sigma2 := rss / f64(n - p)

	// 6. XtX⁻¹
	XtX_inv := l.spd_inverse(&XtX, allocator)

	// 7. vcov = σ² * XtX⁻¹
	vcov := l.matrix_new(f64, p, p, allocator)
	for i := 0; i < p * p; i += 1 {
		vcov.data[i] = XtX_inv.data[i] * sigma2
	}

	// 8. stderr
	stderr := make([]f64, p, allocator)
	for j := 0; j < p; j += 1 {
		stderr[j] = math.sqrt(vcov.data[j * vcov.cols + j])
	}

	// 9. t-values
	tvalues := make([]f64, p, allocator)
	for j := 0; j < p; j += 1 {
		tvalues[j] = beta[j] / stderr[j]
	}

	// 10. R² and adjusted R²
	mean_y := 0.0
	for i := 0; i < n; i += 1 do mean_y += y[i]
	mean_y /= f64(n)

	tss := 0.0
	for i := 0; i < n; i += 1 {
		dy := y[i] - mean_y
		tss += dy * dy
	}

	r2 := 1.0 - rss / tss
	r2_adj := 1.0 - (1.0 - r2) * (f64(n - 1) / f64(n - p))

	// 11. F-statistic
	fstat := (r2 / f64(p)) / ((1.0 - r2) / f64(n - p))

	// 12. Confidence intervals (95%)
	tcrit := 1.96
	ci_low := make([]f64, p, allocator)
	ci_high := make([]f64, p, allocator)
	for j := 0; j < p; j += 1 {
		delta := tcrit * stderr[j]
		ci_low[j] = beta[j] - delta
		ci_high[j] = beta[j] + delta
	}

	// 13. Package result
	res: OLSResult
	res.beta = beta
	res.vcov = vcov
	res.stderr = stderr
	res.tvalues = tvalues
	res.sigma2 = sigma2
	res.residuals = residuals
	res.fitted = fitted
	res.r2 = r2
	res.r2_adj = r2_adj
	res.fstat = fstat
	res.ci_low = ci_low
	res.ci_high = ci_high

	return res
}
