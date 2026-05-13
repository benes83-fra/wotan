package ML

import w "../core"
import l "../linalg"
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
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	return ols_fit_full(X, y, allocator)
}


ols_from_df :: proc(
	df: ^w.DataFrame,
	y_name: string,
	x_names: []string,
	add_intercept: bool = true,
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

	return ols_fit(&X, y, allocator)
}
ols_fit_full :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	if X.rows != len(y) do panic("ols_fit: dimension mismatch")

	n := X.rows
	p := X.cols

	// 1. Compute XtX and Xty
	XtX := l.xtx(X, allocator)
	Xty := l.xty(X, y, allocator)

	// 2. β = (XtX)⁻¹ Xty  via Cholesky
	beta := l.solve_spd_cholesky(&XtX, Xty, allocator)

	// 3. Fitted values: y_hat = X β
	fitted := l.matvec_dyn_simd(X, beta, allocator)

	// 4. Residuals: r = y - y_hat
	residuals := make([]f64, n, allocator)
	for i := 0; i < n; i += 1 {
		residuals[i] = y[i] - fitted[i]
	}

	// 5. σ² = (rᵀ r) / (n - p)
	sigma2 := l.dot_simd(residuals, residuals) / f64(n - p)

	// 6. XtX⁻¹
	XtX_inv := l.spd_inverse(&XtX, allocator)

	// 7. vcov = σ² * XtX⁻¹
	vcov := l.matrix_new(f64, p, p, allocator)
	for i := 0; i < p * p; i += 1 {
		vcov.data[i] = XtX_inv.data[i] * sigma2
	}

	// 8. Standard errors = sqrt(diagonal(vcov))
	stderr := make([]f64, p, allocator)
	for j := 0; j < p; j += 1 {
		stderr[j] = math.sqrt(vcov.data[j * vcov.cols + j])
	}

	// 9. t-values = beta / stderr
	tvalues := make([]f64, p, allocator)
	for j := 0; j < p; j += 1 {
		tvalues[j] = beta[j] / stderr[j]
	}

	// 10. Package result
	res: OLSResult
	res.beta = beta
	res.vcov = vcov
	res.stderr = stderr
	res.tvalues = tvalues
	res.sigma2 = sigma2
	res.residuals = residuals
	res.fitted = fitted

	return res
}
