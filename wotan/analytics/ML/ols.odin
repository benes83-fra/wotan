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
	return l.xtx_simd(X, allocator)
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
}
ols_fit_full :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	if X.rows != len(y) do panic("ols_fit: dimension mismatch")

	n := X.rows
	p := X.cols

	// =====================================================================
	// 1. Solve for beta + residuals
	// =====================================================================
	beta: []f64
	residuals: []f64

	if method == .QR {
		// Use lstsq for QR path (auto-selects best QR variant)
		lstsq_res := l.lstsq(X, y, .QR, allocator)
		// Transfer ownership: OLSResult now owns these slices
		beta = lstsq_res.beta
		residuals = lstsq_res.residuals
	} else {
		// Use original, proven Cholesky path
		XtX := l.xtx_simd(X, allocator)
		Xty := l.xty(X, y, allocator)
		beta = l.solve_spd_cholesky(&XtX, Xty, .Blocked, allocator)
		mem.free(transmute(rawptr)&Xty[0], allocator)

		// Compute residuals manually
		fitted_tmp := l.matvec_dyn_simd(X, beta, allocator)
		residuals = make([]f64, n, allocator)
		for i in 0 ..< n {
			residuals[i] = y[i] - fitted_tmp[i]
		}
		mem.free(transmute(rawptr)&fitted_tmp[0], allocator)
	}

	// =====================================================================
	// 2. Fitted values
	// =====================================================================
	fitted := make([]f64, n, allocator)
	for i in 0 ..< n {
		fitted[i] = y[i] - residuals[i]
	}

	// =====================================================================
	// 3. Compute XtX for vcov (still needed for inference)
	// =====================================================================
	XtX := l.xtx_simd(X, allocator)
	defer l.matrix_free(&XtX)

	// =====================================================================
	// 4. σ² = RSS / (n - p)
	// =====================================================================
	rss := l.dot_simd(residuals, residuals)
	sigma2 := rss / f64(n - p)

	// =====================================================================
	// 5. vcov = σ² * (XtX)⁻¹ via Cholesky inversion
	// =====================================================================
	vcov := l.matrix_new(f64, p, p, allocator)

	is_spd := true
	for j in 0 ..< p {
		if XtX.data[j * XtX.cols + j] <= 0.0 {
			is_spd = false
			break
		}
	}

	if is_spd {
		L := l.matrix_new(f64, p, p, allocator)
		copy(L.data, XtX.data)
		l.cholesky_decompose(&L)

		e := make([]f64, p, context.temp_allocator)
		for k in 0 ..< p {
			for i in 0 ..< p {e[i] = 0.0}
			e[k] = 1.0

			z := l.forward_subst_unit_lower_simd(&L, e, context.temp_allocator)
			x := l.back_subst_upper_simd(&L, z, context.temp_allocator)

			for i in 0 ..< p {
				vcov.data[i * vcov.cols + k] = x[i] * sigma2
			}
			mem.free(transmute(rawptr)&z[0], context.temp_allocator)
			mem.free(transmute(rawptr)&x[0], context.temp_allocator)
		}
		l.matrix_free(&L)
	} else {
		for i in 0 ..< p * p {vcov.data[i] = 0.0}
	}

	// =====================================================================
	// 6-10. Rest unchanged (stderr, tvalues, R², etc.)
	// =====================================================================
	stderr := make([]f64, p, allocator)
	for j in 0 ..< p {
		stderr[j] = math.sqrt(vcov.data[j * vcov.cols + j])
	}

	tvalues := make([]f64, p, allocator)
	for j in 0 ..< p {
		if stderr[j] > 0.0 {
			tvalues[j] = beta[j] / stderr[j]
		} else {
			tvalues[j] = 0.0
		}
	}

	mean_y := 0.0
	for i in 0 ..< n {mean_y += y[i]}
	mean_y /= f64(n)

	tss := 0.0
	for i in 0 ..< n {
		dy := y[i] - mean_y
		tss += dy * dy
	}

	r2 := 1.0 - rss / tss
	r2_adj := 1.0 - (1.0 - r2) * (f64(n - 1) / f64(n - p))

	fstat := f64(0.0)
	if r2 < 1.0 && n > p {
		fstat = (r2 / f64(p)) / ((1.0 - r2) / f64(n - p))
	}

	tcrit := f64(1.96)
	ci_low := make([]f64, p, allocator)
	ci_high := make([]f64, p, allocator)
	for j in 0 ..< p {
		delta := tcrit * stderr[j]
		ci_low[j] = beta[j] - delta
		ci_high[j] = beta[j] + delta
	}

	res := OLSResult{}
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


// Free OLSResult fields (call when done)
_ols_result_free :: proc(res: ^OLSResult, allocator: mem.Allocator) {
	if len(res.beta) > 0 {
		defer delete(res.beta, allocator)
		// mem.free(transmute(rawptr) &res.beta[0], allocator)
	}
	if len(res.residuals) > 0 {
		defer delete(res.residuals, allocator)
		// mem.free(transmute(rawptr) &res.residuals[0], allocator)
	}
	if len(res.stderr) > 0 {
		defer delete(res.stderr, allocator)
		// mem.free(transmute(rawptr) &res.stderr[0], allocator)
	}
	if len(res.tvalues) > 0 {
		defer delete(res.tvalues, allocator)
		// mem.free(transmute(rawptr) &res.tvalues[0], allocator)
	}
	if len(res.fitted) > 0 {
		defer delete(res.fitted, allocator)
		// mem.free(transmute(rawptr) &res.fitted[0], allocator)
	}
	if len(res.ci_low) > 0 {
		defer delete(res.ci_low, allocator)
		// mem.free(transmute(rawptr) &res.ci_low[0], allocator)
	}
	if len(res.ci_high) > 0 {
		defer delete(res.ci_high, allocator)
		// mem.free(transmute(rawptr) &res.ci_high[0], allocator)
	}
	if res.vcov.data != nil {
		l.matrix_free(&res.vcov)
	}
}
