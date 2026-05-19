package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Ridge Regression (L2 Regularization)
// Solves: min ||y - Xβ||² + λ||β||²
// ============================================================================
ridge_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	lambda: f64, // regularization strength (λ ≥ 0)
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	n, p := X.rows, X.cols

	// Validate
	if len(y) != n do panic("ridge_fit: y length mismatch")
	if lambda < 0.0 do panic("ridge_fit: lambda must be non-negative")

	// =====================================================================
	// 1. Compute regularized XtX: XtX_reg = XᵀX + λI
	// =====================================================================
	XtX := l.xtx_simd(X, allocator)
	defer l.matrix_free(&XtX)

	// Add λ to diagonal (Ridge penalty)
	for j in 0 ..< p {
		XtX.data[j * XtX.cols + j] += lambda
	}

	// =====================================================================
	// 2. Compute Xty
	// =====================================================================
	Xty := l.xty(X, y, allocator)
	// defer delete(Xty)

	// =====================================================================
	// 3. Solve (XtX_reg) β = Xty using chosen method
	// =====================================================================
	beta: []f64
	if method == .Cholesky {
		// Note: XtX_reg is SPD if λ > 0 or X has full rank
		beta = l.solve_spd_cholesky(&XtX, Xty, .Blocked, allocator)
	} else {
		// QR path: decompose X directly (more stable for ill-conditioned)
		Q, R := l.qr_decompose(X, .Blocked, allocator)
		defer l.matrix_free(&Q)
		defer l.matrix_free(&R)

		// Compute Qᵀ y
		Qty := make([]f64, p, allocator)
		for j in 0 ..< p {
			sum := 0.0
			for i in 0 ..< n {
				sum += Q.data[i * Q.cols + j] * y[i]
			}
			Qty[j] = sum
		}
		defer delete(Qty)

		// Solve R[0:p, 0:p] β = Qty
		beta = l.upper_tri_solve(&R, Qty, allocator)
	}

	// =====================================================================
	// 4. Fitted values and residuals
	// =====================================================================
	fitted := l.matvec_dyn_simd(X, beta, allocator)
	residuals := make([]f64, n, allocator)
	for i in 0 ..< n {
		residuals[i] = y[i] - fitted[i]
	}

	// =====================================================================
	// 5. σ² = RSS / (n - p)  (note: effective df is less for Ridge)
	// =====================================================================
	rss := l.dot_simd(residuals, residuals)
	sigma2 := rss / f64(n - p) // Approximate; true df requires trace of hat matrix

	// =====================================================================
	// 6. vcov = σ² * (XtX_reg)⁻¹  (approximate; Ridge bias not captured)
	// =====================================================================
	vcov := l.matrix_new(f64, p, p, allocator)

	// Invert via Cholesky (XtX_reg is SPD if λ > 0)
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
		// delete(z)
		// delete(x)
	}
	// delete(e)
	l.matrix_free(&L)

	// =====================================================================
	// 7. Standard errors, t-values (interpret with caution for Ridge)
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

	// =====================================================================
	// 8. R² and adjusted R² (approximate for Ridge)
	// =====================================================================
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

	// =====================================================================
	// 9. F-statistic (approximate)
	// =====================================================================
	fstat := f64(0.0)
	if r2 < 1.0 && n > p {
		fstat = (r2 / f64(p)) / ((1.0 - r2) / f64(n - p))
	}

	// =====================================================================
	// 10. Confidence intervals (approximate; Ridge is biased)
	// =====================================================================
	tcrit := f64(1.96)
	ci_low := make([]f64, p, allocator)
	ci_high := make([]f64, p, allocator)
	for j in 0 ..< p {
		delta := tcrit * stderr[j]
		ci_low[j] = beta[j] - delta
		ci_high[j] = beta[j] + delta
	}

	// =====================================================================
	// 11. Package result
	// =====================================================================
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
