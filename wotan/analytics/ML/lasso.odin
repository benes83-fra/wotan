package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Soft-thresholding operator for Lasso
// S(z, γ) = sign(z) * max(|z| - γ, 0)
// ============================================================================
soft_threshold :: proc(z, gamma: f64) -> f64 {
	if z > gamma {
		return z - gamma
	} else if z < -gamma {
		return z + gamma
	}
	return 0.0
}

// ============================================================================
// Lasso Regression (L1 Regularization) via Coordinate Descent
// Solves: min ||y - Xβ||² + λ||β||₁
// ============================================================================
lasso_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	lambda: f64, // regularization strength (λ ≥ 0)
	max_iter: int = 1000,
	tol: f64 = 1e-4, // convergence tolerance
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	n, p := X.rows, X.cols

	// Validate
	if len(y) != n do panic("lasso_fit: y length mismatch")
	if lambda < 0.0 do panic("lasso_fit: lambda must be non-negative")

	// =====================================================================
	// 1. Precompute XᵀX diagonal and Xᵀy for efficiency
	// =====================================================================
	XtX_diag := make([]f64, p, allocator) // diagonal of XᵀX
	Xty := make([]f64, p, allocator) // Xᵀy

	for j in 0 ..< p {
		// XᵀX[j,j] = ||X[:,j]||²
		col_norm := 0.0
		for i in 0 ..< n {
			val := X.data[i * X.cols + j]
			col_norm += val * val
		}
		XtX_diag[j] = col_norm

		// Xᵀy[j] = X[:,j]ᵀ y
		sum := 0.0
		for i in 0 ..< n {
			sum += X.data[i * X.cols + j] * y[i]
		}
		Xty[j] = sum
	}
	// defer delete(XtX_diag)
	// defer delete(Xty)

	// =====================================================================
	// 2. Initialize β = 0, residuals = y
	// =====================================================================
	beta := make([]f64, p, allocator)
	residuals := make([]f64, n, allocator)
	for i in 0 ..< n {residuals[i] = y[i]}

	// =====================================================================
	// 3. Coordinate descent loop
	// =====================================================================
	for iter in 0 ..< max_iter {
		beta_old := make([]f64, p, context.temp_allocator)
		copy(beta_old, beta)

		max_change := 0.0

		// Cycle through coefficients
		for j in 0 ..< p {
			if XtX_diag[j] == 0.0 {continue} 	// Skip zero-variance features

			// Compute partial residual: r_j = y - Xβ + X[:,j]*β[j]
			// = residuals + X[:,j]*β[j]
			for i in 0 ..< n {
				residuals[i] += X.data[i * X.cols + j] * beta[j]
			}

			// Compute rho_j = X[:,j]ᵀ * r_j
			rho := 0.0
			for i in 0 ..< n {
				rho += X.data[i * X.cols + j] * residuals[i]
			}

			// Soft-thresholding update
			z := rho / XtX_diag[j]
			gamma := lambda / XtX_diag[j]
			beta[j] = soft_threshold(z, gamma)

			// Update residuals: r = r - X[:,j]*β[j]
			for i in 0 ..< n {
				residuals[i] -= X.data[i * X.cols + j] * beta[j]
			}

			// Track max change for convergence
			change := math.abs(beta[j] - beta_old[j])
			if change > max_change {max_change = change}
		}

		// delete(beta_old)

		// Check convergence
		if max_change < tol {break}
	}

	// =====================================================================
	// 4. Fitted values
	// =====================================================================
	fitted := l.matvec_dyn_simd(X, beta, allocator)

	// =====================================================================
	// 5. σ² = RSS / (n - df)  (approximate; true df requires active set size)
	// =====================================================================
	rss := l.dot_simd(residuals, residuals)
	// Approximate degrees of freedom: count non-zero coefficients
	df := 0.0
	for j in 0 ..< p {if beta[j] != 0.0 {df += 1.0}}
	sigma2 := rss / f64(f64(n) - df)

	// =====================================================================
	// 6. vcov = σ² * (X_activeᵀ X_active)⁻¹  (approximate; Lasso bias not captured)
	// =====================================================================
	vcov := l.matrix_new(f64, p, p, allocator)
	// For simplicity, zero out vcov (Lasso inference is complex)
	for i in 0 ..< p * p {vcov.data[i] = 0.0}

	// =====================================================================
	// 7. Standard errors (approximate; interpret with caution)
	// =====================================================================
	stderr := make([]f64, p, allocator)
	for j in 0 ..< p {
		// Use OLS-style stderr for non-zero coefficients
		if beta[j] != 0.0 && XtX_diag[j] > 0.0 {
			stderr[j] = math.sqrt(sigma2 / XtX_diag[j])
		} else {
			stderr[j] = 0.0
		}
	}

	// =====================================================================
	// 8. t-values (approximate)
	// =====================================================================
	tvalues := make([]f64, p, allocator)
	for j in 0 ..< p {
		if stderr[j] > 0.0 {
			tvalues[j] = beta[j] / stderr[j]
		} else {
			tvalues[j] = 0.0
		}
	}

	// =====================================================================
	// 9. R² and adjusted R²
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
	r2_adj := 1.0 - (1.0 - r2) * (f64(n - 1) / f64(f64(n) - df))

	// =====================================================================
	// 10. F-statistic (approximate)
	// =====================================================================
	fstat := f64(0.0)
	if r2 < 1.0 && f64(n) > df {
		fstat = (r2 / df) / ((1.0 - r2) / f64(f64(n) - df))
	}

	// =====================================================================
	// 11. Confidence intervals (approximate; Lasso is biased)
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
	// 12. Package result
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
