package ML

import l "../../linalg"
import "core:math"
import "core:mem"


RidgeParams :: struct {
	lambda: f64,
	method: OLSMethod,
}

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
// ============================================================================
// Ridge Regression with K-Fold Cross-Validation
// Returns best lambda and corresponding OLSResult
// ============================================================================
ridge_cv :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	lambdas: []f64, // candidate lambda values to search
	k: int = 5, // number of folds
	seed: u64 = 42, // random seed for fold assignment
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> (
	best_lambda: f64,
	best_result: OLSResult,
	cv_errors: []f64,
) {
	n := X.rows

	// Generate fold assignments
	folds := cv_folds(n, k, seed, context.temp_allocator)
	defer delete(folds, context.temp_allocator)

	cv_errors = make([]f64, len(lambdas), allocator)
	best_mse := math.F64_MAX
	best_lambda = lambdas[0]

	// Search over lambda values
	for lambda, li in lambdas {
		fold_mse := 0.0

		// K-fold loop
		for fold in 0 ..< k {
			train_idx, val_idx := cv_split(folds, fold, context.temp_allocator)
			defer {
				delete(train_idx, context.temp_allocator)
				delete(val_idx, context.temp_allocator)
			}

			// Split data
			X_train, y_train := cv_subset(X, y, train_idx, context.temp_allocator)
			X_val, y_val := cv_subset(X, y, val_idx, context.temp_allocator)
			defer {
				cv_subset_free(&X_train, y_train, context.temp_allocator)
				cv_subset_free(&X_val, y_val, context.temp_allocator)
			}

			// Train Ridge on training set
			result := ridge_fit(&X_train, y_train, lambda, method, context.temp_allocator)
			defer _ols_result_free(&result, context.temp_allocator)

			// Evaluate on validation set: MSE = mean((y_val - X_val @ beta)²)
			pred := l.matvec_dyn_simd(&X_val, result.beta, context.temp_allocator)
			defer delete(pred, context.temp_allocator)

			mse := 0.0
			for i in 0 ..< len(val_idx) {
				err := y_val[i] - pred[i]
				mse += err * err
			}
			mse /= f64(len(val_idx))
			fold_mse += mse
		}

		// Average MSE across folds
		avg_mse := fold_mse / f64(k)
		cv_errors[li] = avg_mse

		// Track best lambda
		if avg_mse < best_mse {
			best_mse = avg_mse
			best_lambda = lambda
		}
	}

	// Fit final model on full data with best lambda
	best_result = ridge_fit(X, y, best_lambda, method, allocator)

	return best_lambda, best_result, cv_errors
}
