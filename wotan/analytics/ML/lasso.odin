package ML

import l "../../linalg"
import "core:math"
import "core:mem"


LassoParams :: struct {
	lambda:   f64,
	max_iter: int,
	tol:      f64,
}
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
// Solves: min ½||y - Xβ||² + λ Σ_j m_j |β_j|, m_j = 0 for a constant (intercept)
// column of X and 1 otherwise. Equivalent to sklearn Lasso(alpha=λ/n,
// fit_intercept=True) when X = [1, features].
// df = number of non-zero coefficients (Zou, Hastie & Tibshirani 2007); there
// is no valid closed-form covariance, so stderr / t / p / CI are NaN.
// ============================================================================
lasso_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	lambda: f64, // regularization strength (λ ≥ 0)
	max_iter: int = 1000,
	tol: f64 = 1e-4, // convergence tolerance on max |Δβ_j|
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	n, p := X.rows, X.cols

	// Validate
	if len(y) != n do panic("lasso_fit: y length mismatch")
	if lambda < 0.0 do panic("lasso_fit: lambda must be non-negative")

	mask := _penalty_mask(X, context.temp_allocator)
	defer delete(mask, context.temp_allocator)

	// =====================================================================
	// 1. Precompute ||X[:,j]||²
	// =====================================================================
	XtX_diag := make([]f64, p, context.temp_allocator)
	defer delete(XtX_diag, context.temp_allocator)
	for j in 0 ..< p {
		col_norm := 0.0
		for i in 0 ..< n {
			val := X.data[i * X.cols + j]
			col_norm += val * val
		}
		XtX_diag[j] = col_norm
	}

	// =====================================================================
	// 2. Initialize β = 0, r = y
	// =====================================================================
	beta := make([]f64, p, allocator)
	r := make([]f64, n, context.temp_allocator)
	defer delete(r, context.temp_allocator)
	copy(r, y)

	// =====================================================================
	// 3. Coordinate descent: β_j = S(x_jᵀ r_j, λ m_j) / ||x_j||²
	// =====================================================================
	for _ in 0 ..< max_iter {
		max_change := 0.0
		for j in 0 ..< p {
			if XtX_diag[j] == 0.0 {continue} 	// all-zero column: β_j stays 0

			// rho = x_jᵀ (r + x_j β_j)
			rho := 0.0
			for i in 0 ..< n {
				rho += X.data[i * X.cols + j] * r[i]
			}
			rho += XtX_diag[j] * beta[j]

			new_b := soft_threshold(rho, lambda * mask[j]) / XtX_diag[j]
			delta := new_b - beta[j]
			if delta != 0.0 {
				for i in 0 ..< n {
					r[i] -= X.data[i * X.cols + j] * delta
				}
				beta[j] = new_b
			}
			if math.abs(delta) > max_change {max_change = math.abs(delta)}
		}
		if max_change < tol {break}
	}

	df_fit := 0.0
	for j in 0 ..< p {if beta[j] != 0.0 {df_fit += 1.0}}

	return _penalized_summary(X, y, beta, df_fit, nil, allocator)
}
// ============================================================================
// Lasso Regression with K-Fold Cross-Validation
// Returns best lambda and corresponding OLSResult
// ============================================================================
lasso_cv :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	lambdas: []f64, // candidate lambda values to search
	k: int = 5, // number of folds
	seed: u64 = 42, // random seed for fold assignment
	max_iter: int = 1000,
	tol: f64 = 1e-4,
	allocator: mem.Allocator = context.allocator,
) -> (
	best_lambda: f64,
	best_result: OLSResult,
	cv_errors: []f64,
) {
	n := X.rows
	if len(lambdas) == 0 do panic("lasso_cv: lambdas is empty")

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

			// Train Lasso on training set
			result := lasso_fit(&X_train, y_train, lambda, max_iter, tol, context.temp_allocator)
			defer _ols_result_free(&result, context.temp_allocator)

			// Evaluate on validation set: MSE
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
	best_result = lasso_fit(X, y, best_lambda, max_iter, tol, allocator)

	return best_lambda, best_result, cv_errors
}
