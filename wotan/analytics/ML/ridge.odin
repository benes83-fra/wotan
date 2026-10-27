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
// Solves: min ||y - Xβ||² + λ Σ_j m_j β_j², m_j = 0 for a constant (intercept)
// column of X and 1 otherwise — i.e. sklearn Ridge(alpha=λ, fit_intercept=True)
// when X = [1, features].
// Inference uses the ridge sandwich covariance σ² A⁻¹XᵀX A⁻¹ (A = XᵀX + λM)
// and the effective degrees of freedom tr(H), H = X A⁻¹ Xᵀ.
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

	mask := _penalty_mask(X, context.temp_allocator)
	defer delete(mask, context.temp_allocator)

	// =====================================================================
	// 1. A = XᵀX + λM and its inverse (Cholesky; pseudo-inverse if singular)
	// =====================================================================
	XtX := l.xtx_simd(X, context.temp_allocator)
	defer l.matrix_free(&XtX)
	A := l.matrix_new(f64, p, p, context.temp_allocator)
	defer l.matrix_free(&A)
	copy(A.data, XtX.data)
	for j in 0 ..< p {
		A.data[j * p + j] += lambda * mask[j]
	}
	A_inv := l.matrix_new(f64, p, p, context.temp_allocator)
	defer l.matrix_free(&A_inv)
	_ols_inverse(&A, &A_inv)

	Xty := l.xty(X, y, context.temp_allocator)
	defer delete(Xty, context.temp_allocator)

	// =====================================================================
	// 2. β = A⁻¹ Xᵀy, or for .QR the augmented least-squares problem
	//    [X; √(λM)] β ≈ [y; 0] (better conditioned)
	// =====================================================================
	beta := make([]f64, p, allocator)
	if method == .QR && n > 0 {
		Xa := l.matrix_new(f64, n + p, p, context.temp_allocator)
		defer l.matrix_free(&Xa)
		ya := make([]f64, n + p, context.temp_allocator)
		defer delete(ya, context.temp_allocator)
		copy(Xa.data[:n * p], X.data[:n * p])
		copy(ya[:n], y)
		for j in 0 ..< p {Xa.data[(n + j) * p + j] = math.sqrt(lambda * mask[j])}
		lsq := l.lstsq(&Xa, ya, .QR, context.temp_allocator)
		copy(beta, lsq.beta)
		l.lstsq_result_free(&lsq, context.temp_allocator)
	} else {
		for i in 0 ..< p {
			s := 0.0
			for j in 0 ..< p {s += A_inv.data[i * p + j] * Xty[j]}
			beta[i] = s
		}
	}

	// =====================================================================
	// 3. Effective df = tr(A⁻¹XᵀX); unscaled covariance A⁻¹ XᵀX A⁻¹
	// =====================================================================
	B := l.matrix_new(f64, p, p, context.temp_allocator) // A⁻¹ XᵀX
	defer l.matrix_free(&B)
	for i in 0 ..< p {
		for j in 0 ..< p {
			s := 0.0
			for k in 0 ..< p {s += A_inv.data[i * p + k] * XtX.data[k * p + j]}
			B.data[i * p + j] = s
		}
	}
	df_fit := 0.0
	for j in 0 ..< p {df_fit += B.data[j * p + j]}
	V := l.matrix_new(f64, p, p, context.temp_allocator)
	defer l.matrix_free(&V)
	for i in 0 ..< p {
		for j in 0 ..< p {
			s := 0.0
			for k in 0 ..< p {s += B.data[i * p + k] * A_inv.data[k * p + j]}
			V.data[i * p + j] = s
		}
	}

	return _penalized_summary(X, y, beta, df_fit, &V, allocator)
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
	if len(lambdas) == 0 do panic("ridge_cv: lambdas is empty")

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
