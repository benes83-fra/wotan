package analytics

import w "../core"
import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"


RegressionType :: enum {
	None, // Δy_t = γ y_{t-1} + ...
	Constant, // Δy_t = α + γ y_{t-1} + ...
	Trend, // Δy_t = β t + γ y_{t-1} + ...   (nonstandard; we map to trend-type criticals)
	ConstantTrend, // Δy_t = α + β t + γ y_{t-1} + ...
}

LagSelection :: enum {
	Fixed, // use max_lags as given
	AIC,
	BIC,
}

// If you don't already have this in your core:
@(require_results)
inf_f64 :: proc "contextless" (sign: int) -> f64 {
	if sign >= 0 {
		return 0h7ff00000_00000000
	} else {
		return 0hfff00000_00000000
	}
}

// ------------------------------------------------------------
// Dynamic Gauss–Jordan inverse for k×k matrix in row-major
// ------------------------------------------------------------
matrix_inverse :: proc(A: []f64, k: int, allocator: mem.Allocator = context.allocator) -> []f64 {
	aug := make([]f64, k * k * 2, allocator)

	// [A | I]
	for i in 0 ..< k {
		for j in 0 ..< k {
			aug[i * (2 * k) + j] = A[i * k + j]
		}
		aug[i * (2 * k) + (k + i)] = 1.0
	}

	for col in 0 ..< k {
		pivot := aug[col * (2 * k) + col]
		if math.abs(pivot) < 1e-12 {
			for r in col + 1 ..< k {
				if math.abs(aug[r * (2 * k) + col]) > 1e-12 {
					for c in 0 ..< 2 * k {
						aug[col * (2 * k) + c], aug[r * (2 * k) + c] =
							aug[r * (2 * k) + c], aug[col * (2 * k) + c]
					}
					pivot = aug[col * (2 * k) + col]
					break
				}
			}
		}

		inv_pivot := 1.0 / pivot
		for c in 0 ..< 2 * k {
			aug[col * (2 * k) + c] *= inv_pivot
		}

		for r in 0 ..< k {
			if r == col {
				continue
			}
			factor := aug[r * (2 * k) + col]
			if factor != 0.0 {
				for c in 0 ..< 2 * k {
					aug[r * (2 * k) + c] -= factor * aug[col * (2 * k) + c]
				}
			}
		}
	}

	inv := make([]f64, k * k, allocator)
	for i in 0 ..< k {
		for j in 0 ..< k {
			inv[i * k + j] = aug[i * (2 * k) + (k + j)]
		}
	}

	return inv
}

// ------------------------------------------------------------
// MacKinnon-style asymptotic critical values (T → ∞) per regression type
// Values are standard ADF tau criticals (approx):
//   None          : no constant, no trend
//   Constant      : constant only
//   ConstantTrend : constant + trend
//   Trend         : mapped to trend-type criticals (same as ConstantTrend here)
// ------------------------------------------------------------
adf_critical_values :: proc(
	reg_type: RegressionType,
) -> (
	crit_1pct: f64,
	crit_5pct: f64,
	crit_10pct: f64,
) {
	// Defaults (constant only)
	crit_1pct = -3.43
	crit_5pct = -2.86
	crit_10pct = -2.57

	switch reg_type {
	case .None:
		// no constant, no trend
		crit_1pct = -2.565
		crit_5pct = -1.941
		crit_10pct = -1.616
	case .Constant:
	// already set above
	case .ConstantTrend, .Trend:
		// constant + trend
		crit_1pct = -3.96
		crit_5pct = -3.41
		crit_10pct = -3.12
	}

	return
}

// ------------------------------------------------------------
// Simple MacKinnon-style p-value interpolation using the
// regression-type-specific critical values.
// Monotone piecewise-linear interpolation in (stat, log p)-space.
// This gives continuous p-values instead of a step function.
// ------------------------------------------------------------
adf_pvalue_interp :: proc(stat: f64, reg_type: RegressionType) -> f64 {
	c1, c5, c10 := adf_critical_values(reg_type)

	// anchor points: (stat, p)
	// left tail, 1%, 5%, 10%, "no rejection" at 0
	xs := [5]f64{c1 - 2.0, c1, c5, c10, 0.0}
	ps := [5]f64{0.001, 0.01, 0.05, 0.10, 0.90}

	// clamp extreme right tail
	if stat >= xs[4] {
		return 0.99
	}
	// clamp extreme left tail
	if stat <= xs[0] {
		return ps[0]
	}

	// piecewise-linear in log p
	log_ps := [5]f64{}
	for i in 0 ..< 5 {
		log_ps[i] = math.ln(ps[i])
	}

	for i in 0 ..< 4 {
		if stat >= xs[i] && stat <= xs[i + 1] {
			w := (stat - xs[i]) / (xs[i + 1] - xs[i])
			log_p := log_ps[i] + w * (log_ps[i + 1] - log_ps[i])
			p := math.exp(log_p)
			// safety clamp
			if p < 0.0 {
				p = 0.0
			}
			if p > 1.0 {
				p = 1.0
			}
			return p
		}
	}

	// fallback (should not hit)
	return 0.5
}

// ------------------------------------------------------------
// Core ADF regression for given lag order p and regression type
// ------------------------------------------------------------
adf_core :: proc(
	y: []f64,
	p: int,
	reg_type: RegressionType,
	allocator: mem.Allocator,
) -> (
	adf_stat: f64,
	aic: f64,
	bic: f64,
	lags_used: int,
	n_obs: int,
	sigma2: f64,
) {
	n := len(y)
	if n < 10 {
		return 0.0, inf_f64(1), inf_f64(1), p, 0, 0.0
	}

	// Δy
	dy := make([]f64, n - 1, allocator)
	for i in 1 ..< n {
		dy[i - 1] = y[i] - y[i - 1]
	}

	// y_{t-1}
	y_lag := make([]f64, n - 1, allocator)
	for i in 1 ..< n {
		y_lag[i - 1] = y[i - 1]
	}

	// ---------- SPECIAL CASE p = 0 ----------
	if p == 0 {
		T := n - 1
		if T <= 5 {
			return 0.0, inf_f64(1), inf_f64(1), 0, T, 0.0
		}

		n_obs = T
		lags_used = 0

		// time trend
		t_vals := make([]f64, T, allocator)
		for t in 0 ..< T {
			t_vals[t] = f64(t + 1)
		}

		// base regressors: constant? trend? y_{t-1}
		base_cols := 0
		if reg_type == .Constant || reg_type == .ConstantTrend {
			base_cols += 1
		}
		if reg_type == .Trend || reg_type == .ConstantTrend {
			base_cols += 1
		}
		base_cols += 1 // y_{t-1}

		k := base_cols
		if T <= k {
			return 0.0, inf_f64(1), inf_f64(1), 0, T, 0.0
		}

		X := make([]f64, T * k, allocator)
		Y := make([]f64, T, allocator)

		for t in 0 ..< T {
			row := t * k
			col := 0

			if reg_type == .Constant || reg_type == .ConstantTrend {
				X[row + col] = 1.0
				col += 1
			}
			if reg_type == .Trend || reg_type == .ConstantTrend {
				X[row + col] = t_vals[t]
				col += 1
			}

			// y_{t-1}
			X[row + col] = y_lag[t]
			col += 1

			// dependent variable
			Y[t] = dy[t]
		}

		// X'X and X'Y
		// In adf_core, replace the X'X/X'Y computation loop:

		// Pre-extract columns of X for efficient dot products
		X_cols := make([][]f64, k, allocator)
		for j in 0 ..< k {
			X_cols[j] = make([]f64, T, allocator)
			for t in 0 ..< T {
				X_cols[j][t] = X[t * k + j]
			}
		}

		// Compute X'Y with SIMD
		XtY := make([]f64, k, allocator)
		for j in 0 ..< k {
			XtY[j] = l.dot_simd(X_cols[j], Y)
		}

		// Compute X'X with SIMD (exploit symmetry)
		XtX := make([]f64, k * k, allocator)
		for i in 0 ..< k {
			for j in 0 ..= i { 	// only lower triangle
				sum := l.dot_simd(X_cols[i], X_cols[j])
				XtX[i * k + j] = sum
				XtX[j * k + i] = sum // symmetric
			}
		}

		// Clean up column buffers
		for j in 0 ..< k {
			delete(X_cols[j], allocator)
		}
		delete(X_cols, allocator)

		XtX_mat := l.matrix_from_flat(XtX, k, k, allocator)
		defer l.matrix_free(&XtX_mat)

		// Solve XtX * beta = XtY using Cholesky (XtX is SPD)
		beta := l.solve_spd_cholesky(&XtX_mat, XtY, .Blocked, allocator)
		XtX_inv_mat := l.spd_inverse(&XtX_mat, .Blocked, allocator)
		defer l.matrix_free(&XtX_inv_mat)
		// residual variance
		rss := 0.0
		for t in 0 ..< T {
			row := t * k
			pred := 0.0
			for j in 0 ..< k {
				pred += X[row + j] * beta[j]
			}
			e := Y[t] - pred
			rss += e * e
		}

		sigma2 = rss / f64(T - k)
		if sigma2 <= 0.0 {
			return 0.0, inf_f64(1), inf_f64(1), 0, T, sigma2
		}

		// standard errors
		se := make([]f64, k, allocator)
		for i in 0 ..< k {
			se[i] = math.sqrt_f64(sigma2 * XtX_inv_mat.data[i * k + i])
		}

		// locate γ (coefficient on y_{t-1})
		gamma_col := 0
		switch reg_type {
		case .None:
			gamma_col = 0
		case .Constant:
			gamma_col = 1
		case .Trend:
			gamma_col = 1
		case .ConstantTrend:
			gamma_col = 2
		}

		gamma_hat := beta[gamma_col]
		gamma_se := se[gamma_col]
		adf_stat = gamma_hat / gamma_se

		// log-likelihood (Gaussian)
		log_sigma2 := math.ln(sigma2)
		loglik := -0.5 * f64(T) * (math.ln_f64(2.0 * math.PI) + 1.0 + log_sigma2)

		aic = -2.0 * loglik + 2.0 * f64(k)
		bic = -2.0 * loglik + f64(k) * math.ln(f64(T))

		return
	}
	// ---------- END SPECIAL CASE p = 0 ----------

	// generic p >= 1 branch
	T := (n - 1) - p
	if T <= 5 {
		return 0.0, inf_f64(1), inf_f64(1), p, T, 0.0
	}
	n_obs = T
	lags_used = p

	t_vals := make([]f64, T, allocator)
	for t in 0 ..< T {
		t_vals[t] = f64(t + 1)
	}

	base_cols := 0
	if reg_type == .Constant || reg_type == .ConstantTrend {
		base_cols += 1
	}
	if reg_type == .Trend || reg_type == .ConstantTrend {
		base_cols += 1
	}
	base_cols += 1 // y_{t-1}

	k := base_cols + p
	if T <= k {
		return 0.0, inf_f64(1), inf_f64(1), p, T, 0.0
	}

	X := make([]f64, T * k, allocator)
	Y := make([]f64, T, allocator)

	for t in 0 ..< T {
		row := t * k
		col := 0

		if reg_type == .Constant || reg_type == .ConstantTrend {
			X[row + col] = 1.0
			col += 1
		}
		if reg_type == .Trend || reg_type == .ConstantTrend {
			X[row + col] = t_vals[t]
			col += 1
		}

		// y_{t-1}
		X[row + col] = y_lag[t + p - 1]
		col += 1

		// lagged differences
		for j in 0 ..< p {
			X[row + col + j] = dy[t + p - 1 - j]
		}

		// dependent variable
		Y[t] = dy[t + p]
	}

	// In the generic p >= 1 branch of adf_core, replace the manual solve with:

	// Pre-extract columns of X for efficient dot products
	X_cols := make([][]f64, k, allocator)
	for j in 0 ..< k {
		X_cols[j] = make([]f64, T, allocator)
		for t in 0 ..< T {
			X_cols[j][t] = X[t * k + j]
		}
	}

	// Compute X'Y with SIMD
	XtY := make([]f64, k, allocator)
	for j in 0 ..< k {
		XtY[j] = l.dot_simd(X_cols[j], Y)
	}

	// Compute X'X with SIMD (exploit symmetry)
	XtX := make([]f64, k * k, allocator)
	for i in 0 ..< k {
		for j in 0 ..= i {
			sum := l.dot_simd(X_cols[i], X_cols[j])
			XtX[i * k + j] = sum
			XtX[j * k + i] = sum
		}
	}

	// Clean up column buffers
	for j in 0 ..< k {
		delete(X_cols[j], allocator)
	}
	delete(X_cols, allocator)

	// Solve using Cholesky (XtX is SPD)
	XtX_mat := l.matrix_from_flat(XtX, k, k, allocator)
	defer l.matrix_free(&XtX_mat)
	// DEBUG: Print dimensions before solve

	beta := l.solve_spd_cholesky(&XtX_mat, XtY, .Blocked, allocator)
	XtX_inv_mat := l.spd_inverse(&XtX_mat, .Blocked, allocator)
	defer l.matrix_free(&XtX_inv_mat)
	rss := 0.0
	for t in 0 ..< T {
		row := t * k
		pred := 0.0
		for j in 0 ..< k {
			pred += X[row + j] * beta[j]
		}
		e := Y[t] - pred
		rss += e * e
	}

	sigma2 = rss / f64(T - k)
	if sigma2 <= 0.0 {
		return 0.0, inf_f64(1), inf_f64(1), p, T, sigma2
	}

	se := make([]f64, k, allocator)
	for i in 0 ..< k {
		se[i] = math.sqrt_f64(sigma2 * XtX_inv_mat.data[i * k + i])
	}

	gamma_col := 0
	switch reg_type {
	case .None:
		gamma_col = 0
	case .Constant:
		gamma_col = 1
	case .Trend:
		gamma_col = 1
	case .ConstantTrend:
		gamma_col = 2
	}

	gamma_hat := beta[gamma_col]
	gamma_se := se[gamma_col]
	adf_stat = gamma_hat / gamma_se

	log_sigma2 := math.ln(sigma2)
	loglik := -0.5 * f64(T) * (math.ln_f64(2.0 * math.PI) + 1.0 + log_sigma2)

	aic = -2.0 * loglik + 2.0 * f64(k)
	bic = -2.0 * loglik + f64(k) * math.ln(f64(T))

	return
}


// ------------------------------------------------------------
// High-level ADF test with lag selection
// ------------------------------------------------------------
adf_test :: proc(
	y: []f64,
	max_lags: int,
	reg_type: RegressionType = .Constant,
	lag_sel: LagSelection = .Fixed,
	allocator: mem.Allocator = context.allocator,
) -> (
	adf_stat: f64,
	p_value: f64,
	lags_used: int,
	n_obs: int,
	crit_1pct: f64,
	crit_5pct: f64,
	crit_10pct: f64,
) {
	n := len(y)
	if n < 10 {
		return 0, 1, 0, n, 0, 0, 0
	}

	best_adf := 0.0
	best_aic := inf_f64(1)
	best_bic := inf_f64(1)
	best_p := 0
	best_T := 0
	best_sigma2 := 0.0

	if lag_sel == .Fixed {
		best_adf, best_aic, best_bic, best_p, best_T, best_sigma2 = adf_core(
			y,
			max_lags,
			reg_type,
			allocator,
		)
	} else {
		for p in 0 ..= max_lags {
			adf_p, aic_p, bic_p, p_used, T, sigma2 := adf_core(y, p, reg_type, allocator)

			// Skip invalid models
			if T <= 5 || sigma2 <= 0.0 {
				continue
			}

			#partial switch lag_sel {
			case .AIC:
				if aic_p < best_aic {
					best_aic = aic_p
					best_bic = bic_p
					best_adf = adf_p
					best_p = p_used
					best_T = T
					best_sigma2 = sigma2
				}
			case .BIC:
				if bic_p < best_bic {
					best_aic = aic_p
					best_bic = bic_p
					best_adf = adf_p
					best_p = p_used
					best_T = T
					best_sigma2 = sigma2
				}
			}
		}

	}

	adf_stat = best_adf
	lags_used = best_p
	n_obs = best_T

	crit_1pct, crit_5pct, crit_10pct = adf_critical_values(reg_type)
	p_value = adf_pvalue_interp(adf_stat, reg_type)

	return
}

// ------------------------------------------------------------
// DataFrame wrapper
// ------------------------------------------------------------
df_adf :: proc(
	y: []f64,
	max_lags: int,
	reg_type: RegressionType = .Constant,
	lag_sel: LagSelection = .Fixed,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	stat, p, lags, n_obs, c1, c5, c10 := adf_test(y, max_lags, reg_type, lag_sel, allocator)

	out := w.dataframe_new()
	w.add_column(&out, w.column_from_floats("adf_stat", []f64{stat}))
	w.add_column(&out, w.column_from_floats("p_value", []f64{p}))
	w.add_column(&out, w.column_from_ints("lags_used", []int{lags}))
	w.add_column(&out, w.column_from_ints("n_obs", []int{n_obs}))
	w.add_column(&out, w.column_from_floats("crit_1pct", []f64{c1}))
	w.add_column(&out, w.column_from_floats("crit_5pct", []f64{c5}))
	w.add_column(&out, w.column_from_floats("crit_10pct", []f64{c10}))
	out.rows = 1
	return out
}
