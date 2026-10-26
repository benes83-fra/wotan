package analytics

import w "../core"
import l "../linalg"
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
// MacKinnon (2010) critical values per regression type, finite-sample
// response surface crit(T) = b0 + b1/T + b2/T^2 + b3/T^3 (statsmodels
// mackinnoncrit, N = 1). nobs <= 0 gives the asymptotic values b0.
//   None          : no constant, no trend      ("n")
//   Constant      : constant only              ("c")
//   ConstantTrend : constant + trend           ("ct")
//   Trend         : no MacKinnon table; mapped to "ct"
// ------------------------------------------------------------
adf_critical_values :: proc(
	reg_type: RegressionType,
	nobs: int = 0,
) -> (
	crit_1pct: f64,
	crit_5pct: f64,
	crit_10pct: f64,
) {
	// rows: 1%, 5%, 10%; columns: b0..b3
	tau: [3][4]f64
	switch reg_type {
	case .None:
		tau = {
			{-2.56574, -2.2358, -3.627, 0.0},
			{-1.94100, -0.2686, -3.365, 31.223},
			{-1.61682, 0.2656, -2.714, 25.364},
		}
	case .Constant:
		tau = {
			{-3.43035, -6.5393, -16.786, -79.433},
			{-2.86154, -2.8903, -4.234, -40.040},
			{-2.56677, -1.5384, -2.809, 0.0},
		}
	case .ConstantTrend, .Trend:
		tau = {
			{-3.95877, -9.0531, -28.428, -134.155},
			{-3.41049, -4.3904, -9.036, -45.374},
			{-3.12705, -2.5856, -3.925, -22.380},
		}
	}

	crit: [3]f64
	for i in 0 ..< 3 {
		crit[i] = tau[i][0]
		if nobs > 0 {
			u := 1.0 / f64(nobs)
			crit[i] += u * (tau[i][1] + u * (tau[i][2] + u * tau[i][3]))
		}
	}
	return crit[0], crit[1], crit[2]
}

// ------------------------------------------------------------
// MacKinnon (1994) approximate asymptotic p-value for the ADF tau
// statistic (statsmodels mackinnonp, N = 1):
//   p = Phi(c0 + c1*tau + c2*tau^2 [+ c3*tau^3])
// with the "small p" polynomial for tau <= tau_star, the "large p" one
// above it, and p = 0 / 1 beyond tau_min / tau_max.
// ------------------------------------------------------------
adf_pvalue_interp :: proc(stat: f64, reg_type: RegressionType) -> f64 {
	tau_max, tau_min, tau_star: f64
	small: [3]f64
	large: [4]f64
	switch reg_type {
	case .None:
		tau_max, tau_min, tau_star = math.inf_f64(1), -19.04, -1.04
		small = {0.6344, 1.2378, 3.2496e-2}
		large = {0.4797, 9.3557e-1, -0.6999e-1, 3.3066e-2}
	case .Constant:
		tau_max, tau_min, tau_star = 2.74, -18.83, -1.61
		small = {2.1659, 1.4412, 3.8269e-2}
		large = {1.7339, 9.3202e-1, -1.2745e-1, -1.0368e-2}
	case .ConstantTrend, .Trend:
		tau_max, tau_min, tau_star = 0.7, -16.18, -2.89
		small = {3.2512, 1.6047, 4.9588e-2}
		large = {2.5261, 6.1654e-1, -3.7956e-1, -6.0285e-2}
	}

	if math.is_nan(stat) {
		return math.nan_f64()
	}
	if stat > tau_max {
		return 1.0
	}
	if stat < tau_min {
		return 0.0
	}

	z := 0.0
	if stat <= tau_star {
		z = small[0] + stat * (small[1] + stat * small[2])
	} else {
		z = large[0] + stat * (large[1] + stat * (large[2] + stat * large[3]))
	}
	// standard normal CDF
	return 0.5 * math.erfc(-z / math.SQRT_TWO)
}

// ------------------------------------------------------------
// Core ADF regression for given lag order p and regression type
// ------------------------------------------------------------
// True if X'X is (numerically) positive definite, i.e. the Cholesky solves
// below will not panic. Works on a temp copy; A is unchanged.
_adf_spd_ok :: proc(A: ^l.Matrix(f64)) -> bool {
	tmp := l.matrix_new(f64, A.rows, A.cols, context.temp_allocator)
	defer l.matrix_free(&tmp)
	copy(tmp.data, A.data)
	return l.cholesky_try(&tmp)
}

// Regression of Δy_t on [const] [trend] y_{t-1} Δy_{t-1} .. Δy_{t-p}.
// The sample starts after max(p, sample_lags) differences, so a lag search
// can fit every p on the same observations (statsmodels autolag); with the
// default sample_lags = -1 it uses all n-1-p usable rows.
// AIC/BIC use the Gaussian log-likelihood at the MLE sigma^2 = RSS/T and k =
// number of regressors (as statsmodels OLS); the returned sigma2 is the
// unbiased RSS/(T-k) used for the t-statistic.
adf_core :: proc(
	y: []f64,
	p: int,
	reg_type: RegressionType,
	allocator: mem.Allocator,
	sample_lags: int = -1,
) -> (
	adf_stat: f64,
	aic: f64,
	bic: f64,
	lags_used: int,
	n_obs: int,
	sigma2: f64,
) {
	n := len(y)
	if n < 10 || p < 0 {
		return 0.0, inf_f64(1), inf_f64(1), max(p, 0), 0, 0.0
	}

	// Δy
	dy := make([]f64, n - 1, allocator)
	defer delete(dy, allocator)
	for i in 1 ..< n {
		dy[i - 1] = y[i] - y[i - 1]
	}

	m := max(p, sample_lags)
	T := (n - 1) - m
	if T <= 5 {
		return 0.0, inf_f64(1), inf_f64(1), p, T, 0.0
	}
	n_obs = T
	lags_used = p

	has_const := reg_type == .Constant || reg_type == .ConstantTrend
	has_trend := reg_type == .Trend || reg_type == .ConstantTrend

	base_cols := 0
	if has_const {
		base_cols += 1
	}
	if has_trend {
		base_cols += 1
	}
	gamma_col := base_cols // y_{t-1} follows the deterministic terms
	base_cols += 1

	k := base_cols + p
	if T <= k {
		return 0.0, inf_f64(1), inf_f64(1), p, T, 0.0
	}

	X := make([]f64, T * k, allocator)
	defer delete(X, allocator)
	Y := make([]f64, T, allocator)
	defer delete(Y, allocator)

	for t in 0 ..< T {
		s := t + m // dependent Δy_t = dy[s] = y[s+1] - y[s]
		row := t * k
		col := 0

		if has_const {
			X[row + col] = 1.0
			col += 1
		}
		if has_trend {
			X[row + col] = f64(t + 1)
			col += 1
		}

		// y_{t-1}
		X[row + col] = y[s]
		col += 1

		// lagged differences Δy_{t-1-j}
		for j in 0 ..< p {
			X[row + col + j] = dy[s - 1 - j]
		}

		// dependent variable
		Y[t] = dy[s]
	}

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
	defer delete(XtY, allocator)
	for j in 0 ..< k {
		XtY[j] = l.dot_simd(X_cols[j], Y)
	}

	// Compute X'X with SIMD (exploit symmetry)
	XtX := make([]f64, k * k, allocator)
	defer delete(XtX, allocator)
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
	// Singular X'X (constant/all-zero/NaN series, exact trend): no valid regression
	if !_adf_spd_ok(&XtX_mat) {
		return 0.0, inf_f64(1), inf_f64(1), p, T, 0.0
	}

	beta := l.solve_spd_cholesky(&XtX_mat, XtY, .Blocked, allocator)
	defer delete(beta, allocator)
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

	gamma_hat := beta[gamma_col]
	gamma_se := math.sqrt_f64(sigma2 * XtX_inv_mat.data[gamma_col * k + gamma_col])
	adf_stat = gamma_hat / gamma_se

	loglik := -0.5 * f64(T) * (math.ln_f64(2.0 * math.PI) + 1.0 + math.ln(rss / f64(T)))

	aic = -2.0 * loglik + 2.0 * f64(k)
	bic = -2.0 * loglik + f64(k) * math.ln(f64(T))

	return
}


// ------------------------------------------------------------
// High-level ADF test with lag selection
// ------------------------------------------------------------
// Matches statsmodels adfuller (regression "n"/"c"/"ct"):
//   .Fixed   : regression with exactly max_lags lagged differences
//   .AIC/.BIC: every p in 0..max_lags is fitted on the common sample of
//              n-1-max_lags rows, the IC minimiser (smallest p on ties) is
//              then refitted on its full n-1-p rows. max_lags is capped at
//              n/2 - ntrend - 1 (where statsmodels raises).
// p_value is MacKinnon (1994), critical values MacKinnon (2010) at n_obs.
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
	max_lags := max(max_lags, 0)

	best_adf := 0.0
	best_p := 0
	best_T := 0
	best_sigma2 := 0.0

	if lag_sel == .Fixed {
		best_adf, _, _, best_p, best_T, best_sigma2 = adf_core(y, max_lags, reg_type, allocator)
	} else {
		ntrend := 0
		switch reg_type {
		case .None:
			ntrend = 0
		case .Constant, .Trend:
			ntrend = 1
		case .ConstantTrend:
			ntrend = 2
		}
		search_max := max(min(max_lags, n / 2 - ntrend - 1), 0)

		best_ic := inf_f64(1)
		found := false
		for p in 0 ..= search_max {
			_, aic_p, bic_p, _, T, sigma2 := adf_core(y, p, reg_type, allocator, search_max)

			// Skip invalid models
			if T <= 5 || sigma2 <= 0.0 {
				continue
			}

			ic := aic_p if lag_sel == .AIC else bic_p
			if ic < best_ic {
				best_ic = ic
				best_p = p
				found = true
			}
		}

		if found {
			// refit the selected lag on its full sample
			best_adf, _, _, best_p, best_T, best_sigma2 = adf_core(y, best_p, reg_type, allocator)
		}
	}

	adf_stat = best_adf
	lags_used = best_p
	n_obs = best_T

	crit_1pct, crit_5pct, crit_10pct = adf_critical_values(reg_type, n_obs)
	if best_sigma2 <= 0.0 {
		// no valid regression (constant / degenerate series)
		p_value = 1.0
	} else {
		p_value = adf_pvalue_interp(adf_stat, reg_type)
	}

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
