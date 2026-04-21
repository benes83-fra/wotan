package core

import "core:math"
import "core:mem"

df_residual_diagnostics :: proc(
	residuals: []f64,
	max_lag: int,
	dof_adj: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	n := len(residuals)
	if n == 0 {
		return dataframe_new()
	}

	// --- Basic statistics ---
	mean := 0.0
	for x in residuals {
		mean += x
	}
	mean /= f64(n)

	var := 0.0
	m3 := 0.0
	m4 := 0.0

	for x in residuals {
		d := x - mean
		d2 := d * d
		var += d2
		m3 += d2 * d
		m4 += d2 * d2
	}

	var /= f64(n)
	m3 /= f64(n)
	m4 /= f64(n)

	skew := 0.0
	kurt := 0.0
	if var > 0 {
		skew = m3 / math.pow(var, 1.5)
		kurt = m4 / (var * var)
	}

	// --- Jarque–Bera ---
	JB, JB_p := jarque_bera(residuals, allocator)

	// --- Ljung–Box ---
	Q, df, LB_p := ljung_box(residuals, max_lag, dof_adj, allocator)

	// --- Build DataFrame ---
	out := dataframe_new()

	add_column(&out, column_from_floats("mean", []f64{mean}))
	add_column(&out, column_from_floats("variance", []f64{var}))
	add_column(&out, column_from_floats("skewness", []f64{skew}))
	add_column(&out, column_from_floats("kurtosis", []f64{kurt}))

	add_column(&out, column_from_floats("JB", []f64{JB}))
	add_column(&out, column_from_floats("JB_p", []f64{JB_p}))

	add_column(&out, column_from_floats("LB_Q", []f64{Q}))
	add_column(&out, column_from_ints("LB_df", []int{df}))
	add_column(&out, column_from_floats("LB_p", []f64{LB_p}))

	out.rows = 1
	return out
}


// ------------------------------------------------------------
// DataFrame ACF of residuals
// ------------------------------------------------------------
df_residual_acf :: proc(
	residuals: []f64,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	ac := acf(residuals, max_lag, allocator)

	out := dataframe_new()
	col := column_new("residual_acf", .Float, max_lag + 1)

	for i in 0 ..= max_lag {
		append_float(&col, ac[i])
	}

	add_column(&out, col)
	out.rows = max_lag + 1
	return out
}


// ------------------------------------------------------------
// DataFrame PACF of residuals
// ------------------------------------------------------------
df_residual_pacf :: proc(
	residuals: []f64,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	pc := pacf(residuals, max_lag, allocator)

	out := dataframe_new()
	col := column_new("residual_pacf", .Float, max_lag + 1)

	for i in 0 ..= max_lag {
		append_float(&col, pc[i])
	}

	add_column(&out, col)
	out.rows = max_lag + 1
	return out
}
df_residuals :: proc(
	y: []f64,
	fit: ArimaFitResult,
	p, d, q: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	// 1) Differencing if needed
	y_eff := y
	if d > 0 {
		y_eff = difference(y, d, allocator)
	}

	// 2) Build state-space from fitted parameters
	F, Q, P0, H, R, x0, N := arima_state_space(fit.phi, fit.theta, d, fit.sigma2, allocator)

	// 3) Run Kalman filter to get residuals + innovation variances
	v, S := kalman_filter_residuals(y_eff, F, Q, P0, H, R, x0, N, allocator)

	// 4) Build DataFrame
	out := dataframe_new()

	add_column(&out, column_from_floats("residual", v))
	add_column(&out, column_from_floats("innovation_var", S))

	out.rows = len(v)
	return out
}
