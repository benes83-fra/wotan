package core

import "core:math"
import "core:mem"

KPSS_Type :: enum {
	Level, // test for level stationarity (μ-stationary)
	Trend, // test for trend stationarity (τ-stationary)
}

kpss_critical_values :: proc(
	kind: KPSS_Type,
) -> (
	crit_10: f64,
	crit_5: f64,
	crit_2_5: f64,
	crit_1: f64,
) {
	#partial switch kind {
	case .Level:
		return 0.347, 0.463, 0.574, 0.739
	case .Trend:
		return 0.119, 0.146, 0.176, 0.216
	}

	// fallback (should never happen)
	return 0.0, 0.0, 0.0, 0.0
}


kpss_test :: proc(
	y: []f64,
	kind: KPSS_Type = .Level,
	nw_lags: int = -1, // automatic Newey–West lag selection
	allocator: mem.Allocator = context.allocator,
) -> (
	kpss_stat: f64,
	p_value: f64,
	crit_10: f64,
	crit_5: f64,
	crit_2_5: f64,
	crit_1: f64,
) {
	T := len(y)
	if T < 10 {
		return 0, 1, 0, 0, 0, 0
	}

	// 1) Detrend or demean
	X_cols := 1
	if kind == .Trend {
		X_cols = 2
	}

	X := make([]f64, T * X_cols, allocator)
	Y := y

	for t in 0 ..< T {
		row := t * X_cols
		X[row + 0] = 1.0
		if kind == .Trend {
			X[row + 1] = f64(t + 1)
		}
	}

	// Compute OLS residuals
	// X'X
	XtX := make([]f64, X_cols * X_cols, allocator)
	XtY := make([]f64, X_cols, allocator)

	for t in 0 ..< T {
		row := t * X_cols
		for i in 0 ..< X_cols {
			xi := X[row + i]
			XtY[i] += xi * Y[t]
			for j in 0 ..< X_cols {
				XtX[i * X_cols + j] += xi * X[row + j]
			}
		}
	}

	XtX_inv := matrix_inverse(XtX, X_cols, allocator)

	beta := make([]f64, X_cols, allocator)
	for i in 0 ..< X_cols {
		s := 0.0
		for j in 0 ..< X_cols {
			s += XtX_inv[i * X_cols + j] * XtY[j]
		}
		beta[i] = s
	}

	// residuals u_t
	u := make([]f64, T, allocator)
	for t in 0 ..< T {
		row := t * X_cols
		pred := 0.0
		for j in 0 ..< X_cols {
			pred += X[row + j] * beta[j]
		}
		u[t] = Y[t] - pred
	}

	// 2) Partial sum process S_t
	S := make([]f64, T, allocator)
	S[0] = u[0]
	for t in 1 ..< T {
		S[t] = S[t - 1] + u[t]
	}

	// 3) Long-run variance (Newey–West)
	lags := nw_lags
	if lags < 0 {
		lags = int(math.floor(4.0 * math.pow(f64(T) / 100.0, 1.0 / 4.0)))
	}


	eta := kpss_long_run_variance(u, nw_lags)

	// 4) KPSS statistic
	sum_sq := 0.0
	for t in 0 ..< T {
		sum_sq += S[t] * S[t]
	}

	kpss_stat = sum_sq / (f64(T * T) * eta)

	// 5) Critical values
	crit_10, crit_5, crit_2_5, crit_1 = kpss_critical_values(kind)

	// 6) p-value (piecewise interpolation)
	if kpss_stat < crit_10 {
		p_value = 0.10
	} else if kpss_stat < crit_5 {
		p_value = 0.05
	} else if kpss_stat < crit_2_5 {
		p_value = 0.025
	} else if kpss_stat < crit_1 {
		p_value = 0.01
	} else {
		p_value = 0.005
	}

	return
}


df_kpss :: proc(
	y: []f64,
	kind: KPSS_Type = .Level,
	nw_lags: int = -1,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	stat, p, c10, c5, c25, c1 := kpss_test(y, kind, nw_lags, allocator)

	df := dataframe_new()
	add_column(&df, column_from_floats("kpss_stat", []f64{stat}))
	add_column(&df, column_from_floats("p_value", []f64{p}))
	add_column(&df, column_from_floats("crit_10pct", []f64{c10}))
	add_column(&df, column_from_floats("crit_5pct", []f64{c5}))
	add_column(&df, column_from_floats("crit_2.5pct", []f64{c25}))
	add_column(&df, column_from_floats("crit_1pct", []f64{c1}))
	df.rows = 1
	return df
}
kpss_long_run_variance :: proc(u: []f64, lag: int) -> f64 {
	T := len(u)
	if T <= 1 {
		return 0.0
	}

	// s0 = variance
	s0 := 0.0
	for i in 0 ..< T {
		s0 += u[i] * u[i]
	}
	s0 /= f64(T)

	s := s0

	// add weighted autocovariances
	for k in 1 ..= lag {
		w := 1.0 - f64(k) / f64(lag + 1)
		cov := 0.0
		for t in k ..< T {
			cov += u[t] * u[t - k]
		}
		cov /= f64(T)
		s += 2.0 * w * cov
	}

	return s
}
