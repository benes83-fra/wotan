package analytics

import w "../core"
import l "../linalg"
import ml "./ML"
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

	X := l.matrix_new(f64, T, X_cols, allocator)
	Y := y

	for t in 0 ..< T {
		row := t * X_cols
		X.data[row + 0] = 1.0
		if kind == .Trend {
			X.data[row + 1] = f64(t + 1)
		}
	}

	// Compute OLS residuals
	// X'X
	ols_res := ml.ols_fit(&X, y, .Cholesky, allocator)
	defer ml._ols_result_free(&ols_res, allocator)

	// Extract residuals directly
	u := ols_res.residuals

	// 2) Partial sum process S_t
	S := make([]f64, T, allocator)
	S[0] = u[0]
	for t in 1 ..< T {
		S[t] = S[t - 1] + u[t]
	}

	// 3) Long-run variance (Newey–West, Bartlett kernel)
	// nw_lags < 0: data-dependent bandwidth of Hobijn et al. (1998), the
	// default of statsmodels kpss (nlags="auto"); capped at T-1 like statsmodels.
	lags := nw_lags
	if lags < 0 {
		lags = kpss_autolag(u)
	}
	lags = min(lags, T - 1)

	eta := kpss_long_run_variance(u, lags)
	crit_10, crit_5, crit_2_5, crit_1 = kpss_critical_values(kind)
	if !(eta > 0) {
		// constant series: no deviations from the fitted level/trend
		return 0, 0.10, crit_10, crit_5, crit_2_5, crit_1
	}

	// 4) KPSS statistic
	sum_sq := 0.0
	for t in 0 ..< T {
		sum_sq += S[t] * S[t]
	}

	kpss_stat = sum_sq / (f64(T * T) * eta)

	// 5) p-value: linear interpolation in the KPSS (1992) table, clamped to
	// [0.01, 0.10] outside it (as statsmodels: np.interp(stat, crit, pvals))
	crits := [4]f64{crit_10, crit_5, crit_2_5, crit_1}
	pvals := [4]f64{0.10, 0.05, 0.025, 0.01}
	if kpss_stat <= crits[0] {
		p_value = pvals[0]
	} else if kpss_stat >= crits[3] {
		p_value = pvals[3]
	} else {
		for i in 0 ..< 3 {
			if kpss_stat <= crits[i + 1] {
				w := (kpss_stat - crits[i]) / (crits[i + 1] - crits[i])
				p_value = pvals[i] + w * (pvals[i + 1] - pvals[i])
				break
			}
		}
	}

	return
}


df_kpss :: proc(
	y: []f64,
	kind: KPSS_Type = .Level,
	nw_lags: int = -1,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	stat, p, c10, c5, c25, c1 := kpss_test(y, kind, nw_lags, allocator)

	df := w.dataframe_new()
	w.add_column(&df, w.column_from_floats("kpss_stat", []f64{stat}))
	w.add_column(&df, w.column_from_floats("p_value", []f64{p}))
	w.add_column(&df, w.column_from_floats("crit_10pct", []f64{c10}))
	w.add_column(&df, w.column_from_floats("crit_5pct", []f64{c5}))
	w.add_column(&df, w.column_from_floats("crit_2.5pct", []f64{c25}))
	w.add_column(&df, w.column_from_floats("crit_1pct", []f64{c1}))
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

// Hobijn, Franses & Ooms (1998) automatic Bartlett bandwidth, as in
// statsmodels _kpss_autolag: n = floor(T^(2/9)) autocovariances give
// s_hat = s1/s0 and lags = floor(1.1447 * (s_hat^2 * T)^(1/3)).
// (The fixed alternatives are statsmodels nlags="legacy",
// ceil(12 (T/100)^(1/4)), or the KPSS paper's l4 = floor(4 (T/100)^(1/4));
// pass them via nw_lags >= 0.)
kpss_autolag :: proc(u: []f64) -> int {
	T := len(u)
	if T <= 1 {
		return 0
	}
	cov_lags := int(math.pow(f64(T), 2.0 / 9.0))
	s0 := 0.0
	for v in u {
		s0 += v * v
	}
	s0 /= f64(T)
	s1 := 0.0
	for i in 1 ..= cov_lags {
		prod := 0.0
		for t in i ..< T {
			prod += u[t] * u[t - i]
		}
		prod /= f64(T) / 2.0
		s0 += prod
		s1 += f64(i) * prod
	}
	if s0 == 0 {
		return 0
	}
	s_hat := s1 / s0
	gamma_hat := 1.1447 * math.pow(s_hat * s_hat, 1.0 / 3.0)
	return max(int(gamma_hat * math.pow(f64(T), 1.0 / 3.0)), 0)
}
