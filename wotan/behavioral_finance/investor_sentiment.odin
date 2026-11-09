package behavioral_finance

import an "../analytics"
import ml "../analytics/ML"
import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Investor sentiment index: Baker & Wurgler (2006, 2007)
// ============================================================================
// Baker & Wurgler, "Investor Sentiment and the Cross-Section of Stock Returns",
// JF 61 (2006); "Investor Sentiment in the Stock Market", JEP 21 (2007).
// Sentiment is not observed, so the index is the common component of several
// imperfect proxies: the first principal component of the standardized proxies
// (their correlation matrix), scaled to mean 0 and unit variance,
//   SENT_t = sum_j c_j z_jt                                    (2006, eq. 2)
// and, to remove business-cycle variation, the same index built from the
// residuals of each proxy on macro variables (SENT-perp, eq. 3). A proxy can lead
// or lag the others; the 2006 paper's first stage takes the first component of
// all proxies and their 12-month lags and keeps whichever version of each proxy is
// more correlated with it.
// Cross-sectional prediction: when sentiment is high, stocks that are hard to value
// and hard to arbitrage (small, young, volatile, unprofitable, non-dividend-paying,
// distressed or extreme-growth) earn low subsequent returns relative to others, and
// high returns after low sentiment.
//
// The maintained monthly index (Wurgler's SENTIMENT.xlsx, Stata code on the
// BWInvestorSentimentIndex GitHub) has five proxies, NYSE turnover having been
// dropped: the closed-end fund discount (cefd), the number of IPOs summed over the
// last 12 months (nipo), their IPO-weighted average first-day return over the last
// 12 months lagged 12 months (ripo), the dividend premium lagged 12 months (pdnd),
// and the equity share in new issues (s). baker_wurgler_sentiment reproduces it
// from the published monthly raw series.

// x[t - k] (NaN for t < k)
lag_series :: proc(x: []f64, k: int, allocator := context.allocator) -> []f64 {
	out := make([]f64, len(x), allocator)
	for t in 0 ..< len(x) {out[t] = x[t - k] if t >= k && t - k < len(x) else math.nan_f64()}
	return out
}

// Sum of x over the `window` observations ending at t; NaN unless all are finite
trailing_sum :: proc(x: []f64, window: int, allocator := context.allocator) -> []f64 {
	out := make([]f64, len(x), allocator)
	for t in 0 ..< len(x) {
		out[t] = math.nan_f64()
		if t + 1 < window {continue}
		s := 0.0
		for v in x[t + 1 - window:t + 1] {s += v}
		if _finite(s) {out[t] = s}
	}
	return out
}

// Weighted mean of x over the `window` observations ending at t with weights w
// (e.g. average IPO first-day returns weighted by the number of IPOs). NaN unless
// all window weights are finite and their sum is positive; an x with zero weight
// is ignored (it may be NaN).
trailing_weighted_mean :: proc(x, w: []f64, window: int, allocator := context.allocator) -> []f64 {
	assert(len(x) == len(w), "trailing_weighted_mean: x and w differ in length")
	out := make([]f64, len(x), allocator)
	outer: for t in 0 ..< len(x) {
		out[t] = math.nan_f64()
		if t + 1 < window {continue}
		sw, sx := 0.0, 0.0
		for i in t + 1 - window ..= t {
			if !_finite(w[i]) {continue outer}
			if w[i] == 0 {continue}
			sw += w[i]
			sx += w[i] * x[i]
		}
		if sw > 0 {out[t] = sx / sw}
	}
	return out
}

// Percentage deviation of a positive series from its trailing trend,
// exp(ln x_t - m_t) - 1 with m_t the mean of the finite ln x over the previous
// `window` observations (Stata tssmooth ma, window(#): lags only). NaN where x_t is
// missing or nothing precedes it. Used by the maintained index (2026 revision) to
// detrend the macro controls.
log_ma_deviation :: proc(x: []f64, window: int, allocator := context.allocator) -> []f64 {
	out := make([]f64, len(x), allocator)
	for t in 0 ..< len(x) {
		out[t] = math.nan_f64()
		if !(x[t] > 0) || !_finite(x[t]) {continue}
		s, n := 0.0, 0
		for i in max(0, t - window) ..< t {
			if x[i] > 0 && _finite(x[i]) {s += math.ln(x[i]); n += 1}
		}
		if n > 0 {out[t] = math.exp(math.ln(x[t]) - s / f64(n)) - 1}
	}
	return out
}

// Residuals of y on a constant and the controls (controls[j] = series j), fitted
// over the rows with use[t] (all rows if nil) where y and every control are
// finite; NaN on the other rows
orthogonalize :: proc(
	y: []f64,
	controls: [][]f64,
	use: []bool = nil,
	allocator := context.allocator,
) -> []f64 {
	n, k := len(y), len(controls) + 1
	out := make([]f64, n, allocator)
	rows := make([dynamic]int, context.temp_allocator)
	defer delete(rows)
	for t in 0 ..< n {
		out[t] = math.nan_f64()
		ok := (use == nil || use[t]) && _finite(y[t])
		for c in controls {ok &&= _finite(c[t])}
		if ok {append(&rows, t)}
	}
	if len(rows) <= k {return out}
	X := l.matrix_new(f64, len(rows), k, context.temp_allocator)
	defer l.matrix_free(&X)
	yy := make([]f64, len(rows), context.temp_allocator)
	defer delete(yy, context.temp_allocator)
	for t, i in rows {
		X.data[k * i] = 1
		for c, j in controls {X.data[k * i + j + 1] = c[t]}
		yy[i] = y[t]
	}
	fit := ml.ols_fit(&X, yy, .QR, context.temp_allocator)
	defer ml._ols_result_free(&fit, context.temp_allocator)
	for t, i in rows {out[t] = fit.residuals[i]}
	return out
}

Sentiment_Index :: struct {
	index:        []f64, // mean 0, sd 1 over the rows used; NaN elsewhere
	coefficients: []f64, // index = sum_j c_j z_j on the standardized proxies (2006, eq. 2)
	loadings:     []f64, // first eigenvector of the proxies' correlation matrix (unit length)
	explained:    f64, // share of the proxies' total variance it explains
	n_obs:        int,
}

sentiment_index_free :: proc(s: ^Sentiment_Index, allocator := context.allocator) {
	delete(s.index, allocator)
	delete(s.coefficients, allocator)
	delete(s.loadings, allocator)
	s^ = {}
}

// First principal component of the standardized proxies (proxies[j] = series j)
// over the rows with use[t] (all if nil) where every proxy is finite, scaled to
// mean 0 and unit variance (sd with n - 1). Sign: the index is positive at
// sign_row when sign_row >= 0 (the maintained index: December 2000), otherwise
// the largest loading is positive.
sentiment_index :: proc(
	proxies: [][]f64,
	use: []bool = nil,
	sign_row := -1,
	allocator := context.allocator,
) -> Sentiment_Index {
	k := len(proxies)
	assert(k >= 2, "sentiment_index: needs at least two proxies")
	n := len(proxies[0])
	res := Sentiment_Index {
		index        = make([]f64, n, allocator),
		coefficients = make([]f64, k, allocator),
		loadings     = make([]f64, k, allocator),
		explained    = math.nan_f64(),
	}
	for &v in res.index {v = math.nan_f64()}
	rows := make([dynamic]int, context.temp_allocator)
	defer delete(rows)
	for t in 0 ..< n {
		ok := use == nil || use[t]
		for p in proxies {ok &&= _finite(p[t])}
		if ok {append(&rows, t)}
	}
	m := len(rows)
	res.n_obs = m
	if m <= k {
		for j in 0 ..< k {res.coefficients[j], res.loadings[j] = math.nan_f64(), math.nan_f64()}
		return res
	}
	// standardized proxies, row-major [m, k]
	Z := make([]f64, m * k, context.temp_allocator)
	defer delete(Z, context.temp_allocator)
	for p, j in proxies {
		mu, ss := 0.0, 0.0
		for t in rows {mu += p[t] / f64(m)}
		for t in rows {ss += (p[t] - mu) * (p[t] - mu)}
		sd := math.sqrt(ss / f64(m - 1))
		for t, i in rows {Z[i * k + j] = (p[t] - mu) / sd}
	}
	C := make([][]f64, k, context.temp_allocator)
	defer {for r in C {delete(r, context.temp_allocator)}; delete(C, context.temp_allocator)}
	for a in 0 ..< k {
		C[a] = make([]f64, k, context.temp_allocator)
		for b in 0 ..< k {
			s := 0.0
			for i in 0 ..< m {s += Z[i * k + a] * Z[i * k + b]}
			C[a][b] = s / f64(m - 1)
		}
	}
	pca := an.pca_from_cov(C, context.temp_allocator)
	defer an.destroy_pca_result(pca, context.temp_allocator)
	lambda := pca.eigenvalues[0]
	res.explained = lambda / f64(k)
	// var(Z v) = v'Cv = lambda, so c = v / sqrt(lambda) gives unit variance
	for j in 0 ..< k {
		res.loadings[j] = pca.eigenvectors[0][j]
		res.coefficients[j] = res.loadings[j] / math.sqrt(lambda)
	}
	for t, i in rows {
		s := 0.0
		for j in 0 ..< k {s += res.coefficients[j] * Z[i * k + j]}
		res.index[t] = s
	}
	if sign_row >= 0 && sign_row < n && res.index[sign_row] < 0 {
		for &v in res.index {v = -v}
		for j in 0 ..< k {res.coefficients[j], res.loadings[j] = -res.coefficients[j], -res.loadings[j]}
	}
	return res
}

// Baker & Wurgler (2006) first stage: the first component of all proxies in both
// their current and lagged versions; lagged[j] is true where proxy j's lagged
// version is the more correlated with it (in absolute value). Rows as in
// sentiment_index. A heuristic: it finds leading proxies reliably when they share
// only sentiment, but a second common factor in the proxies (e.g. the business
// cycle) can make it pick the wrong version of a weak proxy.
sentiment_select_lags :: proc(
	current, lagged: [][]f64,
	use: []bool = nil,
	allocator := context.allocator,
) -> []bool {
	k := len(current)
	assert(len(lagged) == k, "sentiment_select_lags: current and lagged differ in count")
	all := make([][]f64, 2 * k, context.temp_allocator)
	defer delete(all, context.temp_allocator)
	for j in 0 ..< k {all[j], all[k + j] = current[j], lagged[j]}
	first := sentiment_index(all, use, -1, context.temp_allocator)
	defer sentiment_index_free(&first, context.temp_allocator)
	out := make([]bool, k, allocator)
	for j in 0 ..< k {
		out[j] =
			abs(_sentiment_corr(first.index, lagged[j])) >
			abs(_sentiment_corr(first.index, current[j]))
	}
	return out
}

// Correlation over the rows where both are finite
@(private = "file")
_sentiment_corr :: proc(a, b: []f64) -> f64 {
	n, ma, mb := 0.0, 0.0, 0.0
	for t in 0 ..< len(a) {
		if _finite(a[t]) && _finite(b[t]) {n += 1; ma += a[t]; mb += b[t]}
	}
	ma /= n
	mb /= n
	sab, saa, sbb := 0.0, 0.0, 0.0
	for t in 0 ..< len(a) {
		if !_finite(a[t]) || !_finite(b[t]) {continue}
		sab += (a[t] - ma) * (b[t] - mb)
		saa += (a[t] - ma) * (a[t] - ma)
		sbb += (b[t] - mb) * (b[t] - mb)
	}
	return sab / math.sqrt(saa * sbb)
}

// Monthly series as published in Wurgler's SENTIMENT.xlsx (NaN = missing), all
// of the same length and aligned in time
BW_Sentiment_Data :: struct {
	cefd, nipo, ripo, pdnd, s:                               []f64, // proxies, monthly raw
	indpro, consdur, consnon, consserv, employ, cpi, recess: []f64, // macro controls
}

// The maintained Baker-Wurgler indexes over rows first ..= last: SENT from the
// five proxies and SENT_ORTH from their residuals on the macro controls (the
// detrended industrial production, real durable / nondurable / services
// consumption and employment, and the NBER recession dummy; contemporaneous for
// the current proxies, lagged 12 months, and the recession dummy 1 month, for the
// lagged ones). Proxy order in the coefficients: cefd, nipo, ripo, pdnd, s.
// sign_row: the row made positive (December 2000 in the maintained index).
baker_wurgler_sentiment :: proc(
	data: BW_Sentiment_Data,
	first, last: int,
	sign_row: int,
	allocator := context.allocator,
) -> (
	sent, sent_orth: Sentiment_Index,
) {
	n := len(data.cefd)
	ta := context.temp_allocator
	use := make([]bool, n, ta)
	defer delete(use, ta)
	for t in max(first, 0) ..= min(last, n - 1) {use[t] = true}

	// IPO volume and first-day returns smoothed over 12 months (months without
	// IPOs carry no first-day return)
	ripo := make([]f64, n, ta)
	defer delete(ripo, ta)
	for t in 0 ..< n {ripo[t] = 0 if data.nipo[t] == 0 || !_finite(data.ripo[t]) else data.ripo[t]}
	nipo12 := trailing_sum(data.nipo, 12, ta)
	ripo12 := trailing_weighted_mean(ripo, data.nipo, 12, ta)
	ripo_l := lag_series(ripo12, 12, ta)
	pdnd_l := lag_series(data.pdnd, 12, ta)
	defer {delete(nipo12, ta); delete(ripo12, ta); delete(ripo_l, ta); delete(pdnd_l, ta)}
	raw := [5][]f64{data.cefd, nipo12, ripo_l, pdnd_l, data.s}
	sent = sentiment_index(raw[:], use, sign_row, allocator)

	// macro controls: percentage deviations from a 12-month trailing trend
	real :: proc(x, cpi: []f64) -> []f64 {
		out := make([]f64, len(x), context.temp_allocator)
		for t in 0 ..< len(x) {out[t] = x[t] / cpi[t]}
		return out
	}
	levels := [5][]f64 {
		data.indpro,
		real(data.consdur, data.cpi),
		real(data.consnon, data.cpi),
		real(data.consserv, data.cpi),
		data.employ,
	}
	c0, c1: [6][]f64
	for x, j in levels {
		c0[j] = log_ma_deviation(x, 12, ta)
		c1[j] = lag_series(c0[j], 12, ta)
	}
	c0[5] = data.recess
	c1[5] = lag_series(data.recess, 1, ta)
	defer {
		for j in 0 ..< 5 {delete(c0[j], ta); delete(c1[j], ta)}
		for j in 1 ..< 4 {delete(levels[j], ta)}
		delete(c1[5], ta)
	}
	e := [5][]f64 {
		orthogonalize(data.cefd, c0[:], use, ta),
		orthogonalize(nipo12, c0[:], use, ta),
		orthogonalize(ripo_l, c1[:], use, ta),
		orthogonalize(pdnd_l, c1[:], use, ta),
		orthogonalize(data.s, c0[:], use, ta),
	}
	defer for x in e {delete(x, ta)}
	sent_orth = sentiment_index(e[:], use, sign_row, allocator)
	return
}

// Returns conditional on sentiment: returns[t] paired with sentiment[t] (pass the
// sentiment known before the return, e.g. the previous month's index). Means
// after positive and non-positive sentiment and their difference (2006, Table 3),
// and the predictive slope r_t = a + d S_t (Table 5); t-statistics from OLS with
// Newey-West se when nw_lags > 0. Rows where either is not finite are skipped.
Sentiment_Conditional :: struct {
	mean_high, mean_low:      f64,
	difference, t_difference: f64,
	slope, t_slope:           f64,
	n_high, n_low:            int,
}

sentiment_conditional_returns :: proc(
	returns, sentiment: []f64,
	nw_lags := 0,
) -> Sentiment_Conditional {
	assert(len(returns) == len(sentiment), "sentiment_conditional_returns: lengths differ")
	res := Sentiment_Conditional{}
	nan := math.nan_f64()
	res.mean_high, res.mean_low, res.difference, res.t_difference, res.slope, res.t_slope =
		nan, nan, nan, nan, nan, nan
	ta := context.temp_allocator
	r := make([dynamic]f64, ta)
	s := make([dynamic]f64, ta)
	defer {delete(r); delete(s)}
	for t in 0 ..< len(returns) {
		if _finite(returns[t]) &&
		   _finite(sentiment[t]) {append(&r, returns[t]); append(&s, sentiment[t])}
	}
	sh, sl := 0.0, 0.0
	for v, i in r {
		if s[i] > 0 {sh += v; res.n_high += 1} else {sl += v; res.n_low += 1}
	}
	if res.n_high > 0 {res.mean_high = sh / f64(res.n_high)}
	if res.n_low > 0 {res.mean_low = sl / f64(res.n_low)}
	if res.n_high < 2 || res.n_low < 2 {return res}
	fit_slope :: proc(x, y: []f64, nw_lags: int) -> (b, t: f64) {
		n := len(x)
		X := l.matrix_new(f64, n, 2, context.temp_allocator)
		defer l.matrix_free(&X)
		for i in 0 ..< n {X.data[2 * i], X.data[2 * i + 1] = 1, x[i]}
		fit :=
			ml.ols_fit_robust(&X, y, {type = .HAC, maxlags = nw_lags}, .QR, context.temp_allocator) if nw_lags > 0 else ml.ols_fit(&X, y, .QR, context.temp_allocator)
		defer ml._ols_result_free(&fit, context.temp_allocator)
		return fit.beta[1], fit.tvalues[1]
	}
	d := make([]f64, len(s), ta)
	defer delete(d, ta)
	for v, i in s {d[i] = 1 if v > 0 else 0}
	res.difference, res.t_difference = fit_slope(d, r[:], nw_lags)
	res.slope, res.t_slope = fit_slope(s[:], r[:], nw_lags)
	return res
}
