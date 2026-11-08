package behavioral_finance

import ml "../analytics/ML"
import l "../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Herding: return dispersion in extreme markets
// ============================================================================
// Rational pricing (e.g. the CAPM with different betas) makes individual returns
// spread further from the market return the larger the market move. Investors who
// suppress their own information and follow the market pull returns toward it, so
// herding shows up as dispersion that rises less than linearly, or falls, in large
// market moves.
//   Christie & Huang (1995): CSSD_t = sqrt(sum_i (R_it - R_mt)^2 / (N - 1)) regressed
//     on dummies for the extreme lower and upper tails of the market return,
//     CSSD_t = a + b_L D_L,t + b_U D_U,t + e_t; herding: b_L, b_U < 0 (rational: > 0).
//   Chang, Cheng & Khorana (2000): CSAD_t = sum_i |R_it - R_mt| / N,
//     CSAD_t = a + g1 |R_mt| + g2 R_mt^2 + e_t; rational: g1 > 0, g2 = 0;
//     herding: g2 < 0. Their asymmetric version fits up (R_m > 0) and down (R_m < 0)
//     markets separately. They report Newey-West t-statistics.
// R_mt is the equally weighted mean of the period's returns unless a market return is
// given.

Herding_Dispersion :: struct {
	market:    []f64, // R_mt (NaN with fewer than 2 valid returns)
	cssd:      []f64,
	csad:      []f64,
	n_assets:  []int, // valid (finite) returns in the period
	allocator: mem.Allocator,
}

herding_dispersion_free :: proc(d: ^Herding_Dispersion) {
	delete(d.market, d.allocator)
	delete(d.cssd, d.allocator)
	delete(d.csad, d.allocator)
	delete(d.n_assets, d.allocator)
}

// CSSD and CSAD of returns [asset][time]. NaN returns are left out of the period;
// market (one return per period, nil = equally weighted mean of the valid returns).
cross_sectional_dispersion :: proc(
	returns: [][]f64,
	market: []f64 = nil,
	allocator := context.allocator,
) -> Herding_Dispersion {
	assert(len(returns) > 0, "cross_sectional_dispersion: no assets")
	T := len(returns[0])
	for r in returns {assert(len(r) == T, "cross_sectional_dispersion: ragged returns")}
	assert(
		market == nil || len(market) == T,
		"cross_sectional_dispersion: one market return per period",
	)
	d := Herding_Dispersion {
		market    = make([]f64, T, allocator),
		cssd      = make([]f64, T, allocator),
		csad      = make([]f64, T, allocator),
		n_assets  = make([]int, T, allocator),
		allocator = allocator,
	}
	for t in 0 ..< T {
		n, s := 0, 0.0
		for r in returns {
			if _finite(r[t]) {n += 1; s += r[t]}
		}
		d.n_assets[t] = n
		rm := market[t] if market != nil else s / f64(n)
		if n < 2 || !_finite(rm) {
			d.market[t], d.cssd[t], d.csad[t] = math.nan_f64(), math.nan_f64(), math.nan_f64()
			continue
		}
		ss, sa := 0.0, 0.0
		for r in returns {
			if !_finite(r[t]) {continue}
			ss += (r[t] - rm) * (r[t] - rm)
			sa += abs(r[t] - rm)
		}
		d.market[t], d.cssd[t], d.csad[t] = rm, math.sqrt(ss / f64(n - 1)), sa / f64(n)
	}
	return d
}

// Coefficients of a herding regression: Christie-Huang (a, b_L, b_U), CCK (a, g1, g2)
Herding_Regression :: struct {
	coef:    [3]f64,
	std_err: [3]f64,
	t_stat:  [3]f64,
	r2:      f64,
	n_obs:   int,
}

// OLS of y on (1, x1, x2) over the rows with finite values and keep[t]; classical
// standard errors, or Newey-West with nw_lags > 0 (statsmodels cov_type="HAC", no
// small-sample correction)
@(private = "file")
_herding_ols :: proc(y, x1, x2: []f64, keep: []bool, nw_lags: int) -> Herding_Regression {
	rows := make([dynamic]int, context.temp_allocator)
	defer delete(rows)
	for t in 0 ..< len(y) {
		if keep[t] && _finite(y[t]) && _finite(x1[t]) && _finite(x2[t]) {
			append(&rows, t)
		}
	}
	res := Herding_Regression {
		n_obs = len(rows),
	}
	for j in 0 ..< 3 {res.coef[j], res.std_err[j], res.t_stat[j] = math.nan_f64(), math.nan_f64(), math.nan_f64()}
	res.r2 = math.nan_f64()
	if len(rows) <= 3 {return res}
	X := l.matrix_new(f64, len(rows), 3, context.temp_allocator)
	defer l.matrix_free(&X)
	yy := make([]f64, len(rows), context.temp_allocator)
	defer delete(yy, context.temp_allocator)
	for t, i in rows {
		X.data[3 * i], X.data[3 * i + 1], X.data[3 * i + 2] = 1, x1[t], x2[t]
		yy[i] = y[t]
	}
	// a regressor without variation (e.g. no tail observations): leave NaN
	for j in 1 ..< 3 {
		v0 := X.data[j]
		varies := false
		for i in 1 ..< len(rows) {varies ||= X.data[3 * i + j] != v0}
		if !varies {return res}
	}
	fit :=
		ml.ols_fit_robust(&X, yy, {type = .HAC, maxlags = nw_lags}, .QR, context.temp_allocator) if nw_lags > 0 else ml.ols_fit(&X, yy, .QR, context.temp_allocator)
	defer ml._ols_result_free(&fit, context.temp_allocator)
	for j in 0 ..< 3 {res.coef[j], res.std_err[j], res.t_stat[j] = fit.beta[j], fit.stderr[j], fit.tvalues[j]}
	res.r2 = fit.r2
	return res
}

// Christie & Huang (1995): CSSD on dummies for the `tail` share of periods with the
// lowest and the highest market returns (floor(tail n) periods each among the n with
// a finite CSSD and market return; ties broken by period). Coefficients (a, b_L, b_U).
christie_huang_regression :: proc(
	cssd, market: []f64,
	tail := 0.05,
	nw_lags := 0,
) -> Herding_Regression {
	assert(len(cssd) == len(market), "christie_huang_regression: one market return per CSSD")
	assert(tail > 0 && tail < 0.5, "christie_huang_regression: tail must be in (0, 0.5)")
	T := len(cssd)
	idx := make([dynamic]int, context.temp_allocator)
	defer delete(idx)
	for t in 0 ..< T {
		if _finite(cssd[t]) && _finite(market[t]) {append(&idx, t)}
	}
	_sort_by_market(idx[:], market)
	k := int(math.floor(tail * f64(len(idx))))
	dl := make([]f64, T, context.temp_allocator)
	du := make([]f64, T, context.temp_allocator)
	keep := make([]bool, T, context.temp_allocator)
	defer {delete(dl, context.temp_allocator); delete(du, context.temp_allocator)
		delete(keep, context.temp_allocator)}
	for t in idx {keep[t] = true}
	for i in 0 ..< k {
		dl[idx[i]] = 1
		du[idx[len(idx) - 1 - i]] = 1
	}
	return _herding_ols(cssd, dl, du, keep, nw_lags)
}

// sort period indices by market return (ties by period)
@(private = "file")
_sort_by_market :: proc(idx: []int, market: []f64) {
	Key :: struct {
		m: f64,
		t: int,
	}
	keys := make([]Key, len(idx), context.temp_allocator)
	defer delete(keys, context.temp_allocator)
	for t, i in idx {keys[i] = {market[t], t}}
	slice.sort_by(keys, proc(a, b: Key) -> bool {return a.m < b.m || (a.m == b.m && a.t < b.t)})
	for k, i in keys {idx[i] = k.t}
}

Herding_Market_State :: enum {
	All, // CCK eq. (4): every period
	Up, // R_m > 0 only (CCK's asymmetric up-market regression)
	Down, // R_m < 0 only
}

// Chang, Cheng & Khorana (2000): CSAD on |R_m| and R_m^2. Coefficients (a, g1, g2);
// herding: g2 < 0. CCK use Newey-West standard errors (nw_lags > 0).
cck_regression :: proc(
	csad, market: []f64,
	state := Herding_Market_State.All,
	nw_lags := 0,
) -> Herding_Regression {
	assert(len(csad) == len(market), "cck_regression: one market return per CSAD")
	T := len(csad)
	am := make([]f64, T, context.temp_allocator)
	m2 := make([]f64, T, context.temp_allocator)
	keep := make([]bool, T, context.temp_allocator)
	defer {delete(am, context.temp_allocator); delete(m2, context.temp_allocator)
		delete(keep, context.temp_allocator)}
	for t in 0 ..< T {
		am[t], m2[t] = abs(market[t]), market[t] * market[t]
		switch state {
		case .All:
			keep[t] = true
		case .Up:
			keep[t] = market[t] > 0
		case .Down:
			keep[t] = market[t] < 0
		}
	}
	return _herding_ols(csad, am, m2, keep, nw_lags)
}

@(private = "file")
_finite :: #force_inline proc(x: f64) -> bool {return !math.is_nan(x) && !math.is_inf(x)}
