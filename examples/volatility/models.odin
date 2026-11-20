package main

// Stage 2a: the econometric forecasts of the volatility over the next HORIZON
// days (annualised %), refitted at the start of every fold on the days before.
// - GJR-GARCH(1,1) with Student-t errors: the variance reacts to every return,
//   more to losses than to gains (the leverage term γ), and reverts to its
//   long-run level. The forecast averages the expected variance path over the
//   next HORIZON days. The same fit gives the one-day 99% VaR.
// - HAR + VIX (Corsi 2009): a linear regression of the coming month's realised
//   volatility on the realised volatility of the last day, week and month,
//   plus the VIX.

import ts "../../wotan/analytics"
import ml "../../wotan/analytics/ML"
import fin "../../wotan/finance"
import l "../../wotan/linalg"
import "core:math"

Garch_Summary :: struct {
	alpha, beta, gamma, nu, persistence: f64,
}

// garch_fold fits the model to the returns before day s (in %, demeaned) and
// writes the forecasts made at the close of days s ..< e: the volatility over
// the next HORIZON days and the VaR of the next day's return
garch_fold :: proc(m: ^Market, s, e: int, f: ^Forecasts) -> (g: Garch_Summary) {
	n := s - 1 // returns 1 ..< s
	x := make([]f64, n, context.temp_allocator)
	mu := 0.0
	for i in 0 ..< n {x[i] = 100 * m.ret[i + 1]; mu += x[i]}
	mu /= f64(n)
	for &v in x {v -= mu}
	fit := ts.garch_fit_spec(
		x,
		{variance = .GJR, dist = .StudentT, p = 1, q = 1},
		allocator = context.temp_allocator,
	)
	defer ts.garch_result_free(&fit, context.temp_allocator)
	om, al, be, ga := fit.params.omega, fit.params.alpha[0], fit.params.beta[0], fit.params.gamma
	z99 := ts.garch_std_quantile(&fit.params, 0.99) // of the unit-variance t distribution

	// Continue the fit's variance recursion: v is the variance of the next
	// day's return, known at the close of day d
	v := fit.conditional_var[n - 1]
	for d in s - 1 ..< e {
		r := 100 * m.ret[d] - mu
		v = om + (al + (ga if r < 0 else 0)) * r * r + be * v
		if d < s {continue}
		// expected variance path: losses and gains equally likely, so the
		// leverage term adds γ/2 to the persistence
		f.vol[.GARCH][d] = fin.garch_term_structure_vol(om, al + ga / 2, be, v, HORIZON)
		f.var99[d] = z99 * math.sqrt(v) / 100 // a loss, as a fraction
	}
	return {al, be, ga, fit.params.nu, fit.persistence}
}

HAR_FIRST_DAY :: 22 // the monthly component needs 22 returns

har_row :: proc(m: ^Market, d: int) -> [5]f64 {
	return {1, realised_vol(m, d, 1), realised_vol(m, d, 5), realised_vol(m, d, 22), m.vix[d]}
}

// har_fold fits the regression on the days whose target was known before day s
// (purged: the target of day d ends HORIZON days later) and forecasts days
// s ..< e. Returns the coefficients.
har_fold :: proc(m: ^Market, s, e: int, f: ^Forecasts) -> (beta: [5]f64) {
	last := s - HORIZON - 1
	n := last - HAR_FIRST_DAY + 1
	X := l.matrix_new(f64, n, 5, context.temp_allocator)
	y := make([]f64, n, context.temp_allocator)
	for d in HAR_FIRST_DAY ..= last {
		i := d - HAR_FIRST_DAY
		row := har_row(m, d)
		copy(X.data[i * 5:][:5], row[:])
		y[i] = m.rv_fwd[d]
	}
	res := ml.ols_fit(&X, y, allocator = context.temp_allocator)
	defer ml.ols_result_free(&res, context.temp_allocator)
	copy(beta[:], res.beta)
	for d in s ..< e {
		row := har_row(m, d)
		p := 0.0
		for k in 0 ..< 5 {p += beta[k] * row[k]}
		f.vol[.HAR][d] = max(p, 1) // a linear forecast can, in principle, go below 0
	}
	return
}
