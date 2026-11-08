package behavioral_finance

import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Lottery demand: MAX and salience
// ============================================================================
// Investors who like lottery-like payoffs overpay for stocks whose recent returns
// look like lotteries, which then earn lower returns.
//   Bali, Cakici & Whitelaw (2011): MAX, a stock's maximum daily return within the
//     month (MAX(n): the mean of its n highest), predicts the next month's return with a
//     negative sign.
//   Bordalo, Gennaioli & Shleifer (2012, 2013) salience theory, measured by Cosemans &
//     Frehen (2021): the states are the month's daily returns, each with probability
//     pi_s = 1 / S. The salience of the stock's return r_s against the market's
//     (equally weighted mean of all stocks) rbar_s is
//       sigma(r_s, rbar_s) = |r_s - rbar_s| / (|r_s| + |rbar_s| + theta)   (CF eq. 8)
//     the days are ranked by salience, k_s = 1 the most salient, and weighted
//       omega_s = delta^k_s / sum_s' delta^k_s' pi_s'                      (CF eq. 3)
//     and ST = cov(omega, r) = sum_s pi_s omega_s r_s - sum_s pi_s r_s        (CF eq. 9),
//     the salience-weighted minus the equally weighted mean return. theta = 0.1 and
//     delta = 0.7 (BGS 2012's calibration, used by CF). High ST (a salient upside)
//     predicts low returns. CF do not say how tied salience is ranked; tied days here
//     share the mean of their weights, so the result does not depend on day order.
// Both measures need min_obs valid daily returns in the month (CF: 15).

// Salience weights omega_s (mean 1) of returns r against market returns m, both
// finite; out has len(r) entries
salience_weights :: proc(r, m: []f64, out: []f64, theta := 0.1, delta := 0.7) {
	S := len(r)
	assert(
		len(m) == S && len(out) == S,
		"salience_weights: one market return and one weight per return",
	)
	assert(theta > 0 && delta > 0 && delta <= 1, "salience_weights: theta > 0, delta in (0, 1]")
	if S == 0 {return}
	Day :: struct {
		sal: f64,
		s:   int,
	}
	days := make([]Day, S, context.temp_allocator)
	defer delete(days, context.temp_allocator)
	for s in 0 ..< S {
		days[s] = {abs(r[s] - m[s]) / (abs(r[s]) + abs(m[s]) + theta), s}
	}
	slice.sort_by(
		days,
		proc(a, b: Day) -> bool {return a.sal > b.sal || (a.sal == b.sal && a.s < b.s)},
	)
	total := 0.0
	for i := 0; i < S; {
		j := i // tied block i .. j - 1, ranks i + 1 .. j
		w := 0.0
		for j < S && days[j].sal == days[i].sal {
			w += math.pow(delta, f64(j + 1))
			j += 1
		}
		w /= f64(j - i)
		for q in i ..< j {out[days[q].s] = w}
		total += w * f64(j - i)
		i = j
	}
	for &w in out {w *= f64(S) / total} 	// divide by sum_s delta^k_s pi_s
}

// ST of one month: returns r and market returns m over its days, NaN days skipped;
// NaN with fewer than min_obs valid days
salience_value :: proc(r, m: []f64, theta := 0.1, delta := 0.7, min_obs := 15) -> f64 {
	assert(len(r) == len(m), "salience_value: one market return per return")
	rv := make([dynamic]f64, context.temp_allocator)
	mv := make([dynamic]f64, context.temp_allocator)
	defer {delete(rv); delete(mv)}
	for s in 0 ..< len(r) {
		if _finite(r[s]) && _finite(m[s]) {append(&rv, r[s]); append(&mv, m[s])}
	}
	S := len(rv)
	if S < max(min_obs, 1) {return math.nan_f64()}
	w := make([]f64, S, context.temp_allocator)
	defer delete(w, context.temp_allocator)
	salience_weights(rv[:], mv[:], w, theta, delta)
	st := 0.0
	for s in 0 ..< S {st += (w[s] - 1) * rv[s]}
	return st / f64(S)
}

// Monthly measures from daily returns [asset][day]: months[k] .. months[k+1] - 1 are
// the days of month k (len(months) = number of months + 1); result [asset][month]
// (the value for month k predicts the return of month k + 1)

// ST for every asset and month; market (one return per day, nil = the equally
// weighted mean of the valid returns that day)
salience_theory_value :: proc(
	returns: [][]f64,
	months: []int,
	market: []f64 = nil,
	theta := 0.1,
	delta := 0.7,
	min_obs := 15,
	allocator := context.allocator,
) -> [][]f64 {
	_check_months(returns, months)
	mkt := market
	d: Herding_Dispersion
	if market == nil {
		d = cross_sectional_dispersion(returns, allocator = context.temp_allocator)
		mkt = d.market
	}
	defer if market == nil {herding_dispersion_free(&d)}
	assert(len(mkt) == len(returns[0]), "salience_theory_value: one market return per day")
	M := len(months) - 1
	st := make([][]f64, len(returns), allocator)
	for r, i in returns {
		st[i] = make([]f64, M, allocator)
		for k in 0 ..< M {
			a, b := months[k], months[k + 1]
			st[i][k] = salience_value(r[a:b], mkt[a:b], theta, delta, min_obs)
		}
	}
	return st
}

// MAX(n): the mean of the n highest valid daily returns of each asset in each month
max_daily_return :: proc(
	returns: [][]f64,
	months: []int,
	n := 1,
	min_obs := 15,
	allocator := context.allocator,
) -> [][]f64 {
	_check_months(returns, months)
	assert(n >= 1, "max_daily_return: n must be positive")
	M := len(months) - 1
	buf := make([dynamic]f64, context.temp_allocator)
	defer delete(buf)
	out := make([][]f64, len(returns), allocator)
	for r, i in returns {
		out[i] = make([]f64, M, allocator)
		for k in 0 ..< M {
			clear(&buf)
			for v in r[months[k]:months[k + 1]] {if _finite(v) {append(&buf, v)}}
			if len(buf) < max(min_obs, n) {
				out[i][k] = math.nan_f64()
				continue
			}
			slice.sort(buf[:])
			s := 0.0
			for v in buf[len(buf) - n:] {s += v}
			out[i][k] = s / f64(n)
		}
	}
	return out
}

// free a [asset][month] result
lottery_measure_free :: signal_panel_free

// months: increasing day bounds within the returns (shared by the monthly measures)
@(private)
_check_months :: proc(returns: [][]f64, months: []int) {
	assert(len(returns) > 0 && len(months) >= 2, "monthly measures: no assets or no month")
	T := len(returns[0])
	for r in returns {assert(len(r) == T, "monthly measures: ragged returns")}
	assert(
		months[0] >= 0 && months[len(months) - 1] <= T,
		"monthly measures: month bounds outside the days",
	)
	for k in 1 ..< len(months) {assert(months[k] >= months[k - 1], "monthly measures: month bounds must increase")}
}
