package behavioral_finance

import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Anchoring on the 52-week high (George & Hwang 2004)
// ============================================================================
// PTH_{i,t} = P_{i,t} / high_{i,t}, high the highest price over the `lookback` days
// ending at t (the current price included, so PTH <= 1). The strategy: each month
// rank the stocks on PTH, buy the top 30% and sell the bottom 30% (equally weighted),
// hold for 6 months with overlapping portfolios as Jegadeesh & Titman (1993).
// George & Hwang: 0.45% a month on CRSP 1963 - 2001 (1.23% excluding January).

// PTH for prices [assets][time]: NaN until a full window of valid (finite, > 0)
// prices is available, and while the window holds an invalid one (pandas:
// p / p.rolling(lookback).max()). Monotone queue: O(time) per asset.
compute_anchoring_signal :: proc(
	prices: [][]f64, // [assets][time]
	lookback: int = 252,
	allocator := context.allocator,
) -> [][]f64 {
	assert(lookback > 0, "compute_anchoring_signal: lookback must be positive")
	signal := make([][]f64, len(prices), allocator)
	for p, a in prices {
		s := make([]f64, len(p), allocator)
		signal[a] = s
		// indices of the window's candidate maxima, prices decreasing
		q := make([]int, len(p), context.temp_allocator)
		defer delete(q, context.temp_allocator)
		head, tail := 0, 0
		last_bad := -1
		for v, t in p {
			if !(v > 0) || math.is_inf(v) {
				last_bad = t
				head, tail = 0, 0
				s[t] = math.nan_f64()
				continue
			}
			for tail > head && p[q[tail - 1]] <= v {tail -= 1}
			q[tail] = t
			tail += 1
			for q[head] <= t - lookback {head += 1}
			s[t] = v / p[q[head]] if t - last_bad >= lookback else math.nan_f64()
		}
	}
	return signal
}

compute_anchoring_signal_free :: proc(signal: [][]f64, allocator := context.allocator) {
	for s in signal {delete(s, allocator)}
	delete(signal, allocator)
}

// One formation date: equally weighted long the top `fraction` and short the bottom
// `fraction` of the assets with a finite signal (k = floor(fraction n) on each side;
// ties broken by asset index). Weights sum to +1 and -1; zero when k = 0.
anchoring_portfolio :: proc(
	signal: []f64,
	fraction := 0.3,
	allocator := context.allocator,
) -> []f64 {
	assert(fraction > 0 && fraction <= 0.5, "anchoring_portfolio: fraction in (0, 0.5]")
	weights := make([]f64, len(signal), allocator)
	idx := make([dynamic]int, 0, len(signal), context.temp_allocator)
	defer delete(idx)
	for v, i in signal {if !math.is_nan(v) {append(&idx, i)}}
	k := int(math.floor(fraction * f64(len(idx))))
	if k == 0 {return weights}
	Ctx :: struct {
		s: []f64,
	}
	context.user_ptr = &Ctx{signal}
	slice.sort_by(idx[:], proc(a, b: int) -> bool {
		s := (^Ctx)(context.user_ptr).s
		return s[a] < s[b] || (s[a] == s[b] && a < b)
	})
	n := len(idx)
	for j in 0 ..< k {
		weights[idx[j]] = -1 / f64(k)
		weights[idx[n - 1 - j]] = 1 / f64(k)
	}
	return weights
}

Overlapping_Returns :: struct {
	long:      []f64, // return of the long basket per period (NaN before all cohorts exist)
	short:     []f64, // return of the short basket (what the shorted assets earned)
	spread:    []f64, // long - short: the self-financing strategy
	allocator: mem.Allocator,
}

overlapping_returns_free :: proc(r: ^Overlapping_Returns) {
	delete(r.long, r.allocator)
	delete(r.short, r.allocator)
	delete(r.spread, r.allocator)
}

// Jegadeesh & Titman (1993) overlapping holding. weights[t] is formed at the end of
// period t; returns[s] is the return over period s (end s-1 to end s). Each cohort is
// held for `hold` periods after skipping `skip`: period s holds the cohorts formed at
// s - skip - 1 .. s - skip - hold, each with weight 1 / hold (JT's fixed weights within
// a cohort). Assets with a NaN return in a period contribute zero. Periods before the
// first full set of cohorts are NaN.
overlapping_strategy_returns :: proc(
	weights: [][]f64, // [periods][assets]
	returns: [][]f64, // [periods][assets]
	hold: int = 6,
	skip: int = 0,
	allocator := context.allocator,
) -> Overlapping_Returns {
	assert(
		hold > 0 && skip >= 0 && len(weights) == len(returns),
		"overlapping_strategy_returns: bad inputs",
	)
	T := len(returns)
	out := Overlapping_Returns {
		long      = make([]f64, T, allocator),
		short     = make([]f64, T, allocator),
		spread    = make([]f64, T, allocator),
		allocator = allocator,
	}
	for s in 0 ..< T {
		if s - skip - hold < 0 {
			out.long[s], out.short[s], out.spread[s] =
				math.nan_f64(), math.nan_f64(), math.nan_f64()
			continue
		}
		lr, sr := 0.0, 0.0
		for c in s - skip - hold ..= s - skip - 1 {
			for wi, i in weights[c] {
				r := returns[s][i]
				if wi == 0 || math.is_nan(r) {continue}
				if wi > 0 {lr += wi * r} else {sr -= wi * r}
			}
		}
		out.long[s], out.short[s] = lr / f64(hold), sr / f64(hold)
		out.spread[s] = out.long[s] - out.short[s]
	}
	return out
}

// ============================================================================
// Behavioral Kalman filter (heuristic: anchoring as a sticky prior)
// ============================================================================
// Not from the literature. A local-level Kalman filter (random-walk state with
// process variance q, observation noise variance R) whose gain is multiplied by
// (1 - alpha) when an observation pushes the estimate up while it is still below the
// anchor: slow updating of beliefs toward the anchor. The variance update is the
// Joseph form P = (1 - K)^2 P_pred + K^2 R, the error variance for the gain actually
// applied (for a gain chosen from the innovation's sign it is an approximation).
// anchors: one anchor per step (e.g. the 52-week high); nil = `anchor` throughout.
// alpha = 0 is the standard Kalman filter.
behavioral_kalman_filter :: proc(
	observations: []f64,
	anchor: f64,
	alpha: f64 = 0.5, // anchoring strength (0 = none, 1 = no upward updating below it)
	process_var: f64 = 0.01,
	measurement_var: f64 = 0.1,
	anchors: []f64 = nil,
	allocator := context.allocator,
) -> (
	estimates, gains, variances: []f64,
) {
	n := len(observations)
	assert(
		anchors == nil || len(anchors) == n,
		"behavioral_kalman_filter: one anchor per observation",
	)
	estimates = make([]f64, n, allocator)
	gains = make([]f64, n, allocator)
	variances = make([]f64, n, allocator)
	if n == 0 {return}
	x := observations[0]
	P := 1.0
	for t in 0 ..< n {
		x_pred := x
		P_pred := P + process_var
		K := P_pred / (P_pred + measurement_var)
		innovation := observations[t] - x_pred
		a := anchors[t] if anchors != nil else anchor
		if innovation > 0 && x_pred < a {K *= 1 - alpha}
		x = x_pred + K * innovation
		P = (1 - K) * (1 - K) * P_pred + K * K * measurement_var
		estimates[t], gains[t], variances[t] = x, K, P
	}
	return
}
