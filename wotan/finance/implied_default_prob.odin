package finance

import "core:math"
import "core:mem"

// ============================================================================
// IMPLIED DEFAULT PROBABILITY (Breeden-Litzenberger)
// ============================================================================
// With C(K) the call price curve at maturity T:
//   risk-neutral density   f(K)        = e^{rT} ∂²C/∂K²
//   distribution           P(S_T < K)  = 1 + e^{rT} ∂C/∂K
// The PD for a default barrier K_d is P(S_T < K_d), taken from the slope of the
// call curve at K_d. That includes the probability mass below the lowest quoted
// strike, which integrating the density over the strike grid would miss.
// Finite differences amplify price noise, so Monte Carlo prices should come from
// one set of paths for all strikes (e.g. rbergomi_mc_calls).

DefaultProbResult :: struct {
	rnd_strikes:     []f64, // Grid of strikes (a copy of the input)
	rnd_values:      []f64, // Risk-neutral density at each strike, clamped at 0
	pd_risk_neutral: f64, // P(S_T < K_default); NaN if K_default is outside the strike grid
	expected_loss:   f64, // PV of E[(K_default - S_T)^+], the put at K_default (Merton's expected loss to the debt)
}

// Lagrange quadratic through (x[i], y[i]), i = j-1, j, j+1: value and slope at t
@(private = "file")
_quad_interp :: proc(x, y: []f64, j: int, t: f64) -> (value, slope: f64) {
	x0, x1, x2 := x[j - 1], x[j], x[j + 1]
	l0 := (t - x1) * (t - x2) / ((x0 - x1) * (x0 - x2))
	l1 := (t - x0) * (t - x2) / ((x1 - x0) * (x1 - x2))
	l2 := (t - x0) * (t - x1) / ((x2 - x0) * (x2 - x1))
	d0 := ((t - x1) + (t - x2)) / ((x0 - x1) * (x0 - x2))
	d1 := ((t - x0) + (t - x2)) / ((x1 - x0) * (x1 - x2))
	d2 := ((t - x0) + (t - x1)) / ((x2 - x0) * (x2 - x1))
	return l0 * y[j - 1] + l1 * y[j] + l2 * y[j + 1], d0 * y[j - 1] + d1 * y[j] + d2 * y[j + 1]
}

// Risk-neutral density and default probability from a call price curve.
// spot is S e^{-qT} (S without dividends); when given, the expected loss uses
// put-call parity exactly. With spot = 0 it assumes the put at the lowest
// strike is worthless, which makes the expected loss a lower bound.
implied_default_probability :: proc(
	strikes: []f64, // Strictly ascending
	call_prices: []f64, // Corresponding call prices (or synthetic from IV)
	r: f64, // Risk-free rate
	T: f64, // Time to maturity
	K_default: f64, // Default barrier (e.g., face value of debt)
	allocator: mem.Allocator = context.allocator,
	spot: f64 = 0.0,
) -> DefaultProbResult {
	n := len(strikes)
	if n < 3 || len(call_prices) != n {
		return DefaultProbResult{}
	}
	growth := math.exp(r * T)

	// 1. Density at the strikes. The 3-point second difference on a non-uniform
	// grid; at each end the nearest interior value (the same parabola's curvature)
	rnd := make([]f64, n, allocator)
	for i in 1 ..< n - 1 {
		h_prev := strikes[i] - strikes[i - 1]
		h_next := strikes[i + 1] - strikes[i]
		d2 :=
			2.0 *
			(call_prices[i - 1] / (h_prev * (h_prev + h_next)) -
					call_prices[i] / (h_prev * h_next) +
					call_prices[i + 1] / (h_next * (h_prev + h_next)))
		// A negative butterfly is an arbitrage (or noise): clamp it
		rnd[i] = math.max(growth * d2, 0.0)
	}
	rnd[0] = rnd[1]
	rnd[n - 1] = rnd[n - 2]

	// 2. PD and expected loss from the quadratic through the three strikes
	// nearest K_default
	pd, expected_loss := math.QNAN_F64, math.QNAN_F64
	if K_default >= strikes[0] && K_default <= strikes[n - 1] {
		j := 1
		for j < n - 2 && strikes[j + 1] < K_default {j += 1}
		if j < n - 2 && K_default - strikes[j] > 0.5 * (strikes[j + 1] - strikes[j]) {j += 1}
		C_d, dC_d := _quad_interp(strikes, call_prices, j, K_default)
		pd = math.clamp(1.0 + growth * dC_d, 0.0, 1.0)

		df := 1.0 / growth
		if spot > 0.0 {
			expected_loss = C_d - spot + K_default * df
		} else {
			expected_loss = C_d - call_prices[0] + (K_default - strikes[0]) * df
		}
		expected_loss = math.max(expected_loss, 0.0)
	}

	out_strikes := make([]f64, n, allocator)
	copy(out_strikes, strikes)

	return DefaultProbResult {
		rnd_strikes = out_strikes,
		rnd_values = rnd,
		pd_risk_neutral = pd,
		expected_loss = expected_loss,
	}
}
