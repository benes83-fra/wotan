package finance

import "core:math"
import "core:mem"

// ============================================================================
// IMPLIED DEFAULT PROBABILITY (Breeden-Litzenberger)
// ============================================================================
// The Breeden-Litzenberger formula states that the risk-neutral density (RND)
// of the underlying asset at maturity T is proportional to the second derivative
// of the call price with respect to the strike:
//   f(K) = e^{rT} * (∂²C / ∂K²)
// The Risk-Neutral Probability of Default is the integral of f(K) from 0 to K_default.

DefaultProbResult :: struct {
	rnd_strikes:     []f64, // Grid of strikes
	rnd_values:      []f64, // Risk-Neutral Density at each strike
	pd_risk_neutral: f64, // Probability of Default (area under RND up to K_default)
	expected_loss:   f64, // Expected loss given default (integral of K * f(K))
}

// Extract RND and compute PD using central finite differences
implied_default_probability :: proc(
	strikes: []f64, // Must be sorted ascending
	call_prices: []f64, // Corresponding call prices (or synthetic from IV)
	r: f64, // Risk-free rate
	T: f64, // Time to maturity
	K_default: f64, // Default barrier (e.g., face value of debt)
	allocator: mem.Allocator = context.allocator,
) -> DefaultProbResult {
	n := len(strikes)
	if n < 3 {
		return DefaultProbResult{}
	}

	// 1. Compute second derivative of call prices w.r.t strike (∂²C / ∂K²)
	// We use central differences for interior points, forward/backward for edges
	d2C_dK2 := make([]f64, n, allocator)
	defer delete(d2C_dK2, allocator)

	for i in 0 ..< n {
		if i == 0 {
			// Forward difference
			h1 := strikes[1] - strikes[0]
			h2 := strikes[2] - strikes[0]
			d2C_dK2[i] =
				2.0 *
				(call_prices[0] / (h1 * (h1 + h2)) -
						call_prices[1] / (h1 * h2) +
						call_prices[2] / (h2 * (h1 + h2)))
		} else if i == n - 1 {
			// Backward difference
			h1 := strikes[n - 1] - strikes[n - 2]
			h2 := strikes[n - 1] - strikes[n - 3]
			d2C_dK2[i] =
				2.0 *
				(call_prices[n - 3] / (h1 * (h1 + h2)) -
						call_prices[n - 2] / (h1 * h2) +
						call_prices[n - 1] / (h2 * (h1 + h2)))
		} else {
			// Central difference
			h_prev := strikes[i] - strikes[i - 1]
			h_next := strikes[i + 1] - strikes[i]
			h_avg := (h_prev + h_next) / 2.0

			// General non-uniform grid second derivative
			d2C_dK2[i] =
				2.0 *
				(call_prices[i - 1] / (h_prev * (h_prev + h_next)) -
						call_prices[i] / (h_prev * h_next) +
						call_prices[i + 1] / (h_next * (h_prev + h_next)))
		}
	}

	// 2. Convert to Risk-Neutral Density: f(K) = e^{rT} * ∂²C / ∂K²
	// Enforce non-negativity (no-arbitrage condition: butterfly spread >= 0)
	rnd := make([]f64, n, allocator)
	for i in 0 ..< n {
		rnd[i] = math.exp(r * T) * d2C_dK2[i]
		if rnd[i] < 0.0 {
			rnd[i] = 0.0 // Clamp to prevent negative probabilities from noisy data
		}
	}

	// 3. Integrate RND to find Probability of Default (PD)
	// PD = ∫_{0}^{K_default} f(K) dK
	pd := 0.0
	expected_loss := 0.0

	for i in 0 ..< n - 1 {
		// Trapezoidal rule for integration
		width := strikes[i + 1] - strikes[i]

		// Only integrate up to the default barrier
		k1 := strikes[i]
		k2 := strikes[i + 1]

		if k2 <= K_default {
			// Entire interval is below default barrier
			pd += 0.5 * (rnd[i] + rnd[i + 1]) * width
			expected_loss += 0.5 * (k1 * rnd[i] + k2 * rnd[i + 1]) * width
		} else if k1 < K_default {
			// Interval crosses the default barrier; interpolate
			frac := (K_default - k1) / width
			rnd_at_default := rnd[i] + frac * (rnd[i + 1] - rnd[i])

			pd += 0.5 * (rnd[i] + rnd_at_default) * (K_default - k1)
			expected_loss += 0.5 * (k1 * rnd[i] + K_default * rnd_at_default) * (K_default - k1)
			break // We've reached the barrier
		}
	}

	// Clone slices for the result
	out_strikes := make([]f64, n, allocator)
	out_rnd := make([]f64, n, allocator)
	copy(out_strikes, strikes)
	copy(out_rnd, rnd)

	return DefaultProbResult {
		rnd_strikes = out_strikes,
		rnd_values = out_rnd,
		pd_risk_neutral = pd,
		expected_loss = expected_loss,
	}
}
