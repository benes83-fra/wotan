package finance

import "core:math"
import "core:mem"

// ============================================================================
// YIELD CURVE STRUCTURES
// ============================================================================

YieldCurve :: struct {
	tenors:     []f64, // In years
	zero_rates: []f64, // Continuously compounded zero rates
}

// ============================================================================
// YIELD CURVE BOOTSTRAPPING
// ============================================================================

// Bootstrap a yield curve from a set of market par swap rates.
// This iteratively solves for the zero rate at each tenor such that a swap
// priced with these zero rates would have an NPV of exactly 0 (i.e., it's at par).
bootstrap_yield_curve_from_swaps :: proc(
	tenors: []f64,
	par_swap_rates: []f64,
	payment_frequency: f64 = 0.25, // Quarterly
	day_count: DayCountConvention = .ACT_360,
	allocator: mem.Allocator = context.allocator,
) -> YieldCurve {
	n := len(tenors)
	if n == 0 {
		return YieldCurve{}
	}

	curve := YieldCurve {
		tenors     = make([]f64, n, allocator),
		zero_rates = make([]f64, n, allocator),
	}

	for i in 0 ..< n {
		curve.tenors[i] = tenors[i]
		T := tenors[i]
		par_rate := par_swap_rates[i]

		// 1. Calculate the annuity (PV01) using already bootstrapped zero rates
		annuity := 0.0
		n_payments := int(T / payment_frequency)
		for j in 1 ..< n_payments + 1 {
			t_j := f64(j) * payment_frequency
			if t_j > T {t_j = T}

			// Interpolate zero rate for t_j using only previously bootstrapped points
			r_j := _interpolate_zero_rate(curve.tenors[:i], curve.zero_rates[:i], t_j)
			df_j := math.exp_f64(-r_j * t_j)

			dcf := day_count_fraction(t_j - payment_frequency, t_j, day_count)
			annuity += dcf * df_j
		}

		// 2. Solve for the final discount factor.
		// Par swap formula: par_rate = (1.0 - DF_T) / annuity
		// => DF_T = 1.0 - par_rate * annuity
		df_T := 1.0 - par_rate * annuity

		// Safety clamp to prevent log errors from bad market data
		if df_T <= 1e-10 {
			df_T = 1e-10
		}

		// 3. Convert discount factor to continuously compounded zero rate:
		// DF_T = exp(-r_T * T)  =>  r_T = -ln(DF_T) / T
		if T > 1e-10 {
			curve.zero_rates[i] = -math.ln_f64(df_T) / T
		} else {
			curve.zero_rates[i] = par_rate // Fallback for very short tenor (e.g., O/N)
		}
	}

	return curve
}

// Helper: Linear interpolation of zero rates (standard, arbitrage-free enough for basic use)
_interpolate_zero_rate :: proc(tenors: []f64, zero_rates: []f64, t: f64) -> f64 {
	n := len(tenors)
	if n == 0 {return 0.0}
	if t <= tenors[0] {return zero_rates[0]}
	if t >= tenors[n - 1] {return zero_rates[n - 1]}

	for i in 0 ..< n - 1 {
		if t >= tenors[i] && t <= tenors[i + 1] {
			w := (t - tenors[i]) / (tenors[i + 1] - tenors[i])
			return zero_rates[i] * (1.0 - w) + zero_rates[i + 1] * w
		}
	}
	return zero_rates[n - 1]
}

// ============================================================================
// YIELD CURVE QUERIES
// ============================================================================

// Get discount factor for a given time t
yield_curve_discount_factor :: proc(curve: ^YieldCurve, t: f64) -> f64 {
	if t <= 0.0 {return 1.0}
	r := _interpolate_zero_rate(curve.tenors, curve.zero_rates, t)
	return math.exp_f64(-r * t)
}

// Get zero rate for a given time t
yield_curve_zero_rate :: proc(curve: ^YieldCurve, t: f64) -> f64 {
	if t <= 0.0 {return 0.0}
	return _interpolate_zero_rate(curve.tenors, curve.zero_rates, t)
}

// Get forward rate between t1 and t2 (continuously compounded)
yield_curve_forward_rate :: proc(curve: ^YieldCurve, t1: f64, t2: f64) -> f64 {
	if t2 <= t1 {return 0.0}
	r1 := _interpolate_zero_rate(curve.tenors, curve.zero_rates, t1)
	r2 := _interpolate_zero_rate(curve.tenors, curve.zero_rates, t2)
	return (r2 * t2 - r1 * t1) / (t2 - t1)
}

// Free the yield curve memory
free_yield_curve :: proc(curve: ^YieldCurve, allocator: mem.Allocator) {
	delete(curve.tenors, allocator)
	delete(curve.zero_rates, allocator)
}
