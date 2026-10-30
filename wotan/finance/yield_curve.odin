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

// Bootstrap a yield curve from market par swap rates (spot-starting swaps paying
// fixed every payment_frequency years, coupon dates backwards from each tenor
// with a short first period, floating leg worth 1 − DF(T)). Tenor by tenor the
// zero rate z_i is solved so that the par swap reprices exactly on the final
// curve, i.e. with the linear zero-rate interpolation between z_{i−1} and z_i
// used for its intermediate coupons; a tenor shorter than one period is a
// deposit, DF(T) = 1 / (1 + S · dcf(0, T)). Tenors must be increasing.
bootstrap_yield_curve_from_swaps :: proc(
	tenors: []f64,
	par_swap_rates: []f64,
	payment_frequency: f64 = 0.25, // Quarterly
	day_count: DayCountConvention = .ACT_360,
	allocator: mem.Allocator = context.allocator,
) -> YieldCurve {
	n := min(len(tenors), len(par_swap_rates))
	if n == 0 {
		return YieldCurve{}
	}

	curve := YieldCurve {
		tenors     = make([]f64, n, allocator),
		zero_rates = make([]f64, n, allocator),
	}
	copy(curve.tenors, tenors[:n])

	// Par swap value per unit notional with z at pillar i: S·A − (1 − DF(T)),
	// decreasing in z
	par_pv :: proc(c: ^YieldCurve, i: int, z, S, freq: f64, dc: DayCountConvention) -> f64 {
		c.zero_rates[i] = z
		sub := YieldCurve{c.tenors[:i + 1], c.zero_rates[:i + 1]}
		T := c.tenors[i]
		dates := _swap_schedule(0.0, T, freq, context.temp_allocator)
		defer delete(dates, context.temp_allocator)
		annuity := 0.0
		for k in 1 ..< len(dates) {
			annuity +=
				day_count_fraction(dates[k - 1], dates[k], dc) *
				yield_curve_discount_factor(&sub, dates[k])
		}
		return S * annuity - (1.0 - yield_curve_discount_factor(&sub, T))
	}

	for i in 0 ..< n {
		S := par_swap_rates[i]
		lo, hi := -1.0, 1.0
		for par_pv(&curve, i, hi, S, payment_frequency, day_count) > 0.0 && hi < 100.0 {hi *= 2.0}
		for par_pv(&curve, i, lo, S, payment_frequency, day_count) < 0.0 && lo > -100.0 {lo *= 2.0}
		for _ in 0 ..< 200 {
			mid := 0.5 * (lo + hi)
			if mid == lo || mid == hi {break}
			if par_pv(&curve, i, mid, S, payment_frequency, day_count) >
			   0.0 {lo = mid} else {hi = mid}
		}
		curve.zero_rates[i] = 0.5 * (lo + hi)
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
