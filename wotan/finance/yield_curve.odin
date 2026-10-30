package finance

import w "../core"
import "core:math"
import "core:mem"

// ============================================================================
// YIELD CURVE STRUCTURES
// ============================================================================

CurveInterpolation :: enum u8 {
	LinearZero, // zero rates linear in time, flat before the first and after the last pillar
	FlatForward, // piecewise-constant forwards (log-linear discount factors), the
	// first forward back to t = 0 and the last one carried beyond the last pillar
}

YieldCurve :: struct {
	tenors:         []f64, // In years
	zero_rates:     []f64, // Continuously compounded zero rates
	reference_date: w.Date, // date of t = 0 for dated instruments (times ACT/365F); unset for time-only curves
	interpolation:  CurveInterpolation,
}

// ============================================================================
// YIELD CURVE BOOTSTRAPPING
// ============================================================================

// Solves the zero rate at pillar i so that pv, the value of the pillar's
// instrument on the curve through pillars 0 … i (decreasing in z_i), is zero
@(private = "file")
_solve_pillar :: proc(
	curve: ^YieldCurve,
	i: int,
	pv: proc(sub: ^YieldCurve, data: rawptr) -> f64,
	data: rawptr,
) {
	f :: proc(
		c: ^YieldCurve,
		i: int,
		z: f64,
		pv: proc(sub: ^YieldCurve, data: rawptr) -> f64,
		data: rawptr,
	) -> f64 {
		c.zero_rates[i] = z
		sub := YieldCurve {
			c.tenors[:i + 1],
			c.zero_rates[:i + 1],
			c.reference_date,
			c.interpolation,
		}
		return pv(&sub, data)
	}
	lo, hi := -1.0, 1.0
	for f(curve, i, hi, pv, data) > 0.0 && hi < 100.0 {hi *= 2.0}
	for f(curve, i, lo, pv, data) < 0.0 && lo > -100.0 {lo *= 2.0}
	for _ in 0 ..< 200 {
		mid := 0.5 * (lo + hi)
		if mid == lo || mid == hi {break}
		if f(curve, i, mid, pv, data) > 0.0 {lo = mid} else {hi = mid}
	}
	curve.zero_rates[i] = 0.5 * (lo + hi)
}

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

	// Par swap value per unit notional: S·A − (1 − DF(T))
	Quote :: struct {
		S, freq: f64,
		dc:      DayCountConvention,
	}
	par_pv :: proc(sub: ^YieldCurve, data: rawptr) -> f64 {
		q := (^Quote)(data)
		T := sub.tenors[len(sub.tenors) - 1]
		dates := _swap_schedule(0.0, T, q.freq, context.temp_allocator)
		defer delete(dates, context.temp_allocator)
		annuity := 0.0
		for k in 1 ..< len(dates) {
			annuity +=
				day_count_fraction(dates[k - 1], dates[k], q.dc) *
				yield_curve_discount_factor(sub, dates[k])
		}
		return q.S * annuity - (1.0 - yield_curve_discount_factor(sub, T))
	}

	for i in 0 ..< n {
		q := Quote{par_swap_rates[i], payment_frequency, day_count}
		_solve_pillar(&curve, i, par_pv, &q)
	}

	return curve
}

CurveQuoteKind :: enum u8 {
	Deposit, // simple rate from the spot date over the tenor
	Swap, // par rate of a spot-starting fixed-float swap
}

CurveQuote :: struct {
	kind:  CurveQuoteKind,
	tenor: Period,
	rate:  f64,
}

// Bootstraps a curve on real dates from deposits and par swaps (fixed-float or
// OIS) quoted under `conv`; deposits use the floating-leg calendar, convention
// and day count. Each instrument starts on the spot date; its pillar is the
// last date it depends on (maturity, last payment, end of the last forward
// period). Zero rates are linear in ACT/365F time from reference_date (flat
// outside the pillars), or flat forwards with `.FlatForward`, and each pillar
// is solved so that its instrument reprices exactly on the final curve, as
// QuantLib's PiecewiseLinearZero / PiecewiseFlatForward.
// With `discount` the result is a projection curve: swaps are discounted on the
// given curve (e.g. OIS) and project their floating coupons on the curve being
// built; without it the curve discounts itself. Quotes must be in increasing
// maturity.
bootstrap_yield_curve_dated :: proc(
	reference_date: w.Date,
	quotes: []CurveQuote,
	conv: SwapConventions,
	discount: ^YieldCurve = nil,
	interpolation: CurveInterpolation = .LinearZero,
	allocator: mem.Allocator = context.allocator,
) -> YieldCurve {
	n := len(quotes)
	curve := YieldCurve {
		tenors         = make([]f64, n, allocator),
		zero_rates     = make([]f64, n, allocator),
		reference_date = reference_date,
		interpolation  = interpolation,
	}
	spot := swap_spot_date(conv, reference_date)

	Deposit :: struct {
		start, end: f64,
		rate, tau:  f64,
	}
	// DF(end)(1 + r τ) − DF(start)
	deposit_pv :: proc(sub: ^YieldCurve, data: rawptr) -> f64 {
		q := (^Deposit)(data)
		return(
			yield_curve_discount_factor(sub, q.end) * (1.0 + q.rate * q.tau) -
			yield_curve_discount_factor(sub, q.start) \
		)
	}
	Swap :: struct {
		swap:     DatedSwap,
		discount: ^YieldCurve,
	}
	// receiver swap at the quoted rate, per unit notional
	swap_pv :: proc(sub: ^YieldCurve, data: rawptr) -> f64 {
		q := (^Swap)(data)
		disc := q.discount if q.discount != nil else sub
		return price_dated_swap(q.swap, disc, sub, context.temp_allocator).npv
	}

	for q, i in quotes {
		switch q.kind {
		case .Deposit:
			end := calendar_advance_period(
				conv.calendar,
				spot,
				q.tenor,
				conv.float_convention,
				conv.end_of_month,
			)
			curve.tenors[i] = date_to_time(reference_date, end)
			d := Deposit {
				date_to_time(reference_date, spot),
				curve.tenors[i],
				q.rate,
				year_fraction(conv.float_day_count, spot, end),
			}
			_solve_pillar(&curve, i, deposit_pv, &d)
		case .Swap:
			sq := Swap {
				make_dated_swap(
					conv,
					spot,
					q.tenor,
					q.rate,
					1.0,
					.Receiver,
					allocator = context.temp_allocator,
				),
				discount,
			}
			defer free_dated_swap(&sq.swap, context.temp_allocator)
			curve.tenors[i] = date_to_time(reference_date, dated_swap_last_relevant_date(sq.swap))
			_solve_pillar(&curve, i, swap_pv, &sq)
		}
	}
	return curve
}

// Discount factor to a date on a curve with a reference_date
yield_curve_discount_factor_date :: proc(curve: ^YieldCurve, d: w.Date) -> f64 {
	return yield_curve_discount_factor(curve, date_to_time(curve.reference_date, d))
}

// Zero rate at t under the curve's interpolation
@(private = "file")
_interpolate_zero_rate :: proc(curve: ^YieldCurve, t: f64) -> f64 {
	tenors, z := curve.tenors, curve.zero_rates
	n := len(tenors)
	if n == 0 {return 0.0}
	if t <= tenors[0] {return z[0]}
	k := n - 2 // segment [tenors[k], tenors[k+1]] containing t (the last one beyond)
	for i in 0 ..< n - 1 {
		if t <= tenors[i + 1] {k = i; break}
	}
	switch curve.interpolation {
	case .LinearZero:
		if t >= tenors[n - 1] {return z[n - 1]}
		wt := (t - tenors[k]) / (tenors[k + 1] - tenors[k])
		return z[k] * (1.0 - wt) + z[k + 1] * wt
	case .FlatForward:
		if n == 1 {return z[0]}
		fwd := (z[k + 1] * tenors[k + 1] - z[k] * tenors[k]) / (tenors[k + 1] - tenors[k])
		return (z[k] * tenors[k] + fwd * (t - tenors[k])) / t
	}
	return z[n - 1]
}

// ============================================================================
// YIELD CURVE QUERIES
// ============================================================================

// Get discount factor for a given time t
yield_curve_discount_factor :: proc(curve: ^YieldCurve, t: f64) -> f64 {
	if t <= 0.0 {return 1.0}
	r := _interpolate_zero_rate(curve, t)
	return math.exp_f64(-r * t)
}

// Get zero rate for a given time t
yield_curve_zero_rate :: proc(curve: ^YieldCurve, t: f64) -> f64 {
	if t <= 0.0 {return 0.0}
	return _interpolate_zero_rate(curve, t)
}

// Get forward rate between t1 and t2 (continuously compounded)
yield_curve_forward_rate :: proc(curve: ^YieldCurve, t1: f64, t2: f64) -> f64 {
	if t2 <= t1 {return 0.0}
	r1 := _interpolate_zero_rate(curve, t1)
	r2 := _interpolate_zero_rate(curve, t2)
	return (r2 * t2 - r1 * t1) / (t2 - t1)
}

// Free the yield curve memory
free_yield_curve :: proc(curve: ^YieldCurve, allocator: mem.Allocator) {
	delete(curve.tenors, allocator)
	delete(curve.zero_rates, allocator)
}
