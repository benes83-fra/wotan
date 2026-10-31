package finance

import w "../core"
import "core:math"
import "core:mem"

// ============================================================================
// INFLATION: CPI INDEX, ZERO-INFLATION CURVE, ZERO-COUPON INFLATION SWAPS,
// INFLATION-LINKED BONDS
// ============================================================================
// Conventions of QuantLib's zero-inflation framework (verified against 1.44):
// - a monthly index has one fixing per month, keyed by the first of the month;
// - an observation on date d with an L-month lag reads the fixing of the month
//   of d − L (Flat) or interpolates linearly between that month and the next
//   by the day of d within its own month (Linear, the TIPS / ILB convention);
// - fixings after the last published one are forecast from a zero-inflation
//   curve: I(m) = I(base) · (1 + z(m))^{t(base, m)}, z linear in time between
//   the pillar months (flat back to the base month, linear beyond the last).

CPIInterpolation :: enum u8 {
	Flat,
	Linear,
}

ZeroInflationCurve :: struct {
	base_date: w.Date, // first of the base month (normally the last published fixing)
	day_count: DayCountConvention,
	months:    []w.Date, // pillar months (first of month), increasing
	rates:     []f64, // annually compounded zero inflation rates at the pillars
}

// Published fixings by month, forecast beyond them from `curve` (if set)
CPIIndex :: struct {
	months: []w.Date, // first of month, increasing, consecutive
	values: []f64,
	curve:  ^ZeroInflationCurve,
}

free_zero_inflation_curve :: proc(c: ^ZeroInflationCurve, allocator := context.allocator) {
	delete(c.months, allocator)
	delete(c.rates, allocator)
	c^ = {}
}

@(private = "file")
_month_start :: proc(d: w.Date) -> w.Date {return w.Date{d.year, d.month, 1}}

// Zero inflation rate at the month of d
zero_inflation_rate :: proc(c: ^ZeroInflationCurve, d: w.Date) -> f64 {
	n := len(c.rates)
	if n == 0 {return 0}
	t := year_fraction(c.day_count, c.base_date, _month_start(d))
	// nodes: the base month (with the first rate) and the pillars
	node_t :: proc(c: ^ZeroInflationCurve, i: int) -> f64 {return(
			0 if i == 0 else year_fraction(c.day_count, c.base_date, c.months[i - 1]) \
		)}
	node_z :: proc(c: ^ZeroInflationCurve, i: int) -> f64 {return(
			c.rates[0] if i == 0 else c.rates[i - 1] \
		)}
	if t <= 0 {return c.rates[0]}
	k := n - 1 // segment [node k, node k+1], the last one beyond the last pillar
	for i in 0 ..< n {
		if t <= node_t(c, i + 1) {k = i; break}
	}
	t0, t1 := node_t(c, k), node_t(c, k + 1)
	z0, z1 := node_z(c, k), node_z(c, k + 1)
	return z0 + (z1 - z0) * (t - t0) / (t1 - t0)
}

// Index level of the month of d: the published fixing, else the forecast
cpi_fixing :: proc(index: ^CPIIndex, d: w.Date) -> f64 {
	m := _month_start(d)
	n := len(index.months)
	if n > 0 && !w.date_less(index.months[n - 1], m) && !w.date_less(m, index.months[0]) {
		k := int(i64(m.year - index.months[0].year) * 12 + i64(m.month - index.months[0].month))
		return index.values[k]
	}
	c := index.curve
	if c == nil {return math.QNAN_F64}
	base := cpi_fixing(&CPIIndex{index.months, index.values, nil}, c.base_date)
	return(
		base *
		math.pow(1.0 + zero_inflation_rate(c, m), year_fraction(c.day_count, c.base_date, m)) \
	)
}

// Index observation for date d with a lag of `lag_months`
cpi_observation :: proc(
	index: ^CPIIndex,
	d: w.Date,
	lag_months: int,
	interp: CPIInterpolation,
) -> f64 {
	obs := w.add_month_date(d, i32(-lag_months))
	I0 := cpi_fixing(index, obs)
	if interp == .Flat || d.day == 1 {return I0}
	I1 := cpi_fixing(index, w.add_month_date(_month_start(obs), 1))
	days := f64(w.get_days_in_month(d.year, w.Months(d.month)))
	return I0 + (I1 - I0) * f64(d.day - 1) / days
}

// ----------------------------------------------------------------------------
// Zero-coupon inflation swap: at maturity the inflation leg pays
// N (I(T − L)/I(S − L) − 1) and the fixed leg N ((1 + K)^τ − 1), τ the year
// fraction from start to maturity
// ----------------------------------------------------------------------------

ZeroCouponInflationSwap :: struct {
	notional:      f64,
	fixed_rate:    f64,
	start:         w.Date,
	maturity:      w.Date,
	lag_months:    int,
	interpolation: CPIInterpolation,
	calendar:      Calendar, // payment date = maturity adjusted
	convention:    BusinessDayConvention,
	day_count:     DayCountConvention,
}

@(private = "file")
_zcis_tau :: proc(s: ZeroCouponInflationSwap) -> f64 {
	return year_fraction(s.day_count, s.start, s.maturity)
}

// Inflation and fixed amounts and their payment date
zcis_cashflows :: proc(
	s: ZeroCouponInflationSwap,
	index: ^CPIIndex,
) -> (
	inflation_amount, fixed_amount: f64,
	pay: w.Date,
) {
	ratio :=
		cpi_observation(index, s.maturity, s.lag_months, s.interpolation) /
		cpi_observation(index, s.start, s.lag_months, s.interpolation)
	inflation_amount = s.notional * (ratio - 1.0)
	fixed_amount = s.notional * (math.pow(1.0 + s.fixed_rate, _zcis_tau(s)) - 1.0)
	pay = calendar_adjust(s.calendar, s.maturity, s.convention)
	return
}

// Fixed rate with zero value: (I(T − L)/I(S − L))^{1/τ} − 1 (no discounting needed)
zcis_fair_rate :: proc(s: ZeroCouponInflationSwap, index: ^CPIIndex) -> f64 {
	ratio :=
		cpi_observation(index, s.maturity, s.lag_months, s.interpolation) /
		cpi_observation(index, s.start, s.lag_months, s.interpolation)
	return math.pow(ratio, 1.0 / _zcis_tau(s)) - 1.0
}

// Value to the inflation receiver (fixed payer) on a nominal curve
zcis_npv :: proc(s: ZeroCouponInflationSwap, index: ^CPIIndex, nominal: ^YieldCurve) -> f64 {
	inf, fix, pay := zcis_cashflows(s, index)
	return (inf - fix) * yield_curve_discount_factor_date(nominal, pay)
}

ZCISQuote :: struct {
	tenor: Period,
	rate:  f64,
}

// Bootstraps the zero-inflation curve from zero-coupon swap quotes starting on
// `start` (QuantLib ZeroCouponInflationSwapHelper / PiecewiseZeroInflation):
// the pillar of each quote is the month of its lagged maturity, and its rate
// is solved so that the swap's fair rate is the quote. Under Linear
// observation a quote also reads the month after its pillar, which the next
// pillar moves, so the pillars are re-solved on the full curve until they no
// longer change (QuantLib's iterative bootstrap). `index` holds the published
// fixings up to base_date; its curve is restored on return.
bootstrap_zero_inflation_curve :: proc(
	index: ^CPIIndex,
	start: w.Date,
	base_date: w.Date,
	quotes: []ZCISQuote,
	lag_months: int,
	interpolation: CPIInterpolation,
	calendar: Calendar,
	convention: BusinessDayConvention = .ModifiedFollowing,
	day_count: DayCountConvention = .ACT_365,
	allocator: mem.Allocator = context.allocator,
) -> ZeroInflationCurve {
	n := len(quotes)
	c := ZeroInflationCurve {
		base_date = _month_start(base_date),
		day_count = day_count,
		months    = make([]w.Date, n, allocator),
		rates     = make([]f64, n, allocator),
	}
	saved := index.curve
	defer index.curve = saved
	swaps := make([]ZeroCouponInflationSwap, n, context.temp_allocator)
	defer delete(swaps, context.temp_allocator)
	for q, i in quotes {
		swaps[i] = ZeroCouponInflationSwap {
			notional      = 1,
			fixed_rate    = q.rate,
			start         = start,
			maturity      = calendar_advance_period(calendar, start, q.tenor, convention),
			lag_months    = lag_months,
			interpolation = interpolation,
			calendar      = calendar,
			convention    = convention,
			day_count     = day_count,
		}
		c.months[i] = _month_start(w.add_month_date(swaps[i].maturity, i32(-lag_months)))
	}
	// pillar i's rate on a curve through `upto` pillars (fair rate − quote increases in it)
	solve :: proc(
		c: ^ZeroInflationCurve,
		index: ^CPIIndex,
		sw: ZeroCouponInflationSwap,
		i, upto: int,
	) -> f64 {
		sub := ZeroInflationCurve{c.base_date, c.day_count, c.months[:upto], c.rates[:upto]}
		index.curve = &sub
		old := c.rates[i]
		lo, hi := -0.5, 0.5
		for _ in 0 ..< 200 {
			mid := 0.5 * (lo + hi)
			if mid == lo || mid == hi {break}
			c.rates[i] = mid
			if zcis_fair_rate(sw, index) < sw.fixed_rate {lo = mid} else {hi = mid}
		}
		c.rates[i] = 0.5 * (lo + hi)
		return abs(c.rates[i] - old)
	}
	for i in 0 ..< n {solve(&c, index, swaps[i], i, i + 1)}
	if interpolation == .Linear {
		for _ in 0 ..< 100 {
			change := 0.0
			for i in 0 ..< n {change = max(change, solve(&c, index, swaps[i], i, n))}
			if change < 1e-15 {break}
		}
	}
	return c
}

// ----------------------------------------------------------------------------
// Inflation-linked bond (QuantLib CPIBond): fixed real coupons on the schedule,
// each scaled by I(accrual end, lagged)/base_cpi, and the face value scaled by
// I(maturity, lagged)/base_cpi, all paid on the adjusted accrual end
// ----------------------------------------------------------------------------

InflationLinkedBond :: struct {
	face_value:         f64,
	real_coupon:        f64,
	base_cpi:           f64,
	schedule:           Schedule, // accrual dates (not owned)
	day_count:          DayCountConvention,
	payment_convention: BusinessDayConvention,
	lag_months:         int,
	interpolation:      CPIInterpolation,
}

// Cash flows in schedule order, then the indexed redemption
inflation_linked_bond_cashflows :: proc(
	b: InflationLinkedBond,
	index: ^CPIIndex,
	allocator := context.allocator,
) -> []DatedCashFlow {
	s := b.schedule
	n := len(s.dates) - 1
	flows := make([]DatedCashFlow, n + 1, allocator)
	last := s.dates[n]
	for k in 0 ..< n {
		rs, re := schedule_reference_period(s, k)
		ratio := cpi_observation(index, s.dates[k + 1], b.lag_months, b.interpolation) / b.base_cpi
		flows[k] = DatedCashFlow {
			payment_date  = calendar_adjust(s.calendar, s.dates[k + 1], b.payment_convention),
			accrual_start = s.dates[k],
			accrual_end   = s.dates[k + 1],
			ref_start     = rs,
			ref_end       = re,
			amount        = b.face_value * b.real_coupon * year_fraction(b.day_count, s.dates[k], s.dates[k + 1], rs, re, last) * ratio,
		}
	}
	flows[n] = DatedCashFlow {
		payment_date = calendar_adjust(s.calendar, last, b.payment_convention),
		amount       = b.face_value * cpi_observation(index, last, b.lag_months, b.interpolation) / b.base_cpi,
	}
	return flows
}

// Dirty value at settlement on a nominal curve: Σ cf DF(pay) / DF(settlement)
// over the flows after settlement
inflation_linked_bond_dirty_price :: proc(
	b: InflationLinkedBond,
	index: ^CPIIndex,
	nominal: ^YieldCurve,
	settlement: w.Date,
) -> f64 {
	flows := inflation_linked_bond_cashflows(b, index, context.temp_allocator)
	defer delete(flows, context.temp_allocator)
	v := 0.0
	for f in flows {
		if w.date_less(
			settlement,
			f.payment_date,
		) {v += f.amount * yield_curve_discount_factor_date(nominal, f.payment_date)}
	}
	return v / yield_curve_discount_factor_date(nominal, settlement)
}
