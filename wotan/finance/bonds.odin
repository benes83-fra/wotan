package finance

import w "../core"
import "core:math"
import "core:mem"

// ============================================================================
// BOND STRUCTURES
// ============================================================================

BondType :: enum {
	Fixed,
	Floating, // priced like Fixed with coupon_rate as the (projected) coupon
	ZeroCoupon,
}

CouponFrequency :: enum {
	Annual,
	SemiAnnual,
	Quarterly,
	Monthly,
}

// Times are year fractions; the day_count field is informational (accrual is
// linear in time within each coupon period).
Bond :: struct {
	face_value:       f64,
	coupon_rate:      f64, // Annual coupon rate (e.g., 0.05 for 5%)
	maturity:         f64, // Years from issuance
	issue_date:       f64, // Years from today (0 = issued today, < 0 = issued in the past)
	settlement_date:  f64, // Years from today (when bond is purchased)
	bond_type:        BondType,
	coupon_frequency: CouponFrequency,
	day_count:        DayCountConvention,
}

// Prices are values at the settlement date. Yield, duration and convexity use
// the street convention P(y) = Σ cf (1 + y/f)^(−f τ), τ = years from settlement.
BondPriceResult :: struct {
	clean_price:       f64, // Price excluding accrued interest
	dirty_price:       f64, // Price including accrued interest
	accrued_interest:  f64, // Interest accrued since last coupon
	yield_to_maturity: f64, // YTM (annualized, compounded at the coupon frequency)
	modified_duration: f64, // −P'(y) / P
	convexity:         f64, // P''(y) / P
}

// ============================================================================
// COUPON SCHEDULING
// ============================================================================

@(private = "file")
_bond_freq :: proc(f: CouponFrequency) -> f64 {
	switch f {
	case .Annual:
		return 1.0
	case .SemiAnnual:
		return 2.0
	case .Quarterly:
		return 4.0
	case .Monthly:
		return 12.0
	}
	return 1.0
}

// Coupon dates (years from today), generated backwards from maturity: when the
// maturity is not a whole number of periods the first period is a short stub.
get_coupon_dates :: proc(bond: Bond, allocator: mem.Allocator) -> []f64 {
	freq := _bond_freq(bond.coupon_frequency)
	m := bond.maturity * freq
	n := int(math.round(m))
	if abs(m - f64(n)) > 1e-9 * max(1.0, m) {n = int(math.ceil(m))}
	n = max(n, 1)

	end := bond.issue_date + bond.maturity
	dates := make([]f64, n, allocator)
	for i in 0 ..< n {
		dates[i] = end - f64(n - 1 - i) / freq
	}
	return dates
}

// Payment times and amounts (coupon plus redemption). A short first stub pays a
// pro-rata coupon; a zero-coupon bond pays the face value at maturity only.
@(private = "file")
_bond_cashflows :: proc(bond: Bond, allocator: mem.Allocator) -> (times: []f64, amounts: []f64) {
	if bond.bond_type == .ZeroCoupon {
		times = make([]f64, 1, allocator)
		amounts = make([]f64, 1, allocator)
		times[0] = bond.issue_date + bond.maturity
		amounts[0] = bond.face_value
		return
	}
	freq := _bond_freq(bond.coupon_frequency)
	times = get_coupon_dates(bond, allocator)
	amounts = make([]f64, len(times), allocator)
	coupon := bond.face_value * bond.coupon_rate / freq
	for i in 0 ..< len(times) {amounts[i] = coupon}
	stub := (times[0] - bond.issue_date) * freq
	if abs(stub - 1.0) > 1e-9 {amounts[0] = coupon * stub}
	amounts[len(amounts) - 1] += bond.face_value
	return
}

// ============================================================================
// ACCRUED INTEREST CALCULATION
// ============================================================================

// Coupon of the current period × elapsed fraction of the period; zero before
// issue, on a coupon date and after maturity.
calculate_accrued_interest :: proc(bond: Bond) -> f64 {
	if bond.bond_type == .ZeroCoupon {
		return 0.0
	}
	s := bond.settlement_date
	if s <= bond.issue_date || s >= bond.issue_date + bond.maturity {
		return 0.0
	}

	freq := _bond_freq(bond.coupon_frequency)
	dates := get_coupon_dates(bond, context.temp_allocator)
	defer delete(dates, context.temp_allocator)

	last := bond.issue_date
	for d, i in dates {
		if d <= s {
			last = d
			continue
		}
		coupon := bond.face_value * bond.coupon_rate / freq
		if i == 0 {coupon *= (d - bond.issue_date) * freq}
		return coupon * (s - last) / (d - last)
	}
	return 0.0
}

// ============================================================================
// PRICE / YIELD KERNEL
// ============================================================================

// P(y) = Σ cf (1 + y/f)^(−f τ) and its first two derivatives over the flows
// with τ ≥ 0, τ the discounting time from settlement (τ < 0 marks a flow
// already paid)
@(private = "file")
_bond_yield_pv :: proc(taus, amounts: []f64, y, freq: f64) -> (p, dp, d2p: f64) {
	g := 1.0 + y / freq
	for tau, i in taus {
		if tau < 0.0 {continue}
		pv := amounts[i] * math.pow_f64(g, -tau * freq)
		p += pv
		dp -= tau * pv / g
		d2p += tau * (tau + 1.0 / freq) * pv / (g * g)
	}
	return
}

// τ = t − settlement for the flows after settlement, −1 for the others
@(private = "file")
_bond_taus :: proc(times: []f64, settle: f64, allocator: mem.Allocator) -> []f64 {
	taus := make([]f64, len(times), allocator)
	for t, i in times {taus[i] = t - settle if t > settle else -1.0}
	return taus
}

// Yield y with P(y) = target (a dirty price). Safeguarded Newton on a bracket
// (P is decreasing for y > −f); NaN when no flows remain or the target is not
// positive.
@(private = "file")
_solve_yield :: proc(taus, amounts: []f64, freq, target, guess: f64) -> f64 {
	remaining := false
	for tau in taus {remaining ||= tau >= 0.0}
	if !(target > 0.0) || !remaining {
		return math.QNAN_F64
	}

	// Bracket: P(lo) > target > P(hi)
	lo := -freq * (1.0 - 1e-9)
	hi := 1.0
	for _ in 0 ..< 60 {
		p, _, _ := _bond_yield_pv(taus, amounts, hi, freq)
		if p < target {break}
		hi *= 2.0
	}
	for _ in 0 ..< 60 {
		p, _, _ := _bond_yield_pv(taus, amounts, lo, freq)
		if p > target {break}
		lo = -freq + (lo + freq) * 1e-3
	}

	y := guess if guess > lo && guess < hi else 0.5 * (lo + hi)
	for _ in 0 ..< 200 {
		p, dp, _ := _bond_yield_pv(taus, amounts, y, freq)
		err := p - target
		if err > 0.0 {lo = y} else {hi = y}
		if abs(err) <= 1e-14 * target {return y}

		y_new := y - err / dp
		if !(y_new > lo && y_new < hi) {y_new = 0.5 * (lo + hi)}
		if abs(y_new - y) <= 1e-15 * max(1.0, abs(y)) {return y_new}
		y = y_new
	}
	return y
}

// Accrued interest, clean price, yield and risk measures from a dirty price at settlement
@(private = "file")
_bond_finish :: proc(bond: Bond, dirty_price: f64, allocator: mem.Allocator) -> BondPriceResult {
	accrued := calculate_accrued_interest(bond)
	clean := dirty_price - accrued
	ytm := solve_ytm(bond, clean, allocator)
	mod_duration, conv := calculate_duration_convexity(bond, ytm, allocator)
	return BondPriceResult {
		clean_price = clean,
		dirty_price = dirty_price,
		accrued_interest = accrued,
		yield_to_maturity = ytm,
		modified_duration = mod_duration,
		convexity = conv,
	}
}

// ============================================================================
// BOND PRICING FROM YIELD CURVE STRUCT (Production Standard)
// ============================================================================
// Prices a bond using a bootstrapped YieldCurve. The dirty price is the value at
// the settlement date: Σ cf DF(t) / DF(settlement) over flows after settlement.

price_bond_from_yield_curve :: proc(
	bond: Bond,
	curve: ^YieldCurve,
	allocator: mem.Allocator = context.allocator,
) -> BondPriceResult {
	times, amounts := _bond_cashflows(bond, allocator)
	defer {delete(times, allocator); delete(amounts, allocator)}

	dirty := 0.0
	for t, i in times {
		if t <= bond.settlement_date {continue}
		dirty += amounts[i] * yield_curve_discount_factor(curve, t)
	}
	dirty /= yield_curve_discount_factor(curve, bond.settlement_date)
	return _bond_finish(bond, dirty, allocator)
}

// ============================================================================
// BOND PRICING FROM YIELD CURVE CALLBACK (Legacy / Custom Curves)
// ============================================================================
// `yield_curve(t, user_data)` must return the discount factor to time t (years
// from today).

price_bond_from_curve :: proc(
	bond: Bond,
	yield_curve: proc(t: f64, user_data: rawptr) -> f64,
	user_data: rawptr,
	allocator: mem.Allocator = context.allocator,
) -> BondPriceResult {
	times, amounts := _bond_cashflows(bond, allocator)
	defer {delete(times, allocator); delete(amounts, allocator)}

	dirty := 0.0
	for t, i in times {
		if t <= bond.settlement_date {continue}
		dirty += amounts[i] * yield_curve(t, user_data)
	}
	if bond.settlement_date > 0.0 {
		dirty /= yield_curve(bond.settlement_date, user_data)
	}
	return _bond_finish(bond, dirty, allocator)
}

// ============================================================================
// YIELD TO MATURITY SOLVER
// ============================================================================

// Yield y with P(y) = clean_price + accrued interest. Safeguarded Newton on a
// bracket (P is decreasing for y > −f); NaN when no flows remain or the dirty
// price is not positive.
solve_ytm :: proc(bond: Bond, clean_price: f64, allocator: mem.Allocator) -> f64 {
	times, amounts := _bond_cashflows(bond, allocator)
	defer {delete(times, allocator); delete(amounts, allocator)}
	taus := _bond_taus(times, bond.settlement_date, allocator)
	defer delete(taus, allocator)
	target := clean_price + calculate_accrued_interest(bond)
	return _solve_yield(taus, amounts, _bond_freq(bond.coupon_frequency), target, bond.coupon_rate)
}

// ============================================================================
// DURATION AND CONVEXITY
// ============================================================================

// Modified duration −P'(y)/P and convexity P''(y)/P of the dirty price at
// settlement, at the yield `ytm` (compounded at the coupon frequency).
calculate_duration_convexity :: proc(
	bond: Bond,
	ytm: f64,
	allocator: mem.Allocator,
) -> (
	modified_duration: f64,
	convexity: f64,
) {
	times, amounts := _bond_cashflows(bond, allocator)
	defer {delete(times, allocator); delete(amounts, allocator)}
	taus := _bond_taus(times, bond.settlement_date, allocator)
	defer delete(taus, allocator)

	p, dp, d2p := _bond_yield_pv(taus, amounts, ytm, _bond_freq(bond.coupon_frequency))
	if !(p > 0.0) {
		return 0.0, 0.0
	}
	return -dp / p, d2p / p
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

create_fixed_bond :: proc(
	face_value: f64,
	coupon_rate: f64,
	maturity: f64,
	coupon_frequency: CouponFrequency = .SemiAnnual,
	day_count: DayCountConvention = .ACT_ACT,
) -> Bond {
	return Bond {
		face_value = face_value,
		coupon_rate = coupon_rate,
		maturity = maturity,
		issue_date = 0.0,
		settlement_date = 0.0,
		bond_type = .Fixed,
		coupon_frequency = coupon_frequency,
		day_count = day_count,
	}
}

create_zero_coupon_bond :: proc(face_value: f64, maturity: f64) -> Bond {
	return Bond {
		face_value = face_value,
		coupon_rate = 0.0,
		maturity = maturity,
		issue_date = 0.0,
		settlement_date = 0.0,
		bond_type = .ZeroCoupon,
		coupon_frequency = .Annual,
		day_count = .ACT_365,
	}
}

// ============================================================================
// DATED FIXED-RATE BONDS
// ============================================================================
// A fixed-rate bond on real dates (QuantLib FixedRateBond conventions): coupons
// face × rate × year_fraction over each accrual period of the schedule, with
// the reference periods of irregular stubs (ACT/ACT ICMA), paid on the accrual
// end adjusted with payment_convention; the face value is redeemed with the
// last coupon. Flows paid on the settlement date belong to the seller.

DatedBond :: struct {
	face_value:         f64,
	coupon_rate:        f64, // annual
	schedule:           Schedule, // accrual dates (not owned)
	day_count:          DayCountConvention,
	payment_convention: BusinessDayConvention,
	settlement_days:    int,
}

DatedCashFlow :: struct {
	payment_date:  w.Date,
	accrual_start: w.Date, // zero for the redemption
	accrual_end:   w.Date,
	ref_start:     w.Date, // reference period for the day count
	ref_end:       w.Date,
	amount:        f64,
}

// Coupons in schedule order, then the redemption
dated_bond_cashflows :: proc(bond: DatedBond, allocator := context.allocator) -> []DatedCashFlow {
	s := bond.schedule
	n := len(s.dates) - 1
	flows := make([]DatedCashFlow, n + 1, allocator)
	last := s.dates[n]
	for k in 0 ..< n {
		rs, re := schedule_reference_period(s, k)
		flows[k] = DatedCashFlow {
			payment_date  = calendar_adjust(s.calendar, s.dates[k + 1], bond.payment_convention),
			accrual_start = s.dates[k],
			accrual_end   = s.dates[k + 1],
			ref_start     = rs,
			ref_end       = re,
			amount        = bond.face_value * bond.coupon_rate * year_fraction(bond.day_count, s.dates[k], s.dates[k + 1], rs, re, last),
		}
	}
	flows[n] = DatedCashFlow {
		payment_date = calendar_adjust(s.calendar, last, bond.payment_convention),
		amount       = bond.face_value,
	}
	return flows
}

// settlement_days business days after the trade date
dated_bond_settlement_date :: proc(bond: DatedBond, trade_date: w.Date) -> w.Date {
	return calendar_advance(bond.schedule.calendar, trade_date, bond.settlement_days, .Days)
}

// Coupon accrued from the start of the current period to settlement
dated_bond_accrued_interest :: proc(bond: DatedBond, settlement: w.Date) -> f64 {
	s := bond.schedule
	last := s.dates[len(s.dates) - 1]
	for k in 0 ..< len(s.dates) - 1 {
		start, end := s.dates[k], s.dates[k + 1]
		pay := calendar_adjust(s.calendar, end, bond.payment_convention)
		if !w.date_less(start, settlement) || !w.date_less(settlement, pay) {continue}
		rs, re := schedule_reference_period(s, k)
		until := settlement if w.date_less(settlement, end) else end
		return bond.face_value * bond.coupon_rate * year_fraction(bond.day_count, start, until, rs, re, last)
	}
	return 0.0
}

// Compounding frequency of the yield: the coupon frequency (annual if irregular)
@(private = "file")
_dated_bond_freq :: proc(bond: DatedBond) -> f64 {
	f := period_frequency(bond.schedule.tenor)
	return f64(f) if f > 0 else 1.0
}

// Discounting times from settlement, built period by period in the bond's day
// count as QuantLib's CashFlows::npv with an InterestRate (−1 for flows paid on
// or before settlement); also returns the amounts
@(private = "file")
_dated_bond_taus :: proc(bond: DatedBond, settlement: w.Date, allocator: mem.Allocator) -> (taus, amounts: []f64) {
	flows := dated_bond_cashflows(bond, context.temp_allocator)
	defer delete(flows, context.temp_allocator)
	taus = make([]f64, len(flows), allocator)
	amounts = make([]f64, len(flows), allocator)
	last := settlement
	tau := 0.0
	termination := bond.schedule.dates[len(bond.schedule.dates) - 1]
	for f, i in flows {
		amounts[i] = f.amount
		if !w.date_less(settlement, f.payment_date) {
			taus[i] = -1.0
			continue
		}
		rs, re := f.ref_start, f.ref_end
		if f.accrual_start == {} {
			rs = date_add_period(f.payment_date, Period{-1, .Years}) if last == settlement else last
			re = f.payment_date
		}
		tau += year_fraction(bond.day_count, last, f.payment_date, rs, re, termination)
		taus[i] = tau
		last = f.payment_date
	}
	return
}

// Dirty price at settlement from a yield compounded at the coupon frequency in
// the bond's day count
dated_bond_dirty_price :: proc(bond: DatedBond, yield: f64, settlement: w.Date) -> f64 {
	taus, amounts := _dated_bond_taus(bond, settlement, context.temp_allocator)
	defer {delete(taus, context.temp_allocator); delete(amounts, context.temp_allocator)}
	p, _, _ := _bond_yield_pv(taus, amounts, yield, _dated_bond_freq(bond))
	return p
}

dated_bond_clean_price :: proc(bond: DatedBond, yield: f64, settlement: w.Date) -> f64 {
	return dated_bond_dirty_price(bond, yield, settlement) - dated_bond_accrued_interest(bond, settlement)
}

// Yield (compounded at the coupon frequency) of a clean price at settlement
dated_bond_yield :: proc(bond: DatedBond, clean_price: f64, settlement: w.Date) -> f64 {
	taus, amounts := _dated_bond_taus(bond, settlement, context.temp_allocator)
	defer {delete(taus, context.temp_allocator); delete(amounts, context.temp_allocator)}
	target := clean_price + dated_bond_accrued_interest(bond, settlement)
	return _solve_yield(taus, amounts, _dated_bond_freq(bond), target, bond.coupon_rate)
}

// Modified duration −P'(y)/P and convexity P''(y)/P of the dirty price
dated_bond_duration_convexity :: proc(
	bond: DatedBond,
	yield: f64,
	settlement: w.Date,
) -> (
	modified_duration: f64,
	convexity: f64,
) {
	taus, amounts := _dated_bond_taus(bond, settlement, context.temp_allocator)
	defer {delete(taus, context.temp_allocator); delete(amounts, context.temp_allocator)}
	p, dp, d2p := _bond_yield_pv(taus, amounts, yield, _dated_bond_freq(bond))
	if !(p > 0.0) {return 0.0, 0.0}
	return -dp / p, d2p / p
}

// Prices on a curve with a reference_date: dirty = Σ cf DF(pay) / DF(settlement)
// over the flows after settlement, with the yield and risk measures implied
price_dated_bond :: proc(bond: DatedBond, curve: ^YieldCurve, settlement: w.Date) -> BondPriceResult {
	flows := dated_bond_cashflows(bond, context.temp_allocator)
	defer delete(flows, context.temp_allocator)
	dirty := 0.0
	for f in flows {
		if !w.date_less(settlement, f.payment_date) {continue}
		dirty += f.amount * yield_curve_discount_factor_date(curve, f.payment_date)
	}
	dirty /= yield_curve_discount_factor_date(curve, settlement)
	accrued := dated_bond_accrued_interest(bond, settlement)
	ytm := dated_bond_yield(bond, dirty - accrued, settlement)
	dur, conv := dated_bond_duration_convexity(bond, ytm, settlement)
	return BondPriceResult {
		clean_price = dirty - accrued,
		dirty_price = dirty,
		accrued_interest = accrued,
		yield_to_maturity = ytm,
		modified_duration = dur,
		convexity = conv,
	}
}
