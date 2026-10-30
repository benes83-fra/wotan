package finance

import w "../core"
import "core:math"
import "core:mem"

// ============================================================================
// INTEREST RATE SWAPS (IRS) - CURVE-AWARE
// ============================================================================

SwapLegType :: enum {
	Fixed,
	Floating,
}

SwapDirection :: enum {
	Payer, // Pay fixed, receive floating
	Receiver, // Receive fixed, pay floating
}

DayCountConvention :: enum {
	ACT_360, // Actual/360 (money market standard)
	ACT_365, // Actual/365 Fixed (UK gilts)
	ACT_ACT, // Actual/Actual (ISDA)
	THIRTY_360, // 30/360 Bond Basis
	ACT_ACT_ICMA, // Actual/Actual (ICMA): per coupon reference period (bonds)
	THIRTY_360_US, // 30/360 US, with the end-of-February rules
	THIRTY_E_360, // 30E/360 (Eurobond basis)
	THIRTY_E_360_ISDA, // 30E/360 ISDA (German)
}

SwapLeg :: struct {
	leg_type:          SwapLegType,
	fixed_rate:        f64, // Only for fixed legs (e.g., 0.03 = 3%)
	floating_spread:   f64, // Spread over floating rate (e.g., 0.005 = 50 bps)
	payment_frequency: f64, // Years between payments (e.g., 0.25 = quarterly)
	day_count:         DayCountConvention,
}

InterestRateSwap :: struct {
	notional:       f64,
	effective_date: f64, // Years from today
	maturity:       f64, // Years from today
	payer_leg:      SwapLeg, // The leg you pay
	receiver_leg:   SwapLeg, // The leg you receive
	direction:      SwapDirection,
}

// Sensitivities are to a parallel shift of the continuously compounded zero
// curve: modified_duration = −(1/P) ∂P/∂z and convexity = (1/P) ∂²P/∂z² of the
// fixed leg's coupons (for continuous compounding they are Σ t·cf·DF / P and
// Σ t²·cf·DF / P).
SwapValuationResult :: struct {
	pv_fixed_leg:      f64, // PV of the fixed coupons (positive)
	pv_floating_leg:   f64, // PV of the floating leg incl. spread (positive)
	npv:               f64, // Net present value (receiver - payer)
	par_swap_rate:     f64, // Fixed rate that makes NPV = 0
	pv01:              f64, // notional × fixed-leg annuity × 1bp
	modified_duration: f64,
	convexity:         f64,
}

// ============================================================================
// DAY COUNT FRACTION CALCULATIONS
// ============================================================================

// Times are ACT/365 year fractions, so ACT/360 scales them by 365/360 and the
// other conventions use them as they are (see year_fraction for the exact
// conventions on dates)
day_count_fraction :: proc(t1: f64, t2: f64, convention: DayCountConvention) -> f64 {
	dt := t2 - t1
	if dt <= 0.0 {
		return 0.0
	}

	switch convention {
	case .ACT_360:
		return dt * 365.0 / 360.0
	case .ACT_365, .ACT_ACT, .THIRTY_360, .ACT_ACT_ICMA, .THIRTY_360_US, .THIRTY_E_360, .THIRTY_E_360_ISDA:
		return dt
	}

	return dt
}

// Accrual dates effective = d_0 < d_1 < … < d_n = maturity, generated backwards
// from maturity every `frequency` years (a short first period if the length is
// not a whole number of periods)
@(private)
_swap_schedule :: proc(effective, maturity, frequency: f64, allocator: mem.Allocator) -> []f64 {
	length := maturity - effective
	if length <= 0.0 || frequency <= 0.0 {return make([]f64, 0, allocator)}
	n := max(1, int(math.ceil(length / frequency - 1e-9)))
	dates := make([]f64, n + 1, allocator)
	dates[0] = effective
	for k in 1 ..= n {dates[k] = maturity - f64(n - k) * frequency}
	return dates
}

// ============================================================================
// SWAP LEG VALUATION (CURVE-AWARE)
// ============================================================================

// One accrual period on the curve's time axis, paid at its end
@(private = "file")
_SwapPeriod :: struct {
	start, end: f64, // years from the curve's t = 0
	accrual:    f64, // day-count fraction
}

@(private = "file")
_time_periods :: proc(effective, maturity: f64, leg: SwapLeg, allocator: mem.Allocator) -> []_SwapPeriod {
	dates := _swap_schedule(effective, maturity, leg.payment_frequency, context.temp_allocator)
	defer delete(dates, context.temp_allocator)
	periods := make([]_SwapPeriod, max(len(dates) - 1, 0), allocator)
	for k in 1 ..< len(dates) {
		periods[k - 1] = {dates[k - 1], dates[k], day_count_fraction(dates[k - 1], dates[k], leg.day_count)}
	}
	return periods
}

// Fixed leg: Σ N · K · dcf_k · DF(t_k); also returns the annuity Σ dcf_k DF(t_k)
// and Σ t·dcf·DF, Σ t²·dcf·DF for the duration and convexity
@(private = "file")
_fixed_leg_pv :: proc(periods: []_SwapPeriod, notional, rate: f64, curve: ^YieldCurve) -> (pv, annuity, t1, t2: f64) {
	for p in periods {
		df := yield_curve_discount_factor(curve, p.end)
		annuity += p.accrual * df
		t1 += p.end * p.accrual * df
		t2 += p.end * p.end * p.accrual * df
	}
	pv = notional * rate * annuity
	return
}

// Floating leg (single curve): the forward (DF(s)/DF(e) − 1)/dcf makes each
// coupon worth N (DF(s) − DF(e)), plus the spread N · spread · dcf · DF(e)
@(private = "file")
_float_leg_pv :: proc(periods: []_SwapPeriod, notional, spread: f64, curve: ^YieldCurve) -> (pv: f64) {
	for p in periods {
		df_end := yield_curve_discount_factor(curve, p.end)
		pv += notional * (yield_curve_discount_factor(curve, p.start) - df_end + spread * p.accrual * df_end)
	}
	return
}

// Values a fixed leg against another leg (floating, or fixed at other_rate)
@(private = "file")
_swap_result :: proc(
	fixed, other: []_SwapPeriod,
	notional, fixed_rate, other_rate: f64,
	other_is_floating, fixed_is_payer: bool,
	curve: ^YieldCurve,
) -> SwapValuationResult {
	pv_fixed, annuity, t1, t2 := _fixed_leg_pv(fixed, notional, fixed_rate, curve)
	pv_other: f64
	if other_is_floating {
		pv_other = _float_leg_pv(other, notional, other_rate, curve)
	} else {
		pv_other, _, _, _ = _fixed_leg_pv(other, notional, other_rate, curve)
	}
	dur, conv := 0.0, 0.0
	if annuity > 0.0 {
		dur = t1 / annuity
		conv = t2 / annuity
	}
	return SwapValuationResult {
		pv_fixed_leg = pv_fixed,
		pv_floating_leg = pv_other,
		npv = pv_other - pv_fixed if fixed_is_payer else pv_fixed - pv_other,
		par_swap_rate = pv_other / (notional * annuity) if annuity > 0.0 && notional != 0.0 else 0.0,
		pv01 = annuity * notional * 0.0001,
		modified_duration = dur,
		convexity = conv,
	}
}

// ============================================================================
// MAIN SWAP PRICING FUNCTION
// ============================================================================

// Values both legs on the curve. The par rate and the PV01 use the fixed leg's
// schedule and day count (the payer leg if both legs are of the same type).
price_swap :: proc(
	swap: InterestRateSwap,
	curve: ^YieldCurve,
	allocator: mem.Allocator = context.allocator,
) -> SwapValuationResult {
	fixed_is_payer := swap.payer_leg.leg_type == .Fixed
	fixed := swap.payer_leg if fixed_is_payer else swap.receiver_leg
	other := swap.receiver_leg if fixed_is_payer else swap.payer_leg

	fixed_periods := _time_periods(swap.effective_date, swap.maturity, fixed, allocator)
	defer delete(fixed_periods, allocator)
	other_periods := _time_periods(swap.effective_date, swap.maturity, other, allocator)
	defer delete(other_periods, allocator)
	other_is_floating := other.leg_type == .Floating
	return _swap_result(
		fixed_periods,
		other_periods,
		swap.notional,
		fixed.fixed_rate,
		other.floating_spread if other_is_floating else other.fixed_rate,
		other_is_floating,
		fixed_is_payer,
		curve,
	)
}

// ============================================================================
// DATED SWAPS
// ============================================================================

// Market conventions of a vanilla fixed-float swap (the deposits of a dated
// curve bootstrap use the floating leg's), e.g. EUR: TARGET, 2 days, fixed 1Y
// 30/360 modified following, float 6M ACT/360 modified following
SwapConventions :: struct {
	calendar:         Calendar,
	settlement_days:  int,
	fixed_tenor:      Period,
	fixed_convention: BusinessDayConvention,
	fixed_day_count:  DayCountConvention,
	float_tenor:      Period,
	float_convention: BusinessDayConvention,
	float_day_count:  DayCountConvention,
	end_of_month:     bool,
}

// Swap on real dates. Coupons are paid on the (adjusted) accrual end dates; the
// floating coupons are projected over their accrual periods on the same curve
// (no fixing lag); times are ACT/365F from the curve's reference_date.
DatedSwap :: struct {
	notional:        f64,
	direction:       SwapDirection, // Payer = pay fixed
	fixed_rate:      f64,
	fixed_schedule:  Schedule,
	fixed_day_count: DayCountConvention,
	float_spread:    f64,
	float_schedule:  Schedule,
	float_day_count: DayCountConvention,
}

// Spot date: settlement_days business days after the trade date (moved to a
// business day first)
swap_spot_date :: proc(conv: SwapConventions, trade_date: w.Date) -> w.Date {
	return calendar_advance(conv.calendar, calendar_adjust(conv.calendar, trade_date), conv.settlement_days, .Days)
}

// Maturity of a swap starting on `start`: start + tenor, unadjusted (the
// schedules adjust it), or the end-of-month advance under the end-of-month rule
swap_maturity_date :: proc(conv: SwapConventions, start: w.Date, tenor: Period) -> w.Date {
	if conv.end_of_month {
		return calendar_advance_period(conv.calendar, start, tenor, .ModifiedFollowing, true)
	}
	return date_add_period(start, tenor)
}

// Swap from `start` (usually swap_spot_date) for `tenor` under the conventions;
// free with free_dated_swap
make_dated_swap :: proc(
	conv: SwapConventions,
	start: w.Date,
	tenor: Period,
	fixed_rate: f64,
	notional: f64 = 1.0,
	direction: SwapDirection = .Payer,
	float_spread: f64 = 0.0,
	allocator := context.allocator,
) -> DatedSwap {
	end := swap_maturity_date(conv, start, tenor)
	fixed := make_schedule(
		start,
		end,
		conv.fixed_tenor,
		conv.calendar,
		conv.fixed_convention,
		conv.fixed_convention,
		end_of_month = conv.end_of_month,
		allocator = allocator,
	)
	float := make_schedule(
		start,
		end,
		conv.float_tenor,
		conv.calendar,
		conv.float_convention,
		conv.float_convention,
		end_of_month = conv.end_of_month,
		allocator = allocator,
	)
	return DatedSwap {
		notional = notional,
		direction = direction,
		fixed_rate = fixed_rate,
		fixed_schedule = fixed,
		fixed_day_count = conv.fixed_day_count,
		float_spread = float_spread,
		float_schedule = float,
		float_day_count = conv.float_day_count,
	}
}

free_dated_swap :: proc(swap: ^DatedSwap, allocator := context.allocator) {
	free_schedule(&swap.fixed_schedule, allocator)
	free_schedule(&swap.float_schedule, allocator)
}

@(private = "file")
_dated_periods :: proc(s: Schedule, dc: DayCountConvention, reference: w.Date, allocator: mem.Allocator) -> []_SwapPeriod {
	periods := make([]_SwapPeriod, max(len(s.dates) - 1, 0), allocator)
	for k in 0 ..< len(periods) {
		rs, re := schedule_reference_period(s, k)
		periods[k] = {
			date_to_time(reference, s.dates[k]),
			date_to_time(reference, s.dates[k + 1]),
			year_fraction(dc, s.dates[k], s.dates[k + 1], rs, re, s.dates[len(s.dates) - 1]),
		}
	}
	return periods
}

// Leg values, NPV for the direction, par rate, PV01 and fixed-leg duration /
// convexity as price_swap, on a curve with a reference_date
price_dated_swap :: proc(swap: DatedSwap, curve: ^YieldCurve, allocator := context.allocator) -> SwapValuationResult {
	fixed := _dated_periods(swap.fixed_schedule, swap.fixed_day_count, curve.reference_date, allocator)
	defer delete(fixed, allocator)
	float := _dated_periods(swap.float_schedule, swap.float_day_count, curve.reference_date, allocator)
	defer delete(float, allocator)
	return _swap_result(fixed, float, swap.notional, swap.fixed_rate, swap.float_spread, true, swap.direction == .Payer, curve)
}

// ============================================================================
// SWAP ANNUITY & METRICS (CURVE-AWARE)
// ============================================================================

_compute_swap_annuity :: proc(
	payment_frequency: f64,
	effective_date: f64,
	maturity: f64,
	day_count: DayCountConvention,
	curve: ^YieldCurve,
) -> f64 {
	dates := _swap_schedule(effective_date, maturity, payment_frequency, context.temp_allocator)
	defer delete(dates, context.temp_allocator)
	annuity := 0.0
	for k in 1 ..< len(dates) {
		annuity +=
			day_count_fraction(dates[k - 1], dates[k], day_count) *
			yield_curve_discount_factor(curve, dates[k])
	}
	return annuity
}

// Par rate (1 − DF(T)) / A of a spot-starting swap
compute_par_swap_rate :: proc(
	maturity: f64,
	curve: ^YieldCurve,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	df_maturity := yield_curve_discount_factor(curve, maturity)
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, curve)

	if annuity > 1e-10 {
		return (1.0 - df_maturity) / annuity
	}

	return 0.0
}

compute_swap_pv01 :: proc(
	maturity: f64,
	notional: f64,
	curve: ^YieldCurve,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, curve)
	return annuity * notional * 0.0001
}

// Duration of the fixed side of a spot-starting par swap seen as a par bond
// (coupons at the par rate plus the notional at maturity, worth 1):
//   D = S Σ t_k dcf_k DF(t_k) + T DF(T),  the parallel-zero-shift duration
compute_swap_duration :: proc(
	maturity: f64,
	curve: ^YieldCurve,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	S := compute_par_swap_rate(maturity, curve, payment_frequency, day_count)
	dates := _swap_schedule(0.0, maturity, payment_frequency, context.temp_allocator)
	defer delete(dates, context.temp_allocator)
	d := maturity * yield_curve_discount_factor(curve, maturity)
	for k in 1 ..< len(dates) {
		d +=
			S *
			dates[k] *
			day_count_fraction(dates[k - 1], dates[k], day_count) *
			yield_curve_discount_factor(curve, dates[k])
	}
	return d
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

create_payer_swap :: proc(
	notional: f64,
	fixed_rate: f64,
	maturity: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> InterestRateSwap {
	return InterestRateSwap {
		notional = notional,
		effective_date = 0.0,
		maturity = maturity,
		payer_leg = SwapLeg {
			leg_type = .Fixed,
			fixed_rate = fixed_rate,
			floating_spread = 0.0,
			payment_frequency = payment_frequency,
			day_count = day_count,
		},
		receiver_leg = SwapLeg {
			leg_type = .Floating,
			fixed_rate = 0.0,
			floating_spread = 0.0,
			payment_frequency = payment_frequency,
			day_count = day_count,
		},
		direction = .Payer,
	}
}

create_receiver_swap :: proc(
	notional: f64,
	fixed_rate: f64,
	maturity: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> InterestRateSwap {
	return InterestRateSwap {
		notional = notional,
		effective_date = 0.0,
		maturity = maturity,
		payer_leg = SwapLeg {
			leg_type = .Floating,
			fixed_rate = 0.0,
			floating_spread = 0.0,
			payment_frequency = payment_frequency,
			day_count = day_count,
		},
		receiver_leg = SwapLeg {
			leg_type = .Fixed,
			fixed_rate = fixed_rate,
			floating_spread = 0.0,
			payment_frequency = payment_frequency,
			day_count = day_count,
		},
		direction = .Receiver,
	}
}
