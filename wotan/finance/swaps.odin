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
	case .ACT_365,
	     .ACT_ACT,
	     .THIRTY_360,
	     .ACT_ACT_ICMA,
	     .THIRTY_360_US,
	     .THIRTY_E_360,
	     .THIRTY_E_360_ISDA:
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

// One coupon period on the curves' time axis: accrual fraction, payment time
// and (floating legs) the forward period on the projection curve with its own
// fraction. Without fixing or payment lags fwd_start/fwd_end are the accrual
// start/end and pay the accrual end.
@(private = "file")
_SwapPeriod :: struct {
	fwd_start, fwd_end: f64, // years from the curves' t = 0
	pay:                f64,
	accrual:            f64, // day-count fraction of the accrual period
	fwd_accrual:        f64, // day-count fraction of the forward period
	rate:               f64, // projected floating rate when has_rate (daily compounding)
	has_rate:           bool,
}

@(private = "file")
_time_periods :: proc(
	effective, maturity: f64,
	leg: SwapLeg,
	allocator: mem.Allocator,
) -> []_SwapPeriod {
	dates := _swap_schedule(effective, maturity, leg.payment_frequency, context.temp_allocator)
	defer delete(dates, context.temp_allocator)
	periods := make([]_SwapPeriod, max(len(dates) - 1, 0), allocator)
	for k in 1 ..< len(dates) {
		dcf := day_count_fraction(dates[k - 1], dates[k], leg.day_count)
		periods[k - 1] = {dates[k - 1], dates[k], dates[k], dcf, dcf, 0.0, false}
	}
	return periods
}

// Fixed leg: Σ N · K · dcf_k · DF(pay_k); also returns the annuity
// Σ dcf_k DF(pay_k) and Σ t·dcf·DF, Σ t²·dcf·DF for the duration and convexity
@(private = "file")
_fixed_leg_pv :: proc(
	periods: []_SwapPeriod,
	notional, rate: f64,
	discount: ^YieldCurve,
) -> (
	pv, annuity, t1, t2: f64,
) {
	for p in periods {
		df := yield_curve_discount_factor(discount, p.pay)
		annuity += p.accrual * df
		t1 += p.pay * p.accrual * df
		t2 += p.pay * p.pay * p.accrual * df
	}
	pv = notional * rate * annuity
	return
}

// Floating leg: coupon N (F + spread) dcf with the forward
// F = (P(s)/P(e) − 1) / dcf_fwd on the projection curve P, discounted at the
// payment date. With one curve and no lags each coupon is worth
// N (DF(s) − DF(e)) plus the spread; over the accrual period the same formula
// is the compounded overnight rate of an OIS leg.
@(private = "file")
_float_leg_pv :: proc(
	periods: []_SwapPeriod,
	notional, spread: f64,
	discount, forecast: ^YieldCurve,
) -> (
	pv: f64,
) {
	for p in periods {
		growth :=
			yield_curve_discount_factor(forecast, p.fwd_start) /
			yield_curve_discount_factor(forecast, p.fwd_end)
		coupon := (growth - 1.0) * p.accrual / p.fwd_accrual if p.fwd_accrual > 0.0 else 0.0
		if p.has_rate {coupon = p.rate * p.accrual}
		pv +=
			notional * (coupon + spread * p.accrual) * yield_curve_discount_factor(discount, p.pay)
	}
	return
}

// Values a fixed leg against another leg (floating, or fixed at other_rate)
@(private = "file")
_swap_result :: proc(
	fixed, other: []_SwapPeriod,
	notional, fixed_rate, other_rate: f64,
	other_is_floating, fixed_is_payer: bool,
	discount, forecast: ^YieldCurve,
) -> SwapValuationResult {
	pv_fixed, annuity, t1, t2 := _fixed_leg_pv(fixed, notional, fixed_rate, discount)
	pv_other: f64
	if other_is_floating {
		pv_other = _float_leg_pv(other, notional, other_rate, discount, forecast)
	} else {
		pv_other, _, _, _ = _fixed_leg_pv(other, notional, other_rate, discount)
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
		curve,
	)
}

// ============================================================================
// DATED SWAPS
// ============================================================================

// Market conventions of a vanilla fixed-float or OIS swap (the deposits of a
// dated curve bootstrap use the floating leg's calendar, convention and day
// count). Examples:
//   EUR 6M Euribor: TARGET, 2 days, fixed 1Y 30/360, float 6M ACT/360, both
//                   modified following, fixing_days 2, indexed_coupons
//   €STR OIS:       TARGET, 2 days, fixed and float 1Y ACT/360, payment_lag 1
SwapConventions :: struct {
	calendar:          Calendar,
	settlement_days:   int,
	fixed_tenor:       Period,
	fixed_convention:  BusinessDayConvention,
	fixed_day_count:   DayCountConvention,
	float_tenor:       Period, // also the index tenor of indexed coupons
	float_convention:  BusinessDayConvention,
	float_day_count:   DayCountConvention,
	end_of_month:      bool,
	payment_lag:       int, // business days from accrual end to payment (both legs)
	fixing_days:       int, // index fixing lag in business days (indexed coupons)
	// Projection period of a floating coupon: false = its accrual period (par
	// coupons; the compounded overnight rate of an OIS leg); true = the index
	// period from the value date fixing_days after the fixing, which is
	// fixing_days before the accrual start (QuantLib's indexed Ibor coupons)
	indexed_coupons:   bool,
	// Overnight legs (indexed_coupons false), compounded daily over the
	// business days of the accrual period with the float day count, as
	// QuantLib's OvernightIndexedCoupon: each day's rate is observed
	// lookback_days business days earlier; with observation_shift the day
	// weights (and the year fraction dividing the compounded growth) follow the
	// observation dates instead of the interest dates; the last lockout_days
	// rates repeat the one observed lockout_days days before the end.
	lookback_days:     int,
	lockout_days:      int,
	observation_shift: bool,
}

// Swap on real dates; times are ACT/365F from the curves' reference_date
DatedSwap :: struct {
	notional:       f64,
	direction:      SwapDirection, // Payer = pay fixed
	fixed_rate:     f64,
	float_spread:   f64,
	fixed_schedule: Schedule,
	float_schedule: Schedule,
	conventions:    SwapConventions,
}

// Spot date: settlement_days business days after the trade date (moved to a
// business day first)
swap_spot_date :: proc(conv: SwapConventions, trade_date: w.Date) -> w.Date {
	return calendar_advance(
		conv.calendar,
		calendar_adjust(conv.calendar, trade_date),
		conv.settlement_days,
		.Days,
	)
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
	return DatedSwap{notional, direction, fixed_rate, float_spread, fixed, float, conv}
}

free_dated_swap :: proc(swap: ^DatedSwap, allocator := context.allocator) {
	free_schedule(&swap.fixed_schedule, allocator)
	free_schedule(&swap.float_schedule, allocator)
}

// Payment date of an accrual period ending on `end`
@(private = "file")
_swap_payment_date :: proc(conv: SwapConventions, end: w.Date) -> w.Date {
	return calendar_advance(conv.calendar, end, conv.payment_lag, .Days)
}

// Forward period of the floating coupon accruing from `start` to `end`
@(private = "file")
_swap_forward_period :: proc(conv: SwapConventions, start, end: w.Date) -> (w.Date, w.Date) {
	if !conv.indexed_coupons {
		if conv.observation_shift && conv.lookback_days > 0 {
			return calendar_advance(
				conv.calendar,
				start,
				-conv.lookback_days,
				.Days,
				.Preceding,
			), calendar_advance(conv.calendar, end, -conv.lookback_days, .Days, .Preceding)
		}
		return start, end
	}
	fixing := calendar_advance(conv.calendar, start, -conv.fixing_days, .Days, .Preceding)
	value := calendar_advance(conv.calendar, fixing, conv.fixing_days, .Days)
	return value, calendar_advance_period(
		conv.calendar,
		value,
		conv.float_tenor,
		conv.float_convention,
		conv.end_of_month,
	)
}

@(private = "file")
_dated_periods :: proc(
	swap: DatedSwap,
	floating: bool,
	forecast: ^YieldCurve,
	allocator: mem.Allocator,
) -> []_SwapPeriod {
	conv := swap.conventions
	reference := forecast.reference_date
	s := swap.float_schedule if floating else swap.fixed_schedule
	dc := conv.float_day_count if floating else conv.fixed_day_count
	last := s.dates[len(s.dates) - 1]
	// overnight compounding that does not telescope into one forward period
	daily :=
		floating &&
		!conv.indexed_coupons &&
		(conv.lockout_days > 0 || (conv.lookback_days > 0 && !conv.observation_shift))
	periods := make([]_SwapPeriod, max(len(s.dates) - 1, 0), allocator)
	for k in 0 ..< len(periods) {
		start, end := s.dates[k], s.dates[k + 1]
		rs, re := schedule_reference_period(s, k)
		fs, fe := start, end
		if floating {fs, fe = _swap_forward_period(conv, start, end)}
		periods[k] = {
			fwd_start   = date_to_time(reference, fs),
			fwd_end     = date_to_time(reference, fe),
			pay         = date_to_time(reference, _swap_payment_date(conv, end)),
			accrual     = year_fraction(dc, start, end, rs, re, last),
			fwd_accrual = year_fraction(dc, fs, fe, rs, re, last) if floating else 0.0,
		}
		if daily {
			periods[k].rate = _overnight_rate(conv, start, end, forecast)
			periods[k].has_rate = true
		}
	}
	return periods
}

// Compounded overnight rate of the accrual period [start, end] projected on
// the curve: Π (1 + r_i τ_i) − 1 over the business days, divided by the year
// fraction spanned by the weight dates (see SwapConventions)
@(private = "file")
_overnight_rate :: proc(conv: SwapConventions, start, end: w.Date, curve: ^YieldCurve) -> f64 {
	cal, dc := conv.calendar, conv.float_day_count
	interest := make([dynamic]w.Date, 0, 400, context.temp_allocator)
	defer delete(interest)
	for d := start;
	    !w.date_less(end, d);
	    d = calendar_advance(cal, d, 1, .Days) {append(&interest, d)}
	n := len(interest) - 1
	if n < 1 {return 0.0}
	observed := make([]w.Date, n + 1, context.temp_allocator)
	defer delete(observed, context.temp_allocator)
	for d, i in interest {observed[i] = calendar_advance(cal, d, -conv.lookback_days, .Days, .Preceding)}
	weights := observed if conv.observation_shift else interest[:]
	lock := n - 1 - conv.lockout_days if conv.lockout_days > 0 && conv.lockout_days < n else n
	growth := 1.0
	for i in 0 ..< n {
		f := observed[min(i, lock)]
		next := calendar_advance(cal, f, 1, .Days)
		r :=
			(yield_curve_discount_factor_date(curve, f) /
					yield_curve_discount_factor_date(curve, next) -
				1.0) /
			year_fraction(dc, f, next)
		growth *= 1.0 + r * year_fraction(dc, weights[i], weights[i + 1])
	}
	return (growth - 1.0) / year_fraction(dc, weights[0], weights[n])
}

// Last date the swap depends on: final payment or end of the last forward period
dated_swap_last_relevant_date :: proc(swap: DatedSwap) -> w.Date {
	conv := swap.conventions
	fs, ff := swap.fixed_schedule.dates, swap.float_schedule.dates
	last := _swap_payment_date(conv, fs[len(fs) - 1])
	pay := _swap_payment_date(conv, ff[len(ff) - 1])
	if w.date_less(last, pay) {last = pay}
	_, fwd_end := _swap_forward_period(conv, ff[len(ff) - 2], ff[len(ff) - 1])
	if w.date_less(last, fwd_end) {last = fwd_end}
	return last
}

// Leg values, NPV for the direction, par rate, PV01 and fixed-leg duration /
// convexity as price_swap. Cash flows are discounted on `discount`; floating
// coupons are projected on `forecast` (the discount curve when nil). Both
// curves must share the reference_date.
price_dated_swap :: proc(
	swap: DatedSwap,
	discount: ^YieldCurve,
	forecast: ^YieldCurve = nil,
	allocator := context.allocator,
) -> SwapValuationResult {
	fc := forecast if forecast != nil else discount
	fixed := _dated_periods(swap, false, fc, allocator)
	defer delete(fixed, allocator)
	float := _dated_periods(swap, true, fc, allocator)
	defer delete(float, allocator)
	return _swap_result(
		fixed,
		float,
		swap.notional,
		swap.fixed_rate,
		swap.float_spread,
		true,
		swap.direction == .Payer,
		discount,
		forecast if forecast != nil else discount,
	)
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
