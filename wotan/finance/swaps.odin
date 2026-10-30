package finance

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
	THIRTY_360, // 30/360 (US corporate bonds)
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

SwapCashFlow :: struct {
	payment_date: f64, // Years from today
	amount:       f64, // Cash flow amount
}

// ============================================================================
// DAY COUNT FRACTION CALCULATIONS
// ============================================================================

// Times are ACT/365 year fractions, so ACT/360 scales them by 365/360 and the
// other conventions use them as they are
day_count_fraction :: proc(t1: f64, t2: f64, convention: DayCountConvention) -> f64 {
	dt := t2 - t1
	if dt <= 0.0 {
		return 0.0
	}

	switch convention {
	case .ACT_360:
		return dt * 365.0 / 360.0
	case .ACT_365, .ACT_ACT, .THIRTY_360:
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

// Fixed leg: Σ N · K · dcf_k · DF(d_k); also returns the annuity Σ dcf_k DF(d_k)
// and Σ t·dcf·DF, Σ t²·dcf·DF for the duration and convexity
@(private = "file")
_valuate_fixed_leg :: proc(
	leg: SwapLeg,
	notional: f64,
	effective_date: f64,
	maturity: f64,
	curve: ^YieldCurve,
	allocator: mem.Allocator,
) -> (
	pv: f64,
	cashflows: []SwapCashFlow,
	annuity, t1, t2: f64,
) {
	dates := _swap_schedule(
		effective_date,
		maturity,
		leg.payment_frequency,
		context.temp_allocator,
	)
	defer delete(dates, context.temp_allocator)
	cashflows = make([]SwapCashFlow, max(len(dates) - 1, 0), allocator)
	for k in 1 ..< len(dates) {
		dcf := day_count_fraction(dates[k - 1], dates[k], leg.day_count)
		df := yield_curve_discount_factor(curve, dates[k])
		cashflows[k - 1] = SwapCashFlow {
			payment_date = dates[k],
			amount       = notional * leg.fixed_rate * dcf,
		}
		annuity += dcf * df
		t1 += dates[k] * dcf * df
		t2 += dates[k] * dates[k] * dcf * df
	}
	pv = notional * leg.fixed_rate * annuity
	return
}

// Floating leg (single curve): each coupon's forward makes it worth
// N (DF(d_{k−1}) − DF(d_k)), plus the spread N · s · dcf_k · DF(d_k)
@(private = "file")
_valuate_floating_leg :: proc(
	leg: SwapLeg,
	notional: f64,
	effective_date: f64,
	maturity: f64,
	curve: ^YieldCurve,
	allocator: mem.Allocator,
) -> (
	pv: f64,
	cashflows: []SwapCashFlow,
) {
	dates := _swap_schedule(
		effective_date,
		maturity,
		leg.payment_frequency,
		context.temp_allocator,
	)
	defer delete(dates, context.temp_allocator)
	cashflows = make([]SwapCashFlow, max(len(dates) - 1, 0), allocator)
	for k in 1 ..< len(dates) {
		dcf := day_count_fraction(dates[k - 1], dates[k], leg.day_count)
		df_start := yield_curve_discount_factor(curve, dates[k - 1])
		df_end := yield_curve_discount_factor(curve, dates[k])
		pv += notional * (df_start - df_end + leg.floating_spread * dcf * df_end)
		fwd := (df_start / df_end - 1.0) / dcf if dcf > 1e-12 else 0.0
		cashflows[k - 1] = SwapCashFlow {
			payment_date = dates[k],
			amount       = notional * (fwd + leg.floating_spread) * dcf,
		}
	}
	return
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
	float := swap.receiver_leg if fixed_is_payer else swap.payer_leg

	pv_fixed, cf_fixed, annuity, t1, t2 := _valuate_fixed_leg(
		fixed,
		swap.notional,
		swap.effective_date,
		swap.maturity,
		curve,
		allocator,
	)
	defer delete(cf_fixed, allocator)
	pv_float: f64
	cf_float: []SwapCashFlow
	if float.leg_type == .Floating {
		pv_float, cf_float = _valuate_floating_leg(
			float,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			curve,
			allocator,
		)
	} else {
		pv_float, cf_float, _, _, _ = _valuate_fixed_leg(
			float,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			curve,
			allocator,
		)
	}
	defer delete(cf_float, allocator)

	npv := pv_float - pv_fixed if fixed_is_payer else pv_fixed - pv_float
	par := pv_float / (swap.notional * annuity) if annuity > 0.0 && swap.notional != 0.0 else 0.0
	dur, conv := 0.0, 0.0
	if annuity > 0.0 {
		dur = t1 / annuity
		conv = t2 / annuity
	}

	return SwapValuationResult {
		pv_fixed_leg = pv_fixed,
		pv_floating_leg = pv_float,
		npv = npv,
		par_swap_rate = par,
		pv01 = annuity * swap.notional * 0.0001,
		modified_duration = dur,
		convexity = conv,
	}
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
