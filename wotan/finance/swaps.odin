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

SwapValuationResult :: struct {
	pv_fixed_leg:      f64,
	pv_floating_leg:   f64,
	npv:               f64, // Net present value (receiver - payer)
	par_swap_rate:     f64, // Fixed rate that makes NPV = 0
	pv01:              f64, // PV of 1bp shift (in currency units)
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

day_count_fraction :: proc(t1: f64, t2: f64, convention: DayCountConvention) -> f64 {
	dt := t2 - t1
	if dt <= 0.0 {
		return 0.0
	}

	switch convention {
	case .ACT_360:
		return dt * 365.0 / 360.0
	case .ACT_365:
		return dt
	case .ACT_ACT:
		return dt
	case .THIRTY_360:
		months := dt * 12.0
		return months / 12.0
	}

	return dt
}

// ============================================================================
// SWAP LEG VALUATION (CURVE-AWARE)
// ============================================================================

// Value a fixed leg using the yield curve
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
) {
	pv = 0.0
	n_payments := int((maturity - effective_date) / leg.payment_frequency)

	cashflows = make([]SwapCashFlow, n_payments, allocator)

	for i in 0 ..< n_payments {
		t_start := effective_date + f64(i) * leg.payment_frequency
		t_end := t_start + leg.payment_frequency
		if t_end > maturity {
			t_end = maturity
		}

		dcf := day_count_fraction(t_start, t_end, leg.day_count)
		payment := notional * leg.fixed_rate * dcf

		cashflows[i] = SwapCashFlow {
			payment_date = t_end,
			amount       = payment,
		}

		// ✅ UPGRADE: Use curve discount factor instead of flat rate
		df := yield_curve_discount_factor(curve, t_end)
		pv += payment * df
	}

	return pv, cashflows
}

// Value a floating leg using the yield curve
// ✅ UPGRADE: Uses the exact no-arbitrage identity: PV_float = N * (DF_start - DF_end)
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
	pv = 0.0
	n_payments := int((maturity - effective_date) / leg.payment_frequency)

	cashflows = make([]SwapCashFlow, n_payments, allocator)

	for i in 0 ..< n_payments {
		t_start := effective_date + f64(i) * leg.payment_frequency
		t_end := t_start + leg.payment_frequency
		if t_end > maturity {
			t_end = maturity
		}

		dcf := day_count_fraction(t_start, t_end, leg.day_count)

		// ✅ UPGRADE: Exact no-arbitrage PV of floating leg
		df_start := yield_curve_discount_factor(curve, t_start)
		df_end := yield_curve_discount_factor(curve, t_end)

		// PV of the floating rate payment (without spread)
		pv += notional * (df_start - df_end)

		// PV of the floating spread payment
		pv += notional * leg.floating_spread * dcf * df_end

		// For reporting: calculate the implied forward rate
		fwd_rate := 0.0
		if dcf > 1e-10 {
			fwd_rate = (df_start / df_end - 1.0) / dcf
		}
		payment := notional * (fwd_rate + leg.floating_spread) * dcf

		cashflows[i] = SwapCashFlow {
			payment_date = t_end,
			amount       = payment,
		}
	}

	return pv, cashflows
}

// ============================================================================
// INTERNAL PRICING HELPER
// ============================================================================

_price_swap_internal :: proc(
	swap: InterestRateSwap,
	curve: ^YieldCurve,
	allocator: mem.Allocator,
) -> SwapValuationResult {
	// Value payer leg
	pv_payer: f64
	cashflows_payer: []SwapCashFlow
	if swap.payer_leg.leg_type == .Fixed {
		pv_payer, cashflows_payer = _valuate_fixed_leg(
			swap.payer_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			curve,
			allocator,
		)
	} else {
		pv_payer, cashflows_payer = _valuate_floating_leg(
			swap.payer_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			curve,
			allocator,
		)
	}
	defer delete(cashflows_payer, allocator)

	// Value receiver leg
	pv_receiver: f64
	cashflows_receiver: []SwapCashFlow
	if swap.receiver_leg.leg_type == .Fixed {
		pv_receiver, cashflows_receiver = _valuate_fixed_leg(
			swap.receiver_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			curve,
			allocator,
		)
	} else {
		pv_receiver, cashflows_receiver = _valuate_floating_leg(
			swap.receiver_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			curve,
			allocator,
		)
	}
	defer delete(cashflows_receiver, allocator)

	// NPV = Receiver - Payer
	npv := pv_receiver - pv_payer

	// Par swap rate
	annuity := _compute_swap_annuity(
		swap.receiver_leg.payment_frequency,
		swap.effective_date,
		swap.maturity,
		swap.receiver_leg.day_count,
		curve,
	)

	par_swap_rate := 0.0
	if annuity > 1e-10 {
		if swap.receiver_leg.leg_type == .Floating {
			par_swap_rate = pv_receiver / (swap.notional * annuity)
		} else {
			par_swap_rate = pv_payer / (swap.notional * annuity)
		}
	}

	// PV01 (per 1bp shift)
	pv01 := annuity * swap.notional * 0.0001

	// Modified Duration of the fixed leg
	mac_duration := 0.0
	pv_fixed_unit := 0.0

	fixed_leg: SwapLeg
	if swap.payer_leg.leg_type == .Fixed {
		fixed_leg = swap.payer_leg
	} else {
		fixed_leg = swap.receiver_leg
	}

	n_payments := int((swap.maturity - swap.effective_date) / fixed_leg.payment_frequency)
	for i in 0 ..< n_payments {
		t_start := swap.effective_date + f64(i) * fixed_leg.payment_frequency
		t_end := t_start + fixed_leg.payment_frequency
		if t_end > swap.maturity {
			t_end = swap.maturity
		}
		dcf := day_count_fraction(t_start, t_end, fixed_leg.day_count)
		df := yield_curve_discount_factor(curve, t_end)

		pv_cf := dcf * df
		pv_fixed_unit += pv_cf
		mac_duration += t_end * pv_cf
	}

	modified_duration := 0.0
	if pv_fixed_unit > 1e-10 {
		mac_duration /= pv_fixed_unit
		// Approximate modified duration
		modified_duration = mac_duration / (1.0 + par_swap_rate * fixed_leg.payment_frequency)
	}

	return SwapValuationResult {
		pv_fixed_leg      = 0.0, // Can be populated if needed
		pv_floating_leg   = 0.0,
		npv               = npv,
		par_swap_rate     = par_swap_rate,
		pv01              = pv01,
		modified_duration = modified_duration,
		convexity         = 0.0, // Can be added via finite differences if needed
	}
}

// ============================================================================
// MAIN SWAP PRICING FUNCTION
// ============================================================================

price_swap :: proc(
	swap: InterestRateSwap,
	curve: ^YieldCurve,
	allocator: mem.Allocator = context.allocator,
) -> SwapValuationResult {
	return _price_swap_internal(swap, curve, allocator)
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
	annuity := 0.0
	n_payments := int((maturity - effective_date) / payment_frequency)

	for i in 0 ..< n_payments {
		t_start := effective_date + f64(i) * payment_frequency
		t_end := t_start + payment_frequency
		if t_end > maturity {
			t_end = maturity
		}

		dcf := day_count_fraction(t_start, t_end, day_count)
		df := yield_curve_discount_factor(curve, t_end)
		annuity += dcf * df
	}

	return annuity
}

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

compute_swap_duration :: proc(
	maturity: f64,
	curve: ^YieldCurve,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	df_maturity := yield_curve_discount_factor(curve, maturity)
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, curve)
	par_rate := compute_par_swap_rate(maturity, curve, payment_frequency, day_count)

	if par_rate * annuity > 1e-10 {
		return (1.0 - df_maturity) / (par_rate * annuity)
	}

	return maturity / 2.0
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
