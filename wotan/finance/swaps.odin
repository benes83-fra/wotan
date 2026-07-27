package finance

import "core:math"
import "core:mem"

// ============================================================================
// INTEREST RATE SWAPS (IRS)
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
// ZERO COUPON BOND PRICING (FLAT CURVE)
// ============================================================================

_discount_factor_flat :: proc(t: f64, r: f64) -> f64 {
	if t <= 0.0 {
		return 1.0
	}
	return math.exp_f64(-r * t)
}

// ============================================================================
// SWAP LEG VALUATION
// ============================================================================

_valuate_fixed_leg :: proc(
	leg: SwapLeg,
	notional: f64,
	effective_date: f64,
	maturity: f64,
	r: f64,
) -> (
	pv: f64,
	cashflows: []SwapCashFlow,
) {
	pv = 0.0
	n_payments := int((maturity - effective_date) / leg.payment_frequency)

	cashflows = make([]SwapCashFlow, n_payments, context.allocator)

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

		df := _discount_factor_flat(t_end, r)
		pv += payment * df
	}

	return pv, cashflows
}

_valuate_floating_leg :: proc(
	leg: SwapLeg,
	notional: f64,
	effective_date: f64,
	maturity: f64,
	r: f64,
) -> (
	pv: f64,
	cashflows: []SwapCashFlow,
) {
	pv = 0.0
	n_payments := int((maturity - effective_date) / leg.payment_frequency)

	cashflows = make([]SwapCashFlow, n_payments, context.allocator)

	for i in 0 ..< n_payments {
		t_start := effective_date + f64(i) * leg.payment_frequency
		t_end := t_start + leg.payment_frequency
		if t_end > maturity {
			t_end = maturity
		}

		dcf := day_count_fraction(t_start, t_end, leg.day_count)
		fwd_rate := r + leg.floating_spread
		payment := notional * fwd_rate * dcf

		cashflows[i] = SwapCashFlow {
			payment_date = t_end,
			amount       = payment,
		}

		df := _discount_factor_flat(t_end, r)
		pv += payment * df
	}

	return pv, cashflows
}
// ============================================================================
// INTERNAL PRICING HELPER (No Greeks - prevents infinite recursion)
// ============================================================================
_price_swap_internal :: proc(
	swap: InterestRateSwap,
	r: f64,
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
			r,
		)
	} else {
		pv_payer, cashflows_payer = _valuate_floating_leg(
			swap.payer_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			r,
		)
	}
	defer delete(cashflows_payer, context.allocator)

	// Value receiver leg
	pv_receiver: f64
	cashflows_receiver: []SwapCashFlow
	if swap.receiver_leg.leg_type == .Fixed {
		pv_receiver, cashflows_receiver = _valuate_fixed_leg(
			swap.receiver_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			r,
		)
	} else {
		pv_receiver, cashflows_receiver = _valuate_floating_leg(
			swap.receiver_leg,
			swap.notional,
			swap.effective_date,
			swap.maturity,
			r,
		)
	}
	defer delete(cashflows_receiver, context.allocator)

	// NPV = Receiver - Payer
	npv := pv_receiver - pv_payer

	// Par swap rate (per unit notional)
	annuity := _compute_swap_annuity(
		swap.receiver_leg.payment_frequency,
		swap.effective_date,
		swap.maturity,
		swap.receiver_leg.day_count,
		r,
	)

	par_swap_rate := 0.0
	if annuity > 1e-10 {
		// ✅ FIX 1: Divide dollar PV by notional to get unit PV before dividing by annuity
		if swap.receiver_leg.leg_type == .Floating {
			par_swap_rate = (pv_receiver / swap.notional) / annuity
		} else {
			par_swap_rate = (pv_payer / swap.notional) / annuity
		}
	}

	// PV01
	pv01 := annuity * swap.notional * 0.0001

	// ✅ FIX 2: Calculate Modified Duration of the FIXED LEG (Standard Swap Metric)
	// Traders don't care about the duration of the NPV; they care about the fixed leg's duration.
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
		df := _discount_factor_flat(t_end, r)

		pv_cf := dcf * df
		pv_fixed_unit += pv_cf
		mac_duration += t_end * pv_cf
	}

	modified_duration := 0.0
	if pv_fixed_unit > 1e-10 {
		mac_duration /= pv_fixed_unit
		// Modified Duration ≈ Macaulay Duration / (1 + r * freq)
		modified_duration = mac_duration / (1.0 + r * fixed_leg.payment_frequency)
	}

	return SwapValuationResult {
		pv_fixed_leg = 0.0,
		pv_floating_leg = 0.0,
		npv = npv,
		par_swap_rate = par_swap_rate,
		pv01 = pv01,
		modified_duration = modified_duration,
		convexity = 0.0,
	}
}

// ============================================================================
// MAIN SWAP PRICING FUNCTION (with Greeks via finite differences)
// ============================================================================
price_swap :: proc(
	swap: InterestRateSwap,
	r: f64,
	allocator: mem.Allocator = context.allocator,
) -> SwapValuationResult {
	// Get base price using internal helper (which now includes correct duration)
	result := _price_swap_internal(swap, r, allocator)
	npv := result.npv

	// Compute Convexity using finite differences on NPV
	h := 0.0001 // 1bp shift
	price_up := _price_swap_internal(swap, r + h, allocator)
	price_dn := _price_swap_internal(swap, r - h, allocator)

	convexity := 0.0
	if math.abs(npv) > 1e-10 {
		convexity = (price_up.npv + price_dn.npv - 2.0 * npv) / (h * h * npv)
	}

	return SwapValuationResult {
		pv_fixed_leg = result.pv_fixed_leg,
		pv_floating_leg = result.pv_floating_leg,
		npv = npv,
		par_swap_rate = result.par_swap_rate,
		pv01 = result.pv01,
		modified_duration = result.modified_duration,
		convexity = convexity,
	}
}

// ============================================================================
// SWAP ANNUITY (PV01 per unit notional)
// ============================================================================

_compute_swap_annuity :: proc(
	payment_frequency: f64,
	effective_date: f64,
	maturity: f64,
	day_count: DayCountConvention,
	r: f64,
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
		df := _discount_factor_flat(t_end, r)
		annuity += dcf * df
	}

	return annuity
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

compute_par_swap_rate :: proc(
	maturity: f64,
	r: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	df_maturity := _discount_factor_flat(maturity, r)
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, r)

	if annuity > 1e-10 {
		return (1.0 - df_maturity) / annuity
	}

	return r
}

// ============================================================================
// SWAP RISK METRICS
// ============================================================================

compute_swap_pv01 :: proc(
	maturity: f64,
	notional: f64,
	r: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, r)
	return annuity * notional * 0.0001
}

compute_swap_duration :: proc(
	maturity: f64,
	r: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	df_maturity := _discount_factor_flat(maturity, r)
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, r)

	if r * annuity > 1e-10 {
		return (1.0 - df_maturity) / (r * annuity)
	}

	return maturity / 2.0
}
