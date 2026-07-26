package finance

import "core:math"
import "core:mem"

// ============================================================================
// INTEREST RATE SWAPS (IRS)
// ============================================================================
// The workhorse of fixed income. An IRS exchanges fixed-rate cash flows for
// floating-rate cash flows (or vice versa) on a notional amount.
//
// Key concepts:
// - Fixed Leg: Pays/receives fixed rate × day count fraction × notional
// - Floating Leg: Pays/receives floating rate (e.g., SOFR, EURIBOR) × DCF × notional
// - Par Swap Rate: The fixed rate that makes the swap worth zero at inception
// - PV01 (DV01): Present value of a 1 basis point parallel shift
// - Swap Duration/Convexity: Risk metrics

// ============================================================================
// ENUMS AND STRUCTURES
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

// Compute day count fraction between two dates (in years)
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
		// Simplified 30/360: assume 30 days per month, 360 days per year
		months := dt * 12.0
		return months / 12.0
	}

	return dt
}

// ============================================================================
// ZERO COUPON BOND PRICING (FLAT CURVE)
// ============================================================================

// Simple flat curve discount factor
// In production, you'd use a bootstrapped curve (Nelson-Siegel, cubic spline, etc.)
_discount_factor_flat :: proc(t: f64, r: f64) -> f64 {
	if t <= 0.0 {
		return 1.0
	}
	return math.exp_f64(-r * t)
}

// ============================================================================
// SWAP LEG VALUATION
// ============================================================================

// Value a fixed leg
_valuate_fixed_leg :: proc(
	leg: SwapLeg,
	notional: f64,
	effective_date: f64,
	maturity: f64,
	r: f64, // Flat rate for discounting
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

// Value a floating leg (assuming flat forward curve)
// Under a flat curve, the floating leg is worth: notional × (DF_start - DF_end)
// This is a well-known result from swap pricing theory.
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

	// Under a flat curve, forward rate = spot rate
	// Floating payment = notional × r × DCF
	for i in 0 ..< n_payments {
		t_start := effective_date + f64(i) * leg.payment_frequency
		t_end := t_start + leg.payment_frequency
		if t_end > maturity {
			t_end = maturity
		}

		dcf := day_count_fraction(t_start, t_end, leg.day_count)
		fwd_rate := r + leg.floating_spread // Forward rate + spread
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
// MAIN SWAP PRICING FUNCTION
// ============================================================================

// Price an interest rate swap
price_swap :: proc(
	swap: InterestRateSwap,
	r: f64, // Flat discount rate
	allocator: mem.Allocator = context.allocator,
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

	// Par swap rate: the fixed rate that makes NPV = 0
	// For a standard payer swap (pay fixed, receive floating):
	// PV_fixed = PV_floating at par
	// fixed_rate × annuity = PV_floating
	// fixed_rate = PV_floating / annuity
	annuity := _compute_swap_annuity(
		swap.receiver_leg.payment_frequency,
		swap.effective_date,
		swap.maturity,
		swap.receiver_leg.day_count,
		r,
	)

	par_swap_rate := 0.0
	if annuity > 1e-10 {
		// If receiver is floating, par rate = PV_floating / annuity
		if swap.receiver_leg.leg_type == .Floating {
			par_swap_rate = pv_receiver / annuity
		} else {
			// If payer is floating, par rate = PV_floating / annuity
			par_swap_rate = pv_payer / annuity
		}
	}

	// PV01: PV of a 1bp parallel shift
	// = annuity × notional × 0.0001
	pv01 := annuity * swap.notional * 0.0001

	// Modified duration: -dNPV/dr / NPV (approximation)
	// For a swap, duration ≈ (duration_receiver - duration_payer)
	h := 0.0001 // 1bp shift
	price_up := price_swap(swap, r + h, allocator)
	price_dn := price_swap(swap, r - h, allocator)

	modified_duration := 0.0
	if math.abs(npv) > 1e-10 {
		modified_duration = -(price_up.npv - price_dn.npv) / (2.0 * h * npv)
	}

	// Convexity
	convexity := 0.0
	if math.abs(npv) > 1e-10 {
		convexity = (price_up.npv + price_dn.npv - 2.0 * npv) / (h * h * npv)
	}

	return SwapValuationResult {
		pv_fixed_leg      = 0.0, // Would need to track which leg is fixed
		pv_floating_leg   = 0.0,
		npv               = npv,
		par_swap_rate     = par_swap_rate,
		pv01              = pv01,
		modified_duration = modified_duration,
		convexity         = convexity,
	}
}

// ============================================================================
// SWAP ANNUITY (PV01 per unit notional)
// ============================================================================

// Compute the swap annuity (sum of discounted day count fractions)
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

// Create a standard payer swap (pay fixed, receive floating)
create_payer_swap :: proc(
	notional: f64,
	fixed_rate: f64,
	maturity: f64,
	payment_frequency: f64 = 0.25, // Quarterly
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

// Create a standard receiver swap (receive fixed, pay floating)
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

// Compute the par swap rate for a given maturity and curve
compute_par_swap_rate :: proc(
	maturity: f64,
	r: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	// Par swap rate = (1 - DF_maturity) / annuity
	// This is the standard formula for a swap starting today
	df_maturity := _discount_factor_flat(maturity, r)
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, r)

	if annuity > 1e-10 {
		return (1.0 - df_maturity) / annuity
	}

	return r // Fallback to flat rate
}

// ============================================================================
// SWAP RISK METRICS
// ============================================================================

// Compute PV01 (DV01) for a swap
compute_swap_pv01 :: proc(
	maturity: f64,
	notional: f64,
	r: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, r)
	return annuity * notional * 0.0001 // 1bp = 0.0001
}

// Compute swap duration (modified duration)
compute_swap_duration :: proc(
	maturity: f64,
	r: f64,
	payment_frequency: f64 = 0.25,
	day_count: DayCountConvention = .ACT_360,
) -> f64 {
	// For a par swap, duration ≈ (1 - DF_maturity) / (r × annuity)
	// This is a simplified approximation
	df_maturity := _discount_factor_flat(maturity, r)
	annuity := _compute_swap_annuity(payment_frequency, 0.0, maturity, day_count, r)

	if r * annuity > 1e-10 {
		return (1.0 - df_maturity) / (r * annuity)
	}

	return maturity / 2.0 // Rough approximation
}
