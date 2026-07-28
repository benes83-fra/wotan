package finance

import "core:math"
import "core:mem"

// ============================================================================
// BOND STRUCTURES
// ============================================================================

BondType :: enum {
	Fixed,
	Floating,
	ZeroCoupon,
}

CouponFrequency :: enum {
	Annual,
	SemiAnnual,
	Quarterly,
	Monthly,
}

Bond :: struct {
	face_value:       f64,
	coupon_rate:      f64, // Annual coupon rate (e.g., 0.05 for 5%)
	maturity:         f64, // Years from issuance
	issue_date:       f64, // Years from today (0 = issued today)
	settlement_date:  f64, // Years from today (when bond is purchased)
	bond_type:        BondType,
	coupon_frequency: CouponFrequency,
	day_count:        DayCountConvention,
}

BondPriceResult :: struct {
	clean_price:       f64, // Price excluding accrued interest
	dirty_price:       f64, // Price including accrued interest
	accrued_interest:  f64, // Interest accrued since last coupon
	yield_to_maturity: f64, // YTM (annualized)
	modified_duration: f64,
	convexity:         f64,
}

FlatCurveParams :: struct {
	rate: f64,
}

// ✅ Idiomatic Odin callback: takes rawptr for context
flat_curve_proc :: proc(t: f64, user_data: rawptr) -> f64 {
	params := (^FlatCurveParams)(user_data)
	return math.exp_f64(-params.rate * t)
}

// ============================================================================
// COUPON SCHEDULING
// ============================================================================

get_coupon_dates :: proc(bond: Bond, allocator: mem.Allocator) -> []f64 {
	freq: f64
	switch bond.coupon_frequency {
	case .Annual:
		freq = 1.0
	case .SemiAnnual:
		freq = 2.0
	case .Quarterly:
		freq = 4.0
	case .Monthly:
		freq = 12.0
	}

	n_coupons := int(bond.maturity * freq)
	dates := make([]f64, n_coupons, allocator)

	for i in 0 ..< n_coupons {
		dates[i] = bond.issue_date + f64(i + 1) / freq
	}

	return dates
}

// ============================================================================
// ACCRUED INTEREST CALCULATION
// ============================================================================

calculate_accrued_interest :: proc(bond: Bond) -> f64 {
	if bond.bond_type == .ZeroCoupon {
		return 0.0
	}

	freq: f64
	switch bond.coupon_frequency {
	case .Annual:
		freq = 1.0
	case .SemiAnnual:
		freq = 2.0
	case .Quarterly:
		freq = 4.0
	case .Monthly:
		freq = 12.0
	}

	coupon_payment := bond.face_value * bond.coupon_rate / freq

	coupon_dates := get_coupon_dates(bond, context.temp_allocator)
	defer delete(coupon_dates, context.temp_allocator)

	last_coupon := bond.issue_date
	next_coupon := bond.issue_date + 1.0 / freq

	for i in 0 ..< len(coupon_dates) {
		if coupon_dates[i] <= bond.settlement_date {
			last_coupon = coupon_dates[i]
			if i + 1 < len(coupon_dates) {
				next_coupon = coupon_dates[i + 1]
			}
		}
	}

	days_elapsed := bond.settlement_date - last_coupon
	days_in_period := next_coupon - last_coupon

	if days_in_period <= 0.0 {
		return 0.0
	}

	fraction := days_elapsed / days_in_period
	return coupon_payment * fraction
}

// ============================================================================
// BOND PRICING FROM YIELD CURVE
// ============================================================================

// ✅ FIX: Added user_data parameter to pass context to the callback
price_bond_from_curve :: proc(
	bond: Bond,
	yield_curve: proc(t: f64, user_data: rawptr) -> f64,
	user_data: rawptr,
	allocator: mem.Allocator = context.allocator,
) -> BondPriceResult {

	coupon_dates := get_coupon_dates(bond, allocator)
	defer delete(coupon_dates, allocator)

	freq: f64
	switch bond.coupon_frequency {
	case .Annual:
		freq = 1.0
	case .SemiAnnual:
		freq = 2.0
	case .Quarterly:
		freq = 4.0
	case .Monthly:
		freq = 12.0
	}

	coupon_payment := bond.face_value * bond.coupon_rate / freq
	dirty_price := 0.0

	for i in 0 ..< len(coupon_dates) {
		t := coupon_dates[i]
		if t <= bond.settlement_date {
			continue
		}

		// ✅ FIX: Pass user_data to the yield_curve function
		df := yield_curve(t, user_data)

		if bond.bond_type != .ZeroCoupon {
			dirty_price += coupon_payment * df
		}

		if i == len(coupon_dates) - 1 {
			dirty_price += bond.face_value * df
		}
	}

	accrued_interest := calculate_accrued_interest(bond)
	clean_price := dirty_price - accrued_interest
	ytm := solve_ytm(bond, clean_price, allocator)
	mod_duration, conv := calculate_duration_convexity(bond, ytm, allocator)

	return BondPriceResult {
		clean_price = clean_price,
		dirty_price = dirty_price,
		accrued_interest = accrued_interest,
		yield_to_maturity = ytm,
		modified_duration = mod_duration,
		convexity = conv,
	}
}

// ============================================================================
// YIELD TO MATURITY SOLVER
// ============================================================================

solve_ytm :: proc(bond: Bond, clean_price: f64, allocator: mem.Allocator) -> f64 {
	ytm := bond.coupon_rate

	coupon_dates := get_coupon_dates(bond, allocator)
	defer delete(coupon_dates, allocator)

	freq: f64
	switch bond.coupon_frequency {
	case .Annual:
		freq = 1.0
	case .SemiAnnual:
		freq = 2.0
	case .Quarterly:
		freq = 4.0
	case .Monthly:
		freq = 12.0
	}

	coupon_payment := bond.face_value * bond.coupon_rate / freq

	for iter in 0 ..< 100 {
		price := 0.0
		deriv := 0.0

		for i in 0 ..< len(coupon_dates) {
			t := coupon_dates[i]
			if t <= bond.settlement_date {
				continue
			}

			time_to_payment := t - bond.settlement_date
			df := math.pow_f64(1.0 + ytm / freq, -time_to_payment * freq)

			if bond.bond_type != .ZeroCoupon {
				price += coupon_payment * df
				deriv -= coupon_payment * time_to_payment * df / (1.0 + ytm / freq)
			}

			if i == len(coupon_dates) - 1 {
				price += bond.face_value * df
				deriv -= bond.face_value * time_to_payment * df / (1.0 + ytm / freq)
			}
		}

		error := price - clean_price
		if math.abs(error) < 1e-8 {
			return ytm
		}

		ytm -= error / deriv

		if ytm < -0.5 {ytm = -0.5}
		if ytm > 2.0 {ytm = 2.0}
	}

	return ytm
}

// ============================================================================
// DURATION AND CONVEXITY
// ============================================================================

calculate_duration_convexity :: proc(
	bond: Bond,
	ytm: f64,
	allocator: mem.Allocator,
) -> (
	modified_duration: f64,
	convexity: f64,
) {

	coupon_dates := get_coupon_dates(bond, allocator)
	defer delete(coupon_dates, allocator)

	freq: f64
	switch bond.coupon_frequency {
	case .Annual:
		freq = 1.0
	case .SemiAnnual:
		freq = 2.0
	case .Quarterly:
		freq = 4.0
	case .Monthly:
		freq = 12.0
	}

	coupon_payment := bond.face_value * bond.coupon_rate / freq
	price := 0.0
	mac_duration := 0.0
	convexity_sum := 0.0

	for i in 0 ..< len(coupon_dates) {
		t := coupon_dates[i]
		if t <= bond.settlement_date {
			continue
		}

		time_to_payment := t - bond.settlement_date
		df := math.pow_f64(1.0 + ytm / freq, -time_to_payment * freq)

		cf := coupon_payment
		if bond.bond_type == .ZeroCoupon {
			cf = 0.0
		}
		if i == len(coupon_dates) - 1 {
			cf += bond.face_value
		}

		pv := cf * df
		price += pv
		mac_duration += time_to_payment * pv
		convexity_sum += time_to_payment * (time_to_payment + 1.0 / freq) * pv
	}

	if price <= 0.0 {
		return 0.0, 0.0
	}

	mac_duration /= price
	convexity_sum /= price

	modified_duration = mac_duration / (1.0 + ytm / freq)
	convexity = convexity_sum / (price * (1.0 + ytm / freq) * (1.0 + ytm / freq))

	return modified_duration, convexity
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
