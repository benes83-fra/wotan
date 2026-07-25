package finance

import l "../linalg"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// PUTTABLE BOND STRUCTURES
// ============================================================================

PuttableBond :: struct {
	face_value:       f64,
	coupon_rate:      f64, // Annual coupon rate
	coupon_frequency: int, // Payments per year (e.g., 2 for semi-annual)
	maturity:         f64, // Bond maturity in years
	put_schedule:     []PutDate, // Put dates and prices
	settlement:       f64, // Settlement date (usually 0)
}

PutDate :: struct {
	time:  f64, // Time from today (in years)
	price: f64, // Put price (usually par or par + premium)
}

PuttableBondResult :: struct {
	price:               f64,
	straight_bond_price: f64,
	embedded_put_value:  f64, // Puttable bonds are worth MORE than straight bonds
	oas:                 f64, // Option-Adjusted Spread (in decimal, e.g., 0.0050 = 50 bps)
	effective_duration:  f64,
	effective_convexity: f64,
	put_probability:     f64, // Probability of being put
}

// ============================================================================
// STRAIGHT BOND PRICE FOR PUTTABLE BONDS (Fixes the type mismatch)
// ============================================================================
_price_straight_bond_puttable :: proc(
	bond: ^PuttableBond,
	r_t: f64,
	t: f64,
	a: f64,
	sigma: f64,
	P0T_func: proc(_: f64) -> f64,
) -> f64 {
	price := 0.0
	coupon_payment := bond.face_value * bond.coupon_rate / f64(bond.coupon_frequency)

	n_periods := int(bond.maturity * f64(bond.coupon_frequency))
	for i in 1 ..< n_periods + 1 {
		t_i := f64(i) / f64(bond.coupon_frequency)
		if t_i <= t {continue}

		cf: f64
		if i == n_periods {
			cf = coupon_payment + bond.face_value
		} else {
			cf = coupon_payment
		}

		P_t_Ti := _hw1f_zero_coupon_bond(r_t, t, t_i, a, sigma, P0T_func)
		price += cf * P_t_Ti
	}

	return price
}

// ============================================================================
// CORE PRICING ENGINE FOR PUTTABLE BONDS
// ============================================================================
_puttable_bond_price_and_prob :: proc(
	bond: PuttableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int,
	n_steps: int,
	poly_degree: int,
	norm_data: []f64,
	oas: f64,
	allocator: mem.Allocator,
) -> (
	f64,
	f64,
) {
	bond := bond // Idiomatic Odin: create mutable local copy

	T := bond.maturity
	dt := T / f64(n_steps)

	// Calibrate theta(t) to the initial discount curve
	theta := make([]f64, n_steps + 1, allocator)
	defer delete(theta, allocator)
	for i in 0 ..< n_steps + 1 {
		t_i := f64(i) * dt
		h := 0.0001
		P0t := P0T_func(t_i)
		P0t_h := P0T_func(t_i + h)
		f_0t := -(math.ln(P0t_h) - math.ln(P0t)) / h

		P0t_h2 := P0T_func(t_i + 2.0 * h)
		f_0t_h := -(math.ln(P0t_h2) - math.ln(P0t_h)) / h
		df_0t := (f_0t_h - f_0t) / h

		theta[i] =
			df_0t + a * f_0t + (sigma * sigma / (2.0 * a)) * (1.0 - math.exp_f64(-2.0 * a * t_i))
	}

	r_paths := make([]f64, n_paths * (n_steps + 1), allocator)
	defer delete(r_paths, allocator)

	rand_idx := 0
	for path in 0 ..< n_paths {
		r_paths[path * (n_steps + 1) + 0] = r0
		for step in 1 ..< n_steps + 1 {
			r_prev := r_paths[path * (n_steps + 1) + (step - 1)]
			Z := norm_data[rand_idx]
			rand_idx += 1
			dr := (theta[step] - a * r_prev) * dt + sigma * math.sqrt_f64(dt) * Z
			r_paths[path * (n_steps + 1) + step] = r_prev + dr
		}
	}

	put_dates := bond.put_schedule
	n_put_dates := len(put_dates)

	// Precalculate time-0 PV of coupons for each put date
	put_coupon_pv := make([]f64, n_put_dates, allocator)
	defer delete(put_coupon_pv, allocator)
	coupon_payment := bond.face_value * bond.coupon_rate / f64(bond.coupon_frequency)

	for put_idx in 0 ..< n_put_dates {
		t_put := put_dates[put_idx].time
		pv := 0.0
		n_periods := int(t_put * f64(bond.coupon_frequency))
		for i in 1 ..< n_periods + 1 {
			t_i := f64(i) / f64(bond.coupon_frequency)
			pv += coupon_payment * P0T_func(t_i)
		}
		put_coupon_pv[put_idx] = pv
	}

	cashflows := make([]f64, n_paths, allocator)
	defer delete(cashflows, allocator)

	for path in 0 ..< n_paths {
		r_0_oas := r_paths[path * (n_steps + 1) + 0] + oas
		// ✅ FIXED: Use the puttable-specific straight bond pricer
		cashflows[path] = _price_straight_bond_puttable(&bond, r_0_oas, 0.0, a, sigma, P0T_func)
	}

	put := make([]bool, n_paths, allocator)
	defer delete(put, allocator)
	put_count := 0

	// ✅ CRITICAL FLIP: Work backwards through put dates
	for put_idx := n_put_dates - 1; put_idx >= 0; put_idx -= 1 {
		put_date := put_dates[put_idx]
		t_put := put_date.time
		put_price := put_date.price

		step_put := int(t_put / dt)
		if step_put > n_steps {step_put = n_steps}
		if step_put < 1 {step_put = 1}

		// ✅ CRITICAL FLIP: Investor puts when bond_value < put_price
		itm_count := 0
		for path in 0 ..< n_paths {
			if put[path] {continue}
			r_t_oas := r_paths[path * (n_steps + 1) + step_put] + oas
			bond_value := _price_straight_bond_puttable(&bond, r_t_oas, t_put, a, sigma, P0T_func)
			if bond_value < put_price { 	// ✅ FLIPPED: < instead of >
				itm_count += 1
			}
		}

		if itm_count < poly_degree + 1 {continue}

		X := make([]f64, itm_count * (poly_degree + 1), context.temp_allocator)
		y := make([]f64, itm_count, context.temp_allocator)
		itm_idx := 0

		for path in 0 ..< n_paths {
			if put[path] {continue}
			r_t_oas := r_paths[path * (n_steps + 1) + step_put] + oas
			bond_value := _price_straight_bond_puttable(&bond, r_t_oas, t_put, a, sigma, P0T_func)

			if bond_value < put_price { 	// ✅ FLIPPED
				x_normalized := (r_t_oas - (r0 + oas)) / (sigma + 1e-8)
				X[itm_idx * (poly_degree + 1) + 0] = 1.0
				if poly_degree >= 1 {X[itm_idx * (poly_degree + 1) + 1] = x_normalized}
				if poly_degree >=
				   2 {X[itm_idx * (poly_degree + 1) + 2] = x_normalized * x_normalized}
				if poly_degree >=
				   3 {X[itm_idx * (poly_degree + 1) + 3] = x_normalized * x_normalized * x_normalized}

				integral_r := 0.0
				for s in 0 ..< step_put {
					integral_r += (r_paths[path * (n_steps + 1) + s] + oas) * dt
				}
				y[itm_idx] = cashflows[path] * math.exp_f64(integral_r)
				itm_idx += 1
			}
		}

		beta := _least_squares_regression(X, y, itm_count, poly_degree + 1, allocator)
		defer delete(beta, allocator)

		for path in 0 ..< n_paths {
			if put[path] {continue}
			r_t_oas := r_paths[path * (n_steps + 1) + step_put] + oas
			bond_value := _price_straight_bond_puttable(&bond, r_t_oas, t_put, a, sigma, P0T_func)

			if bond_value < put_price { 	// ✅ FLIPPED
				x_normalized := (r_t_oas - (r0 + oas)) / (sigma + 1e-8)
				cont_val := beta[0]
				if poly_degree >= 1 {cont_val += beta[1] * x_normalized}
				if poly_degree >= 2 {cont_val += beta[2] * x_normalized * x_normalized}
				if poly_degree >=
				   3 {cont_val += beta[3] * x_normalized * x_normalized * x_normalized}

				// ✅ CRITICAL FLIP: Investor puts when put_price > cont_val
				// (Investor maximizes value by selling back to issuer)
				if put_price > cont_val { 	// ✅ FLIPPED: > instead of <
					integral_r := 0.0
					for s in 0 ..< step_put {
						integral_r += (r_paths[path * (n_steps + 1) + s] + oas) * dt
					}
					disc_factor := math.exp_f64(-integral_r)

					// Investor receives put_price + coupons up to put date
					cashflows[path] = put_coupon_pv[put_idx] + put_price * disc_factor
					put[path] = true
					put_count += 1
				}
			}
		}
	}

	price := 0.0
	for path in 0 ..< n_paths {
		price += cashflows[path]
	}

	return price / f64(n_paths), f64(put_count) / f64(n_paths)
}

// ============================================================================
// OAS SOLVER FOR PUTTABLE BONDS
// ============================================================================
puttable_bond_oas :: proc(
	bond: PuttableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	market_price: f64,
	n_paths: int = 5000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	rand_count := n_paths * n_steps
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	low := -0.05
	high := 0.05
	tol := 1e-6
	max_iter := 50

	price_low, _ := _puttable_bond_price_and_prob(
		bond,
		a,
		sigma,
		r0,
		P0T_func,
		n_paths,
		n_steps,
		poly_degree,
		norm_data,
		low,
		allocator,
	)
	price_high, _ := _puttable_bond_price_and_prob(
		bond,
		a,
		sigma,
		r0,
		P0T_func,
		n_paths,
		n_steps,
		poly_degree,
		norm_data,
		high,
		allocator,
	)

	if market_price > price_low {
		return low
	}
	if market_price < price_high {
		return high
	}

	for _ in 0 ..< max_iter {
		mid := (low + high) / 2.0
		price_mid, _ := _puttable_bond_price_and_prob(
			bond,
			a,
			sigma,
			r0,
			P0T_func,
			n_paths,
			n_steps,
			poly_degree,
			norm_data,
			mid,
			allocator,
		)

		if math.abs(price_mid - market_price) < tol {
			return mid
		}

		if price_mid > market_price {
			low = mid
		} else {
			high = mid
		}
	}

	return (low + high) / 2.0
}

// ============================================================================
// MAIN API: LSM FOR PUTTABLE BONDS (HULL-WHITE 1F)
// ============================================================================
puttable_bond_lsm_hw1f :: proc(
	bond: PuttableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int = 10000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> PuttableBondResult {
	bond := bond

	rand_count := n_paths * n_steps
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	price, put_prob := _puttable_bond_price_and_prob(
		bond,
		a,
		sigma,
		r0,
		P0T_func,
		n_paths,
		n_steps,
		poly_degree,
		norm_data,
		0.0,
		allocator,
	)

	// ✅ FIXED: Use the puttable-specific straight bond pricer
	straight_price := _price_straight_bond_puttable(&bond, r0, 0.0, a, sigma, P0T_func)

	greek_paths := n_paths / 2
	if greek_paths < 1000 {greek_paths = 1000}

	h := 0.001
	greek_norm_data := norm_data[0:greek_paths * n_steps]

	price_up, _ := _puttable_bond_price_and_prob(
		bond,
		a,
		sigma,
		r0 + h,
		P0T_func,
		greek_paths,
		n_steps,
		poly_degree,
		greek_norm_data,
		0.0,
		allocator,
	)
	price_dn, _ := _puttable_bond_price_and_prob(
		bond,
		a,
		sigma,
		r0 - h,
		P0T_func,
		greek_paths,
		n_steps,
		poly_degree,
		greek_norm_data,
		0.0,
		allocator,
	)

	effective_duration := -(price_up - price_dn) / (2.0 * h * price)
	effective_convexity := (price_up + price_dn - 2.0 * price) / (h * h * price)

	return PuttableBondResult {
		price               = price,
		straight_bond_price = straight_price,
		embedded_put_value  = price - straight_price, // ✅ Puttable bonds are worth MORE
		oas                 = 0.0,
		effective_duration  = effective_duration,
		effective_convexity = effective_convexity,
		put_probability     = put_prob,
	}
}
