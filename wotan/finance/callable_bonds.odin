package finance

import l "../linalg"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// CALLABLE BOND STRUCTURES
// ============================================================================

CallableBond :: struct {
	face_value:       f64,
	coupon_rate:      f64, // Annual coupon rate
	coupon_frequency: int, // Payments per year (e.g., 2 for semi-annual)
	maturity:         f64, // Bond maturity in years
	call_schedule:    []CallDate, // Call dates and prices
	settlement:       f64, // Settlement date (usually 0)
}

CallDate :: struct {
	time:  f64, // Time from today (in years)
	price: f64, // Call price (usually par or par + premium)
}

CallableBondResult :: struct {
	price:               f64,
	straight_bond_price: f64,
	embedded_call_value: f64,
	oas:                 f64, // Option-Adjusted Spread (in decimal, e.g., 0.0050 = 50 bps)
	effective_duration:  f64,
	effective_convexity: f64,
	call_probability:    f64, // Probability of being called
}

// ============================================================================
// ZERO-COUPON BOND PRICE UNDER HULL-WHITE 1F
// ============================================================================
_hw1f_zero_coupon_bond :: proc(
	r_t: f64, // Current short rate
	t: f64, // Current time
	T: f64, // Maturity
	a: f64, // Mean reversion
	sigma: f64, // Volatility
	P0T_func: proc(_: f64) -> f64, // Market discount curve P(0,.)
) -> f64 {
	if t >= T {return 1.0}

	tau := T - t
	B := (1.0 - math.exp_f64(-a * tau)) / a

	v_sq :=
		(sigma * sigma) /
		(2.0 * a * a * a) *
		(1.0 - math.exp_f64(-a * tau)) *
		(1.0 - math.exp_f64(-a * tau)) *
		(math.exp_f64(2.0 * a * t) - 1.0)

	h := 0.0001
	P0t := P0T_func(t)
	P0t_h := P0T_func(t + h)
	f_0t := -(math.ln(P0t_h) - math.ln(P0t)) / h

	P0T := P0T_func(T)
	log_A := math.ln(P0T / P0t) + B * f_0t - 0.5 * v_sq
	A := math.exp_f64(log_A)

	return A * math.exp_f64(-B * r_t)
}

// ============================================================================
// COUPON BOND PRICE (STRAIGHT BOND)
// ============================================================================
_price_straight_bond :: proc(
	bond: ^CallableBond,
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
// CORE PRICING ENGINE (Accepts OAS shift and pre-generated norm_data for CRN)
// ============================================================================
_callable_bond_price_and_prob :: proc(
	bond: CallableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int,
	n_steps: int,
	poly_degree: int,
	norm_data: []f64,
	oas: f64, // ✅ ADDED: Option-Adjusted Spread (parallel shift to short rate)
	allocator: mem.Allocator,
) -> (
	f64,
	f64,
) {
	bond := bond // Idiomatic Odin: create mutable local copy of parameter

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

	call_dates := bond.call_schedule
	n_call_dates := len(call_dates)

	// Precalculate time-0 PV of coupons for each call date
	call_coupon_pv := make([]f64, n_call_dates, allocator)
	defer delete(call_coupon_pv, allocator)
	coupon_payment := bond.face_value * bond.coupon_rate / f64(bond.coupon_frequency)

	for call_idx in 0 ..< n_call_dates {
		t_call := call_dates[call_idx].time
		pv := 0.0
		n_periods := int(t_call * f64(bond.coupon_frequency))
		for i in 1 ..< n_periods + 1 {
			t_i := f64(i) / f64(bond.coupon_frequency)
			pv += coupon_payment * P0T_func(t_i)
		}
		call_coupon_pv[call_idx] = pv
	}

	cashflows := make([]f64, n_paths, allocator)
	defer delete(cashflows, allocator)

	for path in 0 ..< n_paths {
		r_0_oas := r_paths[path * (n_steps + 1) + 0] + oas
		cashflows[path] = _price_straight_bond(&bond, r_0_oas, 0.0, a, sigma, P0T_func)
	}

	called := make([]bool, n_paths, allocator)
	defer delete(called, allocator)
	called_count := 0

	for call_idx := n_call_dates - 1; call_idx >= 0; call_idx -= 1 {
		call_date := call_dates[call_idx]
		t_call := call_date.time
		call_price := call_date.price

		step_call := int(t_call / dt)
		if step_call > n_steps {step_call = n_steps}
		if step_call < 1 {step_call = 1}

		itm_count := 0
		for path in 0 ..< n_paths {
			if called[path] {continue}
			r_t_oas := r_paths[path * (n_steps + 1) + step_call] + oas
			bond_value := _price_straight_bond(&bond, r_t_oas, t_call, a, sigma, P0T_func)
			if bond_value > call_price {
				itm_count += 1
			}
		}

		if itm_count < poly_degree + 1 {continue}

		X := make([]f64, itm_count * (poly_degree + 1), context.temp_allocator)
		y := make([]f64, itm_count, context.temp_allocator)
		itm_idx := 0

		for path in 0 ..< n_paths {
			if called[path] {continue}
			r_t_oas := r_paths[path * (n_steps + 1) + step_call] + oas
			bond_value := _price_straight_bond(&bond, r_t_oas, t_call, a, sigma, P0T_func)

			if bond_value > call_price {
				x_normalized := (r_t_oas - (r0 + oas)) / (sigma + 1e-8)
				X[itm_idx * (poly_degree + 1) + 0] = 1.0
				if poly_degree >= 1 {X[itm_idx * (poly_degree + 1) + 1] = x_normalized}
				if poly_degree >=
				   2 {X[itm_idx * (poly_degree + 1) + 2] = x_normalized * x_normalized}
				if poly_degree >=
				   3 {X[itm_idx * (poly_degree + 1) + 3] = x_normalized * x_normalized * x_normalized}

				integral_r := 0.0
				for s in 0 ..< step_call {
					// ✅ Apply OAS shift to the discounting integral
					integral_r += (r_paths[path * (n_steps + 1) + s] + oas) * dt
				}
				y[itm_idx] = cashflows[path] * math.exp_f64(integral_r)
				itm_idx += 1
			}
		}

		beta := _least_squares_regression(X, y, itm_count, poly_degree + 1, allocator)
		defer delete(beta, allocator)

		for path in 0 ..< n_paths {
			if called[path] {continue}
			r_t_oas := r_paths[path * (n_steps + 1) + step_call] + oas
			bond_value := _price_straight_bond(&bond, r_t_oas, t_call, a, sigma, P0T_func)

			if bond_value > call_price {
				x_normalized := (r_t_oas - (r0 + oas)) / (sigma + 1e-8)
				cont_val := beta[0]
				if poly_degree >= 1 {cont_val += beta[1] * x_normalized}
				if poly_degree >= 2 {cont_val += beta[2] * x_normalized * x_normalized}
				if poly_degree >=
				   3 {cont_val += beta[3] * x_normalized * x_normalized * x_normalized}

				if call_price < cont_val {
					integral_r := 0.0
					for s in 0 ..< step_call {
						integral_r += (r_paths[path * (n_steps + 1) + s] + oas) * dt
					}
					disc_factor := math.exp_f64(-integral_r)

					cashflows[path] = call_coupon_pv[call_idx] + call_price * disc_factor
					called[path] = true
					called_count += 1
				}
			}
		}
	}

	price := 0.0
	for path in 0 ..< n_paths {
		price += cashflows[path]
	}

	return price / f64(n_paths), f64(called_count) / f64(n_paths)
}

// ============================================================================
// OAS SOLVER: Bisection Method
// ============================================================================
callable_bond_oas :: proc(
	bond: CallableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	market_price: f64,
	n_paths: int = 5000, // Slightly lower default for speed during root-finding
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	// Generate random numbers ONCE for Common Random Numbers (CRN) across all iterations
	rand_count := n_paths * n_steps
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	// Bisection bounds: -500 bps to +500 bps
	low := -0.05
	high := 0.05
	tol := 1e-6 // 0.1 bps tolerance
	max_iter := 50

	// Check if market price is even within bounds
	price_low, _ := _callable_bond_price_and_prob(
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
	price_high, _ := _callable_bond_price_and_prob(
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
		return low // Market price is too high, OAS is below -500 bps
	}
	if market_price < price_high {
		return high // Market price is too low, OAS is above +500 bps
	}

	// Bisection loop
	for _ in 0 ..< max_iter {
		mid := (low + high) / 2.0
		price_mid, _ := _callable_bond_price_and_prob(
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

		// Higher spread -> lower price.
		// If model price > market price, we need a higher spread to bring it down.
		if price_mid > market_price {
			low = mid
		} else {
			high = mid
		}
	}

	return (low + high) / 2.0 // Return best estimate if max iterations reached
}

// ============================================================================
// MAIN API: LSM FOR CALLABLE BONDS (HULL-WHITE 1F)
// ============================================================================
callable_bond_lsm_hw1f :: proc(
	bond: CallableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int = 10000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> CallableBondResult {
	bond := bond

	// 1. Generate random numbers ONCE for Common Random Numbers (CRN)
	rand_count := n_paths * n_steps
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	// 2. Get main price and call probability (with 0.0 OAS)
	price, call_prob := _callable_bond_price_and_prob(
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
	straight_price := _price_straight_bond(&bond, r0, 0.0, a, sigma, P0T_func)

	// 3. Compute Greeks using CRN and a larger shift to dominate Monte Carlo noise
	greek_paths := n_paths / 2
	if greek_paths < 1000 {greek_paths = 1000}

	h := 0.001 // 10 bps shift for numerical stability
	greek_norm_data := norm_data[0:greek_paths * n_steps]

	price_up, _ := _callable_bond_price_and_prob(
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
	price_dn, _ := _callable_bond_price_and_prob(
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

	return CallableBondResult {
		price               = price,
		straight_bond_price = straight_price,
		embedded_call_value = straight_price - price,
		oas                 = 0.0, // This function returns Z-spread/OAS=0 by default
		effective_duration  = effective_duration,
		effective_convexity = effective_convexity,
		call_probability    = call_prob,
	}
}
