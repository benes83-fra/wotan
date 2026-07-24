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
	oas:                 f64, // Option-Adjusted Spread (in bps)
	effective_duration:  f64,
	effective_convexity: f64,
	call_probability:    f64, // Probability of being called
}

// ============================================================================
// ZERO-COUPON BOND PRICE UNDER HULL-WHITE 1F
// ============================================================================
// Analytical formula: P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
// where:
//   B(t,T) = (1 - exp(-a*(T-t))) / a
//   A(t,T) = P_market(0,T) / P_market(0,t) * exp(B(t,T)*f(0,t) - v(t,T)^2/2)
//   v(t,T)^2 = sigma^2/(2*a^3) * (1-exp(-a*(T-t)))^2 * (exp(2*a*t)-1)

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

	// B(t,T)
	B := (1.0 - math.exp_f64(-a * tau)) / a

	// Variance term v(t,T)^2
	v_sq :=
		(sigma * sigma) /
		(2.0 * a * a * a) *
		(1.0 - math.exp_f64(-a * tau)) *
		(1.0 - math.exp_f64(-a * tau)) *
		(math.exp_f64(2.0 * a * t) - 1.0)

	// Forward rate f(0,t) ≈ -d/dt ln(P(0,t))
	// Approximate with finite difference
	h := 0.0001
	P0t := P0T_func(t)
	P0t_h := P0T_func(t + h)
	f_0t := -(math.ln(P0t_h) - math.ln(P0t)) / h

	// A(t,T)
	P0T := P0T_func(T)
	log_A := math.ln(P0T / P0t) + B * f_0t - 0.5 * v_sq
	A := math.exp_f64(log_A)

	// P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
	return A * math.exp_f64(-B * r_t)
}

// ============================================================================
// COUPON BOND PRICE (STRAIGHT BOND)
// ============================================================================
// Sums discounted cash flows: coupons + principal

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

	// Generate all cash flow dates
	n_periods := int(bond.maturity * f64(bond.coupon_frequency))
	for i in 1 ..< n_periods + 1 {
		t_i := f64(i) / f64(bond.coupon_frequency)
		if t_i <= t {continue} 	// Skip past cash flows
		cf: f64
		if i == n_periods {
			// Final payment: coupon + principal
			cf = coupon_payment + bond.face_value
		} else {
			// Coupon only
			cf = coupon_payment
		}

		P_t_Ti := _hw1f_zero_coupon_bond(r_t, t, t_i, a, sigma, P0T_func)
		price += cf * P_t_Ti
	}

	return price
}

// ============================================================================
// LSM FOR CALLABLE BONDS (HULL-WHITE 1F)
// ============================================================================
// Prices the embedded call option using Longstaff-Schwartz

callable_bond_lsm_hw1f :: proc(
	bond: CallableBond,
	a: f64, // HW mean reversion
	sigma: f64, // HW volatility
	r0: f64, // Initial short rate
	P0T_func: proc(_: f64) -> f64, // Market discount curve
	n_paths: int = 10000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> CallableBondResult {
	bond := bond
	T := bond.maturity
	dt := T / f64(n_steps)

	// ========================================================================
	// 1. SIMULATE SHORT RATE PATHS (Hull-White 1F)
	// ========================================================================
	// dr(t) = [theta(t) - a*r(t)]dt + sigma*dW(t)
	// where theta(t) is calibrated to the initial curve

	// Precompute theta(t) on the grid (simplified: assume flat forward curve)
	// In production, calibrate theta(t) to match P_market(0,t) exactly
	theta := make([]f64, n_steps + 1, allocator)
	defer delete(theta, allocator)
	for i in 0 ..< n_steps + 1 {
		t_i := f64(i) * dt
		// Simplified: theta(t) = a*r0 + sigma^2/(2a)*(1-exp(-2at)) + df/dt
		// For now, use constant approximation
		theta[i] = a * r0 + 0.02 // Adjust based on curve slope
	}

	// Generate random paths
	rand_count := n_paths * n_steps
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	// Simulate short rate paths using Euler-Maruyama
	r_paths := make([]f64, n_paths * (n_steps + 1), allocator)
	defer delete(r_paths, allocator)

	rand_idx := 0
	for path in 0 ..< n_paths {
		r_paths[path * (n_steps + 1) + 0] = r0

		for step in 1 ..< n_steps + 1 {
			r_prev := r_paths[path * (n_steps + 1) + (step - 1)]
			t_prev := f64(step - 1) * dt
			Z := norm_data[rand_idx]
			rand_idx += 1

			// Euler-Maruyama: r(t+dt) = r(t) + [theta(t) - a*r(t)]*dt + sigma*sqrt(dt)*Z
			dr := (theta[step] - a * r_prev) * dt + sigma * math.sqrt_f64(dt) * Z
			r_paths[path * (n_steps + 1) + step] = r_prev + dr
		}
	}

	// ========================================================================
	// 2. COMPUTE CASH FLOWS AT EACH CALL DATE
	// ========================================================================
	// We work backwards from maturity, determining optimal call strategy

	// Create a map of call dates for quick lookup
	call_dates := bond.call_schedule
	n_call_dates := len(call_dates)

	// Cashflow array: stores the time-0 value of the bond for each path
	// Initially, assume no early call (straight bond)
	cashflows := make([]f64, n_paths, allocator)
	defer delete(cashflows, allocator)

	// Initialize with straight bond value at t=0 for all paths
	for path in 0 ..< n_paths {
		r_0 := r_paths[path * (n_steps + 1) + 0]
		straight_value := _price_straight_bond(&bond, r_0, 0.0, a, sigma, P0T_func)
		cashflows[path] = straight_value
	}

	// ========================================================================
	// 3. BACKWARD INDUCTION (LSM)
	// ========================================================================
	// Work backwards through call dates, determining optimal exercise

	for call_idx := n_call_dates - 1; call_idx >= 0; call_idx -= 1 {
		call_date := call_dates[call_idx]
		t_call := call_date.time
		call_price := call_date.price

		// Find the time step closest to this call date
		step_call := int(t_call / dt)
		if step_call > n_steps {step_call = n_steps}

		// Collect paths that are ITM for calling (bond value > call price)
		itm_count := 0
		for path in 0 ..< n_paths {
			r_t := r_paths[path * (n_steps + 1) + step_call]
			bond_value := _price_straight_bond(&bond, r_t, t_call, a, sigma, P0T_func)
			if bond_value > call_price {
				itm_count += 1
			}
		}

		if itm_count < poly_degree + 1 {continue}

		// Build regression matrices
		X := make([]f64, itm_count * (poly_degree + 1), context.temp_allocator)
		y := make([]f64, itm_count, context.temp_allocator)
		itm_idx := 0

		for path in 0 ..< n_paths {
			r_t := r_paths[path * (n_steps + 1) + step_call]
			bond_value := _price_straight_bond(&bond, r_t, t_call, a, sigma, P0T_func)

			if bond_value > call_price {
				// Basis functions: polynomials in r(t)
				x_normalized := (r_t - r0) / sigma // Normalize short rate
				X[itm_idx * (poly_degree + 1) + 0] = 1.0
				if poly_degree >= 1 {X[itm_idx * (poly_degree + 1) + 1] = x_normalized}
				if poly_degree >=
				   2 {X[itm_idx * (poly_degree + 1) + 2] = x_normalized * x_normalized}
				if poly_degree >=
				   3 {X[itm_idx * (poly_degree + 1) + 3] = x_normalized * x_normalized * x_normalized}

				// Target: discounted continuation value (from cashflows array)
				// Discount from t_call to t=0 using path-dependent discounting
				// Simplified: use average short rate along path
				sum_r := 0.0
				for s in 0 ..< step_call + 1 {
					sum_r += r_paths[path * (n_steps + 1) + s]
				}
				avg_r := sum_r / f64(step_call + 1)
				disc_factor := math.exp_f64(-avg_r * t_call)

				y[itm_idx] = cashflows[path] / disc_factor // Bring to t_call value
				itm_idx += 1
			}
		}

		// Fit regression
		beta := _least_squares_regression(X, y, itm_count, poly_degree + 1, allocator)
		defer delete(beta, allocator)

		// Update cashflows: call if immediate value > continuation value
		called_count := 0
		for path in 0 ..< n_paths {
			r_t := r_paths[path * (n_steps + 1) + step_call]
			bond_value := _price_straight_bond(&bond, r_t, t_call, a, sigma, P0T_func)

			if bond_value > call_price {
				// Continuation value at t_call
				x_normalized := (r_t - r0) / sigma
				cont_val := beta[0]
				if poly_degree >= 1 {cont_val += beta[1] * x_normalized}
				if poly_degree >= 2 {cont_val += beta[2] * x_normalized * x_normalized}
				if poly_degree >=
				   3 {cont_val += beta[3] * x_normalized * x_normalized * x_normalized}

				// Call if call_price < cont_val (issuer minimizes cost)
				// Equivalently: call if bond_value > cont_val
				if call_price < cont_val {
					// Issuer calls the bond
					// Discount call_price back to t=0
					sum_r := 0.0
					for s in 0 ..< step_call + 1 {
						sum_r += r_paths[path * (n_steps + 1) + s]
					}
					avg_r := sum_r / f64(step_call + 1)
					disc_factor := math.exp_f64(-avg_r * t_call)

					cashflows[path] = call_price * disc_factor
					called_count += 1
				}
				// Else: continue holding (cashflows[path] already has continuation value)
			}
		}
	}

	// ========================================================================
	// 4. COMPUTE FINAL RESULTS
	// ========================================================================
	// Average over all paths
	price := 0.0
	for path in 0 ..< n_paths {
		price += cashflows[path]
	}
	price /= f64(n_paths)

	// Estimate call probability (simplified: count paths called at any date)
	// In production, track this more carefully during backward induction
	call_prob := 0.3 // Placeholder - would track during backward induction

	// Compute OAS (Option-Adjusted Spread)
	// OAS is the spread that makes the model price equal to market price
	// For now, return 0 (would require root-finding)
	oas := 0.0

	// Compute effective duration and convexity
	h := 0.0001 // 1 bp parallel shift
	price_up := _callable_bond_helper_shift(
		&bond,
		a,
		sigma,
		r0 + h,
		P0T_func,
		n_paths,
		n_steps,
		poly_degree,
		allocator,
	)
	price_dn := _callable_bond_helper_shift(
		&bond,
		a,
		sigma,
		r0 - h,
		P0T_func,
		n_paths,
		n_steps,
		poly_degree,
		allocator,
	)

	effective_duration := -(price_up - price_dn) / (2.0 * h * price)
	effective_convexity := (price_up + price_dn - 2.0 * price) / (h * h * price)

	return CallableBondResult {
		price = price,
		oas = oas,
		effective_duration = effective_duration,
		effective_convexity = effective_convexity,
		call_probability = call_prob,
	}
}

// Helper for duration/convexity calculation
_callable_bond_helper_shift :: proc(
	bond: ^CallableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int,
	n_steps: int,
	poly_degree: int,
	allocator: mem.Allocator,
) -> f64 {
	// Simplified: just rerun LSM with shifted r0
	// In production, reuse the same random paths for CRN
	result := callable_bond_lsm_hw1f(
		bond^,
		a,
		sigma,
		r0,
		P0T_func,
		n_paths,
		n_steps,
		poly_degree,
		allocator,
	)
	return result.price
}
