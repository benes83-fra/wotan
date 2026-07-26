package finance

import l "../linalg"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// UNIFIED MONTE CARLO ENGINE - SHARED PRIMITIVES
// ============================================================================
// These primitives eliminate the massive duplication across your MC engines:
// - GBM path generation (used in ~15 different pricers)
// - Heston/MJD/HW path generation
// - Asian averaging, barrier checking, lookback max/min
// - LSM regression steps
// - Variance reduction (antithetic, control variate)
// - CRN Greeks computation
//
// USAGE: Refactor existing MC pricers to call these primitives instead of
// duplicating the logic. All public APIs remain unchanged.

// ============================================================================
// 1. SHARED PATH GENERATION
// ============================================================================

// GBM Path Generation (Geometric Brownian Motion)
// Used in: vanilla, asian, barrier, lookback, basket, rbergomi, digital
// Returns flattened array: [n_paths * (n_steps + 1)]
// S_paths[path * (n_steps + 1) + step] = S at that path/step
mc_generate_gbm_paths :: proc(
	S_0: f64,
	r: f64,
	sigma: f64,
	T: f64,
	n_paths: int,
	n_steps: int,
	norm_data: []f64, // Pre-generated normals (length: n_paths * n_steps)
	allocator: mem.Allocator,
) -> []f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	drift := (r - 0.5 * sigma * sigma) * dt

	S_paths := make([]f64, n_paths * (n_steps + 1), allocator)

	rand_idx := 0
	for path in 0 ..< n_paths {
		S_paths[path * (n_steps + 1) + 0] = S_0
		for step in 1 ..< n_steps + 1 {
			Z := norm_data[rand_idx]
			rand_idx += 1
			S_paths[path * (n_steps + 1) + step] =
				S_paths[path * (n_steps + 1) + (step - 1)] *
				math.exp_f64(drift + sigma * sqrt_dt * Z)
		}
	}

	return S_paths
}

// GBM with Antithetic Variates
// First half: original paths, Second half: negated Z's
mc_generate_gbm_paths_antithetic :: proc(
	S_0: f64,
	r: f64,
	sigma: f64,
	T: f64,
	n_paths: int, // Must be even
	n_steps: int,
	norm_data: []f64, // Length: (n_paths/2) * n_steps
	allocator: mem.Allocator,
) -> []f64 {
	half_paths := n_paths / 2
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	drift := (r - 0.5 * sigma * sigma) * dt

	S_paths := make([]f64, n_paths * (n_steps + 1), allocator)

	// Original paths
	rand_idx := 0
	for path in 0 ..< half_paths {
		S_paths[path * (n_steps + 1) + 0] = S_0
		for step in 1 ..< n_steps + 1 {
			Z := norm_data[rand_idx]
			rand_idx += 1
			S_paths[path * (n_steps + 1) + step] =
				S_paths[path * (n_steps + 1) + (step - 1)] *
				math.exp_f64(drift + sigma * sqrt_dt * Z)
		}
	}

	// Antithetic paths (negate Z's)
	rand_idx = 0
	for path in half_paths ..< n_paths {
		antithetic_path := path - half_paths
		S_paths[path * (n_steps + 1) + 0] = S_0
		for step in 1 ..< n_steps + 1 {
			Z := -norm_data[rand_idx] // ← NEGATED
			rand_idx += 1
			S_paths[path * (n_steps + 1) + step] =
				S_paths[path * (n_steps + 1) + (step - 1)] *
				math.exp_f64(drift + sigma * sqrt_dt * Z)
		}
	}

	return S_paths
}

// Heston Stochastic Volatility Path Generation
// Returns both S_paths and V_paths (variance paths)
mc_generate_heston_paths :: proc(
	S_0: f64,
	r: f64,
	T: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64, // Length: n_paths * n_steps * 2
	allocator: mem.Allocator,
) -> (
	S_paths: []f64,
	V_paths: []f64,
) {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)

	S_paths = make([]f64, n_paths * (n_steps + 1), allocator)
	V_paths = make([]f64, n_paths * (n_steps + 1), allocator)

	rand_idx := 0
	for path in 0 ..< n_paths {
		S_paths[path * (n_steps + 1) + 0] = S_0
		V_paths[path * (n_steps + 1) + 0] = params.v0

		for step in 1 ..< n_steps + 1 {
			Z_v := norm_data[rand_idx]
			Z_indep := norm_data[rand_idx + 1]
			rand_idx += 2

			v_prev := V_paths[path * (n_steps + 1) + (step - 1)]
			S_prev := S_paths[path * (n_steps + 1) + (step - 1)]

			v_sqrt := math.sqrt_f64(math.max(0.0, v_prev))
			Z_s := params.rho * Z_v + math.sqrt_f64(1.0 - params.rho * params.rho) * Z_indep

			// Euler-Maruyama for variance
			v_new :=
				v_prev +
				params.kappa * (params.theta - v_prev) * dt +
				params.sigma * v_sqrt * sqrt_dt * Z_v
			v_new = math.max(0.0, v_new)

			// Log-Euler for stock
			S_new := S_prev * math.exp_f64((r - 0.5 * v_new) * dt + v_sqrt * sqrt_dt * Z_s)

			V_paths[path * (n_steps + 1) + step] = v_new
			S_paths[path * (n_steps + 1) + step] = S_new
		}
	}

	return S_paths, V_paths
}

// Merton Jump Diffusion Path Generation
mc_generate_mjd_paths :: proc(
	S_0: f64,
	r: f64,
	T: f64,
	params: MJD_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64, // Length: n_paths * n_steps * 2
	unif_data: []f64, // Length: n_paths * n_steps
	allocator: mem.Allocator,
) -> []f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	kappa := math.exp_f64(params.mu_j + 0.5 * params.sigma_j * params.sigma_j) - 1.0
	drift := r - params.lambda * kappa - 0.5 * params.sigma * params.sigma
	lambda_dt := params.lambda * dt

	S_paths := make([]f64, n_paths * (n_steps + 1), allocator)

	norm_idx := 0
	unif_idx := 0
	for path in 0 ..< n_paths {
		S_paths[path * (n_steps + 1) + 0] = S_0

		for step in 1 ..< n_steps + 1 {
			Z := norm_data[norm_idx]
			norm_idx += 1
			U := unif_data[unif_idx]
			unif_idx += 1
			Z_J := norm_data[norm_idx]
			norm_idx += 1

			S_prev := S_paths[path * (n_steps + 1) + (step - 1)]
			S_new := S_prev * math.exp_f64(drift * dt + params.sigma * sqrt_dt * Z)

			// Jump
			if U < lambda_dt {
				jump_multiplier := math.exp_f64(params.mu_j + params.sigma_j * Z_J)
				S_new = S_new * jump_multiplier
			}

			S_paths[path * (n_steps + 1) + step] = S_new
		}
	}

	return S_paths
}

// Hull-White 1F Short Rate Path Generation
mc_generate_hw1f_paths :: proc(
	r0: f64,
	T: f64,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64,
	allocator: mem.Allocator,
) -> []f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)

	r_paths := make([]f64, n_paths * (n_steps + 1), allocator)

	// Calibrate theta(t) for flat initial curve
	theta := make([]f64, n_steps + 1, allocator)
	defer delete(theta, allocator)
	for i in 0 ..< n_steps + 1 {
		t_i := f64(i) * dt
		theta[i] =
			params.a * r0 +
			(params.sigma * params.sigma / (2.0 * params.a)) *
				(1.0 - math.exp_f64(-2.0 * params.a * t_i))
	}

	rand_idx := 0
	for path in 0 ..< n_paths {
		r_paths[path * (n_steps + 1) + 0] = r0

		for step in 1 ..< n_steps + 1 {
			r_prev := r_paths[path * (n_steps + 1) + (step - 1)]
			Z := norm_data[rand_idx]
			rand_idx += 1

			dr := (theta[step] - params.a * r_prev) * dt + params.sigma * sqrt_dt * Z
			r_paths[path * (n_steps + 1) + step] = r_prev + dr
		}
	}

	return r_paths
}

// ============================================================================
// 2. SHARED PAYOFF PRIMITIVES
// ============================================================================

// Asian Arithmetic Average (continuous monitoring)
mc_compute_asian_average :: proc(S_paths: []f64, path: int, n_steps: int) -> f64 {
	sum := 0.0
	for step in 1 ..< n_steps + 1 {
		sum += S_paths[path * (n_steps + 1) + step]
	}
	return sum / f64(n_steps)
}

// Asian with Running Average (for early exercise)
mc_compute_asian_average_to_step :: proc(S_paths: []f64, path: int, step: int) -> f64 {
	sum := S_paths[path * (step + 1) + 0] // Include S_0
	for s in 1 ..< step + 1 {
		sum += S_paths[path * (step + 1) + s]
	}
	return sum / f64(step + 1)
}

// Barrier Check (returns true if barrier is breached)
mc_check_barrier_breached :: proc(
	S_paths: []f64,
	path: int,
	n_steps: int,
	barrier: f64,
	is_up: bool,
) -> bool {
	for step in 1 ..< n_steps + 1 {
		S_t := S_paths[path * (n_steps + 1) + step]
		if is_up {
			if S_t >= barrier {
				return true
			}
		} else {
			if S_t <= barrier {
				return true
			}
		}
	}
	return false
}

// Double Barrier Check (returns true if EITHER barrier is breached)
mc_check_double_barrier_breached :: proc(
	S_paths: []f64,
	path: int,
	n_steps: int,
	lower: f64,
	upper: f64,
) -> bool {
	for step in 1 ..< n_steps + 1 {
		S_t := S_paths[path * (n_steps + 1) + step]
		if S_t <= lower || S_t >= upper {
			return true
		}
	}
	return false
}

// Lookback Running Maximum
mc_compute_lookback_max :: proc(S_paths: []f64, path: int, n_steps: int) -> f64 {
	max_S := S_paths[path * (n_steps + 1) + 0]
	for step in 1 ..< n_steps + 1 {
		S_t := S_paths[path * (n_steps + 1) + step]
		if S_t > max_S {
			max_S = S_t
		}
	}
	return max_S
}

// Lookback Running Minimum
mc_compute_lookback_min :: proc(S_paths: []f64, path: int, n_steps: int) -> f64 {
	min_S := S_paths[path * (n_steps + 1) + 0]
	for step in 1 ..< n_steps + 1 {
		S_t := S_paths[path * (n_steps + 1) + step]
		if S_t < min_S {
			min_S = S_t
		}
	}
	return min_S
}

// ============================================================================
// 3. SHARED LSM PRIMITIVES
// ============================================================================

// LSM Regression Step (updates cashflows in-place)
// Used in: American vanilla, Asian, callable/puttable bonds, Bermudan swaptions
mc_lsm_regression_step :: proc(
	S_paths: []f64,
	cashflows: []f64,
	n_paths: int,
	ex_idx: int,
	n_exercise_dates: int,
	t_ex: f64,
	r: f64,
	K: f64,
	opt: OptionType,
	poly_degree: int,
	allocator: mem.Allocator,
) {
	// Count ITM paths
	itm_count := 0
	for path in 0 ..< n_paths {
		S_ex := S_paths[path * (n_exercise_dates + 1) + ex_idx]
		if opt == .Call && S_ex > K {
			itm_count += 1
		} else if opt == .Put && S_ex < K {
			itm_count += 1
		}
	}

	if itm_count < poly_degree + 1 {
		return // Not enough ITM paths for regression
	}

	// Build regression matrices
	X := make([]f64, itm_count * (poly_degree + 1), context.temp_allocator)
	y := make([]f64, itm_count, context.temp_allocator)
	itm_idx := 0

	S_0 := S_paths[0] // Initial spot
	for path in 0 ..< n_paths {
		S_ex := S_paths[path * (n_exercise_dates + 1) + ex_idx]
		is_itm := false
		if opt == .Call && S_ex > K {
			is_itm = true
		} else if opt == .Put && S_ex < K {
			is_itm = true
		}

		if is_itm {
			x_moneyness := (S_ex / S_0) - 1.0
			X[itm_idx * (poly_degree + 1) + 0] = 1.0
			if poly_degree >= 1 {
				X[itm_idx * (poly_degree + 1) + 1] = x_moneyness
			}
			if poly_degree >= 2 {
				X[itm_idx * (poly_degree + 1) + 2] = x_moneyness * x_moneyness
			}
			if poly_degree >= 3 {
				X[itm_idx * (poly_degree + 1) + 3] = x_moneyness * x_moneyness * x_moneyness
			}

			// Bring time-0 cashflow forward to t_ex for regression
			y[itm_idx] = cashflows[path] * math.exp_f64(r * t_ex)
			itm_idx += 1
		}
	}

	// Fit regression
	beta := _least_squares_regression(X, y, itm_count, poly_degree + 1, allocator)
	defer delete(beta, allocator)

	// Update cashflows based on exercise decision
	for path in 0 ..< n_paths {
		S_ex := S_paths[path * (n_exercise_dates + 1) + ex_idx]
		is_itm := false
		if opt == .Call && S_ex > K {
			is_itm = true
		} else if opt == .Put && S_ex < K {
			is_itm = true
		}

		if is_itm {
			exercise_val := 0.0
			if opt == .Call {
				exercise_val = S_ex - K
			} else {
				exercise_val = K - S_ex
			}

			x_moneyness := (S_ex / S_0) - 1.0
			cont_val := beta[0]
			if poly_degree >= 1 {
				cont_val += beta[1] * x_moneyness
			}
			if poly_degree >= 2 {
				cont_val += beta[2] * x_moneyness * x_moneyness
			}
			if poly_degree >= 3 {
				cont_val += beta[3] * x_moneyness * x_moneyness * x_moneyness
			}

			if exercise_val > cont_val {
				cashflows[path] = exercise_val * math.exp_f64(-r * t_ex)
			}
		}
	}
}

// ============================================================================
// 4. SHARED VARIANCE REDUCTION
// ============================================================================

// Control Variate Adjustment
// Returns adjusted price
mc_apply_control_variate :: proc(
	payoffs: []f64, // MC payoffs
	control_payoffs: []f64, // Control variate payoffs
	control_analytical: f64, // Analytical price of control
	n_paths: int,
) -> (
	adjusted_price: f64,
	variance_reduction: f64,
) {
	// Compute means
	mean_payoff := 0.0
	mean_control := 0.0
	for i in 0 ..< n_paths {
		mean_payoff += payoffs[i]
		mean_control += control_payoffs[i]
	}
	mean_payoff /= f64(n_paths)
	mean_control /= f64(n_paths)

	// Compute beta = Cov(payoff, control) / Var(control)
	cov := 0.0
	var_control := 0.0
	for i in 0 ..< n_paths {
		d_payoff := payoffs[i] - mean_payoff
		d_control := control_payoffs[i] - mean_control
		cov += d_payoff * d_control
		var_control += d_control * d_control
	}
	cov /= f64(n_paths)
	var_control /= f64(n_paths)

	beta_cv := 0.0
	if var_control > 1e-12 {
		beta_cv = cov / var_control
	}

	// Apply adjustment
	adjusted_price = mean_payoff - beta_cv * (mean_control - control_analytical)

	// Estimate variance reduction
	var_plain := 0.0
	var_adjusted := 0.0
	for i in 0 ..< n_paths {
		d_plain := payoffs[i] - mean_payoff
		var_plain += d_plain * d_plain

		adjusted_i := payoffs[i] - beta_cv * (control_payoffs[i] - control_analytical)
		d_adj := adjusted_i - adjusted_price
		var_adjusted += d_adj * d_adj
	}
	var_plain /= f64(n_paths)
	var_adjusted /= f64(n_paths)

	variance_reduction = 1.0
	if var_adjusted > 1e-20 {
		variance_reduction = var_plain / var_adjusted
	}

	return adjusted_price, variance_reduction
}

// ============================================================================
// 5. SHARED GREEKS (CRN - Common Random Numbers)
// ============================================================================

// CRN Delta/Gamma via finite differences
// price_fn: function that computes price given S_0 (must reuse same norm_data)
mc_compute_crn_delta_gamma :: proc(
	S_0: f64,
	price_fn: proc(_: f64) -> f64,
) -> (
	delta: f64,
	gamma: f64,
) {
	h_S := 0.01 * S_0
	price_up := price_fn(S_0 + h_S)
	price_dn := price_fn(S_0 - h_S)
	delta = (price_up - price_dn) / (2.0 * h_S)

	h_S2 := 0.02 * S_0
	delta_up := (price_fn(S_0 + h_S2) - price_dn) / (2.0 * h_S2)
	delta_dn := (price_up - price_fn(S_0 - h_S2)) / (2.0 * h_S2)
	gamma = (delta_up - delta_dn) / (2.0 * h_S2)

	return delta, gamma
}

// CRN Vega via finite differences
mc_compute_crn_vega :: proc(S_0: f64, sigma: f64, price_fn: proc(_: f64, _: f64) -> f64) -> f64 {
	h_sigma := 0.01 * sigma
	if h_sigma < 0.001 {
		h_sigma = 0.001
	}
	price_sig_up := price_fn(S_0, sigma + h_sigma)
	price_sig_dn := price_fn(S_0, sigma - h_sigma)
	return (price_sig_up - price_sig_dn) / (2.0 * h_sigma)
}

// ============================================================================
// 6. UTILITY: Random Number Generation
// ============================================================================

// Generate standard normal random numbers
mc_generate_normals :: proc(n: int, allocator: mem.Allocator) -> []f64 {
	norm_data := make([]f64, n, allocator)
	for i in 0 ..< n {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}
	return norm_data
}

// Generate uniform random numbers
mc_generate_uniforms :: proc(n: int, allocator: mem.Allocator) -> []f64 {
	unif_data := make([]f64, n, allocator)
	for i in 0 ..< n {
		unif_data[i] = rand.float64()
	}
	return unif_data
}
