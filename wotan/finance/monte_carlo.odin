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

// GBM with Antithetic Variates: paths 0 ..< h use the normals, paths h ..< n_paths
// their negatives, h = ceil(n_paths / 2) (an odd count drops the last mirror)
mc_generate_gbm_paths_antithetic :: proc(
	S_0: f64,
	r: f64,
	sigma: f64,
	T: f64,
	n_paths: int,
	n_steps: int,
	norm_data: []f64, // Length: ceil(n_paths/2) * n_steps
	allocator: mem.Allocator,
) -> []f64 {
	half_paths := (n_paths + 1) / 2
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	drift := (r - 0.5 * sigma * sigma) * dt

	S_paths := make([]f64, n_paths * (n_steps + 1), allocator)
	for path in 0 ..< n_paths {
		base := path if path < half_paths else path - half_paths
		sgn := 1.0 if path < half_paths else -1.0
		row := path * (n_steps + 1)
		S_paths[row] = S_0
		for step in 1 ..< n_steps + 1 {
			Z := sgn * norm_data[base * n_steps + step - 1]
			S_paths[row + step] =
				S_paths[row + step - 1] * math.exp_f64(drift + sigma * sqrt_dt * Z)
		}
	}
	return S_paths
}

// Heston Stochastic Volatility Path Generation, full-truncation Euler (Lord,
// Koekkoek & van Dijk 2010): with v⁺ = max(v, 0) from the start of the step,
//   v ← v + κ(θ − v⁺)Δt + σ √v⁺ √Δt Z_v
//   S ← S exp((r − v⁺/2)Δt + √v⁺ √Δt Z_s),  Z_s = ρ Z_v + √(1−ρ²) Z_⊥
// so the discounted stock is a martingale step by step. V_paths holds v⁺.
mc_generate_heston_paths :: proc(
	S_0: f64,
	r: f64,
	T: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64, // Length: n_paths * n_steps * 2 (Z_v, Z_⊥ per step)
	allocator: mem.Allocator,
) -> (
	S_paths: []f64,
	V_paths: []f64,
) {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	rho_c := math.sqrt_f64(max(1.0 - params.rho * params.rho, 0.0))

	S_paths = make([]f64, n_paths * (n_steps + 1), allocator)
	V_paths = make([]f64, n_paths * (n_steps + 1), allocator)

	rand_idx := 0
	for path in 0 ..< n_paths {
		row := path * (n_steps + 1)
		S, v := S_0, params.v0
		S_paths[row] = S
		V_paths[row] = max(v, 0.0)
		for step in 1 ..< n_steps + 1 {
			Z_v := norm_data[rand_idx]
			Z_s := params.rho * Z_v + rho_c * norm_data[rand_idx + 1]
			rand_idx += 2
			vp := max(v, 0.0)
			sv := math.sqrt_f64(vp)
			S *= math.exp_f64((r - 0.5 * vp) * dt + sv * sqrt_dt * Z_s)
			v += params.kappa * (params.theta - vp) * dt + params.sigma * sv * sqrt_dt * Z_v
			S_paths[row + step] = S
			V_paths[row + step] = max(v, 0.0)
		}
	}

	return S_paths, V_paths
}

// Merton Jump Diffusion Path Generation, exact on the grid: per step the
// number of jumps is Poisson(λΔt) (inverse CDF of the uniform) and their total
// log size N μ_J + σ_J √N Z_J; the drift r − λκ − σ²/2, κ = e^{μ_J + σ_J²/2} − 1,
// makes the discounted stock a martingale.
mc_generate_mjd_paths :: proc(
	S_0: f64,
	r: f64,
	T: f64,
	params: MJD_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64, // Length: n_paths * n_steps * 2 (diffusion, jump size per step)
	unif_data: []f64, // Length: n_paths * n_steps
	allocator: mem.Allocator,
) -> []f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	kappa := math.exp_f64(params.mu_j + 0.5 * params.sigma_j * params.sigma_j) - 1.0
	drift := r - params.lambda * kappa - 0.5 * params.sigma * params.sigma
	lambda_dt := params.lambda * dt
	p0 := math.exp_f64(-lambda_dt)

	S_paths := make([]f64, n_paths * (n_steps + 1), allocator)

	norm_idx := 0
	unif_idx := 0
	for path in 0 ..< n_paths {
		row := path * (n_steps + 1)
		S_paths[row] = S_0
		for step in 1 ..< n_steps + 1 {
			Z := norm_data[norm_idx]
			Z_J := norm_data[norm_idx + 1]
			norm_idx += 2
			U := unif_data[unif_idx]
			unif_idx += 1

			// N = smallest k with P(N ≤ k) ≥ U
			n_jumps := 0
			pk, cdf := p0, p0
			for cdf < U && n_jumps < 1000 {
				n_jumps += 1
				pk *= lambda_dt / f64(n_jumps)
				cdf += pk
			}
			log_jump :=
				f64(n_jumps) * params.mu_j + params.sigma_j * math.sqrt_f64(f64(n_jumps)) * Z_J
			S_paths[row + step] =
				S_paths[row + step - 1] *
				math.exp_f64(drift * dt + params.sigma * sqrt_dt * Z + log_jump)
		}
	}

	return S_paths
}

// Hull-White 1F short-rate paths fitted to the flat curve e^{−r0 t}, sampled
// exactly: r(t) = x(t) + α(t), x an OU process from 0,
//   x_{k+1} = x_k e^{−aΔ} + σ √((1 − e^{−2aΔ})/(2a)) Z,  α(t) = r0 + σ²/(2a²)(1 − e^{−at})²
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
	a, sigma := params.a, params.sigma
	decay := math.exp_f64(-a * dt)
	sd := sigma * math.sqrt_f64(_hw_b(2.0 * a, dt))

	r_paths := make([]f64, n_paths * (n_steps + 1), allocator)
	alpha := make([]f64, n_steps + 1, allocator)
	defer delete(alpha, allocator)
	for i in 0 ..< n_steps + 1 {
		b := _hw_b(a, f64(i) * dt)
		alpha[i] = r0 + 0.5 * sigma * sigma * b * b
	}

	rand_idx := 0
	for path in 0 ..< n_paths {
		row := path * (n_steps + 1)
		x := 0.0
		r_paths[row] = r0
		for step in 1 ..< n_steps + 1 {
			x = x * decay + sd * norm_data[rand_idx]
			rand_idx += 1
			r_paths[row + step] = x + alpha[step]
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

// Running average of S_0 … S_step on a path of n_steps steps (for early exercise)
mc_compute_asian_average_to_step :: proc(
	S_paths: []f64,
	path: int,
	step: int,
	n_steps: int,
) -> f64 {
	row := path * (n_steps + 1)
	sum := 0.0
	for s in 0 ..= step {sum += S_paths[row + s]}
	return sum / f64(step + 1)
}

// Barrier Check: true if the barrier is touched at any grid date 1 … n_steps
// (discrete monitoring)
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

// One Longstaff–Schwartz step (updates cashflows in place). S_paths holds the
// spot at the exercise dates, stride n_exercise_dates + 1 (index 0 = S_0);
// cashflows are time-0 values. In-the-money paths regress the cashflow carried
// to t_ex on powers 0 … poly_degree of the moneyness S/S_0 − 1 and exercise
// when the payoff beats the fitted continuation value.
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
			pw := 1.0
			for q in 0 ..= poly_degree {
				X[itm_idx * (poly_degree + 1) + q] = pw
				pw *= x_moneyness
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
			cont_val := 0.0
			for q := poly_degree; q >= 0; q -= 1 {cont_val = cont_val * x_moneyness + beta[q]}

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

// CRN Delta/Gamma by central differences with h = 1% of S_0:
//   Δ = (P(S+h) − P(S−h)) / 2h,  Γ = (P(S+h) − 2P(S) + P(S−h)) / h²
// price_fn must reuse the same random numbers for every call
mc_compute_crn_delta_gamma :: proc(
	S_0: f64,
	price_fn: proc(_: f64) -> f64,
) -> (
	delta: f64,
	gamma: f64,
) {
	h := 0.01 * S_0
	up, mid, dn := price_fn(S_0 + h), price_fn(S_0), price_fn(S_0 - h)
	delta = (up - dn) / (2.0 * h)
	gamma = (up - 2.0 * mid + dn) / (h * h)
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
