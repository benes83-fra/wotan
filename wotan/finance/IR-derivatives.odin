package finance

import l "../linalg"
import opt "../optimize"
import t "../tensor"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// HULL-WHITE (1-FACTOR) INTEREST RATE MODEL
// ============================================================================

HW_Params :: struct {
	a:     f64, // Mean reversion speed
	sigma: f64, // Short rate volatility
}

// ============================================================================
// Tensorized Analytical Caplet Pricing (SIMD Optimized)
// ============================================================================
hull_white_caplet_price_tensor :: proc(
	a: f64,
	sigma: f64,
	T_start: ^t.Tensor,
	T_end: ^t.Tensor,
	F: ^t.Tensor,
	K: ^t.Tensor,
	P: ^t.Tensor,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	n := T_start.data.rows

	// Create an Nx1 tensor of 1.0s to strictly avoid any 1x1 broadcasting assumptions
	one_data := l.matrix_new(f64, n, 1, allocator)
	for i in 0 ..< n {one_data.data[i] = 1.0}
	one := t.tensor_new(one_data, false, allocator)
	defer t.tensor_free(one)

	delta_T := t.tensor_sub(T_end, T_start)

	// B(0, delta_T) = (1 - exp(-a * delta_T)) / a
	// Using tensor_scale for scalar * tensor to avoid dimension mismatch
	a_delta := t.tensor_scale(delta_T, a)
	exp_a_delta := t.tensor_exp(t.tensor_neg(a_delta))
	B_delta := t.tensor_scale(t.tensor_sub(one, exp_a_delta), 1.0 / a)

	// Variance of the forward rate
	two_a := 2.0 * a
	exp_2a_T := t.tensor_exp(t.tensor_scale(T_start, -two_a))
	var_factor := t.tensor_scale(t.tensor_sub(one, exp_2a_T), 1.0 / two_a)

	sigma_sq_val := sigma * sigma
	B_sq := t.tensor_mul(B_delta, B_delta) // Nx1 * Nx1

	// sigma_P_sq = sigma^2 * B_sq * var_factor
	sigma_P_sq := t.tensor_mul(t.tensor_scale(B_sq, sigma_sq_val), var_factor)
	sigma_P := t.tensor_sqrt(sigma_P_sq)

	// Black-like formula for Caplet
	F_over_K := t.tensor_div(F, K)
	ln_F_K := t.tensor_log(F_over_K)
	half_var := t.tensor_scale(sigma_P_sq, 0.5)

	numer := t.tensor_add(ln_F_K, half_var)
	d1 := t.tensor_div(numer, sigma_P)
	d2 := t.tensor_sub(d1, sigma_P)

	N_d1 := t.tensor_norm_cdf(d1)
	N_d2 := t.tensor_norm_cdf(d2)

	term1 := t.tensor_mul(F, N_d1)
	term2 := t.tensor_mul(K, N_d2)
	intrinsic := t.tensor_sub(term1, term2)

	payoff := t.tensor_relu(intrinsic)

	return t.tensor_mul(P, payoff)
}

// ============================================================================
// Hull-White Calibration to Caplet Strip (Finite Difference Optimizer)
// ============================================================================

HW_CalibrationResult :: struct {
	params:     HW_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

// Standalone helper to avoid illegal closure captures in Odin
_hw_eval_loss :: proc(
	a_val: f64,
	sigma_val: f64,
	T_start_t: ^t.Tensor,
	T_end_t: ^t.Tensor,
	F_t: ^t.Tensor,
	K_t: ^t.Tensor,
	P_t: ^t.Tensor,
	market_t: ^t.Tensor,
	allocator: mem.Allocator,
) -> f64 {
	loss_t := hull_white_caplet_price_tensor(
		a_val,
		sigma_val,
		T_start_t,
		T_end_t,
		F_t,
		K_t,
		P_t,
		allocator,
	)

	d := t.tensor_sub(loss_t, market_t)
	d_sq := t.tensor_mul(d, d)
	mse := t.tensor_mean(d_sq)
	r := t.tensor_sqrt(mse)

	res := r.data.data[0]

	t.tensor_free_graph(r)
	return res
}

calibrate_hull_white :: proc(
	market_prices: []f64,
	T_start: []f64,
	T_end: []f64,
	F: []f64,
	K: []f64,
	P: []f64,
	n_caplets: int,
	allocator: mem.Allocator = context.allocator,
) -> HW_CalibrationResult {
	T_start_t := _make_const_tensor(T_start, n_caplets, allocator)
	T_end_t := _make_const_tensor(T_end, n_caplets, allocator)
	F_t := _make_const_tensor(F, n_caplets, allocator)
	K_t := _make_const_tensor(K, n_caplets, allocator)
	P_t := _make_const_tensor(P, n_caplets, allocator)
	market_t := _make_const_tensor(market_prices, n_caplets, allocator)

	opt_config := opt.OptimizerConfig {
		type          = .Adam,
		learning_rate = 0.05,
		beta1         = 0.9,
		beta2         = 0.999,
		epsilon       = 1e-8,
	}
	optimizer := opt.optimizer_init(opt_config, 2, allocator)
	defer opt.optimizer_free(&optimizer)

	gradient := make([]f64, 2, allocator)
	defer delete(gradient, allocator)

	eps := 1e-5
	best_loss: f64 = 1e10 // Explicitly typed as f64
	best_a := 0.05
	best_sigma := 0.01
	max_iter := 300
	converged := false

	a_val := 0.05
	sigma_val := 0.01

	for iter in 0 ..< max_iter {
		current_loss := _hw_eval_loss(
			a_val,
			sigma_val,
			T_start_t,
			T_end_t,
			F_t,
			K_t,
			P_t,
			market_t,
			allocator,
		)

		if current_loss < best_loss {
			best_loss = current_loss
			best_a = a_val
			best_sigma = sigma_val
		}

		if current_loss < 1e-6 {
			converged = true
			break
		}

		a_plus := a_val + eps
		a_minus := a_val - eps
		sigma_plus := sigma_val + eps
		sigma_minus := sigma_val - eps

		gradient[0] =
			(_hw_eval_loss(
					a_plus,
					sigma_val,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				) -
				_hw_eval_loss(
					a_minus,
					sigma_val,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		gradient[1] =
			(_hw_eval_loss(
					a_val,
					sigma_plus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				) -
				_hw_eval_loss(
					a_val,
					sigma_minus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		params := make([]f64, 2, allocator)
		params[0] = a_val
		params[1] = sigma_val
		opt.optimizer_step(&optimizer, params, gradient)

		// ✅ FIXED: Using standard math.max and math.min as per Odin's library
		a_val = math.max(0.01, math.min(0.5, params[0]))
		sigma_val = math.max(0.001, math.min(0.1, params[1]))

		delete(params, allocator)
	}

	t.tensor_free(T_start_t)
	t.tensor_free(T_end_t)
	t.tensor_free(F_t)
	t.tensor_free(K_t)
	t.tensor_free(P_t)
	t.tensor_free(market_t)

	return HW_CalibrationResult {
		params = HW_Params{a = best_a, sigma = best_sigma},
		rmse = best_loss,
		converged = converged,
		iterations = max_iter,
	}
}

_make_const_tensor :: proc(data: []f64, n: int, allocator: mem.Allocator) -> ^t.Tensor {
	mat := l.matrix_new(f64, n, 1, allocator)
	for i in 0 ..< n {
		mat.data[i] = data[i]
	}
	return t.tensor_new(mat, false, allocator)
}
// ============================================================================
// Hull-White Monte Carlo: Cap Option (with CRN Greeks)
// ============================================================================
_hw_mc_cap_price_helper :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_caplets: int,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64,
) -> f64 {
	T_max := T_end[n_caplets - 1]
	dt := T_max / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	total_payoff := 0.0

	norm_idx := 0
	for path in 0 ..< n_paths {
		r := r0
		cap_payoff := 0.0
		next_caplet_idx := 0

		// Simulate ONE continuous path up to T_max
		for step in 1 ..< n_steps + 1 {
			Z := norm_data[norm_idx]
			norm_idx += 1

			// Euler step for Hull-White / Vasicek
			dr := params.a * (r0 - r) * dt + params.sigma * sqrt_dt * Z
			r = r + dr

			t_current := f64(step) * dt

			// Check if we have reached or passed the start of the next caplet
			for next_caplet_idx < n_caplets && t_current >= T_start[next_caplet_idx] {
				i := next_caplet_idx

				// Caplet payoff at T_start[i]
				disc := math.exp_f64(-r * (T_end[i] - T_start[i]))
				fwd_rate := (1.0 / disc - 1.0) / delta[i]
				payoff := delta[i] * math.max(fwd_rate - K, 0.0) * disc

				// Discount back to t=0
				cap_payoff += payoff * math.exp_f64(-r0 * T_start[i])

				next_caplet_idx += 1
			}
		}
		total_payoff += cap_payoff
	}
	return total_payoff / f64(n_paths)
}

hw_mc_cap_option :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_caplets: int,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	// ✅ FIXED: Allocates exactly n_paths * n_steps random numbers
	norm_count := n_paths * n_steps
	norm_data := make([]f64, norm_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< norm_count {norm_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _hw_mc_cap_price_helper(
		r0,
		K,
		T_start,
		T_end,
		delta,
		n_caplets,
		params,
		n_paths,
		n_steps,
		norm_data,
	)

	h_r0 := 0.01 * r0
	delta_r =
		(_hw_mc_cap_price_helper(
				r0 + h_r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw_mc_cap_price_helper(
				r0 - h_r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params,
				n_paths,
				n_steps,
				norm_data,
			)) /
		(2.0 * h_r0)

	h_sigma := 0.01 * params.sigma
	if h_sigma < 0.001 {h_sigma = 0.001}
	params_up := params; params_up.sigma = params.sigma + h_sigma
	params_dn := params; params_dn.sigma = params.sigma - h_sigma

	vega =
		(_hw_mc_cap_price_helper(
				r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params_up,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw_mc_cap_price_helper(
				r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params_dn,
				n_paths,
				n_steps,
				norm_data,
			)) /
		(2.0 * h_sigma)

	return price, delta_r, vega
}

// ============================================================================
// HULL-WHITE 2-FACTOR INTEREST RATE MODEL (Brigo-Mercurio formulation)
// ============================================================================

HW2_Params :: struct {
	a:     f64, // Mean reversion of short rate (fast)
	b:     f64, // Mean reversion of second factor (slow)
	sigma: f64, // Volatility of short rate
	eta:   f64, // Volatility of second factor
	rho:   f64, // Correlation between Brownian motions
}

// ============================================================================
// Tensorized Analytical Caplet Pricing (SIMD Optimized) - 2 Factor
// ============================================================================
hull_white_2f_caplet_price_tensor :: proc(
	params: HW2_Params,
	T_start: ^t.Tensor,
	T_end: ^t.Tensor,
	F: ^t.Tensor,
	K: ^t.Tensor,
	P: ^t.Tensor,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	n := T_start.data.rows

	// Create Nx1 tensor of 1.0s
	one_data := l.matrix_new(f64, n, 1, allocator)
	for i in 0 ..< n {one_data.data[i] = 1.0}
	one := t.tensor_new(one_data, false, allocator)
	defer t.tensor_free(one)

	delta_T := t.tensor_sub(T_end, T_start)

	// B(a; T_start, T_end) = (1 - exp(-a * delta_T)) / a
	a_delta := t.tensor_scale(delta_T, params.a)
	exp_a_delta := t.tensor_exp(t.tensor_neg(a_delta))
	B_a := t.tensor_scale(t.tensor_sub(one, exp_a_delta), 1.0 / params.a)

	// B(b; T_start, T_end) = (1 - exp(-b * delta_T)) / b
	b_delta := t.tensor_scale(delta_T, params.b)
	exp_b_delta := t.tensor_exp(t.tensor_neg(b_delta))
	B_b := t.tensor_scale(t.tensor_sub(one, exp_b_delta), 1.0 / params.b)

	// Variance of forward rate (Brigo-Mercurio formula)
	// Term 1: sigma^2 * B_a^2 * T_start
	sigma_sq := params.sigma * params.sigma
	B_a_sq := t.tensor_mul(B_a, B_a)
	term1 := t.tensor_scale(t.tensor_mul(B_a_sq, T_start), sigma_sq)

	// Term 2: eta^2 * B_b^2 * (1 - exp(-2*b*T_start)) / (2*b)
	eta_sq := params.eta * params.eta
	B_b_sq := t.tensor_mul(B_b, B_b)
	exp_2b_T := t.tensor_exp(t.tensor_scale(T_start, -2.0 * params.b))
	var_factor_b := t.tensor_scale(t.tensor_sub(one, exp_2b_T), 1.0 / (2.0 * params.b))
	term2 := t.tensor_scale(t.tensor_mul(B_b_sq, var_factor_b), eta_sq)

	// Term 3: 2*rho*sigma*eta * B_a * B_b * (1 - exp(-(a+b)*T_start)) / (a+b)
	cross_coeff := 2.0 * params.rho * params.sigma * params.eta
	B_a_B_b := t.tensor_mul(B_a, B_b)
	exp_ab_T := t.tensor_exp(t.tensor_scale(T_start, -(params.a + params.b)))
	cross_factor := t.tensor_scale(t.tensor_sub(one, exp_ab_T), 1.0 / (params.a + params.b))
	term3 := t.tensor_scale(t.tensor_mul(B_a_B_b, cross_factor), cross_coeff)

	// Total variance
	sigma_P_sq := t.tensor_add(t.tensor_add(term1, term2), term3)
	sigma_P := t.tensor_sqrt(sigma_P_sq)

	// Black-like formula for Caplet
	F_over_K := t.tensor_div(F, K)
	ln_F_K := t.tensor_log(F_over_K)
	half_var := t.tensor_scale(sigma_P_sq, 0.5)

	numer := t.tensor_add(ln_F_K, half_var)
	d1 := t.tensor_div(numer, sigma_P)
	d2 := t.tensor_sub(d1, sigma_P)

	N_d1 := t.tensor_norm_cdf(d1)
	N_d2 := t.tensor_norm_cdf(d2)

	term1_payoff := t.tensor_mul(F, N_d1)
	term2_payoff := t.tensor_mul(K, N_d2)
	intrinsic := t.tensor_sub(term1_payoff, term2_payoff)

	payoff := t.tensor_relu(intrinsic)

	return t.tensor_mul(P, payoff)
}

// ============================================================================
// Hull-White 2-Factor Calibration (Finite Difference Optimizer)
// ============================================================================

HW2_CalibrationResult :: struct {
	params:     HW2_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

// Standalone helper to avoid illegal closure captures in Odin
_hw2_eval_loss :: proc(
	p: HW2_Params,
	T_start_t: ^t.Tensor,
	T_end_t: ^t.Tensor,
	F_t: ^t.Tensor,
	K_t: ^t.Tensor,
	P_t: ^t.Tensor,
	market_t: ^t.Tensor,
	allocator: mem.Allocator,
) -> f64 {
	loss_t := hull_white_2f_caplet_price_tensor(p, T_start_t, T_end_t, F_t, K_t, P_t, allocator)

	d := t.tensor_sub(loss_t, market_t)
	d_sq := t.tensor_mul(d, d)
	mse := t.tensor_mean(d_sq)
	r := t.tensor_sqrt(mse)

	res := r.data.data[0]

	t.tensor_free_graph(r)
	return res
}

calibrate_hull_white_2f :: proc(
	market_prices: []f64,
	T_start: []f64,
	T_end: []f64,
	F: []f64,
	K: []f64,
	P: []f64,
	n_caplets: int,
	allocator: mem.Allocator = context.allocator,
) -> HW2_CalibrationResult {
	T_start_t := _make_const_tensor(T_start, n_caplets, allocator)
	T_end_t := _make_const_tensor(T_end, n_caplets, allocator)
	F_t := _make_const_tensor(F, n_caplets, allocator)
	K_t := _make_const_tensor(K, n_caplets, allocator)
	P_t := _make_const_tensor(P, n_caplets, allocator)
	market_t := _make_const_tensor(market_prices, n_caplets, allocator)

	opt_config := opt.OptimizerConfig {
		type          = .Adam,
		learning_rate = 0.02,
		beta1         = 0.9,
		beta2         = 0.999,
		epsilon       = 1e-8,
	}
	optimizer := opt.optimizer_init(opt_config, 5, allocator)
	defer opt.optimizer_free(&optimizer)

	gradient := make([]f64, 5, allocator)
	defer delete(gradient, allocator)

	eps := 1e-4
	best_loss: f64 = 1e10
	best_params := HW2_Params {
		a     = 0.1,
		b     = 0.01,
		sigma = 0.01,
		eta   = 0.005,
		rho   = -0.3,
	}
	max_iter := 400
	converged := false

	p := HW2_Params {
		a     = 0.1,
		b     = 0.01,
		sigma = 0.01,
		eta   = 0.005,
		rho   = -0.3,
	}

	for iter in 0 ..< max_iter {
		current_loss := _hw2_eval_loss(p, T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator)

		if current_loss < best_loss {
			best_loss = current_loss
			best_params = p
		}

		if current_loss < 1e-6 {
			converged = true
			break
		}

		// Finite difference gradients for all 5 parameters
		p_a_plus := p; p_a_plus.a += eps
		p_a_minus := p; p_a_minus.a -= eps
		gradient[0] =
			(_hw2_eval_loss(p_a_plus, T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator) -
				_hw2_eval_loss(
					p_a_minus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		p_b_plus := p; p_b_plus.b += eps
		p_b_minus := p; p_b_minus.b -= eps
		gradient[1] =
			(_hw2_eval_loss(p_b_plus, T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator) -
				_hw2_eval_loss(
					p_b_minus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		p_sigma_plus := p; p_sigma_plus.sigma += eps
		p_sigma_minus := p; p_sigma_minus.sigma -= eps
		gradient[2] =
			(_hw2_eval_loss(p_sigma_plus, T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator) -
				_hw2_eval_loss(
					p_sigma_minus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		p_eta_plus := p; p_eta_plus.eta += eps
		p_eta_minus := p; p_eta_minus.eta -= eps
		gradient[3] =
			(_hw2_eval_loss(p_eta_plus, T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator) -
				_hw2_eval_loss(
					p_eta_minus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		p_rho_plus := p; p_rho_plus.rho += eps
		p_rho_minus := p; p_rho_minus.rho -= eps
		gradient[4] =
			(_hw2_eval_loss(p_rho_plus, T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator) -
				_hw2_eval_loss(
					p_rho_minus,
					T_start_t,
					T_end_t,
					F_t,
					K_t,
					P_t,
					market_t,
					allocator,
				)) /
			(2.0 * eps)

		params := make([]f64, 5, allocator)
		params[0] = p.a
		params[1] = p.b
		params[2] = p.sigma
		params[3] = p.eta
		params[4] = p.rho
		opt.optimizer_step(&optimizer, params, gradient)

		// Enforce bounds
		p.a = math.max(0.01, math.min(1.0, params[0]))
		p.b = math.max(0.01, math.min(1.0, params[1]))
		p.sigma = math.max(0.001, math.min(0.05, params[2]))
		p.eta = math.max(0.001, math.min(0.05, params[3]))
		p.rho = math.max(-0.99, math.min(0.99, params[4]))

		delete(params, allocator)
	}

	t.tensor_free(T_start_t)
	t.tensor_free(T_end_t)
	t.tensor_free(F_t)
	t.tensor_free(K_t)
	t.tensor_free(P_t)
	t.tensor_free(market_t)

	return HW2_CalibrationResult {
		params = best_params,
		rmse = best_loss,
		converged = converged,
		iterations = max_iter,
	}
}

// ============================================================================
// Hull-White 2-Factor Monte Carlo: Cap Option (with CRN Greeks)
// ============================================================================
_hw2_mc_cap_price_helper :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_caplets: int,
	params: HW2_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64,
) -> f64 {
	T_max := T_end[n_caplets - 1]
	dt := T_max / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	total_payoff := 0.0

	// Cholesky decomposition for correlated Brownian motions
	cholesky_21 := params.rho
	cholesky_22 := math.sqrt_f64(1.0 - params.rho * params.rho)

	norm_idx := 0
	for path in 0 ..< n_paths {
		r := r0
		x := 0.0 // Second factor starts at 0
		cap_payoff := 0.0
		next_caplet_idx := 0

		for step in 1 ..< n_steps + 1 {
			U1 := norm_data[norm_idx]
			U2 := norm_data[norm_idx + 1]
			norm_idx += 2

			// Correlated Brownian motions
			Z1 := U1
			Z2 := cholesky_21 * U1 + cholesky_22 * U2

			t_current := f64(step) * dt

			// Theta function for flat initial curve
			theta_t :=
				params.a * r0 +
				(params.sigma * params.sigma / (2.0 * params.a)) *
					(1.0 - math.exp_f64(-2.0 * params.a * t_current))

			// Euler step for r(t)
			dr := (theta_t + x - params.a * r) * dt + params.sigma * sqrt_dt * Z1
			r = r + dr

			// Euler step for x(t)
			dx := -params.b * x * dt + params.eta * sqrt_dt * Z2
			x = x + dx

			// Check caplet dates
			for next_caplet_idx < n_caplets && t_current >= T_start[next_caplet_idx] {
				i := next_caplet_idx

				// Caplet payoff at T_start[i]
				disc := math.exp_f64(-r * (T_end[i] - T_start[i]))
				fwd_rate := (1.0 / disc - 1.0) / delta[i]
				payoff := delta[i] * math.max(fwd_rate - K, 0.0) * disc

				// Discount back to t=0
				cap_payoff += payoff * math.exp_f64(-r0 * T_start[i])

				next_caplet_idx += 1
			}
		}
		total_payoff += cap_payoff
	}
	return total_payoff / f64(n_paths)
}

hw2_mc_cap_option :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_caplets: int,
	params: HW2_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	// 2 random numbers per step per path (for Cholesky)
	norm_count := n_paths * n_steps * 2
	norm_data := make([]f64, norm_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< norm_count {norm_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _hw2_mc_cap_price_helper(
		r0,
		K,
		T_start,
		T_end,
		delta,
		n_caplets,
		params,
		n_paths,
		n_steps,
		norm_data,
	)

	h_r0 := 0.01 * r0
	delta_r =
		(_hw2_mc_cap_price_helper(
				r0 + h_r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw2_mc_cap_price_helper(
				r0 - h_r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params,
				n_paths,
				n_steps,
				norm_data,
			)) /
		(2.0 * h_r0)

	h_sigma := 0.01 * params.sigma
	if h_sigma < 0.001 {h_sigma = 0.001}
	params_up := params; params_up.sigma = params.sigma + h_sigma
	params_dn := params; params_dn.sigma = params.sigma - h_sigma

	vega =
		(_hw2_mc_cap_price_helper(
				r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params_up,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw2_mc_cap_price_helper(
				r0,
				K,
				T_start,
				T_end,
				delta,
				n_caplets,
				params_dn,
				n_paths,
				n_steps,
				norm_data,
			)) /
		(2.0 * h_sigma)

	return price, delta_r, vega
}
