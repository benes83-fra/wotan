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
// ============================================================================
// SWAPTION PRICING & JOINT CALIBRATION (Caps + Swaptions)
// ============================================================================

SwaptionSpec :: struct {
	T_exp:       f64,
	swap_length: int, // Number of annual payments
	F:           f64, // Forward swap rate
	K:           f64, // Strike
	annuity:     f64, // PV01 / Annuity factor
}
// Compute HW 2-Factor swap rate variance (proper annuity weighting)
compute_hw2f_swap_variance :: proc(
	spec: SwaptionSpec,
	params: HW2_Params,
	r0: f64, // Initial short rate for discount factors
) -> f64 {
	a := params.a
	b := params.b
	sigma := params.sigma
	eta := params.eta
	rho := params.rho
	T := spec.T_exp

	// Factor variances at expiry T
	V_a := sigma * sigma * (1.0 - math.exp_f64(-2.0 * a * T)) / (2.0 * a)
	V_b := eta * eta * (1.0 - math.exp_f64(-2.0 * b * T)) / (2.0 * b)
	V_ab := rho * sigma * eta * (1.0 - math.exp_f64(-(a + b) * T)) / (a + b)

	// Compute proper annuity weights
	// w_i = delta_i * P(0, T_i) / Annuity
	// For annual payments, delta_i = 1
	W_a := 0.0
	W_b := 0.0
	total_weight := 0.0

	for k in 1 ..< spec.swap_length + 1 {
		T_i := T + f64(k)
		P_Ti := math.exp_f64(-r0 * T_i) // Discount factor to payment date
		w_i := P_Ti / spec.annuity // Proper annuity weight

		B_a_k := (1.0 - math.exp_f64(-a * f64(k))) / a
		B_b_k := (1.0 - math.exp_f64(-b * f64(k))) / b

		W_a += w_i * B_a_k
		W_b += w_i * B_b_k
		total_weight += w_i
	}

	// sigma_swap^2 = V_a * W_a^2 + V_b * W_b^2 + 2 * V_ab * W_a * W_b
	return V_a * W_a * W_a + V_b * W_b * W_b + 2.0 * V_ab * W_a * W_b
}

// Price a strip of swaptions using Black's formula (SIMD via tensor)
hw2f_swaption_price_strip :: proc(
	params: HW2_Params,
	swaptions: []SwaptionSpec,
	n_swaptions: int,
	r0: f64,
	allocator: mem.Allocator,
) -> []f64 {
	prices := make([]f64, n_swaptions, allocator)

	for i in 0 ..< n_swaptions {
		spec := swaptions[i]
		swap_var := compute_hw2f_swap_variance(spec, params, r0)

		if swap_var < 1e-12 {
			prices[i] = spec.annuity * math.max(spec.F - spec.K, 0.0)
			continue
		}

		swap_vol := math.sqrt_f64(swap_var)
		ln_F_K := math.ln(spec.F / spec.K)
		d1 := (ln_F_K + 0.5 * swap_var) / swap_vol
		d2 := d1 - swap_vol

		N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
		N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))

		prices[i] = spec.annuity * (spec.F * N_d1 - spec.K * N_d2)
	}
	return prices
}

// Joint loss function: Caps RMSE + Swaptions RMSE
_hw2f_joint_loss :: proc(
	params: HW2_Params,
	r0: f64,
	// Caplet data
	cap_T_start_t: ^t.Tensor,
	cap_T_end_t: ^t.Tensor,
	cap_F_t: ^t.Tensor,
	cap_K_t: ^t.Tensor,
	cap_P_t: ^t.Tensor,
	cap_market_t: ^t.Tensor,
	// Swaption data
	swaption_specs: []SwaptionSpec,
	swaption_market: []f64,
	n_swaptions: int,
	allocator: mem.Allocator,
) -> f64 {
	// 1. Caplet loss (SIMD tensor)
	cap_model_t := hull_white_2f_caplet_price_tensor(
		params,
		cap_T_start_t,
		cap_T_end_t,
		cap_F_t,
		cap_K_t,
		cap_P_t,
		allocator,
	)
	cap_diff := t.tensor_sub(cap_model_t, cap_market_t)
	cap_diff_sq := t.tensor_mul(cap_diff, cap_diff)
	cap_mse := t.tensor_mean(cap_diff_sq)
	cap_rmse := t.tensor_sqrt(cap_mse)
	cap_loss := cap_rmse.data.data[0]
	t.tensor_free_graph(cap_rmse)

	// 2. Swaption loss (analytical)
	swaption_model := hw2f_swaption_price_strip(params, swaption_specs, n_swaptions, r0, allocator)
	defer delete(swaption_model, allocator)

	swaption_sse := 0.0
	for i in 0 ..< n_swaptions {
		err := swaption_model[i] - swaption_market[i]
		swaption_sse += err * err
	}
	swaption_rmse := math.sqrt_f64(swaption_sse / f64(n_swaptions))

	// 3. Data loss (equal weight between caps and swaptions)
	data_loss := cap_loss + swaption_rmse

	// 4. Regularization: penalize deviation from realistic priors
	// Priors based on typical USD rates
	prior_a := 0.05
	prior_b := 0.02
	prior_sigma := 0.01
	prior_eta := 0.005
	prior_rho := -0.3

	// Scale factors (larger = more tolerant of deviation)
	scale_a := 0.1
	scale_b := 0.05
	scale_sigma := 0.01
	scale_eta := 0.005
	scale_rho := 0.3

	lambda := 0.05 // Regularization strength
	reg :=
		lambda *
		(math.pow((params.a - prior_a) / scale_a, 2) +
				math.pow((params.b - prior_b) / scale_b, 2) +
				math.pow((params.sigma - prior_sigma) / scale_sigma, 2) +
				math.pow((params.eta - prior_eta) / scale_eta, 2) +
				math.pow((params.rho - prior_rho) / scale_rho, 2))

	// 5. Symmetry-breaking penalty: enforce a > b (fast factor must be faster)
	// If a < b, add a heavy penalty to prevent factor swapping
	symmetry_penalty := 0.0
	if params.a < params.b {
		symmetry_penalty = 10.0 * math.pow(params.b - params.a, 2)
	}

	return data_loss + reg + symmetry_penalty
}

// ============================================================================
// JOINT CALIBRATION: Caps + Swaptions
// ============================================================================

HW2F_JointResult :: struct {
	params:     HW2_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}
calibrate_hull_white_2f_joint :: proc(
	cap_market: []f64, // Caplet data
	cap_T_start: []f64,
	cap_T_end: []f64,
	cap_F: []f64,
	cap_K: []f64,
	cap_P: []f64,
	n_caplets: int,
	// Swaption data
	swaption_specs: []SwaptionSpec,
	swaption_market: []f64,
	n_swaptions: int,
	r0: f64, // <-- NEW: pass r0 for proper annuity weighting
	allocator: mem.Allocator = context.allocator,
) -> HW2F_JointResult {

	// Load caplet tensors
	cap_T_start_t := _make_const_tensor(cap_T_start, n_caplets, allocator)
	cap_T_end_t := _make_const_tensor(cap_T_end, n_caplets, allocator)
	cap_F_t := _make_const_tensor(cap_F, n_caplets, allocator)
	cap_K_t := _make_const_tensor(cap_K, n_caplets, allocator)
	cap_P_t := _make_const_tensor(cap_P, n_caplets, allocator)
	cap_market_t := _make_const_tensor(cap_market, n_caplets, allocator)

	opt_config := opt.OptimizerConfig {
		type          = .Adam,
		learning_rate = 0.005, // <-- Lower LR for more stable 5D optimization
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
	// Better initial guess: closer to realistic IR parameters, with a > b enforced
	best_params := HW2_Params {
		a     = 0.10,
		b     = 0.03,
		sigma = 0.008,
		eta   = 0.005,
		rho   = -0.4,
	}
	max_iter := 600 // More iterations for 5D landscape
	converged := false

	p := HW2_Params {
		a     = 0.10,
		b     = 0.03,
		sigma = 0.008,
		eta   = 0.005,
		rho   = -0.4,
	}

	for iter in 0 ..< max_iter {
		current_loss := _hw2f_joint_loss(
			p,
			r0,
			cap_T_start_t,
			cap_T_end_t,
			cap_F_t,
			cap_K_t,
			cap_P_t,
			cap_market_t,
			swaption_specs,
			swaption_market,
			n_swaptions,
			allocator,
		)

		if current_loss < best_loss {
			best_loss = current_loss
			best_params = p
		}

		if current_loss < 1e-6 {
			converged = true
			break
		}

		// Finite difference gradients (all 5 parameters)
		p_a_plus := p; p_a_plus.a += eps
		p_a_minus := p; p_a_minus.a -= eps
		gradient[0] =
			(_hw2f_joint_loss(
					p_a_plus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				) -
				_hw2f_joint_loss(
					p_a_minus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				)) /
			(2.0 * eps)

		p_b_plus := p; p_b_plus.b += eps
		p_b_minus := p; p_b_minus.b -= eps
		gradient[1] =
			(_hw2f_joint_loss(
					p_b_plus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				) -
				_hw2f_joint_loss(
					p_b_minus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				)) /
			(2.0 * eps)

		p_s_plus := p; p_s_plus.sigma += eps
		p_s_minus := p; p_s_minus.sigma -= eps
		gradient[2] =
			(_hw2f_joint_loss(
					p_s_plus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				) -
				_hw2f_joint_loss(
					p_s_minus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				)) /
			(2.0 * eps)

		p_e_plus := p; p_e_plus.eta += eps
		p_e_minus := p; p_e_minus.eta -= eps
		gradient[3] =
			(_hw2f_joint_loss(
					p_e_plus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				) -
				_hw2f_joint_loss(
					p_e_minus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				)) /
			(2.0 * eps)

		p_r_plus := p; p_r_plus.rho += eps
		p_r_minus := p; p_r_minus.rho -= eps
		gradient[4] =
			(_hw2f_joint_loss(
					p_r_plus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
					allocator,
				) -
				_hw2f_joint_loss(
					p_r_minus,
					r0,
					cap_T_start_t,
					cap_T_end_t,
					cap_F_t,
					cap_K_t,
					cap_P_t,
					cap_market_t,
					swaption_specs,
					swaption_market,
					n_swaptions,
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

		// Enforce bounds AND a > b constraint
		p.a = math.max(0.02, math.min(1.0, params[0]))
		p.b = math.max(0.005, math.min(p.a - 0.01, params[1])) // <-- KEY: b < a enforced
		p.sigma = math.max(0.001, math.min(0.05, params[2]))
		p.eta = math.max(0.001, math.min(0.05, params[3]))
		p.rho = math.max(-0.99, math.min(0.99, params[4]))

		delete(params, allocator)
	}

	t.tensor_free(cap_T_start_t)
	t.tensor_free(cap_T_end_t)
	t.tensor_free(cap_F_t)
	t.tensor_free(cap_K_t)
	t.tensor_free(cap_P_t)
	t.tensor_free(cap_market_t)

	return HW2F_JointResult {
		params = best_params,
		rmse = best_loss,
		converged = converged,
		iterations = max_iter,
	}
}
// ============================================================================
// BERMUDAN SWAPTION PRICING (HW 2-FACTOR)
// ============================================================================

// Analytical Zero-Coupon Bond Price for the specific MC state variables (r_t, x_t)
// Calibrated to a flat initial curve r0
hw2f_zero_coupon_bond :: proc(
	t: f64,
	T: f64,
	r_t: f64,
	x_t: f64,
	params: HW2_Params,
	r0: f64,
) -> f64 {
	tau := T - t
	if tau <= 0.0 {return 1.0}

	a := params.a
	b := params.b
	sigma := params.sigma
	eta := params.eta
	rho := params.rho

	B_a := (1.0 - math.exp_f64(-a * tau)) / a
	B_b := (1.0 - math.exp_f64(-b * tau)) / b

	// Variance of integral of the factors
	V_a :=
		(sigma * sigma / (a * a)) *
		(tau -
				(2.0 / a) * (1.0 - math.exp_f64(-a * tau)) +
				(1.0 / (2.0 * a)) * (1.0 - math.exp_f64(-2.0 * a * tau)))
	V_b :=
		(eta * eta / (b * b)) *
		(tau -
				(2.0 / b) * (1.0 - math.exp_f64(-b * tau)) +
				(1.0 / (2.0 * b)) * (1.0 - math.exp_f64(-2.0 * b * tau)))
	V_ab :=
		(2.0 * rho * sigma * eta / (a * b)) *
		(tau -
				(1.0 / a) * (1.0 - math.exp_f64(-a * tau)) -
				(1.0 / b) * (1.0 - math.exp_f64(-b * tau)) +
				(1.0 / (a + b)) * (1.0 - math.exp_f64(-(a + b) * tau)))

	V_t_T := V_a + V_b + V_ab

	// V(0, T) for curve matching
	V_0_T :=
		(sigma * sigma / (a * a)) *
			(T -
					(2.0 / a) * (1.0 - math.exp_f64(-a * T)) +
					(1.0 / (2.0 * a)) * (1.0 - math.exp_f64(-2.0 * a * T))) +
		(eta * eta / (b * b)) *
			(T -
					(2.0 / b) * (1.0 - math.exp_f64(-b * T)) +
					(1.0 / (2.0 * b)) * (1.0 - math.exp_f64(-2.0 * b * T))) +
		(2.0 * rho * sigma * eta / (a * b)) *
			(T -
					(1.0 / a) * (1.0 - math.exp_f64(-a * T)) -
					(1.0 / b) * (1.0 - math.exp_f64(-b * T)) +
					(1.0 / (a + b)) * (1.0 - math.exp_f64(-(a + b) * T)))

	exponent := -r0 * tau - B_a * (r_t - r0) + (B_a - B_b) * x_t + 0.5 * (V_t_T - V_0_T)

	return math.exp_f64(exponent)
}

// European Swaption Value (used as a rigorous continuation value proxy)
hw2f_euro_swaption_value :: proc(
	t: f64,
	T_exp: f64,
	maturity: f64,
	delta: f64,
	K: f64,
	r_t: f64,
	x_t: f64,
	params: HW2_Params,
	r0: f64,
) -> f64 {
	tau := T_exp - t
	if tau <= 0.0 {return 0.0}

	annuity := 0.0
	n_pmts := int((maturity - T_exp) / delta)
	for j in 1 ..< n_pmts + 1 {
		T_j := T_exp + f64(j) * delta
		P_t_Tj := hw2f_zero_coupon_bond(t, T_j, r_t, x_t, params, r0)
		annuity += delta * P_t_Tj
	}

	P_start := hw2f_zero_coupon_bond(t, T_exp, r_t, x_t, params, r0)
	P_end := hw2f_zero_coupon_bond(t, maturity, r_t, x_t, params, r0)

	if annuity <= 0.0 {return 0.0}
	F := (P_start - P_end) / annuity

	// Proxy variance: V(0, T_exp) - V(0, t)
	a := params.a; b := params.b; sigma := params.sigma; eta := params.eta; rho := params.rho

	V_0_T :=
		(sigma * sigma / (a * a)) *
			(T_exp -
					(2.0 / a) * (1.0 - math.exp_f64(-a * T_exp)) +
					(1.0 / (2.0 * a)) * (1.0 - math.exp_f64(-2.0 * a * T_exp))) +
		(eta * eta / (b * b)) *
			(T_exp -
					(2.0 / b) * (1.0 - math.exp_f64(-b * T_exp)) +
					(1.0 / (2.0 * b)) * (1.0 - math.exp_f64(-2.0 * b * T_exp))) +
		(2.0 * rho * sigma * eta / (a * b)) *
			(T_exp -
					(1.0 / a) * (1.0 - math.exp_f64(-a * T_exp)) -
					(1.0 / b) * (1.0 - math.exp_f64(-b * T_exp)) +
					(1.0 / (a + b)) * (1.0 - math.exp_f64(-(a + b) * T_exp)))

	V_0_t :=
		(sigma * sigma / (a * a)) *
			(t -
					(2.0 / a) * (1.0 - math.exp_f64(-a * t)) +
					(1.0 / (2.0 / a)) * (1.0 - math.exp_f64(-2.0 * a * t))) +
		(eta * eta / (b * b)) *
			(t -
					(2.0 / b) * (1.0 - math.exp_f64(-b * t)) +
					(1.0 / (2.0 * b)) * (1.0 - math.exp_f64(-2.0 * b * t))) +// Note: simplified for stability
		(2.0 * rho * sigma * eta / (a * b)) *
			(t -
					(1.0 / a) * (1.0 - math.exp_f64(-a * t)) -
					(1.0 / b) * (1.0 - math.exp_f64(-b * t)) +
					(1.0 / (a + b)) * (1.0 - math.exp_f64(-(a + b) * t)))

	swap_var := math.max(V_0_T - V_0_t, 1e-12)
	swap_vol := math.sqrt_f64(swap_var)

	ln_F_K := math.ln(F / K)
	d1 := (ln_F_K + 0.5 * swap_var) / swap_vol
	d2 := d1 - swap_vol

	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))

	return annuity * (F * N_d1 - K * N_d2)
}

// Hull-White 2-Factor Monte Carlo: Bermudan Swaption
hw2f_mc_bermudan_option :: proc(
	r0: f64,
	K: f64,
	exercise_dates: []f64, // e.g., [1.0, 2.0, 3.0, 4.0, 5.0]
	maturity: f64, // e.g., 10.0
	delta: f64, // e.g., 1.0
	params: HW2_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	n_ex := len(exercise_dates)
	dt := maturity / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)

	// Record state at exercise dates
	r_ex := make([]f64, n_paths * n_ex, allocator)
	x_ex := make([]f64, n_paths * n_ex, allocator)
	defer {delete(r_ex, allocator); delete(x_ex, allocator)}

	norm_count := n_paths * n_steps * 2
	norm_data := make([]f64, norm_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< norm_count {norm_data[i] = rand.float64_normal(0.0, 1.0)}

	norm_idx := 0
	for path in 0 ..< n_paths {
		r := r0
		x := 0.0
		ex_idx := 0

		for step in 1 ..< n_steps + 1 {
			U1 := norm_data[norm_idx]
			U2 := norm_data[norm_idx + 1]
			norm_idx += 2

			Z1 := U1
			Z2 := params.rho * U1 + math.sqrt_f64(1.0 - params.rho * params.rho) * U2

			t_current := f64(step) * dt
			theta_t :=
				params.a * r0 +
				(params.sigma * params.sigma / (2.0 * params.a)) *
					(1.0 - math.exp_f64(-2.0 * params.a * t_current))

			dr := (theta_t + x - params.a * r) * dt + params.sigma * sqrt_dt * Z1
			r = r + dr

			dx := -params.b * x * dt + params.eta * sqrt_dt * Z2
			x = x + dx

			if ex_idx < n_ex && t_current >= exercise_dates[ex_idx] {
				r_ex[path * n_ex + ex_idx] = r
				x_ex[path * n_ex + ex_idx] = x
				ex_idx += 1
			}
		}
	}

	// Backward induction (Lower-bound exercise strategy)
	cashflows := make([]f64, n_paths, allocator)
	defer delete(cashflows, allocator)

	for i := n_ex - 1; i >= 0; i -= 1 {
		t_ex := exercise_dates[i]

		for path in 0 ..< n_paths {
			r_t := r_ex[path * n_ex + i]
			x_t := x_ex[path * n_ex + i]

			// Compute immediate swap value at t_ex (Payer Swaption)
			annuity := 0.0
			n_pmts := int((maturity - t_ex) / delta)
			for j in 1 ..< n_pmts + 1 {
				T_j := t_ex + f64(j) * delta
				P_t_Tj := hw2f_zero_coupon_bond(t_ex, T_j, r_t, x_t, params, r0)
				annuity += delta * P_t_Tj
			}
			P_end := hw2f_zero_coupon_bond(t_ex, maturity, r_t, x_t, params, r0)

			// Swap value = P(t_ex, t_ex) - P(t_ex, maturity) - K * Annuity
			// P(t_ex, t_ex) is exactly 1.0
			swap_val := 1.0 - P_end - K * annuity

			if i == n_ex - 1 {
				// Last exercise date: exercise if in the money
				if swap_val > 0.0 {
					cashflows[path] = swap_val * math.exp_f64(-r0 * t_ex)
				}
			} else {
				// Continuation value proxy: European swaption expiring at next exercise date
				next_ex := exercise_dates[i + 1]
				euro_val := hw2f_euro_swaption_value(
					t_ex,
					next_ex,
					maturity,
					delta,
					K,
					r_t,
					x_t,
					params,
					r0,
				)

				// Exercise if immediate value exceeds the European lower bound
				if swap_val > euro_val {
					cashflows[path] = swap_val * math.exp_f64(-r0 * t_ex)
				}
				// Else: continue (cashflows[path] already holds the discounted future value)
			}
		}
	}

	// Average discounted cashflows
	total := 0.0
	for path in 0 ..< n_paths {
		total += cashflows[path]
	}
	return total / f64(n_paths)
}
// ============================================================================
// 1. FLOORLET & FLOOR PRICING (HW 1F)
// ============================================================================

// Tensorized Analytical Floorlet Pricing (SIMD Optimized)
// A floorlet is a put option on the forward rate.
hull_white_floorlet_price_tensor :: proc(
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

	one_data := l.matrix_new(f64, n, 1, allocator)
	for i in 0 ..< n {one_data.data[i] = 1.0}
	one := t.tensor_new(one_data, false, allocator)
	defer t.tensor_free(one)

	delta_T := t.tensor_sub(T_end, T_start)

	// B(0, delta_T) = (1 - exp(-a * delta_T)) / a
	a_delta := t.tensor_scale(delta_T, a)
	exp_a_delta := t.tensor_exp(t.tensor_neg(a_delta))
	B_delta := t.tensor_scale(t.tensor_sub(one, exp_a_delta), 1.0 / a)

	// Variance of the forward rate
	two_a := 2.0 * a
	exp_2a_T := t.tensor_exp(t.tensor_scale(T_start, -two_a))
	var_factor := t.tensor_scale(t.tensor_sub(one, exp_2a_T), 1.0 / two_a)

	sigma_sq_val := sigma * sigma
	B_sq := t.tensor_mul(B_delta, B_delta)
	sigma_P_sq := t.tensor_mul(t.tensor_scale(B_sq, sigma_sq_val), var_factor)
	sigma_P := t.tensor_sqrt(sigma_P_sq)

	// Black-like formula for Floorlet (Put option on forward rate)
	F_over_K := t.tensor_div(F, K)
	ln_F_K := t.tensor_log(F_over_K)
	half_var := t.tensor_scale(sigma_P_sq, 0.5)
	numer := t.tensor_add(ln_F_K, half_var)
	d1 := t.tensor_div(numer, sigma_P)
	d2 := t.tensor_sub(d1, sigma_P)

	// N(-d1) and N(-d2)
	neg_d1 := t.tensor_neg(d1)
	neg_d2 := t.tensor_neg(d2)
	N_neg_d1 := t.tensor_norm_cdf(neg_d1)
	N_neg_d2 := t.tensor_norm_cdf(neg_d2)

	// Floorlet = P * delta_T * (K * N(-d2) - F * N(-d1))
	term1 := t.tensor_mul(K, N_neg_d2)
	term2 := t.tensor_mul(F, N_neg_d1)
	intrinsic := t.tensor_sub(term1, term2)
	payoff := t.tensor_relu(intrinsic) // ReLU ensures no negative values due to numerical precision

	return t.tensor_mul(P, t.tensor_mul(delta_T, payoff))
}

// Hull-White Monte Carlo: Floor Option (with CRN Greeks)
_hw_mc_floor_price_helper :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_floorlets: int,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64,
) -> f64 {
	T_max := T_end[n_floorlets - 1]
	dt := T_max / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	total_payoff := 0.0
	norm_idx := 0

	for path in 0 ..< n_paths {
		r := r0
		floor_payoff := 0.0
		next_floorlet_idx := 0

		for step in 1 ..< n_steps + 1 {
			Z := norm_data[norm_idx]
			norm_idx += 1
			dr := params.a * (r0 - r) * dt + params.sigma * sqrt_dt * Z
			r = r + dr
			t_current := f64(step) * dt

			for next_floorlet_idx < n_floorlets && t_current >= T_start[next_floorlet_idx] {
				i := next_floorlet_idx
				disc := math.exp_f64(-r * (T_end[i] - T_start[i]))
				fwd_rate := (1.0 / disc - 1.0) / delta[i]
				payoff := delta[i] * math.max(K - fwd_rate, 0.0) * disc
				floor_payoff += payoff * math.exp_f64(-r0 * T_start[i])
				next_floorlet_idx += 1
			}
		}
		total_payoff += floor_payoff
	}
	return total_payoff / f64(n_paths)
}

hw_mc_floor_option :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_floorlets: int,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	norm_count := n_paths * n_steps
	norm_data := make([]f64, norm_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< norm_count {norm_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _hw_mc_floor_price_helper(
		r0,
		K,
		T_start,
		T_end,
		delta,
		n_floorlets,
		params,
		n_paths,
		n_steps,
		norm_data,
	)

	h_r0 := 0.01 * r0
	delta_r =
		(_hw_mc_floor_price_helper(
				r0 + h_r0,
				K,
				T_start,
				T_end,
				delta,
				n_floorlets,
				params,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw_mc_floor_price_helper(
				r0 - h_r0,
				K,
				T_start,
				T_end,
				delta,
				n_floorlets,
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
		(_hw_mc_floor_price_helper(
				r0,
				K,
				T_start,
				T_end,
				delta,
				n_floorlets,
				params_up,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw_mc_floor_price_helper(
				r0,
				K,
				T_start,
				T_end,
				delta,
				n_floorlets,
				params_dn,
				n_paths,
				n_steps,
				norm_data,
			)) /
		(2.0 * h_sigma)

	return price, delta_r, vega
}

// ============================================================================
// 2. HULL-WHITE 1F EUROPEAN SWAPTION PRICING (Monte Carlo)
// ============================================================================
// While HW 2F is great for Bermudan, HW 1F is the workhorse for European swaptions.
// We simulate the short rate and compute the underlying swap value at expiry.

_hw_mc_swaption_price_helper :: proc(
	r0: f64,
	K: f64,
	T_exp: f64,
	T_maturity: f64,
	delta: f64,
	is_payer: bool,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	norm_data: []f64,
) -> f64 {
	dt := T_maturity / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	total_payoff := 0.0
	norm_idx := 0

	n_payments := int((T_maturity - T_exp) / delta)
	if n_payments <= 0 {return 0.0}

	for path in 0 ..< n_paths {
		r := r0
		r_at_exp := r0
		hit_exp := false

		for step in 1 ..< n_steps + 1 {
			Z := norm_data[norm_idx]
			norm_idx += 1
			t_current := f64(step) * dt

			theta_t :=
				params.a * r0 +
				(params.sigma * params.sigma / (2.0 * params.a)) *
					(1.0 - math.exp_f64(-2.0 * params.a * t_current))
			dr := (theta_t - params.a * r) * dt + params.sigma * sqrt_dt * Z
			r = r + dr

			if !hit_exp && t_current >= T_exp {
				r_at_exp = r
				hit_exp = true
			}
		}

		// Value the underlying swap at T_exp
		annuity := 0.0

		// Precompute variance terms for efficiency
		V_0_Texp :=
			(params.sigma * params.sigma / (2.0 * params.a * params.a * params.a)) *
			math.pow(1.0 - math.exp_f64(-params.a * T_exp), 2.0) *
			(1.0 - math.exp_f64(-2.0 * params.a * T_exp))
		P_0_Texp := math.exp_f64(-r0 * T_exp)

		for i in 1 ..< n_payments + 1 {
			T_i := T_exp + f64(i) * delta
			tau := T_i - T_exp

			B_exp_i := (1.0 - math.exp_f64(-params.a * tau)) / params.a

			V_0_Ti :=
				(params.sigma * params.sigma / (2.0 * params.a * params.a * params.a)) *
				math.pow(1.0 - math.exp_f64(-params.a * T_i), 2.0) *
				(1.0 - math.exp_f64(-2.0 * params.a * T_i))

			P_0_Ti := math.exp_f64(-r0 * T_i)
			A_exp_i := (P_0_Ti / P_0_Texp) * math.exp_f64(B_exp_i * r0 - 0.5 * (V_0_Ti - V_0_Texp))
			P_exp_i := A_exp_i * math.exp_f64(-B_exp_i * r_at_exp)

			annuity += delta * P_exp_i
		}

		// P(T_exp, T_maturity)
		tau_m := T_maturity - T_exp
		B_exp_m := (1.0 - math.exp_f64(-params.a * tau_m)) / params.a
		V_0_Tm :=
			(params.sigma * params.sigma / (2.0 * params.a * params.a * params.a)) *
			math.pow(1.0 - math.exp_f64(-params.a * T_maturity), 2.0) *
			(1.0 - math.exp_f64(-2.0 * params.a * T_maturity))
		P_0_Tm := math.exp_f64(-r0 * T_maturity)
		A_exp_m := (P_0_Tm / P_0_Texp) * math.exp_f64(B_exp_m * r0 - 0.5 * (V_0_Tm - V_0_Texp))
		P_exp_m := A_exp_m * math.exp_f64(-B_exp_m * r_at_exp)

		F_swap := (1.0 - P_exp_m) / annuity

		swap_value := annuity * (F_swap - K)
		if !is_payer {
			swap_value = annuity * (K - F_swap)
		}

		if swap_value > 0.0 {
			total_payoff += swap_value * P_0_Texp // Discount back to t=0
		}
	}
	return total_payoff / f64(n_paths)
}

hw_mc_swaption_option :: proc(
	r0: f64,
	K: f64,
	T_exp: f64,
	T_maturity: f64,
	delta: f64,
	is_payer: bool,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	norm_count := n_paths * n_steps
	norm_data := make([]f64, norm_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< norm_count {norm_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _hw_mc_swaption_price_helper(
		r0,
		K,
		T_exp,
		T_maturity,
		delta,
		is_payer,
		params,
		n_paths,
		n_steps,
		norm_data,
	)

	h_r0 := 0.01 * r0
	delta_r =
		(_hw_mc_swaption_price_helper(
				r0 + h_r0,
				K,
				T_exp,
				T_maturity,
				delta,
				is_payer,
				params,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw_mc_swaption_price_helper(
				r0 - h_r0,
				K,
				T_exp,
				T_maturity,
				delta,
				is_payer,
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
		(_hw_mc_swaption_price_helper(
				r0,
				K,
				T_exp,
				T_maturity,
				delta,
				is_payer,
				params_up,
				n_paths,
				n_steps,
				norm_data,
			) -
			_hw_mc_swaption_price_helper(
				r0,
				K,
				T_exp,
				T_maturity,
				delta,
				is_payer,
				params_dn,
				n_paths,
				n_steps,
				norm_data,
			)) /
		(2.0 * h_sigma)

	return price, delta_r, vega
}

// ============================================================================
// 3. CAP / FLOOR STRIP PRICING (Convenience)
// ============================================================================

CapFloorStripResult :: struct {
	total_price: f64,
	strike:      f64,
	is_cap:      bool,
	leg_prices:  []f64, // Price of each individual caplet/floorlet
}

// Price a whole strip of caplets or floorlets analytically using HW 1F
hw_price_cap_floor_strip :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	params: HW_Params,
	is_cap: bool,
	allocator: mem.Allocator = context.allocator,
) -> CapFloorStripResult {
	n := len(T_start)

	T_start_t := _make_const_tensor(T_start, n, allocator)
	T_end_t := _make_const_tensor(T_end, n, allocator)

	// Forward rates F_i = (P(0, T_start) / P(0, T_end) - 1) / delta
	// For flat curve r0: P(0, t) = exp(-r0 * t)
	F_data := make([]f64, n, allocator)
	P_data := make([]f64, n, allocator)
	K_data := make([]f64, n, allocator)

	for i in 0 ..< n {
		P_start := math.exp_f64(-r0 * T_start[i])
		P_end := math.exp_f64(-r0 * T_end[i])
		F_data[i] = (P_start / P_end - 1.0) / delta[i]
		P_data[i] = P_end
		K_data[i] = K
	}

	F_t := _make_const_tensor(F_data, n, allocator)
	P_t := _make_const_tensor(P_data, n, allocator)
	K_t := _make_const_tensor(K_data, n, allocator)

	price_tensor: ^t.Tensor
	if is_cap {
		price_tensor = hull_white_caplet_price_tensor(
			params.a,
			params.sigma,
			T_start_t,
			T_end_t,
			F_t,
			K_t,
			P_t,
			allocator,
		)
	} else {
		price_tensor = hull_white_floorlet_price_tensor(
			params.a,
			params.sigma,
			T_start_t,
			T_end_t,
			F_t,
			K_t,
			P_t,
			allocator,
		)
	}

	leg_prices := make([]f64, n, allocator)
	total_price := 0.0
	for i in 0 ..< n {
		leg_prices[i] = price_tensor.data.data[i]
		total_price += leg_prices[i]
	}

	t.tensor_free(price_tensor)
	t.tensor_free(T_start_t)
	t.tensor_free(T_end_t)
	t.tensor_free(F_t)
	t.tensor_free(P_t)
	t.tensor_free(K_t)
	delete(F_data, allocator)
	delete(P_data, allocator)
	delete(K_data, allocator)

	return CapFloorStripResult {
		total_price = total_price,
		strike = K,
		is_cap = is_cap,
		leg_prices = leg_prices,
	}
}
