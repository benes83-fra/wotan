package finance

import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// 1. BARRIER OPTIONS
// ============================================================================

_heston_mc_barrier_price_helper :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	is_up: bool,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	rand_data: []f64,
) -> f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	discount := math.exp_f64(-r * T)
	total_payoff := 0.0

	rand_idx := 0
	for path in 0 ..< n_paths {
		S := S_0
		v_curr := params.v0
		hit_barrier := false

		for step in 0 ..< n_steps {
			Z_v := rand_data[rand_idx]
			Z_indep := rand_data[rand_idx + 1]
			rand_idx += 2

			Z_s := params.rho * Z_v + math.sqrt_f64(1.0 - params.rho * params.rho) * Z_indep
			v_sqrt := math.sqrt_f64(math.max(0.0, v_curr))

			v_curr =
				v_curr +
				params.kappa * (params.theta - v_curr) * dt +
				params.sigma * v_sqrt * sqrt_dt * Z_v
			v_curr = math.max(0.0, v_curr)

			S = S * math.exp_f64((r - 0.5 * v_curr) * dt + v_sqrt * sqrt_dt * Z_s)

			if is_up {
				if S >= barrier {hit_barrier = true; break}
			} else {
				if S <= barrier {hit_barrier = true; break}
			}
		}

		if !hit_barrier {
			if opt == .Call {
				total_payoff += math.max(S - K, 0.0)
			} else {
				total_payoff += math.max(K - S, 0.0)
			}
		}
	}
	return (total_payoff / f64(n_paths)) * discount
}

// Generalized Barrier Option (Supports Call/Put and Up/Down)
heston_mc_barrier_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	is_up: bool,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	rand_count := n_paths * n_steps * 2
	rand_data := make([]f64, rand_count, allocator)
	defer delete(rand_data, allocator)
	for i in 0 ..< rand_count {rand_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _heston_mc_barrier_price_helper(
		S_0,
		K,
		T,
		r,
		barrier,
		is_up,
		opt,
		params,
		n_paths,
		n_steps,
		rand_data,
	)

	h_S := 0.01 * S_0
	delta =
		(_heston_mc_barrier_price_helper(
				S_0 + h_S,
				K,
				T,
				r,
				barrier,
				is_up,
				opt,
				params,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_barrier_price_helper(
				S_0 - h_S,
				K,
				T,
				r,
				barrier,
				is_up,
				opt,
				params,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_S)

	h_sigma := 0.01 * params.sigma
	if h_sigma < 0.001 {h_sigma = 0.001}
	params_up := params; params_up.sigma = params.sigma + h_sigma
	params_dn := params; params_dn.sigma = params.sigma - h_sigma

	vega =
		(_heston_mc_barrier_price_helper(
				S_0,
				K,
				T,
				r,
				barrier,
				is_up,
				opt,
				params_up,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_barrier_price_helper(
				S_0,
				K,
				T,
				r,
				barrier,
				is_up,
				opt,
				params_dn,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_sigma)

	return price, delta, vega
}

// Legacy Wrapper: Keeps your existing code working perfectly
heston_mc_barrier_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	return heston_mc_barrier_option(
		S_0,
		K,
		T,
		r,
		barrier,
		true,
		.Call,
		params,
		n_paths,
		n_steps,
		allocator,
	)
}


// ============================================================================
// 2. ASIAN OPTIONS
// ============================================================================

_heston_mc_asian_price_helper :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	rand_data: []f64,
) -> f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	discount := math.exp_f64(-r * T)
	total_payoff := 0.0

	rand_idx := 0
	for path in 0 ..< n_paths {
		S := S_0
		v_curr := params.v0
		sum_S := 0.0

		for step in 0 ..< n_steps {
			Z_v := rand_data[rand_idx]
			Z_indep := rand_data[rand_idx + 1]
			rand_idx += 2

			Z_s := params.rho * Z_v + math.sqrt_f64(1.0 - params.rho * params.rho) * Z_indep
			v_sqrt := math.sqrt_f64(math.max(0.0, v_curr))

			v_curr =
				v_curr +
				params.kappa * (params.theta - v_curr) * dt +
				params.sigma * v_sqrt * sqrt_dt * Z_v
			v_curr = math.max(0.0, v_curr)

			S = S * math.exp_f64((r - 0.5 * v_curr) * dt + v_sqrt * sqrt_dt * Z_s)
			sum_S += S
		}

		avg_S := sum_S / f64(n_steps)
		if opt == .Call {
			total_payoff += math.max(avg_S - K, 0.0)
		} else {
			total_payoff += math.max(K - avg_S, 0.0)
		}
	}
	return (total_payoff / f64(n_paths)) * discount
}

heston_mc_asian_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	rand_count := n_paths * n_steps * 2
	rand_data := make([]f64, rand_count, allocator)
	defer delete(rand_data, allocator)
	for i in 0 ..< rand_count {rand_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _heston_mc_asian_price_helper(S_0, K, T, r, opt, params, n_paths, n_steps, rand_data)

	h_S := 0.01 * S_0
	delta =
		(_heston_mc_asian_price_helper(
				S_0 + h_S,
				K,
				T,
				r,
				opt,
				params,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_asian_price_helper(
				S_0 - h_S,
				K,
				T,
				r,
				opt,
				params,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_S)

	h_sigma := 0.01 * params.sigma
	if h_sigma < 0.001 {h_sigma = 0.001}
	params_up := params; params_up.sigma = params.sigma + h_sigma
	params_dn := params; params_dn.sigma = params.sigma - h_sigma

	vega =
		(_heston_mc_asian_price_helper(S_0, K, T, r, opt, params_up, n_paths, n_steps, rand_data) -
			_heston_mc_asian_price_helper(
				S_0,
				K,
				T,
				r,
				opt,
				params_dn,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_sigma)

	return price, delta, vega
}

heston_mc_asian_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	return heston_mc_asian_option(S_0, K, T, r, .Call, params, n_paths, n_steps, allocator)
}


// ============================================================================
// 3. LOOKBACK OPTIONS
// ============================================================================

_heston_mc_lookback_price_helper :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	rand_data: []f64,
) -> f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	discount := math.exp_f64(-r * T)
	total_payoff := 0.0

	rand_idx := 0
	for path in 0 ..< n_paths {
		S := S_0
		v_curr := params.v0
		max_S := S_0
		min_S := S_0

		for step in 0 ..< n_steps {
			Z_v := rand_data[rand_idx]
			Z_indep := rand_data[rand_idx + 1]
			rand_idx += 2

			Z_s := params.rho * Z_v + math.sqrt_f64(1.0 - params.rho * params.rho) * Z_indep
			v_sqrt := math.sqrt_f64(math.max(0.0, v_curr))

			v_curr =
				v_curr +
				params.kappa * (params.theta - v_curr) * dt +
				params.sigma * v_sqrt * sqrt_dt * Z_v
			v_curr = math.max(0.0, v_curr)

			S = S * math.exp_f64((r - 0.5 * v_curr) * dt + v_sqrt * sqrt_dt * Z_s)

			if S > max_S {max_S = S}
			if S < min_S {min_S = S}
		}

		if opt == .Call {
			total_payoff += math.max(max_S - K, 0.0)
		} else {
			total_payoff += math.max(K - min_S, 0.0)
		}
	}
	return (total_payoff / f64(n_paths)) * discount
}

heston_mc_lookback_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	rand_count := n_paths * n_steps * 2
	rand_data := make([]f64, rand_count, allocator)
	defer delete(rand_data, allocator)
	for i in 0 ..< rand_count {rand_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _heston_mc_lookback_price_helper(
		S_0,
		K,
		T,
		r,
		opt,
		params,
		n_paths,
		n_steps,
		rand_data,
	)

	h_S := 0.01 * S_0
	delta =
		(_heston_mc_lookback_price_helper(
				S_0 + h_S,
				K,
				T,
				r,
				opt,
				params,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_lookback_price_helper(
				S_0 - h_S,
				K,
				T,
				r,
				opt,
				params,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_S)

	h_sigma := 0.01 * params.sigma
	if h_sigma < 0.001 {h_sigma = 0.001}
	params_up := params; params_up.sigma = params.sigma + h_sigma
	params_dn := params; params_dn.sigma = params.sigma - h_sigma

	vega =
		(_heston_mc_lookback_price_helper(
				S_0,
				K,
				T,
				r,
				opt,
				params_up,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_lookback_price_helper(
				S_0,
				K,
				T,
				r,
				opt,
				params_dn,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_sigma)

	return price, delta, vega
}

heston_mc_lookback_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	return heston_mc_lookback_option(S_0, K, T, r, .Call, params, n_paths, n_steps, allocator)
}


// ============================================================================
// 4. 2-ASSET BASKET OPTIONS
// ============================================================================

_heston_mc_basket_price_helper :: proc(
	S1_0: f64,
	S2_0: f64,
	K: f64,
	T: f64,
	r: f64,
	w1: f64,
	w2: f64,
	opt: OptionType,
	params1: Heston_Params,
	params2: Heston_Params,
	corr_12: f64,
	n_paths: int,
	n_steps: int,
	rand_data: []f64,
) -> f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	discount := math.exp_f64(-r * T)
	total_payoff := 0.0

	denom :=
		math.sqrt_f64(1.0 - params1.rho * params1.rho) *
		math.sqrt_f64(1.0 - params2.rho * params2.rho)
	rho_12_star := corr_12 / denom
	if rho_12_star > 1.0 {rho_12_star = 1.0} else if rho_12_star < -1.0 {rho_12_star = -1.0}
	sqrt_1_minus_rho_12_star_sq := math.sqrt_f64(1.0 - rho_12_star * rho_12_star)

	rand_idx := 0
	for path in 0 ..< n_paths {
		S1 := S1_0; S2 := S2_0
		v1 := params1.v0; v2 := params2.v0

		for step in 0 ..< n_steps {
			U1 := rand_data[rand_idx]; U2 := rand_data[rand_idx + 1]
			U3 := rand_data[rand_idx + 2]; U4 := rand_data[rand_idx + 3]
			rand_idx += 4

			v1_sqrt := math.sqrt_f64(math.max(0.0, v1))
			v2_sqrt := math.sqrt_f64(math.max(0.0, v2))

			v1 =
				v1 +
				params1.kappa * (params1.theta - v1) * dt +
				params1.sigma * v1_sqrt * sqrt_dt * U1
			v1 = math.max(0.0, v1)

			v2 =
				v2 +
				params2.kappa * (params2.theta - v2) * dt +
				params2.sigma * v2_sqrt * sqrt_dt * U2
			v2 = math.max(0.0, v2)

			dW1 := params1.rho * U1 + math.sqrt_f64(1.0 - params1.rho * params1.rho) * U3
			dW2 :=
				params2.rho * U2 +
				math.sqrt_f64(1.0 - params2.rho * params2.rho) *
					(rho_12_star * U3 + sqrt_1_minus_rho_12_star_sq * U4)

			S1 = S1 * math.exp_f64((r - 0.5 * v1) * dt + v1_sqrt * sqrt_dt * dW1)
			S2 = S2 * math.exp_f64((r - 0.5 * v2) * dt + v2_sqrt * sqrt_dt * dW2)
		}

		basket_value := w1 * S1 + w2 * S2
		if opt == .Call {
			total_payoff += math.max(basket_value - K, 0.0)
		} else {
			total_payoff += math.max(K - basket_value, 0.0)
		}
	}
	return (total_payoff / f64(n_paths)) * discount
}

heston_mc_basket_option :: proc(
	S1_0: f64,
	S2_0: f64,
	K: f64,
	T: f64,
	r: f64,
	w1: f64,
	w2: f64,
	opt: OptionType,
	params1: Heston_Params,
	params2: Heston_Params,
	corr_12: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta1: f64,
	delta2: f64,
	vega1: f64,
	vega2: f64,
) {
	rand_count := n_paths * n_steps * 4
	rand_data := make([]f64, rand_count, allocator)
	defer delete(rand_data, allocator)
	for i in 0 ..< rand_count {rand_data[i] = rand.float64_normal(0.0, 1.0)}

	price = _heston_mc_basket_price_helper(
		S1_0,
		S2_0,
		K,
		T,
		r,
		w1,
		w2,
		opt,
		params1,
		params2,
		corr_12,
		n_paths,
		n_steps,
		rand_data,
	)

	h_S1 := 0.01 * S1_0
	delta1 =
		(_heston_mc_basket_price_helper(
				S1_0 + h_S1,
				S2_0,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1,
				params2,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_basket_price_helper(
				S1_0 - h_S1,
				S2_0,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1,
				params2,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_S1)

	h_S2 := 0.01 * S2_0
	delta2 =
		(_heston_mc_basket_price_helper(
				S1_0,
				S2_0 + h_S2,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1,
				params2,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_basket_price_helper(
				S1_0,
				S2_0 - h_S2,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1,
				params2,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_S2)

	h_sig1 := 0.01 * params1.sigma
	if h_sig1 < 0.001 {h_sig1 = 0.001}
	params1_up := params1; params1_up.sigma = params1.sigma + h_sig1
	params1_dn := params1; params1_dn.sigma = params1.sigma - h_sig1
	vega1 =
		(_heston_mc_basket_price_helper(
				S1_0,
				S2_0,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1_up,
				params2,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_basket_price_helper(
				S1_0,
				S2_0,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1_dn,
				params2,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_sig1)

	h_sig2 := 0.01 * params2.sigma
	if h_sig2 < 0.001 {h_sig2 = 0.001}
	params2_up := params2; params2_up.sigma = params2.sigma + h_sig2
	params2_dn := params2; params2_dn.sigma = params2.sigma - h_sig2
	vega2 =
		(_heston_mc_basket_price_helper(
				S1_0,
				S2_0,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1,
				params2_up,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			) -
			_heston_mc_basket_price_helper(
				S1_0,
				S2_0,
				K,
				T,
				r,
				w1,
				w2,
				opt,
				params1,
				params2_dn,
				corr_12,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_sig2)

	return price, delta1, delta2, vega1, vega2
}

heston_mc_basket_call :: proc(
	S1_0: f64,
	S2_0: f64,
	K: f64,
	T: f64,
	r: f64,
	w1: f64,
	w2: f64,
	params1: Heston_Params,
	params2: Heston_Params,
	corr_12: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta1: f64,
	delta2: f64,
	vega1: f64,
	vega2: f64,
) {
	return heston_mc_basket_option(
		S1_0,
		S2_0,
		K,
		T,
		r,
		w1,
		w2,
		.Call,
		params1,
		params2,
		corr_12,
		n_paths,
		n_steps,
		allocator,
	)
}
