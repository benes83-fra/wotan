package finance

import "core:math"
import rand "core:math/rand"
import "core:mem"
// ============================================================================
// Helper: Heston MC Pricing Logic (Explicit arguments to avoid closures)
// ============================================================================
_heston_mc_barrier_price_helper :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
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

			// Full truncation scheme for variance
			v_curr =
				v_curr +
				params.kappa * (params.theta - v_curr) * dt +
				params.sigma * v_sqrt * sqrt_dt * Z_v
			v_curr = math.max(0.0, v_curr)

			// Evolve spot
			S = S * math.exp_f64((r - 0.5 * v_curr) * dt + v_sqrt * sqrt_dt * Z_s)

			if S >= barrier {
				hit_barrier = true
				break
			}
		}

		if !hit_barrier {
			total_payoff += math.max(S - K, 0.0)
		}
	}
	return (total_payoff / f64(n_paths)) * discount
}

// ============================================================================
// Heston Monte Carlo: Up-and-Out Call Option (with CRN Greeks)
// ============================================================================
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

	// Pre-generate random numbers ONCE for Common Random Numbers (CRN)
	rand_count := n_paths * n_steps * 2
	rand_data := make([]f64, rand_count, allocator)
	defer delete(rand_data, allocator)

	for i in 0 ..< rand_count {
		rand_data[i] = rand.float64_normal(0.0, 1.0)
	}

	// 1. Base Price
	price = _heston_mc_barrier_price_helper(
		S_0,
		K,
		T,
		r,
		barrier,
		params,
		n_paths,
		n_steps,
		rand_data,
	)

	// 2. Delta via central finite difference (1% bump in S_0) with CRN
	h_S := 0.01 * S_0
	delta =
		(_heston_mc_barrier_price_helper(
				S_0 + h_S,
				K,
				T,
				r,
				barrier,
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
				params,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_S)

	// 3. Vega via central finite difference (1% bump in sigma) with CRN
	h_sigma := 0.01 * params.sigma
	if h_sigma < 0.001 {h_sigma = 0.001}

	params_up := params
	params_up.sigma = params.sigma + h_sigma

	params_dn := params
	params_dn.sigma = params.sigma - h_sigma

	vega =
		(_heston_mc_barrier_price_helper(
				S_0,
				K,
				T,
				r,
				barrier,
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
				params_dn,
				n_paths,
				n_steps,
				rand_data,
			)) /
		(2.0 * h_sigma)

	return price, delta, vega
}
