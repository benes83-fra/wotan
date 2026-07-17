package finance

import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// Heston Monte Carlo: Up-and-Out Call Option
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
) -> f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)

	discount := math.exp(-r * T)
	total_payoff := 0.0

	// Pre-generate all random numbers for performance
	// We need 2 random numbers per step per path (one for vol, one independent for spot)
	rand_count := n_paths * n_steps * 2
	rand_data := make([]f64, rand_count, allocator)
	defer delete(rand_data, allocator)

	for i in 0 ..< rand_count {
		rand_data[i] = rand.float64_normal(0.0, 1.0)
	}

	rand_idx := 0
	for p in 0 ..< n_paths {
		S := S_0
		v := params.v0
		hit_barrier := false

		for step in 0 ..< n_steps {
			Z_v := rand_data[rand_idx]
			Z_indep := rand_data[rand_idx + 1]
			rand_idx += 2

			// Correlated Brownian motion for Spot: dW_S = rho * dW_v + sqrt(1 - rho^2) * dZ_indep
			rho := params.rho
			Z_s := rho * Z_v + math.sqrt_f64(1.0 - rho * rho) * Z_indep

			// Full Truncation scheme for variance (prevents negative v)
			v_sqrt := math.sqrt_f64(math.max(0.0, v))

			// Evolve variance
			v = v + params.kappa * (params.theta - v) * dt + params.sigma * v_sqrt * sqrt_dt * Z_v
			v = math.max(0.0, v) // Full truncation

			// Evolve spot (using v from the *start* of the step for stability)
			v_sqrt_start := math.sqrt_f64(math.max(0.0, v)) // or use previous step's v_sqrt
			S = S * math.exp((r - 0.5 * v) * dt + v_sqrt_start * sqrt_dt * Z_s)

			// Check barrier
			if S >= barrier {
				hit_barrier = true
				break // Knocked out, no need to simulate further
			}
		}

		if !hit_barrier {
			payoff := math.max(S - K, 0.0)
			total_payoff += payoff
		}
	}

	return (total_payoff / f64(n_paths)) * discount
}
