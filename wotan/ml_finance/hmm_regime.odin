package ml_finance

import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// HMM Data Structures
// ============================================================================

HMM :: struct {
	n_states:     int,
	transition:   []f64, // Flattened N x N matrix (row-major: A[i * N + j])
	emission_mu:  []f64, // N means (Gaussian)
	emission_var: []f64, // N variances (Gaussian)
	initial:      []f64, // N initial probabilities
	allocator:    mem.Allocator,
}

hmm_new :: proc(n_states: int, allocator: mem.Allocator = context.allocator) -> HMM {
	hmm: HMM
	hmm.n_states = n_states
	hmm.allocator = allocator
	hmm.transition = make([]f64, n_states * n_states, allocator)
	hmm.emission_mu = make([]f64, n_states, allocator)
	hmm.emission_var = make([]f64, n_states, allocator)
	hmm.initial = make([]f64, n_states, allocator)
	return hmm
}

hmm_free :: proc(hmm: ^HMM) {
	if hmm.transition != nil {delete(hmm.transition, hmm.allocator)}
	if hmm.emission_mu != nil {delete(hmm.emission_mu, hmm.allocator)}
	if hmm.emission_var != nil {delete(hmm.emission_var, hmm.allocator)}
	if hmm.initial != nil {delete(hmm.initial, hmm.allocator)}
}

// ============================================================================
// Initialization & Math Helpers
// ============================================================================

// ln N(x; mu, variance)
_gaussian_log_pdf :: proc(x, mu, variance: f64) -> f64 {
	v := max(variance, 1e-300)
	d := x - mu
	return -0.5 * (d * d / v + math.ln(2.0 * math.PI * v))
}

// Initialize HMM for volatility regimes, with high state-persistence: every state
// starts at the sample mean, state i with the mean squared deviation of the i-th
// N-quantile bucket of squared deviations, so states are ordered from calm (0) to
// turbulent (N - 1). (Means spread from min to max put states on the outliers, and
// EM then stalls in a degenerate optimum.)
hmm_init_simple :: proc(hmm: ^HMM, obs: []f64) {
	N := hmm.n_states
	T := len(obs)
	if T == 0 || N == 0 {return}

	mean := 0.0
	for t in 0 ..< T {mean += obs[t]}
	mean /= f64(T)
	d2 := make([]f64, T, hmm.allocator)
	defer delete(d2, hmm.allocator)
	var := 0.0
	for t in 0 ..< T {
		d2[t] = (obs[t] - mean) * (obs[t] - mean)
		var += d2[t]
	}
	var /= f64(T)
	slice.sort(d2)

	for i in 0 ..< N {hmm.initial[i] = 1.0 / f64(N)}

	// High probability of staying in the same state (sticky regimes)
	for i in 0 ..< N {
		for j in 0 ..< N {
			if i == j {
				hmm.transition[i * N + j] = 0.95 if N > 1 else 1.0
			} else {
				hmm.transition[i * N + j] = 0.05 / f64(N - 1)
			}
		}
	}

	for i in 0 ..< N {
		hmm.emission_mu[i] = mean
		bucket := d2[i * T / N:(i + 1) * T / N]
		hmm.emission_var[i] = var
		if len(bucket) > 0 {hmm.emission_var[i] = math.sum(bucket) / f64(len(bucket))}
	}
}

// Emission likelihoods b[t * N + j] = p(obs[t] | state j), each time step divided by its
// largest one (whose log is log_max[t]), so that an outlier far from every state cannot
// underflow them all to 0; computed once per E-step, from log densities
_emissions :: proc(hmm: ^HMM, obs: []f64, alloc: mem.Allocator) -> (b: []f64, log_max: []f64) {
	T := len(obs)
	N := hmm.n_states
	b = make([]f64, T * N, alloc)
	log_max = make([]f64, T, alloc)
	for t in 0 ..< T {
		m := -math.F64_MAX
		for j in 0 ..< N {
			b[t * N + j] = _gaussian_log_pdf(obs[t], hmm.emission_mu[j], hmm.emission_var[j])
			m = max(m, b[t * N + j])
		}
		for j in 0 ..< N {b[t * N + j] = math.exp(b[t * N + j] - m)}
		log_max[t] = m
	}
	return
}

// ============================================================================
// Forward Algorithm (with Rabiner Scaling to prevent underflow)
// ============================================================================
// b from _emissions; ln P(obs) = sum(log_max[t] - ln scale[t]) (_log_likelihood)
_forward :: proc(
	hmm: ^HMM,
	b: []f64,
	T: int,
	alloc: mem.Allocator,
) -> (
	alpha: []f64,
	scale: []f64,
) {
	N := hmm.n_states
	alpha = make([]f64, T * N, alloc)
	scale = make([]f64, T, alloc)

	c := 0.0
	for i in 0 ..< N {
		alpha[0 * N + i] = hmm.initial[i] * b[0 * N + i]
		c += alpha[0 * N + i]
	}
	if c == 0 {c = 1e-300}
	scale[0] = 1.0 / c
	for i in 0 ..< N {alpha[0 * N + i] *= scale[0]}

	for t in 1 ..< T {
		c = 0.0
		for j in 0 ..< N {
			sum := 0.0
			for i in 0 ..< N {
				sum += alpha[(t - 1) * N + i] * hmm.transition[i * N + j]
			}
			alpha[t * N + j] = sum * b[t * N + j]
			c += alpha[t * N + j]
		}
		if c == 0 {c = 1e-300}
		scale[t] = 1.0 / c
		for j in 0 ..< N {alpha[t * N + j] *= scale[t]}
	}
	return alpha, scale
}

// ============================================================================
// Backward Algorithm (with Rabiner Scaling)
// ============================================================================
_backward :: proc(hmm: ^HMM, b: []f64, scale: []f64, alloc: mem.Allocator) -> []f64 {
	T := len(scale)
	N := hmm.n_states
	beta := make([]f64, T * N, alloc)

	for i in 0 ..< N {beta[(T - 1) * N + i] = 1.0}

	for t := T - 2; t >= 0; t -= 1 {
		for i in 0 ..< N {
			sum := 0.0
			for j in 0 ..< N {
				sum += hmm.transition[i * N + j] * b[(t + 1) * N + j] * beta[(t + 1) * N + j]
			}
			beta[t * N + i] = scale[t + 1] * sum
		}
	}
	return beta
}

_log_likelihood :: proc(scale, log_max: []f64) -> f64 {
	ll := 0.0
	for t in 0 ..< len(scale) {ll += log_max[t] - math.ln(scale[t])}
	return ll
}

// ln P(obs) under the current parameters (hmmlearn's score)
hmm_log_likelihood :: proc(hmm: ^HMM, obs: []f64) -> f64 {
	if len(obs) == 0 || hmm.n_states == 0 {return 0}
	alloc := hmm.allocator
	b, log_max := _emissions(hmm, obs, alloc)
	alpha, scale := _forward(hmm, b, len(obs), alloc)
	ll := _log_likelihood(scale, log_max)
	delete(b, alloc)
	delete(log_max, alloc)
	delete(alpha, alloc)
	delete(scale, alloc)
	return ll
}

// Filtered state probabilities P(state at t | obs[0..t]) [T, N] row-major: what is known of
// the regime at t without later observations (hmm_decode's Viterbi path and smoothed
// posteriors use the whole sample, so they look ahead when used to trade)
hmm_filter :: proc(hmm: ^HMM, obs: []f64, alloc: mem.Allocator = context.allocator) -> []f64 {
	if len(obs) == 0 || hmm.n_states == 0 {return make([]f64, 0, alloc)}
	b, log_max := _emissions(hmm, obs, context.temp_allocator)
	alpha, scale := _forward(hmm, b, len(obs), alloc) // rows normalised to sum 1
	delete(b, context.temp_allocator)
	delete(log_max, context.temp_allocator)
	delete(scale, alloc)
	return alpha
}

// ============================================================================
// Baum-Welch (EM Algorithm) for Training
// ============================================================================
// The returned log-likelihood is that of the parameters before the last update (as
// hmmlearn's monitor). Variances are kept above 1e-3 x the sample variance, so that
// a state cannot collapse onto a single observation whatever the scale of the data
// (hmmlearn's min_covar is the same floor, absolute, meant for standardized data).
hmm_fit :: proc(
	hmm: ^HMM,
	obs: []f64,
	max_iter: int = 100,
	tol: f64 = 1e-4,
) -> (
	log_likelihood: f64,
	converged: bool,
) {
	T := len(obs)
	N := hmm.n_states
	if T == 0 || N == 0 {return 0, false}
	alloc := hmm.allocator

	mean, var := 0.0, 0.0
	for x in obs {mean += x}
	mean /= f64(T)
	for x in obs {var += (x - mean) * (x - mean)}
	var /= f64(T)
	min_var := max(1e-3 * var, 1e-300)

	prev_ll := -math.F64_MAX

	gamma := make([]f64, T * N, alloc)
	xi_sum := make([]f64, N * N, alloc) // transition posteriors, summed over t
	defer {
		delete(gamma, alloc)
		delete(xi_sum, alloc)
	}

	for iter in 0 ..< max_iter {
		// E-Step: Forward-Backward
		b, log_max := _emissions(hmm, obs, alloc)
		alpha, scale := _forward(hmm, b, T, alloc)
		beta := _backward(hmm, b, scale, alloc)
		ll := _log_likelihood(scale, log_max)

		// Compute gamma (state posteriors)
		for t in 0 ..< T {
			sum_g := 0.0
			for i in 0 ..< N {
				g := alpha[t * N + i] * beta[t * N + i]
				gamma[t * N + i] = g
				sum_g += g
			}
			if sum_g > 0 {
				for i in 0 ..< N {gamma[t * N + i] /= sum_g}
			}
		}

		// Compute xi (transition posteriors), each t normalized, and sum them
		for &v in xi_sum {v = 0}
		for t in 0 ..< T - 1 {
			sum_xi := 0.0
			for i in 0 ..< N {
				for j in 0 ..< N {
					sum_xi +=
						alpha[t * N + i] *
						hmm.transition[i * N + j] *
						b[(t + 1) * N + j] *
						beta[(t + 1) * N + j]
				}
			}
			if sum_xi > 0 {
				for i in 0 ..< N {
					for j in 0 ..< N {
						xi_sum[i * N + j] +=
							alpha[t * N + i] *
							hmm.transition[i * N + j] *
							b[(t + 1) * N + j] *
							beta[(t + 1) * N + j] /
							sum_xi
					}
				}
			}
		}

		delete(b, alloc)
		delete(log_max, alloc)
		delete(alpha, alloc)
		delete(scale, alloc)
		delete(beta, alloc)

		// M-Step: Update Parameters
		for i in 0 ..< N {hmm.initial[i] = gamma[0 * N + i]}

		for i in 0 ..< N {
			sum_den := 0.0
			for j in 0 ..< N {sum_den += xi_sum[i * N + j]}
			if sum_den > 0 {
				for j in 0 ..< N {hmm.transition[i * N + j] = xi_sum[i * N + j] / sum_den}
			}
		}

		for i in 0 ..< N {
			sum_gamma := 0.0
			sum_x := 0.0
			for t in 0 ..< T {
				sum_gamma += gamma[t * N + i]
				sum_x += gamma[t * N + i] * obs[t]
			}
			if sum_gamma > 0 {
				mu := sum_x / sum_gamma
				sum_d2 := 0.0
				for t in 0 ..< T {sum_d2 += gamma[t * N + i] * (obs[t] - mu) * (obs[t] - mu)}
				hmm.emission_mu[i] = mu
				hmm.emission_var[i] = max(sum_d2 / sum_gamma, min_var)
			}
		}

		if math.abs(ll - prev_ll) < tol {return ll, true}
		prev_ll = ll
	}

	return prev_ll, false
}

// ============================================================================
// Viterbi Algorithm (Most Likely Regime Sequence)
// ============================================================================
hmm_decode :: proc(hmm: ^HMM, obs: []f64, alloc: mem.Allocator = context.allocator) -> []int {
	T := len(obs)
	N := hmm.n_states
	if T == 0 || N == 0 {return make([]int, 0, alloc)}

	delta := make([]f64, T * N, alloc)
	psi := make([]int, T * N, alloc)
	defer {
		delete(delta, alloc)
		delete(psi, alloc)
	}

	for i in 0 ..< N {
		delta[0 * N + i] =
			math.ln(hmm.initial[i] + 1e-300) +
			_gaussian_log_pdf(obs[0], hmm.emission_mu[i], hmm.emission_var[i])
		psi[0 * N + i] = 0
	}

	for t in 1 ..< T {
		for j in 0 ..< N {
			max_val := -math.F64_MAX
			max_idx := 0
			for i in 0 ..< N {
				val := delta[(t - 1) * N + i] + math.ln(hmm.transition[i * N + j] + 1e-300)
				if val > max_val {
					max_val = val
					max_idx = i
				}
			}
			delta[t * N + j] =
				max_val + _gaussian_log_pdf(obs[t], hmm.emission_mu[j], hmm.emission_var[j])
			psi[t * N + j] = max_idx
		}
	}

	states := make([]int, T, alloc)
	max_val := -math.F64_MAX
	max_idx := 0
	for i in 0 ..< N {
		if delta[(T - 1) * N + i] > max_val {
			max_val = delta[(T - 1) * N + i]
			max_idx = i
		}
	}
	states[T - 1] = max_idx

	for t := T - 2; t >= 0; t -= 1 {
		states[t] = psi[(t + 1) * N + states[t + 1]]
	}

	return states
}
