package ml_finance

import "core:fmt"
import "core:math"
import "core:mem"

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

_gaussian_pdf :: proc(x, mu, variance: f64) -> f64 {
	v := variance
	if v < 1e-5 {v = 1e-5}
	diff := x - mu
	return math.exp(-0.5 * diff * diff / v) / math.sqrt(2.0 * math.PI * v)
}

// Initialize HMM with spread means and high state-persistence
hmm_init_simple :: proc(hmm: ^HMM, obs: []f64) {
	N := hmm.n_states
	T := len(obs)
	if T == 0 {return}

	mean := 0.0
	for t in 0 ..< T {mean += obs[t]}
	mean /= f64(T)
	var := 0.0
	for t in 0 ..< T {var += (obs[t] - mean) * (obs[t] - mean)}
	var /= f64(T)

	for i in 0 ..< N {hmm.initial[i] = 1.0 / f64(N)}

	// High probability of staying in the same state (sticky regimes)
	for i in 0 ..< N {
		for j in 0 ..< N {
			if i == j {
				hmm.transition[i * N + j] = 0.95
			} else {
				hmm.transition[i * N + j] = 0.05 / f64(N - 1)
			}
		}
	}

	min_val := obs[0]
	max_val := obs[0]
	for t in 1 ..< T {
		if obs[t] < min_val {min_val = obs[t]}
		if obs[t] > max_val {max_val = obs[t]}
	}

	for i in 0 ..< N {
		hmm.emission_mu[i] = min_val + (max_val - min_val) * f64(i) / f64(N - 1)
		hmm.emission_var[i] = var / f64(N)
	}
}

// ============================================================================
// Forward Algorithm (with Rabiner Scaling to prevent underflow)
// ============================================================================
_forward :: proc(hmm: ^HMM, obs: []f64, alloc: mem.Allocator) -> (alpha: []f64, scale: []f64) {
	T := len(obs)
	N := hmm.n_states
	alpha = make([]f64, T * N, alloc)
	scale = make([]f64, T, alloc)

	c := 0.0
	for i in 0 ..< N {
		b := _gaussian_pdf(obs[0], hmm.emission_mu[i], hmm.emission_var[i])
		alpha[0 * N + i] = hmm.initial[i] * b
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
			b := _gaussian_pdf(obs[t], hmm.emission_mu[j], hmm.emission_var[j])
			alpha[t * N + j] = sum * b
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
_backward :: proc(hmm: ^HMM, obs: []f64, scale: []f64, alloc: mem.Allocator) -> []f64 {
	T := len(obs)
	N := hmm.n_states
	beta := make([]f64, T * N, alloc)

	for i in 0 ..< N {beta[(T - 1) * N + i] = 1.0}

	for t := T - 2; t >= 0; t -= 1 {
		for i in 0 ..< N {
			sum := 0.0
			for j in 0 ..< N {
				b := _gaussian_pdf(obs[t + 1], hmm.emission_mu[j], hmm.emission_var[j])
				sum += hmm.transition[i * N + j] * b * beta[(t + 1) * N + j]
			}
			beta[t * N + i] = scale[t + 1] * sum
		}
	}
	return beta
}

// ============================================================================
// Baum-Welch (EM Algorithm) for Training
// ============================================================================
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
	alloc := hmm.allocator

	prev_ll := -math.F64_MAX

	gamma := make([]f64, T * N, alloc)
	xi := make([]f64, (T - 1) * N * N, alloc)
	defer {
		delete(gamma, alloc)
		delete(xi, alloc)
	}

	for iter in 0 ..< max_iter {
		// E-Step: Forward-Backward
		alpha, scale := _forward(hmm, obs, alloc)
		beta := _backward(hmm, obs, scale, alloc)

		// Compute Log-Likelihood: ln(P(O)) = -sum(ln(scale[t]))
		ll := 0.0
		for t in 0 ..< T {
			if scale[t] > 0 {ll -= math.ln(scale[t])}
		}

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

		// Compute xi (transition posteriors)
		for t in 0 ..< T - 1 {
			sum_xi := 0.0
			for i in 0 ..< N {
				for j in 0 ..< N {
					b := _gaussian_pdf(obs[t + 1], hmm.emission_mu[j], hmm.emission_var[j])
					val :=
						scale[t + 1] *
						alpha[t * N + i] *
						hmm.transition[i * N + j] *
						b *
						beta[(t + 1) * N + j]
					xi[t * N * N + i * N + j] = val
					sum_xi += val
				}
			}
			if sum_xi > 0 {
				for i in 0 ..< N {
					for j in 0 ..< N {xi[t * N * N + i * N + j] /= sum_xi}
				}
			}
		}

		// M-Step: Update Parameters
		for i in 0 ..< N {hmm.initial[i] = gamma[0 * N + i]}

		for i in 0 ..< N {
			sum_den := 0.0
			for t in 0 ..< T - 1 {sum_den += gamma[t * N + i]}
			if sum_den > 0 {
				for j in 0 ..< N {
					num := 0.0
					for t in 0 ..< T - 1 {num += xi[t * N * N + i * N + j]}
					hmm.transition[i * N + j] = num / sum_den
				}
			}
		}

		for i in 0 ..< N {
			sum_gamma := 0.0
			sum_x := 0.0
			sum_x2 := 0.0
			for t in 0 ..< T {
				g := gamma[t * N + i]
				sum_gamma += g
				sum_x += g * obs[t]
				sum_x2 += g * obs[t] * obs[t]
			}
			if sum_gamma > 0 {
				mu := sum_x / sum_gamma
				var := (sum_x2 / sum_gamma) - (mu * mu)
				if var < 1e-5 {var = 1e-5}
				hmm.emission_mu[i] = mu
				hmm.emission_var[i] = var
			}
		}

		if math.abs(ll - prev_ll) < tol {
			delete(alpha, alloc)
			delete(scale, alloc)
			delete(beta, alloc)
			return ll, true
		}
		prev_ll = ll

		delete(alpha, alloc)
		delete(scale, alloc)
		delete(beta, alloc)
	}

	return prev_ll, false
}

// ============================================================================
// Viterbi Algorithm (Most Likely Regime Sequence)
// ============================================================================
hmm_decode :: proc(hmm: ^HMM, obs: []f64, alloc: mem.Allocator = context.allocator) -> []int {
	T := len(obs)
	N := hmm.n_states
	if T == 0 {return make([]int, 0, alloc)}

	delta := make([]f64, T * N, alloc)
	psi := make([]int, T * N, alloc)
	defer {
		delete(delta, alloc)
		delete(psi, alloc)
	}

	for i in 0 ..< N {
		b := _gaussian_pdf(obs[0], hmm.emission_mu[i], hmm.emission_var[i])
		val := math.ln(hmm.initial[i] + 1e-300) + math.ln(b + 1e-300)
		delta[0 * N + i] = val
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
			b := _gaussian_pdf(obs[t], hmm.emission_mu[j], hmm.emission_var[j])
			delta[t * N + j] = max_val + math.ln(b + 1e-300)
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
