package finance

import "core:math"
import "core:mem"

// ============================================================================
// AMERICAN OPTIONS - BINOMIAL TREE PRICING
// ============================================================================
//
// Binomial trees are the gold standard for American options because they
// naturally handle early exercise (optimal stopping) via backward induction.
//
// This implementation uses the Cox-Ross-Rubinstein (CRR) model, which is:
// - Simple and intuitive
// - Converges to Black-Scholes as n_steps → ∞
// - Naturally handles American exercise
// - Provides Greeks directly from the tree structure

AmericanOptionResult :: struct {
	price:                  f64,
	delta:                  f64,
	gamma:                  f64,
	theta:                  f64,
	early_exercise_premium: f64, // Difference between American and European price
}

// ============================================================================
// CRR Binomial Tree for American Options
// ============================================================================
//
// The CRR model uses:
//   u = exp(σ√Δt)
//   d = 1/u
//   p = (exp((r-q)Δt) - d) / (u - d)
//
// where q is the continuous dividend yield.

american_option_binomial :: proc(
	S: f64, // Spot price
	K: f64, // Strike price
	T: f64, // Time to expiry (years)
	r: f64, // Risk-free rate
	sigma: f64, // Volatility
	q: f64, // Continuous dividend yield (0.0 for non-dividend stocks)
	opt: OptionType,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {

	// Edge case: expired option
	if T <= 0.0 {
		intrinsic := 0.0
		if opt == .Call {
			intrinsic = math.max(S - K, 0.0)
		} else {
			intrinsic = math.max(K - S, 0.0)
		}
		return AmericanOptionResult {
			price = intrinsic,
			delta = 0.0,
			gamma = 0.0,
			theta = 0.0,
			early_exercise_premium = 0.0,
		}
	}

	dt := T / f64(n_steps)
	u := math.exp_f64(sigma * math.sqrt_f64(dt))
	d := 1.0 / u
	p := (math.exp_f64((r - q) * dt) - d) / (u - d)
	disc := math.exp_f64(-r * dt)

	// Allocate tree as flat 1D array for cache efficiency
	// Total nodes = (n_steps + 1) * (n_steps + 2) / 2
	n_nodes := (n_steps + 1) * (n_steps + 2) / 2
	tree := make([]f64, n_nodes, allocator)
	defer delete(tree, allocator)

	// Node index formula: node(i, j) = i*(i+1)/2 + j
	// where i is the time level and j is the number of up moves

	// Step 1: Fill terminal payoffs at expiration
	for j in 0 ..< n_steps + 1 {
		S_T := S * math.pow_f64(u, f64(j)) * math.pow_f64(d, f64(n_steps - j))
		payoff := 0.0
		if opt == .Call {
			payoff = math.max(S_T - K, 0.0)
		} else {
			payoff = math.max(K - S_T, 0.0)
		}
		tree[n_steps * (n_steps + 1) / 2 + j] = payoff
	}

	// Step 2: Backward induction with early exercise check
	for i := n_steps - 1; i >= 0; i -= 1 {
		for j in 0 ..< i + 1 {
			S_ij := S * math.pow_f64(u, f64(j)) * math.pow_f64(d, f64(i - j))

			// Continuation value (holding the option)
			node_idx := i * (i + 1) / 2 + j
			cont := disc * (p * tree[node_idx + i + 1] + (1.0 - p) * tree[node_idx + i])

			// Early exercise value
			exercise := 0.0
			if opt == .Call {
				exercise = math.max(S_ij - K, 0.0)
			} else {
				exercise = math.max(K - S_ij, 0.0)
			}

			// American option: max of continuation and exercise
			tree[node_idx] = math.max(cont, exercise)
		}
	}

	american_price := tree[0]

	// Step 3: Calculate European price for early exercise premium
	// (Same backward induction but without early exercise check)
	euro_tree := make([]f64, n_nodes, allocator)
	defer delete(euro_tree, allocator)

	// Terminal payoffs (same as American)
	for j in 0 ..< n_steps + 1 {
		S_T := S * math.pow_f64(u, f64(j)) * math.pow_f64(d, f64(n_steps - j))
		payoff := 0.0
		if opt == .Call {
			payoff = math.max(S_T - K, 0.0)
		} else {
			payoff = math.max(K - S_T, 0.0)
		}
		euro_tree[n_steps * (n_steps + 1) / 2 + j] = payoff
	}

	// Backward induction WITHOUT early exercise
	for i := n_steps - 1; i >= 0; i -= 1 {
		for j in 0 ..< i + 1 {
			node_idx := i * (i + 1) / 2 + j
			euro_tree[node_idx] =
				disc * (p * euro_tree[node_idx + i + 1] + (1.0 - p) * euro_tree[node_idx + i])
		}
	}

	european_price := euro_tree[0]
	early_exercise_premium := american_price - european_price

	// Step 4: Extract Greeks from the tree structure
	// Delta: sensitivity to spot price
	// Δ = (f(1,1) - f(1,0)) / (S*u - S*d)
	f_1_1 := tree[1 * 2 / 2 + 1] // node (1, 1)
	f_1_0 := tree[1 * 2 / 2 + 0] // node (1, 0)
	delta := (f_1_1 - f_1_0) / (S * u - S * d)

	// Gamma: second derivative with respect to spot
	// Need values at level 2
	f_2_2 := tree[2 * 3 / 2 + 2] // node (2, 2)
	f_2_1 := tree[2 * 3 / 2 + 1] // node (2, 1)
	f_2_0 := tree[2 * 3 / 2 + 0] // node (2, 0)

	// Stock prices at level 2
	S_2_2 := S * u * u
	S_2_1 := S * u * d // = S (since u*d = 1 in CRR)
	S_2_0 := S * d * d

	// Deltas at level 2
	delta_up := (f_2_2 - f_2_1) / (S_2_2 - S_2_1)
	delta_dn := (f_2_1 - f_2_0) / (S_2_1 - S_2_0)

	// Gamma = (delta_up - delta_dn) / h, where h = 0.5 * (S_2_2 - S_2_0)
	h := 0.5 * (S_2_2 - S_2_0)
	gamma := (delta_up - delta_dn) / h

	// Theta: time decay
	// θ = (f(2,1) - f(0,0)) / (2*Δt)
	theta := (f_2_1 - tree[0]) / (2.0 * dt)

	return AmericanOptionResult {
		price = american_price,
		delta = delta,
		gamma = gamma,
		theta = theta,
		early_exercise_premium = early_exercise_premium,
	}
}

// ============================================================================
// Convenience wrappers
// ============================================================================

american_call_binomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_binomial(S, K, T, r, sigma, q, .Call, n_steps, allocator)
}

american_put_binomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_binomial(S, K, T, r, sigma, q, .Put, n_steps, allocator)
}
