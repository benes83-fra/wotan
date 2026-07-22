package finance

import "core:math"
import "core:mem"

// ============================================================================
// AMERICAN OPTIONS - BINOMIAL TREE PRICING
// ============================================================================

AmericanOptionResult :: struct {
	price:                  f64,
	delta:                  f64,
	gamma:                  f64,
	theta:                  f64,
	early_exercise_premium: f64,
}

// ============================================================================
// CRR Binomial Tree for American Options
// ============================================================================
american_option_binomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64,
	opt: OptionType,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {

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

	n_nodes := (n_steps + 1) * (n_steps + 2) / 2
	tree := make([]f64, n_nodes, allocator)
	defer delete(tree, allocator)

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

			node_idx := i * (i + 1) / 2 + j

			// ✅ FIXED: Correct child node indexing
			// Children of (i, j) are at (i+1, j) and (i+1, j+1)
			// (i+1, j) = (i+1)*(i+2)/2 + j = i*(i+1)/2 + (i+1) + j = node_idx + i + 1
			// (i+1, j+1) = node_idx + i + 2
			up_child := tree[node_idx + i + 1]
			down_child := tree[node_idx + i + 2]

			cont := disc * (p * up_child + (1.0 - p) * down_child)

			exercise := 0.0
			if opt == .Call {
				exercise = math.max(S_ij - K, 0.0)
			} else {
				exercise = math.max(K - S_ij, 0.0)
			}

			tree[node_idx] = math.max(cont, exercise)
		}
	}

	american_price := tree[0]

	// Step 3: Calculate European price
	euro_tree := make([]f64, n_nodes, allocator)
	defer delete(euro_tree, allocator)

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

	for i := n_steps - 1; i >= 0; i -= 1 {
		for j in 0 ..< i + 1 {
			node_idx := i * (i + 1) / 2 + j
			// ✅ FIXED: Same correction for European tree
			up_child := euro_tree[node_idx + i + 1]
			down_child := euro_tree[node_idx + i + 2]
			euro_tree[node_idx] = disc * (p * up_child + (1.0 - p) * down_child)
		}
	}

	european_price := euro_tree[0]
	early_exercise_premium := american_price - european_price

	// Step 4: Extract Greeks
	f_1_1 := tree[1 * 2 / 2 + 1]
	f_1_0 := tree[1 * 2 / 2 + 0]
	delta := (f_1_1 - f_1_0) / (S * u - S * d)

	f_2_2 := tree[2 * 3 / 2 + 2]
	f_2_1 := tree[2 * 3 / 2 + 1]
	f_2_0 := tree[2 * 3 / 2 + 0]

	S_2_2 := S * u * u
	S_2_1 := S
	S_2_0 := S * d * d

	delta_up := (f_2_2 - f_2_1) / (S_2_2 - S_2_1)
	delta_dn := (f_2_1 - f_2_0) / (S_2_1 - S_2_0)

	h := 0.5 * (S_2_2 - S_2_0)
	gamma := (delta_up - delta_dn) / h

	theta := (f_2_1 - tree[0]) / (2.0 * dt)

	return AmericanOptionResult {
		price = american_price,
		delta = delta,
		gamma = gamma,
		theta = theta,
		early_exercise_premium = early_exercise_premium,
	}
}

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

// ============================================================================
// TRINOMIAL TREE FOR AMERICAN OPTIONS
// ============================================================================
//
// Why Trinomial over Binomial?
// 1. Converges at O(1/n²) vs O(1/n) for binomial
// 2. No even/odd step parity oscillations
// 3. The "middle" node (no movement) is more realistic
// 4. Greeks are more stable (centered finite differences)
//
// Uses the Kamrad-Ritchken formulation which matches the first two moments
// of the log-normal distribution:
//   Δx = σ√(3Δt)
//   p_u = 1/6 + νΔt/(2Δx) + σ²Δt/(2Δx²)
//   p_m = 2/3 - σ²Δt/Δx²
//   p_d = 1/6 - νΔt/(2Δx) + σ²Δt/(2Δx²)
// where ν = r - q - σ²/2

american_option_trinomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64,
	opt: OptionType,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {

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
	dx := sigma * math.sqrt_f64(3.0 * dt)
	nu := r - q - 0.5 * sigma * sigma

	// Kamrad-Ritchken probabilities (match first two moments of log-normal)
	p_u := 1.0 / 6.0 + (nu * dt) / (2.0 * dx) + (sigma * sigma * dt) / (2.0 * dx * dx)
	p_d := 1.0 / 6.0 - (nu * dt) / (2.0 * dx) + (sigma * sigma * dt) / (2.0 * dx * dx)
	p_m := 1.0 - p_u - p_d

	disc := math.exp_f64(-r * dt)

	// Total nodes in trinomial tree: (n_steps + 1)²
	// Level i has 2i+1 nodes (j ∈ [-i, i])
	// Level i starts at index i², node j at level i is at i² + (j + i)
	n_nodes := (n_steps + 1) * (n_steps + 1)
	tree := make([]f64, n_nodes, allocator)
	defer delete(tree, allocator)

	// Step 1: Fill terminal payoffs at level n_steps
	for j in -n_steps ..= n_steps {
		S_T := S * math.exp_f64(f64(j) * dx)
		payoff := 0.0
		if opt == .Call {
			payoff = math.max(S_T - K, 0.0)
		} else {
			payoff = math.max(K - S_T, 0.0)
		}
		tree[n_steps * n_steps + (j + n_steps)] = payoff
	}

	// Step 2: Backward induction with early exercise check
	for i := n_steps - 1; i >= 0; i -= 1 {
		for j in -i ..= i {
			S_ij := S * math.exp_f64(f64(j) * dx)

			idx := i * i + (j + i)
			// Children at level i+1: (j+1), (j), (j-1)
			idx_up := (i + 1) * (i + 1) + (j + 1 + i + 1)
			idx_mid := (i + 1) * (i + 1) + (j + i + 1)
			idx_dn := (i + 1) * (i + 1) + (j - 1 + i + 1)

			cont := disc * (p_u * tree[idx_up] + p_m * tree[idx_mid] + p_d * tree[idx_dn])

			exercise := 0.0
			if opt == .Call {
				exercise = math.max(S_ij - K, 0.0)
			} else {
				exercise = math.max(K - S_ij, 0.0)
			}

			tree[idx] = math.max(cont, exercise)
		}
	}

	american_price := tree[0]

	// Step 3: European price (no early exercise) for premium calculation
	euro_tree := make([]f64, n_nodes, allocator)
	defer delete(euro_tree, allocator)

	for j in -n_steps ..= n_steps {
		S_T := S * math.exp_f64(f64(j) * dx)
		payoff := 0.0
		if opt == .Call {
			payoff = math.max(S_T - K, 0.0)
		} else {
			payoff = math.max(K - S_T, 0.0)
		}
		euro_tree[n_steps * n_steps + (j + n_steps)] = payoff
	}

	for i := n_steps - 1; i >= 0; i -= 1 {
		for j in -i ..= i {
			idx := i * i + (j + i)
			idx_up := (i + 1) * (i + 1) + (j + 1 + i + 1)
			idx_mid := (i + 1) * (i + 1) + (j + i + 1)
			idx_dn := (i + 1) * (i + 1) + (j - 1 + i + 1)
			euro_tree[idx] =
				disc *
				(p_u * euro_tree[idx_up] + p_m * euro_tree[idx_mid] + p_d * euro_tree[idx_dn])
		}
	}

	european_price := euro_tree[0]
	early_exercise_premium := american_price - european_price

	// Step 4: Extract Greeks using centered differences on the tree
	// At level 1: j ∈ {-1, 0, 1}
	// f(1, 1)  = tree[1 + 2] = tree[3]
	// f(1, 0)  = tree[1 + 1] = tree[2]
	// f(1, -1) = tree[1 + 0] = tree[1]
	f_1_1 := tree[3]
	f_1_0 := tree[2]
	f_1_minus1 := tree[1]

	// Delta: (f(1,1) - f(1,-1)) / (2*S*sinh(Δx))
	sinh_dx := math.sinh(dx)
	delta := (f_1_1 - f_1_minus1) / (2.0 * S * sinh_dx)

	// Gamma: (f(1,1) - 2*f(1,0) + f(1,-1)) / (S²*sinh²(Δx))
	gamma := (f_1_1 - 2.0 * f_1_0 + f_1_minus1) / (S * S * sinh_dx * sinh_dx)

	// Theta: (f(2,0) - f(0,0)) / (2*Δt)
	// f(2, 0) = tree[4 + 2] = tree[6]
	f_2_0 := tree[6]
	theta := (f_2_0 - tree[0]) / (2.0 * dt)

	return AmericanOptionResult {
		price = american_price,
		delta = delta,
		gamma = gamma,
		theta = theta,
		early_exercise_premium = early_exercise_premium,
	}
}

american_call_trinomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_trinomial(S, K, T, r, sigma, q, .Call, n_steps, allocator)
}

american_put_trinomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_trinomial(S, K, T, r, sigma, q, .Put, n_steps, allocator)
}
