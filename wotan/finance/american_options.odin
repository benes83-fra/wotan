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
			// Children of (i, j) are at (i+1, j) [down] and (i+1, j+1) [up]
			// idx(i+1, j+1) = (i+1)*(i+2)/2 + j + 1 = i*(i+1)/2 + i + 1 + j + 1 = node_idx + i + 2
			// idx(i+1, j)   = (i+1)*(i+2)/2 + j     = i*(i+1)/2 + i + 1 + j     = node_idx + i + 1
			up_child := tree[node_idx + i + 2] // ✅ SWAPPED
			down_child := tree[node_idx + i + 1] // ✅ SWAPPED

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
			up_child := euro_tree[node_idx + i + 2] // ✅ SWAPPED
			down_child := euro_tree[node_idx + i + 1] // ✅ SWAPPED
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

FiniteDifferenceResult :: struct {
	price: f64,
	delta: f64,
	gamma: f64,
	theta: f64,
}

// ============================================================================
// Thomas Algorithm for Tridiagonal Systems
// ============================================================================
// Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
// Returns solution in d (overwrites input)
_thomas_algorithm :: proc(
	a: []f64, // Lower diagonal (length n-1)
	b: []f64, // Main diagonal (length n)
	c: []f64, // Upper diagonal (length n-1)
	d: []f64, // Right-hand side (length n, solution overwrites this)
	n: int,
) {
	// Forward elimination
	c_prime := make([]f64, n - 1, context.temp_allocator)
	defer delete(c_prime, context.temp_allocator)

	c_prime[0] = c[0] / b[0]
	d[0] = d[0] / b[0]

	for i in 1 ..< n {
		m := b[i] - a[i - 1] * c_prime[i - 1]
		if i < n - 1 {
			c_prime[i] = c[i] / m
		}
		d[i] = (d[i] - a[i - 1] * d[i - 1]) / m
	}

	// Back substitution
	for i := n - 2; i >= 0; i -= 1 {
		d[i] = d[i] - c_prime[i] * d[i + 1]
	}
}
crank_nicolson_european :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {

	// Grid parameters
	x_min := math.ln(S) - 5.0 * sigma * math.sqrt_f64(T)
	x_max := math.ln(S) + 5.0 * sigma * math.sqrt_f64(T)

	dx := (x_max - x_min) / f64(n_space - 1)
	dt := T / f64(n_time)

	// PDE coefficients: ∂V/∂τ = α∂²V/∂x² + β∂V/∂x + γV
	alpha := 0.5 * sigma * sigma
	beta_coef := r - 0.5 * sigma * sigma
	gamma_coef := -r

	// Crank-Nicolson parameter (θ = 0.5)
	theta := 0.5

	// Discretization coefficients
	rx := alpha * dt / (dx * dx) // α*dt/dx²
	r_beta := beta_coef * dt / (2.0 * dx) // β*dt/(2*dx)
	r_gamma := gamma_coef * dt // γ*dt (this is negative)

	// Allocate grid
	V := make([]f64, n_space, allocator)
	defer delete(V, allocator)

	// Terminal condition (τ = 0, i.e., t = T)
	for i in 0 ..< n_space {
		x_i := x_min + f64(i) * dx
		S_i := math.exp_f64(x_i)

		if opt == .Call {
			V[i] = math.max(S_i - K, 0.0)
		} else {
			V[i] = math.max(K - S_i, 0.0)
		}
	}

	n_unknowns := n_space - 2

	// Tridiagonal system arrays
	a := make([]f64, n_unknowns - 1, context.temp_allocator)
	b := make([]f64, n_unknowns, context.temp_allocator)
	c := make([]f64, n_unknowns - 1, context.temp_allocator)
	d := make([]f64, n_unknowns, context.temp_allocator)
	defer {
		delete(a, context.temp_allocator)
		delete(b, context.temp_allocator)
		delete(c, context.temp_allocator)
		delete(d, context.temp_allocator)
	}

	// Time-stepping (backward from τ = T to τ = 0)
	for n in 0 ..< n_time {
		tau := f64(n + 1) * dt

		if opt == .Call {
			V[0] = 0.0
			V[n_space - 1] = math.exp_f64(x_max) - K * math.exp_f64(-r * tau)
		} else {
			V[0] = K * math.exp_f64(-r * tau) - math.exp_f64(x_min)
			V[n_space - 1] = 0.0
		}

		// Build tridiagonal system with correct coefficients
		for i in 1 ..< n_space - 1 {
			idx := i - 1

			// Lower diagonal (coefficient for V[i-1])
			if idx > 0 {
				a[idx - 1] = -theta * (rx - r_beta)
			}

			// ✅ FIXED: Main diagonal (coefficient for V[i])
			// Must be (2.0 * rx - r_gamma), NOT + r_gamma
			b[idx] = 1.0 + theta * (2.0 * rx - r_gamma)

			// Upper diagonal (coefficient for V[i+1])
			if idx < n_unknowns - 1 {
				c[idx] = -theta * (rx + r_beta)
			}

			// ✅ FIXED: Right-hand side (explicit part)
			// Must be (2.0 * rx - r_gamma), NOT + r_gamma
			d[idx] =
				(1.0 - theta) * (rx - r_beta) * V[i - 1] +
				(1.0 - (1.0 - theta) * (2.0 * rx - r_gamma)) * V[i] +
				(1.0 - theta) * (rx + r_beta) * V[i + 1]
		}

		// Adjust RHS for boundary conditions
		d[0] -= (-theta * (rx - r_beta)) * V[0]
		d[n_unknowns - 1] -= (-theta * (rx + r_beta)) * V[n_space - 1]

		// Solve tridiagonal system
		_thomas_algorithm(a, b, c, d, n_unknowns)

		// Update interior points
		for i in 1 ..< n_space - 1 {
			V[i] = d[i - 1]
		}
	}

	// Extract option price at S
	x_target := math.ln(S)
	i_target := int((x_target - x_min) / dx)

	price: f64
	if i_target >= 0 && i_target < n_space - 1 {
		w := (x_target - (x_min + f64(i_target) * dx)) / dx
		price = V[i_target] * (1.0 - w) + V[i_target + 1] * w
	} else if i_target == n_space - 1 {
		price = V[i_target]
	} else {
		price = 0.0
	}

	// Calculate Greeks
	delta: f64
	if i_target > 0 && i_target < n_space - 1 {
		S_plus := math.exp_f64(x_min + f64(i_target + 1) * dx)
		S_minus := math.exp_f64(x_min + f64(i_target - 1) * dx)
		delta = (V[i_target + 1] - V[i_target - 1]) / (S_plus - S_minus)
	}

	gamma: f64
	if i_target > 0 && i_target < n_space - 1 {
		S_plus := math.exp_f64(x_min + f64(i_target + 1) * dx)
		S_i := math.exp_f64(x_min + f64(i_target) * dx)
		S_minus := math.exp_f64(x_min + f64(i_target - 1) * dx)
		gamma =
			(V[i_target + 1] - 2.0 * V[i_target] + V[i_target - 1]) /
			((S_plus - S_i) * (S_i - S_minus))
	}

	theta_val := -0.5 * sigma * sigma * S * S * gamma - r * S * delta + r * price

	return FiniteDifferenceResult{price = price, delta = delta, gamma = gamma, theta = theta_val}
}

// ============================================================================
// Convenience Wrappers
// ============================================================================
crank_nicolson_call :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return crank_nicolson_european(S, K, T, r, sigma, .Call, n_space, n_time, allocator)
}

crank_nicolson_put :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return crank_nicolson_european(S, K, T, r, sigma, .Put, n_space, n_time, allocator)
}

// ============================================================================
// AMERICAN OPTIONS: Fully Implicit Finite Difference (Brennan-Schwartz)
// ============================================================================
//
// Why Fully Implicit (θ = 1.0) instead of Crank-Nicolson (θ = 0.5)?
// Crank-Nicolson can produce spurious oscillations near the early exercise
// boundary because it is not strictly monotonic (not L-stable).
// Fully Implicit is L-stable and guarantees no oscillations, making it the
// industry standard for American free-boundary problems.
finite_difference_american :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {

	// ✅ FIX 1: Extend grid further to the left (10σ instead of 5σ)
	// This makes S_min much smaller, so V[0] = K is more accurate
	x_min := math.ln(S) - 10.0 * sigma * math.sqrt_f64(T) // Changed from 5.0 to 10.0
	x_max := math.ln(S) + 5.0 * sigma * math.sqrt_f64(T)

	dx := (x_max - x_min) / f64(n_space - 1)
	dt := T / f64(n_time)

	alpha := 0.5 * sigma * sigma
	beta_coef := r - 0.5 * sigma * sigma
	gamma_coef := -r

	rx := alpha * dt / (dx * dx)
	r_beta := beta_coef * dt / (2.0 * dx)
	r_gamma := gamma_coef * dt

	// Pre-compute intrinsic values
	intrinsic := make([]f64, n_space, allocator)
	defer delete(intrinsic, allocator)
	for i in 0 ..< n_space {
		S_i := math.exp_f64(x_min + f64(i) * dx)
		if opt == .Call {
			intrinsic[i] = math.max(S_i - K, 0.0)
		} else {
			intrinsic[i] = math.max(K - S_i, 0.0)
		}
	}

	V := make([]f64, n_space, allocator)
	defer delete(V, allocator)

	// Terminal condition
	for i in 0 ..< n_space {
		V[i] = intrinsic[i]
	}

	n_unknowns := n_space - 2

	a := make([]f64, n_unknowns - 1, context.temp_allocator)
	b := make([]f64, n_unknowns, context.temp_allocator)
	c := make([]f64, n_unknowns - 1, context.temp_allocator)
	d := make([]f64, n_unknowns, context.temp_allocator)
	V_old := make([]f64, n_space, context.temp_allocator)
	defer {
		delete(a, context.temp_allocator)
		delete(b, context.temp_allocator)
		delete(c, context.temp_allocator)
		delete(d, context.temp_allocator)
		delete(V_old, context.temp_allocator)
	}

	// Time-stepping
	for n in 0 ..< n_time {
		tau := f64(n + 1) * dt

		// ✅ FIX 2: Use correct boundary conditions
		if opt == .Call {
			V[0] = 0.0
			V[n_space - 1] = math.exp_f64(x_max) - K * math.exp_f64(-r * tau)
		} else {
			// ✅ For American put at S ≈ 0, value is exactly K (immediate exercise)
			V[0] = K // Changed from K - S_min to K
			V[n_space - 1] = 0.0
		}

		// Build tridiagonal system
		for i in 1 ..< n_space - 1 {
			idx := i - 1

			if idx > 0 {
				a[idx - 1] = -(rx - r_beta)
			}
			b[idx] = 1.0 + 2.0 * rx - r_gamma
			if idx < n_unknowns - 1 {
				c[idx] = -(rx + r_beta)
			}

			d[idx] = V[i]
		}

		// Adjust RHS for boundary conditions
		d[0] -= (-(rx - r_beta)) * V[0]
		d[n_unknowns - 1] -= (-(rx + r_beta)) * V[n_space - 1]

		// Solve
		_thomas_algorithm(a, b, c, d, n_unknowns)

		// Update interior points
		for i in 1 ..< n_space - 1 {
			V[i] = d[i - 1]
		}

		// Project: enforce early exercise constraint
		for i in 0 ..< n_space {
			if V[i] < intrinsic[i] {
				V[i] = intrinsic[i]
			}
		}
	}

	// Extract option price at S
	x_target := math.ln(S)
	i_target := int((x_target - x_min) / dx)

	price: f64
	if i_target >= 0 && i_target < n_space - 1 {
		w := (x_target - (x_min + f64(i_target) * dx)) / dx
		price = V[i_target] * (1.0 - w) + V[i_target + 1] * w
	} else if i_target == n_space - 1 {
		price = V[i_target]
	} else {
		price = 0.0
	}

	// Calculate Greeks
	delta: f64
	if i_target > 0 && i_target < n_space - 1 {
		S_plus := math.exp_f64(x_min + f64(i_target + 1) * dx)
		S_minus := math.exp_f64(x_min + f64(i_target - 1) * dx)
		delta = (V[i_target + 1] - V[i_target - 1]) / (S_plus - S_minus)
	}

	gamma: f64
	if i_target > 0 && i_target < n_space - 1 {
		S_plus := math.exp_f64(x_min + f64(i_target + 1) * dx)
		S_i := math.exp_f64(x_min + f64(i_target) * dx)
		S_minus := math.exp_f64(x_min + f64(i_target - 1) * dx)
		gamma =
			(V[i_target + 1] - 2.0 * V[i_target] + V[i_target - 1]) /
			((S_plus - S_i) * (S_i - S_minus))
	}

	theta_val := -0.5 * sigma * sigma * S * S * gamma - r * S * delta + r * price

	return FiniteDifferenceResult{price = price, delta = delta, gamma = gamma, theta = theta_val}
}

// Convenience Wrappers
fd_american_call :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return finite_difference_american(S, K, T, r, sigma, .Call, n_space, n_time, allocator)
}

fd_american_put :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return finite_difference_american(S, K, T, r, sigma, .Put, n_space, n_time, allocator)
}
