package finance

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Option Types
// ============================================================================
OptionType :: enum {
	Call,
	Put,
}

// ============================================================================
// Greeks Container
// ============================================================================
Greeks :: struct {
	price: f64,
	delta: f64, // ∂V/∂S
	gamma: f64, // ∂²V/∂S²
	vega:  f64, // ∂V/∂σ  (per 1% move, divide by 100)
	theta: f64, // ∂V/∂T  (per day, divide by 365)
	rho:   f64, // ∂V/∂r  (per 1% move, divide by 100)
}

// ============================================================================
// Black-Scholes Price (as a tensor for autograd)
// ============================================================================
// d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)
// d2 = d1 - σ*√T
// Call = S*N(d1) - K*exp(-r*T)*N(d2)
// Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)
// ============================================================================
black_scholes_price :: proc(
	S_t: ^t.Tensor, // requires_grad = true
	K_t: ^t.Tensor,
	T_t: ^t.Tensor,
	r_t: ^t.Tensor,
	sigma_t: ^t.Tensor,
	opt: OptionType,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	// ln(S/K)
	S_over_K := t.tensor_div(S_t, K_t)
	ln_S_K := t.tensor_log(S_over_K)

	// σ²/2
	sigma_sq := t.tensor_mul(sigma_t, sigma_t)
	half := _scalar_tensor(0.5, allocator)
	half_sig := t.tensor_mul(sigma_sq, half)

	// r + σ²/2
	r_plus_half_sig := t.tensor_add(r_t, half_sig)

	// (r + σ²/2) * T
	numer_term2 := t.tensor_mul(r_plus_half_sig, T_t)

	// numerator = ln(S/K) + (r + σ²/2)*T
	numer := t.tensor_add(ln_S_K, numer_term2)

	// √T
	sqrt_T := t.tensor_sqrt(T_t)

	// σ * √T
	sig_sqrt_T := t.tensor_mul(sigma_t, sqrt_T)

	// d1 = numer / (σ*√T)
	d1 := t.tensor_div(numer, sig_sqrt_T)

	// d2 = d1 - σ*√T
	d2 := t.tensor_sub(d1, sig_sqrt_T)

	// N(d1), N(d2)
	N_d1 := t.tensor_norm_cdf(d1)
	N_d2 := t.tensor_norm_cdf(d2)

	// exp(-r*T)
	neg_r := t.tensor_neg(r_t)
	rT := t.tensor_mul(neg_r, T_t)
	disc := t.tensor_exp(rT)

	// K * exp(-r*T)
	K_disc := t.tensor_mul(K_t, disc)

	if opt == .Call {
		// C = S*N(d1) - K*exp(-r*T)*N(d2)
		term1 := t.tensor_mul(S_t, N_d1)
		term2 := t.tensor_mul(K_disc, N_d2)
		return t.tensor_sub(term1, term2)
	} else {
		// P = K*exp(-r*T)*N(-d2) - S*N(-d1)
		neg_d1 := t.tensor_neg(d1)
		neg_d2 := t.tensor_neg(d2)
		N_neg_d1 := t.tensor_norm_cdf(neg_d1)
		N_neg_d2 := t.tensor_norm_cdf(neg_d2)

		term1 := t.tensor_mul(K_disc, N_neg_d2)
		term2 := t.tensor_mul(S_t, N_neg_d1)
		return t.tensor_sub(term1, term2)
	}
}
// Helper: create a scalar tensor (1x1)
_scalar_tensor :: proc(val: f64, allocator: mem.Allocator) -> ^t.Tensor {
	data := l.matrix_new(f64, 1, 1, allocator)
	data.data[0] = val

	t := t.tensor_new(data, false, allocator)
	t.owned_by_graph = true // ✅ FIX: Tell the graph cleaner to free this internal constant

	return t
}
// ============================================================================
// Compute All Greeks via Autograd
// ============================================================================
// Creates S, K, T, r, σ as tensors, computes price, and reads off gradients.
// For gamma, we use a finite difference on delta (since autograd gives first order).
// ============================================================================
compute_greeks :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
) -> Greeks {
	// Create parameter tensors with requires_grad = true
	S_t := _make_param(S, allocator)
	K_t := _make_param(K, allocator)
	T_t := _make_param(T, allocator)
	r_t := _make_param(r, allocator)
	sig_t := _make_param(sigma, allocator)

	// Compute price
	price_t := black_scholes_price(S_t, K_t, T_t, r_t, sig_t, opt, allocator)

	// Backward pass
	t.tensor_backward(price_t)

	// Read off Greeks
	price := price_t.data.data[0]
	delta := S_t.grad.data[0]
	vega := sig_t.grad.data[0]
	rho := r_t.grad.data[0]
	theta := T_t.grad.data[0]

	// Gamma via finite difference on delta
	h := 0.01 * S // 1% bump
	delta_up := _compute_delta(S + h, K, T, r, sigma, opt, allocator)
	delta_dn := _compute_delta(S - h, K, T, r, sigma, opt, allocator)
	gamma := (delta_up - delta_dn) / (2.0 * h)

	// Convert theta to per-day (divide by 365)
	theta_per_day := -theta / 365.0

	// Convert vega and rho to per-1% (divide by 100)
	vega_per_pct := vega / 100.0
	rho_per_pct := rho / 100.0

	// ✅ FIX: Free the graph AND the explicitly created leaf nodes
	t.tensor_free_graph(price_t)
	t.tensor_free(S_t)
	t.tensor_free(K_t)
	t.tensor_free(T_t)
	t.tensor_free(r_t)
	t.tensor_free(sig_t)

	return Greeks {
		price = price,
		delta = delta,
		gamma = gamma,
		vega = vega_per_pct,
		theta = theta_per_day,
		rho = rho_per_pct,
	}
}

// Helper: create a parameter tensor
_make_param :: proc(val: f64, allocator: mem.Allocator) -> ^t.Tensor {
	data := l.matrix_new(f64, 1, 1, allocator)
	data.data[0] = val
	return t.tensor_new(data, true, allocator)
}

// Helper: compute delta at a specific S (for gamma finite difference)
_compute_delta :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	allocator: mem.Allocator,
) -> f64 {
	S_t := _make_param(S, allocator)
	K_t := _make_param(K, allocator)
	T_t := _make_param(T, allocator)
	r_t := _make_param(r, allocator)
	sig_t := _make_param(sigma, allocator)

	price_t := black_scholes_price(S_t, K_t, T_t, r_t, sig_t, opt, allocator)
	t.tensor_backward(price_t)
	delta := S_t.grad.data[0]

	// ✅ FIX: Free the graph AND the leaf nodes
	t.tensor_free_graph(price_t)
	t.tensor_free(S_t)
	t.tensor_free(K_t)
	t.tensor_free(T_t)
	t.tensor_free(r_t)
	t.tensor_free(sig_t)

	return delta
}

// ============================================================================
// Implied Volatility via Newton-Raphson
// ============================================================================
// Uses autograd to compute vega (the derivative) at each step.
// Converges in 3-6 iterations typically.
// ============================================================================
implied_volatility :: proc(
	market_price: f64,
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
	max_iter: int = 50,
	tol: f64 = 1e-8,
) -> (
	sigma: f64,
	converged: bool,
	iterations: int,
) {
	// Initial guess: Brenner-Subrahmanyam approximation for ATM
	sigma = math.sqrt(2.0 * math.PI / T) * market_price / S
	if sigma <= 0.0 || sigma > 5.0 {
		sigma = 0.2 // fallback: 20% vol
	}

	for iter in 0 ..< max_iter {
		// Compute price and vega at current sigma
		S_t := _make_param(S, allocator)
		K_t := _make_param(K, allocator)
		T_t := _make_param(T, allocator)
		r_t := _make_param(r, allocator)
		sig_t := _make_param(sigma, allocator)

		price_t := black_scholes_price(S_t, K_t, T_t, r_t, sig_t, opt, allocator)
		t.tensor_backward(price_t)

		model_price := price_t.data.data[0]
		vega := sig_t.grad.data[0] // dV/dσ

		// ✅ FIX: Free the graph AND the leaf nodes inside the loop!
		t.tensor_free_graph(price_t)
		t.tensor_free(S_t)
		t.tensor_free(K_t)
		t.tensor_free(T_t)
		t.tensor_free(r_t)
		t.tensor_free(sig_t)

		// Check convergence
		err := model_price - market_price
		if math.abs(err) < tol {
			return sigma, true, iter + 1
		}

		// Newton step: σ_new = σ - (price - market) / vega
		if math.abs(vega) < 1e-12 {
			// Vega too small, can't continue
			return sigma, false, iter + 1
		}

		sigma = sigma - err / vega

		// Bounds check
		if sigma <= 0.0 {
			sigma = 0.001
		}
		if sigma > 10.0 {
			sigma = 10.0
		}
	}

	return sigma, false, max_iter
}

// ============================================================================
// Convenience: Price + Greeks in one call
// ============================================================================
price_and_greeks :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	greeks: Greeks,
) {
	greeks = compute_greeks(S, K, T, r, sigma, opt, allocator)
	return greeks.price, greeks
}
// ============================================================================
// GARCH-Powered Option Pricing
// ============================================================================

// Forecast the average annualized volatility over a given horizon using GARCH(1,1)
// This creates a dynamic volatility term structure, replacing the flat BS assumption.
garch_term_structure_vol :: proc(
	omega: f64,
	alpha: f64,
	beta: f64,
	current_var: f64, // Today's conditional variance (e.g., garch_result.conditional_var[last])
	horizon_days: int,
) -> f64 {
	if horizon_days <= 0 {return math.sqrt_f64(current_var) * math.sqrt_f64(252.0)}

	persistence := alpha + beta
	if persistence >= 1.0 {persistence = 0.999} 	// Prevent explosion

	long_run_var := omega / (1.0 - persistence)

	// Sum of expected variances over the horizon
	sum_var := 0.0
	for h in 1 ..< horizon_days + 1 {
		// E[σ²_{t+h}] = V_L + (α+β)^{h-1} * (σ²_t - V_L)
		expected_var :=
			long_run_var + math.pow(persistence, f64(h - 1)) * (current_var - long_run_var)
		sum_var += expected_var
	}

	// Average daily variance over the horizon
	avg_daily_var := sum_var / f64(horizon_days)

	// Annualize
	return math.sqrt_f64(avg_daily_var * 252.0)
}

// Price an option and compute Greeks using GARCH-forecasted volatility
garch_price_and_greeks :: proc(
	S: f64,
	K: f64,
	T_years: f64,
	r: f64,
	omega: f64,
	alpha: f64,
	beta: f64,
	current_var: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	greeks: Greeks,
) {
	// 1. Calculate horizon in days (e.g., 0.25 years = 63 days)
	horizon_days := int(math.max(1.0, T_years * 252.0))

	// 2. Get the GARCH-implied volatility for this specific maturity
	sigma_garch := garch_term_structure_vol(omega, alpha, beta, current_var, horizon_days)

	// 3. Plug into your existing autograd Black-Scholes engine
	return price_and_greeks(S, K, T_years, r, sigma_garch, opt, allocator)
}
// ============================================================================
// Advanced Derivatives: Differentiable Monte Carlo
// ============================================================================

// Helper: Create a tensor of standard normal random variables (constant, no grad)
_tensor_randn :: proc(rows, cols: int, allocator: mem.Allocator) -> ^t.Tensor {
	data := l.matrix_new(f64, rows, cols, allocator)
	for i in 0 ..< len(data.data) {
		data.data[i] = rand.float64_normal(0.0, 1.0)
	}
	// requires_grad = false because randomness is not a parameter we optimize
	return t.tensor_new(data, false, allocator)
}


// Price an Asian Call Option and compute exact Greeks via Autograd
monte_carlo_asian_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)

	// 1. Initialize current_S and sigma_tensor as n_paths×1 column vectors
	S_broadcast_data := l.matrix_new(f64, n_paths, 1, allocator)
	sigma_broadcast_data := l.matrix_new(f64, n_paths, 1, allocator)
	for p in 0 ..< n_paths {
		S_broadcast_data.data[p] = S_0
		sigma_broadcast_data.data[p] = sigma
	}

	current_S := t.tensor_new(S_broadcast_data, true, allocator)
	sigma_tensor := t.tensor_new(sigma_broadcast_data, true, allocator)

	// 2. Generate random paths (constant tensor, no grad)
	Z := _tensor_randn(n_paths, n_steps, allocator)

	// Pre-allocate constant n_paths×1 tensor for r*dt
	r_dt_data := l.matrix_new(f64, n_paths, 1, allocator)
	for p in 0 ..< n_paths {
		r_dt_data.data[p] = r * dt
	}
	r_dt_tensor := t.tensor_new(r_dt_data, false, allocator)

	// 3. Simulate paths
	sum_S: ^t.Tensor = nil

	for step in 0 ..< n_steps {
		// Extract column of Z for this step (n_paths×1)
		z_col_data := l.matrix_new(f64, n_paths, 1, allocator)
		for p in 0 ..< n_paths {
			z_col_data.data[p] = Z.data.data[p * n_steps + step]
		}
		z_step := t.tensor_new(z_col_data, false, allocator)

		// vol_term = sigma_tensor * z_step
		vol_term := t.tensor_mul(sigma_tensor, z_step)
		vol_term_scaled := t.tensor_scale(vol_term, sqrt_dt)
		// ✅ DO NOT FREE vol_term or z_step. They are part of the graph.

		// drift_tensor = r*dt - 0.5 * dt * sigma_tensor^2
		sigma_sq := t.tensor_mul(sigma_tensor, sigma_tensor)
		sigma_sq_scaled := t.tensor_scale(sigma_sq, -0.5 * dt)
		// ✅ DO NOT FREE sigma_sq.

		drift_tensor := t.tensor_add(r_dt_tensor, sigma_sq_scaled)
		// ✅ DO NOT FREE sigma_sq_scaled.

		// exponent = drift_tensor + vol_term_scaled
		exponent := t.tensor_add(drift_tensor, vol_term_scaled)
		// ✅ DO NOT FREE drift_tensor or vol_term_scaled.

		// growth = exp(exponent)
		growth := t.tensor_exp(exponent)
		// ✅ DO NOT FREE exponent.

		// next_S = current_S * growth
		next_S := t.tensor_mul(current_S, growth)
		// ✅ DO NOT FREE growth.

		// Accumulate sum safely.
		// We DO NOT free current_S or sum_S here. They are part of the active
		// computation graph needed for the backward pass!
		if step == 0 {
			sum_S = next_S
		} else {
			sum_S = t.tensor_add(sum_S, next_S)
		}

		current_S = next_S
	}

	// 4. Calculate Asian Average (n_paths×1)
	avg_S := t.tensor_scale(sum_S, 1.0 / f64(n_steps))

	// 5. Payoff = ReLU(avg_S - K) (n_paths×1)
	K_data := l.matrix_new(f64, n_paths, 1, allocator)
	for p in 0 ..< n_paths {
		K_data.data[p] = K
	}
	K_tensor := t.tensor_new(K_data, false, allocator)

	payoff_diff := t.tensor_sub(avg_S, K_tensor)
	payoff := t.tensor_relu(payoff_diff)

	// 6. Discount (n_paths×1)
	discount_factor := math.exp(-r * T)
	discounted_payoff := t.tensor_scale(payoff, discount_factor)

	// 7. Mean over paths (scalar)
	price_tensor := t.tensor_mean(discounted_payoff)

	// 8. Backward pass
	t.tensor_backward(price_tensor)

	// 9. Extract results
	price = price_tensor.data.data[0]

	delta_sum := 0.0
	vega_sum := 0.0
	for p in 0 ..< n_paths {
		delta_sum += current_S.grad.data[p]
		vega_sum += sigma_tensor.grad.data[p]
	}

	delta = delta_sum
	vega = vega_sum

	// 10. Cleanup: ONE CALL TO RULE THEM ALL
	// tensor_free_graph recursively frees the ENTIRE computation graph,
	// including current_S, sigma_tensor, r_dt_tensor, K_tensor, and ALL intermediate nodes.
	// We only manually free 'Z' because it was never attached to the grad graph.
	t.tensor_free_graph(price_tensor)
	t.tensor_free(Z)

	return price, delta, vega
}
