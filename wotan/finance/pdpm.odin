// wotan/finance/pdpm.odin
package finance

import w "../core"
import l "../linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Core Structures
// ============================================================================

SDF_Model :: enum {
	Power_Utility, // M = β(C/C0)^(-γ)
	Exponential_Utility, // M = exp(-γ*C)
	Linear_Factor, // M = a + b*R_m
	Quadratic, // M = a + b*R_m + c*R_m²
	Nonparametric, // Estimated from data
}

StochasticDiscountFactor :: struct {
	model_type:      SDF_Model,
	values:          []f64, // Realized SDF values across states
	parameters:      []f64, // Model parameters [a, b, c, ...]
	risk_aversion:   f64, // γ parameter
	discount_factor: f64, // β parameter
	mean:            f64,
	variance:        f64,
}

PayoffDistribution :: struct {
	payoffs:       []f64, // Payoff values in each state
	probabilities: []f64, // State probabilities (risk-neutral or physical)
	n_states:      int,
	mean:          f64,
	variance:      f64,
	skewness:      f64,
	kurtosis:      f64,
	min_payoff:    f64,
	max_payoff:    f64,
}

StatePriceDensity :: struct {
	states:    []f64, // State values (e.g., market returns)
	densities: []f64, // State price densities (Arrow-Debreu prices)
	n_points:  int,
}

PDPMResult :: struct {
	price:                f64, // Fair price of the payoff
	expected_payoff:      f64, // E[payoff]
	risk_premium:         f64, // Price - Expected payoff
	state_prices:         []f64, // Arrow-Debreu state prices
	correlation_with_sdf: f64, // Corr(payoff, SDF)
	certainty_equivalent: f64, // Risk-adjusted value
}

DistributionMetrics :: struct {
	mean:                  f64,
	variance:              f64,
	skewness:              f64,
	kurtosis:              f64,
	value_at_risk_95:      f64,
	value_at_risk_99:      f64,
	expected_shortfall_95: f64,
}

// ============================================================================
// Stochastic Discount Factor Construction
// ============================================================================

// Create SDF from power utility: M = β * (C/C0)^(-γ)
sdf_power_utility :: proc(
	consumption_growth: []f64, // C(t+1)/C(t)
	risk_aversion: f64, // γ (relative risk aversion)
	discount_factor: f64 = 0.99, // β
	allocator: mem.Allocator = context.allocator,
) -> StochasticDiscountFactor {
	n := len(consumption_growth)
	values := make([]f64, n, allocator)
	parameters := make([]f64, 2, allocator)

	for i in 0 ..< n {
		// M = β * (C/C0)^(-γ)
		values[i] = discount_factor * math.pow(consumption_growth[i], -risk_aversion)
	}

	// Compute statistics
	mean := 0.0
	for v in values {mean += v}
	mean /= f64(n)

	variance := 0.0
	for v in values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= f64(n - 1)

	parameters[0] = risk_aversion
	parameters[1] = discount_factor

	return StochasticDiscountFactor {
		model_type = .Power_Utility,
		values = values,
		parameters = parameters,
		risk_aversion = risk_aversion,
		discount_factor = discount_factor,
		mean = mean,
		variance = variance,
	}
}

// Create SDF from exponential utility: M = exp(-γ*C)
sdf_exponential_utility :: proc(
	consumption: []f64, // Absolute consumption levels
	risk_aversion: f64, // γ (absolute risk aversion)
	discount_factor: f64 = 0.99,
	allocator: mem.Allocator = context.allocator,
) -> StochasticDiscountFactor {
	n := len(consumption)
	values := make([]f64, n, allocator)
	parameters := make([]f64, 2, allocator)

	for i in 0 ..< n {
		values[i] = discount_factor * math.exp(-risk_aversion * consumption[i])
	}

	mean := 0.0
	for v in values {mean += v}
	mean /= f64(n)

	variance := 0.0
	for v in values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= f64(n - 1)

	parameters[0] = risk_aversion
	parameters[1] = discount_factor

	return StochasticDiscountFactor {
		model_type = .Exponential_Utility,
		values = values,
		parameters = parameters,
		risk_aversion = risk_aversion,
		discount_factor = discount_factor,
		mean = mean,
		variance = variance,
	}
}

// Create SDF from linear factor model: M = a + b*R_m
sdf_linear_factor :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	allocator: mem.Allocator = context.allocator,
) -> StochasticDiscountFactor {
	n := len(market_returns)

	// Convert to gross returns: R = 1 + r
	gross_returns := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	// Compute statistics
	mean_R := 0.0
	mean_R2 := 0.0
	for r in gross_returns {
		mean_R += r
		mean_R2 += r * r
	}
	mean_R /= f64(n)
	mean_R2 /= f64(n)

	var_R := mean_R2 - mean_R * mean_R

	// Two fundamental pricing constraints:
	// E[M] = 1/(1+rf)
	// E[M * R_m] = 1 (market portfolio prices to 1)

	rf_discount := 1.0 / (1.0 + risk_free_rate)

	// Solve 2x2 system using Cramer's rule:
	// a + b*E[R] = rf_discount
	// a*E[R] + b*E[R²] = 1

	det := var_R // E[R²] - E[R]²

	// Declare a and b in outer scope
	a: f64
	b: f64

	if math.abs(det) < 1e-10 {
		// Degenerate case: use constant SDF
		a = rf_discount
		b = 0.0
	} else {
		// Cramer's rule
		a = (rf_discount * mean_R2 - mean_R) / det
		b = (1.0 - rf_discount * mean_R) / det
	}

	values := make([]f64, n, allocator)
	parameters := make([]f64, 2, allocator)

	for i in 0 ..< n {
		values[i] = a + b * gross_returns[i]
	}

	// Verify constraints
	mean := 0.0
	for v in values {mean += v}
	mean /= f64(n)

	// Verify E[M * R_m] = 1
	pricing_check := 0.0
	for i in 0 ..< n {
		pricing_check += values[i] * gross_returns[i]
	}
	pricing_check /= f64(n)

	variance := 0.0
	for v in values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= f64(n - 1)

	parameters[0] = a
	parameters[1] = b

	fmt.printf("Linear SDF Calibration Check: E[M*R_m] = %.4f (Target: 1.0)\n", pricing_check)

	return StochasticDiscountFactor {
		model_type = .Linear_Factor,
		values = values,
		parameters = parameters,
		risk_aversion = 0.0,
		discount_factor = rf_discount,
		mean = mean,
		variance = variance,
	}
}

// Create SDF from quadratic model: M = a + b*R_m + c*R_m²
sdf_quadratic :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	allocator: mem.Allocator = context.allocator,
) -> StochasticDiscountFactor {
	n := len(market_returns)

	// Compute moments
	mean_rm := 0.0
	mean_rm2 := 0.0
	mean_rm3 := 0.0
	mean_rm4 := 0.0

	for r in market_returns {
		r2 := r * r
		mean_rm += r
		mean_rm2 += r2
		mean_rm3 += r2 * r
		mean_rm4 += r2 * r2
	}

	mean_rm /= f64(n)
	mean_rm2 /= f64(n)
	mean_rm3 /= f64(n)
	mean_rm4 /= f64(n)

	// Solve for quadratic SDF parameters using GMM
	// This is a simplified version - full implementation would use optimization
	a := 1.0 / (1.0 + risk_free_rate)
	b := -0.5 * mean_rm
	c := 0.1

	values := make([]f64, n, allocator)
	parameters := make([]f64, 3, allocator)

	for i in 0 ..< n {
		r := market_returns[i]
		values[i] = a + b * r + c * r * r
	}

	mean := 0.0
	for v in values {mean += v}
	mean /= f64(n)

	variance := 0.0
	for v in values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= f64(n - 1)

	parameters[0] = a
	parameters[1] = b
	parameters[2] = c

	return StochasticDiscountFactor {
		model_type = .Quadratic,
		values = values,
		parameters = parameters,
		risk_aversion = 0.0,
		discount_factor = 1.0 / (1.0 + risk_free_rate),
		mean = mean,
		variance = variance,
	}
}

// ============================================================================
// Payoff Distribution Analysis
// ============================================================================

// Compute distribution metrics for a payoff
analyze_payoff_distribution :: proc(
	payoffs: []f64,
	probabilities: []f64 = nil, // nil = uniform
	allocator: mem.Allocator = context.allocator,
) -> PayoffDistribution {
	n := len(payoffs)

	// Use uniform probabilities if not provided
	probs := probabilities
	if probs == nil {
		probs = make([]f64, n, context.temp_allocator)
		uniform_prob := 1.0 / f64(n)
		for i in 0 ..< n {
			probs[i] = uniform_prob
		}
	}

	// Compute mean
	mean := 0.0
	for i in 0 ..< n {
		mean += probs[i] * payoffs[i]
	}

	// Compute variance
	variance := 0.0
	for i in 0 ..< n {
		diff := payoffs[i] - mean
		variance += probs[i] * diff * diff
	}

	// Compute skewness
	skewness := 0.0
	if variance > 1e-10 {
		std_dev := math.sqrt(variance)
		for i in 0 ..< n {
			diff := payoffs[i] - mean
			skewness += probs[i] * math.pow(diff / std_dev, 3)
		}
	}

	// Compute kurtosis
	kurtosis := 0.0
	if variance > 1e-10 {
		std_dev := math.sqrt(variance)
		for i in 0 ..< n {
			diff := payoffs[i] - mean
			kurtosis += probs[i] * math.pow(diff / std_dev, 4)
		}
		kurtosis -= 3.0 // Excess kurtosis
	}

	// Find min/max
	min_payoff := payoffs[0]
	max_payoff := payoffs[0]
	for p in payoffs {
		if p < min_payoff {min_payoff = p}
		if p > max_payoff {max_payoff = p}
	}

	return PayoffDistribution {
		payoffs = payoffs,
		probabilities = probs,
		n_states = n,
		mean = mean,
		variance = variance,
		skewness = skewness,
		kurtosis = kurtosis,
		min_payoff = min_payoff,
		max_payoff = max_payoff,
	}
}

// ============================================================================
// State Price Density Estimation
// ============================================================================

// Estimate state price density using kernel density estimation
estimate_state_price_density :: proc(
	sdf: ^StochasticDiscountFactor,
	market_returns: []f64,
	bandwidth: f64 = 0.01,
	n_points: int = 100,
	allocator: mem.Allocator = context.allocator,
) -> StatePriceDensity {
	// Find range of market returns
	min_r := market_returns[0]
	max_r := market_returns[0]
	for r in market_returns {
		if r < min_r {min_r = r}
		if r > max_r {max_r = r}
	}

	// Extend range slightly
	range := max_r - min_r
	min_r -= 0.1 * range
	max_r += 0.1 * range

	// Create grid points
	states := make([]f64, n_points, allocator)
	densities := make([]f64, n_points, allocator)

	step := (max_r - min_r) / f64(n_points - 1)
	for i in 0 ..< n_points {
		states[i] = min_r + f64(i) * step
	}

	// Kernel density estimation with SDF weighting
	n_obs := len(market_returns)
	kernel_norm := 1.0 / (bandwidth * math.sqrt_f64(2.0 * math.PI))

	for i in 0 ..< n_points {
		state := states[i]
		density := 0.0

		for j in 0 ..< n_obs {
			// Gaussian kernel
			z := (market_returns[j] - state) / bandwidth
			kernel := kernel_norm * math.exp(-0.5 * z * z)

			// Weight by SDF
			density += sdf.values[j] * kernel
		}

		densities[i] = density / f64(n_obs)

		// Enforce non-negativity (critical for no-arbitrage)
		if densities[i] < 0.0 {
			densities[i] = 0.0
		}
	}

	return StatePriceDensity{states = states, densities = densities, n_points = n_points}
}

// ============================================================================
// Core PDPM Pricing
// ============================================================================

// Price a payoff using the stochastic discount factor
// This is the fundamental pricing equation: P = E[M * X]
price_payoff :: proc(
	payoff: []f64,
	sdf: ^StochasticDiscountFactor,
	probabilities: []f64 = nil, // nil = uniform
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(payoff)

	// Use uniform probabilities if not provided
	probs := probabilities
	if probs == nil {
		probs = make([]f64, n, context.temp_allocator)
		uniform_prob := 1.0 / f64(n)
		for i in 0 ..< n {
			probs[i] = uniform_prob
		}
	}

	// Compute expected payoff: E[X]
	expected_payoff := 0.0
	for i in 0 ..< n {
		expected_payoff += probs[i] * payoff[i]
	}

	// Compute price: P = E[M * X]
	price := 0.0
	for i in 0 ..< n {
		price += probs[i] * sdf.values[i] * payoff[i]
	}

	// Compute correlation between payoff and SDF
	mean_payoff := expected_payoff
	mean_sdf := sdf.mean

	cov := 0.0
	var_payoff := 0.0
	var_sdf := 0.0

	for i in 0 ..< n {
		diff_payoff := payoff[i] - mean_payoff
		diff_sdf := sdf.values[i] - mean_sdf

		cov += probs[i] * diff_payoff * diff_sdf
		var_payoff += probs[i] * diff_payoff * diff_payoff
		var_sdf += probs[i] * diff_sdf * diff_sdf
	}

	correlation := 0.0
	if var_payoff > 1e-10 && var_sdf > 1e-10 {
		correlation = cov / (math.sqrt(var_payoff) * math.sqrt(var_sdf))
	}

	// Risk premium = Price - Expected payoff (discounted)
	risk_premium := price - expected_payoff * sdf.mean

	// Certainty equivalent: CE such that U(CE) = E[U(X)]
	// For simplicity, use: CE = Price / E[M]
	certainty_equivalent := price / sdf.mean

	// Compute state prices (Arrow-Debreu prices)
	state_prices := make([]f64, n, allocator)
	for i in 0 ..< n {
		state_prices[i] = probs[i] * sdf.values[i]
	}

	return PDPMResult {
		price = price,
		expected_payoff = expected_payoff,
		risk_premium = risk_premium,
		state_prices = state_prices,
		correlation_with_sdf = correlation,
		certainty_equivalent = certainty_equivalent,
	}
}

// Price a payoff using state price density (continuous states)
price_payoff_continuous :: proc(
	payoff_function: proc(state: f64) -> f64,
	spd: ^StatePriceDensity,
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := spd.n_points

	// Compute price using trapezoidal rule
	price := 0.0
	expected_payoff := 0.0

	for i in 0 ..< n {
		payoff := payoff_function(spd.states[i])
		price += spd.densities[i] * payoff

		// For expected payoff, we need the physical density, not state prices
		// This is a simplification - in practice you'd need separate physical density
		expected_payoff += payoff / f64(n)
	}

	// Trapezoidal integration
	if n > 1 {
		step := spd.states[1] - spd.states[0]
		price *= step
		expected_payoff *= (spd.states[n - 1] - spd.states[0])
	}

	// Compute state prices
	state_prices := make([]f64, n, allocator)
	for i in 0 ..< n {
		state_prices[i] = spd.densities[i]
	}

	risk_premium := price - expected_payoff

	return PDPMResult {
		price = price,
		expected_payoff = expected_payoff,
		risk_premium = risk_premium,
		state_prices = state_prices,
		correlation_with_sdf = 0.0,
		certainty_equivalent = price,
	}
}

// ============================================================================
// Dybvig's Distributional Pricing
// ============================================================================

// Dybvig's key insight: price depends on the joint distribution of payoff and SDF
// This function implements the full distributional pricing approach
dybvig_distributional_price :: proc(
	payoff: []f64,
	sdf: ^StochasticDiscountFactor,
	n_quantiles: int = 100,
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(payoff)

	// Sort payoffs and corresponding SDF values
	sorted_payoffs := make([]f64, n, context.temp_allocator)
	sorted_sdf := make([]f64, n, context.temp_allocator)
	indices := make([]int, n, context.temp_allocator)

	for i in 0 ..< n {
		sorted_payoffs[i] = payoff[i]
		sorted_sdf[i] = sdf.values[i]
		indices[i] = i
	}

	// Simple bubble sort (replace with quicksort for large n)
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if sorted_payoffs[j] > sorted_payoffs[j + 1] {
				sorted_payoffs[j], sorted_payoffs[j + 1] = sorted_payoffs[j + 1], sorted_payoffs[j]
				sorted_sdf[j], sorted_sdf[j + 1] = sorted_sdf[j + 1], sorted_sdf[j]
			}
		}
	}

	// Compute quantile-based pricing
	// Dybvig showed that price depends on how payoff quantiles align with SDF quantiles
	quantile_size := n / n_quantiles
	quantile_prices := make([]f64, n_quantiles, allocator)

	for q in 0 ..< n_quantiles {
		start_idx := q * quantile_size
		end_idx := min((q + 1) * quantile_size, n)

		// Average payoff and SDF in this quantile
		avg_payoff := 0.0
		avg_sdf := 0.0
		count := 0

		for i in start_idx ..< end_idx {
			avg_payoff += sorted_payoffs[i]
			avg_sdf += sorted_sdf[i]
			count += 1
		}

		avg_payoff /= f64(count)
		avg_sdf /= f64(count)

		// Price contribution from this quantile
		quantile_prices[q] = avg_payoff * avg_sdf * f64(count) / f64(n)
	}

	// Total price
	price := 0.0
	for qp in quantile_prices {
		price += qp
	}

	// Expected payoff
	expected_payoff := 0.0
	for p in payoff {
		expected_payoff += p
	}
	expected_payoff /= f64(n)

	// Risk premium
	risk_premium := price - expected_payoff * sdf.mean

	// State prices
	state_prices := make([]f64, n, allocator)
	for i in 0 ..< n {
		state_prices[i] = sdf.values[i] / f64(n)
	}

	return PDPMResult {
		price = price,
		expected_payoff = expected_payoff,
		risk_premium = risk_premium,
		state_prices = state_prices,
		correlation_with_sdf = 0.0,
		certainty_equivalent = price / sdf.mean,
	}
}

// ============================================================================
// Applications: Option Pricing
// ============================================================================

// Price a European call option using PDPM
price_european_call :: proc(
	spot_price: f64,
	strike_price: f64,
	time_to_maturity: f64,
	sdf: ^StochasticDiscountFactor,
	terminal_prices: []f64, // Simulated terminal prices
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(terminal_prices)

	// Compute payoffs: max(S_T - K, 0)
	payoffs := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		payoffs[i] = math.max(terminal_prices[i] - strike_price, 0.0)
	}

	// Price using PDPM
	return price_payoff(payoffs, sdf, nil, allocator)
}

// Price a European put option using PDPM
price_european_put :: proc(
	spot_price: f64,
	strike_price: f64,
	time_to_maturity: f64,
	sdf: ^StochasticDiscountFactor,
	terminal_prices: []f64,
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(terminal_prices)

	// Compute payoffs: max(K - S_T, 0)
	payoffs := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		payoffs[i] = math.max(strike_price - terminal_prices[i], 0.0)
	}

	return price_payoff(payoffs, sdf, nil, allocator)
}

// Price a structured product (e.g., bull spread)
price_bull_spread :: proc(
	spot_price: f64,
	strike_low: f64,
	strike_high: f64,
	sdf: ^StochasticDiscountFactor,
	terminal_prices: []f64,
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(terminal_prices)

	// Bull spread: long call at K1, short call at K2
	payoffs := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		s_t := terminal_prices[i]
		long_call := math.max(s_t - strike_low, 0.0)
		short_call := math.max(s_t - strike_high, 0.0)
		payoffs[i] = long_call - short_call
	}

	return price_payoff(payoffs, sdf, nil, allocator)
}

// ============================================================================
// Monte Carlo Simulation for PDPM
// ============================================================================

// Simulate terminal prices under physical measure
simulate_terminal_prices :: proc(
	spot_price: f64,
	drift: f64, // Expected return
	volatility: f64, // Annual volatility
	time_to_maturity: f64, // In years
	n_simulations: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	terminal_prices := make([]f64, n_simulations, allocator)

	dt := time_to_maturity
	sqrt_dt := math.sqrt(dt)

	for i in 0 ..< n_simulations {
		// Geometric Brownian Motion
		z := rand.float64_normal(0.0, 1.0)
		terminal_prices[i] =
			spot_price *
			math.exp((drift - 0.5 * volatility * volatility) * dt + volatility * sqrt_dt * z)
	}

	return terminal_prices
}

// ============================================================================
// Utility Functions
// ============================================================================

// Compute distribution metrics
compute_distribution_metrics :: proc(
	values: []f64,
	allocator: mem.Allocator = context.allocator,
) -> DistributionMetrics {
	n := len(values)

	// Mean
	mean := 0.0
	for v in values {mean += v}
	mean /= f64(n)

	// Variance
	variance := 0.0
	for v in values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= f64(n - 1)

	// Skewness
	skewness := 0.0
	if variance > 1e-10 {
		std_dev := math.sqrt(variance)
		for v in values {
			diff := v - mean
			skewness += math.pow(diff / std_dev, 3)
		}
		skewness /= f64(n)
	}

	// Kurtosis (excess)
	kurtosis := 0.0
	if variance > 1e-10 {
		std_dev := math.sqrt(variance)
		for v in values {
			diff := v - mean
			kurtosis += math.pow(diff / std_dev, 4)
		}
		kurtosis /= f64(n)
		kurtosis -= 3.0
	}

	// Sort for VaR
	sorted := make([]f64, n, context.temp_allocator)
	copy(sorted, values)

	// Simple bubble sort (replace with quicksort for large n)
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if sorted[j] > sorted[j + 1] {
				sorted[j], sorted[j + 1] = sorted[j + 1], sorted[j]
			}
		}
	}

	// VaR (95% and 99%)
	idx_95 := int(0.05 * f64(n))
	idx_99 := int(0.01 * f64(n))

	value_at_risk_95 := sorted[idx_95]
	value_at_risk_99 := sorted[idx_99]

	// Expected Shortfall (95%)
	expected_shortfall_95 := 0.0
	for i in 0 ..= idx_95 {
		expected_shortfall_95 += sorted[i]
	}
	expected_shortfall_95 /= f64(idx_95 + 1)

	return DistributionMetrics {
		mean = mean,
		variance = variance,
		skewness = skewness,
		kurtosis = kurtosis,
		value_at_risk_95 = value_at_risk_95,
		value_at_risk_99 = value_at_risk_99,
		expected_shortfall_95 = expected_shortfall_95,
	}
}
