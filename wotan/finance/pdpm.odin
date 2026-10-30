// wotan/finance/pdpm.odin
package finance

import w "../core"

import l "../linalg"
import opt "../optimize"
import p "../plot"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

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

// On a uniform grid of states: q(s) = E[M | R = s] f(s), the state-price
// density (∫ q ds = E[M]), and f(s), the physical density of the states.
// Free with free_state_price_density.
StatePriceDensity :: struct {
	states:    []f64, // State values (e.g., market returns)
	densities: []f64, // State price densities (Arrow-Debreu prices per unit of state)
	n_points:  int,
	physical:  []f64, // Physical density of the states (nil: unknown)
}

free_state_price_density :: proc(
	spd: ^StatePriceDensity,
	allocator: mem.Allocator = context.allocator,
) {
	delete(spd.states, allocator)
	delete(spd.densities, allocator)
	delete(spd.physical, allocator)
	spd^ = {}
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

// Create SDF from linear factor model: M = a + b R, R = 1 + r the gross market
// return, with a and b solving E[M] = 1/(1+rf) and E[M R] = 1 (sample moments)
sdf_linear_factor :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	allocator: mem.Allocator = context.allocator,
) -> StochasticDiscountFactor {
	sdf := sdf_quadratic(market_returns, risk_free_rate, allocator, 0.0)
	a, b, _ := _sdf_quadratic_coefficients(market_returns, risk_free_rate, 0.0)
	delete(sdf.parameters, allocator)
	sdf.parameters = make([]f64, 2, allocator)
	sdf.parameters[0], sdf.parameters[1] = a, b
	sdf.model_type = .Linear_Factor
	return sdf
}

// Create SDF from quadratic model: M = a + b R + c R², R = 1 + r the gross market
// return. For the given curvature c, a and b solve E[M] = 1/(1+rf) and
// E[M R] = 1 (sample moments), so the SDF prices the bond and the market; c = 0
// is sdf_linear_factor. (Two assets cannot identify c; calibrate_sdf_quadratic
// fits it to option prices.) parameters = [a, b, c].
sdf_quadratic :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	allocator: mem.Allocator = context.allocator,
	c: f64 = 0.0,
) -> StochasticDiscountFactor {
	a, b, d := _sdf_quadratic_coefficients(market_returns, risk_free_rate, c)
	n := len(market_returns)
	values := make([]f64, n, allocator)
	for r, i in market_returns {
		R := 1.0 + r
		values[i] = a + b * R + c * R * R
	}
	mean := 0.0
	for x in values {mean += x}
	mean /= f64(max(n, 1))
	variance := 0.0
	for x in values {variance += (x - mean) * (x - mean)}
	variance /= f64(max(n - 1, 1))

	parameters := make([]f64, 3, allocator)
	parameters[0], parameters[1], parameters[2] = a, b, c
	return StochasticDiscountFactor {
		model_type = .Quadratic,
		values = values,
		parameters = parameters,
		risk_aversion = 0.0,
		discount_factor = d,
		mean = mean,
		variance = variance,
	}
}

// a, b of M = a + b R + c R² with E[M] = d = 1/(1+rf) and E[M R] = 1
@(private = "file")
_sdf_quadratic_coefficients :: proc(
	market_returns: []f64,
	risk_free_rate, c: f64,
) -> (
	a, b, d: f64,
) {
	n := f64(max(len(market_returns), 1))
	m1, m2, m3 := 0.0, 0.0, 0.0
	for r in market_returns {
		R := 1.0 + r
		m1 += R
		m2 += R * R
		m3 += R * R * R
	}
	m1 /= n
	m2 /= n
	m3 /= n
	// a + b m1 = d − c m2,  a m1 + b m2 = 1 − c m3
	d = 1.0 / (1.0 + risk_free_rate)
	u, v := d - c * m2, 1.0 - c * m3
	det := m2 - m1 * m1
	if abs(det) < 1e-14 * max(m2, 1e-300) {
		return u, 0.0, d
	}
	b = (v - m1 * u) / det
	a = u - b * m1
	return
}

// ============================================================================
// Payoff Distribution Analysis
// ============================================================================

// Probability-weighted (population) moments of a payoff: mean, variance,
// skewness and excess kurtosis. With probabilities = nil the states are
// equally likely and `probabilities` in the result is nil.
analyze_payoff_distribution :: proc(
	payoffs: []f64,
	probabilities: []f64 = nil, // nil = uniform
	allocator: mem.Allocator = context.allocator,
) -> PayoffDistribution {
	n := len(payoffs)
	if n == 0 {return {}}
	wt :: #force_inline proc(probs: []f64, i, n: int) -> f64 {return(
			probs[i] if probs != nil else 1.0 / f64(n) \
		)}

	mean := 0.0
	for i in 0 ..< n {mean += wt(probabilities, i, n) * payoffs[i]}
	m2, m3, m4 := 0.0, 0.0, 0.0
	for i in 0 ..< n {
		d := payoffs[i] - mean
		q := wt(probabilities, i, n)
		m2 += q * d * d
		m3 += q * d * d * d
		m4 += q * d * d * d * d
	}
	skewness, kurtosis := 0.0, 0.0
	if m2 > 1e-300 {
		skewness = m3 / math.pow(m2, 1.5)
		kurtosis = m4 / (m2 * m2) - 3.0
	}
	min_payoff, max_payoff := payoffs[0], payoffs[0]
	for x in payoffs {
		min_payoff = min(min_payoff, x)
		max_payoff = max(max_payoff, x)
	}
	return PayoffDistribution {
		payoffs = payoffs,
		probabilities = probabilities,
		n_states = n,
		mean = mean,
		variance = m2,
		skewness = skewness,
		kurtosis = kurtosis,
		min_payoff = min_payoff,
		max_payoff = max_payoff,
	}
}

// ============================================================================
// State Price Density Estimation
// ============================================================================

// Gaussian-kernel estimates on a uniform grid spanning the observed returns ± 10%
// of their range: q(s) = (1/n) Σ M_j K_h(s − r_j) and f(s) = (1/n) Σ K_h(s − r_j).
// q is floored at 0 (a linear SDF can be negative in extreme states).
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
	physical := make([]f64, n_points, allocator)

	step := (max_r - min_r) / f64(n_points - 1)
	for i in 0 ..< n_points {
		states[i] = min_r + f64(i) * step
	}

	// Kernel density estimation with SDF weighting
	n_obs := len(market_returns)
	kernel_norm := 1.0 / (bandwidth * math.sqrt_f64(2.0 * math.PI))

	for i in 0 ..< n_points {
		state := states[i]
		density, phys := 0.0, 0.0

		for j in 0 ..< n_obs {
			// Gaussian kernel
			z := (market_returns[j] - state) / bandwidth
			kernel := kernel_norm * math.exp(-0.5 * z * z)

			// Weight by SDF
			density += sdf.values[j] * kernel
			phys += kernel
		}

		densities[i] = density / f64(n_obs)
		physical[i] = phys / f64(n_obs)

		// Enforce non-negativity (critical for no-arbitrage)
		if densities[i] < 0.0 {
			densities[i] = 0.0
		}
	}

	return StatePriceDensity {
		states = states,
		densities = densities,
		n_points = n_points,
		physical = physical,
	}
}

// ============================================================================
// Core PDPM Pricing
// ============================================================================

// Price a payoff using the stochastic discount factor: P = E[M X] = Σ p_i M_i X_i.
// risk_premium = P − E[X] E[M] = Cov(M, X) (the price of the payoff's risk),
// certainty_equivalent = P / E[M] (the risk-neutral expected payoff); all
// moments under `probabilities` (nil = uniform).
price_payoff :: proc(
	payoff: []f64,
	sdf: ^StochasticDiscountFactor,
	probabilities: []f64 = nil, // nil = uniform
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(payoff)
	wt :: #force_inline proc(probs: []f64, i, n: int) -> f64 {return(
			probs[i] if probs != nil else 1.0 / f64(n) \
		)}

	expected_payoff, mean_sdf, price := 0.0, 0.0, 0.0
	for i in 0 ..< n {
		q := wt(probabilities, i, n)
		expected_payoff += q * payoff[i]
		mean_sdf += q * sdf.values[i]
		price += q * sdf.values[i] * payoff[i]
	}

	cov, var_payoff, var_sdf := 0.0, 0.0, 0.0
	for i in 0 ..< n {
		q := wt(probabilities, i, n)
		dx := payoff[i] - expected_payoff
		dm := sdf.values[i] - mean_sdf
		cov += q * dx * dm
		var_payoff += q * dx * dx
		var_sdf += q * dm * dm
	}
	correlation := 0.0
	if var_payoff > 0.0 && var_sdf > 0.0 {
		correlation = cov / math.sqrt(var_payoff * var_sdf)
	}

	state_prices := make([]f64, n, allocator)
	for i in 0 ..< n {state_prices[i] = wt(probabilities, i, n) * sdf.values[i]}

	return PDPMResult {
		price = price,
		expected_payoff = expected_payoff,
		risk_premium = price - expected_payoff * mean_sdf,
		state_prices = state_prices,
		correlation_with_sdf = correlation,
		certainty_equivalent = price / mean_sdf,
	}
}

// Price a payoff on a state-price density: P = ∫ X(s) q(s) ds by the trapezoidal
// rule. E[X] uses the physical density when the SPD has one (otherwise the
// risk-neutral density q / ∫q); risk_premium = P − E[X] ∫q, as in price_payoff.
// state_prices are the Arrow-Debreu prices of the grid cells, q_i · w_i.
price_payoff_continuous :: proc(
	payoff_function: proc(state: f64) -> f64,
	spd: ^StatePriceDensity,
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := spd.n_points
	state_prices := make([]f64, n, allocator)
	if n < 2 {return PDPMResult{state_prices = state_prices}}

	step := spd.states[1] - spd.states[0]
	price, bond, ex, mass := 0.0, 0.0, 0.0, 0.0
	for i in 0 ..< n {
		w := step * (0.5 if i == 0 || i == n - 1 else 1.0)
		x := payoff_function(spd.states[i])
		state_prices[i] = spd.densities[i] * w
		price += x * state_prices[i]
		bond += state_prices[i]
		f := spd.physical[i] if spd.physical != nil else spd.densities[i]
		ex += x * f * w
		mass += f * w
	}
	expected_payoff := ex / mass if mass > 0.0 else 0.0

	return PDPMResult {
		price = price,
		expected_payoff = expected_payoff,
		risk_premium = price - expected_payoff * bond,
		state_prices = state_prices,
		correlation_with_sdf = 0.0,
		certainty_equivalent = price / bond if bond > 0.0 else 0.0,
	}
}

// ============================================================================
// Dybvig's Distributional Pricing
// ============================================================================

// Dybvig's (1988) payoff distribution pricing model: with equally likely states
// the cheapest payoff having the same distribution as X pays its order
// statistics in the reverse order of the SDF, so the distributional price is
//   P_D = (1/n) Σ_i X_(i) M_(n+1−i)   (X ascending, M descending)
// and P − P_D ≥ 0 is the efficiency loss of the actual payoff. Result: price =
// P_D, certainty_equivalent = P_D / E[M], risk_premium = P_D − E[X] E[M].
// n_quantiles is unused (kept for compatibility).
dybvig_distributional_price :: proc(
	payoff: []f64,
	sdf: ^StochasticDiscountFactor,
	n_quantiles: int = 100,
	allocator: mem.Allocator = context.allocator,
) -> PDPMResult {
	n := len(payoff)
	xs := make([]f64, n, context.temp_allocator)
	ms := make([]f64, n, context.temp_allocator)
	defer {delete(xs, context.temp_allocator); delete(ms, context.temp_allocator)}
	copy(xs, payoff)
	copy(ms, sdf.values[:n])
	slice.sort(xs)
	slice.sort(ms)

	price, expected_payoff, mean_sdf := 0.0, 0.0, 0.0
	for i in 0 ..< n {
		price += xs[i] * ms[n - 1 - i]
		expected_payoff += xs[i]
		mean_sdf += ms[i]
	}
	inv := 1.0 / f64(max(n, 1))
	price *= inv
	expected_payoff *= inv
	mean_sdf *= inv

	state_prices := make([]f64, n, allocator)
	for i in 0 ..< n {state_prices[i] = sdf.values[i] * inv}

	return PDPMResult {
		price = price,
		expected_payoff = expected_payoff,
		risk_premium = price - expected_payoff * mean_sdf,
		state_prices = state_prices,
		correlation_with_sdf = 0.0,
		certainty_equivalent = price / mean_sdf,
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

// Sample distribution metrics: mean, unbiased variance (n − 1), population
// skewness m3/m2^1.5 and excess kurtosis m4/m2² − 3 (scipy.stats defaults),
// VaR_α as the lower empirical α-quantile x_(⌈αn⌉) and ES_95 as the mean of
// the ⌈0.05 n⌉ smallest values (both in the units of `values`, ≤ 0 for losses).
compute_distribution_metrics :: proc(
	values: []f64,
	allocator: mem.Allocator = context.allocator,
) -> DistributionMetrics {
	n := len(values)
	if n == 0 {return {}}
	mean := 0.0
	for v in values {mean += v}
	mean /= f64(n)
	m2, m3, m4 := 0.0, 0.0, 0.0
	for v in values {
		d := v - mean
		m2 += d * d
		m3 += d * d * d
		m4 += d * d * d * d
	}
	variance := m2 / f64(max(n - 1, 1))
	m2 /= f64(n)
	m3 /= f64(n)
	m4 /= f64(n)
	skewness, kurtosis := 0.0, 0.0
	if m2 > 1e-300 {
		skewness = m3 / math.pow(m2, 1.5)
		kurtosis = m4 / (m2 * m2) - 3.0
	}

	sorted := make([]f64, n, context.temp_allocator)
	defer delete(sorted, context.temp_allocator)
	copy(sorted, values)
	slice.sort(sorted)
	k95 := clamp(int(math.ceil(0.05 * f64(n) - 1e-9)), 1, n)
	k99 := clamp(int(math.ceil(0.01 * f64(n) - 1e-9)), 1, n)
	es := 0.0
	for i in 0 ..< k95 {es += sorted[i]}

	return DistributionMetrics {
		mean = mean,
		variance = variance,
		skewness = skewness,
		kurtosis = kurtosis,
		value_at_risk_95 = sorted[k95 - 1],
		value_at_risk_99 = sorted[k99 - 1],
		expected_shortfall_95 = es / f64(k95),
	}
}

// Plot the complete PDPM analysis: payoff distribution, SDF, and state prices
plot_pdpm_analysis :: proc(
	payoffs: []f64,
	sdf: ^StochasticDiscountFactor,
	market_returns: []f64,
	title: string,
	output_path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	n := len(payoffs)

	// ========================================================================
	// Plot 1: Payoff Distribution (Physical Measure)
	// ========================================================================
	n_bins := 50
	min_payoff := payoffs[0]
	max_payoff := payoffs[0]
	for p in payoffs {
		if p < min_payoff {min_payoff = p}
		if p > max_payoff {max_payoff = p}
	}

	bin_width := (max_payoff - min_payoff) / f64(n_bins)
	bin_counts := make([]f64, n_bins, context.temp_allocator)
	bin_centers := make([]f64, n_bins, context.temp_allocator)

	for i in 0 ..< n_bins {
		bin_centers[i] = min_payoff + (f64(i) + 0.5) * bin_width
		bin_counts[i] = 0.0
	}

	for p in payoffs {
		bin_idx := int((p - min_payoff) / bin_width)
		if bin_idx >= n_bins {bin_idx = n_bins - 1}
		if bin_idx < 0 {bin_idx = 0}
		bin_counts[bin_idx] += 1.0
	}

	// Normalize to density
	for i in 0 ..< n_bins {
		bin_counts[i] /= f64(n) * bin_width
	}

	// ========================================================================
	// Plot 2: SDF Values vs Market Returns
	// ========================================================================
	sorted_returns := make([]f64, n, context.temp_allocator)
	sorted_sdf := make([]f64, n, context.temp_allocator)

	for i in 0 ..< n {
		sorted_returns[i] = market_returns[i]
		sorted_sdf[i] = sdf.values[i]
	}

	// Simple bubble sort
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if sorted_returns[j] > sorted_returns[j + 1] {
				sorted_returns[j], sorted_returns[j + 1] = sorted_returns[j + 1], sorted_returns[j]
				sorted_sdf[j], sorted_sdf[j + 1] = sorted_sdf[j + 1], sorted_sdf[j]
			}
		}
	}

	// ========================================================================
	// Plot 3: State Price Density
	// ========================================================================
	spd := estimate_state_price_density(sdf, market_returns, 0.02, 100, context.temp_allocator)
	defer free_state_price_density(&spd, context.temp_allocator)

	// ========================================================================
	// Plot 4: Risk-Neutral Distribution (SDF-weighted)
	// ========================================================================
	risk_neutral_density := make([]f64, n_bins, context.temp_allocator)
	for i in 0 ..< n_bins {
		bin_sum := 0.0
		bin_count := 0
		for j in 0 ..< n {
			bin_idx := int((payoffs[j] - min_payoff) / bin_width)
			if bin_idx >= n_bins {bin_idx = n_bins - 1}
			if bin_idx < 0 {bin_idx = 0}
			if bin_idx == i {
				bin_sum += sdf.values[j]
				bin_count += 1
			}
		}

		if bin_count > 0 {
			avg_sdf := bin_sum / f64(bin_count)
			risk_neutral_density[i] = bin_counts[i] * avg_sdf / sdf.mean
		}
	}

	// ========================================================================
	// Create 4 separate plots
	// ========================================================================

	// Use fmt.tprintf for string formatting
	path1 := fmt.tprintf("%s_1_payoff_dist.png", output_path)
	path2 := fmt.tprintf("%s_2_sdf_values.png", output_path)
	path3 := fmt.tprintf("%s_3_state_price_density.png", output_path)
	path4 := fmt.tprintf("%s_4_risk_neutral_dist.png", output_path)

	// Plot 1: Payoff Distribution
	xs1 := make([]f64, n_bins, allocator)
	ys1 := make([]f64, n_bins, allocator)
	defer {
		delete(xs1, allocator)
		delete(ys1, allocator)
	}

	for i in 0 ..< n_bins {
		xs1[i] = bin_centers[i]
		ys1[i] = bin_counts[i]
	}

	lines1 := []p.LineData {
		p.LineData{xs = xs1, ys = ys1, color = p.BLUE, style = .Solid, label = "Physical Density"},
	}

	config1 := p.DEFAULT_PLOT_CONFIG
	config1.title = fmt.tprintf("%s - Payoff Distribution (Physical Measure P)", title)
	config1.x_label = "Payoff Value"
	config1.y_label = "Density"
	config1.show_grid = true

	ok1 := p.multi_line_png(lines1, path1, config1, allocator)

	// Plot 2: SDF Values
	xs2 := make([]f64, n, allocator)
	ys2 := make([]f64, n, allocator)
	defer {
		delete(xs2, allocator)
		delete(ys2, allocator)
	}

	for i in 0 ..< n {
		xs2[i] = sorted_returns[i]
		ys2[i] = sorted_sdf[i]
	}

	lines2 := []p.LineData {
		p.LineData{xs = xs2, ys = ys2, color = p.RED, style = .Solid, label = "SDF M"},
	}

	config2 := p.DEFAULT_PLOT_CONFIG
	config2.title = fmt.tprintf("%s - Stochastic Discount Factor", title)
	config2.x_label = "Market Return"
	config2.y_label = "SDF Value M"
	config2.show_grid = true

	ok2 := p.multi_line_png(lines2, path2, config2, allocator)

	// Plot 3: State Price Density
	xs3 := make([]f64, spd.n_points, allocator)
	ys3 := make([]f64, spd.n_points, allocator)
	defer {
		delete(xs3, allocator)
		delete(ys3, allocator)
	}

	for i in 0 ..< spd.n_points {
		xs3[i] = spd.states[i]
		ys3[i] = spd.densities[i]
	}

	lines3 := []p.LineData {
		p.LineData {
			xs = xs3,
			ys = ys3,
			color = p.Color{0, 150, 0, 255},
			style = .Solid,
			label = "State Price Density",
		},
	}

	config3 := p.DEFAULT_PLOT_CONFIG
	config3.title = fmt.tprintf("%s - State Price Density (Arrow-Debreu Prices)", title)
	config3.x_label = "State (Market Return)"
	config3.y_label = "State Price Density"
	config3.show_grid = true

	ok3 := p.multi_line_png(lines3, path3, config3, allocator)

	// Plot 4: Risk-Neutral Distribution
	xs4 := make([]f64, n_bins, allocator)
	ys4 := make([]f64, n_bins, allocator)
	ys4_phys := make([]f64, n_bins, allocator)
	defer {
		delete(xs4, allocator)
		delete(ys4, allocator)
		delete(ys4_phys, allocator)
	}

	for i in 0 ..< n_bins {
		xs4[i] = bin_centers[i]
		ys4[i] = risk_neutral_density[i]
		ys4_phys[i] = bin_counts[i]
	}

	lines4 := []p.LineData {
		p.LineData{xs = xs4, ys = ys4_phys, color = p.BLUE, style = .Solid, label = "Physical P"},
		p.LineData{xs = xs4, ys = ys4, color = p.RED, style = .Solid, label = "Risk-Neutral Q"},
	}

	config4 := p.DEFAULT_PLOT_CONFIG
	config4.title = fmt.tprintf("%s - Physical vs Risk-Neutral Distribution", title)
	config4.x_label = "Payoff Value"
	config4.y_label = "Density"
	config4.show_grid = true

	ok4 := p.multi_line_png(lines4, path4, config4, allocator)

	return ok1 && ok2 && ok3 && ok4
}
plot_option_payoff_profile :: proc(
	terminal_prices: []f64,
	strike: f64,
	option_type: string, // "call" or "put"
	sdf: ^StochasticDiscountFactor,
	output_path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	n := len(terminal_prices)

	// Compute payoffs
	payoffs := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		if option_type == "call" {
			payoffs[i] = math.max(terminal_prices[i] - strike, 0.0)
		} else {
			payoffs[i] = math.max(strike - terminal_prices[i], 0.0)
		}
	}

	// Price the option
	price := 0.0
	for i in 0 ..< n {
		price += sdf.values[i] * payoffs[i]
	}
	price /= f64(n)

	// Sort by terminal price for clean plot
	sorted_prices := make([]f64, n, context.temp_allocator)
	sorted_payoffs := make([]f64, n, context.temp_allocator)

	for i in 0 ..< n {
		sorted_prices[i] = terminal_prices[i]
		sorted_payoffs[i] = payoffs[i]
	}

	// Simple sort
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if sorted_prices[j] > sorted_prices[j + 1] {
				sorted_prices[j], sorted_prices[j + 1] = sorted_prices[j + 1], sorted_prices[j]
				sorted_payoffs[j], sorted_payoffs[j + 1] = sorted_payoffs[j + 1], sorted_payoffs[j]
			}
		}
	}

	// Create plot data
	xs := make([]f64, n, allocator)
	ys := make([]f64, n, allocator)
	defer {
		delete(xs, allocator)
		delete(ys, allocator)
	}

	for i in 0 ..< n {
		xs[i] = sorted_prices[i]
		ys[i] = sorted_payoffs[i]
	}

	// Add horizontal line for option price
	xs_price := []f64{xs[0], xs[n - 1]}
	ys_price := []f64{price, price}

	lines := []p.LineData {
		p.LineData{xs = xs, ys = ys, color = p.BLUE, style = .Solid, label = "Payoff at Maturity"},
		p.LineData {
			xs = xs_price,
			ys = ys_price,
			color = p.RED,
			style = .Dashed,
			label = fmt.tprintf("PDPM Price: $%.2f", price),
		},
	}

	config := p.DEFAULT_PLOT_CONFIG
	title_suffix := "Call"
	if option_type == "put" {
		title_suffix = "Put"
	}
	config.title = fmt.tprintf(
		"European %s Option (K=%.0f) - Payoff Profile",
		title_suffix,
		strike,
	)
	config.x_label = "Terminal Stock Price"
	config.y_label = "Option Payoff ($)"
	config.show_grid = true

	return p.multi_line_png(lines, output_path, config, allocator)
}
MarketData :: struct {
	dates:   []w.Date,
	prices:  []f64,
	volumes: []f64,
	returns: []f64,
	n_obs:   int,
}

OptionChain :: struct {
	strikes:       []f64,
	expiries:      []f64, // Time to expiry in years
	implied_vols:  []f64,
	market_prices: []f64,
	option_types:  []string, // "call" or "put"
	n_options:     int,
}


// ============================================================================
// SDF Calibration to Market Prices
// ============================================================================
// ============================================================================
// SDF Calibration to Market Prices
// ============================================================================

CalibrationResult :: struct {
	sdf:          StochasticDiscountFactor,
	rmse:         f64,
	max_error:    f64,
	n_iterations: int,
	converged:    bool,
}

// Objective function: sum of squared pricing errors
// Objective function: sum of squared pricing errors
// Enforces E[M] = 1/(1+rf) by expressing 'a' as a function of 'b'
_calibration_loss :: proc(
	b: f64,
	mean_R: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	// pricing scratch (payoffs, state prices) lives only for this evaluation
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if horizon.returns_period > 0.0 {
		Ctx :: struct {
			b, spot:   f64,
			allocator: mem.Allocator,
		}
		ctx := Ctx{b, spot_price, allocator}
		return _horizon_mse(
			market_returns,
			option_chain,
			horizon,
			proc(rets: []f64, disc: f64, sub: ^OptionChain, ud: rawptr) -> f64 {
				c := (^Ctx)(ud)
				mean_R, _ := _mean_gross(rets)
				return _calibration_loss(c.b, mean_R, disc, rets, sub, c.spot, c.allocator)
			},
			&ctx,
		)
	}
	n := len(market_returns)

	// Enforce the constraint: a = 1/(1+rf) - b * E[R]
	a := rf_discount - b * mean_R

	gross_returns := make([]f64, n, allocator)
	defer delete(gross_returns, allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	sdf_values := make([]f64, n, allocator)
	for i in 0 ..< n {
		sdf_values[i] = a + b * gross_returns[i]
	}

	mean := 0.0
	for v in sdf_values {mean += v}
	mean /= f64(n)

	sdf := StochasticDiscountFactor {
		model_type = .Linear_Factor,
		values     = sdf_values,
		mean       = mean,
	}

	// Use historical returns as terminal states
	terminal_prices := make([]f64, n, allocator)
	defer delete(terminal_prices, allocator)
	for i in 0 ..< n {
		terminal_prices[i] = spot_price * gross_returns[i]
	}

	total_error := 0.0
	for i in 0 ..< option_chain.n_options {
		theo_price: f64

		if option_chain.option_types[i] == "call" {
			result := price_european_call(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		} else {
			result := price_european_put(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		}

		error := theo_price - option_chain.market_prices[i]
		total_error += error * error
	}

	delete(sdf_values, allocator)

	return total_error / f64(option_chain.n_options)
}

// Calibrate Linear SDF using SGD optimizer (1D optimization for 'b')
// Updated calibration function using grid search
calibrate_sdf_linear :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	max_iterations: int = 100,
	tolerance: f64 = 1e-6,
	allocator: mem.Allocator = context.allocator,
	returns_period: f64 = 0.0, // years per return; > 0 prices each option at its own expiry (see PDPM_Horizon)
) -> CalibrationResult {
	n := len(market_returns)
	gross_returns := make([]f64, n, context.temp_allocator)
	defer delete(gross_returns, context.temp_allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	mean_R := 0.0
	for r in gross_returns {mean_R += r}
	mean_R /= f64(n)

	rf_discount := _period_discount(risk_free_rate, returns_period)
	horizon := PDPM_Horizon{returns_period, risk_free_rate}

	// Use grid search instead of gradient descent
	b_lo, b_hi := -10.0, 0.0
	fmt.println("Running PDPM grid search...")
	best_b, best_rmse := pdpm_grid_search(
		mean_R,
		rf_discount,
		market_returns,
		option_chain,
		spot_price,
		b_lo,
		b_hi,
		100,
		allocator,
		horizon,
	)

	best_a := rf_discount - best_b * mean_R

	fmt.printf(
		"Grid search complete: RMSE = %.6f, a = %.4f, b = %.4f\n",
		best_rmse,
		best_a,
		best_b,
	)

	// Construct final SDF
	final_sdf := _construct_sdf_from_params(best_a, best_b, rf_discount, market_returns, allocator)

	return CalibrationResult {
		sdf          = final_sdf,
		rmse         = best_rmse,
		max_error    = 0.0,
		n_iterations = 200, // 100 coarse + 100 fine
		converged    = best_rmse < tolerance || _grid_interior(best_b, b_lo, b_hi),
	}
}


_construct_sdf_from_params :: proc(
	a: f64,
	b: f64,
	rf_discount: f64,
	market_returns: []f64,
	allocator: mem.Allocator,
) -> StochasticDiscountFactor {
	n := len(market_returns)
	gross_returns := make([]f64, n, context.temp_allocator)
	defer delete(gross_returns, context.temp_allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	values := make([]f64, n, allocator)
	for i in 0 ..< n {
		values[i] = a + b * gross_returns[i]
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

	parameters := make([]f64, 2, allocator)
	parameters[0] = a
	parameters[1] = b

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

compare_to_black_scholes :: proc(
	spot_price: f64,
	strike: f64,
	time_to_expiry: f64,
	risk_free_rate: f64,
	volatility: f64,
	sdf: ^StochasticDiscountFactor,
	terminal_prices: []f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	f64,
	f64,
) {
	pdpm_price: f64
	result := price_european_call(
		spot_price,
		strike,
		time_to_expiry,
		sdf,
		terminal_prices,
		allocator,
	)
	pdpm_price = result.price
	delete(result.state_prices, allocator)

	d1 :=
		(math.ln(spot_price / strike) +
			(risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry) /
		(volatility * math.sqrt(time_to_expiry))
	d2 := d1 - volatility * math.sqrt(time_to_expiry)

	nd1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	nd2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))

	bs_price := spot_price * nd1 - strike * math.exp(-risk_free_rate * time_to_expiry) * nd2

	return pdpm_price, bs_price
}


// ============================================================================
// PDPM Grid Search Calibration
// ============================================================================

// Fine-grid point i of n: centred on the coarse best c, covering ±1 coarse step,
// clipped to the coarse search range [lo, hi]
_fine_grid_point :: proc(c, coarse_step, lo, hi: f64, i, n: int) -> f64 {
	a, b := max(c - coarse_step, lo), min(c + coarse_step, hi)
	return a + (b - a) * f64(i) / f64(n - 1)
}

// A grid-search optimum is only trustworthy away from the edge of the search
// range; on the edge the true optimum is likely outside it
_grid_interior :: proc(x, lo, hi: f64) -> bool {
	tol := 1e-9 * (hi - lo)
	return x > lo + tol && x < hi - tol
}

// Horizon-consistent pricing (opt-in). market_returns are one observation per
// `returns_period` years (e.g. 1/252 for daily data); each option is then priced
// on the states mapped to its own expiry T and discounted with e^(-r T), with
// risk_free_rate read as an annual, continuously compounded rate.
// returns_period = 0 keeps the legacy one-period pricing (states = 1 + r_i,
// discount 1/(1+rf), option expiry unused).
PDPM_Horizon :: struct {
	returns_period: f64,
	risk_free_rate: f64,
}

// Per-state-period discount used for the SDF a calibrator returns
_period_discount :: proc(risk_free_rate, returns_period: f64) -> f64 {
	if returns_period > 0.0 {return math.exp(-risk_free_rate * returns_period)}
	return 1.0 / (1.0 + risk_free_rate)
}

// Map one-period net returns to horizon T by square-root-of-time scaling of the
// log returns around their mean: ln R_T = h m + (ln R - m) sqrt(h), h = T / period.
// Preserves the empirical shape of the shocks; the drift scales with h.
pdpm_horizon_returns :: proc(
	returns: []f64,
	returns_period, T: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	out := make([]f64, n, allocator)
	m := 0.0
	for r in returns {m += math.ln(1.0 + r)}
	m /= f64(n)
	h := T / returns_period
	sh := math.sqrt(h)
	for r, i in returns {
		out[i] = math.exp(h * m + (math.ln(1.0 + r) - m) * sh) - 1.0
	}
	return out
}

// Option-count-weighted MSE over the distinct expiries of the chain: `eval` is a
// one-period loss called with the horizon states and e^(-r T) for each expiry group
_Horizon_Eval :: proc(
	horizon_returns: []f64,
	discount: f64,
	sub: ^OptionChain,
	user_data: rawptr,
) -> f64

_horizon_mse :: proc(
	market_returns: []f64,
	option_chain: ^OptionChain,
	horizon: PDPM_Horizon,
	eval: _Horizon_Eval,
	user_data: rawptr,
) -> f64 {
	n_opt := option_chain.n_options
	done := make([]bool, n_opt, context.temp_allocator)
	idx := make([dynamic]int, 0, n_opt, context.temp_allocator)
	total := 0.0
	for k in 0 ..< n_opt {
		if done[k] {continue}
		T := option_chain.expiries[k]
		clear(&idx)
		for j in k ..< n_opt {
			if !done[j] && option_chain.expiries[j] == T {
				append(&idx, j)
				done[j] = true
			}
		}
		m := len(idx)
		sub := OptionChain {
			strikes       = make([]f64, m, context.temp_allocator),
			expiries      = make([]f64, m, context.temp_allocator),
			implied_vols  = make([]f64, m, context.temp_allocator),
			market_prices = make([]f64, m, context.temp_allocator),
			option_types  = make([]string, m, context.temp_allocator),
			n_options     = m,
		}
		for j, q in idx {
			sub.strikes[q] = option_chain.strikes[j]
			sub.expiries[q] = T
			sub.implied_vols[q] =
				option_chain.implied_vols[j] if len(option_chain.implied_vols) > j else 0.0
			sub.market_prices[q] = option_chain.market_prices[j]
			sub.option_types[q] = option_chain.option_types[j]
		}
		rets := pdpm_horizon_returns(
			market_returns,
			horizon.returns_period,
			T,
			context.temp_allocator,
		)
		total += f64(m) * eval(rets, math.exp(-horizon.risk_free_rate * T), &sub, user_data)
	}
	return total / f64(n_opt)
}

_mean_gross :: proc(returns: []f64) -> (mean_R, mean_R2: f64) {
	for r in returns {
		R := 1.0 + r
		mean_R += R
		mean_R2 += R * R
	}
	return mean_R / f64(len(returns)), mean_R2 / f64(len(returns))
}

PDPMCalibrationParams :: struct {
	b: f64, // The only parameter to optimize
}

// PDPM evaluator - computes negative RMSE (we want to maximize this)
_pdpm_grid_evaluator :: proc(
	b: f64,
	mean_R: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	loss := _calibration_loss(
		b,
		mean_R,
		rf_discount,
		market_returns,
		option_chain,
		spot_price,
		allocator,
		horizon,
	)
	return -math.sqrt_f64(loss) // Return negative RMSE (higher is better)
}

// 1D Grid Search for PDPM calibration
pdpm_grid_search :: proc(
	mean_R: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	b_min: f64 = -10.0,
	b_max: f64 = 0.0,
	n_grid: int = 100,
	allocator: mem.Allocator = context.allocator,
	horizon := PDPM_Horizon{},
) -> (
	best_b: f64,
	best_rmse: f64,
) {
	best_score := -math.F64_MAX
	best_b = b_min
	b_min := b_min
	b_max := b_max
	// Coarse grid search
	b_step := (b_max - b_min) / f64(n_grid - 1)

	for i in 0 ..< n_grid {
		b_test := b_min + f64(i) * b_step
		score := _pdpm_grid_evaluator(
			b_test,
			mean_R,
			rf_discount,
			market_returns,
			option_chain,
			spot_price,
			allocator,
			horizon,
		)

		if score > best_score {
			best_score = score
			best_b = b_test
		}
	}

	// Fine grid search around best_b
	b_center := best_b

	for i in 0 ..< n_grid {
		b_test := _fine_grid_point(b_center, b_step, b_min, b_max, i, n_grid)
		score := _pdpm_grid_evaluator(
			b_test,
			mean_R,
			rf_discount,
			market_returns,
			option_chain,
			spot_price,
			allocator,
			horizon,
		)

		if score > best_score {
			best_score = score
			best_b = b_test
		}
	}

	best_rmse = -best_score
	return best_b, best_rmse
}
// ============================================================================
// Non-Linear SDF Calibration
// ============================================================================

// Power Utility SDF: M = β * (C/C0)^(-γ)
// Here we use R (gross return) as proxy for consumption growth.
// β is set so that E[M] = 1/(1+rf), like every other SDF here
_power_utility_beta :: proc(gamma, rf_discount: f64, market_returns: []f64) -> f64 {
	mean_pow := 0.0
	for r in market_returns {mean_pow += math.pow(1.0 + r, -gamma)}
	mean_pow /= f64(len(market_returns))
	return rf_discount / mean_pow
}

_calibration_loss_power_utility :: proc(
	gamma: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	// pricing scratch (payoffs, state prices) lives only for this evaluation
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if horizon.returns_period > 0.0 {
		Ctx :: struct {
			gamma, spot: f64,
			allocator:   mem.Allocator,
		}
		ctx := Ctx{gamma, spot_price, allocator}
		return _horizon_mse(
			market_returns,
			option_chain,
			horizon,
			proc(rets: []f64, disc: f64, sub: ^OptionChain, ud: rawptr) -> f64 {
				c := (^Ctx)(ud)
				return _calibration_loss_power_utility(
					c.gamma,
					disc,
					rets,
					sub,
					c.spot,
					c.allocator,
				)
			},
			&ctx,
		)
	}
	n := len(market_returns)

	gross_returns := make([]f64, n, allocator)
	defer delete(gross_returns, allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	beta := _power_utility_beta(gamma, rf_discount, market_returns)
	sdf_values := make([]f64, n, allocator)
	for i in 0 ..< n {
		sdf_values[i] = beta * math.pow(gross_returns[i], -gamma)
	}

	mean := 0.0
	for v in sdf_values {mean += v}
	mean /= f64(n)

	sdf := StochasticDiscountFactor {
		model_type = .Power_Utility,
		values     = sdf_values,
		mean       = mean,
	}

	// Use historical returns as terminal states
	terminal_prices := make([]f64, n, allocator)
	defer delete(terminal_prices, allocator)
	for i in 0 ..< n {
		terminal_prices[i] = spot_price * gross_returns[i]
	}

	total_error := 0.0
	for i in 0 ..< option_chain.n_options {
		theo_price: f64

		if option_chain.option_types[i] == "call" {
			result := price_european_call(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		} else {
			result := price_european_put(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		}

		error := theo_price - option_chain.market_prices[i]
		total_error += error * error
	}

	delete(sdf_values, allocator)

	return total_error / f64(option_chain.n_options)
}

// Grid search for power utility parameter γ
pdpm_grid_search_power_utility :: proc(
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	gamma_min: f64 = 0.5,
	gamma_max: f64 = 10.0,
	n_grid: int = 100,
	allocator: mem.Allocator = context.allocator,
	horizon := PDPM_Horizon{},
) -> (
	best_gamma: f64,
	best_rmse: f64,
) {
	best_loss := math.F64_MAX
	best_gamma = gamma_min
	gamma_min := gamma_min
	gamma_max := gamma_max
	// Coarse grid search
	gamma_step := (gamma_max - gamma_min) / f64(n_grid - 1)

	for i in 0 ..< n_grid {
		gamma_test := gamma_min + f64(i) * gamma_step
		loss := _calibration_loss_power_utility(
			gamma_test,
			rf_discount,
			market_returns,
			option_chain,
			spot_price,
			allocator,
			horizon,
		)

		if loss < best_loss {
			best_loss = loss
			best_gamma = gamma_test
		}
	}

	// Fine grid search around best_gamma
	gamma_center := best_gamma

	for i in 0 ..< n_grid {
		gamma_test := _fine_grid_point(gamma_center, gamma_step, gamma_min, gamma_max, i, n_grid)
		if gamma_test > 0.0 { 	// γ must be positive
			loss := _calibration_loss_power_utility(
				gamma_test,
				rf_discount,
				market_returns,
				option_chain,
				spot_price,
				allocator,
				horizon,
			)

			if loss < best_loss {
				best_loss = loss
				best_gamma = gamma_test
			}
		}
	}

	best_rmse = math.sqrt_f64(best_loss)
	return best_gamma, best_rmse
}

// Calibrate Power Utility SDF
calibrate_sdf_power_utility :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator = context.allocator,
	returns_period: f64 = 0.0, // years per return; > 0 prices each option at its own expiry (see PDPM_Horizon)
) -> CalibrationResult {
	rf_discount := _period_discount(risk_free_rate, returns_period)
	horizon := PDPM_Horizon{returns_period, risk_free_rate}
	gamma_lo, gamma_hi := 0.5, 10.0

	fmt.println("Running Power Utility SDF grid search...")
	best_gamma, best_rmse := pdpm_grid_search_power_utility(
		rf_discount,
		market_returns,
		option_chain,
		spot_price,
		gamma_lo,
		gamma_hi,
		100,
		allocator,
		horizon,
	)
	beta := _power_utility_beta(best_gamma, rf_discount, market_returns)

	fmt.printf(
		"Power Utility calibration complete: γ = %.4f, RMSE = %.6f\n",
		best_gamma,
		best_rmse,
	)

	// Construct final SDF
	n := len(market_returns)
	gross_returns := make([]f64, n, context.temp_allocator)
	defer delete(gross_returns, context.temp_allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	values := make([]f64, n, allocator)
	for i in 0 ..< n {
		values[i] = beta * math.pow(gross_returns[i], -best_gamma)
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

	parameters := make([]f64, 2, allocator)
	parameters[0] = best_gamma
	parameters[1] = beta

	final_sdf := StochasticDiscountFactor {
		model_type      = .Power_Utility,
		values          = values,
		parameters      = parameters,
		risk_aversion   = best_gamma,
		discount_factor = beta,
		mean            = mean,
		variance        = variance,
	}

	return CalibrationResult {
		sdf = final_sdf,
		rmse = best_rmse,
		max_error = 0.0,
		n_iterations = 200,
		converged = _grid_interior(best_gamma, gamma_lo, gamma_hi),
	}
}

// Quadratic SDF: M = a + b*R + c*R²
// Constraint: E[M] = 1/(1+rf) => a = rf_discount - b*E[R] - c*E[R²]
_calibration_loss_quadratic :: proc(
	b: f64,
	c: f64,
	mean_R: f64,
	mean_R2: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	// pricing scratch (payoffs, state prices) lives only for this evaluation
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if horizon.returns_period > 0.0 {
		Ctx :: struct {
			b, c, spot: f64,
			allocator:  mem.Allocator,
		}
		ctx := Ctx{b, c, spot_price, allocator}
		return _horizon_mse(
			market_returns,
			option_chain,
			horizon,
			proc(rets: []f64, disc: f64, sub: ^OptionChain, ud: rawptr) -> f64 {
				q := (^Ctx)(ud)
				mean_R, mean_R2 := _mean_gross(rets)
				return _calibration_loss_quadratic(
					q.b,
					q.c,
					mean_R,
					mean_R2,
					disc,
					rets,
					sub,
					q.spot,
					q.allocator,
				)
			},
			&ctx,
		)
	}
	n := len(market_returns)

	// Enforce constraint: a = rf_discount - b*E[R] - c*E[R²]
	a := rf_discount - b * mean_R - c * mean_R2

	gross_returns := make([]f64, n, allocator)
	defer delete(gross_returns, allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	sdf_values := make([]f64, n, allocator)
	for i in 0 ..< n {
		R := gross_returns[i]
		sdf_values[i] = a + b * R + c * R * R
	}

	mean := 0.0
	for v in sdf_values {mean += v}
	mean /= f64(n)

	sdf := StochasticDiscountFactor {
		model_type = .Quadratic,
		values     = sdf_values,
		mean       = mean,
	}

	// Use historical returns as terminal states
	terminal_prices := make([]f64, n, allocator)
	defer delete(terminal_prices, allocator)
	for i in 0 ..< n {
		terminal_prices[i] = spot_price * gross_returns[i]
	}

	total_error := 0.0
	for i in 0 ..< option_chain.n_options {
		theo_price: f64

		if option_chain.option_types[i] == "call" {
			result := price_european_call(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		} else {
			result := price_european_put(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		}

		error := theo_price - option_chain.market_prices[i]
		total_error += error * error
	}

	delete(sdf_values, allocator)

	return total_error / f64(option_chain.n_options)
}

// 2D Grid search for quadratic SDF parameters (b, c)
pdpm_grid_search_quadratic :: proc(
	mean_R: f64,
	mean_R2: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	b_range: f64 = 20.0,
	c_range: f64 = 10.0,
	n_grid: int = 20,
	allocator: mem.Allocator = context.allocator,
	horizon := PDPM_Horizon{},
) -> (
	best_b: f64,
	best_c: f64,
	best_rmse: f64,
) {
	best_loss := math.F64_MAX
	best_b = 0.0
	best_c = 0.0

	// Coarse 2D grid search
	b_step := (2.0 * b_range) / f64(n_grid - 1)
	c_step := (2.0 * c_range) / f64(n_grid - 1)

	for i in 0 ..< n_grid {
		b_test := -b_range + f64(i) * b_step
		for j in 0 ..< n_grid {
			c_test := -c_range + f64(j) * c_step
			loss := _calibration_loss_quadratic(
				b_test,
				c_test,
				mean_R,
				mean_R2,
				rf_discount,
				market_returns,
				option_chain,
				spot_price,
				allocator,
				horizon,
			)

			if loss < best_loss {
				best_loss = loss
				best_b = b_test
				best_c = c_test
			}
		}
	}

	// Fine 2D grid search around best (b, c)
	b_center, c_center := best_b, best_c

	for i in 0 ..< n_grid {
		b_test := _fine_grid_point(b_center, b_step, -b_range, b_range, i, n_grid)
		for j in 0 ..< n_grid {
			c_test := _fine_grid_point(c_center, c_step, -c_range, c_range, j, n_grid)
			loss := _calibration_loss_quadratic(
				b_test,
				c_test,
				mean_R,
				mean_R2,
				rf_discount,
				market_returns,
				option_chain,
				spot_price,
				allocator,
				horizon,
			)

			if loss < best_loss {
				best_loss = loss
				best_b = b_test
				best_c = c_test
			}
		}
	}

	best_rmse = math.sqrt_f64(best_loss)
	return best_b, best_c, best_rmse
}

// Calibrate Quadratic SDF
calibrate_sdf_quadratic :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator = context.allocator,
	returns_period: f64 = 0.0, // years per return; > 0 prices each option at its own expiry (see PDPM_Horizon)
) -> CalibrationResult {
	n := len(market_returns)
	gross_returns := make([]f64, n, context.temp_allocator)
	defer delete(gross_returns, context.temp_allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	mean_R := 0.0
	mean_R2 := 0.0
	for r in gross_returns {
		mean_R += r
		mean_R2 += r * r
	}
	mean_R /= f64(n)
	mean_R2 /= f64(n)

	rf_discount := _period_discount(risk_free_rate, returns_period)
	horizon := PDPM_Horizon{returns_period, risk_free_rate}
	b_range, c_range := 20.0, 10.0

	fmt.println("Running Quadratic SDF 2D grid search...")
	best_b, best_c, best_rmse := pdpm_grid_search_quadratic(
		mean_R,
		mean_R2,
		rf_discount,
		market_returns,
		option_chain,
		spot_price,
		b_range,
		c_range,
		20,
		allocator,
		horizon,
	)

	best_a := rf_discount - best_b * mean_R - best_c * mean_R2

	fmt.printf(
		"Quadratic SDF calibration complete: a = %.4f, b = %.4f, c = %.4f, RMSE = %.6f\n",
		best_a,
		best_b,
		best_c,
		best_rmse,
	)

	// Construct final SDF
	values := make([]f64, n, allocator)
	for i in 0 ..< n {
		R := gross_returns[i]
		values[i] = best_a + best_b * R + best_c * R * R
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

	parameters := make([]f64, 3, allocator)
	parameters[0] = best_a
	parameters[1] = best_b
	parameters[2] = best_c

	final_sdf := StochasticDiscountFactor {
		model_type      = .Quadratic,
		values          = values,
		parameters      = parameters,
		risk_aversion   = 0.0,
		discount_factor = rf_discount,
		mean            = mean,
		variance        = variance,
	}

	return CalibrationResult {
		sdf          = final_sdf,
		rmse         = best_rmse,
		max_error    = 0.0,
		n_iterations = 800, // 20*20 coarse + 20*20 fine
		converged    = _grid_interior(
			best_b,
			-b_range,
			b_range,
		) && _grid_interior(best_c, -c_range, c_range),
	}
}


// ============================================================================
// Exponential SDF Calibration (CARA Utility)
// ============================================================================

// Exponential SDF: M = exp(a - γ*R)
// Constraint E[M] = 1/(1+rf) is enforced by setting exp(a) = rf_discount / E[exp(-γ*R)]
_calibration_loss_exponential :: proc(
	gamma: f64,
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	// pricing scratch (payoffs, state prices) lives only for this evaluation
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if horizon.returns_period > 0.0 {
		Ctx :: struct {
			gamma, spot: f64,
			allocator:   mem.Allocator,
		}
		ctx := Ctx{gamma, spot_price, allocator}
		return _horizon_mse(
			market_returns,
			option_chain,
			horizon,
			proc(rets: []f64, disc: f64, sub: ^OptionChain, ud: rawptr) -> f64 {
				c := (^Ctx)(ud)
				return _calibration_loss_exponential(c.gamma, disc, rets, sub, c.spot, c.allocator)
			},
			&ctx,
		)
	}
	n := len(market_returns)

	gross_returns := make([]f64, n, allocator)
	defer delete(gross_returns, allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	// Compute mean of exp(-γ*R) to enforce constraint
	sum_exp := 0.0
	for i in 0 ..< n {
		// Clamp to prevent overflow
		val := -gamma * gross_returns[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n)

	// Enforce constraint: E[M] = rf_discount
	// M_i = (rf_discount / mean_exp) * exp(-γ * R_i)
	scale := rf_discount / mean_exp

	sdf_values := make([]f64, n, allocator)
	for i in 0 ..< n {
		val := -gamma * gross_returns[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sdf_values[i] = scale * math.exp(val)
	}

	mean := 0.0
	for v in sdf_values {mean += v}
	mean /= f64(n)

	sdf := StochasticDiscountFactor {
		model_type = .Exponential_Utility,
		values     = sdf_values,
		mean       = mean,
	}

	terminal_prices := make([]f64, n, allocator)
	defer delete(terminal_prices, allocator)
	for i in 0 ..< n {
		terminal_prices[i] = spot_price * gross_returns[i]
	}

	total_error := 0.0
	for i in 0 ..< option_chain.n_options {
		theo_price: f64

		if option_chain.option_types[i] == "call" {
			result := price_european_call(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		} else {
			result := price_european_put(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		}

		error := theo_price - option_chain.market_prices[i]
		total_error += error * error
	}

	delete(sdf_values, allocator)

	return total_error / f64(option_chain.n_options)
}

// 1D Grid search for Exponential SDF parameter γ
pdpm_grid_search_exponential :: proc(
	rf_discount: f64,
	market_returns: []f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	gamma_min: f64 = -5.0,
	gamma_max: f64 = 5.0,
	n_grid: int = 100,
	allocator: mem.Allocator = context.allocator,
	horizon := PDPM_Horizon{},
) -> (
	best_gamma: f64,
	best_rmse: f64,
) {
	best_loss := math.F64_MAX
	best_gamma = gamma_min
	gamma_min := gamma_min
	gamma_max := gamma_max


	// Coarse grid search
	gamma_step := (gamma_max - gamma_min) / f64(n_grid - 1)

	for i in 0 ..< n_grid {
		gamma_test := gamma_min + f64(i) * gamma_step
		loss := _calibration_loss_exponential(
			gamma_test,
			rf_discount,
			market_returns,
			option_chain,
			spot_price,
			allocator,
			horizon,
		)

		if loss < best_loss {
			best_loss = loss
			best_gamma = gamma_test
		}
	}

	// Fine grid search around best_gamma
	gamma_center := best_gamma

	for i in 0 ..< n_grid {
		gamma_test := _fine_grid_point(gamma_center, gamma_step, gamma_min, gamma_max, i, n_grid)
		loss := _calibration_loss_exponential(
			gamma_test,
			rf_discount,
			market_returns,
			option_chain,
			spot_price,
			allocator,
			horizon,
		)

		if loss < best_loss {
			best_loss = loss
			best_gamma = gamma_test
		}
	}

	best_rmse = math.sqrt_f64(best_loss)
	return best_gamma, best_rmse
}

// Calibrate Exponential SDF
calibrate_sdf_exponential :: proc(
	market_returns: []f64,
	risk_free_rate: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator = context.allocator,
	returns_period: f64 = 0.0, // years per return; > 0 prices each option at its own expiry (see PDPM_Horizon)
) -> CalibrationResult {
	rf_discount := _period_discount(risk_free_rate, returns_period)
	horizon := PDPM_Horizon{returns_period, risk_free_rate}
	n := len(market_returns)

	gamma_lo, gamma_hi := -5.0, 5.0
	fmt.println("Running Exponential SDF grid search...")
	best_gamma, best_rmse := pdpm_grid_search_exponential(
		rf_discount,
		market_returns,
		option_chain,
		spot_price,
		gamma_lo,
		gamma_hi,
		100,
		allocator,
		horizon,
	)

	fmt.printf(
		"Exponential SDF calibration complete: γ = %.4f, RMSE = %.6f\n",
		best_gamma,
		best_rmse,
	)

	// Construct final SDF
	gross_returns := make([]f64, n, context.temp_allocator)
	defer delete(gross_returns, context.temp_allocator)
	for i in 0 ..< n {
		gross_returns[i] = 1.0 + market_returns[i]
	}

	sum_exp := 0.0
	for i in 0 ..< n {sum_exp += math.exp(-best_gamma * gross_returns[i])}
	mean_exp := sum_exp / f64(n)
	scale := rf_discount / mean_exp

	values := make([]f64, n, allocator)
	for i in 0 ..< n {
		values[i] = scale * math.exp(-best_gamma * gross_returns[i])
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

	parameters := make([]f64, 1, allocator)
	parameters[0] = best_gamma

	final_sdf := StochasticDiscountFactor {
		model_type      = .Exponential_Utility,
		values          = values,
		parameters      = parameters,
		risk_aversion   = best_gamma,
		discount_factor = rf_discount,
		mean            = mean,
		variance        = variance,
	}

	return CalibrationResult {
		sdf = final_sdf,
		rmse = best_rmse,
		max_error = 0.0,
		n_iterations = 200,
		converged = _grid_interior(best_gamma, gamma_lo, gamma_hi),
	}
}
// ============================================================================
// Multi-Factor SDF Calibration
// ============================================================================

MultiFactorSDF :: struct {
	factors:   [][]f64, // Each row is a factor's returns
	weights:   []f64, // Factor loadings (betas)
	intercept: f64, // Alpha
	values:    []f64, // SDF values for each state
	mean:      f64,
	variance:  f64,
	n_factors: int,
	n_states:  int,
}

// Multi-factor SDF: M = a + b1*F1 + b2*F2 + ... + bn*Fn
// Constraint: E[M] = 1/(1+rf)
// ============================================================================
// Multi-Factor SDF Calibration (Exponential Form)
// ============================================================================

// Exponential Multi-Factor SDF: M = exp(a + b1*F1 + b2*F2 + ...)
// This guarantees M > 0 (no-arbitrage)
// Constraint: E[M] = 1/(1+rf) enforced by adjusting 'a'
_calibration_loss_multifactor :: proc(
	weights: []f64,
	factors: [][]f64,
	rf_discount: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	// pricing scratch (payoffs, state prices) lives only for this evaluation
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if horizon.returns_period > 0.0 {
		// only the market factor (factors[0], the terminal-price driver) is mapped to the horizon
		Ctx :: struct {
			weights:   []f64,
			factors:   [][]f64,
			spot:      f64,
			allocator: mem.Allocator,
		}
		ctx := Ctx{weights, factors, spot_price, allocator}
		return _horizon_mse(
			factors[0],
			option_chain,
			horizon,
			proc(rets: []f64, disc: f64, sub: ^OptionChain, ud: rawptr) -> f64 {
				c := (^Ctx)(ud)
				fT := make([][]f64, len(c.factors), context.temp_allocator)
				copy(fT, c.factors)
				fT[0] = rets
				return _calibration_loss_multifactor(c.weights, fT, disc, sub, c.spot, c.allocator)
			},
			&ctx,
		)
	}
	n := len(factors[0])
	n_factors := len(factors)

	// Compute sum of weighted factors for each state
	weighted_factors := make([]f64, n, allocator)
	defer delete(weighted_factors, allocator)

	for i in 0 ..< n {
		weighted_factors[i] = 0.0
		for j in 0 ..< n_factors {
			weighted_factors[i] += weights[j] * factors[j][i]
		}
	}

	// Compute mean of exp(weighted_factors) to enforce constraint
	sum_exp := 0.0
	for i in 0 ..< n {
		// Clamp to prevent overflow
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n)

	// Enforce constraint: E[M] = rf_discount
	// M_i = (rf_discount / mean_exp) * exp(weighted_factors[i])
	scale := rf_discount / mean_exp

	sdf_values := make([]f64, n, allocator)
	defer delete(sdf_values, allocator)

	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sdf_values[i] = scale * math.exp(val)
	}

	mean := 0.0
	for v in sdf_values {mean += v}
	mean /= f64(n)

	sdf := StochasticDiscountFactor {
		model_type = .Exponential_Utility,
		values     = sdf_values,
		mean       = mean,
	}

	// Use market returns as terminal states
	terminal_prices := make([]f64, n, allocator)
	defer delete(terminal_prices, allocator)
	for i in 0 ..< n {
		terminal_prices[i] = spot_price * (1.0 + factors[0][i])
	}

	total_error := 0.0
	for i in 0 ..< option_chain.n_options {
		theo_price: f64

		if option_chain.option_types[i] == "call" {
			result := price_european_call(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		} else {
			result := price_european_put(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		}

		error := theo_price - option_chain.market_prices[i]
		total_error += error * error
	}

	return total_error / f64(option_chain.n_options)
}

// 2D Grid search for 2-factor exponential model
pdpm_grid_search_multifactor_2d :: proc(
	factors: [][]f64,
	rf_discount: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	range1: f64 = 5.0,
	range2: f64 = 5.0,
	n_grid: int = 20,
	allocator: mem.Allocator = context.allocator,
	horizon := PDPM_Horizon{},
) -> (
	best_weights: []f64,
	best_rmse: f64,
) {
	best_loss := math.F64_MAX
	best_weights = make([]f64, 2, allocator)
	range1 := range1
	range2 := range2
	// Coarse 2D grid search
	step1 := (2.0 * range1) / f64(n_grid - 1)
	step2 := (2.0 * range2) / f64(n_grid - 1)

	test_weights := make([]f64, 2, allocator)
	defer delete(test_weights, allocator)

	for i in 0 ..< n_grid {
		test_weights[0] = -range1 + f64(i) * step1
		for j in 0 ..< n_grid {
			test_weights[1] = -range2 + f64(j) * step2
			loss := _calibration_loss_multifactor(
				test_weights,
				factors,
				rf_discount,
				option_chain,
				spot_price,
				allocator,
				horizon,
			)

			if loss < best_loss {
				best_loss = loss
				best_weights[0] = test_weights[0]
				best_weights[1] = test_weights[1]
			}
		}
	}

	// Fine 2D grid search: ±1 coarse step around the (frozen) coarse best
	center1, center2 := best_weights[0], best_weights[1]

	for i in 0 ..< n_grid {
		test_weights[0] = _fine_grid_point(center1, step1, -range1, range1, i, n_grid)
		for j in 0 ..< n_grid {
			test_weights[1] = _fine_grid_point(center2, step2, -range2, range2, j, n_grid)
			loss := _calibration_loss_multifactor(
				test_weights,
				factors,
				rf_discount,
				option_chain,
				spot_price,
				allocator,
				horizon,
			)

			if loss < best_loss {
				best_loss = loss
				best_weights[0] = test_weights[0]
				best_weights[1] = test_weights[1]
			}
		}
	}

	best_rmse = math.sqrt_f64(best_loss)
	return best_weights, best_rmse
}

// Calibrate 2-Factor Exponential SDF
calibrate_sdf_multifactor_2d :: proc(
	market_returns: []f64,
	volatility_proxy: []f64,
	risk_free_rate: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator = context.allocator,
	returns_period: f64 = 0.0, // years per return; > 0 prices each option at its own expiry (see PDPM_Horizon)
) -> CalibrationResult {
	n := len(market_returns)
	rf_discount := _period_discount(risk_free_rate, returns_period)
	horizon := PDPM_Horizon{returns_period, risk_free_rate}

	// Prepare factors
	factors := make([][]f64, 2, allocator)
	factors[0] = market_returns
	factors[1] = volatility_proxy

	range1, range2 := 5.0, 5.0
	fmt.println("Running 2-Factor Exponential SDF grid search...")
	best_weights, best_rmse := pdpm_grid_search_multifactor_2d(
		factors,
		rf_discount,
		option_chain,
		spot_price,
		range1,
		range2,
		20,
		allocator,
		horizon,
	)

	fmt.printf("2-Factor Exponential SDF calibration complete:\n")
	fmt.printf("  Market beta (b1): %.4f\n", best_weights[0])
	fmt.printf("  Vol beta (b2): %.4f\n", best_weights[1])
	fmt.printf("  RMSE: $%.6f\n", best_rmse)

	// Construct final SDF
	weighted_factors := make([]f64, n, context.temp_allocator)
	defer delete(weighted_factors, context.temp_allocator)

	for i in 0 ..< n {
		weighted_factors[i] =
			best_weights[0] * market_returns[i] + best_weights[1] * volatility_proxy[i]
	}

	sum_exp := 0.0
	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n)
	scale := rf_discount / mean_exp

	values := make([]f64, n, allocator)
	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		values[i] = scale * math.exp(val)
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

	parameters := make([]f64, 2, allocator)
	parameters[0] = best_weights[0]
	parameters[1] = best_weights[1]

	final_sdf := StochasticDiscountFactor {
		model_type      = .Exponential_Utility,
		values          = values,
		parameters      = parameters,
		risk_aversion   = 0.0,
		discount_factor = rf_discount,
		mean            = mean,
		variance        = variance,
	}

	// Clean up
	delete(factors, allocator)

	return CalibrationResult {
		sdf          = final_sdf,
		rmse         = best_rmse,
		max_error    = 0.0,
		n_iterations = 800, // 20*20 coarse + 20*20 fine
		converged    = _grid_interior(
			best_weights[0],
			-range1,
			range1,
		) && _grid_interior(best_weights[1], -range2, range2),
	}
}
// ============================================================================
// 4-Factor SDF Calibration (Market, Volatility, Momentum, Macro)
// ============================================================================

// Calculate momentum factor (12-month rolling return)
calculate_momentum :: proc(
	returns: []f64,
	window: int = 252, // 12 months of trading days
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	momentum := make([]f64, n, allocator)

	for i in 0 ..< n {
		if i < window {
			momentum[i] = 0.0
		} else {
			// Calculate cumulative return over window
			cum_return := 1.0
			for j in (i - window) ..< i {
				cum_return *= (1.0 + returns[j])
			}
			momentum[i] = cum_return - 1.0
		}
	}

	return momentum
}

// Calculate macro factor (6-month rolling average of returns)
calculate_macro_factor :: proc(
	returns: []f64,
	window: int = 126, // 6 months of trading days
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	macro := make([]f64, n, allocator)

	for i in 0 ..< n {
		if i < window {
			macro[i] = 0.0
		} else {
			// Calculate average return over window
			sum := 0.0
			for j in (i - window) ..< i {
				sum += returns[j]
			}
			macro[i] = sum / f64(window)
		}
	}

	return macro
}

// 4-Factor Exponential SDF loss function
_calibration_loss_multifactor_4d :: proc(
	weights: []f64,
	factors: [][]f64,
	rf_discount: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator,
	horizon: PDPM_Horizon = {},
) -> f64 {
	// pricing scratch (payoffs, state prices) lives only for this evaluation
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if horizon.returns_period > 0.0 {
		// only the market factor (factors[0], the terminal-price driver) is mapped to the horizon
		Ctx :: struct {
			weights:   []f64,
			factors:   [][]f64,
			spot:      f64,
			allocator: mem.Allocator,
		}
		ctx := Ctx{weights, factors, spot_price, allocator}
		return _horizon_mse(
			factors[0],
			option_chain,
			horizon,
			proc(rets: []f64, disc: f64, sub: ^OptionChain, ud: rawptr) -> f64 {
				c := (^Ctx)(ud)
				fT := make([][]f64, len(c.factors), context.temp_allocator)
				copy(fT, c.factors)
				fT[0] = rets
				return _calibration_loss_multifactor_4d(
					c.weights,
					fT,
					disc,
					sub,
					c.spot,
					c.allocator,
				)
			},
			&ctx,
		)
	}
	n := len(factors[0])
	n_factors := len(factors)

	// Compute sum of weighted factors for each state
	weighted_factors := make([]f64, n, allocator)
	defer delete(weighted_factors, allocator)

	for i in 0 ..< n {
		weighted_factors[i] = 0.0
		for j in 0 ..< n_factors {
			weighted_factors[i] += weights[j] * factors[j][i]
		}
	}

	// Compute mean of exp(weighted_factors) to enforce constraint
	sum_exp := 0.0
	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n)

	// Enforce constraint: E[M] = rf_discount
	scale := rf_discount / mean_exp

	sdf_values := make([]f64, n, allocator)
	defer delete(sdf_values, allocator)

	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sdf_values[i] = scale * math.exp(val)
	}

	mean := 0.0
	for v in sdf_values {mean += v}
	mean /= f64(n)

	sdf := StochasticDiscountFactor {
		model_type = .Exponential_Utility,
		values     = sdf_values,
		mean       = mean,
	}

	terminal_prices := make([]f64, n, allocator)
	defer delete(terminal_prices, allocator)
	for i in 0 ..< n {
		terminal_prices[i] = spot_price * (1.0 + factors[0][i])
	}

	total_error := 0.0
	for i in 0 ..< option_chain.n_options {
		theo_price: f64

		if option_chain.option_types[i] == "call" {
			result := price_european_call(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		} else {
			result := price_european_put(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				&sdf,
				terminal_prices,
				context.temp_allocator,
			)
			theo_price = result.price
			delete(result.state_prices, context.temp_allocator)
		}

		error := theo_price - option_chain.market_prices[i]
		total_error += error * error
	}

	return total_error / f64(option_chain.n_options)
}

// 4D Grid search (coarse then fine)
pdpm_grid_search_multifactor_4d :: proc(
	factors: [][]f64,
	rf_discount: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	range: f64 = 5.0,
	n_grid: int = 10,
	allocator: mem.Allocator = context.allocator,
	horizon := PDPM_Horizon{},
) -> (
	best_weights: []f64,
	best_rmse: f64,
) {
	best_loss := math.F64_MAX
	best_weights = make([]f64, 4, allocator)
	range := range

	// Coarse 4D grid search
	step := (2.0 * range) / f64(n_grid - 1)
	test_weights := make([]f64, 4, allocator)
	defer delete(test_weights, allocator)

	for i in 0 ..< n_grid {
		test_weights[0] = -range + f64(i) * step
		for j in 0 ..< n_grid {
			test_weights[1] = -range + f64(j) * step
			for k in 0 ..< n_grid {
				test_weights[2] = -range + f64(k) * step
				for l in 0 ..< n_grid {
					test_weights[3] = -range + f64(l) * step
					loss := _calibration_loss_multifactor_4d(
						test_weights,
						factors,
						rf_discount,
						option_chain,
						spot_price,
						allocator,
						horizon,
					)

					if loss < best_loss {
						best_loss = loss
						best_weights[0] = test_weights[0]
						best_weights[1] = test_weights[1]
						best_weights[2] = test_weights[2]
						best_weights[3] = test_weights[3]
					}
				}
			}
		}
	}

	// Fine grid search: ±1 coarse step around the coarse best, frozen so the
	// window does not move while best_weights improves inside the scan
	center := [4]f64{best_weights[0], best_weights[1], best_weights[2], best_weights[3]}

	for i in 0 ..< n_grid {
		test_weights[0] = _fine_grid_point(center[0], step, -range, range, i, n_grid)
		for j in 0 ..< n_grid {
			test_weights[1] = _fine_grid_point(center[1], step, -range, range, j, n_grid)
			for k in 0 ..< n_grid {
				test_weights[2] = _fine_grid_point(center[2], step, -range, range, k, n_grid)
				for l in 0 ..< n_grid {
					test_weights[3] = _fine_grid_point(center[3], step, -range, range, l, n_grid)
					loss := _calibration_loss_multifactor_4d(
						test_weights,
						factors,
						rf_discount,
						option_chain,
						spot_price,
						allocator,
						horizon,
					)

					if loss < best_loss {
						best_loss = loss
						best_weights[0] = test_weights[0]
						best_weights[1] = test_weights[1]
						best_weights[2] = test_weights[2]
						best_weights[3] = test_weights[3]
					}
				}
			}
		}
	}

	best_rmse = math.sqrt_f64(best_loss)
	return best_weights, best_rmse
}

// Calibrate 4-Factor Exponential SDF
calibrate_sdf_multifactor_4d :: proc(
	market_returns: []f64,
	volatility_proxy: []f64,
	momentum_factor: []f64,
	macro_factor: []f64,
	risk_free_rate: f64,
	option_chain: ^OptionChain,
	spot_price: f64,
	allocator: mem.Allocator = context.allocator,
	returns_period: f64 = 0.0, // years per return; > 0 prices each option at its own expiry (see PDPM_Horizon)
) -> CalibrationResult {
	n := len(market_returns)
	rf_discount := _period_discount(risk_free_rate, returns_period)
	horizon := PDPM_Horizon{returns_period, risk_free_rate}

	// Prepare 4 factors
	factors := make([][]f64, 4, allocator)
	factors[0] = market_returns
	factors[1] = volatility_proxy
	factors[2] = momentum_factor
	factors[3] = macro_factor

	fmt.println("Running 4-Factor Exponential SDF grid search...")
	fmt.println("Note: This will take longer due to 4D optimization")

	weight_range := 5.0
	best_weights, best_rmse := pdpm_grid_search_multifactor_4d(
		factors,
		rf_discount,
		option_chain,
		spot_price,
		weight_range,
		10,
		allocator,
		horizon,
	)

	fmt.printf("4-Factor Exponential SDF calibration complete:\n")
	fmt.printf("  Market beta (b1): %.4f\n", best_weights[0])
	fmt.printf("  Vol beta (b2): %.4f\n", best_weights[1])
	fmt.printf("  Momentum beta (b3): %.4f\n", best_weights[2])
	fmt.printf("  Macro beta (b4): %.4f\n", best_weights[3])
	fmt.printf("  RMSE: $%.6f\n", best_rmse)

	// Construct final SDF
	weighted_factors := make([]f64, n, context.temp_allocator)
	defer delete(weighted_factors, context.temp_allocator)

	for i in 0 ..< n {
		weighted_factors[i] =
			best_weights[0] * market_returns[i] +
			best_weights[1] * volatility_proxy[i] +
			best_weights[2] * momentum_factor[i] +
			best_weights[3] * macro_factor[i]
	}

	sum_exp := 0.0
	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n)
	scale := rf_discount / mean_exp

	values := make([]f64, n, allocator)
	for i in 0 ..< n {
		val := weighted_factors[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		values[i] = scale * math.exp(val)
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

	parameters := make([]f64, 4, allocator)
	parameters[0] = best_weights[0]
	parameters[1] = best_weights[1]
	parameters[2] = best_weights[2]
	parameters[3] = best_weights[3]

	final_sdf := StochasticDiscountFactor {
		model_type      = .Exponential_Utility,
		values          = values,
		parameters      = parameters,
		risk_aversion   = 0.0,
		discount_factor = rf_discount,
		mean            = mean,
		variance        = variance,
	}

	delete(factors, allocator)

	return CalibrationResult {
		sdf          = final_sdf,
		rmse         = best_rmse,
		max_error    = 0.0,
		n_iterations = 20000, // 10^4 coarse + 10^4 fine
		converged    = _grid_interior(
			best_weights[0],
			-weight_range,
			weight_range,
		) && _grid_interior(best_weights[1], -weight_range, weight_range) && _grid_interior(best_weights[2], -weight_range, weight_range) && _grid_interior(best_weights[3], -weight_range, weight_range),
	}
} // ============================================================================
// SDF Surface Visualization (Using Your Plot Package)
// ============================================================================

// Generate SDF surface data as a 2D heatmap
plot_sdf_surface_2d :: proc(
	calibration: ^CalibrationResult,
	market_returns: []f64,
	volatility_proxy: []f64,
	output_path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(calibration.sdf.parameters) < 2 {
		fmt.println("Error: Need at least 2 parameters for 2D surface")
		return false
	}

	// Create a grid of factor values
	grid_size := 20
	min_ret := -0.10 // -10%
	max_ret := 0.10 // +10%
	min_vol := -0.05 // -5%
	max_vol := 0.05 // +5%

	// Calculate normalization constant from historical data
	weights := calibration.sdf.parameters
	n_hist := len(market_returns)

	sum_exp := 0.0
	for i in 0 ..< n_hist {
		val := weights[0] * market_returns[i] + weights[1] * volatility_proxy[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n_hist)
	scale := calibration.sdf.discount_factor / mean_exp

	// Generate heatmap data
	data := make([]f64, grid_size * grid_size, allocator)
	defer delete(data, allocator)

	min_sdf := math.F64_MAX
	max_sdf := -math.F64_MAX

	for i in 0 ..< grid_size {
		market_ret := min_ret + (max_ret - min_ret) * f64(i) / f64(grid_size - 1)
		for j in 0 ..< grid_size {
			vol_ret := min_vol + (max_vol - min_vol) * f64(j) / f64(grid_size - 1)

			// Calculate SDF value
			val := weights[0] * market_ret + weights[1] * vol_ret
			if val > 100.0 {val = 100.0}
			if val < -100.0 {val = -100.0}
			sdf_value := scale * math.exp(val)

			idx := i * grid_size + j
			data[idx] = sdf_value

			if sdf_value < min_sdf {min_sdf = sdf_value}
			if sdf_value > max_sdf {max_sdf = sdf_value}
		}
	}

	// Create labels
	row_labels := make([]string, grid_size, allocator)
	col_labels := make([]string, grid_size, allocator)
	defer {
		delete(row_labels, allocator)
		delete(col_labels, allocator)
	}

	for i in 0 ..< grid_size {
		market_ret := min_ret + (max_ret - min_ret) * f64(i) / f64(grid_size - 1)
		vol_ret := min_vol + (max_vol - min_vol) * f64(i) / f64(grid_size - 1)
		row_labels[i] = fmt.tprintf("%.1f%%", market_ret * 100)
		col_labels[i] = fmt.tprintf("%.1f%%", vol_ret * 100)
	}

	// Create plot config
	config := p.DEFAULT_PLOT_CONFIG
	config.title = "SDF Surface (Market Return vs Volatility)"
	config.width = 1000
	config.height = 800

	// Generate heatmap
	ok := p.heatmap_png(
		data,
		grid_size,
		grid_size,
		output_path,
		row_labels,
		col_labels,
		config,
		allocator,
	)

	if ok {
		fmt.printf("SDF surface plot saved to: %s\n", output_path)
		fmt.printf("SDF range: %.4f to %.4f\n", min_sdf, max_sdf)
	}

	return ok
}

// Plot SDF values against market returns (1D slice)
plot_sdf_vs_returns :: proc(
	calibration: ^CalibrationResult,
	market_returns: []f64,
	volatility_proxy: []f64,
	output_path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	n := len(market_returns)
	if n == 0 {return false}

	// Calculate SDF values for each historical state
	sdf_values := make([]f64, n, allocator)
	defer delete(sdf_values, allocator)

	weights := calibration.sdf.parameters

	// Calculate normalization
	sum_exp := 0.0
	for i in 0 ..< n {
		val := weights[0] * market_returns[i] + weights[1] * volatility_proxy[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sum_exp += math.exp(val)
	}
	mean_exp := sum_exp / f64(n)
	scale := calibration.sdf.discount_factor / mean_exp

	// Calculate SDF for each state
	for i in 0 ..< n {
		val := weights[0] * market_returns[i] + weights[1] * volatility_proxy[i]
		if val > 100.0 {val = 100.0}
		if val < -100.0 {val = -100.0}
		sdf_values[i] = scale * math.exp(val)
	}

	// Sort by market return for clean plot
	sorted_returns := make([]f64, n, allocator)
	sorted_sdf := make([]f64, n, allocator)
	defer {
		delete(sorted_returns, allocator)
		delete(sorted_sdf, allocator)
	}

	copy(sorted_returns, market_returns)
	copy(sorted_sdf, sdf_values)

	// Simple bubble sort (replace with quicksort for large n)
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if sorted_returns[j] > sorted_returns[j + 1] {
				sorted_returns[j], sorted_returns[j + 1] = sorted_returns[j + 1], sorted_returns[j]
				sorted_sdf[j], sorted_sdf[j + 1] = sorted_sdf[j + 1], sorted_sdf[j]
			}
		}
	}

	// Create plot
	lines := []p.LineData {
		p.LineData {
			xs = sorted_returns,
			ys = sorted_sdf,
			color = p.RED,
			style = .Solid,
			label = "SDF M(R_mkt, R_vol)",
		},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Stochastic Discount Factor vs Market Return"
	config.x_label = "Market Return"
	config.y_label = "SDF Value M"
	config.show_grid = true

	ok := p.multi_line_png(lines, output_path, config, allocator)

	if ok {
		fmt.printf("SDF vs Returns plot saved to: %s\n", output_path)
	}

	return ok
}
