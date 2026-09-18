package ml_finance

import "core:fmt"
import "core:math"
import "core:mem"

DispersionSignal :: struct {
	index_iv:            f64,
	index_rv:            f64,
	avg_comp_iv:         f64,
	avg_comp_rv:         f64,
	implied_corr:        f64,
	realized_corr:       f64,
	correlation_premium: f64,
	signal:              f64,
	recommendation:      string,
}

// normalize_weights ensures weights sum to 1.0
normalize_weights :: proc(weights: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(weights)
	out := make([]f64, n, allocator)
	sum := 0.0
	for w in weights {sum += w}
	if sum < 1e-12 {
		for i in 0 ..< n {out[i] = 1.0 / f64(n)}
		return out
	}
	for i in 0 ..< n {out[i] = weights[i] / sum}
	return out
}

// calculate_pairwise_correlation derives the average pairwise correlation
// using proper index weights (NOT equal weights).
//
// σ_I² = Σᵢ wᵢ²σᵢ² + Σ_{i≠j} wᵢwⱼσᵢσⱼρ
// Assuming constant average pairwise correlation ρ:
// σ_I² = Σᵢ wᵢ²σᵢ² + ρ * [(Σᵢ wᵢσᵢ)² - Σᵢ wᵢ²σᵢ²]
// Solving for ρ:
// ρ = (σ_I² - Σᵢ wᵢ²σᵢ²) / [(Σᵢ wᵢσᵢ)² - Σᵢ wᵢ²σᵢ²]
calculate_pairwise_correlation :: proc(index_var: f64, comp_vars: []f64, weights: []f64) -> f64 {
	n := len(comp_vars)
	if n == 0 {return 0.0}

	// Normalize weights to sum to 1
	w := normalize_weights(weights)
	defer delete(w)

	// Σᵢ wᵢ²σᵢ² (weighted sum of component variances)
	weighted_comp_var := 0.0
	for i in 0 ..< n {
		weighted_comp_var += w[i] * w[i] * comp_vars[i]
	}

	// (Σᵢ wᵢσᵢ)² (square of weighted sum of vols)
	sum_w_vol := 0.0
	for i in 0 ..< n {
		sum_w_vol += w[i] * math.sqrt_f64(math.max(comp_vars[i], 0.0))
	}
	sum_w_vol_sq := sum_w_vol * sum_w_vol

	// Cross-term: (Σwᵢσᵢ)² - Σwᵢ²σᵢ²
	denominator := sum_w_vol_sq - weighted_comp_var

	if math.abs(denominator) < 1e-10 {
		return 0.0
	}

	rho := (index_var - weighted_comp_var) / denominator

	// Clamp to valid correlation bounds [-1, 1]
	if rho > 1.0 {rho = 1.0}
	if rho < -1.0 {rho = -1.0}

	return rho
}

compute_dispersion_signal :: proc(
	index_iv: f64,
	index_rv: f64,
	comp_ivs: []f64,
	comp_rvs: []f64,
	weights: []f64,
	hist_corr_mean: f64,
	hist_corr_std: f64,
) -> DispersionSignal {
	sig: DispersionSignal
	n := len(comp_ivs)

	sig.index_iv = index_iv
	sig.index_rv = index_rv

	// Calculate average component vols
	sum_iv := 0.0
	sum_rv := 0.0
	for i in 0 ..< n {
		sum_iv += comp_ivs[i]
		sum_rv += comp_rvs[i]
	}
	sig.avg_comp_iv = sum_iv / f64(n)
	sig.avg_comp_rv = sum_rv / f64(n)

	// Calculate variances
	index_var_iv := index_iv * index_iv
	index_var_rv := index_rv * index_rv

	comp_vars_iv := make([]f64, n)
	comp_vars_rv := make([]f64, n)
	defer {delete(comp_vars_iv); delete(comp_vars_rv)}

	for i in 0 ..< n {
		comp_vars_iv[i] = comp_ivs[i] * comp_ivs[i]
		comp_vars_rv[i] = comp_rvs[i] * comp_rvs[i]
	}

	// Use weighted correlation formula
	sig.implied_corr = calculate_pairwise_correlation(index_var_iv, comp_vars_iv, weights)
	sig.realized_corr = calculate_pairwise_correlation(index_var_rv, comp_vars_rv, weights)

	// The Correlation Risk Premium (CRP)
	sig.correlation_premium = sig.implied_corr - sig.realized_corr

	// Z-Score the premium
	if hist_corr_std > 1e-6 {
		z := (sig.correlation_premium - hist_corr_mean) / hist_corr_std
		sig.signal = math.tanh(z / 1.5)
	} else {
		sig.signal = 0.0
	}

	// Recommendation
	if sig.signal > 0.5 {
		sig.recommendation = "SHORT CORRELATION (Sell Index Vol / Buy Component Vol)"
	} else if sig.signal < -0.5 {
		sig.recommendation = "LONG CORRELATION (Buy Index Vol / Sell Component Vol)"
	} else {
		sig.recommendation = "NEUTRAL (Correlation fairly priced)"
	}

	return sig
}

print_dispersion_signal :: proc(sig: DispersionSignal) {
	fmt.println(
		"\n╔════════════════════════════════════════════════════════════╗",
	)
	fmt.println("║          CROSS-ASSET DISPERSION TRADING DASHBOARD          ║")
	fmt.println(
		"╠════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║  Index IV (VIX):        %6.2f%%                            ║\n",
		sig.index_iv * 100.0,
	)
	fmt.printf(
		"║  Index Forecast RV:     %6.2f%%                            ║\n",
		sig.index_rv * 100.0,
	)
	fmt.println(
		"╠════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║  Avg Component IV:      %6.2f%%                            ║\n",
		sig.avg_comp_iv * 100.0,
	)
	fmt.printf(
		"║  Avg Component RV:      %6.2f%%                            ║\n",
		sig.avg_comp_rv * 100.0,
	)
	fmt.println(
		"╠════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║  IMPLIED CORRELATION:   %+6.2f%% (Market Pricing)           ║\n",
		sig.implied_corr * 100.0,
	)
	fmt.printf(
		"║  REALIZED CORRELATION:  %+6.2f%% (Model Forecast)           ║\n",
		sig.realized_corr * 100.0,
	)
	fmt.println(
		"╠════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║  CORRELATION PREMIUM:   %+6.2f%%                           ║\n",
		sig.correlation_premium * 100.0,
	)
	fmt.printf("║  NORMALIZED SIGNAL:     %+6.3f                            ║\n", sig.signal)
	fmt.println(
		"╠════════════════════════════════════════════════════════════╣",
	)
	fmt.printf("║  ACTION: %s  ║\n", sig.recommendation)
	fmt.println(
		"╚════════════════════════════════════════════════════════════╝",
	)
}
