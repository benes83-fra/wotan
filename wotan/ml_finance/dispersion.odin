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

// Implied (or realised) average pairwise correlation, the CBOE implied-correlation formula:
// with index weights w (normalised to sum 1) and a common correlation rho,
//   sigma_I^2 = sum w_i^2 sigma_i^2 + rho [(sum w_i sigma_i)^2 - sum w_i^2 sigma_i^2], so
//   rho = (sigma_I^2 - sum w_i^2 sigma_i^2) / [(sum w_i sigma_i)^2 - sum w_i^2 sigma_i^2].
// Clamped to [-1, 1]. NaN when undefined (fewer than two components with weight and
// variance, i.e. a zero denominator).
calculate_pairwise_correlation :: proc(index_var: f64, comp_vars: []f64, weights: []f64) -> f64 {
	n := len(comp_vars)
	assert(
		len(weights) == n,
		"calculate_pairwise_correlation: weights and comp_vars differ in length",
	)
	if n == 0 {return math.nan_f64()}

	sum_w := 0.0
	for x in weights {sum_w += x}
	w :: proc(weights: []f64, i: int, sum_w: f64, n: int) -> f64 {
		return sum_w < 1e-12 ? 1.0 / f64(n) : weights[i] / sum_w // as normalize_weights
	}
	weighted_comp_var, sum_w_vol := 0.0, 0.0
	for v, i in comp_vars {
		wi := w(weights, i, sum_w, n)
		weighted_comp_var += wi * wi * v
		sum_w_vol += wi * math.sqrt(math.max(v, 0.0))
	}
	denominator := sum_w_vol * sum_w_vol - weighted_comp_var
	if !(denominator > 1e-12 * sum_w_vol * sum_w_vol) {return math.nan_f64()}

	return clamp((index_var - weighted_comp_var) / denominator, -1.0, 1.0)
}

// Volatilities in, not variances. Signal = tanh(z / 1.5) of the z-scored premium
// implied - realised correlation against its history; > 0.5 sells correlation.
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
	assert(
		len(comp_rvs) == n && len(weights) == n,
		"compute_dispersion_signal: component inputs differ in length",
	)
	if n == 0 {
		sig.recommendation = "NEUTRAL (no components)"
		return sig
	}

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

	comp_vars_iv := make([]f64, n, context.temp_allocator)
	comp_vars_rv := make([]f64, n, context.temp_allocator)
	defer {delete(comp_vars_iv, context.temp_allocator)
		delete(comp_vars_rv, context.temp_allocator)}

	for i in 0 ..< n {
		comp_vars_iv[i] = comp_ivs[i] * comp_ivs[i]
		comp_vars_rv[i] = comp_rvs[i] * comp_rvs[i]
	}

	// Use weighted correlation formula
	sig.implied_corr = calculate_pairwise_correlation(index_var_iv, comp_vars_iv, weights)
	sig.realized_corr = calculate_pairwise_correlation(index_var_rv, comp_vars_rv, weights)

	// The Correlation Risk Premium (CRP)
	sig.correlation_premium = sig.implied_corr - sig.realized_corr

	// Z-Score the premium (NaN premium: signal NaN, recommendation neutral)
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
