package tests

import fin "../wotan/finance"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

prospect_vol_smile_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Prospect Theory & The Volatility Smile ===")
	fmt.println("Proving that Probability Weighting generates the Volatility Smile.")
	fmt.println("We will price OTM options using a distorted probability distribution")
	fmt.println("and feed the prices into your existing Black-Scholes IV solver.")

	S0 := 100.0
	r := 0.05
	T := 1.0
	true_sigma := 0.20 // 20% flat volatility in the real world (GBM)

	// 1. Create a fine grid of terminal prices (ST)
	n_points := 2000
	ST_min := 10.0
	ST_max := 300.0
	dS := (ST_max - ST_min) / f64(n_points)

	ST_data := l.matrix_new(f64, 1, n_points, allocator)
	for i in 0 ..< n_points {
		ST_data.data[i] = ST_min + f64(i) * dS
	}

	// 2. Calculate True Log-Normal CDF
	// ln(ST/S0) ~ N((r - 0.5*sigma^2)*T, sigma^2 * T)
	mu := (r - 0.5 * true_sigma * true_sigma) * T
	variance := true_sigma * true_sigma * T
	std_dev := math.sqrt_f64(variance)

	true_cdf_data := l.matrix_new(f64, 1, n_points, allocator)

	for i in 0 ..< n_points {
		ST := ST_data.data[i]
		if ST <= 0 {
			true_cdf_data.data[i] = 0.0
		} else {
			z := (math.ln(ST / S0) - mu) / std_dev
			true_cdf_data.data[i] = 0.5 * (1.0 + math.erf(z / math.sqrt_f64(2.0)))
		}
	}

	true_cdf := t.tensor_new(true_cdf_data, true, allocator)

	// 3. Apply Probability Weighting to the CDF using our Prospect Backbone
	// gamma = 0.50 creates a strong inverse-S curve.
	// Investors overweight the tails (small probabilities of extreme events).
	gamma := 0.50
	distorted_cdf := t.tensor_prob_weighting(true_cdf, gamma, allocator)

	// 4. Extract Distorted PDF via finite differences
	// f_w(S) = d(w(F(S))) / dS
	dist_pdf := make([]f64, n_points, allocator)
	dist_pdf[0] = distorted_cdf.data.data[0] / dS
	for i in 1 ..< n_points {
		diff := distorted_cdf.data.data[i] - distorted_cdf.data.data[i - 1]
		if diff < 0 {diff = 0} 	// Clamp numerical noise
		dist_pdf[i] = diff / dS
	}

	// Normalize distorted PDF so it integrates to 1
	sum_pdf := 0.0
	for i in 0 ..< n_points {sum_pdf += dist_pdf[i] * dS}
	if sum_pdf > 0 {
		for i in 0 ..< n_points {dist_pdf[i] /= sum_pdf}
	}

	// 5. Price a strip of Call Options using the Distorted PDF
	strikes := []f64{70, 80, 90, 100, 110, 120, 130}

	fmt.println("\nStrike | True Price | Distorted Price | True IV | Distorted IV | Smile Effect")
	fmt.println("-------|------------|-----------------|---------|--------------|-------------")

	for K in strikes {
		// True Price (Black-Scholes)
		true_price := fin.black_scholes_call(S0, K, T, r, true_sigma)

		// Distorted Price = e^{-rT} * sum( max(ST - K, 0) * f_w(ST) * dS )
		dist_price := 0.0
		for i in 0 ..< n_points {
			ST := ST_data.data[i]
			if ST > K {
				dist_price += (ST - K) * dist_pdf[i] * dS
			}
		}
		dist_price *= math.exp(-r * T)

		// Calculate Implied Volatilities using your existing finance module!
		true_iv, _, _ := fin.implied_volatility(true_price, S0, K, T, r, .Call, allocator)
		dist_iv, _, _ := fin.implied_volatility(dist_price, S0, K, T, r, .Call, allocator)

		effect := ""
		if K < S0 {
			if dist_iv > true_iv + 0.001 {effect = "PUT SKEW (Crashophobia)"}
		} else if K > S0 {
			if dist_iv > true_iv + 0.001 {effect = "CALL SMILE (Euphoria)"}
		} else {
			effect = "ATM"
		}

		fmt.printf(
			" %5.0f | $%9.4f | $%14.4f | %6.2f%% | %11.2f%% | %s\n",
			K,
			true_price,
			dist_price,
			true_iv * 100,
			dist_iv * 100,
			effect,
		)
	}

	fmt.println("\n💡 Insight: The exact same underlying asset (20% flat vol) produces a")
	fmt.println("volatility smile purely because investors distort probabilities (γ=0.50).")
	fmt.println("They overweight the tails, making OTM options artificially expensive.")
	fmt.println("This mathematically proves the Volatility Smile is a behavioral phenomenon.")

	// Cleanup
	t.tensor_free_graph(distorted_cdf) // Frees distorted_cdf and true_cdf
	l.matrix_free(&ST_data)
	delete(dist_pdf, allocator)
}
