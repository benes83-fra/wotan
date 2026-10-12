package tests

import bfin "../wotan/behavoral_finance"
import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

market_implied_psychology_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Market-Implied Psychology Calibration ===")
	fmt.println("Reverse-engineering investor bias from a synthetic option chain.")

	S := 100.0
	r := 0.05
	T := 0.5 // 6 months
	true_sigma := 0.20 // The actual, physical volatility of the asset

	// 1. Simulate a "Panicked" Market
	// The true vol is 20%, but investors are terrified of a crash.
	// They distort probabilities with gamma = 0.40 (extreme tail overweighting).
	panic_params := bfin.Prospect_Params {
		alpha  = 0.88,
		beta   = 0.88,
		lambda = 2.25,
		gamma  = 0.40,
	}

	strikes := []f64{70, 80, 90, 100, 110, 120, 130}
	market_data := make([]fin.VolSurfacePoint, len(strikes), allocator)
	defer delete(market_data, allocator)

	fmt.println("\nGenerating synthetic 'Panicked' market prices...")
	for K, i in strikes {
		// The market prices options using distorted probabilities
		panicked_price := bfin.prospect_theory_price(
			S,
			K,
			T,
			r,
			true_sigma,
			panic_params,
			.Call,
			500,
			allocator,
		)

		market_data[i] = fin.VolSurfacePoint {
			strike       = K,
			expiry       = T,
			market_price = panicked_price,
			implied_vol  = 0.0, // We will let the calibrator use the raw price
		}
	}

	// 2. Calibrate Prospect Theory to the Market
	fmt.println("Running Adam optimizer to extract Market-Implied Psychology...")
	result := bfin.calibrate_prospect_theory(market_data, S, r, true_sigma, allocator)

	fmt.println("\n--- Calibration Results ---")
	fmt.printf("Converged: %v (RMSE: $%.4f)\n", result.converged, result.rmse)
	fmt.println("\nParameter | True (TK92) | Market-Implied | Interpretation")
	fmt.println("----------|-------------|----------------|---------------")
	fmt.printf(
		"Alpha     |    0.88     |      %.2f      | Gain sensitivity\n",
		result.params.alpha,
	)
	fmt.printf("Beta      |    0.88     |      %.2f      | Loss sensitivity\n", result.params.beta)
	fmt.printf("Lambda    |    2.25     |      %.2f      | Loss Aversion\n", result.params.lambda)
	fmt.printf(
		"Gamma     |    0.61     |      %.2f      | Probability Weighting\n",
		result.params.gamma,
	)

	fmt.println("\n💡 Insight:")
	if result.params.gamma < 0.55 {
		fmt.println("The market is exhibiting EXTREME CRASHOPHOBIA.")
		fmt.println("Investors are heavily overweighting the probability of a left-tail event.")
	} else if result.params.gamma > 0.70 {
		fmt.println("The market is highly rational / risk-neutral.")
	}

	fmt.println("\n=== Market-Implied Psychology Test Complete ===\n")
}
