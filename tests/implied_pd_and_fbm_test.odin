package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

implied_pd_and_fbm_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    IMPLIED DEFAULT PROBABILITY & FRACTIONAL BROWNIAN MOTION")
	fmt.println("======================================================================\n")

	// ========================================================================
	// 1. IMPLIED DEFAULT PROBABILITY (Breeden-Litzenberger)
	// ========================================================================
	fmt.println("1. Implied Default Probability from Option Surface")
	fmt.println("   ----------------------------------------------------------------------")

	// Synthetic option chain (e.g., a distressed firm with high skew)
	S := 50.0
	r := 0.05
	T := 1.0
	K_default := 40.0 // Default barrier (e.g., debt face value)

	// Dense grid of strikes and synthetic call prices (in-the-money to out-of-the-money)
	strikes := []f64{20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0}

	// Synthetic prices reflecting high left-tail risk (distressed firm)
	call_prices := []f64{30.15, 25.20, 20.35, 15.60, 11.10, 7.20, 4.30, 2.40, 1.20, 0.55, 0.25}

	result := fin.implied_default_probability(strikes, call_prices, r, T, K_default, allocator)

	fmt.printf("   Underlying Price (S):  $%.2f\n", S)
	fmt.printf("   Default Barrier (K):   $%.2f\n", K_default)
	fmt.printf("   Risk-Neutral PD:       %.2f%%\n", result.pd_risk_neutral * 100.0)
	fmt.printf("   Expected Loss (PV):    $%.4f\n", result.expected_loss)

	fmt.println("\n   Risk-Neutral Density (RND) Snapshot:")
	fmt.println("   Strike   |   RND")
	fmt.println("   ---------|---------")
	for i in 0 ..< len(result.rnd_strikes) {
		fmt.printf("   $%-7.2f | %.6f\n", result.rnd_strikes[i], result.rnd_values[i])
	}

	// ========================================================================
	// 2. FRACTIONAL BROWNIAN MOTION (fBm) PRICING
	// ========================================================================
	fmt.println("\n2. Fractional Black-Scholes Pricing (Monte Carlo)")
	fmt.println("   ----------------------------------------------------------------------")

	S_fbm := 100.0
	K_fbm := 100.0
	T_fbm := 1.0
	r_fbm := 0.05
	sigma_fbm := 0.20
	n_paths := 10000
	n_steps := 252

	fmt.printf(
		"   Parameters: S=%.2f, K=%.2f, T=%.1f, r=%.2f, σ=%.2f\n",
		S_fbm,
		K_fbm,
		T_fbm,
		r_fbm,
		sigma_fbm,
	)
	fmt.println("   Hurst (H) | fBm Call Price | Diff vs Standard BS (H=0.5)")
	fmt.println("   ----------|----------------|-----------------------------")

	// Standard Black-Scholes for baseline (H = 0.5)
	bs_price := fin.fbm_european_call_price(
		S_fbm,
		K_fbm,
		T_fbm,
		r_fbm,
		sigma_fbm,
		0.5,
		n_paths,
		n_steps,
		allocator,
	)
	fmt.printf("   H = 0.50  | $%-13.4f | (Baseline Standard Brownian Motion)\n", bs_price)

	// Persistent fBm (H > 0.5): Trends are more likely to continue
	h_persistent := 0.7
	price_persistent := fin.fbm_european_call_price(
		S_fbm,
		K_fbm,
		T_fbm,
		r_fbm,
		sigma_fbm,
		h_persistent,
		n_paths,
		n_steps,
		allocator,
	)
	fmt.printf(
		"   H = 0.70  | $%-13.4f | %+f (Persistent / Trending)\n",
		price_persistent,
		price_persistent - bs_price,
	)

	// Anti-persistent / Rough fBm (H < 0.5): Mean-reverting, sharp local moves
	h_rough := 0.3
	price_rough := fin.fbm_european_call_price(
		S_fbm,
		K_fbm,
		T_fbm,
		r_fbm,
		sigma_fbm,
		h_rough,
		n_paths,
		n_steps,
		allocator,
	)
	fmt.printf(
		"   H = 0.30  | $%-13.4f | %+f (Anti-persistent / Rough)\n",
		price_rough,
		price_rough - bs_price,
	)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Breeden-Litzenberger extracts the market's implied probability")
	fmt.println("     distribution directly from the cross-section of option prices.")
	fmt.println("   • Integrating the left tail of the RND yields the Risk-Neutral PD.")
	fmt.println("   • fBm with H ≠ 0.5 captures long-range dependence.")
	fmt.println("   • While raw fBm allows arbitrage, its modern descendant,")
	fmt.println("     'Rough Volatility' (H ≈ 0.1 for volatility), is the industry")
	fmt.println("     standard for fitting the steep short-term implied vol skew.")
	fmt.println("======================================================================\n")
}
