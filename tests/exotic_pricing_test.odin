package tests

import w "../wotan/core"
import fin "../wotan/finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

exotic_pricing_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("              EXOTIC PRICING: THE 'SMILE EFFECT' DEMO")
	fmt.println("======================================================================\n")

	symbol := "AAPL"
	r := 0.05

	// 1. Fetch and Calibrate (Reusing your robust pipeline)
	fmt.printf("1. Fetching live data and calibrating Heston for %s...\n", symbol)
	chain := fin.fetch_yahoo_options(symbol, allocator)

	df := net.read_yahoo(symbol, .Daily, .OneMonth, allocator)
	defer w.destroy_dataframe(&df)

	last_idx := df.rows - 1
	spot, _ := w.column_at_float(&df.columns[4], last_idx)

	surface := make([dynamic]fin.VolSurfacePoint, 0, allocator)
	defer delete(surface)

	for i in 0 ..< chain.n_options {
		iv := chain.implied_vols[i]
		price := chain.market_prices[i]
		strike := chain.strikes[i]
		days_to_exp := chain.expiries[i] * 365.25

		if days_to_exp >= 20.0 &&
		   days_to_exp <= 60.0 &&
		   strike >= spot * 0.85 &&
		   strike <= spot * 1.15 &&
		   iv >= 0.05 &&
		   iv <= 0.80 &&
		   price > 0.0 {
			append(
				&surface,
				fin.VolSurfacePoint {
					strike = strike,
					expiry = chain.expiries[i],
					implied_vol = iv,
					market_price = price,
				},
			)
		}
	}

	heston_res := fin.calibrate_heston(surface[:], spot, r, allocator)
	fmt.printf(
		"   Calibrated Heston Params: v0=%.4f, κ=%.4f, θ=%.4f, σ=%.4f, ρ=%.4f\n\n",
		heston_res.params.v0,
		heston_res.params.kappa,
		heston_res.params.theta,
		heston_res.params.sigma,
		heston_res.params.rho,
	)

	// 2. Define the Exotic Option
	// Up-and-Out Call: Pays max(S_T - K, 0) ONLY if S_t never touches the Barrier
	T := 30.0 / 365.25
	K := spot * 1.05 // Slightly Out-of-The-Money
	barrier := spot * 1.10 // 10% above current spot

	n_paths := 50000
	n_steps := 100

	fmt.println("2. Pricing Instrument: 30-Day Up-and-Out Call")
	fmt.printf(
		"   Spot: $%.2f | Strike: $%.2f | Barrier: $%.2f | T: %.2f years\n\n",
		spot,
		K,
		barrier,
		T,
	)

	// 3. Price using Black-Scholes MC (Flat Volatility = ATM Vol)
	atm_iv := 0.30 // Fallback, or extract from surface
	for pt in surface {
		if math.abs(pt.strike - spot) < 2.0 {
			atm_iv = pt.implied_vol
			break
		}
	}

	// We can reuse your existing BS MC or just write a quick inline one for comparison


	// 4. Price using Black-Scholes MC (Skew-Adjusted Volatility)
	// Find the IV at the strike K to approximate the skew
	skew_iv := atm_iv
	for pt in surface {
		if math.abs(pt.strike - K) < 2.0 {
			skew_iv = pt.implied_vol
			break
		}
	}
	// 4. Price using Black-Scholes MC (Flat Volatility = ATM Vol)
	bs_price_flat, delta_bs_flat, vega_bs_flat := fin.monte_carlo_barrier_call_bs(
		spot,
		K,
		T,
		r,
		atm_iv,
		barrier,
		n_paths,
		n_steps,
		allocator,
	)

	// 5. Price using Black-Scholes MC (Skew-Adjusted Volatility)
	bs_price_skew, _, _ := fin.monte_carlo_barrier_call_bs(
		spot,
		K,
		T,
		r,
		skew_iv,
		barrier,
		n_paths,
		n_steps,
		allocator,
	)

	// 6. Price using Calibrated Heston MC
	heston_price, delta_heston, vega_heston := fin.heston_mc_barrier_call(
		spot,
		K,
		T,
		r,
		barrier,
		heston_res.params,
		n_paths,
		n_steps,
		allocator,
	)

	// 7. Display Results
	fmt.println("3. Pricing Results Comparison:")
	fmt.println("----------------------------------------------------------------------")
	fmt.printf(" %-35s | %12s\n", "Model / Assumption", "Option Price")
	fmt.println("----------------------------------------------------------------------")
	fmt.printf(" %-35s | $%10.4f\n", "Black-Scholes (Flat ATM Vol)", bs_price_flat)
	fmt.printf(" %-35s | $%10.4f\n", "Black-Scholes (Skew-Adjusted Vol)", bs_price_skew)
	fmt.printf(" %-35s | $%10.4f\n", "Heston Model (Calibrated)", heston_price)
	fmt.println("----------------------------------------------------------------------")

	fmt.println("\n4. Autograd Greeks Comparison (Delta / Vega):")
	fmt.println("----------------------------------------------------------------------")
	fmt.printf(
		" %-35s | Delta: %8.4f | Vega: %8.4f\n",
		"Black-Scholes (Flat ATM)",
		delta_bs_flat,
		vega_bs_flat,
	)
	fmt.printf(
		" %-35s | Delta: %8.4f | Vega: %8.4f\n",
		"Heston Model (Calibrated)",
		delta_heston,
		vega_heston,
	)
	fmt.println("----------------------------------------------------------------------")

	diff := math.abs(heston_price - bs_price_flat)
	pct_diff := 0.0
	if heston_price > 0.0 {
		pct_diff = (diff / heston_price) * 100.0
	}

	fmt.printf(
		"\n💡 The Smile Effect: Black-Scholes (ATM) differs from Heston by %.2f%%\n",
		pct_diff,
	)
	fmt.println("   Reason: BS assumes flat volatility, failing to capture the true")
	fmt.println(
		"   probability of the asset hitting the barrier due to the negative skew (ρ < 0).",
	)
	fmt.println("======================================================================\n")
	// =========================================================================
	// 8. Expand: Asian and Lookback Options
	// =========================================================================
	fmt.println("5. Expanding the Library: Asian & Lookback Options")
	fmt.println("======================================================================")

	// Asian Option
	asian_bs_price, _, _ := fin.monte_carlo_asian_option(
		spot,
		K,
		T,
		r,
		atm_iv,
		n_paths,
		n_steps,
		allocator,
	)
	asian_heston_price, _, _ := fin.heston_mc_asian_call(
		spot,
		K,
		T,
		r,
		heston_res.params,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.printf(
		" %-35s | BS: $%8.4f | Heston: $%8.4f | Diff: %6.2f%%\n",
		"Arithmetic Asian Call",
		asian_bs_price,
		asian_heston_price,
		math.abs(asian_heston_price - asian_bs_price) / asian_heston_price * 100.0,
	)

	// Lookback Option
	lookback_bs_price, _, _ := fin.monte_carlo_lookback_call_option(
		spot,
		K,
		T,
		r,
		atm_iv,
		n_paths,
		n_steps,
		allocator,
	)
	lookback_heston_price, _, _ := fin.heston_mc_lookback_call(
		spot,
		K,
		T,
		r,
		heston_res.params,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.printf(
		" %-35s | BS: $%8.4f | Heston: $%8.4f | Diff: %6.2f%%\n",
		"Fixed-Strike Lookback Call",
		lookback_bs_price,
		lookback_heston_price,
		math.abs(lookback_heston_price - lookback_bs_price) / lookback_heston_price * 100.0,
	)

	fmt.println("======================================================================\n")
}

// Helper: Simple BS Monte Carlo for Barrier Options (for baseline comparison)
// Add this to wotan/finance/vol_surface.odin or keep it local in the test
monte_carlo_barrier_call_bs :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	barrier: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator,
) -> f64 {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	discount := math.exp(-r * T)
	total_payoff := 0.0

	for p in 0 ..< n_paths {
		S := S_0
		hit := false
		for step in 0 ..< n_steps {
			Z := rand.float64_normal(0.0, 1.0)
			S = S * math.exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * Z)
			if S >= barrier {
				hit = true
				break
			}
		}
		if !hit {
			total_payoff += math.max(S - K, 0.0)
		}
	}
	return (total_payoff / f64(n_paths)) * discount
}
