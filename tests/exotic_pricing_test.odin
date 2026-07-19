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
	// =========================================================================
	// 9. Expand: 2-Asset Basket Option
	// =========================================================================
	fmt.println("6. Expanding the Library: 2-Asset Basket Call (Equal Weight)")
	fmt.println("======================================================================")

	// For demonstration, we'll use AAPL and a hypothetical Asset 2 with same params
	// In production, you would calibrate params2 separately for MSFT, etc.
	S2_0 := 400.00 // e.g., MSFT price
	w1 := 0.5
	w2 := 0.5
	corr_12 := 0.65 // Typical tech sector correlation

	basket_K := w1 * spot + w2 * S2_0 // ATM basket strike

	// Black-Scholes Basket (using your existing tensor engine)
	bs_basket_price, _, _, _, _ := fin.monte_carlo_basket_option(
		spot,
		S2_0,
		basket_K,
		T,
		r,
		atm_iv,
		atm_iv,
		corr_12,
		w1,
		w2,
		n_paths,
		n_steps,
		allocator,
	)

	// Heston Basket
	heston_basket_price, delta1, delta2, vega1, vega2 := fin.heston_mc_basket_call(
		spot,
		S2_0,
		basket_K,
		T,
		r,
		w1,
		w2,
		heston_res.params,
		heston_res.params,
		corr_12,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.printf(
		" %-35s | BS: $%8.4f | Heston: $%8.4f | Diff: %6.2f%%\n",
		"2-Asset Basket Call",
		bs_basket_price,
		heston_basket_price,
		math.abs(heston_basket_price - bs_basket_price) / heston_basket_price * 100.0,
	)

	fmt.printf(
		" %-35s | Delta1: %6.4f | Delta2: %6.4f | Vega1: %6.4f | Vega2: %6.4f\n",
		"Heston Basket Greeks",
		delta1,
		delta2,
		vega1,
		vega2,
	)

	fmt.println("======================================================================\n")
	// =========================================================================
	// 7. The Downside Smile Effect: Down-and-Out Put
	// =========================================================================
	fmt.println("7. The Downside Smile Effect: Down-and-Out Put")
	fmt.println("======================================================================")

	put_K := spot * 0.95 // Slightly Out-of-The-Money Put
	put_barrier := spot * 0.90 // 10% below current spot

	// Black-Scholes Down-and-Out Put (using your existing tensor engine)
	// Note: Your existing tensor engine might need a quick `.Put` and `is_up=false` branch if you want to test it,
	// but for now we will just show the Heston price and the conceptual difference.
	// For a fair comparison, we can just price a vanilla Put and note the barrier effect.

	heston_do_put_price, delta_do_put, vega_do_put := fin.heston_mc_barrier_option(
		spot,
		put_K,
		T,
		r,
		put_barrier,
		false,
		.Put, // is_up = false, opt = .Put
		heston_res.params,
		n_paths,
		n_steps,
		allocator,
	)

	// For baseline, price a Vanilla Put in Heston to show the barrier discount
	heston_vanilla_put_price, _, _ := fin.heston_mc_barrier_option(
		spot,
		put_K,
		T,
		r,
		0.0,
		false,
		.Put, // barrier = 0.0 ensures it never knocks out
		heston_res.params,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.printf(" %-35s | $%10.4f\n", "Heston Vanilla Put", heston_vanilla_put_price)
	fmt.printf(" %-35s | $%10.4f\n", "Heston Down-and-Out Put", heston_do_put_price)
	fmt.printf(
		" %-35s | Delta: %6.4f | Vega: %6.4f\n",
		"Down-and-Out Put Greeks",
		delta_do_put,
		vega_do_put,
	)

	barrier_discount :=
		(heston_vanilla_put_price - heston_do_put_price) / heston_vanilla_put_price * 100.0
	fmt.printf(
		"\n💡 Barrier Discount: The Down-and-Out feature reduces the Put's value by %.2f%%\n",
		barrier_discount,
	)
	fmt.println("   Reason: The strong negative skew (ρ < 0) fattens the left tail,")
	fmt.println("   making the downside barrier much more likely to be hit than BS predicts.")
	fmt.println("======================================================================\n")
	// =========================================================================
	// 8. The Jump Diffusion Effect: Merton Model
	// =========================================================================
	fmt.println("8. The Jump Diffusion Effect: Merton Jump Diffusion (MJD)")
	fmt.println("======================================================================")

	// Typical MJD parameters for a tech stock (e.g., AAPL)
	// lambda = 0.15 (15% chance of a jump per year)
	// mu_j = -0.05 (Jumps tend to be downward, avg -5%)
	// sigma_j = 0.10 (Jump size volatility)
	mjd_params := fin.MJD_Params {
		sigma   = 0.25, // Diffusion vol (lower than total vol because jumps add risk)
		lambda  = 0.15,
		mu_j    = -0.05,
		sigma_j = 0.10,
	}

	// Price Barrier Option under MJD
	mjd_barrier_price, mjd_delta, mjd_vega := fin.mjd_mc_barrier_option(
		spot,
		K,
		T,
		r,
		barrier,
		true,
		.Call,
		mjd_params,
		n_paths,
		n_steps,
		allocator,
	)

	// Price Asian Option under MJD
	mjd_asian_price, _, _ := fin.mjd_mc_asian_option(
		spot,
		K,
		T,
		r,
		.Call,
		mjd_params,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.println(" Model Comparison (Up-and-Out Call):")
	fmt.printf("  %-35s | $%8.4f\n", "Black-Scholes", bs_price_flat) // From earlier in script
	fmt.printf("  %-35s | $%8.4f\n", "Heston (Stochastic Vol)", heston_price) // From earlier
	fmt.printf("  %-35s | $%8.4f\n", "Merton Jump Diffusion", mjd_barrier_price)

	fmt.println("\n Model Comparison (Asian Call):")
	fmt.printf("  %-35s | $%8.4f\n", "Black-Scholes", asian_bs_price) // From earlier
	fmt.printf("  %-35s | $%8.4f\n", "Heston (Stochastic Vol)", asian_heston_price) // From earlier
	fmt.printf("  %-35s | $%8.4f\n", "Merton Jump Diffusion", mjd_asian_price)

	fmt.printf("\n MJD Barrier Greeks | Delta: %6.4f | Vega: %6.4f\n", mjd_delta, mjd_vega)

	fmt.println("\n💡 The Jump Effect: MJD prices are often lower for Up-and-Out calls")
	fmt.println("   because a single downward jump doesn't hurt the call, but the *fear*")
	fmt.println("   of an upward jump hitting the barrier is priced in via the jump intensity.")
	fmt.println(
		"   Conversely, for Asian options, jumps can spike the average, altering the price.",
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
