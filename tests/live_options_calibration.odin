package tests

import w "../wotan/core"
import fin "../wotan/finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

live_options_calibration_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=======================================================")
	fmt.println("         LIVE OPTIONS VOLATILITY CALIBRATION")
	fmt.println("=======================================================\n")

	symbol := "AAPL" // Change to "SPY", "MSFT", etc.
	r := 0.05 // Simplified risk-free rate

	// 1. Fetch live options chain using YOUR existing parser
	fmt.printf("1. Fetching live options chain for %s...\n", symbol)
	chain := fin.fetch_yahoo_options(symbol, allocator)

	if chain.n_options == 0 {
		fmt.println("⚠️  No options data found. Aborting.")

		// DEBUG: Let's see what Yahoo actually returned
		url := fmt.tprintf("https://query1.finance.yahoo.com/v7/finance/options/%s", symbol)
		text, ok := net.http_get(url, allocator)
		if ok && len(text) > 0 {
			max_len := 400
			if len(text) < max_len {max_len = len(text)}
			fmt.printf("\n--- DEBUG: Raw Yahoo Response (first %d chars) ---\n", max_len)
			fmt.println(text[:max_len])
			fmt.println("---------------------------------------------------\n")
		}
		if ok {delete(text, allocator)}

		return
	}
	fmt.printf("   Raw options fetched: %d\n", chain.n_options)

	// 2. Fetch current spot price using your existing DataFrame ingestion
	fmt.println("2. Fetching underlying spot price...")
	df := net.read_yahoo(symbol, .Daily, .OneMonth, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("⚠️  Failed to fetch spot price. Aborting.")
		return
	}

	// Get the last close price (Index 4 is "Close" based on your yahoo_json_to_dataframe)
	last_idx := df.rows - 1
	spot, _ := w.column_at_float(&df.columns[4], last_idx)
	fmt.printf("   Live Spot Price: $%.2f\n\n", spot)


	// 3. Filter and build the VolSurfacePoint slice
	fmt.println("3. Filtering for liquid, near-the-money options...")
	surface := make([dynamic]fin.VolSurfacePoint, 0, allocator)
	defer delete(surface)

	for i in 0 ..< chain.n_options {
		iv := chain.implied_vols[i]
		price := chain.market_prices[i]
		strike := chain.strikes[i]
		days_to_exp := chain.expiries[i] * 365.25

		// 1. Filter by Days to Expiry (e.g., 7 to 180 days)
		if days_to_exp < 7.0 || days_to_exp > 180.0 {
			continue
		}

		// 2. Filter by Moneyness: Keep options within 70% to 130% of the spot price.
		// This eliminates illiquid, deeply ITM/OTM options with stale prices.
		if strike < spot * 0.70 || strike > spot * 1.30 {
			continue
		}

		// 3. Filter by reasonable Implied Volatility (5% to 150%)
		// Anything above 150% is almost certainly a stale price artifact.
		if iv < 0.05 || iv > 1.50 {
			continue
		}

		// 4. Ensure price is positive
		if price <= 0.0 {
			continue
		}

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

	fmt.printf("   Filtered to %d high-quality data points.\n\n", len(surface))

	if len(surface) < 10 {
		fmt.println("⚠️  Insufficient filtered data points for robust calibration.")
		return
	}

	// 4. Calibrate SABR
	fmt.println("4. Calibrating SABR Model...")
	sabr_res := fin.calibrate_sabr(surface[:], spot, r, allocator)
	fmt.printf("   Status:   %v (Iterations: %d)\n", sabr_res.converged, sabr_res.iterations)
	fmt.printf("   RMSE:     %.4f%%\n", sabr_res.rmse * 100.0)
	fmt.printf(
		"   Params:   α=%.4f, β=%.4f, ρ=%.4f, ν=%.4f\n\n",
		sabr_res.params.alpha,
		sabr_res.params.beta,
		sabr_res.params.rho,
		sabr_res.params.nu,
	)

	// 5. Calibrate Heston
	fmt.println("5. Calibrating Heston Model...")
	heston_res := fin.calibrate_heston(surface[:], spot, r, allocator)
	fmt.printf("   Status:   %v (Iterations: %d)\n", heston_res.converged, heston_res.iterations)
	fmt.printf("   RMSE:     %.4f%%\n", heston_res.rmse * 100.0)
	fmt.printf(
		"   Params:   v0=%.4f, κ=%.4f, θ=%.4f, σ=%.4f, ρ=%.4f\n\n",
		heston_res.params.v0,
		heston_res.params.kappa,
		heston_res.params.theta,
		heston_res.params.sigma,
		heston_res.params.rho,
	)

	// 6. Surface Fit Comparison
	fmt.println("6. Surface Fit Comparison (Sample Strikes):")
	fmt.printf(
		"%-10s | %-10s | %-10s | %-10s | %-10s\n",
		"Strike",
		"Mkt IV",
		"SABR IV",
		"Heston IV",
		"Heston Err",
	)
	fmt.println("-----------|------------|------------|------------|------------")

	// Print a representative sample of the fitted surface
	step := len(surface) / 5
	if step < 1 {step = 1}

	// ✅ FIXED: Standard Odin for-loop with explicit step increment
	for i := 0; i < len(surface); i += step {
		pt := surface[i]
		F := spot * math.exp(r * pt.expiry)

		sabr_iv := fin.sabr_implied_vol(F, pt.strike, pt.expiry, sabr_res.params)

		// Price with Heston and invert to IV for an apples-to-apples comparison
		h_price := fin.heston_price(spot, pt.strike, pt.expiry, r, heston_res.params, .Call, 2000)
		h_iv, conv, _ := fin.implied_volatility(
			h_price,
			spot,
			pt.strike,
			pt.expiry,
			r,
			.Call,
			allocator,
		)

		err := math.abs(h_iv - pt.implied_vol) * 100.0

		h_iv_pct := 0.0
		if conv {
			h_iv_pct = h_iv * 100.0
		}

		fmt.printf(
			"%-10.2f | %-10.2f%% | %-10.2f%% | %-10.2f%% | %-10.2f%%\n",
			pt.strike,
			pt.implied_vol * 100.0,
			sabr_iv * 100.0,
			h_iv_pct,
			err,
		)
	}
	fmt.println("=======================================================\n")
}
