package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import fin "../wotan/finance"
import yahoo "../wotan/net"
import p "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

garch_options_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GARCH-Powered Option Pricing Test ===\n")

	main_alloc := context.allocator

	// 1. Fetch SPY Data and Fit GARCH
	fmt.println("Fetching SPY data and fitting GARCH(1,1)...")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	n := spy_df.rows
	returns := make([]f64, n - 1, main_alloc)
	defer delete(returns, main_alloc)

	close_prices := make([]f64, n, main_alloc)
	defer delete(close_prices, main_alloc)

	for i in 0 ..< n {
		close_prices[i], _ = w.column_at_float(&spy_df.columns[4], i)
		if i > 0 {
			prev_close := close_prices[i - 1]
			curr_close := close_prices[i]
			returns[i - 1] = math.ln_f64(curr_close / prev_close)
		}
	}

	current_price := close_prices[n - 1]
	residuals := ts.extract_residuals(returns, main_alloc)
	defer delete(residuals, main_alloc)

	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 2000, 1e-4, main_alloc)
	defer {
		delete(garch_result.params.alpha, main_alloc)
		delete(garch_result.params.beta, main_alloc)
		delete(garch_result.conditional_var, main_alloc)
		delete(garch_result.standardized_resid, main_alloc)
	}

	omega := garch_result.params.omega
	alpha := garch_result.params.alpha[0]
	beta := garch_result.params.beta[0]
	current_var := garch_result.conditional_var[len(garch_result.conditional_var) - 1]

	fmt.printf("Current SPY Price: $%.2f\n", current_price)
	fmt.printf("Current Daily Variance: %.8f\n", current_var)

	// 2. Price Options Across Different Maturities (Term Structure)
	fmt.println("\n--- GARCH Volatility Term Structure & Pricing ---")
	fmt.printf(
		"%-10s %-15s %-15s %-15s %-10s\n",
		"Maturity",
		"GARCH Vol",
		"BS Price",
		"Vega",
		"Delta",
	)
	fmt.printf(
		"%-10s %-15s %-15s %-15s %-10s\n",
		"----------",
		"---------------",
		"---------------",
		"---------------",
		"----------",
	)

	maturities_years := []f64{7.0 / 252.0, 30.0 / 252.0, 90.0 / 252.0, 252.0 / 252.0} // 1 week, 1 mo, 3 mo, 1 yr
	strike := current_price // At-The-Money (ATM)
	r := 0.05 // 5% risk-free rate

	vols := make([]f64, len(maturities_years), main_alloc)
	prices := make([]f64, len(maturities_years), main_alloc)
	defer {
		delete(vols, main_alloc)
		delete(prices, main_alloc)
	}

	for T, i in maturities_years {
		price, greeks := fin.garch_price_and_greeks(
			current_price,
			strike,
			T,
			r,
			omega,
			alpha,
			beta,
			current_var,
			.Call,
			main_alloc,
		)

		// Extract the vol that was actually used
		horizon_days := int(math.max(1.0, T * 252.0))
		sigma_garch := fin.garch_term_structure_vol(omega, alpha, beta, current_var, horizon_days)

		vols[i] = sigma_garch
		prices[i] = price

		fmt.printf(
			"%-10.1f %-15.2f%% %-15.2f %-15.4f %-10.4f\n",
			T * 252.0,
			sigma_garch * 100.0,
			price,
			greeks.vega,
			greeks.delta,
		)
	}

	// 3. Visualization: GARCH Volatility Term Structure
	fmt.println("\nGenerating Volatility Term Structure Plot...")

	// Create X axis (days to maturity)
	days_to_mat := make([]f64, len(maturities_years), main_alloc)
	defer delete(days_to_mat, main_alloc)
	for T, i in maturities_years {
		days_to_mat[i] = T * 252.0
	}

	lines := []p.LineData {
		p.LineData {
			xs = days_to_mat,
			ys = vols,
			color = p.BLUE,
			style = .Solid,
			label = "GARCH Forecast Vol",
		},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "SPY ATM Option: GARCH Volatility Term Structure"
	config.x_label = "Days to Maturity"
	config.y_label = "Annualized Volatility (%)"
	config.show_grid = true

	p.multi_line_png(lines, "garch_vol_term_structure.png", config, allocator)
	fmt.printf("✓ Saved: garch_vol_term_structure.png\n")

	fmt.println("\n✓ GARCH Option Pricing test completed!")
}
