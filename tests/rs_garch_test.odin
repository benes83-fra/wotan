package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import yahoo "../wotan/net"
import p "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

rs_garch_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Regime-Switching GARCH Test (SPY) ===\n")

	main_alloc := context.allocator

	// 1. Fetch Data
	fmt.println("Fetching SPY data...")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	n := spy_df.rows
	returns := make([]f64, n - 1, main_alloc)
	defer delete(returns, main_alloc)

	dates := make([]f64, n - 1, main_alloc)
	defer delete(dates, main_alloc)

	for i in 1 ..< n {
		prev_close, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_close, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_close / prev_close)
		dates[i - 1] = f64(i)
	}

	// 2. Fit RS-GARCH
	fmt.println("Fitting Regime-Switching GARCH (2 Regimes)...")
	fmt.println("Note: This involves a grid search and may take a few seconds...")

	rs_result := ts.rs_garch_fit(returns, main_alloc)
	defer {
		for row in rs_result.filtered_probs {delete(row, main_alloc)}
		delete(rs_result.filtered_probs, main_alloc)
		delete(rs_result.most_likely_path, main_alloc)
	}

	// 3. Display Results
	fmt.printf("\n--- RS-GARCH Parameters ---\n")
	fmt.printf("Regime 1 (Calm):\n")
	fmt.printf(
		"  ω: %.8f, α: %.4f, β: %.4f\n",
		rs_result.params.omega_1,
		rs_result.params.alpha_1,
		rs_result.params.beta_1,
	)
	fmt.printf("Regime 2 (Crisis):\n")
	fmt.printf(
		"  ω: %.8f, α: %.4f, β: %.4f\n",
		rs_result.params.omega_2,
		rs_result.params.alpha_2,
		rs_result.params.beta_2,
	)

	fmt.printf("\nTransition Probabilities:\n")
	fmt.printf("  P(Calm -> Calm):   %.2f%%\n", rs_result.params.p_11 * 100)
	fmt.printf("  P(Crisis -> Crisis): %.2f%%\n", rs_result.params.p_22 * 100)

	fmt.printf("\nLog-Likelihood: %.2f\n", rs_result.log_likelihood)

	// Count days in each regime
	crisis_days := 0
	for state in rs_result.most_likely_path {
		if state == 1 {crisis_days += 1}
	}
	fmt.printf(
		"Crisis Regime Detected: %d / %d days (%.1f%%)\n",
		crisis_days,
		len(returns),
		f64(crisis_days) / f64(len(returns)) * 100,
	)

	// 4. Visualization
	fmt.println("\nGenerating Visualization...")

	// Create a "Crisis Indicator" series (1 if in Regime 2, 0 otherwise)
	crisis_indicator := make([]f64, len(returns), main_alloc)
	defer delete(crisis_indicator, main_alloc)

	for i, state in rs_result.most_likely_path {
		if state == 1 {
			crisis_indicator[i] = 1.0
		}
	}

	// Plot Returns with Crisis Regimes Highlighted
	lines := []p.LineData {
		p.LineData{xs = dates, ys = returns, color = p.BLUE, style = .Solid, label = "Returns"},
		p.LineData {
			xs = dates,
			ys = crisis_indicator,
			color = p.RED,
			style = .Solid,
			label = "Crisis Regime (Scaled)",
		},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "SPY Returns with Regime-Switching GARCH (Crisis Detection)"
	config.x_label = "Time (days)"
	config.y_label = "Daily Return / Regime State"
	config.show_grid = true

	p.multi_line_png(lines, "rs_garch_regimes.png", config, allocator)
	fmt.printf("✓ Saved: rs_garch_regimes.png\n")

	fmt.println("\n✓ RS-GARCH test completed!")
}
