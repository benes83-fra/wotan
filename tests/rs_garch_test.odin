package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import yahoo "../wotan/net"
import p "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

// RS-GARCH reference checks on a deterministic 2-regime series (numpy / scipy):
// Hamilton log-likelihood and filtered probabilities at fixed parameters, the
// exact Viterbi path, and a fit that reaches the L-BFGS-B + Nelder-Mead MLE
rs_garch_reference_checks :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== RS-GARCH Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// simulate: Haas-type regime variances, S_t Markov with p11 = .98, p22 = .95
	n := 1000
	y := make([]f64, n)
	defer delete(y)
	state: u64 = 21
	s := 0
	v1, v2 := 1e-4, 1e-4
	for t in 0 ..< n {
		if t > 0 {
			v1 = 2e-6 + 0.04 * y[t - 1] * y[t - 1] + 0.9 * v1
			v2 = 4e-5 + 0.1 * y[t - 1] * y[t - 1] + 0.8 * v2
			stay := 0.98 if s == 0 else 0.95
			if garch_ref_uniform(&state) > stay {s = 1 - s}
		}
		y[t] = math.sqrt_f64(v1 if s == 0 else v2) * garch_ref_normal(&state)
	}
	expect(&passed, &total, abs(y[999] / 0.0081514132230485863 - 1) < 1e-9, "reference series reproduced")

	P := ts.RS_GARCH_Params{omega_1 = 3e-6, alpha_1 = 0.05, beta_1 = 0.88, omega_2 = 3e-5, alpha_2 = 0.12, beta_2 = 0.75, p_11 = 0.97, p_22 = 0.9}
	ll, probs := ts._rs_garch_hamilton_filter(P, y, allocator, true)
	defer {for r in probs {delete(r)}; delete(probs)}
	sums_ok := true
	for r in probs {sums_ok &= abs(r[0] + r[1] - 1) < 1e-15 && r[0] >= 0 && r[1] >= 0}
	expect(&passed, &total,
		abs(ll - 3192.820761805702) < 1e-8 && abs(probs[500][0] - 0.94123712621829558) < 1e-12 &&
		abs(probs[999][0] - 0.84442553391881348) < 1e-12 && sums_ok,
		fmt.tprintf("Hamilton filter LL %.9f and filtered probabilities match numpy", ll))

	path := ts._viterbi_path(P, y, probs, allocator)
	defer delete(path)
	count, first := 0, -1
	for v, t in path {
		if v == 1 {count += 1; if first < 0 {first = t}}
	}
	expect(&passed, &total, count == 231 && first == 119,
		fmt.tprintf("Viterbi path (regime 2 on %d days from t = %d; filtered argmax: 181)", count, first))

	fit := ts.rs_garch_fit(y, allocator)
	defer {
		for r in fit.filtered_probs {delete(r)}
		delete(fit.filtered_probs)
		delete(fit.most_likely_path)
	}
	fp := fit.params
	expect(&passed, &total,
		fit.log_likelihood > 3210.1868856751 - 1e-4 && abs(fp.p_11 - 0.986383537) < 5e-3 && abs(fp.p_22 - 0.9491122403) < 1e-2 &&
		fp.omega_1 / (1 - fp.alpha_1 - fp.beta_1) < fp.omega_2 / (1 - fp.alpha_2 - fp.beta_2),
		fmt.tprintf("fit reaches the MLE (LL %.6f vs 3210.186886, p11 %.4f, p22 %.4f), regime 1 = calm", fit.log_likelihood, fp.p_11, fp.p_22))

	fmt.printf("RS-GARCH reference summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

rs_garch_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	rs_garch_reference_checks(context.allocator)
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
		if curr_close / prev_close == 0.0 || prev_close == 0 {
			continue
		}
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

	p.multi_line_png(lines, "plots/rs_garch_regimes.png", config, allocator)
	fmt.printf("✓ Saved: rs_garch_regimes.png\n")

	fmt.println("\n✓ RS-GARCH test completed!")
}
