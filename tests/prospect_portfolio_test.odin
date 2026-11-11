package tests

import bfin "../wotan/behavioral_finance"
import fin "../wotan/finance"
import l "../wotan/linalg"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// CPT portfolios (TK92, reference return 0):
//   A. myopic loss aversion (Benartzi & Thaler 1995): stocks (10% a year, 20%
//      volatility) and bonds (5%, 6%), 500 simulated years; the stock share that
//      maximizes the CPT value of the returns over the evaluation period rises with the
//      period (1 < 3 < 12 < 60 months)
//   B. 11 sector ETFs, 5 years of monthly adjusted returns: the CPT portfolio against
//      the maximum Sharpe portfolio (printed)

prospect_portfolio_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Prospect Theory Portfolio Choice ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. myopic loss aversion
	MONTHS :: 6000
	logr := make([]f64, 2 * MONTHS)
	defer delete(logr)
	for t in 0 ..< MONTHS {
		logr[2 * t] =
			(0.10 - 0.5 * 0.20 * 0.20) / 12 + 0.20 / math.sqrt(12.0) * rand.float64_normal(0, 1)
		logr[2 * t + 1] =
			(0.05 - 0.5 * 0.06 * 0.06) / 12 + 0.06 / math.sqrt(12.0) * rand.float64_normal(0, 1)
	}
	horizons := []int{1, 3, 12, 60}
	shares: [4]f64
	for h, k in horizons {
		n := MONTHS / h
		R := l.matrix_new(f64, n, 2)
		defer l.matrix_free(&R)
		for p in 0 ..< n {
			for a in 0 ..< 2 {
				s := 0.0
				for t in p * h ..< (p + 1) * h {s += logr[2 * t + a]}
				R.data[p * 2 + a] = math.exp(s) - 1
			}
		}
		w := bfin.prospect_portfolio_optimize(&R, bfin.TK92_PARAMS)
		shares[k] = w[0]
		delete(w)
	}
	expect(
		&passed,
		&total,
		shares[0] < shares[1] && shares[1] < shares[2] && shares[2] < shares[3],
		fmt.tprintf(
			"myopic loss aversion: optimal stock share %.0f%% / %.0f%% / %.0f%% / %.0f%% evaluating every 1 / 3 / 12 / 60 months",
			100 * shares[0],
			100 * shares[1],
			100 * shares[2],
			100 * shares[3],
		),
	)

	// B. sector ETFs
	symbols := []string {
		"XLK",
		"XLF",
		"XLE",
		"XLV",
		"XLI",
		"XLY",
		"XLP",
		"XLB",
		"XLRE",
		"XLU",
		"XLC",
	}
	NA := len(symbols)
	panel, ok, bad := net.read_yahoo_panel(symbols, .Daily, .FiveYears)
	defer net.yahoo_panel_free(&panel)
	P, dates := panel.adj_close, panel.dates
	ok &&= len(dates) >= 600
	if !ok {
		fail_missing_data(
			#procedure,
			fmt.tprintf("Yahoo data for %s", symbols[bad]) if bad >= 0 else fmt.tprintf("600 common days of Yahoo data (got %d)", len(dates)),
		)
	} else {
		M := (len(dates) - 1) / 21
		R := l.matrix_new(f64, M, NA)
		defer l.matrix_free(&R)
		for m in 0 ..< M {
			for a in 0 ..< NA {R.data[m * NA + a] = P[(21 * (m + 1)) * NA + a] / P[(21 * m) * NA + a] - 1}
		}
		mu := fin.expected_returns_from_history(&R)
		defer delete(mu)
		cov := fin.covariance_matrix(&R)
		defer l.matrix_free(&cov)
		w_sharpe := fin.constrained_max_sharpe_portfolio(
			mu,
			&cov,
			0,
			{no_short_selling = true},
			allocator,
		)
		defer delete(w_sharpe)
		w_cpt := bfin.prospect_portfolio_optimize(&R, bfin.TK92_PARAMS)
		defer delete(w_cpt)
		ms := bfin.prospect_metrics(w_sharpe, &R, bfin.TK92_PARAMS)
		mc := bfin.prospect_metrics(w_cpt, &R, bfin.TK92_PARAMS)
		fmt.printf("%d months. %-5s | %-10s | %-10s\n", M, "ETF", "Max Sharpe", "CPT")
		for a in 0 ..< NA {fmt.printf("  %-5s | %9.1f%% | %9.1f%%\n", symbols[a], 100 * w_sharpe[a], 100 * w_cpt[a])}
		fmt.printf(
			"  monthly mean %.2f%% / %.2f%%, volatility %.2f%% / %.2f%%, P(loss) %.0f%% / %.0f%%, CPT value %.5f / %.5f\n",
			100 * ms.expected_return,
			100 * mc.expected_return,
			100 * ms.volatility,
			100 * mc.volatility,
			100 * ms.loss_probability,
			100 * mc.loss_probability,
			ms.cpt_value,
			mc.cpt_value,
		)
		expect(
			&passed,
			&total,
			mc.cpt_value >= ms.cpt_value,
			"the CPT portfolio's CPT value is at least the max-Sharpe portfolio's",
		)
	}

	fmt.printf("Prospect portfolio: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
