package tests

import bfin "../wotan/behavioral_finance"
import fin "../wotan/finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// Behavioral portfolio theory with mental accounts (Das, Markowitz, Scheid & Statman
// 2010) on 9 ETFs (bonds, gold, utilities, the market, sectors), 5 years of monthly
// adjusted returns, annualized (x 12), long only (with short sales the in-sample
// frontier is so steep that every account's expected return is unbounded; the
// short-sale case is checked against DMSS in behavioral_portfolio_reference_test).
// Accounts: safety (H -10%, alpha 5%, 40% of wealth), aspiration (-20%, 10%, 50%),
// speculative (-35%, 20%, 10%).
//   bpt_test     every asset in every account
//   bpt_v2_test  each account restricted to its own assets (safety: TLT, GLD, XLU;
//                aspiration: SPY, XLF, XLI, XLK; speculative: XLK, XLE, XLY)
// Checked: the statuses agree with the shortfall probabilities, the weights are
// nonnegative and inside the universes, the aggregate is not above the long-only
// frontier (its distance printed); the portfolios are printed.

@(private = "file")
_BPT_SYMBOLS := []string{"TLT", "GLD", "XLU", "SPY", "XLF", "XLI", "XLK", "XLE", "XLY"}

@(private = "file")
_bpt_inputs :: proc() -> (mu: []f64, cov: l.Matrix(f64), ok: bool) {
	NA := len(_BPT_SYMBOLS)
	P, _, dates, got := yahoo_adjclose_panel(_BPT_SYMBOLS, .FiveYears, 600)
	if !got {return}
	M := (len(dates) - 1) / 21
	R := l.matrix_new(f64, M, NA)
	defer l.matrix_free(&R)
	for m in 0 ..< M {
		for a in 0 ..< NA {R.data[m * NA + a] = P[(21 * (m + 1)) * NA + a] / P[(21 * m) * NA + a] - 1}
	}
	mu = fin.expected_returns_from_history(&R)
	cov = fin.covariance_matrix(&R)
	for &v in mu {v *= 12}
	for &v in cov.data {v *= 12}
	return mu, cov, true
}

@(private = "file")
_bpt_print :: proc(res: ^bfin.BPT_Result, mu: []f64, cov: ^l.Matrix(f64)) {
	fmt.printf(
		"%-5s | %8s | %10s | %11s | %9s\n",
		"Asset",
		"Safety",
		"Aspiration",
		"Speculative",
		"Aggregate",
	)
	for s, a in _BPT_SYMBOLS {
		fmt.printf(
			"%-5s | %7.1f%% | %9.1f%% | %10.1f%% | %8.1f%%\n",
			s,
			100 * res.accounts[0].weights[a],
			100 * res.accounts[1].weights[a],
			100 * res.accounts[2].weights[a],
			100 * res.aggregate[a],
		)
	}
	for acc, k in res.accounts {
		fmt.printf(
			"  account %d: %v, implied gamma %.3f, mean %.2f%%, volatility %.2f%%\n",
			k,
			acc.status,
			acc.gamma,
			100 * acc.expected_return,
			100 * acc.volatility,
		)
	}
	fmt.printf(
		"  aggregate: mean %.2f%%, volatility %.2f%%\n",
		100 * l.dot_simd(res.aggregate, mu),
		100 * math.sqrt(fin.portfolio_variance(res.aggregate, cov)),
	)
}

@(private = "file")
_bpt_accounts :: proc(universes: [3][]bool = {}) -> [3]bfin.Mental_Account {
	return {
		{threshold = -0.10, probability = 0.05, share = 0.4, universe = universes[0]},
		{threshold = -0.20, probability = 0.10, share = 0.5, universe = universes[1]},
		{threshold = -0.35, probability = 0.20, share = 0.1, universe = universes[2]},
	}
}

// Every account's result agrees with its status: Optimal meets P(r < H) = alpha,
// Max_Return stays within alpha, Infeasible (the closest portfolio) misses it; the
// weights are nonnegative and inside the account's universe
@(private = "file")
_bpt_check :: proc(
	res: ^bfin.BPT_Result,
	accounts: []bfin.Mental_Account,
	mu: []f64,
	cov: ^l.Matrix(f64),
) -> int {
	consistent, in_universe := true, true
	for acc, k in res.accounts {
		a := accounts[k]
		p := bfin.shortfall_probability(acc.weights, mu, cov, a.threshold)
		switch acc.status {
		case .Optimal:
			consistent &&= abs(p - a.probability) < 1e-6
		case .Max_Return:
			consistent &&= p <= a.probability + 1e-9
		case .Infeasible:
			consistent &&= p > a.probability
		case .Unbounded:
			consistent = false // long only
		}
		for v, i in acc.weights {in_universe &&= v >= 0 && (a.universe == nil || a.universe[i] || v == 0)}
	}
	// the aggregate against the long-only frontier at its volatility (DMSS section V)
	lm := l.dot_simd(res.aggregate, mu)
	ls := math.sqrt(fin.portfolio_variance(res.aggregate, cov))
	lo, hi := lm, lm
	for m in mu {hi = max(hi, m)}
	for _ in 0 ..< 100 {
		mid := 0.5 * (lo + hi)
		wf := fin.constrained_mean_variance_portfolio(
			mu,
			cov,
			mid,
			{no_short_selling = true},
			context.allocator,
		)
		if math.sqrt(fin.portfolio_variance(wf, cov)) <= ls {lo = mid} else {hi = mid}
		delete(wf)
	}
	ok := consistent && in_universe && lo - lm >= -1e-9
	fmt.printf(
		"%s statuses consistent with P(r < H) (Optimal = alpha, Max_Return <= alpha, Infeasible > alpha), weights nonnegative and in each universe; the aggregate %.1f bp below the long-only frontier\n",
		"OK  " if ok else "FAIL",
		1e4 * (lo - lm),
	)
	return 0 if ok else 1
}

bpt_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Behavioral Portfolio Theory: Mental Accounts (long only) ===")
	context.allocator = allocator
	mu, cov, ok := _bpt_inputs()
	if !ok {
		fmt.println("Not enough data (offline?): skipped")
		return
	}
	defer {delete(mu); l.matrix_free(&cov)}
	accounts := _bpt_accounts()
	res := bfin.bpt_optimize(mu, &cov, accounts[:], long_only = true)
	defer bfin.bpt_result_free(&res)
	_bpt_print(&res, mu, &cov)
	record_failures(_bpt_check(&res, accounts[:], mu, &cov))
}

bpt_v2_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Behavioral Portfolio Theory: Mental Accounts with Mandates (long only) ===")
	context.allocator = allocator
	mu, cov, ok := _bpt_inputs()
	if !ok {
		fmt.println("Not enough data (offline?): skipped")
		return
	}
	defer {delete(mu); l.matrix_free(&cov)}
	safety := []bool{true, true, true, false, false, false, false, false, false}
	aspiration := []bool{false, false, false, true, true, true, true, false, false}
	speculative := []bool{false, false, false, false, false, false, true, true, true}
	accounts := _bpt_accounts({safety, aspiration, speculative})
	res := bfin.bpt_optimize(mu, &cov, accounts[:], long_only = true)
	defer bfin.bpt_result_free(&res)
	_bpt_print(&res, mu, &cov)
	record_failures(_bpt_check(&res, accounts[:], mu, &cov))
}
