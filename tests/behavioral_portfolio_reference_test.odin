package tests

import bfin "../wotan/behavioral_finance"
import fin "../wotan/finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// Behavioral portfolios (behavioral_finance/portfolio.odin):
// - CPT portfolio: 3 assets (bond, stock, a lottery paying +25% one period in ten),
//   60 LCG periods; the CPT value against a vectorised numpy CPT, the optimum against
//   the global one from a 1/600 simplex grid refined by Nelder-Mead (TK92 with and
//   without probability weighting)
// - Mental accounts: Das, Markowitz, Scheid & Statman (2010), mu (5, 10, 25%), Sigma
//   of their eq. (4), accounts (H, alpha) = (-10%, 5%), (-5%, 15%), (-15%, 20%), wealth
//   60:20:20. Table 1 (gammas 3.7950, 2.7063, 0.8773; the text's 3.9750 is a typo),
//   weights and moments, Table 2 (the 44 shortfall probabilities, 2 decimals); long
//   only (section V) against scipy SLSQP: the third account (0, 0.0889, 0.9111), the
//   aggregate 13.30% / 19.38% (the paper prints 13.31% / 19.89%; its 13.43% frontier
//   return and 12 bp loss match sigma 19.38%), 12.4 bp below the constrained frontier.

@(private = "file")
_DM_W := [3][3]f64 {
	{0.53943223, 0.2656202, 0.19494757},
	{0.37873288, 0.34985776, 0.27140935},
	{-0.78904292, 0.96199831, 0.82704462},
}

// Table 2: thresholds -25% .. 25%; retirement, education, bequest, aggregate
@(private = "file")
_DM_T2 := [11][4]f64 {
	{0.00, 0.01, 0.15, 0.03},
	{0.01, 0.03, 0.17, 0.05},
	{0.02, 0.05, 0.20, 0.08},
	{0.05, 0.09, 0.23, 0.12},
	{0.11, 0.15, 0.26, 0.18},
	{0.20, 0.23, 0.30, 0.25},
	{0.34, 0.33, 0.33, 0.33},
	{0.49, 0.45, 0.37, 0.42},
	{0.65, 0.57, 0.41, 0.52},
	{0.79, 0.68, 0.45, 0.62},
	{0.89, 0.78, 0.49, 0.71},
}

@(private = "file")
_bp_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

behavioral_portfolio_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Behavioral Portfolio Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// --- CPT portfolio
	T :: 60
	R := l.matrix_new(f64, T, 3)
	defer l.matrix_free(&R)
	x: u64 = 2024
	for t in 0 ..< T {
		R.data[t * 3 + 0] = 0.004 + 0.01 * (_bp_u(&x) - 0.5)
		R.data[t * 3 + 1] = 0.01 + 0.12 * (_bp_u(&x) - 0.5)
		R.data[t * 3 + 2] = -0.01 + 0.03 * (_bp_u(&x) - 0.5) + (0.25 if _bp_u(&x) < 0.1 else 0)
	}
	tk := bfin.TK92_PARAMS
	flat := tk
	flat.gamma, flat.gamma_minus = 1, 1
	ew := []f64{1.0 / 3, 1.0 / 3, 1.0 / 3}
	m_tk := bfin.prospect_metrics(ew, &R, tk)
	m_flat := bfin.prospect_metrics(ew, &R, flat)
	expect(
		&passed,
		&total,
		abs(m_tk.cpt_value - 0.02003740390454819) < 1e-15 &&
		abs(m_flat.cpt_value - 0.018559424324964907) < 1e-15,
		fmt.tprintf(
			"CPT value of equal weights = numpy (TK92 %.10f, without weighting %.10f)",
			m_tk.cpt_value,
			m_flat.cpt_value,
		),
	)
	want := [2][4]f64 {
		{0.04017614059086945, 0, 3.49561014e-02, 9.65043899e-01},
		{0.024033872051852357, 0, 3.24154212e-01, 6.75845788e-01},
	}
	ev, ew_err := 0.0, 0.0
	for p, k in ([]bfin.Prospect_Params{tk, flat}) {
		w := bfin.prospect_portfolio_optimize(&R, p)
		defer delete(w)
		v := bfin.prospect_metrics(w, &R, p).cpt_value
		ev = max(ev, (want[k][0] - v) / want[k][0])
		for i in 0 ..< 3 {ew_err = max(ew_err, abs(w[i] - want[k][1 + i]))}
	}
	expect(
		&passed,
		&total,
		ev < 1e-9 && ew_err < 1e-4,
		fmt.tprintf(
			"CPT optimum = the global one (value short by %.1e relative, weights within %.1e): TK92 holds 96.5%% of the lottery, without weighting 67.6%%",
			max(ev, 0),
			ew_err,
		),
	)

	// --- mental accounts
	mu := []f64{0.05, 0.10, 0.25}
	cov := l.matrix_new(f64, 3, 3)
	defer l.matrix_free(&cov)
	copy(cov.data, []f64{0.0025, 0, 0, 0, 0.04, 0.02, 0, 0.02, 0.25})
	accounts := []bfin.Mental_Account {
		{threshold = -0.10, probability = 0.05, share = 0.6},
		{threshold = -0.05, probability = 0.15, share = 0.2},
		{threshold = -0.15, probability = 0.20, share = 0.2},
	}
	res := bfin.bpt_optimize(mu, &cov, accounts)
	defer bfin.bpt_result_free(&res)
	gammas := [3]f64{3.7950149028, 2.7062624855, 0.8772940719}
	moments := [3][2]f64 {
		{0.1022705236, 0.1229717467},
		{0.1217747588, 0.1657364192},
		{0.2635088389, 0.4913241520},
	}
	e1 := 0.0
	all_optimal := true
	for a, k in res.accounts {
		all_optimal &&= a.status == .Optimal
		e1 = max(
			e1,
			abs(a.gamma - gammas[k]) / gammas[k],
			abs(a.expected_return - moments[k][0]),
			abs(a.volatility - moments[k][1]),
		)
		for i in 0 ..< 3 {e1 = max(e1, abs(a.weights[i] - _DM_W[k][i]))}
	}
	agg_m := l.dot_simd(res.aggregate, mu)
	agg_s := math.sqrt(fin.portfolio_variance(res.aggregate, &cov))
	e1 = max(e1, abs(agg_m - 0.1384190337), abs(agg_s - 0.2031621296))
	expect(
		&passed,
		&total,
		all_optimal && e1 < 1e-8,
		fmt.tprintf(
			"Table 1: implied gammas %.4f / %.4f / %.4f, weights, moments and the 60:20:20 aggregate (%.2f%% / %.2f%%) = DMSS (max error %.1e)",
			res.accounts[0].gamma,
			res.accounts[1].gamma,
			res.accounts[2].gamma,
			100 * agg_m,
			100 * agg_s,
			e1,
		),
	)
	e2 := 0.0
	for row, h in _DM_T2 {
		H := -0.25 + 0.05 * f64(h)
		for k in 0 ..< 4 {
			w := res.aggregate if k == 3 else res.accounts[k].weights
			e2 = max(e2, abs(bfin.shortfall_probability(w, mu, &cov, H) - row[k]))
		}
	}
	// 0.0050: the aggregate at H 10% is 0.42497, printed 0.42
	expect(
		&passed,
		&total,
		e2 < 0.0051,
		fmt.tprintf(
			"Table 2: all 44 shortfall probabilities round to the printed ones (max deviation %.4f)",
			e2,
		),
	)

	// long only
	lo := bfin.bpt_optimize(mu, &cov, accounts, long_only = true)
	defer bfin.bpt_result_free(&lo)
	e3 := abs(lo.accounts[2].gamma - 0.7219589940) / 0.7219589940
	w3 := [3]f64{0, 0.08892787, 0.91107213}
	for i in 0 ..< 3 {
		e3 = max(e3, abs(lo.accounts[2].weights[i] - w3[i]))
		for k in 0 ..< 2 {e3 = max(e3, abs(lo.accounts[k].weights[i] - _DM_W[k][i]))}
	}
	lm := l.dot_simd(lo.aggregate, mu)
	ls := math.sqrt(fin.portfolio_variance(lo.aggregate, &cov))
	e3 = max(e3, abs(lm - 0.1330494298), abs(ls - 0.1938301707))
	// the constrained frontier's return at the aggregate's volatility (bisection on the
	// target return of the long-only minimum-variance portfolio)
	a, b := lm, 0.25
	for _ in 0 ..< 100 {
		mid := 0.5 * (a + b)
		wf := fin.constrained_mean_variance_portfolio(
			mu,
			&cov,
			mid,
			{no_short_selling = true},
			allocator,
		)
		if math.sqrt(fin.portfolio_variance(wf, &cov)) < ls {a = mid} else {b = mid}
		delete(wf)
	}
	loss_bp := 1e4 * (a - lm)
	expect(
		&passed,
		&total,
		e3 < 1e-6 && abs(loss_bp - 12.4189) < 0.01,
		fmt.tprintf(
			"long only: bequest account gamma %.4f, weights (0, %.4f, %.4f), aggregate %.2f%% / %.2f%%, %.1f bp below the constrained frontier = scipy (max error %.1e)",
			lo.accounts[2].gamma,
			lo.accounts[2].weights[1],
			lo.accounts[2].weights[2],
			100 * lm,
			100 * ls,
			loss_bp,
			e3,
		),
	)

	// feasibility and universes
	inf := bfin.mental_account_portfolio(mu, &cov, {threshold = 0.2, probability = 0.05})
	unb := bfin.mental_account_portfolio(mu, &cov, {threshold = -0.5, probability = 0.4})
	mx := bfin.mental_account_portfolio(
		mu,
		&cov,
		{threshold = -0.5, probability = 0.4},
		long_only = true,
	)
	uni := []bool{true, true, false}
	sub := bfin.mental_account_portfolio(
		mu,
		&cov,
		{threshold = -0.05, probability = 0.1, universe = uni},
	)
	defer {delete(inf.weights); delete(unb.weights); delete(mx.weights); delete(sub.weights)}
	// two-asset check: the bond / low-risk stock account reaches P(r < H) = alpha
	sp := bfin.shortfall_probability(sub.weights, mu, &cov, -0.05)
	expect(
		&passed,
		&total,
		inf.status == .Infeasible &&
		unb.status == .Unbounded &&
		mx.status == .Max_Return &&
		abs(mx.weights[2] - 1) < 1e-9 &&
		sub.status == .Optimal &&
		sub.weights[2] == 0 &&
		abs(sp - 0.1) < 1e-10,
		fmt.tprintf(
			"H 20%% at 5%%: infeasible; H -50%% at 40%%: unbounded with short sales, all in the 25%% asset long only; an account limited to two assets meets P(r < H) = %.6f",
			sp,
		),
	)

	fmt.printf("Behavioral portfolio reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
