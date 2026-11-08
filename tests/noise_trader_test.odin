package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Noise trader risk: De Long, Shleifer, Summers & Waldmann (1990),
// behavioral_finance/noise_trader.odin
//   noise_trader_reference_test: the equilibrium conditions from first principles (the
//     next price's mean and variance taken from the price function itself, demands
//     clear the market and are the CARA optima); price level, slope and expected gap =
//     a sympy solution of the model by undetermined coefficients (scratch, three
//     parameter sets incl. AR(1) misperceptions and dividend risk); no noise -> p = 1;
//     eq. 21's mu* and Appendix I's mu_L, mu_H are zeros of eq. 18 / A5 with the
//     paper's sign pattern; Monte Carlo of the basic and the persistent model: mean and
//     variance of the price (eqs. 24, 11, 22) and the mean realized return gap (eqs.
//     18, 23) within 4 standard errors
//     On 10 seeds: within 2.8 se.
//   noise_trader_test: A. mean reversion (Section III) with persistent (phi 0.9) and
//     iid misperceptions plus dividend risk, 5000 periods: the slope of next-period
//     excess returns on the price = the model's -(1 + r - phi) and the lag-1
//     autocorrelation of price changes = -(1 - phi) / 2, within 4 Newey-West se
//     (on 9 seeds: slopes within 2.1 se, autocorrelations within 3.2 se). B. survival
//     with imitation (eq. 20): without fundamental risk the noise traders die out from
//     below mu* and take over from above (their Fig. 1); with dividend risk the share
//     settles at mu_L from below it and from between the roots (0.0611 - 0.0632 on 9
//     seeds, mu_L 0.0615) and takes over from above mu_H (Appendix I). The parameters
//     are ours: the paper gives no calibration.

NT_SET1 :: bfin.Noise_Trader_Params{0.05, 90, 0.3, 0.2, 0.05, 0, 0}
NT_SET2 :: bfin.Noise_Trader_Params{0.05, 90, 0.3, 0.2, 0.05, 0, 0.008}
NT_SET3 :: bfin.Noise_Trader_Params{0.03, 40, 0.25, 0.1, 0.08, 0.8, 0.02}

// mean and its standard error from means of `block`-long batches (serial dependence)
@(private = "file")
_nt_batch_mean :: proc(x: []f64, block: int) -> (mean, se: f64) {
	nb := len(x) / block
	bm := make([]f64, nb)
	defer delete(bm)
	for b in 0 ..< nb {
		for v in x[b * block:(b + 1) * block] {bm[b] += v / f64(block)}
		mean += bm[b] / f64(nb)
	}
	for v in bm {se += (v - mean) * (v - mean) / f64(nb - 1)}
	return mean, math.sqrt(se / f64(nb))
}

// misperceptions rho* + AR(1) deviations started from their stationary distribution
@(private = "file")
_nt_rho :: proc(T: int, p: bfin.Noise_Trader_Params) -> []f64 {
	rho := make([]f64, T)
	d := rand.float64_normal(0, p.sigma / math.sqrt(1 - p.phi * p.phi))
	for t in 0 ..< T {
		if t > 0 {d = p.phi * d + rand.float64_normal(0, p.sigma)}
		rho[t] = p.rho_star + d
	}
	return rho
}

noise_trader_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Noise Trader Risk (DSSW 1990) Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	sets := [3]bfin.Noise_Trader_Params{NT_SET1, NT_SET2, NT_SET3}

	// 1. equilibrium from first principles: for this generation's rho_t, the next
	// price is p(rho*+ phi (rho_t - rho*) +- sigma) with equal odds (exact for the
	// mean and variance of a linear price); with V = sigma_eps^2 + its variance the
	// CARA demands (r + E p' - (1 + r) p) / (2 gamma V) (+ rho / (2 gamma V)) must
	// equal dssw_demands and clear the market
	{
		worst := 0.0
		for p in sets {
			for k in 0 ..< 7 {
				rho := p.rho_star + p.sigma * (f64(k) - 3)
				pt := bfin.dssw_price(rho, p)
				m := p.rho_star + p.phi * (rho - p.rho_star)
				up, dn := bfin.dssw_price(m + p.sigma, p), bfin.dssw_price(m - p.sigma, p)
				Ep, Vp := 0.5 * (up + dn), 0.25 * (up - dn) * (up - dn)
				V := p.sigma_eps * p.sigma_eps + Vp
				li := (p.r + Ep - (1 + p.r) * pt) / (2 * p.gamma * V)
				ln := li + rho / (2 * p.gamma * V)
				si, sn := bfin.dssw_demands(rho, p)
				worst = max(
					worst,
					abs(si - li) / abs(li),
					abs(sn - ln) / abs(ln),
					abs((1 - p.mu) * li + p.mu * ln - 1),
				)
				worst = max(
					worst,
					abs(bfin.dssw_expected_excess_return(rho, p) - (p.r + Ep - (1 + p.r) * pt)),
				)
				// E_t dR = (lambda_n - lambda_i) E_t excess
				worst = max(
					worst,
					abs(
						bfin.dssw_conditional_return_gap(rho, p) -
						(ln - li) * (p.r + Ep - (1 + p.r) * pt),
					),
				)
			}
		}
		expect(
			&passed,
			&total,
			worst < 1e-12,
			fmt.tprintf(
				"equilibrium: demands are the CARA optima for the next price's mean and variance, clear the market, E_t gap = demand gap x expected excess (3 sets x 7 rho), max error %.1e",
				worst,
			),
		)
	}

	// 2. sympy: price at rho = rho* (= E p), slope, E dR, and the zeros of E dR in mu
	{
		WANT_P := [3]f64{1.4653061224489792, 1.2349061224489795, -19.397164461247648}
		WANT_B := [3]f64{0.2857142857142857, 0.2857142857142857, 1.0869565217391306}
		WANT_G := [3]f64{-0.14708333333333337, -0.064223000406008965, 0.089096747343316351}
		err := 0.0
		for p, i in sets {
			err = max(
				err,
				abs(bfin.dssw_mean_price(p) - WANT_P[i]),
				abs(bfin.dssw_price(p.rho_star, p) - WANT_P[i]),
			)
			err = max(
				err,
				abs(bfin.dssw_price_slope(p) - WANT_B[i]),
				abs(bfin.dssw_expected_return_gap(p) - WANT_G[i]),
			)
		}
		lo1, hi1, n1 := bfin.dssw_critical_shares(NT_SET1)
		lo2, hi2, n2 := bfin.dssw_critical_shares(NT_SET2)
		_, _, n3 := bfin.dssw_critical_shares(NT_SET3)
		ok := n1 == 2 && lo1 == 0 && abs(hi1 - 0.5206250000000001) < 1e-14
		ok &&=
			n2 == 2 &&
			abs(lo2 - 0.06146935162357237) < 1e-14 &&
			abs(hi2 - 0.45915564837642764) < 1e-14 &&
			n3 == 0
		expect(
			&passed,
			&total,
			ok && err < 1e-12,
			fmt.tprintf(
				"= sympy (undetermined coefficients): E p, slope, E gap on 3 sets within %.1e; zeros of E gap in mu: 0, %.6f (eq. 21); %.6f, %.6f (Appendix I); none with phi 0.8",
				err,
				hi1,
				lo2,
				hi2,
			),
		)
	}

	// 3. the paper's statements: no noise -> price 1; eq. 21 by hand; the sign of E dR
	// below / between / above the zeros (A: noise traders shrink below mu*, grow
	// above; with fundamental risk they grow below mu_L, shrink between, grow above
	// mu_H); mean price below 1 whenever noise traders earn more (eq. 24's discussion)
	{
		q := NT_SET1
		q.rho_star, q.sigma = 0, 0
		ok := bfin.dssw_price(0, q) == 1 && bfin.dssw_mean_price(q) == 1
		q = NT_SET1
		q.mu = 0
		ok &&= bfin.dssw_price(0.7, q) == 1
		p := NT_SET1
		mstar :=
			(1 + p.r) *
			(1 + p.r) *
			(p.rho_star * p.rho_star + p.sigma * p.sigma) /
			(2 * p.gamma * p.rho_star * p.sigma * p.sigma)
		_, hi1, _ := bfin.dssw_critical_shares(p)
		ok &&= abs(hi1 - mstar) < 1e-15
		sign_at :: proc(p: bfin.Noise_Trader_Params, mu: f64) -> f64 {
			q := p
			q.mu = mu
			return bfin.dssw_expected_return_gap(q)
		}
		ok &&= sign_at(p, 0.5 * mstar) < 0 && sign_at(p, 0.5 * (1 + mstar)) > 0
		lo2, hi2, _ := bfin.dssw_critical_shares(NT_SET2)
		ok &&=
			sign_at(NT_SET2, 0.5 * lo2) > 0 &&
			sign_at(NT_SET2, 0.5 * (lo2 + hi2)) < 0 &&
			sign_at(NT_SET2, 0.5 * (hi2 + 1)) > 0
		ok &&= sign_at(NT_SET2, 1e-9) > 0 // small populations always earn more with fundamental risk
		// whenever E gap > 0 the mean price is below 1 (scan the shares)
		below := true
		for s in 1 ..= 99 {
			for ps in ([2]bfin.Noise_Trader_Params{NT_SET1, NT_SET2}) {
				q := ps
				q.mu = f64(s) / 100
				if bfin.dssw_expected_return_gap(q) > 0 {below &&= bfin.dssw_mean_price(q) < 1}
			}
		}
		expect(
			&passed,
			&total,
			ok && below,
			fmt.tprintf(
				"no noise -> p = 1; mu* = (1+r)^2 (rho*^2 + s^2) / (2 gamma rho* s^2) = %.4f; E gap < 0 below mu*, > 0 above; with dividend risk + / - / + around mu_L, mu_H; E p < 1 wherever noise traders earn more",
				mstar,
			),
		)
	}

	// 4. Monte Carlo: the basic model (set 1, iid) and the persistent one (set 3,
	// phi 0.8, dividend risk), 10^6 generations: mean and variance of the price, the
	// mean realized gap (batch means, 1000-period batches)
	{
		T :: 1_000_000
		zs: [2][3]f64 // [set][z of mean price, z of price variance, z of gap]
		gaps: [2][2]f64
		for p, k in ([2]bfin.Noise_Trader_Params{NT_SET1, NT_SET3}) {
			rho := _nt_rho(T, p)
			eps := make([]f64, T)
			defer {delete(rho); delete(eps)}
			for &e in eps {e = rand.float64_normal(0, p.sigma_eps)}
			path := bfin.dssw_simulate(rho, eps, p)
			defer bfin.dssw_path_free(&path)
			mp, sp := _nt_batch_mean(path.price, 1000)
			Ep := bfin.dssw_mean_price(p)
			dev := make([]f64, T)
			defer delete(dev)
			for v, t in path.price {dev[t] = (v - Ep) * (v - Ep)}
			mv, sv := _nt_batch_mean(dev, 1000)
			mg, sg := _nt_batch_mean(path.gap[:T - 1], 1000)
			zs[k] = {
				abs(mp - Ep) / sp,
				abs(mv - bfin.dssw_price_variance(p)) / sv,
				abs(mg - bfin.dssw_expected_return_gap(p)) / sg,
			}
			gaps[k] = {mg, bfin.dssw_expected_return_gap(p)}
		}
		worst := 0.0
		for z in zs {for v in z {worst = max(worst, v)}}
		expect(
			&passed,
			&total,
			worst < 4,
			fmt.tprintf(
				"Monte Carlo, 10^6 generations: mean price, price variance and mean gap within %.1f se (basic: gap %.4f vs eq. 18 %.4f; persistent with dividend risk: %.4f vs %.4f)",
				worst,
				gaps[0][0],
				gaps[0][1],
				gaps[1][0],
				gaps[1][1],
			),
		)
	}

	fmt.printf("Noise trader reference: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// OLS of y on a constant and x: slope and its Newey-West t (and se)
@(private = "file")
_nt_slope :: proc(x, y: []f64, lags: int) -> (slope, se: f64) {
	n := len(x)
	X := l.matrix_new(f64, n, 2)
	defer l.matrix_free(&X)
	for i in 0 ..< n {X.data[2 * i], X.data[2 * i + 1] = 1, x[i]}
	fit := ml.ols_fit_robust(&X, y, {type = .HAC, maxlags = lags}, .QR, context.temp_allocator)
	defer ml._ols_result_free(&fit, context.temp_allocator)
	return fit.beta[1], fit.stderr[1]
}

noise_trader_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Noise Trader Risk: Mean Reversion and Survival ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. mean reversion (Section III): misperceptions persistent (phi 0.9) or iid, with
	// dividend risk. The price has no fundamental news yet moves; high prices predict
	// low excess returns with slope -(1 + r - phi) (E_t excess = 2 gamma V - mu rho_t,
	// p_t linear in rho_t), and price changes are negatively autocorrelated,
	// corr(dp_t, dp_{t-1}) = -(1 - phi) / 2. 5000 periods each.
	{
		T :: 5000
		res: [2][4]f64 // [phi 0.9, 0][slope, its se, lag-1 autocorrelation of dp, its se]
		for phi, k in ([2]f64{0.9, 0}) {
			p := bfin.Noise_Trader_Params{0.05, 5, 0.2, 0.05, 0.02, phi, 0.02}
			rho := _nt_rho(T, p)
			eps := make([]f64, T)
			defer {delete(rho); delete(eps)}
			for &e in eps {e = rand.float64_normal(0, p.sigma_eps)}
			path := bfin.dssw_simulate(rho, eps, p)
			defer bfin.dssw_path_free(&path)
			res[k][0], res[k][1] = _nt_slope(path.price[:T - 1], path.excess[:T - 1], 6)
			// lag-1 autocorrelation of price changes (mean 0), delta-method se via a
			// regression of dp_t on dp_{t-1}
			dp := make([]f64, T - 1)
			defer delete(dp)
			for t in 1 ..< T {dp[t - 1] = path.price[t] - path.price[t - 1]}
			res[k][2], res[k][3] = _nt_slope(dp[:T - 2], dp[1:], 6)
		}
		ok := true
		for phi, k in ([2]f64{0.9, 0}) {
			ok &&= abs(res[k][0] + (1.05 - phi)) < 4 * res[k][1] && res[k][0] / res[k][1] < -3
			ok &&= abs(res[k][2] + (1 - phi) / 2) < 4 * res[k][3]
		}
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"mean reversion: excess return on price, slope %.3f (se %.3f; model -0.15) with phi 0.9, %.3f (%.3f; -1.05) iid; dp autocorrelation %.3f (%.3f; -0.05), %.3f (%.3f; -0.5)",
				res[0][0],
				res[0][1],
				res[1][0],
				res[1][1],
				res[0][2],
				res[0][3],
				res[1][2],
				res[1][3],
			),
		)
	}

	// B. survival with imitation (eq. 20, alpha 0.002, 50,000 generations): without
	// fundamental risk noise traders die out from below mu* and take over from above
	// (their Fig. 1); with dividend risk the share settles at mu_L from below it and
	// from between mu_L and mu_H, and takes over from above mu_H (Appendix I). Settled:
	// the mean share over the last 10,000 generations within 0.01 of mu_L.
	{
		T, ALPHA :: 50_000, 0.002
		run :: proc(p: bfin.Noise_Trader_Params, mu0: f64) -> (final, tail_mean: f64) {
			q := p
			q.mu = mu0
			rho := _nt_rho(T, q)
			eps := make([]f64, T)
			defer {delete(rho); delete(eps)}
			for &e in eps {e = rand.float64_normal(0, q.sigma_eps)}
			path := bfin.dssw_simulate(rho, eps, q, ALPHA)
			defer bfin.dssw_path_free(&path)
			for v in path.mu[T - 10_000:] {tail_mean += v / 10_000}
			return path.mu[T - 1], tail_mean
		}
		_, mstar, _ := bfin.dssw_critical_shares(NT_SET1)
		lo, hi, _ := bfin.dssw_critical_shares(NT_SET2)
		a, _ := run(NT_SET1, mstar - 0.07)
		b, _ := run(NT_SET1, mstar + 0.07)
		_, c := run(NT_SET2, 0.5 * lo)
		_, d := run(NT_SET2, 0.5 * (lo + hi))
		e, _ := run(NT_SET2, hi + 0.07)
		expect(
			&passed,
			&total,
			a == 0 && b == 1 && abs(c - lo) < 0.01 && abs(d - lo) < 0.01 && e == 1,
			fmt.tprintf(
				"survival: no dividend risk, from mu* -+ 0.07 (mu* %.3f) the share ends at %.2f / %.2f; with dividend risk (mu_L %.4f, mu_H %.4f) it settles at %.4f from below and %.4f from between, and ends at %.2f from above mu_H",
				mstar,
				a,
				b,
				lo,
				hi,
				c,
				d,
				e,
			),
		)
	}

	fmt.printf("Noise trader: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
