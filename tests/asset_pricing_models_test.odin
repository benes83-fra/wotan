package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Latent factor models with characteristic-driven betas, out of sample:
//   A. linear betas (IPCA's own model): 200 assets, 120 dates, constant + 5
//      characteristics, K = 2. IPCA fitted on the first 80 dates recovers the span of
//      the true Gamma and gets most of the true model's out-of-sample total R².
//   B. nonlinear betas (1 + 6 (z1² - 1/12) and 12 z2 z3): IPCA, linear in z, misses them;
//      the conditional autoencoder (beta network 6 -> 32 ReLU -> 2, early stopping on
//      dates 80-99) beats it on the last 20 dates (by 8-24% over 9 seeds). Neither gets
//      near the true model: the factor network sees the linear managed portfolios, and
//      12 z2 z3 is uncorrelated with every linear characteristic, so they carry almost
//      nothing of the second factor (GKX's design, not a bug).
//   C. 24 large caps, monthly (21-day) returns, characteristics known at the start of
//      the month (12-1 momentum, 1-month reversal, 60-day volatility, log dollar
//      volume, ranked to [-0.5, 0.5]): the single IPCA factor is a market factor
//      (correlation with the equal-weighted market above 0.8; its betas rise with
//      volatility, so it is not the equal-weighted portfolio itself).

@(private = "file")
APM_N, APM_T, APM_L :: 200, 120, 6

// One synthetic panel; nonlinear selects the betas. Dates 0 ..< APM_T. Also the true
// model's fitted returns (oracle) for the R² bound.
@(private = "file")
_apm_panel :: proc(
	nonlinear: bool,
	gamma: ^[APM_L][2]f64,
) -> (
	X: l.Matrix(f64),
	y: []f64,
	times: []int,
	oracle: []f64,
) {
	n := APM_N * APM_T
	X = l.matrix_new(f64, n, APM_L)
	y = make([]f64, n)
	times = make([]int, n)
	oracle = make([]f64, n)
	for d in 0 ..< APM_T {
		f := [2]f64{rand.float64_normal(0.02, 0.08), rand.float64_normal(0.01, 0.06)}
		for i in 0 ..< APM_N {
			k := d * APM_N + i
			z := X.data[k * APM_L:][:APM_L]
			z[0] = 1
			for c in 1 ..< APM_L {z[c] = rand.float64() - 0.5}
			beta: [2]f64
			if nonlinear {
				beta = {1 + 6 * (z[1] * z[1] - 1.0 / 12), 12 * z[2] * z[3]}
			} else {
				for c in 0 ..< APM_L {beta[0] += z[c] * gamma[c][0]; beta[1] += z[c] * gamma[c][1]}
			}
			oracle[k] = beta[0] * f[0] + beta[1] * f[1]
			y[k] = oracle[k] + rand.float64_normal(0, 0.1)
			times[k] = d
		}
	}
	return
}

// Rows of dates [d0, d1), dates renumbered from 0
@(private = "file")
_apm_slice :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	d0, d1: int,
) -> (
	Xs: l.Matrix(f64),
	ys: []f64,
	ts: []int,
) {
	a, b := d0 * APM_N, d1 * APM_N
	Xs = l.Matrix(f64) {
		data = X.data[a * APM_L:b * APM_L],
		rows = b - a,
		cols = APM_L,
	}
	ys = y[a:b]
	ts = make([]int, b - a, context.temp_allocator)
	for &v, i in ts {v = i / APM_N}
	return
}

@(private = "file")
_apm_r2 :: proc(y, fit: []f64) -> f64 {
	ss, e := 0.0, 0.0
	for v, i in y {ss += v * v; e += (v - fit[i]) * (v - fit[i])}
	return 1 - e / ss
}

asset_pricing_models_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== IPCA and the conditional autoencoder, out of sample ===")
	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}

	// A. linear betas
	{
		gamma: [APM_L][2]f64
		for c in 0 ..< APM_L {gamma[c] = {rand.float64_normal(0, 1), rand.float64_normal(0, 1)}}
		gamma[0] = {1, 0} // market-like first factor
		X, y, times, oracle := _apm_panel(false, &gamma)
		defer {l.matrix_free(&X); delete(y); delete(times); delete(oracle)}
		Xa, ya, ta := _apm_slice(&X, y, 0, 80)
		Xt, yt, tt := _apm_slice(&X, y, 100, APM_T)
		m := ml_fin.ipca_fit(&Xa, ya, ta, {n_factors = 2})
		defer ml_fin.ipca_free(&m)
		// span: smallest singular value of Q' Gamma_hat (Q orthonormal basis of the true Gamma)
		q: [APM_L][2]f64
		n0, d01, n1 := 0.0, 0.0, 0.0
		for c in 0 ..< APM_L {n0 += gamma[c][0] * gamma[c][0]}
		for c in 0 ..< APM_L {q[c][0] = gamma[c][0] / math.sqrt(n0); d01 += q[c][0] * gamma[c][1]}
		for c in 0 ..< APM_L {q[c][1] = gamma[c][1] - d01 * q[c][0]; n1 += q[c][1] * q[c][1]}
		for c in 0 ..< APM_L {q[c][1] /= math.sqrt(n1)}
		B: [2][2]f64
		for i in 0 ..< 2 {for j in 0 ..< 2 {for c in 0 ..< APM_L {B[i][j] += q[c][i] * m.gamma.data[c * 2 + j]}}}
		// singular values of 2x2 B: sqrt of the eigenvalues of B'B
		a := B[0][0] * B[0][0] + B[1][0] * B[1][0]
		dd := B[0][1] * B[0][1] + B[1][1] * B[1][1]
		b := B[0][0] * B[0][1] + B[1][0] * B[1][1]
		smin := math.sqrt((a + dd) / 2 - math.sqrt((a - dd) * (a - dd) / 4 + b * b))
		tot, pred := ml_fin.ipca_r2_out_of_sample(&m, &Xt, yt, tt)
		orc := _apm_r2(yt, oracle[100 * APM_N:])
		check(
			&failures,
			smin > 0.99 && tot > 0.9 * orc,
			fmt.tprintf(
				"linear betas: IPCA's Gamma spans the true one (cosine of the largest principal angle %.4f); out-of-sample total R² %.3f of the true model's %.3f (predictive %.4f)",
				smin,
				tot,
				orc,
				pred,
			),
		)
	}

	// B. nonlinear betas
	{
		X, y, times, oracle := _apm_panel(true, nil)
		defer {l.matrix_free(&X); delete(y); delete(times); delete(oracle)}
		Xa, ya, ta := _apm_slice(&X, y, 0, 80)
		Xv, yv, tv := _apm_slice(&X, y, 80, 100)
		Xt, yt, tt := _apm_slice(&X, y, 100, APM_T)
		m := ml_fin.ipca_fit(&Xa, ya, ta, {n_factors = 2})
		defer ml_fin.ipca_free(&m)
		ip_tot, _ := ml_fin.ipca_r2_out_of_sample(&m, &Xt, yt, tt)
		train := ml_fin.ca_panel_new(&Xa, ya, ta)
		defer ml_fin.ca_panel_free(&train)
		valid := ml_fin.ca_panel_new(&Xv, yv, tv)
		defer ml_fin.ca_panel_free(&valid)
		test := ml_fin.ca_panel_new(&Xt, yt, tt)
		defer ml_fin.ca_panel_free(&test)
		ca := ml_fin.ca_new({n_chars = APM_L, n_factors = 2, hidden = []int{32}, l1 = 1e-5})
		defer ml_fin.ca_free(ca)
		fit := ml_fin.ca_fit(
			ca,
			&train,
			&valid,
			epochs = 5000,
			learning_rate = 3e-3,
			patience = 500,
		)
		lam := ml_fin.ca_mean_factor(ca, &train)
		defer delete(lam)
		ca_tot, ca_pred := ml_fin.ca_r2(ca, &test, lam)
		orc := _apm_r2(yt, oracle[100 * APM_N:])
		check(
			&failures,
			ca_tot > 1.05 * ip_tot,
			fmt.tprintf(
				"nonlinear betas, last 20 dates: conditional autoencoder total R² %.3f (predictive %.4f; %d epochs, best %d) vs IPCA %.3f; true model %.3f",
				ca_tot,
				ca_pred,
				fit.epochs_run,
				fit.best_epoch,
				ip_tot,
				orc,
			),
		)
	}

	// C. real data
	{
		tickers := []string {
			"AAPL",
			"MSFT",
			"AMZN",
			"GOOG",
			"META",
			"NVDA",
			"JPM",
			"BAC",
			"XOM",
			"CVX",
			"JNJ",
			"PFE",
			"KO",
			"PEP",
			"WMT",
			"HD",
			"DIS",
			"INTC",
			"CSCO",
			"ORCL",
			"VZ",
			"MRK",
			"UNH",
			"PG",
		}
		NA := len(tickers)
		price_maps := make([]map[w.Date]f64, NA)
		dvol_maps := make([]map[w.Date]f64, NA)
		dates := make([dynamic]w.Date)
		for sym, k in tickers {
			df := yahoo.read_yahoo(sym, .Daily, .FiveYears, allocator)
			defer w.destroy_dataframe(&df)
			if df.rows < 600 {
				fmt.printf("Not enough %s data (offline?): real-data check skipped\n", sym)
				fmt.printf("Asset pricing models: %d failure(s)\n", failures)
				record_failures(failures)
				return
			}
			price_maps[k] = make(map[w.Date]f64)
			dvol_maps[k] = make(map[w.Date]f64)
			dc, pc, vc := w.column(&df, "Date"), w.column(&df, "AdjClose"), w.column(&df, "Volume")
			for i in 0 ..< df.rows {
				d, dn := w.column_at_date(dc, i)
				p, pn := w.column_at_float(pc, i)
				v, vn := w.column_at_numeric(vc, i)
				if dn || pn || vn || p <= 0 {continue}
				price_maps[k][d] = p
				dvol_maps[k][d] = p * v
				if k == 0 {append(&dates, d)}
			}
		}
		common := make([dynamic]w.Date)
		for d in dates {
			ok := true
			for m in price_maps {ok &&= d in m}
			if ok {append(&common, d)}
		}
		D := len(common)
		P := make([]f64, D * NA)
		V := make([]f64, D * NA)
		for d, t in common {for k in 0 ..< NA {P[t * NA + k], V[t * NA + k] = price_maps[k][d], dvol_maps[k][d]}}
		// months of 21 days; characteristics at the start s of each month from data up to s
		M :: 21
		rows := make([dynamic]f64)
		ys := make([dynamic]f64)
		ts := make([dynamic]int)
		mkt := make([dynamic]f64)
		raw := make([]f64, NA * 4)
		month := 0
		for s := 252; s + M < D; s += M {
			for k in 0 ..< NA {
				p := proc(P: []f64, t, k, NA: int) -> f64 {return P[t * NA + k]}
				mom := p(P, s - M, k, NA) / p(P, s - 252, k, NA) - 1
				rev := p(P, s, k, NA) / p(P, s - M, k, NA) - 1
				ss, dv := 0.0, 0.0
				for t in s - 59 ..= s {
					r := math.ln(p(P, t, k, NA) / p(P, t - 1, k, NA))
					ss += r * r
				}
				for t in s - 19 ..= s {dv += V[t * NA + k] / 20}
				raw[k * 4 + 0], raw[k * 4 + 1], raw[k * 4 + 2], raw[k * 4 + 3] =
					mom, rev, math.sqrt(ss / 60), math.ln(dv)
			}
			// cross-sectional ranks to [-0.5, 0.5]
			ranks := make([]f64, NA * 4, context.temp_allocator)
			for c in 0 ..< 4 {
				for k in 0 ..< NA {
					below := 0
					for j in 0 ..< NA {if raw[j * 4 + c] < raw[k * 4 + c] {below += 1}}
					ranks[k * 4 + c] = f64(below) / f64(NA - 1) - 0.5
				}
			}
			mr := 0.0
			for k in 0 ..< NA {
				append(
					&rows,
					1,
					ranks[k * 4],
					ranks[k * 4 + 1],
					ranks[k * 4 + 2],
					ranks[k * 4 + 3],
				)
				r := P[(s + M) * NA + k] / P[s * NA + k] - 1
				append(&ys, r)
				append(&ts, month)
				mr += r / f64(NA)
			}
			append(&mkt, mr)
			month += 1
		}
		X := l.Matrix(f64) {
			data = rows[:],
			rows = len(ys),
			cols = 5,
		}
		m := ml_fin.ipca_fit(&X, ys[:], ts[:], {n_factors = 1})
		defer ml_fin.ipca_free(&m)
		tot, pred := ml_fin.ipca_r2(&m, &X, ys[:], ts[:])
		corr := fin.correlation(m.factors.data[:month], mkt[:])
		check(
			&failures,
			corr > 0.8 && tot > 0.1,
			fmt.tprintf(
				"%d large caps, %d months: the single IPCA factor correlates %.3f with the equal-weighted market; total R² %.3f, predictive %.4f; Gamma (const, mom, rev, vol, size) %.2f %.2f %.2f %.2f %.2f",
				NA,
				month,
				corr,
				tot,
				pred,
				m.gamma.data[0],
				m.gamma.data[1],
				m.gamma.data[2],
				m.gamma.data[3],
				m.gamma.data[4],
			),
		)
	}
	fmt.printf("Asset pricing models: %d failure(s)\n", failures)
	record_failures(failures)
}
