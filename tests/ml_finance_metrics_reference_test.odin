package tests

import ts "../wotan/analytics"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"

// Rank IC vs scipy.stats.spearmanr + pandas (alphalens conventions), trading features vs
// TA-Lib RSI and pandas rolling volume statistics, CARs vs numpy + scipy.stats.ttest_1samp,
// and the GAT cross-validation on a tracking allocator.
ml_finance_metrics_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println(
		"\n=== ML-Finance Metrics Reference Checks (scipy 1.10, pandas 1.5.3, TA-Lib 0.8.3) ===",
	)
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}
	normal :: garch_ref_normal

	{ 	// --- rank IC: 6 periods x 8 assets; ties in period 2, constant predictions in 3, a NaN in 4
		B, A :: 6, 8
		pd := l.matrix_new(f64, 1, B * A)
		yd := l.matrix_new(f64, 1, B * A)
		state: u64 = 21
		for i in 0 ..< B * A {
			p := normal(&state)
			pd.data[i] = p
			yd.data[i] = 0.3 * p + normal(&state)
		}
		for a in 0 ..< A {
			pd.data[2 * A + a] = math.round(pd.data[2 * A + a])
			pd.data[3 * A + a] = 0.25
		}
		yd.data[4 * A + 5] = math.nan_f64()
		P := t.tensor_new(pd, false)
		Y := t.tensor_new(yd, false)
		P.shape = {B, A, 1, 1}
		Y.shape = {B, A, 1, 1}
		defer {t.tensor_free(P); t.tensor_free(Y)}
		ties := []f64{1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0, -0.0}
		same := true
		for v, a in ties {same = same && pd.data[2 * A + a] == v}
		r := ml_fin.cross_sectional_rank_ic(P, Y, allocator)
		err := max(
			rel(r.mean_ic, 0.15772650931036025),
			rel(r.std_ic, 0.21416303090563202),
			rel(r.icir, 0.7364786940275433),
		)
		expect(
			&passed,
			&total,
			same && r.num_periods == 5 && r.positive_ic_ratio == 0.6 && err < 1e-14,
			fmt.tprintf(
				"rank IC: ties, constant period skipped, NaN asset dropped, n - 1 std (max rel err %.1e)",
				err,
			),
		)
	}

	{ 	// --- trading features on 120 days
		N :: 120
		p := make([]f64, N)
		v := make([]f64, N)
		v6 := make([]f64, N)
		defer {delete(p); delete(v); delete(v6)}
		state: u64 = 5
		px := 100.0
		for i in 0 ..< N {
			if i > 0 {px *= math.exp(0.02 * normal(&state))}
			p[i] = px
			v[i] = 1e6 * math.exp(0.3 * normal(&state))
			v6[i] = v[i] * 1e-6 // volume in millions
		}
		f, k := ml_fin.compute_trading_features(p, v, allocator)
		f6, _ := ml_fin.compute_trading_features(p, v6, allocator)
		defer {delete(f); delete(f6)}
		Row :: struct {
			i:              int,
			lr, rsi, z, vr: f64,
		}
		rows := []Row {
			{1, 0.027712790505641408, 0, 0, 0},
			{14, -0.007527093999660071, 0.561251883507276, 0, 0},
			{15, -0.001761003241549651, 0.5483443174070576, 0, 0},
			{
				60,
				0.011639608123704771,
				0.11160100946042477,
				0.06316478428727385,
				0.01990424360527665,
			},
			{
				119,
				-0.006938275423936183,
				0.24217404122903546,
				-0.022738724174598653,
				-0.011005096332255049,
			},
		}
		err := 0.0
		for r in rows {
			b := r.i * k
			err = max(
				err,
				rel(f[b], r.lr),
				abs(f[b + 1] - r.rsi),
				abs(f[b + 2] - r.z),
				abs(f[b + 3] - r.vr),
			)
		}
		s1, s2, s3 := 0.0, 0.0, 0.0
		for i in 0 ..< N {s1 += f[i * k + 1]; s2 += f[i * k + 2]; s3 += f[i * k + 3]}
		err = max(
			err,
			rel(s1, 29.739898610271617),
			rel(s2, 0.3056554968651706),
			rel(s3, 0.315203578570636),
		)
		expect(
			&passed,
			&total,
			k == 4 && p[119] == 168.9640556819164 && err < 1e-12,
			fmt.tprintf("features vs TA-Lib RSI(14) and pandas rolling(20) (max err %.1e)", err),
		)
		d := 0.0
		for i in 0 ..< N * k {d = max(d, abs(f[i] - f6[i]))}
		expect(
			&passed,
			&total,
			d < 1e-12,
			fmt.tprintf("volume features scale-invariant (volume in millions: max diff %.1e)", d),
		)
	}

	{ 	// --- event study: 300 days, events on rows 150, 200, 260; row 30 (estimation window
		// before the data), 297 (event window past the end) and an unknown date are not used
		N :: 300
		ret := make([]f64, N)
		names := make([]string, N)
		defer {delete(ret); delete(names)}
		state: u64 = 9
		for i in 0 ..< N {
			ret[i] = 0.0005 + 0.01 * normal(&state)
			names[i] = fmt.tprintf("D%03d", i)
		}
		for e in ([]int{150, 200, 260}) {ret[e] += 0.03; ret[e + 1] += 0.01}
		events := []string{"D150", "D200", "D260", "D030", "D297", "X"}
		r := ml_fin.compute_car(names, ret, events, -120, -10, -5, 5, allocator)
		defer ml_fin.event_study_result_free(&r, allocator)
		cars := []f64{0.027045193286241827, 0.008429695365148097, 0.01921195065341619}
		ok :=
			r.num_events == 3 &&
			len(r.car_values) == 3 &&
			len(r.dates) == 3 &&
			r.dates[2] == "D260"
		err := 0.0
		if ok {
			for c, i in cars {err = max(err, rel(r.car_values[i], c))}
			err = max(
				err,
				rel(r.mean_car, 0.018228946434935372),
				rel(r.t_statistic, 3.37806952521587),
			)
		}
		expect(
			&passed,
			&total,
			ok && ret[299] == -0.00013589855582405417 && err < 1e-12,
			fmt.tprintf(
				"CARs and t vs ttest_1samp, incomplete windows dropped (max rel err %.1e)",
				err,
			),
		)
	}

	{ 	// --- GAT cross-validation: 40 windows x 4 assets x 5 steps, on a tracking allocator
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.heap_allocator()) // frees must reach a real heap
		defer mem.tracking_allocator_destroy(&track)
		ta := mem.tracking_allocator(&track)
		context.allocator = ta
		W, A, S :: 40, 4, 5
		fd := l.matrix_new(f64, 1, W * A * S, ta)
		td := l.matrix_new(f64, 1, W * A, ta)
		state: u64 = 3
		for i in 0 ..< W * A * S {fd.data[i] = normal(&state)}
		for i in 0 ..< W * A {td.data[i] = 0.5 * fd.data[i * S + S - 1] + 0.5 * normal(&state)}
		feat := t.tensor_new(fd, false, ta)
		targ := t.tensor_new(td, false, ta)
		feat.shape = {W, A, S, 1}
		targ.shape = {W, A, 1, 1}
		res := ml_fin.gat_cross_val_score(feat, targ, 4, 4, 1, 30, 0.01, ta)
		finite := len(res) == 4
		for r in res {finite = finite && !math.is_nan(r.mean_ic) && r.val_loss < math.F64_MAX && r.early_stop_epoch >= 1}
		delete(res, ta)
		t.tensor_free(feat)
		t.tensor_free(targ)
		context.allocator = allocator
		expect(
			&passed,
			&total,
			finite && len(track.allocation_map) == 0 && len(track.bad_free_array) == 0,
			fmt.tprintf(
				"GAT CV: 4 blocked folds, %d leaked, %d bad frees (the adjacency and the validation graphs were freed twice)",
				len(track.allocation_map),
				len(track.bad_free_array),
			),
		)
	}

	{ 	// --- correlation adjacency, all 30 windows pooled (np.corrcoef): two sectors (0-2, 3-4)
		// whose shock is constant inside a window, so a single window shows only noise;
		// asset 5 is constant (no edges but its self-loop)
		B, A, S :: 30, 6, 5
		xd := l.matrix_new(f64, 1, B * A * S)
		state: u64 = 13
		for b in 0 ..< B {
			sa, sb := 0.02 * normal(&state), 0.02 * normal(&state)
			for a in 0 ..< 5 {
				for s in 0 ..< S {xd.data[(b * A + a) * S + s] = (sa if a < 3 else sb) + 0.005 * normal(&state)}
			}
		}
		X := t.tensor_new(xd, false)
		X.shape = {B, A, S, 1}
		adj := ml_fin.compute_correlation_adjacency(X, A, S, allocator)
		defer {t.tensor_free(X); t.tensor_free(adj)}
		ref := []f64 {
			0.342821611922084,
			0.3295616852352213,
			0.3276167028426948,
			0.0,
			0.0,
			0.0,
			0.32933459628682105,
			0.3425853860413822,
			0.3280800176717967,
			0.0,
			0.0,
			0.0,
			0.32802852418533657,
			0.32871892964752003,
			0.3432525461671434,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.5148473590048009,
			0.4851526409951991,
			0.0,
			0.0,
			0.0,
			0.0,
			0.4851526409951991,
			0.5148473590048009,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			1.0,
		}
		err := 0.0
		for v, i in ref {err = max(err, abs(adj.data.data[i] - v))}
		expect(
			&passed,
			&total,
			len(adj.data.data) == A * A && err < 1e-14,
			fmt.tprintf(
				"correlation adjacency over all windows vs np.corrcoef (max err %.1e)",
				err,
			),
		)
	}

	{ 	// --- split conformal: Q = k-th smallest score, k = ceil((n + 1)(1 - alpha))
		// (np.quantile(scores, k / n, method="inverted_cdf")); +Inf when k > n
		N :: 100
		y := make([]f64, N)
		yh := make([]f64, N)
		defer {delete(y); delete(yh)}
		state: u64 = 17
		for i in 0 ..< N {y[i] = normal(&state)}
		for i in 0 ..< N {yh[i] = 0.5 * normal(&state)}
		Case :: struct {
			n:     int,
			alpha: f64,
			q:     f64,
		}
		cases := []Case {
			{100, 0.05, 2.133292933842421}, // k = 96 (was the 97th: 2.2029)
			{100, 0.1, 1.8905596136607878}, // k = 91
			{19, 0.05, 2.3338264495059535}, // k = 19 = n exactly
			{19, 0.1, 1.9716490169588818},
			{10, 0.05, math.inf_f64(1)}, // k = 11 > n (was the maximum)
		}
		ok := true
		for c in cases {
			cp := ml_fin.conformal_new(allocator)
			ml_fin.conformal_calibrate(&cp, y[:c.n], yh[:c.n], c.alpha)
			ok = ok && cp.quantile == c.q
			ml_fin.conformal_free(&cp)
		}
		expect(&passed, &total, ok, "split conformal quantile (k-th smallest, +Inf for k > n)")
	}
	{ 	// --- implied correlation (CBOE formula): an index built from a constant-correlation
		// covariance gives that correlation back; a general one vs numpy
		raw := []f64{3, 1, 2, 2, 2}
		w := []f64{0.3, 0.1, 0.2, 0.2, 0.2}
		s := []f64{0.2, 0.3, 0.25, 0.4, 0.18}
		R2 := [5][5]f64 {
			{1, .6, .2, .1, .4},
			{.6, 1, .3, .2, .1},
			{.2, .3, 1, .5, .3},
			{.1, .2, .5, 1, .2},
			{.4, .1, .3, .2, 1},
		}
		index_var :: proc(
			w, s: []f64,
			R: proc(i, j: int, data: rawptr) -> f64,
			data: rawptr,
		) -> f64 {
			v := 0.0
			for i in 0 ..< len(w) {for j in 0 ..< len(w) {v += w[i] * w[j] * s[i] * s[j] * R(i, j, data)}}
			return v
		}
		const_r :: proc(i, j: int, _: rawptr) -> f64 {return i == j ? 1.0 : 0.37}
		mat_r :: proc(i, j: int, data: rawptr) -> f64 {return (^[5][5]f64)(data)[i][j]}
		vars := make([]f64, 5)
		defer delete(vars)
		for v, i in s {vars[i] = v * v}
		v1 := index_var(w, s, const_r, nil)
		v2 := index_var(w, s, mat_r, &R2)
		r1 := ml_fin.calculate_pairwise_correlation(v1, vars, raw) // raw weights are normalised
		r2 := ml_fin.calculate_pairwise_correlation(v2, vars, raw)
		civ := make([]f64, 5)
		defer delete(civ)
		for v, i in s {civ[i] = 1.05 * v}
		sig := ml_fin.compute_dispersion_signal(
			math.sqrt(1.1 * v2),
			math.sqrt(v2),
			civ,
			s,
			raw,
			0.02,
			0.05,
		)
		one := ml_fin.calculate_pairwise_correlation(0.04, vars[:1], raw[:1])
		err := max(
			rel(r1, 0.37),
			rel(r2, 0.28237608182533436),
			rel(sig.implied_corr, 0.2810802999405899),
			rel(sig.correlation_premium, -0.0012957818847444402),
			rel(sig.signal, -0.2765511759134744),
		)
		expect(
			&passed,
			&total,
			err < 1e-12 && math.is_nan(one),
			fmt.tprintf(
				"implied correlation and dispersion signal, NaN for one component (max rel err %.1e)",
				err,
			),
		)
	}
	{ 	// --- volatility ensemble helpers
		// GARCH recursion = the conditional variance of the fit it comes from
		state: u64 = 41
		NR :: 400
		r := make([]f64, NR)
		defer delete(r)
		h := 1e-4
		for i in 0 ..< NR {
			r[i] = math.sqrt(h) * normal(&state)
			h = 2e-6 + 0.08 * r[i] * r[i] + 0.9 * h
		}
		res := ts.extract_residuals(r)
		defer delete(res)
		fit := ts.garch_fit(res, .GARCH, 1, 1, 2000, 1e-6)
		defer ts.garch_result_free(&fit)
		vol := ml_fin.garch_recursive_forecast(
			res,
			fit.params.omega,
			fit.params.alpha[0],
			fit.params.beta[0],
		)
		defer delete(vol)
		gerr := 0.0
		for v, i in vol {gerr = max(gerr, rel(v * v, fit.conditional_var[i]))}
		vol_pct := make([]f64, NR) // returns in percent: same path x 100 (no absolute fallback)
		defer delete(vol_pct)
		{
			rp := make([]f64, NR)
			defer delete(rp)
			for x, i in res {rp[i] = 100 * x}
			v2 := ml_fin.garch_recursive_forecast(
				rp,
				1e4 * fit.params.omega,
				fit.params.alpha[0],
				fit.params.beta[0],
			)
			defer delete(v2)
			for v, i in v2 {gerr = max(gerr, rel(v, 100 * vol[i]))}
		}
		expect(
			&passed,
			&total,
			gerr < 1e-12,
			fmt.tprintf(
				"GARCH recursion = ts.garch_fit conditional variance, scale-free (max rel err %.1e)",
				gerr,
			),
		)

		// optimal convex weight vs scipy minimize_scalar(bounded)
		N :: 200
		y := make([]f64, N)
		g := make([]f64, N)
		lv := make([]f64, N)
		mir := make([]f64, N)
		far := make([]f64, N)
		defer {delete(y); delete(g); delete(lv); delete(mir); delete(far)}
		state = 31
		for i in 0 ..< N {y[i] = 0.15 + 0.03 * normal(&state)}
		for i in 0 ..< N {g[i] = y[i] + 0.02 * normal(&state)}
		for i in 0 ..< N {lv[i] = y[i] + 0.01 + 0.04 * normal(&state)}
		for i in 0 ..< N {mir[i] = 2 * y[i] - g[i]; far[i] = y[i] + 0.5}
		w1 := ml_fin.compute_optimal_weight(g, lv, y)
		w2 := ml_fin.compute_optimal_weight(g, mir, y)
		w3 := ml_fin.compute_optimal_weight(far, lv, y)
		expect(
			&passed,
			&total,
			rel(w1, 0.841757971498239) < 1e-10 && abs(w2 - 0.5) < 1e-12 && w3 == 0,
			fmt.tprintf(
				"optimal ensemble weight in closed form (%.12f; was a 0.01 grid), clamped at 0",
				w1,
			),
		)

		// backtest metrics vs numpy: first-day loss, n - 1 Sharpe, entries
		bt: ml_fin.VRPBacktestResult
		bt.daily_pnl = make([]f64, N)
		bt.equity_curve = make([]f64, N)
		bt.positions = make([]f64, N)
		defer {delete(bt.daily_pnl); delete(bt.equity_curve); delete(bt.positions)}
		for i in 0 ..< N {
			z := normal(&state)
			bt.positions[i] = z > 0.8 ? 1.0 : (z < -0.8 ? -1.0 : 0.0)
		}
		eq := 0.0
		for i in 0 ..< N {
			bt.daily_pnl[i] = (i > 0 ? bt.positions[i - 1] : 1.0) * 0.01 * normal(&state)
			if i == 0 {bt.daily_pnl[0] = -0.05}
			eq += bt.daily_pnl[i]
			bt.equity_curve[i] = eq
		}
		ml_fin.compute_backtest_metrics(&bt)
		berr := max(
			rel(bt.total_pnl, -0.06230283262263201),
			rel(bt.sharpe_ratio, -0.6498517367552993),
			rel(bt.max_drawdown, 0.11607941309659585),
			rel(bt.win_rate, 0.4457831325301205),
		)
		expect(
			&passed,
			&total,
			berr < 1e-12 && bt.num_trades == 67 && bt.positions[5] == -1,
			fmt.tprintf(
				"backtest metrics: drawdown incl. day 0, Sharpe (n - 1), 67 entries (max rel err %.1e)",
				berr,
			),
		)
	}
	fmt.printf("ML-finance metrics reference checks: %d/%d passed\n", passed, total)
}
