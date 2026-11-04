package tests

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
	fmt.printf("ML-finance metrics reference checks: %d/%d passed\n", passed, total)
}
