package tests

import ml_finance "../wotan/ml_finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// TimeGAN judged on what it is for, the temporal structure: lag-1 auto- and
// cross-correlations, the marginal moments, and train-on-synthetic / test-on-real (a
// linear one-step predictor fitted on synthetic sequences, scored on real ones), next to a
// baseline with the right marginals and no dynamics (each value drawn at random from the
// real data). The old test compared the mean and std of all features pooled and only
// printed a warning.

// Statistics of sequences [n, T, F]: per feature mean and std; lag-1 autocorrelation of
// feature a; lag-1 cross-correlation corr(x_b[t], x_a[t-1])
@(private = "file")
TGStats :: struct {
	mean, std:    [2]f64,
	acf, cross:   f64,
	fit_a, fit_b: f64, // x_a[t] ~ fit_a + fit_b x_a[t-1]
}

@(private = "file")
_tg_stats :: proc(seqs: []f64, T, F: int, a := 0, b := 1, abs_a := false) -> (s: TGStats) {
	n := len(seqs) / (T * F)
	get :: proc(seqs: []f64, T, F, i, t, f: int, absolute: bool) -> f64 {
		v := seqs[(i * T + t) * F + f]
		return abs(v) if absolute else v
	}
	for f in 0 ..< 2 {
		m, m2 := 0.0, 0.0
		for i in 0 ..< n {for t in 0 ..< T {v := get(seqs, T, F, i, t, f, false); m += v; m2 += v * v}}
		k := f64(n * T)
		s.mean[f], s.std[f] = m / k, math.sqrt(max(m2 / k - (m / k) * (m / k), 0))
	}
	corr :: proc(x, y: []f64) -> f64 {
		mx, my := 0.0, 0.0
		for v, i in x {mx += v; my += y[i]}
		mx, my = mx / f64(len(x)), my / f64(len(x))
		sxy, sxx, syy := 0.0, 0.0, 0.0
		for v, i in x {sxy += (v - mx) * (y[i] - my); sxx += (v - mx) * (v - mx); syy += (y[i] - my) * (y[i] - my)}
		return sxy / math.sqrt(sxx * syy) if sxx > 0 && syy > 0 else 0
	}
	prev, cur, other :=
		make([]f64, n * (T - 1)), make([]f64, n * (T - 1)), make([]f64, n * (T - 1))
	defer {delete(prev); delete(cur); delete(other)}
	for i in 0 ..< n {
		for t in 1 ..< T {
			k := i * (T - 1) + t - 1
			prev[k] = get(seqs, T, F, i, t - 1, a, abs_a)
			cur[k] = get(seqs, T, F, i, t, a, abs_a)
			other[k] = get(seqs, T, F, i, t, b, false)
		}
	}
	s.acf, s.cross = corr(prev, cur), corr(prev, other)
	mp, mc, sxy, sxx := 0.0, 0.0, 0.0, 0.0
	for v, i in prev {mp += v; mc += cur[i]}
	mp, mc = mp / f64(len(prev)), mc / f64(len(prev))
	for v, i in prev {sxy += (v - mp) * (cur[i] - mc); sxx += (v - mp) * (v - mp)}
	s.fit_b = sxy / sxx if sxx > 0 else 0
	s.fit_a = mc - s.fit_b * mp
	return
}

// Mean squared one-step error on real sequences of the predictor fitted elsewhere
@(private = "file")
_tg_tstr :: proc(real: []f64, T, F: int, fit: TGStats, a := 0, abs_a := false) -> f64 {
	n := len(real) / (T * F)
	se := 0.0
	for i in 0 ..< n {
		for t in 1 ..< T {
			p, c := real[(i * T + t - 1) * F + a], real[(i * T + t) * F + a]
			if abs_a {p, c = abs(p), abs(c)}
			e := c - (fit.fit_a + fit.fit_b * p)
			se += e * e
		}
	}
	return se / f64(n * (T - 1))
}

// The no-dynamics baseline: every value drawn from the real values of its feature
@(private = "file")
_tg_shuffled :: proc(real: []f64, n_out, T, F: int) -> []f64 {
	n := len(real) / (T * F)
	out := make([]f64, n_out * T * F)
	for k in 0 ..< len(out) {
		f := k % F
		out[k] = real[(rand.int_max(n) * T + rand.int_max(T)) * F + f]
	}
	return out
}

// Trains in the three phases; returns synthetic sequences in the data's units
@(private = "file")
_tg_train :: proc(
	data: []f64,
	T, F: int,
	iters: [3]int,
	n_out: int,
	allocator: mem.Allocator,
) -> []f64 {
	scaled := make([]f64, len(data))
	copy(scaled, data)
	sc := ml_finance.timegan_scaler_fit(data, F, allocator)
	ml_finance.timegan_scale(scaled, sc)
	tg := ml_finance.new_timegan(T, F, 8, 24, 1e-3, allocator)
	defer ml_finance.timegan_free(tg)
	ml_finance.timegan_init_weights(tg)
	BATCH :: 64
	// the variance of the scaled data, to judge the reconstruction error by
	sm, sm2 := 0.0, 0.0
	for v in scaled {sm += v; sm2 += v * v}
	var_scaled := sm2 / f64(len(scaled)) - (sm / f64(len(scaled))) * (sm / f64(len(scaled)))
	for it in 0 ..< iters[0] {
		loss := ml_finance.train_embedder_recovery(tg, scaled, BATCH)
		if it % (iters[0] / 2) == 0 || it == iters[0] - 1 {
			fmt.printf(
				"   embedding  %5d  reconstruction MSE %.5f (%.1f%% of the variance)\n",
				it,
				loss,
				100 * loss / var_scaled,
			)
		}
	}
	for it in 0 ..< iters[1] {
		loss := ml_finance.train_supervisor(tg, scaled, BATCH)
		if it % (iters[1] / 2) == 0 ||
		   it ==
			   iters[1] -
				   1 {fmt.printf("   supervised %5d  next-step latent MSE %.6f\n", it, loss)}
	}
	for it in 0 ..< iters[2] {
		g, e, d := ml_finance.timegan_train_joint_step(tg, scaled, BATCH)
		if it % (iters[2] / 4) == 0 ||
		   it ==
			   iters[2] -
				   1 {fmt.printf("   joint      %5d  G %.4f  E %.4f  D %.4f\n", it, g, e, d)}
	}
	synth := ml_finance.generate_synthetic_data(tg, n_out, allocator)
	ml_finance.timegan_unscale(synth, sc)
	return synth
}

@(private = "file")
_tg_report :: proc(
	names: [2]string,
	lag_name: string,
	real, synth, shuffled: TGStats,
	tstr: [3]f64,
) {
	fmt.printf("   %-26s %10s %10s %10s\n", "", "real", "TimeGAN", "shuffled")
	row :: proc(name: string, a, b, c: f64) {fmt.printf(
			"   %-26s %10.4f %10.4f %10.4f\n",
			name,
			a,
			b,
			c,
		)}
	for f in 0 ..< 2 {
		row(fmt.tprintf("mean %s", names[f]), real.mean[f], synth.mean[f], shuffled.mean[f])
		row(fmt.tprintf("std %s", names[f]), real.std[f], synth.std[f], shuffled.std[f])
	}
	row(fmt.tprintf("lag-1 autocorr %s", lag_name), real.acf, synth.acf, shuffled.acf)
	row(
		fmt.tprintf("corr(%s[t], %s[t-1])", names[1], lag_name),
		real.cross,
		synth.cross,
		shuffled.cross,
	)
	fmt.printf(
		"   %-26s %10.3e %10.3e %10.3e\n",
		"one-step MSE on real data",
		tstr[0],
		tstr[1],
		tstr[2],
	)
	fmt.println("   (one-step MSE: a linear predictor fitted on each data set, scored on real)")
}

// x1 an AR(1) with φ = 0.8, x2[t] = 0.7 x1[t-1] + 0.5 ε: windows of 12 steps
timegan_synthetic_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== TimeGAN on a known process (AR(1) and a lagged copy) ===")
	N_STEPS, T, F :: 6000, 12, 2
	x1, x2 := make([]f64, N_STEPS), make([]f64, N_STEPS)
	x1[0] = rand.float64_normal(0, 1)
	for i in 1 ..< N_STEPS {
		x1[i] = 0.8 * x1[i - 1] + 0.6 * rand.float64_normal(0, 1)
		x2[i] = 0.7 * x1[i - 1] + 0.5 * rand.float64_normal(0, 1)
	}
	n := N_STEPS - T
	data := make([]f64, n * T * F)
	for s in 0 ..< n {
		for t in 0 ..< T {
			data[(s * T + t) * F + 0] = x1[1 + s + t]
			data[(s * T + t) * F + 1] = x2[1 + s + t]
		}
	}
	synth := _tg_train(data, T, F, {1000, 500, 500}, 2000, allocator)
	shuffled := _tg_shuffled(data, 2000, T, F)
	rs, ss, hs := _tg_stats(data, T, F), _tg_stats(synth, T, F), _tg_stats(shuffled, T, F)
	tstr := [3]f64{_tg_tstr(data, T, F, rs), _tg_tstr(data, T, F, ss), _tg_tstr(data, T, F, hs)}
	_tg_report({"x1", "x2"}, "x1", rs, ss, hs, tstr)

	passed, total := 0, 0
	check :: proc(passed, total: ^int, ok: bool, msg: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}
	check(
		&passed,
		&total,
		abs(ss.acf - rs.acf) < 0.15 && abs(ss.cross - rs.cross) < 0.2,
		fmt.tprintf(
			"lag-1 autocorrelation %.3f (real %.3f, within 0.15) and cross-correlation %.3f (real %.3f, within 0.2)",
			ss.acf,
			rs.acf,
			ss.cross,
			rs.cross,
		),
	)
	std_ok := true
	for f in 0 ..< 2 {std_ok &&= abs(ss.std[f] / rs.std[f] - 1) < 0.25 && abs(ss.mean[f] - rs.mean[f]) < 0.25 * rs.std[f]}
	check(&passed, &total, std_ok, "means within 0.25 std and stds within 25% of the real ones")
	mid := 0.5 * (tstr[0] + tstr[2])
	check(
		&passed,
		&total,
		tstr[1] < mid,
		fmt.tprintf(
			"predictor fitted on TimeGAN data: one-step MSE %.3f on real data, below %.3f (halfway between fitted on real %.3f and on shuffled %.3f)",
			tstr[1],
			mid,
			tstr[0],
			tstr[2],
		),
	)
	fmt.printf("TimeGAN on a known process: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// SPY: daily log returns and log volume changes, 20-day windows over five years. The
// stylised facts reported: returns nearly uncorrelated, |returns| autocorrelated
// (volatility clustering); no pass / fail on live data. With this budget TimeGAN gets the
// marginals but makes returns far too persistent (lag-1 autocorrelation 0.7 to 0.99 against
// ~0): its sigmoid recurrent generator favours smooth paths, and near-white-noise
// returns are the hard case for it.
timegan_test :: proc(allocator: mem.Allocator = context.allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== TimeGAN on SPY daily returns and volume changes ===")
	spy, spy_ok := net.read_yahoo_series("SPY", .Daily, .FiveYears, allocator)
	if !spy_ok {fail_missing_data(#procedure, "Yahoo data for SPY"); return}
	defer net.yahoo_series_free(&spy)
	if len(spy.adj_close) < 500 {
		fail_missing_data(
			#procedure,
			fmt.tprintf("500 days of Yahoo data for SPY (got %d)", len(spy.adj_close)),
		)
		return
	}
	T :: 20
	// a NaN volume gives a zero volume change, like a missing volume did before
	data, n_seq, F := ml_finance.prepare_market_data(
		spy.adj_close,
		spy.volume,
		nil,
		0,
		T,
		allocator,
	)
	fmt.printf("%d windows of %d days, features: log return, log volume change\n", n_seq, T)
	// winsorised at 4 std: min-max scaling to [0, 1] otherwise squeezes the typical day
	// into a sliver of the range and the autoencoder learns only the mean
	for f in 0 ..< F {
		m, m2 := 0.0, 0.0
		for k := f; k < len(data); k += F {m += data[k]; m2 += data[k] * data[k]}
		k_n := f64(len(data) / F)
		sd := math.sqrt(m2 / k_n - (m / k_n) * (m / k_n))
		for k := f;
		    k < len(data);
		    k += F {data[k] = clamp(data[k], m / k_n - 4 * sd, m / k_n + 4 * sd)}
	}
	synth := _tg_train(data, T, F, {1000, 500, 500}, 1000, allocator)
	shuffled := _tg_shuffled(data, 1000, T, F)
	// |return| as the lagged variable: volatility clustering
	rs, ss, hs :=
		_tg_stats(data, T, F, abs_a = true),
		_tg_stats(synth, T, F, abs_a = true),
		_tg_stats(shuffled, T, F, abs_a = true)
	tstr := [3]f64 {
		_tg_tstr(data, T, F, rs, abs_a = true),
		_tg_tstr(data, T, F, ss, abs_a = true),
		_tg_tstr(data, T, F, hs, abs_a = true),
	}
	_tg_report({"return", "volume chg"}, "|return|", rs, ss, hs, tstr)
	r_real, r_syn := _tg_stats(data, T, F), _tg_stats(synth, T, F)
	fmt.printf(
		"   lag-1 autocorrelation of returns: real %.3f, TimeGAN %.3f\n",
		r_real.acf,
		r_syn.acf,
	)
}
