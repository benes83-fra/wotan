package tests

import ts "../wotan/analytics"
import ml "../wotan/analytics/ML"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Labels and weights doing their job.
//   A. Sequential bootstrap (López de Prado's Monte Carlo, snippet 4.9): 10 labels of
//      1 - 5 bars at random starts in 100 bars, 1000 trials; the mean uniqueness of a
//      sequential-bootstrap sample beats the standard bootstrap's.
//   B. Meta-labeling: a primary model picks sides at random; a feature tells when the
//      side is likely right (75% if f > 0, 35% otherwise). Triple-barrier meta-labels,
//      a logistic meta-model on the first half, bets sized by bet_size_from_probability
//      on the second half: higher precision and Sharpe than taking every primary bet.
//   C. Fractional differencing on 5 years of SPY log prices: the smallest d that passes
//      ADF and how much of the price series it keeps (correlation), against returns
//      (d = 1).

@(private = "file")
_lab_subset :: proc(
	labels: []ml_fin.TripleBarrierLabel,
	idx: []int,
) -> []ml_fin.TripleBarrierLabel {
	out := make([]ml_fin.TripleBarrierLabel, len(idx), context.temp_allocator)
	for i, k in idx {out[k] = labels[i]}
	return out
}

labeling_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Labels, weights and fractional differencing ===")
	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}

	// A. sequential vs standard bootstrap
	{
		TRIALS, NOBS, NBARS, MAXH :: 1000, 10, 100, 5
		labels := make([]ml_fin.TripleBarrierLabel, NOBS)
		defer delete(labels)
		idx := make([]int, NOBS)
		defer delete(idx)
		mean_u :: proc(set: []ml_fin.TripleBarrierLabel) -> f64 {
			c := ml_fin.label_concurrency(NBARS + MAXH + 1, set, context.temp_allocator)
			u := ml_fin.average_uniqueness(set, c, context.temp_allocator)
			m := 0.0
			for v in u {m += v / f64(len(u))}
			return m
		}
		seq_sum, std_sum := 0.0, 0.0
		seq_wins := 0
		for _ in 0 ..< TRIALS {
			for &lab in labels {
				lab.start = rand.int_max(NBARS)
				lab.end = lab.start + 1 + rand.int_max(MAXH)
			}
			for &i in idx {i = rand.int_max(NOBS)}
			us := mean_u(_lab_subset(labels, idx))
			seq := ml_fin.sequential_bootstrap(
				labels,
				NBARS + MAXH + 1,
				allocator = context.temp_allocator,
			)
			uq := mean_u(_lab_subset(labels, seq))
			seq_sum += uq / TRIALS
			std_sum += us / TRIALS
			if uq > us {seq_wins += 1}
		}
		check(
			&failures,
			seq_sum > std_sum + 0.03,
			fmt.tprintf(
				"sequential bootstrap: mean uniqueness %.3f vs standard bootstrap %.3f over %d trials (higher in %d)",
				seq_sum,
				std_sum,
				TRIALS,
				seq_wins,
			),
		)
	}

	// B. meta-labeling
	{
		E, BARS :: 2000, 10
		n := E * BARS + 1
		close := make([]f64, n)
		defer delete(close)
		feature := make([]f64, E)
		defer delete(feature)
		side := make([]f64, E)
		defer delete(side)
		events := make([]int, E)
		defer delete(events)
		close[0] = 100
		for e in 0 ..< E {
			f := rand.float64_normal(0, 1)
			s := 1.0 if rand.float64() < 0.5 else -1.0
			right := rand.float64() < (0.75 if f > 0 else 0.35)
			dir := s if right else -s
			feature[e], side[e], events[e] = f, s, e * BARS
			for b in 1 ..= BARS {
				t := e * BARS + b
				close[t] = close[t - 1] * (1 + dir * 0.002 + rand.float64_normal(0, 0.005))
			}
		}
		target := make([]f64, n)
		defer delete(target)
		for &v in target {v = 0.01}
		labels := ml_fin.triple_barrier_labels(
			close,
			events,
			target,
			{profit_taking = 1, stop_loss = 1, max_holding = BARS},
			side,
		)
		defer delete(labels)
		half := len(labels) / 2
		X := l.matrix_new(f64, half, 1)
		defer l.matrix_free(&X)
		y := make([]f64, half)
		defer delete(y)
		for i in 0 ..< half {
			X.data[i] = feature[i]
			y[i] = f64(labels[i].label)
		}
		model := ml.logistic_fit(
			&X,
			y,
			{
				C = 1,
				max_iter = 200,
				tol = 1e-8,
				learning_rate = 1,
				fit_intercept = true,
				optimizer_type = .LBFGS,
			},
		)
		defer ml.logistic_free(&model)
		Xt := l.matrix_new(f64, len(labels) - half, 1)
		defer l.matrix_free(&Xt)
		for i in half ..< len(labels) {Xt.data[i - half] = feature[i]}
		p := ml.logistic_predict_proba(&model, &Xt)
		defer delete(p)
		prim := make([]f64, len(labels) - half)
		defer delete(prim)
		sized := make([]f64, len(labels) - half)
		defer delete(sized)
		hits, taken, prim_hits := 0, 0, 0
		for i in half ..< len(labels) {
			k := i - half
			lab := labels[i]
			prim[k] = lab.ret // the side is already applied
			size := max(ml_fin.bet_size_from_probability(p[k]), 0) // bet only when the meta-model expects a win
			sized[k] = size * lab.ret
			prim_hits += lab.label
			if size > 0 {
				taken += 1
				hits += lab.label
			}
		}
		prec_prim := f64(prim_hits) / f64(len(prim))
		prec_meta := f64(hits) / f64(max(taken, 1))
		sr_prim := ml_fin.sharpe_stats(prim).sharpe
		sr_meta := ml_fin.sharpe_stats(sized).sharpe
		check(
			&failures,
			prec_meta > prec_prim + 0.07 && sr_meta > 2 * sr_prim,
			fmt.tprintf(
				"meta-labeling, out of sample: primary precision %.1f%%, Sharpe per bet %.3f; meta-model takes %d of %d bets, precision %.1f%%, sized Sharpe per bet %.3f",
				100 * prec_prim,
				sr_prim,
				taken,
				len(prim),
				100 * prec_meta,
				sr_meta,
			),
		)
	}

	// C. fractional differencing on SPY
	{
		spy, spy_ok := yahoo.read_yahoo_series("SPY", .Daily, .FiveYears, allocator)
		if !spy_ok {fail_missing_data(#procedure, "Yahoo data for SPY"); return}
		defer yahoo.yahoo_series_free(&spy)
		if len(spy.adj_close) < 600 {
			fail_missing_data(
				#procedure,
				fmt.tprintf("600 days of Yahoo data for SPY (got %d)", len(spy.adj_close)),
			)
		} else {
			lp := make([]f64, len(spy.adj_close))
			defer delete(lp)
			for v, i in spy.adj_close {lp[i] = math.ln(v)}
			md := ml_fin.frac_diff_min_d(lp[:], 0.05, 1, 1e-4)
			// d = 1: the log returns
			r := ts.log_returns(spy.adj_close)
			defer delete(r)
			corr_ret := fin.correlation(lp[1:], r)
			check(
				&failures,
				md.found && md.d < 1 && md.correlation > abs(corr_ret),
				fmt.tprintf(
					"SPY log prices (%d days): smallest d passing ADF at 5%% is %.2f (stat %.2f, p %.3f), correlation with the prices %.2f; returns (d = 1) %.2f",
					len(lp),
					md.d,
					md.adf_stat,
					md.p_value,
					md.correlation,
					corr_ret,
				),
			)
		}
	}
	fmt.printf("Labeling: %d failure(s)\n", failures)
	record_failures(failures)
}
