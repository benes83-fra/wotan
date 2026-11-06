package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Do the backtest statistics do their job on known ground truth?
//   A. PSR calibration: negatively skewed returns (a reflected lognormal) with a true
//      per-period SR of 0.1; the PSR against that true SR should exceed 0.95 in ~5% of
//      samples. The normal-returns formula (skewness 0, kurtosis 3) understates the
//      SR's estimation error under negative skew and rejects more often.
//   B. Selection: the best of 50 strategies without skill looks significant by its PSR
//      most of the time; its DSR rarely does, and PBO is around 1/2.
//   C. One strategy with skill (annual SR 2.5) among 49 without: the DSR finds it, PBO
//      is near 0.
//   D. Applied: 40 SMA-crossover rules on 5 years of SPY; the best rule's PSR, DSR and
//      the grid's PBO (reported, not asserted beyond consistency).

@(private = "file")
_bst_noise_trials :: proc(T, N: int, skilled_sr: f64) -> l.Matrix(f64) {
	m := l.matrix_new(f64, T, N)
	for t in 0 ..< T {
		for j in 0 ..< N {
			m.data[t * N + j] = rand.float64_normal(0, 0.01)
		}
		m.data[t * N] += 0.01 * skilled_sr // trial 0 carries the skill (0 for none)
	}
	return m
}

backtest_stats_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Backtest statistics on known ground truth ===")
	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}

	// A. PSR calibration under skew
	{
		T, REPS :: 1000, 2000
		SIGMA :: 0.6 // of the underlying normal
		SR :: 0.1
		mean_ln := math.exp(SIGMA * SIGMA / 2)
		sd_ln := math.sqrt((math.exp(SIGMA * SIGMA) - 1) * math.exp(SIGMA * SIGMA))
		r := make([]f64, T)
		defer delete(r)
		hit_psr, hit_naive := 0, 0
		skew_sum := 0.0
		for _ in 0 ..< REPS {
			for &v in r {v = SR - (math.exp(SIGMA * rand.float64_normal(0, 1)) - mean_ln) / sd_ln}
			s := ml_fin.sharpe_stats(r)
			skew_sum += s.skewness / REPS
			if ml_fin.probabilistic_sharpe_ratio(s.sharpe, SR, T, s.skewness, s.kurtosis) >
			   0.95 {hit_psr += 1}
			if ml_fin.probabilistic_sharpe_ratio(s.sharpe, SR, T) > 0.95 {hit_naive += 1}
		}
		rate, naive := f64(hit_psr) / REPS, f64(hit_naive) / REPS
		check(
			&failures,
			abs(rate - 0.05) < 0.015 && naive > rate,
			fmt.tprintf(
				"PSR calibration, skewness %.2f, true SR %.1f, %d samples of %d: PSR > 0.95 in %.1f%% (nominal 5%%); with the normal-returns formula %.1f%%",
				skew_sum,
				SR,
				REPS,
				T,
				100 * rate,
				100 * naive,
			),
		)
	}

	// B and C. Selection among 50 trials
	for skilled in ([]f64{0, 2.5 / math.sqrt(252.0)}) {
		T, N, REPS :: 1500, 50, 20
		psr_hits, dsr_hits, found := 0, 0, 0
		pbo_sum := 0.0
		for _ in 0 ..< REPS {
			M := _bst_noise_trials(T, N, skilled)
			defer l.matrix_free(&M)
			d := ml_fin.deflated_sharpe_trials(&M)
			p := ml_fin.backtest_overfitting(&M, 16)
			defer ml_fin.pbo_result_free(&p)
			if d.psr > 0.95 {psr_hits += 1}
			if d.dsr > 0.95 {dsr_hits += 1}
			if d.best == 0 {found += 1}
			pbo_sum += p.pbo / REPS
		}
		if skilled == 0 {
			check(
				&failures,
				psr_hits >= REPS * 3 / 4 && dsr_hits <= 2 && abs(pbo_sum - 0.5) < 0.15,
				fmt.tprintf(
					"best of %d unskilled trials (%d samples of %d days): PSR > 0.95 in %d, DSR > 0.95 in %d; mean PBO %.2f",
					N,
					REPS,
					T,
					psr_hits,
					dsr_hits,
					pbo_sum,
				),
			)
		} else {
			check(
				&failures,
				found == REPS && dsr_hits >= REPS * 4 / 5 && pbo_sum < 0.05,
				fmt.tprintf(
					"one trial with annual SR 2.5 among %d: selected in %d of %d samples, DSR > 0.95 in %d; mean PBO %.3f",
					N,
					found,
					REPS,
					dsr_hits,
					pbo_sum,
				),
			)
		}
	}

	// D. SMA crossovers on SPY
	{
		df := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
		defer w.destroy_dataframe(&df)
		if df.rows < 600 {
			fmt.println("Not enough SPY data (offline?): applied example skipped")
		} else {
			pc := w.column(&df, "AdjClose")
			px := make([dynamic]f64)
			defer delete(px)
			for i in 0 ..< df.rows {
				if p, null := w.column_at_float(pc, i); !null && p > 0 {append(&px, p)}
			}
			n := len(px)
			cum := make([]f64, n + 1) // prefix sums for the moving averages
			defer delete(cum)
			for p, i in px {cum[i + 1] = cum[i] + p}
			sma :: proc(cum: []f64, t, k: int) -> f64 {return(
					(cum[t + 1] - cum[t + 1 - k]) /
					f64(k) \
				)}
			fasts := []int{5, 10, 20, 30, 50}
			slows := []int{60, 100, 150, 200}
			WARM :: 200
			T := n - 1 - WARM // decisions at the close of days WARM ..< n - 1
			N := 2 * len(fasts) * len(slows) // long / flat and long / short
			M := l.matrix_new(f64, T, N)
			defer l.matrix_free(&M)
			for t in WARM ..< n - 1 {
				next := px[t + 1] / px[t] - 1
				j := 0
				for f in fasts {
					for s in slows {
						up := sma(cum[:], t, f) > sma(cum[:], t, s)
						M.data[(t - WARM) * N + j] = next if up else 0
						M.data[(t - WARM) * N + j + 1] = next if up else -next
						j += 2
					}
				}
			}
			d := ml_fin.deflated_sharpe_trials(&M)
			p := ml_fin.backtest_overfitting(&M, 16)
			defer ml_fin.pbo_result_free(&p)
			b := d.best / 2
			ann := math.sqrt(252.0)
			fmt.printf(
				"   best of %d SMA rules on %d SPY days: fast %d / slow %d %s, annual SR %.2f, PSR %.3f; SR0 %.2f annual -> DSR %.3f\n",
				N,
				T,
				fasts[b / len(slows)],
				slows[b % len(slows)],
				"long/flat" if d.best % 2 == 0 else "long/short",
				ann * d.stats.sharpe,
				d.psr,
				ann * d.expected_max,
				d.dsr,
			)
			fmt.printf(
				"   CSCV with 16 blocks: PBO %.2f, P[selected rule loses out of sample] %.2f, OOS SR = %.2f %+.2f × IS SR (annual)\n",
				p.pbo,
				p.prob_oos_loss,
				ann * p.intercept,
				p.slope,
			)
			ok :=
				d.dsr <= d.psr &&
				d.dsr >= 0 &&
				d.psr <= 1 &&
				p.pbo >= 0 &&
				p.pbo <= 1 &&
				len(p.logits) == 12870
			check(
				&failures,
				ok,
				"SPY grid: DSR <= PSR, PBO in [0, 1] over all C(16, 8) = 12870 splits",
			)
		}
	}
	fmt.printf("Backtest statistics: %d failure(s)\n", failures)
	record_failures(failures)
}
