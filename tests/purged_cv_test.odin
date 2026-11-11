package tests

import ml "../wotan/analytics/ML"
import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Purged cross-validation where it matters.
//   A. Leakage: a random walk; the label is the sign of the next 20 steps, the features
//      the last 20, 50 and 200 steps, so they carry no information about the label. But
//      neighbouring samples have nearly the same features and overlapping labels, and a
//      1-nearest-neighbour classifier (ml.knn_cv_evaluator via ml.cross_val_score_cv)
//      finds them: shuffled K-fold reports a skill that is not there, purged K-fold
//      (label horizon 20, embargo 20) reports ~50%.
//   B. CPCV on SPY: a momentum rule (long / short on the sign of the trailing L-day
//      return) with L chosen on each training set; 6 groups, 2 test groups -> 15 splits
//      stitched into 5 out-of-sample paths, a distribution of backtest Sharpe ratios
//      instead of one.

@(private = "file")
_PCV_H :: 20

purged_cv_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Purged cross-validation: leakage and CPCV paths ===")
	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}

	// A. leakage
	{
		T, REPS, WARM :: 2000, 10, 200
		lags := [3]int{20, 50, 200}
		splitters := [3]ml.CVSplitter {
			{kind = .KFold, n_splits = 5, shuffle = true, random_state = 7},
			{kind = .KFold, n_splits = 5},
			{kind = .Purged_KFold, n_splits = 5, label_horizon = _PCV_H, embargo = _PCV_H},
		}
		names := [3]string{"shuffled K-fold", "contiguous K-fold", "purged K-fold"}
		acc: [3]f64
		params := ml.KNNParams {
			k = 1,
		}
		walk := make([]f64, WARM + T + _PCV_H)
		defer delete(walk)
		X := l.matrix_new(f64, T, 3)
		defer l.matrix_free(&X)
		y := make([]f64, T)
		defer delete(y)
		for _ in 0 ..< REPS {
			for i in 1 ..< len(walk) {walk[i] = walk[i - 1] + rand.float64_normal(0, 1)}
			for i in 0 ..< T {
				t := WARM + i
				for L, c in lags {X.data[i * 3 + c] = walk[t] - walk[t - L]}
				y[i] = 1 if walk[t + _PCV_H] > walk[t] else -1
			}
			for cv, s in splitters {
				scores := ml.cross_val_score_cv(&X, y, cv, ml.knn_cv_evaluator, &params)
				for v in scores {acc[s] += v / f64(len(scores) * REPS)}
				delete(scores)
			}
		}
		for a, s in acc {fmt.printf("   %-18s 1-NN accuracy %.3f\n", names[s], a)}
		check(
			&failures,
			acc[0] > 0.62 && abs(acc[2] - 0.5) < 0.05,
			fmt.tprintf(
				"features without information, labels overlapping 20 steps: shuffled K-fold %.1f%% (leakage), purged %.1f%% (chance is 50%%)",
				100 * acc[0],
				100 * acc[2],
			),
		)
	}

	// B. CPCV paths on SPY
	{
		df, df_ok := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
		if !df_ok {fail_missing_data(#procedure, "Yahoo data for SPY"); return}
		defer w.destroy_dataframe(&df)
		if df.rows < 600 {
			fmt.println("Not enough SPY data (offline?): CPCV example skipped")
		} else {
			pc := w.column(&df, "AdjClose")
			px := make([dynamic]f64)
			defer delete(px)
			for i in 0 ..< df.rows {
				if p, null := w.column_at_float(pc, i); !null && p > 0 {append(&px, p)}
			}
			lookbacks := []int{5, 10, 20, 40, 60, 120, 250}
			WARM :: 250
			n := len(px) - 1 - WARM // sample i: decision at the close of day WARM + i, earns the next day
			// rule returns per lookback
			rule := make([][]f64, len(lookbacks))
			defer {for r in rule {delete(r)}; delete(rule)}
			for L, j in lookbacks {
				rule[j] = make([]f64, n)
				for i in 0 ..< n {
					t := WARM + i
					next := px[t + 1] / px[t] - 1
					rule[j][i] = next if px[t] > px[t - L] else -next
				}
			}
			NG, KT :: 6, 2
			folds := ml.cv_make_folds(
				ml.CVSplitter {
					kind = .Combinatorial_Purged,
					n_splits = NG,
					n_test_groups = KT,
					label_horizon = 1,
					embargo = 5,
				},
				n,
			)
			defer ml.cv_folds_free(folds)
			split_vals := make([][]f64, len(folds))
			defer {for v in split_vals {delete(v)}; delete(split_vals)}
			buf := make([]f64, n)
			defer delete(buf)
			for f, s in folds {
				// lookback with the best Sharpe ratio on the (purged) training samples
				best, best_sr := 0, math.inf_f64(-1)
				for j in 0 ..< len(lookbacks) {
					for i, k in f.train {buf[k] = rule[j][i]}
					sr := ml_fin.sharpe_stats(buf[:len(f.train)]).sharpe
					if sr > best_sr {best, best_sr = j, sr}
				}
				split_vals[s] = make([]f64, n)
				for &v in split_vals[s] {v = math.nan_f64()}
				for i in f.test {split_vals[s][i] = rule[best][i]}
			}
			paths, n_paths := ml.cv_cpcv_paths(NG, KT)
			defer delete(paths)
			complete := true
			mean, m2 := 0.0, 0.0
			ann := math.sqrt(252.0)
			fmt.printf(
				"   %d samples, %d splits, %d paths; annual Sharpe of each out-of-sample path:",
				n,
				len(folds),
				n_paths,
			)
			for p in 0 ..< n_paths {
				path := ml.cv_cpcv_path(split_vals, paths, NG, p)
				defer delete(path)
				for v in path {complete &&= !math.is_nan(v)}
				sr := ann * ml_fin.sharpe_stats(path).sharpe
				mean += sr / f64(n_paths)
				m2 += sr * sr / f64(n_paths)
				fmt.printf(" %.2f", sr)
			}
			full_best := math.inf_f64(-1)
			for r in rule {full_best = max(full_best, ann * ml_fin.sharpe_stats(r).sharpe)}
			fmt.printf(
				"\n   path Sharpe %.2f ± %.2f; the best lookback in sample over all %d days: %.2f\n",
				mean,
				math.sqrt(max(m2 - mean * mean, 0)),
				n,
				full_best,
			)
			check(
				&failures,
				len(folds) == 15 && n_paths == 5 && complete,
				"CPCV on SPY: 15 splits, 5 complete out-of-sample paths (every sample predicted once per path)",
			)
		}
	}
	fmt.printf("Purged CV: %d failure(s)\n", failures)
	record_failures(failures)
}
