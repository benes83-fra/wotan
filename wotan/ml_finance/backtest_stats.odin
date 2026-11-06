package ml_finance

import ana "../analytics"
import ml "../analytics/ML"
import fin "../finance"
import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Backtest statistics: is a Sharpe ratio evidence of skill?
// ============================================================================
// Sharpe ratios here are per period (mean / sample standard deviation of the period
// returns, not annualised: divide an annual Sharpe ratio by sqrt(periods per year)).
//
//   probabilistic_sharpe_ratio  P[true SR > benchmark] given the observed SR, the number
//                               of returns and their skewness and kurtosis
//                               (Bailey & López de Prado 2012, "The Sharpe ratio
//                               efficient frontier")
//   min_track_record_length     returns needed for the PSR to reach a confidence level
//   expected_max_sharpe         the largest SR expected from n_trials unskilled
//                               strategies with SR variance V across the trials
//   deflated_sharpe_ratio       the PSR against that expected maximum: corrects for
//                               having picked the best of many trials (Bailey &
//                               López de Prado 2014, "The deflated Sharpe ratio")
//   backtest_overfitting        probability of backtest overfitting by combinatorially
//                               symmetric cross-validation (Bailey, Borwein,
//                               López de Prado & Zhu 2017, "The probability of
//                               backtest overfitting")

EULER_MASCHERONI :: 0.5772156649015329

SharpeStats :: struct {
	sharpe:   f64, // per period: mean / sample std (divisor n - 1)
	skewness: f64, // m3 / m2^1.5
	kurtosis: f64, // m4 / m2^2 (3 for a normal, not excess)
	n:        int,
}

// The Sharpe ratio of a return series and the moments its PSR needs
sharpe_stats :: proc(returns: []f64) -> SharpeStats {
	n := len(returns)
	mean, variance, skew, kurt := ana.sample_moments(returns)
	sd := math.sqrt(variance * f64(n) / f64(n - 1)) if n > 1 else 0
	return SharpeStats {
		sharpe = mean / sd if sd > 0 else 0,
		skewness = skew,
		kurtosis = kurt,
		n = n,
	}
}

@(private = "file")
_norm_cdf :: proc(x: f64) -> f64 {return 0.5 * math.erfc(-x / math.SQRT_TWO)}

// Variance of the SR estimate times (n - 1): 1 - skew SR + (kurt - 1) / 4 SR^2 (Mertens;
// 1 + SR^2 / 2 for normal returns). Never negative, as kurt >= 1 + skew^2.
@(private = "file")
_sharpe_estimate_variance :: proc(sharpe, skewness, kurtosis: f64) -> f64 {
	return max(1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe * sharpe, 0)
}

// P[true SR > benchmark]: Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))
probabilistic_sharpe_ratio :: proc(
	sharpe, benchmark: f64,
	n: int,
	skewness := 0.0,
	kurtosis := 3.0,
) -> f64 {
	assert(n > 1, "probabilistic_sharpe_ratio: needs at least 2 returns")
	v := _sharpe_estimate_variance(sharpe, skewness, kurtosis)
	return _norm_cdf((sharpe - benchmark) * math.sqrt(f64(n - 1)) / math.sqrt(v))
}

// The PSR of a return series
probabilistic_sharpe_ratio_returns :: proc(returns: []f64, benchmark := 0.0) -> f64 {
	s := sharpe_stats(returns)
	return probabilistic_sharpe_ratio(s.sharpe, benchmark, s.n, s.skewness, s.kurtosis)
}

// Number of returns for PSR(benchmark) to reach `probability` at the observed SR, skewness
// and kurtosis: 1 + (1 - skew SR + (kurt - 1)/4 SR^2) (z_p / (SR - SR*))^2.
// +Inf if SR <= benchmark.
min_track_record_length :: proc(
	sharpe, benchmark: f64,
	skewness := 0.0,
	kurtosis := 3.0,
	probability := 0.95,
) -> f64 {
	if sharpe <= benchmark {return math.INF_F64}
	z := fin.norm_inv(probability) / (sharpe - benchmark)
	return 1 + _sharpe_estimate_variance(sharpe, skewness, kurtosis) * z * z
}

// Expected maximum of n_trials SR estimates of strategies without skill, whose SRs vary
// across the trials with variance sharpe_variance:
// sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))), gamma Euler's constant.
// 0 for a single trial.
expected_max_sharpe :: proc(n_trials: int, sharpe_variance: f64) -> f64 {
	if n_trials <= 1 {return 0}
	N := f64(n_trials)
	g :: EULER_MASCHERONI
	return(
		math.sqrt(sharpe_variance) *
		((1 - g) * fin.norm_inv(1 - 1 / N) + g * fin.norm_inv(1 - 1 / (N * math.E))) \
	)
}

// The PSR of the selected strategy against the SR expected from the best of n_trials
// unskilled ones
deflated_sharpe_ratio :: proc(
	sharpe: f64,
	n: int,
	skewness, kurtosis: f64,
	n_trials: int,
	sharpe_variance: f64,
) -> f64 {
	return probabilistic_sharpe_ratio(
		sharpe,
		expected_max_sharpe(n_trials, sharpe_variance),
		n,
		skewness,
		kurtosis,
	)
}

DeflatedSharpeResult :: struct {
	best:            int, // column with the highest SR
	stats:           SharpeStats, // of that column
	sharpe_variance: f64, // of the column SRs (divisor N - 1)
	expected_max:    f64, // SR0, the benchmark the DSR uses
	psr:             f64, // PSR against 0 (ignores the selection)
	dsr:             f64,
}

// Per-column Sharpe ratios of a [T, N] matrix of returns (one column per trial)
sharpe_ratios :: proc(returns: ^l.Matrix(f64), allocator := context.allocator) -> []f64 {
	T, N := returns.rows, returns.cols
	out := make([]f64, N, allocator)
	col := make([]f64, T, context.temp_allocator)
	defer delete(col, context.temp_allocator)
	for j in 0 ..< N {
		for t in 0 ..< T {col[t] = returns.data[t * returns.cols + j]}
		out[j] = sharpe_stats(col).sharpe
	}
	return out
}

// The DSR of the best of the N trials in the columns of `returns` [T, N], N being the
// number of trials and V the variance of their SRs. (Trials that were run but are not
// in the matrix make N too small and the DSR too optimistic.)
deflated_sharpe_trials :: proc(returns: ^l.Matrix(f64)) -> (res: DeflatedSharpeResult) {
	T, N := returns.rows, returns.cols
	assert(N >= 2 && T >= 2, "deflated_sharpe_trials: needs at least 2 trials and 2 returns")
	srs := sharpe_ratios(returns, context.temp_allocator)
	defer delete(srs, context.temp_allocator)
	mean := 0.0
	for s, j in srs {
		mean += s / f64(N)
		if s > srs[res.best] {res.best = j}
	}
	for s in srs {res.sharpe_variance += (s - mean) * (s - mean) / f64(N - 1)}
	col := make([]f64, T, context.temp_allocator)
	defer delete(col, context.temp_allocator)
	for t in 0 ..< T {col[t] = returns.data[t * returns.cols + res.best]}
	res.stats = sharpe_stats(col)
	s := res.stats
	res.expected_max = expected_max_sharpe(N, res.sharpe_variance)
	res.psr = probabilistic_sharpe_ratio(s.sharpe, 0, s.n, s.skewness, s.kurtosis)
	res.dsr = probabilistic_sharpe_ratio(s.sharpe, res.expected_max, s.n, s.skewness, s.kurtosis)
	return
}

// ============================================================================
// Probability of backtest overfitting (CSCV)
// ============================================================================
// The T rows of the [T, N] trial returns are cut into S contiguous blocks (sizes as
// numpy.array_split: the first T mod S blocks one row longer). For each of the C(S, S/2)
// choices of S/2 blocks as the in-sample set (the rest out of sample):
//   n*     the trial with the best in-sample metric (the first one on ties)
//   omega  the rank of n*'s out-of-sample metric among the N trials (1 = worst, ties
//          averaged) over N + 1
//   lambda ln(omega / (1 - omega)), the logit
// PBO is the fraction of lambda <= 0: how often the in-sample winner is no better than
// the out-of-sample median. Also the performance degradation (OLS of n*'s
// out-of-sample metric on its in-sample one) and the probability that n* loses out of
// sample (metric < 0). The metric is the per-period Sharpe ratio unless one is given.

PBOResult :: struct {
	pbo:           f64,
	logits:        []f64, // per combination, in lexicographic order of the in-sample blocks
	is_metric:     []f64, // n*'s in-sample metric per combination
	oos_metric:    []f64, // n*'s out-of-sample metric per combination
	prob_oos_loss: f64, // fraction of combinations with oos_metric < 0
	slope:         f64, // OLS oos_metric = intercept + slope is_metric
	intercept:     f64,
	allocator:     mem.Allocator,
}

pbo_result_free :: proc(r: ^PBOResult) {
	delete(r.logits, r.allocator)
	delete(r.is_metric, r.allocator)
	delete(r.oos_metric, r.allocator)
	r^ = {}
}

@(private = "file")
_sharpe_from_sums :: proc(n, s1, s2: f64) -> f64 {
	if n < 2 {return 0}
	m := s1 / n
	v := (s2 - s1 * m) / (n - 1)
	return m / math.sqrt(v) if v > 0 else 0
}

backtest_overfitting :: proc(
	returns: ^l.Matrix(f64),
	n_blocks := 16,
	metric: proc(returns: []f64) -> f64 = nil, // nil: per-period Sharpe ratio
	allocator := context.allocator,
) -> (
	res: PBOResult,
) {
	T, N, S := returns.rows, returns.cols, n_blocks
	assert(S >= 2 && S % 2 == 0, "backtest_overfitting: n_blocks must be even")
	assert(T >= S, "backtest_overfitting: fewer rows than blocks")
	assert(N >= 2, "backtest_overfitting: needs at least 2 trials")
	res.allocator = allocator
	temp := context.temp_allocator

	// block b holds rows starts[b] ..< starts[b + 1]
	starts := ml.cv_block_starts(T, S, temp)
	defer delete(starts, temp)

	// Sharpe: per-block sums, so a combination costs N S instead of N T
	sums1, sums2: []f64
	if metric == nil {
		sums1, sums2 = make([]f64, S * N, temp), make([]f64, S * N, temp)
		for b in 0 ..< S {
			for t in starts[b] ..< starts[b + 1] {
				row := returns.data[t * returns.cols:][:N]
				for x, j in row {
					sums1[b * N + j] += x
					sums2[b * N + j] += x * x
				}
			}
		}
	}
	defer if metric == nil {delete(sums1, temp); delete(sums2, temp)}
	buf := make([]f64, T, temp) // one trial's rows of one side (custom metric)
	defer delete(buf, temp)
	side_metric :: proc(
		returns: ^l.Matrix(f64),
		starts: []int,
		in_side: []bool,
		side: bool,
		j: int,
		metric: proc(_: []f64) -> f64,
		sums1, sums2, buf: []f64,
	) -> f64 {
		N := returns.cols
		if metric == nil {
			n, s1, s2 := 0.0, 0.0, 0.0
			for b in 0 ..< len(in_side) {
				if in_side[b] != side {continue}
				n += f64(starts[b + 1] - starts[b])
				s1 += sums1[b * N + j]
				s2 += sums2[b * N + j]
			}
			return _sharpe_from_sums(n, s1, s2)
		}
		k := 0
		for b in 0 ..< len(in_side) {
			if in_side[b] != side {continue}
			for t in starts[b] ..< starts[b + 1] {
				buf[k] = returns.data[t * returns.cols + j]
				k += 1
			}
		}
		return metric(buf[:k])
	}

	// combinations of S/2 of the S blocks, lexicographic
	H := S / 2
	n_comb := 1
	for i in 0 ..< H {n_comb = n_comb * (S - i) / (i + 1)}
	res.logits = make([]f64, n_comb, allocator)
	res.is_metric = make([]f64, n_comb, allocator)
	res.oos_metric = make([]f64, n_comb, allocator)
	idx := make([]int, H, temp)
	defer delete(idx, temp)
	for i in 0 ..< H {idx[i] = i}
	in_sample := make([]bool, S, temp)
	defer delete(in_sample, temp)
	oos := make([]f64, N, temp)
	defer delete(oos, temp)

	overfit, losses := 0, 0
	for c in 0 ..< n_comb {
		for &v in in_sample {v = false}
		for i in idx {in_sample[i] = true}
		best, best_is := 0, math.inf_f64(-1)
		for j in 0 ..< N {
			v := side_metric(returns, starts, in_sample, true, j, metric, sums1, sums2, buf)
			if v > best_is {best, best_is = j, v}
			oos[j] = side_metric(returns, starts, in_sample, false, j, metric, sums1, sums2, buf)
		}
		below, ties := 0, 0
		for v in oos {
			if v < oos[best] {below += 1} else if v == oos[best] {ties += 1}
		}
		rank := f64(below) + (f64(ties) + 1) / 2 // average rank, 1 = worst
		omega := rank / f64(N + 1)
		res.logits[c] = math.ln(omega / (1 - omega))
		res.is_metric[c], res.oos_metric[c] = best_is, oos[best]
		if res.logits[c] <= 0 {overfit += 1}
		if oos[best] < 0 {losses += 1}

		// next combination
		i := H - 1
		for i >= 0 && idx[i] == S - H + i {i -= 1}
		if i < 0 {break}
		idx[i] += 1
		for k in i + 1 ..< H {idx[k] = idx[k - 1] + 1}
	}
	res.pbo = f64(overfit) / f64(n_comb)
	res.prob_oos_loss = f64(losses) / f64(n_comb)
	mx, my := 0.0, 0.0
	for c in 0 ..< n_comb {
		mx += res.is_metric[c] / f64(n_comb)
		my += res.oos_metric[c] / f64(n_comb)
	}
	sxy, sxx := 0.0, 0.0
	for c in 0 ..< n_comb {
		dx := res.is_metric[c] - mx
		sxy += dx * (res.oos_metric[c] - my)
		sxx += dx * dx
	}
	res.slope = sxy / sxx if sxx > 0 else 0
	res.intercept = my - res.slope * mx
	return
}

// Selection report for a grid of N trials ([T, N] returns, one column per configuration):
// the best trial's annualised SR with its PSR, the SR0 that N trials of no skill would
// reach, the DSR against it, and the PBO by CSCV over n_blocks blocks
print_selection_stats :: proc(
	trials: ^l.Matrix(f64),
	periods_per_year := 252.0,
	n_blocks := 16,
	indent := "   ",
) -> (
	dsr: DeflatedSharpeResult,
	pbo: f64,
) {
	dsr = deflated_sharpe_trials(trials)
	p := backtest_overfitting(trials, n_blocks, allocator = context.temp_allocator)
	defer pbo_result_free(&p)
	ann := math.sqrt(periods_per_year)
	fmt.printf(
		"%sselection over %d trials of %d periods: best trial %d, annual SR %.2f, PSR %.3f; the best of %d without skill would reach SR %.2f -> DSR %.3f\n",
		indent,
		trials.cols,
		trials.rows,
		dsr.best,
		ann * dsr.stats.sharpe,
		dsr.psr,
		trials.cols,
		ann * dsr.expected_max,
		dsr.dsr,
	)
	fmt.printf(
		"%sCSCV over %d blocks: PBO %.2f, P[the in-sample best loses out of sample] %.2f, OOS SR = %.2f %+.2f × IS SR (annual)\n",
		indent,
		n_blocks,
		p.pbo,
		p.prob_oos_loss,
		ann * p.intercept,
		p.slope,
	)
	return dsr, p.pbo
}
