package stats

import w "../core"
import "core:math"
import "core:slice"

// Correlation coefficients with tests of zero correlation: Pearson,
// Spearman and Kendall.

// Result of a correlation test.
Correlation_Result :: struct {
	coefficient: f64, // r, rho or tau
	p_value:     f64, // test of zero correlation
	ci_low:      f64, // confidence interval (Pearson only, by Fisher's z with 4 or more pairs;
	ci_high:     f64, // NaN for Spearman and Kendall)
}

// Pearson's correlation r of the pairs (x[i], y[i]), with the exact test of
// zero correlation for normal data (r has a beta distribution under the
// null) and a Fisher-z confidence interval, open on one side for one-sided
// alternatives. Greater tests for positive correlation. Needs 2 pairs;
// constant x or y gives NaN. As scipy's pearsonr (whose one-sided p-values
// in 1.10.0 are wrong for negative r).
pearson :: proc(
	x, y: []f64,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Correlation_Result,
	ok: bool,
) {
	n := len(x)
	if n != len(y) ||
	   n < 2 ||
	   !_all_finite(x) ||
	   !_all_finite(y) ||
	   !_valid_confidence(confidence) {return {}, false}
	res = _pearson(x, y, alternative)
	// Fisher z interval
	nan := math.nan_f64()
	res.ci_low, res.ci_high = -1, 1
	if math.is_nan(res.coefficient) {
		res.ci_low, res.ci_high = nan, nan
	} else if n > 3 {
		r := res.coefficient
		zr := math.atanh(r) if abs(r) < 1 else math.copy_sign(math.inf_f64(1), r)
		se := math.sqrt(1 / f64(n - 3))
		switch alternative {
		case .Two_Sided:
			h := normal_ppf(STANDARD_NORMAL, 0.5 + confidence / 2)
			res.ci_low, res.ci_high = math.tanh(zr - h * se), math.tanh(zr + h * se)
		case .Less:
			res.ci_high = math.tanh(zr + normal_ppf(STANDARD_NORMAL, confidence) * se)
		case .Greater:
			res.ci_low = math.tanh(zr - normal_ppf(STANDARD_NORMAL, confidence) * se)
		}
	}
	return res, true
}

// r and its p-value; no interval
@(private)
_pearson :: proc(x, y: []f64, alternative: Alternative) -> (res: Correlation_Result) {
	n := len(x)
	nan := math.nan_f64()
	res.ci_low, res.ci_high = nan, nan
	if _constant(x) || _constant(y) {
		res.coefficient, res.p_value = nan, nan
		return
	}
	if n == 2 {
		res.coefficient = math.sign(x[1] - x[0]) * math.sign(y[1] - y[0])
		res.p_value = 1
		return
	}
	cx, cy := _center(x), _center(y)
	// compensated sums: far-tail p-values are sensitive to the last bits of r
	sxx, syy, sxy: _Sum
	for i in 0 ..< n {
		dx, dy := _dev(cx, x[i]), _dev(cy, y[i])
		_sum_add(&sxx, dx * dx)
		_sum_add(&syy, dy * dy)
		_sum_add(&sxy, dx * dy)
	}
	r := clamp(_sum(sxy) / math.sqrt(_sum(sxx) * _sum(syy)), -1, 1)
	res.coefficient = r
	// under the null (1 + r) / 2 is Beta(n/2 - 1, n/2 - 1)
	ab := f64(n) / 2 - 1
	switch alternative {
	case .Two_Sided:
		lo, _ := beta_inc_pair(ab, ab, (1 - abs(r)) / 2, (1 + abs(r)) / 2)
		res.p_value = min(1, 2 * lo)
	case .Less:
		res.p_value, _ = beta_inc_pair(ab, ab, (1 + r) / 2, (1 - r) / 2)
	case .Greater:
		res.p_value, _ = beta_inc_pair(ab, ab, (1 - r) / 2, (1 + r) / 2)
	}
	return
}

@(private)
_constant :: proc(x: []f64) -> bool {
	for v in x {if v != x[0] {return false}}
	return true
}

// Spearman's rank correlation rho: Pearson's r of the ranks (ties get
// their average rank), tested with the t approximation on n - 2 degrees of
// freedom (scipy's spearmanr). Needs 3 pairs; constant x or y gives NaN.
spearman :: proc(
	x, y: []f64,
	alternative := Alternative.Two_Sided,
) -> (
	res: Correlation_Result,
	ok: bool,
) {
	n := len(x)
	if n != len(y) || n < 3 || !_all_finite(x) || !_all_finite(y) {return {}, false}
	w.TEMP_GUARD()
	rx := make([]f64, n, context.temp_allocator)
	ry := make([]f64, n, context.temp_allocator)
	_rank(x, rx)
	_rank(y, ry)
	r := _pearson(rx, ry, .Two_Sided)
	rs := r.coefficient
	res.coefficient = rs
	res.ci_low, res.ci_high = math.nan_f64(), math.nan_f64()
	df := f64(n - 2)
	t := rs * math.sqrt(max(0, df / ((rs + 1) * (1 - rs))))
	if abs(rs) == 1 {t = math.copy_sign(math.inf_f64(1), rs)}
	res.p_value = _p_t(t, df, alternative)
	return res, true
}

// Kendall's tau-b, which accounts for ties in x and in y. Method Auto uses
// the exact null distribution without ties for up to 33 pairs (or when at
// most one pair is discordant or concordant), otherwise the normal
// approximation with the variance corrected for ties (scipy's kendalltau).
// Exact with ties gives ok = false. Needs 2 pairs; constant x or y gives
// NaN. O(n log n).
kendall :: proc(
	x, y: []f64,
	alternative := Alternative.Two_Sided,
	method := Method.Auto,
) -> (
	res: Correlation_Result,
	ok: bool,
) {
	n := len(x)
	if n != len(y) || n < 2 || !_all_finite(x) || !_all_finite(y) {return {}, false}
	w.TEMP_GUARD()
	nan := math.nan_f64()
	res.ci_low, res.ci_high = nan, nan
	// dense ranks of y, then pairs sorted by (x, y)
	P :: struct {
		x: f64,
		y: int,
	}
	ry := _dense_ranks(y)
	ps := make([]P, n, context.temp_allocator)
	for i in 0 ..< n {ps[i] = {x[i], ry[i]}}
	slice.sort_by(ps, proc(a, b: P) -> bool {return a.x < b.x || (a.x == b.x && a.y < b.y)})
	xr := make([]int, n, context.temp_allocator)
	yr := make([]int, n, context.temp_allocator)
	for i in 0 ..< n {
		yr[i] = ps[i].y
		xr[i] = 0 if i == 0 else (xr[i - 1] + (1 if ps[i].x != ps[i - 1].x else 0))
	}
	dis := _count_inversions(yr)
	ntie := 0
	for i := 0; i < n; {
		j := i + 1
		for j < n && xr[j] == xr[i] && yr[j] == yr[i] {j += 1}
		c := j - i
		ntie += c * (c - 1) / 2
		i = j
	}
	xtie, x0, x1 := _rank_ties(xr)
	ytie, y0, y1 := _rank_ties(_dense_ranks(y))
	tot := n * (n - 1) / 2
	if xtie == tot || ytie == tot {
		res.coefficient, res.p_value = nan, nan
		return res, true
	}
	cmd := f64(tot - xtie - ytie + ntie - 2 * dis)
	tau := cmd / math.sqrt(f64(tot - xtie)) / math.sqrt(f64(tot - ytie))
	res.coefficient = clamp(tau, -1, 1)
	m := method
	if m == .Exact && (xtie != 0 || ytie != 0) {return {}, false}
	if m == .Auto {
		m =
			.Exact if xtie == 0 && ytie == 0 && (n <= 33 || min(dis, tot - dis) <= 1) else .Asymptotic
	}
	if m == .Exact {
		res.p_value = _kendall_p_exact(n, tot - dis, alternative)
	} else {
		nf := f64(n)
		mm := nf * (nf - 1)
		v :=
			(mm * (2 * nf + 5) - x1 - y1) / 18 +
			2 * f64(xtie) * f64(ytie) / mm +
			x0 * y0 / (9 * mm * (nf - 2))
		res.p_value = _p_normal(cmd / math.sqrt(v), alternative)
	}
	return res, true
}

// dense ranks 0, 1, ... of x (equal values share one), temp allocator
@(private)
_dense_ranks :: proc(x: []f64) -> []int {
	idx := _order(x)
	r := make([]int, len(x), context.temp_allocator)
	k := 0
	for i in 0 ..< len(idx) {
		if i > 0 && x[idx[i]] != x[idx[i - 1]] {k += 1}
		r[idx[i]] = k
	}
	return r
}

// tied pairs Σ t(t-1)/2 and the variance terms Σ t(t-1)(t-2), Σ t(t-1)(2t+5)
@(private)
_rank_ties :: proc(r: []int) -> (pairs: int, v0, v1: f64) {
	cnt := make(map[int]int, context.temp_allocator)
	for v in r {cnt[v] += 1}
	for _, c in cnt {
		if c > 1 {
			cf := f64(c)
			pairs += c * (c - 1) / 2
			v0 += cf * (cf - 1) * (cf - 2)
			v1 += cf * (cf - 1) * (2 * cf + 5)
		}
	}
	return
}

// pairs i < j with a[i] > a[j], by merge sort
@(private)
_count_inversions :: proc(a_in: []int) -> int {
	n := len(a_in)
	a := make([]int, n, context.temp_allocator)
	copy(a, a_in)
	buf := make([]int, n, context.temp_allocator)
	inv := 0
	for width := 1; width < n; width *= 2 {
		for lo := 0; lo < n - width; lo += 2 * width {
			mid, hi := lo + width, min(lo + 2 * width, n)
			i, j, k := lo, mid, lo
			for i < mid && j < hi {
				if a[j] < a[i] {
					buf[k] = a[j]
					inv += mid - i
					j += 1
				} else {
					buf[k] = a[i]
					i += 1
				}
				k += 1
			}
			for i < mid {buf[k] = a[i]; i += 1; k += 1}
			for j < hi {buf[k] = a[j]; j += 1; k += 1}
			copy(a[lo:hi], buf[lo:hi])
		}
	}
	return inv
}

// exact p-value with c concordant pairs out of n (n - 1) / 2, no ties; the
// count of permutations by concordant pairs are the Mahonian numbers,
// built up one element at a time (scipy's _kendall_p_exact)
@(private)
_kendall_p_exact :: proc(n, c_in: int, alternative: Alternative) -> f64 {
	tot := n * (n - 1) / 2
	in_right_tail := c_in >= tot - c_in
	c := min(c_in, tot - c_in)
	prob, pmass: f64
	if n <= 2 {
		prob, pmass = 1, (1 if n == 1 else 0.5)
	} else if c == 0 {
		prob = 2 / _factorial(n)
		pmass = prob / 2
	} else if c == 1 {
		prob = 2 / _factorial(n - 1)
		pmass = f64(n - 1) / _factorial(n)
	} else if 4 * c == n * (n - 1) && alternative == .Two_Sided {
		prob = 1
	} else {
		v := make([]f64, c + 1, context.temp_allocator)
		v[0], v[1] = 1, 1
		cs := make([]f64, c + 1, context.temp_allocator)
		for j in 3 ..= n {
			s := 0.0
			for k in 0 ..= c {
				s += v[k]
				cs[k] = s / f64(j)
			}
			for k in 0 ..= c {v[k] = cs[k] - (cs[k - j] if k >= j else 0)}
		}
		for e in v {prob += e}
		pmass = v[c] / 2
	}
	if alternative != .Two_Sided {
		if in_right_tail == (alternative == .Greater) {
			prob /= 2
		} else {
			prob = 1 - prob / 2 + pmass
		}
	}
	return clamp(prob, 0, 1)
}

// n! (0 beyond the range of f64)
@(private)
_factorial :: proc(n: int) -> f64 {
	if n > 170 {return math.inf_f64(1)}
	f := 1.0
	for i in 2 ..= n {f *= f64(i)}
	return f
}
