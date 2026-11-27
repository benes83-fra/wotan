package stats

import w "../core"
import "core:math"

// Rank tests: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis.

// Mann-Whitney U test (Wilcoxon rank-sum): do x and y come from the same
// distribution, against x tending to be smaller (Less) or larger (Greater)?
// The statistic is U of x: the number of pairs with x[i] > y[j], ties
// counting 1/2. Method Auto uses the exact null distribution when one
// sample has at most 8 values and there are no ties, otherwise the normal
// approximation with tie correction and, if `continuity`, a continuity
// correction (scipy's mannwhitneyu). Exact ignores ties.
mann_whitney_u :: proc(
	x, y: []f64,
	alternative := Alternative.Two_Sided,
	method := Method.Auto,
	continuity := true,
) -> (
	res: Test_Result,
	ok: bool,
) {
	n1, n2 := len(x), len(y)
	if n1 == 0 || n2 == 0 || !_all_finite(x) || !_all_finite(y) {return {}, false}
	w.TEMP_GUARD()
	n := n1 + n2
	xy := make([]f64, n, context.temp_allocator)
	copy(xy, x)
	copy(xy[n1:], y)
	ranks := make([]f64, n, context.temp_allocator)
	tie_term := _rank(xy, ranks)
	r1: f64
	for r in ranks[:n1] {r1 += r}
	a, b := f64(n1), f64(n2)
	u1 := r1 - a * (a + 1) / 2
	u2 := a * b - u1
	m := method
	if m == .Auto {
		m = .Exact if (n1 <= 8 || n2 <= 8) && tie_term == 0 else .Asymptotic
	}
	u, f: f64
	switch alternative {
	case .Greater:
		u, f = u1, 1
	case .Less:
		u, f = u2, 1
	case .Two_Sided:
		u, f = max(u1, u2), 2
	}
	p: f64
	if m == .Exact {
		// P(U >= u) = P(U <= mn - u) by symmetry
		p = _mwu_cdf(n1, n2, int(a * b) - int(u))
	} else {
		nf := f64(n)
		s := math.sqrt(a * b / 12 * ((nf + 1) - tie_term / (nf * (nf - 1))))
		num := u - a * b / 2
		if continuity {num -= 0.5}
		p = normal_sf(STANDARD_NORMAL, num / s)
	}
	res.statistic = u1
	res.p_value = clamp(p * f, 0, 1)
	return res, true
}

// P(U <= k) under the null for sample sizes m and n: the count of
// arrangements with U = j is the coefficient of q^j in the Gaussian binomial
// [m + n choose m]_q = Π_{i=1..m} (1 - q^(n+i)) / (1 - q^i), built up to
// degree k. Exact in 128-bit integers while C(m + n, m) < 2^126; beyond
// that in floating point.
@(private)
_mwu_cdf :: proc(m_in, n_in: int, k: int) -> f64 {
	if k < 0 {return 0}
	m, n := min(m_in, n_in), max(m_in, n_in)
	if k >= m * n {return 1}
	lchoose := log_gamma(f64(m + n + 1)) - log_gamma(f64(m + 1)) - log_gamma(f64(n + 1))
	if lchoose + math.ln(f64(m)) < 125 * math.LN2 {
		c := _gauss_binom_coeffs(i128, m, n, k)
		s: i128
		for v in c {s += v}
		total := i128(1)
		for i in 1 ..= m {total = total * i128(n + i) / i128(i)}
		return f64(s) / f64(total)
	}
	c := _gauss_binom_coeffs(f64, m, n, k)
	s: f64
	for v in c {s += v}
	return s / math.exp(lchoose)
}

@(private)
_gauss_binom_coeffs :: proc($T: typeid, m, n, k: int) -> []T {
	c := make([]T, k + 1, context.temp_allocator)
	c[0] = 1
	for i in 1 ..= m {
		// times (1 - q^(n+i)), then divided by (1 - q^i)
		for j := k; j >= n + i; j -= 1 {c[j] -= c[j - n - i]}
		for j in i ..= k {c[j] += c[j - i]}
	}
	return c
}

// How wilcoxon_signed_rank treats zero differences.
Zero_Method :: enum {
	Wilcox, // drop them
	Pratt, // rank them with the others, then drop them
	Zsplit, // rank them and split their ranks between the two sums
}

// Wilcoxon signed-rank test: is the distribution of x (or of the
// differences x - y, if y is given) symmetric about zero? The statistic is
// the smaller of the positive and negative rank sums for a two-sided test,
// the positive sum for a one-sided one. Method Auto uses the exact null
// distribution for up to 50 differences without zeros or ties, otherwise the
// normal approximation with tie correction (and a continuity correction if
// `correction`). As scipy's wilcoxon, except that scipy 1.10's Auto stays
// exact with tied differences.
wilcoxon_signed_rank :: proc(
	x: []f64,
	y: []f64 = nil,
	alternative := Alternative.Two_Sided,
	method := Method.Auto,
	zero_method := Zero_Method.Wilcox,
	correction := false,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if y != nil && len(y) != len(x) {return {}, false}
	if len(x) == 0 || !_all_finite(x) || (y != nil && !_all_finite(y)) {return {}, false}
	w.TEMP_GUARD()
	d := make([dynamic]f64, 0, len(x), context.temp_allocator)
	n_zero := 0
	for v, i in x {
		di := v - y[i] if y != nil else v
		if di == 0 {n_zero += 1}
		append(&d, di)
	}
	if n_zero == len(d) && zero_method != .Zsplit {return {}, false}
	m := method
	if m == .Auto {m = .Exact if len(d) <= 50 else .Asymptotic}
	if n_zero > 0 && m == .Exact {m = .Asymptotic}
	if zero_method == .Wilcox && m == .Asymptotic && n_zero > 0 {
		k := 0
		for v in d {
			if v != 0 {
				d[k] = v
				k += 1
			}
		}
		resize(&d, k)
	}
	count := len(d)
	absd := make([]f64, count, context.temp_allocator)
	for v, i in d {absd[i] = abs(v)}
	r := make([]f64, count, context.temp_allocator)
	tie_term := _rank(absd[:], r)
	if method == .Auto && m == .Exact && tie_term != 0 {m = .Asymptotic}
	r_plus, r_minus, r_zero: f64
	for v, i in d {
		if v > 0 {r_plus += r[i]} else if v < 0 {r_minus += r[i]} else {r_zero += r[i]}
	}
	if zero_method == .Zsplit {
		r_plus += r_zero / 2
		r_minus += r_zero / 2
	}
	t := min(r_plus, r_minus) if alternative == .Two_Sided else r_plus
	res.statistic = t
	if m == .Asymptotic {
		cf := f64(count)
		mn := cf * (cf + 1) * 0.25
		se := cf * (cf + 1) * (2 * cf + 1)
		tt := tie_term
		if zero_method == .Pratt {
			// ties among the non-zero ranks only; Cureton's adjustment
			zf := f64(n_zero)
			mn -= zf * (zf + 1) * 0.25
			se -= zf * (zf + 1) * (2 * zf + 1)
			nz := make([dynamic]f64, 0, count, context.temp_allocator)
			for v, i in d {if v != 0 {append(&nz, r[i])}}
			tt = _tie_term_of(nz[:])
		}
		se = math.sqrt((se - 0.5 * tt) / 24)
		cc := 0.0
		if correction {
			switch alternative {
			case .Two_Sided:
				cc = 0.5 * math.sign(t - mn)
			case .Less:
				cc = -0.5
			case .Greater:
				cc = 0.5
			}
		}
		res.p_value = _p_normal((t - mn - cc) / se, alternative)
	} else {
		res.p_value = _wilcoxon_exact_p(count, r_plus, alternative)
	}
	return res, true
}

// Σ (t³ - t) over groups of t equal values
@(private)
_tie_term_of :: proc(v: []f64) -> f64 {
	r := make([]f64, len(v), context.temp_allocator)
	return _rank(v, r)
}

// exact p-value of the positive rank sum of n differences without ties;
// the distribution is that of the sum of a random subset of 1 .. n
@(private)
_wilcoxon_exact_p :: proc(n: int, r_plus_f: f64, alternative: Alternative) -> f64 {
	top := n * (n + 1) / 2
	pmf := make([]f64, top + 1, context.temp_allocator)
	pmf[0] = 1
	hi := 0
	for i in 1 ..= n {
		// times (1 + q^i) / 2
		hi += i
		for j := hi; j >= 0; j -= 1 {
			pmf[j] = 0.5 * (pmf[j] + (pmf[j - i] if j >= i else 0))
		}
	}
	r := int(r_plus_f)
	lower, upper: f64
	for j in 0 ..= r {lower += pmf[j]}
	for j in r ..= top {upper += pmf[j]}
	switch alternative {
	case .Two_Sided:
		if r == top / 2 {return 1}
		return clamp(2 * min(lower, upper), 0, 1)
	case .Greater:
		return clamp(upper, 0, 1)
	case .Less:
		return clamp(lower, 0, 1)
	}
	return math.nan_f64()
}

// Kruskal-Wallis H test: do the samples come from the same distribution?
// H is corrected for ties and compared with chi-squared on k - 1 degrees of
// freedom (scipy's kruskal). Needs two samples, none empty; all values
// equal gives ok = false.
kruskal_wallis :: proc(samples: [][]f64) -> (res: Test_Result, ok: bool) {
	k := len(samples)
	if k < 2 {return {}, false}
	n := 0
	for s in samples {
		if len(s) == 0 || !_all_finite(s) {return {}, false}
		n += len(s)
	}
	w.TEMP_GUARD()
	all := make([]f64, n, context.temp_allocator)
	o := 0
	for s in samples {
		copy(all[o:], s)
		o += len(s)
	}
	ranks := make([]f64, n, context.temp_allocator)
	tie_term := _rank(all, ranks)
	nf := f64(n)
	ties := 1 - tie_term / (nf * nf * nf - nf)
	if ties == 0 {return {}, false}
	ssbn: f64
	o = 0
	for s in samples {
		rs: f64
		for r in ranks[o:o + len(s)] {rs += r}
		ssbn += rs * rs / f64(len(s))
		o += len(s)
	}
	h := (12 / (nf * (nf + 1)) * ssbn - 3 * (nf + 1)) / ties
	res.statistic = h
	res.df = f64(k - 1)
	res.p_value = chi_squared_sf(Chi_Squared{res.df}, h)
	return res, true
}
