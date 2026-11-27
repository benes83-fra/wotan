package stats

import w "../core"
import "core:math"

// Tests on counts: chi-squared goodness of fit, tests of independence in a
// contingency table, and Fisher's exact test for 2 x 2 tables.

// The divergence measure of chi_square_test and chi_square_contingency, the
// Cressie-Read family with parameter λ (scipy's power_divergence).
Divergence :: enum {
	Pearson, // λ = 1: Σ (o - e)² / e, the usual chi-squared statistic
	Log_Likelihood, // λ = 0: the G-test, 2 Σ o log(o / e)
	Freeman_Tukey, // λ = -1/2
	Mod_Log_Likelihood, // λ = -1: 2 Σ e log(e / o)
	Neyman, // λ = -2: Σ (o - e)² / o
	Cressie_Read, // λ = 2/3
}

// Chi-squared goodness-of-fit test: do the observed counts follow the
// expected ones? expected = nil means equal frequencies; otherwise it must
// have the same total as observed (relative difference at most 1e-8). ddof
// reduces the degrees of freedom len(observed) - 1, for parameters
// estimated from the data.
chi_square_test :: proc(
	observed: []f64,
	expected: []f64 = nil,
	ddof := 0,
	divergence := Divergence.Pearson,
) -> (
	res: Test_Result,
	ok: bool,
) {
	k := len(observed)
	if k < 1 ||
	   (expected != nil && len(expected) != k) ||
	   !_all_finite(observed) {return {}, false}
	df := k - 1 - ddof
	if df < 1 {return {}, false}
	so: f64
	for o in observed {
		if o < 0 {return {}, false}
		so += o
	}
	mean := so / f64(k)
	if expected != nil {
		if !_all_finite(expected) {return {}, false}
		se: f64
		for e in expected {
			if e <= 0 {return {}, false}
			se += e
		}
		if abs(so - se) / min(so, se) > 1e-8 {return {}, false}
	}
	stat: f64
	for o, i in observed {
		e := expected[i] if expected != nil else mean
		stat += _divergence_term(o, e, divergence)
	}
	res.statistic = stat
	res.df = f64(df)
	res.p_value = chi_squared_sf(Chi_Squared{res.df}, stat)
	return res, true
}

// Chi-squared test of independence of the rows and columns of a contingency
// table of counts (row-major, `cols` columns; crosstab builds one from two
// DataFrame columns). Degrees of freedom (rows - 1)(cols - 1); with one
// degree of freedom (a 2 x 2 table) and `correction`, Yates' correction
// moves each count 0.5 towards its expected value. If `expected` is given
// (rows * cols), it receives the expected counts. Every row and column must
// have a positive total. As scipy's chi2_contingency.
chi_square_contingency :: proc(
	observed: []f64,
	cols: int,
	correction := true,
	divergence := Divergence.Pearson,
	expected: []f64 = nil,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if cols < 1 ||
	   len(observed) == 0 ||
	   len(observed) % cols != 0 ||
	   !_all_finite(observed) {return {}, false}
	rows := len(observed) / cols
	if expected != nil && len(expected) != len(observed) {return {}, false}
	w.TEMP_GUARD()
	rs := make([]f64, rows, context.temp_allocator)
	cs := make([]f64, cols, context.temp_allocator)
	total: f64
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			o := observed[i * cols + j]
			if o < 0 {return {}, false}
			rs[i] += o
			cs[j] += o
			total += o
		}
	}
	for v in rs {if v <= 0 {return {}, false}}
	for v in cs {if v <= 0 {return {}, false}}
	df := (rows - 1) * (cols - 1)
	res.df = f64(df)
	stat: f64
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			e := rs[i] * cs[j] / total
			if expected != nil {expected[i * cols + j] = e}
			o := observed[i * cols + j]
			if df == 1 && correction {
				// Yates: towards the expected count by at most 0.5
				d := e - o
				o += math.sign(d) * min(0.5, abs(d))
			}
			stat += _divergence_term(o, e, divergence)
		}
	}
	if df == 0 {
		res.statistic, res.p_value = 0, 1
		return res, true
	}
	res.statistic = stat
	res.p_value = chi_squared_sf(Chi_Squared{res.df}, stat)
	return res, true
}

@(private)
_divergence_term :: proc(o, e: f64, d: Divergence) -> f64 {
	switch d {
	case .Pearson:
		return (o - e) * (o - e) / e
	case .Log_Likelihood:
		return 0 if o == 0 else 2 * o * math.ln(o / e)
	case .Mod_Log_Likelihood:
		return 2 * e * math.ln(e / o)
	case .Freeman_Tukey:
		return _cressie_read(o, e, -0.5)
	case .Neyman:
		return _cressie_read(o, e, -2)
	case .Cressie_Read:
		return _cressie_read(o, e, 2.0 / 3)
	}
	return math.nan_f64()
}

@(private)
_cressie_read :: proc(o, e, l: f64) -> f64 {
	return o * (math.pow(o / e, l) - 1) / (0.5 * l * (l + 1))
}

// Result of fisher_exact.
Fisher_Result :: struct {
	odds_ratio: f64, // sample odds ratio a d / (b c); +Inf when b c = 0, NaN for a zero row or column total
	p_value:    f64,
}

// Fisher's exact test on the 2 x 2 table {{a, b}, {c, d}}: are rows and
// columns independent? The p-value comes from the hypergeometric
// distribution of the top-left count given the margins; the two-sided one
// adds all tables no more likely than the observed one. Less tests for an
// odds ratio below 1. As scipy's fisher_exact; a table with a zero row or
// column total gives p = 1.
fisher_exact :: proc(
	table: [2][2]int,
	alternative := Alternative.Two_Sided,
) -> (
	res: Fisher_Result,
	ok: bool,
) {
	a, b, c, d := table[0][0], table[0][1], table[1][0], table[1][1]
	if a < 0 || b < 0 || c < 0 || d < 0 {return {}, false}
	if a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0 {
		return {math.nan_f64(), 1}, true
	}
	res.odds_ratio = f64(a) * f64(d) / (f64(c) * f64(b)) if c > 0 && b > 0 else math.inf_f64(1)
	// the top-left count: draws a + c from a + b "row 1" and c + d "row 2" items
	h := _Hyper{f64(a + b), f64(c + d), f64(a + c)}
	x := f64(a)
	switch alternative {
	case .Less:
		res.p_value = _hyper_cdf(h, x)
	case .Greater:
		res.p_value = _hyper_sf(h, x - 1)
	case .Two_Sided:
		n := h.n
		mode := math.floor((n + 1) * (h.r + 1) / (h.r + h.b + 2))
		pexact := _hyper_pmf(h, x)
		pmode := _hyper_pmf(h, mode)
		gamma := 1 + 1e-14
		if abs(pexact - pmode) / max(pexact, pmode) <= 1e-14 {
			res.p_value = 1
		} else if x < mode {
			plower := _hyper_cdf(h, x)
			if _hyper_pmf(h, n) > pexact * gamma {
				res.p_value = plower
			} else {
				guess := _search_increasing(_neg_hyper_pmf, &h, -pexact * gamma, mode, n)
				res.p_value = plower + _hyper_sf(h, guess)
			}
		} else {
			pupper := _hyper_sf(h, x - 1)
			if _hyper_pmf(h, 0) > pexact * gamma {
				res.p_value = pupper
			} else {
				guess := _search_increasing(_hyper_pmf_fn, &h, pexact * gamma, 0, mode)
				res.p_value = pupper + _hyper_cdf(h, guess)
			}
		}
	}
	res.p_value = min(1, res.p_value)
	return res, true
}

// ---------------------------------------------------------------------------
// Hypergeometric distribution: successes in n draws without replacement
// from r "red" and b "black" items. Probabilities as in R (dhyper through
// Loader's binomial terms, phyper summing the smaller tail).

@(private)
_Hyper :: struct {
	r, b, n: f64,
}

@(private)
_hyper_logpmf :: proc(h: _Hyper, x: f64) -> f64 {
	if x < 0 || x > h.r || x > h.n || h.n - x > h.b || x != math.floor(x) {return math.inf_f64(-1)}
	t := h.r + h.b
	p := h.n / t
	q := (t - h.n) / t
	return _log_dbinom(x, h.r, p, q) + _log_dbinom(h.n - x, h.b, p, q) - _log_dbinom(h.n, t, p, q)
}

@(private)
_hyper_pmf :: proc(h: _Hyper, x: f64) -> f64 {return math.exp(_hyper_logpmf(h, x))}

@(private)
_hyper_pmf_fn :: proc(data: rawptr, x: f64) -> f64 {return _hyper_pmf((^_Hyper)(data)^, x)}

@(private)
_neg_hyper_pmf :: proc(data: rawptr, x: f64) -> f64 {return -_hyper_pmf((^_Hyper)(data)^, x)}

// P(X <= x) (lower) or P(X > x)
@(private)
_hyper_tail :: proc(h_in: _Hyper, x_in: f64, lower_in: bool) -> f64 {
	h, lower := h_in, lower_in
	x := math.floor(x_in + 1e-7)
	if x * (h.r + h.b) > h.n * h.r {
		// the other tail is the smaller one: count the black items instead
		h.r, h.b = h.b, h.r
		x = h.n - x - 1
		lower = !lower
	}
	if x < 0 {return 0 if lower else 1}
	if x >= h.r || x >= h.n {return 1 if lower else 0}
	// P(X <= x) / P(X = x) = 1 + Σ of the pmf ratios below x
	sum, term := 0.0, 1.0
	for k := x; k > 0 && term >= 2.220446049250313e-16 * sum; k -= 1 {
		term *= k * (h.b - h.n + k) / (h.n + 1 - k) / (h.r + 1 - k)
		sum += term
	}
	p := _hyper_pmf(h, x) * (1 + sum)
	return p if lower else 1 - p
}

@(private)
_hyper_cdf :: proc(h: _Hyper, x: f64) -> f64 {return _hyper_tail(h, x, true)}

@(private)
_hyper_sf :: proc(h: _Hyper, x: f64) -> f64 {return _hyper_tail(h, x, false)}
