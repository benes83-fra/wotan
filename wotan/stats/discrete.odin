package stats

import "core:math"

// Discrete distributions, with the same procedures as the continuous ones
// but pmf / logpmf in place of pdf / logpdf. The probabilities use Loader's
// saddle-point form (as R does), which keeps full precision for large counts
// where the factorials of the textbook formula overflow or cancel. cdf and
// sf take any real k (rounded down); ppf(p) is the smallest k with
// cdf(k) >= p and isf(q) the smallest k with sf(k) <= q, as in scipy, so
// ppf(0) is -1 and ppf(1) the largest value (+Inf without one).

// ---------------------------------------------------------------------------
// Loader's helpers

// log(n!) - log(√(2π n) (n/e)^n), the error of Stirling's formula; for
// n > 15 that is the correction of log Γ(n) (log n! = log n + log Γ(n))
@(private)
_stirlerr :: proc(n: f64) -> f64 {
	if n > 15 {return _lgamma_corr(n)}
	if n == 0 {return 0}
	return log_gamma(n + 1) - (n + 0.5) * math.ln(n) + n - LOG_SQRT_2PI
}

// x log(x / m) + m - x, accurate when x is near m
@(private)
_bd0 :: proc(x, m: f64) -> f64 {
	if abs(x - m) < 0.1 * (x + m) {
		v := (x - m) / (x + m)
		s := (x - m) * v
		ej := 2 * x * v
		v2 := v * v
		for j in 1 ..< 1000 {
			ej *= v2
			s1 := s + ej / f64(2 * j + 1)
			if s1 == s {return s1}
			s = s1
		}
		return s
	}
	return x * math.ln(x / m) + m - x
}

// log of the binomial probability of x successes in n trials (both real),
// success probability p, q = 1 - p
@(private)
_log_dbinom :: proc(x, n, p, q: f64) -> f64 {
	if p == 0 {return 0 if x == 0 else math.inf_f64(-1)}
	if q == 0 {return 0 if x == n else math.inf_f64(-1)}
	if x == 0 {
		if n == 0 {return 0}
		return -_bd0(n, n * q) - n * p if p < 0.1 else n * math.log1p(-p)
	}
	if x == n {
		return -_bd0(n, n * p) - n * q if q < 0.1 else n * math.ln(p)
	}
	if x < 0 || x > n {return math.inf_f64(-1)}
	lc := _stirlerr(n) - _stirlerr(x) - _stirlerr(n - x) - _bd0(x, n * p) - _bd0(n - x, n * q)
	lf := math.ln(2 * math.PI) + math.ln(x) + math.log1p(-x / n)
	return lc - 0.5 * lf
}

// smallest k in [lo, hi] with ok(k); ok is monotone in k (false .. true)
@(private)
_Disc_Fn :: #type proc(data: rawptr, k: f64) -> bool

@(private)
_discrete_search :: proc(fn: _Disc_Fn, data: rawptr, guess, lo_in, hi_in: f64) -> f64 {
	lo, hi := lo_in, hi_in
	k := clamp(math.floor(guess), lo, hi if !math.is_inf(hi) else max(lo, math.floor(guess)))
	if math.is_nan(k) {k = lo}
	// gallop from the guess to a bracket [lo, hi] with fn(lo - 1) false, fn(hi) true
	step := 1.0
	if fn(data, k) {
		hi = k
		for {
			nk := max(lo, k - step)
			if nk == k {return k}
			if !fn(data, nk) {
				lo = nk + 1
				break
			}
			hi = nk
			k = nk
			step *= 2
		}
	} else {
		lo = k + 1
		for {
			nk := k + step
			if !math.is_inf(hi) && nk >= hi {break}
			if fn(data, nk) {
				hi = nk
				break
			}
			lo = nk + 1
			k = nk
			step *= 2
		}
	}
	for lo < hi {
		mid := math.floor(0.5 * (lo + hi))
		if fn(data, mid) {hi = mid} else {lo = mid + 1}
	}
	return lo
}

// ---------------------------------------------------------------------------
// Binomial

// The number of successes in `n` independent trials, each a success with
// probability `p`
Binomial :: struct {
	n: int,
	p: f64,
}

@(private)
_binom_ok :: #force_inline proc(d: Binomial) -> bool {
	return d.n >= 0 && d.p >= 0 && d.p <= 1
}

binomial_pmf :: proc(d: Binomial, k: f64) -> f64 {
	return math.exp(binomial_logpmf(d, k))
}

binomial_logpmf :: proc(d: Binomial, k: f64) -> f64 {
	if !_binom_ok(d) || math.is_nan(k) {return math.nan_f64()}
	if k != math.floor(k) || k < 0 || k > f64(d.n) {return math.inf_f64(-1)}
	return _log_dbinom(k, f64(d.n), d.p, 1 - d.p)
}

binomial_cdf :: proc(d: Binomial, k: f64) -> f64 {
	if !_binom_ok(d) || math.is_nan(k) {return math.nan_f64()}
	k := math.floor(k)
	n := f64(d.n)
	if k < 0 {return 0}
	if k >= n {return 1}
	if d.p == 0 {return 1}
	if d.p == 1 {return 0}
	// P(X <= k) = I_{1-p}(n - k, k + 1)
	i, _ := beta_inc_pair(n - k, k + 1, 1 - d.p, d.p)
	return i
}

binomial_sf :: proc(d: Binomial, k: f64) -> f64 {
	if !_binom_ok(d) || math.is_nan(k) {return math.nan_f64()}
	k := math.floor(k)
	n := f64(d.n)
	if k < 0 {return 1}
	if k >= n {return 0}
	if d.p == 0 {return 0}
	if d.p == 1 {return 1}
	_, ic := beta_inc_pair(n - k, k + 1, 1 - d.p, d.p)
	return ic
}

@(private)
Disc_Target :: struct {
	d:      rawptr,
	target: f64,
}

binomial_ppf :: proc(d: Binomial, p: f64) -> f64 {
	if !_binom_ok(d) || math.is_nan(p) || p < 0 || p > 1 {return math.nan_f64()}
	if p == 0 {return -1}
	dd := d
	t := Disc_Target{&dd, p}
	fn :: proc(data: rawptr, k: f64) -> bool {
		t := (^Disc_Target)(data)
		return binomial_cdf((^Binomial)(t.d)^, k) >= t.target
	}
	return _discrete_search(
		fn,
		&t,
		_cf_guess(binomial_mean(d), binomial_variance(d), p),
		0,
		f64(d.n),
	)
}

binomial_isf :: proc(d: Binomial, q: f64) -> f64 {
	if !_binom_ok(d) || math.is_nan(q) || q < 0 || q > 1 {return math.nan_f64()}
	if q == 1 {return -1}
	dd := d
	t := Disc_Target{&dd, q}
	fn :: proc(data: rawptr, k: f64) -> bool {
		t := (^Disc_Target)(data)
		return binomial_sf((^Binomial)(t.d)^, k) <= t.target
	}
	return _discrete_search(
		fn,
		&t,
		_cf_guess(binomial_mean(d), binomial_variance(d), 1 - q),
		0,
		f64(d.n),
	)
}

binomial_mean :: proc(d: Binomial) -> f64 {
	return f64(d.n) * d.p if _binom_ok(d) else math.nan_f64()
}

binomial_variance :: proc(d: Binomial) -> f64 {
	return f64(d.n) * d.p * (1 - d.p) if _binom_ok(d) else math.nan_f64()
}

// a first guess of a quantile from the mean and variance
@(private)
_cf_guess :: proc(mean, variance, p: f64) -> f64 {
	z := _std_normal_ppf(clamp(p, 1e-300, 1 - 1e-16))
	return max(0, math.floor(mean + math.sqrt(variance) * z))
}

// ---------------------------------------------------------------------------
// Poisson

// The Poisson distribution with mean `lambda` >= 0
Poisson :: struct {
	lambda: f64,
}

@(private)
_pois_ok :: #force_inline proc(d: Poisson) -> bool {
	return d.lambda >= 0 && !math.is_inf(d.lambda)
}

poisson_pmf :: proc(d: Poisson, k: f64) -> f64 {
	return math.exp(poisson_logpmf(d, k))
}

poisson_logpmf :: proc(d: Poisson, k: f64) -> f64 {
	if !_pois_ok(d) || math.is_nan(k) {return math.nan_f64()}
	if k != math.floor(k) || k < 0 {return math.inf_f64(-1)}
	if d.lambda == 0 {return 0 if k == 0 else math.inf_f64(-1)}
	if k == 0 {return -d.lambda}
	if math.is_inf(k) {return math.inf_f64(-1)}
	return -_stirlerr(k) - _bd0(k, d.lambda) - 0.5 * math.ln(2 * math.PI * k)
}

poisson_cdf :: proc(d: Poisson, k: f64) -> f64 {
	if !_pois_ok(d) || math.is_nan(k) {return math.nan_f64()}
	k := math.floor(k)
	if k < 0 {return 0}
	if d.lambda == 0 || math.is_inf(k) {return 1}
	return gamma_q(k + 1, d.lambda)
}

poisson_sf :: proc(d: Poisson, k: f64) -> f64 {
	if !_pois_ok(d) || math.is_nan(k) {return math.nan_f64()}
	k := math.floor(k)
	if k < 0 {return 1}
	if d.lambda == 0 || math.is_inf(k) {return 0}
	return gamma_p(k + 1, d.lambda)
}

poisson_ppf :: proc(d: Poisson, p: f64) -> f64 {
	if !_pois_ok(d) || math.is_nan(p) || p < 0 || p > 1 {return math.nan_f64()}
	if p == 0 {return -1}
	if p == 1 {return math.inf_f64(1) if d.lambda > 0 else 0}
	dd := d
	t := Disc_Target{&dd, p}
	fn :: proc(data: rawptr, k: f64) -> bool {
		t := (^Disc_Target)(data)
		return poisson_cdf((^Poisson)(t.d)^, k) >= t.target
	}
	return _discrete_search(fn, &t, _cf_guess(d.lambda, d.lambda, p), 0, math.inf_f64(1))
}

poisson_isf :: proc(d: Poisson, q: f64) -> f64 {
	if !_pois_ok(d) || math.is_nan(q) || q < 0 || q > 1 {return math.nan_f64()}
	if q == 1 {return -1}
	if q == 0 {return math.inf_f64(1) if d.lambda > 0 else 0}
	dd := d
	t := Disc_Target{&dd, q}
	fn :: proc(data: rawptr, k: f64) -> bool {
		t := (^Disc_Target)(data)
		return poisson_sf((^Poisson)(t.d)^, k) <= t.target
	}
	return _discrete_search(fn, &t, _cf_guess(d.lambda, d.lambda, 1 - q), 0, math.inf_f64(1))
}

poisson_mean :: proc(d: Poisson) -> f64 {
	return d.lambda if _pois_ok(d) else math.nan_f64()
}

poisson_variance :: proc(d: Poisson) -> f64 {
	return d.lambda if _pois_ok(d) else math.nan_f64()
}

// ---------------------------------------------------------------------------
// Negative binomial

// The number of failures before the `n`-th success (n > 0, any real) in
// trials that succeed with probability `p` (scipy's nbinom)
Negative_Binomial :: struct {
	n, p: f64,
}

@(private)
_nb_ok :: #force_inline proc(d: Negative_Binomial) -> bool {
	return d.n > 0 && !math.is_inf(d.n) && d.p > 0 && d.p <= 1
}

negative_binomial_pmf :: proc(d: Negative_Binomial, k: f64) -> f64 {
	return math.exp(negative_binomial_logpmf(d, k))
}

negative_binomial_logpmf :: proc(d: Negative_Binomial, k: f64) -> f64 {
	if !_nb_ok(d) || math.is_nan(k) {return math.nan_f64()}
	if k != math.floor(k) || k < 0 || math.is_inf(k) {return math.inf_f64(-1)}
	if d.p == 1 {return 0 if k == 0 else math.inf_f64(-1)}
	if k == 0 {return d.n * math.ln(d.p)}
	// n / (n + k) times the binomial probability of n successes in n + k trials
	return math.ln(d.n / (d.n + k)) + _log_dbinom(d.n, d.n + k, d.p, 1 - d.p)
}

negative_binomial_cdf :: proc(d: Negative_Binomial, k: f64) -> f64 {
	if !_nb_ok(d) || math.is_nan(k) {return math.nan_f64()}
	k := math.floor(k)
	if k < 0 {return 0}
	if d.p == 1 || math.is_inf(k) {return 1}
	i, _ := beta_inc_pair(d.n, k + 1, d.p, 1 - d.p)
	return i
}

negative_binomial_sf :: proc(d: Negative_Binomial, k: f64) -> f64 {
	if !_nb_ok(d) || math.is_nan(k) {return math.nan_f64()}
	k := math.floor(k)
	if k < 0 {return 1}
	if d.p == 1 || math.is_inf(k) {return 0}
	_, ic := beta_inc_pair(d.n, k + 1, d.p, 1 - d.p)
	return ic
}

negative_binomial_ppf :: proc(d: Negative_Binomial, p: f64) -> f64 {
	if !_nb_ok(d) || math.is_nan(p) || p < 0 || p > 1 {return math.nan_f64()}
	if p == 0 {return -1}
	if p == 1 {return math.inf_f64(1) if d.p < 1 else 0}
	dd := d
	t := Disc_Target{&dd, p}
	fn :: proc(data: rawptr, k: f64) -> bool {
		t := (^Disc_Target)(data)
		return negative_binomial_cdf((^Negative_Binomial)(t.d)^, k) >= t.target
	}
	return _discrete_search(
		fn,
		&t,
		_cf_guess(negative_binomial_mean(d), negative_binomial_variance(d), p),
		0,
		math.inf_f64(1),
	)
}

negative_binomial_isf :: proc(d: Negative_Binomial, q: f64) -> f64 {
	if !_nb_ok(d) || math.is_nan(q) || q < 0 || q > 1 {return math.nan_f64()}
	if q == 1 {return -1}
	if q == 0 {return math.inf_f64(1) if d.p < 1 else 0}
	dd := d
	t := Disc_Target{&dd, q}
	fn :: proc(data: rawptr, k: f64) -> bool {
		t := (^Disc_Target)(data)
		return negative_binomial_sf((^Negative_Binomial)(t.d)^, k) <= t.target
	}
	return _discrete_search(
		fn,
		&t,
		_cf_guess(negative_binomial_mean(d), negative_binomial_variance(d), 1 - q),
		0,
		math.inf_f64(1),
	)
}

negative_binomial_mean :: proc(d: Negative_Binomial) -> f64 {
	return d.n * (1 - d.p) / d.p if _nb_ok(d) else math.nan_f64()
}

negative_binomial_variance :: proc(d: Negative_Binomial) -> f64 {
	return d.n * (1 - d.p) / (d.p * d.p) if _nb_ok(d) else math.nan_f64()
}
