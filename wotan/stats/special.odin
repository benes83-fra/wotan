package stats

import "base:intrinsics"
import "core:math"

// Special functions behind the distributions: the regularized incomplete
// gamma and beta functions and their inverses, the log of the beta function,
// expm1 and the inverse error function. The error function, the gamma
// function and its log are Odin's (core:math erf, erfc, gamma, lgamma).
//
// The forms that lose precision in the textbook formulas are computed
// directly: x^a e^-x / Γ(a) and x^a (1-x)^b / B(a, b) through Stirling's
// series for large parameters, each tail of P / Q and I / 1 - I from its own
// expansion (never 1 - the other), and the complement 1 - x passed in where a
// caller knows it exactly.

// ---------------------------------------------------------------------------
// Elementary helpers

// e^x - 1, accurate for small x
expm1 :: proc(x: f64) -> f64 {
	if math.is_nan(x) {return x}
	if abs(x) < 1e-5 {return x + x * x / 2 + x * x * x / 6}
	u := math.exp(x)
	if u == 1 {return x}
	if u - 1 == -1 {return -1}
	if math.is_inf(u) {return u}
	return (u - 1) * x / math.ln(u) // Kahan's correction of the rounding of u
}

// log(1 + x) - x, accurate for small x
log1pmx :: proc(x: f64) -> f64 {
	if abs(x) > 0.5 {return math.log1p(x) - x}
	// log1p(x) = 2 atanh(r), r = x / (2 + x); x - 2r = r x
	r := x / (2 + x)
	r2 := r * r
	term := r * r2
	sum := 0.0
	for k := 3.0; k < 200; k += 2 {
		t := term / k
		sum += t
		if abs(t) <= abs(sum) * 1e-17 {break}
		term *= r2
	}
	return 2 * sum - r * x
}

// log Γ(x) for x > 0 (Odin's lgamma, without the sign)
log_gamma :: proc(x: f64) -> f64 {
	v, _ := math.lgamma(x)
	return v
}

// log Γ(x) - Stirling's approximation, for x >= 10:
// log Γ(x) = (x - 1/2) log x - x + log √(2π) + _lgamma_corr(x)
@(private)
_lgamma_corr :: proc(x: f64) -> f64 {
	r := 1 / x
	r2 := r * r
	return(
		r *
		(1.0 / 12 +
				r2 *
					(-1.0 / 360 +
							r2 *
								(1.0 / 1260 +
										r2 *
											(-1.0 / 1680 +
													r2 *
														(1.0 / 1188 +
																r2 *
																	(-691.0 / 360360 +
																			r2 *
																				(1.0 / 156 +
																						r2 *
																							(-3617.0 /
																									122400)))))))) \
	)
}

@(private)
LOG_SQRT_2PI :: 0.918938533204672741780329736406

// log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b), without the cancellation of
// the three logs for large arguments
log_beta :: proc(a, b: f64) -> f64 {
	if math.is_nan(a) || math.is_nan(b) {return math.nan_f64()}
	p := min(a, b)
	q := max(a, b)
	if p <= 0 {return math.nan_f64()}
	if p >= 10 {
		corr := _lgamma_corr(p) + _lgamma_corr(q) - _lgamma_corr(p + q)
		return(
			-0.5 * math.ln(q) +
			LOG_SQRT_2PI +
			corr +
			(p - 0.5) * math.ln(p / (p + q)) +
			q * math.log1p(-p / (p + q)) \
		)
	}
	if q >= 10 {
		corr := _lgamma_corr(q) - _lgamma_corr(p + q)
		return log_gamma(p) + corr + p - p * math.ln(p + q) + (q - 0.5) * math.log1p(-p / (p + q))
	}
	return log_gamma(p) + log_gamma(q) - log_gamma(p + q)
}

// ---------------------------------------------------------------------------
// Incomplete gamma

// log(x^a e^-x / Γ(a))
@(private)
_gamma_log_prefix :: proc(a, x: f64) -> f64 {
	if x <= 0 {return math.inf_f64(-1)}
	if a >= 10 {
		// log(sqrt(a / 2π)) + a log(x / a) + a - x - corr(a); near x = a the
		// last terms as a log1pmx((x - a) / a), which does not cancel
		d := (x - a) / a
		core := a * log1pmx(d) if abs(d) < 0.5 else a * math.ln(x / a) + (a - x)
		return 0.5 * math.ln(a / (2 * math.PI)) + core - _lgamma_corr(a)
	}
	return a * math.ln(x) - x - log_gamma(a)
}

// x^a e^-x / Γ(a)
@(private)
_gamma_prefix :: proc(a, x: f64) -> f64 {
	return math.exp(_gamma_log_prefix(a, x))
}

// The regularized incomplete gamma functions P(a, x) = γ(a, x) / Γ(a) and
// Q(a, x) = 1 - P(a, x), each to full relative precision
gamma_pq :: proc(a, x: f64) -> (p, q: f64) {
	if math.is_nan(a) || math.is_nan(x) || a <= 0 || x < 0 {
		return math.nan_f64(), math.nan_f64()
	}
	if x == 0 {return 0, 1}
	if math.is_inf(x) {return 1, 0}
	iters := 100 + int(16 * math.sqrt(a))
	if a < 1 && x < 1.1 {
		// small a: Q = 1 - x^a / Γ(a + 1) - x^a / Γ(a + 1) * a Σ_{n≥1} (-x)^n / (n! (a + n))
		lu := a * math.ln(x) - log_gamma(a + 1)
		u := math.exp(lu)
		s := 0.0
		term := 1.0
		for n in 1 ..< 200 {
			term *= -x / f64(n)
			t := term / (a + f64(n))
			s += t
			if abs(t) <= abs(s) * 1e-17 {break}
		}
		q = -expm1(lu) - u * a * s
		p = u * (1 + a * s)
		return
	}
	if x < a + 1 {
		// series for P
		sum := 1.0 / a
		term := sum
		ap := a
		for _ in 0 ..< iters {
			ap += 1
			term *= x / ap
			sum += term
			if term <= sum * 1e-17 {break}
		}
		p = sum * _gamma_prefix(a, x)
		return p, 1 - p
	}
	// Lentz's continued fraction for Q
	FPMIN :: 1e-300
	b := x + 1 - a
	c := 1 / FPMIN
	d := 1 / b
	h := d
	for i in 1 ..= iters {
		an := -f64(i) * (f64(i) - a)
		b += 2
		d = an * d + b
		if abs(d) < FPMIN {d = FPMIN}
		c = b + an / c
		if abs(c) < FPMIN {c = FPMIN}
		d = 1 / d
		delta := d * c
		h *= delta
		if abs(delta - 1) < 1e-16 {break}
	}
	q = h * _gamma_prefix(a, x)
	return 1 - q, q
}

// P(a, x), the lower regularized incomplete gamma function (scipy's gammainc)
gamma_p :: proc(a, x: f64) -> f64 {
	p, _ := gamma_pq(a, x)
	return p
}

// Q(a, x) = 1 - P(a, x), the upper one (scipy's gammaincc)
gamma_q :: proc(a, x: f64) -> f64 {
	_, q := gamma_pq(a, x)
	return q
}

// The x with P(a, x) = p (scipy's gammaincinv)
gamma_p_inv :: proc(a, p: f64) -> f64 {
	return _gamma_inv(a, p, 1 - p, false)
}

// The x with Q(a, x) = q (scipy's gammainccinv); keeps its precision for
// small q, where 1 - q would round
gamma_q_inv :: proc(a, q: f64) -> f64 {
	return _gamma_inv(a, 1 - q, q, true)
}

@(private)
Gamma_Inv :: struct {
	a, target: f64,
	upper:     bool,
}

// log(P) - log(p), or log(q) - log(Q) for the upper tail, so that Newton
// steps keep their size in the far tails; the derivative is the density
// over P (or Q)
@(private)
_gamma_inv_fn :: proc(data: rawptr, x: f64) -> (f, df: f64) {
	g := (^Gamma_Inv)(data)
	p, q := gamma_pq(g.a, x)
	ld := _gamma_log_prefix(g.a, x) - math.ln(x)
	if g.upper {return math.ln(g.target) - math.ln(q), math.exp(ld - math.ln(q))}
	return math.ln(p) - math.ln(g.target), math.exp(ld - math.ln(p))
}

// solves on whichever of P = p, Q = q has the smaller target
@(private)
_gamma_inv :: proc(a, p, q: f64, from_q: bool) -> f64 {
	if math.is_nan(a) ||
	   math.is_nan(p) ||
	   math.is_nan(q) ||
	   a <= 0 ||
	   p < 0 ||
	   q < 0 ||
	   p > 1 ||
	   q > 1 {
		return math.nan_f64()
	}
	if (from_q && q == 1) || (!from_q && p == 0) {return 0}
	if (from_q && q == 0) || (!from_q && p == 1) {return math.INF_F64}
	upper := from_q && q < 0.5 || !from_q && p > 0.5
	// a first guess: Wilson-Hilferty, or the small- and large-x tails
	z := _std_normal_ppf(p) if !upper else -_std_normal_ppf(q)
	x0 := a * math.pow(1 - 1 / (9 * a) + z / (3 * math.sqrt(a)), 3)
	if !(x0 > 0) || a < 1 {
		if !upper || p < 0.5 {
			x0 = math.exp((math.ln(max(p, 1e-300)) + log_gamma(a + 1)) / a)
		} else {
			x0 = -math.ln(q) - log_gamma(a)
			x0 = max(1e-300, -math.ln(q) - log_gamma(a) + (a - 1) * math.ln(max(x0, 1)))
		}
	}
	g := Gamma_Inv {
		a      = a,
		target = q if upper else p,
		upper  = upper,
	}
	return _solve(_gamma_inv_fn, &g, x0, 0, math.INF_F64)
}

// ---------------------------------------------------------------------------
// Incomplete beta

// log(x^a y^b / B(a, b)), y = 1 - x. The smaller of x and y is taken as
// exact and the logarithm of the other as log1p of it: a caller's 1 - x is
// off by a rounding, which x^a would magnify a-fold.
@(private)
_beta_log_prefix :: proc(a, b, x, y: f64) -> f64 {
	if x <= 0 || y <= 0 {return math.inf_f64(-1)}
	lx, ly: f64
	if x <= y {
		lx, ly = math.ln(x), math.log1p(-x)
	} else {
		lx, ly = math.log1p(-y), math.ln(y)
	}
	if a >= 10 && b >= 10 {
		// with x0 = a / c: a log(x / x0) + b log(y / y0) + log(x0^a y0^b / B);
		// n = x c - a (= b - y c) by fused multiply-add from the exact one
		c := a + b
		n1: f64
		if x <= y {
			n1 = intrinsics.fused_mul_add(x, c, -a)
		} else {
			n1 = -intrinsics.fused_mul_add(y, c, -b)
		}
		corr := _lgamma_corr(a) + _lgamma_corr(b) - _lgamma_corr(c)
		r := 0.5 * math.ln(a * b / (2 * math.PI * c)) - corr
		d1 := n1 / a
		d2 := -n1 / b
		// a log1p(d1) + b log1p(d2) with a d1 + b d2 = 0: near the mean as
		// log1pmx, whose linear parts n1 and -n1 cancel and are left out;
		// a term far from the mean from its logarithm instead
		near1 := abs(d1) < 0.5
		near2 := abs(d2) < 0.5
		if near1 && near2 {
			return r + a * log1pmx(d1) + b * log1pmx(d2)
		}
		s1 := a * log1pmx(d1) + n1 if near1 else a * (lx - math.ln(a / c))
		s2 := b * log1pmx(d2) - n1 if near2 else b * (ly - math.ln(b / c))
		return r + s1 + s2
	}
	return a * lx + b * ly - log_beta(a, b)
}

// x^a y^b / B(a, b), y = 1 - x
@(private)
_beta_prefix :: proc(a, b, x, y: f64) -> f64 {
	return math.exp(_beta_log_prefix(a, b, x, y))
}

// I_x(a, b) by the continued fraction of Didonato and Morris, as Boost's
// ibeta_fraction2: written in x and y = 1 - x separately, so that it keeps
// its precision near x = a / (a + b), where the textbook form cancels.
// Converges fast for x below the mean a / (a + b).
@(private)
_beta_cf :: proc(a, b, x, y: f64) -> f64 {
	prefix := _beta_prefix(a, b, x, y)
	if prefix == 0 {return 0}
	TINY :: 1e-300
	iters := 100 + int(16 * math.sqrt(max(a, b)))
	// a y - b x + 1 and 2 - x from whichever of x, y is exact (the smaller),
	// as the prefix does: these terms cancel near the mean
	t, two_x: f64
	if y <= x {
		t = intrinsics.fused_mul_add(a + b, y, 1 - b)
		two_x = 1 + y
	} else {
		t = intrinsics.fused_mul_add(-(a + b), x, a + 1)
		two_x = 2 - x
	}
	f := a * t / (a + 1)
	if f == 0 {f = TINY}
	c := f
	d := 0.0
	for m := 1.0; m <= f64(iters); m += 1 {
		den := a + 2 * m - 1
		an := (a + m - 1) * (a + b + m - 1) * m * (b - m) * x * x / (den * den)
		bn := m + m * (b - m) * x / den + (a + m) * (t + m * two_x) / (a + 2 * m + 1)
		d = bn + an * d
		if d == 0 {d = TINY}
		c = bn + an / c
		if c == 0 {c = TINY}
		d = 1 / d
		delta := c * d
		f *= delta
		if abs(delta - 1) <= 1e-16 {break}
	}
	return prefix / f
}

// I_x(a, b) by its power series (cephes' pseries), for b x <= 1, x <= 0.95:
// x^a / B(a, b) (1/a + Σ_{n≥1} (1-b)(2-b)...(n-b) x^n / (n! (a + n)))
@(private)
_beta_series :: proc(a, b, x, y: f64) -> f64 {
	s := 1 / a
	t := 1.0
	for n := 1.0; n < 10000; n += 1 {
		t *= (n - b) * x / n
		v := t / (a + n)
		s += v
		if abs(v) <= abs(s) * 1e-17 {break}
	}
	lx := math.ln(x) if x <= y else math.log1p(-y)
	return math.exp(a * lx - log_beta(a, b)) * s
}

// I_x(a, b) and 1 - I_x(a, b) with y = 1 - x given exactly; each to full
// relative precision
beta_inc_pair :: proc(a, b, x, y: f64) -> (i, ic: f64) {
	if math.is_nan(a) || math.is_nan(b) || math.is_nan(x) || a <= 0 || b <= 0 || x < 0 || x > 1 {
		return math.nan_f64(), math.nan_f64()
	}
	if x == 0 {return 0, 1}
	if y == 0 {return 1, 0}
	// below the mean the tail I is the small one and is computed directly
	// (series for small b x, else the continued fraction); above it 1 - I,
	// as I_y(b, a)
	below := (a - (a + b) * x > 0) if a < b else ((a + b) * y - b > 0)
	if below {
		i = _beta_series(a, b, x, y) if b * x <= 1 && x <= 0.95 else _beta_cf(a, b, x, y)
		return i, 1 - i
	}
	if a * y <= 1 && y <= 0.95 {
		ic = _beta_series(b, a, y, x)
	} else if b * x <= 1 && x <= 0.95 {
		i = _beta_series(a, b, x, y)
		return i, 1 - i
	} else {
		ic = _beta_cf(b, a, y, x)
	}
	return 1 - ic, ic
}

// I_x(a, b), the regularized incomplete beta function (scipy's betainc)
beta_inc :: proc(a, b, x: f64) -> f64 {
	i, _ := beta_inc_pair(a, b, x, 1 - x)
	return i
}

// 1 - I_x(a, b)
beta_inc_c :: proc(a, b, x: f64) -> f64 {
	_, ic := beta_inc_pair(a, b, x, 1 - x)
	return ic
}

// The x with I_x(a, b) = p (scipy's betaincinv)
beta_inc_inv :: proc(a, b, p: f64) -> f64 {
	x, _ := beta_inc_inv_pair(a, b, p, 1 - p)
	return x
}

@(private)
Beta_Inv :: struct {
	a, b, target: f64,
	upper:        bool, // target is 1 - I_x(a, b)
}

// log(I) - log(p), or log(q) - log(1 - I) for an upper target, with the
// density over I (or 1 - I) as derivative (see _gamma_inv_fn)
@(private)
_beta_inv_fn :: proc(data: rawptr, x: f64) -> (f, df: f64) {
	g := (^Beta_Inv)(data)
	y := 1 - x
	i, ic := beta_inc_pair(g.a, g.b, x, y)
	ld := _beta_log_prefix(g.a, g.b, x, y) - math.ln(x) - math.log1p(-x)
	if g.upper {return math.ln(g.target) - math.ln(ic), math.exp(ld - math.ln(ic))}
	return math.ln(i) - math.ln(g.target), math.exp(ld - math.ln(i))
}

// solves I_x(a, b) = p for x, aiming at the smaller of p and q = 1 - p
@(private)
_beta_solve :: proc(a, b, p, q, x0: f64) -> f64 {
	g := Beta_Inv{a, b, p, false} if p <= q else Beta_Inv{a, b, q, true}
	return _solve(_beta_inv_fn, &g, x0, 0, 1)
}

// The x with I_x(a, b) = p, and y = 1 - x, given p and q = 1 - p: solves
// for whichever of x and y is small, so both keep their precision
beta_inc_inv_pair :: proc(a, b, p, q: f64) -> (x, y: f64) {
	if math.is_nan(a) ||
	   math.is_nan(b) ||
	   math.is_nan(p) ||
	   math.is_nan(q) ||
	   a <= 0 ||
	   b <= 0 ||
	   p < 0 ||
	   p > 1 ||
	   q < 0 ||
	   q > 1 {
		return math.nan_f64(), math.nan_f64()
	}
	if p == 0 {return 0, 1}
	if q == 0 {return 1, 0}
	// first guess (Numerical Recipes' invbetai)
	x0: f64
	if a >= 1 && b >= 1 {
		pp := p if p < 0.5 else q
		t := math.sqrt(-2 * math.ln(pp))
		z := (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t
		if p < 0.5 {z = -z}
		al := (z * z - 3) / 6
		h := 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1))
		w :=
			z * math.sqrt(al + h) / h -
			(1 / (2 * b - 1) - 1 / (2 * a - 1)) * (al + 5.0 / 6 - 2 / (3 * h))
		x0 = a / (a + b * math.exp(2 * w))
	} else {
		lna := math.ln(a / (a + b))
		lnb := math.ln(b / (a + b))
		t := math.exp(a * lna) / a
		u := math.exp(b * lnb) / b
		w := t + u
		if p < t / w {
			x0 = math.pow(a * w * p, 1 / a)
		} else {
			x0 = 1 - math.pow(b * w * q, 1 / b)
		}
	}
	if !(x0 > 0 && x0 < 1) {x0 = 0.5}
	// x solves I_x(a, b) = p and y = 1 - x solves I_y(b, a) = q; solve for
	// whichever is below 1/2, switching if the first guess was on the wrong side
	if x0 <= 0.5 {
		x = _beta_solve(a, b, p, q, x0)
		if x <= 0.5 {return x, 1 - x}
		x0 = x
	}
	// 1 - x solves I_y(b, a) = q
	y = _beta_solve(b, a, q, p, 1 - x0)
	if y > 0.5 {
		// the root is below 1/2 after all
		x = _beta_solve(a, b, p, q, 1 - y)
		return x, 1 - x
	}
	return 1 - y, y
}

// ---------------------------------------------------------------------------
// Root finding

@(private)
_Root_Fn :: #type proc(data: rawptr, x: f64) -> (f, df: f64)

// The root of an increasing f in [lo, hi] (hi may be +Inf; lo >= 0): Newton
// steps from x0, falling back to bisection (geometric across decades) when a
// step leaves the bracket
@(private)
_solve :: proc(fn: _Root_Fn, data: rawptr, x0, lo_in, hi_in: f64) -> f64 {
	lo, hi := lo_in, hi_in
	x := x0
	if !(x > lo && x < hi) {x = lo + 1 if math.is_inf(hi) else 0.5 * (lo + hi)}
	for _ in 0 ..< 400 {
		f, df := fn(data, x)
		if f == 0 {return x}
		if math.is_nan(f) {return f}
		if f < 0 {lo = x} else {hi = x}
		nx := x - f / df
		if !(df > 0) || !(nx > lo && nx < hi) {
			// bisect
			switch {
			case math.is_inf(hi):
				nx = max(2 * x, 1.0)
			case lo == 0:
				nx = hi / 16 if hi > 1e-300 else 0.5 * hi
			case hi / lo > 4:
				nx = math.sqrt(lo) * math.sqrt(hi)
			case:
				nx = 0.5 * (lo + hi)
			}
		}
		if abs(nx - x) <= 2e-16 * abs(x) || nx == lo || nx == hi {
			// converged, or the bracket is down to neighbouring numbers
			if !(nx > lo && nx < hi) {return x}
			return nx
		}
		if !math.is_inf(hi) && hi - lo <= 4e-16 * hi {return 0.5 * (lo + hi)}
		x = nx
	}
	return x
}

// ---------------------------------------------------------------------------
// Inverse error function

// The x with erf(x) = y, for -1 <= y <= 1
erf_inv :: proc(y: f64) -> f64 {
	if math.is_nan(y) || y < -1 || y > 1 {return math.nan_f64()}
	if y == 0 {return y}
	// erf(x) = y  <=>  Φ(x √2) = (1 + y) / 2; the smaller tail keeps precision
	x :=
		_std_normal_ppf(0.5 * (1 + y)) / math.SQRT_TWO if y < 0 else -_std_normal_ppf(0.5 * (1 - y)) / math.SQRT_TWO
	if abs(y) < 0.5 {
		// near 0, 1 ± y rounds: Newton on erf itself
		for _ in 0 ..< 3 {
			x -= (math.erf(x) - y) / (2 / math.sqrt(math.PI) * math.exp(-x * x))
		}
	}
	return x
}

// The x with erfc(x) = y, for 0 <= y <= 2
erfc_inv :: proc(y: f64) -> f64 {
	if math.is_nan(y) || y < 0 || y > 2 {return math.nan_f64()}
	return -_std_normal_ppf(0.5 * y) / math.SQRT_TWO
}
