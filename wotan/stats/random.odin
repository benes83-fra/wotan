package stats

import "core:math"
import "core:math/rand"

// Random draws. <name>_random(d) gives one value and <name>_sample(d, n)
// a slice of n (free it with delete), from context.random_generator, so a
// seeded generator gives the same values every run. Invalid parameters give
// NaN (continuous) or -1 (discrete).
//
// Methods: the normal from core:math/rand; gamma by Marsaglia and Tsang
// (shape < 1 boosted by U^(1/shape)); chi-squared, beta, t and F from
// gammas and normals; Poisson by multiplication below a mean of 10 and
// Hörmann's PTRS above; binomial by inversion below n·min(p, 1-p) = 10 and
// Hörmann's BTRS above; negative binomial as a Poisson with a gamma mean.

// a uniform in (0, 1]
@(private)
_u01 :: #force_inline proc() -> f64 {
	return 1 - rand.float64()
}

@(private)
_std_gamma :: proc(k: f64) -> f64 {
	if k < 1 {
		return _std_gamma(k + 1) * math.pow(_u01(), 1 / k)
	}
	d := k - 1.0 / 3
	c := 1 / math.sqrt(9 * d)
	for {
		x := rand.norm_float64()
		v := 1 + c * x
		if v <= 0 {continue}
		v = v * v * v
		u := _u01()
		x2 := x * x
		if u < 1 - 0.0331 * x2 * x2 {return d * v}
		if math.ln(u) < 0.5 * x2 + d * (1 - v + math.ln(v)) {return d * v}
	}
}

normal_random :: proc(d: Normal) -> f64 {
	if !_normal_ok(d) {return math.nan_f64()}
	return d.mu + d.sigma * rand.norm_float64()
}

log_normal_random :: proc(d: Log_Normal) -> f64 {
	return math.exp(normal_random(Normal{d.mu, d.sigma}))
}

gamma_random :: proc(d: Gamma) -> f64 {
	if !_gamma_ok(d) {return math.nan_f64()}
	return d.scale * _std_gamma(d.shape)
}

chi_squared_random :: proc(d: Chi_Squared) -> f64 {
	return gamma_random(_chi2_gamma(d))
}

exponential_random :: proc(d: Exponential) -> f64 {
	if !_exp_ok(d) {return math.nan_f64()}
	return -math.ln(_u01()) / d.rate
}

student_t_random :: proc(d: Student_T) -> f64 {
	if !_t_ok(d) {return math.nan_f64()}
	z := rand.norm_float64()
	if math.is_inf(d.df) {return z}
	return z / math.sqrt(2 * _std_gamma(0.5 * d.df) / d.df)
}

beta_random :: proc(d: Beta) -> f64 {
	if !_beta_ok(d) {return math.nan_f64()}
	x := _std_gamma(d.a)
	y := _std_gamma(d.b)
	return x / (x + y)
}

fisher_f_random :: proc(d: Fisher_F) -> f64 {
	if !_f_ok(d) {return math.nan_f64()}
	return (_std_gamma(0.5 * d.d1) / d.d1) / (_std_gamma(0.5 * d.d2) / d.d2)
}

poisson_random :: proc(d: Poisson) -> int {
	if !_pois_ok(d) {return -1}
	return _poisson_draw(d.lambda)
}

@(private)
_poisson_draw :: proc(lam: f64) -> int {
	if lam == 0 {return 0}
	if lam < 10 {
		// multiply uniforms until the product drops below e^-λ
		limit := math.exp(-lam)
		k := 0
		prod := _u01()
		for prod > limit {
			k += 1
			prod *= _u01()
		}
		return k
	}
	// PTRS (Hörmann 1993), as numpy
	slam := math.sqrt(lam)
	loglam := math.ln(lam)
	b := 0.931 + 2.53 * slam
	a := -0.059 + 0.02483 * b
	invalpha := 1.1239 + 1.1328 / (b - 3.4)
	vr := 0.9277 - 3.6224 / (b - 2)
	for {
		u := rand.float64() - 0.5
		v := rand.float64()
		us := 0.5 - abs(u)
		k := math.floor((2 * a / us + b) * u + lam + 0.43)
		if us >= 0.07 && v <= vr {return int(k)}
		if k < 0 || (us < 0.013 && v > us) {continue}
		if math.ln(v) + math.ln(invalpha) - math.ln(a / (us * us) + b) <=
		   -lam + k * loglam - log_gamma(k + 1) {
			return int(k)
		}
	}
}

binomial_random :: proc(d: Binomial) -> int {
	if !_binom_ok(d) {return -1}
	if d.n == 0 || d.p == 0 {return 0}
	if d.p == 1 {return d.n}
	if d.p > 0.5 {return d.n - binomial_random(Binomial{d.n, 1 - d.p})}
	n := f64(d.n)
	p := d.p
	q := 1 - p
	if n * p < 10 {
		// inversion: walk up the probabilities from k = 0
		u := rand.float64()
		k := 0
		pk := math.exp(n * math.log1p(-p))
		r := p / q
		for u > pk && k < d.n {
			u -= pk
			pk *= r * f64(d.n - k) / f64(k + 1)
			k += 1
		}
		return k
	}
	// BTRS (Hörmann 1993)
	spq := math.sqrt(n * p * q)
	b := 1.15 + 2.53 * spq
	a := -0.0873 + 0.0248 * b + 0.01 * p
	c := n * p + 0.5
	alpha := (2.83 + 5.1 / b) * spq
	vr := 0.92 - 4.2 / b
	m := math.floor((n + 1) * p)
	h := log_gamma(m + 1) + log_gamma(n - m + 1)
	lpq := math.ln(p / q)
	for {
		u := rand.float64() - 0.5
		v := rand.float64()
		us := 0.5 - abs(u)
		k := math.floor((2 * a / us + b) * u + c)
		if k < 0 || k > n {continue}
		if us >= 0.07 && v <= vr {return int(k)}
		lv := math.ln(v * alpha / (a / (us * us) + b))
		if lv <= h - log_gamma(k + 1) - log_gamma(n - k + 1) + (k - m) * lpq {return int(k)}
	}
}

negative_binomial_random :: proc(d: Negative_Binomial) -> int {
	if !_nb_ok(d) {return -1}
	if d.p == 1 {return 0}
	return _poisson_draw(_std_gamma(d.n) * (1 - d.p) / d.p)
}

// n draws of a continuous distribution; free with delete
@(private)
_sample_f :: proc(
	n: int,
	d: $D,
	draw: proc(d: D) -> f64,
	allocator := context.allocator,
) -> []f64 {
	out := make([]f64, max(n, 0), allocator)
	for &v in out {v = draw(d)}
	return out
}

@(private)
_sample_i :: proc(
	n: int,
	d: $D,
	draw: proc(d: D) -> int,
	allocator := context.allocator,
) -> []int {
	out := make([]int, max(n, 0), allocator)
	for &v in out {v = draw(d)}
	return out
}

normal_sample :: proc(d: Normal, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, normal_random, allocator)
}
log_normal_sample :: proc(d: Log_Normal, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, log_normal_random, allocator)
}
student_t_sample :: proc(d: Student_T, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, student_t_random, allocator)
}
gamma_sample :: proc(d: Gamma, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, gamma_random, allocator)
}
chi_squared_sample :: proc(d: Chi_Squared, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, chi_squared_random, allocator)
}
exponential_sample :: proc(d: Exponential, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, exponential_random, allocator)
}
beta_sample :: proc(d: Beta, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, beta_random, allocator)
}
fisher_f_sample :: proc(d: Fisher_F, n: int, allocator := context.allocator) -> []f64 {
	return _sample_f(n, d, fisher_f_random, allocator)
}
binomial_sample :: proc(d: Binomial, n: int, allocator := context.allocator) -> []int {
	return _sample_i(n, d, binomial_random, allocator)
}
poisson_sample :: proc(d: Poisson, n: int, allocator := context.allocator) -> []int {
	return _sample_i(n, d, poisson_random, allocator)
}
negative_binomial_sample :: proc(
	d: Negative_Binomial,
	n: int,
	allocator := context.allocator,
) -> []int {
	return _sample_i(n, d, negative_binomial_random, allocator)
}
