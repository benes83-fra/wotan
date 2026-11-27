package stats

import "core:math"

// Continuous distributions. Each is a struct of its parameters with
// procedures named after it: <name>_pdf, _logpdf, _cdf, _sf (1 - cdf), _ppf
// (the quantile, inverse of cdf), _isf (inverse of sf), _mean, _variance;
// the groups pdf, cdf, ... in stats.odin take any of them. sf and isf keep
// their precision in the upper tail, where 1 - cdf and ppf(1 - q) round.
// Invalid parameters, and probabilities outside [0, 1], give NaN, as in
// scipy.

// ---------------------------------------------------------------------------
// Normal

// The normal distribution with mean `mu` and standard deviation `sigma` > 0
Normal :: struct {
	mu, sigma: f64,
}

STANDARD_NORMAL :: Normal{0, 1}

// Acklam's rational approximation (relative error ~1e-9) refined by one
// Halley step on Φ(x) - p, with Φ from erfc, which is accurate in the lower
// tail; the upper half by symmetry
@(private)
_std_normal_ppf :: proc(p: f64) -> f64 {
	if math.is_nan(p) || p < 0 || p > 1 {return math.nan_f64()}
	if p == 0 {return math.inf_f64(-1)}
	if p == 1 {return math.inf_f64(1)}
	if p > 0.5 {return -_std_normal_ppf(1 - p)}
	A := [6]f64 {
		-3.969683028665376e+01,
		2.209460984245205e+02,
		-2.759285104469687e+02,
		1.383577518672690e+02,
		-3.066479806614716e+01,
		2.506628277459239e+00,
	}
	B := [5]f64 {
		-5.447609879822406e+01,
		1.615858368580409e+02,
		-1.556989798598866e+02,
		6.680131188771972e+01,
		-1.328068155288572e+01,
	}
	C := [6]f64 {
		-7.784894002430293e-03,
		-3.223964580411365e-01,
		-2.400758277161838e+00,
		-2.549732539343734e+00,
		4.374664141464968e+00,
		2.938163982698783e+00,
	}
	D := [4]f64 {
		7.784695709041462e-03,
		3.224671290700398e-01,
		2.445134137142996e+00,
		3.754408661907416e+00,
	}
	x: f64
	if p < 0.02425 {
		q := math.sqrt(-2 * math.ln(p))
		x =
			(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
			((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1)
	} else {
		q := p - 0.5
		r := q * q
		x =
			(((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) *
			q /
			(((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1)
	}
	// Halley: e = Φ(x) - p, u = e / φ(x)
	if x * x < 1400 {
		e := 0.5 * math.erfc(-x / math.SQRT_TWO) - p
		u := e * math.sqrt(2 * math.PI) * math.exp(0.5 * x * x)
		x = x - u / (1 + 0.5 * x * u)
	}
	return x
}

@(private)
_normal_ok :: #force_inline proc(d: Normal) -> bool {
	return d.sigma > 0 && !math.is_nan(d.mu) && !math.is_inf(d.mu) && !math.is_inf(d.sigma)
}

normal_pdf :: proc(d: Normal, x: f64) -> f64 {
	if !_normal_ok(d) || math.is_nan(x) {return math.nan_f64()}
	return _std_normal_pdf((x - d.mu) / d.sigma) / d.sigma
}

// e^(-z²/2) / √(2π); for large |z| with z split into a short head and a
// tail, so that z² does not round (R's dnorm)
@(private)
_std_normal_pdf :: proc(z: f64) -> f64 {
	z := abs(z)
	if z < 5 {return math.exp(-0.5 * z * z - LOG_SQRT_2PI)}
	hi := math.floor(z * 65536) / 65536
	lo := z - hi
	return math.exp(-0.5 * hi * hi - LOG_SQRT_2PI) * math.exp((-0.5 * lo - hi) * lo)
}

normal_logpdf :: proc(d: Normal, x: f64) -> f64 {
	if !_normal_ok(d) || math.is_nan(x) {return math.nan_f64()}
	z := (x - d.mu) / d.sigma
	return -0.5 * z * z - LOG_SQRT_2PI - math.ln(d.sigma)
}

normal_cdf :: proc(d: Normal, x: f64) -> f64 {
	if !_normal_ok(d) || math.is_nan(x) {return math.nan_f64()}
	return 0.5 * math.erfc(-(x - d.mu) / (d.sigma * math.SQRT_TWO))
}

normal_sf :: proc(d: Normal, x: f64) -> f64 {
	if !_normal_ok(d) || math.is_nan(x) {return math.nan_f64()}
	return 0.5 * math.erfc((x - d.mu) / (d.sigma * math.SQRT_TWO))
}

normal_ppf :: proc(d: Normal, p: f64) -> f64 {
	if !_normal_ok(d) {return math.nan_f64()}
	return d.mu + d.sigma * _std_normal_ppf(p)
}

normal_isf :: proc(d: Normal, q: f64) -> f64 {
	if !_normal_ok(d) {return math.nan_f64()}
	return d.mu - d.sigma * _std_normal_ppf(q)
}

normal_mean :: proc(d: Normal) -> f64 {
	return d.mu if _normal_ok(d) else math.nan_f64()
}

normal_variance :: proc(d: Normal) -> f64 {
	return d.sigma * d.sigma if _normal_ok(d) else math.nan_f64()
}

// ---------------------------------------------------------------------------
// Log-normal

// The distribution of e^X for X normal with mean `mu` and standard deviation
// `sigma` (scipy's lognorm with s = sigma, scale = e^mu)
Log_Normal :: struct {
	mu, sigma: f64,
}

log_normal_pdf :: proc(d: Log_Normal, x: f64) -> f64 {
	if !_normal_ok(Normal{d.mu, d.sigma}) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 0}
	return _log_normal_pdf_direct(d, x)
}

log_normal_logpdf :: proc(d: Log_Normal, x: f64) -> f64 {
	if !_normal_ok(Normal{d.mu, d.sigma}) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return math.inf_f64(-1)}
	lx := math.ln(x)
	return normal_logpdf(Normal{d.mu, d.sigma}, lx) - lx
}

@(private)
_log_normal_pdf_direct :: proc(d: Log_Normal, x: f64) -> f64 {
	return _std_normal_pdf((math.ln(x) - d.mu) / d.sigma) / (d.sigma * x)
}

log_normal_cdf :: proc(d: Log_Normal, x: f64) -> f64 {
	if !_normal_ok(Normal{d.mu, d.sigma}) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 0}
	return normal_cdf(Normal{d.mu, d.sigma}, math.ln(x))
}

log_normal_sf :: proc(d: Log_Normal, x: f64) -> f64 {
	if !_normal_ok(Normal{d.mu, d.sigma}) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 1}
	return normal_sf(Normal{d.mu, d.sigma}, math.ln(x))
}

log_normal_ppf :: proc(d: Log_Normal, p: f64) -> f64 {
	return math.exp(normal_ppf(Normal{d.mu, d.sigma}, p))
}

log_normal_isf :: proc(d: Log_Normal, q: f64) -> f64 {
	return math.exp(normal_isf(Normal{d.mu, d.sigma}, q))
}

log_normal_mean :: proc(d: Log_Normal) -> f64 {
	if !_normal_ok(Normal{d.mu, d.sigma}) {return math.nan_f64()}
	return math.exp(d.mu + 0.5 * d.sigma * d.sigma)
}

log_normal_variance :: proc(d: Log_Normal) -> f64 {
	if !_normal_ok(Normal{d.mu, d.sigma}) {return math.nan_f64()}
	s2 := d.sigma * d.sigma
	return expm1(s2) * math.exp(2 * d.mu + s2)
}

// ---------------------------------------------------------------------------
// Student's t

// Student's t distribution with `df` > 0 degrees of freedom (df = +Inf is
// the standard normal)
Student_T :: struct {
	df: f64,
}

@(private)
_t_ok :: #force_inline proc(d: Student_T) -> bool {
	return d.df > 0
}

// x = df / (df + t²) and y = t² / (df + t²), each without rounding the other
@(private)
_t_xy :: proc(df, t: f64) -> (x, y: f64) {
	at := abs(t)
	r := (df / at) / at // df / t²
	if r < 1 {return r / (1 + r), 1 / (1 + r)}
	s := (at / df) * at // t² / df
	return 1 / (1 + s), s / (1 + s)
}

student_t_pdf :: proc(d: Student_T, x: f64) -> f64 {
	return math.exp(student_t_logpdf(d, x))
}

student_t_logpdf :: proc(d: Student_T, x: f64) -> f64 {
	if !_t_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if math.is_inf(d.df) {return normal_logpdf(STANDARD_NORMAL, x)}
	if math.is_inf(x) {return math.inf_f64(-1)}
	return(
		-0.5 * math.ln(d.df) -
		log_beta(0.5 * d.df, 0.5) -
		0.5 * (d.df + 1) * math.log1p((x / d.df) * x) \
	)
}

student_t_cdf :: proc(d: Student_T, x: f64) -> f64 {
	if !_t_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if math.is_inf(d.df) {return normal_cdf(STANDARD_NORMAL, x)}
	if x == 0 {return 0.5}
	if math.is_inf(x) {return 1 if x > 0 else 0}
	bx, by := _t_xy(d.df, x)
	i, ic := beta_inc_pair(0.5 * d.df, 0.5, bx, by)
	// P(|T| > |x|) = I; the tail is I / 2
	if x < 0 {return 0.5 * i}
	return 0.5 + 0.5 * ic
}

student_t_sf :: proc(d: Student_T, x: f64) -> f64 {
	return student_t_cdf(d, -x)
}

student_t_ppf :: proc(d: Student_T, p: f64) -> f64 {
	if !_t_ok(d) || math.is_nan(p) || p < 0 || p > 1 {return math.nan_f64()}
	if math.is_inf(d.df) {return _std_normal_ppf(p)}
	if p == 0.5 {return 0}
	if p > 0.5 {return -_t_lower(d.df, 1 - p)}
	return _t_lower(d.df, p)
}

student_t_isf :: proc(d: Student_T, q: f64) -> f64 {
	if !_t_ok(d) || math.is_nan(q) || q < 0 || q > 1 {return math.nan_f64()}
	if math.is_inf(d.df) {return -_std_normal_ppf(q)}
	if q == 0.5 {return 0}
	if q > 0.5 {return _t_lower(d.df, 1 - q)}
	return -_t_lower(d.df, q)
}

// the t <= 0 with cdf(t) = p, p <= 0.5: I_x(df/2, 1/2) = 2p with
// x = df / (df + t²), t² = df y / x
@(private)
_t_lower :: proc(df, p: f64) -> f64 {
	if p == 0 {return math.inf_f64(-1)}
	if df == 1 {return -1 / math.tan(math.PI * p)} 	// Cauchy
	x, y := beta_inc_inv_pair(0.5 * df, 0.5, 2 * p, 1 - 2 * p)
	if x == 0 {return math.inf_f64(-1)}
	return -math.sqrt(df * (y / x))
}

// 0, or +Inf for df <= 1 (where the mean does not exist), as scipy
student_t_mean :: proc(d: Student_T) -> f64 {
	if !_t_ok(d) {return math.nan_f64()}
	return 0 if d.df > 1 else math.inf_f64(1)
}

student_t_variance :: proc(d: Student_T) -> f64 {
	if !_t_ok(d) {return math.nan_f64()}
	if math.is_inf(d.df) {return 1}
	if d.df > 2 {return d.df / (d.df - 2)}
	if d.df > 1 {return math.inf_f64(1)}
	return math.nan_f64()
}

// ---------------------------------------------------------------------------
// Gamma, chi-squared, exponential

// The gamma distribution with `shape` > 0 and `scale` > 0 (scipy's gamma with
// a = shape, scale = scale; the rate is 1 / scale)
Gamma :: struct {
	shape, scale: f64,
}

@(private)
_gamma_ok :: #force_inline proc(d: Gamma) -> bool {
	return d.shape > 0 && d.scale > 0 && !math.is_inf(d.shape) && !math.is_inf(d.scale)
}

gamma_pdf :: proc(d: Gamma, x: f64) -> f64 {
	if !_gamma_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x < 0 {return 0}
	if x == 0 {
		if d.shape < 1 {return math.inf_f64(1)}
		return 1 / d.scale if d.shape == 1 else 0
	}
	return math.exp(gamma_logpdf(d, x))
}

gamma_logpdf :: proc(d: Gamma, x: f64) -> f64 {
	if !_gamma_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return math.ln(gamma_pdf(d, x))}
	y := x / d.scale
	if d.shape < 10 {
		return (d.shape - 1) * math.ln(y) - y - log_gamma(d.shape) - math.ln(d.scale)
	}
	// x^(k-1) e^(-x/θ) / (Γ(k) θ^k) = prefix(k, x/θ) / x
	return _gamma_log_prefix(d.shape, y) - math.ln(x)
}

gamma_cdf :: proc(d: Gamma, x: f64) -> f64 {
	if !_gamma_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 0}
	return gamma_p(d.shape, x / d.scale)
}

gamma_sf :: proc(d: Gamma, x: f64) -> f64 {
	if !_gamma_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 1}
	return gamma_q(d.shape, x / d.scale)
}

gamma_ppf :: proc(d: Gamma, p: f64) -> f64 {
	if !_gamma_ok(d) {return math.nan_f64()}
	return d.scale * gamma_p_inv(d.shape, p)
}

gamma_isf :: proc(d: Gamma, q: f64) -> f64 {
	if !_gamma_ok(d) {return math.nan_f64()}
	return d.scale * gamma_q_inv(d.shape, q)
}

gamma_mean :: proc(d: Gamma) -> f64 {
	return d.shape * d.scale if _gamma_ok(d) else math.nan_f64()
}

gamma_variance :: proc(d: Gamma) -> f64 {
	return d.shape * d.scale * d.scale if _gamma_ok(d) else math.nan_f64()
}

// The chi-squared distribution with `df` > 0 degrees of freedom: Gamma with
// shape df/2 and scale 2
Chi_Squared :: struct {
	df: f64,
}

@(private)
_chi2_gamma :: #force_inline proc(d: Chi_Squared) -> Gamma {
	return Gamma{0.5 * d.df, 2} if d.df > 0 else Gamma{math.nan_f64(), 2}
}

chi_squared_pdf :: proc(d: Chi_Squared, x: f64) -> f64 {return gamma_pdf(_chi2_gamma(d), x)}
chi_squared_logpdf :: proc(d: Chi_Squared, x: f64) -> f64 {return gamma_logpdf(_chi2_gamma(d), x)}
chi_squared_cdf :: proc(d: Chi_Squared, x: f64) -> f64 {return gamma_cdf(_chi2_gamma(d), x)}
chi_squared_sf :: proc(d: Chi_Squared, x: f64) -> f64 {return gamma_sf(_chi2_gamma(d), x)}
chi_squared_ppf :: proc(d: Chi_Squared, p: f64) -> f64 {return gamma_ppf(_chi2_gamma(d), p)}
chi_squared_isf :: proc(d: Chi_Squared, q: f64) -> f64 {return gamma_isf(_chi2_gamma(d), q)}
chi_squared_mean :: proc(d: Chi_Squared) -> f64 {return gamma_mean(_chi2_gamma(d))}
chi_squared_variance :: proc(d: Chi_Squared) -> f64 {return gamma_variance(_chi2_gamma(d))}

// The exponential distribution with `rate` λ > 0 (scipy's expon with
// scale = 1 / λ)
Exponential :: struct {
	rate: f64,
}

@(private)
_exp_ok :: #force_inline proc(d: Exponential) -> bool {
	return d.rate > 0 && !math.is_inf(d.rate)
}

exponential_pdf :: proc(d: Exponential, x: f64) -> f64 {
	if !_exp_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x < 0 {return 0}
	return d.rate * math.exp(-d.rate * x)
}

exponential_logpdf :: proc(d: Exponential, x: f64) -> f64 {
	if !_exp_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x < 0 {return math.inf_f64(-1)}
	return math.ln(d.rate) - d.rate * x
}

exponential_cdf :: proc(d: Exponential, x: f64) -> f64 {
	if !_exp_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 0}
	return -expm1(-d.rate * x)
}

exponential_sf :: proc(d: Exponential, x: f64) -> f64 {
	if !_exp_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 1}
	return math.exp(-d.rate * x)
}

exponential_ppf :: proc(d: Exponential, p: f64) -> f64 {
	if !_exp_ok(d) || math.is_nan(p) || p < 0 || p > 1 {return math.nan_f64()}
	return -math.log1p(-p) / d.rate
}

exponential_isf :: proc(d: Exponential, q: f64) -> f64 {
	if !_exp_ok(d) || math.is_nan(q) || q < 0 || q > 1 {return math.nan_f64()}
	return -math.ln(q) / d.rate
}

exponential_mean :: proc(d: Exponential) -> f64 {
	return 1 / d.rate if _exp_ok(d) else math.nan_f64()
}

exponential_variance :: proc(d: Exponential) -> f64 {
	return 1 / (d.rate * d.rate) if _exp_ok(d) else math.nan_f64()
}

// ---------------------------------------------------------------------------
// Beta and F

// The beta distribution on [0, 1] with shapes a > 0 and b > 0
Beta :: struct {
	a, b: f64,
}

@(private)
_beta_ok :: #force_inline proc(d: Beta) -> bool {
	return d.a > 0 && d.b > 0 && !math.is_inf(d.a) && !math.is_inf(d.b)
}

beta_pdf :: proc(d: Beta, x: f64) -> f64 {
	if !_beta_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x < 0 || x > 1 {return 0}
	if x == 0 {
		if d.a < 1 {return math.inf_f64(1)}
		return d.b if d.a == 1 else 0
	}
	if x == 1 {
		if d.b < 1 {return math.inf_f64(1)}
		return d.a if d.b == 1 else 0
	}
	return math.exp(beta_logpdf(d, x))
}

beta_logpdf :: proc(d: Beta, x: f64) -> f64 {
	if !_beta_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 || x >= 1 {return math.ln(beta_pdf(d, x))}
	y := 1 - x
	return _beta_log_prefix(d.a, d.b, x, y) - math.ln(x) - math.log1p(-x)
}

beta_cdf :: proc(d: Beta, x: f64) -> f64 {
	if !_beta_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 0}
	if x >= 1 {return 1}
	i, _ := beta_inc_pair(d.a, d.b, x, 1 - x)
	return i
}

beta_sf :: proc(d: Beta, x: f64) -> f64 {
	if !_beta_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 1}
	if x >= 1 {return 0}
	_, ic := beta_inc_pair(d.a, d.b, x, 1 - x)
	return ic
}

beta_ppf :: proc(d: Beta, p: f64) -> f64 {
	if !_beta_ok(d) {return math.nan_f64()}
	x, _ := beta_inc_inv_pair(d.a, d.b, p, 1 - p)
	return x
}

beta_isf :: proc(d: Beta, q: f64) -> f64 {
	if !_beta_ok(d) {return math.nan_f64()}
	x, _ := beta_inc_inv_pair(d.a, d.b, 1 - q, q)
	return x
}

beta_mean :: proc(d: Beta) -> f64 {
	return d.a / (d.a + d.b) if _beta_ok(d) else math.nan_f64()
}

beta_variance :: proc(d: Beta) -> f64 {
	if !_beta_ok(d) {return math.nan_f64()}
	s := d.a + d.b
	return d.a * d.b / (s * s * (s + 1))
}

// The F distribution with `d1` > 0 and `d2` > 0 degrees of freedom (scipy's
// f with dfn = d1, dfd = d2)
Fisher_F :: struct {
	d1, d2: f64,
}

@(private)
_f_ok :: #force_inline proc(d: Fisher_F) -> bool {
	return d.d1 > 0 && d.d2 > 0 && !math.is_inf(d.d1) && !math.is_inf(d.d2)
}

// x = d1 t / (d1 t + d2), y = d2 / (d1 t + d2)
@(private)
_f_xy :: proc(d: Fisher_F, t: f64) -> (x, y: f64) {
	u := d.d1 * t
	if u > d.d2 {
		r := d.d2 / u
		return 1 / (1 + r), r / (1 + r)
	}
	r := u / d.d2
	return r / (1 + r), 1 / (1 + r)
}

fisher_f_pdf :: proc(d: Fisher_F, x: f64) -> f64 {
	if !_f_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x < 0 {return 0}
	if x == 0 {
		if d.d1 < 2 {return math.inf_f64(1)}
		return 1 if d.d1 == 2 else 0
	}
	return math.exp(fisher_f_logpdf(d, x))
}

fisher_f_logpdf :: proc(d: Fisher_F, x: f64) -> f64 {
	if !_f_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return math.ln(fisher_f_pdf(d, x))}
	if math.is_inf(x) {return math.inf_f64(-1)}
	bx, by := _f_xy(d, x)
	return _beta_log_prefix(0.5 * d.d1, 0.5 * d.d2, bx, by) - math.ln(x)
}

fisher_f_cdf :: proc(d: Fisher_F, x: f64) -> f64 {
	if !_f_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 0}
	if math.is_inf(x) {return 1}
	bx, by := _f_xy(d, x)
	i, _ := beta_inc_pair(0.5 * d.d1, 0.5 * d.d2, bx, by)
	return i
}

fisher_f_sf :: proc(d: Fisher_F, x: f64) -> f64 {
	if !_f_ok(d) || math.is_nan(x) {return math.nan_f64()}
	if x <= 0 {return 1}
	if math.is_inf(x) {return 0}
	bx, by := _f_xy(d, x)
	_, ic := beta_inc_pair(0.5 * d.d1, 0.5 * d.d2, bx, by)
	return ic
}

fisher_f_ppf :: proc(d: Fisher_F, p: f64) -> f64 {
	if !_f_ok(d) {return math.nan_f64()}
	return _f_from_beta(d, p, 1 - p)
}

fisher_f_isf :: proc(d: Fisher_F, q: f64) -> f64 {
	if !_f_ok(d) {return math.nan_f64()}
	return _f_from_beta(d, 1 - q, q)
}

@(private)
_f_from_beta :: proc(d: Fisher_F, p, q: f64) -> f64 {
	x, y := beta_inc_inv_pair(0.5 * d.d1, 0.5 * d.d2, p, q)
	if math.is_nan(x) {return x}
	if y == 0 {return math.inf_f64(1)}
	return d.d2 * x / (d.d1 * y)
}

fisher_f_mean :: proc(d: Fisher_F) -> f64 {
	if !_f_ok(d) || d.d2 <= 2 {return math.nan_f64() if !_f_ok(d) else math.inf_f64(1)}
	return d.d2 / (d.d2 - 2)
}

// +Inf for d2 <= 4, as scipy
fisher_f_variance :: proc(d: Fisher_F) -> f64 {
	if !_f_ok(d) {return math.nan_f64()}
	if d.d2 <= 4 {return math.inf_f64(1)}
	m := d.d2 - 2
	return 2 * d.d2 * d.d2 * (d.d1 + d.d2 - 2) / (d.d1 * m * m * (d.d2 - 4))
}
