package finance

import "core:math"

// ============================================================================
// FOURIER-COSINE (COS) OPTION PRICING
// ============================================================================
// Fang & Oosterlee (2008): European prices from the characteristic function of
// the log price X = ln S_T. The density of X is expanded in a cosine series on
// [a, b] = c1 ± L·√(c2 + √|c4|) (cumulants of X); a put is
//   DF · K · Σ'_k Re{φ(u_k) e^{−i u_k a}} · V_k(K),   u_k = kπ/(b − a),
// where V_k are the closed-form cosine coefficients of the payoff (1 − e^y)⁺ in
// y = X − ln K. φ does not depend on the strike, so it is evaluated once per
// term for all strikes; calls follow by put-call parity with the model forward
// E[S_T] = φ(−i). The series is extended (doubling) until |φ(u_N)| is
// negligible, so the error is that of the truncation range, ~1e-12 relative.

// E[exp(iu ln S_T)] under the pricing measure; must accept complex u in the
// strip −1 ≤ Im(u) ≤ 0 (the forward and the cumulants use u = −i s)
CharacteristicFunction :: proc(u: complex128, data: rawptr) -> complex128

// Cumulants c1, c2, c4 of X from finite differences of the cumulant generating
// function K(s) = ln E[e^{sX}] = ln φ(−is) around 0
@(private = "file")
_cos_cumulants :: proc(cf: CharacteristicFunction, data: rawptr) -> (c1, c2, c4: f64) {
	h := 0.01
	K :: proc(cf: CharacteristicFunction, data: rawptr, s: f64) -> f64 {
		return math.ln(real(cf(complex(0.0, -s), data)))
	}
	k1, km1 := K(cf, data, h), K(cf, data, -h)
	k2, km2 := K(cf, data, 2 * h), K(cf, data, -2 * h)
	c1 = (8.0 * (k1 - km1) - (k2 - km2)) / (12.0 * h)
	c2 = (16.0 * (k1 + km1) - (k2 + km2)) / (12.0 * h * h) // K(0) = 0
	c4 = (k2 + km2 - 4.0 * (k1 + km1)) / (h * h * h * h)
	return
}

// European prices for every strike, written to `out`; `discount` is the
// discount factor to expiry. L is the half-width of the truncation range in
// standard deviations (20 keeps short-dated jump tails inside the range).
cos_option_prices :: proc(
	cf: CharacteristicFunction,
	data: rawptr,
	strikes: []f64,
	discount: f64,
	opt: OptionType,
	out: []f64,
	L: f64 = 20.0,
	min_terms: int = 64,
	max_terms: int = 1 << 16,
) {
	assert(len(out) == len(strikes), "cos_option_prices: out must have one slot per strike")
	c1, c2, c4 := _cos_cumulants(cf, data)
	half := L * math.sqrt(max(c2, 0.0) + math.sqrt(abs(c4)))
	a, b := c1 - half, c1 + half
	width := b - a
	forward := real(cf(complex(0.0, -1.0), data))

	// number of terms: until the characteristic function has decayed
	n := max(min_terms, 16)
	for n < max_terms && abs(cf(complex(f64(n) * math.PI / width, 0.0), data)) > 1e-16 {n *= 2}

	// Per strike: y = X − ln K on [aK, bK], put payoff over [aK, d], d = min(0, bK);
	// cos / sin of u_k (d − aK) advance by a fixed rotation per term
	Strike :: struct {
		delta, ed, ec: f64, // d − aK, e^d, e^aK
		cr, sr:        f64, // rotation by π (d − aK) / width
		c, s:          f64, // cos, sin of u_k (d − aK)
		sum:           f64,
		live:          bool,
	}
	st := make([]Strike, len(strikes), context.temp_allocator)
	defer delete(st, context.temp_allocator)
	for K, j in strikes {
		lk := math.ln(K)
		aK := a - lk
		d := min(0.0, b - lk)
		if d <= aK {continue} 	// the put is worthless on the whole range
		delta := d - aK
		th := math.PI * delta / width
		st[j] = {delta, math.exp(d), math.exp(aK), math.cos(th), math.sin(th), 1, 0, 0, true}
	}
	for k in 0 ..< n {
		u := f64(k) * math.PI / width
		phi := cf(complex(u, 0.0), data)
		// Re{φ(u) e^{−iua}}, halved for k = 0
		A := real(phi) * math.cos(u * a) + imag(phi) * math.sin(u * a)
		if k == 0 {A *= 0.5}
		for &q in st {
			if !q.live {continue}
			if k > 0 && k % 128 == 0 {q.c, q.s = math.cos(u * q.delta), math.sin(u * q.delta)} 	// re-sync
			chi := (q.c * q.ed - q.ec + u * q.s * q.ed) / (1.0 + u * u)
			psi := q.delta if k == 0 else q.s / u
			q.sum += A * (psi - chi)
			q.c, q.s = q.c * q.cr - q.s * q.sr, q.s * q.cr + q.c * q.sr
		}
	}
	for K, j in strikes {
		put := discount * K * 2.0 / width * st[j].sum
		put = max(put, discount * K - discount * forward, 0.0)
		out[j] =
			put if opt == .Put else max(put + discount * (forward - K), discount * (forward - K), 0.0)
	}
}

// ============================================================================
// CHARACTERISTIC FUNCTIONS
// ============================================================================

@(private = "file")
_Heston_CF :: struct {
	S, r, T: f64,
	p:       Heston_Params,
}

@(private = "file")
_heston_cf :: proc(u: complex128, data: rawptr) -> complex128 {
	c := (^_Heston_CF)(data)
	return heston_char_func(u, c.S, c.r, c.T, c.p)
}

// Heston European prices by COS for strikes sharing S, T and r
heston_prices_cos :: proc(
	S: f64,
	strikes: []f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	opt: OptionType,
	out: []f64,
	min_terms: int = 64,
) {
	if T <= 0.0 {
		for K, k in strikes {out[k] = _payoff_intrinsic(S, K, opt)}
		return
	}
	c := _Heston_CF{S, r, T, params}
	cos_option_prices(_heston_cf, &c, strikes, math.exp(-r * T), opt, out, min_terms = min_terms)
}

// Merton jumps: λT (E[e^{iuJ}] − 1) − iuλκT for lognormal jumps J ~ N(μ_J, σ_J²),
// κ = e^{μ_J + σ_J²/2} − 1 (the compensator keeps E[S_T] = S e^{rT})
@(private = "file")
_merton_jump_exponent :: proc(u: complex128, T, lambda, mu_j, sigma_j: f64) -> complex128 {
	i := complex(0.0, 1.0)
	kappa := math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
	jump := cmath_exp(i * u * complex(mu_j, 0) - complex(0.5 * sigma_j * sigma_j, 0) * u * u)
	return complex(lambda * T, 0) * (jump - 1.0) - i * u * complex(lambda * kappa * T, 0)
}

@(private = "file")
_Merton_CF :: struct {
	S, r, T: f64,
	p:       MJD_Params,
}

@(private = "file")
_merton_cf :: proc(u: complex128, data: rawptr) -> complex128 {
	c := (^_Merton_CF)(data)
	i := complex(0.0, 1.0)
	s2 := c.p.sigma * c.p.sigma
	drift := math.ln(c.S) + (c.r - 0.5 * s2) * c.T
	e := i * u * complex(drift, 0) - complex(0.5 * s2 * c.T, 0) * u * u
	return cmath_exp(e + _merton_jump_exponent(u, c.T, c.p.lambda, c.p.mu_j, c.p.sigma_j))
}

// Merton jump-diffusion European prices by COS (λ = 0 is Black-Scholes)
merton_prices_cos :: proc(
	S: f64,
	strikes: []f64,
	T: f64,
	r: f64,
	params: MJD_Params,
	opt: OptionType,
	out: []f64,
) {
	if T <= 0.0 {
		for K, k in strikes {out[k] = _payoff_intrinsic(S, K, opt)}
		return
	}
	c := _Merton_CF{S, r, T, params}
	cos_option_prices(_merton_cf, &c, strikes, math.exp(-r * T), opt, out)
}

// Bates (1996): Heston stochastic volatility with Merton lognormal jumps
Bates_Params :: struct {
	heston:  Heston_Params,
	lambda:  f64, // jump intensity
	mu_j:    f64, // mean log jump
	sigma_j: f64, // volatility of the log jump
}

@(private = "file")
_Bates_CF :: struct {
	S, r, T: f64,
	p:       Bates_Params,
}

@(private = "file")
_bates_cf :: proc(u: complex128, data: rawptr) -> complex128 {
	c := (^_Bates_CF)(data)
	return(
		heston_char_func(u, c.S, c.r, c.T, c.p.heston) *
		cmath_exp(_merton_jump_exponent(u, c.T, c.p.lambda, c.p.mu_j, c.p.sigma_j)) \
	)
}

bates_prices_cos :: proc(
	S: f64,
	strikes: []f64,
	T: f64,
	r: f64,
	params: Bates_Params,
	opt: OptionType,
	out: []f64,
) {
	if T <= 0.0 {
		for K, k in strikes {out[k] = _payoff_intrinsic(S, K, opt)}
		return
	}
	c := _Bates_CF{S, r, T, params}
	cos_option_prices(_bates_cf, &c, strikes, math.exp(-r * T), opt, out)
}

@(private = "file")
_payoff_intrinsic :: proc(S, K: f64, opt: OptionType) -> f64 {
	return max(S - K, 0.0) if opt == .Call else max(K - S, 0.0)
}
