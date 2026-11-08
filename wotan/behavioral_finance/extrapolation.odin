package behavioral_finance

import "core:math"
import "core:mem"

// ============================================================================
// Extrapolative expectations: the X-CAPM of Barberis, Greenwood, Jin & Shleifer (2015)
// ============================================================================
// Continuous time. A risk-free asset pays r; the stock market (supply Q) pays a
// dividend whose level is an arithmetic Brownian motion, dD = g_D dt + sigma_D dw.
// Extrapolators (share 1 - mu) expect the price to change at lambda0 + lambda1 S_t
// (eq. 3), sentiment S_t = beta int e^{-beta (t - s)} dP_s a weighted average of past
// price changes (eq. 2); rational traders (share mu) know the true process. Both have
// CARA utility (gamma) and discount at delta. Proposition 1: P_t = A + B S_t + D_t / r,
// sentiment is an Ornstein-Uhlenbeck process
//   dS = -k (S - g_D / r) dt + sigma_S dw,  k = beta / (1 - beta B),
//   sigma_S = beta sigma_D / ((1 - beta B) r),  sigma_P = sigma_D / ((1 - beta B) r),
// demands N^i = eta^i_0 + eta^i_1 S, value functions -exp(-delta t - r gamma W + a S^2
// + b S + c). The coefficients solve the paper's Appendix A system: B from market
// clearing in the S-terms (A34) with a^e, a^r in closed form (A40) and the demand
// slopes (A32); then b^e, b^r (A25, A29, linear in each), A (A33, linear) and c^e, c^r
// (A26, A30). mu = 1 is the rational benchmark P = (g_D - gamma sigma_D^2 Q) / r^2 +
// D / r (Corollary 1). Propositions 2 - 6 give expectations, the predictive power of
// D/r - P (1 - e^{-k t}), the autocorrelation of P - D/r (e^{-k t}), the volatility
// of price changes and of P - D/r, and the (negative) autocorrelation of price
// changes. Time is in years.

XCAPM_Params :: struct {
	r:       f64, // risk-free rate
	g_D:     f64, // mean dividend change per unit time
	sigma_D: f64, // sd of dividend changes per unit time
	Q:       f64, // supply of the risky asset per capita
	gamma:   f64, // absolute risk aversion
	delta:   f64, // time discount rate
	lambda0: f64, // extrapolators expect the price to change at lambda0 + lambda1 S
	lambda1: f64,
	beta:    f64, // weight decay of past price changes in sentiment
	mu:      f64, // share of rational traders
}

// Table 2 with beta = 0.5, mu = 0.25 (the case of their eq. 28)
XCAPM_2015 :: XCAPM_Params{0.025, 0.05, 0.25, 5, 0.1, 0.015, 0, 1, 0.5, 0.25}

XCAPM_Solution :: struct {
	A, B:           f64, // P = A + B S + D / r
	sigma_P:        f64, // sd of price changes per unit time
	k, sigma_S:     f64, // sentiment mean reversion and volatility
	eta_e0, eta_e1: f64, // extrapolators' demand eta0 + eta1 S
	eta_r0, eta_r1: f64, // rational traders' demand
	a_e, b_e, c_e:  f64, // value function coefficients (A19)
	a_r, b_r, c_r:  f64,
}

// The equilibrium of Proposition 1; ok = false when no B in (0, 1 / beta) clears the
// market (see their Corollary 2 for sufficient conditions)
xcapm_solve :: proc(p: XCAPM_Params) -> (s: XCAPM_Solution, ok: bool) {
	assert(
		p.r > 0 && p.gamma > 0 && p.sigma_D > 0 && p.beta > 0 && p.mu >= 0 && p.mu <= 1,
		"xcapm_solve: r, gamma, sigma_D, beta > 0, mu in [0, 1]",
	)
	B := 0.0
	if p.mu < 1 {
		F :: proc(B: f64, p: XCAPM_Params) -> f64 {
			s := _xcapm_slopes(B, p)
			return p.mu * s.eta_r1 + (1 - p.mu) * s.eta_e1
		}
		// the first sign change on a grid over (0, 1 / beta), then bisection
		N :: 4000
		lo, flo := 0.0, F(0, p)
		if !(flo > 0) {return}
		found := false
		hi: f64
		for i in 1 ..< N {
			hi = f64(i) / f64(N) / p.beta
			if F(hi, p) <= 0 {found = true; break}
			lo = hi
		}
		if !found {return}
		for _ in 0 ..< 200 {
			mid := 0.5 * (lo + hi)
			if mid == lo || mid == hi {break}
			if F(mid, p) > 0 {lo = mid} else {hi = mid}
		}
		B = 0.5 * (lo + hi)
	}
	s = _xcapm_slopes(B, p)
	// A: market clearing in the constants (A33), linear in A once b^e, b^r solve
	// (A25), (A29), which are linear in b given A
	lin :: proc(f0, f1: f64) -> f64 {return -f0 / (f1 - f0)} 	// root of a line through (0, f0), (1, f1)
	at_A :: proc(s: ^XCAPM_Solution, A: f64, p: XCAPM_Params) {
		s.A = A
		s.b_e = 0
		f0 := _xcapm_A25(s^, p)
		s.b_e = 1
		s.b_e = lin(f0, _xcapm_A25(s^, p))
		s.b_r = 0
		g0 := _xcapm_A29(s^, p)
		s.b_r = 1
		s.b_r = lin(g0, _xcapm_A29(s^, p))
		_xcapm_levels(s, p)
	}
	clear :: proc(s: ^XCAPM_Solution, A: f64, p: XCAPM_Params) -> f64 {
		at_A(s, A, p)
		return p.mu * s.eta_r0 + (1 - p.mu) * s.eta_e0 - p.Q
	}
	c0 := clear(&s, 0, p)
	at_A(&s, lin(c0, clear(&s, 1, p)), p)
	// c^e, c^r: (A26), (A30), linear
	s.c_e = 0
	e0 := _xcapm_A26(s, p)
	s.c_e = 1
	s.c_e = lin(e0, _xcapm_A26(s, p))
	s.c_r = 0
	r0 := _xcapm_A30(s, p)
	s.c_r = 1
	s.c_r = lin(r0, _xcapm_A30(s, p))
	return s, true
}

// B, sigma_P, k, sigma_S, a^e, a^r (A40) and the demand slopes (A32) at a given B
@(private = "file")
_xcapm_slopes :: proc(B: f64, p: XCAPM_Params) -> (s: XCAPM_Solution) {
	r, b, g := p.r, p.beta, p.gamma
	s.B = B
	s.sigma_P = p.sigma_D / ((1 - b * B) * r)
	s.k = b / (1 - b * B)
	s.sigma_S = b * s.sigma_P
	sp2 := s.sigma_P * s.sigma_P
	den := 2 * p.sigma_D * p.sigma_D * (r + 2 * b - 2 * b * B * r)
	q := (1 - b * B) * (1 - b * B) * r * r / den
	s.a_e = -q * (p.lambda1 - r * B) * (p.lambda1 - r * B)
	s.a_r = -q * B * B * (s.k + r) * (s.k + r)
	s.eta_e1 = (p.lambda1 - r * B) / (r * g * sp2) + 2 * b * s.a_e / (r * g)
	s.eta_r1 = -B / (r * g * sp2) * (s.k + r) + 2 * b * s.a_r / (r * g)
	return
}

// the demand intercepts (A31) given A, b^e, b^r
@(private = "file")
_xcapm_levels :: proc(s: ^XCAPM_Solution, p: XCAPM_Params) {
	r, g := p.r, p.gamma
	sp2 := s.sigma_P * s.sigma_P
	s.eta_e0 = (p.lambda0 - r * s.A) / (r * g * sp2) + p.beta * s.b_e / (r * g)
	s.eta_r0 =
		(p.g_D / ((1 - p.beta * s.B) * r) - r * s.A) / (r * g * sp2) + p.beta * s.b_r / (r * g)
}

// The Bellman-equation conditions of Appendix A as printed (residuals, 0 at the
// solution), with the intercepts recomputed from (A31)
@(private = "file")
_xcapm_A25 :: proc(s: XCAPM_Solution, p: XCAPM_Params) -> f64 {
	q := s
	_xcapm_levels(&q, p)
	r, g, b, sp2 := p.r, p.gamma, p.beta, s.sigma_P * s.sigma_P
	return(
		-r * g * q.eta_e0 * (p.lambda1 - r * s.B) -
		r * g * s.eta_e1 * (p.lambda0 - r * s.A) -
		r * s.b_e +
		r * r * g * g * sp2 * q.eta_e0 * s.eta_e1 +
		2 * b * s.a_e * p.lambda0 +
		b * s.b_e * (p.lambda1 - 1) +
		2 * b * b * sp2 * s.a_e * s.b_e -
		b * r * g * sp2 * (2 * s.a_e * q.eta_e0 + s.b_e * s.eta_e1) \
	)
}

@(private = "file")
_xcapm_A29 :: proc(s: XCAPM_Solution, p: XCAPM_Params) -> f64 {
	q := s
	_xcapm_levels(&q, p)
	r, g, b, sp2 := p.r, p.gamma, p.beta, s.sigma_P * s.sigma_P
	m := p.g_D / ((1 - b * s.B) * r)
	return(
		r * g * s.B * q.eta_r0 * (s.k + r) -
		r * g * s.eta_r1 * (m - r * s.A) -
		r * s.b_r +
		r * r * g * g * sp2 * q.eta_r0 * s.eta_r1 -
		s.k * (s.b_r - 2 * p.g_D * s.a_r / r) +
		2 * b * b * sp2 * s.a_r * s.b_r -
		b * r * g * sp2 * (2 * s.a_r * q.eta_r0 + s.b_r * s.eta_r1) \
	)
}

@(private = "file")
_xcapm_A26 :: proc(s: XCAPM_Solution, p: XCAPM_Params) -> f64 {
	r, g, b, sp2 := p.r, p.gamma, p.beta, s.sigma_P * s.sigma_P
	return(
		r -
		p.delta -
		r * g * s.eta_e0 * (p.lambda0 - r * s.A) -
		r * s.c_e -
		r * math.ln(r * g) +
		0.5 * r * r * g * g * sp2 * s.eta_e0 * s.eta_e0 +
		b * s.b_e * p.lambda0 +
		0.5 * b * b * sp2 * s.b_e * s.b_e +
		b * b * sp2 * s.a_e -
		b * r * g * sp2 * s.b_e * s.eta_e0 \
	)
}

@(private = "file")
_xcapm_A30 :: proc(s: XCAPM_Solution, p: XCAPM_Params) -> f64 {
	r, g, b, sp2 := p.r, p.gamma, p.beta, s.sigma_P * s.sigma_P
	m := p.g_D / ((1 - b * s.B) * r)
	return(
		r -
		p.delta -
		r * g * s.eta_r0 * (m - r * s.A) -
		r * s.c_r -
		r * math.ln(r * g) +
		0.5 * r * r * g * g * sp2 * s.eta_r0 * s.eta_r0 +
		b * p.g_D * s.b_r / ((1 - b * s.B) * r) +
		0.5 * b * b * sp2 * s.b_r * s.b_r +
		b * b * sp2 * s.a_r -
		b * r * g * sp2 * s.b_r * s.eta_r0 \
	)
}

// Residuals of all of Appendix A's conditions at a solution: (A24) - (A26), (A28) -
// (A30) and market clearing (A33), (A34)
xcapm_residuals :: proc(s: XCAPM_Solution, p: XCAPM_Params) -> [8]f64 {
	r, g, b, sp2 := p.r, p.gamma, p.beta, s.sigma_P * s.sigma_P
	A24 :=
		-r * g * s.eta_e1 * (p.lambda1 - r * s.B) -
		r * s.a_e +
		0.5 * r * r * g * g * sp2 * s.eta_e1 * s.eta_e1 +
		2 * b * s.a_e * (p.lambda1 - 1) +
		2 * b * b * sp2 * s.a_e * s.a_e -
		2 * b * r * g * sp2 * s.a_e * s.eta_e1
	A28 :=
		r * g * s.eta_r1 * s.B * (s.k + r) -
		r * s.a_r +
		0.5 * r * r * g * g * sp2 * s.eta_r1 * s.eta_r1 -
		2 * s.k * s.a_r +
		2 * b * b * sp2 * s.a_r * s.a_r -
		2 * b * r * g * sp2 * s.a_r * s.eta_r1
	return {
		A24,
		_xcapm_A25(s, p),
		_xcapm_A26(s, p),
		A28,
		_xcapm_A29(s, p),
		_xcapm_A30(s, p),
		p.mu * s.eta_r0 + (1 - p.mu) * s.eta_e0 - p.Q,
		p.mu * s.eta_r1 + (1 - p.mu) * s.eta_e1,
	}
}

// P = A + B S + D / r
xcapm_price :: proc(s: XCAPM_Solution, p: XCAPM_Params, S, D: f64) -> f64 {
	return s.A + s.B * S + D / p.r
}

// Expected price change over (0, t) given S_0 = S: rational traders (eq. 25) and
// extrapolators (eq. 26)
xcapm_expected_change_rational :: proc(s: XCAPM_Solution, p: XCAPM_Params, S, t: f64) -> f64 {
	return s.B * (1 - math.exp(-s.k * t)) * (p.g_D / p.r - S) + p.g_D * t / p.r
}

xcapm_expected_change_extrapolator :: proc(p: XCAPM_Params, S, t: f64) -> f64 {
	m := p.beta * (1 - p.lambda1)
	if m == 0 {return (p.lambda0 + p.lambda1 * S) * t}
	return(
		(p.lambda0 + p.lambda1 * S) * t +
		p.lambda1 * (p.beta * p.lambda0 - m * S) * (m * t + math.exp(-m * t) - 1) / (m * m) \
	)
}

// Slope of P_t - P_0 on D_0 / r - P_0 (Proposition 3) and the autocorrelation of
// P - D / r at lag t (Proposition 4)
xcapm_dp_coefficient :: proc(s: XCAPM_Solution, t: f64) -> f64 {return 1 - math.exp(-s.k * t)}
xcapm_pd_autocorrelation :: proc(s: XCAPM_Solution, t: f64) -> f64 {return math.exp(-s.k * t)}

// var(P_t - P_0) (Proposition 5, eq. 32 squared) and the sd of P - D / r (eq. 33)
xcapm_price_change_variance :: proc(s: XCAPM_Solution, p: XCAPM_Params, t: f64) -> f64 {
	if s.B == 0 {return p.sigma_D * p.sigma_D / (p.r * p.r) * t}
	return(
		s.B *
			s.sigma_S /
			s.k *
			(s.B * s.sigma_S + 2 * p.sigma_D / p.r) *
			(1 - math.exp(-s.k * t)) +
		p.sigma_D * p.sigma_D / (p.r * p.r) * t \
	)
}

xcapm_pd_sd :: proc(s: XCAPM_Solution) -> f64 {return s.B * s.sigma_S / math.sqrt(2 * s.k)}

// cov(P_{t1} - P_0, P_{t3} - P_{t2}), t2 >= t1 (Proposition 6, eq. 35)
xcapm_price_change_covariance :: proc(s: XCAPM_Solution, p: XCAPM_Params, t1, t2, t3: f64) -> f64 {
	assert(t2 >= t1 && t3 >= t2, "xcapm_price_change_covariance: t1 <= t2 <= t3")
	return(
		-s.B *
		s.sigma_S /
		(2 * s.k) *
		(s.B * s.sigma_S + 2 * p.sigma_D / p.r) *
		(math.exp(-s.k * t2) - math.exp(-s.k * t3)) *
		(math.exp(s.k * t1) - 1) \
	)
}

XCAPM_Path :: struct {
	dividend, sentiment, price: []f64,
	allocator:                  mem.Allocator,
}

xcapm_path_free :: proc(x: ^XCAPM_Path) {
	delete(x.dividend, x.allocator)
	delete(x.sentiment, x.allocator)
	delete(x.price, x.allocator)
}

// The economy on a grid of step dt from D0, S0, driven by standard normals z (two per
// step): exact transitions of the joint Gaussian (D, S), D' = D + g_D dt + sigma_D dW,
// S' = g_D / r + (S - g_D / r) e^{-k dt} + sigma_S int_0^dt e^{-k (dt - u)} dW_u (the
// same Brownian motion; var dt, (1 - e^{-2 k dt}) / (2 k), cov (1 - e^{-k dt}) / k)
xcapm_simulate :: proc(
	s: XCAPM_Solution,
	p: XCAPM_Params,
	D0, S0, dt: f64,
	z: []f64,
	allocator := context.allocator,
) -> XCAPM_Path {
	assert(len(z) % 2 == 0 && dt > 0, "xcapm_simulate: two normals per step, dt > 0")
	n := len(z) / 2
	x := XCAPM_Path {
		dividend  = make([]f64, n + 1, allocator),
		sentiment = make([]f64, n + 1, allocator),
		price     = make([]f64, n + 1, allocator),
		allocator = allocator,
	}
	e := math.exp(-s.k * dt)
	vw, vi, c := dt, (1 - e * e) / (2 * s.k), (1 - e) / s.k
	// dW = sqrt(vw) z1, I = c / sqrt(vw) z1 + sqrt(vi - c^2 / vw) z2
	l11, l21 := math.sqrt(vw), c / math.sqrt(vw)
	l22 := math.sqrt(max(vi - l21 * l21, 0))
	m := p.g_D / p.r
	D, S := D0, S0
	x.dividend[0], x.sentiment[0], x.price[0] = D, S, xcapm_price(s, p, S, D)
	for i in 0 ..< n {
		z1, z2 := z[2 * i], z[2 * i + 1]
		D += p.g_D * dt + p.sigma_D * l11 * z1
		S = m + (S - m) * e + s.sigma_S * (l21 * z1 + l22 * z2)
		x.dividend[i + 1], x.sentiment[i + 1], x.price[i + 1] = D, S, xcapm_price(s, p, S, D)
	}
	return x
}
