package behavioral_finance

import l "../linalg"
import "core:math"

// ============================================================================
// Newswatchers and momentum traders: Hong & Stein (1999)
// ============================================================================
// A risky asset pays D_T = D_0 + sum_t eps_t. News about each innovation diffuses
// across z groups of newswatchers over z periods (each sees one of z subinnovations
// a period, rotating), and newswatchers do not condition on prices, so on their own
// prices underreact: the shock e_t (the innovation that starts to diffuse at t) moves
// the price by 1/z a period for z periods (eq. 1). Momentum traders enter every
// period, hold for j periods and buy phi dP_{t-1} (eq. 3); the newswatchers absorb
// their orders, so (eq. A2, constants dropped)
//   dP_t = (e_t + e_{t-1} + .. + e_{t-z+1}) / z + phi dP_{t-1} - phi dP_{t-j-1}
// an ARMA(j+1, z-1) in the news. In equilibrium phi is the momentum traders' best
// response given the dynamics it creates (eq. 7, risk tolerance gamma):
//   phi = gamma cov(P_{t+j} - P_t, dP_{t-1}) / (var(dP) var(P_{t+j} - P_t | dP_{t-1}))
// with the univariate regression variance var - cov^2 / var(dP). The process is
// covariance stationary when the roots of 1 - phi x + phi x^(j+1) lie outside the unit
// circle (Appendix A.B). Their numerical results (Tables AI - AIII) use the smallest
// stationary fixed point, found by moving up from phi = 0. Prices first underreact,
// then overshoot (Proposition 1: the cumulative impulse response peaks above 1, at
// lag j when j >= z - 1) and with risk-neutral momentum traders (gamma = inf, phi
// solves cov = 0) returns are negatively autocorrelated within j + 1 lags
// (Proposition 2). The rational benchmark P*_t = D_{t+z-1} is a random walk.

Hong_Stein_Params :: struct {
	z:     int, // periods for news to diffuse (z groups)
	j:     int, // momentum traders' horizon
	gamma: f64, // momentum traders' aggregate risk tolerance (math.inf_f64(1): risk neutral)
	sigma: f64, // sd of the news shocks
}

// Their base case (Tables AI - AIII, months): z = j = 12, gamma = 1/3, sigma = 0.5
HONG_STEIN_1999 :: Hong_Stein_Params{12, 12, 1.0 / 3, 0.5}

// Spectral radius of the AR part 1 - phi L + phi L^(j+1) (companion matrix); the
// process is covariance stationary iff it is < 1
hong_stein_ar_radius :: proc(phi: f64, j: int) -> f64 {
	assert(j >= 1, "hong_stein_ar_radius: j >= 1")
	n := j + 1
	C := l.matrix_new(f64, n, n, context.temp_allocator)
	defer l.matrix_free(&C)
	C.data[0] = phi
	C.data[n - 1] = -phi // first row: phi, 0, .., 0, -phi
	for i in 1 ..< n {C.data[i * n + i - 1] = 1}
	return l.eigvals_modulus_max(&C)
}

// (a root within 1e-10 of the unit circle, up to rounding of the eigenvalues, counts
// as not stationary)
hong_stein_stationary :: proc(phi: f64, j: int) -> bool {
	r := hong_stein_ar_radius(phi, j)
	return !math.is_nan(r) && r < 1 - 1e-10
}

// Price changes after a unit shock e_t starting to diffuse at t: h[k] = dP_{t+k}, k <
// n (the cumulative sums are the cumulative impulse response of Tables AI - AIII)
hong_stein_impulse :: proc(
	phi: f64,
	p := HONG_STEIN_1999,
	n: int,
	allocator := context.allocator,
) -> []f64 {
	h := make([]f64, n, allocator)
	for k in 0 ..< n {
		v := 1 / f64(p.z) if k < p.z else 0
		if k >= 1 {v += phi * h[k - 1]}
		if k >= p.j + 1 {v -= phi * h[k - p.j - 1]}
		h[k] = v
	}
	return h
}

// Impulse-response length that captures the process to ~1e-17 (0 if not stationary)
@(private = "file")
_hs_horizon :: proc(phi: f64, p: Hong_Stein_Params) -> int {
	if phi == 0 {return p.z}
	if !hong_stein_stationary(phi, p.j) {return 0}
	r := hong_stein_ar_radius(phi, p.j)
	// h_k ~ k^m r^k (m < j + 1 for repeated roots): 40 e-folds plus a margin
	return p.z + 4 * (p.j + 1) + int(math.ceil(48 / -math.ln(max(r, 1e-300))))
}

// Autocovariances a_0 .. a_max_lag of dP (sum over the impulse response, sigma^2
// sum_m h_m h_{m+k}); NaN when phi is not stationary
hong_stein_autocovariances :: proc(
	phi: f64,
	p := HONG_STEIN_1999,
	max_lag: int,
	allocator := context.allocator,
) -> []f64 {
	a := make([]f64, max_lag + 1, allocator)
	K := _hs_horizon(phi, p)
	if K == 0 {
		for &v in a {v = math.nan_f64()}
		return a
	}
	h := hong_stein_impulse(phi, p, K + max_lag, context.temp_allocator)
	defer delete(h, context.temp_allocator)
	s2 := p.sigma * p.sigma
	for k in 0 ..= max_lag {
		s := 0.0
		for m in 0 ..< K {s += h[m] * h[m + k]}
		a[k] = s2 * s
	}
	return a
}

// sd of the pricing error P_t - P*_t (P* = D_{t+z-1}): sigma sqrt(sum_k (C_k - 1)^2),
// C the cumulative impulse response (phi = 0: sigma sqrt(sum_{i<z} (i/z)^2))
hong_stein_pricing_error_sd :: proc(phi: f64, p := HONG_STEIN_1999) -> f64 {
	K := _hs_horizon(phi, p)
	if K == 0 {return math.nan_f64()}
	h := hong_stein_impulse(phi, p, K, context.temp_allocator)
	defer delete(h, context.temp_allocator)
	c, s := 0.0, 0.0
	for v in h {c += v; s += (c - 1) * (c - 1)}
	return p.sigma * math.sqrt(s)
}

// cov(P_{t+j} - P_t, dP_{t-1}), var(dP) and var(P_{t+j} - P_t | dP_{t-1}) under phi
@(private = "file")
_hs_moments :: proc(phi: f64, p: Hong_Stein_Params) -> (cov, var0, var_m: f64) {
	a := hong_stein_autocovariances(phi, p, p.j + 1, context.temp_allocator)
	defer delete(a, context.temp_allocator)
	for k in 2 ..= p.j + 1 {cov += a[k]}
	v := f64(p.j) * a[0]
	for k in 1 ..< p.j {v += 2 * f64(p.j - k) * a[k]}
	return cov, a[0], v - cov * cov / a[0]
}

// The momentum traders' best response (eq. 7) when the market runs at phi
hong_stein_best_response :: proc(phi: f64, p := HONG_STEIN_1999) -> f64 {
	cov, v0, vm := _hs_moments(phi, p)
	return p.gamma * cov / (v0 * vm)
}

// Equilibrium phi: the smallest phi >= 0 with best response = phi (gamma = inf: cov =
// 0), scanning up from 0 in steps of `step` inside the stationary region, then
// bisecting. ok = false when there is none before the process stops being stationary.
hong_stein_equilibrium :: proc(p := HONG_STEIN_1999, step := 1e-3) -> (phi: f64, ok: bool) {
	assert(
		p.z >= 1 && p.j >= 1 && p.gamma > 0 && p.sigma > 0,
		"hong_stein_equilibrium: bad parameters",
	)
	// same sign as best response - phi (var0 var_m > 0)
	F :: proc(phi: f64, p: Hong_Stein_Params) -> f64 {
		cov, v0, vm := _hs_moments(phi, p)
		if math.is_inf(p.gamma) {return cov}
		return p.gamma * cov - phi * v0 * vm
	}
	lo := 0.0
	flo := F(lo, p)
	if flo <= 0 {return 0, flo == 0}
	for {
		hi := lo + step
		if !hong_stein_stationary(hi, p.j) {return math.nan_f64(), false}
		fhi := F(hi, p)
		if math.is_nan(fhi) {return math.nan_f64(), false}
		if fhi <= 0 {
			for _ in 0 ..< 200 {
				mid := 0.5 * (lo + hi)
				if mid == lo || mid == hi {break}
				if F(mid, p) > 0 {lo = mid} else {hi = mid}
			}
			return 0.5 * (lo + hi), true
		}
		lo = hi
	}
}

// Price changes from news shocks e[t] (the innovation that starts to diffuse at t,
// e.g. sigma N(0, 1)), starting from rest: dP_t = (e_t + .. + e_{t-z+1}) / z +
// phi dP_{t-1} - phi dP_{t-j-1}
hong_stein_simulate :: proc(
	e: []f64,
	phi: f64,
	p := HONG_STEIN_1999,
	allocator := context.allocator,
) -> []f64 {
	n := len(e)
	dP := make([]f64, n, allocator)
	news := 0.0 // e_t + .. + e_{t-z+1}
	for t in 0 ..< n {
		news += e[t]
		if t >= p.z {news -= e[t - p.z]}
		v := news / f64(p.z)
		if t >= 1 {v += phi * dP[t - 1]}
		if t >= p.j + 1 {v -= phi * dP[t - p.j - 1]}
		dP[t] = v
	}
	return dP
}
