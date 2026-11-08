package behavioral_finance

import "core:math"
import "core:mem"

// ============================================================================
// Noise trader risk: De Long, Shleifer, Summers & Waldmann (1990)
// ============================================================================
// Overlapping generations live two periods and choose a portfolio when young. A safe
// asset pays r and is in elastic supply (price 1); an unsafe asset in unit supply pays
// the same dividend r (plus a shock eps, sd sigma_eps, in their Appendix I), price p_t.
// A share mu of the young are noise traders who misperceive the expected price of
// the unsafe asset by rho_t ~ N(rho*, sigma^2) (iid in the basic model; AR(1) with
// persistence phi and innovation sd sigma in Section III); the rest are sophisticated.
// With CARA utility -exp(-2 gamma w) and normal returns the demands are (eqs. 7, 8)
//   lambda_i = (r + E_t p_{t+1} - (1 + r) p_t) / (2 gamma V),  lambda_n = lambda_i +
//   rho_t / (2 gamma V),  V = Var_t(eps + p_{t+1}),
// and (1 - mu) lambda_i + mu lambda_n = 1 gives the steady-state price
//   p_t = 1 + mu (rho_t - rho*) / (1 + r - phi) + mu rho* / r - 2 gamma V / r,
//   V = sigma_eps^2 + mu^2 sigma^2 / (1 + r - phi)^2
// (phi = sigma_eps = 0: eqs. 11, 12; sigma_eps: A4; phi: Section III, eq. 22). The
// noise traders' expected return in excess of the sophisticated investors', per unit
// invested in the unsafe asset at equal wealth (eqs. 17, 18, 23, A5):
//   E_t dR = rho_t - mu rho_t^2 / (2 gamma V),
//   E dR   = rho* - mu (rho*^2 + sigma^2 / (1 - phi^2)) / (2 gamma V)
// "hold more" (rho*) against "price pressure" (rho*^2), "buy high - sell low" (sigma^2)
// and "create space" (V grows with mu). With imitation (eq. 20) the share moves with
// the realized gap, mu_{t+1} = mu_t + alpha dR; E dR = 0 at mu* = (1 + r)^2 (rho*^2 +
// sigma^2) / (2 gamma rho* sigma^2) (eq. 21, unstable: noise traders die out below it
// and take over above it); with fundamental risk a stable share mu_L < mu_H appears
// (Appendix I). The phi and sigma_eps cases combined follow the same derivation.

Noise_Trader_Params :: struct {
	r:         f64, // dividend of both assets = the safe rate
	gamma:     f64, // risk aversion (utility -exp(-2 gamma w))
	mu:        f64, // share of noise traders
	rho_star:  f64, // mean misperception (bullishness)
	sigma:     f64, // sd of the misperception's innovations (of rho itself if phi = 0)
	phi:       f64, // AR(1) persistence of the misperception (0: the basic model)
	sigma_eps: f64, // sd of the dividend shock (0: no fundamental risk)
}

@(private = "file")
_nt_check :: proc(p: Noise_Trader_Params) {
	assert(
		p.r > 0 &&
		p.gamma > 0 &&
		p.mu >= 0 &&
		p.mu <= 1 &&
		p.sigma >= 0 &&
		p.sigma_eps >= 0 &&
		abs(p.phi) < 1,
		"noise trader model: r > 0, gamma > 0, mu in [0, 1], sigma >= 0, |phi| < 1",
	)
}

// dp_t / d rho_t = mu / (1 + r - phi)
dssw_price_slope :: proc(p: Noise_Trader_Params) -> f64 {
	_nt_check(p)
	return p.mu / (1 + p.r - p.phi)
}

// V = Var_t(eps_{t+1} + p_{t+1}): the one-period risk of holding the unsafe asset
dssw_price_risk :: proc(p: Noise_Trader_Params) -> f64 {
	b := dssw_price_slope(p)
	return p.sigma_eps * p.sigma_eps + b * b * p.sigma * p.sigma
}

// The equilibrium price given this generation's misperception rho_t
dssw_price :: proc(rho: f64, p: Noise_Trader_Params) -> f64 {
	return dssw_mean_price(p) + dssw_price_slope(p) * (rho - p.rho_star)
}

// E p = 1 + mu rho* / r - 2 gamma V / r (eq. 24)
dssw_mean_price :: proc(p: Noise_Trader_Params) -> f64 {
	_nt_check(p)
	return 1 + p.mu * p.rho_star / p.r - 2 * p.gamma * dssw_price_risk(p) / p.r
}

// Unconditional variance of p: slope^2 sigma^2 / (1 - phi^2) (eqs. 11, 22)
dssw_price_variance :: proc(p: Noise_Trader_Params) -> f64 {
	b := dssw_price_slope(p)
	return b * b * p.sigma * p.sigma / (1 - p.phi * p.phi)
}

// Holdings of the unsafe asset per sophisticated investor and per noise trader
// (eqs. 7, 8); +-inf when V = 0
dssw_demands :: proc(rho: f64, p: Noise_Trader_Params) -> (sophisticated, noise: f64) {
	V := dssw_price_risk(p)
	x := dssw_expected_excess_return(rho, p)
	return x / (2 * p.gamma * V), (x + rho) / (2 * p.gamma * V)
}

// E_t(r + eps + p_{t+1} - (1 + r) p_t) = 2 gamma V - mu rho_t (eq. 16)
dssw_expected_excess_return :: proc(rho: f64, p: Noise_Trader_Params) -> f64 {
	return 2 * p.gamma * dssw_price_risk(p) - p.mu * rho
}

// E_t dR given rho_t (eq. 17): rho_t - mu rho_t^2 / (2 gamma V)
dssw_conditional_return_gap :: proc(rho: f64, p: Noise_Trader_Params) -> f64 {
	return rho - p.mu * rho * rho / (2 * p.gamma * dssw_price_risk(p))
}

// E dR (eqs. 18, 23, A5)
dssw_expected_return_gap :: proc(p: Noise_Trader_Params) -> f64 {
	m2 := p.rho_star * p.rho_star + p.sigma * p.sigma / (1 - p.phi * p.phi)
	return p.rho_star - p.mu * m2 / (2 * p.gamma * dssw_price_risk(p))
}

// Shares mu where E dR = 0 with the other parameters fixed: the roots of
//   2 gamma rho* s mu^2 - m2 mu + 2 gamma rho* sigma_eps^2 = 0,
// s = sigma^2 / (1 + r - phi)^2, m2 = rho*^2 + sigma^2 / (1 - phi^2); n real roots
// (lo <= hi). sigma_eps = 0: lo = 0 and hi = mu* (eq. 21); with fundamental risk lo =
// mu_L (stable under imitation) and hi = mu_H (Appendix I), or none (noise traders
// always earn more). The roots need not lie in [0, 1].
dssw_critical_shares :: proc(p: Noise_Trader_Params) -> (lo, hi: f64, n: int) {
	q := p
	q.mu = 0
	_nt_check(q)
	s := p.sigma * p.sigma / ((1 + p.r - p.phi) * (1 + p.r - p.phi))
	m2 := p.rho_star * p.rho_star + p.sigma * p.sigma / (1 - p.phi * p.phi)
	A := 2 * p.gamma * p.rho_star * s
	C := 2 * p.gamma * p.rho_star * p.sigma_eps * p.sigma_eps
	if A == 0 {
		if m2 == 0 {return math.nan_f64(), math.nan_f64(), 0}
		x := C / m2
		return x, x, 1
	}
	D := m2 * m2 - 4 * A * C
	if D < 0 {return math.nan_f64(), math.nan_f64(), 0}
	sq := math.sqrt(D)
	// stable form: the root with the larger magnitude first
	big := (m2 + math.copy_sign(sq, m2)) / (2 * A)
	small := C / (A * big) if big != 0 else 0
	return min(big, small), max(big, small), 2
}

DSSW_Path :: struct {
	price:     []f64, // p_t
	mu:        []f64, // noise trader share among the young at t
	excess:    []f64, // r + eps_{t+1} + p_{t+1} - (1 + r) p_t (NaN at the end)
	gap:       []f64, // realized dR of the generation young at t (NaN at the end)
	allocator: mem.Allocator,
}

dssw_path_free :: proc(path: ^DSSW_Path) {
	delete(path.price, path.allocator)
	delete(path.mu, path.allocator)
	delete(path.excess, path.allocator)
	delete(path.gap, path.allocator)
}

// The economy along misperceptions rho[t] and dividend shocks eps[t] (paid at t; nil =
// none), the share starting at p.mu. The price at t uses the share mu_t (with
// imitation, eq. 12'); the generation young at t - 1 sells at p_t and realizes its gap
// dR_{t-1}, judged with the risk V it faced; alpha > 0: imitation (eq. 20), mu_{t+1} =
// clamp(mu_t + alpha dR_{t-1}, 0, 1). The gap is NaN (and the share stays) where V = 0.
dssw_simulate :: proc(
	rho: []f64,
	eps: []f64,
	p: Noise_Trader_Params,
	alpha := 0.0,
	allocator := context.allocator,
) -> DSSW_Path {
	_nt_check(p)
	T := len(rho)
	assert(eps == nil || len(eps) == T, "dssw_simulate: one dividend shock per period")
	path := DSSW_Path {
		price     = make([]f64, T, allocator),
		mu        = make([]f64, T, allocator),
		excess    = make([]f64, T, allocator),
		gap       = make([]f64, T, allocator),
		allocator = allocator,
	}
	q, prev := p, p
	for t in 0 ..< T {
		path.mu[t] = q.mu
		path.price[t] = dssw_price(rho[t], q)
		next := q.mu
		if t >= 1 {
			x :=
				p.r + (eps[t] if eps != nil else 0) + path.price[t] - (1 + p.r) * path.price[t - 1]
			V := dssw_price_risk(prev)
			path.excess[t - 1] = x
			path.gap[t - 1] = rho[t - 1] / (2 * p.gamma * V) * x if V > 0 else math.nan_f64()
			if alpha > 0 && V > 0 {next = clamp(q.mu + alpha * path.gap[t - 1], 0, 1)}
		}
		prev = q
		q.mu = next
	}
	if T > 0 {path.excess[T - 1], path.gap[T - 1] = math.nan_f64(), math.nan_f64()}
	return path
}
