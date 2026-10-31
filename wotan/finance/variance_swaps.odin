package finance

import "core:math"
import "core:slice"

// ============================================================================
// VARIANCE AND VOLATILITY SWAPS
// ============================================================================
// A variance swap pays N_var (σ²_realized − K_var) at maturity; its fair strike
// is the risk-neutral expected annualized variance, replicated statically by
// out-of-the-money options weighted by 1/K² (Demeterfi, Derman, Kamal & Zou
// 1999). A volatility swap pays N_vol (σ_realized − K_vol); E[σ] < √E[σ²], the
// difference being the convexity adjustment. Variances are annualized (σ²).

// Price of a European option of strike K at the swap's maturity (any model:
// Black-Scholes on a smile, Heston COS, ...)
OptionPriceFn :: proc(K: f64, opt: OptionType, data: rawptr) -> f64

// Fair variance from a finite strip of calls (strikes ≥ the boundary S*, the
// lowest call strike) and puts (≤ S*, the highest put strike), as QuantLib's
// ReplicatingVarianceSwapEngine: the log payoff (2/T)((K − S*)/S* − ln(K/S*))
// is matched piecewise linearly between strikes (one extra strike dk beyond
// each end), and
//   K_var = 2/T [(r − q)T − (F/S* − 1) − ln(S*/S)] + e^{rT} Σ w_i O_i,
// F = S e^{(r−q)T} (QuantLib omits q).
variance_swap_replication :: proc(
	S, T, r, q: f64,
	call_strikes, put_strikes: []f64,
	price: OptionPriceFn,
	data: rawptr,
	dk: f64 = 5.0,
) -> f64 {
	log_payoff :: proc(K, f, T: f64) -> f64 {return 2.0 / T * ((K - f) / f - math.ln(K / f))}
	portfolio := 0.0
	boundary := 0.0
	// weights of one wing: slope changes of the piecewise-linear log payoff
	wing :: proc(
		strikes: []f64,
		opt: OptionType,
		T, dk: f64,
		price: OptionPriceFn,
		data: rawptr,
		boundary: ^f64,
		set_boundary: bool,
	) -> f64 {
		if len(strikes) == 0 {return 0}
		ks := make([dynamic]f64, 0, len(strikes) + 1, context.temp_allocator)
		defer delete(ks)
		append(&ks, ..strikes)
		if opt == .Call {
			slice.sort(ks[:])
			append(&ks, ks[len(ks) - 1] + dk)
		} else {
			slice.reverse_sort(ks[:])
			append(&ks, max(ks[len(ks) - 1] - dk, 0.0))
		}
		// drop duplicates
		n := 1
		for i in 1 ..< len(ks) {
			if ks[i] != ks[n - 1] {ks[n] = ks[i]; n += 1}
		}
		f := ks[0]
		if set_boundary {boundary^ = f}
		value, prev := 0.0, 0.0
		for i in 0 ..< n - 1 {
			slope := abs(
				(log_payoff(ks[i + 1], f, T) - log_payoff(ks[i], f, T)) / (ks[i + 1] - ks[i]),
			)
			value += (slope - prev) * price(ks[i], opt, data)
			prev = slope
		}
		return value
	}
	portfolio += wing(call_strikes, .Call, T, dk, price, data, &boundary, true)
	portfolio += wing(put_strikes, .Put, T, dk, price, data, &boundary, len(call_strikes) == 0)
	F := S * math.exp((r - q) * T)
	f := boundary
	return 2.0 / T * ((r - q) * T - (F / f - 1.0) - math.ln(f / S)) + math.exp(r * T) * portfolio
}

// 10-point Gauss-Legendre nodes (positive half) and weights on [−1, 1]
@(private = "file")
_GL_X := [5]f64 {
	0.1488743389816312,
	0.4333953941292472,
	0.6794095682990244,
	0.8650633666889845,
	0.9739065285171717,
}
@(private = "file")
_GL_W := [5]f64 {
	0.2955242247147529,
	0.2692667193099963,
	0.2190863625159820,
	0.1494513491505806,
	0.0666713443086881,
}

// Fair variance from a continuum of strikes (the exact limit of the strip):
//   K_var = (2 e^{rT} / T) [∫_0^F P(K)/K² dK + ∫_F^∞ C(K)/K² dK],
// integrated in log-strike x = ln(K/F) by 10-point Gauss-Legendre on
// n_panels panels per side of the forward (where the integrand has a kink),
// over a range widened until the out-of-the-money prices are negligible
variance_swap_fair_variance :: proc(
	S, T, r, q: f64,
	price: OptionPriceFn,
	data: rawptr,
	n_panels: int = 40,
) -> f64 {
	F := S * math.exp((r - q) * T)
	// O(K)/K² dK = O(F e^x) / (F e^x) dx
	g :: proc(x, F: f64, price: OptionPriceFn, data: rawptr) -> f64 {
		K := F * math.exp(x)
		return price(K, .Put if x < 0 else .Call, data) / K
	}
	atm := g(0, F, price, data)
	lo, hi := -1.0, 1.0
	for lo > -40 && abs(g(lo, F, price, data)) > 1e-15 * atm {lo *= 1.5}
	for hi < 40 && abs(g(hi, F, price, data)) > 1e-15 * atm {hi *= 1.5}
	total := 0.0
	for side in 0 ..< 2 {
		a := lo if side == 0 else 0.0
		b := 0.0 if side == 0 else hi
		h := (b - a) / f64(n_panels)
		for k in 0 ..< n_panels {
			m := a + (f64(k) + 0.5) * h
			for j in 0 ..< 5 {
				total +=
					0.5 *
					h *
					_GL_W[j] *
					(g(m - 0.5 * h * _GL_X[j], F, price, data) +
							g(m + 0.5 * h * _GL_X[j], F, price, data))
			}
		}
	}
	return 2.0 * math.exp(r * T) / T * total
}

// Annualized realized variance of a price series, (A/n) Σ ln²(S_i/S_{i−1})
// (zero-mean convention of variance-swap term sheets; A observations a year)
realized_variance :: proc(prices: []f64, periods_per_year: f64 = 252.0) -> f64 {
	n := len(prices) - 1
	if n < 1 {return 0}
	s := 0.0
	for i in 1 ..= n {
		x := math.ln(prices[i] / prices[i - 1])
		s += x * x
	}
	return periods_per_year * s / f64(n)
}

// Value to the long side of a variance swap at time t into its life:
//   N_var · DF(T) · [(t/T) σ²_realized + ((T − t)/T) K_var(t, T) − K]
// with the fair variance K_var(t, T) of the remaining period
variance_swap_value :: proc(
	notional_var, strike_var, realized_var, elapsed, T, remaining_fair_var, discount: f64,
) -> f64 {
	w := elapsed / T
	return(
		notional_var *
		discount *
		(w * realized_var + (1.0 - w) * remaining_fair_var - strike_var) \
	)
}

// Variance notional of a vega notional (P&L per volatility point at the strike)
variance_notional_from_vega :: proc(vega_notional, strike_vol: f64) -> f64 {
	return vega_notional / (2.0 * strike_vol)
}

// ----------------------------------------------------------------------------
// Heston: expected variance and volatility
// ----------------------------------------------------------------------------

// E[(1/T) ∫_0^T v dt] = θ + (v0 − θ)(1 − e^{−κT})/(κT), the fair variance
heston_fair_variance :: proc(T: f64, p: Heston_Params) -> f64 {
	x := p.kappa * T
	w := (1.0 - math.exp(-x)) / x if x > 1e-8 else 1.0 - 0.5 * x
	return p.theta + (p.v0 - p.theta) * w
}

// Laplace transform E[exp(−s ∫_0^T v dt)] = exp(ln A − B v0) of the integrated
// variance (the CIR bond price with rate s·v), written without the 1/σ²
// cancellation of the textbook form (γ − κ = 2σ²s/(γ + κ)):
//   γ = √(κ² + 2σ²s),  e = e^{−γT},  den = (γ + κ)(1 − e) + 2γ e,
//   B = 2s(1 − e)/den,
//   ln A = 2κθ [−sT/(γ + κ) − ln(1 + y)/σ²],  y = −σ² s(1 − e)/(γ(γ + κ)),
// so σ → 0 is exact (deterministic variance)
heston_integrated_variance_laplace :: proc(s, T: f64, p: Heston_Params) -> f64 {
	return math.exp(_heston_iv_log_laplace(s, T, p))
}

@(private = "file")
_heston_iv_log_laplace :: proc(s, T: f64, p: Heston_Params) -> f64 {
	if s == 0 {return 0}
	k, th, s2 := p.kappa, p.theta, p.sigma * p.sigma
	g := math.sqrt(k * k + 2.0 * s2 * s)
	e := math.exp(-g * T)
	den := (g + k) * (1.0 - e) + 2.0 * g * e
	B := 2.0 * s * (1.0 - e) / den
	Y := -s * (1.0 - e) / (g * (g + k)) // y = σ² Y
	y := s2 * Y
	log_term := Y if y == 0 else Y * math.log1p(y) / y
	lnA := 2.0 * k * th * (-s * T / (g + k) - log_term)
	return lnA - B * p.v0
}

// Fair volatility-swap strike E[√((1/T) ∫_0^T v dt)] by
//   E[√I] = (1/(2√π)) ∫_0^∞ (1 − E[e^{−sI}]) s^{−3/2} ds,
// with s = u²: ∫_0^∞ 2(1 − L(u²))/u² du on [0, u_max] (Gauss-Legendre panels)
// plus the tail 2/u_max where L has vanished
heston_fair_volatility :: proc(T: f64, p: Heston_Params) -> f64 {
	EV := heston_fair_variance(T, p) * T // E[I]
	if p.sigma == 0 {return math.sqrt(EV / T)}
	L :: proc(u, T: f64, p: Heston_Params) -> f64 {return heston_integrated_variance_laplace(
			u * u,
			T,
			p,
		)}
	// 1 − L = −expm1(ln L), free of cancellation for small s
	f :: proc(u, T, EV: f64, p: Heston_Params) -> f64 {
		if u == 0 {return 2.0 * EV}
		return -2.0 * _expm1(_heston_iv_log_laplace(u * u, T, p)) / (u * u)
	}
	// u_max: L(u²) below 1e-17 (it decays like e^{−c u})
	u_max := 1.0 / math.sqrt(EV)
	for L(u_max, T, p) > 1e-17 && u_max < 1e12 {u_max *= 2}
	// 10-point Gauss-Legendre on geometric panels (the integrand varies on the
	// scale 1/√E[I] near 0 and decays like 2/u² beyond)
	total := 0.0
	a := 0.0
	b := 0.05 / math.sqrt(EV)
	for a < u_max {
		b = min(b, u_max)
		m, h := 0.5 * (a + b), 0.5 * (b - a)
		for j in 0 ..< 5 {total += _GL_W[j] * h * (f(m - h * _GL_X[j], T, EV, p) + f(m + h * _GL_X[j], T, EV, p))}
		a, b = b, b * 1.25
	}
	total += 2.0 / u_max
	return total / (2.0 * math.sqrt_f64(math.PI)) / math.sqrt(T)
}
