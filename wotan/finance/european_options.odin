package finance

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Option Types
// ============================================================================
OptionType :: enum {
	Call,
	Put,
}

// ============================================================================
// Greeks Container
// ============================================================================
Greeks :: struct {
	price: f64,
	delta: f64, // ∂V/∂S
	gamma: f64, // ∂²V/∂S²
	vega:  f64, // ∂V/∂σ  (per 1% move, divide by 100)
	theta: f64, // ∂V/∂T  (per day, divide by 365)
	rho:   f64, // ∂V/∂r  (per 1% move, divide by 100)
}

// ============================================================================
// Black-Scholes Price (as a tensor for autograd)
// ============================================================================
// d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)
// d2 = d1 - σ*√T
// Call = S*N(d1) - K*exp(-r*T)*N(d2)
// Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)
// ============================================================================
black_scholes_price :: proc(
	S_t: ^t.Tensor, // requires_grad = true
	K_t: ^t.Tensor,
	T_t: ^t.Tensor,
	r_t: ^t.Tensor,
	sigma_t: ^t.Tensor,
	opt: OptionType,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	// ln(S/K)
	S_over_K := t.tensor_div(S_t, K_t)
	ln_S_K := t.tensor_log(S_over_K)

	// σ²/2
	sigma_sq := t.tensor_mul(sigma_t, sigma_t)
	half := _scalar_tensor(0.5, allocator)
	half_sig := t.tensor_mul(sigma_sq, half)

	// r + σ²/2
	r_plus_half_sig := t.tensor_add(r_t, half_sig)

	// (r + σ²/2) * T
	numer_term2 := t.tensor_mul(r_plus_half_sig, T_t)

	// numerator = ln(S/K) + (r + σ²/2)*T
	numer := t.tensor_add(ln_S_K, numer_term2)

	// √T
	sqrt_T := t.tensor_sqrt(T_t)

	// σ * √T
	sig_sqrt_T := t.tensor_mul(sigma_t, sqrt_T)

	// d1 = numer / (σ*√T)
	d1 := t.tensor_div(numer, sig_sqrt_T)

	// d2 = d1 - σ*√T
	d2 := t.tensor_sub(d1, sig_sqrt_T)

	// N(d1), N(d2)


	// exp(-r*T)
	neg_r := t.tensor_neg(r_t)
	rT := t.tensor_mul(neg_r, T_t)
	disc := t.tensor_exp(rT)

	// K * exp(-r*T)
	K_disc := t.tensor_mul(K_t, disc)

	if opt == .Call {
		// C = S*N(d1) - K*exp(-r*T)*N(d2)
		N_d1 := t.tensor_norm_cdf(d1)
		N_d2 := t.tensor_norm_cdf(d2)
		term1 := t.tensor_mul(S_t, N_d1)
		term2 := t.tensor_mul(K_disc, N_d2)
		return t.tensor_sub(term1, term2)
	} else {
		// P = K*exp(-r*T)*N(-d2) - S*N(-d1)
		neg_d1 := t.tensor_neg(d1)
		neg_d2 := t.tensor_neg(d2)
		N_neg_d1 := t.tensor_norm_cdf(neg_d1)
		N_neg_d2 := t.tensor_norm_cdf(neg_d2)

		term1 := t.tensor_mul(K_disc, N_neg_d2)
		term2 := t.tensor_mul(S_t, N_neg_d1)
		return t.tensor_sub(term1, term2)
	}
}
// Helper: create a scalar tensor (1x1)
_scalar_tensor :: proc(val: f64, allocator: mem.Allocator) -> ^t.Tensor {
	data := l.matrix_new(f64, 1, 1, allocator)
	data.data[0] = val

	t := t.tensor_new(data, false, allocator)
	t.owned_by_graph = true // ✅ FIX: Tell the graph cleaner to free this internal constant

	return t
}
// ============================================================================
// Closed-form Black-Scholes Price and Greeks
// ============================================================================
// Analytic price, delta, gamma, vega, theta and rho. Vega and rho are per 1%
// move, theta per calendar day (−∂V/∂T / 365). At T ≤ 0 the option is worth its
// payoff; with σ ≤ 0 the forward is certain and the option is worth the
// discounted payoff of the forward.
// ============================================================================
compute_greeks :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
) -> Greeks {
	call := opt == .Call
	if T <= 0.0 {
		if call {return Greeks{price = max(S - K, 0), delta = 1 if S > K else 0}}
		return Greeks{price = max(K - S, 0), delta = -1 if S < K else 0}
	}
	D := math.exp(-r * T)
	if sigma <= 0.0 {
		itm := S > K * D if call else S < K * D
		if !itm {return Greeks{}}
		sgn := 1.0 if call else -1.0
		return Greeks {
			price = sgn * (S - K * D),
			delta = sgn,
			theta = -sgn * r * K * D / 365.0,
			rho = sgn * K * T * D / 100.0,
		}
	}

	sqrt_T := math.sqrt(T)
	sig_sqrt_T := sigma * sqrt_T
	d1 := (math.ln(S / K) + (r + 0.5 * sigma * sigma) * T) / sig_sqrt_T
	d2 := d1 - sig_sqrt_T
	pdf_d1 := _bs_pdf(d1)
	decay := S * pdf_d1 * sigma / (2.0 * sqrt_T) // −∂V/∂T of the volatility term

	g := Greeks {
		gamma = pdf_d1 / (S * sig_sqrt_T),
		vega  = S * pdf_d1 * sqrt_T / 100.0,
	}
	if call {
		Nd1, Nd2 := _bs_cdf(d1), _bs_cdf(d2)
		g.price = S * Nd1 - K * D * Nd2
		g.delta = Nd1
		g.theta = (-decay - r * K * D * Nd2) / 365.0
		g.rho = K * T * D * Nd2 / 100.0
	} else {
		Nm1, Nm2 := _bs_cdf(-d1), _bs_cdf(-d2)
		g.price = K * D * Nm2 - S * Nm1
		g.delta = -Nm1
		g.theta = (-decay + r * K * D * Nm2) / 365.0
		g.rho = -K * T * D * Nm2 / 100.0
	}
	return g
}

// Standard normal CDF, PDF and quantile at full double precision
@(private = "file")
_bs_cdf :: #force_inline proc(x: f64) -> f64 {
	return 0.5 * math.erfc(-x / math.SQRT_TWO)
}

@(private = "file")
_bs_pdf :: #force_inline proc(x: f64) -> f64 {
	return math.exp(-0.5 * x * x) * 0.3989422804014327
}

@(private = "file")
_bs_ppf :: proc(p: f64) -> f64 {
	x := norm_inv(p) // Acklam, ~1e-9 relative
	// One Halley step on Φ(x) − p
	u := (_bs_cdf(x) - p) * 2.5066282746310002 * math.exp(0.5 * x * x)
	return x - u / (1.0 + 0.5 * x * u)
}

// ============================================================================
// Implied Volatility
// ============================================================================
// Newton steps on the log of the closed-form price, safeguarded by a bracket:
// the price increases in σ, so every evaluation narrows [lo, hi], and a step
// that leaves the bracket is replaced by bisection. Starts at the inflection
// point of the price in σ, √(2|ln(S/(K·e^{-rT}))|/T) (Manaster & Koehler).
// Converged when the price error is below `tol` and the
// next step would change σ by less than 1e-10 relative (or the bracket has
// shrunk to rounding). Prices outside the no-arbitrage bounds (intrinsic value
// below; S for a call, K·e^{-rT} for a put above) have no implied volatility and
// return (0, false, 0).
// ============================================================================
implied_volatility :: proc(
	market_price: f64,
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
	max_iter: int = 50,
	tol: f64 = 1e-8,
) -> (
	sigma: f64,
	converged: bool,
	iterations: int,
) {
	if !(T > 0.0 && S > 0.0 && K > 0.0) {return 0, false, 0}
	KD := K * math.exp(-r * T)
	lower := max(S - KD, 0) if opt == .Call else max(KD - S, 0)
	upper := S if opt == .Call else KD
	if !(market_price > lower && market_price < upper) {return 0, false, 0}

	// Solve on the out-of-the-money option, whose price is the time value of the
	// in-the-money one (put-call parity): the errors are the same in price units
	otm := OptionType.Call if S < KD else OptionType.Put
	target := market_price - lower
	ln_target := math.ln(target)

	lo, hi := 0.0, 10.0
	for compute_greeks(S, K, T, r, hi, otm).price < target {
		lo, hi = hi, 2.0 * hi
		if hi > 1e6 {return lo, false, 0}
	}

	sigma = math.sqrt(2.0 * math.abs(math.ln(S / KD)) / T)
	if sigma <= 0.0 {sigma = math.sqrt(2.0 * math.PI / T) * market_price / S} 	// at the money forward
	if !(sigma > lo && sigma < hi) {sigma = 0.5 * (lo + hi)}

	for iter in 1 ..= max_iter {
		g := compute_greeks(S, K, T, r, sigma, otm)
		err := g.price - target
		if err > 0 {hi = sigma} else {lo = sigma}
		// Newton on ln V: close to linear in 1/σ², where V itself is far too
		// steep for Newton when the price is small
		vega := 100.0 * g.vega
		step := (math.ln(g.price) - ln_target) * g.price / vega if vega > 0 && g.price > 0 else math.INF_F64
		if err == 0 ||
		   math.abs(err) <= tol && (math.abs(step) <= 1e-10 * sigma || hi - lo <= 1e-14 * hi) {
			return sigma, true, iter
		}
		next := sigma - step
		if !(next > lo && next < hi) {next = 0.5 * (lo + hi)}
		sigma = next
	}
	return sigma, false, max_iter
}

// ============================================================================
// Convenience: Price + Greeks in one call
// ============================================================================
price_and_greeks :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	greeks: Greeks,
) {
	greeks = compute_greeks(S, K, T, r, sigma, opt, allocator)
	return greeks.price, greeks
}
// ============================================================================
// GARCH-Powered Option Pricing
// ============================================================================

// Forecast the average annualized volatility over a given horizon using GARCH(1,1)
// This creates a dynamic volatility term structure, replacing the flat BS assumption.
garch_term_structure_vol :: proc(
	omega: f64,
	alpha: f64,
	beta: f64,
	current_var: f64, // Today's conditional variance (e.g., garch_result.conditional_var[last])
	horizon_days: int,
) -> f64 {
	if horizon_days <= 0 {return math.sqrt_f64(current_var) * math.sqrt_f64(252.0)}

	persistence := alpha + beta
	if persistence >= 1.0 {persistence = 0.999} 	// Prevent explosion

	long_run_var := omega / (1.0 - persistence)

	// Sum of expected variances over the horizon
	sum_var := 0.0
	for h in 1 ..< horizon_days + 1 {
		// E[σ²_{t+h}] = V_L + (α+β)^{h-1} * (σ²_t - V_L)
		expected_var :=
			long_run_var + math.pow(persistence, f64(h - 1)) * (current_var - long_run_var)
		sum_var += expected_var
	}

	// Average daily variance over the horizon
	avg_daily_var := sum_var / f64(horizon_days)

	// Annualize
	return math.sqrt_f64(avg_daily_var * 252.0)
}

// Price an option and compute Greeks using GARCH-forecasted volatility
garch_price_and_greeks :: proc(
	S: f64,
	K: f64,
	T_years: f64,
	r: f64,
	omega: f64,
	alpha: f64,
	beta: f64,
	current_var: f64,
	opt: OptionType,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	greeks: Greeks,
) {
	// 1. Calculate horizon in days (e.g., 0.25 years = 63 days)
	horizon_days := int(math.max(1.0, T_years * 252.0))

	// 2. Get the GARCH-implied volatility for this specific maturity
	sigma_garch := garch_term_structure_vol(omega, alpha, beta, current_var, horizon_days)

	// 3. Plug into your existing autograd Black-Scholes engine
	return price_and_greeks(S, K, T_years, r, sigma_garch, opt, allocator)
}
// ============================================================================
// Monte Carlo Pricers with Pathwise Greeks
// ============================================================================
// Geometric Brownian motion under the risk-neutral measure on the grid
// t_i = i·T/n_steps, i = 1..n_steps. Averages and barrier monitoring use
// S_1..S_n (not S_0); the lookback maximum includes S_0.
//
// Delta (∂V/∂S_0) and vega (∂V/∂σ) are exact pathwise derivatives of each
// simulated payoff, accumulated path by path, so memory is O(1) in paths and
// steps. Normals are drawn path by path, step by step.
//
// Barrier options use one-step survival (Glasserman & Staum 2001): every step is
// drawn conditional on staying below the barrier and the path carries the
// product of the survival probabilities as its weight. This is unbiased for the
// discretely monitored barrier and has lower variance. The weighted payoff is
// smooth in S_0 and σ, so its pathwise Greeks include the effect of knock-outs,
// which differentiating a 0/1 knock-out indicator misses. These pricers need
// σ > 0 and T > 0.
// ============================================================================

// Price an arithmetic-average Asian call: max(mean(S_1..S_n) − K, 0)
monte_carlo_asian_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 {return}
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt(dt)
	drift := (r - 0.5 * sigma * sigma) * dt
	inv_n := 1.0 / f64(n_steps)

	for _ in 0 ..< n_paths {
		S, W := S_0, 0.0
		sum, dsum := 0.0, 0.0 // Σ S_i and Σ ∂S_i/∂σ
		for i in 1 ..= n_steps {
			z := rand.float64_normal(0.0, 1.0)
			S *= math.exp(drift + sigma * sqrt_dt * z)
			W += sqrt_dt * z
			sum += S
			dsum += S * (W - sigma * f64(i) * dt)
		}
		A := sum * inv_n
		if A > K {
			price += A - K
			delta += A / S_0
			vega += dsum * inv_n
		}
	}
	scale := math.exp(-r * T) / f64(n_paths)
	return price * scale, delta * scale, vega * scale
}

// Up-and-Out Asian Call: pays max(mean(S_1..S_n) − K, 0) unless some S_i > barrier
monte_carlo_barrier_asian_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	barrier: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(sigma > 0 && T > 0 && barrier > 0) {return}
	m := _oss_model(S_0, T, r, sigma, barrier, n_steps)
	inv_n := 1.0 / f64(n_steps)

	for _ in 0 ..< n_paths {
		p := _oss_start(&m)
		sum: f64
		dsum: [2]f64
		for _ in 0 ..< n_steps {
			if !_oss_step(&m, &p) {break}
			S := math.exp(p.x)
			sum += S
			dsum += S * p.dx
		}
		A := sum * inv_n
		if p.w == 0 || A <= K {continue}
		f := A - K
		price += p.w * f
		delta += p.w * (p.dlw[0] * f + dsum[0] * inv_n)
		vega += p.w * (p.dlw[1] * f + dsum[1] * inv_n)
	}
	scale := math.exp(-r * T) / f64(n_paths)
	return price * scale, delta * scale, vega * scale
}

// 2-Asset Basket Call: pays max(w1·mean(S1) + w2·mean(S2) − K, 0)
monte_carlo_basket_option :: proc(
	S1_0: f64,
	S2_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma1: f64,
	sigma2: f64,
	corr: f64, // Correlation between assets
	w1: f64, // Weight for asset 1
	w2: f64, // Weight for asset 2
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta1: f64,
	delta2: f64,
	vega1: f64,
	vega2: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(math.abs(corr) <= 1) {return}
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt(dt)
	drift1 := (r - 0.5 * sigma1 * sigma1) * dt
	drift2 := (r - 0.5 * sigma2 * sigma2) * dt
	corr_perp := math.sqrt(1.0 - corr * corr) // Cholesky factor of [[1, ρ], [ρ, 1]]
	inv_n := 1.0 / f64(n_steps)

	for _ in 0 ..< n_paths {
		S1, S2, W1, W2 := S1_0, S2_0, 0.0, 0.0
		sum1, sum2, dsum1, dsum2: f64
		for i in 1 ..= n_steps {
			z1 := rand.float64_normal(0.0, 1.0)
			z2 := corr * z1 + corr_perp * rand.float64_normal(0.0, 1.0)
			S1 *= math.exp(drift1 + sigma1 * sqrt_dt * z1)
			S2 *= math.exp(drift2 + sigma2 * sqrt_dt * z2)
			W1 += sqrt_dt * z1
			W2 += sqrt_dt * z2
			t_i := f64(i) * dt
			sum1 += S1
			sum2 += S2
			dsum1 += S1 * (W1 - sigma1 * t_i)
			dsum2 += S2 * (W2 - sigma2 * t_i)
		}
		basket := (w1 * sum1 + w2 * sum2) * inv_n
		if basket > K {
			price += basket - K
			delta1 += w1 * sum1 * inv_n / S1_0
			delta2 += w2 * sum2 * inv_n / S2_0
			vega1 += w1 * dsum1 * inv_n
			vega2 += w2 * dsum2 * inv_n
		}
	}
	scale := math.exp(-r * T) / f64(n_paths)
	return price * scale, delta1 * scale, delta2 * scale, vega1 * scale, vega2 * scale
}

// ============================================================================
// Lookback Options
// ============================================================================

// Helper: Smooth differentiable maximum of two tensors
// max(a, b) ≈ (a + b + sqrt((a - b)^2 + ε)) / 2
// Fully compatible with strict dimension-matching tensor operations.
tensor_smooth_max :: proc(a: ^t.Tensor, b: ^t.Tensor, allocator: mem.Allocator) -> ^t.Tensor {
	diff := t.tensor_sub(a, b)
	diff_sq := t.tensor_mul(diff, diff)

	// Create an epsilon tensor with the EXACT same dimensions as diff_sq
	// to avoid 1x1 broadcasting issues in tensor_add
	eps_data := l.matrix_new(f64, diff_sq.data.rows, diff_sq.data.cols, allocator)
	for i in 0 ..< len(eps_data.data) {
		eps_data.data[i] = 1e-6
	}
	eps_tensor := t.tensor_new(eps_data, false, allocator)
	eps_tensor.owned_by_graph = true // an input of the graph: tensor_free_graph frees it

	diff_sq_eps := t.tensor_add(diff_sq, eps_tensor)
	sqrt_diff := t.tensor_sqrt(diff_sq_eps)
	a_plus_b := t.tensor_add(a, b)
	num := t.tensor_add(a_plus_b, sqrt_diff)

	// Use tensor_scale for division by 2 to avoid 1x1 tensor multiplication
	out := t.tensor_scale(num, 0.5)
	if !out.requires_grad {
		// No graph was recorded, so nothing reachable from out holds the intermediates
		for x in ([]^t.Tensor{diff, diff_sq, eps_tensor, diff_sq_eps, sqrt_diff, a_plus_b, num}) {
			t.tensor_free(x)
		}
	}
	return out
}

// Fixed Strike Lookback Call: Pays max(max(S_0, S_1..S_n) − K, 0)
monte_carlo_lookback_call_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 {return}
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt(dt)
	drift := (r - 0.5 * sigma * sigma) * dt

	for _ in 0 ..< n_paths {
		S, W := S_0, 0.0
		M, dM := S_0, 0.0 // running maximum and its ∂/∂σ
		for i in 1 ..= n_steps {
			z := rand.float64_normal(0.0, 1.0)
			S *= math.exp(drift + sigma * sqrt_dt * z)
			W += sqrt_dt * z
			if S > M {
				M = S
				dM = S * (W - sigma * f64(i) * dt)
			}
		}
		if M > K {
			price += M - K
			delta += M / S_0 // every S_i scales with S_0
			vega += dM
		} else if M == K {
			// M = S_0 = K: paths that never rise above S_0 have the kink of the
			// payoff at S_0 itself (a positive fraction under discrete
			// monitoring), so the price has a kink at K = S_0. Count the
			// symmetric derivative, ½·M/S_0.
			delta += 0.5
		}
	}
	scale := math.exp(-r * T) / f64(n_paths)
	return price * scale, delta * scale, vega * scale
}

// ============================================================================
// Black-Scholes Monte Carlo: Up-and-Out Call Option
// ============================================================================
// Pays max(S_n − K, 0) unless some S_i ≥ barrier
monte_carlo_barrier_call_bs :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	barrier: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(sigma > 0 && T > 0 && barrier > 0) {return}
	m := _oss_model(S_0, T, r, sigma, barrier, n_steps)

	for _ in 0 ..< n_paths {
		p := _oss_start(&m)
		for _ in 0 ..< n_steps {
			if !_oss_step(&m, &p) {break}
		}
		if p.w == 0 {continue}
		S := math.exp(p.x)
		if S <= K {continue}
		f := S - K
		price += p.w * f
		delta += p.w * (p.dlw[0] * f + S * p.dx[0])
		vega += p.w * (p.dlw[1] * f + S * p.dx[1])
	}
	scale := math.exp(-r * T) / f64(n_paths)
	return price * scale, delta * scale, vega * scale
}

// ----------------------------------------------------------------------------
// One-step survival below an up barrier, in log-price x = ln S, with tangents
// with respect to θ = (S_0, σ). Per step: a = (ln B − x − μ)/s with μ = (r −
// σ²/2)·dt and s = σ·√dt, survival p = Φ(a), z = Φ⁻¹(U·p) for U ~ U(0,1), so
// x + μ + s·z < ln B, and the weight is multiplied by p. Differentiating with U
// held fixed: ∂z = (U·p/φ(z))·∂ln p.
// ----------------------------------------------------------------------------
@(private = "file")
_OSS_Model :: struct {
	lb, mu, s: f64, // log barrier, drift and volatility per step
	dmu, ds:   [2]f64, // ∂μ/∂θ, ∂s/∂θ
	x0:        f64,
	dx0:       [2]f64,
}

@(private = "file")
_OSS_Path :: struct {
	x:   f64, // log-price
	dx:  [2]f64, // ∂x/∂θ
	w:   f64, // survival weight Π p_i
	dlw: [2]f64, // ∂ln w/∂θ
}

@(private = "file")
_oss_model :: proc(S_0, T, r, sigma, barrier: f64, n_steps: int) -> _OSS_Model {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt(dt)
	return _OSS_Model {
		lb = math.ln(barrier),
		mu = (r - 0.5 * sigma * sigma) * dt,
		s = sigma * sqrt_dt,
		dmu = {0, -sigma * dt},
		ds = {0, sqrt_dt},
		x0 = math.ln(S_0),
		dx0 = {1.0 / S_0, 0},
	}
}

@(private = "file")
_oss_start :: #force_inline proc(m: ^_OSS_Model) -> _OSS_Path {
	return _OSS_Path{x = m.x0, dx = m.dx0, w = 1}
}

// Advances one step; false (and weight 0) once survival is negligible
@(private = "file")
_oss_step :: proc(m: ^_OSS_Model, p: ^_OSS_Path) -> bool {
	a := (m.lb - p.x - m.mu) / m.s
	surv := _bs_cdf(a)
	if surv == 1 {
		// Far from the barrier (a > 8.3): Φ⁻¹(U) is a plain normal draw and
		// ∂ln p is below 1e-15
		z := rand.float64_normal(0.0, 1.0)
		p.dx += m.dmu + m.ds * z
		p.x += m.mu + m.s * z
		return true
	}
	if surv < 1e-280 {
		p.w = 0
		return false
	}
	u := rand.float64()
	if u <= 0 {u = 5.551115123125783e-17} // 2^-54
	v := u * surv
	z := _bs_ppf(v)
	hazard := _bs_pdf(a) / surv // ∂ln p/∂a
	dz_dlnp := v / _bs_pdf(z)
	for k in 0 ..< 2 {
		dlnp := hazard * (-p.dx[k] - m.dmu[k] - a * m.ds[k]) / m.s
		p.dx[k] += m.dmu[k] + m.ds[k] * z + m.s * dz_dlnp * dlnp
		p.dlw[k] += dlnp
	}
	p.x += m.mu + m.s * z
	p.w *= surv
	return true
}
