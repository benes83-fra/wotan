package finance

import l "../linalg"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// FRACTIONAL BROWNIAN MOTION (fBm)
// ============================================================================
// Exact simulation on an arbitrary time grid via the Cholesky factor of
// Cov(B^H_t, B^H_s) = 0.5 * (t^{2H} + s^{2H} - |t-s|^{2H}),  0 < H < 1

@(private = "file")
_fbm_cov :: #force_inline proc(t, s, H: f64) -> f64 {
	h2 := 2.0 * H
	return 0.5 * (math.pow_f64(t, h2) + math.pow_f64(s, h2) - math.pow_f64(math.abs(t - s), h2))
}

// In-place lower Cholesky that accepts positive semi-definite matrices: a pivot
// at round-off level relative to its diagonal means the variable is a linear
// combination of the earlier ones (t = 0, repeated times, or H → 0.5 in the
// rBergomi drivers), and its column is set to zero instead of failing
@(private = "file")
_psd_cholesky :: proc(A: ^l.Matrix(f64)) {
	n := A.rows
	d := A.data
	for j in 0 ..< n {
		diag := d[j * n + j]
		row_j := d[j * n:j * n + j]
		pivot := diag - l.dot_simd(row_j, row_j)
		if pivot <= 1e-12 * diag {
			for i in j ..< n {d[i * n + j] = 0.0}
			continue
		}
		ljj := math.sqrt(pivot)
		d[j * n + j] = ljj
		for i in j + 1 ..< n {
			d[i * n + j] = (d[i * n + j] - l.dot_simd(d[i * n:i * n + j], row_j)) / ljj
		}
	}
	for i in 0 ..< n {
		for j in i + 1 ..< n {d[i * n + j] = 0.0}
	}
}

// y = L z for a lower-triangular L (only the lower triangle is read)
@(private = "file")
_lower_matvec :: proc(L: ^l.Matrix(f64), z, y: []f64) {
	n := L.rows
	for i in 0 ..< n {
		y[i] = l.dot_simd(L.data[i * n:i * n + i + 1], z[:i + 1])
	}
}

// Cholesky factor of the fBm covariance at the given times (t ≥ 0). Build it once
// and call fbm_path_from_cholesky per path: the factorisation is O(n³), a path O(n²)
fbm_cholesky :: proc(times: []f64, H: f64, allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	assert(H > 0.0 && H < 1.0, "fbm_cholesky: H must be in (0, 1)")
	n := len(times)
	L := l.matrix_new(f64, n, n, allocator)
	for i in 0 ..< n {
		for j in 0 ..= i {
			c := _fbm_cov(times[i], times[j], H)
			L.data[i * n + j] = c
			L.data[j * n + i] = c
		}
	}
	_psd_cholesky(&L)
	return L
}

// One fBm path from a factor built by fbm_cholesky (B^H = 0 exactly where t = 0)
fbm_path_from_cholesky :: proc(L: ^l.Matrix(f64), allocator: mem.Allocator = context.allocator) -> []f64 {
	n := L.rows
	z := make([]f64, n, allocator)
	for i in 0 ..< n {z[i] = rand.float64_normal(0.0, 1.0)}
	path := make([]f64, n, allocator)
	_lower_matvec(L, z, path)
	delete(z, allocator)
	return path
}

// One fBm path at the given times; caller deletes it. For many paths on the same
// grid use fbm_cholesky + fbm_path_from_cholesky instead
generate_fbm_path :: proc(times: []f64, H: f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	L := fbm_cholesky(times, H, context.temp_allocator)
	defer l.matrix_free(&L)
	return fbm_path_from_cholesky(&L, allocator)
}

// ============================================================================
// FRACTIONAL BLACK-SCHOLES
// ============================================================================
// Wick-Itô fractional Black-Scholes (Necula 2002; Hu & Øksendal 2003):
//   S_T = S exp(r T - ½ σ² T^{2H} + σ B^H_T)
// so E[e^{-rT} S_T] = S and the call is Black-Scholes with total variance σ² T^{2H}
// (H = 0.5 is Black-Scholes). Raw fBm is not a semimartingale, so the model admits
// arbitrage under pathwise trading; it is meant for empirical fitting and as a
// baseline. For arbitrage-free fractional modelling see rough Bergomi below.

fbm_european_call_closed_form :: proc(S, K, T, r, sigma, H: f64) -> f64 {
	if T <= 0.0 {return math.max(S - K, 0.0)}
	return _bs_call_price(S, K, T, r, sigma * math.pow_f64(T, H - 0.5))
}

// Monte Carlo of the same model. The payoff depends only on B^H_T ~ N(0, T^{2H}),
// so that is sampled directly; n_steps no longer affects the result
fbm_european_call_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	H: f64, // Hurst exponent (0.5 = standard GBM)
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	if T <= 0.0 || n_paths <= 0 {return math.max(S - K, 0.0)}
	sd := sigma * math.pow_f64(T, H)
	drift := math.ln(S) + r * T - 0.5 * sd * sd
	total_payoff := 0.0
	for _ in 0 ..< n_paths {
		total_payoff += math.max(math.exp(drift + sd * rand.float64_normal(0.0, 1.0)) - K, 0.0)
	}
	return total_payoff / f64(n_paths) * math.exp(-r * T)
}

// ============================================================================
// ROUGH BERGOMI (rBergomi) MODEL (Bayer, Friz & Gatheral 2016)
// ============================================================================
//   V_t = ξ₀ exp(η Ṽ_t - ½ η² t^{2H}),   Ṽ_t = √(2H) ∫₀ᵗ (t-s)^{H-½} dW_s
//   d ln S = (r - ½ V) dt + √V (ρ dW + √(1-ρ²) dW⊥)
// The asset's Brownian motion W is the one driving the Volterra process Ṽ, which
// is what produces the short-maturity skew. Simulation is exact for the Gaussian
// pair (W, Ṽ) on the grid (joint Cholesky) and log-Euler for S.

rBergomi_Params :: struct {
	xi_0: f64, // Initial forward variance (typically ATM variance, e.g., 0.04 for 20% vol)
	eta:  f64, // Volatility of volatility (typically 1 - 2.5 for SPX; 0.3 - 0.6 is mild)
	H:    f64, // Hurst parameter (typically 0.05 - 0.15)
	rho:  f64, // Correlation between asset and volatility shocks (typically -0.7 to -0.9)
}

// 16-point Gauss-Legendre rule on [-1, 1] (nodes ±x, weights w)
@(private = "file")
_GL16_X := [8]f64 {
	0.0950125098376374401853193,
	0.2816035507792589132304605,
	0.4580167776572273863424194,
	0.6178762444026437484466718,
	0.7554044083550030338951012,
	0.8656312023878317438804679,
	0.9445750230732325760779884,
	0.9894009349916499325961542,
}
@(private = "file")
_GL16_W := [8]f64 {
	0.1894506104550684962853967,
	0.1826034150449235888667637,
	0.1691565193950025381893121,
	0.1495959888165767320815017,
	0.1246289712555338720524763,
	0.0951585116824927848099251,
	0.0622535239386478928628438,
	0.0271524594117540948517806,
}

// ∫_lo^hi x^a (x+k)^a dx for 0 < lo (integrand analytic on the interval)
@(private = "file")
_volterra_gl :: proc(lo, hi, k, a: f64) -> f64 {
	c, h := 0.5 * (lo + hi), 0.5 * (hi - lo)
	s := 0.0
	for i in 0 ..< 8 {
		x1, x2 := c - h * _GL16_X[i], c + h * _GL16_X[i]
		s += _GL16_W[i] * (math.pow_f64(x1 * (x1 + k), a) + math.pow_f64(x2 * (x2 + k), a))
	}
	return h * s
}

// ∫_0^½ x^a (x+k)^a dx for k ≥ 1: binomial series in x/k ≤ ½
@(private = "file")
_volterra_head :: proc(k, a: f64) -> f64 {
	sum, binom, q := 0.0, 1.0, math.pow_f64(0.5, a + 1.0) // binom = C(a, n), q = ½^{a+n+1}
	for n in 0 ..< 200 {
		nf := f64(n)
		term := binom * q / (a + nf + 1.0)
		sum += term
		if math.abs(term) <= 1e-17 * math.abs(sum) {break}
		binom *= (a - nf) / (nf + 1.0)
		q *= 0.5 / k
	}
	return math.pow_f64(k, a) * sum
}

// Joint covariance of the rBergomi Gaussian drivers on t_i = i T/n:
//   variables 0 .. n-1:    ΔW_i = W_{t_{i+1}} - W_{t_i}
//   variables n .. 2n-2:   Ṽ_{t_j}, j = 1 .. n-1   (Ṽ_0 = 0; Ṽ_{t_n} is never used)
// with Cov(Ṽ_u, Ṽ_v) = 2H ∫₀ᵘ (u-s)^{H-½} (v-s)^{H-½} ds (u ≤ v), so Var(Ṽ_t) = t^{2H}
rbergomi_covariance :: proc(n_steps: int, T, H: f64, allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	assert(H > 0.0 && H < 1.0 && n_steps >= 1, "rbergomi_covariance: need 0 < H < 1 and n_steps ≥ 1")
	n := n_steps
	dim := 2 * n - 1
	C := l.matrix_new(f64, dim, dim, allocator)
	dt := T / f64(n)
	a := H - 0.5
	alpha := H + 0.5
	set :: proc(C: ^l.Matrix(f64), i, j: int, v: f64) {
		C.data[i * C.cols + j] = v
		C.data[j * C.cols + i] = v
	}

	for i in 0 ..< n {set(&C, i, i, dt)}

	// Cov(Ṽ_{t_j}, ΔW_i) = √(2H)/α dt^α ((j-i)^α - (j-i-1)^α) for i < j, else 0
	cross := math.sqrt(2.0 * H) / alpha * math.pow_f64(dt, alpha)
	for j in 1 ..< n {
		for i in 0 ..< j {
			set(&C, n + j - 1, i, cross * (math.pow_f64(f64(j - i), alpha) - math.pow_f64(f64(j - i - 1), alpha)))
		}
	}

	// Cov(Ṽ_{t_j}, Ṽ_{t_{j+k}}) = 2H dt^{2H} J(j, k),  J(j, k) = ∫_0^j x^a (x+k)^a dx,
	// accumulated over unit intervals; J(j, 0) = j^{2H} / (2H)
	scale := 2.0 * H * math.pow_f64(dt, 2.0 * H)
	for j in 1 ..< n {
		set(&C, n + j - 1, n + j - 1, math.pow_f64(dt * f64(j), 2.0 * H))
	}
	for k in 1 ..< n - 1 {
		kf := f64(k)
		J := _volterra_head(kf, a) + _volterra_gl(0.5, 1.0, kf, a)
		for j in 1 ..< n - k {
			if j > 1 {J += _volterra_gl(f64(j - 1), f64(j), kf, a)}
			set(&C, n + j - 1, n + j + k - 1, scale * J)
		}
	}
	return C
}

// Price a European call under rBergomi by Monte Carlo
rbergomi_mc_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	params: rBergomi_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	strikes := [1]f64{K}
	prices := rbergomi_mc_calls(S_0, strikes[:], T, r, params, n_paths, n_steps, context.temp_allocator)
	price := prices[0]
	delete(prices, context.temp_allocator)
	return price
}

// Calls at several strikes from one set of paths (cheaper than one run per strike,
// and the prices across strikes are consistent: a smile without strike-to-strike noise)
rbergomi_mc_calls :: proc(
	S_0: f64,
	strikes: []f64,
	T: f64,
	r: f64,
	params: rBergomi_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := n_steps
	dt := T / f64(n)
	sqrt_dt := math.sqrt(dt)
	rho_perp := math.sqrt(1.0 - params.rho * params.rho)

	L := rbergomi_covariance(n, T, params.H, context.temp_allocator)
	defer l.matrix_free(&L)
	_psd_cholesky(&L)

	dim := 2 * n - 1
	z := make([]f64, dim, context.temp_allocator)
	y := make([]f64, dim, context.temp_allocator)
	comp := make([]f64, n, context.temp_allocator) // ½ η² t_i^{2H}
	defer {
		delete(z, context.temp_allocator)
		delete(y, context.temp_allocator)
		delete(comp, context.temp_allocator)
	}
	for i in 0 ..< n {
		comp[i] = 0.5 * params.eta * params.eta * math.pow_f64(f64(i) * dt, 2.0 * params.H)
	}

	prices := make([]f64, len(strikes), allocator)
	ln_S0 := math.ln(S_0)
	for _ in 0 ..< n_paths {
		for j in 0 ..< dim {z[j] = rand.float64_normal(0.0, 1.0)}
		_lower_matvec(&L, z, y)

		ln_S := ln_S0
		for i in 0 ..< n {
			// Variance at the left end t_i of the step (Ṽ_0 = 0)
			V := params.xi_0
			if i > 0 {V = params.xi_0 * math.exp(params.eta * y[n + i - 1] - comp[i])}
			dW := y[i]
			dW_perp := rand.float64_normal(0.0, 1.0) * sqrt_dt
			ln_S += (r - 0.5 * V) * dt + math.sqrt(V) * (params.rho * dW + rho_perp * dW_perp)
		}
		S_T := math.exp(ln_S)
		for K, k in strikes {prices[k] += math.max(S_T - K, 0.0)}
	}

	if n_paths > 0 {
		df := math.exp(-r * T) / f64(n_paths)
		for &p in prices {p *= df}
	}
	return prices
}

// ============================================================================
// Black-Scholes Baseline (for comparison)
// ============================================================================
_bs_call_price :: proc(S: f64, K: f64, T: f64, r: f64, sigma: f64) -> f64 {
	if T <= 0.0 {return math.max(S - K, 0.0)}
	sd := sigma * math.sqrt(T)
	df := math.exp(-r * T)
	if sd <= 0.0 {return math.max(S - K * df, 0.0)}
	d1 := (math.ln(S / K) + r * T) / sd + 0.5 * sd
	return S * _bs_cdf(d1) - K * df * _bs_cdf(d1 - sd)
}
