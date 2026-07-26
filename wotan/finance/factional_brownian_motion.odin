package finance

import l "../linalg"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// FRACTIONAL BROWNIAN MOTION (fBm)
// ============================================================================
// Generates fBm paths using the Cholesky decomposition of the covariance matrix.
// Cov(B^H_t, B^H_s) = 0.5 * (t^{2H} + s^{2H} - |t-s|^{2H})

// Compute the covariance matrix for fBm
_fbm_covariance_matrix :: proc(times: []f64, H: f64, allocator: mem.Allocator) -> l.Matrix(f64) {
	n := len(times)
	cov := l.matrix_new(f64, n, n, allocator)

	for i in 0 ..< n {
		for j in 0 ..< n {
			t := times[i]
			s := times[j]
			// fBm covariance formula
			cov_val :=
				0.5 *
				(math.pow(t, 2.0 * H) + math.pow(s, 2.0 * H) - math.pow(math.abs(t - s), 2.0 * H))

			// ✅ FIX: Add a tiny epsilon to the diagonal to ensure strict positive definiteness.
			// This prevents Cholesky from panicking when t=0 (where covariance is exactly 0)
			// or due to floating-point roundoff errors in the covariance calculation.
			if i == j {
				cov_val += 1e-12
			}

			cov.data[i * n + j] = cov_val
		}
	}
	return cov
}
// Generate a single fBm path using Cholesky decomposition
generate_fbm_path :: proc(
	times: []f64,
	H: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(times)

	// 1. Build covariance matrix
	cov := _fbm_covariance_matrix(times, H, allocator)
	defer l.matrix_free(&cov)

	// 2. Cholesky decomposition: Cov = L * L^T
	// Note: We use a copy because Cholesky modifies the matrix in-place
	L := l.matrix_new(f64, n, n, allocator)
	copy(L.data, cov.data)
	l.cholesky_decompose(&L)
	defer l.matrix_free(&L)

	// 3. Generate independent standard normals
	Z := make([]f64, n, allocator)
	defer delete(Z, allocator)
	for i in 0 ..< n {
		Z[i] = rand.float64_normal(0.0, 1.0)
	}

	// 4. Multiply L * Z to get correlated fBm increments
	// Since L is lower triangular, we can use matvec
	fbm_path := l.matvec_dyn_simd(&L, Z, allocator)

	// The result from matvec is allocated by the caller's context,
	// but we want to return it cleanly. The caller must delete it.
	return fbm_path
}

// ============================================================================
// FRACTIONAL BLACK-SCHOLES PRICING (Monte Carlo)
// ============================================================================
// WARNING: Raw fBm for the underlying asset violates the semimartingale property,
// meaning standard no-arbitrage arguments fail. This is used for empirical
// fitting or as a stepping stone to Rough Volatility models.
// For arbitrage-free fractional modeling, see Rough Bergomi (fractional volatility).

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
	// Time grid
	times := make([]f64, n_steps + 1, context.temp_allocator)
	defer delete(times, context.temp_allocator)
	for i in 0 ..< n_steps + 1 {
		times[i] = f64(i) * (T / f64(n_steps))
	}

	// Pre-allocate for efficiency
	total_payoff := 0.0

	for path in 0 ..< n_paths {
		// Generate fBm path
		W_H := generate_fbm_path(times, H, context.temp_allocator)

		// Simulate asset price: S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma * W_H_T)
		// Note: In strict fractional calculus (Wick-Itô), the drift adjustment differs,
		// but this is the standard "naive" fractional Black-Scholes approximation.
		W_H_T := W_H[n_steps]
		S_T := S * math.exp_f64((r - 0.5 * sigma * sigma) * T + sigma * W_H_T)

		delete(W_H, context.temp_allocator)

		payoff := math.max(S_T - K, 0.0)
		total_payoff += payoff
	}

	return (total_payoff / f64(n_paths)) * math.exp_f64(-r * T)
}

// ============================================================================
// ROUGH VOLATILITY HELPER (Conceptual)
// ============================================================================
// In modern finance, we apply fBm to the *volatility* process, not the asset.
// E.g., d(log v_t) = ... dt + nu * dB^H_t, where H ≈ 0.1 (Rough Volatility).
// This preserves the martingale property of the asset while capturing the
// steep short-term implied volatility skew observed in real markets.

// ============================================================================
// ROUGH BERGOMI (rBergomi) MODEL
// ============================================================================
// A state-of-the-art rough volatility model where the volatility process is
// driven by a fractional Brownian motion with Hurst parameter H ≈ 0.1.
// This naturally captures the steep short-term implied volatility skew observed
// in real markets without requiring jump components.

rBergomi_Params :: struct {
	xi_0: f64, // Initial forward variance (typically ATM variance, e.g., 0.04 for 20% vol)
	eta:  f64, // Volatility of volatility (vol-of-vol, typically 0.3 - 0.6)
	H:    f64, // Hurst parameter (typically 0.1 - 0.2)
	rho:  f64, // Correlation between asset and volatility shocks (typically -0.7 to -0.9)
}

// Precompute the Cholesky decomposition of the fBm covariance matrix.
// Cov(W^H_{t_i}, W^H_{t_j}) = 0.5 * (t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})
// This guarantees Var(W^H_{t_i}) = t_i^{2H}, making the compensator exact.
_rbergomi_compute_cholesky :: proc(
	n_steps: int,
	T: f64,
	H: f64,
	allocator: mem.Allocator,
) -> l.Matrix(f64) {
	dt := T / f64(n_steps)
	cov := l.matrix_new(f64, n_steps + 1, n_steps + 1, allocator)

	for i in 0 ..< n_steps + 1 {
		t_i := f64(i) * dt
		for j in 0 ..< n_steps + 1 {
			t_j := f64(j) * dt
			cov.data[i * (n_steps + 1) + j] =
				0.5 *
				(math.pow_f64(t_i, 2.0 * H) +
						math.pow_f64(t_j, 2.0 * H) -
						math.pow_f64(math.abs(t_i - t_j), 2.0 * H))
		}
	}

	// Add tiny jitter to diagonal for numerical stability in Cholesky
	for i in 0 ..< n_steps + 1 {
		cov.data[i * (n_steps + 1) + i] += 1e-12
	}

	// Cholesky decomposes in-place, leaving L in the lower triangle
	l.cholesky_decompose(&cov)
	return cov
}

// Price a European Call Option under rBergomi using Monte Carlo
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
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)

	// 1. Precompute Cholesky decomposition of fBm covariance matrix (done ONCE)
	L := _rbergomi_compute_cholesky(n_steps, T, params.H, context.temp_allocator)
	defer l.matrix_free(&L)

	total_payoff := 0.0

	for path in 0 ..< n_paths {
		// 2. Generate independent normals for volatility driver and orthogonal asset driver
		Z_vol := make([]f64, n_steps + 1, context.temp_allocator)
		Z_perp := make([]f64, n_steps + 1, context.temp_allocator)
		for j in 0 ..< n_steps + 1 {
			Z_vol[j] = rand.float64_normal(0.0, 1.0)
			Z_perp[j] = rand.float64_normal(0.0, 1.0)
		}

		// 3. Simulate fractional Brownian motion path: W^H = L * Z_vol
		// Uses your existing SIMD-optimized matrix-vector multiplication
		W_H := l.matvec_dyn_simd(&L, Z_vol, context.temp_allocator)
		defer delete(W_H, context.temp_allocator)

		ln_S := math.ln(S_0)

		// 4. Euler-Maruyama simulation of the asset price
		for i in 1 ..< n_steps + 1 {
			t_i := f64(i) * dt
			t_prev := f64(i - 1) * dt

			// Variance at t_{i-1}
			// Compensator is exactly 0.5 * eta^2 * t^{2H} because Var(W^H_t) = t^{2H}
			W_H_prev := W_H[i - 1]
			V_prev :=
				params.xi_0 *
				math.exp_f64(
					params.eta * W_H_prev -
					0.5 * params.eta * params.eta * math.pow_f64(t_prev, 2.0 * params.H),
				)

			sqrt_V_prev := math.sqrt_f64(V_prev)

			// Correlated Brownian increments
			dW := Z_vol[i] * sqrt_dt
			dW_perp := Z_perp[i] * sqrt_dt

			// Log-return: d ln S = (r - 0.5 V) dt + sqrt(V) (rho dW + sqrt(1-rho^2) dW_perp)
			ln_S +=
				(r - 0.5 * V_prev) * dt +
				sqrt_V_prev *
					(params.rho * dW + math.sqrt_f64(1.0 - params.rho * params.rho) * dW_perp)
		}

		S_T := math.exp_f64(ln_S)
		payoff := math.max(S_T - K, 0.0)
		total_payoff += payoff

		delete(Z_vol, context.temp_allocator)
		delete(Z_perp, context.temp_allocator)
	}

	return (total_payoff / f64(n_paths)) * math.exp_f64(-r * T)
}

// ============================================================================
// Black-Scholes Baseline (for comparison)
// ============================================================================
_bs_call_price :: proc(S: f64, K: f64, T: f64, r: f64, sigma: f64) -> f64 {
	if T <= 0.0 {return math.max(S - K, 0.0)}
	sqrt_T := math.sqrt_f64(T)
	d1 := (math.ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2 := d1 - sigma * sqrt_T
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
	return S * N_d1 - K * math.exp_f64(-r * T) * N_d2
}
