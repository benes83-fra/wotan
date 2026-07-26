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
