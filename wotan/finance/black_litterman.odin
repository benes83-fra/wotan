package finance

import l "../linalg"
import "core:mem"

// ============================================================================
// BLACK-LITTERMAN
// ============================================================================
// Prior: equilibrium excess returns Π = δ Σ w_mkt implied by the market
// portfolio (reverse optimization). Views: P μ = Q + ε, ε ~ N(0, Ω), with P
// (k × n) picking assets or spreads. The posterior (He & Litterman 1999) is
//   μ_BL = Π + τΣPᵀ (PτΣPᵀ + Ω)⁻¹ (Q − PΠ)
//   M    = τΣ − τΣPᵀ (PτΣPᵀ + Ω)⁻¹ PτΣ      (uncertainty of the mean)
//   Σ_BL = Σ + M                             (covariance of returns)
// Ω defaults to He-Litterman's diag(P τΣ Pᵀ); Idzorek's confidences c_k ∈
// [0, 1] give ω_k = (1/c_k − 1) p_k τΣ p_kᵀ (100% confidence: the view holds
// exactly; 0%: the view is ignored). μ_BL and Σ_BL feed the optimizers of
// portfolio.odin / constraints.odin.

BlackLittermanResult :: struct {
	posterior_returns: []f64,
	posterior_cov:     l.Matrix(f64), // Σ + M
	omega:             l.Matrix(f64), // view uncertainty used
	allocator:         mem.Allocator,
}

black_litterman_result_free :: proc(r: ^BlackLittermanResult) {
	delete(r.posterior_returns, r.allocator)
	l.matrix_free(&r.posterior_cov)
	l.matrix_free(&r.omega)
	r^ = {}
}

// Risk aversion implied by the market: δ = (E[R_m] − r_f) / σ_m²
bl_market_risk_aversion :: proc(market_return, market_variance: f64, risk_free: f64 = 0.0) -> f64 {
	return (market_return - risk_free) / market_variance
}

// Equilibrium excess returns Π = δ Σ w_mkt
bl_implied_returns :: proc(
	cov: ^l.Matrix(f64),
	market_weights: []f64,
	risk_aversion: f64,
	allocator := context.allocator,
) -> []f64 {
	n := cov.rows
	pi := make([]f64, n, allocator)
	for i in 0 ..< n {
		s := 0.0
		for j in 0 ..< n {s += cov.data[i * n + j] * market_weights[j]}
		pi[i] = risk_aversion * s
	}
	return pi
}

// Posterior returns and covariance. `omega` (k × k) takes precedence, then
// `confidences` (Idzorek), else He-Litterman's default.
black_litterman :: proc(
	cov: ^l.Matrix(f64),
	prior: []f64,
	P: ^l.Matrix(f64),
	Q: []f64,
	tau: f64 = 0.05,
	omega: ^l.Matrix(f64) = nil,
	confidences: []f64 = nil,
	allocator := context.allocator,
) -> BlackLittermanResult {
	n, k := cov.rows, P.rows
	assert(
		cov.cols == n && P.cols == n && len(prior) == n && len(Q) == k,
		"black_litterman: dimension mismatch",
	)
	S := cov.data

	// B = P τΣ (k × n)
	B := make([]f64, k * n, context.temp_allocator)
	defer delete(B, context.temp_allocator)
	for v in 0 ..< k {
		for j in 0 ..< n {
			s := 0.0
			for i in 0 ..< n {s += P.data[v * n + i] * S[i * n + j]}
			B[v * n + j] = tau * s
		}
	}
	// Ω
	om := l.matrix_new(f64, k, k, allocator)
	if omega != nil {
		copy(om.data, omega.data)
	} else {
		for v in 0 ..< k {
			pSp := 0.0 // p_v τΣ p_vᵀ
			for j in 0 ..< n {pSp += B[v * n + j] * P.data[v * n + j]}
			om.data[v * k + v] = pSp
			if confidences != nil {
				c := confidences[v]
				om.data[v * k + v] = pSp * (1.0 / c - 1.0) if c > 0 else 1e6
			}
		}
	}
	// A = P τΣ Pᵀ + Ω (k × k), Cholesky
	A := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&A)
	for a in 0 ..< k {
		for b in 0 ..< k {
			s := 0.0
			for j in 0 ..< n {s += B[a * n + j] * P.data[b * n + j]}
			A.data[a * k + b] = s + om.data[a * k + b]
		}
	}
	if !l.cholesky_try(&A) {panic("black_litterman: P τΣ Pᵀ + Ω is not positive definite")}
	solve :: proc(L: ^l.Matrix(f64), b: []f64) -> []f64 {
		z := l.forward_substitute_simd(L, b, context.temp_allocator)
		defer delete(z, context.temp_allocator)
		return l.backward_substitute_simd(L, z, context.temp_allocator)
	}

	// μ_BL = Π + Bᵀ A⁻¹ (Q − PΠ)
	resid := make([]f64, k, context.temp_allocator)
	defer delete(resid, context.temp_allocator)
	for v in 0 ..< k {
		s := 0.0
		for j in 0 ..< n {s += P.data[v * n + j] * prior[j]}
		resid[v] = Q[v] - s
	}
	x := solve(&A, resid)
	defer delete(x, context.temp_allocator)
	mu := make([]f64, n, allocator)
	for j in 0 ..< n {
		s := prior[j]
		for v in 0 ..< k {s += B[v * n + j] * x[v]}
		mu[j] = s
	}

	// Σ_BL = Σ + τΣ − Bᵀ A⁻¹ B
	post := l.matrix_new(f64, n, n, allocator)
	Y := make([]f64, k * n, context.temp_allocator) // A⁻¹ B, column by column
	defer delete(Y, context.temp_allocator)
	col := make([]f64, k, context.temp_allocator)
	defer delete(col, context.temp_allocator)
	for j in 0 ..< n {
		for v in 0 ..< k {col[v] = B[v * n + j]}
		y := solve(&A, col)
		for v in 0 ..< k {Y[v * n + j] = y[v]}
		delete(y, context.temp_allocator)
	}
	for i in 0 ..< n {
		for j in 0 ..< n {
			s := 0.0
			for v in 0 ..< k {s += B[v * n + i] * Y[v * n + j]}
			post.data[i * n + j] = (1.0 + tau) * S[i * n + j] - s
		}
	}
	return BlackLittermanResult{mu, post, om, allocator}
}

// Unconstrained mean-variance weights w = (δ Σ)⁻¹ μ (not normalized; with the
// posterior these are He-Litterman's (w_mkt + Pᵀ Λ)/(1 + τ))
bl_optimal_weights :: proc(
	cov: ^l.Matrix(f64),
	mu: []f64,
	risk_aversion: f64,
	allocator := context.allocator,
) -> []f64 {
	n := cov.rows
	L := l.matrix_new(f64, n, n, context.temp_allocator)
	defer l.matrix_free(&L)
	copy(L.data, cov.data)
	if !l.cholesky_try(
		&L,
	) {panic("bl_optimal_weights: covariance matrix is not positive definite")}
	z := l.forward_substitute_simd(&L, mu, context.temp_allocator)
	defer delete(z, context.temp_allocator)
	w := l.backward_substitute_simd(&L, z, allocator)
	for &x in w {x /= risk_aversion}
	return w
}
