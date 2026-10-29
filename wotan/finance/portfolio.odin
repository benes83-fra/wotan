package finance

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Portfolio Metrics
// ============================================================================

PortfolioMetrics :: struct {
	weights:         []f64, // The caller's weights (not copied)
	expected_return: f64,
	volatility:      f64,
	sharpe_ratio:    f64,
}

// Calculate portfolio expected return: w^T * mu
portfolio_return :: proc(weights: []f64, expected_returns: []f64) -> f64 {
	if len(weights) != len(expected_returns) {
		panic("portfolio_return: dimension mismatch")
	}
	return l.dot_simd(weights, expected_returns)
}

// Calculate portfolio volatility: sqrt(w^T * Sigma * w)
portfolio_volatility :: proc(weights: []f64, cov_matrix: ^l.Matrix(f64)) -> f64 {
	n := len(weights)
	if cov_matrix.rows != n || cov_matrix.cols != n {
		panic("portfolio_volatility: dimension mismatch")
	}
	sigma_w := l.matvec_dyn_simd(cov_matrix, weights, context.temp_allocator)
	defer delete(sigma_w, context.temp_allocator)
	// clamp the rounding noise of a (near) zero variance
	return math.sqrt(math.max(l.dot_simd(weights, sigma_w), 0.0))
}

// Calculate Sharpe ratio: (return - risk_free) / volatility
sharpe_ratio :: proc(expected_return: f64, volatility: f64, risk_free_rate: f64) -> f64 {
	if volatility == 0.0 {
		return 0.0
	}
	return (expected_return - risk_free_rate) / volatility
}

// Comprehensive portfolio metrics
portfolio_metrics :: proc(
	weights: []f64,
	expected_returns: []f64,
	cov_matrix: ^l.Matrix(f64),
	risk_free_rate: f64 = 0.0,
) -> PortfolioMetrics {
	ret := portfolio_return(weights, expected_returns)
	vol := portfolio_volatility(weights, cov_matrix)
	sr := sharpe_ratio(ret, vol, risk_free_rate)

	return PortfolioMetrics {
		weights = weights,
		expected_return = ret,
		volatility = vol,
		sharpe_ratio = sr,
	}
}

// ============================================================================
// Mean-variance building blocks (fully invested, short sales allowed)
// ============================================================================
// With x = Σ⁻¹1 and y = Σ⁻¹μ:  A = μᵀy, B = 1ᵀy = μᵀx, D = 1ᵀx, det = AD - B².
//   minimum variance      w = x / D                       (return B/D)
//   tangency              w = (y - rf x) / (B - rf D)
//   frontier at return t  w = ((tD - B) y + (A - tB) x) / det

@(private = "file")
_MV_Core :: struct {
	x, y:    []f64, // Σ⁻¹1 and Σ⁻¹μ (y is nil without μ)
	A, B, D: f64,
}

@(private = "file")
_mv_core :: proc(mu: []f64, cov: ^l.Matrix(f64), caller: string) -> _MV_Core {
	n := cov.rows
	if cov.cols != n || (mu != nil && len(mu) != n) {
		panic(caller)
	}
	L := l.matrix_new(f64, n, n, context.temp_allocator)
	defer l.matrix_free(&L)
	copy(L.data, cov.data)
	if !l.cholesky_try(&L) {
		panic("mean-variance: covariance matrix is not positive definite")
	}
	solve :: proc(L: ^l.Matrix(f64), b: []f64) -> []f64 {
		z := l.forward_substitute_simd(L, b, context.temp_allocator)
		defer delete(z, context.temp_allocator)
		return l.backward_substitute_simd(L, z, context.temp_allocator)
	}
	ones := make([]f64, n, context.temp_allocator)
	defer delete(ones, context.temp_allocator)
	for i in 0 ..< n {ones[i] = 1.0}

	c: _MV_Core
	c.x = solve(&L, ones)
	c.D = math.sum(c.x)
	if mu != nil {
		c.y = solve(&L, mu)
		c.A = l.dot_simd(mu, c.y)
		c.B = math.sum(c.y)
	}
	return c
}

@(private = "file")
_mv_core_free :: proc(c: ^_MV_Core) {
	delete(c.x, context.temp_allocator)
	delete(c.y, context.temp_allocator)
}

// ============================================================================
// Minimum Variance Portfolio
// ============================================================================

// Solve: min w^T * Sigma * w  subject to: sum(w) = 1
// Solution: w = Sigma^{-1} * 1 / (1^T * Sigma^{-1} * 1)
// Panics if the covariance is not positive definite
min_variance_portfolio :: proc(
	cov_matrix: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	c := _mv_core(nil, cov_matrix, "min_variance_portfolio: covariance matrix must be square")
	defer _mv_core_free(&c)
	weights := make([]f64, cov_matrix.rows, allocator)
	for v, i in c.x {weights[i] = v / c.D}
	return weights
}

// ============================================================================
// Maximum Sharpe Ratio Portfolio (Tangency Portfolio)
// ============================================================================

// Solve: max (w^T * mu - rf) / sqrt(w^T * Sigma * w)  subject to: sum(w) = 1
// Solution: w = Sigma^{-1} (mu - rf 1) / (1^T Sigma^{-1} (mu - rf 1)).
// That is the maximum only when the minimum-variance portfolio earns more than
// rf (the denominator is then positive). Otherwise the formula gives the
// portfolio with the LOWEST Sharpe ratio and no finite maximum exists, so the
// weights are NaN
max_sharpe_portfolio :: proc(
	expected_returns: []f64,
	cov_matrix: ^l.Matrix(f64),
	risk_free_rate: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	c := _mv_core(expected_returns, cov_matrix, "max_sharpe_portfolio: dimension mismatch")
	defer _mv_core_free(&c)
	n := len(expected_returns)
	weights := make([]f64, n, allocator)
	s := c.B - risk_free_rate * c.D
	if !(s > 0.0) {
		for i in 0 ..< n {weights[i] = math.QNAN_F64}
		return weights
	}
	for i in 0 ..< n {weights[i] = (c.y[i] - risk_free_rate * c.x[i]) / s}
	return weights
}

// ============================================================================
// Efficient Frontier
// ============================================================================

EfficientFrontierPoint :: struct {
	expected_return: f64,
	volatility:      f64,
	sharpe_ratio:    f64,
	weights:         []f64, // Allocated with the frontier's allocator
}

// Efficient frontier (min w'Σw s.t. w'μ = t, w'1 = 1, shorting allowed) at
// n_points target returns from the minimum-variance portfolio's return up to the
// highest single-asset expected return; returns below the minimum-variance one
// lie on the inefficient lower branch and are not included. If all expected
// returns are equal the frontier is the single minimum-variance portfolio
efficient_frontier :: proc(
	expected_returns: []f64,
	cov_matrix: ^l.Matrix(f64),
	risk_free_rate: f64 = 0.0,
	n_points: int = 20,
	allocator: mem.Allocator = context.allocator,
) -> []EfficientFrontierPoint {
	n := len(expected_returns)
	c := _mv_core(expected_returns, cov_matrix, "efficient_frontier: dimension mismatch")
	defer _mv_core_free(&c)

	ret_gmv := c.B / c.D
	max_ret := ret_gmv
	for m in expected_returns {max_ret = math.max(max_ret, m)}
	det := c.A * c.D - c.B * c.B
	// det = D · Σ⁻¹-weighted dispersion of μ: zero when μ ∝ 1
	degenerate := det <= 1e-12 * c.A * c.D

	frontier := make([]EfficientFrontierPoint, max(n_points, 0), allocator)
	for k in 0 ..< len(frontier) {
		weights := make([]f64, n, allocator)
		if degenerate {
			for i in 0 ..< n {weights[i] = c.x[i] / c.D}
		} else {
			t := ret_gmv
			if n_points > 1 {t += (max_ret - ret_gmv) * f64(k) / f64(n_points - 1)}
			a, b := (t * c.D - c.B) / det, (c.A - t * c.B) / det
			for i in 0 ..< n {weights[i] = a * c.y[i] + b * c.x[i]}
		}
		ret := portfolio_return(weights, expected_returns)
		vol := portfolio_volatility(weights, cov_matrix)
		frontier[k] = EfficientFrontierPoint {
			expected_return = ret,
			volatility      = vol,
			sharpe_ratio    = sharpe_ratio(ret, vol, risk_free_rate),
			weights         = weights,
		}
	}
	return frontier
}

// ============================================================================
// Utility Functions
// ============================================================================

// Sample covariance matrix (n - 1) of returns data: rows = time periods, cols = assets
covariance_matrix :: proc(
	returns_data: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	return l.covariance(returns_data, allocator)
}

// Arithmetic mean return of each asset (column)
expected_returns_from_history :: proc(
	returns_data: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_assets := returns_data.cols
	n_periods := returns_data.rows

	means := make([]f64, n_assets, allocator)
	if n_periods == 0 {return means}
	for i in 0 ..< n_periods {
		row := returns_data.data[i * n_assets:(i + 1) * n_assets]
		for v, j in row {means[j] += v}
	}
	for &m in means {m /= f64(n_periods)}
	return means
}

// Annualize an arithmetic mean daily return (simple scaling, not compounding)
annualize_return :: proc(daily_return: f64, trading_days: int = 252) -> f64 {
	return daily_return * f64(trading_days)
}

// Annualize volatility (square-root-of-time scaling)
annualize_volatility :: proc(daily_volatility: f64, trading_days: int = 252) -> f64 {
	return daily_volatility * math.sqrt(f64(trading_days))
}
