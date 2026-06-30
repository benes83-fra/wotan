package finance

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Portfolio Metrics
// ============================================================================

PortfolioMetrics :: struct {
	weights:         []f64,
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

	// Calculate Sigma * w
	sigma_w := l.matvec_dyn_simd(cov_matrix, weights, context.temp_allocator)
	defer delete(sigma_w, context.temp_allocator)

	// Calculate w^T * Sigma * w
	variance := l.dot_simd(weights, sigma_w)

	return math.sqrt(variance)
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
// Minimum Variance Portfolio
// ============================================================================

// Solve: min w^T * Sigma * w  subject to: sum(w) = 1
// Solution: w = Sigma^{-1} * 1 / (1^T * Sigma^{-1} * 1)
min_variance_portfolio :: proc(
	cov_matrix: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := cov_matrix.rows
	if cov_matrix.cols != n {
		panic("min_variance_portfolio: covariance matrix must be square")
	}

	// Create vector of ones
	ones := make([]f64, n, allocator)
	defer delete(ones, allocator)
	for i in 0 ..< n {
		ones[i] = 1.0
	}

	// Solve Sigma * x = ones using Cholesky decomposition
	// First, make a copy since Cholesky modifies the matrix
	cov_copy := l.matrix_new(f64, n, n, allocator)
	defer l.matrix_free(&cov_copy)
	copy(cov_copy.data, cov_matrix.data)

	// Cholesky decomposition
	l.cholesky_decompose(&cov_copy)

	// Forward substitution: L * y = ones
	y := l.forward_subst_unit_lower_simd(&cov_copy, ones, allocator)
	defer delete(y, allocator)

	// Back substitution: L^T * x = y
	x := l.back_subst_upper_simd(&cov_copy, y, allocator)
	defer delete(x, allocator)

	// Calculate sum = 1^T * x
	sum := l.dot_simd(ones, x)

	// Normalize: w = x / sum
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = x[i] / sum
	}

	return weights
}

// ============================================================================
// Maximum Sharpe Ratio Portfolio (Tangency Portfolio)
// ============================================================================

// Solve: max (w^T * mu - rf) / sqrt(w^T * Sigma * w)  subject to: sum(w) = 1
// Solution: w = Sigma^{-1} * (mu - rf*1) / (1^T * Sigma^{-1} * (mu - rf*1))
max_sharpe_portfolio :: proc(
	expected_returns: []f64,
	cov_matrix: ^l.Matrix(f64),
	risk_free_rate: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(expected_returns)
	if cov_matrix.rows != n || cov_matrix.cols != n {
		panic("max_sharpe_portfolio: dimension mismatch")
	}

	// Create vector (mu - rf*1)
	excess_returns := make([]f64, n, allocator)
	defer delete(excess_returns, allocator)
	for i in 0 ..< n {
		excess_returns[i] = expected_returns[i] - risk_free_rate
	}

	// Solve Sigma * x = excess_returns using Cholesky
	cov_copy := l.matrix_new(f64, n, n, allocator)
	defer l.matrix_free(&cov_copy)
	copy(cov_copy.data, cov_matrix.data)

	l.cholesky_decompose(&cov_copy)

	y := l.forward_subst_unit_lower_simd(&cov_copy, excess_returns, allocator)
	defer delete(y, allocator)

	x := l.back_subst_upper_simd(&cov_copy, y, allocator)
	defer delete(x, allocator)

	// Normalize so weights sum to 1
	ones := make([]f64, n, allocator)
	defer delete(ones, allocator)
	for i in 0 ..< n {
		ones[i] = 1.0
	}

	sum := l.dot_simd(ones, x)

	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = x[i] / sum
	}

	return weights
}

// ============================================================================
// Efficient Frontier
// ============================================================================

EfficientFrontierPoint :: struct {
	expected_return: f64,
	volatility:      f64,
	sharpe_ratio:    f64,
	weights:         []f64,
}

// Generate efficient frontier by solving:
// min w^T * Sigma * w  subject to: w^T * mu = target_return, sum(w) = 1
efficient_frontier :: proc(
	expected_returns: []f64,
	cov_matrix: ^l.Matrix(f64),
	risk_free_rate: f64 = 0.0,
	n_points: int = 20,
	allocator: mem.Allocator = context.allocator,
) -> []EfficientFrontierPoint {
	n := len(expected_returns)
	if cov_matrix.rows != n || cov_matrix.cols != n {
		panic("efficient_frontier: dimension mismatch")
	}

	// Find min and max returns for the frontier
	min_ret := expected_returns[0]
	max_ret := expected_returns[0]
	for i in 1 ..< n {
		if expected_returns[i] < min_ret {
			min_ret = expected_returns[i]
		}
		if expected_returns[i] > max_ret {
			max_ret = expected_returns[i]
		}
	}

	// Generate target returns
	frontier := make([]EfficientFrontierPoint, n_points, allocator)

	for k in 0 ..< n_points {
		target_return := min_ret + (max_ret - min_ret) * f64(k) / f64(n_points - 1)

		// Solve constrained optimization using Lagrange multipliers
		// This is a simplified approach - for production, use quadratic programming
		weights := _solve_constrained_portfolio(
			expected_returns,
			cov_matrix,
			target_return,
			allocator,
		)

		ret := portfolio_return(weights, expected_returns)
		vol := portfolio_volatility(weights, cov_matrix)
		sr := sharpe_ratio(ret, vol, risk_free_rate)

		frontier[k] = EfficientFrontierPoint {
			expected_return = ret,
			volatility      = vol,
			sharpe_ratio    = sr,
			weights         = weights,
		}
	}

	return frontier
}

// Helper: solve min w^T*Sigma*w subject to w^T*mu=target, sum(w)=1
_solve_constrained_portfolio :: proc(
	expected_returns: []f64,
	cov_matrix: ^l.Matrix(f64),
	target_return: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(expected_returns)

	// Use a simplified approach: start with equal weights and adjust
	// For a proper implementation, use quadratic programming
	weights := make([]f64, n, allocator)

	// Start with equal weights
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Iterative adjustment (simplified gradient descent)
	learning_rate := 0.01
	for _ in 0 ..< 100 {
		// Calculate current return
		current_return := portfolio_return(weights, expected_returns)

		// Adjust weights to move toward target return
		error := target_return - current_return
		for i in 0 ..< n {
			weights[i] += learning_rate * error * expected_returns[i]
		}

		// Normalize to sum to 1
		sum := 0.0
		for i in 0 ..< n {
			sum += weights[i]
		}
		for i in 0 ..< n {
			weights[i] /= sum
		}
	}

	return weights
}

// ============================================================================
// Utility Functions
// ============================================================================

// Calculate covariance matrix from returns data
covariance_matrix :: proc(
	returns_data: ^l.Matrix(f64), // rows = time periods, cols = assets
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	return l.covariance(returns_data, allocator)
}

// Calculate expected returns from historical data
expected_returns_from_history :: proc(
	returns_data: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_assets := returns_data.cols
	n_periods := returns_data.rows

	means := make([]f64, n_assets, allocator)

	for j in 0 ..< n_assets {
		sum := 0.0
		for i in 0 ..< n_periods {
			sum += returns_data.data[i * n_assets + j]
		}
		means[j] = sum / f64(n_periods)
	}

	return means
}

// Annualize returns (assuming daily data)
annualize_return :: proc(daily_return: f64, trading_days: int = 252) -> f64 {
	return daily_return * f64(trading_days)
}

// Annualize volatility
annualize_volatility :: proc(daily_volatility: f64, trading_days: int = 252) -> f64 {
	return daily_volatility * math.sqrt(f64(trading_days))
}
