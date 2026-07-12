package finance

import a "../analytics"
import l "../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Risk-Based Portfolio Optimization
// ============================================================================

// Portfolio CVaR metrics
PortfolioCVaR :: struct {
	weights:         []f64,
	expected_return: f64,
	volatility:      f64,
	cvar_95:         f64, // 95% Conditional VaR
	cvar_99:         f64, // 99% Conditional VaR
	sharpe_ratio:    f64,
}

// ============================================================================
// GARCH-Adjusted Covariance Matrix
// ============================================================================

// Compute GARCH-adjusted covariance matrix using EWMA volatility
// This gives more weight to recent observations
garch_adjusted_covariance :: proc(
	returns_data: ^l.Matrix(f64), // rows = time, cols = assets
	lambda: f64 = 0.94, // EWMA decay factor (RiskMetrics standard)
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_periods := returns_data.rows
	n_assets := returns_data.cols

	cov := l.matrix_new(f64, n_assets, n_assets, allocator)

	// Initialize with simple covariance
	for i in 0 ..< n_assets {
		for j in 0 ..< n_assets {
			sum := 0.0
			for t in 0 ..< n_periods {
				sum += returns_data.data[t * n_assets + i] * returns_data.data[t * n_assets + j]
			}
			cov.data[i * n_assets + j] = sum / f64(n_periods)
		}
	}

	// Apply EWMA weighting (more weight to recent observations)
	ewma_cov := l.matrix_new(f64, n_assets, n_assets, allocator)
	weight_sum := 0.0

	for t in 0 ..< n_periods {
		weight := math.pow(lambda, f64(n_periods - 1 - t))
		weight_sum += weight

		for i in 0 ..< n_assets {
			for j in 0 ..< n_assets {
				ewma_cov.data[i * n_assets + j] +=
					weight *
					returns_data.data[t * n_assets + i] *
					returns_data.data[t * n_assets + j]
			}
		}
	}

	// Normalize
	for i in 0 ..< n_assets {
		for j in 0 ..< n_assets {
			cov.data[i * n_assets + j] = ewma_cov.data[i * n_assets + j] / weight_sum
		}
	}

	l.matrix_free(&ewma_cov)
	return cov
}

// ============================================================================
// Portfolio CVaR Calculation
// ============================================================================

// Compute portfolio returns from asset returns
portfolio_returns :: proc(
	weights: []f64,
	returns_data: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> []f64 {
	n_periods := returns_data.rows
	n_assets := returns_data.cols

	port_ret := make([]f64, n_periods, allocator)

	for t in 0 ..< n_periods {
		sum := 0.0
		for i in 0 ..< n_assets {
			sum += weights[i] * returns_data.data[t * n_assets + i]
		}
		port_ret[t] = sum
	}

	return port_ret
}

// Compute CVaR using historical simulation
portfolio_cvar_historical :: proc(portfolio_returns: []f64, confidence: f64 = 0.95) -> f64 {
	// CVaR is the average of losses beyond VaR
	sorted := make([]f64, len(portfolio_returns), context.temp_allocator)
	copy(sorted, portfolio_returns)
	slice.sort(sorted)

	cutoff := int((1.0 - confidence) * f64(len(sorted)))
	if cutoff < 1 {cutoff = 1}

	sum := 0.0
	for i in 0 ..< cutoff {
		sum += sorted[i] // Negative returns are losses
	}

	return -sum / f64(cutoff) // Return as positive loss
}

// Compute CVaR using EVT (Peak-over-Threshold)
portfolio_cvar_evt :: proc(
	portfolio_returns: []f64,
	confidence: f64 = 0.95,
	allocator: mem.Allocator,
) -> f64 {
	evt_result := evt_fit(portfolio_returns, confidence, allocator)
	return -evt_result.cvar_95 // Convert to positive loss
}

// ============================================================================
// Minimum CVaR Portfolio Optimization
// ============================================================================

// Compute CVaR gradient using finite differences
cvar_gradient :: proc(
	weights: []f64,
	returns_data: ^l.Matrix(f64),
	confidence: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	grad := make([]f64, n, allocator)
	eps := 1e-5

	// Base CVaR
	base_port_ret := portfolio_returns(weights, returns_data, context.temp_allocator)
	defer delete(base_port_ret, context.temp_allocator)
	base_cvar := portfolio_cvar_historical(base_port_ret, confidence)

	for i in 0 ..< n {
		// Perturb weight i
		weights_plus := make([]f64, n, context.temp_allocator)
		copy(weights_plus, weights)
		weights_plus[i] += eps

		// Renormalize
		sum := 0.0
		for w in weights_plus {sum += w}
		for j in 0 ..< n {weights_plus[j] /= sum}

		// Compute CVaR
		port_ret_plus := portfolio_returns(weights_plus, returns_data, context.temp_allocator)
		cvar_plus := portfolio_cvar_historical(port_ret_plus, confidence)

		grad[i] = (cvar_plus - base_cvar) / eps

		delete(weights_plus, context.temp_allocator)
		delete(port_ret_plus, context.temp_allocator)
	}

	return grad
}

// Minimum CVaR portfolio using projected gradient descent
min_cvar_portfolio :: proc(
	returns_data: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	confidence: f64 = 0.95,
	allocator: mem.Allocator,
	max_iter: int = 500,
	tol: f64 = 1e-5,
) -> []f64 {
	n := returns_data.cols

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	weights = project_constraints(weights, constraints, allocator)

	// Projected gradient descent
	lr := 0.01
	prev_cvar: f64 = 1000000.0 // FIX: Explicitly typed as f64 (or use a decimal point)

	for iter in 0 ..< max_iter {
		// Compute CVaR gradient
		grad := cvar_gradient(weights, returns_data, confidence, allocator)

		// Gradient step
		for i in 0 ..< n {
			weights[i] -= lr * grad[i]
		}

		// Project onto constraints
		weights = project_constraints(weights, constraints, allocator)

		// Check convergence
		port_ret := portfolio_returns(weights, returns_data, context.temp_allocator)
		current_cvar := portfolio_cvar_historical(port_ret, confidence)
		delete(port_ret, context.temp_allocator)

		if math.abs(current_cvar - prev_cvar) < tol {
			delete(grad, allocator)
			break
		}
		prev_cvar = current_cvar

		delete(grad, allocator)
	}

	return weights
}

// ============================================================================
// Risk Parity Portfolio
// ============================================================================

// Compute risk contributions: w_i * (Σw)_i / sqrt(w'Σw)
risk_contributions :: proc(
	weights: []f64,
	cov_matrix: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	contrib := make([]f64, n, allocator)

	// Compute Σw
	sigma_w := l.matvec_dyn_simd(cov_matrix, weights, allocator)
	defer delete(sigma_w, allocator)

	// Portfolio volatility
	port_var := l.dot_simd(weights, sigma_w)
	port_vol := math.sqrt(port_var)

	// Risk contribution of each asset
	for i in 0 ..< n {
		contrib[i] = weights[i] * sigma_w[i] / port_vol
	}

	return contrib
}

// Risk parity portfolio: equalize risk contributions
risk_parity_portfolio :: proc(
	cov_matrix: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := cov_matrix.rows

	// Initialize with inverse volatility weights
	weights := make([]f64, n, allocator)
	sum_inv_vol := 0.0

	for i in 0 ..< n {
		vol := math.sqrt(cov_matrix.data[i * n + i])
		weights[i] = 1.0 / vol
		sum_inv_vol += weights[i]
	}

	for i in 0 ..< n {
		weights[i] /= sum_inv_vol
	}

	weights = project_constraints(weights, constraints, allocator)

	// Iterative optimization to equalize risk contributions
	target_risk := 1.0 / f64(n)

	for iter in 0 ..< max_iter {
		contrib := risk_contributions(weights, cov_matrix, allocator)

		// Check convergence
		max_diff := 0.0
		for i in 0 ..< n {
			diff := math.abs(contrib[i] - target_risk)
			if diff > max_diff {max_diff = diff}
		}

		if max_diff < tol {
			delete(contrib, allocator)
			break
		}

		// Adjust weights to equalize risk
		for i in 0 ..< n {
			// Scale weight by ratio of target to actual risk contribution
			if contrib[i] > 1e-10 {
				weights[i] *= target_risk / contrib[i]
			}
		}

		// Project onto constraints
		weights = project_constraints(weights, constraints, allocator)

		delete(contrib, allocator)
	}

	return weights
}

// ============================================================================
// Portfolio Metrics with CVaR
// ============================================================================

// Comprehensive portfolio metrics including CVaR
portfolio_metrics_with_cvar :: proc(
	weights: []f64,
	expected_returns: []f64,
	returns_data: ^l.Matrix(f64),
	cov_matrix: ^l.Matrix(f64),
	risk_free_rate: f64 = 0.0,
	allocator: mem.Allocator,
) -> PortfolioCVaR {
	ret := portfolio_return(weights, expected_returns)
	vol := portfolio_volatility(weights, cov_matrix)
	sr := sharpe_ratio(ret, vol, risk_free_rate)

	// Compute CVaR from historical returns
	port_ret := portfolio_returns(weights, returns_data, allocator)
	cvar_95 := portfolio_cvar_historical(port_ret, 0.95)
	cvar_99 := portfolio_cvar_historical(port_ret, 0.99)
	delete(port_ret, allocator)

	return PortfolioCVaR {
		weights = weights,
		expected_return = ret,
		volatility = vol,
		cvar_95 = cvar_95,
		cvar_99 = cvar_99,
		sharpe_ratio = sr,
	}
}
