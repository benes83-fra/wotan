// wotan/finance/constraints.odin
package finance

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Portfolio Constraints
// ============================================================================
PortfolioConstraints :: struct {
	no_short_selling: bool, // weights >= 0
	max_weight:       f64, // weights <= max_weight (0 = no limit)
	min_weight:       f64, // weights >= min_weight if included (0 = no limit)
	group_limits:     []GroupLimit, // sector/asset class limits
}

GroupLimit :: struct {
	asset_indices: []int, // indices of assets in this group
	max_weight:    f64, // sum of weights in group <= max_weight
}

// ============================================================================
// Projection Operators
// ============================================================================

// Project onto simplex: weights sum to 1, weights >= 0
// Uses Duchi et al. algorithm (O(n log n))
project_simplex :: proc(weights: []f64, allocator: mem.Allocator) -> []f64 {
	n := len(weights)
	result := make([]f64, n, allocator)
	copy(result, weights)

	// Sort in descending order
	sorted := make([]f64, n, allocator)
	copy(sorted, result)

	// Simple insertion sort for small n, or use proper sort
	for i in 1 ..< n {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] < key {
			sorted[j + 1] = sorted[j]
			j -= 1
		}
		sorted[j + 1] = key
	}

	// Find rho
	cumsum := 0.0
	rho := -1
	for j in 0 ..< n {
		cumsum += sorted[j]
		if sorted[j] + (1.0 - cumsum) / f64(j + 1) > 0.0 {
			rho = j
		} else {
			break
		}
	}

	if rho < 0 {
		// All weights should be zero (shouldn't happen with sum=1)
		for i in 0 ..< n {result[i] = 1.0 / f64(n)}
		return result
	}

	// Compute theta
	cumsum = 0.0
	for j in 0 ..= rho {
		cumsum += sorted[j]
	}
	theta := (cumsum - 1.0) / f64(rho + 1)

	// Project
	for i in 0 ..< n {
		result[i] = math.max(weights[i] - theta, 0.0)
	}

	delete(sorted, allocator)
	return result
}

// Project onto box constraints: min_weight <= weights <= max_weight
project_box :: proc(weights: []f64, min_w: f64, max_w: f64) {
	for i in 0 ..< len(weights) {
		if weights[i] < min_w {
			weights[i] = min_w
		}
		if max_w > 0.0 && weights[i] > max_w {
			weights[i] = max_w
		}
	}
}

// Project onto group constraints
project_groups :: proc(weights: []f64, groups: []GroupLimit) {
	for group in groups {
		// Sum current weights in group
		sum := 0.0
		for idx in group.asset_indices {
			if idx >= 0 && idx < len(weights) {
				sum += weights[idx]
			}
		}

		// If sum exceeds limit, scale down proportionally
		if sum > group.max_weight && sum > 1e-10 {
			scale := group.max_weight / sum
			for idx in group.asset_indices {
				if idx >= 0 && idx < len(weights) {
					weights[idx] *= scale
				}
			}
		}
	}
}

// Combined projection: apply all constraints iteratively
project_constraints :: proc(
	weights: []f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
) -> []f64 {
	result := weights

	// Project onto simplex first (ensures sum=1, >=0 if no_short_selling)
	if constraints.no_short_selling {
		result = project_simplex(result, allocator)
	} else {
		// Just ensure sum=1
		sum := 0.0
		for w in result {sum += w}
		if math.abs(sum) > 1e-10 {
			for i in 0 ..< len(result) {
				result[i] /= sum
			}
		}
	}

	// Apply box constraints
	if constraints.max_weight > 0.0 || constraints.min_weight > 0.0 {
		project_box(result, constraints.min_weight, constraints.max_weight)
	}

	// Apply group constraints
	if len(constraints.group_limits) > 0 {
		project_groups(result, constraints.group_limits)
	}

	// Re-normalize if needed
	sum := 0.0
	for w in result {sum += w}
	if math.abs(sum - 1.0) > 1e-6 {
		for i in 0 ..< len(result) {
			result[i] /= sum
		}
	}

	return result
}

// ============================================================================
// Constrained Portfolio Optimization
// ============================================================================

// Compute portfolio variance: w' Σ w
portfolio_variance :: proc(weights: []f64, cov: ^l.Matrix(f64)) -> f64 {
	n := len(weights)
	var := 0.0

	for i in 0 ..< n {
		for j in 0 ..< n {
			var += weights[i] * weights[j] * cov.data[i * cov.cols + j]
		}
	}

	return var
}

// Compute portfolio variance gradient: 2 Σ w
portfolio_variance_gradient :: proc(
	weights: []f64,
	cov: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	grad := make([]f64, n, allocator)

	for i in 0 ..< n {
		for j in 0 ..< n {
			grad[i] += 2.0 * cov.data[i * cov.cols + j] * weights[j]
		}
	}

	return grad
}

// Compute negative Sharpe ratio
negative_sharpe :: proc(weights: []f64, returns: []f64, cov: ^l.Matrix(f64), rf: f64) -> f64 {
	port_return := 0.0
	for i in 0 ..< len(weights) {
		port_return += weights[i] * returns[i]
	}

	port_var := portfolio_variance(weights, cov)
	port_vol := math.sqrt(port_var)

	if port_vol < 1e-10 {
		return 1e10 // Avoid division by zero
	}

	return -(port_return - rf) / port_vol
}

// Compute negative Sharpe gradient
negative_sharpe_gradient :: proc(
	weights: []f64,
	returns: []f64,
	cov: ^l.Matrix(f64),
	rf: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)

	port_return := 0.0
	for i in 0 ..< n {
		port_return += weights[i] * returns[i]
	}

	port_var := portfolio_variance(weights, cov)
	port_vol := math.sqrt(port_var)

	if port_vol < 1e-10 {
		grad := make([]f64, n, allocator)
		return grad
	}

	// Gradient of -(μ - rf) / σ
	// = -[σ * μ - (μ - rf) * (Σw)/σ] / σ²
	grad := make([]f64, n, allocator)

	var_grad := portfolio_variance_gradient(weights, cov, allocator)

	for i in 0 ..< n {
		grad[i] =
			-(returns[i] * port_vol - (port_return - rf) * var_grad[i] / port_vol) /
			(port_vol * port_vol)
	}

	delete(var_grad, allocator)
	return grad
}

// Constrained minimum variance portfolio
constrained_min_variance_portfolio :: proc(
	cov: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := cov.rows

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Project initial weights
	weights = project_constraints(weights, constraints, allocator)

	// Projected gradient descent
	lr := 0.01
	prev_var := portfolio_variance(weights, cov)

	for iter in 0 ..< max_iter {
		// Compute gradient
		grad := portfolio_variance_gradient(weights, cov, allocator)

		// Gradient step
		for i in 0 ..< n {
			weights[i] -= lr * grad[i]
		}

		// Project onto constraints
		weights = project_constraints(weights, constraints, allocator)

		// Check convergence
		var := portfolio_variance(weights, cov)
		if math.abs(var - prev_var) < tol {
			delete(grad, allocator)
			break
		}
		prev_var = var

		delete(grad, allocator)
	}

	return weights
}

// Constrained maximum Sharpe ratio portfolio
constrained_max_sharpe_portfolio :: proc(
	returns: []f64,
	cov: ^l.Matrix(f64),
	rf: f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := len(returns)

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Project initial weights
	weights = project_constraints(weights, constraints, allocator)

	// Projected gradient descent
	lr := 0.01
	prev_sharpe := negative_sharpe(weights, returns, cov, rf)

	for iter in 0 ..< max_iter {
		// Compute gradient
		grad := negative_sharpe_gradient(weights, returns, cov, rf, allocator)

		// Gradient step
		for i in 0 ..< n {
			weights[i] -= lr * grad[i]
		}

		// Project onto constraints
		weights = project_constraints(weights, constraints, allocator)

		// Check convergence
		sharpe := negative_sharpe(weights, returns, cov, rf)
		if math.abs(sharpe - prev_sharpe) < tol {
			delete(grad, allocator)
			break
		}
		prev_sharpe = sharpe

		delete(grad, allocator)
	}

	return weights
}

// Constrained mean-variance portfolio (target return)
constrained_mean_variance_portfolio :: proc(
	returns: []f64,
	cov: ^l.Matrix(f64),
	target_return: f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := len(returns)

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Project initial weights
	weights = project_constraints(weights, constraints, allocator)

	// Projected gradient descent with penalty for target return
	lr := 0.01
	penalty := 10.0
	prev_obj := 0.0

	for iter in 0 ..< max_iter {
		// Compute variance gradient
		var_grad := portfolio_variance_gradient(weights, cov, allocator)

		// Compute return gradient
		ret_grad := make([]f64, n, allocator)
		for i in 0 ..< n {
			ret_grad[i] = returns[i]
		}

		// Combined gradient with penalty
		current_return := 0.0
		for i in 0 ..< n {
			current_return += weights[i] * returns[i]
		}

		return_error := current_return - target_return

		for i in 0 ..< n {
			weights[i] -= lr * (var_grad[i] + penalty * return_error * ret_grad[i])
		}

		// Project onto constraints
		weights = project_constraints(weights, constraints, allocator)

		// Check convergence
		obj := portfolio_variance(weights, cov) + penalty * return_error * return_error
		if math.abs(obj - prev_obj) < tol {
			delete(var_grad, allocator)
			delete(ret_grad, allocator)
			break
		}
		prev_obj = obj

		delete(var_grad, allocator)
		delete(ret_grad, allocator)
	}

	return weights
}
// ============================================================================
// Strict Constraint Enforcement
// ============================================================================

// enforce_constraints strictly applies all constraints to the weights
// Call this AFTER the optimization solver returns
enforce_constraints :: proc(weights: []f64, config: PortfolioConstraints) -> []f64 {
	result := make([]f64, len(weights), context.temp_allocator)
	copy(result, weights)

	// 1. Apply no-short-selling (clamp to >= 0)
	if config.no_short_selling {
		for i in 0 ..< len(result) {
			if result[i] < 0.0 {
				result[i] = 0.0
			}
		}
	}

	// 2. Apply max position size (clamp to <= max_weight)
	if config.max_weight > 0.0 {
		for i in 0 ..< len(result) {
			if result[i] > config.max_weight {
				result[i] = config.max_weight
			}
		}
	}

	// 3. Apply sector constraints
	if len(config.group_limits) > 0 {
		for group in config.group_limits {
			// Calculate current sum for this group
			group_sum := 0.0
			for idx in group.asset_indices {
				if idx >= 0 && idx < len(result) {
					group_sum += result[idx]
				}
			}

			// If sum exceeds limit, scale down proportionally
			if group_sum > group.max_weight && group_sum > 1e-10 {
				scale := group.max_weight / group_sum
				for idx in group.asset_indices {
					if idx >= 0 && idx < len(result) {
						result[idx] *= scale
					}
				}
			}
		}
	}

	// 4. Re-normalize to sum to 1.0
	total := 0.0
	for w in result {
		total += w
	}

	if total > 1e-10 {
		inv_total := 1.0 / total
		for i in 0 ..< len(result) {
			result[i] *= inv_total
		}
	} else {
		// Fallback: equal weights
		equal_w := 1.0 / f64(len(result))
		for i in 0 ..< len(result) {
			result[i] = equal_w
		}
	}

	// 5. Final pass: re-apply box constraints after normalization
	// (normalization might have pushed weights slightly out of bounds)
	if config.no_short_selling || config.max_weight > 0.0 {
		for i in 0 ..< len(result) {
			if config.no_short_selling && result[i] < 0.0 {
				result[i] = 0.0
			}
			if config.max_weight > 0.0 && result[i] > config.max_weight {
				result[i] = config.max_weight
			}
		}
	}

	return result
}
