package finance

import l "../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Risk-Based Portfolio Optimization
// ============================================================================

// Portfolio CVaR metrics
PortfolioCVaR :: struct {
	weights:         []f64, // The caller's weights (not copied)
	expected_return: f64,
	volatility:      f64,
	cvar_95:         f64, // 95% historical CVaR (Acerbi-Tasche), positive loss
	cvar_99:         f64, // 99% historical CVaR (Acerbi-Tasche), positive loss
	sharpe_ratio:    f64,
}

// ============================================================================
// EWMA Covariance Matrix
// ============================================================================

// RiskMetrics EWMA covariance (zero mean): Σ_t λ^{T-1-t} r_t r_tᵀ / Σ_t λ^{T-1-t},
// the most recent row weighted 1. Despite the name it is not fitted by GARCH; it
// is the IGARCH(1,1) / RiskMetrics special case with a fixed λ
garch_adjusted_covariance :: proc(
	returns_data: ^l.Matrix(f64), // rows = time, cols = assets
	lambda: f64 = 0.94, // EWMA decay factor (RiskMetrics standard)
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_periods := returns_data.rows
	n := returns_data.cols
	cov := l.matrix_new(f64, n, n, allocator)

	weight, weight_sum := 1.0, 0.0
	for t := n_periods - 1; t >= 0; t -= 1 {
		row := returns_data.data[t * n:(t + 1) * n]
		for i in 0 ..< n {
			wi := weight * row[i]
			for j in 0 ..= i {cov.data[i * n + j] += wi * row[j]}
		}
		weight_sum += weight
		weight *= lambda
	}
	for i in 0 ..< n {
		for j in 0 ..= i {
			v := cov.data[i * n + j] / weight_sum
			cov.data[i * n + j] = v
			cov.data[j * n + i] = v
		}
	}
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
		port_ret[t] = l.dot_simd(weights, returns_data.data[t * n_assets:(t + 1) * n_assets])
	}
	return port_ret
}

// Historical CVaR as a positive loss: the Acerbi-Tasche mean of the worst
// (1 - confidence)·n returns, the boundary return weighted by the fractional part
// (same estimator as conditional_var / expected_shortfall)
portfolio_cvar_historical :: proc(portfolio_returns: []f64, confidence: f64 = 0.95) -> f64 {
	return -conditional_var(portfolio_returns, confidence)
}

// Compute CVaR using EVT (Peak-over-Threshold)
portfolio_cvar_evt :: proc(
	portfolio_returns: []f64,
	confidence: f64 = 0.95,
	allocator: mem.Allocator,
) -> f64 {
	// Threshold at the confidence level, ES from the GPD tail at that level
	evt_result := evt_fit(portfolio_returns, confidence, allocator)
	_, es := evt_var_es(&evt_result, confidence)
	return -es // positive loss
}

// ============================================================================
// Minimum CVaR Portfolio Optimization
// ============================================================================

// Gradient of the historical CVaR (positive loss) with respect to the weights:
// minus the Acerbi-Tasche tail average of each asset's returns over the worst
// portfolio scenarios. CVaR is homogeneous of degree 1, so w_i · grad_i are the
// Euler (component) CVaR contributions and sum to the CVaR. Exact where the
// ordering of the scenarios has no ties
cvar_gradient :: proc(
	weights: []f64,
	returns_data: ^l.Matrix(f64),
	confidence: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	T := returns_data.rows
	grad := make([]f64, n, allocator)
	if T == 0 {return grad}

	port := portfolio_returns(weights, returns_data, context.temp_allocator)
	defer delete(port, context.temp_allocator)
	order := make([]int, T, context.temp_allocator)
	defer delete(order, context.temp_allocator)
	for i in 0 ..< T {order[i] = i}
	_sort_scenarios(order, port)

	m := clamp((1.0 - confidence) * f64(T), 0.0, f64(T))
	k := min(int(math.floor(m + 1e-9)), T)
	frac := max(m - f64(k), 0.0)
	denom := f64(k) + frac
	if denom == 0 {k, frac, denom = 1, 0, 1}
	for s in 0 ..< min(k + 1, T) {
		wt := 1.0 if s < k else frac
		if wt == 0 {continue}
		row := returns_data.data[order[s] * n:(order[s] + 1) * n]
		for i in 0 ..< n {grad[i] -= wt * row[i]}
	}
	for &g in grad {g /= denom}
	return grad
}

// Scenario indices sorted by ascending portfolio return
@(private = "file")
_sort_scenarios :: proc(order: []int, port: []f64) {
	context.user_ptr = raw_data(port)
	slice.sort_by(order, proc(a, b: int) -> bool {
		p := ([^]f64)(context.user_ptr)
		return p[a] < p[b]
	})
}

// Minimum historical CVaR portfolio under the given constraints.
//
// Rockafellar & Uryasev: CVaR(w) = min_ζ ζ + E[(L(w) - ζ)^+] / α with losses
// L_t = -r_tᵀw and α = 1 - confidence, a convex problem in (w, ζ). The kink is
// smoothed with μ·softplus(·/μ) (error ≤ μ·ln 2 / α) and minimised by accelerated
// projected gradient (FISTA with backtracking and restarts) onto the exact
// constraint projection, for a decreasing sequence of μ down to 1e-9 of the
// return scale. tol is the stopping tolerance on the weights per stage
min_cvar_portfolio :: proc(
	returns_data: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	confidence: f64 = 0.95,
	allocator: mem.Allocator,
	max_iter: int = 5000,
	tol: f64 = 1e-10,
) -> []f64 {
	n := returns_data.cols
	T := returns_data.rows
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {weights[i] = 1.0 / f64(n)}
	_project_constraints_exact(weights, constraints)
	if T == 0 || n == 0 {return weights}
	alpha := 1.0 - confidence

	// Return scale for the smoothing and the step size
	scale := 0.0
	for v in returns_data.data {scale += v * v}
	scale = math.sqrt(scale / f64(len(returns_data.data)))
	if scale == 0 {return weights}

	dim := n + 1 // x = (w, ζ)
	buf := make([]f64, 5 * dim, context.temp_allocator)
	defer delete(buf, context.temp_allocator)
	x, y, xn := buf[0:dim], buf[dim:2 * dim], buf[2 * dim:3 * dim]
	g, x_prev := buf[3 * dim:4 * dim], buf[4 * dim:5 * dim]

	// ζ starts at the VaR of the starting portfolio (its optimal value for fixed w)
	copy(x[:n], weights)
	{
		port := portfolio_returns(weights, returns_data, context.temp_allocator)
		slice.sort(port)
		x[n] = -port[clamp(int(math.ceil(alpha * f64(T) - 1e-9)) - 1, 0, T - 1)]
		delete(port, context.temp_allocator)
	}

	// Smoothed objective and gradient at z
	eval :: proc(z, grad: []f64, X: ^l.Matrix(f64), alpha, mu: f64, want_grad: bool) -> f64 {
		n := X.cols
		T := X.rows
		zeta := z[n]
		c := 1.0 / (alpha * f64(T))
		f := zeta
		if want_grad {
			for i in 0 ..< n {grad[i] = 0}
			grad[n] = 1.0
		}
		for t in 0 ..< T {
			row := X.data[t * n:(t + 1) * n]
			u := (-l.dot_simd(z[:n], row) - zeta) / mu
			// softplus and sigmoid, overflow-safe
			sp := math.max(u, 0.0) + math.log1p(math.exp(-math.abs(u)))
			f += c * mu * sp
			if want_grad {
				sg := 1.0 / (1.0 + math.exp(-u)) if u >= 0 else math.exp(u) / (1.0 + math.exp(u))
				for i in 0 ..< n {grad[i] -= c * sg * row[i]}
				grad[n] -= c * sg
			}
		}
		return f
	}

	project :: proc(z: []f64, n: int, c: PortfolioConstraints) {_project_constraints_exact(z[:n], c)}

	// Lipschitz estimate of the smoothed gradient: ‖(r_t, 1)‖² / (4 μ α T) summed
	row_sq := 0.0
	for t in 0 ..< T {
		row := returns_data.data[t * n:(t + 1) * n]
		row_sq = math.max(row_sq, l.dot_simd(row, row) + 1.0)
	}

	mu := 1e-2 * scale
	for mu >= 1e-9 * scale * 0.999 {
		copy(y, x)
		t_k := 1.0
		Lc := row_sq / (4.0 * mu * alpha * f64(T)) * 0.01
		f_x := eval(x, g, returns_data, alpha, mu, false)
		for _ in 0 ..< max_iter {
			f_y := eval(y, g, returns_data, alpha, mu, true)
			// backtracking on the quadratic upper bound
			f_n: f64
			for {
				for i in 0 ..< dim {xn[i] = y[i] - g[i] / Lc}
				project(xn, n, constraints)
				f_n = eval(xn, nil, returns_data, alpha, mu, false)
				q := f_y
				for i in 0 ..< dim {
					d := xn[i] - y[i]
					q += g[i] * d + 0.5 * Lc * d * d
				}
				if f_n <= q + 1e-15 * math.abs(q) {break}
				Lc *= 2.0
			}
			copy(x_prev, x)
			copy(x, xn)
			step := 0.0
			for i in 0 ..< n {step = math.max(step, math.abs(x[i] - x_prev[i]))}
			// restart the momentum when the objective goes up
			if f_n > f_x {
				t_k = 1.0
				copy(y, x)
			} else {
				t_next := 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
				beta := (t_k - 1.0) / t_next
				for i in 0 ..< dim {y[i] = x[i] + beta * (x[i] - x_prev[i])}
				project(y, n, constraints)
				t_k = t_next
			}
			f_x = f_n
			Lc *= 0.9 // let the step grow again
			if step < tol {break}
		}
		mu *= 0.1
	}

	copy(weights, x[:n])
	return weights
}

// ============================================================================
// Risk Parity Portfolio
// ============================================================================

// Risk contributions w_i (Σw)_i / sqrt(w'Σw); they sum to the portfolio volatility
// (all 0 when the portfolio variance is 0)
risk_contributions :: proc(
	weights: []f64,
	cov_matrix: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	contrib := make([]f64, n, allocator)
	sigma_w := l.matvec_dyn_simd(cov_matrix, weights, context.temp_allocator)
	defer delete(sigma_w, context.temp_allocator)
	port_vol := math.sqrt(math.max(l.dot_simd(weights, sigma_w), 0.0))
	if port_vol == 0 {return contrib}
	for i in 0 ..< n {contrib[i] = weights[i] * sigma_w[i] / port_vol}
	return contrib
}

// Long-only risk parity (equal risk contribution, or the given risk budgets b,
// Σb = 1): every asset's share of the portfolio volatility equals b_i.
// Cyclical coordinate descent on ½yᵀΣy - Σ b_i ln y_i (Griveau-Billion, Richard &
// Roncalli 2013), which converges for any positive definite Σ; w = y / Σy.
// The constraints are applied afterwards by exact projection: with binding
// bounds or group limits the contributions are then no longer equal
risk_parity_portfolio :: proc(
	cov_matrix: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-12,
	budgets: []f64 = nil,
) -> []f64 {
	n := cov_matrix.rows
	S := cov_matrix.data
	weights := make([]f64, n, allocator)
	if n == 0 {return weights}

	b := make([]f64, n, context.temp_allocator)
	defer delete(b, context.temp_allocator)
	for i in 0 ..< n {b[i] = budgets[i] if budgets != nil else 1.0 / f64(n)}

	// start from inverse volatility, Σy kept up to date
	y := weights
	for i in 0 ..< n {y[i] = 1.0 / math.sqrt(S[i * n + i])}
	sy := l.matvec_dyn_simd(cov_matrix, y, context.temp_allocator)
	defer delete(sy, context.temp_allocator)

	for _ in 0 ..< max_iter {
		max_rel := 0.0
		for i in 0 ..< n {
			sii := S[i * n + i]
			c := sy[i] - sii * y[i] // Σ_{j≠i} σ_ij y_j
			yi := (-c + math.sqrt(c * c + 4.0 * sii * b[i])) / (2.0 * sii)
			d := yi - y[i]
			if d != 0 {
				for k in 0 ..< n {sy[k] += S[k * n + i] * d}
				max_rel = math.max(max_rel, math.abs(d) / yi)
				y[i] = yi
			}
		}
		if max_rel < tol {break}
	}

	total := 0.0
	for v in y {total += v}
	for &v in y {v /= total}
	_project_constraints_exact(weights, constraints)
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
