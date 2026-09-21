package ml_finance

import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// ALMGREN-CHRISS OPTIMAL EXECUTION
// ============================================================================
// Minimizes Implementation Shortfall: E[Cost] + lambda * Var[Cost]
// Uses the exact discrete-time formulation for precise trading intervals.

AlmgrenChrissParams :: struct {
	initial_shares: f64, // X: Total shares to liquidate
	n_steps:        int, // N: Number of trading intervals
	T:              f64, // Total time horizon (e.g., 1.0 for 1 day)
	sigma:          f64, // Volatility of the asset price per unit time
	eta:            f64, // Temporary market impact coefficient
	gamma:          f64, // Permanent market impact coefficient
	lambda:         f64, // Risk aversion parameter (higher = trade faster)
}

AlmgrenChrissTrajectory :: struct {
	times:         []f64, // Time points [0, tau, 2*tau, ..., T]
	shares_held:   []f64, // Optimal inventory X_j at each time step
	trade_sizes:   []f64, // Shares traded v_j = X_{j-1} - X_j
	expected_cost: f64, // Expected Implementation Shortfall
	variance:      f64, // Variance of the execution cost
	allocator:     mem.Allocator,
}

almgren_chriss_free :: proc(traj: ^AlmgrenChrissTrajectory) {
	if traj.times != nil {delete(traj.times, traj.allocator)}
	if traj.shares_held != nil {delete(traj.shares_held, traj.allocator)}
	if traj.trade_sizes != nil {delete(traj.trade_sizes, traj.allocator)}
}

almgren_chriss_solve :: proc(
	params: AlmgrenChrissParams,
	allocator: mem.Allocator = context.allocator,
) -> AlmgrenChrissTrajectory {
	X := params.initial_shares
	N := params.n_steps
	T := params.T
	sigma := params.sigma
	eta := params.eta
	gamma := params.gamma
	lambda := params.lambda

	tau := T / f64(N)

	// Discrete kappa: cosh(kappa * tau) = 1 + (lambda * sigma^2 * tau^2) / (2 * eta)
	arg := 1.0 + (lambda * sigma * sigma * tau * tau) / (2.0 * eta)
	if arg < 1.0 {arg = 1.0} 	// Safety for acosh
	kappa := math.acosh(arg) / tau

	traj: AlmgrenChrissTrajectory
	traj.allocator = allocator
	traj.times = make([]f64, N + 1, allocator)
	traj.shares_held = make([]f64, N + 1, allocator)
	traj.trade_sizes = make([]f64, N, allocator)

	sinh_kT := math.sinh(kappa * T)

	// 1. Compute Optimal Trajectory
	for j in 0 ..= N {
		t_j := f64(j) * tau
		traj.times[j] = t_j
		traj.shares_held[j] = X * math.sinh(kappa * (T - t_j)) / sinh_kT
	}

	// Enforce exact boundary conditions
	traj.shares_held[0] = X
	traj.shares_held[N] = 0.0

	// 2. Compute Trades
	for j in 0 ..< N {
		traj.trade_sizes[j] = traj.shares_held[j] - traj.shares_held[j + 1]
	}

	// 3. Compute Expected Cost and Variance
	// E[Cost] = 0.5 * gamma * X^2 + sum( (eta / tau) * v_j^2 )
	// Var[Cost] = sigma^2 * sum( tau * X_j^2 )
	expected_temp_cost := 0.0
	variance_cost := 0.0

	for j in 0 ..< N {
		v_j := traj.trade_sizes[j]
		X_j := traj.shares_held[j + 1] // Inventory after trade j

		expected_temp_cost += (eta / tau) * v_j * v_j
		variance_cost += tau * X_j * X_j
	}

	traj.expected_cost = 0.5 * gamma * X * X + expected_temp_cost
	traj.variance = sigma * sigma * variance_cost

	return traj
}

// ============================================================================
// AVELLANEDA-STOIKOV MARKET MAKING
// ============================================================================
// Computes optimal bid/ask quotes based on current inventory and time to horizon.
// Maximizes expected utility of terminal wealth while penalizing inventory risk.

AvellanedaStoikovParams :: struct {
	sigma: f64, // Volatility of the mid-price
	gamma: f64, // Risk aversion parameter
	k:     f64, // Order book density / arrival rate decay parameter
	T:     f64, // Terminal time
}

ASQuotes :: struct {
	reservation_price: f64, // Indifference price
	bid_quote:         f64, // Absolute bid price
	ask_quote:         f64, // Absolute ask price
	bid_spread:        f64, // mid - bid
	ask_spread:        f64, // ask - mid
	optimal_spread:    f64, // ask - bid
}

avellaneda_stoikov_quote :: proc(
	mid_price: f64,
	inventory: f64,
	current_time: f64,
	params: AvellanedaStoikovParams,
) -> ASQuotes {
	dt := params.T - current_time
	if dt < 1e-8 {dt = 1e-8} 	// Prevent division by zero at terminal time

	sigma := params.sigma
	gamma := params.gamma
	k := params.k
	q := inventory

	// 1. Reservation Price
	// r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
	r := mid_price - q * gamma * sigma * sigma * dt

	// 2. Optimal Spread
	// The spread is symmetric around the reservation price.
	shift := (1.0 / gamma) * math.ln(1.0 + gamma / k)

	// 3. Individual Quotes
	bid := r - shift
	ask := r + shift

	return ASQuotes {
		reservation_price = r,
		bid_quote = bid,
		ask_quote = ask,
		bid_spread = mid_price - bid,
		ask_spread = ask - mid_price,
		optimal_spread = ask - bid,
	}
}
