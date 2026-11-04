package ml_finance

import "core:math"
import "core:mem"

// ============================================================================
// ALMGREN-CHRISS OPTIMAL EXECUTION
// ============================================================================
// Minimizes Implementation Shortfall: E[Cost] + lambda * Var[Cost]
// Exact discrete-time solution (Almgren & Chriss 2000, section 3).

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

// Almgren & Chriss (2000), "Optimal execution of portfolio transactions", linear impact:
//   S_k = S_{k-1} + sigma sqrt(tau) xi_k - gamma n_k,   received S_{k-1} - (eta / tau) n_k.
// With eta~ = eta - gamma tau / 2 the shortfall has
//   E = gamma X^2 / 2 + (eta~ / tau) sum n_k^2,   V = sigma^2 tau sum_{k=1..N} x_k^2,
// minimised by x_j = X sinh(kappa (T - t_j)) / sinh(kappa T), with
//   cosh(kappa tau) = 1 + lambda sigma^2 tau^2 / (2 eta~);  lambda = 0 gives TWAP.
// Needs n_steps >= 1, lambda >= 0 and eta~ > 0 (else the cost is not convex): otherwise
// the trajectory is empty and the costs NaN. The fixed cost epsilon |X| is not modelled.
almgren_chriss_solve :: proc(
	params: AlmgrenChrissParams,
	allocator: mem.Allocator = context.allocator,
) -> AlmgrenChrissTrajectory {
	X := params.initial_shares
	N := params.n_steps
	T := params.T
	sigma := params.sigma
	gamma := params.gamma
	lambda := params.lambda

	traj := AlmgrenChrissTrajectory {
		allocator     = allocator,
		expected_cost = math.nan_f64(),
		variance      = math.nan_f64(),
	}
	if N < 1 || !(T > 0.0) {return traj}
	tau := T / f64(N)
	eta_t := params.eta - 0.5 * gamma * tau
	if !(eta_t > 0.0) || !(lambda >= 0.0) {return traj}

	// kappa tau = acosh(1 + d) = log1p(d + sqrt(d (2 + d))), accurate for small d
	d := lambda * sigma * sigma * tau * tau / (2.0 * eta_t)
	kappa := math.log1p(d + math.sqrt(d * (2.0 + d))) / tau

	traj.times = make([]f64, N + 1, allocator)
	traj.shares_held = make([]f64, N + 1, allocator)
	traj.trade_sizes = make([]f64, N, allocator)

	// sinh(a) / sinh(b) = e^(a-b) (1 - e^-2a) / (1 - e^-2b), and 1 - e^-2a = 2 tanh a / (1 + tanh a):
	// no overflow for large kappa T, no cancellation for small kappa T
	g :: proc(a: f64) -> f64 {th := math.tanh(a); return th / (1.0 + th)}
	kT := kappa * T
	for j in 0 ..= N {
		t_j := f64(j) * tau
		traj.times[j] = t_j
		if kappa == 0.0 {
			traj.shares_held[j] = X * (1.0 - f64(j) / f64(N))
		} else {
			a := kappa * (T - t_j)
			traj.shares_held[j] = X * math.exp(a - kT) * g(a) / g(kT)
		}
	}
	traj.shares_held[0] = X
	traj.shares_held[N] = 0.0

	sum_n2, sum_x2 := 0.0, 0.0
	for j in 0 ..< N {
		n_j := traj.shares_held[j] - traj.shares_held[j + 1]
		traj.trade_sizes[j] = n_j
		sum_n2 += n_j * n_j
		x_j := traj.shares_held[j + 1] // inventory after trade j
		sum_x2 += x_j * x_j
	}
	traj.expected_cost = 0.5 * gamma * X * X + eta_t / tau * sum_n2
	traj.variance = sigma * sigma * tau * sum_x2
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

// Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book":
//   r = s - q gamma sigma^2 (T - t),   delta_a + delta_b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k),
// quotes symmetric around r. gamma -> 0 gives the risk-neutral half-spread 1/k; past T, T - t = 0.
avellaneda_stoikov_quote :: proc(
	mid_price: f64,
	inventory: f64,
	current_time: f64,
	params: AvellanedaStoikovParams,
) -> ASQuotes {
	dt := math.max(params.T - current_time, 0.0)
	sigma := params.sigma
	gamma := params.gamma
	k := params.k
	q := inventory

	r := mid_price - q * gamma * sigma * sigma * dt
	// (1/gamma) ln(1 + gamma/k) via log1p, exact limit 1/k at gamma = 0
	liquidity := gamma == 0.0 ? 1.0 / k : math.log1p(gamma / k) / gamma
	half := 0.5 * gamma * sigma * sigma * dt + liquidity

	bid := r - half
	ask := r + half
	return ASQuotes {
		reservation_price = r,
		bid_quote = bid,
		ask_quote = ask,
		bid_spread = mid_price - bid,
		ask_spread = ask - mid_price,
		optimal_spread = ask - bid,
	}
}
