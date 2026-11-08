package behavioral_finance

import "core:math"
import la "core:math/linalg"

// ============================================================================
// Investor sentiment: Barberis, Shleifer & Vishny (1998)
// ============================================================================
// Earnings follow a random walk, N_t = N_{t-1} + y_t with y_t = +y or -y equally
// likely, and are paid out. The risk-neutral representative investor (discount rate
// delta) wrongly believes y_t comes from one of two regimes:
//   Model 1 (mean-reverting): the next shock has the same sign with probability pi_L < 1/2
//   Model 2 (trending):       the same sign with probability pi_H > 1/2
// switching from 1 to 2 with probability lambda1 and from 2 to 1 with lambda2 a period.
// q_t = Pr(regime 1 | shocks so far) is updated by Bayes' rule: it falls after two
// shocks of the same sign and rises after a reversal. The price (Proposition 1) is
//   P_t = N_t / delta + y_t (p1 - p2 q_t),
// p1 = (1/delta) c0' (1 + delta) [I (1 + delta) - Q]^-1 Q c1 and p2 the same with -c2
// (appendix), so prices underreact to a single shock (conservatism, regime 1 believed
// on average) and overreact to a series of shocks of the same sign (representativeness,
// regime 2 after a series). Proposition 2 gives sufficient conditions on
// (pi_L, pi_H, lambda1, lambda2) for both.

BSV_Params :: struct {
	pi_L:    f64, // Model 1: probability that the next shock has the same sign
	pi_H:    f64, // Model 2
	lambda1: f64, // regime 1 -> 2 a period
	lambda2: f64, // regime 2 -> 1
	delta:   f64, // discount rate (prices only)
}

// The paper's parameters (Table 1, simulations): pi_L = 1/3, pi_H = 3/4,
// lambda1 = 0.1, lambda2 = 0.3. BSV do not state delta; 0.1 here.
BSV_1998 :: BSV_Params{1.0 / 3, 0.75, 0.1, 0.3, 0.1}

// q_{t+1} from q_t after a shock with the same sign as the last (same_sign) or the
// opposite sign
bsv_update_belief :: proc(q: f64, same_sign: bool, p := BSV_1998) -> f64 {
	a := (1 - p.lambda1) * q + p.lambda2 * (1 - q) // Pr(regime 1 next period)
	b := p.lambda1 * q + (1 - p.lambda2) * (1 - q)
	l1, l2 := p.pi_L, p.pi_H
	if !same_sign {l1, l2 = 1 - p.pi_L, 1 - p.pi_H}
	return a * l1 / (a * l1 + b * l2)
}

// q_t along a series of shocks (only their signs matter): q[0] = q0 is the belief after
// shocks[0]
bsv_beliefs :: proc(
	shocks: []f64,
	q0 := 0.5,
	p := BSV_1998,
	allocator := context.allocator,
) -> []f64 {
	q := make([]f64, len(shocks), allocator)
	if len(shocks) == 0 {return q}
	q[0] = q0
	for t in 1 ..< len(shocks) {
		q[t] = bsv_update_belief(q[t - 1], (shocks[t] > 0) == (shocks[t - 1] > 0), p)
	}
	return q
}

// p1, p2 of Proposition 1 (appendix): states (regime, next shock same / opposite sign
// as y_t), Q the transpose of their transition matrix
bsv_price_coefficients :: proc(p := BSV_1998) -> (p1, p2: f64) {
	assert(p.delta > 0, "bsv_price_coefficients: delta must be positive")
	l1, l2, L, H := p.lambda1, p.lambda2, p.pi_L, p.pi_H
	Q := matrix[4, 4]f64{
		(1 - l1) * L, (1 - l1) * (1 - L), l2 * L, l2 * (1 - L),
		(1 - l1) * (1 - L), (1 - l1) * L, l2 * (1 - L), l2 * L,
		l1 * H, l1 * (1 - H), (1 - l2) * H, (1 - l2) * (1 - H),
		l1 * (1 - H), l1 * H, (1 - l2) * (1 - H), (1 - l2) * H,
	}
	c0 := [4]f64{1, -1, 1, -1}
	c1 := [4]f64{0, 0, 1, 0}
	c2 := [4]f64{1, 0, -1, 0}
	d := p.delta
	A := la.inverse((1 + d) * la.MATRIX4F64_IDENTITY - Q) * Q
	p1 = (1 + d) / d * la.dot(c0, A * c1)
	p2 = -(1 + d) / d * la.dot(c0, A * c2)
	return
}

// P_t = N_t / delta + y_t (p1 - p2 q_t)
bsv_price :: proc(N, y, q, p1, p2, delta: f64) -> f64 {
	return N / delta + y * (p1 - p2 * q)
}

// Earnings N_t, beliefs q_t and prices P_t along a series of shocks y_t (N_t = N0 +
// y_0 + .. + y_t); q_0 = q0
BSV_Path :: struct {
	earnings, beliefs, prices: []f64,
}

bsv_path_free :: proc(b: ^BSV_Path, allocator := context.allocator) {
	delete(b.earnings, allocator)
	delete(b.beliefs, allocator)
	delete(b.prices, allocator)
}

bsv_simulate_path :: proc(
	shocks: []f64,
	N0: f64,
	q0 := 0.5,
	p := BSV_1998,
	allocator := context.allocator,
) -> BSV_Path {
	p1, p2 := bsv_price_coefficients(p)
	path := BSV_Path {
		earnings = make([]f64, len(shocks), allocator),
		beliefs  = bsv_beliefs(shocks, q0, p, allocator),
		prices   = make([]f64, len(shocks), allocator),
	}
	n := N0
	for y, t in shocks {
		n += y
		path.earnings[t] = n
		path.prices[t] = bsv_price(n, y, path.beliefs[t], p1, p2, p.delta)
	}
	return path
}

// The limits of q_t after an endless series of same-sign shocks (q_low) and of
// alternating shocks (q_high); q_t stays in [q_low, q_high] once inside (BSV: 0.28,
// 0.95 for their parameters)
bsv_belief_bounds :: proc(p := BSV_1998) -> (q_low, q_high: f64) {
	fixed :: proc(same: bool, p: BSV_Params) -> f64 {
		q := 0.5
		for _ in 0 ..< 100_000 {
			n := bsv_update_belief(q, same, p)
			if abs(n - q) < 1e-16 {return n}
			q = n
		}
		return q
	}
	return fixed(true, p), fixed(false, p)
}

// E[P_{t+1} - P_t | y_t = +y, q_t = q] - E[.. | y_t = -y, q_t = q] per unit y (BSV's
// f(q) / y): 2 (p2 q - p1) + p2 (rise(q) + fall(q)), rise the increase of q after a
// reversal, fall its decrease after a continuation
bsv_reaction_gap :: proc(q, p1, p2: f64, p := BSV_1998) -> f64 {
	rise := bsv_update_belief(q, false, p) - q
	fall := q - bsv_update_belief(q, true, p)
	return 2 * (p2 * q - p1) + p2 * (rise + fall)
}

BSV_Conditions :: struct {
	p1, p2:        f64,
	q_low, q_high: f64,
	k_low, k_high: f64, // overreaction if p1 > k_low p2, underreaction if p1 < k_high p2 (p2 >= 0)
	overreaction:  bool,
	underreaction: bool,
}

// Proposition 2. k_low = q_low + rise(q_low) / 2 (exact: overreaction iff
// f(q_low) < 0). k_high bounds E[q] + E[rise + fall] / 2 from below: g(q) = E[q_{t+1} |
// q_t = q] = q + (rise - fall) / 2 is bounded by lines of slope b = (g(q_high) -
// g(q_low)) / (q_high - q_low) through its lowest and highest points, so E[q] lies in
// [a_lo / (1 - b), a_hi / (1 - b)] (a_lo = min, a_hi = max of g(q) - b q, found on a
// grid of n_grid points); rise + fall, concave, lies above the chord c1 + c2 q.
bsv_sufficient_conditions :: proc(p := BSV_1998, n_grid := 20_000) -> BSV_Conditions {
	c: BSV_Conditions
	c.p1, c.p2 = bsv_price_coefficients(p)
	c.q_low, c.q_high = bsv_belief_bounds(p)
	ql, qh := c.q_low, c.q_high
	rise :: proc(q: f64, p: BSV_Params) -> f64 {return bsv_update_belief(q, false, p) - q}
	fall :: proc(q: f64, p: BSV_Params) -> f64 {return q - bsv_update_belief(q, true, p)}
	g :: proc(q: f64, p: BSV_Params) -> f64 {return q + 0.5 * (rise(q, p) - fall(q, p))}
	c.k_low = ql + 0.5 * rise(ql, p)
	b := (g(qh, p) - g(ql, p)) / (qh - ql)
	a_lo, a_hi := math.inf_f64(1), math.inf_f64(-1)
	for i in 0 ..= n_grid {
		q := ql + (qh - ql) * f64(i) / f64(n_grid)
		v := g(q, p) - b * q
		a_lo, a_hi = min(a_lo, v), max(a_hi, v)
	}
	qe_lo, qe_hi := a_lo / (1 - b), a_hi / (1 - b)
	c1 := (rise(ql, p) * qh - fall(qh, p) * ql) / (qh - ql)
	c2 := (fall(qh, p) - rise(ql, p)) / (qh - ql)
	q_star := qe_hi if c2 < 0 else qe_lo
	c.k_high = qe_lo + 0.5 * (c1 + c2 * q_star)
	c.overreaction = c.p2 >= 0 && c.p1 > c.k_low * c.p2
	c.underreaction = c.p2 >= 0 && c.p1 < c.k_high * c.p2
	return c
}
