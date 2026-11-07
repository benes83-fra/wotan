package behavioral_finance

import fin "../finance"
import l "../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Portfolio choice under cumulative prospect theory
// ============================================================================
// The portfolio return R_t = w'r_t over T equally likely scenarios (e.g. the
// returns of past evaluation periods, Benartzi & Thaler 1995) is valued by CPT
// relative to a reference return: outcomes x_t = R_t - reference sorted, losses
// weighted by w-(i/T) differences counted from the worst, gains by w+ differences
// counted from the best (Tversky & Kahneman 1992), valued by v (prospect.odin).
// gamma = gamma_minus = 1 gives the plain average of v(x_t).

// CPT value of equally likely outcomes (already relative to the reference)
cpt_value :: proc(outcomes: []f64, params: Prospect_Params) -> f64 {
	pi := make([]f64, len(outcomes), context.temp_allocator)
	defer delete(pi, context.temp_allocator)
	_cpt_decision_weights(outcomes, params, pi)
	V := 0.0
	for x, t in outcomes {V += pi[t] * prospect_value(x, params)}
	return V
}

// Decision weight of each outcome (in the outcomes' order)
@(private = "file")
_cpt_decision_weights :: proc(outcomes: []f64, params: Prospect_Params, pi: []f64) {
	T := len(outcomes)
	order := make([]int, T, context.temp_allocator)
	defer delete(order, context.temp_allocator)
	for i in 0 ..< T {order[i] = i}
	context.user_ptr = raw_data(outcomes)
	slice.sort_by(order, proc(a, b: int) -> bool {
		x := ([^]f64)(context.user_ptr)
		return x[a] < x[b] || (x[a] == x[b] && a < b)
	})
	gp, gm := params.gamma, prospect_gamma_minus(params)
	nT := f64(T)
	for k, rank in order {
		if outcomes[k] < 0 {
			// rank-th worst: P(at least as bad) goes from rank/T to (rank+1)/T
			pi[k] =
				prospect_weight(f64(rank + 1) / nT, gm, params.weighting) -
				prospect_weight(f64(rank) / nT, gm, params.weighting)
		} else {
			up := T - rank // P(at least as good) = up / T
			pi[k] =
				prospect_weight(f64(up) / nT, gp, params.weighting) -
				prospect_weight(f64(up - 1) / nT, gp, params.weighting)
		}
	}
}

ProspectMetrics :: struct {
	cpt_value:        f64, // CPT value of the period returns relative to the reference
	expected_return:  f64,
	volatility:       f64, // population standard deviation
	loss_probability: f64, // share of periods below the reference
}

// returns_data: rows = periods, cols = assets
prospect_metrics :: proc(
	weights: []f64,
	returns_data: ^l.Matrix(f64),
	params: Prospect_Params,
	reference := 0.0,
) -> ProspectMetrics {
	T, N := returns_data.rows, returns_data.cols
	x := make([]f64, T, context.temp_allocator)
	defer delete(x, context.temp_allocator)
	m := ProspectMetrics{}
	s1, s2 := 0.0, 0.0
	for t in 0 ..< T {
		R := l.dot_simd(weights, returns_data.data[t * N:][:N])
		x[t] = R - reference
		s1 += R
		s2 += R * R
		if x[t] < 0 {m.loss_probability += 1 / f64(T)}
	}
	m.expected_return = s1 / f64(T)
	m.volatility = math.sqrt(max(0, s2 / f64(T) - m.expected_return * m.expected_return))
	m.cpt_value = cpt_value(x, params)
	return m
}

@(private = "file")
_CPT_Problem :: struct {
	R:         ^l.Matrix(f64),
	params:    Prospect_Params,
	reference: f64,
	scale:     f64,
}

@(private = "file")
_cpt_neg_value :: proc(w: []f64, user: rawptr) -> f64 {
	p := (^_CPT_Problem)(user)
	T, N := p.R.rows, p.R.cols
	x := make([]f64, T, context.temp_allocator)
	defer delete(x, context.temp_allocator)
	for t in 0 ..< T {x[t] = l.dot_simd(w, p.R.data[t * N:][:N]) - p.reference}
	return -cpt_value(x, p.params) / p.scale
}

// Gradient with the ranking of the outcomes held fixed: -sum_t pi_t v'(x_t) r_t
// (v' at |x| < 1e-12 evaluated at 1e-12: v' is unbounded at 0 for alpha, beta < 1)
@(private = "file")
_cpt_neg_grad :: proc(w, g: []f64, user: rawptr) {
	p := (^_CPT_Problem)(user)
	T, N := p.R.rows, p.R.cols
	x := make([]f64, T, context.temp_allocator)
	pi := make([]f64, T, context.temp_allocator)
	defer delete(x, context.temp_allocator)
	defer delete(pi, context.temp_allocator)
	for t in 0 ..< T {x[t] = l.dot_simd(w, p.R.data[t * N:][:N]) - p.reference}
	_cpt_decision_weights(x, p.params, pi)
	for &v in g {v = 0}
	a := p.params
	for t in 0 ..< T {
		d: f64
		if x[t] >= 0 {
			d = a.alpha * math.pow(max(x[t], 1e-12), a.alpha - 1)
		} else {
			d = a.lambda * a.beta * math.pow(max(-x[t], 1e-12), a.beta - 1)
		}
		c := pi[t] * d / p.scale
		row := p.R.data[t * N:][:N]
		for i in 0 ..< N {g[i] -= c * row[i]}
	}
}

// The portfolio maximizing the CPT value of the period returns relative to the
// reference over the feasible set (default long only, fully invested). The objective
// is not concave (v is convex over losses, the weights rank-dependent): spectral
// projected gradient (finance.constrained_minimize) from every vertex, the centre and
// every edge midpoint, keeping the best.
prospect_portfolio_optimize :: proc(
	returns_data: ^l.Matrix(f64),
	params: Prospect_Params,
	reference := 0.0,
	constraints := fin.PortfolioConstraints{no_short_selling = true},
	allocator := context.allocator,
) -> []f64 {
	N := returns_data.cols
	prob := _CPT_Problem {
		R         = returns_data,
		params    = params,
		reference = reference,
		scale     = 1,
	}
	// scale the objective to O(1): the mean |outcome| of equal weights, through v
	ew := make([]f64, N, context.temp_allocator)
	defer delete(ew, context.temp_allocator)
	for &v in ew {v = 1 / f64(N)}
	s := 0.0
	for t in 0 ..< returns_data.rows {
		s +=
			abs(l.dot_simd(ew, returns_data.data[t * N:][:N]) - reference) / f64(returns_data.rows)
	}
	if s > 0 {prob.scale = math.pow(s, params.alpha)}

	starts := make([dynamic][]f64, context.temp_allocator)
	defer {
		for st in starts {delete(st, context.temp_allocator)}
		delete(starts)
	}
	append(&starts, slice.clone(ew, context.temp_allocator))
	for i in 0 ..< N {
		st := make([]f64, N, context.temp_allocator)
		st[i] = 1
		append(&starts, st)
		for j in i + 1 ..< N {
			e := make([]f64, N, context.temp_allocator)
			e[i], e[j] = 0.5, 0.5
			append(&starts, e)
		}
	}
	best: []f64
	best_f := math.INF_F64
	for st in starts {
		w := fin.constrained_minimize(
			N,
			_cpt_neg_value,
			_cpt_neg_grad,
			&prob,
			constraints,
			allocator,
			st,
		)
		f := _cpt_neg_value(w, &prob)
		if f < best_f {
			if best != nil {delete(best, allocator)}
			best, best_f = w, f
		} else {
			delete(w, allocator)
		}
	}
	return best
}

// ============================================================================
// Behavioral portfolio theory with mental accounts
// (Shefrin & Statman 2000; Das, Markowitz, Scheid & Statman 2010)
// ============================================================================
// Each mental account (retirement, education, bequest, ...) has a threshold
// return H and a maximum probability alpha of falling short of it, and maximizes
// its expected return subject to P(r < H) <= alpha. With normal returns the
// constraint is  w'mu + Phi^-1(alpha) sqrt(w'Sigma w) >= H, and the optimum is the
// mean-variance portfolio max w'mu - (gamma/2) w'Sigma w (fully invested) at the
// implied risk aversion gamma that makes the constraint bind (DMSS eqs. 3 - 8): the
// smallest gamma with Q(gamma) = w(gamma)'mu + Phi^-1(alpha) sigma(w(gamma)) >= H.
// Q is unimodal in gamma (it is concave along the frontier); H above its maximum is
// infeasible (eqs. 9 - 19). Long only (DMSS section V): w(gamma) from the
// constrained problem. The aggregate portfolio weights the accounts by wealth share.

Mental_Account :: struct {
	threshold:   f64, // H: the return the account must reach
	probability: f64, // alpha: the maximum probability of falling short of H
	share:       f64, // share of wealth in the account
	universe:    []bool, // assets the account may hold (nil = all)
}

Mental_Account_Status :: enum {
	Optimal, // the shortfall constraint binds at the implied gamma
	Max_Return, // long only: even the highest-return portfolio meets the constraint
	Unbounded, // short sales: the expected return can grow without limit
	Infeasible, // no portfolio reaches H with probability 1 - alpha
}

Mental_Account_Portfolio :: struct {
	weights:         []f64, // over all assets
	gamma:           f64, // implied risk aversion
	expected_return: f64,
	volatility:      f64,
	status:          Mental_Account_Status,
}

BPT_Result :: struct {
	aggregate: []f64, // sum of share * account weights
	accounts:  []Mental_Account_Portfolio,
	allocator: mem.Allocator,
}

bpt_result_free :: proc(r: ^BPT_Result) {
	for a in r.accounts {delete(a.weights, r.allocator)}
	delete(r.accounts, r.allocator)
	delete(r.aggregate, r.allocator)
}

// P(r < H) for normal returns with the portfolio's mean and volatility
shortfall_probability :: proc(weights, mu: []f64, cov: ^l.Matrix(f64), threshold: f64) -> f64 {
	m := l.dot_simd(weights, mu)
	s := math.sqrt(fin.portfolio_variance(weights, cov))
	return 0.5 * math.erfc(-(threshold - m) / (s * math.SQRT_TWO))
}

@(private = "file")
_MA_Problem :: struct {
	mu:        []f64,
	cov:       l.Matrix(f64),
	z:         f64,
	long_only: bool,
}

@(private = "file")
_ma_weights :: proc(p: ^_MA_Problem, gamma: f64, alloc: mem.Allocator) -> []f64 {
	if p.long_only {
		return fin.constrained_risk_aversion_portfolio(
			p.mu,
			&p.cov,
			gamma,
			{no_short_selling = true},
			alloc,
		)
	}
	return fin.risk_aversion_portfolio(p.mu, &p.cov, gamma, alloc)
}

@(private = "file")
_ma_q :: proc(p: ^_MA_Problem, log_gamma: f64) -> f64 {
	w := _ma_weights(p, math.exp(log_gamma), context.temp_allocator)
	defer delete(w, context.temp_allocator)
	return l.dot_simd(w, p.mu) + p.z * math.sqrt(fin.portfolio_variance(w, &p.cov))
}

// One mental account: the weights over all assets (zero outside its universe)
mental_account_portfolio :: proc(
	mu: []f64,
	cov: ^l.Matrix(f64),
	account: Mental_Account,
	long_only := false,
	allocator := context.allocator,
) -> Mental_Account_Portfolio {
	N := len(mu)
	assert(cov.rows == N && cov.cols == N, "mental_account_portfolio: dimension mismatch")
	assert(
		account.probability > 0 && account.probability < 1,
		"mental_account_portfolio: probability in (0, 1)",
	)
	assert(
		account.universe == nil || len(account.universe) == N,
		"mental_account_portfolio: one flag per asset",
	)
	idx := make([dynamic]int, context.temp_allocator)
	defer delete(idx)
	for i in 0 ..< N {if account.universe == nil || account.universe[i] {append(&idx, i)}}
	n := len(idx)
	assert(n > 0, "mental_account_portfolio: empty universe")
	p := _MA_Problem {
		mu        = make([]f64, n, context.temp_allocator),
		cov       = l.matrix_new(f64, n, n, context.temp_allocator),
		z         = fin.norm_inv(account.probability),
		long_only = long_only,
	}
	defer {delete(p.mu, context.temp_allocator); l.matrix_free(&p.cov)}
	for a, i in idx {
		p.mu[i] = mu[a]
		for b, j in idx {p.cov.data[i * n + j] = cov.data[a * N + b]}
	}

	LO, HI :: -14.0, 14.0 // ln gamma range
	// the maximum of Q over gamma (golden section on ln gamma)
	INV_PHI :: 0.6180339887498949
	a, b := LO, HI
	c, d := b - INV_PHI * (b - a), a + INV_PHI * (b - a)
	qc, qd := _ma_q(&p, c), _ma_q(&p, d)
	for b - a > 1e-10 {
		if qc >= qd {
			b, d, qd = d, c, qc
			c = b - INV_PHI * (b - a)
			qc = _ma_q(&p, c)
		} else {
			a, c, qc = c, d, qd
			d = a + INV_PHI * (b - a)
			qd = _ma_q(&p, d)
		}
	}
	x_star := 0.5 * (a + b)
	H := account.threshold
	res := Mental_Account_Portfolio{}
	x: f64
	if _ma_q(&p, x_star) < H {
		res.status, x = .Infeasible, x_star // the portfolio closest to feasible
	} else if _ma_q(&p, LO) >= H {
		res.status = .Max_Return if long_only else .Unbounded
		x = LO
	} else {
		// the root of Q - H between LO and x_star (Q rises on that side)
		lo, hi := LO, x_star
		for _ in 0 ..< 200 {
			mid := 0.5 * (lo + hi)
			if _ma_q(&p, mid) < H {lo = mid} else {hi = mid}
			if hi - lo < 1e-14 {break}
		}
		res.status, x = .Optimal, hi
	}
	res.gamma = math.exp(x)
	sub := _ma_weights(&p, res.gamma, context.temp_allocator)
	defer delete(sub, context.temp_allocator)
	res.weights = make([]f64, N, allocator)
	for a, i in idx {res.weights[a] = sub[i]}
	res.expected_return = l.dot_simd(res.weights, mu)
	res.volatility = math.sqrt(fin.portfolio_variance(res.weights, cov))
	return res
}

// All mental accounts and the aggregate portfolio (shares must sum to 1). mu and cov
// are per evaluation period (the thresholds are returns over the same period).
bpt_optimize :: proc(
	mu: []f64,
	cov: ^l.Matrix(f64),
	accounts: []Mental_Account,
	long_only := false,
	allocator := context.allocator,
) -> BPT_Result {
	total := 0.0
	for a in accounts {total += a.share}
	assert(abs(total - 1) < 1e-9, "bpt_optimize: account shares must sum to 1")
	r := BPT_Result {
		aggregate = make([]f64, len(mu), allocator),
		accounts  = make([]Mental_Account_Portfolio, len(accounts), allocator),
		allocator = allocator,
	}
	for acc, k in accounts {
		r.accounts[k] = mental_account_portfolio(mu, cov, acc, long_only, allocator)
		for v, i in r.accounts[k].weights {r.aggregate[i] += acc.share * v}
	}
	return r
}
