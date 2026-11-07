package behavioral_finance

import fin "../finance"
import opt "../optimize"
import "core:math"
import "core:mem"

// ============================================================================
// Option prices under cumulative prospect theory
// (Versluis, Lehnert & Wolff 2010; Nardon & Pianca 2012)
// ============================================================================
// The price of an option is the premium c at which a CPT investor is indifferent
// between taking the position and not: the prospect value V of the position is
// zero. Outcomes are valued at maturity T relative to zero, the premium carried at
// the risk-free rate (C = c e^{rT}), the terminal price S_T drawn from a physical
// distribution (log-normal with drift mu, or a distribution given on a grid).
//
//   value function       v(x) = x^alpha (x >= 0), -lambda (-x)^beta (x < 0)
//   decision weights     gains: dw+(P[outcome at least as good]), losses:
//                        dw-(P[outcome at least as bad]) (rank-dependent, Tversky &
//                        Kahneman 1992); w = TK92  p^g / (p^g + (1-p)^g)^(1/g)  or
//                        Prelec  exp(-(-ln p)^g), g = gamma (gains) / gamma_minus (losses)
//   writer               receives C, pays the payoff;  holder: pays C, receives it
//   segregated frame     V = v(+-C) + V(-+payoff): premium and payoff valued apart,
//                        closed form, e.g. writer call c = e^{-rT} (lambda J)^(1/alpha),
//                        J = int_K^inf Psi-(1 - F) f (S - K)^beta dS (N&P eq. 16)
//   aggregated frame     V = V(net outcome +-C -+ payoff) = 0, solved for C (eq. 17)
// alpha = beta = lambda = gamma = 1 and mu = r give the Black-Scholes price.
//
// Numerics: the terminal-price axis is cut into cells (the strike a cell edge);
// a cell's decision weight is the exact difference of w at its edges, its outcome
// the payoff at its midpoint (second-order accurate); the cells below / above the
// grid's ends carry the tail probability (cdf forced to 0 and 1 at the ends).

CPT_Weighting :: enum {
	Tversky_Kahneman,
	Prelec,
}

CPT_Side :: enum {
	Writer, // receives the premium (Versluis et al., the market price)
	Holder,
}

CPT_Frame :: enum {
	Aggregated, // premium and payoff netted into one outcome
	Segregated, // valued as two separate prospects
}

Prospect_Params :: struct {
	alpha:       f64, // gain curvature (TK92: 0.88)
	beta:        f64, // loss curvature (TK92: 0.88)
	lambda:      f64, // loss aversion (TK92: 2.25)
	gamma:       f64, // probability weighting of gains, gamma+ (TK92: 0.61)
	gamma_minus: f64, // ... of losses, gamma- (TK92: 0.69); 0 = gamma
	weighting:   CPT_Weighting,
}

// Tversky & Kahneman (1992) estimates
TK92_PARAMS :: Prospect_Params {
	alpha       = 0.88,
	beta        = 0.88,
	lambda      = 2.25,
	gamma       = 0.61,
	gamma_minus = 0.69,
}

// Risk and probability neutral: the Black-Scholes investor
RATIONAL_PARAMS :: Prospect_Params {
	alpha  = 1,
	beta   = 1,
	lambda = 1,
	gamma  = 1,
}

prospect_gamma_minus :: proc(p: Prospect_Params) -> f64 {
	return p.gamma_minus if p.gamma_minus > 0 else p.gamma
}

// The value function v(x)
prospect_value :: proc(x: f64, p: Prospect_Params) -> f64 {
	return math.pow(x, p.alpha) if x >= 0 else -p.lambda * math.pow(-x, p.beta)
}

// The probability weighting function w(prob). TK92 is increasing for gamma > ~0.28.
prospect_weight :: proc(prob, gamma: f64, kind: CPT_Weighting = .Tversky_Kahneman) -> f64 {
	if prob <= 0 {return 0}
	if prob >= 1 {return 1}
	switch kind {
	case .Tversky_Kahneman:
		pg := math.pow(prob, gamma)
		return pg / math.pow(pg + math.pow(1 - prob, gamma), 1 / gamma)
	case .Prelec:
		return math.exp(-math.pow(-math.ln(prob), gamma))
	}
	return prob
}

// The distribution of S_T on a grid: cell i = [nodes[i], nodes[i+1]] has probability
// cdf[i+1] - cdf[i]; cdf[0] = 0, cdf[n-1] = 1 (the tails are folded into the end cells)
Prospect_Precomputed :: struct {
	nodes:     []f64,
	cdf:       []f64,
	allocator: mem.Allocator,
}

prospect_precomputed_free :: proc(pre: ^Prospect_Precomputed) {
	delete(pre.nodes, pre.allocator)
	delete(pre.cdf, pre.allocator)
	pre^ = {}
}

// Log-normal S_T (drift mu, volatility sigma): a uniform grid in the standard normal
// z of ln S_T over +-10 standard deviations, n_points cells, the strike K a node
prospect_lognormal_grid :: proc(
	S, K, T, mu, sigma: f64,
	n_points := 2000,
	allocator := context.allocator,
) -> Prospect_Precomputed {
	assert(
		S > 0 && K > 0 && T > 0 && sigma > 0 && n_points >= 2,
		"prospect_lognormal_grid: bad inputs",
	)
	m := math.ln(S) + (mu - 0.5 * sigma * sigma) * T
	s := sigma * math.sqrt(T)
	ZMAX :: 10.0
	h := 2 * ZMAX / f64(n_points)
	zK := (math.ln(K) - m) / s
	// nodes zK + j h within [-ZMAX, ZMAX] (zK itself when it lies inside)
	j0 := int(math.ceil((-ZMAX - zK) / h))
	j1 := int(math.floor((ZMAX - zK) / h))
	n := j1 - j0 + 1
	pre := Prospect_Precomputed {
		nodes     = make([]f64, n, allocator),
		cdf       = make([]f64, n, allocator),
		allocator = allocator,
	}
	for j in 0 ..< n {
		z := zK + f64(j0 + j) * h
		pre.nodes[j] = math.exp(m + s * z)
		pre.cdf[j] = 0.5 * math.erfc(-z / math.SQRT_TWO)
	}
	pre.cdf[0], pre.cdf[n - 1] = 0, 1
	return pre
}

// The cells of one option position: payoffs and decision weights
@(private = "file")
_CPT_Cells :: struct {
	pay:       []f64, // payoff at the midpoints of the cells where it is positive
	wgain:     []f64, // their decision weights as gains
	wloss:     []f64, // ... as losses
	zero_gain: f64, // total weight of the zero-payoff cells (one outcome) as a gain
	zero_loss: f64, // ... as a loss
	sgn:       f64, // +1 holder (receives the payoff), -1 writer
	params:    Prospect_Params,
}

@(private = "file")
_cpt_cells :: proc(
	pre: ^Prospect_Precomputed,
	K: f64,
	params: Prospect_Params,
	opt_type: fin.OptionType,
	side: CPT_Side,
	alloc: mem.Allocator,
) -> _CPT_Cells {
	n := len(pre.nodes)
	call := opt_type == .Call
	// the position's outcome rises with S_T (holder of a call, writer of a put): gains
	// ranked from high S_T (decumulative 1 - F), losses from low S_T (F); else reversed
	inc := call == (side == .Holder)
	gp, gm := params.gamma, prospect_gamma_minus(params)
	Wg := make([]f64, n, alloc)
	Wl := make([]f64, n, alloc)
	defer delete(Wg, alloc)
	defer delete(Wl, alloc)
	for F, j in pre.cdf {
		Wg[j] = prospect_weight(1 - F if inc else F, gp, params.weighting)
		Wl[j] = prospect_weight(F if inc else 1 - F, gm, params.weighting)
	}
	c := _CPT_Cells {
		pay    = make([]f64, n, alloc),
		wgain  = make([]f64, n, alloc),
		wloss  = make([]f64, n, alloc),
		sgn    = 1 if side == .Holder else -1,
		params = params,
	}
	k := 0
	for i in 0 ..< n - 1 {
		wg := Wg[i] - Wg[i + 1] if inc else Wg[i + 1] - Wg[i]
		wl := Wl[i + 1] - Wl[i] if inc else Wl[i] - Wl[i + 1]
		mid := 0.5 * (pre.nodes[i] + pre.nodes[i + 1])
		pay := max(mid - K, 0) if call else max(K - mid, 0)
		if pay > 0 {
			c.pay[k], c.wgain[k], c.wloss[k] = pay, wg, wl
			k += 1
		} else {
			c.zero_gain += wg
			c.zero_loss += wl
		}
	}
	c.pay, c.wgain, c.wloss = c.pay[:k], c.wgain[:k], c.wloss[:k]
	return c
}

@(private = "file")
_cpt_cells_free :: proc(c: ^_CPT_Cells, alloc: mem.Allocator) {
	delete(c.pay, alloc)
	delete(c.wgain, alloc)
	delete(c.wloss, alloc)
}

// Prospect value of the payoff side plus the constant shift (the carried premium,
// signed) netted into every outcome
@(private = "file")
_cpt_value :: proc(c: ^_CPT_Cells, shift: f64) -> f64 {
	p := c.params
	V := 0.0
	if shift != 0 {
		V = prospect_value(shift, p) * (c.zero_gain if shift > 0 else c.zero_loss)
	}
	for pay, i in c.pay {
		o := c.sgn * pay + shift
		if o > 0 {
			V += math.pow(o, p.alpha) * c.wgain[i]
		} else if o < 0 {
			V -= p.lambda * math.pow(-o, p.beta) * c.wloss[i]
		}
	}
	return V
}

// Aggregated frame: the carried premium C >= 0 with V(+-C) = 0. The writer's V rises
// with C, the holder's falls; bracket from the expected payoff, then Illinois
// (modified regula falsi)
@(private = "file")
_cpt_aggregated_premium :: proc(c: ^_CPT_Cells) -> f64 {
	writer := c.sgn < 0
	g :: proc(c: ^_CPT_Cells, C: f64, writer: bool) -> f64 {
		return _cpt_value(c, C) if writer else -_cpt_value(c, -C)
	}
	guess := 0.0
	for pay, i in c.pay {guess += pay * (c.wgain[i] + c.wloss[i]) * 0.5}
	if len(c.pay) == 0 {return 0}
	guess = max(guess, 1e-12)
	lo, hi := 0.0, guess
	glo := g(c, lo, writer)
	if glo >= 0 {return 0}
	ghi := g(c, hi, writer)
	for ghi < 0 {
		lo, glo = hi, ghi
		hi *= 2
		ghi = g(c, hi, writer)
	}
	side := 0
	for _ in 0 ..< 200 {
		x := (lo * ghi - hi * glo) / (ghi - glo)
		gx := g(c, x, writer)
		if gx == 0 || hi - lo <= 1e-14 * hi {return x}
		if gx < 0 {
			lo, glo = x, gx
			if side == -1 {ghi *= 0.5}
			side = -1
		} else {
			hi, ghi = x, gx
			if side == 1 {glo *= 0.5}
			side = 1
		}
	}
	return 0.5 * (lo + hi)
}

// The CPT option price from a precomputed distribution of S_T
prospect_price_precomputed :: proc(
	pre: ^Prospect_Precomputed,
	K, T, r: f64,
	params: Prospect_Params,
	opt_type: fin.OptionType,
	side: CPT_Side = .Writer,
	frame: CPT_Frame = .Aggregated,
	allocator := context.temp_allocator,
) -> f64 {
	c := _cpt_cells(pre, K, params, opt_type, side, allocator)
	defer _cpt_cells_free(&c, allocator)
	C: f64
	switch frame {
	case .Segregated:
		U := _cpt_value(&c, 0)
		if side == .Writer {
			// v+(C) + U = 0, U <= 0
			C = math.pow(max(-U, 0), 1 / params.alpha)
		} else {
			// v-(-C) + U = 0: lambda C^beta = U
			C = math.pow(max(U, 0) / params.lambda, 1 / params.beta)
		}
	case .Aggregated:
		C = _cpt_aggregated_premium(&c)
	}
	return C * math.exp(-r * T)
}

// The prospect value of a position taken at the given premium (zero at the CPT price)
prospect_option_value :: proc(
	pre: ^Prospect_Precomputed,
	K, T, r, premium: f64,
	params: Prospect_Params,
	opt_type: fin.OptionType,
	side: CPT_Side = .Writer,
	frame: CPT_Frame = .Aggregated,
	allocator := context.temp_allocator,
) -> f64 {
	c := _cpt_cells(pre, K, params, opt_type, side, allocator)
	defer _cpt_cells_free(&c, allocator)
	C := premium * math.exp(r * T) * (1 if side == .Writer else -1)
	if frame == .Aggregated {return _cpt_value(&c, C)}
	return prospect_value(C, params) + _cpt_value(&c, 0)
}

// The CPT price of a European option on a log-normal underlying (physical drift mu,
// volatility sigma), discounted at r
prospect_theory_price :: proc(
	S, K, T, r, mu, sigma: f64,
	params: Prospect_Params,
	opt_type: fin.OptionType,
	side: CPT_Side = .Writer,
	frame: CPT_Frame = .Aggregated,
	n_points := 2000,
) -> f64 {
	if T <= 0 {
		return max(S - K, 0) if opt_type == .Call else max(K - S, 0)
	}
	pre := prospect_lognormal_grid(S, K, T, mu, sigma, n_points, context.temp_allocator)
	defer prospect_precomputed_free(&pre)
	return prospect_price_precomputed(&pre, K, T, r, params, opt_type, side, frame)
}

// ============================================================================
// Calibration to an option chain
// ============================================================================

Prospect_Param :: enum {
	Alpha,
	Beta,
	Lambda,
	Gamma,
	Gamma_Minus,
}

Prospect_Calibration_Setup :: struct {
	side:     CPT_Side,
	frame:    CPT_Frame,
	init:     Prospect_Params, // start and the value of fixed parameters; zero = TK92
	fixed:    bit_set[Prospect_Param], // held at init
	max_iter: int, // 0 = 200
	n_points: int, // grid cells (log-normal); 0 = 1000
}

Prospect_CalibrationResult :: struct {
	params:     Prospect_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

Prospect_CalibrationContext :: struct {
	market_data: []fin.VolSurfacePoint,
	opt_types:   []fin.OptionType, // nil = calls
	S:           f64,
	r:           f64,
	setup:       Prospect_Calibration_Setup,
	free_index:  [Prospect_Param]int, // position in the optimizer's x, -1 = fixed
	precomputed: []Prospect_Precomputed, // one distribution per option
}

@(private = "file")
_ctx_params :: proc(ctx: ^Prospect_CalibrationContext, x: []f64) -> Prospect_Params {
	p := ctx.setup.init
	get :: proc(ctx: ^Prospect_CalibrationContext, x: []f64, k: Prospect_Param, v: f64) -> f64 {
		i := ctx.free_index[k]
		return x[i] if i >= 0 else v
	}
	p.alpha = get(ctx, x, .Alpha, p.alpha)
	p.beta = get(ctx, x, .Beta, p.beta)
	p.lambda = get(ctx, x, .Lambda, p.lambda)
	p.gamma = get(ctx, x, .Gamma, p.gamma)
	p.gamma_minus = get(ctx, x, .Gamma_Minus, prospect_gamma_minus(p))
	return p
}

// RMSE of the CPT prices against the market prices (market_price, or the
// Black-Scholes price at implied_vol when market_price <= 0)
prospect_objective_fn :: proc(x: []f64, user_data: rawptr) -> f64 {
	ctx := (^Prospect_CalibrationContext)(user_data)
	params := _ctx_params(ctx, x)
	sse := 0.0
	for point, i in ctx.market_data {
		ot := ctx.opt_types[i] if ctx.opt_types != nil else fin.OptionType.Call
		model := prospect_price_precomputed(
			&ctx.precomputed[i],
			point.strike,
			point.expiry,
			ctx.r,
			params,
			ot,
			ctx.setup.side,
			ctx.setup.frame,
			context.temp_allocator,
		)
		market := point.market_price
		if market <= 0 {
			market = fin.black_scholes_call(
				ctx.S,
				point.strike,
				point.expiry,
				ctx.r,
				point.implied_vol,
			)
			if ot == .Put {market += point.strike * math.exp(-ctx.r * point.expiry) - ctx.S}
		}
		sse += (model - market) * (model - market)
	}
	return math.sqrt(sse / f64(len(ctx.market_data)))
}

// Calibrates the CPT parameters to an option chain on a log-normal underlying with
// physical drift mu and volatility sigma (e.g. the realised volatility). opt_types:
// the type of each option (nil = all calls). Weakly identified: alpha, beta and
// lambda trade off against each other; fix some (setup.fixed) to read the rest.
calibrate_prospect_theory :: proc(
	market_data: []fin.VolSurfacePoint,
	S, r, mu, sigma: f64,
	setup: Prospect_Calibration_Setup = {},
	opt_types: []fin.OptionType = nil,
	allocator := context.allocator,
) -> Prospect_CalibrationResult {
	n_points := setup.n_points if setup.n_points > 0 else 1000
	pre := make([]Prospect_Precomputed, len(market_data), allocator)
	defer {
		for &p in pre {prospect_precomputed_free(&p)}
		delete(pre, allocator)
	}
	for point, i in market_data {
		pre[i] = prospect_lognormal_grid(
			S,
			point.strike,
			point.expiry,
			mu,
			sigma,
			n_points,
			allocator,
		)
	}
	return _calibrate_prospect(market_data, opt_types, S, r, setup, pre, allocator)
}

// Shared by the log-normal and the behavioral Heston calibrations: fits the free
// parameters with Adam (finite-difference gradients, probes in parallel)
_calibrate_prospect :: proc(
	market_data: []fin.VolSurfacePoint,
	opt_types: []fin.OptionType,
	S, r: f64,
	setup: Prospect_Calibration_Setup,
	pre: []Prospect_Precomputed,
	allocator: mem.Allocator,
) -> Prospect_CalibrationResult {
	assert(
		opt_types == nil || len(opt_types) == len(market_data),
		"calibrate_prospect: one type per option",
	)
	ctx := Prospect_CalibrationContext {
		market_data = market_data,
		opt_types   = opt_types,
		S           = S,
		r           = r,
		setup       = setup,
		precomputed = pre,
	}
	if ctx.setup.init.alpha == 0 {
		w := ctx.setup.init.weighting
		ctx.setup.init = TK92_PARAMS
		ctx.setup.init.weighting = w
	}
	init := ctx.setup.init
	all := [Prospect_Param]opt.Calibration_Param {
		.Alpha       = {"alpha", init.alpha, 0.2, 1.0, .Linear},
		.Beta        = {"beta", init.beta, 0.2, 1.0, .Linear},
		.Lambda      = {"lambda", init.lambda, 0.5, 5.0, .Linear},
		.Gamma       = {"gamma", init.gamma, 0.3, 1.2, .Linear},
		.Gamma_Minus = {"gamma_minus", prospect_gamma_minus(init), 0.3, 1.2, .Linear},
	}
	free := make([dynamic]opt.Calibration_Param, allocator)
	defer delete(free)
	for k in Prospect_Param {
		ctx.free_index[k] = -1
		if k in setup.fixed {continue}
		ctx.free_index[k] = len(free)
		append(&free, all[k])
	}
	if len(free) == 0 {
		return {
			params = _ctx_params(&ctx, nil),
			rmse = prospect_objective_fn(nil, &ctx),
			converged = true,
		}
	}
	config := opt.Calibration_Config {
		optimizer = {
			type = .Adam,
			learning_rate = 0.02,
			beta1 = 0.9,
			beta2 = 0.999,
			epsilon = 1e-8,
		},
		max_iter = setup.max_iter if setup.max_iter > 0 else 200,
		loss_threshold = 1e-6,
		fd_step = 1e-5,
		parallel = true,
	}
	res := opt.calibrate(free[:], prospect_objective_fn, &ctx, config, allocator)
	defer opt.calibration_result_free(&res)
	return {
		params = _ctx_params(&ctx, res.x),
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}

// ============================================================================
// A heuristic trading signal from calibrated parameters
// ============================================================================
// Not from the literature: hand-set thresholds on the calibrated loss-side
// probability weighting (gamma-, tail fear) and the implied-minus-realised
// volatility spread. Illustrative only.

BehavioralSignal :: struct {
	regime:     string,
	action:     string,
	conviction: f64, // 0 .. 1
	bvrp:       f64, // implied minus historical volatility
}

generate_behavioral_signal :: proc(
	market_params: Prospect_Params,
	hist_vol: f64,
	avg_implied_vol: f64,
) -> BehavioralSignal {
	sig: BehavioralSignal
	sig.bvrp = avg_implied_vol - hist_vol
	g := prospect_gamma_minus(market_params)
	switch {
	case g < 0.55:
		sig.regime = "Panic (Crashophobia)"
		if sig.bvrp > 0.05 {
			sig.action = "SHORT VOLATILITY (Sell OTM Puts / Short Strangle)"
			sig.conviction = min(1.0, sig.bvrp / 0.15)
		} else {
			sig.action = "Hold (Fear is high, but IV hasn't decoupled from HV yet)"
			sig.conviction = 0.2
		}
	case g > 0.75:
		sig.regime = "Complacency / Euphoria"
		if sig.bvrp < 0.02 {
			sig.action = "BUY TAIL PROTECTION (Buy cheap OTM Puts)"
			sig.conviction = min(1.0, (g - 0.75) / 0.25)
		} else {
			sig.action = "Hold (Market is rational and fairly priced)"
			sig.conviction = 0.1
		}
	case:
		sig.regime = "Standard Behavioral Bias"
		sig.action = "Delta Neutral / Market Making"
		sig.conviction = 0.3
	}
	return sig
}
