package ml_finance

import "core:math"

// ============================================================================
// TRANSACTION COSTS FOR BACKTESTS
// ============================================================================
// A trade of notional q (currency) in an asset with daily return volatility sigma and
// average daily traded value adv (currency) costs
//   cost(q) = |q| (commission + half_spread + impact sigma (|q| / adv)^impact_exponent)
// impact_exponent 0.5 is the square-root law of market impact (Y sigma sqrt(Q / V) with
// Y of order 1: Torre 1997; Almgren, Thum, Hauptmann & Li 2005; Tóth et al. 2011). With
// impact_exponent 1 the impact is linear and equals the Almgren-Chriss temporary impact
// with eta = impact sigma S^2 / adv (almgren_chriss_eta). The impact term is skipped when
// sigma or adv is not positive (liquidity unknown). Commission and half spread are paid
// on every trade (each side of a round trip).

CostModel :: struct {
	commission:      f64, // fraction of the traded notional
	half_spread:     f64, // half the bid-ask spread, fraction of the traded notional
	impact:          f64, // Y
	impact_exponent: f64, // 0.5: square-root law; 1: linear (Almgren-Chriss)
}

// Assumption-level defaults for liquid US ETFs and large caps: no commission, 1 bp half
// spread, square-root impact with Y = 1
DEFAULT_COST_MODEL :: CostModel {
	commission      = 0,
	half_spread     = 0.0001,
	impact          = 1,
	impact_exponent = 0.5,
}

// Cost per unit of traded notional
trade_cost_rate :: proc(model: CostModel, notional, sigma, adv: f64) -> f64 {
	q := abs(notional)
	rate := model.commission + model.half_spread
	if q > 0 && sigma > 0 && adv > 0 && model.impact != 0 {
		rate += model.impact * sigma * math.pow(q / adv, model.impact_exponent)
	}
	return rate
}

// Cost of trading the notional (currency)
trade_cost :: proc(model: CostModel, notional, sigma, adv: f64) -> f64 {
	return abs(notional) * trade_cost_rate(model, notional, sigma, adv)
}

// Moving the weights from `before` to `after` at the given capital, as a fraction of the
// capital: sum_i |dw_i| rate_i(|dw_i| capital). sigma / adv: per asset, nil for no impact.
rebalance_cost :: proc(
	model: CostModel,
	before, after: []f64,
	capital: f64,
	sigma: []f64 = nil,
	adv: []f64 = nil,
) -> f64 {
	assert(len(before) == len(after), "rebalance_cost: weights differ in length")
	assert(sigma == nil || len(sigma) == len(after), "rebalance_cost: one sigma per asset")
	assert(adv == nil || len(adv) == len(after), "rebalance_cost: one adv per asset")
	c := 0.0
	for a, i in after {
		dw := abs(a - before[i])
		if dw == 0 {continue}
		s := sigma[i] if sigma != nil else 0
		v := adv[i] if adv != nil else 0
		c += dw * trade_cost_rate(model, dw * capital, s, v)
	}
	return c
}

// Per-period costs of a position series, as fractions of the capital (positions in
// units of the capital, 1 = fully long): cost[t] = |dp_t| rate(|dp_t| capital) with
// dp_t = p_t - p_{t-1} and p_{-1} = initial. sigma / adv: per period (the estimates known
// when trading at t), nil for no impact.
position_costs :: proc(
	model: CostModel,
	positions: []f64,
	capital: f64,
	sigma: []f64 = nil,
	adv: []f64 = nil,
	initial := 0.0,
	allocator := context.allocator,
) -> []f64 {
	assert(sigma == nil || len(sigma) == len(positions), "position_costs: one sigma per period")
	assert(adv == nil || len(adv) == len(positions), "position_costs: one adv per period")
	out := make([]f64, len(positions), allocator)
	prev := initial
	for p, t in positions {
		dp := abs(p - prev)
		prev = p
		if dp == 0 {continue}
		s := sigma[t] if sigma != nil else 0
		v := adv[t] if adv != nil else 0
		out[t] = dp * trade_cost_rate(model, dp * capital, s, v)
	}
	return out
}

// The Almgren-Chriss eta (time in days) whose linear temporary impact equals this
// model's impact term with impact_exponent 1: n shares traded within tau days receive
// (eta / tau) n less per share, and S impact sigma n S / (adv tau) = (eta / tau) n gives
// eta = impact sigma S^2 / adv. (For other exponents there is no exact equivalent.)
almgren_chriss_eta :: proc(model: CostModel, price, sigma, adv: f64) -> f64 {
	return model.impact * sigma * price * price / adv
}

// Liquidity known at the close of day t, from the `window` days ending at t:
//   sigma[t]  sample std of the simple returns close[t-window+1 ..= t] over the previous close
//   adv[t]    mean of close * volume over days t-window+1 ..= t
// NaN where the window is not full (sigma needs window + 1 closes). No look-ahead.
liquidity_estimates :: proc(
	close, volume: []f64,
	window := 20,
	allocator := context.allocator,
) -> (
	sigma, adv: []f64,
) {
	assert(len(close) == len(volume), "liquidity_estimates: close and volume differ in length")
	assert(window >= 2, "liquidity_estimates: window >= 2")
	n := len(close)
	sigma = make([]f64, n, allocator)
	adv = make([]f64, n, allocator)
	for t in 0 ..< n {
		sigma[t], adv[t] = math.nan_f64(), math.nan_f64()
		if t >= window - 1 {
			m := 0.0
			for k in t - window + 1 ..= t {m += close[k] * volume[k] / f64(window)}
			adv[t] = m
		}
		if t >= window {
			mean := 0.0
			for k in t - window + 1 ..= t {mean += (close[k] / close[k - 1] - 1) / f64(window)}
			ss := 0.0
			for k in t - window + 1 ..= t {
				d := close[k] / close[k - 1] - 1 - mean
				ss += d * d
			}
			sigma[t] = math.sqrt(ss / f64(window - 1))
		}
	}
	return
}

// Net returns: gross[t] - cost[t] (into a new slice)
net_of_costs :: proc(gross, cost: []f64, allocator := context.allocator) -> []f64 {
	assert(len(gross) == len(cost), "net_of_costs: gross and cost differ in length")
	out := make([]f64, len(gross), allocator)
	for g, t in gross {out[t] = g - cost[t]}
	return out
}
