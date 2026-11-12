package ml_finance

import "core:math"
import "core:mem"

// ActionDef defines a discrete portfolio allocation
// weights: array of length n_assets. Sum must be <= 1.0.
// (Any remainder is automatically kept in Cash)
ActionDef :: struct {
	weights: []f64,
}

// Multi-asset rebalancing environment. prices / volumes are [day, asset] row-major,
// indicators [day, asset, indicator]. Step k observes the window ending at day k - 1,
// rebalances at day k's prices to the target weights of the chosen action (an asset is
// traded only if its change exceeds REBALANCE_BAND of the portfolio value; fee on the
// traded value), and marks the portfolio to day k. The first step trades on day `window`;
// the episode ends on the last day (or below 10% of the initial cash), where everything
// is sold with the fee.
// Reward (shaped): change of portfolio value from day k - 1 to day k / initial cash
// (market move of the held positions minus fees), - 0.5 x drawdown from the peak,
// - 0.5 x the mean absolute daily return over the window of every asset held, times its
// weight |position value| / portfolio value,
// + 1e-4 x the cash share of the initial cash.
MultiAssetEnv :: struct {
	env:             Environment,
	prices:          []f64, // Flattened: [day0_asset0, day0_asset1, day1_asset0...]
	volumes:         []f64,
	indicators:      []f64,
	n_assets:        int,
	n_indicators:    int,
	window:          int,
	cash:            f64,
	initial_cash:    f64,
	peak_cash:       f64, // peak portfolio value
	positions:       []f64, // Fractional shares per asset
	action_defs:     []ActionDef,
	transaction_fee: f64,
	allocator:       mem.Allocator,
}

REBALANCE_BAND :: 0.005 // trade an asset only if its value changes by > 0.5% of the portfolio

multi_asset_env_reset :: proc(env: ^Environment) -> Observation {
	t_env := cast(^MultiAssetEnv)env
	t_env.cash = t_env.initial_cash
	t_env.peak_cash = t_env.initial_cash
	for &p in t_env.positions {p = 0.0}
	t_env.env.current_step = t_env.window - 1 // the first step trades on day `window`
	t_env.env.done = false

	day := t_env.env.current_step
	obs := build_multi_asset_obs(
		t_env,
		day,
		t_env.prices[day * t_env.n_assets:][:t_env.n_assets],
		t_env.initial_cash,
	)
	return Observation{data = obs, shape = [4]int{1, len(obs), 1, 1}}
}

multi_asset_env_step :: proc(env: ^Environment, action: int) -> Step {
	t_env := cast(^MultiAssetEnv)env
	n := t_env.n_assets
	k := t_env.env.current_step
	if k >= len(t_env.prices) / n || k < 1 {
		obs_dim := t_env.env.obs_dim
		return Step {
			observation = Observation {
				data = make([]f64, obs_dim, t_env.allocator),
				shape = [4]int{1, obs_dim, 1, 1},
			},
			done = true,
			info = "Already done",
		}
	}
	prev_prices := t_env.prices[(k - 1) * n:][:n]
	curr_prices := t_env.prices[k * n:][:n]

	// value at the previous close (the reward measures the change from it)
	prev_value := t_env.cash
	for a in 0 ..< n {prev_value += t_env.positions[a] * prev_prices[a]}
	// value at today's prices before trading: the base of the target weights
	value_now := t_env.cash
	for a in 0 ..< n {value_now += t_env.positions[a] * curr_prices[a]}

	if action >= 0 && action < len(t_env.action_defs) {
		target_weights := t_env.action_defs[action].weights
		for a in 0 ..< n {
			delta_val := value_now * target_weights[a] - t_env.positions[a] * curr_prices[a]
			if math.abs(delta_val) > REBALANCE_BAND * math.abs(value_now) {
				fee := math.abs(delta_val) * t_env.transaction_fee
				t_env.positions[a] += delta_val / curr_prices[a]
				t_env.cash -= delta_val + fee
			}
		}
	}

	curr_value := t_env.cash
	for a in 0 ..< n {curr_value += t_env.positions[a] * curr_prices[a]}
	if curr_value < t_env.initial_cash * 0.10 {t_env.env.done = true} 	// bust
	if t_env.env.done {
		// last day (or bust): sell everything, the fee is part of the value change
		for a in 0 ..< n {
			t_env.cash += t_env.positions[a] * curr_prices[a] * (1.0 - t_env.transaction_fee)
			t_env.positions[a] = 0.0
		}
		curr_value = t_env.cash
	}

	t_env.peak_cash = max(t_env.peak_cash, curr_value)
	drawdown := (t_env.peak_cash - curr_value) / t_env.peak_cash if t_env.peak_cash > 0 else 0.0

	reward := (curr_value - prev_value) / t_env.initial_cash
	reward -= drawdown * 0.5
	// volatility penalty: mean absolute daily return over the window of each asset held,
	// weighted by the asset's share of the portfolio value (it used to be charged in full
	// for any holding, however small, so holding cash always earned the best reward)
	vol_penalty := 0.0
	start := max(0, k - t_env.window)
	for a in 0 ..< n {
		if t_env.positions[a] == 0.0 || k <= start || curr_value <= 0 {continue}
		avg_abs_ret := 0.0
		for d in start ..< k {
			p0 := t_env.prices[d * n + a]
			if p0 > 0 {avg_abs_ret += math.abs((t_env.prices[(d + 1) * n + a] - p0) / p0)}
		}
		weight := t_env.positions[a] * curr_prices[a] / curr_value
		vol_penalty += avg_abs_ret / f64(k - start) * 0.5 * math.abs(weight)
	}
	reward -= vol_penalty
	reward += (t_env.cash / t_env.initial_cash) * 0.0001 // cash bonus

	obs := build_multi_asset_obs(t_env, k, curr_prices, curr_value)
	return Step {
		observation = Observation{data = obs, shape = [4]int{1, len(obs), 1, 1}},
		reward = reward,
		done = t_env.env.done,
		info = "",
	}
}

build_multi_asset_obs :: proc(
	t_env: ^MultiAssetEnv,
	step_idx: int,
	curr_prices: []f64,
	total_value: f64,
) -> []f64 {
	total_value := total_value
	obs_dim := t_env.env.obs_dim
	obs := make([]f64, obs_dim, t_env.allocator)
	idx := 0

	start := step_idx - t_env.window + 1
	if start < 0 {
		start = 0
	}

	// 1. Historical window for each asset
	for a in 0 ..< t_env.n_assets {
		base_price := t_env.prices[start * t_env.n_assets + a]
		if base_price == 0 {
			base_price = 1.0
		}
		base_vol := t_env.volumes[start * t_env.n_assets + a]
		if base_vol == 0 {
			base_vol = 1.0
		}

		for w in 0 ..< t_env.window {
			w_idx := start + w
			if w_idx < len(t_env.prices) / t_env.n_assets {
				price_ret := (t_env.prices[w_idx * t_env.n_assets + a] / base_price) - 1.0
				obs[idx] = math.max(-0.1, math.min(0.1, price_ret))
				idx += 1
				obs[idx] = (t_env.volumes[w_idx * t_env.n_assets + a] / base_vol) - 1.0
				idx += 1

				for ind in 0 ..< t_env.n_indicators {
					ind_idx :=
						w_idx * t_env.n_assets * t_env.n_indicators + a * t_env.n_indicators + ind
					if ind_idx < len(t_env.indicators) {
						obs[idx] = t_env.indicators[ind_idx]
					}
					idx += 1
				}
			} else {
				for _ in 0 ..< (2 + t_env.n_indicators) {
					obs[idx] = 0.0
					idx += 1
				}
			}
		}
	}

	// 2. Current portfolio weights and cash share (value floored at 1 against division by 0)
	if total_value < 1.0 {
		total_value = 1.0
	}

	for a in 0 ..< t_env.n_assets {
		asset_val := t_env.positions[a] * curr_prices[a]
		weight := asset_val / total_value
		// Clamp weights to [-2.0, 2.0] to prevent NaN/Inf from breaking the NN
		obs[idx] = math.max(-2.0, math.min(2.0, weight))
		idx += 1
	}
	cash_ratio := t_env.cash / total_value
	obs[idx] = math.max(-2.0, math.min(2.0, cash_ratio))
	idx += 1

	return obs
}

new_multi_asset_env :: proc(
	prices: []f64,
	volumes: []f64,
	indicators: []f64,
	n_assets: int,
	n_indicators: int,
	window: int,
	action_defs: []ActionDef,
	transaction_fee: f64 = 0.001,
	alloc: mem.Allocator = context.allocator,
) -> ^MultiAssetEnv {
	assert(
		n_assets >= 1 &&
		len(prices) % n_assets == 0 &&
		len(prices) / n_assets > window &&
		window >= 1,
		"new_multi_asset_env: need more days than the window",
	)
	assert(len(volumes) == len(prices), "new_multi_asset_env: volumes and prices differ in length")
	for ad in action_defs {assert(len(ad.weights) == n_assets, "new_multi_asset_env: every action needs n_assets weights")}
	env := new(MultiAssetEnv, alloc)
	env.prices = prices
	env.volumes = volumes
	env.indicators = indicators
	env.n_assets = n_assets
	env.n_indicators = n_indicators
	env.window = window
	env.action_defs = action_defs
	env.transaction_fee = transaction_fee
	env.positions = make([]f64, n_assets, alloc)

	env.env.action_space = len(action_defs)
	// Obs Dim = (window * (price + vol + indicators) * n_assets) + n_assets (weights) + 1 (cash)
	env.env.obs_dim = window * (2 + n_indicators) * n_assets + n_assets + 1
	env.env.max_steps = len(prices) / n_assets - 1 // the last step trades on the last day
	env.initial_cash = 100000.0
	env.env.reset_fn = multi_asset_env_reset
	env.env.step_fn = multi_asset_env_step
	env.allocator = alloc
	return env
}

multi_asset_env_free :: proc(env: ^MultiAssetEnv) {
	delete(env.positions, env.allocator)
	free(env, env.allocator)
}
