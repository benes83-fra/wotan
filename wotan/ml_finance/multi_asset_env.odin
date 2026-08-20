package ml_finance

import l "../linalg"
import nn "../nn"
import tensor "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ActionDef defines a discrete portfolio allocation
// weights: array of length n_assets. Sum must be <= 1.0.
// (Any remainder is automatically kept in Cash)
ActionDef :: struct {
	weights: []f64,
}

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
	peak_cash:       f64,
	positions:       []f64, // Fractional shares per asset
	action_defs:     []ActionDef,
	transaction_fee: f64,
	allocator:       mem.Allocator,
}

multi_asset_env_reset :: proc(env: ^Environment) -> Observation {
	t_env := cast(^MultiAssetEnv)env
	t_env.cash = t_env.initial_cash
	t_env.peak_cash = t_env.initial_cash
	for i in 0 ..< t_env.n_assets {
		t_env.positions[i] = 0.0
	}
	t_env.env.current_step = t_env.window
	t_env.env.done = false

	curr_prices := make([]f64, t_env.n_assets, t_env.allocator)
	defer delete(curr_prices, t_env.allocator)
	for a in 0 ..< t_env.n_assets {
		curr_prices[a] = t_env.prices[t_env.window * t_env.n_assets + a]
	}

	obs := build_multi_asset_obs(t_env, t_env.window, curr_prices, t_env.initial_cash)
	return Observation{data = obs, shape = [4]int{1, len(obs), 1, 1}}
}

multi_asset_env_step :: proc(env: ^Environment, action: int) -> Step {
	t_env := cast(^MultiAssetEnv)env
	step_idx := t_env.env.current_step

	// 1. Get current prices for all assets
	curr_prices := make([]f64, t_env.n_assets, t_env.allocator)
	defer delete(curr_prices, t_env.allocator)
	for a in 0 ..< t_env.n_assets {
		curr_prices[a] = t_env.prices[step_idx * t_env.n_assets + a]
	}

	// 2. Calculate total portfolio value BEFORE action
	prev_value := t_env.cash
	for a in 0 ..< t_env.n_assets {
		prev_value += t_env.positions[a] * curr_prices[a]
	}

	// 3. Execute Rebalancing Action
	if action >= 0 && action < len(t_env.action_defs) {
		target_weights := t_env.action_defs[action].weights

		for a in 0 ..< t_env.n_assets {
			target_val := prev_value * target_weights[a]
			curr_val := t_env.positions[a] * curr_prices[a]
			delta_val := target_val - curr_val

			// Threshold ($50) to prevent churning on tiny weight drifts
			if math.abs(delta_val) > 50.0 {
				shares := delta_val / curr_prices[a]
				fee := math.abs(delta_val) * t_env.transaction_fee
				t_env.positions[a] += shares
				t_env.cash -= delta_val + fee
			}
		}
	}

	// 4. Calculate total portfolio value AFTER action
	curr_value := t_env.cash
	for a in 0 ..< t_env.n_assets {
		curr_value += t_env.positions[a] * curr_prices[a]
	}

	// 5. Update peak and calculate drawdown
	if curr_value > t_env.peak_cash {
		t_env.peak_cash = curr_value
	}
	drawdown := 0.0
	if t_env.peak_cash > 0.0 {
		drawdown = (t_env.peak_cash - curr_value) / t_env.peak_cash
	}

	// 6. Reward Shaping
	reward := (curr_value - prev_value) / t_env.initial_cash
	reward -= drawdown * 2.0 // Drawdown penalty

	// 7. Build observation for the NEXT step
	obs := build_multi_asset_obs(t_env, step_idx, curr_prices, curr_value)

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
				obs[idx] = (t_env.prices[w_idx * t_env.n_assets + a] / base_price) - 1.0
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

	// 2. Current Portfolio State (Crucial for Multi-Asset!)
	// The agent MUST know its current weights to decide if it needs to rebalance
	for a in 0 ..< t_env.n_assets {
		asset_val := t_env.positions[a] * curr_prices[a]
		obs[idx] = asset_val / total_value
		idx += 1
	}
	obs[idx] = t_env.cash / total_value // Cash ratio
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
	env.env.max_steps = len(prices) / n_assets - 1
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
