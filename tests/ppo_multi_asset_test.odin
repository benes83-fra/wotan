package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_finance "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import tensor "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

ppo_multi_asset_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Multi-Asset PPO Trading Test (SPY vs QQQ) ===")

	// 1. Fetch Data
	spy_df := net.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	qqq_df := net.read_yahoo("QQQ", .Daily, .FiveYears, allocator)
	defer {
		w.destroy_dataframe(&spy_df)
		w.destroy_dataframe(&qqq_df)
	}

	n_days := min(spy_df.rows, qqq_df.rows)
	n_assets := 2
	n_indicators := 4
	window := 10

	// 2. Extract and compute features per asset
	spy_prices := make([]f64, n_days, allocator)
	spy_vols := make([]f64, n_days, allocator)
	qqq_prices := make([]f64, n_days, allocator)
	qqq_vols := make([]f64, n_days, allocator)
	defer {
		delete(spy_prices, allocator)
		delete(spy_vols, allocator)
		delete(qqq_prices, allocator)
		delete(qqq_vols, allocator)
	}

	spy_close := w.column(&spy_df, "Close")
	spy_vol := w.column(&spy_df, "Volume")
	qqq_close := w.column(&qqq_df, "Close")
	qqq_vol := w.column(&qqq_df, "Volume")

	for i in 0 ..< n_days {
		spy_prices[i], _ = w.column_at_float(spy_close, i)
		spy_vols[i], _ = w.column_at_float(spy_vol, i)
		qqq_prices[i], _ = w.column_at_float(qqq_close, i)
		qqq_vols[i], _ = w.column_at_float(qqq_vol, i)
	}

	spy_inds, _ := ml_finance.compute_trading_features(spy_prices, spy_vols, allocator)
	qqq_inds, _ := ml_finance.compute_trading_features(qqq_prices, qqq_vols, allocator)
	defer {
		delete(spy_inds, allocator)
		delete(qqq_inds, allocator)
	}

	// Interleave data for the environment
	prices := make([]f64, n_days * n_assets, allocator)
	volumes := make([]f64, n_days * n_assets, allocator)
	indicators := make([]f64, n_days * n_assets * n_indicators, allocator)
	defer {
		delete(prices, allocator)
		delete(volumes, allocator)
		delete(indicators, allocator)
	}

	for t in 0 ..< n_days {
		prices[t * 2 + 0] = spy_prices[t]
		prices[t * 2 + 1] = qqq_prices[t]
		volumes[t * 2 + 0] = spy_vols[t]
		volumes[t * 2 + 1] = qqq_vols[t]

		for ind in 0 ..< n_indicators {
			indicators[t * 2 * n_indicators + 0 * n_indicators + ind] =
				spy_inds[t * n_indicators + ind]
			indicators[t * 2 * n_indicators + 1 * n_indicators + ind] =
				qqq_inds[t * n_indicators + ind]
		}
	}

	// 3. Define Discrete Portfolio Actions (Target Weights)
	// Action 0: 100% Cash
	// Action 1: 100% SPY
	// Action 2: 100% QQQ
	// Action 3: 50% SPY / 50% QQQ
	// Action 4: 75% SPY / 25% QQQ
	// Action 5: 25% SPY / 75% QQQ
	actions := make([]ml_finance.ActionDef, 6, allocator)
	actions[0] = ml_finance.ActionDef {
		weights = []f64{0.0, 0.0},
	}
	actions[1] = ml_finance.ActionDef {
		weights = []f64{1.0, 0.0},
	}
	actions[2] = ml_finance.ActionDef {
		weights = []f64{0.0, 1.0},
	}
	actions[3] = ml_finance.ActionDef {
		weights = []f64{0.5, 0.5},
	}
	actions[4] = ml_finance.ActionDef {
		weights = []f64{0.75, 0.25},
	}
	actions[5] = ml_finance.ActionDef {
		weights = []f64{0.25, 0.75},
	}

	env := ml_finance.new_multi_asset_env(
		prices,
		volumes,
		indicators,
		n_assets,
		n_indicators,
		window,
		actions,
		0.001,
		allocator,
	)
	defer ml_finance.multi_asset_env_free(env)

	agent := ml_finance.new_ppo_agent(
		env.env.obs_dim,
		env.env.action_space,
		128,
		0.99,
		0.95,
		0.2,
		0.5,
		0.05,
		3e-4,
		allocator,
	)
	defer ml_finance.ppo_agent_free(agent)

	fmt.printf("Obs Dim: %d | Action Space: %d\n", env.env.obs_dim, env.env.action_space)

	// 4. Train
	n_episodes := 20
	for ep in 0 ..< n_episodes {
		if ep > 0 {
			ml_finance.rollout_buffer_clear(agent.buffer)
		}
		state := ml_finance.env_reset(&env.env)
		done := false

		for !done {
			state_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, len(state.data), allocator),
				false,
				allocator,
			)
			copy(state_tensor.data.data, state.data)
			state_tensor.shape = [4]int{1, len(state.data), 1, 1}

			action, log_prob, value := ml_finance.ppo_agent_select_action(agent, state_tensor)
			step := ml_finance.env_step(&env.env, action)

			ml_finance.rollout_buffer_add(
				agent.buffer,
				state_tensor,
				action,
				log_prob,
				step.reward,
				value,
				step.done,
			)
			done = step.done
			state = step.observation

			if agent.buffer.size >= 512 {
				_ = ml_finance.ppo_agent_update(agent, 10, 64)
			}
		}

		pnl := env.cash - 100000.0
		fmt.printf(
			"Ep %2d/%2d | PnL: $%+9.2f (%+.2f%%) | Cash: $%.2f | Pos: [%.2f, %.2f]\n",
			ep + 1,
			n_episodes,
			pnl,
			pnl / 1000.0,
			env.cash,
			env.positions[0],
			env.positions[1],
		)
	}
}
