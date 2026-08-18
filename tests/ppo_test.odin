// ===== ./tests/ppo_trading_test.odin =====
package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_finance "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import tensor "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// 1. SYNTHETIC DATA TEST (Ornstein-Uhlenbeck)
// ============================================================================
ppo_trading_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== PPO Trading Test (Synthetic Data) ===")

	n_days := 2500
	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	indicators := make([]f64, n_days, allocator)
	defer {
		delete(prices, allocator)
		delete(volumes, allocator)
		delete(indicators, allocator)
	}

	price := 100.0
	mean_price := 100.0

	for i in 0 ..< n_days {
		reversion_speed := 0.05
		noise := rand.float64_normal(0.0, 1.5)
		price += reversion_speed * (mean_price - price) + noise

		prices[i] = price
		volumes[i] = rand.float64_uniform(1000.0, 10000.0)
		indicators[i] = (price - mean_price) / 10.0
	}

	env := ml_finance.new_trading_env(prices, volumes, indicators, 1, 10, 0.001, 10, allocator)
	defer ml_finance.trading_env_free(env)

	agent := ml_finance.new_ppo_agent(
		env.env.obs_dim,
		env.env.action_space,
		64,
		0.99,
		0.95,
		0.2,
		0.5,
		0.01,
		3e-4,
		allocator,
	)
	defer ml_finance.ppo_agent_free(agent)

	fmt.printf(
		"Environment: %d steps | Obs Dim: %d | Action Space: %d\n",
		n_days,
		env.env.obs_dim,
		env.env.action_space,
	)

	for ep in 0 ..< 5 {
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
		fmt.printf("Ep %d/%d | PnL: $%+.2f\n", ep + 1, 5, env.cash - 100000.0)
	}
}

// ============================================================================
// 2. REAL MARKET DATA TEST (Yahoo Finance)
// ============================================================================
ppo_trading_real_data_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== PPO Trading Test (REAL MARKET DATA) ===")

	fmt.println("\n[1/4] Fetching real market data from Yahoo Finance...")
	symbol := "SPY"
	spy_df := net.read_yahoo(symbol, .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	if spy_df.rows < 100 {
		fmt.println("ERROR: Failed to fetch enough SPY data.")
		return
	}
	fmt.printf("  Fetched %d days of %s data\n", spy_df.rows, symbol)

	fmt.println("\n[2/4] Extracting prices, volumes, and computing features...")
	n_days := spy_df.rows
	close_col := w.column(&spy_df, "Close")
	volume_col := w.column(&spy_df, "Volume")

	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	defer {delete(prices, allocator); delete(volumes, allocator)}

	for i in 0 ..< n_days {
		prices[i], _ = w.column_at_float(close_col, i)
		volumes[i], _ = w.column_at_float(volume_col, i)
	}

	// ✅ Use the new centralized feature engineering function
	indicators, n_indicators := ml_finance.compute_trading_features(prices, volumes, allocator)
	defer delete(indicators, allocator)

	window := 10
	fmt.printf(
		"  Features per step: %d | Observation dim: %d\n",
		n_indicators,
		window * (2 + n_indicators),
	)

	fmt.println("\n[3/4] Creating environment and PPO agent...")
	env := ml_finance.new_trading_env(
		prices,
		volumes,
		indicators,
		n_indicators,
		window,
		0.001,
		200,
		allocator,
	)
	defer ml_finance.trading_env_free(env)

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

	fmt.println("\n[4/4] Training...")
	n_episodes := 20
	for ep in 0 ..< n_episodes {
		if ep > 0 {ml_finance.rollout_buffer_clear(agent.buffer)}
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
			"  Ep %2d/%2d | PnL: $%+9.2f (%+.2f%%)\n",
			ep + 1,
			n_episodes,
			pnl,
			pnl / 1000.0,
		)
	}
}

// ============================================================================
// 3. WALK-FORWARD VALIDATION TEST
// ============================================================================
WalkForwardWindow :: struct {
	train_start, train_end, test_start, test_end, window_index: int,
}

WalkForwardResult :: struct {
	window_index:                                       int,
	train_pnl, test_pnl, test_max_dd, walk_forward_eff: f64,
	test_trades:                                        int,
}

walk_forward_validation_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    WALK-FORWARD VALIDATION: PPO TRADING AGENT")
	fmt.println("======================================================================")

	spy_df := net.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)
	if spy_df.rows < 400 {return}

	n_days := spy_df.rows
	close_col := w.column(&spy_df, "Close")
	volume_col := w.column(&spy_df, "Volume")

	all_prices := make([]f64, n_days, allocator)
	all_volumes := make([]f64, n_days, allocator)
	defer {delete(all_prices, allocator); delete(all_volumes, allocator)}

	for i in 0 ..< n_days {
		all_prices[i], _ = w.column_at_float(close_col, i)
		all_volumes[i], _ = w.column_at_float(volume_col, i)
	}

	indicators, n_indicators := ml_finance.compute_trading_features(
		all_prices,
		all_volumes,
		allocator,
	)
	defer delete(indicators, allocator)

	window := 10
	train_len := 252
	test_len := 63

	// Simple window builder
	n_windows := (n_days - train_len) / test_len
	results := make([dynamic]WalkForwardResult, 0, allocator)
	defer delete(results)

	for w_idx in 0 ..< n_windows {
		test_start := train_len + w_idx * test_len
		test_end := min(test_start + test_len, n_days)
		train_start := max(0, test_start - train_len)

		train_env := ml_finance.new_trading_env(
			all_prices[train_start:test_start],
			all_volumes[train_start:test_start],
			indicators[train_start * n_indicators:test_start * n_indicators],
			n_indicators,
			window,
			0.001,
			10,
			allocator,
		)

		test_env := ml_finance.new_trading_env(
			all_prices[test_start:test_end],
			all_volumes[test_start:test_end],
			indicators[test_start * n_indicators:test_end * n_indicators],
			n_indicators,
			window,
			0.001,
			10,
			allocator,
		)

		w_agent := ml_finance.new_ppo_agent(
			window * (2 + n_indicators),
			3,
			64,
			0.99,
			0.95,
			0.2,
			0.5,
			0.01,
			3e-4,
			allocator,
		)

		// Train
		for ep in 0 ..< 3 {
			state := ml_finance.env_reset(&train_env.env)
			done := false
			for !done {
				st := tensor.tensor_new(
					l.matrix_new(f64, 1, len(state.data), allocator),
					false,
					allocator,
				)
				copy(st.data.data, state.data); st.shape = [4]int{1, len(state.data), 1, 1}
				a, lp, v := ml_finance.ppo_agent_select_action(w_agent, st)
				step := ml_finance.env_step(&train_env.env, a)
				ml_finance.rollout_buffer_add(w_agent.buffer, st, a, lp, step.reward, v, step.done)
				done = step.done; state = step.observation
				if w_agent.buffer.size >= 256 {_ = ml_finance.ppo_agent_update(w_agent, 5, 64)}
			}
		}
		is_pnl := train_env.cash - 100000.0

		// Test
		state := ml_finance.env_reset(&test_env.env)
		done := false
		trades := 0; prev_inv := test_env.inventory; peak := 100000.0; max_dd := 0.0
		for !done {
			st := tensor.tensor_new(
				l.matrix_new(f64, 1, len(state.data), allocator),
				false,
				allocator,
			)
			copy(st.data.data, state.data); st.shape = [4]int{1, len(state.data), 1, 1}
			a, _, _ := ml_finance.ppo_agent_select_action(w_agent, st)
			step := ml_finance.env_step(&test_env.env, a)
			if test_env.cash > peak {peak = test_env.cash}
			dd := (peak - test_env.cash) / peak
			if dd > max_dd {max_dd = dd}
			if test_env.inventory != prev_inv {trades += 1; prev_inv = test_env.inventory}
			done = step.done; state = step.observation
		}

		test_pnl := test_env.cash - 100000.0
		wfe := 0.0
		if math.abs(is_pnl) > 1e-6 {wfe = test_pnl / is_pnl}

		append(&results, WalkForwardResult{w_idx, is_pnl, test_pnl, max_dd, wfe, trades})
		fmt.printf(
			"Win %2d | Train: $%+8.2f | Test: $%+8.2f | Trades: %3d | MaxDD: %5.2f%% | WFE: %5.2f\n",
			w_idx,
			is_pnl,
			test_pnl,
			trades,
			max_dd * 100.0,
			wfe,
		)

		ml_finance.trading_env_free(train_env)
		ml_finance.trading_env_free(test_env)
		ml_finance.ppo_agent_free(w_agent)
	}

	fmt.println("\n--- Walk-Forward Summary ---")
	total_test := 0.0; profits := 0
	for r in results {
		total_test += r.test_pnl
		if r.test_pnl > 0 {profits += 1}
	}
	fmt.printf(
		"Avg OOS PnL: $%+.2f | Win Rate: %.1f%%\n",
		total_test / f64(len(results)),
		f64(profits) / f64(len(results)) * 100.0,
	)
}
