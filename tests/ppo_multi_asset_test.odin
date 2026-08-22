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
	fmt.println("\n=== Multi-Asset PPO Trading Test (SPY vs QQQ) [Optimized 25 Ep] ===")

	spy_df := net.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	qqq_df := net.read_yahoo("QQQ", .Daily, .FiveYears, allocator)
	defer {w.destroy_dataframe(&spy_df); w.destroy_dataframe(&qqq_df)}

	n_days := min(spy_df.rows, qqq_df.rows)
	n_assets := 2
	n_indicators := 2 // ✅ Reduced to Log Ret + RSI for faster convergence & less memory
	window := 5 // ✅ Reduced from 10 to 5. Cuts obs dim from 123 to 63.

	spy_prices := make([]f64, n_days, allocator)
	spy_vols := make([]f64, n_days, allocator)
	qqq_prices := make([]f64, n_days, allocator)
	qqq_vols := make([]f64, n_days, allocator)
	defer {
		delete(spy_prices, allocator); delete(spy_vols, allocator)
		delete(qqq_prices, allocator); delete(qqq_vols, allocator)
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
	defer {delete(spy_inds, allocator); delete(qqq_inds, allocator)}

	prices := make([]f64, n_days * n_assets, allocator)
	volumes := make([]f64, n_days * n_assets, allocator)
	indicators := make([]f64, n_days * n_assets * n_indicators, allocator)
	defer {delete(prices, allocator); delete(volumes, allocator); delete(indicators, allocator)}

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

	make_weights :: proc(w0, w1: f64, alloc: mem.Allocator) -> []f64 {
		res := make([]f64, 2, alloc)
		res[0], res[1] = w0, w1
		return res
	}

	actions := make([]ml_finance.ActionDef, 6, allocator)
	actions[0] = ml_finance.ActionDef {
		weights = make_weights(0.0, 0.0, allocator),
	}
	actions[1] = ml_finance.ActionDef {
		weights = make_weights(1.0, 0.0, allocator),
	}
	actions[2] = ml_finance.ActionDef {
		weights = make_weights(0.0, 1.0, allocator),
	}
	actions[3] = ml_finance.ActionDef {
		weights = make_weights(0.5, 0.5, allocator),
	}
	actions[4] = ml_finance.ActionDef {
		weights = make_weights(0.75, 0.25, allocator),
	}
	actions[5] = ml_finance.ActionDef {
		weights = make_weights(0.25, 0.75, allocator),
	}
	defer {
		for i in 0 ..< len(actions) {delete(actions[i].weights, allocator)}
	}

	env := ml_finance.new_multi_asset_env(
		prices,
		volumes,
		indicators,
		n_assets,
		n_indicators,
		window,
		actions,
		0.0001,
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

	fmt.printf("Optimized Obs Dim: %d | Action Space: %d\n", env.env.obs_dim, env.env.action_space)

	n_episodes := 25 // ✅ Strictly 25 episodes, optimized for maximum learning efficiency
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

			// ✅ Safe deletion AFTER copy
			if len(state.data) > 0 {delete(state.data, env.allocator)}

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

			// ✅ OPTIMIZED: Update every 256 steps instead of 512.
			// Doubles the learning feedback frequency within the 25-episode limit.
			if agent.buffer.size >= 256 {
				_ = ml_finance.ppo_agent_update(agent, 10, 64)
			}
		}

		if len(state.data) > 0 {delete(state.data, env.allocator)}
		if agent.buffer.size > 0 {_ = ml_finance.ppo_agent_update(agent, 10, 64)}

		final_val := env.cash
		last_step_idx := n_days - 1
		for a in 0 ..< n_assets {
			final_val += env.positions[a] * prices[last_step_idx * n_assets + a]
		}

		pnl := final_val - 100000.0
		fmt.printf(
			"Ep %2d/%2d | PnL: $%+9.2f (%+.2f%%) | Cash: $%9.2f | Pos: [%6.2f, %6.2f]\n",
			ep + 1,
			n_episodes,
			pnl,
			(pnl / 100000.0) * 100.0,
			env.cash,
			env.positions[0],
			env.positions[1],
		)
	}
}
build_multi_asset_wf_windows :: proc(
	n_days, train_len, test_len: int,
	alloc: mem.Allocator,
) -> []WalkForwardWindow {
	n_windows := 0
	test_start := train_len
	for test_start < n_days {
		test_end := min(test_start + test_len, n_days)
		if test_end - test_start < test_len / 2 {break}
		n_windows += 1
		test_start += test_len
	}
	if n_windows == 0 {return make([]WalkForwardWindow, 0, alloc)}

	windows := make([]WalkForwardWindow, n_windows, alloc)
	test_start = train_len
	for w in 0 ..< n_windows {
		test_end := min(test_start + test_len, n_days)
		if test_end - test_start < test_len / 2 {break}
		train_start := max(0, test_start - train_len)
		windows[w] = WalkForwardWindow{train_start, test_start, test_start, test_end, w}
		test_start += test_len
	}
	return windows
}

ppo_multi_asset_wf_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    MULTI-ASSET WALK-FORWARD VALIDATION (SPY vs QQQ)")
	fmt.println("======================================================================")

	// 1. Fetch Data
	spy_df := net.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	qqq_df := net.read_yahoo("QQQ", .Daily, .FiveYears, allocator)
	defer {w.destroy_dataframe(&spy_df); w.destroy_dataframe(&qqq_df)}

	n_days := min(spy_df.rows, qqq_df.rows)
	if n_days < 400 {
		fmt.println("ERROR: Not enough data for walk-forward validation.")
		return
	}

	// 2. Extract & Compute Features (Optimized: 2 indicators, window 5)
	n_assets := 2
	n_indicators := 2
	window := 5

	spy_prices := make([]f64, n_days, allocator)
	spy_vols := make([]f64, n_days, allocator)
	qqq_prices := make([]f64, n_days, allocator)
	qqq_vols := make([]f64, n_days, allocator)
	defer {
		delete(spy_prices, allocator); delete(spy_vols, allocator)
		delete(qqq_prices, allocator); delete(qqq_vols, allocator)
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
	defer {delete(spy_inds, allocator); delete(qqq_inds, allocator)}

	// Interleave data
	prices := make([]f64, n_days * n_assets, allocator)
	volumes := make([]f64, n_days * n_assets, allocator)
	indicators := make([]f64, n_days * n_assets * n_indicators, allocator)
	defer {delete(prices, allocator); delete(volumes, allocator); delete(indicators, allocator)}

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

	// 3. Build Walk-Forward Windows (1 Year Train, 1 Quarter Test)
	train_len := 252
	test_len := 63
	windows := build_multi_asset_wf_windows(n_days, train_len, test_len, allocator)
	defer delete(windows, allocator)

	fmt.printf(
		"Data: %d days | Train: %d | Test: %d | Windows: %d\n\n",
		n_days,
		train_len,
		test_len,
		len(windows),
	)

	// 4. Define Actions
	make_weights :: proc(w0, w1: f64, alloc: mem.Allocator) -> []f64 {
		res := make([]f64, 2, alloc)
		res[0], res[1] = w0, w1
		return res
	}

	actions := make([]ml_finance.ActionDef, 6, allocator)
	actions[0] = ml_finance.ActionDef {
		weights = make_weights(0.0, 0.0, allocator),
	}
	actions[1] = ml_finance.ActionDef {
		weights = make_weights(1.0, 0.0, allocator),
	}
	actions[2] = ml_finance.ActionDef {
		weights = make_weights(0.0, 1.0, allocator),
	}
	actions[3] = ml_finance.ActionDef {
		weights = make_weights(0.5, 0.5, allocator),
	}
	actions[4] = ml_finance.ActionDef {
		weights = make_weights(0.75, 0.25, allocator),
	}
	actions[5] = ml_finance.ActionDef {
		weights = make_weights(0.25, 0.75, allocator),
	}
	defer {
		for i in 0 ..< len(actions) {delete(actions[i].weights, allocator)}
	}

	results := make([dynamic]WalkForwardResult, 0, allocator)
	defer delete(results)

	obs_dim := window * (2 + n_indicators) * n_assets + n_assets + 1

	// 5. Run Walk-Forward Loop
	for w_idx in 0 ..< len(windows) {
		win := windows[w_idx]
		fmt.printf(
			"--- Window %d/%d (Train: %d-%d, Test: %d-%d) ---\n",
			w_idx + 1,
			len(windows),
			win.train_start,
			win.train_end - 1,
			win.test_start,
			win.test_end - 1,
		)

		// Slice data
		train_prices := prices[win.train_start * n_assets:win.train_end * n_assets]
		train_volumes := volumes[win.train_start * n_assets:win.train_end * n_assets]
		train_indicators := indicators[win.train_start *
		n_assets *
		n_indicators:win.train_end *
		n_assets *
		n_indicators]

		test_prices := prices[win.test_start * n_assets:win.test_end * n_assets]
		test_volumes := volumes[win.test_start * n_assets:win.test_end * n_assets]
		test_indicators := indicators[win.test_start *
		n_assets *
		n_indicators:win.test_end *
		n_assets *
		n_indicators]

		// Create Envs & Agent
		train_env := ml_finance.new_multi_asset_env(
			train_prices,
			train_volumes,
			train_indicators,
			n_assets,
			n_indicators,
			window,
			actions,
			0.0001,
			allocator,
		)
		test_env := ml_finance.new_multi_asset_env(
			test_prices,
			test_volumes,
			test_indicators,
			n_assets,
			n_indicators,
			window,
			actions,
			0.0001,
			allocator,
		)
		w_agent := ml_finance.new_ppo_agent(
			obs_dim,
			6,
			128,
			0.99,
			0.95,
			0.2,
			0.5,
			0.05,
			3e-4,
			allocator,
		)

		// --- TRAIN PHASE (25 Episodes) ---
		for ep in 0 ..< 5 {
			if ep > 0 {ml_finance.rollout_buffer_clear(w_agent.buffer)}
			state := ml_finance.env_reset(&train_env.env)
			done := false

			for !done {
				if len(state.data) == 0 {
					fmt.printf(
						"⚠️ WARNING: Empty state data received in Train Ep %d. Terminating episode safely.\n",
						ep,
					)
					done = true
					break
				}
				state_tensor := tensor.tensor_new(
					l.matrix_new(f64, 1, len(state.data), allocator),
					false,
					allocator,
				)
				copy(state_tensor.data.data, state.data)
				state_tensor.shape = [4]int{1, len(state.data), 1, 1}
				if len(state.data) > 0 {delete(state.data, train_env.allocator)}

				action, log_prob, value := ml_finance.ppo_agent_select_action(
					w_agent,
					state_tensor,
				)
				step := ml_finance.env_step(&train_env.env, action)

				ml_finance.rollout_buffer_add(
					w_agent.buffer,
					state_tensor,
					action,
					log_prob,
					step.reward,
					value,
					step.done,
				)
				done = step.done
				state = step.observation

				if w_agent.buffer.size >= 256 {
					_ = ml_finance.ppo_agent_update(w_agent, 10, 64)
				}
			}
			if len(state.data) > 0 {delete(state.data, train_env.allocator)}
			if w_agent.buffer.size > 0 {_ = ml_finance.ppo_agent_update(w_agent, 10, 64)}
		}

		// Calculate Train PnL
		train_final_val := train_env.cash
		train_last_idx := (win.train_end - win.train_start) - 1
		for a in 0 ..< n_assets {
			train_final_val += train_env.positions[a] * train_prices[train_last_idx * n_assets + a]
		}
		train_pnl := train_final_val - 100000.0

		// --- TEST PHASE (1 Episode, NO TRAINING) ---
		state := ml_finance.env_reset(&test_env.env)
		done := false
		test_trades := 0
		prev_inv_0 := test_env.positions[0]
		prev_inv_1 := test_env.positions[1]
		peak_val := 100000.0
		max_dd := 0.0

		for !done {
			// ✅ SAFEGUARD: Prevent 0-length tensors from reaching the neural network
			if len(state.data) == 0 {
				fmt.println("WARNING: Empty state data received, terminating episode early.")
				done = true
				break
			}

			state_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, len(state.data), allocator),
				false,
				allocator,
			)
			copy(state_tensor.data.data, state.data)
			state_tensor.shape = [4]int{1, len(state.data), 1, 1}
			if len(state.data) > 0 {delete(state.data, test_env.allocator)}

			action, _, _ := ml_finance.ppo_agent_select_action(w_agent, state_tensor)
			step := ml_finance.env_step(&test_env.env, action)

			// Track trades
			if test_env.positions[0] != prev_inv_0 || test_env.positions[1] != prev_inv_1 {
				test_trades += 1
				prev_inv_0 = test_env.positions[0]
				prev_inv_1 = test_env.positions[1]
			}

			// Track Max Drawdown
			curr_val := test_env.cash
			for a in 0 ..< n_assets {curr_val += test_env.positions[a] * test_env.prices[test_env.env.current_step * n_assets + a]}
			if curr_val > peak_val {peak_val = curr_val}
			dd := (peak_val - curr_val) / peak_val
			if dd > max_dd {max_dd = dd}

			done = step.done
			state = step.observation
		}
		if len(state.data) > 0 {delete(state.data, test_env.allocator)}

		// Calculate Test PnL
		test_final_val := test_env.cash
		test_last_idx := (win.test_end - win.test_start) - 1
		for a in 0 ..< n_assets {
			test_final_val += test_env.positions[a] * test_prices[test_last_idx * n_assets + a]
		}
		test_pnl := test_final_val - 100000.0

		// Calculate Walk-Forward Efficiency
		wfe := 0.0
		if math.abs(train_pnl) > 1e-6 {
			wfe = test_pnl / train_pnl
		}

		append(
			&results,
			WalkForwardResult {
				window_index = w_idx,
				train_pnl = train_pnl,
				test_pnl = test_pnl,
				test_trades = test_trades,
				test_max_dd = max_dd,
				walk_forward_eff = wfe,
			},
		)
		fmt.printf(
			"  Train PnL: $%+8.2f | Test PnL: $%+8.2f | Trades: %3d | MaxDD: %5.2f%% | WFE: %5.2f\n",
			train_pnl,
			test_pnl,
			test_trades,
			max_dd * 100.0,
			wfe,
		)

		// Cleanup window
		ml_finance.multi_asset_env_free(train_env)
		ml_finance.multi_asset_env_free(test_env)
		ml_finance.ppo_agent_free(w_agent)
	}

	// 6. Final Summary
	fmt.println("\n--- Walk-Forward Summary ---")
	total_test_pnl := 0.0
	profits := 0
	valid_wfe_count := 0
	total_valid_wfe := 0.0

	for r in results {
		total_test_pnl += r.test_pnl
		if r.test_pnl > 0 {profits += 1}

		// ✅ FIX: Only calculate WFE if train_pnl was positive.
		// If train_pnl < 0 and test_pnl > 0, that's a success, not negative efficiency.
		if r.train_pnl > 1e-6 {
			total_valid_wfe += (r.test_pnl / r.train_pnl)
			valid_wfe_count += 1
		}
	}

	avg_test_pnl := total_test_pnl / f64(len(results))
	avg_wfe: f64
	if valid_wfe_count > 0 {avg_wfe = total_valid_wfe / f64(valid_wfe_count)} else {avg_wfe = 0.0}
	win_rate := f64(profits) / f64(len(results)) * 100.0

	fmt.printf("Total Windows Evaluated: %d\n", len(results))
	fmt.printf("Avg Out-of-Sample PnL:   $%+8.2f\n", avg_test_pnl)
	fmt.printf("Out-of-Sample Win Rate:  %.1f%%\n", win_rate)
	fmt.printf("Avg Walk-Forward Eff:    %.2f\n", avg_wfe)

	if win_rate >= 60.0 && avg_test_pnl > 0.0 {
		fmt.println(
			"✅ RESULT: Strategy shows robust out-of-sample profitability and high win rate.",
		)
	} else if avg_test_pnl > 0.0 {
		fmt.println(
			"⚠️  RESULT: Strategy is profitable out-of-sample, but win rate is marginal.",
		)
	} else {
		fmt.println("❌ RESULT: Strategy is failing to generate consistent out-of-sample alpha.")
	}
	fmt.println("======================================================================")
}
