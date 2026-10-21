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

// Helper to apply Prospect Value to a scalar reward using our new tensor backbone.
// This bridges the gap between the scalar RL environment and the tensor autograd engine.
apply_prospect_value :: proc(reward: f64, alloc: mem.Allocator) -> f64 {
	data := l.matrix_new(f64, 1, 1, alloc)
	data.data[0] = reward
	t := tensor.tensor_new(data, false, alloc)

	// α=0.88, β=0.88, λ=2.25 (Tversky & Kahneman 1992 calibration)
	v := tensor.tensor_prospect_value(t, 0.88, 0.88, 2.25, alloc)

	out := v.data.data[0]
	tensor.tensor_free(t)
	tensor.tensor_free(v)
	return out
}

prospect_ppo_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Prospect Theory PPO Trading Test (Disposition Effect) ===")

	// 1. Fetch Data
	fmt.println("- Fetching Market Data (AAPL) -")
	df := net.read_yahoo("AAPL", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df)

	n_days := df.rows - 1
	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	defer {delete(prices, allocator); delete(volumes, allocator)}

	close_col := w.column(&df, "Close")
	volume_col := w.column(&df, "Volume")

	for i in 0 ..< n_days {
		prices[i], _ = w.column_at_float(close_col, i)
		volumes[i], _ = w.column_at_float(volume_col, i)
	}

	indicators, n_indicators := ml_finance.compute_trading_features(prices, volumes, allocator)
	defer delete(indicators, allocator)

	window := 10
	fmt.printf("Data loaded: %d days, %d indicators\n", n_days, n_indicators)

	// 2. Setup Environment and Agent (Using your existing, untouched PPO architecture)
	env := ml_finance.new_trading_env(
		prices,
		volumes,
		indicators,
		n_indicators,
		window,
		0.001,
		10,
		allocator,
	)
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

	// 3. Tracking variables for Disposition Effect
	winner_holds := make([dynamic]int, 0, allocator)
	loser_holds := make([dynamic]int, 0, allocator)
	defer {delete(winner_holds); delete(loser_holds)}

	// 4. Training Loop
	fmt.println("- Training Prospect-Optimized PPO Agent -")
	n_episodes := 20
	for ep in 0 ..< n_episodes {
		if ep > 0 {ml_finance.rollout_buffer_clear(agent.buffer)}
		state := ml_finance.env_reset(&env.env)
		done := false

		current_hold_steps := 0
		entry_price := 0.0
		position_dir := 0
		in_position := false

		for !done {
			state_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, len(state.data), allocator),
				false,
				allocator,
			)
			copy(state_tensor.data.data, state.data)
			state_tensor.shape = [4]int{1, len(state.data), 1, 1}
			if len(state.data) > 0 {delete(state.data, env.allocator)}

			action, log_prob, value := ml_finance.ppo_agent_select_action(agent, state_tensor)
			step := ml_finance.env_step(&env.env, action)

			// ✅ FIX: Clamp the price index. On the final step, env_step increments
			// current_step to len(prices) (500), which is out of bounds for the array.
			// The environment internally liquidates at len(prices)-1, so we must match that.
			price_idx := min(env.env.current_step, len(prices) - 1)

			// --- TRACK DISPOSITION EFFECT ---
			if env.inventory != 0 && !in_position {
				// Just entered a position
				in_position = true
				entry_price = prices[price_idx]
				if env.inventory > 0 {position_dir = 1} else {position_dir = -1}
				current_hold_steps = 1
			} else if env.inventory == 0 && in_position {
				// Just exited a position (or force-liquidated at end of episode)
				exit_price := prices[price_idx]
				profit := (exit_price - entry_price) * f64(position_dir)
				if profit > 0 {
					append(&winner_holds, current_hold_steps)
				} else if profit < 0 {
					append(&loser_holds, current_hold_steps)
				}
				in_position = false
				current_hold_steps = 0
			} else if in_position {
				current_hold_steps += 1
			}

			// --- APPLY PROSPECT THEORY TO REWARD ---
			// We intercept the raw financial reward and warp it through the S-curve.
			// Losses are amplified by λ=2.25, gains are subjected to diminishing sensitivity.
			prospect_reward := apply_prospect_value(step.reward, allocator)

			ml_finance.rollout_buffer_add(
				agent.buffer,
				state_tensor,
				action,
				log_prob,
				prospect_reward,
				value,
				step.done,
			)
			done = step.done
			state = step.observation

			if agent.buffer.size >= 512 {
				_ = ml_finance.ppo_agent_update(agent, 10, 64)
			}
		}

		if len(state.data) > 0 {delete(state.data, env.allocator)}
		if agent.buffer.size > 0 {
			_ = ml_finance.ppo_agent_update(agent, 10, 64)
		}

		pnl := env.cash - 100000.0
		if ep % 5 == 0 || ep == n_episodes - 1 {
			fmt.printf(
				"Ep %2d/%2d | PnL: $%+9.2f | Wins: %d | Losses: %d\n",
				ep + 1,
				n_episodes,
				pnl,
				len(winner_holds),
				len(loser_holds),
			)
		}
	}

	// 5. Disposition Effect Analysis
	fmt.println("\n--- Disposition Effect Analysis ---")
	avg_winner_hold := 0.0
	for h in winner_holds {avg_winner_hold += f64(h)}
	if len(winner_holds) > 0 {avg_winner_hold /= f64(len(winner_holds))}

	avg_loser_hold := 0.0
	for h in loser_holds {avg_loser_hold += f64(h)}
	if len(loser_holds) > 0 {avg_loser_hold /= f64(len(loser_holds))}

	fmt.printf(
		"Total Winning Trades: %d | Avg Hold: %.1f steps\n",
		len(winner_holds),
		avg_winner_hold,
	)
	fmt.printf(
		"Total Losing Trades:   %d | Avg Hold: %.1f steps\n",
		len(loser_holds),
		avg_loser_hold,
	)

	if len(winner_holds) > 0 && len(loser_holds) > 0 {
		if avg_loser_hold > avg_winner_hold {
			fmt.println("✓ DISPOSITION EFFECT DETECTED: Agent holds losers longer than winners!")
			fmt.println(
				"  (This is the exact behavioral bias predicted by Prospect Theory's loss aversion).",
			)
		} else {
			fmt.println("✗ No Disposition Effect: Agent cuts losers faster (rational behavior).")
		}
	} else {
		fmt.println("Not enough trades to calculate disposition effect.")
	}

	fmt.println("=== Prospect Theory PPO Test Complete ===\n")
}
