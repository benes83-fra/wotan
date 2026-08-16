package tests

import l "../wotan/linalg"
import ml_finance "../wotan/ml_finance"
import nn "../wotan/nn"
import tensor "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

ppo_trading_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== PPO Trading Test (Verbose) ===")

	n_days := 2500
	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	indicators := make([]f64, n_days, allocator)

	price := 100.0
	mean_price := 100.0

	for i in 0 ..< n_days {
		// Ornstein-Uhlenbeck process: pulls price back to mean_price
		reversion_speed := 0.05
		noise := rand.float64_normal(0.0, 1.5)
		price += reversion_speed * (mean_price - price) + noise

		prices[i] = price
		volumes[i] = rand.float64_uniform(1000.0, 10000.0)

		// ✅ Predictive Indicator: Normalized distance from the mean
		// The agent can learn: "If this is high, SELL. If low, BUY."
		indicators[i] = (price - mean_price) / 10.0
	}

	env := ml_finance.new_trading_env(prices, volumes, indicators, 1, 10, 0.001, 10, allocator)
	defer ml_finance.trading_env_free(env)

	obs_dim := env.env.obs_dim
	action_space := env.env.action_space

	agent := ml_finance.new_ppo_agent(
		obs_dim,
		action_space,
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

	// ✅ Verbose Setup Info
	fmt.println("\n--- Configuration ---")
	fmt.printf(
		"Environment: %d steps | Window: %d | Indicators: %d\n",
		n_days,
		env.window,
		env.n_indicators,
	)
	fmt.printf(
		"Agent:       Obs Dim: %d | Action Space: %d | Hidden Dim: 64\n",
		obs_dim,
		action_space,
	)
	fmt.printf("Buffer Size: %d | Update Threshold: 2048\n", agent.buffer.capacity)
	fmt.println("---------------------\n")

	n_episodes := 10 // ✅ Multiple episodes to see learning
	update_count := 0

	for ep in 0 ..< n_episodes {
		fmt.printf("=== Episode %d/%d ===\n", ep + 1, n_episodes)

		state := ml_finance.env_reset(&env.env)
		ep_reward := 0.0
		done := false
		step_count := 0

		for !done {
			// Create as row vector [1, features] for neural network compatibility
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

			ep_reward += step.reward
			step_count += 1
			done = step.done

			// ✅ Periodic Step Logging (every 500 steps)
			if step_count % 500 == 0 {
				action_str := "Hold"
				if action == 1 {action_str = "Buy"}
				if action == 2 {action_str = "Sell"}

				fmt.printf(
					"  Step %04d | Action: %-4s | Reward: %+.4f | Inventory: %2d | Cash: $%9.2f | Buffer: %d\n",
					step_count,
					action_str,
					step.reward,
					env.inventory,
					env.cash,
					agent.buffer.size,
				)
			}

			// ✅ PPO Update Logging
			if agent.buffer.size >= 512 {
				update_count += 1
				fmt.printf(
					"  [Update #%d] Buffer full (%d samples). Running PPO update...\n",
					update_count,
					agent.buffer.size,
				)

				stats := ml_finance.ppo_agent_update(agent, 10, 64)

				fmt.printf(
					"    Policy Loss: %.4f | Value Loss: %.4f | Entropy: %.4f | Total Loss: %.4f\n",
					stats.policy_loss,
					stats.value_loss,
					stats.entropy,
					stats.total_loss,
				)
			}

			// ✅ REMOVED: tensor.tensor_free(state_tensor)
			// The buffer takes ownership and will safely free it later in rollout_buffer_clear/free

			state = step.observation
		}

		// ✅ Episode Summary
		fmt.println("\n--- Episode Summary ---")
		fmt.printf("Total Steps:     %d\n", step_count)
		fmt.printf("Final Cash:      $%.2f\n", env.cash)
		fmt.printf("Final Inventory: %d\n", env.inventory)
		fmt.printf("Episode Reward:  %.4f\n", ep_reward)
		fmt.printf(
			"Net PnL:         $%.2f (%.2f%%)\n",
			env.cash - 100000.0,
			(env.cash - 100000.0) / 100000.0 * 100.0,
		)
		fmt.println("=========================\n")
	}

	fmt.printf("Training complete! Total PPO updates: %d\n", update_count)
}
