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
	fmt.println("\n=== PPO Trading Test ===")

	// Dummy environment setup for testing
	prices := make([]f64, 1000, allocator)
	volumes := make([]f64, 1000, allocator)
	indicators := make([]f64, 1000, allocator)
	for i in 0 ..< 1000 {
		prices[i] = 100.0 + f64(i) * 0.1 + rand.float64_normal(0.0, 1.0)
		volumes[i] = 1000.0 + rand.float64_normal(0.0, 100.0)
		indicators[i] = rand.float64_normal(0.0, 1.0)
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
	fmt.printf(
		"Environment: %d steps | Window: %d | Indicators: %d\n",
		len(prices),
		env.window,
		env.n_indicators,
	)
	fmt.printf(
		"Agent:       Obs Dim: %d | Action Space: %d | Hidden Dim: 64\n",
		obs_dim,
		action_space,
	)
	fmt.println("Starting episode...\n")

	state := ml_finance.env_reset(&env.env)
	ep_reward := 0.0
	done := false
	step_count := 0

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

		ep_reward += step.reward
		step_count += 1
		done = step.done

		// ✅ Periodic Step Logging
		if step_count % 100 == 0 || done {
			fmt.printf(
				"  Step %04d | Action: %-6v | Reward: %+.4f | Inventory: %2d | Cash: $%10.2f | Ep Reward: %+.4f\n",
				step_count,
				action,
				step.reward,
				env.inventory,
				env.cash,
				ep_reward,
			)
		}

		// ✅ Verbose Update Logging
		if agent.buffer.size >= 2048 {
			stats := ml_finance.ppo_agent_update(agent, 10, 64)
			fmt.printf(
				"  [Update] Buffer cleared (%d samples). Policy Loss: %.4f | Value Loss: %.4f | Entropy: %.4f | Total Loss: %.4f\n",
				agent.buffer.capacity,
				stats.policy_loss,
				stats.value_loss,
				stats.entropy,
				stats.total_loss,
			)
		}

		tensor.tensor_free(state_tensor)
		state = step.observation
	}

	// ✅ Final Episode Summary
	fmt.println("\n--- Episode Summary ---")
	fmt.printf("Total Steps:     %d\n", step_count)
	fmt.printf("Final Cash:      $%.2f\n", env.cash)
	fmt.printf("Final Inventory: %d\n", env.inventory)
	fmt.printf("Episode Reward:  %.4f\n", ep_reward)
	fmt.printf("Initial Cash:    $100,000.00\n")
	fmt.printf(
		"Net PnL:         $%.2f (%.2f%%)\n",
		env.cash - 100000.0,
		(env.cash - 100000.0) / 100000.0 * 100.0,
	)
	fmt.println("=========================")
}
