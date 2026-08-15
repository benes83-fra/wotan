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

	state := ml_finance.env_reset(&env.env)
	ep_reward := 0.0
	done := false

	for !done {
		// ✅ FIXED: Convert Observation to Tensor before passing to agent
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
		done = step.done

		if agent.buffer.size >= 2048 {
			ml_finance.ppo_agent_update(agent, 10, 64)
		}

		// ✅ FIXED: Free the tensor we created
		tensor.tensor_free(state_tensor)
		state = step.observation
	}

	fmt.printf("Episode Reward: %.2f\n", ep_reward)
}
