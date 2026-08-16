package ml_finance

import l "../linalg"
import nn "../nn"
import tensor "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Observation represents the state seen by the agent
Observation :: struct {
	data:  []f64, // Flattened observation vector
	shape: [4]int, // Optional shape metadata
}

// Step represents a single transition in the environment
Step :: struct {
	observation: Observation,
	reward:      f64,
	done:        bool,
	info:        string,
}

// Environment is the base interface for all RL environments
Environment :: struct {
	action_space: int,
	obs_dim:      int,
	max_steps:    int,
	current_step: int,
	done:         bool,
	reset_fn:     proc(env: ^Environment) -> Observation,
	step_fn:      proc(env: ^Environment, action: int) -> Step,
}

env_reset :: proc(env: ^Environment) -> Observation {
	env.current_step = 0
	env.done = false
	return env.reset_fn(env)
}

env_step :: proc(env: ^Environment, action: int) -> Step {
	env.current_step += 1
	if env.current_step >= env.max_steps {
		env.done = true
	}
	return env.step_fn(env, action)
}

// TradingEnv simulates a financial market for RL agents
TradingEnv :: struct {
	env:             Environment,
	prices:          []f64,
	volumes:         []f64,
	indicators:      []f64,
	n_indicators:    int,
	window:          int,
	cash:            f64,
	initial_cash:    f64,
	inventory:       i32,
	entry_price:     f64,
	transaction_fee: f64,
	max_position:    i32,
	allocator:       mem.Allocator,
}

trading_env_reset :: proc(env: ^Environment) -> Observation {
	t_env := cast(^TradingEnv)env
	t_env.initial_cash = 100000.0 //
	t_env.cash = 100000.0
	t_env.inventory = 0
	t_env.entry_price = 0.0
	t_env.env.current_step = t_env.window
	t_env.env.done = false

	obs := make([]f64, t_env.window * (2 + t_env.n_indicators), t_env.allocator)
	idx := 0
	for w in 0 ..< t_env.window {
		obs[idx] = t_env.prices[t_env.env.current_step - t_env.window + w]
		idx += 1
	}
	for w in 0 ..< t_env.window {
		obs[idx] = t_env.volumes[t_env.env.current_step - t_env.window + w]
		idx += 1
	}
	for w in 0 ..< t_env.window {
		obs[idx] =
			t_env.indicators[(t_env.env.current_step - t_env.window + w) * t_env.n_indicators]
		idx += 1
	}

	return Observation{data = obs, shape = [4]int{1, len(obs), 1, 1}}
}

trading_env_step :: proc(env: ^Environment, action: int) -> Step {
	t_env := cast(^TradingEnv)env

	// ✅ CRITICAL FIX: If the environment is already done, liquidate and return safely.
	// This prevents accessing prices[current_step] when current_step == max_steps.
	if t_env.env.done {
		if t_env.inventory != 0 {
			// Liquidate at the last available valid price
			price := t_env.prices[len(t_env.prices) - 1]
			if t_env.inventory > 0 {
				t_env.cash += price * (1.0 - t_env.transaction_fee) * f64(t_env.inventory)
			} else {
				t_env.cash -= price * (1.0 + t_env.transaction_fee) * f64(-t_env.inventory)
			}
			profit := (t_env.cash - t_env.initial_cash) / t_env.initial_cash
			t_env.inventory = 0

			obs_dim := t_env.window * (2 + t_env.n_indicators)
			obs := make([]f64, obs_dim, t_env.allocator)

			return Step {
				observation = Observation{data = obs, shape = [4]int{1, obs_dim, 1, 1}},
				reward = profit,
				done = true,
				info = "Liquidated at end",
			}
		}

		// Already flat, just return a terminal step
		obs_dim := t_env.window * (2 + t_env.n_indicators)
		obs := make([]f64, obs_dim, t_env.allocator)
		return Step {
			observation = Observation{data = obs, shape = [4]int{1, obs_dim, 1, 1}},
			reward = 0.0,
			done = true,
			info = "Already done",
		}
	}

	// 1. Calculate portfolio value BEFORE action
	prev_price := t_env.prices[max(0, t_env.env.current_step - 1)]
	prev_value := t_env.cash + f64(t_env.inventory) * prev_price

	price := t_env.prices[t_env.env.current_step]
	reward := 0.0
	info := ""

	// 2. Execute action
	if action == 1 && t_env.inventory < t_env.max_position {
		cost := price * (1.0 + t_env.transaction_fee)
		if t_env.cash >= cost {
			t_env.cash -= cost
			t_env.inventory += 1
			if t_env.inventory == 1 {
				t_env.entry_price = price
			}
			info = "Bought"
		}
	} else if action == 2 && t_env.inventory > -t_env.max_position {
		revenue := price * (1.0 - t_env.transaction_fee)
		t_env.cash += revenue
		t_env.inventory -= 1
		if t_env.inventory == 0 {
			profit := (price - t_env.entry_price) / t_env.entry_price
			reward += profit * 0.1
			info = fmt.tprintf("Sold, PnL=%.4f", profit)
		}
	}

	// 3. Calculate portfolio value AFTER action
	curr_value := t_env.cash + f64(t_env.inventory) * price

	// 4. Reward is the normalized change in portfolio value
	reward += (curr_value - prev_value) / t_env.initial_cash

	// 5. Tiny penalty for holding
	reward -= math.abs(f64(t_env.inventory)) * 0.00001

	// 6. Build observation for the NEXT step
	obs_dim := t_env.window * (2 + t_env.n_indicators)
	obs := make([]f64, obs_dim, t_env.allocator)
	idx := 0
	// +1 because current_step was already incremented by env_step
	start := max(0, t_env.env.current_step - t_env.window + 1)
	for w in 0 ..< t_env.window {
		idx_in_data := start + w
		if idx_in_data < len(t_env.prices) {
			obs[idx] = t_env.prices[idx_in_data]
		}
		idx += 1
	}
	for w in 0 ..< t_env.window {
		idx_in_data := start + w
		if idx_in_data < len(t_env.volumes) {
			obs[idx] = t_env.volumes[idx_in_data]
		}
		idx += 1
	}
	for w in 0 ..< t_env.window {
		idx_in_data := (start + w) * t_env.n_indicators
		if idx_in_data < len(t_env.indicators) {
			obs[idx] = t_env.indicators[idx_in_data]
		}
		idx += 1
	}

	return Step {
		observation = Observation{data = obs, shape = [4]int{1, obs_dim, 1, 1}},
		reward = reward,
		done = t_env.env.done,
		info = info,
	}
}

new_trading_env :: proc(
	prices: []f64,
	volumes: []f64,
	indicators: []f64,
	n_indicators: int,
	window: int,
	transaction_fee: f64 = 0.001,
	max_position: i32 = 10,
	alloc: mem.Allocator = context.allocator,
) -> ^TradingEnv {
	env := new(TradingEnv, alloc)
	env.prices = prices
	env.volumes = volumes
	env.indicators = indicators
	env.n_indicators = n_indicators
	env.window = window
	env.transaction_fee = transaction_fee
	env.max_position = max_position
	env.env.action_space = 3
	env.env.obs_dim = window * (2 + n_indicators)
	env.env.max_steps = len(prices)
	env.initial_cash = 100000.0
	env.env.reset_fn = trading_env_reset
	env.env.step_fn = trading_env_step
	env.allocator = alloc
	return env
}

trading_env_free :: proc(env: ^TradingEnv) {
	free(env, env.allocator)
}

// RolloutBuffer stores experience for PPO updates
RolloutBuffer :: struct {
	states:        []^tensor.Tensor,
	actions:       []int,
	old_log_probs: []f64,
	rewards:       []f64,
	values:        []f64,
	dones:         []bool,
	capacity:      int,
	size:          int,
	allocator:     mem.Allocator,
}

new_rollout_buffer :: proc(capacity: int, alloc: mem.Allocator) -> ^RolloutBuffer {
	buf := new(RolloutBuffer, alloc)
	buf.capacity = capacity
	buf.states = make([]^tensor.Tensor, capacity, alloc)
	buf.actions = make([]int, capacity, alloc)
	buf.old_log_probs = make([]f64, capacity, alloc)
	buf.rewards = make([]f64, capacity, alloc)
	buf.values = make([]f64, capacity, alloc)
	buf.dones = make([]bool, capacity, alloc)
	buf.size = 0
	buf.allocator = alloc
	return buf
}

rollout_buffer_add :: proc(
	buf: ^RolloutBuffer,
	state: ^tensor.Tensor,
	action: int,
	log_prob: f64,
	reward: f64,
	value: f64,
	done: bool,
) {
	if buf.size >= buf.capacity {
		for i in 0 ..< buf.capacity {
			if buf.states[i] != nil {
				tensor.tensor_free(buf.states[i])
			}
		}
		buf.size = 0
	}
	buf.states[buf.size] = state
	buf.actions[buf.size] = action
	buf.old_log_probs[buf.size] = log_prob
	buf.rewards[buf.size] = reward
	buf.values[buf.size] = value
	buf.dones[buf.size] = done
	buf.size += 1
}

rollout_buffer_clear :: proc(buf: ^RolloutBuffer) {
	for i in 0 ..< buf.size {
		if buf.states[i] != nil {
			tensor.tensor_free(buf.states[i])
		}
	}
	buf.size = 0
}

rollout_buffer_free :: proc(buf: ^RolloutBuffer) {
	rollout_buffer_clear(buf)
	delete(buf.states, buf.allocator)
	delete(buf.actions, buf.allocator)
	delete(buf.old_log_probs, buf.allocator)
	delete(buf.rewards, buf.allocator)
	delete(buf.values, buf.allocator)
	delete(buf.dones, buf.allocator)
	free(buf, buf.allocator)
}

// PPOAgent implements Proximal Policy Optimization
PPOAgent :: struct {
	actor:        ^nn.Sequential,
	critic:       ^nn.Sequential,
	optimizer:    nn.Adam, // ✅ FIXED: Use nn.Adam directly
	gamma:        f64,
	gae_lambda:   f64,
	clip_epsilon: f64,
	value_coef:   f64,
	entropy_coef: f64,
	action_space: int,
	obs_dim:      int,
	buffer:       ^RolloutBuffer,
	allocator:    mem.Allocator,
}
PPOUpdateStats :: struct {
	policy_loss: f64,
	value_loss:  f64,
	entropy:     f64,
	total_loss:  f64,
	n_updates:   int,
}
new_ppo_agent :: proc(
	obs_dim: int,
	action_space: int,
	hidden_dim: int = 64,
	gamma: f64 = 0.99,
	gae_lambda: f64 = 0.95,
	clip_epsilon: f64 = 0.2,
	value_coef: f64 = 0.5,
	entropy_coef: f64 = 0.01,
	lr: f64 = 3e-4,
	alloc: mem.Allocator = context.allocator,
) -> ^PPOAgent {
	agent := new(PPOAgent, alloc)
	agent.gamma = gamma
	agent.gae_lambda = gae_lambda
	agent.clip_epsilon = clip_epsilon
	agent.value_coef = value_coef
	agent.entropy_coef = entropy_coef
	agent.action_space = action_space
	agent.obs_dim = obs_dim
	agent.allocator = alloc

	agent.actor = nn.sequential_new(alloc)
	nn.sequential_add(agent.actor, nn.linear_layer_new(obs_dim, hidden_dim, alloc))
	nn.sequential_add(agent.actor, nn.Activation.ReLU)
	nn.sequential_add(agent.actor, nn.linear_layer_new(hidden_dim, action_space, alloc))

	agent.critic = nn.sequential_new(alloc)
	nn.sequential_add(agent.critic, nn.linear_layer_new(obs_dim, hidden_dim, alloc))
	nn.sequential_add(agent.critic, nn.Activation.ReLU)
	nn.sequential_add(agent.critic, nn.linear_layer_new(hidden_dim, 1, alloc))

	// ✅ FIXED: Use nn.adam_new
	agent.optimizer = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)

	nn.sequential_add_to_adam(agent.actor, &agent.optimizer)
	nn.sequential_add_to_adam(agent.critic, &agent.optimizer)

	agent.buffer = new_rollout_buffer(2048, alloc)

	return agent
}

ppo_agent_select_action :: proc(
	agent: ^PPOAgent,
	state: ^tensor.Tensor,
) -> (
	action: int,
	log_prob: f64,
	value: f64,
) {
	logits := nn.sequential_forward(agent.actor, state)
	probs := tensor_softmax(logits)
	action = tensor_categorical_sample(probs)
	log_prob = tensor_categorical_log_prob(probs, action)

	val_tensor := nn.sequential_forward(agent.critic, state)
	value = val_tensor.data.data[0]

	tensor.tensor_free_graph(logits)
	tensor.tensor_free_graph(probs)
	tensor.tensor_free_graph(val_tensor)

	return action, log_prob, value
}
compute_gae :: proc(
	rewards: []f64,
	values: []f64,
	dones: []bool,
	gamma: f64,
	gae_lambda: f64,
	alloc: mem.Allocator,
) -> []f64 {
	advantages := make([]f64, len(rewards), alloc)
	last_gae := 0.0

	for t := len(rewards) - 1; t >= 0; t -= 1 {
		if dones[t] {
			delta := rewards[t] - values[t]
			last_gae = delta
		} else {
			// ✅ FIXED: Safely check bounds before accessing t + 1
			next_value := 0.0
			if t + 1 < len(values) {
				next_value = values[t + 1]
			}

			delta := rewards[t] + gamma * next_value - values[t]
			last_gae = delta + gamma * gae_lambda * last_gae
		}
		advantages[t] = last_gae
	}

	return advantages
}

// ===== ./wotan/nn/ppo.odin — replace ppo_agent_update =====

ppo_agent_update :: proc(
	agent: ^PPOAgent,
	epochs: int = 10,
	batch_size: int = 64,
) -> PPOUpdateStats {
	stats: PPOUpdateStats
	if agent.buffer.size == 0 {return stats}

	advantages := compute_gae(
		agent.buffer.rewards[:agent.buffer.size],
		agent.buffer.values[:agent.buffer.size],
		agent.buffer.dones[:agent.buffer.size],
		agent.gamma,
		agent.gae_lambda,
		agent.allocator,
	)
	defer delete(advantages, agent.allocator)

	adv_mean := 0.0
	adv_var := 0.0
	for a in advantages {
		adv_mean += a
		adv_var += a * a
	}
	adv_mean /= f64(len(advantages))
	adv_var = adv_var / f64(len(advantages)) - adv_mean * adv_mean
	adv_std := math.sqrt_f64(adv_var + 1e-8)
	for i in 0 ..< len(advantages) {
		advantages[i] = (advantages[i] - adv_mean) / adv_std
	}

	returns := make([]f64, len(advantages), agent.allocator)
	defer delete(returns, agent.allocator)
	for i in 0 ..< len(advantages) {
		returns[i] = advantages[i] + agent.buffer.values[i]
	}

	for _ in 0 ..< epochs {
		indices := make([]int, agent.buffer.size, agent.allocator)
		for i in 0 ..< agent.buffer.size {indices[i] = i}
		for i := agent.buffer.size - 1; i > 0; i -= 1 {
			j := rand.int_range(0, i)
			indices[i], indices[j] = indices[j], indices[i]
		}

		for start := 0; start < agent.buffer.size; start += batch_size {
			end := min(start + batch_size, agent.buffer.size)
			b_len := end - start

			nn.adam_zero_grad(&agent.optimizer)

			total_policy_loss := tensor.tensor_new(
				l.matrix_new(f64, 1, 1, agent.allocator),
				true,
				agent.allocator,
			)
			total_policy_loss.data.data[0] = 0.0

			total_value_loss := tensor.tensor_new(
				l.matrix_new(f64, 1, 1, agent.allocator),
				true,
				agent.allocator,
			)
			total_value_loss.data.data[0] = 0.0

			total_entropy := tensor.tensor_new(
				l.matrix_new(f64, 1, 1, agent.allocator),
				true,
				agent.allocator,
			)
			total_entropy.data.data[0] = 0.0

			for i in 0 ..< b_len {
				idx := indices[start + i]
				state := agent.buffer.states[idx]
				action := agent.buffer.actions[idx]
				old_log_prob := agent.buffer.old_log_probs[idx]
				advantage := advantages[idx]
				target_return := returns[idx]

				// ── Actor forward ──
				logits := nn.sequential_forward(agent.actor, state)
				probs := tensor_softmax(logits)

				// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
				// ✅ FIX: Extract action prob via one-hot mask
				//   (keeps the autograd graph intact)
				// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
				one_hot_data := l.matrix_new(
					f64,
					probs.data.rows,
					probs.data.cols,
					agent.allocator,
				)
				one_hot_data.data[action] = 1.0
				one_hot := tensor.tensor_new(one_hot_data, false, agent.allocator)
				one_hot.owned_by_graph = true

				masked_probs := tensor.tensor_mul(probs, one_hot)
				action_prob := tensor.tensor_sum(masked_probs)
				// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

				new_log_prob_tensor := tensor.tensor_log(action_prob)
				new_log_prob := new_log_prob_tensor.data.data[0]

				ratio := math.exp_f64(new_log_prob - old_log_prob)

				surr1 := ratio * advantage
				clamped_ratio := math.max(
					1.0 - agent.clip_epsilon,
					math.min(1.0 + agent.clip_epsilon, ratio),
				)
				surr2 := clamped_ratio * advantage
				min_surr := math.min(surr1, surr2)

				policy_loss_scalar := -min_surr

				policy_loss_tensor := tensor.tensor_scale(action_prob, policy_loss_scalar)
				policy_loss_sum := tensor.tensor_sum(policy_loss_tensor)
				total_policy_loss = tensor.tensor_add(total_policy_loss, policy_loss_sum)

				// ── Critic forward ──
				value_tensor := nn.sequential_forward(agent.critic, state)

				target_data := l.matrix_new(f64, 1, 1, agent.allocator)
				target_data.data[0] = target_return
				target_tensor := tensor.tensor_new(target_data, false, agent.allocator)
				target_tensor.owned_by_graph = true

				value_loss_tensor := tensor.tensor_mse_loss(value_tensor, target_tensor)
				total_value_loss = tensor.tensor_add(total_value_loss, value_loss_tensor)

				// ── Entropy ──
				entropy_val := 0.0
				for j in 0 ..< len(probs.data.data) {
					p := probs.data.data[j]
					if p > 1e-10 {
						entropy_val += -p * math.ln_f64(p)
					}
				}
				entropy_tensor := tensor.tensor_new(
					l.matrix_new(f64, 1, 1, agent.allocator),
					true,
					agent.allocator,
				)
				entropy_tensor.data.data[0] = entropy_val
				entropy_scaled := tensor.tensor_scale(probs, entropy_val)
				entropy_sum := tensor.tensor_sum(entropy_scaled)
				total_entropy = tensor.tensor_add(total_entropy, entropy_sum)

				stats.policy_loss += policy_loss_scalar
				stats.value_loss += value_loss_tensor.data.data[0]
				stats.entropy += entropy_val
				stats.total_loss +=
					policy_loss_scalar +
					agent.value_coef * value_loss_tensor.data.data[0] -
					agent.entropy_coef * entropy_val
				stats.n_updates += 1
			}

			batch_len_f := f64(b_len)
			inv_batch := tensor.tensor_new(
				l.matrix_new(f64, 1, 1, agent.allocator),
				false,
				agent.allocator,
			)
			inv_batch.data.data[0] = 1.0 / batch_len_f
			inv_batch.owned_by_graph = true

			total_policy_loss = tensor.tensor_mul(total_policy_loss, inv_batch)
			total_value_loss = tensor.tensor_mul(total_value_loss, inv_batch)
			total_entropy = tensor.tensor_mul(total_entropy, inv_batch)

			value_coef_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, 1, agent.allocator),
				false,
				agent.allocator,
			)
			value_coef_tensor.data.data[0] = agent.value_coef
			value_coef_tensor.owned_by_graph = true

			entropy_coef_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, 1, agent.allocator),
				false,
				agent.allocator,
			)
			entropy_coef_tensor.data.data[0] = agent.entropy_coef
			entropy_coef_tensor.owned_by_graph = true

			value_scaled := tensor.tensor_mul(total_value_loss, value_coef_tensor)
			entropy_scaled := tensor.tensor_mul(total_entropy, entropy_coef_tensor)

			total_loss := tensor.tensor_add(total_policy_loss, value_scaled)
			total_loss = tensor.tensor_sub(total_loss, entropy_scaled)

			tensor.tensor_backward(total_loss)
			nn.adam_step(&agent.optimizer)
			tensor.tensor_free_graph(total_loss)

			delete(indices, agent.allocator)
		}
	}

	rollout_buffer_clear(agent.buffer)

	if stats.n_updates > 0 {
		stats.policy_loss /= f64(stats.n_updates)
		stats.value_loss /= f64(stats.n_updates)
		stats.entropy /= f64(stats.n_updates)
		stats.total_loss /= f64(stats.n_updates)
	}
	return stats
}
ppo_agent_free :: proc(agent: ^PPOAgent) {
	nn.sequential_free(agent.actor)
	nn.sequential_free(agent.critic)
	nn.adam_free(&agent.optimizer) // ✅ FIXED
	rollout_buffer_free(agent.buffer)
	free(agent, agent.allocator)
}

tensor_softmax :: proc(logits: ^tensor.Tensor) -> ^tensor.Tensor {
	max_val := -math.F64_MAX
	for v in logits.data.data {
		if v > max_val {max_val = v}
	}
	exp_sum := 0.0
	n := len(logits.data.data)
	exps := make([]f64, n, context.allocator)

	// ✅ FIXED: Use explicit index loop to avoid type inference issues
	for i in 0 ..< n {
		v := logits.data.data[i]
		exps[i] = math.exp_f64(v - max_val)
		exp_sum += exps[i]
	}

	out_data := l.matrix_new(f64, logits.data.rows, logits.data.cols, logits.allocator)
	for i in 0 ..< n {
		out_data.data[i] = exps[i] / exp_sum
	}
	delete(exps, context.allocator)
	out := tensor.tensor_new(out_data, logits.requires_grad, logits.allocator)
	out.shape = logits.shape
	return out
}

tensor_categorical_sample :: proc(probs: ^tensor.Tensor) -> int {
	r := rand.float64()
	cum_prob := 0.0
	for i in 0 ..< len(probs.data.data) {
		p := probs.data.data[i]
		cum_prob += p
		if r < cum_prob {
			return i
		}
	}
	return len(probs.data.data) - 1
}

tensor_categorical_log_prob :: proc(probs: ^tensor.Tensor, action: int) -> f64 {
	if action < 0 || action >= len(probs.data.data) {
		return -math.F64_MAX
	}
	p := probs.data.data[action]
	if p <= 1e-10 {
		return -30.0
	}
	return math.ln_f64(p)
}
