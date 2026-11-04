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
	max_steps:    int, // the episode ends when current_step reaches it
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

// ============================================================================
// Single-asset trading environment
// ============================================================================
// Actions: 0 hold, 1 buy (10 shares if cash allows, else 1), 2 sell (up to 10 of the
// held shares; long only), at most one trade per 5 steps, up to max_position shares.
// Step k trades at prices[k] after observing the window ending at prices[k - 1], then
// marks the portfolio to prices[k]. The episode starts with the window prices[0 ..
// window - 1] and ends at the last price, where any position is sold (with the fee).
// Observation: window values each of price / first price - 1, volume / first volume - 1
// and every indicator (indicators is row-major [day, n_indicators]), 2 + n_indicators
// blocks of `window`.
// Reward (shaped): change of portfolio value / initial cash, + 0.1 x the return of a
// position when it is closed, - 2 x fee for any non-hold action, - 1e-5 per share held,
// - 2 x the drawdown from the peak value.
TradingEnv :: struct {
	env:             Environment,
	prices:          []f64,
	volumes:         []f64,
	indicators:      []f64,
	n_indicators:    int,
	window:          int,
	cash:            f64,
	peak_cash:       f64, // peak portfolio value (for the drawdown penalty)
	initial_cash:    f64,
	inventory:       i32,
	entry_price:     f64, // average cost of the position
	cooldown:        int,
	transaction_fee: f64,
	max_position:    i32,
	allocator:       mem.Allocator,
}

// Observation of the window of `window` days ending at day `last`
@(private = "file")
_trading_obs :: proc(t_env: ^TradingEnv, last: int) -> Observation {
	W, K := t_env.window, t_env.n_indicators
	obs := make([]f64, W * (2 + K), t_env.allocator)
	start := max(0, last - W + 1)
	base_price := t_env.prices[start] if t_env.prices[start] != 0 else 1.0
	base_vol := t_env.volumes[start] if t_env.volumes[start] != 0 else 1.0
	for w in 0 ..< W {
		d := start + w
		if d >= len(t_env.prices) {break}
		obs[w] = t_env.prices[d] / base_price - 1.0
		obs[W + w] = t_env.volumes[d] / base_vol - 1.0
		for k in 0 ..< K {
			if d * K + k <
			   len(t_env.indicators) {obs[(2 + k) * W + w] = t_env.indicators[d * K + k]}
		}
	}
	return Observation{data = obs, shape = [4]int{1, len(obs), 1, 1}}
}

trading_env_reset :: proc(env: ^Environment) -> Observation {
	t_env := cast(^TradingEnv)env
	t_env.initial_cash = 100000.0
	t_env.cash = 100000.0
	t_env.peak_cash = 100000.0
	t_env.inventory = 0
	t_env.entry_price = 0.0
	t_env.cooldown = 0
	t_env.env.current_step = t_env.window - 1 // the first step trades at prices[window]
	t_env.env.done = false
	return _trading_obs(t_env, t_env.env.current_step)
}

trading_env_step :: proc(env: ^Environment, action: int) -> Step {
	t_env := cast(^TradingEnv)env
	k := t_env.env.current_step
	if k >= len(t_env.prices) {
		// called after the end: nothing happens
		return Step {
			observation = Observation {
				data = make([]f64, t_env.env.obs_dim, t_env.allocator),
				shape = [4]int{1, t_env.env.obs_dim, 1, 1},
			},
			done = true,
			info = "Already done",
		}
	}
	t_env.cooldown += 1

	prev_value := t_env.cash + f64(t_env.inventory) * t_env.prices[k - 1]
	price := t_env.prices[k]
	reward := 0.0
	info := ""

	if action == 1 && t_env.inventory < t_env.max_position && t_env.cooldown >= 5 {
		cost := price * (1.0 + t_env.transaction_fee)
		trade_size: i32 = 10 if t_env.cash > cost * 10 else 1
		trade_size = min(trade_size, t_env.max_position - t_env.inventory)
		if t_env.cash >= cost * f64(trade_size) {
			t_env.cash -= cost * f64(trade_size)
			held := f64(t_env.inventory)
			t_env.entry_price =
				(t_env.entry_price * held + price * f64(trade_size)) / (held + f64(trade_size))
			t_env.inventory += trade_size
			info = fmt.tprintf("Bought %d", trade_size)
			t_env.cooldown = 0
		}
	} else if action == 2 && t_env.cooldown >= 5 {
		trade_size := min(t_env.inventory, 10)
		if trade_size > 0 {
			t_env.cash += price * (1.0 - t_env.transaction_fee) * f64(trade_size)
			t_env.inventory -= trade_size
			if t_env.inventory == 0 {
				profit := (price - t_env.entry_price) / t_env.entry_price
				reward += profit * 0.1
				info = fmt.tprintf("Sold %d, PnL=%.4f", trade_size, profit)
				t_env.entry_price = 0
			}
			t_env.cooldown = 0
		}
	}

	// last price: close the position (the fee is part of the value change below)
	if t_env.env.done && t_env.inventory != 0 {
		t_env.cash += price * (1.0 - t_env.transaction_fee) * f64(t_env.inventory)
		t_env.inventory = 0
		info = "Liquidated at end"
	}

	curr_value := t_env.cash + f64(t_env.inventory) * price
	t_env.peak_cash = max(t_env.peak_cash, curr_value)
	drawdown := (t_env.peak_cash - curr_value) / t_env.peak_cash if t_env.peak_cash > 0 else 0.0

	reward += (curr_value - prev_value) / t_env.initial_cash // includes the fees paid
	if action != 0 {reward -= t_env.transaction_fee * 2.0} 	// discourages churn
	reward -= math.abs(f64(t_env.inventory)) * 0.00001
	reward -= 2.0 * drawdown

	return Step {
		observation = _trading_obs(t_env, k),
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
	assert(
		window >= 1 && len(prices) > window && len(volumes) == len(prices),
		"new_trading_env: need more prices than the window",
	)
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
	env.env.max_steps = len(prices) - 1 // the last step trades at the last price
	env.initial_cash = 100000.0
	env.env.reset_fn = trading_env_reset
	env.env.step_fn = trading_env_step
	env.allocator = alloc
	return env
}

trading_env_free :: proc(env: ^TradingEnv) {
	free(env, env.allocator)
}

// ============================================================================
// Rollout buffer
// ============================================================================
// Transitions in order; dones[t] marks the last transition of an episode. Grows when
// full (no data is dropped). The buffer owns the state tensors.
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
	buf.capacity = max(capacity, 1)
	buf.states = make([]^tensor.Tensor, buf.capacity, alloc)
	buf.actions = make([]int, buf.capacity, alloc)
	buf.old_log_probs = make([]f64, buf.capacity, alloc)
	buf.rewards = make([]f64, buf.capacity, alloc)
	buf.values = make([]f64, buf.capacity, alloc)
	buf.dones = make([]bool, buf.capacity, alloc)
	buf.allocator = alloc
	return buf
}

@(private = "file")
_grow :: proc(s: ^[]$T, n: int, alloc: mem.Allocator) {
	g := make([]T, n, alloc)
	copy(g, s^)
	delete(s^, alloc)
	s^ = g
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
		n := 2 * buf.capacity
		_grow(&buf.states, n, buf.allocator)
		_grow(&buf.actions, n, buf.allocator)
		_grow(&buf.old_log_probs, n, buf.allocator)
		_grow(&buf.rewards, n, buf.allocator)
		_grow(&buf.values, n, buf.allocator)
		_grow(&buf.dones, n, buf.allocator)
		buf.capacity = n
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
			buf.states[i] = nil
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

// ============================================================================
// PPO (Schulman et al. 2017, clipped surrogate; conventions as CleanRL / SB3)
// ============================================================================
PPOAgent :: struct {
	actor:        ^nn.Sequential,
	critic:       ^nn.Sequential,
	optimizer:    nn.Adam,
	gamma:        f64,
	gae_lambda:   f64,
	clip_epsilon: f64,
	value_coef:   f64,
	entropy_coef: f64,
	action_space: int,
	obs_dim:      int,
	buffer:       ^RolloutBuffer,
	allocator:    mem.Allocator,
	// One arena per action selection / minibatch update, reused across them
	scope:        tensor.Graph_Scope,
}

PPOUpdateStats :: struct {
	policy_loss: f64, // mean of -min(r A, clip(r) A)
	value_loss:  f64, // mean of (V - R)^2
	entropy:     f64,
	total_loss:  f64, // policy + value_coef value - entropy_coef entropy
	n_updates:   int, // samples summed over (epochs x buffer size)
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
	tensor.graph_begin(&agent.scope, no_grad = true)
	defer tensor.graph_end(&agent.scope)

	probs := tensor.tensor_softmax(nn.sequential_forward(agent.actor, state))
	action = tensor_categorical_sample(probs)
	log_prob = tensor_categorical_log_prob(probs, action)
	value = nn.sequential_forward(agent.critic, state).data.data[0]
	return action, log_prob, value
}

// Critic value V(state), e.g. of the state after the last transition, to bootstrap
// ppo_agent_update when the rollout ends inside an episode
ppo_agent_value :: proc(agent: ^PPOAgent, state: ^tensor.Tensor) -> f64 {
	tensor.graph_begin(&agent.scope, no_grad = true)
	defer tensor.graph_end(&agent.scope)
	return nn.sequential_forward(agent.critic, state).data.data[0]
}

// Generalised advantage estimation (Schulman et al. 2016), as CleanRL:
//   delta_t = r_t + gamma V_{t+1} (1 - done_t) - V_t,  A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}
// V_{t+1} after the last transition is last_value (the value of the state the rollout
// stopped in; without it V of the last state is used, an approximation).
compute_gae :: proc(
	rewards: []f64,
	values: []f64,
	dones: []bool,
	gamma: f64,
	gae_lambda: f64,
	alloc: mem.Allocator,
	last_value: Maybe(f64) = nil,
) -> []f64 {
	n := len(rewards)
	advantages := make([]f64, n, alloc)
	if n == 0 {return advantages}
	bootstrap, ok := last_value.?
	if !ok {bootstrap = values[n - 1]}
	gae := 0.0
	for t := n - 1; t >= 0; t -= 1 {
		not_done := 0.0 if dones[t] else 1.0
		next_value := values[t + 1] if t + 1 < n else bootstrap
		delta := rewards[t] + gamma * next_value * not_done - values[t]
		gae = delta + gamma * gae_lambda * not_done * gae
		advantages[t] = gae
	}
	return advantages
}

// Loss of one minibatch (buffer rows `rows`), for the advantages (already normalised)
// and returns of the whole buffer:
//   L = mean[-min(r A, clip(r, 1 - eps, 1 + eps) A)] + value_coef mean[(V - R)^2]
//       - entropy_coef mean[H(pi)],   r = pi(a) / pi_old(a)
// The returned tensor has the gradient of L (the surrogate's gradient is A r grad log pi
// where the unclipped term is the minimum, else 0, applied through the cross entropy
// -log pi(a)); `objective` is the value of L. Build it inside a graph scope.
ppo_minibatch_loss :: proc(
	agent: ^PPOAgent,
	rows: []int,
	advantages, returns: []f64,
) -> (
	loss: ^tensor.Tensor,
	objective: f64,
	stats: PPOUpdateStats,
) {
	ga := tensor._scope_or(agent.allocator)
	total_policy := tensor.tensor_new(l.matrix_new(f64, 1, 1, ga), true, ga)
	total_value := tensor.tensor_new(l.matrix_new(f64, 1, 1, ga), true, ga)
	total_entropy := tensor.tensor_new(l.matrix_new(f64, 1, 1, ga), true, ga)

	for idx in rows {
		state := agent.buffer.states[idx]
		action := agent.buffer.actions[idx]
		advantage := advantages[idx]

		logits := nn.sequential_forward(agent.actor, state)
		probs := tensor.tensor_softmax(logits)
		ce_loss := tensor.tensor_cross_entropy_loss(logits, []int{action}) // -log pi(a)

		ratio := math.exp(math.ln(probs.data.data[action]) - agent.buffer.old_log_probs[idx])
		surr1 := ratio * advantage
		surr2 := clamp(ratio, 1.0 - agent.clip_epsilon, 1.0 + agent.clip_epsilon) * advantage
		scale := advantage * ratio if surr1 <= surr2 else 0.0
		total_policy = tensor.tensor_add(total_policy, tensor.tensor_scale(ce_loss, scale))

		target_data := l.matrix_new(f64, 1, 1, ga)
		target_data.data[0] = returns[idx]
		value_loss := tensor.tensor_mse_loss(
			nn.sequential_forward(agent.critic, state),
			tensor.tensor_new(target_data, false, ga),
		)
		total_value = tensor.tensor_add(total_value, value_loss)

		entropy := tensor.tensor_entropy(probs)
		total_entropy = tensor.tensor_add(total_entropy, entropy)

		pl := -math.min(surr1, surr2)
		stats.policy_loss += pl
		stats.value_loss += value_loss.data.data[0]
		stats.entropy += entropy.data.data[0]
		stats.n_updates += 1
	}
	inv := 1.0 / f64(len(rows))
	loss = tensor.tensor_add(
		tensor.tensor_add(
			tensor.tensor_scale(total_policy, inv),
			tensor.tensor_scale(total_value, agent.value_coef * inv),
		),
		tensor.tensor_scale(total_entropy, -agent.entropy_coef * inv),
	)
	stats.total_loss =
		stats.policy_loss +
		agent.value_coef * stats.value_loss -
		agent.entropy_coef * stats.entropy
	objective = stats.total_loss * inv
	return
}

// PPO update on the buffer: GAE advantages (raw advantages + values are the critic's
// targets; the advantages are normalised for the policy loss), then `epochs` passes over
// shuffled minibatches of batch_size, one Adam step each. Clears the buffer.
// last_value: V of the state after the last transition when the rollout stopped inside
// an episode (ppo_agent_value); without it V of the last state is used.
ppo_agent_update :: proc(
	agent: ^PPOAgent,
	epochs: int = 10,
	batch_size: int = 64,
	last_value: Maybe(f64) = nil,
) -> PPOUpdateStats {
	stats: PPOUpdateStats
	n := agent.buffer.size
	if n == 0 {return stats}

	advantages := compute_gae(
		agent.buffer.rewards[:n],
		agent.buffer.values[:n],
		agent.buffer.dones[:n],
		agent.gamma,
		agent.gae_lambda,
		agent.allocator,
		last_value,
	)
	defer delete(advantages, agent.allocator)
	returns := make([]f64, n, agent.allocator)
	defer delete(returns, agent.allocator)
	for i in 0 ..< n {returns[i] = advantages[i] + agent.buffer.values[i]}

	adv_mean, adv_var := 0.0, 0.0
	for a in advantages {adv_mean += a}
	adv_mean /= f64(n)
	for a in advantages {adv_var += (a - adv_mean) * (a - adv_mean)}
	adv_std := math.sqrt(adv_var / f64(n)) + 1e-8
	for &a in advantages {a = (a - adv_mean) / adv_std}

	indices := make([]int, n, agent.allocator)
	defer delete(indices, agent.allocator)
	for _ in 0 ..< epochs {
		for i in 0 ..< n {indices[i] = i}
		for i := n - 1; i > 0; i -= 1 {
			j := rand.int_max(i + 1) // Fisher-Yates: j uniform in [0, i]
			indices[i], indices[j] = indices[j], indices[i]
		}
		for start := 0; start < n; start += batch_size {
			rows := indices[start:min(start + batch_size, n)]
			nn.adam_zero_grad(&agent.optimizer)
			tensor.graph_begin(&agent.scope)
			loss, _, s := ppo_minibatch_loss(agent, rows, advantages, returns)
			tensor.tensor_backward(loss)
			tensor.graph_end(&agent.scope)
			nn.adam_step(&agent.optimizer)
			stats.policy_loss += s.policy_loss
			stats.value_loss += s.value_loss
			stats.entropy += s.entropy
			stats.total_loss += s.total_loss
			stats.n_updates += s.n_updates
		}
	}
	rollout_buffer_clear(agent.buffer)

	if stats.n_updates > 0 {
		k := f64(stats.n_updates)
		stats.policy_loss /= k
		stats.value_loss /= k
		stats.entropy /= k
		stats.total_loss /= k
	}
	return stats
}

ppo_agent_free :: proc(agent: ^PPOAgent) {
	nn.sequential_free(agent.actor)
	nn.sequential_free(agent.critic)
	nn.adam_free(&agent.optimizer)
	rollout_buffer_free(agent.buffer)
	tensor.graph_scope_destroy(&agent.scope)
	free(agent, agent.allocator)
}

tensor_categorical_sample :: proc(probs: ^tensor.Tensor) -> int {
	r := rand.float64()
	cum_prob := 0.0
	for i in 0 ..< len(probs.data.data) {
		cum_prob += probs.data.data[i]
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
