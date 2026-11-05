package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Conservative Q-learning (Kumar et al. 2020, CQL(H)) for discrete actions
// ============================================================================
// Offline Q-learning from a fixed log of transitions (s, a, r, s', done):
//   loss = mean_i (Q(s_i, a_i) - y_i)^2 + alpha * mean_i (logsumexp_a Q(s_i, a) - Q(s_i, a_i))
//   y_i  = r_i + gamma * (1 - done_i) * max_a Q_target(s'_i, a)   (a constant)
// The regulariser pushes down the Q-values of actions the log rarely takes and up those it
// takes, so the greedy policy does not exploit over-estimated, unseen actions. The paper
// weights the TD term by 1/2, so alpha here is twice the paper's alpha.

CQLConfig :: struct {
	state_dim:   int,
	num_actions: int,
	hidden_dim:  int,
	gamma:       f64, // discount factor
	alpha:       f64, // CQL regularisation weight
	tau:         f64, // soft target update rate
}

CQLNetwork :: struct {
	fc1: nn.LinearLayer,
	fc2: nn.LinearLayer,
}

cql_network_new :: proc(
	config: CQLConfig,
	allocator: mem.Allocator = context.allocator,
) -> CQLNetwork {
	net: CQLNetwork
	net.fc1 = nn.linear_layer_new(config.state_dim, config.hidden_dim, allocator)
	net.fc2 = nn.linear_layer_new(config.hidden_dim, config.num_actions, allocator)
	return net
}

cql_network_free :: proc(net: ^CQLNetwork) {
	nn.linear_layer_free(&net.fc1)
	nn.linear_layer_free(&net.fc2)
}

// Q-values [batch, num_actions] of states [batch, state_dim]
cql_network_forward :: proc(net: ^CQLNetwork, states: ^t.Tensor) -> ^t.Tensor {
	h := nn.linear_forward(&net.fc1, states)
	h = t.tensor_relu(h)
	return nn.linear_forward(&net.fc2, h)
}

// The weights and biases (fc1, fc2), in the same order for any two networks
@(private = "file")
_cql_params :: proc(net: ^CQLNetwork) -> [4]^t.Tensor {
	return {net.fc1.weights, net.fc1.bias, net.fc2.weights, net.fc2.bias}
}

// Registers the network's weights and biases with Adam
cql_network_add_to_adam :: proc(net: ^CQLNetwork, opt: ^nn.Adam) {
	for p in _cql_params(net) {if p != nil {nn.adam_add_param(opt, p)}}
}

// cql_network_copy performs a hard copy of weights from src to dst
cql_network_copy :: proc(src: ^CQLNetwork, dst: ^CQLNetwork) {
	cql_soft_update(src, dst, 1)
}

// cql_soft_update performs: target = tau * main + (1 - tau) * target
cql_soft_update :: proc(main_net: ^CQLNetwork, target_net: ^CQLNetwork, tau: f64) {
	src, dst := _cql_params(main_net), _cql_params(target_net)
	for k in 0 ..< len(src) {
		if src[k] == nil || dst[k] == nil {continue}
		for &v, i in dst[k].data.data {v = tau * src[k].data.data[i] + (1 - tau) * v}
	}
}

// The greedy action argmax_a Q(state, a) (ties: the lowest index); no graph is kept
cql_greedy_action :: proc(net: ^CQLNetwork, state: []f64) -> int {
	scope: t.Graph_Scope
	t.graph_begin(&scope, no_grad = true)
	defer {t.graph_end(&scope); t.graph_scope_destroy(&scope)}
	alloc := t._scope_or(context.allocator)
	x := l.matrix_new(f64, 1, len(state), alloc)
	copy(x.data, state)
	q := cql_network_forward(net, t.tensor_new(x, false, alloc))
	best := 0
	for v, a in q.data.data {if v > q.data.data[best] {best = a}}
	return best
}

// ============================================================================
// CQL Loss
// ============================================================================
// q_values: Q(s, .) [batch, num_actions] (graph); next_q_values: Q_target(s', .), only read
// (no gradient flows into the target); rewards, dones: one entry per action. Returns the
// total loss and the TD and regulariser values.
cql_loss :: proc(
	q_values: ^t.Tensor,
	actions: []int,
	rewards: ^t.Tensor,
	next_q_values: ^t.Tensor,
	dones: ^t.Tensor,
	config: CQLConfig,
	allocator: mem.Allocator,
) -> (
	^t.Tensor,
	f64,
	f64,
) {
	allocator := t._scope_or(allocator) // the active graph scope's arena, if any
	A := config.num_actions
	batch := len(actions)
	assert(
		len(q_values.data.data) == batch * A && len(next_q_values.data.data) == batch * A,
		"cql_loss: q_values and next_q_values need batch x num_actions entries",
	)
	assert(
		len(rewards.data.data) == batch && len(dones.data.data) == batch,
		"cql_loss: one reward and done flag per action",
	)

	// TD target r + gamma (1 - done) max_a Q_target(s', a), a constant
	target_data := l.matrix_new(f64, batch, 1, allocator)
	for i in 0 ..< batch {
		max_nq := next_q_values.data.data[i * A]
		for a in 1 ..< A {max_nq = max(max_nq, next_q_values.data.data[i * A + a])}
		target_data.data[i] =
			rewards.data.data[i] + config.gamma * max_nq * (1.0 - dones.data.data[i])
	}
	target := t.tensor_new(target_data, false, allocator)

	// Q(s_i, a_i) = sum_a Q(s_i, a) one_hot(a_i); the mask has the layout of q_values
	one_hot_data := l.matrix_new(f64, q_values.data.rows, q_values.data.cols, allocator)
	for a, i in actions {
		assert(a >= 0 && a < A, "cql_loss: action out of range")
		one_hot_data.data[i * A + a] = 1.0
	}
	q_masked := t.tensor_mul(q_values, t.tensor_new(one_hot_data, false, allocator))
	q_masked.shape = [4]int{batch, A, 1, 1}
	q_taken := t.tensor_sum_dim1(q_masked) // [batch, 1]

	td_loss := t.tensor_mse_loss(q_taken, target)
	cql_reg := t.tensor_mean(t.tensor_sub(t.tensor_logsumexp_dim1(q_values, A), q_taken))
	total_loss := t.tensor_add(td_loss, t.tensor_scale(cql_reg, config.alpha))
	return total_loss, td_loss.data.data[0], cql_reg.data.data[0]
}

// ============================================================================
// Trading datasets
// ============================================================================
// The decision on day i (after the close) sets the position for the next day: action 0
// short, 1 flat, 2 long (position a - 1). The state holds only what is known on day i:
// the last CQL_LAGS returns in percent, 100 (close_i / close_{i-1} - 1), and the position
// held into day i (the trading cost depends on it). Reward, in percent of notional:
//   100 * (position_i * (close_{i+1} / close_i - 1) - cost * |position_i - position_{i-1}|)
CQL_LAGS :: 3
CQL_STATE_DIM :: CQL_LAGS + 1

cql_trading_state :: proc(closes: []f64, i: int, prev_position: f64) -> (s: [CQL_STATE_DIM]f64) {
	assert(i >= CQL_LAGS && i < len(closes), "cql_trading_state: needs CQL_LAGS earlier closes")
	for k in 0 ..< CQL_LAGS {s[k] = 100 * (closes[i - k] / closes[i - k - 1] - 1)}
	s[CQL_LAGS] = prev_position
	return
}

// Transitions of the decisions on days CQL_LAGS .. len(closes) - 2; actions[k] is the
// logged action on day CQL_LAGS + k (flat before the first). The last transition is
// terminal. states, next_states [M, CQL_STATE_DIM]; rewards, dones [M, 1]; M = len(actions).
cql_trading_dataset :: proc(
	closes: []f64,
	actions: []int,
	cost: f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	states, rewards, next_states, dones: ^t.Tensor,
) {
	M := len(actions)
	assert(
		M >= 1 && M == len(closes) - 1 - CQL_LAGS,
		"cql_trading_dataset: needs len(closes) - 1 - CQL_LAGS actions",
	)
	S :: CQL_STATE_DIM
	s_data := l.matrix_new(f64, M, S, allocator)
	ns_data := l.matrix_new(f64, M, S, allocator)
	r_data := l.matrix_new(f64, M, 1, allocator)
	d_data := l.matrix_new(f64, M, 1, allocator)
	prev := 0.0
	for a, k in actions {
		assert(a >= 0 && a <= 2, "cql_trading_dataset: actions are 0, 1, 2")
		i := CQL_LAGS + k
		pos := f64(a - 1)
		s := cql_trading_state(closes, i, prev)
		ns := cql_trading_state(closes, i + 1, pos)
		copy(s_data.data[k * S:][:S], s[:])
		copy(ns_data.data[k * S:][:S], ns[:])
		r_data.data[k] = 100 * (pos * (closes[i + 1] / closes[i] - 1) - cost * abs(pos - prev))
		prev = pos
	}
	d_data.data[M - 1] = 1
	states = t.tensor_new(s_data, false, allocator)
	next_states = t.tensor_new(ns_data, false, allocator)
	rewards = t.tensor_new(r_data, false, allocator)
	dones = t.tensor_new(d_data, false, allocator)
	return
}

// n closes from 100 with AR(1) daily returns
//   r_t = mu + phi (r_{t-1} - mu) + sigma eps,  mu = 0.0005, sigma = 0.01
generate_ar1_closes :: proc(n: int, phi: f64, allocator: mem.Allocator) -> []f64 {
	MU, SIGMA :: 0.0005, 0.01
	closes := make([]f64, n, allocator)
	closes[0] = 100
	r := MU
	for i in 1 ..< n {
		r = MU + phi * (r - MU) + SIGMA * rand.float64_normal(0, 1)
		closes[i] = closes[i - 1] * (1 + r)
	}
	return closes
}

// Synthetic log of num_steps transitions on AR(1) returns (phi > 0: momentum a policy can
// earn), traded by a noisy momentum rule (long when r_i + N(0, 0.005) > 0.005, short below
// -0.005, else flat) with a 10 bp cost. Also returns the closes (free with delete).
generate_trading_dataset :: proc(
	num_steps: int,
	allocator: mem.Allocator,
	phi: f64 = 0.3,
) -> (
	states: ^t.Tensor,
	actions: []int,
	rewards: ^t.Tensor,
	next_states: ^t.Tensor,
	dones: ^t.Tensor,
	closes: []f64,
) {
	closes = generate_ar1_closes(num_steps + CQL_LAGS + 1, phi, allocator)
	actions = make([]int, num_steps, allocator)
	for &a, k in actions {
		i := CQL_LAGS + k
		signal := closes[i] / closes[i - 1] - 1 + rand.float64_normal(0.0, 0.005)
		a = 2 if signal > 0.005 else (0 if signal < -0.005 else 1)
	}
	states, rewards, next_states, dones = cql_trading_dataset(closes, actions, 0.001, allocator)
	return
}

// Runs the greedy policy over the closes (decisions on days CQL_LAGS .. len - 2, its own
// previous position in the state); returns the summed reward in percent and the number of
// decisions per action.
cql_backtest :: proc(net: ^CQLNetwork, closes: []f64, cost: f64) -> (pnl: f64, counts: [3]int) {
	prev := 0.0
	for i in CQL_LAGS ..< len(closes) - 1 {
		s := cql_trading_state(closes, i, prev)
		a := cql_greedy_action(net, s[:])
		pos := f64(a - 1)
		pnl += 100 * (pos * (closes[i + 1] / closes[i] - 1) - cost * abs(pos - prev))
		counts[a] += 1
		prev = pos
	}
	return
}
