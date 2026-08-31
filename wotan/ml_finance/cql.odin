package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// CQL Configuration & Structures
// ============================================================================

CQLConfig :: struct {
	state_dim:   int,
	num_actions: int,
	hidden_dim:  int,
	gamma:       f64, // Discount factor
	alpha:       f64, // CQL regularization weight
	tau:         f64,
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

cql_network_forward :: proc(net: ^CQLNetwork, states: ^t.Tensor) -> ^t.Tensor {
	h := nn.linear_forward(&net.fc1, states)
	h = t.tensor_relu(h)
	q_values := nn.linear_forward(&net.fc2, h)
	return q_values
}

// ============================================================================
// CQL Loss Function (Bulletproof Adaptive Version)
// ============================================================================

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
	batch := q_values.shape[0]
	num_actions := config.num_actions

	// 1. Compute max_next_q manually
	max_next_q_data := l.matrix_new(f64, batch, 1, allocator)
	for i in 0 ..< batch {
		max_val := -math.F64_MAX
		for a in 0 ..< num_actions {
			val := next_q_values.data.data[i * num_actions + a]
			if val > max_val {
				max_val = val
			}
		}
		max_next_q_data.data[i] = max_val
	}
	max_next_q := t.tensor_new(max_next_q_data, false, allocator)

	// 2. Compute TD Target: r + gamma * max_next_q * (1 - done)
	target_data := l.matrix_new(f64, batch, 1, allocator)
	for i in 0 ..< batch {
		r := rewards.data.data[i]
		max_nq := max_next_q_data.data[i]
		done := dones.data.data[i]
		target_data.data[i] = r + config.gamma * max_nq * (1.0 - done)
	}
	target := t.tensor_new(target_data, false, allocator)
	target.shape = [4]int{batch, 1, 1, 1}

	// 3. q_taken = sum(q_values * one_hot(actions), dim=1)
	// ✅ BULLETPROOF FIX: Dynamically match the physical shape of q_values
	one_hot_data: l.Matrix(f64)
	if q_values.data.rows == 1 {
		// Flattened batch case (e.g., 1 x 1500)
		one_hot_data = l.matrix_new(f64, 1, batch * num_actions, allocator)
		for i in 0 ..< batch {
			a := actions[i]
			one_hot_data.data[i * num_actions + a] = 1.0
		}
	} else {
		// Standard 2D case (e.g., 250 x 3)
		one_hot_data = l.matrix_new(f64, batch, num_actions, allocator)
		for i in 0 ..< batch {
			a := actions[i]
			one_hot_data.data[i * num_actions + a] = 1.0
		}
	}
	one_hot := t.tensor_new(one_hot_data, false, allocator)

	q_masked := t.tensor_mul(q_values, one_hot)

	// Explicitly set logical shape so tensor_sum_dim1 interprets dimensions correctly
	q_masked.shape = [4]int{batch, num_actions, 1, 1}

	q_taken := t.tensor_sum_dim1(q_masked) // Results in [Batch, 1, 1, 1]

	// 4. TD Loss
	td_loss := t.tensor_mse_loss(q_taken, target)

	// 5. CQL Regularization: alpha * (logsumexp(q_values) - q_taken)
	lse_q := t.tensor_logsumexp_dim1(q_values, num_actions, allocator)
	diff := t.tensor_sub(lse_q, q_taken)
	cql_reg := t.tensor_mean(diff)

	// Total Loss = td_loss + alpha * cql_reg
	alpha_data := l.matrix_new(f64, 1, 1, allocator)
	alpha_data.data[0] = config.alpha
	alpha_tensor := t.tensor_new(alpha_data, false, allocator)

	scaled_cql := t.tensor_mul(cql_reg, alpha_tensor)
	total_loss := t.tensor_add(td_loss, scaled_cql)

	// Capture scalar values for logging before returning
	td_val := td_loss.data.data[0]
	cql_val := cql_reg.data.data[0]

	return total_loss, td_val, cql_val
}
// ============================================================================
// Financial Dataset Generator (Historical Trade Execution)
// ============================================================================

generate_trading_dataset :: proc(
	num_steps: int,
	allocator: mem.Allocator,
) -> (
	states: ^t.Tensor,
	actions: []int,
	rewards: ^t.Tensor,
	next_states: ^t.Tensor,
	dones: ^t.Tensor,
) {

	state_dim := 4
	num_actions := 3 // 0: Short, 1: Hold, 2: Long

	// ✅ FIX: Use rows=1 to ensure tensor_matmul hits the flattened batch handling
	states_data := l.matrix_new(f64, 1, num_steps * state_dim, allocator)
	rewards_data := l.matrix_new(f64, 1, num_steps, allocator)
	next_states_data := l.matrix_new(f64, 1, num_steps * state_dim, allocator)
	dones_data := l.matrix_new(f64, 1, num_steps, allocator)
	actions_slice := make([]int, num_steps, allocator)

	prev_action := 1 // Start with Hold

	for i in 0 ..< num_steps {
		// Synthetic market: random walk returns
		ret := rand.float64_normal(0.0005, 0.01)
		vol := math.abs(ret) * 10.0

		// State: [current_ret, lag_1, lag_2, volatility]
		states_data.data[i * state_dim + 0] = ret
		states_data.data[i * state_dim + 1] = ret * 0.8
		states_data.data[i * state_dim + 2] = ret * 0.6
		states_data.data[i * state_dim + 3] = vol

		// Behavioral policy (historical data): noisy momentum
		signal := ret + rand.float64_normal(0.0, 0.005)
		action := 1 // Hold
		if signal > 0.005 {
			action = 2 // Long
		} else if signal < -0.005 {
			action = 0 // Short
		}

		actions_slice[i] = action

		// Reward: return * direction - transaction cost for flipping
		direction := f64(action - 1) // -1, 0, 1
		prev_direction := f64(prev_action - 1)
		tx_cost := 0.001 * math.abs(direction - prev_direction)
		reward := direction * ret - tx_cost

		rewards_data.data[i] = reward

		// Next state setup
		if i < num_steps - 1 {
			for s in 0 ..< state_dim {
				next_states_data.data[i * state_dim + s] =
					states_data.data[(i + 1) * state_dim + s]
			}
			dones_data.data[i] = 0.0
		} else {
			for s in 0 ..< state_dim {
				next_states_data.data[i * state_dim + s] = 0.0
			}
			dones_data.data[i] = 1.0
		}

		prev_action = action
	}

	// ✅ FIX: Use '=' instead of ':=' to assign to named return values
	states = t.tensor_new(states_data, true, allocator)
	states.shape = [4]int{num_steps, state_dim, 1, 1}

	next_states = t.tensor_new(next_states_data, false, allocator)
	next_states.shape = [4]int{num_steps, state_dim, 1, 1}

	rewards = t.tensor_new(rewards_data, false, allocator)
	rewards.shape = [4]int{num_steps, 1, 1, 1}

	dones = t.tensor_new(dones_data, false, allocator)
	dones.shape = [4]int{num_steps, 1, 1, 1}

	return states, actions_slice, rewards, next_states, dones
}

// cql_network_copy performs a hard copy of weights from src to dst
cql_network_copy :: proc(src: ^CQLNetwork, dst: ^CQLNetwork) {
	copy(dst.fc1.weights.data.data, src.fc1.weights.data.data)
	if src.fc1.bias != nil && dst.fc1.bias != nil {
		copy(dst.fc1.bias.data.data, src.fc1.bias.data.data)
	}
	copy(dst.fc2.weights.data.data, src.fc2.weights.data.data)
	if src.fc2.bias != nil && dst.fc2.bias != nil {
		copy(dst.fc2.bias.data.data, src.fc2.bias.data.data)
	}
}

// cql_soft_update performs: target = tau * main + (1 - tau) * target
cql_soft_update :: proc(main_net: ^CQLNetwork, target_net: ^CQLNetwork, tau: f64) {
	inv_tau := 1.0 - tau

	// Update fc1 weights
	for i in 0 ..< len(main_net.fc1.weights.data.data) {
		target_net.fc1.weights.data.data[i] =
			tau * main_net.fc1.weights.data.data[i] + inv_tau * target_net.fc1.weights.data.data[i]
	}
	if main_net.fc1.bias != nil && target_net.fc1.bias != nil {
		for i in 0 ..< len(main_net.fc1.bias.data.data) {
			target_net.fc1.bias.data.data[i] =
				tau * main_net.fc1.bias.data.data[i] + inv_tau * target_net.fc1.bias.data.data[i]
		}
	}

	// Update fc2 weights
	for i in 0 ..< len(main_net.fc2.weights.data.data) {
		target_net.fc2.weights.data.data[i] =
			tau * main_net.fc2.weights.data.data[i] + inv_tau * target_net.fc2.weights.data.data[i]
	}
	if main_net.fc2.bias != nil && target_net.fc2.bias != nil {
		for i in 0 ..< len(main_net.fc2.bias.data.data) {
			target_net.fc2.bias.data.data[i] =
				tau * main_net.fc2.bias.data.data[i] + inv_tau * target_net.fc2.bias.data.data[i]
		}
	}
}
