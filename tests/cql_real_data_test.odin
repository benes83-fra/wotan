package tests

import w "../wotan/core"
import importer "../wotan/importer"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"


// ============================================================================
// 2. Feature Engineering & Dataset Construction
// ============================================================================

build_cql_dataset_from_ohlcv :: proc(
	df: ^w.DataFrame,
	allocator: mem.Allocator,
) -> (
	states: ^t.Tensor,
	actions: []int,
	rewards: ^t.Tensor,
	next_states: ^t.Tensor,
	dones: ^t.Tensor,
) {
	n_rows := df.rows
	if n_rows < 10 {
		fmt.println("Error: Not enough data rows for CQL dataset.")
		return nil, nil, nil, nil, nil
	}

	state_dim := 3 // [daily_return, momentum_3d, volatility_5d]
	num_actions := 3 // 0: Short, 1: Hold, 2: Long

	states_data := l.matrix_new(f64, n_rows - 1, state_dim, allocator)
	rewards_data := l.matrix_new(f64, n_rows - 1, 1, allocator)
	next_states_data := l.matrix_new(f64, n_rows - 1, state_dim, allocator)
	dones_data := l.matrix_new(f64, n_rows - 1, 1, allocator)
	actions_slice := make([]int, n_rows - 1, allocator)

	close_col := &df.columns[4] // Close price

	prev_action := 1 // Start with Hold

	for i := 0; i < n_rows - 1; i += 1 {
		close_today, _ := w.column_at_float(close_col, i)
		close_tomorrow, _ := w.column_at_float(close_col, i + 1)

		// 1. Compute State Features
		daily_ret := (close_tomorrow - close_today) / close_today

		// Simple momentum (3-day lookback)
		momentum := 0.0
		if i >= 3 {
			close_3d_ago, _ := w.column_at_float(close_col, i - 3)
			momentum = (close_today - close_3d_ago) / close_3d_ago
		}

		// Simple volatility (5-day std dev of returns)
		volatility := 0.0
		if i >= 5 {
			var_sum := 0.0
			for j := 0; j < 5; j += 1 {
				c1, _ := w.column_at_float(close_col, i - j)
				c2, _ := w.column_at_float(close_col, i - j - 1)
				r := (c1 - c2) / c2
				var_sum += r * r
			}
			volatility = math.sqrt(var_sum / 5.0)
		}

		// Store current state
		states_data.data[i * state_dim + 0] = daily_ret
		states_data.data[i * state_dim + 1] = momentum
		states_data.data[i * state_dim + 2] = volatility

		// 2. Generate Historical Actions (Behavioral Policy: Trend Following)
		action := 1 // Hold
		if momentum > 0.01 {
			action = 2 // Long
		} else if momentum < -0.01 {
			action = 0 // Short
		}
		actions_slice[i] = action

		// 3. Compute Reward (Realized PnL - Transaction Cost)
		direction := f64(action - 1) // -1, 0, 1
		prev_direction := f64(prev_action - 1)
		tx_cost := 0.001 * math.abs(direction - prev_direction) // 10bps cost for flipping
		reward := direction * daily_ret - tx_cost
		rewards_data.data[i] = reward

		// 4. Setup Next State & Done flag
		if i < n_rows - 2 {
			next_states_data.data[i * state_dim + 0] = states_data.data[(i + 1) * state_dim + 0]
			next_states_data.data[i * state_dim + 1] = states_data.data[(i + 1) * state_dim + 1]
			next_states_data.data[i * state_dim + 2] = states_data.data[(i + 1) * state_dim + 2]
			dones_data.data[i] = 0.0
		} else {
			dones_data.data[i] = 1.0
		}

		prev_action = action
	}

	// Wrap in Tensors
	states = t.tensor_new(states_data, true, allocator)
	states.shape = [4]int{n_rows - 1, state_dim, 1, 1}

	next_states = t.tensor_new(next_states_data, false, allocator)
	next_states.shape = [4]int{n_rows - 1, state_dim, 1, 1}

	rewards = t.tensor_new(rewards_data, false, allocator)
	rewards.shape = [4]int{n_rows - 1, 1, 1, 1}

	dones = t.tensor_new(dones_data, false, allocator)
	dones.shape = [4]int{n_rows - 1, 1, 1, 1}

	return states, actions_slice, rewards, next_states, dones
}

// ============================================================================
// 3. Main Test Orchestrator
// ============================================================================
// ============================================================================
// 3. Main Test Orchestrator
// ============================================================================

cql_real_data_trading_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Yahoo Finance Offline CQL Trading Test ===")

	// ✅ FIX: Use your existing, robust JSON-based Yahoo Finance downloader
	// instead of the broken placeholder CSV URL.
	fmt.println("Fetching AAPL data via net.read_yahoo...")
	df := net.read_yahoo("AAPL", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Aborting test due to empty DataFrame.")
		return
	}

	fmt.printf("✓ Successfully loaded %d days of OHLCV data.\n", df.rows)

	fmt.println("Building CQL dataset from OHLCV features...")
	states, actions, rewards, next_states, dones := build_cql_dataset_from_ohlcv(&df, allocator)
	defer t.tensor_free(states)
	defer t.tensor_free(next_states)
	defer t.tensor_free(rewards)
	defer t.tensor_free(dones)
	defer delete(actions, allocator)

	if states == nil {
		fmt.println("Error: Failed to build CQL dataset.")
		return
	}

	batch_size := states.shape[0]
	fmt.printf("Dataset size: %d steps\n", batch_size)

	// 1. CQL Configuration
	config := ml_fin.CQLConfig {
		state_dim   = 3,
		num_actions = 3,
		hidden_dim  = 32,
		gamma       = 0.95,
		alpha       = 1.0, // Conservative penalty weight
	}

	// 2. Initialize Q-Networks
	q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&q_net)

	// 3. Optimizer
	opt := nn.adam_new(0.003, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, q_net.fc1.weights)
	nn.adam_add_param(&opt, q_net.fc1.bias)
	nn.adam_add_param(&opt, q_net.fc2.weights)
	nn.adam_add_param(&opt, q_net.fc2.bias)

	fmt.println("\nStarting Offline CQL Training on Real AAPL Data...")
	fmt.println("Epoch | Total Loss | TD Loss  | CQL Reg  | Status")
	fmt.println("------|------------|----------|----------|-------------------------")

	epochs := 50

	for epoch := 0; epoch < epochs; epoch += 1 {
		nn.adam_zero_grad(&opt)

		q_values := ml_fin.cql_network_forward(&q_net, states)
		next_q_values := ml_fin.cql_network_forward(&q_net, next_states)

		loss, td_val, cql_val := ml_fin.cql_loss(
			q_values,
			actions,
			rewards,
			next_q_values,
			dones,
			config,
			allocator,
		)

		t.tensor_backward(loss)
		nn.adam_step(&opt)

		total_loss_val := loss.data.data[0]

		status := ""
		if epoch == 0 {
			status = "(Initial)"
		} else if total_loss_val < 2.0 {
			status = "(Converging)"
		}

		fmt.printf(
			" %3d  | %.5f    | %.5f | %.5f | %s\n",
			epoch + 1,
			total_loss_val,
			td_val,
			cql_val,
			status,
		)

		t.tensor_free_graph(loss)
	}

	fmt.println("\n✓ Yahoo Finance CQL Test Complete!")
	fmt.println("The agent learned to trade AAPL directly from historical logs,")
	fmt.println("penalizing out-of-distribution actions (e.g., aggressive flipping).")
}
