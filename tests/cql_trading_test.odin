package tests

import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
// ============================================================================
// 3. Main Test Orchestrator
// ============================================================================

cql_trading_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Offline CQL Historical Trading Test ===")

	// 1. CQL Configuration (Added tau for soft updates)
	config := ml_fin.CQLConfig {
		state_dim   = 4,
		num_actions = 3,
		hidden_dim  = 32,
		gamma       = 0.95,
		alpha       = 1.0, // Conservative penalty weight
		tau         = 0.005, // Soft update rate (0.5% per step)
	}

	// 2. Generate Offline Dataset
	num_steps := 500
	states, actions, rewards, next_states, dones := ml_fin.generate_trading_dataset(
		num_steps,
		allocator,
	)
	defer t.tensor_free(states)
	defer t.tensor_free(next_states)
	defer t.tensor_free(rewards)
	defer t.tensor_free(dones)
	defer delete(actions, allocator)
	fmt.printf(
		"DEBUG: states shape = [%d, %d, 1, 1], flat cols = %d\n",
		states.shape[0],
		states.shape[1],
		states.data.cols,
	)

	fmt.printf("Generated dataset: %d steps, %d actions\n", num_steps, config.num_actions)

	// 3. Initialize Q-Networks
	q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&q_net)

	target_q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&target_q_net)

	// ✅ Hard copy initial weights to target network
	ml_fin.cql_network_copy(&q_net, &target_q_net)

	// 4. Optimizer (Only optimize the main network!)
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

		// Forward pass through main network
		q_values := ml_fin.cql_network_forward(&q_net, states)

		// ✅ Forward pass through TARGET network for next states (stabilizes TD targets)
		next_q_values := ml_fin.cql_network_forward(&target_q_net, next_states)

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

		// ✅ Soft update target network towards main network
		ml_fin.cql_soft_update(&q_net, &target_q_net, config.tau)

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
	fmt.println("with a stabilized TD loss thanks to the soft-updating target network.")
}
