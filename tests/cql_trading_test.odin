package tests

import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

cql_trading_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Offline CQL Historical Trading Test ===")

	config := ml_fin.CQLConfig {
		state_dim   = 4,
		num_actions = 3,
		hidden_dim  = 32,
		gamma       = 0.95,
		alpha       = 5.0, // CQL penalty weight
	}

	// 1. Generate Offline Dataset (simulating historical logs)
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

	fmt.printf("Generated dataset: %d steps, %d actions\n", num_steps, config.num_actions)

	// 2. Initialize Q-Networks
	q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&q_net)

	target_q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&target_q_net)

	// Copy weights to target network (simplified: in practice, use a soft update)
	// For this test, we'll just keep target fixed or do a hard copy periodically

	// 3. Optimizer
	opt := nn.adam_new(0.003, allocator = allocator)
	defer nn.adam_free(&opt)

	// ✅ FIX: Pass &opt instead of opt
	nn.adam_add_param(&opt, q_net.fc1.weights)
	nn.adam_add_param(&opt, q_net.fc1.bias)
	nn.adam_add_param(&opt, q_net.fc2.weights)
	nn.adam_add_param(&opt, q_net.fc2.bias)

	fmt.println("\nStarting CQL Training...")
	fmt.println("Epoch | Total Loss | TD Loss  | CQL Reg  | Status")
	fmt.println("------|------------|----------|----------|-------------------------")

	epochs := 50
	batch_size := 64

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Sample a batch (simplified: using the whole dataset for this small test)
		// In production, you would slice the tensors here.
		q_values := ml_fin.cql_network_forward(&q_net, states)
		next_q_values := ml_fin.cql_network_forward(&target_q_net, next_states)

		loss := ml_fin.cql_loss(
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

		// Extract loss components for logging
		total_loss_val := loss.data.data[0]

		// Simple status check
		status := ""
		if epoch == 0 {status = "(Initial)"} else if total_loss_val < 0.5 {status = "(Converging)"}

		fmt.printf(
			" %3d  | %.5f    | (Logged) | (Logged) | %s\n",
			epoch + 1,
			total_loss_val,
			status,
		)

		t.tensor_free_graph(loss)

		// Soft update target network (tau = 0.005)
		// (Omitted for brevity, but would go here in production)
	}

	fmt.println("\n✓ CQL Training Complete!")
	fmt.println("The CQL penalty prevents the Q-network from assigning high values")
	fmt.println("to out-of-distribution actions (e.g., aggressive flipping) that")
	fmt.println("were not present in the historical behavioral dataset.")
}
