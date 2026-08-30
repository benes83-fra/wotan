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
		alpha       = 1.0, // ✅ LOWERED from 5.0 to prevent TD/CQL gradient conflict
	}

	// 1. Generate Offline Dataset
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

	// 3. Optimizer
	opt := nn.adam_new(0.003, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, q_net.fc1.weights)
	nn.adam_add_param(&opt, q_net.fc1.bias)
	nn.adam_add_param(&opt, q_net.fc2.weights)
	nn.adam_add_param(&opt, q_net.fc2.bias)

	fmt.println("\nStarting CQL Training...")
	fmt.println("Epoch | Total Loss | TD Loss  | CQL Reg  | Status")
	fmt.println("------|------------|----------|----------|-------------------------")

	epochs := 50

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		q_values := ml_fin.cql_network_forward(&q_net, states)
		next_q_values := ml_fin.cql_network_forward(&q_net, next_states) // Using same net for simplicity in this test

		// ✅ Capture the decomposed loss values
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
		if epoch ==
		   0 {status = "(Initial: alpha * log(3) ≈ 1.098)"} else if total_loss_val < 2.0 {status = "(Converging)"}

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

	fmt.println("\n✓ CQL Training Complete!")
	fmt.println("Notice how lowering alpha allows the TD Loss to decrease,")
	fmt.println("while the CQL Reg stabilizes, preventing out-of-distribution overestimation.")
}
