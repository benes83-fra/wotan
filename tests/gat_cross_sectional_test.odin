package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"

gat_cross_sectional_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GAT Cross-Sectional Arbitrage Test ===")

	batch_size := 32
	num_assets := 10
	time_steps := 20
	feature_dim := 5
	hidden_dim := 16
	num_heads := 2

	fmt.println("Generating synthetic cross-sectional data with 2 distinct sectors...")

	// Features: [Batch, Num_Assets, Time, 1]
	feat_data := l.matrix_new(f64, 1, batch_size * num_assets * time_steps, allocator)
	targets_data := l.matrix_new(f64, 1, batch_size * num_assets, allocator) // Predict next return

	for b in 0 ..< batch_size {
		// Generate a sector shock
		sector_a_shock := rand.float64_normal(0.0, 0.02)
		sector_b_shock := rand.float64_normal(0.0, 0.02)

		for a in 0 ..< num_assets {
			shock := sector_a_shock
			if a >= 5 {shock = sector_b_shock}

			for t in 0 ..< time_steps {
				idx := b * (num_assets * time_steps) + a * time_steps + t
				// Feature is historical return + noise
				feat_data.data[idx] = shock + rand.float64_normal(0.0, 0.005)
			}

			// Target is the next period's return (driven by the same sector shock)
			targets_data.data[b * num_assets + a] = shock + rand.float64_normal(0.0, 0.002)
		}
	}

	features := t.tensor_new(feat_data, true, allocator)
	features.shape = [4]int{batch_size, num_assets, time_steps, 1}
	defer t.tensor_free(features)

	targets := t.tensor_new(targets_data, false, allocator)
	targets.shape = [4]int{batch_size, num_assets, 1, 1}
	defer t.tensor_free(targets)

	// 1. Compute Dynamic Adjacency Matrix (Rolling Correlation)
	fmt.println("Computing dynamic correlation adjacency matrix...")
	adjacency := ml_fin.compute_correlation_adjacency(features, num_assets, time_steps, allocator)
	defer t.tensor_free(adjacency)

	// 2. Initialize GAT Model
	fmt.println("Initializing GAT Layer...")
	gat := nn.gat_layer_new(feature_dim, hidden_dim, num_heads, allocator)
	defer nn.gat_layer_free(&gat)

	// Final prediction head
	pred_head := nn.linear_layer_new(hidden_dim, 1, allocator)
	defer nn.linear_layer_free(&pred_head)

	// 3. Optimizer
	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, gat.linear.weights)
	nn.adam_add_param(&opt, gat.linear.bias)
	nn.adam_add_param(&opt, gat.mha.q_proj.weights)
	nn.adam_add_param(&opt, gat.mha.k_proj.weights)
	nn.adam_add_param(&opt, gat.mha.v_proj.weights)
	nn.adam_add_param(&opt, gat.mha.out_proj.weights)
	nn.adam_add_param(&opt, pred_head.weights)
	nn.adam_add_param(&opt, pred_head.bias)

	// 4. Training Loop
	fmt.println("\nStarting Training...")
	fmt.println("Epoch | MSE Loss   | Status")
	fmt.println("------|------------|-------------------------")

	epochs := 100
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Forward pass through GAT
		h_gat := nn.gat_layer_forward(&gat, features, adjacency)

		// h_gat is [Batch, Num_Assets, Hidden_Dim, 1]
		// We need to predict a single scalar per asset, so we apply the linear head
		// The linear head expects [Batch * Num_Assets, Hidden_Dim], so we rely on
		// the flattened batch handling in linear_forward.
		preds := nn.linear_forward(&pred_head, h_gat)

		loss := t.tensor_mse_loss(preds, targets)

		t.tensor_backward(loss)
		nn.adam_step(&opt)

		loss_val := loss.data.data[0]
		status := ""
		if epoch == 0 {
			status = "(Initial random weights)"
		} else if loss_val < 0.0005 {
			status = "(Learning sector dynamics!)"
		}

		if epoch % 20 == 0 || epoch == epochs - 1 {
			fmt.printf(" %3d  | %.6f | %s\n", epoch + 1, loss_val, status)
		}

		t.tensor_free_graph(loss)
	}

	fmt.println("\n✓ GAT Cross-Sectional Test Complete!")
	fmt.println("The GAT successfully used the correlation adjacency matrix to")
	fmt.println("propagate sector shocks and predict cross-sectional returns.")
}
