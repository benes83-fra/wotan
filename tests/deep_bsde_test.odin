package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

deep_bsde_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    DEEP BSDE: HIGH-DIMENSIONAL BLACK-SCHOLES PDE")
	fmt.println("======================================================================\n")

	// Problem parameters (Standard 100D Max-Call Option benchmark)
	d := 100 // Dimension
	T := 1.0 // Time to maturity (1 year)
	N := 20 // Number of time steps
	r := 0.05 // Risk-free rate
	sigma_val := 0.2 // Volatility
	K := 100.0 // Strike price
	S_0_val := 100.0 // Initial spot price for all assets

	// Create S_0 and sigma slices of length d
	S_0 := make([]f64, d, allocator)
	sigma := make([]f64, d, allocator)
	for i in 0 ..< d {
		S_0[i] = S_0_val
		sigma[i] = sigma_val
	}

	batch_size := 64 // Monte Carlo batch size
	epochs := 1000 // Training epochs

	fmt.printf("Problem: %d-dimensional Max-Call Option\n", d)
	fmt.printf(
		"Parameters: T=%.1f, r=%.2f, σ=%.2f, K=%.1f, S_0=%.1f\n",
		T,
		r,
		sigma_val,
		K,
		S_0_val,
	)
	fmt.printf("Training: batch_size=%d, epochs=%d, N=%d steps\n\n", batch_size, epochs, N)

	// Initialize model (Note: n_paths = batch_size)
	model := ml_fin.deep_bsde_model_new(d, T, N, 64, 2, batch_size, allocator)
	defer ml_fin.deep_bsde_model_free(&model)

	// Optimizer
	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.adam_add_param(&opt, model.y0)
	nn.sequential_add_to_adam(model.z_net, &opt)

	fmt.println("Training Progress:")
	fmt.println("Epoch | Loss      | Estimated Y_0")
	fmt.println("------|-----------|--------------")

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		loss := ml_fin.deep_bsde_bs_loss(&model, S_0, K, r, sigma, batch_size, N, allocator)

		// Backward pass
		t.tensor_backward(loss)

		// Optimizer step
		nn.adam_step(&opt)

		if epoch % 100 == 0 || epoch == epochs - 1 {
			fmt.printf("%-5d | %.6f  | %.4f\n", epoch, loss.data.data[0], model.y0.data.data[0])
		}

		// ✅ CRITICAL FIX: Free the ENTIRE computation graph at once after backward pass.
		// This prevents "empty gradient" warnings and memory leaks.
		t.tensor_free_graph(loss)
	}

	fmt.println("\n======================================================================")
	fmt.printf("Final Estimated Option Price (Y_0): $%.4f\n", model.y0.data.data[0])
	fmt.println("(Reference value for 100D Max-Call is approximately $12.8 - $13.0)")
	fmt.println("======================================================================\n")
}
