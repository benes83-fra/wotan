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


	N := 20


	// Becker et al. 100D Max-Call parameters
	d := 100
	T := 1.0 / 3.0 // 4 months
	K := 16.3 // Not 100!
	r := 0.0 // Zero interest rate in their setup
	sigma_val := 0.2 // Or heterogeneous: sigma[i] = (10 + i/2) / 100
	rho := 0.3 // Correlated assets
	S_0_val := 1.0 // Normalized initial prices

	S_0 := make([]f64, d, allocator)
	sigma := make([]f64, d, allocator)
	for i in 0 ..< d {
		S_0[i] = S_0_val
		sigma[i] = sigma_val
	}

	batch_size := 64
	epochs := 5000

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

	model := ml_fin.deep_bsde_model_new(12.0, d, T, N, 64, 2, batch_size, allocator)
	defer ml_fin.deep_bsde_model_free(&model)

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
		t.tensor_backward(loss)
		nn.adam_step(&opt)

		if epoch % 100 == 0 || epoch == epochs - 1 {
			fmt.printf("%-5d | %.4f  | %.4f\n", epoch, loss.data.data[0], model.y0.data.data[0])
		}

		t.tensor_free_graph(loss)
	}

	fmt.println("\n======================================================================")
	fmt.printf("Final Estimated Option Price (Y_0): $%.4f\n", model.y0.data.data[0])
	fmt.println("(Reference value for 100D Max-Call is approximately $12.8 - $13.0)")
	fmt.println("======================================================================\n")
}
deep_bsde_european_call_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== DEEP BSDE: 1D EUROPEAN CALL VALIDATION ===")

	// Parameters for a simple 1D European Call
	d := 1
	T := 1.0
	N := 10 // Fewer steps for 1D
	r := 0.05
	sigma_val := 0.2
	K := 100.0
	S_0_val := 100.0

	S_0 := make([]f64, d, allocator)
	sigma := make([]f64, d, allocator)
	for i in 0 ..< d {
		S_0[i] = S_0_val
		sigma[i] = sigma_val
	}

	batch_size := 64
	epochs := 3000

	// Create model
	model := ml_fin.deep_bsde_model_new(10.0, d, T, N, 64, 2, batch_size, allocator)
	defer ml_fin.deep_bsde_model_free(&model)

	opt := nn.adam_new(0.005, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.adam_add_param(&opt, model.y0)
	// For 1D, we can try to train Z too, but let's stick to Y0 for stability first

	fmt.println("Training 1D European Call (Target: Black-Scholes Price)...")

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		loss := ml_fin.deep_bsde_bs_loss(&model, S_0, K, r, sigma, batch_size, N, allocator)
		t.tensor_backward(loss)
		nn.adam_step(&opt)

		if epoch % 500 == 0 {
			fmt.printf(
				"Epoch %d | Loss: %.4f | Y0: %.4f\n",
				epoch,
				loss.data.data[0],
				model.y0.data.data[0],
			)
		}

		t.tensor_free_graph(loss)
	}

	// Analytical Black-Scholes Price for comparison
	d1 :=
		(math.ln(S_0_val / K) + (r + 0.5 * sigma_val * sigma_val) * T) / (sigma_val * math.sqrt(T))
	d2 := d1 - sigma_val * math.sqrt(T)
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
	bs_price := S_0_val * N_d1 - K * math.exp(-r * T) * N_d2

	fmt.printf("\nDeep BSDE Result: $%.4f\n", model.y0.data.data[0])
	fmt.printf("Black-Scholes Reference: $%.4f\n", bs_price)
	fmt.printf("Error: %.4f%%\n", math.abs(model.y0.data.data[0] - bs_price) / bs_price * 100)
}
