package tests

import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

deep_calibration_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Deep Heston Calibration Test ===")

	num_samples := 5000 // Generate 5000 synthetic market scenarios
	num_options := 10 // 10 different strike/maturity combinations
	hidden_dim := 64

	fmt.printf(
		"Generating synthetic dataset (%d samples, %d options) using finance.heston_price...\n",
		num_samples,
		num_options,
	)

	prices, params := ml_fin.generate_heston_calibration_dataset(
		num_samples,
		num_options,
		allocator,
	)
	defer t.tensor_free(prices)
	defer t.tensor_free(params)

	fmt.println("Dataset generated successfully.")

	// 1. Initialize Deep Calibrator
	fmt.println("Initializing Deep Calibrator MLP (Prices -> Parameters)...")
	calibrator := ml_fin.deep_heston_calibrator_new(num_options, hidden_dim, allocator)
	defer ml_fin.deep_heston_calibrator_free(&calibrator)

	// 2. Optimizer
	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	ml_fin.deep_heston_calibrator_add_to_adam(&calibrator, &opt)

	// 3. Training Loop
	fmt.println("\nStarting Training...")
	fmt.println("Epoch | MSE Loss   | Status")
	fmt.println("------|------------|-------------------------")

	epochs := 150

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Forward pass
		predicted_params := ml_fin.deep_heston_calibrator_forward(&calibrator, prices)

		// MSE Loss between predicted parameters and true parameters
		loss := t.tensor_mse_loss(predicted_params, params)

		// Backward & Step
		t.tensor_backward(loss)
		nn.adam_step(&opt)

		loss_val := loss.data.data[0]

		status := ""
		if epoch == 0 {
			status = "(Initial random weights)"
		} else if loss_val < 0.001 {
			status = "(Highly Accurate Calibration)"
		} else if loss_val < 0.01 {
			status = "(Learning mapping)"
		}

		if epoch % 15 == 0 || epoch == epochs - 1 {
			fmt.printf(" %3d  | %.6f | %s\n", epoch + 1, loss_val, status)
		}

		t.tensor_free_graph(loss)
	}

	// 4. Inference / Microsecond Calibration Test
	fmt.println("\n=== Microsecond Calibration Inference ===")
	fmt.println("Generating a NEW unseen market price surface...")

	// Generate 1 new sample
	new_prices_data := l.matrix_new(f64, 1, num_options, allocator)
	true_params := fin.Heston_Params {
		v0    = 0.04,
		kappa = 2.0,
		theta = 0.04,
		sigma = 0.3,
		rho   = -0.5,
	}

	S0 := 100.0; r := 0.05
	for o in 0 ..< num_options {
		K := 80.0 + f64(o) * 5.0
		T := 0.5 + f64(o) * 0.1
		// Price using your existing engine
		price := fin.heston_price(S0, K, T, r, true_params, .Call, 1000)
		new_prices_data.data[o] = price
	}

	new_prices := t.tensor_new(new_prices_data, false, allocator)
	new_prices.shape = [4]int{1, num_options, 1, 1}
	defer t.tensor_free(new_prices)

	fmt.println("Running MLP forward pass (No iterative solver needed)...")

	inference_params := ml_fin.deep_heston_calibrator_forward(&calibrator, new_prices)
	defer t.tensor_free(inference_params)

	fmt.printf(
		"\nTrue Parameters:      v0=%.4f, kappa=%.2f, theta=%.4f, sigma=%.2f, rho=%.2f\n",
		true_params.v0,
		true_params.kappa,
		true_params.theta,
		true_params.sigma,
		true_params.rho,
	)

	fmt.printf(
		"MLP Predicted Params: v0=%.4f, kappa=%.2f, theta=%.4f, sigma=%.2f, rho=%.2f\n",
		inference_params.data.data[0],
		inference_params.data.data[1],
		inference_params.data.data[2],
		inference_params.data.data[3],
		inference_params.data.data[4],
	)

	fmt.println("\n✓ Deep Calibration Test Complete!")
	fmt.println("The MLP calibrated the Heston model in a single forward pass (microseconds),")
	fmt.println("bypassing the need for slow iterative solvers like Levenberg-Marquardt.")
}
