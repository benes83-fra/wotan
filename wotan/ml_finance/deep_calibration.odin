package ml_finance

import fin "../finance"
import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Deep Heston Calibrator
// ============================================================================
// Maps a vector of option prices (or implied vols) directly to Heston parameters.
// This replaces slow iterative solvers (like Levenberg-Marquardt) with a
// single, microsecond MLP forward pass.

DeepHestonCalibrator :: struct {
	mlp:        ^nn.Sequential,
	input_dim:  int, // Number of options/strikes in the surface
	output_dim: int, // Always 5 for Heston: [v0, kappa, theta, sigma, rho]
	allocator:  mem.Allocator,
}

deep_heston_calibrator_new :: proc(
	input_dim: int,
	hidden_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> DeepHestonCalibrator {
	model: DeepHestonCalibrator
	model.input_dim = input_dim
	model.output_dim = 5 // Heston has 5 parameters
	model.allocator = allocator

	model.mlp = nn.sequential_new(allocator)

	// Input Layer
	nn.sequential_add(model.mlp, nn.linear_layer_new(input_dim, hidden_dim, allocator))
	nn.sequential_add(model.mlp, nn.Activation.ReLU)

	// Hidden Layer
	nn.sequential_add(model.mlp, nn.linear_layer_new(hidden_dim, hidden_dim, allocator))
	nn.sequential_add(model.mlp, nn.Activation.ReLU)

	// Output Layer (No activation, we want raw parameter values)
	nn.sequential_add(model.mlp, nn.linear_layer_new(hidden_dim, 5, allocator))

	return model
}

deep_heston_calibrator_free :: proc(model: ^DeepHestonCalibrator) {
	if model.mlp != nil {
		nn.sequential_free(model.mlp)
	}
}

deep_heston_calibrator_forward :: proc(
	model: ^DeepHestonCalibrator,
	prices: ^t.Tensor,
) -> ^t.Tensor {
	return nn.sequential_forward(model.mlp, prices)
}

deep_heston_calibrator_add_to_adam :: proc(model: ^DeepHestonCalibrator, opt: ^nn.Adam) {
	nn.sequential_add_to_adam(model.mlp, opt)
}

// ============================================================================
// Dataset Generation (Reuses existing finance.heston_price)
// ============================================================================

generate_heston_calibration_dataset :: proc(
	num_samples: int,
	num_options: int,
	allocator: mem.Allocator,
) -> (
	inputs: ^t.Tensor,
	targets: ^t.Tensor,
) {
	// Standard 2D matrices to safely use the standard tensor_matmul path
	inputs_data := l.matrix_new(f64, num_samples, num_options, allocator)
	targets_data := l.matrix_new(f64, num_samples, 5, allocator)

	S0 := 100.0
	r := 0.05

	// Pre-define strikes and maturities for the "market" surface
	strikes := make([]f64, num_options, allocator)
	maturities := make([]f64, num_options, allocator)
	for i in 0 ..< num_options {
		strikes[i] = 80.0 + f64(i) * 5.0 // Strikes from 80 to 80 + 5*N
		maturities[i] = 0.5 + f64(i) * 0.1 // Maturities from 0.5 to 0.5 + 0.1*N
	}
	defer {
		delete(strikes, allocator)
		delete(maturities, allocator)
	}

	for s in 0 ..< num_samples {
		// 1. Generate random valid Heston parameters
		params := fin.Heston_Params {
			v0    = 0.01 + rand.float64() * 0.05, // [0.01, 0.06]
			kappa = 1.0 + rand.float64() * 4.0, // [1.0, 5.0]
			theta = 0.01 + rand.float64() * 0.05, // [0.01, 0.06]
			sigma = 0.1 + rand.float64() * 0.5, // [0.1, 0.6]
			rho   = -0.9 + rand.float64() * 0.8, // [-0.9, -0.1]
		}

		// Enforce Feller condition to ensure valid dataset: 2 * kappa * theta > sigma^2
		for params.sigma * params.sigma >= 2.0 * params.kappa * params.theta {
			params.sigma = 0.1 + rand.float64() * 0.3 // Reduce sigma until valid
		}

		// 2. Store target parameters
		targets_data.data[s * 5 + 0] = params.v0
		targets_data.data[s * 5 + 1] = params.kappa
		targets_data.data[s * 5 + 2] = params.theta
		targets_data.data[s * 5 + 3] = params.sigma
		targets_data.data[s * 5 + 4] = params.rho

		// 3. Price options using your EXISTING, optimized finance.heston_price
		// (Using 1000 points is fast and accurate enough for dataset generation)
		for o in 0 ..< num_options {
			price := fin.heston_price(S0, strikes[o], maturities[o], r, params, .Call, 1000)
			inputs_data.data[s * num_options + o] = price
		}
	}

	// Wrap in Tensors (Standard 2D layout: rows=num_samples, cols=features)
	in_tensor := t.tensor_new(inputs_data, true, allocator)
	in_tensor.shape = [4]int{num_samples, num_options, 1, 1}

	tgt_tensor := t.tensor_new(targets_data, false, allocator)
	tgt_tensor.shape = [4]int{num_samples, 5, 1, 1}

	return in_tensor, tgt_tensor
}
