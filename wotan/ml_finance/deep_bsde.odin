package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Deep BSDE Model
// ============================================================================
DeepBSDEModel :: struct {
	y0:        ^t.Tensor,
	z_net:     ^nn.Sequential,
	T:         f64,
	N:         int,
	d:         int,
	allocator: mem.Allocator,
}

deep_bsde_model_new :: proc(
	initial_y0: f64,
	d: int,
	T: f64,
	N: int,
	hidden_size: int,
	num_layers: int,
	n_paths: int,
	allocator: mem.Allocator = context.allocator,
) -> DeepBSDEModel {
	model: DeepBSDEModel
	model.T = T
	model.N = N
	model.d = d
	model.allocator = allocator

	y0_data := l.matrix_new(f64, n_paths, 1, allocator)
	for i in 0 ..< n_paths {
		y0_data.data[i] = initial_y0
	}
	model.y0 = t.tensor_new(y0_data, true, allocator)

	model.z_net = nn.sequential_new(allocator)
	nn.sequential_add(model.z_net, nn.linear_layer_new(1 + d, hidden_size, allocator))
	nn.sequential_add(model.z_net, nn.Activation.ReLU)
	for _ in 1 ..< num_layers {
		nn.sequential_add(model.z_net, nn.linear_layer_new(hidden_size, hidden_size, allocator))
		nn.sequential_add(model.z_net, nn.Activation.ReLU)
	}
	nn.sequential_add(model.z_net, nn.linear_layer_new(hidden_size, d, allocator))

	return model
}

deep_bsde_model_free :: proc(model: ^DeepBSDEModel) {
	if model.y0 != nil {t.tensor_free(model.y0)}
	if model.z_net != nil {nn.sequential_free(model.z_net)}
}

// ============================================================================
// Forward Pass
// ============================================================================
deep_bsde_bs_forward :: proc(
	model: ^DeepBSDEModel,
	S_0: []f64,
	K: f64,
	r: f64,
	sigma: []f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	Y_T: ^t.Tensor,
	X_T: ^t.Tensor,
) {
	d := model.d
	dt := model.T / f64(n_steps)
	sqrt_dt := math.sqrt(dt)

	// Initial state X_0 (does not require gradients, just part of the graph)
	X_data := l.matrix_new(f64, n_paths, d, allocator)
	for i in 0 ..< n_paths {
		for j in 0 ..< d {
			X_data.data[i * d + j] = S_0[j]
		}
	}
	X := t.tensor_new(X_data, false, allocator)

	// Y_0 is a trainable parameter (lives on the heap, not in the scope arena)
	Y := model.y0

	for step in 0 ..< n_steps {
		t_k := f64(step) * dt

		// 1. Generate Brownian increments
		dW_data := l.matrix_new(f64, n_paths, d, allocator)
		for i in 0 ..< n_paths {
			for j in 0 ..< d {
				dW_data.data[i * d + j] = rand.float64_normal(0.0, 1.0) * sqrt_dt
			}
		}
		dW := t.tensor_new(dW_data, false, allocator)

		// 2. Build input [t, X_t] for the Z-network
		input_data := l.matrix_new(f64, n_paths, 1 + d, allocator)
		for i in 0 ..< n_paths {
			input_data.data[i * (1 + d) + 0] = t_k
			for j in 0 ..< d {
				input_data.data[i * (1 + d) + 1 + j] = X.data.data[i * d + j]
			}
		}
		input := t.tensor_new(input_data, false, allocator)

		// 3. Predict Z_t
		Z_k := nn.sequential_forward(model.z_net, input)

		// 4. Euler-Maruyama step for X_{t+1}
		X_next_data := l.matrix_new(f64, n_paths, d, allocator)
		for i in 0 ..< n_paths {
			for j in 0 ..< d {
				x_val := X.data.data[i * d + j]
				dw_val := dW.data.data[i * d + j]
				sig_val := sigma[j]
				X_next_data.data[i * d + j] = x_val + r * x_val * dt + sig_val * x_val * dw_val
			}
		}
		X_next := t.tensor_new(X_next_data, false, allocator)

		// 5. Euler step for Y_{t+1} = Y_t + f(t,Y_t,Z_t)dt + Z_t dW_t
		// Under risk-neutral measure, f(t,y,z) = r * y
		f_k := t.tensor_scale(Y, r * dt)

		// Element-wise: Z_k * dW
		Z_k_dW := t.tensor_mul(Z_k, dW)

		// Sum over the d dimensions to get [n_paths, 1]
		ones_data := l.matrix_new(f64, d, 1, allocator)
		for j in 0 ..< d {ones_data.data[j] = 1.0}
		ones := t.tensor_new(ones_data, false, allocator)
		Z_dW := t.tensor_matmul(Z_k_dW, ones)

		Y_next := t.tensor_add(Y, f_k)
		Y_next = t.tensor_add(Y_next, Z_dW)

		// Advance state
		X = X_next
		Y = Y_next
	}

	return Y, X
}

// ============================================================================
// Loss Function
// ============================================================================
deep_bsde_bs_loss :: proc(
	model: ^DeepBSDEModel,
	S_0: []f64,
	K: f64,
	r: f64,
	sigma: []f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> ^t.Tensor {
	d := model.d

	// Run the full BSDE trajectory
	Y_T, X_T := deep_bsde_bs_forward(model, S_0, K, r, sigma, n_paths, n_steps, allocator)

	// Compute the terminal payoff (e.g., max(S_T - K, 0) for a Call option)
	// Note: If it's a basket option, we might sum or max the assets. Here we take the max of the first asset for simplicity.
	max_X_data := l.matrix_new(f64, n_paths, 1, allocator)
	for i in 0 ..< n_paths {
		// Assuming a single asset or taking the max across the basket
		max_val := X_T.data.data[i * d + 0]
		for j in 1 ..< d {
			if X_T.data.data[i * d + j] > max_val {
				max_val = X_T.data.data[i * d + j]
			}
		}
		max_X_data.data[i] = math.max(max_val - K, 0.0)
	}
	max_X := t.tensor_new(max_X_data, false, allocator)

	// BSDE Loss: E[ |Y_T - Payoff(S_T)|^2 ]
	diff := t.tensor_sub(Y_T, max_X)
	diff_sq := t.tensor_mul(diff, diff)
	loss := t.tensor_mean(diff_sq)

	return loss
}
