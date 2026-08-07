package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Neural SDF: M_{t+1} = f_θ(Z_t)² + ε  (positivity via squaring)
// Euler equation: E[M_{t+1} · R_{i,t+1}] = 1  for all assets i
// Loss: Σ_i (mean_t(M_t · R_{i,t}) - 1)²
// ============================================================================

NeuralSDFConfig :: struct {
	input_size:  int, // Dimension of conditioning variables Z_t
	hidden_size: int, // Hidden layer width
	num_layers:  int, // Number of hidden layers
	num_assets:  int, // Number of assets to calibrate to
	epsilon:     f64, // Positivity floor (default 0.01)
}

NeuralSDF :: struct {
	network:   ^nn.Sequential,
	config:    NeuralSDFConfig,
	allocator: mem.Allocator,
}

// ============================================================================
// Initialization
// ============================================================================

neural_sdf_new :: proc(
	config: NeuralSDFConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^NeuralSDF {
	model := new(NeuralSDF, allocator)
	model.config = config
	model.allocator = allocator

	seq := nn.sequential_new(allocator)
	nn.sequential_add(seq, nn.linear_layer_new(config.input_size, config.hidden_size))
	nn.sequential_add(seq, nn.Activation.ReLU)

	for _ in 1 ..< config.num_layers {
		nn.sequential_add(seq, nn.linear_layer_new(config.hidden_size, config.hidden_size))
		nn.sequential_add(seq, nn.Activation.ReLU)
	}

	// Output: single scalar per time step (the raw SDF factor before squaring)
	nn.sequential_add(seq, nn.linear_layer_new(config.hidden_size, 1))

	model.network = seq
	return model
}

neural_sdf_free :: proc(model: ^NeuralSDF) {
	if model.network != nil {
		nn.sequential_free(model.network)
	}
	free(model, model.allocator)
}

// ============================================================================
// Training Step
// ============================================================================
// z: [T, input_size]  — conditioning variables
// r: [T, num_assets]  — gross returns (1 + net_return)
//
// The SDF is M_t = (f_θ(Z_t))² + ε, ensuring M > 0.
// Loss = Σ_i ( (1/T) Σ_t M_t · R_{i,t}  -  1 )²
neural_sdf_train_step :: proc(
	model: ^NeuralSDF,
	z: ^t.Tensor,
	r: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	n := z.shape[0]
	num_assets := model.config.num_assets
	eps := model.config.epsilon
	alloc := z.allocator

	// 1. Forward pass: M = f(Z)² + ε
	raw := nn.sequential_forward(model.network, z)
	raw_sq := t.tensor_mul(raw, raw)

	eps_data := l.matrix_new(f64, n, 1, alloc)
	for i in 0 ..< n {eps_data.data[i] = eps}
	eps_tensor := t.tensor_new(eps_data, false, alloc)
	m := t.tensor_add(raw_sq, eps_tensor)

	// 2. Euler equation loss: Σ_i (E[M·R_i] - 1)²
	euler_loss_data := l.matrix_new(f64, 1, 1, alloc)
	euler_loss_data.data[0] = 0.0
	euler_loss := t.tensor_new(euler_loss_data, false, alloc)

	for asset in 0 ..< num_assets {
		r_col_data := l.matrix_new(f64, n, 1, alloc)
		for row in 0 ..< n {
			r_col_data.data[row] = r.data.data[row * num_assets + asset]
		}
		r_col := t.tensor_new(r_col_data, false, alloc)

		mr := t.tensor_mul(m, r_col)
		mean_mr := t.tensor_mean(mr)

		one_data := l.matrix_new(f64, 1, 1, alloc)
		one_data.data[0] = 1.0
		one := t.tensor_new(one_data, false, alloc)

		diff := t.tensor_sub(mean_mr, one)
		diff_sq := t.tensor_mul(diff, diff)
		euler_loss = t.tensor_add(euler_loss, diff_sq)

		// ✅ FIX: DO NOT free intermediate tensors here!
		// They are part of the computation graph. Freeing them before
		// t.tensor_backward destroys their gradient matrices, causing
		// "empty gradient" warnings and "length mismatch" panics.
	}

	// 3. Regularization: Penalize variance of M
	m_mean := t.tensor_mean(m)

	m_mean_data := l.matrix_new(f64, n, 1, alloc)
	mean_val := m_mean.data.data[0]
	for i in 0 ..< n {
		m_mean_data.data[i] = mean_val
	}
	m_mean_tensor := t.tensor_new(m_mean_data, false, alloc)

	m_centered := t.tensor_sub(m, m_mean_tensor)
	m_centered_sq := t.tensor_mul(m_centered, m_centered)
	m_variance := t.tensor_mean(m_centered_sq)

	reg_strength := 1.0
	reg_loss := t.tensor_scale(m_variance, reg_strength)

	// Total loss = Euler loss + Regularization
	total_loss := t.tensor_add(euler_loss, reg_loss)

	// 4. Backward pass
	t.tensor_backward(total_loss)
	nn.adam_step(opt)

	// 5. Get loss value
	loss_val := total_loss.data.data[0]

	// 6. Cleanup the ENTIRE computation graph at once
	// This safely frees all intermediate nodes and their gradients
	t.tensor_free_graph(total_loss)

	return loss_val
}
// ============================================================================
// Evaluation: compute pricing error per asset
// ============================================================================

neural_sdf_evaluate :: proc(
	model: ^NeuralSDF,
	z: ^t.Tensor,
	r: ^t.Tensor,
	allocator: mem.Allocator,
) -> []f64 {
	T := z.shape[0]
	num_assets := model.config.num_assets
	eps := model.config.epsilon

	// Forward pass (no grad needed)
	raw := nn.sequential_forward(model.network, z)

	// SDF = raw² + ε
	pricing_errors := make([]f64, num_assets, allocator)

	for asset in 0 ..< num_assets {
		sum_mr := 0.0
		for row in 0 ..< T {
			raw_val := raw.data.data[row]
			m_val := raw_val * raw_val + eps
			r_val := r.data.data[row * num_assets + asset]
			sum_mr += m_val * r_val
		}
		mean_mr := sum_mr / f64(T)
		pricing_errors[asset] = math.abs(mean_mr - 1.0)
	}

	t.tensor_free(raw)
	return pricing_errors
}

// ============================================================================
// Save / Load
// ============================================================================

neural_sdf_save :: proc(model: ^NeuralSDF, path: string) -> bool {
	dummy_opt := nn.adam_new(0.001, allocator = model.allocator)
	defer nn.adam_free(&dummy_opt)
	nn.sequential_add_to_adam(model.network, &dummy_opt)
	return nn.save_checkpoint(model.network, &dummy_opt, path, 0, model.allocator)
}

neural_sdf_load :: proc(
	path: string,
	config: NeuralSDFConfig,
	allocator: mem.Allocator = context.allocator,
) -> (
	^NeuralSDF,
	bool,
) {
	loaded_model, loaded_opt, _, ok := nn.load_checkpoint(path, allocator)
	if !ok {
		return nil, false
	}
	if loaded_opt != nil {
		nn.adam_free(loaded_opt)
	}

	model := new(NeuralSDF, allocator)
	model.network = loaded_model
	model.config = config
	model.allocator = allocator
	return model, true
}
