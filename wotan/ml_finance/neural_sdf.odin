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
	T := z.shape[0]
	num_assets := model.config.num_assets
	eps := model.config.epsilon
	alloc := model.allocator

	// 1. Forward pass: raw [T, 1]
	raw := nn.sequential_forward(model.network, z)

	// 2. SDF = raw² + ε  (element-wise, [T, 1])
	raw_sq := t.tensor_mul(raw, raw)

	// Add epsilon for positivity floor
	eps_data := l.matrix_new(f64, T, 1, alloc)
	for i in 0 ..< T {
		eps_data.data[i] = eps
	}
	eps_tensor := t.tensor_new(eps_data, false, alloc)
	m := t.tensor_add(raw_sq, eps_tensor) // [T, 1]

	// 3. Compute Euler equation loss for each asset
	//    loss_i = (mean_t(M_t * R_{i,t}) - 1)²
	//    total_loss = Σ_i loss_i

	// We accumulate the total loss scalar manually
	total_loss_val := 0.0

	// For autograd, we build the loss tensor step by step
	// Start with a zero loss tensor [1,1]
	loss_data := l.matrix_new(f64, 1, 1, alloc)
	loss_data.data[0] = 0.0
	total_loss := t.tensor_new(loss_data, false, alloc)
	total_loss.shape = [4]int{1, 1, 1, 1}

	for asset in 0 ..< num_assets {
		// Extract column `asset` from r -> r_col [T, 1]
		r_col_data := l.matrix_new(f64, T, 1, alloc)
		for row in 0 ..< T {
			r_col_data.data[row] = r.data.data[row * num_assets + asset]
		}
		r_col := t.tensor_new(r_col_data, false, alloc)
		r_col.shape = [4]int{T, 1, 1, 1}

		// M * R_i  (element-wise, [T, 1])
		mr := t.tensor_mul(m, r_col)

		// mean(M * R_i)  -> scalar [1, 1]
		mean_mr := t.tensor_mean(mr)

		// mean(M * R_i) - 1
		one_data := l.matrix_new(f64, 1, 1, alloc)
		one_data.data[0] = 1.0
		one_tensor := t.tensor_new(one_data, false, alloc)
		one_tensor.shape = [4]int{1, 1, 1, 1}

		diff := t.tensor_sub(mean_mr, one_tensor)

		// (mean(M * R_i) - 1)²
		diff_sq := t.tensor_mul(diff, diff)

		// Accumulate into total loss
		total_loss = t.tensor_add(total_loss, diff_sq)

		total_loss_val += diff_sq.data.data[0]
	}

	// 4. Backward pass
	t.tensor_backward(total_loss)
	nn.adam_step(opt)

	// 5. Cleanup
	t.tensor_free_graph(total_loss)

	// Free the epsilon tensor (not part of the graph since we used tensor_add)
	l.matrix_free(&eps_data)
	t.tensor_free(eps_tensor)

	return total_loss_val
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
