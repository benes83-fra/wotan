package ml_finance

import nn "../nn"
import t "../tensor"
import "core:mem"

// ============================================================================
// 1. Simple MLP Volatility Forecaster (Unchanged)
// ============================================================================

VolatilityForecaster :: struct {
	fc1:       nn.LinearLayer,
	fc2:       nn.LinearLayer,
	fc3:       nn.LinearLayer,
	allocator: mem.Allocator,
}

volatility_forecaster_new :: proc(
	flat_input: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> VolatilityForecaster {
	model: VolatilityForecaster
	model.allocator = allocator
	model.fc1 = nn.linear_layer_new(flat_input, hidden_size, allocator)
	model.fc2 = nn.linear_layer_new(hidden_size, hidden_size, allocator)
	model.fc3 = nn.linear_layer_new(hidden_size, 1, allocator)
	return model
}

volatility_forecaster_free :: proc(model: ^VolatilityForecaster) {
	nn.linear_layer_free(&model.fc1)
	nn.linear_layer_free(&model.fc2)
	nn.linear_layer_free(&model.fc3)
}

volatility_forecaster_forward :: proc(
	model: ^VolatilityForecaster,
	input: ^t.Tensor,
) -> ^t.Tensor {
	h1 := nn.linear_forward(&model.fc1, input)
	h1_act := t.tensor_relu(h1)

	h2 := nn.linear_forward(&model.fc2, h1_act)
	h2_act := t.tensor_relu(h2)

	out := nn.linear_forward(&model.fc3, h2_act)
	return out
}

// ============================================================================
// 2. Unified Sequence Volatility Forecaster (LSTM / GRU / Mamba)
// ============================================================================

// ✅ NEW: Configuration struct to hold the backend switch
LSTMVolaConfig :: struct {
	input_size:  int,
	hidden_size: int,
	seq_len:     int,
	seq_backend: nn.SeqBackend, // ✅ .LSTM, .GRU, or .Mamba
}

LSTMVolatilityForecaster :: struct {
	seq_layer: ^nn.Sequential, // ✅ Replaces hardcoded lstm: nn.LSTMLayer
	fc1:       nn.LinearLayer,
	fc2:       nn.LinearLayer,
	config:    LSTMVolaConfig,
	allocator: mem.Allocator,
}

lstm_volatility_forecaster_new :: proc(
	config: LSTMVolaConfig,
	allocator: mem.Allocator = context.allocator,
) -> LSTMVolatilityForecaster {
	model: LSTMVolatilityForecaster
	model.allocator = allocator
	model.config = config

	// ✅ Unified Sequence Layer (LSTM / GRU / Mamba)
	model.seq_layer = nn.sequential_new(allocator)
	nn.sequential_add_seq_block(
		model.seq_layer,
		config.seq_backend,
		config.input_size,
		config.hidden_size,
		16, // d_state for Mamba
		allocator,
	)

	model.fc1 = nn.linear_layer_new(
		config.seq_len * config.hidden_size,
		config.hidden_size,
		allocator,
	)
	model.fc2 = nn.linear_layer_new(config.hidden_size, 1, allocator)
	return model
}

lstm_volatility_forecaster_free :: proc(model: ^LSTMVolatilityForecaster) {
	if model.seq_layer != nil {nn.sequential_free(model.seq_layer)}
	nn.linear_layer_free(&model.fc1)
	nn.linear_layer_free(&model.fc2)
}

// ✅ MASSIVE API SIMPLIFICATION:
// We no longer need to pass h_0 and c_0! sequential_forward handles
// zero-initialization of hidden states internally based on the backend.
lstm_volatility_forecaster_forward :: proc(
	model: ^LSTMVolatilityForecaster,
	input: ^t.Tensor,
) -> ^t.Tensor {
	// 1. Sequence Layer: [batch, seq_len, input_size] → [batch, seq_len, hidden_size]
	lstm_out := nn.sequential_forward(model.seq_layer, input)

	// 2. Flatten: [batch, seq_len, hidden_size, 1] → [batch, seq_len*hidden_size, 1, 1]
	flat := t.tensor_flatten(lstm_out)

	// 3. FC1 + ReLU: [batch, seq_len*hidden_size] → [batch, hidden_size]
	h1 := nn.linear_forward(&model.fc1, flat)
	h1_act := t.tensor_relu(h1)

	// 4. FC2: [batch, hidden_size] → [batch, 1]
	out := nn.linear_forward(&model.fc2, h1_act)

	return out
}

// ✅ NEW: Centralized Optimizer Registration
lstm_volatility_add_to_optimizer :: proc(model: ^LSTMVolatilityForecaster, opt: ^nn.Adam) {
	// One line replaces the manual weight extraction!
	nn.sequential_add_to_adam(model.seq_layer, opt)

	nn.adam_add_param(opt, model.fc1.weights)
	if model.fc1.bias != nil {nn.adam_add_param(opt, model.fc1.bias)}

	nn.adam_add_param(opt, model.fc2.weights)
	if model.fc2.bias != nil {nn.adam_add_param(opt, model.fc2.bias)}
}
