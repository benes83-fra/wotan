package ml_finance

import nn "../nn"
import t "../tensor"
import "core:mem"

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
// LSTMVolatilityForecaster predicts next-period volatility from a sequence
// of [realized_vol, vix_proxy, sentiment] observations.
//
// Architecture:
//   LSTM(input_size → hidden_size) over seq_len timesteps
//   Flatten(seq_len × hidden_size)
//   Linear(seq_len × hidden_size → hidden_size) + ReLU
//   Linear(hidden_size → 1)
LSTMVolatilityForecaster :: struct {
	lstm:      nn.LSTMLayer,
	fc1:       nn.LinearLayer, // seq_len * hidden_size → hidden_size
	fc2:       nn.LinearLayer, // hidden_size → 1
	seq_len:   int,
	allocator: mem.Allocator,
}

lstm_volatility_forecaster_new :: proc(
	input_size: int,
	hidden_size: int,
	seq_len: int,
	allocator: mem.Allocator = context.allocator,
) -> LSTMVolatilityForecaster {
	model: LSTMVolatilityForecaster
	model.allocator = allocator
	model.seq_len = seq_len
	model.lstm = nn.lstm_layer_new(input_size, hidden_size, allocator)
	model.fc1 = nn.linear_layer_new(seq_len * hidden_size, hidden_size, allocator)
	model.fc2 = nn.linear_layer_new(hidden_size, 1, allocator)
	return model
}

lstm_volatility_forecaster_free :: proc(model: ^LSTMVolatilityForecaster) {
	nn.lstm_layer_free(&model.lstm)
	nn.linear_layer_free(&model.fc1)
	nn.linear_layer_free(&model.fc2)
}

// input:  shape [batch, seq_len, input_size, 1], data [1, batch*seq_len*input_size]
// h_0:    data length batch*hidden_size (zeros for stateless inference)
// c_0:    data length batch*hidden_size
// returns: shape [batch, 1, 1, 1], data [batch, 1]
lstm_volatility_forecaster_forward :: proc(
	model: ^LSTMVolatilityForecaster,
	input: ^t.Tensor,
	h_0: ^t.Tensor,
	c_0: ^t.Tensor,
) -> ^t.Tensor {
	// 1. LSTM: [batch, seq_len, input_size] → [batch, seq_len, hidden_size]
	//    output data: [1, batch * seq_len * hidden_size]
	lstm_out := nn.lstm_layer_forward(&model.lstm, input, h_0, c_0)

	// 2. Flatten: [batch, seq_len, hidden_size, 1] → [batch, seq_len*hidden_size, 1, 1]
	//    Uses existing .Flatten op with correct backward (copy)
	flat := t.tensor_flatten(lstm_out)

	// 3. FC1 + ReLU: [batch, seq_len*hidden_size] → [batch, hidden_size]
	h1 := nn.linear_forward(&model.fc1, flat)
	h1_act := t.tensor_relu(h1)

	// 4. FC2: [batch, hidden_size] → [batch, 1]
	out := nn.linear_forward(&model.fc2, h1_act)

	return out
}
