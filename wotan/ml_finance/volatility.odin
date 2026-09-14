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
