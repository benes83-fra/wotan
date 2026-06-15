package nn

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Activation Enum
// ============================================================================

Activation :: enum {
	None,
	ReLU,
	Sigmoid,
	Tanh,
	LeakyReLU,
}

// apply_activation applies an activation function to a tensor
apply_activation :: proc(x: ^t.Tensor, act: Activation) -> ^t.Tensor {
	switch act {
	case .None:
		return x
	case .ReLU:
		return t.tensor_relu(x)
	case .Sigmoid:
		return t.tensor_sigmoid(x)
	case .Tanh:
		return t.tensor_tanh(x)
	case .LeakyReLU:
		return t.tensor_leaky_relu(x, 0.01)
	}
	return x
}

// ============================================================================
// Conv2d Layer
// ============================================================================

Conv2dLayer :: struct {
	weight:       ^t.Tensor, // (C_out, C_in, kH, kW)
	bias:         ^t.Tensor, // (C_out,) or nil
	in_channels:  int,
	out_channels: int,
	kernel_size:  int,
	stride:       int,
	padding:      int,
}

conv2d_layer_new :: proc(
	in_channels: int,
	out_channels: int,
	kernel_size: int,
	stride: int = 1,
	padding: int = 0,
	use_bias: bool = true,
	allocator: mem.Allocator = context.allocator,
) -> Conv2dLayer {
	layer: Conv2dLayer
	layer.in_channels = in_channels
	layer.out_channels = out_channels
	layer.kernel_size = kernel_size
	layer.stride = stride
	layer.padding = padding

	// Initialize weights with Xavier initialization
	col_w := in_channels * kernel_size * kernel_size
	w_data := l.matrix_new(f64, out_channels, col_w, allocator)
	fan_in := in_channels * kernel_size * kernel_size
	fan_out := out_channels * kernel_size * kernel_size
	limit := math.sqrt(6.0 / f64(fan_in + fan_out))

	for i in 0 ..< len(w_data.data) {
		w_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit
	}

	layer.weight = t.tensor_new(w_data, true, allocator)
	layer.weight.shape = [4]int{out_channels, in_channels, kernel_size, kernel_size}

	// Initialize bias to zeros
	if use_bias {
		b_data := l.matrix_new(f64, 1, out_channels, allocator)
		layer.bias = t.tensor_new(b_data, true, allocator)
	}

	return layer
}

conv2d_layer_forward :: proc(layer: ^Conv2dLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding)
}

conv2d_layer_free :: proc(layer: ^Conv2dLayer) {
	if layer.weight != nil {t.tensor_free(layer.weight)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

// ============================================================================
// MaxPool2d Layer
// ============================================================================

MaxPool2dLayer :: struct {
	kernel_size: int,
	stride:      int,
}

maxpool2d_layer_new :: proc(kernel_size: int, stride: int = 0) -> MaxPool2dLayer {
	layer: MaxPool2dLayer
	layer.kernel_size = kernel_size
	layer.stride = stride == 0 ? kernel_size : stride
	return layer
}

maxpool2d_layer_forward :: proc(layer: ^MaxPool2dLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_max_pool2d(x, layer.kernel_size, layer.kernel_size, layer.stride)
}
AvgPool2dLayer :: struct {
	kernel_size: int,
	stride:      int,
}

avgpool2d_layer_new :: proc(kernel_size: int, stride: int = 0) -> AvgPool2dLayer {
	layer: AvgPool2dLayer
	layer.kernel_size = kernel_size
	layer.stride = stride == 0 ? kernel_size : stride
	return layer
}

// ============================================================================
// Dropout Layer (NEW)
// ============================================================================

DropoutLayer :: struct {
	drop_prob: f64,
}

dropout_layer_new :: proc(drop_prob: f64 = 0.5) -> DropoutLayer {
	layer: DropoutLayer
	layer.drop_prob = drop_prob
	return layer
}

BatchNorm2dLayer :: struct {
	num_features: int,
	eps:          f64,
	momentum:     f64,
	weight:       ^t.Tensor, // gamma (scale)
	bias:         ^t.Tensor, // beta (shift)
	running_mean: ^t.Tensor, // Saved for eval mode
	running_var:  ^t.Tensor, // Saved for eval mode
}

batch_norm_2d_layer_new :: proc(
	num_features: int,
	eps: f64 = 1e-5,
	momentum: f64 = 0.1,
	allocator: mem.Allocator = context.allocator,
) -> BatchNorm2dLayer {
	layer: BatchNorm2dLayer
	layer.num_features = num_features
	layer.eps = eps
	layer.momentum = momentum

	// Initialize weight (gamma) to 1.0
	w_data := l.matrix_new(f64, 1, num_features, allocator)
	for i in 0 ..< num_features {w_data.data[i] = 1.0}
	layer.weight = t.tensor_new(w_data, true, allocator)

	// Initialize bias (beta) to 0.0
	b_data := l.matrix_new(f64, 1, num_features, allocator)
	// (already zeroed by matrix_new)
	layer.bias = t.tensor_new(b_data, true, allocator)

	// Running stats (not requiring grad)
	rm_data := l.matrix_new(f64, 1, num_features, allocator)
	layer.running_mean = t.tensor_new(rm_data, false, allocator)

	rv_data := l.matrix_new(f64, 1, num_features, allocator)
	layer.running_var = t.tensor_new(rv_data, false, allocator)

	return layer
}

batch_norm_2d_layer_free :: proc(layer: ^BatchNorm2dLayer) {
	if layer.weight != nil {t.tensor_free(layer.weight)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
	if layer.running_mean != nil {t.tensor_free(layer.running_mean)}
	if layer.running_var != nil {t.tensor_free(layer.running_var)}
}
// ============================================================================
// RNN Layer
// ============================================================================

RNNLayer :: struct {
	input_size:  int,
	hidden_size: int,
	w_ih:        ^t.Tensor, // [input_size, hidden_size]
	w_hh:        ^t.Tensor, // [hidden_size, hidden_size]
	bias:        ^t.Tensor, // [1, hidden_size]
}

rnn_layer_new :: proc(
	input_size: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> RNNLayer {
	layer: RNNLayer
	layer.input_size = input_size
	layer.hidden_size = hidden_size

	// Xavier initialization for w_ih
	w_ih_data := l.matrix_new(f64, input_size, hidden_size, allocator)
	limit_ih := math.sqrt(6.0 / f64(input_size + hidden_size))
	for i in 0 ..< len(w_ih_data.data) {
		w_ih_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_ih
	}
	layer.w_ih = t.tensor_new(w_ih_data, true, allocator)
	// ✅ FIX: Set shape so shape[0]=input_size, shape[1]=hidden_size
	layer.w_ih.shape = [4]int{input_size, hidden_size, 1, 1}

	// Orthogonal/Xavier initialization for w_hh
	w_hh_data := l.matrix_new(f64, hidden_size, hidden_size, allocator)
	limit_hh := math.sqrt(6.0 / f64(hidden_size + hidden_size))
	for i in 0 ..< len(w_hh_data.data) {
		w_hh_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_hh
	}
	layer.w_hh = t.tensor_new(w_hh_data, true, allocator)
	// ✅ FIX: Set shape so shape[0]=hidden_size, shape[1]=hidden_size
	layer.w_hh.shape = [4]int{hidden_size, hidden_size, 1, 1}

	// Zero bias
	bias_data := l.matrix_new(f64, 1, hidden_size, allocator)
	layer.bias = t.tensor_new(bias_data, true, allocator)
	// ✅ FIX: Set shape so shape[1]=hidden_size
	layer.bias.shape = [4]int{1, hidden_size, 1, 1}

	return layer
}
rnn_layer_free :: proc(layer: ^RNNLayer) {
	if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
	if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

rnn_layer_forward :: proc(layer: ^RNNLayer, x: ^t.Tensor, h_0: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_rnn(x, h_0, layer.w_ih, layer.w_hh, layer.bias)
}
GRULayer :: struct {
	input_size:  int,
	hidden_size: int,
	w_ih:        ^t.Tensor,
	w_hh:        ^t.Tensor,
	bias:        ^t.Tensor,
}

gru_layer_new :: proc(
	input_size: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> GRULayer {
	layer: GRULayer
	layer.input_size = input_size
	layer.hidden_size = hidden_size
	H3 := 3 * hidden_size

	w_ih_data := l.matrix_new(f64, input_size, H3, allocator)
	limit_ih := math.sqrt(6.0 / f64(input_size + hidden_size))
	for i in 0 ..< len(w_ih_data.data) {
		w_ih_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_ih
	}
	layer.w_ih = t.tensor_new(w_ih_data, true, allocator)
	layer.w_ih.shape = [4]int{input_size, H3, 1, 1}

	w_hh_data := l.matrix_new(f64, hidden_size, H3, allocator)
	limit_hh := math.sqrt(6.0 / f64(hidden_size + hidden_size))
	for i in 0 ..< len(w_hh_data.data) {
		w_hh_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_hh
	}
	layer.w_hh = t.tensor_new(w_hh_data, true, allocator)
	layer.w_hh.shape = [4]int{hidden_size, H3, 1, 1}

	bias_data := l.matrix_new(f64, 1, H3, allocator)
	layer.bias = t.tensor_new(bias_data, true, allocator)
	layer.bias.shape = [4]int{1, H3, 1, 1}

	return layer
}

gru_layer_free :: proc(layer: ^GRULayer) {
	if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
	if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

gru_layer_forward :: proc(layer: ^GRULayer, x: ^t.Tensor, h_0: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_gru(x, h_0, layer.w_ih, layer.w_hh, layer.bias)
}
LSTMLayer :: struct {
	input_size:  int,
	hidden_size: int,
	w_ih:        ^t.Tensor,
	w_hh:        ^t.Tensor,
	bias:        ^t.Tensor,
}

lstm_layer_new :: proc(
	input_size: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> LSTMLayer {
	layer: LSTMLayer
	layer.input_size = input_size
	layer.hidden_size = hidden_size
	H4 := 4 * hidden_size

	w_ih_data := l.matrix_new(f64, input_size, H4, allocator)
	limit_ih := math.sqrt(6.0 / f64(input_size + hidden_size))
	for i in 0 ..< len(w_ih_data.data) {
		w_ih_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_ih
	}
	layer.w_ih = t.tensor_new(w_ih_data, true, allocator)
	layer.w_ih.shape = [4]int{input_size, H4, 1, 1}

	w_hh_data := l.matrix_new(f64, hidden_size, H4, allocator)
	limit_hh := math.sqrt(6.0 / f64(hidden_size + hidden_size))
	for i in 0 ..< len(w_hh_data.data) {
		w_hh_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_hh
	}
	layer.w_hh = t.tensor_new(w_hh_data, true, allocator)
	layer.w_hh.shape = [4]int{hidden_size, H4, 1, 1}

	bias_data := l.matrix_new(f64, 1, H4, allocator)
	layer.bias = t.tensor_new(bias_data, true, allocator)
	layer.bias.shape = [4]int{1, H4, 1, 1}

	return layer
}

lstm_layer_free :: proc(layer: ^LSTMLayer) {
	if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
	if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

lstm_layer_forward :: proc(
	layer: ^LSTMLayer,
	x: ^t.Tensor,
	h_0: ^t.Tensor,
	c_0: ^t.Tensor,
) -> ^t.Tensor {
	return t.tensor_lstm(x, h_0, c_0, layer.w_ih, layer.w_hh, layer.bias)
}
