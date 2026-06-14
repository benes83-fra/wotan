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
