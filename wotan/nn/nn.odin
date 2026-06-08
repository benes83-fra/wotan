package nn

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// 1. Linear Layer (Dense Layer)
// ============================================================================

LinearLayer :: struct {
	weights:      ^t.Tensor, // [in_features, out_features]
	bias:         ^t.Tensor, // [1, out_features]
	in_features:  int,
	out_features: int,
}

// linear_layer_new creates a new linear layer.
// For now, we initialize weights to small random values and bias to zero.
linear_layer_new :: proc(
	in_features: int,
	out_features: int,
	allocator: mem.Allocator = context.allocator,
) -> LinearLayer {
	layer: LinearLayer
	layer.in_features = in_features
	layer.out_features = out_features

	// Initialize Weights (Xavier/Glorot initialization simplified)
	// We use a small uniform random distribution
	w_data := l.matrix_new(f64, in_features, out_features, allocator)
	limit := math.sqrt(2.0 / f64(in_features + out_features))
	for i in 0 ..< len(w_data.data) {
		// Random between -limit and limit
		w_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit
	}
	layer.weights = t.tensor_new(w_data, true, allocator)

	// Initialize Bias to zeros
	b_data := l.matrix_new(f64, 1, out_features, allocator)
	// (matrix_new usually zeroes memory, but let's be explicit if needed)
	layer.bias = t.tensor_new(b_data, true, allocator)

	return layer
}

// linear_forward performs the forward pass: Y = X @ W + b
// X: [batch_size, in_features]
// W: [in_features, out_features]
// b: [1, out_features]
// Y: [batch_size, out_features]
linear_forward :: proc(layer: ^LinearLayer, x: ^t.Tensor) -> ^t.Tensor {
	// 1. Matrix Multiplication: X @ W
	out := t.tensor_matmul(x, layer.weights)

	// 2. Add Bias: (X @ W) + b
	out = t.tensor_add_bias(out, layer.bias)

	return out
}

linear_layer_free :: proc(layer: ^LinearLayer) {
	if layer.weights != nil {t.tensor_free(layer.weights)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}
