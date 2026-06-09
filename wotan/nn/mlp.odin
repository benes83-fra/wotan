package nn

import l "../linalg"
import t "../tensor"
import "core:mem"

// ============================================================================
// 1. Multi-Layer Perceptron (MLP)
// ============================================================================

MLP :: struct {
	layers:    [dynamic]LinearLayer,
	allocator: mem.Allocator,
}

// mlp_new creates a new MLP with the specified layer sizes.
// e.g., sizes = {2, 4, 1} creates:
// Layer 1: 2 inputs -> 4 outputs (followed by Relu)
// Layer 2: 4 inputs -> 1 output (no activation, raw output)
mlp_new :: proc(sizes: []int, allocator: mem.Allocator = context.allocator) -> MLP {
	net: MLP
	net.allocator = allocator
	net.layers = make([dynamic]LinearLayer, 0, allocator)

	for i in 0 ..< len(sizes) - 1 {
		layer := linear_layer_new(sizes[i], sizes[i + 1], allocator)
		append(&net.layers, layer)
	}

	return net
}

// mlp_forward performs the forward pass through the network
mlp_forward :: proc(net: ^MLP, x: ^t.Tensor) -> ^t.Tensor {
	out := x
	n_layers := len(net.layers)

	// Hidden layers: Linear + Relu
	for i in 0 ..< n_layers - 1 {
		out = linear_forward(&net.layers[i], out)
		out = t.tensor_relu(out)
	}

	// Output layer: Linear only (for regression)
	out = linear_forward(&net.layers[n_layers - 1], out)

	return out
}

// mlp_add_to_opt registers all weights and biases of the MLP with an optimizer
mlp_add_to_opt :: proc(net: ^MLP, opt: ^SGD) {
	for i in 0 ..< len(net.layers) {
		sgd_add_param(opt, net.layers[i].weights)
		sgd_add_param(opt, net.layers[i].bias)
	}
}

// mlp_free cleans up the MLP and all its layers
mlp_free :: proc(net: ^MLP) {
	for i in 0 ..< len(net.layers) {
		linear_layer_free(&net.layers[i])
	}
	delete(net.layers)
}
