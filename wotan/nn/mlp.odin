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
// ✅ Added drop_prob and training flag
mlp_forward :: proc(
	net: ^MLP,
	x: ^t.Tensor,
	drop_prob: f64 = 0.0,
	training: bool = true,
) -> ^t.Tensor {
	out := x
	n_layers := len(net.layers)

	for i in 0 ..< n_layers - 1 {
		out = linear_forward(&net.layers[i], out)
		out = t.tensor_relu(out)

		// ✅ Apply Dropout after ReLU if training and drop_prob > 0
		if training && drop_prob > 0.0 {
			out = t.tensor_dropout(out, drop_prob, training)
		}
	}

	out = linear_forward(&net.layers[n_layers - 1], out)
	return out
}

// ✅ OVERLOADED: Add parameters to SGD optimizer
mlp_add_to_sgd :: proc(net: ^MLP, opt: ^SGD) {
	for i in 0 ..< len(net.layers) {
		sgd_add_param(opt, net.layers[i].weights)
		sgd_add_param(opt, net.layers[i].bias)
	}
}

// ✅ OVERLOADED: Add parameters to Adam optimizer
mlp_add_to_adam :: proc(net: ^MLP, opt: ^Adam) {
	for i in 0 ..< len(net.layers) {
		adam_add_param(opt, net.layers[i].weights)
		adam_add_param(opt, net.layers[i].bias)
	}
}

// ✅ PROCEDURE GROUP: Allows calling mlp_add_to_opt with either optimizer type
mlp_add_to_opt :: proc {
	mlp_add_to_sgd,
	mlp_add_to_adam,
}

// mlp_free cleans up the MLP and all its layers
mlp_free :: proc(net: ^MLP) {
	for i in 0 ..< len(net.layers) {
		linear_layer_free(&net.layers[i])
	}
	delete(net.layers)
}
