package nn

import l "../linalg"
import t "../tensor"
import "core:mem"

MLP :: struct {
	layers:      [dynamic]LinearLayer,
	activations: [dynamic]Activation, // ✅ ADD THIS
	allocator:   mem.Allocator,
}

// mlp_new creates an MLP with configurable activations
mlp_new :: proc(
	sizes: []int,
	activation: Activation = .ReLU, // ✅ Configurable!
	allocator: mem.Allocator = context.allocator,
) -> MLP {
	net: MLP
	net.allocator = allocator
	net.layers = make([dynamic]LinearLayer, 0, allocator)
	net.activations = make([dynamic]Activation, 0, allocator)

	for i in 0 ..< len(sizes) - 1 {
		layer := linear_layer_new(sizes[i], sizes[i + 1], allocator)
		append(&net.layers, layer)

		// Apply activation to all layers except the last
		if i < len(sizes) - 2 {
			append(&net.activations, activation)
		} else {
			append(&net.activations, Activation.None) // No activation on output
		}
	}

	return net
}

// mlp_forward with configurable activations
mlp_forward :: proc(
	net: ^MLP,
	x: ^t.Tensor,
	drop_prob: f64 = 0.0,
	training: bool = true,
) -> ^t.Tensor {
	out := x
	n_layers := len(net.layers)

	for i in 0 ..< n_layers {
		out = linear_forward(&net.layers[i], out)

		// Apply the configured activation
		if i < n_layers - 1 {
			out = apply_activation(out, net.activations[i])

			if training && drop_prob > 0.0 {
				out = t.tensor_dropout(out, drop_prob, training)
			}
		}
	}

	return out
}

// Add parameters to optimizer (unchanged)
mlp_add_to_sgd :: proc(net: ^MLP, opt: ^SGD) {
	for i in 0 ..< len(net.layers) {
		sgd_add_param(opt, net.layers[i].weights)
		sgd_add_param(opt, net.layers[i].bias)
	}
}

mlp_add_to_adam :: proc(net: ^MLP, opt: ^Adam) {
	for i in 0 ..< len(net.layers) {
		adam_add_param(opt, net.layers[i].weights)
		adam_add_param(opt, net.layers[i].bias)
	}
}

mlp_add_to_opt :: proc {
	mlp_add_to_sgd,
	mlp_add_to_adam,
}

mlp_free :: proc(net: ^MLP) {
	for i in 0 ..< len(net.layers) {
		linear_layer_free(&net.layers[i])
	}
	delete(net.layers)
	delete(net.activations)
}
