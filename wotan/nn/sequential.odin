package nn

import t "../tensor"
import "core:mem"

// ============================================================================
// Layer Union Type
// ============================================================================

// FlattenLayer is a marker struct for the flatten operation
FlattenLayer :: struct {}

// Layer is a tagged union that can hold any layer type
Layer :: union {
	LinearLayer,
	Conv2dLayer,
	MaxPool2dLayer,
	Activation, // enum: ReLU, Sigmoid, Tanh, LeakyReLU, None
	FlattenLayer, // marker struct
}

// ============================================================================
// Sequential Container
// ============================================================================

Sequential :: struct {
	layers:    [dynamic]Layer,
	allocator: mem.Allocator,
}

// sequential_new creates an empty Sequential container
sequential_new :: proc(allocator: mem.Allocator = context.allocator) -> Sequential {
	return Sequential{layers = make([dynamic]Layer, 0, allocator), allocator = allocator}
}

// sequential_add adds one or more layers to the sequence
sequential_add :: proc(seq: ^Sequential, layers: ..Layer) {
	for layer in layers {
		append(&seq.layers, layer)
	}
}

// sequential_forward applies all layers in sequence
sequential_forward :: proc(seq: ^Sequential, x: ^t.Tensor) -> ^t.Tensor {
	out := x
	for layer in seq.layers {
		switch &l in layer {
		case LinearLayer:
			out = linear_forward(&l, out)
		case Conv2dLayer:
			out = conv2d_layer_forward(&l, out)
		case MaxPool2dLayer:
			out = maxpool2d_layer_forward(&l, out)
		case Activation:
			out = apply_activation(out, l)
		case FlattenLayer:
			out = t.tensor_flatten(out)
		}
	}
	return out
}
// sequential_add_to_sgd registers all parameters with SGD optimizer
sequential_add_to_sgd :: proc(seq: ^Sequential, opt: ^SGD) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			sgd_add_param(opt, l.weights)
			sgd_add_param(opt, l.bias)
		case Conv2dLayer:
			sgd_add_param(opt, l.weight)
			if l.bias != nil {
				sgd_add_param(opt, l.bias)
			}
		case MaxPool2dLayer:
		// No parameters
		case Activation:
		// No parameters
		case FlattenLayer:
		// No parameters
		}
	}
}

// sequential_add_to_adam registers all parameters with Adam optimizer
sequential_add_to_adam :: proc(seq: ^Sequential, opt: ^Adam) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			adam_add_param(opt, l.weights)
			adam_add_param(opt, l.bias)
		case Conv2dLayer:
			adam_add_param(opt, l.weight)
			if l.bias != nil {
				adam_add_param(opt, l.bias)
			}
		case MaxPool2dLayer:
		// No parameters
		case Activation:
		// No parameters
		case FlattenLayer:
		// No parameters
		}
	}
}

// sequential_free cleans up all layers
sequential_free :: proc(seq: ^Sequential) {
	for layer in seq.layers {
		switch &l in layer {
		case LinearLayer:
			linear_layer_free(&l)
		case Conv2dLayer:
			conv2d_layer_free(&l)
		case MaxPool2dLayer:
		// No heap allocations
		case Activation:
		// No heap allocations
		case FlattenLayer:
		// No heap allocations
		}
	}
	delete(seq.layers)
}
// Procedure group for optimizer registration
sequential_add_to_opt :: proc {
	sequential_add_to_sgd,
	sequential_add_to_adam,
}
