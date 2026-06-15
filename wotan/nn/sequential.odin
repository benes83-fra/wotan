package nn

import la "../linalg"
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
	AvgPool2dLayer, // ✅ ADDED
	DropoutLayer, // ✅ ADDED
	BatchNorm2dLayer, // ✅ ADDED
	Activation,
	FlattenLayer,
	RNNLayer,
	GRULayer,
}

// ============================================================================
// Sequential Container
// ============================================================================

Sequential :: struct {
	layers:    [dynamic]Layer,
	training:  bool, // ✅ ADD THIS: Controls BatchNorm and Dropout behavior
	allocator: mem.Allocator,
}

sequential_new :: proc(allocator: mem.Allocator = context.allocator) -> ^Sequential {
	s := new(Sequential, allocator)
	s.layers = make([dynamic]Layer, 0, allocator)
	s.training = true // Default to training mode
	s.allocator = allocator
	return s
}

// ✅ ADD: Methods to switch modes
sequential_train :: proc(s: ^Sequential) {
	s.training = true
}

sequential_eval :: proc(s: ^Sequential) {
	s.training = false
}
sequential_add :: proc(s: ^Sequential, layers: ..Layer) {
	for layer in layers {
		append(&s.layers, layer)
	}
}

// sequential_forward applies all layers in sequence
// sequential_forward applies all layers in sequence
sequential_forward :: proc(s: ^Sequential, input: ^t.Tensor) -> ^t.Tensor {
	x := input
	for layer in s.layers {
		// ✅ FIX: Use '&l' to get a pointer to the layer variant
		switch &l in layer {
		case LinearLayer:
			x = t.tensor_matmul(x, l.weights)
			if l.bias != nil {
				x = t.tensor_add_bias(x, l.bias)
			}
		case Conv2dLayer:
			x = t.tensor_conv2d(x, l.weight, l.bias, l.stride, l.padding)
		case MaxPool2dLayer:
			x = t.tensor_max_pool2d(x, l.kernel_size, l.kernel_size, l.stride)
		case AvgPool2dLayer:
			x = t.tensor_avg_pool2d(x, l.kernel_size, l.kernel_size, l.stride)
		case FlattenLayer:
			x = t.tensor_flatten(x)
		case Activation:
			if l == .ReLU {x = t.tensor_relu(x)}
		// ... other activations ...
		case DropoutLayer:
			x = t.tensor_dropout(x, l.drop_prob, s.training)
		case BatchNorm2dLayer:
			x = t.tensor_batch_norm_2d(
				x,
				l.weight,
				l.bias,
				l.running_mean,
				l.running_var,
				s.training,
				l.momentum,
				l.eps,
			)
		case RNNLayer:
			batch := x.shape[0]
			hidden := l.hidden_size

			// ✅ FIX: Use 'la' for the linalg package!
			h_0_data := la.matrix_new(f64, 1, batch * hidden, x.allocator)
			h_0 := t.tensor_new(h_0_data, false, x.allocator)

			// ✅ FIX: 'l' is now a pointer, so this matches the ^RNNLayer signature
			x = rnn_layer_forward(&l, x, h_0)
			t.tensor_free(h_0)
		case GRULayer:
			batch := x.shape[0]
			hidden := l.hidden_size
			h_0_data := la.matrix_new(f64, 1, batch * hidden, x.allocator)
			h_0 := t.tensor_new(h_0_data, false, x.allocator)
			x = gru_layer_forward(&l, x, h_0)
			t.tensor_free(h_0)
		}
	}
	return x
}
// sequential_add_to_sgd registers all parameters with SGD optimizer
// sequential_add_to_sgd registers all parameters with SGD optimizer
sequential_add_to_sgd :: proc(seq: ^Sequential, opt: ^SGD) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			sgd_add_param(opt, l.weights)
			sgd_add_param(opt, l.bias)
		case Conv2dLayer:
			sgd_add_param(opt, l.weight)
			if l.bias != nil {sgd_add_param(opt, l.bias)}
		case BatchNorm2dLayer:
			sgd_add_param(opt, l.weight)
			sgd_add_param(opt, l.bias)

		// ✅ FIX: Added missing case for RNN
		case RNNLayer:
			sgd_add_param(opt, l.w_ih)
			sgd_add_param(opt, l.w_hh)
			sgd_add_param(opt, l.bias)

		case MaxPool2dLayer:
		case AvgPool2dLayer:
		case DropoutLayer:
		case Activation:
		case FlattenLayer:

		case GRULayer:
			sgd_add_param(opt, l.w_ih) // or sgd_add_param
			sgd_add_param(opt, l.w_hh)
			sgd_add_param(opt, l.bias)
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
		case BatchNorm2dLayer:
			adam_add_param(opt, l.weight)
			adam_add_param(opt, l.bias)
		case MaxPool2dLayer:
		// No parameters
		case AvgPool2dLayer:
		// No parameters
		case DropoutLayer:
		// No parameters
		case Activation:
		// No parameters
		case FlattenLayer:
		case GRULayer:
			adam_add_param(opt, l.w_ih) // or sgd_add_param
			adam_add_param(opt, l.w_hh)
			adam_add_param(opt, l.bias)
		// No parameters
		case RNNLayer:
			adam_add_param(opt, l.w_ih)
			adam_add_param(opt, l.w_hh)
			adam_add_param(opt, l.bias)
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
		case BatchNorm2dLayer:
			batch_norm_2d_layer_free(&l)
		case MaxPool2dLayer:
		// No heap allocations
		case AvgPool2dLayer:
		// No heap allocations
		case DropoutLayer:
		// No heap allocations
		case Activation:
		// No heap allocations
		case FlattenLayer:
		case GRULayer:
			gru_layer_free(&l)
		// No heap allocations
		case RNNLayer:
			rnn_layer_free(&l)
		}
	}
	delete(seq.layers)
}
// Procedure group for optimizer registration
sequential_add_to_opt :: proc {
	sequential_add_to_sgd,
	sequential_add_to_adam,
}
