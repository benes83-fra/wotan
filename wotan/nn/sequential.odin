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
	LSTMLayer,
	EmbeddingLayer,
	MultiHeadAttentionLayer,
	LayerNormLayer,
	FFNLayer,
	TransformerEncoderBlock,
	TransformerEncoder,
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
		case LSTMLayer:
			batch := x.shape[0]
			hidden := l.hidden_size
			h_0_data := la.matrix_new(f64, 1, batch * hidden, x.allocator)
			c_0_data := la.matrix_new(f64, 1, batch * hidden, x.allocator)
			h_0 := t.tensor_new(h_0_data, false, x.allocator)
			c_0 := t.tensor_new(c_0_data, false, x.allocator)
			x = lstm_layer_forward(&l, x, h_0, c_0)
			t.tensor_free(h_0)
			t.tensor_free(c_0)
		case EmbeddingLayer:
			x = embedding_layer_forward(&l, x)
		case MultiHeadAttentionLayer:
			x = multi_head_attention_layer_forward(&l, x)
		case LayerNormLayer:
			x = layer_norm_layer_forward(&l, x)
		case FFNLayer:
			x = ffn_layer_forward(&l, x)
		case TransformerEncoderBlock:
			x = transformer_encoder_block_forward(&l, x)
		case TransformerEncoder:
			x = transformer_encoder_forward(&l, x)
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
		case LSTMLayer:
			sgd_add_param(opt, l.w_ih)
			sgd_add_param(opt, l.w_hh)
			sgd_add_param(opt, l.bias)
		case EmbeddingLayer:
			sgd_add_param(opt, l.weight) // or sgd_add_param
		case MultiHeadAttentionLayer:
			sgd_add_param(opt, l.q_proj.weights)
			sgd_add_param(opt, l.q_proj.bias)
			sgd_add_param(opt, l.k_proj.weights)
			sgd_add_param(opt, l.k_proj.bias)
			sgd_add_param(opt, l.v_proj.weights)
			sgd_add_param(opt, l.v_proj.bias)
			sgd_add_param(opt, l.out_proj.weights)
			sgd_add_param(opt, l.out_proj.bias)
		case LayerNormLayer:
			sgd_add_param(opt, l.gamma)
			sgd_add_param(opt, l.beta)
		case FFNLayer:
			sgd_add_param(opt, l.fc1.weights)
			sgd_add_param(opt, l.fc1.bias)
			sgd_add_param(opt, l.fc2.weights)
			sgd_add_param(opt, l.fc2.bias)
		case TransformerEncoderBlock:
			// MHA parameters
			sgd_add_param(opt, l.mha.q_proj.weights)
			sgd_add_param(opt, l.mha.q_proj.bias)
			sgd_add_param(opt, l.mha.k_proj.weights)
			sgd_add_param(opt, l.mha.k_proj.bias)
			sgd_add_param(opt, l.mha.v_proj.weights)
			sgd_add_param(opt, l.mha.v_proj.bias)
			sgd_add_param(opt, l.mha.out_proj.weights)
			sgd_add_param(opt, l.mha.out_proj.bias)

			// FFN parameters
			sgd_add_param(opt, l.ffn.fc1.weights)
			sgd_add_param(opt, l.ffn.fc1.bias)
			sgd_add_param(opt, l.ffn.fc2.weights)
			sgd_add_param(opt, l.ffn.fc2.bias)

			// LayerNorm parameters
			sgd_add_param(opt, l.ln1.gamma)
			sgd_add_param(opt, l.ln1.beta)
			sgd_add_param(opt, l.ln2.gamma)
			sgd_add_param(opt, l.ln2.beta)
		case TransformerEncoder:
			// Register all parameters from all blocks
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]
				// MHA parameters
				sgd_add_param(opt, block.mha.q_proj.weights)
				sgd_add_param(opt, block.mha.q_proj.bias)
				sgd_add_param(opt, block.mha.k_proj.weights)
				sgd_add_param(opt, block.mha.k_proj.bias)
				sgd_add_param(opt, block.mha.v_proj.weights)
				sgd_add_param(opt, block.mha.v_proj.bias)
				sgd_add_param(opt, block.mha.out_proj.weights)
				sgd_add_param(opt, block.mha.out_proj.bias)

				// FFN parameters
				sgd_add_param(opt, block.ffn.fc1.weights)
				sgd_add_param(opt, block.ffn.fc1.bias)
				sgd_add_param(opt, block.ffn.fc2.weights)
				sgd_add_param(opt, block.ffn.fc2.bias)

				// LayerNorm parameters
				sgd_add_param(opt, block.ln1.gamma)
				sgd_add_param(opt, block.ln1.beta)
				sgd_add_param(opt, block.ln2.gamma)
				sgd_add_param(opt, block.ln2.beta)
			}
		}
	}
}

// sequential_add_to_adam registers all parameters with Adam optimizer
sequential_add_to_adam :: proc(seq: ^Sequential, opt: ^Adam) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			if l.weights.requires_grad {
				adam_add_param(opt, l.weights)
			}
			if l.bias != nil && l.bias.requires_grad {
				adam_add_param(opt, l.bias)
			}
		case Conv2dLayer:
			if l.weight.requires_grad {
				adam_add_param(opt, l.weight)
			}
			if l.bias != nil && l.bias.requires_grad {
				adam_add_param(opt, l.bias)
			}
		case BatchNorm2dLayer:
			if l.weight.requires_grad {
				adam_add_param(opt, l.weight)
			}
			if l.bias.requires_grad {
				adam_add_param(opt, l.bias)
			}
		case LayerNormLayer:
			if l.gamma.requires_grad {
				adam_add_param(opt, l.gamma)
			}
			if l.beta.requires_grad {
				adam_add_param(opt, l.beta)
			}
		case EmbeddingLayer:
			if l.weight.requires_grad {
				adam_add_param(opt, l.weight)
			}
		case RNNLayer:
			if l.w_ih.requires_grad {
				adam_add_param(opt, l.w_ih)
			}
			if l.w_hh.requires_grad {
				adam_add_param(opt, l.w_hh)
			}
			if l.bias.requires_grad {
				adam_add_param(opt, l.bias)
			}
		case GRULayer:
			if l.w_ih.requires_grad {
				adam_add_param(opt, l.w_ih)
			}
			if l.w_hh.requires_grad {
				adam_add_param(opt, l.w_hh)
			}
			if l.bias.requires_grad {
				adam_add_param(opt, l.bias)
			}
		case LSTMLayer:
			if l.w_ih.requires_grad {
				adam_add_param(opt, l.w_ih)
			}
			if l.w_hh.requires_grad {
				adam_add_param(opt, l.w_hh)
			}
			if l.bias.requires_grad {
				adam_add_param(opt, l.bias)
			}
		case MultiHeadAttentionLayer:
			if l.q_proj.weights.requires_grad {adam_add_param(opt, l.q_proj.weights)}
			if l.q_proj.bias != nil &&
			   l.q_proj.bias.requires_grad {adam_add_param(opt, l.q_proj.bias)}
			if l.k_proj.weights.requires_grad {adam_add_param(opt, l.k_proj.weights)}
			if l.k_proj.bias != nil &&
			   l.k_proj.bias.requires_grad {adam_add_param(opt, l.k_proj.bias)}
			if l.v_proj.weights.requires_grad {adam_add_param(opt, l.v_proj.weights)}
			if l.v_proj.bias != nil &&
			   l.v_proj.bias.requires_grad {adam_add_param(opt, l.v_proj.bias)}
			if l.out_proj.weights.requires_grad {adam_add_param(opt, l.out_proj.weights)}
			if l.out_proj.bias != nil &&
			   l.out_proj.bias.requires_grad {adam_add_param(opt, l.out_proj.bias)}
		case FFNLayer:
			if l.fc1.weights.requires_grad {adam_add_param(opt, l.fc1.weights)}
			if l.fc1.bias != nil && l.fc1.bias.requires_grad {adam_add_param(opt, l.fc1.bias)}
			if l.fc2.weights.requires_grad {adam_add_param(opt, l.fc2.weights)}
			if l.fc2.bias != nil && l.fc2.bias.requires_grad {adam_add_param(opt, l.fc2.bias)}
		case TransformerEncoderBlock:
			if l.ln1.gamma.requires_grad {adam_add_param(opt, l.ln1.gamma)}
			if l.ln1.beta.requires_grad {adam_add_param(opt, l.ln1.beta)}
			if l.mha.q_proj.weights.requires_grad {adam_add_param(opt, l.mha.q_proj.weights)}
			if l.mha.q_proj.bias != nil &&
			   l.mha.q_proj.bias.requires_grad {adam_add_param(opt, l.mha.q_proj.bias)}
			if l.mha.k_proj.weights.requires_grad {adam_add_param(opt, l.mha.k_proj.weights)}
			if l.mha.k_proj.bias != nil &&
			   l.mha.k_proj.bias.requires_grad {adam_add_param(opt, l.mha.k_proj.bias)}
			if l.mha.v_proj.weights.requires_grad {adam_add_param(opt, l.mha.v_proj.weights)}
			if l.mha.v_proj.bias != nil &&
			   l.mha.v_proj.bias.requires_grad {adam_add_param(opt, l.mha.v_proj.bias)}
			if l.mha.out_proj.weights.requires_grad {adam_add_param(opt, l.mha.out_proj.weights)}
			if l.mha.out_proj.bias != nil &&
			   l.mha.out_proj.bias.requires_grad {adam_add_param(opt, l.mha.out_proj.bias)}
			if l.ln2.gamma.requires_grad {adam_add_param(opt, l.ln2.gamma)}
			if l.ln2.beta.requires_grad {adam_add_param(opt, l.ln2.beta)}
			if l.ffn.fc1.weights.requires_grad {adam_add_param(opt, l.ffn.fc1.weights)}
			if l.ffn.fc1.bias != nil &&
			   l.ffn.fc1.bias.requires_grad {adam_add_param(opt, l.ffn.fc1.bias)}
			if l.ffn.fc2.weights.requires_grad {adam_add_param(opt, l.ffn.fc2.weights)}
			if l.ffn.fc2.bias != nil &&
			   l.ffn.fc2.bias.requires_grad {adam_add_param(opt, l.ffn.fc2.bias)}
		case TransformerEncoder:
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]
				if block.ln1.gamma.requires_grad {adam_add_param(opt, block.ln1.gamma)}
				if block.ln1.beta.requires_grad {adam_add_param(opt, block.ln1.beta)}
				if block.mha.q_proj.weights.requires_grad {adam_add_param(opt, block.mha.q_proj.weights)}
				if block.mha.q_proj.bias != nil &&
				   block.mha.q_proj.bias.requires_grad {adam_add_param(opt, block.mha.q_proj.bias)}
				if block.mha.k_proj.weights.requires_grad {adam_add_param(opt, block.mha.k_proj.weights)}
				if block.mha.k_proj.bias != nil &&
				   block.mha.k_proj.bias.requires_grad {adam_add_param(opt, block.mha.k_proj.bias)}
				if block.mha.v_proj.weights.requires_grad {adam_add_param(opt, block.mha.v_proj.weights)}
				if block.mha.v_proj.bias != nil &&
				   block.mha.v_proj.bias.requires_grad {adam_add_param(opt, block.mha.v_proj.bias)}
				if block.mha.out_proj.weights.requires_grad {adam_add_param(opt, block.mha.out_proj.weights)}
				if block.mha.out_proj.bias != nil &&
				   block.mha.out_proj.bias.requires_grad {adam_add_param(opt, block.mha.out_proj.bias)}
				if block.ln2.gamma.requires_grad {adam_add_param(opt, block.ln2.gamma)}
				if block.ln2.beta.requires_grad {adam_add_param(opt, block.ln2.beta)}
				if block.ffn.fc1.weights.requires_grad {adam_add_param(opt, block.ffn.fc1.weights)}
				if block.ffn.fc1.bias != nil &&
				   block.ffn.fc1.bias.requires_grad {adam_add_param(opt, block.ffn.fc1.bias)}
				if block.ffn.fc2.weights.requires_grad {adam_add_param(opt, block.ffn.fc2.weights)}
				if block.ffn.fc2.bias != nil &&
				   block.ffn.fc2.bias.requires_grad {adam_add_param(opt, block.ffn.fc2.bias)}
			}
		case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
		// No trainable parameters
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
		case LSTMLayer:
			lstm_layer_free(&l)
		case EmbeddingLayer:
			embedding_layer_free(&l)
		case MultiHeadAttentionLayer:
			multi_head_attention_layer_free(&l)
		case LayerNormLayer:
			layer_norm_layer_free(&l)
		case FFNLayer:
			ffn_layer_free(&l)
		case TransformerEncoderBlock:
			transformer_encoder_block_free(&l)
		case TransformerEncoder:
			transformer_encoder_free(&l)
		}
	}
	delete(seq.layers)
}
// Procedure group for optimizer registration
sequential_add_to_opt :: proc {
	sequential_add_to_sgd,
	sequential_add_to_adam,
}
sequential_replace_last_layer :: proc(
	seq: ^Sequential,
	new_layer: Layer,
	allocator: mem.Allocator,
) {
	if len(seq.layers) == 0 {return}

	// Free old last layer
	last_idx := len(seq.layers) - 1
	old_layer := &seq.layers[last_idx]
	#partial switch &l in old_layer {
	case LinearLayer:
		linear_layer_free(&l)
	case Conv2dLayer:
		conv2d_layer_free(&l)
	}

	// Replace with new layer
	seq.layers[last_idx] = new_layer
}
