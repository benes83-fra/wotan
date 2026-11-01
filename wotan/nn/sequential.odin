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
	GATLayer,
	MambaLayer,
	MoELayer,
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

// sequential_forward applies all layers in sequence. Recurrent and Mamba
// layers start from a zero state.
sequential_forward :: proc(s: ^Sequential, input: ^t.Tensor) -> ^t.Tensor {
	x := input
	for &layer in s.layers {
		switch &l in layer {
		case LinearLayer:
			x = linear_forward(&l, x)
		case Conv2dLayer:
			x = conv2d_layer_forward(&l, x)
		case MaxPool2dLayer:
			x = maxpool2d_layer_forward(&l, x)
		case AvgPool2dLayer:
			x = t.tensor_avg_pool2d(x, l.kernel_size, l.kernel_size, l.stride)
		case FlattenLayer:
			x = t.tensor_flatten(x)
		case Activation:
			x = apply_activation(x, l)
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
			n := x.shape[0] * l.hidden_size
			h_0 := _constant(n, {1, n, 1, 1}, x.allocator)
			x = rnn_layer_forward(&l, x, h_0)
			_constant_done(h_0, x)
		case GRULayer:
			n := x.shape[0] * l.hidden_size
			h_0 := _constant(n, {1, n, 1, 1}, x.allocator)
			x = gru_layer_forward(&l, x, h_0)
			_constant_done(h_0, x)
		case LSTMLayer:
			n := x.shape[0] * l.hidden_size
			h_0 := _constant(n, {1, n, 1, 1}, x.allocator)
			c_0 := _constant(n, {1, n, 1, 1}, x.allocator)
			x = lstm_layer_forward(&l, x, h_0, c_0)
			_constant_done(h_0, x)
			_constant_done(c_0, x)
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
		case GATLayer:
			x = gat_layer_forward(&l, x, l.adjacency)
		case MoELayer:
			x = moe_layer_forward(&l, x, x.allocator)
		case MambaLayer:
			batch, d_model := x.shape[0], x.shape[2]
			h_0 := _constant(
				batch * d_model * l.d_state,
				{batch, d_model, l.d_state, 1},
				x.allocator,
			)
			x = mamba_layer_forward(&l, x, h_0)
			_constant_done(h_0, x)
		}
	}
	return x
}

// ============================================================================
// Parameters
// ============================================================================
// Trainable tensors (requires_grad) of a layer in a fixed order; checkpoints
// store optimizer moments in this order.

layer_parameters :: proc(layer: ^Layer, out: ^[dynamic]^t.Tensor) {
	add :: proc(out: ^[dynamic]^t.Tensor, p: ^t.Tensor) {
		if p != nil && p.requires_grad {append(out, p)}
	}
	add_linear :: proc(out: ^[dynamic]^t.Tensor, lin: ^LinearLayer) {
		add(out, lin.weights)
		add(out, lin.bias)
	}
	add_mha :: proc(out: ^[dynamic]^t.Tensor, mha: ^MultiHeadAttentionLayer) {
		add_linear(out, &mha.q_proj)
		add_linear(out, &mha.k_proj)
		add_linear(out, &mha.v_proj)
		add_linear(out, &mha.out_proj)
	}
	add_ffn :: proc(out: ^[dynamic]^t.Tensor, ffn: ^FFNLayer) {
		add_linear(out, &ffn.fc1)
		add_linear(out, &ffn.fc2)
	}
	add_block :: proc(out: ^[dynamic]^t.Tensor, b: ^TransformerEncoderBlock) {
		add(out, b.ln1.gamma)
		add(out, b.ln1.beta)
		add_mha(out, &b.mha)
		add(out, b.ln2.gamma)
		add(out, b.ln2.beta)
		add_ffn(out, &b.ffn)
	}

	switch &l in layer {
	case LinearLayer:
		add_linear(out, &l)
	case Conv2dLayer:
		add(out, l.weight)
		add(out, l.bias)
	case BatchNorm2dLayer:
		add(out, l.weight)
		add(out, l.bias)
	case LayerNormLayer:
		add(out, l.gamma)
		add(out, l.beta)
	case EmbeddingLayer:
		add(out, l.weight)
	case RNNLayer:
		add(out, l.w_ih)
		add(out, l.w_hh)
		add(out, l.bias)
	case GRULayer:
		add(out, l.w_ih)
		add(out, l.w_hh)
		add(out, l.bias)
	case LSTMLayer:
		add(out, l.w_ih)
		add(out, l.w_hh)
		add(out, l.bias)
	case MultiHeadAttentionLayer:
		add_mha(out, &l)
	case FFNLayer:
		add_ffn(out, &l)
	case TransformerEncoderBlock:
		add_block(out, &l)
	case TransformerEncoder:
		for &block in l.blocks {add_block(out, &block)}
	case GATLayer:
		add_linear(out, &l.linear)
		add_mha(out, &l.mha)
	case MambaLayer:
		add_linear(out, &l.proj_x)
		add_linear(out, &l.proj_B)
		add_linear(out, &l.proj_C)
		add_linear(out, &l.proj_Delta)
		add_linear(out, &l.proj_out)
		add(out, l.A)
		add(out, l.D)
	case MoELayer:
		add_linear(out, &l.router)
		for i in 0 ..< l.num_experts {add_ffn(out, &l.experts[i])}
	case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
	// No trainable parameters
	}
}

// All trainable tensors of the model, layer by layer
sequential_parameters :: proc(
	seq: ^Sequential,
	allocator: mem.Allocator = context.allocator,
) -> [dynamic]^t.Tensor {
	params := make([dynamic]^t.Tensor, 0, 16, allocator)
	for &layer in seq.layers {layer_parameters(&layer, &params)}
	return params
}

// sequential_add_to_sgd registers all trainable parameters with SGD
sequential_add_to_sgd :: proc(seq: ^Sequential, opt: ^SGD) {
	params := sequential_parameters(seq, context.temp_allocator)
	defer delete(params)
	for p in params {sgd_add_param(opt, p)}
}

// sequential_add_to_adam registers all trainable parameters with Adam
sequential_add_to_adam :: proc(seq: ^Sequential, opt: ^Adam) {
	params := sequential_parameters(seq, context.temp_allocator)
	defer delete(params)
	for p in params {adam_add_param(opt, p)}
}

// sequential_free cleans up all layers
sequential_free :: proc(seq: ^Sequential) {
	for &layer in seq.layers {
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
		case GATLayer:
			gat_layer_free(&l)
		case MoELayer:
			moe_layer_free(&l)
		case MambaLayer:
			mamba_layer_free(&l)
		}

	}

	delete(seq.layers)
	free(seq, seq.allocator)
}
// Procedure group for optimizer registration
sequential_add_to_opt :: proc {
	sequential_add_to_sgd,
	sequential_add_to_adam,
}
sequential_replace_last_layer :: proc(
	seq: ^Sequential,
	new_layer: Layer,
	allocator: mem.Allocator = context.allocator,
) {
	if len(seq.layers) == 0 {return}

	last_idx := len(seq.layers) - 1
	old_layer := &seq.layers[last_idx]

	// Re-use the exact same cleanup logic from sequential_free to prevent memory leaks
	switch &l in old_layer {
	case LinearLayer:
		linear_layer_free(&l)
	case Conv2dLayer:
		conv2d_layer_free(&l)
	case BatchNorm2dLayer:
		batch_norm_2d_layer_free(&l)
	case GRULayer:
		gru_layer_free(&l)
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
	case GATLayer:
		gat_layer_free(&l)
	case MambaLayer:
		mamba_layer_free(&l)
	case MoELayer:
		moe_layer_free(&l)
	case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
	// No heap allocations to free for these marker/stateless layers
	}

	// Replace with new layer
	seq.layers[last_idx] = new_layer
}
