package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:mem"

// ============================================================================
// DeepLOB Configuration & Structures
// ============================================================================

DeepLOBConfig :: struct {
	time_steps:   int,
	price_levels: int,
	num_classes:  int,
	hidden_dim:   int,
	dropout_prob: f64,
}

DeepLOB :: struct {
	cnn:       ^nn.Sequential,
	inception: InceptionBlock,
	lstm:      nn.LSTMLayer,
	fc_head:   nn.LinearLayer,
	config:    DeepLOBConfig,
	allocator: mem.Allocator,
	training:  bool,
}

// ============================================================================
// Inception Block (Multi-Scale Feature Extraction)
// ============================================================================

InceptionBlock :: struct {
	b1_1x1: nn.Conv2dLayer,
	b2_1x1: nn.Conv2dLayer,
	b2_3x3: nn.Conv2dLayer,
	b3_1x1: nn.Conv2dLayer,
	b3_5x5: nn.Conv2dLayer,
}

inception_block_new :: proc(
	in_channels: int,
	out_channels: int,
	allocator: mem.Allocator = context.allocator,
) -> InceptionBlock {
	block: InceptionBlock

	// Split channels to ensure the sum exactly equals out_channels
	c1 := out_channels / 3
	c2 := out_channels / 3
	c3 := out_channels - c1 - c2 // Guarantees c1 + c2 + c3 == out_channels

	block.b1_1x1 = nn.conv2d_layer_new(in_channels, c1, 1, 1, 0, true, allocator)
	block.b2_1x1 = nn.conv2d_layer_new(in_channels, c2, 1, 1, 0, true, allocator)
	block.b2_3x3 = nn.conv2d_layer_new(c2, c2, 3, 1, 1, true, allocator)
	block.b3_1x1 = nn.conv2d_layer_new(in_channels, c3, 1, 1, 0, true, allocator)
	block.b3_5x5 = nn.conv2d_layer_new(c3, c3, 5, 1, 2, true, allocator)

	return block
}

inception_block_free :: proc(block: ^InceptionBlock) {
	nn.conv2d_layer_free(&block.b1_1x1)
	nn.conv2d_layer_free(&block.b2_1x1)
	nn.conv2d_layer_free(&block.b2_3x3)
	nn.conv2d_layer_free(&block.b3_1x1)
	nn.conv2d_layer_free(&block.b3_5x5)
}

inception_block_forward :: proc(
	block: ^InceptionBlock,
	x: ^t.Tensor,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	b1 := t.tensor_conv2d(
		x,
		block.b1_1x1.weight,
		block.b1_1x1.bias,
		block.b1_1x1.stride,
		block.b1_1x1.padding,
	)
	b1 = t.tensor_relu(b1)

	b2 := t.tensor_conv2d(
		x,
		block.b2_1x1.weight,
		block.b2_1x1.bias,
		block.b2_1x1.stride,
		block.b2_1x1.padding,
	)
	b2 = t.tensor_relu(b2)
	b2 = t.tensor_conv2d(
		b2,
		block.b2_3x3.weight,
		block.b2_3x3.bias,
		block.b2_3x3.stride,
		block.b2_3x3.padding,
	)
	b2 = t.tensor_relu(b2)

	b3 := t.tensor_conv2d(
		x,
		block.b3_1x1.weight,
		block.b3_1x1.bias,
		block.b3_1x1.stride,
		block.b3_1x1.padding,
	)
	b3 = t.tensor_relu(b3)
	b3 = t.tensor_conv2d(
		b3,
		block.b3_5x5.weight,
		block.b3_5x5.bias,
		block.b3_5x5.stride,
		block.b3_5x5.padding,
	)
	b3 = t.tensor_relu(b3)

	branches: [3]^t.Tensor
	branches[0] = b1
	branches[1] = b2
	branches[2] = b3

	out := t.tensor_concat(branches[:], 1, alloc)

	// NOTE: Do NOT free b1, b2, or b3 here! They are required for the
	// backward pass. The final t.tensor_free_graph(loss) call in the
	// training loop will safely clean up the entire computation graph.

	return out
}

// ============================================================================
// Initialization & Lifecycle
// ============================================================================

deeplob_new :: proc(
	config: DeepLOBConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^DeepLOB {
	model := new(DeepLOB, allocator)
	model.config = config
	model.allocator = allocator
	model.training = true

	if model.config.dropout_prob == 0.0 {
		model.config.dropout_prob = 0.3
	}

	model.cnn = nn.sequential_new(allocator)

	nn.sequential_add(model.cnn, nn.conv2d_layer_new(4, 16, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	nn.sequential_add(model.cnn, nn.conv2d_layer_new(16, 32, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	// Inception Module: takes 32 channels, outputs exactly 32 channels
	inception_out_channels := 32
	model.inception = inception_block_new(32, inception_out_channels, allocator)

	nn.sequential_add(model.cnn, nn.maxpool2d_layer_new(2, 2))

	// LSTM: input features = C_out * L_out
	// After MaxPool2d, L_out = price_levels / 2
	lstm_feat_dim := inception_out_channels * (config.price_levels / 2)
	model.lstm = nn.lstm_layer_new(lstm_feat_dim, config.hidden_dim, allocator)

	model.fc_head = nn.linear_layer_new(config.hidden_dim, config.num_classes, allocator)

	return model
}

deeplob_free :: proc(model: ^DeepLOB) {
	if model.cnn != nil {
		nn.sequential_free(model.cnn)
	}
	inception_block_free(&model.inception)
	nn.lstm_layer_free(&model.lstm)
	nn.linear_layer_free(&model.fc_head)
	free(model, model.allocator)
}

deeplob_train :: proc(model: ^DeepLOB) {
	model.training = true
}

deeplob_eval :: proc(model: ^DeepLOB) {
	model.training = false
}

// ============================================================================
// Forward Pass
// ============================================================================

deeplob_forward :: proc(model: ^DeepLOB, input: ^t.Tensor) -> ^t.Tensor {
	alloc := model.allocator
	batch := input.shape[0]

	cnn_out := nn.sequential_forward(model.cnn, input)

	inception_out := inception_block_forward(&model.inception, cnn_out, alloc)

	c_out := inception_out.shape[1]
	t_out := inception_out.shape[2]
	l_out := inception_out.shape[3]

	lstm_in := tensor_permute_lob(inception_out, batch, c_out, t_out, l_out, alloc)

	h0_data := l.matrix_new(f64, batch, model.config.hidden_dim, alloc)
	c0_data := l.matrix_new(f64, batch, model.config.hidden_dim, alloc)
	h0 := t.tensor_new(h0_data, false, alloc)
	c0 := t.tensor_new(c0_data, false, alloc)
	h0.owned_by_graph = true
	c0.owned_by_graph = true

	lstm_out := nn.lstm_layer_forward(&model.lstm, lstm_in, h0, c0)

	hidden_dim := model.config.hidden_dim
	last_step_data := l.matrix_new(f64, batch, hidden_dim, alloc)

	for b: int = 0; b < batch; b += 1 {
		src_idx := b * (t_out * hidden_dim) + (t_out - 1) * hidden_dim
		dst_idx := b * hidden_dim
		copy(
			last_step_data.data[dst_idx:dst_idx + hidden_dim],
			lstm_out.data.data[src_idx:src_idx + hidden_dim],
		)
	}

	last_step := t.tensor_new(last_step_data, true, alloc)
	last_step.shape = [4]int{batch, hidden_dim, 1, 1}
	last_step.owned_by_graph = true

	if model.training && model.config.dropout_prob > 0.0 {
		last_step = t.tensor_dropout(last_step, model.config.dropout_prob, model.training)
	}

	logits := nn.linear_forward(&model.fc_head, last_step)

	return logits
}

// ============================================================================
// Custom Permute Operation for Autograd Graph
// ============================================================================

tensor_permute_lob :: proc(
	x: ^t.Tensor,
	batch, c_out, t_out, l_out: int,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	feat_dim := c_out * l_out
	out_data := l.matrix_new(f64, batch * t_out, feat_dim, alloc)

	for b: int = 0; b < batch; b += 1 {
		for tt: int = 0; tt < t_out; tt += 1 {
			for c: int = 0; c < c_out; c += 1 {
				src_idx := b * (c_out * t_out * l_out) + c * (t_out * l_out) + tt * l_out
				dst_idx := (b * t_out + tt) * feat_dim + c * l_out
				copy(out_data.data[dst_idx:dst_idx + l_out], x.data.data[src_idx:src_idx + l_out])
			}
		}
	}

	out := t.tensor_new(out_data, x.requires_grad, alloc)
	out.shape = [4]int{batch, t_out, feat_dim, 1}

	if out.requires_grad {
		out.op = .PermuteLOB
		append(&out.inputs, x)
		append(&out.int_metadata, batch)
		append(&out.int_metadata, c_out)
		append(&out.int_metadata, t_out)
		append(&out.int_metadata, l_out)
	}
	return out
}

// ============================================================================
// Optimizer Integration
// ============================================================================

deeplob_add_to_adam :: proc(model: ^DeepLOB, opt: ^nn.Adam) {
	nn.sequential_add_to_adam(model.cnn, opt)

	nn.adam_add_param(opt, model.inception.b1_1x1.weight)
	if model.inception.b1_1x1.bias != nil {nn.adam_add_param(opt, model.inception.b1_1x1.bias)}

	nn.adam_add_param(opt, model.inception.b2_1x1.weight)
	if model.inception.b2_1x1.bias != nil {nn.adam_add_param(opt, model.inception.b2_1x1.bias)}
	nn.adam_add_param(opt, model.inception.b2_3x3.weight)
	if model.inception.b2_3x3.bias != nil {nn.adam_add_param(opt, model.inception.b2_3x3.bias)}

	nn.adam_add_param(opt, model.inception.b3_1x1.weight)
	if model.inception.b3_1x1.bias != nil {nn.adam_add_param(opt, model.inception.b3_1x1.bias)}
	nn.adam_add_param(opt, model.inception.b3_5x5.weight)
	if model.inception.b3_5x5.bias != nil {nn.adam_add_param(opt, model.inception.b3_5x5.bias)}

	nn.adam_add_param(opt, model.lstm.w_ih)
	nn.adam_add_param(opt, model.lstm.w_hh)
	if model.lstm.bias != nil {
		nn.adam_add_param(opt, model.lstm.bias)
	}

	nn.adam_add_param(opt, model.fc_head.weights)
	if model.fc_head.bias != nil {
		nn.adam_add_param(opt, model.fc_head.bias)
	}
}
