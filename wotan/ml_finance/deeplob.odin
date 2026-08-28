package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:mem"

// ============================================================================
// DeepLOB Configuration & Structures
// ============================================================================

DeepLOBConfig :: struct {
	time_steps:   int, // T (e.g., 100 ticks)
	price_levels: int, // L (e.g., 10 bid/ask levels)
	num_classes:  int, // Usually 3 (Up, Down, Stationary)
	hidden_dim:   int, // LSTM hidden size
}

DeepLOB :: struct {
	cnn:       ^nn.Sequential,
	inception: InceptionBlock, // Multi-scale feature extraction
	lstm:      nn.LSTMLayer,
	fc_head:   nn.LinearLayer,
	config:    DeepLOBConfig,
	allocator: mem.Allocator,
}

// ============================================================================
// Inception Block (Multi-Scale Feature Extraction)
// ============================================================================
// Implements parallel convolutional branches. Outputs are fused via element-wise
// addition to avoid requiring tensor_concat, while maintaining identical shapes.

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

	// Split output channels equally among the 3 main branches so they can be added
	c := out_channels / 3
	if c < 1 {c = 1}

	// Branch 1: Direct 1x1 Conv
	block.b1_1x1 = nn.conv2d_layer_new(in_channels, c, 1, 1, 0, true, allocator)

	// Branch 2: 1x1 Conv (dimensionality reduction) -> 3x3 Conv
	block.b2_1x1 = nn.conv2d_layer_new(in_channels, c, 1, 1, 0, true, allocator)
	block.b2_3x3 = nn.conv2d_layer_new(c, c, 3, 1, 1, true, allocator)

	// Branch 3: 1x1 Conv (dimensionality reduction) -> 5x5 Conv
	block.b3_1x1 = nn.conv2d_layer_new(in_channels, c, 1, 1, 0, true, allocator)
	block.b3_5x5 = nn.conv2d_layer_new(c, c, 5, 1, 2, true, allocator)

	return block
}

inception_block_free :: proc(block: ^InceptionBlock) {
	nn.conv2d_layer_free(&block.b1_1x1)
	nn.conv2d_layer_free(&block.b2_1x1)
	nn.conv2d_layer_free(&block.b2_3x3)
	nn.conv2d_layer_free(&block.b3_1x1)
	nn.conv2d_layer_free(&block.b3_5x5)
}

inception_block_forward :: proc(block: ^InceptionBlock, x: ^t.Tensor, alloc: mem.Allocator) -> ^t.Tensor {
	// Branch 1: 1x1 Conv + ReLU
	b1 := t.tensor_conv2d(x, block.b1_1x1.weight, block.b1_1x1.bias, block.b1_1x1.stride, block.b1_1x1.padding)
	b1 = t.tensor_relu(b1)

	// Branch 2: 1x1 Conv + ReLU -> 3x3 Conv + ReLU
	b2 := t.tensor_conv2d(x, block.b2_1x1.weight, block.b2_1x1.bias, block.b2_1x1.stride, block.b2_1x1.padding)
	b2 = t.tensor_relu(b2)
	b2 = t.tensor_conv2d(b2, block.b2_3x3.weight, block.b2_3x3.bias, block.b2_3x3.stride, block.b2_3x3.padding)
	b2 = t.tensor_relu(b2)

	// Branch 3: 1x1 Conv + ReLU -> 5x5 Conv + ReLU
	b3 := t.tensor_conv2d(x, block.b3_1x1.weight, block.b3_1x1.bias, block.b3_1x1.stride, block.b3_1x1.padding)
	b3 = t.tensor_relu(b3)
	b3 = t.tensor_conv2d(b3, block.b3_5x5.weight, block.b3_5x5.bias, block.b3_5x5.stride, block.b3_5x5.padding)
	b3 = t.tensor_relu(b3)

	// Combine branches via concatenation along the channel dimension (dim = 1)
	// Using a fixed-size array avoids dynamic literal allocation issues
	branches: [3]^t.Tensor
	branches[0] = b1
	branches[1] = b2
	branches[2] = b3
	
	out := t.tensor_concat(branches[:], 1, alloc)

	// Clean up intermediate tensors to prevent memory leaks in the autograd graph
	t.tensor_free_graph(b1)
	t.tensor_free_graph(b2)
	t.tensor_free_graph(b3)

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

	// 1. Initial CNN Feature Extractor
	model.cnn = nn.sequential_new(allocator)

	// Conv1: 4 -> 16 channels, 3x3 kernel, stride 1, padding 1
	nn.sequential_add(model.cnn, nn.conv2d_layer_new(4, 16, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	// Conv2: 16 -> 32 channels, 3x3 kernel, stride 1, padding 1
	nn.sequential_add(model.cnn, nn.conv2d_layer_new(16, 32, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	// 2. Inception Module for Multi-Scale Feature Extraction
	// Takes 32 channels, outputs 32 channels (split across branches)
	model.inception = inception_block_new(32, 32, allocator)

	// 3. MaxPool: 2x2 kernel, stride 2 (downsamples T -> T/2, L -> L/2)
	nn.sequential_add(model.cnn, nn.maxpool2d_layer_new(2, 2))

	// 4. LSTM for Temporal Sequence Modeling
	// The input features will be C_out * L_out (32 * (price_levels / 2))
	lstm_feat_dim := 32 * (config.price_levels / 2)
	model.lstm = nn.lstm_layer_new(lstm_feat_dim, config.hidden_dim, allocator)

	// 5. Classification Head
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

// ============================================================================
// Forward Pass
// ============================================================================

deeplob_forward :: proc(model: ^DeepLOB, input: ^t.Tensor) -> ^t.Tensor {
	alloc := model.allocator
	batch := input.shape[0]

	// 1. Pass through initial CNN
	cnn_out := nn.sequential_forward(model.cnn, input)

	// 2. Pass through Inception Block (Multi-scale extraction)
	inception_out := inception_block_forward(&model.inception, cnn_out, alloc)
	t.tensor_free_graph(cnn_out) // Clean up intermediate to prevent graph bloat

	c_out := inception_out.shape[1]
	t_out := inception_out.shape[2]
	l_out := inception_out.shape[3]

	// 3. Permute for LSTM: [Batch, C, T, L] -> [Batch, T, C*L]
	lstm_in := tensor_permute_lob(inception_out, batch, c_out, t_out, l_out, alloc)
	t.tensor_free_graph(inception_out) // Clean up intermediate

	// 4. Pass through LSTM
	h0_data := l.matrix_new(f64, batch, model.config.hidden_dim, alloc)
	c0_data := l.matrix_new(f64, batch, model.config.hidden_dim, alloc)
	h0 := t.tensor_new(h0_data, false, alloc)
	c0 := t.tensor_new(c0_data, false, alloc)
	h0.owned_by_graph = true
	c0.owned_by_graph = true

	lstm_out := nn.lstm_layer_forward(&model.lstm, lstm_in, h0, c0)
	// lstm_out shape: [Batch, T_out, Hidden, 1]

	// 5. Extract the last time step for classification
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

	// 6. Classification Head
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

	// Manually add Inception block parameters
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

	// Manually add LSTM and FC weights
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
