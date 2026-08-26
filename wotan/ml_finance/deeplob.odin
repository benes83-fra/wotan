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
	lstm:      nn.LSTMLayer,
	fc_head:   nn.LinearLayer,
	config:    DeepLOBConfig,
	allocator: mem.Allocator,
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

	// 1. CNN Feature Extractor
	// Input shape: [Batch, Channels=4, Time=T, Levels=L]
	// Note: Wotan's Conv2d and MaxPool2d only support square kernels/strides.
	// We use 3x3 kernels with padding 1 to preserve spatial dimensions,
	// then a 2x2 MaxPool to downsample both Time and Levels.
	model.cnn = nn.sequential_new(allocator)

	// Conv1: 4 -> 16 channels, 3x3 kernel, stride 1, padding 1
	nn.sequential_add(model.cnn, nn.conv2d_layer_new(4, 16, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	// Conv2: 16 -> 32 channels, 3x3 kernel, stride 1, padding 1
	nn.sequential_add(model.cnn, nn.conv2d_layer_new(16, 32, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	// MaxPool: 2x2 kernel, stride 2 (downsamples T -> T/2, L -> L/2)
	nn.sequential_add(model.cnn, nn.maxpool2d_layer_new(2, 2))

	// 2. LSTM for Temporal Sequence Modeling
	// The input features will be C_out * L_out (32 * (price_levels / 2))
	lstm_feat_dim := 32 * (config.price_levels / 2)
	model.lstm = nn.lstm_layer_new(lstm_feat_dim, config.hidden_dim, allocator)

	// 3. Classification Head
	model.fc_head = nn.linear_layer_new(config.hidden_dim, config.num_classes, allocator)

	return model
}

deeplob_free :: proc(model: ^DeepLOB) {
	if model.cnn != nil {
		nn.sequential_free(model.cnn)
	}
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

	// 1. Pass through CNN
	cnn_out := nn.sequential_forward(model.cnn, input)
	// cnn_out shape: [Batch, 32, T_out, L_out]

	c_out := cnn_out.shape[1]
	t_out := cnn_out.shape[2]
	l_out := cnn_out.shape[3]

	// 2. Permute for LSTM: [Batch, C, T, L] -> [Batch, T, C*L]
	// We use a custom permute operation to maintain the autograd graph.
	lstm_in := tensor_permute_lob(cnn_out, batch, c_out, t_out, l_out, alloc)

	// 3. Pass through LSTM
	// Initialize h0 and c0 with zeros
	h0_data := l.matrix_new(f64, batch, model.config.hidden_dim, alloc)
	c0_data := l.matrix_new(f64, batch, model.config.hidden_dim, alloc)

	h0 := t.tensor_new(h0_data, false, alloc)
	c0 := t.tensor_new(c0_data, false, alloc)
	h0.owned_by_graph = true
	c0.owned_by_graph = true

	lstm_out := nn.lstm_layer_forward(&model.lstm, lstm_in, h0, c0)
	// lstm_out shape: [Batch, T_out, Hidden, 1]

	// 4. Extract the last time step for classification
	// OPTIMIZED: Using Odin's highly optimized `copy` (SIMD/memcpy) instead of manual element-wise loops
	hidden_dim := model.config.hidden_dim
	last_step_data := l.matrix_new(f64, batch, hidden_dim, alloc)

	for b in 0 ..< batch {
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

	// DO NOT call t.tensor_free_graph here. Let the training loop handle it
	// via t.tensor_free_graph(loss) to safely clean up the entire connected graph.

	// 5. Classification Head
	logits := nn.linear_forward(&model.fc_head, last_step)

	return logits
}

// ============================================================================
// Custom Permute Operation for Autograd Graph
// ============================================================================
// NOTE: This requires `.PermuteLOB` to be present in the `Op` enum in `wotan/tensor/tensor.odin`
// and the corresponding backward pass case (which is already present in the repo).

tensor_permute_lob :: proc(
	x: ^t.Tensor,
	batch, c_out, t_out, l_out: int,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	feat_dim := c_out * l_out
	out_data := l.matrix_new(f64, batch * t_out, feat_dim, alloc)

	// OPTIMIZED: Permutation using Odin's `copy` for contiguous blocks.
	// The innermost loop over `l_out` copies a contiguous slice, leveraging SIMD/memcpy.
	for b in 0 ..< batch {
		for tt in 0 ..< t_out {
			for c in 0 ..< c_out {
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

	// Manually add LSTM and FC weights since they are outside the Sequential container
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
