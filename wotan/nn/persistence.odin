package nn

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:mem"
import "core:os"

// Magic bytes for checkpoint validation
CHECKPOINT_MAGIC :: []u8{'W', 'O', 'T', 'A', 'N', '_', 'C', 'K', 'P', 'T'}

// ============================================================================
// Simple Binary I/O Helpers
// ============================================================================

write_i32 :: proc(file: ^os.File, val: i32) {
	bytes := [4]u8 {
		u8(val & 0xFF),
		u8((val >> 8) & 0xFF),
		u8((val >> 16) & 0xFF),
		u8((val >> 24) & 0xFF),
	}
	os.write(file, bytes[:])
}

write_i64 :: proc(file: ^os.File, val: i64) {
	bytes := [8]u8 {
		u8(val & 0xFF),
		u8((val >> 8) & 0xFF),
		u8((val >> 16) & 0xFF),
		u8((val >> 24) & 0xFF),
		u8((val >> 32) & 0xFF),
		u8((val >> 40) & 0xFF),
		u8((val >> 48) & 0xFF),
		u8((val >> 56) & 0xFF),
	}
	os.write(file, bytes[:])
}

write_f64 :: proc(file: ^os.File, val: f64) {
	val_i64 := transmute(i64)val
	bytes := [8]u8 {
		u8(val_i64 & 0xFF),
		u8((val_i64 >> 8) & 0xFF),
		u8((val_i64 >> 16) & 0xFF),
		u8((val_i64 >> 24) & 0xFF),
		u8((val_i64 >> 32) & 0xFF),
		u8((val_i64 >> 40) & 0xFF),
		u8((val_i64 >> 48) & 0xFF),
		u8((val_i64 >> 56) & 0xFF),
	}
	os.write(file, bytes[:])
}

read_i32 :: proc(data: []u8, offset: int) -> (i32, int) {
	val :=
		i32(data[offset]) |
		(i32(data[offset + 1]) << 8) |
		(i32(data[offset + 2]) << 16) |
		(i32(data[offset + 3]) << 24)
	return val, offset + 4
}

read_i64 :: proc(data: []u8, offset: int) -> (i64, int) {
	val :=
		i64(data[offset]) |
		(i64(data[offset + 1]) << 8) |
		(i64(data[offset + 2]) << 16) |
		(i64(data[offset + 3]) << 24) |
		(i64(data[offset + 4]) << 32) |
		(i64(data[offset + 5]) << 40) |
		(i64(data[offset + 6]) << 48) |
		(i64(data[offset + 7]) << 56)
	return val, offset + 8
}

read_f64 :: proc(data: []u8, offset: int) -> (f64, int) {
	val_i64 :=
		i64(data[offset]) |
		(i64(data[offset + 1]) << 8) |
		(i64(data[offset + 2]) << 16) |
		(i64(data[offset + 3]) << 24) |
		(i64(data[offset + 4]) << 32) |
		(i64(data[offset + 5]) << 40) |
		(i64(data[offset + 6]) << 48) |
		(i64(data[offset + 7]) << 56)
	return transmute(f64)val_i64, offset + 8
}

// ============================================================================
// Tensor I/O
// ============================================================================

write_tensor :: proc(file: ^os.File, tensor: ^t.Tensor) {
	if tensor == nil {
		write_i32(file, 0) // rows
		write_i32(file, 0) // cols
		return
	}

	write_i32(file, i32(tensor.data.rows))
	write_i32(file, i32(tensor.data.cols))

	count := tensor.data.rows * tensor.data.cols
	for i in 0 ..< count {
		write_f64(file, tensor.data.data[i])
	}
}

read_tensor :: proc(data: []u8, offset: int, allocator: mem.Allocator) -> (^t.Tensor, int) {
	curr_offset := offset

	rows_i32: i32
	cols_i32: i32
	rows_i32, curr_offset = read_i32(data, curr_offset)
	cols_i32, curr_offset = read_i32(data, curr_offset)

	rows := int(rows_i32)
	cols := int(cols_i32)

	// Handle nil tensor
	if rows == 0 && cols == 0 {
		return nil, curr_offset
	}

	count := rows * cols
	m := l.matrix_new(f64, rows, cols, allocator)

	for i in 0 ..< count {
		val: f64
		val, curr_offset = read_f64(data, curr_offset)
		m.data[i] = val
	}

	tensor := t.tensor_new(m, true, allocator)
	return tensor, curr_offset
}

// ============================================================================
// Save Checkpoint
// ============================================================================

save_checkpoint :: proc(
	model: ^Sequential,
	opt: ^Adam,
	path: string,
	epoch: int,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	// Magic header
	os.write(file, CHECKPOINT_MAGIC)

	// Epoch and layer count
	write_i32(file, i32(epoch))
	write_i32(file, i32(len(model.layers)))

	// Save each layer
	for layer in model.layers {
		switch l in layer {
		case LinearLayer:
			write_i32(file, 0) // type
			write_i32(file, i32(l.in_features))
			write_i32(file, i32(l.out_features))
			write_i32(file, i32(bool(l.bias != nil)))
			write_tensor(file, l.weights)
			if l.bias != nil {
				write_tensor(file, l.bias)
			}

		case Conv2dLayer:
			write_i32(file, 1) // type
			write_i32(file, i32(l.in_channels))
			write_i32(file, i32(l.out_channels))
			write_i32(file, i32(l.kernel_size))
			write_i32(file, i32(l.stride))
			write_i32(file, i32(l.padding))
			write_i32(file, i32(bool(l.bias != nil)))
			write_tensor(file, l.weight)
			if l.bias != nil {
				write_tensor(file, l.bias)
			}

		case MaxPool2dLayer:
			write_i32(file, 2) // type
			write_i32(file, i32(l.kernel_size))
			write_i32(file, i32(l.stride))

		case Activation:
			write_i32(file, 3) // type
			write_i32(file, i32(l))

		case FlattenLayer:
			write_i32(file, 4) // type
		case BatchNorm2dLayer:
			write_i32(file, 5) // Type 5
			write_i32(file, i32(l.num_features))
			write_f64(file, l.eps)
			write_f64(file, l.momentum)
			write_tensor(file, l.weight)
			write_tensor(file, l.bias)
			write_tensor(file, l.running_mean)
			write_tensor(file, l.running_var)
		case AvgPool2dLayer:
			write_i32(file, 6) // Type 6
			write_i32(file, i32(l.kernel_size))
			write_i32(file, i32(l.stride))

		case DropoutLayer:
			write_i32(file, 7) // Type 7
			write_f64(file, l.drop_prob)
		case RNNLayer:
			write_i32(file, 8) // Type 8
			write_i32(file, i32(l.input_size))
			write_i32(file, i32(l.hidden_size))
			write_tensor(file, l.w_ih)
			write_tensor(file, l.w_hh)
			write_tensor(file, l.bias)
		case GRULayer:
			write_i32(file, 9) // Type 9
			write_i32(file, i32(l.input_size))
			write_i32(file, i32(l.hidden_size))
			write_tensor(file, l.w_ih)
			write_tensor(file, l.w_hh)
			write_tensor(file, l.bias)
		case LSTMLayer:
			write_i32(file, 10) // Type 10
			write_i32(file, i32(l.input_size))
			write_i32(file, i32(l.hidden_size))
			write_tensor(file, l.w_ih)
			write_tensor(file, l.w_hh)
			write_tensor(file, l.bias)
		case EmbeddingLayer:
			write_i32(file, 11) // Type 11
			write_i32(file, i32(l.num_embeddings))
			write_i32(file, i32(l.embedding_dim))
			write_tensor(file, l.weight)
		case MultiHeadAttentionLayer:
			write_i32(file, 12) // Type 12
			write_i32(file, i32(l.d_model))
			write_i32(file, i32(l.num_heads))
			write_tensor(file, l.q_proj.weights)
			write_tensor(file, l.q_proj.bias)
			write_tensor(file, l.k_proj.weights)
			write_tensor(file, l.k_proj.bias)
			write_tensor(file, l.v_proj.weights)
			write_tensor(file, l.v_proj.bias)
			write_tensor(file, l.out_proj.weights)
			write_tensor(file, l.out_proj.bias)
		case LayerNormLayer:
			write_i32(file, 13)
			write_i32(file, i32(l.d_model))
			write_f64(file, l.eps)
			write_tensor(file, l.gamma)
			write_tensor(file, l.beta)
		case FFNLayer:
			write_i32(file, 14) // Type 14
			write_i32(file, i32(l.d_model))
			write_i32(file, i32(l.d_ff))
			write_tensor(file, l.fc1.weights)
			write_tensor(file, l.fc1.bias)
			write_tensor(file, l.fc2.weights)
			write_tensor(file, l.fc2.bias)
		case TransformerEncoderBlock:
			write_i32(file, 15) // Type 15
			write_i32(file, i32(l.d_model))
			write_i32(file, i32(l.num_heads))
			write_i32(file, i32(l.d_ff))

			// Save MHA
			write_tensor(file, l.mha.q_proj.weights)
			write_tensor(file, l.mha.q_proj.bias)
			write_tensor(file, l.mha.k_proj.weights)
			write_tensor(file, l.mha.k_proj.bias)
			write_tensor(file, l.mha.v_proj.weights)
			write_tensor(file, l.mha.v_proj.bias)
			write_tensor(file, l.mha.out_proj.weights)
			write_tensor(file, l.mha.out_proj.bias)

			// Save FFN
			write_tensor(file, l.ffn.fc1.weights)
			write_tensor(file, l.ffn.fc1.bias)
			write_tensor(file, l.ffn.fc2.weights)
			write_tensor(file, l.ffn.fc2.bias)

			// Save LayerNorms
			write_tensor(file, l.ln1.gamma)
			write_tensor(file, l.ln1.beta)
			write_tensor(file, l.ln2.gamma)
			write_tensor(file, l.ln2.beta)
		case TransformerEncoder:
			write_i32(file, 16) // Type 16
			write_i32(file, i32(l.num_layers))
			write_i32(file, i32(l.d_model))
			write_i32(file, i32(l.num_heads))
			write_i32(file, i32(l.d_ff))

			// Save each block
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]

				// Save MHA
				write_tensor(file, block.mha.q_proj.weights)
				write_tensor(file, block.mha.q_proj.bias)
				write_tensor(file, block.mha.k_proj.weights)
				write_tensor(file, block.mha.k_proj.bias)
				write_tensor(file, block.mha.v_proj.weights)
				write_tensor(file, block.mha.v_proj.bias)
				write_tensor(file, block.mha.out_proj.weights)
				write_tensor(file, block.mha.out_proj.bias)

				// Save FFN
				write_tensor(file, block.ffn.fc1.weights)
				write_tensor(file, block.ffn.fc1.bias)
				write_tensor(file, block.ffn.fc2.weights)
				write_tensor(file, block.ffn.fc2.bias)

				// Save LayerNorms
				write_tensor(file, block.ln1.gamma)
				write_tensor(file, block.ln1.beta)
				write_tensor(file, block.ln2.gamma)
				write_tensor(file, block.ln2.beta)
			}
		case GATLayer:
			write_i32(file, 24) // Type 24
			write_i32(file, i32(l.linear.in_features))
			write_i32(file, i32(l.linear.out_features))
			write_i32(file, i32(l.num_heads))
			write_i32(file, i32(bool(l.linear.bias != nil)))

			// Save Linear weights
			write_tensor(file, l.linear.weights)
			if l.linear.bias != nil {write_tensor(file, l.linear.bias)}

			// Save MHA weights (same pattern as MultiHeadAttentionLayer)
			write_tensor(file, l.mha.q_proj.weights)
			write_tensor(file, l.mha.q_proj.bias)
			write_tensor(file, l.mha.k_proj.weights)
			write_tensor(file, l.mha.k_proj.bias)
			write_tensor(file, l.mha.v_proj.weights)
			write_tensor(file, l.mha.v_proj.bias)
			write_tensor(file, l.mha.out_proj.weights)
			write_tensor(file, l.mha.out_proj.bias)

			// Save Adjacency Matrix
			write_tensor(file, l.adjacency)
		}

	}

	// Save optimizer state
	write_i32(file, 0) // optimizer type (Adam)
	write_i64(file, i64(opt.timestep))
	write_i32(file, i32(len(opt.parameters)))

	for i in 0 ..< len(opt.parameters) {
		// moment_1
		m1_count := len(opt.moment_1[i])
		write_i32(file, i32(m1_count))
		for j in 0 ..< m1_count {
			write_f64(file, opt.moment_1[i][j])
		}

		// moment_2
		m2_count := len(opt.moment_2[i])
		write_i32(file, i32(m2_count))
		for j in 0 ..< m2_count {
			write_f64(file, opt.moment_2[i][j])
		}
	}

	return true
}

// ============================================================================
// Load Checkpoint
// ============================================================================

load_checkpoint :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	model: ^Sequential,
	opt: ^Adam,
	epoch: int,
	ok: bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, nil, 0, false
	}
	defer delete(data, allocator)

	offset := 0

	// Magic header
	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, nil, 0, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, nil, 0, false
		}
	}
	offset += 10

	// Epoch
	ep_i32: i32
	ep_i32, offset = read_i32(data, offset)
	epoch = int(ep_i32)

	// Model
	model = new(Sequential, allocator)
	model.allocator = allocator
	model.layers = make([dynamic]Layer, 0, allocator)

	num_layers_i32: i32
	num_layers_i32, offset = read_i32(data, offset)

	for _ in 0 ..< int(num_layers_i32) {
		layer_type: i32
		layer_type, offset = read_i32(data, offset)

		if layer_type == 0 { 	// Linear
			in_f: i32
			out_f: i32
			has_bias: i32
			in_f, offset = read_i32(data, offset)
			out_f, offset = read_i32(data, offset)
			has_bias, offset = read_i32(data, offset)

			layer := linear_layer_new(int(in_f), int(out_f), allocator)

			// Read weights
			layer.weights, offset = read_tensor(data, offset, allocator)

			// Read or discard bias
			if has_bias != 0 {
				layer.bias, offset = read_tensor(data, offset, allocator)
			} else {
				// Free the bias that was created by linear_layer_new
				if layer.bias != nil {
					t.tensor_free(layer.bias)
					layer.bias = nil
				}
			}

			append(&model.layers, layer)

		} else if layer_type == 1 { 	// Conv2d
			in_c: i32
			out_c: i32
			k: i32
			stride: i32
			pad: i32
			has_bias: i32
			in_c, offset = read_i32(data, offset)
			out_c, offset = read_i32(data, offset)
			k, offset = read_i32(data, offset)
			stride, offset = read_i32(data, offset)
			pad, offset = read_i32(data, offset)
			has_bias, offset = read_i32(data, offset)

			layer := conv2d_layer_new(
				int(in_c),
				int(out_c),
				int(k),
				int(stride),
				int(pad),
				has_bias != 0,
				allocator,
			)

			// Free default weight and load saved
			if layer.weight != nil {
				t.tensor_free(layer.weight)
			}
			layer.weight, offset = read_tensor(data, offset, allocator)

			// Set the shape field for Conv2d weights
			layer.weight.shape = [4]int{int(out_c), int(in_c), int(k), int(k)}

			// Read or discard bias
			if has_bias != 0 {
				layer.bias, offset = read_tensor(data, offset, allocator)
			} else {
				if layer.bias != nil {
					t.tensor_free(layer.bias)
					layer.bias = nil
				}
			}

			append(&model.layers, layer)

		} else if layer_type == 2 { 	// MaxPool
			k: i32
			s: i32
			k, offset = read_i32(data, offset)
			s, offset = read_i32(data, offset)
			append(&model.layers, maxpool2d_layer_new(int(k), int(s)))

		} else if layer_type == 3 { 	// Activation
			act: i32
			act, offset = read_i32(data, offset)
			append(&model.layers, Activation(act))

		} else if layer_type == 4 { 	// Flatten
			append(&model.layers, FlattenLayer{})
		} else if layer_type == 5 { 	// BatchNorm2d
			num_features: i32
			eps: f64
			momentum: f64
			num_features, offset = read_i32(data, offset)
			eps, offset = read_f64(data, offset)
			momentum, offset = read_f64(data, offset)

			layer := batch_norm_2d_layer_new(int(num_features), eps, momentum, allocator)

			// Free defaults and load saved
			if layer.weight != nil {t.tensor_free(layer.weight)}
			layer.weight, offset = read_tensor(data, offset, allocator)

			if layer.bias != nil {t.tensor_free(layer.bias)}
			layer.bias, offset = read_tensor(data, offset, allocator)

			if layer.running_mean != nil {t.tensor_free(layer.running_mean)}
			layer.running_mean, offset = read_tensor(data, offset, allocator)

			if layer.running_var != nil {t.tensor_free(layer.running_var)}
			layer.running_var, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)

		} else if layer_type == 6 { 	// AvgPool2d
			k: i32
			s: i32
			k, offset = read_i32(data, offset)
			s, offset = read_i32(data, offset)
			append(&model.layers, avgpool2d_layer_new(int(k), int(s)))

		} else if layer_type == 7 { 	// Dropout
			drop_prob: f64
			drop_prob, offset = read_f64(data, offset)
			append(&model.layers, dropout_layer_new(drop_prob))
		} else if layer_type == 8 { 	// RNN
			in_size: i32
			hidden_size: i32
			in_size, offset = read_i32(data, offset)
			hidden_size, offset = read_i32(data, offset)

			layer := rnn_layer_new(int(in_size), int(hidden_size), allocator)

			// Free the randomly initialized defaults and load the saved weights
			if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
			layer.w_ih, offset = read_tensor(data, offset, allocator)

			if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
			layer.w_hh, offset = read_tensor(data, offset, allocator)

			if layer.bias != nil {t.tensor_free(layer.bias)}
			layer.bias, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 9 { 	// GRU
			in_size: i32
			hidden_size: i32
			in_size, offset = read_i32(data, offset)
			hidden_size, offset = read_i32(data, offset)

			layer := gru_layer_new(int(in_size), int(hidden_size), allocator)

			// Free the randomly initialized defaults and load the saved weights
			if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
			layer.w_ih, offset = read_tensor(data, offset, allocator)

			if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
			layer.w_hh, offset = read_tensor(data, offset, allocator)

			if layer.bias != nil {t.tensor_free(layer.bias)}
			layer.bias, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 10 { 	// LSTM
			in_size: i32
			hidden_size: i32
			in_size, offset = read_i32(data, offset)
			hidden_size, offset = read_i32(data, offset)

			layer := lstm_layer_new(int(in_size), int(hidden_size), allocator)

			// Free the randomly initialized defaults and load the saved weights
			if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
			layer.w_ih, offset = read_tensor(data, offset, allocator)

			if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
			layer.w_hh, offset = read_tensor(data, offset, allocator)

			if layer.bias != nil {t.tensor_free(layer.bias)}
			layer.bias, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 11 { 	// Embedding
			num_embeddings: i32
			embedding_dim: i32
			num_embeddings, offset = read_i32(data, offset)
			embedding_dim, offset = read_i32(data, offset)

			layer := embedding_layer_new(int(num_embeddings), int(embedding_dim), allocator)

			if layer.weight != nil {t.tensor_free(layer.weight)}
			layer.weight, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 12 { 	// MultiHeadAttention
			d_model: i32
			num_heads: i32
			d_model, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)

			layer := multi_head_attention_layer_new(int(d_model), int(num_heads), allocator)

			// Load q_proj
			if layer.q_proj.weights != nil {t.tensor_free(layer.q_proj.weights)}
			layer.q_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.q_proj.bias != nil {t.tensor_free(layer.q_proj.bias)}
			layer.q_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load k_proj
			if layer.k_proj.weights != nil {t.tensor_free(layer.k_proj.weights)}
			layer.k_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.k_proj.bias != nil {t.tensor_free(layer.k_proj.bias)}
			layer.k_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load v_proj
			if layer.v_proj.weights != nil {t.tensor_free(layer.v_proj.weights)}
			layer.v_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.v_proj.bias != nil {t.tensor_free(layer.v_proj.bias)}
			layer.v_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load out_proj
			if layer.out_proj.weights != nil {t.tensor_free(layer.out_proj.weights)}
			layer.out_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.out_proj.bias != nil {t.tensor_free(layer.out_proj.bias)}
			layer.out_proj.bias, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 13 { 	// LayerNorm
			d_model: i32
			eps: f64
			d_model, offset = read_i32(data, offset)
			eps, offset = read_f64(data, offset)

			layer := layer_norm_layer_new(int(d_model), eps, allocator)

			if layer.gamma != nil {t.tensor_free(layer.gamma)}
			layer.gamma, offset = read_tensor(data, offset, allocator)

			if layer.beta != nil {t.tensor_free(layer.beta)}
			layer.beta, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 14 { 	// FFN
			d_model: i32
			d_ff: i32
			d_model, offset = read_i32(data, offset)
			d_ff, offset = read_i32(data, offset)

			layer := ffn_layer_new(int(d_model), int(d_ff), allocator)

			// Load fc1
			if layer.fc1.weights != nil {t.tensor_free(layer.fc1.weights)}
			layer.fc1.weights, offset = read_tensor(data, offset, allocator)
			if layer.fc1.bias != nil {t.tensor_free(layer.fc1.bias)}
			layer.fc1.bias, offset = read_tensor(data, offset, allocator)

			// Load fc2
			if layer.fc2.weights != nil {t.tensor_free(layer.fc2.weights)}
			layer.fc2.weights, offset = read_tensor(data, offset, allocator)
			if layer.fc2.bias != nil {t.tensor_free(layer.fc2.bias)}
			layer.fc2.bias, offset = read_tensor(data, offset, allocator)

			append(&model.layers, layer)
		} else if layer_type == 15 { 	// TransformerEncoderBlock
			d_model: i32
			num_heads: i32
			d_ff: i32
			d_model, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)
			d_ff, offset = read_i32(data, offset)

			block := transformer_encoder_block_new(
				int(d_model),
				int(num_heads),
				int(d_ff),
				allocator,
			)

			// Load MHA q_proj
			if block.mha.q_proj.weights != nil {t.tensor_free(block.mha.q_proj.weights)}
			block.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
			if block.mha.q_proj.bias != nil {t.tensor_free(block.mha.q_proj.bias)}
			block.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load MHA k_proj
			if block.mha.k_proj.weights != nil {t.tensor_free(block.mha.k_proj.weights)}
			block.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
			if block.mha.k_proj.bias != nil {t.tensor_free(block.mha.k_proj.bias)}
			block.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load MHA v_proj
			if block.mha.v_proj.weights != nil {t.tensor_free(block.mha.v_proj.weights)}
			block.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
			if block.mha.v_proj.bias != nil {t.tensor_free(block.mha.v_proj.bias)}
			block.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load MHA out_proj
			if block.mha.out_proj.weights != nil {t.tensor_free(block.mha.out_proj.weights)}
			block.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
			if block.mha.out_proj.bias != nil {t.tensor_free(block.mha.out_proj.bias)}
			block.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load FFN fc1
			if block.ffn.fc1.weights != nil {t.tensor_free(block.ffn.fc1.weights)}
			block.ffn.fc1.weights, offset = read_tensor(data, offset, allocator)
			if block.ffn.fc1.bias != nil {t.tensor_free(block.ffn.fc1.bias)}
			block.ffn.fc1.bias, offset = read_tensor(data, offset, allocator)

			// Load FFN fc2
			if block.ffn.fc2.weights != nil {t.tensor_free(block.ffn.fc2.weights)}
			block.ffn.fc2.weights, offset = read_tensor(data, offset, allocator)
			if block.ffn.fc2.bias != nil {t.tensor_free(block.ffn.fc2.bias)}
			block.ffn.fc2.bias, offset = read_tensor(data, offset, allocator)

			// Load LayerNorms
			if block.ln1.gamma != nil {t.tensor_free(block.ln1.gamma)}
			block.ln1.gamma, offset = read_tensor(data, offset, allocator)
			if block.ln1.beta != nil {t.tensor_free(block.ln1.beta)}
			block.ln1.beta, offset = read_tensor(data, offset, allocator)

			if block.ln2.gamma != nil {t.tensor_free(block.ln2.gamma)}
			block.ln2.gamma, offset = read_tensor(data, offset, allocator)
			if block.ln2.beta != nil {t.tensor_free(block.ln2.beta)}
			block.ln2.beta, offset = read_tensor(data, offset, allocator)

			append(&model.layers, block)
		} else if layer_type == 16 { 	// TransformerEncoder
			num_layers: i32
			d_model: i32
			num_heads: i32
			d_ff: i32
			num_layers, offset = read_i32(data, offset)
			d_model, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)
			d_ff, offset = read_i32(data, offset)

			encoder := transformer_encoder_new(
				int(num_layers),
				int(d_model),
				int(num_heads),
				int(d_ff),
				allocator,
			)

			// Load each block
			for i in 0 ..< int(num_layers) {
				block := &encoder.blocks[i]

				// Load MHA
				if block.mha.q_proj.weights != nil {t.tensor_free(block.mha.q_proj.weights)}
				block.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
				if block.mha.q_proj.bias != nil {t.tensor_free(block.mha.q_proj.bias)}
				block.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)

				if block.mha.k_proj.weights != nil {t.tensor_free(block.mha.k_proj.weights)}
				block.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
				if block.mha.k_proj.bias != nil {t.tensor_free(block.mha.k_proj.bias)}
				block.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)

				if block.mha.v_proj.weights != nil {t.tensor_free(block.mha.v_proj.weights)}
				block.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
				if block.mha.v_proj.bias != nil {t.tensor_free(block.mha.v_proj.bias)}
				block.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)

				if block.mha.out_proj.weights != nil {t.tensor_free(block.mha.out_proj.weights)}
				block.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
				if block.mha.out_proj.bias != nil {t.tensor_free(block.mha.out_proj.bias)}
				block.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)

				// Load FFN
				if block.ffn.fc1.weights != nil {t.tensor_free(block.ffn.fc1.weights)}
				block.ffn.fc1.weights, offset = read_tensor(data, offset, allocator)
				if block.ffn.fc1.bias != nil {t.tensor_free(block.ffn.fc1.bias)}
				block.ffn.fc1.bias, offset = read_tensor(data, offset, allocator)

				if block.ffn.fc2.weights != nil {t.tensor_free(block.ffn.fc2.weights)}
				block.ffn.fc2.weights, offset = read_tensor(data, offset, allocator)
				if block.ffn.fc2.bias != nil {t.tensor_free(block.ffn.fc2.bias)}
				block.ffn.fc2.bias, offset = read_tensor(data, offset, allocator)

				// Load LayerNorms
				if block.ln1.gamma != nil {t.tensor_free(block.ln1.gamma)}
				block.ln1.gamma, offset = read_tensor(data, offset, allocator)
				if block.ln1.beta != nil {t.tensor_free(block.ln1.beta)}
				block.ln1.beta, offset = read_tensor(data, offset, allocator)

				if block.ln2.gamma != nil {t.tensor_free(block.ln2.gamma)}
				block.ln2.gamma, offset = read_tensor(data, offset, allocator)
				if block.ln2.beta != nil {t.tensor_free(block.ln2.beta)}
				block.ln2.beta, offset = read_tensor(data, offset, allocator)
			}

			append(&model.layers, encoder)

		} else if layer_type == 24 { 	// GATLayer
			in_f: i32; out_f: i32; num_heads: i32; has_bias: i32
			in_f, offset = read_i32(data, offset)
			out_f, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)
			has_bias, offset = read_i32(data, offset)

			// We need to read adjacency first to pass it to the constructor
			adj_tensor, new_offset := read_tensor(data, offset, allocator)

			layer := gat_layer_new(int(in_f), int(out_f), int(num_heads), adj_tensor, allocator)

			// Overwrite default weights with saved ones
			if layer.linear.weights != nil {t.tensor_free(layer.linear.weights)}
			layer.linear.weights, offset = read_tensor(data, new_offset, allocator)

			if has_bias != 0 {
				if layer.linear.bias != nil {t.tensor_free(layer.linear.bias)}
				layer.linear.bias, offset = read_tensor(data, offset, allocator)
			}

			// Load MHA weights (simplified for brevity, follow the MultiHeadAttentionLayer pattern)
			if layer.mha.q_proj.weights != nil {t.tensor_free(layer.mha.q_proj.weights)}
			layer.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.mha.q_proj.bias != nil {t.tensor_free(layer.mha.q_proj.bias)}
			layer.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load k_proj
			if layer.mha.k_proj.weights != nil {t.tensor_free(layer.mha.k_proj.weights)}
			layer.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.mha.k_proj.bias != nil {t.tensor_free(layer.mha.k_proj.bias)}
			layer.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load v_proj
			if layer.mha.v_proj.weights != nil {t.tensor_free(layer.mha.v_proj.weights)}
			layer.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.mha.v_proj.bias != nil {t.tensor_free(layer.mha.v_proj.bias)}
			layer.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)

			// Load out_proj
			if layer.mha.out_proj.weights != nil {t.tensor_free(layer.mha.out_proj.weights)}
			layer.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
			if layer.mha.out_proj.bias != nil {t.tensor_free(layer.mha.out_proj.bias)}
			layer.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)
			append(&model.layers, layer)
		}
	}

	// Optimizer
	opt_type: i32
	opt_type, offset = read_i32(data, offset)
	if opt_type != 0 {
		fmt.println("Unsupported optimizer type")
		return nil, nil, 0, false
	}

	opt = new(Adam, allocator)
	timestep_i64: i64
	timestep_i64, offset = read_i64(data, offset)
	opt.timestep = int(timestep_i64)

	opt.learning_rate = 0.001
	opt.beta_1 = 0.9
	opt.beta_2 = 0.999
	opt.epsilon = 1e-8
	opt.allocator = allocator

	num_params_i32: i32
	num_params_i32, offset = read_i32(data, offset)

	opt.parameters = make([dynamic]^t.Tensor, 0, allocator)
	opt.moment_1 = make([dynamic][]f64, 0, allocator)
	opt.moment_2 = make([dynamic][]f64, 0, allocator)

	for i in 0 ..< int(num_params_i32) {
		// moment_1
		m1_count_i32: i32
		m1_count_i32, offset = read_i32(data, offset)
		m1_count := int(m1_count_i32)
		m1 := make([]f64, m1_count, allocator)
		for j in 0 ..< m1_count {
			val: f64
			val, offset = read_f64(data, offset)
			m1[j] = val
		}
		append(&opt.moment_1, m1)

		// moment_2
		m2_count_i32: i32
		m2_count_i32, offset = read_i32(data, offset)
		m2_count := int(m2_count_i32)
		m2 := make([]f64, m2_count, allocator)
		for j in 0 ..< m2_count {
			val: f64
			val, offset = read_f64(data, offset)
			m2[j] = val
		}
		append(&opt.moment_2, m2)
	}

	return model, opt, epoch, true
}
// ============================================================================
// GPT Model Persistence
// ============================================================================
// ============================================================================
// GPT Model Persistence
// ============================================================================

save_gpt_model :: proc(model: ^GPTModel, path: string) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	// Magic header
	os.write(file, CHECKPOINT_MAGIC)

	// Model type and hyperparameters
	write_i32(file, 17) // GPT type ID
	write_i32(file, i32(model.vocab_size))
	write_i32(file, i32(model.d_model))
	write_i32(file, i32(model.num_heads))
	write_i32(file, i32(model.d_ff))
	write_i32(file, i32(model.num_layers))
	write_i32(file, i32(model.max_seq_len))

	// Save embeddings
	write_tensor(file, model.token_emb.weight)
	write_tensor(file, model.pos_emb.weight)

	// Save each decoder block
	for i in 0 ..< len(model.blocks) {
		block := &model.blocks[i]

		// Save MHA
		write_tensor(file, block.mha.q_proj.weights)
		write_tensor(file, block.mha.q_proj.bias)
		write_tensor(file, block.mha.k_proj.weights)
		write_tensor(file, block.mha.k_proj.bias)
		write_tensor(file, block.mha.v_proj.weights)
		write_tensor(file, block.mha.v_proj.bias)
		write_tensor(file, block.mha.out_proj.weights)
		write_tensor(file, block.mha.out_proj.bias)

		// Save FFN
		write_tensor(file, block.ffn.fc1.weights)
		write_tensor(file, block.ffn.fc1.bias)
		write_tensor(file, block.ffn.fc2.weights)
		write_tensor(file, block.ffn.fc2.bias)

		// Save LayerNorms
		write_tensor(file, block.ln1.gamma)
		write_tensor(file, block.ln1.beta)
		write_tensor(file, block.ln2.gamma)
		write_tensor(file, block.ln2.beta)
	}

	// Save final layer norm
	write_tensor(file, model.final_ln.gamma)
	write_tensor(file, model.final_ln.beta)

	// Save output projection
	write_tensor(file, model.output_proj.weights)
	write_tensor(file, model.output_proj.bias)

	return true
}

load_gpt_model :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^GPTModel,
	bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, false
	}
	defer delete(data, allocator)

	offset := 0

	// Magic header
	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, false
		}
	}
	offset += 10

	// Model type
	model_type: i32
	model_type, offset = read_i32(data, offset)
	if model_type != 17 {
		fmt.println("Not a GPT model")
		return nil, false
	}

	// Hyperparameters
	vocab_size: i32
	d_model: i32
	num_heads: i32
	d_ff: i32
	num_layers: i32
	max_seq_len: i32
	vocab_size, offset = read_i32(data, offset)
	d_model, offset = read_i32(data, offset)
	num_heads, offset = read_i32(data, offset)
	d_ff, offset = read_i32(data, offset)
	num_layers, offset = read_i32(data, offset)
	max_seq_len, offset = read_i32(data, offset)

	model_ptr := new(GPTModel, allocator)
	model_ptr^ = gpt_model_new(
		int(vocab_size),
		int(d_model),
		int(num_heads),
		int(d_ff),
		int(num_layers),
		int(max_seq_len),
		allocator,
	)

	// Load embeddings
	if model_ptr.token_emb.weight != nil {t.tensor_free(model_ptr.token_emb.weight)}
	model_ptr.token_emb.weight, offset = read_tensor(data, offset, allocator)

	if model_ptr.pos_emb.weight != nil {t.tensor_free(model_ptr.pos_emb.weight)}
	model_ptr.pos_emb.weight, offset = read_tensor(data, offset, allocator)

	// Load each decoder block
	for i in 0 ..< int(num_layers) {
		block := &model_ptr.blocks[i]

		// Load MHA
		if block.mha.q_proj.weights != nil {t.tensor_free(block.mha.q_proj.weights)}
		block.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.q_proj.bias != nil {t.tensor_free(block.mha.q_proj.bias)}
		block.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.mha.k_proj.weights != nil {t.tensor_free(block.mha.k_proj.weights)}
		block.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.k_proj.bias != nil {t.tensor_free(block.mha.k_proj.bias)}
		block.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.mha.v_proj.weights != nil {t.tensor_free(block.mha.v_proj.weights)}
		block.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.v_proj.bias != nil {t.tensor_free(block.mha.v_proj.bias)}
		block.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.mha.out_proj.weights != nil {t.tensor_free(block.mha.out_proj.weights)}
		block.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.out_proj.bias != nil {t.tensor_free(block.mha.out_proj.bias)}
		block.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)

		// Load FFN
		if block.ffn.fc1.weights != nil {t.tensor_free(block.ffn.fc1.weights)}
		block.ffn.fc1.weights, offset = read_tensor(data, offset, allocator)
		if block.ffn.fc1.bias != nil {t.tensor_free(block.ffn.fc1.bias)}
		block.ffn.fc1.bias, offset = read_tensor(data, offset, allocator)

		if block.ffn.fc2.weights != nil {t.tensor_free(block.ffn.fc2.weights)}
		block.ffn.fc2.weights, offset = read_tensor(data, offset, allocator)
		if block.ffn.fc2.bias != nil {t.tensor_free(block.ffn.fc2.bias)}
		block.ffn.fc2.bias, offset = read_tensor(data, offset, allocator)

		// Load LayerNorms
		if block.ln1.gamma != nil {t.tensor_free(block.ln1.gamma)}
		block.ln1.gamma, offset = read_tensor(data, offset, allocator)
		if block.ln1.beta != nil {t.tensor_free(block.ln1.beta)}
		block.ln1.beta, offset = read_tensor(data, offset, allocator)

		if block.ln2.gamma != nil {t.tensor_free(block.ln2.gamma)}
		block.ln2.gamma, offset = read_tensor(data, offset, allocator)
		if block.ln2.beta != nil {t.tensor_free(block.ln2.beta)}
		block.ln2.beta, offset = read_tensor(data, offset, allocator)
	}

	// Load final layer norm
	if model_ptr.final_ln.gamma != nil {t.tensor_free(model_ptr.final_ln.gamma)}
	model_ptr.final_ln.gamma, offset = read_tensor(data, offset, allocator)
	if model_ptr.final_ln.beta != nil {t.tensor_free(model_ptr.final_ln.beta)}
	model_ptr.final_ln.beta, offset = read_tensor(data, offset, allocator)

	// Load output projection
	if model_ptr.output_proj.weights != nil {t.tensor_free(model_ptr.output_proj.weights)}
	model_ptr.output_proj.weights, offset = read_tensor(data, offset, allocator)
	if model_ptr.output_proj.bias != nil {t.tensor_free(model_ptr.output_proj.bias)}
	model_ptr.output_proj.bias, offset = read_tensor(data, offset, allocator)

	return model_ptr, true
}

// ============================================================================
// BERT Model Persistence
// ============================================================================

save_bert_model :: proc(model: ^BERTModel, path: string) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	os.write(file, CHECKPOINT_MAGIC)

	write_i32(file, 18) // BERT type ID
	write_i32(file, i32(model.vocab_size))
	write_i32(file, i32(model.d_model))
	write_i32(file, i32(model.num_heads))
	write_i32(file, i32(model.d_ff))
	write_i32(file, i32(model.num_layers))
	write_i32(file, i32(model.max_seq_len))

	write_tensor(file, model.token_emb.weight)
	write_tensor(file, model.pos_emb.weight)
	write_tensor(file, model.segment_emb.weight)

	write_tensor(file, model.emb_ln.gamma)
	write_tensor(file, model.emb_ln.beta)

	for i in 0 ..< len(model.encoder_blocks) {
		block := &model.encoder_blocks[i]

		write_tensor(file, block.mha.q_proj.weights)
		write_tensor(file, block.mha.q_proj.bias)
		write_tensor(file, block.mha.k_proj.weights)
		write_tensor(file, block.mha.k_proj.bias)
		write_tensor(file, block.mha.v_proj.weights)
		write_tensor(file, block.mha.v_proj.bias)
		write_tensor(file, block.mha.out_proj.weights)
		write_tensor(file, block.mha.out_proj.bias)

		write_tensor(file, block.ffn.fc1.weights)
		write_tensor(file, block.ffn.fc1.bias)
		write_tensor(file, block.ffn.fc2.weights)
		write_tensor(file, block.ffn.fc2.bias)

		write_tensor(file, block.ln1.gamma)
		write_tensor(file, block.ln1.beta)
		write_tensor(file, block.ln2.gamma)
		write_tensor(file, block.ln2.beta)
	}

	write_tensor(file, model.pooler.weights)
	write_tensor(file, model.pooler.bias)

	write_tensor(file, model.mlm_head.weights)
	write_tensor(file, model.mlm_head.bias)

	write_tensor(file, model.nsp_head.weights)
	write_tensor(file, model.nsp_head.bias)

	return true
}

load_bert_model :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^BERTModel,
	bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, false
	}
	defer delete(data, allocator)

	offset := 0

	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, false
		}
	}
	offset += 10

	model_type: i32
	model_type, offset = read_i32(data, offset)
	if model_type != 18 {
		fmt.println("Not a BERT model")
		return nil, false
	}

	vocab_size: i32
	d_model: i32
	num_heads: i32
	d_ff: i32
	num_layers: i32
	max_seq_len: i32
	vocab_size, offset = read_i32(data, offset)
	d_model, offset = read_i32(data, offset)
	num_heads, offset = read_i32(data, offset)
	d_ff, offset = read_i32(data, offset)
	num_layers, offset = read_i32(data, offset)
	max_seq_len, offset = read_i32(data, offset)

	model_ptr := new(BERTModel, allocator)
	model_ptr^ = bert_model_new(
		int(vocab_size),
		int(d_model),
		int(num_heads),
		int(d_ff),
		int(num_layers),
		int(max_seq_len),
		allocator,
	)

	if model_ptr.token_emb.weight != nil {t.tensor_free(model_ptr.token_emb.weight)}
	model_ptr.token_emb.weight, offset = read_tensor(data, offset, allocator)

	if model_ptr.pos_emb.weight != nil {t.tensor_free(model_ptr.pos_emb.weight)}
	model_ptr.pos_emb.weight, offset = read_tensor(data, offset, allocator)

	if model_ptr.segment_emb.weight != nil {t.tensor_free(model_ptr.segment_emb.weight)}
	model_ptr.segment_emb.weight, offset = read_tensor(data, offset, allocator)

	if model_ptr.emb_ln.gamma != nil {t.tensor_free(model_ptr.emb_ln.gamma)}
	model_ptr.emb_ln.gamma, offset = read_tensor(data, offset, allocator)
	if model_ptr.emb_ln.beta != nil {t.tensor_free(model_ptr.emb_ln.beta)}
	model_ptr.emb_ln.beta, offset = read_tensor(data, offset, allocator)

	for i in 0 ..< int(num_layers) {
		block := &model_ptr.encoder_blocks[i]

		if block.mha.q_proj.weights != nil {t.tensor_free(block.mha.q_proj.weights)}
		block.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.q_proj.bias != nil {t.tensor_free(block.mha.q_proj.bias)}
		block.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.mha.k_proj.weights != nil {t.tensor_free(block.mha.k_proj.weights)}
		block.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.k_proj.bias != nil {t.tensor_free(block.mha.k_proj.bias)}
		block.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.mha.v_proj.weights != nil {t.tensor_free(block.mha.v_proj.weights)}
		block.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.v_proj.bias != nil {t.tensor_free(block.mha.v_proj.bias)}
		block.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.mha.out_proj.weights != nil {t.tensor_free(block.mha.out_proj.weights)}
		block.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
		if block.mha.out_proj.bias != nil {t.tensor_free(block.mha.out_proj.bias)}
		block.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)

		if block.ffn.fc1.weights != nil {t.tensor_free(block.ffn.fc1.weights)}
		block.ffn.fc1.weights, offset = read_tensor(data, offset, allocator)
		if block.ffn.fc1.bias != nil {t.tensor_free(block.ffn.fc1.bias)}
		block.ffn.fc1.bias, offset = read_tensor(data, offset, allocator)

		if block.ffn.fc2.weights != nil {t.tensor_free(block.ffn.fc2.weights)}
		block.ffn.fc2.weights, offset = read_tensor(data, offset, allocator)
		if block.ffn.fc2.bias != nil {t.tensor_free(block.ffn.fc2.bias)}
		block.ffn.fc2.bias, offset = read_tensor(data, offset, allocator)

		if block.ln1.gamma != nil {t.tensor_free(block.ln1.gamma)}
		block.ln1.gamma, offset = read_tensor(data, offset, allocator)
		if block.ln1.beta != nil {t.tensor_free(block.ln1.beta)}
		block.ln1.beta, offset = read_tensor(data, offset, allocator)

		if block.ln2.gamma != nil {t.tensor_free(block.ln2.gamma)}
		block.ln2.gamma, offset = read_tensor(data, offset, allocator)
		if block.ln2.beta != nil {t.tensor_free(block.ln2.beta)}
		block.ln2.beta, offset = read_tensor(data, offset, allocator)
	}

	if model_ptr.pooler.weights != nil {t.tensor_free(model_ptr.pooler.weights)}
	model_ptr.pooler.weights, offset = read_tensor(data, offset, allocator)
	if model_ptr.pooler.bias != nil {t.tensor_free(model_ptr.pooler.bias)}
	model_ptr.pooler.bias, offset = read_tensor(data, offset, allocator)

	if model_ptr.mlm_head.weights != nil {t.tensor_free(model_ptr.mlm_head.weights)}
	model_ptr.mlm_head.weights, offset = read_tensor(data, offset, allocator)
	if model_ptr.mlm_head.bias != nil {t.tensor_free(model_ptr.mlm_head.bias)}
	model_ptr.mlm_head.bias, offset = read_tensor(data, offset, allocator)

	if model_ptr.nsp_head.weights != nil {t.tensor_free(model_ptr.nsp_head.weights)}
	model_ptr.nsp_head.weights, offset = read_tensor(data, offset, allocator)
	if model_ptr.nsp_head.bias != nil {t.tensor_free(model_ptr.nsp_head.bias)}
	model_ptr.nsp_head.bias, offset = read_tensor(data, offset, allocator)

	return model_ptr, true
}
// ============================================================================
// Generator Persistence
// ============================================================================

save_generator :: proc(gen: ^Generator, path: string) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	os.write(file, CHECKPOINT_MAGIC)
	write_i32(file, 19) // Generator type ID

	write_tensor(file, gen.fc1.weights)
	write_tensor(file, gen.fc1.bias)
	write_tensor(file, gen.fc2.weights)
	write_tensor(file, gen.fc2.bias)
	write_tensor(file, gen.fc3.weights)
	write_tensor(file, gen.fc3.bias)

	return true
}

load_generator :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^Generator,
	bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, false
	}
	defer delete(data, allocator)

	offset := 0

	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, false
		}
	}
	offset += 10

	model_type: i32
	model_type, offset = read_i32(data, offset)
	if model_type != 19 {
		fmt.println("Not a Generator")
		return nil, false
	}

	gen_ptr := new(Generator, allocator)

	// Load fc1
	gen_ptr.fc1 = linear_layer_new(1, 1, allocator)
	if gen_ptr.fc1.weights != nil {t.tensor_free(gen_ptr.fc1.weights)}
	gen_ptr.fc1.weights, offset = read_tensor(data, offset, allocator)
	if gen_ptr.fc1.bias != nil {t.tensor_free(gen_ptr.fc1.bias)}
	gen_ptr.fc1.bias, offset = read_tensor(data, offset, allocator)

	// Load fc2
	if gen_ptr.fc2.weights != nil {t.tensor_free(gen_ptr.fc2.weights)}
	gen_ptr.fc2.weights, offset = read_tensor(data, offset, allocator)
	if gen_ptr.fc2.bias != nil {t.tensor_free(gen_ptr.fc2.bias)}
	gen_ptr.fc2.bias, offset = read_tensor(data, offset, allocator)

	// Load fc3
	if gen_ptr.fc3.weights != nil {t.tensor_free(gen_ptr.fc3.weights)}
	gen_ptr.fc3.weights, offset = read_tensor(data, offset, allocator)
	if gen_ptr.fc3.bias != nil {t.tensor_free(gen_ptr.fc3.bias)}
	gen_ptr.fc3.bias, offset = read_tensor(data, offset, allocator)

	return gen_ptr, true
}

// ============================================================================
// Discriminator Persistence
// ============================================================================

save_discriminator :: proc(disc: ^Discriminator, path: string) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	os.write(file, CHECKPOINT_MAGIC)
	write_i32(file, 20) // Discriminator type ID

	write_tensor(file, disc.fc1.weights)
	write_tensor(file, disc.fc1.bias)
	write_tensor(file, disc.fc2.weights)
	write_tensor(file, disc.fc2.bias)
	write_tensor(file, disc.fc3.weights)
	write_tensor(file, disc.fc3.bias)

	return true
}

load_discriminator :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^Discriminator,
	bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, false
	}
	defer delete(data, allocator)

	offset := 0

	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, false
		}
	}
	offset += 10

	model_type: i32
	model_type, offset = read_i32(data, offset)
	if model_type != 20 {
		fmt.println("Not a Discriminator")
		return nil, false
	}

	disc_ptr := new(Discriminator, allocator)

	disc_ptr.fc1 = linear_layer_new(1, 1, allocator)
	if disc_ptr.fc1.weights != nil {t.tensor_free(disc_ptr.fc1.weights)}
	disc_ptr.fc1.weights, offset = read_tensor(data, offset, allocator)
	if disc_ptr.fc1.bias != nil {t.tensor_free(disc_ptr.fc1.bias)}
	disc_ptr.fc1.bias, offset = read_tensor(data, offset, allocator)

	if disc_ptr.fc2.weights != nil {t.tensor_free(disc_ptr.fc2.weights)}
	disc_ptr.fc2.weights, offset = read_tensor(data, offset, allocator)
	if disc_ptr.fc2.bias != nil {t.tensor_free(disc_ptr.fc2.bias)}
	disc_ptr.fc2.bias, offset = read_tensor(data, offset, allocator)

	if disc_ptr.fc3.weights != nil {t.tensor_free(disc_ptr.fc3.weights)}
	disc_ptr.fc3.weights, offset = read_tensor(data, offset, allocator)
	if disc_ptr.fc3.bias != nil {t.tensor_free(disc_ptr.fc3.bias)}
	disc_ptr.fc3.bias, offset = read_tensor(data, offset, allocator)

	return disc_ptr, true
}

// ============================================================================
// VAE Encoder Persistence
// ============================================================================

save_vae_encoder :: proc(enc: ^VAEEncoder, path: string) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	os.write(file, CHECKPOINT_MAGIC)
	write_i32(file, 21) // VAEEncoder type ID

	write_tensor(file, enc.fc1.weights)
	write_tensor(file, enc.fc1.bias)
	write_tensor(file, enc.fc_mu.weights)
	write_tensor(file, enc.fc_mu.bias)
	write_tensor(file, enc.fc_logvar.weights)
	write_tensor(file, enc.fc_logvar.bias)

	return true
}

load_vae_encoder :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^VAEEncoder,
	bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, false
	}
	defer delete(data, allocator)

	offset := 0

	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, false
		}
	}
	offset += 10

	model_type: i32
	model_type, offset = read_i32(data, offset)
	if model_type != 21 {
		fmt.println("Not a VAE Encoder")
		return nil, false
	}

	enc_ptr := new(VAEEncoder, allocator)

	enc_ptr.fc1 = linear_layer_new(1, 1, allocator)
	if enc_ptr.fc1.weights != nil {t.tensor_free(enc_ptr.fc1.weights)}
	enc_ptr.fc1.weights, offset = read_tensor(data, offset, allocator)
	if enc_ptr.fc1.bias != nil {t.tensor_free(enc_ptr.fc1.bias)}
	enc_ptr.fc1.bias, offset = read_tensor(data, offset, allocator)

	if enc_ptr.fc_mu.weights != nil {t.tensor_free(enc_ptr.fc_mu.weights)}
	enc_ptr.fc_mu.weights, offset = read_tensor(data, offset, allocator)
	if enc_ptr.fc_mu.bias != nil {t.tensor_free(enc_ptr.fc_mu.bias)}
	enc_ptr.fc_mu.bias, offset = read_tensor(data, offset, allocator)

	if enc_ptr.fc_logvar.weights != nil {t.tensor_free(enc_ptr.fc_logvar.weights)}
	enc_ptr.fc_logvar.weights, offset = read_tensor(data, offset, allocator)
	if enc_ptr.fc_logvar.bias != nil {t.tensor_free(enc_ptr.fc_logvar.bias)}
	enc_ptr.fc_logvar.bias, offset = read_tensor(data, offset, allocator)

	return enc_ptr, true
}

// ============================================================================
// VAE Decoder Persistence
// ============================================================================

save_vae_decoder :: proc(dec: ^VAEDecoder, path: string) -> bool {
	file, err := os.create(path)
	if err != nil {
		fmt.printf("Failed to create file: %v\n", err)
		return false
	}
	defer os.close(file)

	os.write(file, CHECKPOINT_MAGIC)
	write_i32(file, 22) // VAEDecoder type ID

	write_tensor(file, dec.fc1.weights)
	write_tensor(file, dec.fc1.bias)
	write_tensor(file, dec.fc2.weights)
	write_tensor(file, dec.fc2.bias)
	write_tensor(file, dec.fc3.weights)
	write_tensor(file, dec.fc3.bias)

	return true
}

load_vae_decoder :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^VAEDecoder,
	bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return nil, false
	}
	defer delete(data, allocator)

	offset := 0

	if offset + 10 > len(data) {
		fmt.println("File too small")
		return nil, false
	}
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid magic bytes")
			return nil, false
		}
	}
	offset += 10

	model_type: i32
	model_type, offset = read_i32(data, offset)
	if model_type != 22 {
		fmt.println("Not a VAE Decoder")
		return nil, false
	}

	dec_ptr := new(VAEDecoder, allocator)

	dec_ptr.fc1 = linear_layer_new(1, 1, allocator)
	if dec_ptr.fc1.weights != nil {t.tensor_free(dec_ptr.fc1.weights)}
	dec_ptr.fc1.weights, offset = read_tensor(data, offset, allocator)
	if dec_ptr.fc1.bias != nil {t.tensor_free(dec_ptr.fc1.bias)}
	dec_ptr.fc1.bias, offset = read_tensor(data, offset, allocator)

	if dec_ptr.fc2.weights != nil {t.tensor_free(dec_ptr.fc2.weights)}
	dec_ptr.fc2.weights, offset = read_tensor(data, offset, allocator)
	if dec_ptr.fc2.bias != nil {t.tensor_free(dec_ptr.fc2.bias)}
	dec_ptr.fc2.bias, offset = read_tensor(data, offset, allocator)

	if dec_ptr.fc3.weights != nil {t.tensor_free(dec_ptr.fc3.weights)}
	dec_ptr.fc3.weights, offset = read_tensor(data, offset, allocator)
	if dec_ptr.fc3.bias != nil {t.tensor_free(dec_ptr.fc3.bias)}
	dec_ptr.fc3.bias, offset = read_tensor(data, offset, allocator)

	return dec_ptr, true
}

// ============================================================================
// Skip tensor data without loading (for partial loading)
// ============================================================================
skip_tensor :: proc(data: []u8, offset: int) -> int {
	rows_i32, new_offset := read_i32(data, offset)
	cols_i32, newer_offset := read_i32(data, new_offset) // ✅ Use = not :=

	rows := int(rows_i32)
	cols := int(cols_i32)

	if rows == 0 && cols == 0 {
		return newer_offset
	}

	count := rows * cols
	return newer_offset + count * 8
}

// ============================================================================
// Partial Weight Loading for Sequential Models
// ============================================================================
sequential_load_partial :: proc(
	seq: ^Sequential,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	loaded: int,
	skipped: int,
	ok: bool,
) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		fmt.printf("Failed to read file: %v\n", err)
		return 0, 0, false
	}
	defer delete(data, allocator)

	offset := 10
	epoch_i32: i32
	epoch_i32, offset = read_i32(data, offset)

	num_layers_i32: i32
	num_layers_i32, offset = read_i32(data, offset)
	ckpt_layers := int(num_layers_i32)

	model_layers := len(seq.layers)
	loaded = 0
	skipped = 0

	layers_to_process := min(ckpt_layers, model_layers)

	for i in 0 ..< layers_to_process {
		layer_type: i32
		layer_type, offset = read_i32(data, offset)

		model_layer := &seq.layers[i]

		types_match := false
		switch l in model_layer {
		case LinearLayer:
			if layer_type == 0 {types_match = true}
		case Conv2dLayer:
			if layer_type == 1 {types_match = true}
		case MaxPool2dLayer:
			if layer_type == 2 {types_match = true}
		case Activation:
			if layer_type == 3 {types_match = true}
		case FlattenLayer:
			if layer_type == 4 {types_match = true}
		case BatchNorm2dLayer:
			if layer_type == 5 {types_match = true}
		case AvgPool2dLayer:
			if layer_type == 6 {types_match = true}
		case DropoutLayer:
			if layer_type == 7 {types_match = true}
		case RNNLayer:
			if layer_type == 8 {types_match = true}
		case GRULayer:
			if layer_type == 9 {types_match = true}
		case LSTMLayer:
			if layer_type == 10 {types_match = true}
		case EmbeddingLayer:
			if layer_type == 11 {types_match = true}
		case MultiHeadAttentionLayer:
			if layer_type == 12 {types_match = true}
		case LayerNormLayer:
			if layer_type == 13 {types_match = true}
		case FFNLayer:
			if layer_type == 14 {types_match = true}
		case TransformerEncoderBlock:
			if layer_type == 15 {types_match = true}
		case TransformerEncoder:
			if layer_type == 16 {types_match = true}
		case GATLayer:
			if layer_type == 24 {types_match = true}
		}

		if !types_match {
			switch layer_type {
			case 0:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				has_bias: i32
				has_bias, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset)
				if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}
			case 1:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				has_bias: i32
				has_bias, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset)
				if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}
			case 2:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
			case 3:
				_, offset = read_i32(data, offset)
			case 4:
			case 5:
				_, offset = read_i32(data, offset)
				_, offset = read_f64(data, offset)
				_, offset = read_f64(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
			case 6:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
			case 7:
				_, offset = read_f64(data, offset)
			case 8, 9, 10:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
			case 11:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset)
			case 12:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
			case 13:
				_, offset = read_i32(data, offset)
				_, offset = read_f64(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
			case 14:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
			case 15:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				for _ in 0 ..< 12 {
					offset = skip_tensor(data, offset)
				}
			case 16:
				num_layers_skip: i32
				num_layers_skip, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
				for _ in 0 ..< int(num_layers_skip) * 12 {
					offset = skip_tensor(data, offset)
				}
			case 24:
				_, offset = read_i32(data, offset) // in_f
				_, offset = read_i32(data, offset) // out_f
				_, offset = read_i32(data, offset) // num_heads
				has_bias: i32; has_bias, offset = read_i32(data, offset)
				offset = skip_tensor(data, offset) // adjacency
				offset = skip_tensor(data, offset) // linear.weights
				if has_bias != 0 {offset = skip_tensor(data, offset)} 	// linear.bias
				for _ in 0 ..< 8 {offset = skip_tensor(data, offset)} 	// MHA weights/biases
			}
			skipped += 1
			continue
		}

		// Types match - load the layer
		switch &l in model_layer {
		case LinearLayer:
			in_f: i32
			out_f: i32
			has_bias: i32
			in_f, offset = read_i32(data, offset)
			out_f, offset = read_i32(data, offset)
			has_bias, offset = read_i32(data, offset)

			if int(in_f) == l.in_features && int(out_f) == l.out_features {
				if l.weights != nil {t.tensor_free(l.weights)}
				l.weights, offset = read_tensor(data, offset, allocator)
				if has_bias != 0 && l.bias != nil {
					t.tensor_free(l.bias)
					l.bias, offset = read_tensor(data, offset, allocator)
				} else if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}
				skipped += 1
			}
		case Conv2dLayer:
			in_c: i32
			out_c: i32
			k: i32
			stride: i32
			pad: i32
			has_bias: i32
			in_c, offset = read_i32(data, offset)
			out_c, offset = read_i32(data, offset)
			k, offset = read_i32(data, offset)
			stride, offset = read_i32(data, offset)
			pad, offset = read_i32(data, offset)
			has_bias, offset = read_i32(data, offset)

			if int(in_c) == l.in_channels &&
			   int(out_c) == l.out_channels &&
			   int(k) == l.kernel_size {
				if l.weight != nil {t.tensor_free(l.weight)}
				l.weight, offset = read_tensor(data, offset, allocator)
				l.weight.shape = [4]int{int(out_c), int(in_c), int(k), int(k)}
				if has_bias != 0 && l.bias != nil {
					t.tensor_free(l.bias)
					l.bias, offset = read_tensor(data, offset, allocator)
				} else if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}
				skipped += 1
			}
		case BatchNorm2dLayer:
			num_features: i32
			eps: f64
			momentum: f64
			num_features, offset = read_i32(data, offset)
			eps, offset = read_f64(data, offset)
			momentum, offset = read_f64(data, offset)

			if int(num_features) == l.num_features {
				if l.weight != nil {t.tensor_free(l.weight)}
				l.weight, offset = read_tensor(data, offset, allocator)
				if l.bias != nil {t.tensor_free(l.bias)}
				l.bias, offset = read_tensor(data, offset, allocator)
				if l.running_mean != nil {t.tensor_free(l.running_mean)}
				l.running_mean, offset = read_tensor(data, offset, allocator)
				if l.running_var != nil {t.tensor_free(l.running_var)}
				l.running_var, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				skipped += 1
			}
		case RNNLayer:
			in_size: i32
			hidden_size: i32
			in_size, offset = read_i32(data, offset)
			hidden_size, offset = read_i32(data, offset)

			if int(in_size) == l.input_size && int(hidden_size) == l.hidden_size {
				if l.w_ih != nil {t.tensor_free(l.w_ih)}
				l.w_ih, offset = read_tensor(data, offset, allocator)
				if l.w_hh != nil {t.tensor_free(l.w_hh)}
				l.w_hh, offset = read_tensor(data, offset, allocator)
				if l.bias != nil {t.tensor_free(l.bias)}
				l.bias, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				skipped += 1
			}
		case GRULayer:
			in_size: i32
			hidden_size: i32
			in_size, offset = read_i32(data, offset)
			hidden_size, offset = read_i32(data, offset)

			if int(in_size) == l.input_size && int(hidden_size) == l.hidden_size {
				if l.w_ih != nil {t.tensor_free(l.w_ih)}
				l.w_ih, offset = read_tensor(data, offset, allocator)
				if l.w_hh != nil {t.tensor_free(l.w_hh)}
				l.w_hh, offset = read_tensor(data, offset, allocator)
				if l.bias != nil {t.tensor_free(l.bias)}
				l.bias, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				skipped += 1
			}
		case LSTMLayer:
			in_size: i32
			hidden_size: i32
			in_size, offset = read_i32(data, offset)
			hidden_size, offset = read_i32(data, offset)

			if int(in_size) == l.input_size && int(hidden_size) == l.hidden_size {
				if l.w_ih != nil {t.tensor_free(l.w_ih)}
				l.w_ih, offset = read_tensor(data, offset, allocator)
				if l.w_hh != nil {t.tensor_free(l.w_hh)}
				l.w_hh, offset = read_tensor(data, offset, allocator)
				if l.bias != nil {t.tensor_free(l.bias)}
				l.bias, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				skipped += 1
			}
		case MultiHeadAttentionLayer:
			d_model: i32
			num_heads: i32
			d_model, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)

			if int(d_model) == l.d_model && int(num_heads) == l.num_heads {
				if l.q_proj.weights != nil {t.tensor_free(l.q_proj.weights)}
				l.q_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.q_proj.bias != nil {t.tensor_free(l.q_proj.bias)}
				l.q_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.k_proj.weights != nil {t.tensor_free(l.k_proj.weights)}
				l.k_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.k_proj.bias != nil {t.tensor_free(l.k_proj.bias)}
				l.k_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.v_proj.weights != nil {t.tensor_free(l.v_proj.weights)}
				l.v_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.v_proj.bias != nil {t.tensor_free(l.v_proj.bias)}
				l.v_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.out_proj.weights != nil {t.tensor_free(l.out_proj.weights)}
				l.out_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.out_proj.bias != nil {t.tensor_free(l.out_proj.bias)}
				l.out_proj.bias, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				for _ in 0 ..< 8 {
					offset = skip_tensor(data, offset)
				}
				skipped += 1
			}
		case FFNLayer:
			d_model: i32
			d_ff: i32
			d_model, offset = read_i32(data, offset)
			d_ff, offset = read_i32(data, offset)

			if int(d_model) == l.d_model && int(d_ff) == l.d_ff {
				if l.fc1.weights != nil {t.tensor_free(l.fc1.weights)}
				l.fc1.weights, offset = read_tensor(data, offset, allocator)
				if l.fc1.bias != nil {t.tensor_free(l.fc1.bias)}
				l.fc1.bias, offset = read_tensor(data, offset, allocator)
				if l.fc2.weights != nil {t.tensor_free(l.fc2.weights)}
				l.fc2.weights, offset = read_tensor(data, offset, allocator)
				if l.fc2.bias != nil {t.tensor_free(l.fc2.bias)}
				l.fc2.bias, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				for _ in 0 ..< 4 {
					offset = skip_tensor(data, offset)
				}
				skipped += 1
			}
		case TransformerEncoderBlock:
			d_model: i32
			num_heads: i32
			d_ff: i32
			d_model, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)
			d_ff, offset = read_i32(data, offset)

			if int(d_model) == l.d_model && int(num_heads) == l.num_heads && int(d_ff) == l.d_ff {
				if l.ln1.gamma != nil {t.tensor_free(l.ln1.gamma)}
				l.ln1.gamma, offset = read_tensor(data, offset, allocator)
				if l.ln1.beta != nil {t.tensor_free(l.ln1.beta)}
				l.ln1.beta, offset = read_tensor(data, offset, allocator)
				if l.mha.q_proj.weights != nil {t.tensor_free(l.mha.q_proj.weights)}
				l.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.mha.q_proj.bias != nil {t.tensor_free(l.mha.q_proj.bias)}
				l.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.mha.k_proj.weights != nil {t.tensor_free(l.mha.k_proj.weights)}
				l.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.mha.k_proj.bias != nil {t.tensor_free(l.mha.k_proj.bias)}
				l.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.mha.v_proj.weights != nil {t.tensor_free(l.mha.v_proj.weights)}
				l.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.mha.v_proj.bias != nil {t.tensor_free(l.mha.v_proj.bias)}
				l.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.mha.out_proj.weights != nil {t.tensor_free(l.mha.out_proj.weights)}
				l.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
				if l.mha.out_proj.bias != nil {t.tensor_free(l.mha.out_proj.bias)}
				l.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)
				if l.ln2.gamma != nil {t.tensor_free(l.ln2.gamma)}
				l.ln2.gamma, offset = read_tensor(data, offset, allocator)
				if l.ln2.beta != nil {t.tensor_free(l.ln2.beta)}
				l.ln2.beta, offset = read_tensor(data, offset, allocator)
				if l.ffn.fc1.weights != nil {t.tensor_free(l.ffn.fc1.weights)}
				l.ffn.fc1.weights, offset = read_tensor(data, offset, allocator)
				if l.ffn.fc1.bias != nil {t.tensor_free(l.ffn.fc1.bias)}
				l.ffn.fc1.bias, offset = read_tensor(data, offset, allocator)
				if l.ffn.fc2.weights != nil {t.tensor_free(l.ffn.fc2.weights)}
				l.ffn.fc2.weights, offset = read_tensor(data, offset, allocator)
				if l.ffn.fc2.bias != nil {t.tensor_free(l.ffn.fc2.bias)}
				l.ffn.fc2.bias, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				for _ in 0 ..< 12 {
					offset = skip_tensor(data, offset)
				}
				skipped += 1
			}
		case TransformerEncoder:
			num_layers_ckpt: i32
			d_model: i32
			num_heads: i32
			d_ff: i32
			num_layers_ckpt, offset = read_i32(data, offset)
			d_model, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)
			d_ff, offset = read_i32(data, offset)

			if int(num_layers_ckpt) == len(l.blocks) &&
			   int(d_model) == l.d_model &&
			   int(num_heads) == l.num_heads &&
			   int(d_ff) == l.d_ff {
				for block_idx in 0 ..< int(num_layers_ckpt) {
					block := &l.blocks[block_idx]
					if block.ln1.gamma != nil {t.tensor_free(block.ln1.gamma)}
					block.ln1.gamma, offset = read_tensor(data, offset, allocator)
					if block.ln1.beta != nil {t.tensor_free(block.ln1.beta)}
					block.ln1.beta, offset = read_tensor(data, offset, allocator)
					if block.mha.q_proj.weights != nil {t.tensor_free(block.mha.q_proj.weights)}
					block.mha.q_proj.weights, offset = read_tensor(data, offset, allocator)
					if block.mha.q_proj.bias != nil {t.tensor_free(block.mha.q_proj.bias)}
					block.mha.q_proj.bias, offset = read_tensor(data, offset, allocator)
					if block.mha.k_proj.weights != nil {t.tensor_free(block.mha.k_proj.weights)}
					block.mha.k_proj.weights, offset = read_tensor(data, offset, allocator)
					if block.mha.k_proj.bias != nil {t.tensor_free(block.mha.k_proj.bias)}
					block.mha.k_proj.bias, offset = read_tensor(data, offset, allocator)
					if block.mha.v_proj.weights != nil {t.tensor_free(block.mha.v_proj.weights)}
					block.mha.v_proj.weights, offset = read_tensor(data, offset, allocator)
					if block.mha.v_proj.bias != nil {t.tensor_free(block.mha.v_proj.bias)}
					block.mha.v_proj.bias, offset = read_tensor(data, offset, allocator)
					if block.mha.out_proj.weights !=
					   nil {t.tensor_free(block.mha.out_proj.weights)}
					block.mha.out_proj.weights, offset = read_tensor(data, offset, allocator)
					if block.mha.out_proj.bias != nil {t.tensor_free(block.mha.out_proj.bias)}
					block.mha.out_proj.bias, offset = read_tensor(data, offset, allocator)
					if block.ln2.gamma != nil {t.tensor_free(block.ln2.gamma)}
					block.ln2.gamma, offset = read_tensor(data, offset, allocator)
					if block.ln2.beta != nil {t.tensor_free(block.ln2.beta)}
					block.ln2.beta, offset = read_tensor(data, offset, allocator)
					if block.ffn.fc1.weights != nil {t.tensor_free(block.ffn.fc1.weights)}
					block.ffn.fc1.weights, offset = read_tensor(data, offset, allocator)
					if block.ffn.fc1.bias != nil {t.tensor_free(block.ffn.fc1.bias)}
					block.ffn.fc1.bias, offset = read_tensor(data, offset, allocator)
					if block.ffn.fc2.weights != nil {t.tensor_free(block.ffn.fc2.weights)}
					block.ffn.fc2.weights, offset = read_tensor(data, offset, allocator)
					if block.ffn.fc2.bias != nil {t.tensor_free(block.ffn.fc2.bias)}
					block.ffn.fc2.bias, offset = read_tensor(data, offset, allocator)
				}
				loaded += 1
			} else {
				for _ in 0 ..< int(num_layers_ckpt) * 12 {
					offset = skip_tensor(data, offset)
				}
				skipped += 1
			}
		case EmbeddingLayer:
			num_emb: i32
			emb_dim: i32
			num_emb, offset = read_i32(data, offset)
			emb_dim, offset = read_i32(data, offset)

			if int(num_emb) == l.num_embeddings && int(emb_dim) == l.embedding_dim {
				if l.weight != nil {t.tensor_free(l.weight)}
				l.weight, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				skipped += 1
			}
		case LayerNormLayer:
			d_model: i32
			eps: f64
			d_model, offset = read_i32(data, offset)
			eps, offset = read_f64(data, offset)

			if int(d_model) == l.d_model {
				if l.gamma != nil {t.tensor_free(l.gamma)}
				l.gamma, offset = read_tensor(data, offset, allocator)
				if l.beta != nil {t.tensor_free(l.beta)}
				l.beta, offset = read_tensor(data, offset, allocator)
				loaded += 1
			} else {
				offset = skip_tensor(data, offset)
				offset = skip_tensor(data, offset)
				skipped += 1
			}
		case GATLayer:
			in_f: i32; out_f: i32; num_heads: i32; has_bias: i32
			in_f, offset = read_i32(data, offset)
			out_f, offset = read_i32(data, offset)
			num_heads, offset = read_i32(data, offset)
			has_bias, offset = read_i32(data, offset)

			if int(in_f) == l.linear.in_features && int(out_f) == l.linear.out_features {
				// Skip adjacency in partial load (or load it if you prefer)
				offset = skip_tensor(data, offset)

				if l.linear.weights != nil {t.tensor_free(l.linear.weights)}
				l.linear.weights, offset = read_tensor(data, offset, allocator)

				if has_bias != 0 && l.linear.bias != nil {
					t.tensor_free(l.linear.bias)
					l.linear.bias, offset = read_tensor(data, offset, allocator)
				} else if has_bias != 0 {
					offset = skip_tensor(data, offset)
				}

				// Skip MHA weights for partial load (or load them if you want full GAT partial loading)
				for _ in 0 ..< 8 {offset = skip_tensor(data, offset)}

				loaded += 1
			} else {
				offset = skip_tensor(data, offset) // adjacency
				offset = skip_tensor(data, offset) // weights
				if has_bias != 0 {offset = skip_tensor(data, offset)}
				for _ in 0 ..< 8 {offset = skip_tensor(data, offset)}
				skipped += 1
			}
		case MaxPool2dLayer, AvgPool2dLayer, Activation, FlattenLayer, DropoutLayer:
			switch layer_type {
			case 2:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
			case 3:
				_, offset = read_i32(data, offset)
			case 6:
				_, offset = read_i32(data, offset)
				_, offset = read_i32(data, offset)
			case 7:
				_, offset = read_f64(data, offset)
			}
			loaded += 1
		}
	}

	// Skip remaining checkpoint layers
	for i in layers_to_process ..< ckpt_layers {
		layer_type: i32
		layer_type, offset = read_i32(data, offset)
		switch layer_type {
		case 0:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			has_bias: i32
			has_bias, offset = read_i32(data, offset)
			offset = skip_tensor(data, offset)
			if has_bias != 0 {offset = skip_tensor(data, offset)}
		case 1:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			has_bias: i32
			has_bias, offset = read_i32(data, offset)
			offset = skip_tensor(data, offset)
			if has_bias != 0 {offset = skip_tensor(data, offset)}
		case 2:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
		case 3:
			_, offset = read_i32(data, offset)
		case 4:
		case 5:
			_, offset = read_i32(data, offset)
			_, offset = read_f64(data, offset)
			_, offset = read_f64(data, offset)
			offset = skip_tensor(data, offset)
			offset = skip_tensor(data, offset)
			offset = skip_tensor(data, offset)
			offset = skip_tensor(data, offset)
		case 6:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
		case 7:
			_, offset = read_f64(data, offset)
		case 8, 9, 10:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			offset = skip_tensor(data, offset)
			offset = skip_tensor(data, offset)
			offset = skip_tensor(data, offset)
		case 11:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			offset = skip_tensor(data, offset)
		case 12:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			for _ in 0 ..< 8 {
				offset = skip_tensor(data, offset)
			}
		case 13:
			_, offset = read_i32(data, offset)
			_, offset = read_f64(data, offset)
			offset = skip_tensor(data, offset)
			offset = skip_tensor(data, offset)
		case 14:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			for _ in 0 ..< 4 {
				offset = skip_tensor(data, offset)
			}
		case 15:
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			for _ in 0 ..< 12 {
				offset = skip_tensor(data, offset)
			}
		case 16:
			num_layers_skip: i32
			num_layers_skip, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			_, offset = read_i32(data, offset)
			for _ in 0 ..< int(num_layers_skip) * 12 {
				offset = skip_tensor(data, offset)
			}
		}
		skipped += 1
	}

	return loaded, skipped, true
}
