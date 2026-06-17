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
			write_i32(file, 9) // ✅ FIX: Use Type ID 9
			write_i32(file, i32(l.input_size))
			write_i32(file, i32(l.hidden_size))
			write_tensor(file, l.w_ih)
			write_tensor(file, l.w_hh)
			write_tensor(file, l.bias)
		case LSTMLayer:
			write_i32(file, 10) // ✅ FIX: Use Type ID 10
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

			// Read weight
			if layer.weight != nil {
				t.tensor_free(layer.weight)
			}

			// Read weight
			layer.weight, offset = read_tensor(data, offset, allocator)

			// ✅ CRITICAL: Set the shape field for Conv2d weights
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

			// ✅ FIX: Use gru_layer_new, not rnn_layer_new
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

			// ✅ FIX: Use gru_layer_new, not rnn_layer_new
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
