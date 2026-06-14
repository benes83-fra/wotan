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
