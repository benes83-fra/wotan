package nn

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:mem"
import "core:os"

// ✅ FIX: Define magic as a typed byte array directly
CHECKPOINT_MAGIC :: []u8{'W', 'O', 'T', 'A', 'N', '_', 'C', 'K', 'P', 'T'}

// ============================================================================
// Binary Reader Helpers (Little-Endian, dependency-free)
// ============================================================================

_read_i32 :: proc(data: []u8, offset: int) -> (i32, int) {
	val :=
		i32(data[offset]) |
		(i32(data[offset + 1]) << 8) |
		(i32(data[offset + 2]) << 16) |
		(i32(data[offset + 3]) << 24)
	return val, offset + 4
}

_read_i64 :: proc(data: []u8, offset: int) -> (i64, int) {
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

// ✅ FIX: Use a local mutable variable instead of modifying the parameter
_read_f64_slice :: proc(
	data: []u8,
	offset: int,
	expected_count: int,
	allocator: mem.Allocator,
) -> (
	^t.Tensor,
	int,
) {
	curr_offset := offset // ✅ Create mutable local copy

	// 1. Read rows and cols written by _write_tensor
	rows_i32: i32
	cols_i32: i32
	rows_i32, curr_offset = _read_i32(data, curr_offset)
	cols_i32, curr_offset = _read_i32(data, curr_offset)

	count := int(rows_i32) * int(cols_i32)
	if count != expected_count {
		fmt.printf(
			"Warning: Tensor size mismatch in checkpoint. Expected %d, got %d\n",
			expected_count,
			count,
		)
	}

	// 2. Allocate matrix with correct dimensions
	m := l.matrix_new(f64, int(rows_i32), int(cols_i32), allocator)

	// 3. Read the f64 data
	for i in 0 ..< count {
		byte_offset := curr_offset + i * 8
		val :=
			i64(data[byte_offset]) |
			(i64(data[byte_offset + 1]) << 8) |
			(i64(data[byte_offset + 2]) << 16) |
			(i64(data[byte_offset + 3]) << 24) |
			(i64(data[byte_offset + 4]) << 32) |
			(i64(data[byte_offset + 5]) << 40) |
			(i64(data[byte_offset + 6]) << 48) |
			(i64(data[byte_offset + 7]) << 56)
		m.data[i] = transmute(f64)val
	}

	tensor := t.tensor_new(m, true, allocator)
	return tensor, curr_offset + count * size_of(f64)
}

_write_i32 :: proc(file: ^os.File, val: i32) {
	bytes := [4]u8 {
		u8(val & 0xFF),
		u8((val >> 8) & 0xFF),
		u8((val >> 16) & 0xFF),
		u8((val >> 24) & 0xFF),
	}
	os.write(file, bytes[:])
}

_write_i64 :: proc(file: ^os.File, val: i64) {
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

// ✅ FIX: Now writes nil tensors as 0,0 to maintain symmetry
_write_tensor :: proc(file: ^os.File, tensor: ^t.Tensor) {
	if tensor == nil {
		_write_i32(file, 0)
		_write_i32(file, 0)
		return
	}
	_write_i32(file, i32(tensor.data.rows))
	_write_i32(file, i32(tensor.data.cols))

	count := tensor.data.rows * tensor.data.cols
	for i in 0 ..< count {
		val := tensor.data.data[i]
		// ✅ FIX: Write f64 strictly as little-endian bytes to match reader
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
}

// ============================================================================
// Checkpoint Save / Load
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
		fmt.printf("Failed to open file for writing: %v\n", err)
		return false
	}
	defer os.close(file)

	// 1. Magic Header
	os.write(file, CHECKPOINT_MAGIC)

	// 2. Epoch & Hyperparams
	_write_i32(file, i32(epoch))
	_write_i32(file, i32(len(model.layers)))

	// 3. Layers
	for layer in model.layers {
		switch l in layer {
		case LinearLayer:
			_write_i32(file, 0)
			_write_i32(file, i32(l.in_features))
			_write_i32(file, i32(l.out_features))
			_write_i32(file, i32(bool(l.bias != nil)))
			_write_tensor(file, l.weights)
			if l.bias != nil {_write_tensor(file, l.bias)}

		case Conv2dLayer:
			_write_i32(file, 1)
			_write_i32(file, i32(l.in_channels))
			_write_i32(file, i32(l.out_channels))
			_write_i32(file, i32(l.kernel_size))
			_write_i32(file, i32(l.stride))
			_write_i32(file, i32(l.padding))
			_write_i32(file, i32(bool(l.bias != nil)))
			_write_tensor(file, l.weight)
			if l.bias != nil {_write_tensor(file, l.bias)}

		case MaxPool2dLayer:
			_write_i32(file, 2)
			_write_i32(file, i32(l.kernel_size))
			_write_i32(file, i32(l.stride))

		case Activation:
			_write_i32(file, 3)
			_write_i32(file, i32(l))

		case FlattenLayer:
			_write_i32(file, 4)
		}
	}

	// 4. Optimizer State (Adam)
	_write_i32(file, 0)
	_write_i64(file, i64(opt.timestep))
	_write_i32(file, i32(len(opt.parameters)))

	for i in 0 ..< len(opt.parameters) {
		m1_count := len(opt.moment_1[i])
		_write_i32(file, i32(m1_count))
		for j in 0 ..< m1_count {
			val := opt.moment_1[i][j]
			// ✅ FIX: Write f64 strictly as little-endian bytes
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

		m2_count := len(opt.moment_2[i])
		_write_i32(file, i32(m2_count))
		for j in 0 ..< m2_count {
			val := opt.moment_2[i][j]
			// ✅ FIX: Write f64 strictly as little-endian bytes
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
	}

	return true
}

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
		fmt.printf("Failed to read checkpoint: %v\n", err)
		return nil, nil, 0, false
	}
	defer delete(data, allocator)

	offset := 0

	// 1. Magic Header
	if offset + 10 > len(data) {
		fmt.println("Checkpoint too small")
		return nil, nil, 0, false
	}
	// ✅ FIX: Store constant in local variable before indexing
	magic := CHECKPOINT_MAGIC
	for i in 0 ..< 10 {
		if data[offset + i] != magic[i] {
			fmt.println("Invalid checkpoint magic")
			return nil, nil, 0, false
		}
	}
	offset += 10

	// 2. Epoch
	ep_i32: i32
	ep_i32, offset = _read_i32(data, offset)
	epoch = int(ep_i32)

	// 3. Model
	model = new(Sequential, allocator)
	model.allocator = allocator
	model.layers = make([dynamic]Layer, 0, allocator)

	num_layers_i32: i32
	num_layers_i32, offset = _read_i32(data, offset)

	for _ in 0 ..< int(num_layers_i32) {
		layer_type: i32
		layer_type, offset = _read_i32(data, offset)

		if layer_type == 0 { 	// Linear
			in_f: i32
			out_f: i32
			has_bias: i32
			in_f, offset = _read_i32(data, offset)
			out_f, offset = _read_i32(data, offset)
			has_bias, offset = _read_i32(data, offset)

			layer := linear_layer_new(int(in_f), int(out_f), allocator)
			layer.weights, offset = _read_f64_slice(
				data,
				offset,
				int(in_f) * int(out_f),
				allocator,
			)
			if has_bias != 0 {
				layer.bias, offset = _read_f64_slice(data, offset, int(out_f), allocator)
			} else {
				t.tensor_free(layer.bias)
				layer.bias = nil
			}
			append(&model.layers, layer)

		} else if layer_type == 1 { 	// Conv2d
			in_c: i32
			out_c: i32
			k: i32
			stride: i32
			pad: i32
			has_bias: i32
			in_c, offset = _read_i32(data, offset)
			out_c, offset = _read_i32(data, offset)
			k, offset = _read_i32(data, offset)
			stride, offset = _read_i32(data, offset)
			pad, offset = _read_i32(data, offset)
			has_bias, offset = _read_i32(data, offset)

			layer := conv2d_layer_new(
				int(in_c),
				int(out_c),
				int(k),
				int(stride),
				int(pad),
				has_bias != 0,
				allocator,
			)
			layer.weight, offset = _read_f64_slice(
				data,
				offset,
				int(out_c) * int(in_c) * int(k) * int(k),
				allocator,
			)
			if has_bias != 0 {
				layer.bias, offset = _read_f64_slice(data, offset, int(out_c), allocator)
			} else {
				t.tensor_free(layer.bias)
				layer.bias = nil
			}
			append(&model.layers, layer)

		} else if layer_type == 2 { 	// MaxPool
			k: i32
			s: i32
			k, offset = _read_i32(data, offset)
			s, offset = _read_i32(data, offset)
			append(&model.layers, maxpool2d_layer_new(int(k), int(s)))

		} else if layer_type == 3 { 	// Activation
			act: i32
			act, offset = _read_i32(data, offset)
			append(&model.layers, Activation(act))

		} else if layer_type == 4 { 	// Flatten
			append(&model.layers, FlattenLayer{})
		}
	}

	// 4. Optimizer
	opt_type: i32
	opt_type, offset = _read_i32(data, offset)
	if opt_type != 0 {
		fmt.println("Unsupported optimizer type")
		return nil, nil, 0, false
	}

	opt = new(Adam, allocator)
	timestep_i64: i64
	timestep_i64, offset = _read_i64(data, offset)
	opt.timestep = int(timestep_i64)

	opt.learning_rate = 0.001
	opt.beta_1 = 0.9
	opt.beta_2 = 0.999
	opt.epsilon = 1e-8
	opt.allocator = allocator

	num_params_i32: i32
	num_params_i32, offset = _read_i32(data, offset)

	opt.parameters = make([dynamic]^t.Tensor, 0, allocator)
	opt.moment_1 = make([dynamic][]f64, 0, allocator)
	opt.moment_2 = make([dynamic][]f64, 0, allocator)

	for i in 0 ..< int(num_params_i32) {
		// Read moment_1
		m1_count_i32: i32
		m1_count_i32, offset = _read_i32(data, offset)
		m1_count := int(m1_count_i32)
		m1 := make([]f64, m1_count, allocator)

		for j in 0 ..< m1_count {
			byte_offset := offset + j * 8
			val :=
				i64(data[byte_offset]) |
				(i64(data[byte_offset + 1]) << 8) |
				(i64(data[byte_offset + 2]) << 16) |
				(i64(data[byte_offset + 3]) << 24) |
				(i64(data[byte_offset + 4]) << 32) |
				(i64(data[byte_offset + 5]) << 40) |
				(i64(data[byte_offset + 6]) << 48) |
				(i64(data[byte_offset + 7]) << 56)
			m1[j] = transmute(f64)val
		}
		offset += m1_count * size_of(f64)
		append(&opt.moment_1, m1)

		// Read moment_2
		m2_count_i32: i32
		m2_count_i32, offset = _read_i32(data, offset)
		m2_count := int(m2_count_i32)
		m2 := make([]f64, m2_count, allocator)

		for j in 0 ..< m2_count {
			byte_offset := offset + j * 8
			val :=
				i64(data[byte_offset]) |
				(i64(data[byte_offset + 1]) << 8) |
				(i64(data[byte_offset + 2]) << 16) |
				(i64(data[byte_offset + 3]) << 24) |
				(i64(data[byte_offset + 4]) << 32) |
				(i64(data[byte_offset + 5]) << 40) |
				(i64(data[byte_offset + 6]) << 48) |
				(i64(data[byte_offset + 7]) << 56)
			m2[j] = transmute(f64)val
		}
		offset += m2_count * size_of(f64)
		append(&opt.moment_2, m2)
	}

	return model, opt, epoch, true
}
