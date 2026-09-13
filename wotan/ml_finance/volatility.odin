package ml_finance

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

VolatilityForecaster :: struct {
	fc1_w:     ^t.Tensor, // [60, 64]
	fc1_b:     ^t.Tensor, // [1, 64]
	fc2_w:     ^t.Tensor, // [64, 64]
	fc2_b:     ^t.Tensor, // [1, 64]
	fc3_w:     ^t.Tensor, // [64, 1]
	fc3_b:     ^t.Tensor, // [1, 1]
	allocator: mem.Allocator,
}

volatility_forecaster_new :: proc(
	input_size: int,
	hidden_size: int,
	seq_len: int,
	allocator: mem.Allocator = context.allocator,
) -> VolatilityForecaster {
	model: VolatilityForecaster
	model.allocator = allocator

	flat_input := seq_len * input_size

	model.fc1_w = _init_weight(flat_input, hidden_size, allocator)
	model.fc1_b = _init_bias(hidden_size, allocator)
	model.fc2_w = _init_weight(hidden_size, hidden_size, allocator)
	model.fc2_b = _init_bias(hidden_size, allocator)
	model.fc3_w = _init_weight(hidden_size, 1, allocator)
	model.fc3_b = _init_bias(1, allocator)

	return model
}

volatility_forecaster_free :: proc(model: ^VolatilityForecaster) {
	t.tensor_free(model.fc1_w)
	t.tensor_free(model.fc1_b)
	t.tensor_free(model.fc2_w)
	t.tensor_free(model.fc2_b)
	t.tensor_free(model.fc3_w)
	t.tensor_free(model.fc3_b)
}

volatility_forecaster_forward :: proc(
	model: ^VolatilityForecaster,
	input: ^t.Tensor,
) -> ^t.Tensor {
	batch := input.data.rows

	// fc1: [batch, 60] @ [60, 64] -> [batch, 64]
	h1_mat := l.matmul_dyn_simd(&input.data, &model.fc1_w.data, model.allocator)
	defer l.matrix_free(&h1_mat)

	for b in 0 ..< batch {
		for h in 0 ..< 64 {
			h1_mat.data[b * 64 + h] += model.fc1_b.data.data[h]
		}
	}

	h1_data := l.matrix_new(f64, batch, 64, model.allocator)
	l.vec_relu_simd(h1_mat.data, h1_data.data)
	h1 := t.tensor_new(h1_data, true, model.allocator)
	h1.shape = [4]int{batch, 64, 1, 1}
	defer t.tensor_free(h1)

	// fc2: [batch, 64] @ [64, 64] -> [batch, 64]
	h2_mat := l.matmul_dyn_simd(&h1.data, &model.fc2_w.data, model.allocator)
	defer l.matrix_free(&h2_mat)

	for b in 0 ..< batch {
		for h in 0 ..< 64 {
			h2_mat.data[b * 64 + h] += model.fc2_b.data.data[h]
		}
	}

	h2_data := l.matrix_new(f64, batch, 64, model.allocator)
	l.vec_relu_simd(h2_mat.data, h2_data.data)
	h2 := t.tensor_new(h2_data, true, model.allocator)
	h2.shape = [4]int{batch, 64, 1, 1}
	defer t.tensor_free(h2)

	// fc3: [batch, 64] @ [64, 1] -> [batch, 1]
	out_mat := l.matmul_dyn_simd(&h2.data, &model.fc3_w.data, model.allocator)
	defer l.matrix_free(&out_mat)

	for b in 0 ..< batch {
		out_mat.data[b] += model.fc3_b.data.data[0]
	}

	out := t.tensor_new(out_mat, true, model.allocator)
	out.shape = [4]int{batch, 1, 1, 1}

	return out
}

_init_weight :: proc(rows: int, cols: int, allocator: mem.Allocator) -> ^t.Tensor {
	data := l.matrix_new(f64, rows, cols, allocator)
	limit := 1.0 / math.sqrt(f64(rows + cols))
	for i in 0 ..< len(data.data) {
		data.data[i] = (rand.float64() * 2.0 - 1.0) * limit
	}
	t := t.tensor_new(data, true, allocator)
	t.shape = [4]int{rows, cols, 1, 1}
	return t
}

_init_bias :: proc(size: int, allocator: mem.Allocator) -> ^t.Tensor {
	data := l.matrix_new(f64, 1, size, allocator)
	t := t.tensor_new(data, true, allocator)
	t.shape = [4]int{1, size, 1, 1}
	return t
}
