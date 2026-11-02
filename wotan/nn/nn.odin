package nn

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Constant op inputs
// ============================================================================
// Forward passes create constant tensors (zero states, position ids, masks)
// as op inputs. When the op output is part of a graph the backward pass may
// still read them, so the graph owns and frees them; otherwise they are freed
// right away.

// Constant [1, len(values)] tensor with the given logical shape (zeros if values is nil)
@(private)
_constant :: proc(n: int, shape: [4]int, alloc: mem.Allocator, values: []f64 = nil) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, n, alloc)
	if values != nil {copy(m.data, values)}
	c := t.tensor_new(m, false, alloc)
	c.shape = shape
	return c
}

// Hands a constant input of `out` to the graph, or frees it if there is none
@(private)
_constant_done :: proc(c, out: ^t.Tensor) {
	if out.requires_grad {
		c.owned_by_graph = true
	} else {
		t.tensor_free(c)
	}
}

// x as a true [rows, cols] matrix: row-wise ops (softmax, cross-entropy, a
// standard matmul) work on data rows, while sequence tensors are stored flat.
// Gradients flow back to x.
as_matrix :: proc(
	x: ^t.Tensor,
	rows, cols: int,
	allocator: mem.Allocator = context.allocator,
) -> ^t.Tensor {
	assert(rows * cols == len(x.data.data), "as_matrix: size mismatch")
	m := t.tensor_new(
		l.matrix_new(f64, rows, cols, allocator),
		x.requires_grad && t.grad_enabled(),
		allocator,
	)
	copy(m.data.data, x.data.data)
	m.shape = [4]int{rows, cols, 1, 1}
	if m.requires_grad {
		m.op = .Reshape
		append(&m.inputs, x)
	}
	return m
}

// Position ids 0..seq_len-1 for each sequence, shaped [batch, seq_len] for an embedding lookup
@(private)
_position_ids :: proc(batch, seq_len: int, alloc: mem.Allocator) -> ^t.Tensor {
	ids := _constant(batch * seq_len, {batch, seq_len, 1, 1}, alloc)
	for i in 0 ..< batch * seq_len {ids.data.data[i] = f64(i % seq_len)}
	return ids
}

// ============================================================================
// 1. Linear Layer (Dense Layer)
// ============================================================================

LinearLayer :: struct {
	weights:      ^t.Tensor, // [in_features, out_features]
	bias:         ^t.Tensor, // [1, out_features]
	in_features:  int,
	out_features: int,
}

// Fills data with U(-limit, limit)
init_uniform :: proc(data: []f64, limit: f64) {
	for &v in data {v = (rand.float64() * 2.0 - 1.0) * limit}
}

// Xavier/Glorot uniform initialization: U(-a, a) with a = sqrt(6 / (fan_in +
// fan_out)), i.e. variance 2 / (fan_in + fan_out)
xavier_uniform :: proc(data: []f64, fan_in, fan_out: int) {
	init_uniform(data, math.sqrt(6.0 / f64(fan_in + fan_out)))
}

// linear_layer_new creates a linear layer with Xavier-uniform weights and a
// zero bias.
linear_layer_new :: proc(
	in_features: int,
	out_features: int,
	allocator: mem.Allocator = context.allocator,
) -> LinearLayer {
	layer: LinearLayer
	layer.in_features = in_features
	layer.out_features = out_features

	w_data := l.matrix_new(f64, in_features, out_features, allocator)
	xavier_uniform(w_data.data, in_features, out_features)
	layer.weights = t.tensor_new(w_data, true, allocator)
	layer.bias = t.tensor_new(l.matrix_new(f64, 1, out_features, allocator), true, allocator)

	return layer
}

// linear_forward performs the forward pass: Y = X @ W + b
// X: [batch_size, in_features]
// W: [in_features, out_features]
// b: [1, out_features]
// Y: [batch_size, out_features]
linear_forward :: proc(layer: ^LinearLayer, x: ^t.Tensor) -> ^t.Tensor {
	// 1. Matrix Multiplication: X @ W
	out := t.tensor_matmul(x, layer.weights)

	// 2. Add Bias: (X @ W) + b
	out = t.tensor_add_bias(out, layer.bias)

	return out
}

linear_layer_free :: proc(layer: ^LinearLayer) {
	if layer.weights != nil {t.tensor_free(layer.weights)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}
