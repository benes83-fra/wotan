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
// right away. Inside a graph scope they live in the scope's arena like the op
// outputs that use them (graph_end frees them; on the heap they leaked once
// per step, as graph_end does not free heap tensors).

// Constant [1, len(values)] tensor with the given logical shape (zeros if values is nil)
@(private)
_constant :: proc(n: int, shape: [4]int, alloc: mem.Allocator, values: []f64 = nil) -> ^t.Tensor {
	alloc := t._scope_or(alloc)
	m := l.matrix_new(f64, 1, n, alloc)
	if values != nil {copy(m.data, values)}
	c := t.tensor_new(m, false, alloc)
	c.shape = shape
	return c
}

// A forward pass outside a graph scope records its constant intermediates (op
// outputs without gradient, e.g. the outputs of frozen layers on data, which no
// graph links): forward_record_end frees them, or hands those that the graph
// reads to it, and keeps the results passed to it. Inside a graph scope both do
// nothing (graph_end frees everything). Every layer and model forward of the
// library records itself; code that chains layer forwards by hand records the
// whole chain:
//
//	rec := nn.forward_record_begin()
//	h := nn.linear_forward(&frozen, x)
//	out := nn.linear_forward(&head, h)
//	nn.forward_record_end(rec, out)
//
// Results that the code keeps elsewhere (a struct field, a cache) must be
// passed to forward_record_end as well.
forward_record_begin :: proc() -> (recording: bool) {
	if t.graph_active_scope() != nil {return false}
	t.constant_record_begin()
	return true
}

forward_record_end :: proc(recording: bool, keep: ..^t.Tensor) {
	if recording {t.constant_record_end(..keep)}
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
	allocator := t._scope_or(allocator) // an op output: in the active scope, if any
	// an op output (also without gradient: a forward recording then sees it)
	m := t._tensor_new_op(
		l.matrix_new(f64, rows, cols, allocator),
		[4]int{rows, cols, 1, 1},
		x.requires_grad,
		allocator,
	)
	copy(m.data.data, x.data.data)
	if m.requires_grad {
		m.op = .Reshape
		append(&m.inputs, x)
	}
	return m
}

// Position `step` of every sequence of x [batch, seq_len, width] as
// [batch, 1, width] (e.g. BERT's [CLS] token, the last step of an RNN),
// gathered with an embedding lookup into the [batch * seq_len, width] rows of x
// so that gradients reach x
sequence_step :: proc(x: ^t.Tensor, step: int, alloc: mem.Allocator) -> ^t.Tensor {
	batch, seq_len, width := x.shape[0], x.shape[1], x.shape[2]
	assert(x.shape[3] == 1, "sequence_step: x must be [batch, seq_len, width]")
	assert(step >= 0 && step < seq_len, "sequence_step: step out of range")
	rows := t.tensor_reshape(x, [4]int{batch * seq_len, width, 1, 1})
	ids := _constant(batch, {batch, 1, 1, 1}, alloc)
	for b in 0 ..< batch {ids.data.data[b] = f64(b * seq_len + step)}
	out := t.tensor_embedding(ids, rows)
	_constant_done(ids, out)
	return out
}

// Position ids start..start+seq_len-1 for each sequence, shaped [batch, seq_len]
// for an embedding lookup
@(private)
_position_ids :: proc(batch, seq_len: int, alloc: mem.Allocator, start := 0) -> ^t.Tensor {
	ids := _constant(batch * seq_len, {batch, seq_len, 1, 1}, alloc)
	for i in 0 ..< batch * seq_len {ids.data.data[i] = f64(start + i % seq_len)}
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
linear_forward :: proc(layer: ^LinearLayer, x: ^t.Tensor) -> (fwd_out: ^t.Tensor) {
	rec_ := forward_record_begin()
	defer forward_record_end(rec_, fwd_out)
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
