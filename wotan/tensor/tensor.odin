
package tensor

import l "../linalg"
import "../util"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

global_tensors_created: int = 0
global_tensors_freed: int = 0
// ============================================================================
// 1. The Tensor Struct & Graph Nodes
// ============================================================================

// Op defines the operation that created this tensor.
// We use an enum instead of function pointers to keep it Odin-idiomatic.
Op :: enum {
	None,
	Add,
	Sub,
	Mul,
	MatMul,
	Sum,
	Mean,
	Neg,
	Relu,
	Sigmoid, // ✅ ADD
	Tanh, // ✅ ADD
	LeakyReLU, // ✅ ADD
	AddBias,
	MSELoss,
	CrossEntropy,
	Dropout,
	Conv2d,
	Flatten,
	MaxPool2d,
	AvgPool2d,
	BatchNorm2d,
	RNN,
	GRU,
	LSTM,
	Embedding,
	ScaledDotProductAttention,
	FlashAttention,
	PermuteMHA,
	PermuteMHAInverse,
	LayerNorm,
	MaskedScaledDotProductAttention,
	BinaryCrossEntropy,
	KLDivergence,
	Scale,
	Reparameterize,
	Exp,
	Sqrt,
	Clamp,
	Log,
	Div,
	NormCDF,
	SumDim1,
	Softmax,
	GateMul,
	TopKMask,
	Entropy,
	BCELoss,
	PermuteLOB,
	SharpeLoss,
	Concat,
	Reshape,
	NormalizeTime,
	LogSumExpDim1,
	Softplus,
	SSM,
	SliceGate,
	ProspectValue,
	ProbWeighting,
	ProspectUtility,
}

PoolParams :: struct {
	kH:     int,
	kW:     int,
	stride: int,
	pad:    int,
}

Tensor :: struct {
	data:           l.Matrix(f64),
	grad:           l.Matrix(f64),
	requires_grad:  bool,
	op:             Op,
	inputs:         [dynamic]^Tensor,
	allocator:      mem.Allocator,
	int_metadata:   [dynamic]int,
	f64_metadata:   [dynamic]f64, // float op parameters (scalars, eps, alpha, ...) for backward
	dropout_mask:   []f64,
	shape:          [4]int,
	conv_params:    ConvParams,
	pool_params:    PoolParams,
	owned_by_graph: bool, // ✅ ADD THIS
}

ConvParams :: struct {
	kH:     int,
	kW:     int,
	stride: int,
	pad:    int,
}
// ============================================================================
// 2. Construction & Lifecycle
// ============================================================================

tensor_new :: proc(
	data: l.Matrix(f64),
	requires_grad: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	if len(data.data) == 0 {
		fmt.printf(
			"WARNING: Creating tensor with empty data! rows=%d, cols=%d\n",
			data.rows,
			data.cols,
		)
	}
	t := new(Tensor, allocator)
	t.data = data
	t.requires_grad = requires_grad
	t.op = .None
	t.allocator = allocator

	t.inputs = make([dynamic]^Tensor, 0, allocator)
	t.int_metadata = make([dynamic]int, 0, allocator)
	t.f64_metadata = make([dynamic]f64, 0, allocator)
	_scope_note_new(allocator)
	// Default logical shape is the matrix itself; callers override it for 3D/4D data
	t.shape = [4]int{data.rows, data.cols, 1, 1}
	// Initialize gradients to zero
	if requires_grad {
		t.grad = l.matrix_new(f64, data.rows, data.cols, allocator)
		// (In a real implementation, we'd zero it out here, but matrix_new usually does)
	}
	global_tensors_created += 1
	return t
}

// _tensor_new_op creates the output node of an operation. Unlike tensor_new it
// does not allocate a gradient buffer: tensor_backward allocates gradients for
// every node it walks, so forward-only passes (e.g. inference with trainable
// weights) never pay for gradient memory. To seed a custom upstream gradient on
// an op output before tensor_backward, call tensor_ensure_grad on it first.
//
// Every op must state the logical shape of its output; it is validated against
// the data size so `shape` and `data` can never silently disagree.
_tensor_new_op :: proc(
	data: l.Matrix(f64),
	shape: [4]int,
	requires_grad: bool,
	allocator: mem.Allocator = context.allocator,
	loc := #caller_location,
) -> ^Tensor {
	t := new(Tensor, allocator)
	t.data = data
	t.shape = shape
	_check_shape(t, loc)
	t.requires_grad = requires_grad
	t.op = .None
	t.allocator = allocator
	t.inputs = make([dynamic]^Tensor, 0, allocator)
	t.int_metadata = make([dynamic]int, 0, allocator)
	t.f64_metadata = make([dynamic]f64, 0, allocator)
	_scope_note_new(allocator)
	global_tensors_created += 1
	return t
}

SCALAR_SHAPE :: [4]int{1, 1, 1, 1}

// _check_shape panics if a tensor's logical shape does not describe its data.
// Ops call it on every input whose shape they read, and on every output.
_check_shape :: proc(t: ^Tensor, loc := #caller_location) {
	numel := t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3]
	if numel != len(t.data.data) {
		panic(
			fmt.tprintf(
				"tensor shape %v (%d elements) does not match its data (%d x %d = %d elements)",
				t.shape,
				numel,
				t.data.rows,
				t.data.cols,
				len(t.data.data),
			),
			loc,
		)
	}
}

// _flattened_matmul_shape: output shape of [.., in_features] @ [in_features, out_features]
// for a flattened [1, N*in] input. The feature dim is the dim d with
// shape[d] == in_features whose leading dims multiply to N (all later dims are 1),
// e.g. [B, S, in, 1] -> [B, S, out, 1]. Dim 2 (the [B, S, D, 1] sequence convention)
// is tried first, then the rest front to back; this also disambiguates
// in_features == 1 ([B, S, 1, 1] -> [B, S, out, 1]).
// If no such dim exists the result is a plain [N, out].
_flattened_matmul_shape :: proc(shape: [4]int, N, in_features, out_features: int) -> [4]int {
	for d in ([4]int{2, 0, 1, 3}) {
		lead := 1
		for i in 0 ..< d {lead *= shape[i]}
		trail := 1
		for i in d + 1 ..< 4 {trail *= shape[i]}
		if shape[d] == in_features && lead == N && trail == 1 {
			out := shape
			out[d] = out_features
			return out
		}
	}
	return [4]int{N, out_features, 1, 1}
}

tensor_free :: proc(t: ^Tensor) {
	if t == nil {return}
	if t.data.data != nil {l.matrix_free(&t.data)}
	if t.grad.data != nil {l.matrix_free(&t.grad)}

	delete(t.inputs)
	delete(t.int_metadata)
	delete(t.f64_metadata)
	if t.dropout_mask != nil {delete(t.dropout_mask, t.allocator)}
	global_tensors_freed += 1
	_scope_note_free(t.allocator)
	free(t, t.allocator)

}

tensor_zero_grad :: proc(t: ^Tensor) {
	if t.grad.data != nil {
		// Odin's compiler will auto-vectorize this simple zero loop perfectly.
		for i in 0 ..< len(t.grad.data) {
			t.grad.data[i] = 0.0
		}
	}
}

// tensor_dropout randomly zeroes elements with probability drop_prob.
// Uses Inverted Dropout: scales remaining elements by 1/(1-p) during training.
tensor_dropout :: proc(a: ^Tensor, drop_prob: f64, training: bool) -> ^Tensor {
	// If not training, or drop_prob is 0, just return the input tensor directly.
	// This is safe for tensor_free_graph because 'a' will have op == .None (or its original op).
	if !training || drop_prob <= 0.0 {
		return a
	}

	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	n := len(a.data.data)

	// Allocate the mask to remember what we dropped for the backward pass
	mask := make([]f64, n, _op_alloc(a))
	scale := 1.0 / (1.0 - drop_prob)

	for i in 0 ..< n {
		if rand.float64() > drop_prob {
			out_data.data[i] = a.data.data[i] * scale
			mask[i] = scale
		} else {
			out_data.data[i] = 0.0
			mask[i] = 0.0
		}
	}

	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	out.dropout_mask = mask

	if out.requires_grad {
		out.op = .Dropout
		append(&out.inputs, a)
	}

	return out
}
// tensor_sum reduces a tensor to a single scalar value (1x1 matrix)
tensor_sum :: proc(a: ^Tensor) -> ^Tensor {
	// ✅ SIMD Optimization: Fast reduction sum
	total := l.sum_simd(a.data.data)

	// Create a 1x1 matrix for the scalar output
	out_data := l.matrix_new(f64, 1, 1, _op_alloc(a))
	out_data.data[0] = total

	// 2. Create output tensor
	out := _tensor_new_op(out_data, SCALAR_SHAPE, a.requires_grad, _op_alloc(a))
	// 3. Record graph
	if out.requires_grad {
		out.op = .Sum
		append(&out.inputs, a)
	}

	return out
}

// ============================================================================
// 3. Operations (Building the Graph)
// ============================================================================

tensor_add :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	// ✅ FIX: Check dimensions
	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		panic("tensor_add: dimension mismatch")
	}

	// 1. Perform the forward pass using your existing SIMD add!
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_add_simd(a.data.data, b.data.data, out_data.data)

	// 2. Create the output tensor
	out := _tensor_new_op(out_data, a.shape, a.requires_grad || b.requires_grad, _op_alloc(a))
	// 3. Record the graph
	if out.requires_grad {
		out.op = .Add
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}

// tensor_relu applies the Rectified Linear Unit: max(0, x)
tensor_relu :: proc(a: ^Tensor) -> ^Tensor {

	// 1. Forward pass
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_relu_simd(a.data.data, out_data.data) // ✅ SIMD ReLU

	// 2. Create output tensor
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	// 3. Record graph
	if out.requires_grad {
		out.op = .Relu
		append(&out.inputs, a)
	}

	return out
}
tensor_gelu :: proc(a: ^Tensor) -> ^Tensor {
	// GELU(x) = x * Φ(x) where Φ is the standard normal CDF
	cdf := tensor_norm_cdf(a)
	out := tensor_mul(a, cdf)
	// Free intermediate if not building graph
	if !a.requires_grad {
		tensor_free(cdf)
	}
	return out
}

// tensor_mul creates a new tensor C = A * B (element-wise)
tensor_mul :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {


	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		fmt.printf("DEBUG PANIC: tensor_mul dimension mismatch!\n")
		fmt.printf("  Tensor A: rows=%d, cols=%d\n", a.data.rows, a.data.cols)
		fmt.printf("  Tensor B: rows=%d, cols=%d\n", b.data.rows, b.data.cols)
		panic("tensor_mul: dimension mismatch")
	}

	// 1. Forward pass using SIMD
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_mul_simd(a.data.data, b.data.data, out_data.data)

	// 2. Create output tensor
	out := _tensor_new_op(out_data, a.shape, a.requires_grad || b.requires_grad, _op_alloc(a))
	// 3. Record graph
	if out.requires_grad {
		out.op = .Mul
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}
// tensor_add_bias adds a 1xD bias vector to every row of an NxD matrix
// a: [N, D], bias: [1, D] -> out: [N, D]
tensor_add_bias :: proc(a: ^Tensor, bias: ^Tensor) -> ^Tensor {
	D := bias.data.cols
	if bias.data.rows != 1 {
		panic("tensor_add_bias: bias must be a 1xD row vector")
	}
	bias_slice := bias.data.data[0:D]
	// ✅ NEW: Handle flattened 3D tensor
	if a.data.rows == 1 && a.data.cols != D {
		if a.data.cols % D == 0 {
			N := a.data.cols / D
			out_data := l.matrix_new(f64, 1, a.data.cols, _op_alloc(a))

			for i in 0 ..< N {
				row_out := out_data.data[i * D:(i + 1) * D]
				row_a := a.data.data[i * D:(i + 1) * D]
				l.vec_add_simd(row_a, bias_slice, row_out)
			}

			out := _tensor_new_op(
				out_data,
				a.shape,
				a.requires_grad || bias.requires_grad,
				_op_alloc(a),
			)
			if out.requires_grad {
				out.op = .AddBias
				append(&out.inputs, a)
				append(&out.inputs, bias)
				append(&out.int_metadata, N)
				append(&out.int_metadata, D)
			}
			return out
		}
	}

	// Standard 2D path
	if a.data.cols != D {
		panic("tensor_add_bias: column mismatch")
	}

	N := a.data.rows
	out_data := l.matrix_new(f64, N, D, _op_alloc(a))

	for i in 0 ..< N {
		row_out := out_data.data[i * D:(i + 1) * D]
		row_a := a.data.data[i * D:(i + 1) * D]
		l.vec_add_simd(row_a, bias_slice, row_out)
	}

	out := _tensor_new_op(out_data, a.shape, a.requires_grad || bias.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .AddBias
		append(&out.inputs, a)
		append(&out.inputs, bias)
	}

	return out
}


// tensor_mse_loss calculates Mean Squared Error: mean((pred - target)^2)
// This is a Fused SIMD operation that avoids creating intermediate Sub/Mul tensors!
// tensor_mse_loss calculates Mean Squared Error: mean((pred - target)^2)
tensor_mse_loss :: proc(
	pred: ^Tensor,
	target: ^Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	if pred.data.rows != target.data.rows || pred.data.cols != target.data.cols {
		panic("tensor_mse_loss: shape mismatch")
	}

	n := f64(len(pred.data.data))

	diff := make([]f64, len(pred.data.data), allocator)
	l.vec_sub_simd(pred.data.data, target.data.data, diff)
	squared_sum := l.dot_simd(diff, diff)
	defer delete(diff, allocator)

	loss_val := squared_sum / n

	out_data := l.matrix_new(f64, 1, 1, _op_alloc(pred))
	out_data.data[0] = loss_val

	out := _tensor_new_op(
		out_data,
		SCALAR_SHAPE,
		pred.requires_grad || target.requires_grad,
		_op_alloc(pred),
	)
	// ✅ FIX: Restored the correct MSE graph recording!
	if out.requires_grad {
		out.op = .MSELoss
		append(&out.inputs, pred)
		append(&out.inputs, target)
	}
	return out
}


// tensor_cross_entropy_loss calculates the loss for multi-class classification.
tensor_cross_entropy_loss :: proc(logits: ^Tensor, target_indices: []int) -> ^Tensor {
	// ✅ FIX: Handle flattened 3D tensor [1, N*C] where N = batch*seq_len, C = vocab_size
	N := logits.data.rows
	C := logits.data.cols

	// Detect flattened 3D tensor
	if N == 1 && C > len(target_indices) {
		if C % len(target_indices) == 0 {
			C = C / len(target_indices)
			N = len(target_indices)
		}
	}

	if N != len(target_indices) {
		panic(
			fmt.aprintf(
				"tensor_cross_entropy_loss: batch size mismatch (N=%d, targets=%d)",
				N,
				len(target_indices),
			),
		)
	}

	loss := 0.0
	for i in 0 ..< N {
		max_val := -math.F64_MAX
		for j in 0 ..< C {
			v := logits.data.data[i * C + j]
			if v > max_val {max_val = v}
		}

		sum_exp := 0.0
		for j in 0 ..< C {
			sum_exp += math.exp(logits.data.data[i * C + j] - max_val)
		}

		log_sum_exp := math.ln(sum_exp) + max_val

		target_class := target_indices[i]
		log_prob := logits.data.data[i * C + target_class] - log_sum_exp
		loss -= log_prob
	}

	loss /= f64(N)

	out_data := l.matrix_new(f64, 1, 1, _op_alloc(logits))
	out_data.data[0] = loss

	out := _tensor_new_op(out_data, SCALAR_SHAPE, logits.requires_grad, _op_alloc(logits))

	if out.requires_grad {
		out.op = .CrossEntropy
		append(&out.inputs, logits)
		// Save indices and dimensions for the backward pass
		append(&out.int_metadata, N)
		append(&out.int_metadata, C)
		for idx in target_indices {
			append(&out.int_metadata, idx)
		}
	}
	return out
}


// tensor_flatten reshapes a 4D tensor (N, C, H, W) to 2D (N, C*H*W)
tensor_flatten :: proc(input: ^Tensor) -> ^Tensor {
	_check_shape(input)
	N := input.shape[0]
	features := input.shape[1] * input.shape[2] * input.shape[3]

	// Just create a new tensor with the same data but different shape
	out_data := l.matrix_new(f64, N, features, _op_alloc(input))
	copy(out_data.data, input.data.data)

	out := _tensor_new_op(
		out_data,
		[4]int{N, features, 1, 1},
		input.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .Flatten
		append(&out.inputs, input)
	}

	return out
}


// ============================================================================
// 4. The Backward Engine (Chain Rule)
// ============================================================================

// Graph walks use an explicit stack instead of recursion: unrolled recurrent
// nets and long accumulation loops build graphs thousands of nodes deep, which
// overflowed the call stack. Each frame remembers the next input to visit, so
// the visiting order is exactly that of the recursive depth-first walk.
_Walk_Frame :: struct {
	node: ^Tensor,
	next: int,
}

// Helper: Topological Sort to determine the order of operations
// (post-order depth-first: every node after all of its inputs)
_build_topo :: proc(node: ^Tensor, topo: ^[dynamic]^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil || visited[node] {return}
	stack := make([dynamic]_Walk_Frame, 0, 64, context.temp_allocator)
	defer delete(stack)
	visited[node] = true
	append(&stack, _Walk_Frame{node, 0})
	for len(stack) > 0 {
		top := &stack[len(stack) - 1]
		if top.next < len(top.node.inputs) {
			input := top.node.inputs[top.next]
			top.next += 1
			if input != nil && !visited[input] {
				visited[input] = true
				append(&stack, _Walk_Frame{input, 0})
			}
			continue
		}
		append(topo, top.node)
		pop(&stack)
	}
}
// tensor_ensure_grad ensures a tensor has a gradient matrix allocated
tensor_ensure_grad :: proc(t: ^Tensor) {
	if t.requires_grad && t.grad.data == nil {
		t.grad = l.matrix_new(f64, t.data.rows, t.data.cols, t.allocator)
	}
}

// tensor_max_pool2d performs max pooling.
// input: (N, C, H, W) -> output: (N, C, H_out, W_out)
tensor_max_pool2d :: proc(input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	_check_shape(input)
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	out_h := (H - kH) / stride + 1
	out_w := (W - kW) / stride + 1

	out_data := l.matrix_new(f64, 1, N * C * out_h * out_w, _op_alloc(input))

	// We need to store the indices of the max values for the backward pass
	indices := make([]int, N * C * out_h * out_w, _op_alloc(input))

	for n in 0 ..< N {
		for c in 0 ..< C {
			for oh in 0 ..< out_h {
				for ow in 0 ..< out_w {
					max_val := -math.F64_MAX
					max_idx := 0

					for kh in 0 ..< kH {
						for kw in 0 ..< kW {
							ih := oh * stride + kh
							iw := ow * stride + kw
							idx := n * (C * H * W) + c * (H * W) + ih * W + iw
							val := input.data.data[idx]
							if val > max_val {
								max_val = val
								max_idx = idx
							}
						}
					}

					out_idx := n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
					out_data.data[out_idx] = max_val
					indices[out_idx] = max_idx
				}
			}
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{N, C, out_h, out_w},
		input.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .MaxPool2d
		append(&out.inputs, input)
		out.pool_params = PoolParams{kH, kW, stride, 0}

		// Save indices to int_metadata for backward pass
		for idx in indices {
			append(&out.int_metadata, idx)
		}
	}

	delete(indices, _op_alloc(input))
	return out
}
// tensor_avg_pool2d performs average pooling.
tensor_avg_pool2d :: proc(input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	_check_shape(input)
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	out_h := (H - kH) / stride + 1
	out_w := (W - kW) / stride + 1

	out_data := l.matrix_new(f64, 1, N * C * out_h * out_w, _op_alloc(input))
	pool_size := f64(kH * kW)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for oh in 0 ..< out_h {
				for ow in 0 ..< out_w {
					sum := 0.0
					for kh in 0 ..< kH {
						for kw in 0 ..< kW {
							ih := oh * stride + kh
							iw := ow * stride + kw
							idx := n * (C * H * W) + c * (H * W) + ih * W + iw
							sum += input.data.data[idx]
						}
					}
					out_idx := n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
					out_data.data[out_idx] = sum / pool_size
				}
			}
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{N, C, out_h, out_w},
		input.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .AvgPool2d
		append(&out.inputs, input)
		out.pool_params = PoolParams{kH, kW, stride, 0}
	}

	return out
}
// tensor_conv2d performs 2D convolution using im2col + SIMD matmul
// input: (N, C_in, H, W)
// weight: (C_out, C_in, kH, kW)
// bias: (C_out,) or nil
// output: (N, C_out, H_out, W_out)
tensor_conv2d :: proc(
	input: ^Tensor,
	weight: ^Tensor,
	bias: ^Tensor = nil,
	stride: int = 1,
	pad: int = 0,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(input)
	_check_shape(weight)
	N := input.shape[0]
	C_in := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	C_out := weight.shape[0]
	kH := weight.shape[2]
	kW := weight.shape[3]

	out_h := (H + 2 * pad - kH) / stride + 1
	out_w := (W + 2 * pad - kW) / stride + 1

	// ✅ FIX: Use context.allocator for temporary im2col
	col, _, _ := _im2col(input.data.data, N, C_in, H, W, kH, kW, stride, pad, allocator)

	out_data := l.matrix_new(f64, 1, N * C_out * out_h * out_w, _op_alloc(input))
	col_w := C_in * kH * kW
	col_h := out_h * out_w

	weight_2d := l.Matrix(f64) {
		rows = C_out,
		cols = col_w,
		data = weight.data.data,
	}

	for n in 0 ..< N {
		col_start := n * col_w * col_h
		col_2d := l.Matrix(f64) {
			rows = col_w,
			cols = col_h,
			data = col[col_start:col_start + col_w * col_h],
		}

		// Multiply straight into this sample's slice of the (zeroed) output
		out_start := n * C_out * col_h
		out_2d := _mat(C_out, col_h, out_data.data[out_start:out_start + C_out * col_h])
		l.matmul_nn_acc_simd(&weight_2d, &col_2d, &out_2d)
	}

	// ✅ FIX: Delete with context.allocator to match allocation
	delete(col, allocator)

	if bias != nil {
		for n in 0 ..< N {
			for c in 0 ..< C_out {
				b := bias.data.data[c]
				offset := n * C_out * col_h + c * col_h
				for i in 0 ..< col_h {
					out_data.data[offset + i] += b
				}
			}
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{N, C_out, out_h, out_w},
		input.requires_grad || weight.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .Conv2d
		append(&out.inputs, input)
		append(&out.inputs, weight)
		if bias != nil {append(&out.inputs, bias)}
		out.conv_params = ConvParams{kH, kW, stride, pad}
	}

	return out
}


// _matmul returns a new [a.rows, b.cols] matrix A @ B. It uses the transpose-free
// i-k-j kernel, which avoids re-transposing B (usually the weights) on every call.
_matmul :: proc(a, b: ^l.Matrix(f64), allocator: mem.Allocator) -> l.Matrix(f64) {
	out := l.matrix_new(f64, a.rows, b.cols, allocator)
	l.matmul_nn_acc_simd(a, b, &out)
	return out
}

// tensor_matmul creates a new tensor C = A @ B (Matrix Multiplication)
tensor_matmul :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {


	// ✅ NEW: Handle flattened 3D/4D tensor @ 2D weight matrix
	if a.data.rows == 1 && a.data.cols != b.data.rows {
		if a.data.cols % b.data.rows == 0 {
			N := a.data.cols / b.data.rows
			in_features := b.data.rows
			out_features := b.data.cols

			// Create a temporary view to do the SIMD matmul
			a_view := l.Matrix(f64) {
				rows = N,
				cols = in_features,
				data = a.data.data,
			}
			// Result stays flattened as 1 x (N * out_features) to match framework
			// conventions; compute it in place through an [N, out_features] view.
			out_data := l.matrix_new(f64, 1, N * out_features, _op_alloc(a))
			out_view := _mat(N, out_features, out_data.data)
			l.matmul_nn_acc_simd(&a_view, &b.data, &out_view)

			out := _tensor_new_op(
				out_data,
				_flattened_matmul_shape(a.shape, N, in_features, out_features),
				a.requires_grad || b.requires_grad,
				_op_alloc(a),
			)
			if out.requires_grad {
				out.op = .MatMul
				append(&out.inputs, a)
				append(&out.inputs, b)
				// Store metadata so the backward pass knows how to reshape gradients
				append(&out.int_metadata, N)
				append(&out.int_metadata, in_features)
				append(&out.int_metadata, out_features)
			}
			return out
		}
	}


	// Standard 2D matmul path
	if a.data.cols != b.data.rows {
		panic("tensor_matmul: dimension mismatch")
	}

	out_data := _matmul(&a.data, &b.data, _op_alloc(a))
	out := _tensor_new_op(
		out_data,
		[4]int{a.data.rows, b.data.cols, 1, 1},
		a.requires_grad || b.requires_grad,
		_op_alloc(a),
	)
	if out.requires_grad {
		out.op = .MatMul
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}
// _tensor_free_graph_impl frees the intermediate nodes below `node`, inputs
// first (post-order). Leaf nodes (op == .None) are user-managed and kept.
_tensor_free_graph_impl :: proc(node: ^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil || visited[node] {return} 	// visited prevents double-free in DAGs
	visited[node] = true
	if node.op == .None {return}
	stack := make([dynamic]_Walk_Frame, 0, 64, context.temp_allocator)
	defer delete(stack)
	append(&stack, _Walk_Frame{node, 0})
	for len(stack) > 0 {
		top := &stack[len(stack) - 1]
		if top.next < len(top.node.inputs) {
			input := top.node.inputs[top.next]
			top.next += 1
			if input != nil && !visited[input] {
				visited[input] = true
				if input.op != .None {append(&stack, _Walk_Frame{input, 0})}
			}
			continue
		}
		n := top.node
		pop(&stack)
		if n.data.data != nil {l.matrix_free(&n.data)}
		if n.grad.data != nil {l.matrix_free(&n.grad)}
		delete(n.inputs)
		delete(n.int_metadata)
		delete(n.f64_metadata)
		if n.dropout_mask != nil {delete(n.dropout_mask, n.allocator)}
		free(n, n.allocator)
	}
}

// tensor_free_graph automatically cleans up all intermediate tensors
// in the computation graph, starting from 'root'.
// It protects leaf nodes (inputs/weights) from being freed.
tensor_free_graph :: proc(root: ^Tensor, allocator: mem.Allocator = context.allocator) {
	if root == nil {return}

	visited := make(map[^Tensor]bool)
	defer delete(visited)

	nodes := make([dynamic]^Tensor, 0, allocator)
	_collect_graph_nodes(root, &nodes, &visited)
	freed_in_graph := 0
	for i := len(nodes) - 1; i >= 0; i -= 1 {
		node := nodes[i]
		// ✅ FIX: Free nodes that are either intermediate OR owned by the graph
		if node.op != .None || node.owned_by_graph {
			if node.data.data != nil {l.matrix_free(&node.data)}
			if node.grad.data != nil {l.matrix_free(&node.grad)}
			delete(node.inputs)
			delete(node.int_metadata)
			delete(node.f64_metadata)
			if node.dropout_mask != nil {delete(node.dropout_mask, node.allocator)}
			_scope_note_free(node.allocator)
			free(node, node.allocator)
			freed_in_graph += 1
			global_tensors_freed += 1
		}
	}

	delete(nodes)
}

// Pre-order depth-first collection (node before its inputs)
_collect_graph_nodes :: proc(node: ^Tensor, nodes: ^[dynamic]^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil || visited[node] {return}
	stack := make([dynamic]_Walk_Frame, 0, 64, context.temp_allocator)
	defer delete(stack)
	visited[node] = true
	append(nodes, node)
	append(&stack, _Walk_Frame{node, 0})
	for len(stack) > 0 {
		top := &stack[len(stack) - 1]
		if top.next < len(top.node.inputs) {
			input := top.node.inputs[top.next]
			top.next += 1
			if input != nil && !visited[input] {
				visited[input] = true
				append(nodes, input)
				append(&stack, _Walk_Frame{input, 0})
			}
			continue
		}
		pop(&stack)
	}
}

// _im2col unfolds image patches into columns for matrix multiplication
// Input shape: (N, C, H, W)
// Output shape: (N, C*kH*kW, H_out*W_out)
_im2col :: proc(
	input: []f64,
	N, C, H, W: int,
	kH, kW, stride, pad: int,
	allocator: mem.Allocator,
) -> (
	col: []f64,
	out_h: int,
	out_w: int,
) {
	out_h = (H + 2 * pad - kH) / stride + 1
	out_w = (W + 2 * pad - kW) / stride + 1

	col_size := N * (C * kH * kW) * (out_h * out_w)
	col = make([]f64, col_size, allocator)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for kh in 0 ..< kH {
				for kw in 0 ..< kW {
					// Which row in the column matrix is this?
					col_row := c * (kH * kW) + kh * kW + kw

					for oh in 0 ..< out_h {
						for ow in 0 ..< out_w {
							// Which column in the column matrix?
							col_col := oh * out_w + ow

							// Input coordinates
							ih := oh * stride - pad + kh
							iw := ow * stride - pad + kw

							// Handle padding (zero-fill)
							val := 0.0
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								val = input[n * (C * H * W) + c * (H * W) + ih * W + iw]
							}

							// Write to col matrix: (N, C*kH*kW, H_out*W_out)
							idx :=
								n * ((C * kH * kW) * (out_h * out_w)) +
								col_row * (out_h * out_w) +
								col_col
							col[idx] = val
						}
					}
				}
			}
		}
	}

	return col, out_h, out_w
}

// _col2im is the backward pass of im2col: accumulates gradients back to input shape
_col2im :: proc(
	col: []f64,
	N, C, H, W: int,
	kH, kW, stride, pad, out_h, out_w: int,
	allocator: mem.Allocator,
) -> []f64 {
	grad_input := make([]f64, N * C * H * W, allocator)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for kh in 0 ..< kH {
				for kw in 0 ..< kW {
					col_row := c * (kH * kW) + kh * kW + kw

					for oh in 0 ..< out_h {
						for ow in 0 ..< out_w {
							col_col := oh * out_w + ow

							ih := oh * stride - pad + kh
							iw := ow * stride - pad + kw

							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								idx_col :=
									n * ((C * kH * kW) * (out_h * out_w)) +
									col_row * (out_h * out_w) +
									col_col
								idx_in := n * (C * H * W) + c * (H * W) + ih * W + iw
								grad_input[idx_in] += col[idx_col]
							}
						}
					}
				}
			}
		}
	}

	return grad_input
}
tensor_new_4d :: proc(
	N, C, H, W: int,
	requires_grad: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	data := l.matrix_new(f64, 1, N * C * H * W, allocator)
	t := tensor_new(data, requires_grad, allocator)
	t.shape = [4]int{N, C, H, W}
	return t
}
// tensor_validate_graph checks that all tensors in the graph have valid gradients
tensor_validate_graph :: proc(root: ^Tensor) -> bool {
	if root == nil {return false}

	visited := make(map[^Tensor]bool)
	defer delete(visited)

	return _validate_graph_impl(root, &visited)
}

// Pre-order check of every node (explicit stack); stops at the first invalid one
_validate_graph_impl :: proc(node: ^Tensor, visited: ^map[^Tensor]bool) -> bool {
	if node == nil || visited[node] {return true}
	visited[node] = true
	if !_validate_node(node) {return false}
	stack := make([dynamic]_Walk_Frame, 0, 64, context.temp_allocator)
	defer delete(stack)
	append(&stack, _Walk_Frame{node, 0})
	for len(stack) > 0 {
		top := &stack[len(stack) - 1]
		if top.next < len(top.node.inputs) {
			input := top.node.inputs[top.next]
			top.next += 1
			if input != nil && !visited[input] {
				visited[input] = true
				if !_validate_node(input) {return false}
				append(&stack, _Walk_Frame{input, 0})
			}
			continue
		}
		pop(&stack)
	}
	return true
}

_validate_node :: proc(node: ^Tensor) -> bool {
	// Check that this node has valid data
	if len(node.data.data) == 0 {
		fmt.printf("ERROR: Tensor with op %v has empty data\n", node.op)
		return false
	}

	// Leaves (inputs/parameters) must own a gradient buffer. Op outputs get
	// theirs lazily in tensor_backward, so they are not checked here.
	if node.op == .None && node.requires_grad && node.grad.data == nil {
		fmt.printf("ERROR: Leaf tensor requires grad but has nil gradient\n")
		return false
	}

	return true
}

// tensor_batch_norm_2d performs Batch Normalization over a 4D tensor (N, C, H, W)
// ✅ SIMD-optimized with proper memory cleanup
tensor_batch_norm_2d :: proc(
	input: ^Tensor,
	weight: ^Tensor,
	bias: ^Tensor,
	running_mean: ^Tensor,
	running_var: ^Tensor,
	training: bool,
	momentum: f64 = 0.1,
	eps: f64 = 1e-5,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(input)
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]
	N_hw := f64(N * H * W)
	channel_size := N * H * W

	out_data := l.matrix_new(f64, 1, N * C * H * W, _op_alloc(input))

	// ✅ Allocate buffers ONCE outside the loop for reuse
	channel_buf := make([]f64, channel_size, allocator)
	mean_vec := make([]f64, channel_size, allocator)
	centered := make([]f64, channel_size, allocator)
	inv_std_vec := make([]f64, channel_size, allocator)
	normalized := make([]f64, channel_size, allocator)
	gamma_vec := make([]f64, channel_size, allocator)
	beta_vec := make([]f64, channel_size, allocator)
	scaled := make([]f64, channel_size, allocator)

	for c in 0 ..< C {
		// 1. Extract channel to contiguous buffer
		for n in 0 ..< N {
			for h in 0 ..< H {
				for w in 0 ..< W {
					src_idx := n * (C * H * W) + c * (H * W) + h * W + w
					dst_idx := n * (H * W) + h * W + w
					channel_buf[dst_idx] = input.data.data[src_idx]
				}
			}
		}

		// 2. Compute mean (SIMD sum)
		mean := l.sum_simd(channel_buf) / N_hw

		// 3. Compute variance using SIMD
		for i in 0 ..< channel_size {mean_vec[i] = mean}
		l.vec_sub_simd(channel_buf, mean_vec, centered)
		var := l.dot_simd(centered, centered) / N_hw

		// 4. Update running stats
		if training {
			running_mean.data.data[c] =
				(1.0 - momentum) * running_mean.data.data[c] + momentum * mean
			running_var.data.data[c] = (1.0 - momentum) * running_var.data.data[c] + momentum * var
		}

		current_mean := mean
		current_var := var
		if !training {
			current_mean = running_mean.data.data[c]
			current_var = running_var.data.data[c]
		}

		// 5. Normalize and scale using SIMD
		std := math.sqrt(current_var + eps)
		gamma := weight.data.data[c]
		beta := bias.data.data[c]
		inv_std := 1.0 / std

		for i in 0 ..< channel_size {inv_std_vec[i] = inv_std}
		l.vec_mul_simd(centered, inv_std_vec, normalized)

		for i in 0 ..< channel_size {
			gamma_vec[i] = gamma
			beta_vec[i] = beta
		}
		l.vec_mul_simd(normalized, gamma_vec, scaled)
		l.vec_add_simd(scaled, beta_vec, scaled)

		// 6. Write back (de-interleave)
		for n in 0 ..< N {
			for h in 0 ..< H {
				for w in 0 ..< W {
					src_idx := n * (H * W) + h * W + w
					dst_idx := n * (C * H * W) + c * (H * W) + h * W + w
					out_data.data[dst_idx] = scaled[src_idx]
				}
			}
		}
	}

	// ✅ CRITICAL: Free all temporary buffers
	delete(channel_buf, allocator)
	delete(mean_vec, allocator)
	delete(centered, allocator)
	delete(inv_std_vec, allocator)
	delete(normalized, allocator)
	delete(gamma_vec, allocator)
	delete(beta_vec, allocator)
	delete(scaled, allocator)

	out := _tensor_new_op(
		out_data,
		input.shape,
		input.requires_grad || weight.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .BatchNorm2d
		append(&out.inputs, input)
		append(&out.inputs, weight)
		append(&out.inputs, bias)
	}

	return out
}
// ============================================================================
// Recurrent cells (RNN / GRU / LSTM)
// ============================================================================
//
// The input projection x_t @ W_ih does not depend on the recurrent state, so it
// is computed for all timesteps at once (one large, threaded matmul) before the
// time loop. Inside the loop only h_prev @ W_hh remains. Every row of the
// projection is computed exactly as a per-step product would be, so the
// results are bit-identical to stepping through x_t @ W_ih one step at a time.
//
// When the output needs gradients, the forward pass keeps every state and gate
// in out.f64_metadata (layouts below), so the backward pass does not recompute
// them. Without gradients only two state rows are kept.

// _gather_steps lays a [batch, seq_len, width] sequence out step-major:
// block j (batch rows) holds timestep j, or timestep seq_len-1-j if reverse.
_gather_steps :: proc(dst, src: []f64, batch, seq_len, width: int, reverse: bool) {
	for j in 0 ..< seq_len {
		s := seq_len - 1 - j if reverse else j
		for b in 0 ..< batch {
			off := (b * seq_len + s) * width
			dst_off := (j * batch + b) * width
			copy(dst[dst_off:dst_off + width], src[off:off + width])
		}
	}
}

// _input_projection returns xs @ W_ih for step-major xs [seq_len * batch, in_size]
_input_projection :: proc(
	xs: []f64,
	rows, in_size: int,
	w_ih: []f64,
	G: int,
	allocator: mem.Allocator,
) -> []f64 {
	proj := make([]f64, rows * G, allocator)
	xs_mat := l.Matrix(f64) {
		rows = rows,
		cols = in_size,
		data = xs,
	}
	w_mat := l.Matrix(f64) {
		rows = in_size,
		cols = G,
		data = w_ih,
	}
	proj_mat := l.Matrix(f64) {
		rows = rows,
		cols = G,
		data = proj,
	}
	l.matmul_nn_acc_simd(&xs_mat, &w_mat, &proj_mat)
	return proj
}

// _step_projection gathers x into step-major order and projects it with W_ih
_step_projection :: proc(x, w_ih: ^Tensor, G: int, reverse: bool, allocator: mem.Allocator) -> []f64 {
	batch, seq_len, in_size := x.shape[0], x.shape[1], x.shape[2]
	xs := make([]f64, seq_len * batch * in_size, allocator)
	defer delete(xs, allocator)
	_gather_steps(xs, x.data.data, batch, seq_len, in_size, reverse)
	return _input_projection(xs, seq_len * batch, in_size, w_ih.data.data, G, allocator)
}

// _hidden_projection: gate_hh = h_prev [batch, H] @ W_hh [H, G] (overwrites gate_hh)
_hidden_projection :: proc(h_prev, w_hh, gate_hh: []f64, batch, H, G: int) {
	mem.zero_slice(gate_hh)
	h_mat := l.Matrix(f64) {
		rows = batch,
		cols = H,
		data = h_prev,
	}
	w_mat := l.Matrix(f64) {
		rows = H,
		cols = G,
		data = w_hh,
	}
	out_mat := l.Matrix(f64) {
		rows = batch,
		cols = G,
		data = gate_hh,
	}
	l.matmul_nn_acc_simd(&h_mat, &w_mat, &out_mat)
}

// _scatter_step writes src [batch, width] to timestep s of a [batch, seq_len, width] sequence
_scatter_step :: proc(dst, src: []f64, batch, seq_len, width, s: int) {
	for b in 0 ..< batch {
		off := (b * seq_len + s) * width
		copy(dst[off:off + width], src[b * width:(b + 1) * width])
	}
}

// _Recurrent_State: state rows of one cell. With gradients, rows[s] holds the
// state entering step s for s in [0, seq_len] (a view into the saved buffer);
// without, two rows are reused alternately.
_Recurrent_State :: struct {
	rows: []f64,
	BH:   int,
	all:  bool,
}

_state_row :: proc(st: _Recurrent_State, s: int) -> []f64 {
	slot := s if st.all else s % 2
	return st.rows[slot * st.BH:(slot + 1) * st.BH]
}

// _step_block returns block s of a buffer of seq_len blocks (nil if not kept)
_step_block :: proc(buf: []f64, s, size: int) -> []f64 {
	return buf[s * size:(s + 1) * size] if buf != nil else nil
}

// _keep_saved hands the saved buffer to the output node (or frees it)
_keep_saved :: proc(out: ^Tensor, saved: [dynamic]f64) {
	if out.requires_grad {
		delete(out.f64_metadata)
		out.f64_metadata = saved
	} else {
		delete(saved)
	}
}

// _rnn_step_forward: h_t = tanh(h_ih + h_prev @ W_hh + b), where h_ih = x_t @ W_ih
// is precomputed. h_hh is scratch [batch, H].
_rnn_step_forward :: proc(
	h_ih: []f64,
	h_prev: []f64,
	w_hh: []f64,
	bias: []f64,
	h_t: []f64,
	h_hh: []f64,
	batch: int,
	hidden_size: int,
) {
	_hidden_projection(h_prev, w_hh, h_hh, batch, hidden_size, hidden_size)

	for i in 0 ..< batch * hidden_size {
		h_t[i] = h_ih[i] + h_hh[i] + bias[i % hidden_size]
	}
	for i in 0 ..< batch * hidden_size {
		h_t[i] = math.tanh(h_t[i])
	}
}

// tensor_rnn processes a full sequence. Saved for backward (f64_metadata):
// hs [(seq_len + 1) * batch * H], hs[s] = state entering step s.
tensor_rnn :: proc(
	x: ^Tensor,
	h_0: ^Tensor,
	w_ih: ^Tensor,
	w_hh: ^Tensor,
	bias: ^Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(x)
	_check_shape(w_ih)
	batch := x.shape[0]
	seq_len := x.shape[1]
	hidden_size := w_ih.shape[1]
	BH := batch * hidden_size
	rg :=
		x.requires_grad ||
		h_0.requires_grad ||
		w_ih.requires_grad ||
		w_hh.requires_grad ||
		bias.requires_grad

	out_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, _op_alloc(x))

	proj := _step_projection(x, w_ih, hidden_size, false, allocator)
	h_hh := make([]f64, BH, allocator)
	defer {
		delete(proj, allocator)
		delete(h_hh, allocator)
	}

	saved := make([dynamic]f64, ((seq_len + 1) if rg else 2) * BH, _op_alloc(x))
	hs := _Recurrent_State{saved[:], BH, rg}
	copy(_state_row(hs, 0), h_0.data.data)

	for s in 0 ..< seq_len {
		h_next := _state_row(hs, s + 1)
		_rnn_step_forward(
			proj[s * BH:(s + 1) * BH],
			_state_row(hs, s),
			w_hh.data.data,
			bias.data.data,
			h_next,
			h_hh,
			batch,
			hidden_size,
		)
		_scatter_step(out_data.data, h_next, batch, seq_len, hidden_size, s)
	}

	out := _tensor_new_op(out_data, [4]int{batch, seq_len, hidden_size, 1}, rg, _op_alloc(x))
	_keep_saved(out, saved)

	if out.requires_grad {
		out.op = .RNN
		append(&out.inputs, x)
		append(&out.inputs, h_0)
		append(&out.inputs, w_ih)
		append(&out.inputs, w_hh)
		append(&out.inputs, bias)
	}

	return out
}
// ============================================================================
// GRU Operations
// ============================================================================

// _gru_step_forward: gates [r, z, n]. gate_ih = x_t @ W_ih is precomputed and
// gets the bias added in place; gate_hh [batch, 3H] is scratch.
// The r / z / n / hh_n buffers can be nil if only h_t is needed
// (hh_n_buf receives the n part of h_prev @ W_hh).
_gru_step_forward :: proc(
	gate_ih: []f64,
	h_prev: []f64,
	w_hh: []f64,
	bias: []f64,
	h_t: []f64,
	gate_hh: []f64,
	r_buf: []f64,
	z_buf: []f64,
	n_buf: []f64,
	hh_n_buf: []f64,
	batch: int,
	hidden_size: int,
) {
	H := hidden_size
	H3 := 3 * H

	_hidden_projection(h_prev, w_hh, gate_hh, batch, H, H3)

	// Add bias
	for b in 0 ..< batch {
		for i in 0 ..< H3 {
			gate_ih[b * H3 + i] += bias[i]
		}
	}

	for b in 0 ..< batch {
		for i in 0 ..< H {
			idx_r := b * H3 + i
			idx_z := b * H3 + H + i
			idx_n := b * H3 + 2 * H + i

			r_pre := gate_ih[idx_r] + gate_hh[idx_r]
			z_pre := gate_ih[idx_z] + gate_hh[idx_z]

			r_val := 1.0 / (1.0 + math.exp(-r_pre))
			z_val := 1.0 / (1.0 + math.exp(-z_pre))

			if r_buf != nil {r_buf[b * H + i] = r_val}
			if z_buf != nil {z_buf[b * H + i] = z_val}
			if hh_n_buf != nil {hh_n_buf[b * H + i] = gate_hh[idx_n]}

			n_pre := gate_ih[idx_n] + r_val * gate_hh[idx_n]
			n_val := math.tanh(n_pre)
			if n_buf != nil {n_buf[b * H + i] = n_val}

			// h_t = (1 - z) * n + z * h_prev
			h_t[b * H + i] = (1.0 - z_val) * n_val + z_val * h_prev[b * H + i]
		}
	}
}

// tensor_gru: saved for backward (f64_metadata), each block [seq_len * batch * H]
// unless noted: hs [(seq_len + 1) * batch * H], r, z, n, hh_n (n part of h_prev @ W_hh)
tensor_gru :: proc(
	x: ^Tensor,
	h_0: ^Tensor,
	w_ih: ^Tensor,
	w_hh: ^Tensor,
	bias: ^Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(x)
	_check_shape(w_ih)
	batch := x.shape[0]
	seq_len := x.shape[1]
	hidden_size := w_ih.shape[1] / 3
	BH := batch * hidden_size
	BG := batch * 3 * hidden_size
	rg :=
		x.requires_grad ||
		h_0.requires_grad ||
		w_ih.requires_grad ||
		w_hh.requires_grad ||
		bias.requires_grad

	out_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, _op_alloc(x))

	proj := _step_projection(x, w_ih, 3 * hidden_size, false, allocator)
	gate_hh := make([]f64, BG, allocator)
	defer {
		delete(proj, allocator)
		delete(gate_hh, allocator)
	}

	n_states := (seq_len + 1) if rg else 2
	SBH := seq_len * BH
	saved := make([dynamic]f64, n_states * BH + (4 * SBH if rg else 0), _op_alloc(x))
	hs := _Recurrent_State{saved[0:n_states * BH], BH, rg}
	rs, zs, ns, hh_n: []f64
	if rg {
		g := saved[n_states * BH:]
		rs, zs, ns, hh_n = g[0:SBH], g[SBH:2 * SBH], g[2 * SBH:3 * SBH], g[3 * SBH:4 * SBH]
	}
	copy(_state_row(hs, 0), h_0.data.data)

	for s in 0 ..< seq_len {
		h_next := _state_row(hs, s + 1)
		_gru_step_forward(
			proj[s * BG:(s + 1) * BG],
			_state_row(hs, s),
			w_hh.data.data,
			bias.data.data,
			h_next,
			gate_hh,
			_step_block(rs, s, BH),
			_step_block(zs, s, BH),
			_step_block(ns, s, BH),
			_step_block(hh_n, s, BH),
			batch,
			hidden_size,
		)
		_scatter_step(out_data.data, h_next, batch, seq_len, hidden_size, s)
	}

	out := _tensor_new_op(out_data, [4]int{batch, seq_len, hidden_size, 1}, rg, _op_alloc(x))
	_keep_saved(out, saved)

	if out.requires_grad {
		out.op = .GRU
		append(&out.inputs, x)
		append(&out.inputs, h_0)
		append(&out.inputs, w_ih)
		append(&out.inputs, w_hh)
		append(&out.inputs, bias)
	}

	return out
}
// ============================================================================
// LSTM Operations (SIMD-optimized)
// ============================================================================

// _lstm_step_forward: gates [i, f, g, o]. gate_ih = x_t @ W_ih is precomputed
// and gets the bias added in place; gate_hh [batch, 4H] is scratch.
// The gate buffers and tc_buf (tanh(c_t)) can be nil if only h_t / c_t are needed.
_lstm_step_forward :: proc(
	gate_ih: []f64,
	h_prev: []f64,
	c_prev: []f64,
	w_hh: []f64,
	bias: []f64,
	h_t: []f64,
	c_t: []f64,
	gate_hh: []f64,
	i_buf: []f64,
	f_buf: []f64,
	g_buf: []f64,
	o_buf: []f64,
	tc_buf: []f64,
	batch: int,
	hidden_size: int,
) {
	H := hidden_size
	H4 := 4 * H

	_hidden_projection(h_prev, w_hh, gate_hh, batch, H, H4)

	// Add bias
	for b in 0 ..< batch {
		for i in 0 ..< H4 {
			gate_ih[b * H4 + i] += bias[i]
		}
	}

	for b in 0 ..< batch {
		for i in 0 ..< H {
			idx_i := b * H4 + i
			idx_f := b * H4 + H + i
			idx_g := b * H4 + 2 * H + i
			idx_o := b * H4 + 3 * H + i

			i_pre := gate_ih[idx_i] + gate_hh[idx_i]
			f_pre := gate_ih[idx_f] + gate_hh[idx_f]
			g_pre := gate_ih[idx_g] + gate_hh[idx_g]
			o_pre := gate_ih[idx_o] + gate_hh[idx_o]

			i_val := 1.0 / (1.0 + math.exp(-i_pre))
			f_val := 1.0 / (1.0 + math.exp(-f_pre))
			g_val := math.tanh(g_pre)
			o_val := 1.0 / (1.0 + math.exp(-o_pre))

			if i_buf != nil {i_buf[b * H + i] = i_val}
			if f_buf != nil {f_buf[b * H + i] = f_val}
			if g_buf != nil {g_buf[b * H + i] = g_val}
			if o_buf != nil {o_buf[b * H + i] = o_val}

			// c_t = f * c_prev + i * g
			c_t[b * H + i] = f_val * c_prev[b * H + i] + i_val * g_val

			// h_t = o * tanh(c_t)
			tc := math.tanh(c_t[b * H + i])
			if tc_buf != nil {tc_buf[b * H + i] = tc}
			h_t[b * H + i] = o_val * tc
		}
	}
}

// tensor_lstm: saved for backward (f64_metadata), each block [seq_len * batch * H]
// unless noted: hs, cs [(seq_len + 1) * batch * H], i, f, g, o, tanh(c)
tensor_lstm :: proc(
	x: ^Tensor,
	h_0: ^Tensor,
	c_0: ^Tensor,
	w_ih: ^Tensor,
	w_hh: ^Tensor,
	bias: ^Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(x)
	_check_shape(w_ih)
	batch := x.shape[0]
	seq_len := x.shape[1]
	hidden_size := w_ih.shape[1] / 4
	BH := batch * hidden_size
	BG := batch * 4 * hidden_size
	rg :=
		x.requires_grad ||
		h_0.requires_grad ||
		c_0.requires_grad ||
		w_ih.requires_grad ||
		w_hh.requires_grad ||
		bias.requires_grad

	out_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, _op_alloc(x))

	proj := _step_projection(x, w_ih, 4 * hidden_size, false, allocator)
	gate_hh := make([]f64, BG, allocator)
	defer {
		delete(proj, allocator)
		delete(gate_hh, allocator)
	}

	n_states := (seq_len + 1) if rg else 2
	SBH := seq_len * BH
	saved := make([dynamic]f64, 2 * n_states * BH + (5 * SBH if rg else 0), _op_alloc(x))
	hs := _Recurrent_State{saved[0:n_states * BH], BH, rg}
	cs := _Recurrent_State{saved[n_states * BH:2 * n_states * BH], BH, rg}
	ig, fg, gg, og, tc: []f64
	if rg {
		g := saved[2 * n_states * BH:]
		ig, fg, gg = g[0:SBH], g[SBH:2 * SBH], g[2 * SBH:3 * SBH]
		og, tc = g[3 * SBH:4 * SBH], g[4 * SBH:5 * SBH]
	}
	copy(_state_row(hs, 0), h_0.data.data)
	copy(_state_row(cs, 0), c_0.data.data)

	for s in 0 ..< seq_len {
		h_next := _state_row(hs, s + 1)
		_lstm_step_forward(
			proj[s * BG:(s + 1) * BG],
			_state_row(hs, s),
			_state_row(cs, s),
			w_hh.data.data,
			bias.data.data,
			h_next,
			_state_row(cs, s + 1),
			gate_hh,
			_step_block(ig, s, BH),
			_step_block(fg, s, BH),
			_step_block(gg, s, BH),
			_step_block(og, s, BH),
			_step_block(tc, s, BH),
			batch,
			hidden_size,
		)
		_scatter_step(out_data.data, h_next, batch, seq_len, hidden_size, s)
	}

	out := _tensor_new_op(out_data, [4]int{batch, seq_len, hidden_size, 1}, rg, _op_alloc(x))
	_keep_saved(out, saved)

	if out.requires_grad {
		out.op = .LSTM
		append(&out.inputs, x)
		append(&out.inputs, h_0)
		append(&out.inputs, c_0)
		append(&out.inputs, w_ih)
		append(&out.inputs, w_hh)
		append(&out.inputs, bias)
	}

	return out
}
// ============================================================================
// Embedding Operations (SIMD-optimized lookup and scatter-add)
// ============================================================================

tensor_embedding :: proc(input: ^Tensor, weight: ^Tensor) -> ^Tensor {
	_check_shape(input)
	_check_shape(weight)

	batch := input.shape[0]
	seq_len := input.shape[1]
	vocab_size := weight.shape[0]
	embed_dim := weight.shape[1]

	out_data := l.matrix_new(f64, 1, batch * seq_len * embed_dim, _op_alloc(input))

	for b in 0 ..< batch {
		for s in 0 ..< seq_len {
			idx := int(input.data.data[b * seq_len + s])

			// Safety bounds check
			if idx < 0 || idx >= vocab_size {
				panic(fmt.aprintf("Embedding index %d out of bounds [0, %d)", idx, vocab_size))
			}

			src_offset := idx * embed_dim
			dst_offset := (b * seq_len + s) * embed_dim

			// ✅ OPTIMIZATION: Odin's `copy` is highly optimized and uses SIMD/memcpy
			copy(
				out_data.data[dst_offset:dst_offset + embed_dim],
				weight.data.data[src_offset:src_offset + embed_dim],
			)
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{batch, seq_len, embed_dim, 1},
		weight.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .Embedding
		append(&out.inputs, input)
		append(&out.inputs, weight)
	}

	return out
}
// ============================================================================
// Scaled Dot-Product Attention (with Checkpointing for memory efficiency)
// ============================================================================

tensor_scaled_dot_product_attention :: proc(
	Q: ^Tensor,
	K: ^Tensor,
	V: ^Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(Q)
	_check_shape(K)
	_check_shape(V)
	batch := Q.shape[0]
	seq_q := Q.shape[1]
	seq_k := K.shape[1]
	d_k := Q.shape[2]
	d_v := V.shape[2]

	out_data := l.matrix_new(f64, 1, batch * seq_q * d_v, _op_alloc(Q))
	scale := 1.0 / math.sqrt(f64(d_k))

	for b in 0 ..< batch {
		q_b := l.Matrix(f64) {
			rows = seq_q,
			cols = d_k,
			data = Q.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k],
		}
		k_b := l.Matrix(f64) {
			rows = seq_k,
			cols = d_k,
			data = K.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k],
		}
		v_b := l.Matrix(f64) {
			rows = seq_k,
			cols = d_v,
			data = V.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v],
		}

		// 1. S_b = Q_b @ K_b^T (row dot products, no transpose)
		s_b := l.matrix_new(f64, seq_q, seq_k, allocator)
		l.matmul_nt_acc_simd(&q_b, &k_b, &s_b)

		// 2. Scale and Softmax to get P_b
		p_b := l.Matrix(f64) {
			rows = seq_q,
			cols = seq_k,
			data = s_b.data,
		}
		for i in 0 ..< seq_q * seq_k {
			p_b.data[i] *= scale
		}

		// Numerically stable softmax along seq_k (rows of p_b)
		for i in 0 ..< seq_q {
			row_start := i * seq_k
			max_val := p_b.data[row_start]
			for j in 1 ..< seq_k {
				if p_b.data[row_start + j] > max_val {
					max_val = p_b.data[row_start + j]
				}
			}

			sum_exp := 0.0
			for j in 0 ..< seq_k {
				p_b.data[row_start + j] = math.exp(p_b.data[row_start + j] - max_val)
				sum_exp += p_b.data[row_start + j]
			}

			inv_sum := 1.0 / sum_exp
			for j in 0 ..< seq_k {
				p_b.data[row_start + j] *= inv_sum
			}
		}

		// 3. O_b = P_b @ V_b
		o_b := _matmul(&p_b, &v_b, allocator)
		copy(out_data.data[b * seq_q * d_v:(b + 1) * seq_q * d_v], o_b.data)

		l.matrix_free(&o_b)
		l.matrix_free(&s_b)
	}

	out := _tensor_new_op(
		out_data,
		[4]int{batch, seq_q, d_v, 1},
		Q.requires_grad || K.requires_grad || V.requires_grad,
		_op_alloc(Q),
	)

	if out.requires_grad {
		out.op = .ScaledDotProductAttention
		append(&out.inputs, Q)
		append(&out.inputs, K)
		append(&out.inputs, V)
	}

	return out
}
// ============================================================================
// MHA Permute Helpers (to group heads for batched attention)
// ============================================================================

tensor_permute_mha :: proc(
	x: ^Tensor,
	batch: int,
	seq_len: int,
	num_heads: int,
	head_dim: int,
) -> ^Tensor {
	d_model := num_heads * head_dim
	out_data := l.matrix_new(f64, 1, batch * num_heads * seq_len * head_dim, _op_alloc(x))

	for b in 0 ..< batch {
		for s in 0 ..< seq_len {
			for h in 0 ..< num_heads {
				for d in 0 ..< head_dim {
					src := b * (seq_len * d_model) + s * d_model + h * head_dim + d
					dst := (b * num_heads + h) * (seq_len * head_dim) + s * head_dim + d
					out_data.data[dst] = x.data.data[src]
				}
			}
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{batch * num_heads, seq_len, head_dim, 1},
		x.requires_grad,
		_op_alloc(x),
	)

	if out.requires_grad {
		out.op = .PermuteMHA
		append(&out.inputs, x)
		// Store metadata for the backward pass
		append(&out.int_metadata, batch)
		append(&out.int_metadata, seq_len)
		append(&out.int_metadata, num_heads)
		append(&out.int_metadata, head_dim)
	}

	return out
}

tensor_permute_mha_inverse :: proc(
	x: ^Tensor,
	batch: int,
	seq_len: int,
	num_heads: int,
	head_dim: int,
) -> ^Tensor {
	d_model := num_heads * head_dim
	out_data := l.matrix_new(f64, 1, batch * seq_len * d_model, _op_alloc(x))

	for b in 0 ..< batch {
		for s in 0 ..< seq_len {
			for h in 0 ..< num_heads {
				for d in 0 ..< head_dim {
					src := (b * num_heads + h) * (seq_len * head_dim) + s * head_dim + d
					dst := b * (seq_len * d_model) + s * d_model + h * head_dim + d
					out_data.data[dst] = x.data.data[src]
				}
			}
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{batch, seq_len, d_model, 1},
		x.requires_grad,
		_op_alloc(x),
	)

	if out.requires_grad {
		out.op = .PermuteMHAInverse
		append(&out.inputs, x)
		append(&out.int_metadata, batch)
		append(&out.int_metadata, seq_len)
		append(&out.int_metadata, num_heads)
		append(&out.int_metadata, head_dim)
	}

	return out
}
tensor_layer_norm :: proc(
	input: ^Tensor,
	gamma: ^Tensor, // [1, d_model]
	beta: ^Tensor, // [1, d_model]
	eps: f64 = 1e-5,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	d_model := gamma.data.cols
	if beta.data.cols != d_model {
		panic("tensor_layer_norm: beta dimension mismatch")
	}

	// ✅ CRITICAL CHECK: Ensure gamma and beta actually have at least d_model elements
	if len(gamma.data.data) < d_model {
		panic(
			fmt.aprintf(
				"tensor_layer_norm: gamma data length %d is less than d_model %d",
				len(gamma.data.data),
				d_model,
			),
		)
	}
	if len(beta.data.data) < d_model {
		panic(
			fmt.aprintf(
				"tensor_layer_norm: beta data length %d is less than d_model %d",
				len(beta.data.data),
				d_model,
			),
		)
	}

	// ✅ FIX: Detect flattened 3D tensor (rows=1, cols is a multiple of d_model)
	N := input.data.rows
	if N == 1 && input.data.cols != d_model {
		if input.data.cols % d_model != 0 {
			panic("tensor_layer_norm: input size not divisible by d_model")
		}
		N = input.data.cols / d_model
	} else if input.data.cols != d_model && N != 1 {
		// Standard 2D case: rows=N, cols=d_model
		if input.data.cols != d_model {
			panic("tensor_layer_norm: input column mismatch")
		}
	}

	out_data := l.matrix_new(f64, input.data.rows, input.data.cols, _op_alloc(input))

	// Temporary buffers for SIMD operations (reused across rows)
	centered := make([]f64, d_model, allocator)
	x_hat := make([]f64, d_model, allocator)
	inv_std_vec := make([]f64, d_model, allocator)
	defer {
		delete(centered, allocator)
		delete(x_hat, allocator)
		delete(inv_std_vec, allocator)
	}

	for i in 0 ..< N {
		row_start := i * d_model
		row := input.data.data[row_start:row_start + d_model]

		// 1. Compute mean (SIMD sum)
		mean := l.sum_simd(row) / f64(d_model)

		// 2. Compute variance: var = dot(centered, centered) / d_model
		for j in 0 ..< d_model {
			centered[j] = row[j] - mean
		}
		var := l.dot_simd(centered, centered) / f64(d_model)
		std := math.sqrt(var + eps)
		inv_std := 1.0 / std

		// 3. Normalize: x_hat = centered / std (SIMD)
		for j in 0 ..< d_model {inv_std_vec[j] = inv_std}
		l.vec_mul_simd(centered, inv_std_vec, x_hat)

		// 4. Scale and shift: out = gamma * x_hat + beta (SIMD)
		out_row := out_data.data[row_start:row_start + d_model]

		// ✅ BULLETPROOF: Explicitly slice to d_model to prevent vec_mul_simd length mismatch
		gamma_row := gamma.data.data[0:d_model]
		beta_row := beta.data.data[0:d_model]

		l.vec_mul_simd(x_hat, gamma_row, out_row)
		l.vec_add_simd(out_row, beta_row, out_row)
	}

	out := _tensor_new_op(
		out_data,
		input.shape,
		input.requires_grad || gamma.requires_grad || beta.requires_grad,
		_op_alloc(input),
	)

	if out.requires_grad {
		out.op = .LayerNorm
		append(&out.inputs, input)
		append(&out.inputs, gamma)
		append(&out.inputs, beta)
		// ✅ Store metadata for backward pass
		append(&out.int_metadata, N)
		append(&out.int_metadata, d_model)
	}

	return out
}
// ============================================================================
// Masked Scaled Dot-Product Attention (for Decoder self-attention)
// ============================================================================

tensor_masked_scaled_dot_product_attention :: proc(
	Q: ^Tensor,
	K: ^Tensor,
	V: ^Tensor,
	mask: []f64,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(Q)
	_check_shape(K)
	_check_shape(V)
	batch := Q.shape[0]
	seq_q := Q.shape[1]
	seq_k := K.shape[1]
	d_k := Q.shape[2]
	d_v := V.shape[2]

	out_data := l.matrix_new(f64, 1, batch * seq_q * d_v, _op_alloc(Q))
	scale := 1.0 / math.sqrt(f64(d_k))

	for b in 0 ..< batch {
		q_b := l.Matrix(f64) {
			rows = seq_q,
			cols = d_k,
			data = Q.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k],
		}
		k_b := l.Matrix(f64) {
			rows = seq_k,
			cols = d_k,
			data = K.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k],
		}
		v_b := l.Matrix(f64) {
			rows = seq_k,
			cols = d_v,
			data = V.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v],
		}

		// 1. S_b = Q_b @ K_b^T (row dot products, no transpose)
		s_b := l.matrix_new(f64, seq_q, seq_k, allocator)
		l.matmul_nt_acc_simd(&q_b, &k_b, &s_b)

		// 2. Scale and apply mask
		p_b := l.Matrix(f64) {
			rows = seq_q,
			cols = seq_k,
			data = s_b.data,
		}
		for i in 0 ..< seq_q * seq_k {
			p_b.data[i] *= scale
		}

		// ✅ Apply causal mask: add -inf to future positions
		for i in 0 ..< seq_q {
			for j in 0 ..< seq_k {
				if mask[i * seq_k + j] != 0.0 {
					p_b.data[i * seq_k + j] += mask[i * seq_k + j]
				}
			}
		}

		// 3. Numerically stable softmax
		for i in 0 ..< seq_q {
			row_start := i * seq_k
			max_val := p_b.data[row_start]
			for j in 1 ..< seq_k {
				if p_b.data[row_start + j] > max_val {
					max_val = p_b.data[row_start + j]
				}
			}

			sum_exp := 0.0
			for j in 0 ..< seq_k {
				p_b.data[row_start + j] = math.exp(p_b.data[row_start + j] - max_val)
				sum_exp += p_b.data[row_start + j]
			}

			inv_sum := 1.0 / sum_exp
			for j in 0 ..< seq_k {
				p_b.data[row_start + j] *= inv_sum
			}
		}

		// 4. O_b = P_b @ V_b
		o_b := _matmul(&p_b, &v_b, allocator)
		copy(out_data.data[b * seq_q * d_v:(b + 1) * seq_q * d_v], o_b.data)

		l.matrix_free(&o_b)
		l.matrix_free(&s_b)
	}

	out := _tensor_new_op(
		out_data,
		[4]int{batch, seq_q, d_v, 1},
		Q.requires_grad || K.requires_grad || V.requires_grad,
		_op_alloc(Q),
	)

	if out.requires_grad {
		out.op = .MaskedScaledDotProductAttention
		append(&out.inputs, Q)
		append(&out.inputs, K)
		append(&out.inputs, V)
		// Store the exact additive mask for the backward pass
		append(&out.f64_metadata, ..mask)
	}

	return out
}
// In tensor.odin, add this helper function at the top:
tensor_validate :: proc(t: ^Tensor, ctx: string) {
	if t == nil {
		fmt.printf("ERROR: %s - tensor is nil\n", ctx)
		return
	}
	if t.data.data == nil || len(t.data.data) == 0 {
		fmt.printf(
			"ERROR: %s - tensor has empty data (rows=%d, cols=%d)\n",
			ctx,
			t.data.rows,
			t.data.cols,
		)
		return
	}
	if t.requires_grad && (t.grad.data == nil || len(t.grad.data) == 0) {
		fmt.printf("ERROR: %s - tensor requires grad but has empty gradient\n", ctx)
		return
	}
}
tensor_binary_cross_entropy :: proc(pred: ^Tensor, target: ^Tensor) -> ^Tensor {
	if pred.data.rows != target.data.rows || pred.data.cols != target.data.cols {
		panic("tensor_binary_cross_entropy: shape mismatch")
	}

	eps := 1e-7
	n := f64(len(pred.data.data))

	loss := 0.0
	for i in 0 ..< len(pred.data.data) {
		p := math.max(math.min(pred.data.data[i], 1.0 - eps), eps)
		t := target.data.data[i]
		loss += -(t * math.ln(p) + (1.0 - t) * math.ln(1.0 - p))
	}
	loss /= n

	out_data := l.matrix_new(f64, 1, 1, _op_alloc(pred))
	out_data.data[0] = loss

	out := _tensor_new_op(
		out_data,
		SCALAR_SHAPE,
		pred.requires_grad || target.requires_grad,
		_op_alloc(pred),
	)
	if out.requires_grad {
		out.op = .BinaryCrossEntropy
		append(&out.inputs, pred)
		append(&out.inputs, target)
	}
	return out
}

// Add detach operation to break gradient chain:
tensor_detach :: proc(t: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, t.data.rows, t.data.cols, t.allocator)
	copy(out_data.data, t.data.data)
	out := tensor_new(out_data, false, t.allocator)
	out.shape = t.shape
	return out
}


// Add this function to tensor.odin
tensor_clip_weights :: proc(t: ^Tensor, min_val: f64, max_val: f64) {
	if t.data.data == nil {
		return
	}
	for i in 0 ..< len(t.data.data) {
		if t.data.data[i] < min_val {
			t.data.data[i] = min_val
		} else if t.data.data[i] > max_val {
			t.data.data[i] = max_val
		}
	}
}
// tensor_sub computes element-wise subtraction: out = a - b
tensor_sub :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		panic("tensor_sub: dimension mismatch")
	}

	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_sub_simd(a.data.data, b.data.data, out_data.data)

	out := _tensor_new_op(out_data, a.shape, a.requires_grad || b.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Sub
		append(&out.inputs, a)
		append(&out.inputs, b)
	}
	return out
}

// tensor_mean computes the mean of all elements, returning a scalar tensor
tensor_mean :: proc(a: ^Tensor) -> ^Tensor {
	n := f64(len(a.data.data))
	sum := l.sum_simd(a.data.data)
	mean_val := sum / n

	out_data := l.matrix_new(f64, 1, 1, _op_alloc(a))
	out_data.data[0] = mean_val

	out := _tensor_new_op(out_data, SCALAR_SHAPE, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Mean
		append(&out.inputs, a)
	}
	return out
}

// tensor_neg computes element-wise negation: out = -a

tensor_kl_divergence :: proc(mu: ^Tensor, log_var: ^Tensor) -> ^Tensor {
	// KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
	// Returns scalar loss

	n := f64(len(mu.data.data))
	kl_sum := 0.0

	for i in 0 ..< len(mu.data.data) {
		mu_val := mu.data.data[i]
		log_var_val := log_var.data.data[i]

		exp_val := math.exp(log_var_val)
		if exp_val > 1e10 {
			exp_val = 1e10
		}

		kl_sum += 1.0 + log_var_val - mu_val * mu_val - exp_val
	}

	kl_loss := -0.5 * kl_sum / n

	out_data := l.matrix_new(f64, 1, 1, _op_alloc(mu))
	out_data.data[0] = kl_loss

	out := _tensor_new_op(
		out_data,
		SCALAR_SHAPE,
		mu.requires_grad || log_var.requires_grad,
		_op_alloc(mu),
	)
	if out.requires_grad {
		out.op = .KLDivergence
		append(&out.inputs, mu)
		append(&out.inputs, log_var)
	}
	return out
}

tensor_sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_sigmoid_simd(a.data.data, out_data.data) // ✅ SIMD
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Sigmoid
		append(&out.inputs, a)
	}
	return out
}

tensor_tanh :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_tanh_simd(a.data.data, out_data.data) // ✅ SIMD
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Tanh
		append(&out.inputs, a)
	}
	return out
}

tensor_leaky_relu :: proc(a: ^Tensor, alpha: f64 = 0.01) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_leaky_relu_simd(a.data.data, alpha, out_data.data) // ✅ SIMD
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .LeakyReLU
		append(&out.inputs, a)
		append(&out.f64_metadata, alpha)
	}
	return out
}

tensor_neg :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_neg_simd(a.data.data, out_data.data) // ✅ SIMD
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Neg
		append(&out.inputs, a)
	}
	return out
}

tensor_scale :: proc(a: ^Tensor, scalar: f64) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	l.vec_scale_simd(a.data.data, scalar, out_data.data) // ✅ SIMD
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Scale
		append(&out.inputs, a)
		append(&out.f64_metadata, scalar)
	}
	return out
}
// ============================================================================
// Element-wise Exponential: out = exp(a)
// Backward: grad_a = grad_out * exp(a) = grad_out * out
// ============================================================================
tensor_exp :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = math.exp(a.data.data[i])
	}
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Exp
		append(&out.inputs, a)
	}
	return out
}

// ============================================================================
// Element-wise Square Root: out = sqrt(a)
// Backward: grad_a = grad_out / (2 * sqrt(a)) = grad_out / (2 * out)
// ============================================================================
tensor_sqrt :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = math.sqrt(a.data.data[i])
	}
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Sqrt
		append(&out.inputs, a)
	}
	return out
}

// ============================================================================
// Element-wise Natural Log: out = log(a)
// Backward: grad_a = grad_out / a
// ============================================================================
tensor_log :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = math.ln_f64(a.data.data[i])
	}
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Log
		append(&out.inputs, a)
	}
	return out
}

// ============================================================================
// Element-wise Division: out = a / b
// Backward: grad_a = grad_out / b
//           grad_b = -grad_out * a / b²
// ============================================================================
tensor_div :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		panic("tensor_div: dimension mismatch")
	}
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = a.data.data[i] / b.data.data[i]
	}
	out := _tensor_new_op(out_data, a.shape, a.requires_grad || b.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Div
		append(&out.inputs, a)
		append(&out.inputs, b)
	}
	return out
}

// ============================================================================
// Standard Normal CDF: out = N(a) = P(Z ≤ a) where Z ~ N(0,1)
// Uses Hastings approximation (accurate to ~10^-7)
// Backward: grad_a = grad_out * φ(a) where φ is the standard normal PDF
// ============================================================================
tensor_norm_cdf :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))

	// Hastings approximation coefficients
	p := 0.2316419
	b1 := 0.319381530
	b2 := -0.356563782
	b3 := 1.781477937
	b4 := -1.821255978
	b5 := 1.330274429
	inv_sqrt_2pi := 0.3989422804014327 // 1/√(2π)

	for i in 0 ..< len(a.data.data) {
		x := a.data.data[i]
		sign := 1.0
		ax := x
		if x < 0.0 {
			sign = -1.0
			ax = -x
		}

		t := 1.0 / (1.0 + p * ax)
		t2 := t * t
		t3 := t2 * t
		t4 := t3 * t
		t5 := t4 * t

		poly := b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5
		phi := math.exp(-0.5 * ax * ax) * inv_sqrt_2pi
		cdf := 1.0 - phi * poly

		if sign < 0.0 {
			cdf = 1.0 - cdf
		}

		out_data.data[i] = cdf
	}

	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .NormCDF
		append(&out.inputs, a)
	}
	return out
}
tensor_sum_dim1 :: proc(a: ^Tensor) -> ^Tensor {
	_check_shape(a)
	batch_size := a.shape[0]
	num_assets := a.shape[1]

	out_data := l.matrix_new(f64, batch_size, 1, _op_alloc(a))
	for b in 0 ..< batch_size {
		sum := 0.0
		for asset in 0 ..< num_assets {
			sum += a.data.data[b * num_assets + asset]
		}
		out_data.data[b] = sum
	}

	out := _tensor_new_op(out_data, [4]int{batch_size, 1, 1, 1}, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .SumDim1
		append(&out.inputs, a)
	}
	return out
}
// ===== add to ./wotan/tensor/tensor.odin =====

// tensor_clamp: element-wise clamp to [lo, hi]
tensor_clamp :: proc(a: ^Tensor, lo: f64, hi: f64) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	for i in 0 ..< len(a.data.data) {
		v := a.data.data[i]
		if v < lo {v = lo}
		if v > hi {v = hi}
		out_data.data[i] = v
	}
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Clamp // add .Clamp to the Op enum
		append(&out.inputs, a)
		// store lo/hi for backward
		append(&out.f64_metadata, lo)
		append(&out.f64_metadata, hi)
	}
	return out
}
// tensor_softmax applies a numerically stable softmax independently to each row
// of a [N, C] matrix (normalizes over the last dim). A [1, C] tensor is a single row.
tensor_softmax :: proc(logits: ^Tensor, allocator: mem.Allocator = context.allocator) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	N := logits.data.rows
	C := logits.data.cols
	out_data := l.matrix_new(f64, N, C, _op_alloc(logits))

	for r in 0 ..< N {
		row_in := logits.data.data[r * C:(r + 1) * C]
		row_out := out_data.data[r * C:(r + 1) * C]

		max_val := -math.F64_MAX
		for v in row_in {
			if v > max_val {max_val = v}
		}
		exp_sum := 0.0
		for j in 0 ..< C {
			row_out[j] = math.exp_f64(row_in[j] - max_val)
			exp_sum += row_out[j]
		}
		inv_sum := 1.0 / exp_sum
		for j in 0 ..< C {
			row_out[j] *= inv_sum
		}
	}

	out := _tensor_new_op(out_data, logits.shape, logits.requires_grad, _op_alloc(logits))
	if out.requires_grad {
		out.op = .Softmax
		append(&out.inputs, logits)
		append(&out.int_metadata, C) // row width for backward
	}
	return out
}

tensor_entropy :: proc(a: ^Tensor) -> ^Tensor {
	entropy_val := 0.0
	for i in 0 ..< len(a.data.data) {
		p := a.data.data[i]
		if p > 1e-10 {
			entropy_val += -p * math.ln_f64(p)
		}
	}
	out_data := l.matrix_new(f64, 1, 1, _op_alloc(a))
	out_data.data[0] = entropy_val
	out := _tensor_new_op(out_data, SCALAR_SHAPE, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Entropy
		append(&out.inputs, a)
	}
	return out
}
// ============================================================================
// Binary Cross-Entropy Loss
// ============================================================================
tensor_bce_loss :: proc(prediction: ^Tensor, target: ^Tensor) -> ^Tensor {
	// ✅ FIX: Use runtime assert instead of compile-time #assert
	assert(
		len(prediction.data.data) == len(target.data.data),
		"BCE loss: prediction and target must have same size",
	)

	n := len(prediction.data.data)
	loss_sum := 0.0

	// Forward pass: compute BCE loss
	for i in 0 ..< n {
		p := prediction.data.data[i]
		t := target.data.data[i]

		// Clamp prediction to avoid log(0)
		p = math.max(1e-7, math.min(1.0 - 1e-7, p))

		// BCE formula: -[t * log(p) + (1-t) * log(1-p)]
		loss_sum += -(t * math.ln_f64(p) + (1.0 - t) * math.ln_f64(1.0 - p))
	}

	// Average loss
	avg_loss := loss_sum / f64(n)

	// Create output tensor
	out_data := l.matrix_new(f64, 1, 1, _op_alloc(prediction))
	out_data.data[0] = avg_loss
	out := _tensor_new_op(
		out_data,
		SCALAR_SHAPE,
		prediction.requires_grad || target.requires_grad,
		_op_alloc(prediction),
	)

	if out.requires_grad {
		out.op = .BCELoss
		append(&out.inputs, prediction)
		append(&out.inputs, target)
	}

	return out
}
// ============================================================================
// Tensor Concatenation (Multi-dimensional)
// ============================================================================

tensor_concat :: proc(
	tensors: []^Tensor,
	dim: int,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	for x in tensors {_check_shape(x)}
	if len(tensors) == 0 {
		panic("tensor_concat: empty tensor slice")
	}
	if dim < 0 || dim >= 4 {
		panic("tensor_concat: dim out of bounds (0-3)")
	}

	ref_shape := tensors[0].shape
	total_dim_size := 0
	requires_grad := false

	for t in tensors {
		total_dim_size += t.shape[dim]
		if t.requires_grad {
			requires_grad = true
		}
		for d in 0 ..< 4 {
			if d != dim && t.shape[d] != ref_shape[d] {
				panic(
					fmt.tprintf(
						"tensor_concat: shape mismatch at dim %d (expected %d, got %d)",
						d,
						ref_shape[d],
						t.shape[d],
					),
				)
			}
		}
	}

	out_shape := ref_shape
	out_shape[dim] = total_dim_size

	total_elements := out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
	out_data := l.matrix_new(f64, 1, total_elements, allocator)

	// Calculate strides for block copying
	outer_blocks := 1
	for d in 0 ..< dim {
		outer_blocks *= ref_shape[d]
	}

	inner_block_size := 1
	for d in dim + 1 ..< 4 {
		inner_block_size *= ref_shape[d]
	}

	dim_offset := 0
	for t in tensors {
		t_dim_size := t.shape[dim]

		for ob in 0 ..< outer_blocks {
			for c in 0 ..< t_dim_size {
				src_start := ob * (t_dim_size * inner_block_size) + c * inner_block_size
				dst_start :=
					ob * (total_dim_size * inner_block_size) + (dim_offset + c) * inner_block_size

				copy(
					out_data.data[dst_start:dst_start + inner_block_size],
					t.data.data[src_start:src_start + inner_block_size],
				)
			}
		}
		dim_offset += t_dim_size
	}

	out := _tensor_new_op(out_data, out_shape, requires_grad, allocator)

	if requires_grad {
		out.op = .Concat
		for t in tensors {
			append(&out.inputs, t)
		}
		// Store dim and the dim_size of each input tensor for the backward pass
		append(&out.int_metadata, dim)
		for t in tensors {
			append(&out.int_metadata, t.shape[dim])
		}
	}

	return out
}
tensor_permute_lob :: proc(
	x: ^Tensor,
	batch, c_out, t_out, l_out: int,
	alloc: mem.Allocator,
) -> ^Tensor {
	alloc := _scope_or(alloc) // the active graph scope, if any
	feat_dim := c_out * l_out
	out_data := l.matrix_new(f64, 1, batch * t_out * feat_dim, alloc)

	for b: int = 0; b < batch; b += 1 {
		for tt: int = 0; tt < t_out; tt += 1 {
			for c: int = 0; c < c_out; c += 1 {
				src_idx := b * (c_out * t_out * l_out) + c * (t_out * l_out) + tt * l_out
				dst_idx := (b * t_out + tt) * feat_dim + c * l_out
				copy(out_data.data[dst_idx:dst_idx + l_out], x.data.data[src_idx:src_idx + l_out])
			}
		}
	}

	out := _tensor_new_op(out_data, [4]int{batch, t_out, feat_dim, 1}, x.requires_grad, alloc)

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
// Differentiable Financial Loss: Sharpe Ratio
// ============================================================================
// tensor_sharpe_loss computes the negative Sharpe Ratio of a batch of returns.
// Optimizing this loss maximizes the risk-adjusted return.
// returns: 1D or 2D tensor of periodic returns.
// risk_free_rate: scalar f64 (e.g., 0.0).
tensor_sharpe_loss :: proc(
	returns: ^Tensor,
	risk_free_rate: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	n := f64(len(returns.data.data))
	if n == 0 {
		panic("tensor_sharpe_loss: empty returns tensor")
	}

	// 1. Compute mean
	mean := 0.0
	for i in 0 ..< len(returns.data.data) {
		mean += returns.data.data[i]
	}
	mean /= n

	// 2. Compute variance
	var_sum := 0.0
	for i in 0 ..< len(returns.data.data) {
		d := returns.data.data[i] - mean
		var_sum += d * d
	}
	var := var_sum / n
	std := math.sqrt(var + 1e-8)

	// 3. Compute Sharpe Ratio
	excess_mean := mean - risk_free_rate
	sharpe := excess_mean / std

	// 4. Create output tensor (negative sharpe for minimization)
	out_data := l.matrix_new(f64, 1, 1, allocator)
	out_data.data[0] = -sharpe

	out := _tensor_new_op(out_data, SCALAR_SHAPE, returns.requires_grad, allocator)

	if out.requires_grad {
		out.op = .SharpeLoss
		append(&out.inputs, returns)
		// Store metadata for backward pass
		append(&out.int_metadata, int(n))
		append(&out.f64_metadata, risk_free_rate)
		append(&out.f64_metadata, mean)
		append(&out.f64_metadata, std)
	}

	return out
}
// ============================================================================
// Time-Dimension Z-Score Normalization (SIMD-Optimized)
// ============================================================================
// tensor_normalize_time applies Z-score normalization along the Time dimension (dim=2)
// for a 4D tensor of shape [Batch, Channels, Time, Levels].
// This prevents look-ahead bias and is highly optimized using SIMD reductions.
tensor_normalize_time :: proc(
	input: ^Tensor,
	eps: f64 = 1e-8,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(input)
	N := input.shape[0]
	C := input.shape[1]
	T := input.shape[2]
	L := input.shape[3]

	out_data := l.matrix_new(f64, 1, N * C * T * L, allocator)

	// Temporary contiguous buffers for SIMD operations (allocated once)
	time_buf := make([]f64, T, allocator)
	centered := make([]f64, T, allocator)
	mean_vec := make([]f64, T, allocator)
	inv_std_vec := make([]f64, T, allocator)

	defer {
		delete(time_buf, allocator)
		delete(centered, allocator)
		delete(mean_vec, allocator)
		delete(inv_std_vec, allocator)
	}

	for n: int = 0; n < N; n += 1 {
		for c: int = 0; c < C; c += 1 {
			for el: int = 0; el < L; el += 1 {
				// 1. Extract time series to contiguous buffer
				for t: int = 0; t < T; t += 1 {
					src_idx := n * (C * T * L) + c * (T * L) + t * L + el
					time_buf[t] = input.data.data[src_idx]
				}

				// 2. Compute mean (SIMD sum)
				mean := l.sum_simd(time_buf) / f64(T)
				for i: int = 0; i < T; i += 1 {mean_vec[i] = mean}

				// 3. Compute variance (SIMD)
				l.vec_sub_simd(time_buf, mean_vec, centered)
				var := l.dot_simd(centered, centered) / f64(T)
				std := math.sqrt_f64(var + eps)
				inv_std := 1.0 / std

				// 4. Normalize (SIMD)
				for i: int = 0; i < T; i += 1 {inv_std_vec[i] = inv_std}
				l.vec_mul_simd(centered, inv_std_vec, time_buf) // Reuse time_buf for output

				// 5. Write back
				for t: int = 0; t < T; t += 1 {
					dst_idx := n * (C * T * L) + c * (T * L) + t * L + el
					out_data.data[dst_idx] = time_buf[t]
				}
			}
		}
	}

	out := _tensor_new_op(out_data, input.shape, input.requires_grad, allocator)

	if out.requires_grad {
		out.op = .NormalizeTime
		append(&out.inputs, input)
		append(&out.int_metadata, N)
		append(&out.int_metadata, C)
		append(&out.int_metadata, T)
		append(&out.int_metadata, L)
		append(&out.f64_metadata, eps)
	}

	return out
}
// ============================================================================
// LogSumExp over Dimension 1 (Core Math Primitive)
// ============================================================================
tensor_logsumexp_dim1 :: proc(
	q: ^Tensor,
	num_actions: int,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(q)
	batch := q.shape[0]
	out_data := l.matrix_new(f64, batch, 1, allocator)

	for i in 0 ..< batch {
		max_val := -math.F64_MAX
		for a in 0 ..< num_actions {
			val := q.data.data[i * num_actions + a]
			if val > max_val {max_val = val}
		}
		sum_exp := 0.0
		for a in 0 ..< num_actions {
			sum_exp += math.exp(q.data.data[i * num_actions + a] - max_val)
		}
		out_data.data[i] = max_val + math.ln(sum_exp)
	}

	out := _tensor_new_op(out_data, [4]int{batch, 1, 1, 1}, q.requires_grad, allocator)
	if out.requires_grad {
		out.op = .LogSumExpDim1
		append(&out.inputs, q)
		append(&out.int_metadata, num_actions)
	}
	return out
}

// ============================================================================
// Softplus: out = log(1 + exp(x))
// Backward: grad_a = grad_out * sigmoid(x)
// ============================================================================
tensor_softplus :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, _op_alloc(a))
	for i in 0 ..< len(a.data.data) {
		x := a.data.data[i]
		// ✅ FIX: Correct numerically stable softplus
		if x > 20.0 {
			out_data.data[i] = x
		} else if x < -20.0 {
			out_data.data[i] = math.exp(x)
		} else {
			out_data.data[i] = math.ln_f64(1.0 + math.exp(x))
		}
	}
	out := _tensor_new_op(out_data, a.shape, a.requires_grad, _op_alloc(a))
	if out.requires_grad {
		out.op = .Softplus
		append(&out.inputs, a)
	}
	return out
}
// ============================================================================
// Selective State Space Model (Mamba-style) Operations
// ============================================================================

// The scan is independent for every batch row: h[b, d, n] only ever reads
// h[b, d, n] of the previous step. Batch rows are therefore split across
// threads; each row is computed exactly as in a sequential run.

// Below this many state updates per thread, a thread costs more than it saves
SSM_MIN_WORK_PER_THREAD :: 1 << 14

// _ssm_threads: number of tasks to split `batch` rows of `work_per_row` updates into
_ssm_threads :: proc(batch, work_per_row: int) -> int {
	n := min(batch, batch * work_per_row / SSM_MIN_WORK_PER_THREAD)
	if n <= 1 {return 1} // decide before asking the OS for the core count
	return min(n, util.parallel_thread_count())
}

_Ssm_Fwd_Job :: struct {
	x, h_0, A, B, C, Delta, D: []f64,
	out, h:                    []f64, // h: per-row running state [batch, d_model * d_state]
	// Saved for backward when gradients are needed (else nil):
	// exp(delta * A) of every step, [batch, seq_len, d_model * d_state]
	abar:                      []f64,
	seq_len, d_model, d_state: int,
}

// _ssm_forward_rows: y[b, s, d] = x*D + sum_n C*h_n with
// h_n = exp(delta * A) * h_prev + (delta * B) * x, for batch rows [b0, b1)
_ssm_forward_rows :: proc(data: rawptr, b0, b1: int) #no_bounds_check {
	j := (^_Ssm_Fwd_Job)(data)
	S, Dm, N := j.seq_len, j.d_model, j.d_state
	DN := Dm * N
	// Local copies: stores through the slices cannot alias the job struct
	X, H0, A, Bm, C, Dl, D, out := j.x, j.h_0, j.A, j.B, j.C, j.Delta, j.D, j.out
	abar := j.abar
	save := abar != nil
	for b in b0 ..< b1 {
		h := j.h[b * DN:(b + 1) * DN]
		copy(h, H0[b * DN:(b + 1) * DN])
		for s in 0 ..< S {
			row_x := (b * S + s) * Dm
			row_bc := (b * S + s) * N
			st := (b * S + s) * DN
			for d in 0 ..< Dm {
				x_val := X[row_x + d]
				delta := Dl[row_x + d]

				y_val := x_val * D[d] // Skip connection
				for n in 0 ..< N {
					A_bar := math.exp(delta * A[d * N + n])
					if save {abar[st + d * N + n] = A_bar}
					B_bar := delta * Bm[row_bc + n]

					h_val := A_bar * h[d * N + n] + B_bar * x_val
					h[d * N + n] = h_val

					y_val += C[row_bc + n] * h_val
				}
				out[row_x + d] = y_val
			}
		}
	}
}

tensor_ssm :: proc(
	x: ^Tensor, // [batch, seq_len, d_model]
	h_0: ^Tensor, // [batch, d_model, d_state] (flattened)
	A: ^Tensor, // [d_model, d_state]
	B: ^Tensor, // [batch, seq_len, d_state]
	C: ^Tensor, // [batch, seq_len, d_state]
	Delta: ^Tensor, // [batch, seq_len, d_model]
	D: ^Tensor, // [1, d_model]
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(x)
	_check_shape(A)
	batch := x.shape[0]
	seq_len := x.shape[1]
	d_model := x.shape[2]
	d_state := A.shape[1]

	rg :=
		x.requires_grad ||
		A.requires_grad ||
		B.requires_grad ||
		C.requires_grad ||
		Delta.requires_grad ||
		D.requires_grad

	out_data := l.matrix_new(f64, 1, batch * seq_len * d_model, _op_alloc(x))

	h := make([]f64, batch * d_model * d_state, allocator)
	defer delete(h, allocator)

	// Saved for backward (f64_metadata): exp(delta * A), see _Ssm_Fwd_Job.
	// Every element is written by the scan, so the buffer is not zeroed first.
	saved := make([dynamic]f64, 0, _op_alloc(x))
	if rg {non_zero_resize(&saved, batch * seq_len * d_model * d_state)}

	job := _Ssm_Fwd_Job {
		x       = x.data.data,
		h_0     = h_0.data.data,
		A       = A.data.data,
		B       = B.data.data,
		C       = C.data.data,
		Delta   = Delta.data.data,
		D       = D.data.data,
		out     = out_data.data,
		h       = h,
		abar    = saved[:] if rg else nil,
		seq_len = seq_len,
		d_model = d_model,
		d_state = d_state,
	}
	util.parallel_for(batch, &job, _ssm_forward_rows, _ssm_threads(batch, seq_len * d_model * d_state))

	out := _tensor_new_op(out_data, [4]int{batch, seq_len, d_model, 1}, rg, _op_alloc(x))
	_keep_saved(out, saved)

	if out.requires_grad {
		out.op = .SSM
		// ✅ CRITICAL: Append ALL 7 inputs so tensor_free_graph can find them!
		append(&out.inputs, x) // x_proj
		append(&out.inputs, h_0)
		append(&out.inputs, A)
		append(&out.inputs, B) // B_proj
		append(&out.inputs, C) // C_proj
		append(&out.inputs, Delta)
		append(&out.inputs, D) // layer.D

		// If your backward pass uses int_metadata for shapes, keep that too
		append(&out.int_metadata, batch)
		append(&out.int_metadata, seq_len)
		append(&out.int_metadata, d_model)
		append(&out.int_metadata, d_state)
	}

	return out
}
// ============================================================================
// FlashAttention (Memory-Efficient Tiled Attention with Online Softmax)
// ============================================================================
tensor_flash_attention :: proc(
	Q: ^Tensor,
	K: ^Tensor,
	V: ^Tensor,
	causal: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(Q)
	_check_shape(K)
	_check_shape(V)
	batch := Q.shape[0]
	seq_q := Q.shape[1]
	seq_k := K.shape[1]
	d_k := Q.shape[2]
	d_v := V.shape[2]

	out_data := l.matrix_new(f64, 1, batch * seq_q * d_v, _op_alloc(Q))
	scale := 1.0 / math.sqrt(f64(d_k))

	// Block size for K and V tiling (chosen to fit in L1/L2 cache)
	Bc := 64
	if seq_k < Bc {Bc = seq_k}

	// Pre-allocate SRAM-like buffers to avoid allocations in the inner loop
	s_tile := make([]f64, Bc, allocator)
	p_tile := make([]f64, Bc, allocator)
	o_acc := make([]f64, d_v, allocator)
	defer {
		delete(s_tile, allocator)
		delete(p_tile, allocator)
		delete(o_acc, allocator)
	}

	for b in 0 ..< batch {
		for i in 0 ..< seq_q {
			q_offset := (b * seq_q + i) * d_k
			q_row := Q.data.data[q_offset:q_offset + d_k]

			// Online softmax accumulators
			m_i: f64 = -math.F64_MAX
			l_i: f64 = 0.0
			for d in 0 ..< d_v {o_acc[d] = 0.0}
			// Tile over K and V
			for j_start := 0; j_start < seq_k; j_start += Bc {
				j_end := j_start + Bc
				if j_end > seq_k {j_end = seq_k}
				curr_Bc := j_end - j_start

				// Causal optimization: skip entire blocks that are strictly in the future
				if causal && j_start > i {
					break
				}

				// 1. Fused MatMul + Scale: S_ij = Q_i @ K_j^T * scale
				for x in 0 ..< curr_Bc {
					j := j_start + x
					if causal && j > i {
						s_tile[x] = -1e9 // Safe mask value to avoid NaNs during correction
					} else {
						k_offset := (b * seq_k + j) * d_k
						k_row := K.data.data[k_offset:k_offset + d_k]
						s_tile[x] = l.dot_simd(q_row, k_row) * scale
					}
				}

				// 2. Find block max
				block_max := -math.F64_MAX
				for x in 0 ..< curr_Bc {
					if s_tile[x] > block_max {block_max = s_tile[x]}
				}

				// 3. Update global max and compute correction factor
				new_m := m_i
				if block_max > new_m {new_m = block_max}
				correction := math.exp(m_i - new_m)

				// 4. Compute P_ij = exp(S_ij - new_m) and sum
				block_sum: f64 = 0.0
				for x in 0 ..< curr_Bc {
					p_val := math.exp(s_tile[x] - new_m)
					p_tile[x] = p_val
					block_sum += p_val
				}

				// 5. Update denominator
				new_l := l_i * correction + block_sum

				// 6. Fused MAC: O_i = O_i * correction + P_ij @ V_j
				for d in 0 ..< d_v {o_acc[d] *= correction}
				for x in 0 ..< curr_Bc {
					j := j_start + x
					if causal && j > i {
						continue
					}
					v_offset := (b * seq_k + j) * d_v
					v_row := V.data.data[v_offset:v_offset + d_v]
					p_val := p_tile[x]
					l.axpy_simd(p_val, v_row, o_acc)
				}

				m_i = new_m
				l_i = new_l
			}

			// 7. Final normalization: O_i = O_i / l_i
			inv_l := 0.0
			if l_i > 1e-12 {
				inv_l = 1.0 / l_i
			}
			out_offset := (b * seq_q + i) * d_v
			for d in 0 ..< d_v {
				out_data.data[out_offset + d] = o_acc[d] * inv_l
			}
		}
	}

	out := _tensor_new_op(
		out_data,
		[4]int{batch, seq_q, d_v, 1},
		Q.requires_grad || K.requires_grad || V.requires_grad,
		_op_alloc(Q),
	)
	if out.requires_grad {
		out.op = .FlashAttention
		append(&out.inputs, Q)
		append(&out.inputs, K)
		append(&out.inputs, V)
		// Store causal flag for backward pass
		if causal {
			append(&out.int_metadata, 1)
		} else {
			append(&out.int_metadata, 0)
		}
	}
	return out
}
// ============================================================================
// Top-K Mask (Non-Differentiable)
// ============================================================================
tensor_top_k_mask :: proc(
	probs: ^Tensor,
	k: int,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	_check_shape(probs)
	N := probs.shape[0]
	E := probs.shape[1]
	mask_data := l.matrix_new(f64, N, E, allocator)
	top_k_indices := make([]int, k, allocator)
	defer delete(top_k_indices, allocator)
	for n in 0 ..< N {
		row := probs.data.data[n * E:(n + 1) * E]

		// Simple insertion-sort based Top-K for each row
		for i in 0 ..< k {top_k_indices[i] = i}

		for e in 0 ..< E {
			val := row[e]
			// Check if it belongs in top_k
			if val > row[top_k_indices[k - 1]] {
				top_k_indices[k - 1] = e
				// Bubble up to maintain sorted order
				for j := k - 1; j > 0; j -= 1 {
					if row[top_k_indices[j]] > row[top_k_indices[j - 1]] {
						tmp := top_k_indices[j]
						top_k_indices[j] = top_k_indices[j - 1]
						top_k_indices[j - 1] = tmp
					} else {
						break
					}
				}
			}
		}

		for i in 0 ..< k {
			mask_data.data[n * E + top_k_indices[i]] = 1.0
		}
	}

	// Mask is NOT differentiable. It acts as a constant.
	out := tensor_new(mask_data, false, allocator)
	out.shape = probs.shape
	out.owned_by_graph = true
	return out
}

// ============================================================================
// Gate Multiply: out[n, d] = x[n, d] * gate[n, 0]
// ============================================================================
tensor_gate_mul :: proc(
	x: ^Tensor,
	gate: ^Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	_check_shape(x)
	N := x.shape[0]
	D := x.shape[1]
	out_data := l.matrix_new(f64, N, D, allocator)

	for n in 0 ..< N {
		g := gate.data.data[n]
		for d in 0 ..< D {
			out_data.data[n * D + d] = x.data.data[n * D + d] * g
		}
	}

	out := _tensor_new_op(out_data, x.shape, x.requires_grad || gate.requires_grad, allocator)
	if out.requires_grad {
		out.op = .GateMul
		append(&out.inputs, x)
		append(&out.inputs, gate)
	}
	return out
}

// tensor_reshape changes the logical shape of the tensor without altering the flat memory layout.
tensor_reshape :: proc(
	input: ^Tensor,
	new_shape: [4]int,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	old_total := len(input.data.data)
	new_total := new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3]

	if old_total != new_total {
		panic(
			fmt.aprintf(
				"tensor_reshape: cannot reshape tensor of size %d to size %d",
				old_total,
				new_total,
			),
		)
	}

	// Create a new matrix container with the exact same flat data
	out_data := l.matrix_new(f64, 1, new_total, allocator)
	copy(out_data.data, input.data.data)

	out := _tensor_new_op(out_data, new_shape, input.requires_grad, allocator)

	if out.requires_grad {
		out.op = .Reshape
		append(&out.inputs, input)
	}
	return out
}
// tensor_slice_gate extracts a single expert's gate column while preserving the autograd graph.
tensor_slice_gate :: proc(
	gates: ^Tensor,
	expert_idx: int,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	N := gates.data.rows
	E := gates.data.cols

	out_data := l.matrix_new(f64, N, 1, allocator)
	for n in 0 ..< N {
		out_data.data[n] = gates.data.data[n * E + expert_idx]
	}

	out := _tensor_new_op(out_data, [4]int{N, 1, 1, 1}, gates.requires_grad, allocator)

	if out.requires_grad {
		out.op = .SliceGate
		append(&out.inputs, gates)
		append(&out.int_metadata, expert_idx)
		append(&out.int_metadata, E) // Store total experts for backward pass
	}
	return out
}
// ============================================================================
// Prospect Theory: Value Function (Kahneman & Tversky, 1992)
// ============================================================================
// v(x) = x^α           if x ≥ 0  (concave gains)
// v(x) = -λ(-x)^β      if x < 0  (convex losses, loss-averse)
// Calibrated: α = β = 0.88, λ = 2.25

tensor_prospect_value :: proc(
	returns: ^Tensor,
	alpha: f64 = 0.88,
	beta: f64 = 0.88,
	lambda: f64 = 2.25,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	out_data := l.matrix_new(f64, returns.data.rows, returns.data.cols, allocator)

	for i in 0 ..< len(returns.data.data) {
		x := returns.data.data[i]
		if x >= 0.0 {
			out_data.data[i] = math.pow(x, alpha)
		} else {
			out_data.data[i] = -lambda * math.pow(-x, beta)
		}
	}

	out := _tensor_new_op(out_data, returns.shape, returns.requires_grad, allocator)
	if out.requires_grad {
		out.op = .ProspectValue
		append(&out.inputs, returns)
		append(&out.f64_metadata, alpha)
		append(&out.f64_metadata, beta)
		append(&out.f64_metadata, lambda)
	}
	return out
}

// ============================================================================
// Prospect Theory: Probability Weighting (Cumulative PT, Tversky & Kahneman 1992)
// ============================================================================
// w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
// Calibrated: γ_gain = 0.61, γ_loss = 0.69

tensor_prob_weighting :: proc(
	p: ^Tensor,
	gamma: f64 = 0.61,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	eps := 1e-7
	out_data := l.matrix_new(f64, p.data.rows, p.data.cols, allocator)

	for i in 0 ..< len(p.data.data) {
		pi := math.max(eps, math.min(1.0 - eps, p.data.data[i]))
		p_gamma := math.pow(pi, gamma)
		q_gamma := math.pow(1.0 - pi, gamma)
		denom := math.pow(p_gamma + q_gamma, 1.0 / gamma)
		out_data.data[i] = p_gamma / denom
	}

	out := _tensor_new_op(out_data, p.shape, p.requires_grad, allocator)
	if out.requires_grad {
		out.op = .ProbWeighting
		append(&out.inputs, p)
		append(&out.f64_metadata, gamma)
	}
	return out
}

// ============================================================================
// Prospect Theory: Fused Portfolio Utility (like tensor_sharpe_loss)
// ============================================================================
// U = -mean(v(r_i))   (negated for gradient descent minimization)
// This is the RL reward signal that makes agents trade like humans.

tensor_prospect_utility :: proc(
	returns: ^Tensor,
	alpha: f64 = 0.88,
	beta: f64 = 0.88,
	lambda: f64 = 2.25,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	allocator := _scope_or(allocator) // the active graph scope, if any
	n := f64(len(returns.data.data))
	if n == 0 {
		panic("tensor_prospect_utility: empty returns tensor")
	}

	// Fused forward: compute mean prospect value in one pass
	utility_sum := 0.0
	for i in 0 ..< len(returns.data.data) {
		x := returns.data.data[i]
		if x >= 0.0 {
			utility_sum += math.pow(x, alpha)
		} else {
			utility_sum += -lambda * math.pow(-x, beta)
		}
	}

	// Negate for minimization (like sharpe_loss)
	out_data := l.matrix_new(f64, 1, 1, allocator)
	out_data.data[0] = -(utility_sum / n)

	out := _tensor_new_op(out_data, SCALAR_SHAPE, returns.requires_grad, allocator)
	if out.requires_grad {
		out.op = .ProspectUtility
		append(&out.inputs, returns)
		append(&out.f64_metadata, alpha)
		append(&out.f64_metadata, beta)
		append(&out.f64_metadata, lambda)
	}
	return out
}
