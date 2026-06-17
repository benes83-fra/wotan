
package tensor

import l "../linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
// ============================================================================
// 1. The Tensor Struct & Graph Nodes
// ============================================================================

// Op defines the operation that created this tensor.
// We use an enum instead of function pointers to keep it Odin-idiomatic.
Op :: enum {
	None,
	Add,
	Mul,
	MatMul,
	Sum,
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
	PermuteMHA,
	PermuteMHAInverse,
	LayerNorm,
	MaskedScaledDotProductAttention,
}

PoolParams :: struct {
	kH:     int,
	kW:     int,
	stride: int,
	pad:    int,
}

Tensor :: struct {
	data:          l.Matrix(f64),
	grad:          l.Matrix(f64),
	requires_grad: bool,
	op:            Op,
	inputs:        [dynamic]^Tensor,
	allocator:     mem.Allocator,
	int_metadata:  [dynamic]int,
	dropout_mask:  []f64,
	shape:         [4]int,
	conv_params:   ConvParams,
	pool_params:   PoolParams, // ✅ ADD THIS
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
	// Initialize gradients to zero
	if requires_grad {
		t.grad = l.matrix_new(f64, data.rows, data.cols, allocator)
		// (In a real implementation, we'd zero it out here, but matrix_new usually does)
	}

	return t
}

tensor_free :: proc(t: ^Tensor) {
	if t == nil {return}
	if t.data.data != nil {l.matrix_free(&t.data)}
	if t.grad.data != nil {l.matrix_free(&t.grad)}

	delete(t.inputs)
	delete(t.int_metadata)
	if t.dropout_mask != nil {delete(t.dropout_mask, t.allocator)} 	// ✅ ADD THIS
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

	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	// Allocate the mask to remember what we dropped for the backward pass
	mask := make([]f64, n, a.allocator)
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

	out := tensor_new(out_data, a.requires_grad, a.allocator)
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
	out_data := l.matrix_new(f64, 1, 1, a.allocator)
	out_data.data[0] = total

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
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
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_add_simd(a.data.data, b.data.data, out_data.data)

	// 2. Create the output tensor
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)
	out.shape = a.shape
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
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_relu_simd(a.data.data, out_data.data) // ✅ SIMD ReLU

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	// 3. Record graph
	if out.requires_grad {
		out.op = .Relu
		append(&out.inputs, a)
	}

	return out
}


// tensor_mul creates a new tensor C = A * B (element-wise)
tensor_mul :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		panic("tensor_mul: dimension mismatch")
	}

	// 1. Forward pass using SIMD
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_mul_simd(a.data.data, b.data.data, out_data.data)

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)
	out.shape = a.shape
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

	// ✅ NEW: Handle flattened 3D tensor
	if a.data.rows == 1 && a.data.cols != D {
		if a.data.cols % D == 0 {
			N := a.data.cols / D
			out_data := l.matrix_new(f64, 1, a.data.cols, a.allocator)

			for i in 0 ..< N {
				row_out := out_data.data[i * D:(i + 1) * D]
				row_a := a.data.data[i * D:(i + 1) * D]
				l.vec_add_simd(row_a, bias.data.data, row_out)
			}

			out := tensor_new(out_data, a.requires_grad || bias.requires_grad, a.allocator)
			out.shape = a.shape
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
	out_data := l.matrix_new(f64, N, D, a.allocator)

	for i in 0 ..< N {
		row_out := out_data.data[i * D:(i + 1) * D]
		row_a := a.data.data[i * D:(i + 1) * D]
		l.vec_add_simd(row_a, bias.data.data, row_out)
	}

	out := tensor_new(out_data, a.requires_grad || bias.requires_grad, a.allocator)
	out.shape = a.shape
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
tensor_mse_loss :: proc(pred: ^Tensor, target: ^Tensor) -> ^Tensor {
	if pred.data.rows != target.data.rows || pred.data.cols != target.data.cols {
		panic("tensor_mse_loss: shape mismatch")
	}

	n := f64(len(pred.data.data))

	diff := make([]f64, len(pred.data.data), context.temp_allocator)
	l.vec_sub_simd(pred.data.data, target.data.data, diff)
	squared_sum := l.dot_simd(diff, diff)
	defer delete(diff, context.temp_allocator)

	loss_val := squared_sum / n

	out_data := l.matrix_new(f64, 1, 1, pred.allocator)
	out_data.data[0] = loss_val

	out := tensor_new(out_data, pred.requires_grad, pred.allocator)
	// ✅ FIX: Restored the correct MSE graph recording!
	if out.requires_grad {
		out.op = .MSELoss
		append(&out.inputs, pred)
		append(&out.inputs, target)
	}
	return out
}

// tensor_sigmoid applies σ(x) = 1 / (1 + exp(-x))
tensor_sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	for i in 0 ..< n {
		out_data.data[i] = 1.0 / (1.0 + math.exp(-a.data.data[i]))
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape

	if out.requires_grad {
		out.op = .Sigmoid
		append(&out.inputs, a)
	}
	return out
}

// tensor_tanh applies tanh(x)
tensor_tanh :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	for i in 0 ..< n {
		out_data.data[i] = math.tanh(a.data.data[i])
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape

	if out.requires_grad {
		out.op = .Tanh
		append(&out.inputs, a)
	}
	return out
}

// tensor_leaky_relu applies max(α*x, x) where α is small (e.g., 0.01)
tensor_leaky_relu :: proc(a: ^Tensor, alpha: f64 = 0.01) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	for i in 0 ..< n {
		v := a.data.data[i]
		out_data.data[i] = v > 0.0 ? v : alpha * v
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape

	if out.requires_grad {
		out.op = .LeakyReLU
		append(&out.inputs, a)
	}
	return out
}


// tensor_cross_entropy_loss calculates the loss for multi-class classification.
tensor_cross_entropy_loss :: proc(logits: ^Tensor, target_indices: []int) -> ^Tensor {
	if len(target_indices) != logits.data.rows {
		panic("tensor_cross_entropy_loss: batch size mismatch")
	}

	N := logits.data.rows
	C := logits.data.cols

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

		// Note: Use math.log for natural logarithm in Odin
		log_sum_exp := math.ln(sum_exp) + max_val

		target_class := target_indices[i]
		log_prob := logits.data.data[i * C + target_class] - log_sum_exp
		loss -= log_prob
	}

	loss /= f64(N)

	out_data := l.matrix_new(f64, 1, 1, logits.allocator)
	out_data.data[0] = loss

	out := tensor_new(out_data, logits.requires_grad, logits.allocator)

	if out.requires_grad {
		out.op = .CrossEntropy
		append(&out.inputs, logits)
		// ✅ Save indices for the backward pass
		for idx in target_indices {
			append(&out.int_metadata, idx)
		}
	}
	return out
}


// tensor_flatten reshapes a 4D tensor (N, C, H, W) to 2D (N, C*H*W)
tensor_flatten :: proc(input: ^Tensor) -> ^Tensor {
	N := input.shape[0]
	features := input.shape[1] * input.shape[2] * input.shape[3]

	// Just create a new tensor with the same data but different shape
	out_data := l.matrix_new(f64, N, features, input.allocator)
	copy(out_data.data, input.data.data)

	out := tensor_new(out_data, input.requires_grad, input.allocator)
	out.shape = [4]int{N, features, 1, 1} // Treat as 2D

	if out.requires_grad {
		out.op = .Flatten
		append(&out.inputs, input)
	}

	return out
}


// ============================================================================
// 4. The Backward Engine (Chain Rule)
// ============================================================================

// Helper: Topological Sort to determine the order of operations
_build_topo :: proc(node: ^Tensor, topo: ^[dynamic]^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil {return}
	if visited[node] {return}

	visited[node] = true

	for input in node.inputs {
		_build_topo(input, topo, visited)
	}

	append(topo, node)
}
// tensor_ensure_grad ensures a tensor has a gradient matrix allocated
tensor_ensure_grad :: proc(t: ^Tensor) {
	if t.requires_grad && t.grad.data == nil {
		t.grad = l.matrix_new(f64, t.data.rows, t.data.cols, t.allocator)
	}
}

tensor_backward :: proc(root: ^Tensor) {
	if !root.requires_grad {return}

	// 1. Set the gradient of the root node to 1.0
	tensor_ensure_grad(root)
	has_custom_grad := false
	for i in 0 ..< len(root.grad.data) {
		if root.grad.data[i] != 0.0 {
			has_custom_grad = true
			break
		}
	}

	// Only set to 1.0 if user hasn't set a custom gradient
	if !has_custom_grad {
		for i in 0 ..< len(root.grad.data) {
			root.grad.data[i] = 1.0
		}
	}

	// 2. Build the topological sort
	topo := make([dynamic]^Tensor, 0, root.allocator)
	defer delete(topo) // ✅ FIX: Removed allocator

	visited := make(map[^Tensor]bool, root.allocator)
	defer delete(visited) // ✅ FIX: Removed allocator

	_build_topo(root, &topo, &visited)

	// 3. Iterate in reverse topological order
	for i := len(topo) - 1; i >= 0; i -= 1 {
		node := topo[i]

		if !node.requires_grad {continue}
		// ✅ CRITICAL: Ensure this node has a gradient allocated
		tensor_ensure_grad(node)

		// ✅ CRITICAL: Skip if gradient is still empty (shouldn't happen, but defensive)
		if len(node.grad.data) == 0 {
			fmt.printf("WARNING: Skipping node with op %v - empty gradient\n", node.op)
			continue
		}
		switch node.op {
		case .Add:
			for input in node.inputs {
				if input.requires_grad {
					tensor_ensure_grad(input)
					if len(input.grad.data) > 0 {
						l.axpy_simd(1.0, node.grad.data, input.grad.data)
					}
				}
			}
		case .Mul:
			a_in := node.inputs[0]
			b_in := node.inputs[1]

			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 {
					l.vec_fma_inplace_simd(node.grad.data, b_in.data.data, a_in.grad.data)
				}
			}
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 {
					l.vec_fma_inplace_simd(node.grad.data, a_in.data.data, b_in.grad.data)
				}
			}
		case .MatMul:
			a_in := node.inputs[0]
			b_in := node.inputs[1]
			if len(node.int_metadata) == 3 {
				N := node.int_metadata[0]
				in_features := node.int_metadata[1]
				out_features := node.int_metadata[2]

				a_view := l.Matrix(f64) {
					rows = N,
					cols = in_features,
					data = a_in.data.data,
				}
				b_view := b_in.data
				grad_view := l.Matrix(f64) {
					rows = N,
					cols = out_features,
					data = node.grad.data,
				}

				if a_in.requires_grad {
					tensor_ensure_grad(a_in)
					b_t := _matrix_transpose(b_view, context.temp_allocator)
					grad_a_view := l.matmul_dyn_simd(&grad_view, &b_t, context.temp_allocator)
					l.matrix_free(&b_t)

					// Copy flattened gradient back
					copy(a_in.grad.data, grad_a_view.data)
					l.matrix_free(&grad_a_view)
				}

				if b_in.requires_grad {
					tensor_ensure_grad(b_in)
					a_t := _matrix_transpose(a_view, context.temp_allocator)
					grad_b := l.matmul_dyn_simd(&a_t, &grad_view, context.temp_allocator)
					l.matrix_free(&a_t)

					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&grad_b)
				}
				continue // Skip the standard 2D backward pass
			}
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					// ✅ FIX: Use context.allocator for ALL temporary matrices
					bt := _matrix_transpose(b_in.data, context.allocator)
					grad_a := l.matmul_dyn_simd(&node.grad, &bt, context.allocator)
					l.vec_add_simd(a_in.grad.data, grad_a.data, a_in.grad.data)
					l.matrix_free(&bt)
					l.matrix_free(&grad_a)
				}
			}

			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 && len(node.grad.data) > 0 {
					// ✅ FIX: Use context.allocator
					at := _matrix_transpose(a_in.data, context.allocator)
					grad_b := l.matmul_dyn_simd(&at, &node.grad, context.allocator)
					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&at)
					l.matrix_free(&grad_b)
				}
			}
		case .Sum:
			// L = sum(A)
			// dL/dA_ij = 1 for all i, j.
			// We just broadcast the incoming scalar gradient to the shape of A.
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) == 0 {
					fmt.println("ERROR: Sum grad.data is empty")
					continue
				}
				scalar_grad := node.grad.data[0]
				for j in 0 ..< len(a_in.grad.data) {
					a_in.grad.data[j] += scalar_grad
				}
			}
		case .MaskedScaledDotProductAttention:
			Q_in := node.inputs[0]
			K_in := node.inputs[1]
			V_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			batch := Q_in.shape[0]
			seq_q := Q_in.shape[1]
			seq_k := K_in.shape[1]
			d_k := Q_in.shape[2]
			d_v := V_in.shape[2]
			scale := 1.0 / math.sqrt(f64(d_k))

			// Retrieve mask from metadata
			mask := make([]f64, seq_q * seq_k, context.temp_allocator)
			neg_inf: f64 = -1e9
			for i in 0 ..< len(mask) {
				if node.int_metadata[i] == 1 {
					mask[i] = neg_inf
				} else {
					mask[i] = 0.0
				}
			}
			defer delete(mask, context.temp_allocator)

			dQ := make([]f64, len(Q_in.data.data), context.temp_allocator)
			dK := make([]f64, len(K_in.data.data), context.temp_allocator)
			dV := make([]f64, len(V_in.data.data), context.temp_allocator)
			defer {
				delete(dQ, context.temp_allocator)
				delete(dK, context.temp_allocator)
				delete(dV, context.temp_allocator)
			}

			for b in 0 ..< batch {
				q_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_k,
					data = Q_in.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k],
				}
				k_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_k,
					data = K_in.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k],
				}
				v_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_v,
					data = V_in.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v],
				}
				dO_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_v,
					data = node.grad.data[b * seq_q * d_v:(b + 1) * seq_q * d_v],
				}

				// Recompute S_b and P_b with mask
				k_b_t := _matrix_transpose(k_b, context.temp_allocator)
				s_b := l.matmul_dyn_simd(&q_b, &k_b_t, context.temp_allocator)
				l.matrix_free(&k_b_t)

				p_b := l.Matrix(f64) {
					rows = seq_q,
					cols = seq_k,
					data = s_b.data,
				}
				for i in 0 ..< seq_q * seq_k {
					p_b.data[i] *= scale
				}

				// Apply mask
				for i in 0 ..< seq_q {
					for j in 0 ..< seq_k {
						if mask[i * seq_k + j] != 0.0 {
							p_b.data[i * seq_k + j] += mask[i * seq_k + j]
						}
					}
				}

				// Softmax
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					max_val := p_b.data[row_start]
					for j in 1 ..< seq_k {
						if p_b.data[row_start + j] > max_val {max_val = p_b.data[row_start + j]}
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

				// dV = P_b^T @ dO_b
				p_b_t := _matrix_transpose(p_b, context.temp_allocator)
				dV_b := l.matmul_dyn_simd(&p_b_t, &dO_b, context.temp_allocator)
				copy(dV[b * seq_k * d_v:(b + 1) * seq_k * d_v], dV_b.data)
				l.matrix_free(&p_b_t)
				l.matrix_free(&dV_b)

				// dP = dO_b @ V_b^T
				v_b_t := _matrix_transpose(v_b, context.temp_allocator)
				dP_b := l.matmul_dyn_simd(&dO_b, &v_b_t, context.temp_allocator)

				// dS = P_b * (dP_b - sum(dP_b * P_b, dim=-1))
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					sum := 0.0
					for j in 0 ..< seq_k {
						sum += dP_b.data[row_start + j] * p_b.data[row_start + j]
					}
					for j in 0 ..< seq_k {
						dP_b.data[row_start + j] =
							p_b.data[row_start + j] * (dP_b.data[row_start + j] - sum)
					}
				}

				// dQ = (dP_b * scale) @ K_b
				for i in 0 ..< seq_q * seq_k {
					dP_b.data[i] *= scale
				}
				dQ_b := l.matmul_dyn_simd(&dP_b, &k_b, context.temp_allocator)
				copy(dQ[b * seq_q * d_k:(b + 1) * seq_q * d_k], dQ_b.data)
				l.matrix_free(&dQ_b)

				// dK = (dP_b * scale)^T @ Q_b
				dP_b_t := _matrix_transpose(dP_b, context.temp_allocator)
				dK_b := l.matmul_dyn_simd(&dP_b_t, &q_b, context.temp_allocator)
				copy(dK[b * seq_k * d_k:(b + 1) * seq_k * d_k], dK_b.data)
				l.matrix_free(&dP_b_t)
				l.matrix_free(&dK_b)

				l.matrix_free(&dP_b)
				l.matrix_free(&s_b)
			}

			if Q_in.requires_grad &&
			   len(Q_in.grad.data) > 0 {l.vec_add_simd(Q_in.grad.data, dQ, Q_in.grad.data)}
			if K_in.requires_grad &&
			   len(K_in.grad.data) > 0 {l.vec_add_simd(K_in.grad.data, dK, K_in.grad.data)}
			if V_in.requires_grad &&
			   len(V_in.grad.data) > 0 {l.vec_add_simd(V_in.grad.data, dV, V_in.grad.data)}
		case .Relu:
			// y = max(0, x)
			// dy/dx = 1 if x > 0 else 0
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// ✅ SIMD Optimization: Use the SIMD backward function
				l.vec_relu_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data)
			}
		case .Sigmoid:
			// d/dx σ(x) = σ(x) * (1 - σ(x))
			a_in := node.inputs[0]
			if a_in.requires_grad {
				for i in 0 ..< len(a_in.grad.data) {
					s := a_in.data.data[i] // already σ(x) from forward
					a_in.grad.data[i] += node.grad.data[i] * s * (1.0 - s)
				}
			}

		case .Tanh:
			// d/dx tanh(x) = 1 - tanh²(x)
			a_in := node.inputs[0]
			if a_in.requires_grad {
				for i in 0 ..< len(a_in.grad.data) {
					t := a_in.data.data[i] // already tanh(x) from forward
					a_in.grad.data[i] += node.grad.data[i] * (1.0 - t * t)
				}
			}
		case .PermuteMHA:
			x_in := node.inputs[0]
			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				batch := node.int_metadata[0]
				seq_len := node.int_metadata[1]
				num_heads := node.int_metadata[2]
				head_dim := node.int_metadata[3]
				d_model := num_heads * head_dim

				for b in 0 ..< batch {
					for s in 0 ..< seq_len {
						for h in 0 ..< num_heads {
							for d in 0 ..< head_dim {
								src :=
									(b * num_heads + h) * (seq_len * head_dim) + s * head_dim + d
								dst := b * (seq_len * d_model) + s * d_model + h * head_dim + d
								x_in.grad.data[dst] += node.grad.data[src]
							}
						}
					}
				}
			}

		case .PermuteMHAInverse:
			x_in := node.inputs[0]
			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				batch := node.int_metadata[0]
				seq_len := node.int_metadata[1]
				num_heads := node.int_metadata[2]
				head_dim := node.int_metadata[3]
				d_model := num_heads * head_dim

				for b in 0 ..< batch {
					for s in 0 ..< seq_len {
						for h in 0 ..< num_heads {
							for d in 0 ..< head_dim {
								src := b * (seq_len * d_model) + s * d_model + h * head_dim + d
								dst :=
									(b * num_heads + h) * (seq_len * head_dim) + s * head_dim + d
								x_in.grad.data[dst] += node.grad.data[src]
							}
						}
					}
				}
			}
		case .LeakyReLU:
			// d/dx = 1 if x > 0 else α
			a_in := node.inputs[0]
			if a_in.requires_grad {
				alpha := 0.01 // or store in Tensor struct
				for i in 0 ..< len(a_in.grad.data) {
					if a_in.data.data[i] > 0.0 {
						a_in.grad.data[i] += node.grad.data[i]
					} else {
						a_in.grad.data[i] += node.grad.data[i] * alpha
					}
				}
			}
		case .AddBias:
			a_in := node.inputs[0]
			bias_in := node.inputs[1]

			// ✅ CRITICAL: Check gradient dimensions
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: AddBias node has empty gradient, skipping")
				continue
			}
			if len(node.int_metadata) == 2 {
				N := node.int_metadata[0]
				D := node.int_metadata[1]

				if a_in.requires_grad {
					tensor_ensure_grad(a_in)
					l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
				}

				if bias_in.requires_grad {
					tensor_ensure_grad(bias_in)
					// Sum gradients over the N rows
					for i in 0 ..< N {
						row_grad := node.grad.data[i * D:(i + 1) * D]
						l.axpy_simd(1.0, row_grad, bias_in.grad.data)
					}
				}
				continue // Skip standard 2D backward
			}
			N := node.grad.rows
			D := node.grad.cols

			// ✅ CRITICAL: Verify gradient size matches expected dimensions
			if len(node.grad.data) != N * D {
				fmt.printf(
					"WARNING: AddBias gradient size mismatch: %d != %d * %d\n",
					len(node.grad.data),
					N,
					D,
				)
				continue
			}

			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(a_in.grad.data) == len(node.grad.data) {
					l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
				}
			}

			if bias_in.requires_grad {
				tensor_ensure_grad(bias_in)
				if len(bias_in.grad.data) > 0 && len(bias_in.grad.data) == D {
					for i in 0 ..< N {
						start_idx := i * D
						end_idx := (i + 1) * D

						// ✅ CRITICAL: Bounds check before slicing
						if end_idx > len(node.grad.data) {
							fmt.printf(
								"ERROR: AddBias slice out of bounds: %d:%d > %d\n",
								start_idx,
								end_idx,
								len(node.grad.data),
							)
							continue
						}

						row_grad := node.grad.data[start_idx:end_idx]
						l.axpy_simd(1.0, row_grad, bias_in.grad.data)
					}
				}
			}
		case .Embedding:
			input_in := node.inputs[0] // indices (no grad)
			weight_in := node.inputs[1] // weight matrix

			if len(node.grad.data) == 0 || !weight_in.requires_grad {continue}

			batch := input_in.shape[0]
			seq_len := input_in.shape[1]
			vocab_size := weight_in.shape[0]
			embed_dim := weight_in.shape[1]

			// Ensure gradient matrix is allocated
			if weight_in.grad.data == nil {
				weight_in.grad = l.matrix_new(f64, vocab_size, embed_dim, weight_in.allocator)
			}

			// ✅ OPTIMIZATION: Scatter-add using SIMD vector addition
			for b in 0 ..< batch {
				for s in 0 ..< seq_len {
					idx := int(input_in.data.data[b * seq_len + s])
					if idx < 0 || idx >= vocab_size {continue} 	// Safety

					src_offset := (b * seq_len + s) * embed_dim
					dst_offset := idx * embed_dim

					// weight_in.grad[idx] += node.grad[b, s]
					l.vec_add_simd(
						weight_in.grad.data[dst_offset:dst_offset + embed_dim],
						node.grad.data[src_offset:src_offset + embed_dim],
						weight_in.grad.data[dst_offset:dst_offset + embed_dim],
					)
				}
			}
		case .ScaledDotProductAttention:
			Q_in := node.inputs[0]
			K_in := node.inputs[1]
			V_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			batch := Q_in.shape[0]
			seq_q := Q_in.shape[1]
			seq_k := K_in.shape[1]
			d_k := Q_in.shape[2]
			d_v := V_in.shape[2]
			scale := 1.0 / math.sqrt(f64(d_k))

			dQ := make([]f64, len(Q_in.data.data), context.temp_allocator)
			dK := make([]f64, len(K_in.data.data), context.temp_allocator)
			dV := make([]f64, len(V_in.data.data), context.temp_allocator)
			defer {
				delete(dQ, context.temp_allocator)
				delete(dK, context.temp_allocator)
				delete(dV, context.temp_allocator)
			}

			for b in 0 ..< batch {
				q_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_k,
					data = Q_in.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k],
				}
				k_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_k,
					data = K_in.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k],
				}
				v_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_v,
					data = V_in.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v],
				}
				dO_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_v,
					data = node.grad.data[b * seq_q * d_v:(b + 1) * seq_q * d_v],
				}

				// 1. Recompute S_b and P_b (Checkpointing)
				k_b_t := _matrix_transpose(k_b, context.temp_allocator)
				s_b := l.matmul_dyn_simd(&q_b, &k_b_t, context.temp_allocator)
				l.matrix_free(&k_b_t)

				p_b := l.Matrix(f64) {
					rows = seq_q,
					cols = seq_k,
					data = s_b.data,
				}
				for i in 0 ..< seq_q * seq_k {
					p_b.data[i] *= scale
				}

				// Numerically stable softmax
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					max_val := p_b.data[row_start]
					for j in 1 ..< seq_k {
						if p_b.data[row_start + j] > max_val {max_val = p_b.data[row_start + j]}
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

				// 2. dV = P_b^T @ dO_b
				p_b_t := _matrix_transpose(p_b, context.temp_allocator)
				dV_b := l.matmul_dyn_simd(&p_b_t, &dO_b, context.temp_allocator)
				copy(dV[b * seq_k * d_v:(b + 1) * seq_k * d_v], dV_b.data)
				l.matrix_free(&p_b_t)
				l.matrix_free(&dV_b)

				// 3. dP = dO_b @ V_b^T
				v_b_t := _matrix_transpose(v_b, context.temp_allocator)
				dP_b := l.matmul_dyn_simd(&dO_b, &v_b_t, context.temp_allocator)

				// 4. dS = P_b * (dP_b - sum(dP_b * P_b, dim=-1))  [Standard Softmax Backward]
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					sum := 0.0
					for j in 0 ..< seq_k {
						sum += dP_b.data[row_start + j] * p_b.data[row_start + j]
					}
					for j in 0 ..< seq_k {
						dP_b.data[row_start + j] =
							p_b.data[row_start + j] * (dP_b.data[row_start + j] - sum)
					}
				}

				// 5. dQ = (dP_b * scale) @ K_b
				for i in 0 ..< seq_q * seq_k {
					dP_b.data[i] *= scale
				}
				dQ_b := l.matmul_dyn_simd(&dP_b, &k_b, context.temp_allocator)
				copy(dQ[b * seq_q * d_k:(b + 1) * seq_q * d_k], dQ_b.data)
				l.matrix_free(&dQ_b)

				// 6. dK = (dP_b * scale)^T @ Q_b  => dP_b^T @ Q_b
				dP_b_t := _matrix_transpose(dP_b, context.temp_allocator)
				dK_b := l.matmul_dyn_simd(&dP_b_t, &q_b, context.temp_allocator)
				copy(dK[b * seq_k * d_k:(b + 1) * seq_k * d_k], dK_b.data)
				l.matrix_free(&dP_b_t)
				l.matrix_free(&dK_b)

				l.matrix_free(&dP_b)
				l.matrix_free(&s_b)
			}

			if Q_in.requires_grad &&
			   len(Q_in.grad.data) > 0 {l.vec_add_simd(Q_in.grad.data, dQ, Q_in.grad.data)}
			if K_in.requires_grad &&
			   len(K_in.grad.data) > 0 {l.vec_add_simd(K_in.grad.data, dK, K_in.grad.data)}
			if V_in.requires_grad &&
			   len(V_in.grad.data) > 0 {l.vec_add_simd(V_in.grad.data, dV, V_in.grad.data)}
		case .MSELoss:
			// L = mean((pred - target)^2)
			// dL/dpred = 2 * (pred - target) / N
			pred_in := node.inputs[0]
			target_in := node.inputs[1]

			if pred_in.requires_grad {
				if len(node.grad.data) == 0 {
					fmt.println("WARNING: MSELoss node has empty gradient, skipping")
					continue
				}
				n := f64(len(pred_in.data.data))
				scalar_grad := node.grad.data[0] // Usually 1.0
				scale := 2.0 * scalar_grad / n

				// Re-calculate (pred - target) using temp allocator
				diff := make([]f64, len(pred_in.data.data), context.temp_allocator)
				l.vec_sub_simd(pred_in.data.data, target_in.data.data, diff)

				// ✅ SIMD Optimization: grad_pred += scale * diff
				l.axpy_simd(scale, diff, pred_in.grad.data)
				delete(diff, context.temp_allocator)
			}
		case .LayerNorm:
			input_in := node.inputs[0]
			gamma_in := node.inputs[1]
			beta_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			// ✅ FIX: Use stored metadata
			N := node.int_metadata[0]
			d_model := node.int_metadata[1]
			eps := 1e-5

			// ✅ FIX: Ensure gradient matrices are allocated before accumulating
			if input_in.requires_grad {tensor_ensure_grad(input_in)}
			if gamma_in.requires_grad {tensor_ensure_grad(gamma_in)}
			if beta_in.requires_grad {tensor_ensure_grad(beta_in)}

			// Temporary buffers
			centered := make([]f64, d_model, context.temp_allocator)
			x_hat := make([]f64, d_model, context.temp_allocator)
			dx_hat := make([]f64, d_model, context.temp_allocator)
			dx_row := make([]f64, d_model, context.temp_allocator)
			inv_std_vec := make([]f64, d_model, context.temp_allocator)
			defer {
				delete(centered, context.temp_allocator)
				delete(x_hat, context.temp_allocator)
				delete(dx_hat, context.temp_allocator)
				delete(dx_row, context.temp_allocator)
				delete(inv_std_vec, context.temp_allocator)
			}

			for i in 0 ..< N {
				row_start := i * d_model
				row := input_in.data.data[row_start:row_start + d_model]
				dy_row := node.grad.data[row_start:row_start + d_model]
				gamma_row := gamma_in.data.data

				// Recompute forward stats
				mean := l.sum_simd(row) / f64(d_model)
				for j in 0 ..< d_model {
					centered[j] = row[j] - mean
				}
				var := l.dot_simd(centered, centered) / f64(d_model)
				std := math.sqrt(var + eps)
				inv_std := 1.0 / std

				for j in 0 ..< d_model {inv_std_vec[j] = inv_std}
				l.vec_mul_simd(centered, inv_std_vec, x_hat)

				// dx_hat = dy * gamma (SIMD)
				l.vec_mul_simd(dy_row, gamma_row, dx_hat)

				// Stable O(N) backward formula
				sum_dx_hat := l.sum_simd(dx_hat)
				sum_dx_hat_x_hat := l.dot_simd(dx_hat, x_hat)

				// term1 = d_model * dx_hat
				d_model_f := f64(d_model)
				for j in 0 ..< d_model {dx_row[j] = d_model_f * dx_hat[j]}

				// term1 -= sum_dx_hat
				for j in 0 ..< d_model {dx_row[j] -= sum_dx_hat}

				// term1 -= x_hat * sum_dx_hat_x_hat
				for j in 0 ..< d_model {dx_row[j] -= x_hat[j] * sum_dx_hat_x_hat}

				// dx = dx_row / (d_model * std)
				inv_denom := 1.0 / (d_model_f * std)
				for j in 0 ..< d_model {dx_row[j] *= inv_denom}

				// ✅ Accumulate dx to input gradient
				if input_in.requires_grad && len(input_in.grad.data) > 0 {
					dx_grad_row := input_in.grad.data[row_start:row_start + d_model]
					l.vec_add_simd(dx_grad_row, dx_row, dx_grad_row)
				}

				// ✅ Accumulate dgamma: dgamma += dy * x_hat
				if gamma_in.requires_grad && len(gamma_in.grad.data) > 0 {
					dgamma_row := gamma_in.grad.data
					for j in 0 ..< d_model {
						dgamma_row[j] += dy_row[j] * x_hat[j]
					}
				}

				// ✅ Accumulate dbeta: dbeta += dy
				if beta_in.requires_grad && len(beta_in.grad.data) > 0 {
					dbeta_row := beta_in.grad.data
					l.vec_add_simd(dbeta_row, dy_row, dbeta_row)
				}
			}
		case .GRU:
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			w_ih_in := node.inputs[2]
			w_hh_in := node.inputs[3]
			bias_in := node.inputs[4]

			if len(node.grad.data) == 0 {continue}

			batch := x_in.shape[0]
			seq_len := x_in.shape[1]
			in_size := x_in.shape[2]
			hidden_size := w_ih_in.shape[1] / 3
			H := hidden_size
			H3 := 3 * H

			dx := make([]f64, len(x_in.data.data), context.temp_allocator)
			dw_ih := make([]f64, len(w_ih_in.data.data), context.temp_allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), context.temp_allocator)
			dbias := make([]f64, len(bias_in.data.data), context.temp_allocator)

			h_t := make([]f64, batch * H, context.temp_allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * H, context.temp_allocator)
			h_next := make([]f64, batch * H, context.temp_allocator)
			x_t := make([]f64, batch * in_size, context.temp_allocator)

			r_buf := make([]f64, batch * H, context.temp_allocator)
			z_buf := make([]f64, batch * H, context.temp_allocator)
			n_buf := make([]f64, batch * H, context.temp_allocator)
			n_hh_buf := make([]f64, batch * H, context.temp_allocator)

			dh_next := make([]f64, batch * H, context.temp_allocator)
			dh_prev := make([]f64, batch * H, context.temp_allocator)
			d_gate_ih := make([]f64, batch * H3, context.temp_allocator)
			d_gate_hh := make([]f64, batch * H3, context.temp_allocator)

			defer {
				delete(dx, context.temp_allocator); delete(dw_ih, context.temp_allocator)
				delete(dw_hh, context.temp_allocator); delete(dbias, context.temp_allocator)
				delete(h_t, context.temp_allocator); delete(h_prev, context.temp_allocator)
				delete(h_next, context.temp_allocator); delete(x_t, context.temp_allocator)
				delete(r_buf, context.temp_allocator); delete(z_buf, context.temp_allocator)
				delete(n_buf, context.temp_allocator); delete(n_hh_buf, context.temp_allocator)
				delete(dh_next, context.temp_allocator); delete(dh_prev, context.temp_allocator)
				delete(
					d_gate_ih,
					context.temp_allocator,
				); delete(d_gate_hh, context.temp_allocator)
			}

			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(h_prev, h_t)
				for b in 0 ..< batch {
					src := b * seq_len * in_size + s * in_size
					dst := b * in_size
					copy(x_t[dst:dst + in_size], x_in.data.data[src:src + in_size])
				}

				// Recompute forward
				_gru_step_forward(
					x_t,
					h_prev,
					w_ih_in.data.data,
					w_hh_in.data.data,
					bias_in.data.data,
					h_next,
					r_buf,
					z_buf,
					n_buf,
					batch,
					in_size,
					hidden_size,
					context.allocator,
				)

				// Recompute n_hh for backward
				h_prev_mat := l.Matrix(f64) {
					rows = batch,
					cols = H,
					data = h_prev,
				}
				w_hh_mat := l.Matrix(f64) {
					rows = H,
					cols = H3,
					data = w_hh_in.data.data,
				}
				gate_hh := l.matmul_dyn_simd(&h_prev_mat, &w_hh_mat, context.allocator)
				for b in 0 ..< batch {
					for i in 0 ..< H {
						n_hh_buf[b * H + i] = gate_hh.data[b * H3 + 2 * H + i]
					}
				}
				l.matrix_free(&gate_hh)

				copy(h_t, h_next)

				for b in 0 ..< batch {
					src := b * seq_len * H + s * H
					dst := b * H
					for i in 0 ..< H {
						dh_next[dst + i] = node.grad.data[src + i] + dh_prev[dst + i]
					}
				}

				// Backprop through h_t = (1 - z) * n + z * h_prev
				dz := make([]f64, batch * H, context.temp_allocator)
				dn := make([]f64, batch * H, context.temp_allocator)
				dr_from_n := make([]f64, batch * H, context.temp_allocator)

				for i in 0 ..< batch * H {
					dh_prev[i] += dh_next[i] * z_buf[i]
					dz[i] = dh_next[i] * (h_prev[i] - n_buf[i])
					dn[i] = dh_next[i] * (1.0 - z_buf[i])
				}

				// Backprop through n = tanh(n_ih + r * n_hh)
				d_n_pre := make([]f64, batch * H, context.temp_allocator)
				for i in 0 ..< batch * H {
					d_n_pre[i] = dn[i] * (1.0 - n_buf[i] * n_buf[i])
					dr_from_n[i] = d_n_pre[i] * n_hh_buf[i]
				}

				// Backprop through sigmoid gates
				dr_pre := make([]f64, batch * H, context.temp_allocator)
				dz_pre := make([]f64, batch * H, context.temp_allocator)
				for i in 0 ..< batch * H {
					dr_pre[i] = dr_from_n[i] * r_buf[i] * (1.0 - r_buf[i])
					dz_pre[i] = dz[i] * z_buf[i] * (1.0 - z_buf[i])
				}

				// Pack into d_gate matrices
				for b in 0 ..< batch {
					for i in 0 ..< H {
						idx_r := b * H3 + i
						idx_z := b * H3 + H + i
						idx_n := b * H3 + 2 * H + i

						d_gate_ih[idx_r] = dr_pre[b * H + i]
						d_gate_ih[idx_z] = dz_pre[b * H + i]
						d_gate_ih[idx_n] = d_n_pre[b * H + i]

						d_gate_hh[idx_r] = dr_pre[b * H + i]
						d_gate_hh[idx_z] = dz_pre[b * H + i]
						d_gate_hh[idx_n] = d_n_pre[b * H + i] * r_buf[b * H + i]
					}
				}

				delete(dz, context.temp_allocator); delete(dn, context.temp_allocator)
				delete(dr_from_n, context.temp_allocator); delete(d_n_pre, context.temp_allocator)
				delete(dr_pre, context.temp_allocator); delete(dz_pre, context.temp_allocator)

				for b in 0 ..< batch {
					for i in 0 ..< H3 {dbias[i] += d_gate_ih[b * H3 + i]}
				}

				// ✅ SIMD Matmul Backward
				x_mat := l.Matrix(f64) {
					rows = batch,
					cols = in_size,
					data = x_t,
				}
				h_prev_mat_bwd := l.Matrix(f64) {
					rows = batch,
					cols = H,
					data = h_prev,
				}
				d_gate_ih_mat := l.Matrix(f64) {
					rows = batch,
					cols = H3,
					data = d_gate_ih,
				}
				d_gate_hh_mat := l.Matrix(f64) {
					rows = batch,
					cols = H3,
					data = d_gate_hh,
				}

				x_t_t := _matrix_transpose(x_mat, context.allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &d_gate_ih_mat, context.allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t); l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat_bwd, context.allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &d_gate_hh_mat, context.allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t); l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, context.allocator)
				dx_t_res := l.matmul_dyn_simd(&d_gate_ih_mat, &w_ih_t, context.allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {dx[dst + i] += dx_t_res.data[src + i]}
				}
				l.matrix_free(&w_ih_t); l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, context.allocator)
				dh_prev_res := l.matmul_dyn_simd(&d_gate_hh_mat, &w_hh_t, context.allocator)
				for i in 0 ..< len(dh_prev) {dh_prev[i] = dh_prev_res.data[i]}
				l.matrix_free(&w_hh_t); l.matrix_free(&dh_prev_res)
			}

			if x_in.requires_grad &&
			   len(x_in.grad.data) > 0 {l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)}
			if h_0_in.requires_grad &&
			   len(h_0_in.grad.data) >
				   0 {l.vec_add_simd(h_0_in.grad.data, dh_prev, h_0_in.grad.data)}
			if w_ih_in.requires_grad &&
			   len(w_ih_in.grad.data) >
				   0 {l.vec_add_simd(w_ih_in.grad.data, dw_ih, w_ih_in.grad.data)}
			if w_hh_in.requires_grad &&
			   len(w_hh_in.grad.data) >
				   0 {l.vec_add_simd(w_hh_in.grad.data, dw_hh, w_hh_in.grad.data)}
			if bias_in.requires_grad &&
			   len(bias_in.grad.data) >
				   0 {l.vec_add_simd(bias_in.grad.data, dbias, bias_in.grad.data)}
		case .LSTM:
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			c_0_in := node.inputs[2]
			w_ih_in := node.inputs[3]
			w_hh_in := node.inputs[4]
			bias_in := node.inputs[5]

			if len(node.grad.data) == 0 {continue}

			batch := x_in.shape[0]
			seq_len := x_in.shape[1]
			in_size := x_in.shape[2]
			hidden_size := w_ih_in.shape[1] / 4
			H := hidden_size
			H4 := 4 * H

			dx := make([]f64, len(x_in.data.data), context.temp_allocator)
			dw_ih := make([]f64, len(w_ih_in.data.data), context.temp_allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), context.temp_allocator)
			dbias := make([]f64, len(bias_in.data.data), context.temp_allocator)

			h_t := make([]f64, batch * H, context.temp_allocator)
			c_t := make([]f64, batch * H, context.temp_allocator)
			copy(h_t, h_0_in.data.data)
			copy(c_t, c_0_in.data.data)

			h_prev := make([]f64, batch * H, context.temp_allocator)
			c_prev := make([]f64, batch * H, context.temp_allocator)
			h_next := make([]f64, batch * H, context.temp_allocator)
			c_next := make([]f64, batch * H, context.temp_allocator)
			x_t := make([]f64, batch * in_size, context.temp_allocator)

			i_buf := make([]f64, batch * H, context.temp_allocator)
			f_buf := make([]f64, batch * H, context.temp_allocator)
			g_buf := make([]f64, batch * H, context.temp_allocator)
			o_buf := make([]f64, batch * H, context.temp_allocator)

			dh_next := make([]f64, batch * H, context.temp_allocator)
			dc_next := make([]f64, batch * H, context.temp_allocator)
			dh_prev := make([]f64, batch * H, context.temp_allocator)
			dc_prev := make([]f64, batch * H, context.temp_allocator)
			d_gate_ih := make([]f64, batch * H4, context.temp_allocator)
			d_gate_hh := make([]f64, batch * H4, context.temp_allocator)

			defer {
				delete(dx, context.temp_allocator); delete(dw_ih, context.temp_allocator)
				delete(dw_hh, context.temp_allocator); delete(dbias, context.temp_allocator)
				delete(h_t, context.temp_allocator); delete(c_t, context.temp_allocator)
				delete(h_prev, context.temp_allocator); delete(c_prev, context.temp_allocator)
				delete(h_next, context.temp_allocator); delete(c_next, context.temp_allocator)
				delete(x_t, context.temp_allocator)
				delete(i_buf, context.temp_allocator); delete(f_buf, context.temp_allocator)
				delete(g_buf, context.temp_allocator); delete(o_buf, context.temp_allocator)
				delete(dh_next, context.temp_allocator); delete(dc_next, context.temp_allocator)
				delete(dh_prev, context.temp_allocator); delete(dc_prev, context.temp_allocator)
				delete(
					d_gate_ih,
					context.temp_allocator,
				); delete(d_gate_hh, context.temp_allocator)
			}

			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(h_prev, h_t)
				copy(c_prev, c_t)

				for b in 0 ..< batch {
					src := b * seq_len * in_size + s * in_size
					dst := b * in_size
					copy(x_t[dst:dst + in_size], x_in.data.data[src:src + in_size])
				}

				// Recompute forward
				_lstm_step_forward(
					x_t,
					h_prev,
					c_prev,
					w_ih_in.data.data,
					w_hh_in.data.data,
					bias_in.data.data,
					h_next,
					c_next,
					i_buf,
					f_buf,
					g_buf,
					o_buf,
					batch,
					in_size,
					hidden_size,
					context.temp_allocator,
				)
				copy(h_t, h_next)
				copy(c_t, c_next)

				// Accumulate gradients from output and future timestep
				for b in 0 ..< batch {
					src := b * seq_len * H + s * H
					dst := b * H
					for i in 0 ..< H {
						dh_next[dst + i] = node.grad.data[src + i] + dh_prev[dst + i]
					}
				}

				// Backprop through h_t = o * tanh(c_t)
				tanh_c := make([]f64, batch * H, context.temp_allocator)
				dov := make([]f64, batch * H, context.temp_allocator)
				dc := make([]f64, batch * H, context.temp_allocator)

				for i in 0 ..< batch * H {
					tanh_c[i] = math.tanh(c_t[i])
					dov[i] = dh_next[i] * tanh_c[i]
					dc[i] = dh_next[i] * o_buf[i] * (1.0 - tanh_c[i] * tanh_c[i]) + dc_next[i]
				}

				// Backprop through c_t = f * c_prev + i * g
				df := make([]f64, batch * H, context.temp_allocator)
				di := make([]f64, batch * H, context.temp_allocator)
				dg := make([]f64, batch * H, context.temp_allocator)

				for i in 0 ..< batch * H {
					df[i] = dc[i] * c_prev[i]
					di[i] = dc[i] * g_buf[i]
					dg[i] = dc[i] * i_buf[i]
					dc_prev[i] = dc[i] * f_buf[i]
				}

				// Backprop through activation functions
				di_pre := make([]f64, batch * H, context.temp_allocator)
				df_pre := make([]f64, batch * H, context.temp_allocator)
				dg_pre := make([]f64, batch * H, context.temp_allocator)
				do_pre := make([]f64, batch * H, context.temp_allocator)

				for i in 0 ..< batch * H {
					di_pre[i] = di[i] * i_buf[i] * (1.0 - i_buf[i])
					df_pre[i] = df[i] * f_buf[i] * (1.0 - f_buf[i])
					dg_pre[i] = dg[i] * (1.0 - g_buf[i] * g_buf[i])
					do_pre[i] = dov[i] * o_buf[i] * (1.0 - o_buf[i])
				}

				// Pack into d_gate matrices
				for b in 0 ..< batch {
					for i in 0 ..< H {
						idx_i := b * H4 + i
						idx_f := b * H4 + H + i
						idx_g := b * H4 + 2 * H + i
						idx_o := b * H4 + 3 * H + i

						d_gate_ih[idx_i] = di_pre[b * H + i]
						d_gate_ih[idx_f] = df_pre[b * H + i]
						d_gate_ih[idx_g] = dg_pre[b * H + i]
						d_gate_ih[idx_o] = do_pre[b * H + i]

						d_gate_hh[idx_i] = di_pre[b * H + i]
						d_gate_hh[idx_f] = df_pre[b * H + i]
						d_gate_hh[idx_g] = dg_pre[b * H + i]
						d_gate_hh[idx_o] = do_pre[b * H + i]
					}
				}

				delete(tanh_c, context.temp_allocator)
				delete(dov, context.temp_allocator)
				delete(dc, context.temp_allocator)
				delete(df, context.temp_allocator)
				delete(di, context.temp_allocator); delete(dg, context.temp_allocator)
				delete(di_pre, context.temp_allocator); delete(df_pre, context.temp_allocator)
				delete(dg_pre, context.temp_allocator); delete(do_pre, context.temp_allocator)

				// Accumulate bias gradients
				for b in 0 ..< batch {
					for i in 0 ..< H4 {dbias[i] += d_gate_ih[b * H4 + i]}
				}

				// ✅ SIMD Matmul Backward
				x_mat := l.Matrix(f64) {
					rows = batch,
					cols = in_size,
					data = x_t,
				}
				h_prev_mat_bwd := l.Matrix(f64) {
					rows = batch,
					cols = H,
					data = h_prev,
				}
				d_gate_ih_mat := l.Matrix(f64) {
					rows = batch,
					cols = H4,
					data = d_gate_ih,
				}
				d_gate_hh_mat := l.Matrix(f64) {
					rows = batch,
					cols = H4,
					data = d_gate_hh,
				}

				x_t_t := _matrix_transpose(x_mat, context.temp_allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &d_gate_ih_mat, context.temp_allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t); l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat_bwd, context.temp_allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &d_gate_hh_mat, context.temp_allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t); l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, context.temp_allocator)
				dx_t_res := l.matmul_dyn_simd(&d_gate_ih_mat, &w_ih_t, context.temp_allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {dx[dst + i] += dx_t_res.data[src + i]}
				}
				l.matrix_free(&w_ih_t); l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, context.temp_allocator)
				dh_prev_res := l.matmul_dyn_simd(&d_gate_hh_mat, &w_hh_t, context.temp_allocator)
				for i in 0 ..< len(dh_prev) {dh_prev[i] = dh_prev_res.data[i]}
				l.matrix_free(&w_hh_t); l.matrix_free(&dh_prev_res)

				copy(dc_next, dc_prev)
			}

			if x_in.requires_grad &&
			   len(x_in.grad.data) > 0 {l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)}
			if h_0_in.requires_grad &&
			   len(h_0_in.grad.data) >
				   0 {l.vec_add_simd(h_0_in.grad.data, dh_prev, h_0_in.grad.data)}
			if c_0_in.requires_grad &&
			   len(c_0_in.grad.data) >
				   0 {l.vec_add_simd(c_0_in.grad.data, dc_prev, c_0_in.grad.data)}
			if w_ih_in.requires_grad &&
			   len(w_ih_in.grad.data) >
				   0 {l.vec_add_simd(w_ih_in.grad.data, dw_ih, w_ih_in.grad.data)}
			if w_hh_in.requires_grad &&
			   len(w_hh_in.grad.data) >
				   0 {l.vec_add_simd(w_hh_in.grad.data, dw_hh, w_hh_in.grad.data)}
			if bias_in.requires_grad &&
			   len(bias_in.grad.data) >
				   0 {l.vec_add_simd(bias_in.grad.data, dbias, bias_in.grad.data)}
		case .CrossEntropy:
			// The magical simplified gradient: grad = (softmax_prob - target_one_hot) / N
			logits_in := node.inputs[0]
			N := logits_in.data.rows
			C := logits_in.data.cols
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: CrossEntropy node has empty gradient, skipping")
				continue
			}
			scalar_grad := node.grad.data[0] // Usually 1.0
			if len(node.int_metadata) < N {
				fmt.printf(
					"ERROR: CrossEntropy int_metadata length %d < N %d\n",
					len(node.int_metadata),
					N,
				)
				continue
			}
			for i in 0 ..< N {
				// Recompute softmax for stability
				max_val := -math.F64_MAX
				for j in 0 ..< C {
					v := logits_in.data.data[i * C + j]
					if v > max_val {max_val = v}
				}

				sum_exp := 0.0
				for j in 0 ..< C {
					sum_exp += math.exp(logits_in.data.data[i * C + j] - max_val)
				}

				target_class := node.int_metadata[i]

				for j in 0 ..< C {
					prob := math.exp(logits_in.data.data[i * C + j] - max_val) / sum_exp
					target_val := 0.0
					if j == target_class {target_val = 1.0}

					// Apply chain rule
					logit_grad := (prob - target_val) * scalar_grad / f64(N)
					logits_in.grad.data[i * C + j] += logit_grad
				}
			}
		case .Dropout:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// ✅ SIMD Optimization: grad_a += node.grad * mask
				// vec_fma_inplace_simd does: c += a * b
				l.vec_fma_inplace_simd(node.grad.data, node.dropout_mask, a_in.grad.data)
			}
		case .Flatten:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// Just reshape the gradient back to 4D
				copy(a_in.grad.data, node.grad.data)
			}
		case .MaxPool2d:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// ✅ ADD: Bounds check
				if len(node.int_metadata) != len(node.grad.data) {
					fmt.printf(
						"ERROR: MaxPool2d int_metadata length %d != grad length %d\n",
						len(node.int_metadata),
						len(node.grad.data),
					)
					continue
				}

				for i in 0 ..< len(node.grad.data) {
					idx := node.int_metadata[i]
					// ✅ ADD: Bounds check for idx
					if idx >= len(a_in.grad.data) {
						fmt.printf(
							"ERROR: MaxPool2d idx %d >= grad length %d\n",
							idx,
							len(a_in.grad.data),
						)
						continue
					}
					a_in.grad.data[idx] += node.grad.data[i]
				}
			}
		case .BatchNorm2d:
			input_in := node.inputs[0]
			weight_in := node.inputs[1]
			bias_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			N := input_in.shape[0]
			C := input_in.shape[1]
			H := input_in.shape[2]
			W := input_in.shape[3]
			N_hw := f64(N * H * W)
			channel_size := N * H * W
			eps := 1e-5

			// ✅ Temporary contiguous buffers for SIMD operations
			x_buf := make([]f64, channel_size, context.temp_allocator)
			dout_buf := make([]f64, channel_size, context.temp_allocator)
			defer delete(x_buf, context.temp_allocator)
			defer delete(dout_buf, context.temp_allocator)

			for c in 0 ..< C {
				// 1. Extract channel data and grad to contiguous buffers
				for n in 0 ..< N {
					for h in 0 ..< H {
						for w in 0 ..< W {
							idx := n * (C * H * W) + c * (H * W) + h * W + w
							buf_idx := n * (H * W) + h * W + w
							x_buf[buf_idx] = input_in.data.data[idx]
							dout_buf[buf_idx] = node.grad.data[idx]
						}
					}
				}

				// 2. Compute mean (SIMD sum)
				mean := l.sum_simd(x_buf) / N_hw

				// 3. Compute variance (SIMD)
				mean_vec := make([]f64, channel_size, context.temp_allocator)
				for i in 0 ..< channel_size {mean_vec[i] = mean}
				centered := make([]f64, channel_size, context.temp_allocator)
				l.vec_sub_simd(x_buf, mean_vec, centered)
				delete(mean_vec, context.temp_allocator)

				var := l.dot_simd(centered, centered) / N_hw
				std := math.sqrt(var + eps)
				inv_std := 1.0 / std
				gamma := weight_in.data.data[c]

				// 4. Compute x_hat (SIMD mul)
				x_hat := make([]f64, channel_size, context.temp_allocator)
				inv_std_vec := make([]f64, channel_size, context.temp_allocator)
				for i in 0 ..< channel_size {inv_std_vec[i] = inv_std}
				l.vec_mul_simd(centered, inv_std_vec, x_hat)
				delete(inv_std_vec, context.temp_allocator)

				// 5. Compute weight gradient: dgamma = dot(dout, x_hat) (SIMD)
				if weight_in.requires_grad && len(weight_in.grad.data) > 0 {
					dgamma := l.dot_simd(dout_buf, x_hat)
					weight_in.grad.data[c] += dgamma
				}

				// 6. Compute bias gradient: dbeta = sum(dout) (SIMD)
				if bias_in.requires_grad && len(bias_in.grad.data) > 0 {
					dbeta := l.sum_simd(dout_buf)
					bias_in.grad.data[c] += dbeta
				}

				// 7. Compute input gradient (SIMD)
				if input_in.requires_grad && len(input_in.grad.data) > 0 {
					// dout_gamma = dout * gamma (SIMD)
					dout_gamma := make([]f64, channel_size, context.temp_allocator)
					gamma_vec := make([]f64, channel_size, context.temp_allocator)
					for i in 0 ..< channel_size {gamma_vec[i] = gamma}
					l.vec_mul_simd(dout_buf, gamma_vec, dout_gamma)
					delete(gamma_vec, context.temp_allocator)

					// Precompute sums (SIMD)
					sum_dout_gamma := l.sum_simd(dout_gamma)
					sum_dout_gamma_x_hat := l.dot_simd(dout_gamma, x_hat)

					// dx = (1 / (N_hw * std)) * (N_hw * dout_gamma - sum_dout_gamma - x_hat * sum_dout_gamma_x_hat)

					// term1 = N_hw * dout_gamma (SIMD)
					term1 := make([]f64, channel_size, context.temp_allocator)
					n_hw_vec := make([]f64, channel_size, context.temp_allocator)
					for i in 0 ..< channel_size {n_hw_vec[i] = N_hw}
					l.vec_mul_simd(dout_gamma, n_hw_vec, term1)
					delete(n_hw_vec, context.temp_allocator)
					delete(dout_gamma, context.temp_allocator)

					// term2 = x_hat * sum_dout_gamma_x_hat (SIMD)
					term2 := make([]f64, channel_size, context.temp_allocator)
					s2_vec := make([]f64, channel_size, context.temp_allocator)
					for i in 0 ..< channel_size {s2_vec[i] = sum_dout_gamma_x_hat}
					l.vec_mul_simd(x_hat, s2_vec, term2)
					delete(s2_vec, context.temp_allocator)
					delete(x_hat, context.temp_allocator)

					// combined = term1 - sum_dout_gamma (SIMD)
					sum_dg_vec := make([]f64, channel_size, context.temp_allocator)
					for i in 0 ..< channel_size {sum_dg_vec[i] = sum_dout_gamma}
					l.vec_sub_simd(term1, sum_dg_vec, term1)
					delete(sum_dg_vec, context.temp_allocator)

					// combined = combined - term2 (SIMD)
					l.vec_sub_simd(term1, term2, term1)
					delete(term2, context.temp_allocator)

					// dx = combined / (N_hw * std) (SIMD)
					inv_denom := 1.0 / (N_hw * std)
					inv_denom_vec := make([]f64, channel_size, context.temp_allocator)
					for i in 0 ..< channel_size {inv_denom_vec[i] = inv_denom}
					l.vec_mul_simd(term1, inv_denom_vec, term1)
					delete(inv_denom_vec, context.temp_allocator)

					// Write dx back to input gradient (de-interleave)
					for n in 0 ..< N {
						for h in 0 ..< H {
							for w in 0 ..< W {
								idx := n * (C * H * W) + c * (H * W) + h * W + w
								buf_idx := n * (H * W) + h * W + w
								input_in.grad.data[idx] += term1[buf_idx]
							}
						}
					}
					delete(term1, context.temp_allocator)
				} else {
					delete(x_hat, context.temp_allocator)
				}

				delete(centered, context.temp_allocator)
			}
		case .RNN:
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			w_ih_in := node.inputs[2]
			w_hh_in := node.inputs[3]
			bias_in := node.inputs[4]

			if len(node.grad.data) == 0 {continue}

			batch := x_in.shape[0]
			seq_len := x_in.shape[1]
			in_size := x_in.shape[2]
			hidden_size := w_ih_in.shape[1]

			dx := make([]f64, len(x_in.data.data), context.temp_allocator)
			dh_0 := make([]f64, len(h_0_in.data.data), context.temp_allocator) // ✅ FIX: typo corrected
			dw_ih := make([]f64, len(w_ih_in.data.data), context.temp_allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), context.temp_allocator)
			dbias := make([]f64, len(bias_in.data.data), context.temp_allocator)

			h_t := make([]f64, batch * hidden_size, context.temp_allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * hidden_size, context.temp_allocator)
			h_next := make([]f64, batch * hidden_size, context.temp_allocator)
			x_t := make([]f64, batch * in_size, context.temp_allocator)

			dh_next := make([]f64, batch * hidden_size, context.temp_allocator)
			dx_t := make([]f64, batch * in_size, context.temp_allocator)
			dh_prev := make([]f64, batch * hidden_size, context.temp_allocator)
			dw_ih_step := make([]f64, in_size * hidden_size, context.temp_allocator)
			dw_hh_step := make([]f64, hidden_size * hidden_size, context.temp_allocator)
			dbias_step := make([]f64, hidden_size, context.temp_allocator)

			defer {
				delete(dx, context.temp_allocator)
				delete(dh_0, context.temp_allocator)
				delete(dw_ih, context.temp_allocator)
				delete(dw_hh, context.temp_allocator)
				delete(dbias, context.temp_allocator)
				delete(h_t, context.temp_allocator)
				delete(h_prev, context.temp_allocator)
				delete(h_next, context.temp_allocator)
				delete(x_t, context.temp_allocator)
				delete(dh_next, context.temp_allocator)
				delete(dx_t, context.temp_allocator)
				delete(dh_prev, context.temp_allocator)
				delete(dw_ih_step, context.temp_allocator)
				delete(dw_hh_step, context.temp_allocator)
				delete(dbias_step, context.temp_allocator)
			}

			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(h_prev, h_t)
				for b in 0 ..< batch {
					src := b * seq_len * in_size + s * in_size
					dst := b * in_size
					copy(x_t[dst:dst + in_size], x_in.data.data[src:src + in_size])
				}
				_rnn_step_forward(
					x_t,
					h_prev,
					w_ih_in.data.data,
					w_hh_in.data.data,
					bias_in.data.data,
					h_next,
					batch,
					in_size,
					hidden_size,
					context.temp_allocator,
				)
				copy(h_t, h_next)

				for b in 0 ..< batch {
					src := b * seq_len * hidden_size + s * hidden_size
					dst := b * hidden_size
					for i in 0 ..< hidden_size {
						dh_next[dst + i] = node.grad.data[src + i] + dh_prev[dst + i]
					}
				}

				for i in 0 ..< batch * hidden_size {
					one_minus_h2 := 1.0 - h_next[i] * h_next[i]
					dh_next[i] *= one_minus_h2
				}

				for b in 0 ..< batch {
					for i in 0 ..< hidden_size {
						dbias_step[i] += dh_next[b * hidden_size + i]
					}
				}

				// ✅ FIX: Use '=' for struct initialization
				x_mat := l.Matrix(f64) {
					rows = batch,
					cols = in_size,
					data = x_t,
				}
				h_prev_mat := l.Matrix(f64) {
					rows = batch,
					cols = hidden_size,
					data = h_prev,
				}
				dh_mat := l.Matrix(f64) {
					rows = batch,
					cols = hidden_size,
					data = dh_next,
				}

				x_t_t := _matrix_transpose(x_mat, context.temp_allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &dh_mat, context.temp_allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t)
				l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat, context.temp_allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &dh_mat, context.temp_allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t)
				l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, context.temp_allocator)
				dx_t_res := l.matmul_dyn_simd(&dh_mat, &w_ih_t, context.temp_allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {
						dx[dst + i] += dx_t_res.data[src + i]
					}
				}
				l.matrix_free(&w_ih_t)
				l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, context.temp_allocator)
				dh_prev_res := l.matmul_dyn_simd(&dh_mat, &w_hh_t, context.temp_allocator)
				for i in 0 ..< len(dh_prev) {
					dh_prev[i] = dh_prev_res.data[i]
				}
				l.matrix_free(&w_hh_t)
				l.matrix_free(&dh_prev_res)
			}

			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)
			}
			if h_0_in.requires_grad && len(h_0_in.grad.data) > 0 {
				l.vec_add_simd(h_0_in.grad.data, dh_prev, h_0_in.grad.data)
			}
			if w_ih_in.requires_grad && len(w_ih_in.grad.data) > 0 {
				l.vec_add_simd(w_ih_in.grad.data, dw_ih, w_ih_in.grad.data)
			}
			if w_hh_in.requires_grad && len(w_hh_in.grad.data) > 0 {
				l.vec_add_simd(w_hh_in.grad.data, dw_hh, w_hh_in.grad.data)
			}
			if bias_in.requires_grad && len(bias_in.grad.data) > 0 {
				l.vec_add_simd(bias_in.grad.data, dbias, bias_in.grad.data)
			}
		case .AvgPool2d:
			// Distribute gradient equally to all elements in the pooling window
			a_in := node.inputs[0]
			if a_in.requires_grad {
				N := a_in.shape[0]
				C := a_in.shape[1]
				H := a_in.shape[2]
				W := a_in.shape[3]
				kH := node.pool_params.kH
				kW := node.pool_params.kW
				stride := node.pool_params.stride
				out_h := node.shape[2]
				out_w := node.shape[3]
				pool_size := f64(kH * kW)

				for n in 0 ..< N {
					for c in 0 ..< C {
						for oh in 0 ..< out_h {
							for ow in 0 ..< out_w {
								out_idx :=
									n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
								grad := node.grad.data[out_idx] / pool_size

								for kh in 0 ..< kH {
									for kw in 0 ..< kW {
										ih := oh * stride + kh
										iw := ow * stride + kw
										in_idx := n * (C * H * W) + c * (H * W) + ih * W + iw
										a_in.grad.data[in_idx] += grad
									}
								}
							}
						}
					}
				}
			}

		case .Conv2d:
			input_in := node.inputs[0]
			weight_in := node.inputs[1]
			bias_in: ^Tensor = nil
			if len(node.inputs) > 2 {bias_in = node.inputs[2]}

			// ✅ ADD: Comprehensive checks for empty data
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: Conv2d node has empty gradient, skipping")
				continue
			}
			if len(input_in.data.data) == 0 {
				fmt.println("WARNING: Conv2d input has empty data, skipping")
				continue
			}
			if len(weight_in.data.data) == 0 {
				fmt.println("WARNING: Conv2d weight has empty data, skipping")
				continue
			}

			N := input_in.shape[0]
			C_in := input_in.shape[1]
			H := input_in.shape[2]
			W := input_in.shape[3]

			C_out := weight_in.shape[0]
			kH := node.conv_params.kH
			kW := node.conv_params.kW
			stride := node.conv_params.stride
			pad := node.conv_params.pad

			out_h := node.shape[2]
			out_w := node.shape[3]
			col_h := out_h * out_w
			col_w := C_in * kH * kW

			// ✅ ADD: Skip if dimensions are 0
			if col_h == 0 || col_w == 0 || C_out == 0 || N == 0 {
				fmt.printf(
					"WARNING: Conv2d has zero dimensions (col_h=%d, col_w=%d, C_out=%d, N=%d), skipping\n",
					col_h,
					col_w,
					C_out,
					N,
				)
				continue
			}

			col, _, _ := _im2col(
				input_in.data.data,
				N,
				C_in,
				H,
				W,
				kH,
				kW,
				stride,
				pad,
				context.allocator,
			)

			// ✅ ADD: Check if col is empty
			if len(col) == 0 {
				fmt.println("WARNING: Conv2d im2col returned empty slice, skipping")
				continue
			}

			if input_in.requires_grad {
				tensor_ensure_grad(input_in)
				if len(input_in.grad.data) > 0 {
					grad_input_col := make([]f64, N * col_w * col_h, context.allocator)
					weight_2d_t := _matrix_transpose(weight_in.data, context.allocator)

					for n in 0 ..< N {
						grad_out_start := n * C_out * col_h
						if grad_out_start + C_out * col_h > len(node.grad.data) {
							fmt.printf(
								"WARNING: Conv2d grad_out slice out of bounds: %d:%d > %d\n",
								grad_out_start,
								grad_out_start + C_out * col_h,
								len(node.grad.data),
							)
							continue
						}

						grad_out_2d := l.Matrix(f64) {
							rows = C_out,
							cols = col_h,
							data = node.grad.data[grad_out_start:grad_out_start + C_out * col_h],
						}

						// ✅ ADD: Check if grad_out_2d is empty
						if len(grad_out_2d.data) == 0 {
							fmt.println("WARNING: Conv2d grad_out_2d is empty, skipping")
							continue
						}

						grad_col_2d := l.matmul_dyn_simd(
							&weight_2d_t,
							&grad_out_2d,
							context.allocator,
						)

						col_start := n * col_w * col_h
						if col_start + col_w * col_h <= len(grad_input_col) {
							copy(
								grad_input_col[col_start:col_start + col_w * col_h],
								grad_col_2d.data,
							)
						}
						l.matrix_free(&grad_col_2d)
					}

					l.matrix_free(&weight_2d_t)

					grad_input := _col2im(
						grad_input_col,
						N,
						C_in,
						H,
						W,
						kH,
						kW,
						stride,
						pad,
						out_h,
						out_w,
						context.allocator,
					)

					for i in 0 ..< len(input_in.grad.data) {
						if i < len(grad_input) {
							input_in.grad.data[i] += grad_input[i]
						}
					}

					delete(grad_input_col, context.allocator)
					delete(grad_input, context.allocator)
				}
			}

			if weight_in.requires_grad {
				tensor_ensure_grad(weight_in)
				if len(weight_in.grad.data) > 0 {
					grad_weight := make([]f64, len(weight_in.data.data), context.allocator)

					for n in 0 ..< N {
						grad_out_start := n * C_out * col_h
						if grad_out_start + C_out * col_h > len(node.grad.data) {continue}

						grad_out_2d := l.Matrix(f64) {
							rows = C_out,
							cols = col_h,
							data = node.grad.data[grad_out_start:grad_out_start + C_out * col_h],
						}

						col_start := n * col_w * col_h
						if col_start + col_w * col_h > len(col) {
							fmt.printf(
								"WARNING: Conv2d col slice out of bounds: %d:%d > %d\n",
								col_start,
								col_start + col_w * col_h,
								len(col),
							)
							continue
						}

						col_2d := l.Matrix(f64) {
							rows = col_w,
							cols = col_h,
							data = col[col_start:col_start + col_w * col_h],
						}

						col_2d_t := _matrix_transpose(col_2d, context.allocator)
						grad_weight_2d := l.matmul_dyn_simd(
							&grad_out_2d,
							&col_2d_t,
							context.allocator,
						)

						for i in 0 ..< len(grad_weight) {
							if i < len(grad_weight_2d.data) {
								grad_weight[i] += grad_weight_2d.data[i]
							}
						}

						l.matrix_free(&col_2d_t)
						l.matrix_free(&grad_weight_2d)
					}

					for i in 0 ..< len(weight_in.grad.data) {
						weight_in.grad.data[i] += grad_weight[i]
					}

					delete(grad_weight, context.allocator)
				}
			}

			if bias_in != nil && bias_in.requires_grad {
				tensor_ensure_grad(bias_in)
				if len(bias_in.grad.data) > 0 {
					for n in 0 ..< N {
						for c in 0 ..< C_out {
							offset := n * C_out * col_h + c * col_h
							if offset + col_h > len(node.grad.data) {continue}
							sum := 0.0
							for i in 0 ..< col_h {sum += node.grad.data[offset + i]}
							bias_in.grad.data[c] += sum
						}
					}
				}
			}

			delete(col, context.allocator)

		// We don't calculate gradients for the target data.
		case .None:
		// Leaf node, nothing to do
		}
	}
}
// tensor_max_pool2d performs max pooling.
// input: (N, C, H, W) -> output: (N, C, H_out, W_out)
tensor_max_pool2d :: proc(input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	out_h := (H - kH) / stride + 1
	out_w := (W - kW) / stride + 1

	out_data := l.matrix_new(f64, 1, N * C * out_h * out_w, input.allocator)

	// We need to store the indices of the max values for the backward pass
	indices := make([]int, N * C * out_h * out_w, input.allocator)

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

	out := tensor_new(out_data, input.requires_grad, input.allocator)
	out.shape = [4]int{N, C, out_h, out_w}

	if out.requires_grad {
		out.op = .MaxPool2d
		append(&out.inputs, input)
		out.pool_params = PoolParams{kH, kW, stride, 0}

		// Save indices to int_metadata for backward pass
		for idx in indices {
			append(&out.int_metadata, idx)
		}
	}

	delete(indices, input.allocator)
	return out
}
// tensor_avg_pool2d performs average pooling.
tensor_avg_pool2d :: proc(input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	out_h := (H - kH) / stride + 1
	out_w := (W - kW) / stride + 1

	out_data := l.matrix_new(f64, 1, N * C * out_h * out_w, input.allocator)
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

	out := tensor_new(out_data, input.requires_grad, input.allocator)
	out.shape = [4]int{N, C, out_h, out_w}

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
) -> ^Tensor {
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
	col, _, _ := _im2col(input.data.data, N, C_in, H, W, kH, kW, stride, pad, context.allocator)

	out_data := l.matrix_new(f64, 1, N * C_out * out_h * out_w, input.allocator)
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

		// ✅ FIX: Use context.allocator for temporary matmul
		out_2d := l.matmul_dyn_simd(&weight_2d, &col_2d, context.allocator)

		out_start := n * C_out * col_h
		copy(out_data.data[out_start:out_start + C_out * col_h], out_2d.data)

		// ✅ FIX: matrix_free defaults to context.allocator, which now matches!
		l.matrix_free(&out_2d)
	}

	// ✅ FIX: Delete with context.allocator to match allocation
	delete(col, context.allocator)

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

	out := tensor_new(out_data, input.requires_grad || weight.requires_grad, input.allocator)
	out.shape = [4]int{N, C_out, out_h, out_w}

	if out.requires_grad {
		out.op = .Conv2d
		append(&out.inputs, input)
		append(&out.inputs, weight)
		if bias != nil {append(&out.inputs, bias)}
		out.conv_params = ConvParams{kH, kW, stride, pad}
	}

	return out
}

// Internal helper: Transpose a matrix (needed for MatMul backward pass)
_matrix_transpose :: proc(m: l.Matrix(f64), alloc: mem.Allocator) -> l.Matrix(f64) {
	out := l.matrix_new(f64, m.cols, m.rows, alloc)
	for r in 0 ..< m.rows {
		for c in 0 ..< m.cols {
			out.data[c * out.cols + r] = m.data[r * m.cols + c]
		}
	}
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
			temp_out := l.matmul_dyn_simd(&a_view, &b.data, a.allocator)

			// Flatten the result back to 1 x (N * out_features) to match framework conventions
			out_data := l.matrix_new(f64, 1, N * out_features, a.allocator)
			copy(out_data.data, temp_out.data)
			l.matrix_free(&temp_out)

			out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)
			out.shape = a.shape
			out.shape[2] = out_features
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

	out_data := l.matmul_dyn_simd(&a.data, &b.data, a.allocator)
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)
	out.shape = [4]int{a.data.rows, b.data.cols, 1, 1}
	if out.requires_grad {
		out.op = .MatMul
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}
// _tensor_free_graph_impl is the recursive helper
_tensor_free_graph_impl :: proc(node: ^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil {return}
	if visited[node] {return} 	// Prevent double-free in DAGs
	visited[node] = true

	// ✅ CRITICAL: Do NOT free leaf nodes (op == .None).
	// Leaf nodes are user-managed (e.g., input data, weights).
	if node.op == .None {
		return
	}

	// Recursively free inputs first
	for input in node.inputs {
		_tensor_free_graph_impl(input, visited)
	}

	// Free this intermediate node
	if node.data.data != nil {l.matrix_free(&node.data)}
	if node.grad.data != nil {l.matrix_free(&node.grad)}
	delete(node.inputs)
	delete(node.int_metadata)
	if node.dropout_mask != nil {delete(node.dropout_mask, node.allocator)}
	free(node, node.allocator)
}

// tensor_free_graph automatically cleans up all intermediate tensors
// in the computation graph, starting from 'root'.
// It protects leaf nodes (inputs/weights) from being freed.
tensor_free_graph :: proc(root: ^Tensor) {
	if root == nil {return}

	visited := make(map[^Tensor]bool)
	defer delete(visited)

	// Use context.allocator for the nodes array
	nodes := make([dynamic]^Tensor, 0, context.allocator)
	_collect_graph_nodes(root, &nodes, &visited)

	// Free in reverse order
	for i := len(nodes) - 1; i >= 0; i -= 1 {
		node := nodes[i]
		if node.op != .None {
			if node.data.data != nil {l.matrix_free(&node.data)}
			if node.grad.data != nil {l.matrix_free(&node.grad)}
			delete(node.inputs)
			delete(node.int_metadata)
			if node.dropout_mask != nil {delete(node.dropout_mask, node.allocator)}
			free(node, node.allocator)
		}
	}

	// ✅ FIX: Dynamic arrays don't need allocator argument
	delete(nodes)
}

_collect_graph_nodes :: proc(node: ^Tensor, nodes: ^[dynamic]^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil {return}
	if visited[node] {return}
	visited[node] = true

	append(nodes, node)

	for input in node.inputs {
		_collect_graph_nodes(input, nodes, visited)
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

_validate_graph_impl :: proc(node: ^Tensor, visited: ^map[^Tensor]bool) -> bool {
	if node == nil {return true}
	if visited[node] {return true}
	visited[node] = true

	// Check that this node has valid data
	if len(node.data.data) == 0 {
		fmt.printf("ERROR: Tensor with op %v has empty data\n", node.op)
		return false
	}

	// Check that gradient is allocated if requires_grad
	if node.requires_grad && node.grad.data == nil {
		fmt.printf("ERROR: Tensor with op %v requires grad but has nil gradient\n", node.op)
		return false
	}

	// Recursively check inputs
	for input in node.inputs {
		if !_validate_graph_impl(input, visited) {
			return false
		}
	}

	return true
}
// tensor_batch_norm_2d performs Batch Normalization over a 4D tensor (N, C, H, W)
// tensor_batch_norm_2d performs Batch Normalization over a 4D tensor (N, C, H, W)
// ✅ SIMD-optimized version
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
) -> ^Tensor {
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]
	N_hw := f64(N * H * W)
	channel_size := N * H * W

	out_data := l.matrix_new(f64, 1, N * C * H * W, input.allocator)

	// ✅ Allocate buffers ONCE outside the loop for reuse
	channel_buf := make([]f64, channel_size, context.temp_allocator)
	mean_vec := make([]f64, channel_size, context.temp_allocator)
	centered := make([]f64, channel_size, context.temp_allocator)
	inv_std_vec := make([]f64, channel_size, context.temp_allocator)
	normalized := make([]f64, channel_size, context.temp_allocator)
	gamma_vec := make([]f64, channel_size, context.temp_allocator)
	beta_vec := make([]f64, channel_size, context.temp_allocator)
	scaled := make([]f64, channel_size, context.temp_allocator)

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
	delete(channel_buf, context.temp_allocator)
	delete(mean_vec, context.temp_allocator)
	delete(centered, context.temp_allocator)
	delete(inv_std_vec, context.temp_allocator)
	delete(normalized, context.temp_allocator)
	delete(gamma_vec, context.temp_allocator)
	delete(beta_vec, context.temp_allocator)
	delete(scaled, context.temp_allocator)

	out := tensor_new(out_data, input.requires_grad || weight.requires_grad, input.allocator)
	out.shape = input.shape

	if out.requires_grad {
		out.op = .BatchNorm2d
		append(&out.inputs, input)
		append(&out.inputs, weight)
		append(&out.inputs, bias)
	}

	return out
}
// ============================================================================
// RNN Operations (with Checkpointing for memory efficiency)
// ============================================================================

// _rnn_step_forward performs a single RNN step without creating autograd nodes
_rnn_step_forward :: proc(
	x_t: []f64,
	h_prev: []f64,
	w_ih: []f64,
	w_hh: []f64,
	bias: []f64,
	h_t: []f64,
	batch: int,
	in_size: int,
	hidden_size: int,
	allocator: mem.Allocator,
) {
	// ✅ FIX: Use '=' for struct initialization in Odin
	x_mat := l.Matrix(f64) {
		rows = batch,
		cols = in_size,
		data = x_t,
	}
	w_ih_mat := l.Matrix(f64) {
		rows = in_size,
		cols = hidden_size,
		data = w_ih,
	}
	h_prev_mat := l.Matrix(f64) {
		rows = batch,
		cols = hidden_size,
		data = h_prev,
	}
	w_hh_mat := l.Matrix(f64) {
		rows = hidden_size,
		cols = hidden_size,
		data = w_hh,
	}

	// 1. h_ih = x_t @ w_ih
	h_ih := l.matmul_dyn_simd(&x_mat, &w_ih_mat, allocator)
	defer l.matrix_free(&h_ih)

	// 2. h_hh = h_prev @ w_hh
	h_hh := l.matmul_dyn_simd(&h_prev_mat, &w_hh_mat, allocator)
	defer l.matrix_free(&h_hh)

	// 3. h_sum = h_ih + h_hh + bias
	for i in 0 ..< batch * hidden_size {
		row := i / hidden_size
		h_t[i] = h_ih.data[i] + h_hh.data[i] + bias[row]
	}

	// 4. h_t = tanh(h_sum)
	for i in 0 ..< batch * hidden_size {
		h_t[i] = math.tanh(h_t[i])
	}
}

// tensor_rnn processes a full sequence using checkpointing (rematerialization)
tensor_rnn :: proc(
	x: ^Tensor,
	h_0: ^Tensor,
	w_ih: ^Tensor,
	w_hh: ^Tensor,
	bias: ^Tensor,
) -> ^Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	in_size := x.shape[2]
	hidden_size := w_ih.shape[1]

	out_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, x.allocator)

	h_t := make([]f64, batch * hidden_size, context.temp_allocator)
	copy(h_t, h_0.data.data)

	h_next := make([]f64, batch * hidden_size, context.temp_allocator)
	defer delete(h_t, context.temp_allocator)
	defer delete(h_next, context.temp_allocator)

	for s in 0 ..< seq_len {
		x_t := make([]f64, batch * in_size, context.temp_allocator)
		for b in 0 ..< batch {
			src := b * seq_len * in_size + s * in_size
			dst := b * in_size
			copy(x_t[dst:dst + in_size], x.data.data[src:src + in_size])
		}

		_rnn_step_forward(
			x_t,
			h_t,
			w_ih.data.data,
			w_hh.data.data,
			bias.data.data,
			h_next,
			batch,
			in_size,
			hidden_size,
			context.temp_allocator,
		)

		for b in 0 ..< batch {
			src := b * hidden_size
			dst := b * seq_len * hidden_size + s * hidden_size
			copy(out_data.data[dst:dst + hidden_size], h_next[src:src + hidden_size])
			copy(h_t[src:src + hidden_size], h_next[src:src + hidden_size])
		}

		delete(x_t, context.temp_allocator)
	}

	out := tensor_new(out_data, x.requires_grad || w_ih.requires_grad, x.allocator)
	out.shape = [4]int{batch, seq_len, hidden_size, 1}

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

_gru_step_forward :: proc(
	x_t: []f64,
	h_prev: []f64,
	w_ih: []f64,
	w_hh: []f64,
	bias: []f64,
	h_t: []f64,
	r_buf: []f64,
	z_buf: []f64,
	n_buf: []f64, // Can be nil if just computing h_t
	batch: int,
	in_size: int,
	hidden_size: int,
	allocator: mem.Allocator,
) {
	H := hidden_size
	H3 := 3 * H

	x_mat := l.Matrix(f64) {
		rows = batch,
		cols = in_size,
		data = x_t,
	}
	w_ih_mat := l.Matrix(f64) {
		rows = in_size,
		cols = H3,
		data = w_ih,
	}
	h_prev_mat := l.Matrix(f64) {
		rows = batch,
		cols = H,
		data = h_prev,
	}
	w_hh_mat := l.Matrix(f64) {
		rows = H,
		cols = H3,
		data = w_hh,
	}

	// ✅ SIMD: Two large matmuls instead of 6 small ones
	gate_ih := l.matmul_dyn_simd(&x_mat, &w_ih_mat, allocator)
	gate_hh := l.matmul_dyn_simd(&h_prev_mat, &w_hh_mat, allocator)
	defer l.matrix_free(&gate_ih)
	defer l.matrix_free(&gate_hh)

	// Add bias
	for b in 0 ..< batch {
		for i in 0 ..< H3 {
			gate_ih.data[b * H3 + i] += bias[i]
		}
	}

	// Compute gates (Auto-vectorized by Odin)
	for b in 0 ..< batch {
		for i in 0 ..< H {
			idx_r := b * H3 + i
			idx_z := b * H3 + H + i
			idx_n := b * H3 + 2 * H + i

			r_pre := gate_ih.data[idx_r] + gate_hh.data[idx_r]
			z_pre := gate_ih.data[idx_z] + gate_hh.data[idx_z]

			r_val := 1.0 / (1.0 + math.exp(-r_pre))
			z_val := 1.0 / (1.0 + math.exp(-z_pre))

			if r_buf != nil {r_buf[b * H + i] = r_val}
			if z_buf != nil {z_buf[b * H + i] = z_val}

			n_pre := gate_ih.data[idx_n] + r_val * gate_hh.data[idx_n]
			n_val := math.tanh(n_pre)
			if n_buf != nil {n_buf[b * H + i] = n_val}

			// h_t = (1 - z) * n + z * h_prev
			h_t[b * H + i] = (1.0 - z_val) * n_val + z_val * h_prev[b * H + i]
		}
	}
}

tensor_gru :: proc(
	x: ^Tensor,
	h_0: ^Tensor,
	w_ih: ^Tensor,
	w_hh: ^Tensor,
	bias: ^Tensor,
) -> ^Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	in_size := x.shape[2]
	hidden_size := w_ih.shape[1] / 3

	out_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, x.allocator)

	h_t := make([]f64, batch * hidden_size, context.temp_allocator)
	copy(h_t, h_0.data.data)
	h_next := make([]f64, batch * hidden_size, context.temp_allocator)
	defer delete(h_t, context.temp_allocator)
	defer delete(h_next, context.temp_allocator)

	for s in 0 ..< seq_len {
		x_t := make([]f64, batch * in_size, context.temp_allocator)
		for b in 0 ..< batch {
			src := b * seq_len * in_size + s * in_size
			dst := b * in_size
			copy(x_t[dst:dst + in_size], x.data.data[src:src + in_size])
		}

		_gru_step_forward(
			x_t,
			h_t,
			w_ih.data.data,
			w_hh.data.data,
			bias.data.data,
			h_next,
			nil,
			nil,
			nil,
			batch,
			in_size,
			hidden_size,
			context.allocator,
		)

		for b in 0 ..< batch {
			src := b * hidden_size
			dst := b * seq_len * hidden_size + s * hidden_size
			copy(out_data.data[dst:dst + hidden_size], h_next[src:src + hidden_size])
			copy(h_t[src:src + hidden_size], h_next[src:src + hidden_size])
		}
		delete(x_t, context.temp_allocator)
	}

	out := tensor_new(out_data, x.requires_grad || w_ih.requires_grad, x.allocator)
	out.shape = [4]int{batch, seq_len, hidden_size, 1}

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
// LSTM Operations (SIMD-optimized with checkpointing)
// ============================================================================

_lstm_step_forward :: proc(
	x_t: []f64,
	h_prev: []f64,
	c_prev: []f64,
	w_ih: []f64,
	w_hh: []f64,
	bias: []f64,
	h_t: []f64,
	c_t: []f64,
	i_buf: []f64,
	f_buf: []f64,
	g_buf: []f64,
	o_buf: []f64,
	batch: int,
	in_size: int,
	hidden_size: int,
	allocator: mem.Allocator,
) {
	H := hidden_size
	H4 := 4 * H

	x_mat := l.Matrix(f64) {
		rows = batch,
		cols = in_size,
		data = x_t,
	}
	w_ih_mat := l.Matrix(f64) {
		rows = in_size,
		cols = H4,
		data = w_ih,
	}
	h_prev_mat := l.Matrix(f64) {
		rows = batch,
		cols = H,
		data = h_prev,
	}
	w_hh_mat := l.Matrix(f64) {
		rows = H,
		cols = H4,
		data = w_hh,
	}

	// ✅ SIMD: Two massive matmuls for all 4 gates
	gate_ih := l.matmul_dyn_simd(&x_mat, &w_ih_mat, allocator)
	gate_hh := l.matmul_dyn_simd(&h_prev_mat, &w_hh_mat, allocator)
	defer l.matrix_free(&gate_ih)
	defer l.matrix_free(&gate_hh)

	// Add bias
	for b in 0 ..< batch {
		for i in 0 ..< H4 {
			gate_ih.data[b * H4 + i] += bias[i]
		}
	}

	// Compute gates and states (Auto-vectorized by Odin)
	for b in 0 ..< batch {
		for i in 0 ..< H {
			idx_i := b * H4 + i
			idx_f := b * H4 + H + i
			idx_g := b * H4 + 2 * H + i
			idx_o := b * H4 + 3 * H + i

			i_pre := gate_ih.data[idx_i] + gate_hh.data[idx_i]
			f_pre := gate_ih.data[idx_f] + gate_hh.data[idx_f]
			g_pre := gate_ih.data[idx_g] + gate_hh.data[idx_g]
			o_pre := gate_ih.data[idx_o] + gate_hh.data[idx_o]

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
			h_t[b * H + i] = o_val * math.tanh(c_t[b * H + i])
		}
	}
}

tensor_lstm :: proc(
	x: ^Tensor,
	h_0: ^Tensor,
	c_0: ^Tensor,
	w_ih: ^Tensor,
	w_hh: ^Tensor,
	bias: ^Tensor,
) -> ^Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	in_size := x.shape[2]
	hidden_size := w_ih.shape[1] / 4

	out_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, x.allocator)

	h_t := make([]f64, batch * hidden_size, context.temp_allocator)
	c_t := make([]f64, batch * hidden_size, context.temp_allocator)
	copy(h_t, h_0.data.data)
	copy(c_t, c_0.data.data)

	h_next := make([]f64, batch * hidden_size, context.temp_allocator)
	c_next := make([]f64, batch * hidden_size, context.temp_allocator)
	defer {
		delete(h_t, context.temp_allocator)
		delete(c_t, context.temp_allocator)
		delete(h_next, context.temp_allocator)
		delete(c_next, context.temp_allocator)
	}

	for s in 0 ..< seq_len {
		x_t := make([]f64, batch * in_size, context.temp_allocator)
		for b in 0 ..< batch {
			src := b * seq_len * in_size + s * in_size
			dst := b * in_size
			copy(x_t[dst:dst + in_size], x.data.data[src:src + in_size])
		}

		_lstm_step_forward(
			x_t,
			h_t,
			c_t,
			w_ih.data.data,
			w_hh.data.data,
			bias.data.data,
			h_next,
			c_next,
			nil,
			nil,
			nil,
			nil,
			batch,
			in_size,
			hidden_size,
			context.temp_allocator,
		)

		for b in 0 ..< batch {
			src := b * hidden_size
			dst := b * seq_len * hidden_size + s * hidden_size
			copy(out_data.data[dst:dst + hidden_size], h_next[src:src + hidden_size])
			copy(h_t[src:src + hidden_size], h_next[src:src + hidden_size])
			copy(c_t[src:src + hidden_size], c_next[src:src + hidden_size])
		}
		delete(x_t, context.temp_allocator)
	}

	out := tensor_new(out_data, x.requires_grad || w_ih.requires_grad, x.allocator)
	out.shape = [4]int{batch, seq_len, hidden_size, 1}

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
	batch := input.shape[0]
	seq_len := input.shape[1]
	vocab_size := weight.shape[0]
	embed_dim := weight.shape[1]

	out_data := l.matrix_new(f64, 1, batch * seq_len * embed_dim, input.allocator)

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

	out := tensor_new(out_data, weight.requires_grad, input.allocator)
	out.shape = [4]int{batch, seq_len, embed_dim, 1}

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

tensor_scaled_dot_product_attention :: proc(Q: ^Tensor, K: ^Tensor, V: ^Tensor) -> ^Tensor {
	batch := Q.shape[0]
	seq_q := Q.shape[1]
	seq_k := K.shape[1]
	d_k := Q.shape[2]
	d_v := V.shape[2]

	out_data := l.matrix_new(f64, 1, batch * seq_q * d_v, Q.allocator)
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

		// 1. S_b = Q_b @ K_b^T
		k_b_t := _matrix_transpose(k_b, context.temp_allocator)
		s_b := l.matmul_dyn_simd(&q_b, &k_b_t, context.temp_allocator)
		l.matrix_free(&k_b_t)

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
		o_b := l.matmul_dyn_simd(&p_b, &v_b, context.temp_allocator)
		copy(out_data.data[b * seq_q * d_v:(b + 1) * seq_q * d_v], o_b.data)

		l.matrix_free(&o_b)
		l.matrix_free(&s_b)
	}

	out := tensor_new(out_data, Q.requires_grad || K.requires_grad || V.requires_grad, Q.allocator)
	out.shape = [4]int{batch, seq_q, d_v, 1}

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
	out_data := l.matrix_new(f64, 1, batch * num_heads * seq_len * head_dim, x.allocator)

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

	out := tensor_new(out_data, x.requires_grad, x.allocator)
	out.shape = [4]int{batch * num_heads, seq_len, head_dim, 1}

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
	out_data := l.matrix_new(f64, 1, batch * seq_len * d_model, x.allocator)

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

	out := tensor_new(out_data, x.requires_grad, x.allocator)
	out.shape = [4]int{batch, seq_len, d_model, 1}

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
) -> ^Tensor {
	d_model := gamma.data.cols

	if beta.data.cols != d_model {
		panic("tensor_layer_norm: beta dimension mismatch")
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

	out_data := l.matrix_new(f64, input.data.rows, input.data.cols, input.allocator)

	// Temporary buffers for SIMD operations (reused across rows)
	centered := make([]f64, d_model, context.temp_allocator)
	x_hat := make([]f64, d_model, context.temp_allocator)
	inv_std_vec := make([]f64, d_model, context.temp_allocator)
	defer {
		delete(centered, context.temp_allocator)
		delete(x_hat, context.temp_allocator)
		delete(inv_std_vec, context.temp_allocator)
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
		gamma_row := gamma.data.data
		beta_row := beta.data.data

		l.vec_mul_simd(x_hat, gamma_row, out_row)
		l.vec_add_simd(out_row, beta_row, out_row)
	}

	out := tensor_new(
		out_data,
		input.requires_grad || gamma.requires_grad || beta.requires_grad,
		input.allocator,
	)
	out.shape = input.shape

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
) -> ^Tensor {
	batch := Q.shape[0]
	seq_q := Q.shape[1]
	seq_k := K.shape[1]
	d_k := Q.shape[2]
	d_v := V.shape[2]

	out_data := l.matrix_new(f64, 1, batch * seq_q * d_v, Q.allocator)
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

		// 1. S_b = Q_b @ K_b^T
		k_b_t := _matrix_transpose(k_b, context.temp_allocator)
		s_b := l.matmul_dyn_simd(&q_b, &k_b_t, context.temp_allocator)
		l.matrix_free(&k_b_t)

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
		o_b := l.matmul_dyn_simd(&p_b, &v_b, context.temp_allocator)
		copy(out_data.data[b * seq_q * d_v:(b + 1) * seq_q * d_v], o_b.data)

		l.matrix_free(&o_b)
		l.matrix_free(&s_b)
	}

	out := tensor_new(out_data, Q.requires_grad || K.requires_grad || V.requires_grad, Q.allocator)
	out.shape = [4]int{batch, seq_q, d_v, 1}

	if out.requires_grad {
		out.op = .MaskedScaledDotProductAttention
		append(&out.inputs, Q)
		append(&out.inputs, K)
		append(&out.inputs, V)
		// Store mask for backward pass
		for i in 0 ..< len(mask) {
			if mask[i] != 0.0 {
				append(&out.int_metadata, 1) // Masked position
			} else {
				append(&out.int_metadata, 0) // Unmasked position
			}
		}
	}

	return out
}
