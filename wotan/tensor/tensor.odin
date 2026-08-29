
package tensor

import l "../linalg"
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
	Constant,
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
	Softmax, // <--- ADD THIS
	Entropy,
	BCELoss,
	PermuteLOB,
	SharpeLoss,
	Concat,
	NormalizeTime,
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
	// Initialize gradients to zero
	if requires_grad {
		t.grad = l.matrix_new(f64, data.rows, data.cols, allocator)
		// (In a real implementation, we'd zero it out here, but matrix_new usually does)
	}
	global_tensors_created += 1
	return t
}

tensor_free :: proc(t: ^Tensor) {
	if t == nil {return}
	if t.data.data != nil {l.matrix_free(&t.data)}
	if t.grad.data != nil {l.matrix_free(&t.grad)}

	delete(t.inputs)
	delete(t.int_metadata)
	if t.dropout_mask != nil {delete(t.dropout_mask, t.allocator)}
	global_tensors_freed += 1 // ✅ ADD THIS
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
		fmt.printf("DEBUG PANIC: tensor_mul dimension mismatch!\n")
		fmt.printf("  Tensor A: rows=%d, cols=%d\n", a.data.rows, a.data.cols)
		fmt.printf("  Tensor B: rows=%d, cols=%d\n", b.data.rows, b.data.cols)
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

	diff := make([]f64, len(pred.data.data), context.allocator)
	l.vec_sub_simd(pred.data.data, target.data.data, diff)
	squared_sum := l.dot_simd(diff, diff)
	defer delete(diff, context.allocator)

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

	out_data := l.matrix_new(f64, 1, 1, logits.allocator)
	out_data.data[0] = loss

	out := tensor_new(out_data, logits.requires_grad, logits.allocator)

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

tensor_backward :: proc(root: ^Tensor, allocator: mem.Allocator = context.allocator) {
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
		case .Clamp:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				lo := f64(node.int_metadata[0]) / 1_000_000.0
				hi := f64(node.int_metadata[1]) / 1_000_000.0
				for i in 0 ..< len(a_in.grad.data) {
					v := a_in.data.data[i]
					if v >= lo && v <= hi {
						a_in.grad.data[i] += node.grad.data[i]
					}
					// outside [lo,hi] → gradient is 0 (clamped)
				}
			}

		case .PermuteLOB:
			x_in := node.inputs[0]
			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				batch := node.int_metadata[0]
				c_out := node.int_metadata[1]
				t_out := node.int_metadata[2]
				l_out := node.int_metadata[3]
				feat_dim := c_out * l_out

				// Reverse the permutation to route gradients back to [B, C, T, L]
				for b in 0 ..< batch {
					for tt in 0 ..< t_out {
						for c in 0 ..< c_out {
							for ll in 0 ..< l_out {
								src_idx := (b * t_out + tt) * feat_dim + c * l_out + ll
								dst_idx :=
									b * (c_out * t_out * l_out) +
									c * (t_out * l_out) +
									tt * l_out +
									ll
								x_in.grad.data[dst_idx] += node.grad.data[src_idx]
							}
						}
					}
				}
			}
		case .SumDim1:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				batch_size := node.shape[0]
				num_assets := node.shape[1]
				// The gradient of a sum is just the gradient of the output, broadcasted to all summed elements
				for b in 0 ..< batch_size {
					grad_val := node.grad.data[b]
					for asset in 0 ..< num_assets {
						a_in.grad.data[b * num_assets + asset] += grad_val
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
		case .Sub:
			a_in := node.inputs[0]
			b_in := node.inputs[1]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
				}
			}
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 && len(node.grad.data) > 0 {
					l.vec_sub_inplace_simd(b_in.grad.data, node.grad.data) // ✅ SIMD
				}
			}

		case .Mean:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				n := f64(len(a_in.data.data))
				scalar_grad := node.grad.data[0] / n
				l.vec_broadcast_add_simd(scalar_grad, a_in.grad.data) // ✅ SIMD
			}

		case .Neg:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					l.vec_sub_inplace_simd(a_in.grad.data, node.grad.data) // ✅ SIMD
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
					b_t := _matrix_transpose(b_view, allocator)
					grad_a_view := l.matmul_dyn_simd(&grad_view, &b_t, allocator)
					l.matrix_free(&b_t)

					// Copy flattened gradient back
					copy(a_in.grad.data, grad_a_view.data)
					l.matrix_free(&grad_a_view)
				}

				if b_in.requires_grad {
					tensor_ensure_grad(b_in)
					a_t := _matrix_transpose(a_view, allocator)
					grad_b := l.matmul_dyn_simd(&a_t, &grad_view, allocator)
					l.matrix_free(&a_t)

					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&grad_b)
				}
				continue // Skip the standard 2D backward pass
			}
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					// ✅ FIX: Use allocator for ALL temporary matrices
					bt := _matrix_transpose(b_in.data, allocator)
					grad_a := l.matmul_dyn_simd(&node.grad, &bt, allocator)
					l.vec_add_simd(a_in.grad.data, grad_a.data, a_in.grad.data)
					l.matrix_free(&bt)
					l.matrix_free(&grad_a)
				}
			}

			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 && len(node.grad.data) > 0 {
					// ✅ FIX: Use allocator
					at := _matrix_transpose(a_in.data, allocator)
					grad_b := l.matmul_dyn_simd(&at, &node.grad, allocator)
					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&at)
					l.matrix_free(&grad_b)
				}
			}
		case .Sum:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) == 0 {
					fmt.println("ERROR: Sum grad.data is empty")
					continue
				}
				scalar_grad := node.grad.data[0]
				l.vec_broadcast_add_simd(scalar_grad, a_in.grad.data) // ✅ SIMD
			}
		case .KLDivergence:
			mu_in := node.inputs[0]
			log_var_in := node.inputs[1]

			scalar_grad := node.grad.data[0]
			n := f64(len(mu_in.data.data))

			// ∂KL/∂mu = -0.5 * (-2 * mu) / n = mu / n
			if mu_in.requires_grad {
				tensor_ensure_grad(mu_in)
				for i in 0 ..< len(mu_in.grad.data) {
					mu_in.grad.data[i] += scalar_grad * mu_in.data.data[i] / n
				}
			}

			// ∂KL/∂log_var = -0.5 * (1 - exp(log_var)) / n
			if log_var_in.requires_grad {
				tensor_ensure_grad(log_var_in)
				for i in 0 ..< len(log_var_in.grad.data) {
					log_var_in.grad.data[i] +=
						scalar_grad * (1.0 - math.exp(log_var_in.data.data[i])) / (2.0 * n)
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
			mask := make([]f64, seq_q * seq_k, allocator)
			neg_inf: f64 = -1e9
			for i in 0 ..< len(mask) {
				if node.int_metadata[i] == 1 {
					mask[i] = neg_inf
				} else {
					mask[i] = 0.0
				}
			}
			defer delete(mask, allocator)

			dQ := make([]f64, len(Q_in.data.data), allocator)
			dK := make([]f64, len(K_in.data.data), allocator)
			dV := make([]f64, len(V_in.data.data), allocator)
			defer {
				delete(dQ, allocator)
				delete(dK, allocator)
				delete(dV, allocator)
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
				k_b_t := _matrix_transpose(k_b, allocator)
				s_b := l.matmul_dyn_simd(&q_b, &k_b_t, allocator)
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
				p_b_t := _matrix_transpose(p_b, allocator)
				dV_b := l.matmul_dyn_simd(&p_b_t, &dO_b, allocator)
				copy(dV[b * seq_k * d_v:(b + 1) * seq_k * d_v], dV_b.data)
				l.matrix_free(&p_b_t)
				l.matrix_free(&dV_b)

				// dP = dO_b @ V_b^T
				v_b_t := _matrix_transpose(v_b, allocator)
				dP_b := l.matmul_dyn_simd(&dO_b, &v_b_t, allocator)

				// ✅ FIX: Free v_b_t immediately after use
				l.matrix_free(&v_b_t)

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
				dQ_b := l.matmul_dyn_simd(&dP_b, &k_b, allocator)
				copy(dQ[b * seq_q * d_k:(b + 1) * seq_q * d_k], dQ_b.data)
				l.matrix_free(&dQ_b)

				// dK = (dP_b * scale)^T @ Q_b
				dP_b_t := _matrix_transpose(dP_b, allocator)
				dK_b := l.matmul_dyn_simd(&dP_b_t, &q_b, allocator)
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
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// ✅ FIX: Add defensive length checks
				grad_out_len := len(node.grad.data)
				input_len := len(a_in.data.data)
				grad_in_len := len(a_in.grad.data)

				if grad_out_len != input_len || grad_out_len != grad_in_len {
					fmt.printf(
						"WARNING: ReLU backward length mismatch at epoch - " +
						"grad_out=%d, input=%d, grad_in=%d\n",
						grad_out_len,
						input_len,
						grad_in_len,
					)
					continue
				}

				l.vec_relu_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data)
			}
		case .Sigmoid:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) != len(a_in.data.data) ||
				   len(node.grad.data) != len(a_in.grad.data) {
					fmt.printf("WARNING: Sigmoid backward length mismatch\n")
					continue
				}
				l.vec_sigmoid_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data) // ✅ SIMD
			}

		case .Tanh:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) != len(a_in.data.data) ||
				   len(node.grad.data) != len(a_in.grad.data) {
					fmt.printf("WARNING: Tanh backward length mismatch\n")
					continue
				}
				l.vec_tanh_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data) // ✅ SIMD
			}
		case .Exp:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx exp(x) = exp(x) = node.data[i]
					a_in.grad.data[i] += node.grad.data[i] * node.data.data[i]
				}
			}

		case .Sqrt:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx sqrt(x) = 1 / (2*sqrt(x)) = 1 / (2*out)
					a_in.grad.data[i] += node.grad.data[i] / (2.0 * node.data.data[i])
				}
			}

		case .Log:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx log(x) = 1/x
					a_in.grad.data[i] += node.grad.data[i] / a_in.data.data[i]
				}
			}

		case .Div:
			a_in := node.inputs[0]
			b_in := node.inputs[1]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/da (a/b) = 1/b
					a_in.grad.data[i] += node.grad.data[i] / b_in.data.data[i]
				}
			}
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				for i in 0 ..< len(b_in.grad.data) {
					// d/db (a/b) = -a/b²
					b_val := b_in.data.data[i]
					_ = a_in.grad.data[i] // just to reference (not used)
					b_in.grad.data[i] -= node.grad.data[i] * a_in.data.data[i] / (b_val * b_val)
				}
			}

		case .NormCDF:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				inv_sqrt_2pi := 0.3989422804014327
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx N(x) = φ(x) = (1/√(2π)) * exp(-x²/2)
					x := a_in.data.data[i]
					phi := math.exp(-0.5 * x * x) * inv_sqrt_2pi
					a_in.grad.data[i] += node.grad.data[i] * phi
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
		case .Reparameterize:
			mu_in := node.inputs[0]
			log_var_in := node.inputs[1]

			if mu_in.requires_grad {
				tensor_ensure_grad(mu_in)
				// ∂z/∂mu = 1, so gradient flows directly
				for i in 0 ..< len(mu_in.grad.data) {
					mu_in.grad.data[i] += node.grad.data[i]
				}
			}

			if log_var_in.requires_grad {
				tensor_ensure_grad(log_var_in)
				// ∂z/∂log_var = 0.5 * exp(0.5 * log_var) * epsilon
				// But we don't have epsilon stored, so we use the chain rule approximation
				// ∂z/∂log_var ≈ 0.5 * (z - mu) / exp(0.5 * log_var) * exp(0.5 * log_var)
				// Simplified: ∂z/∂log_var = 0.5 * std * epsilon = 0.5 * (z - mu)
				for i in 0 ..< len(log_var_in.grad.data) {
					mu_val := mu_in.data.data[i]
					z_val := node.data.data[i]
					log_var_val := log_var_in.data.data[i]
					log_var_val = max(-10.0, min(10.0, log_var_val))
					std := math.exp(0.5 * log_var_val)

					// Gradient through std
					log_var_in.grad.data[i] +=
						node.grad.data[i] * 0.5 * std * (z_val - mu_val) / std
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
		// In tensor_backward, find case .ScaledDotProductAttention and update it:
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

			dQ := make([]f64, len(Q_in.data.data), allocator)
			dK := make([]f64, len(K_in.data.data), allocator)
			dV := make([]f64, len(V_in.data.data), allocator)
			defer {
				delete(dQ, allocator)
				delete(dK, allocator)
				delete(dV, allocator)
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
				k_b_t := _matrix_transpose(k_b, allocator)
				s_b := l.matmul_dyn_simd(&q_b, &k_b_t, allocator)
				l.matrix_free(&k_b_t) // ✅ Free immediately

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
				p_b_t := _matrix_transpose(p_b, allocator)
				dV_b := l.matmul_dyn_simd(&p_b_t, &dO_b, allocator)
				copy(dV[b * seq_k * d_v:(b + 1) * seq_k * d_v], dV_b.data)
				l.matrix_free(&p_b_t) // ✅ Free immediately
				l.matrix_free(&dV_b) // ✅ Free immediately

				// 3. dP = dO_b @ V_b^T
				v_b_t := _matrix_transpose(v_b, allocator)
				dP_b := l.matmul_dyn_simd(&dO_b, &v_b_t, allocator)
				l.matrix_free(&v_b_t) // ✅ Free immediately

				// 4. dS = P_b * (dP_b - sum(dP_b * P_b, dim=-1))
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
				dQ_b := l.matmul_dyn_simd(&dP_b, &k_b, allocator)
				copy(dQ[b * seq_q * d_k:(b + 1) * seq_q * d_k], dQ_b.data)
				l.matrix_free(&dQ_b) // ✅ Free immediately

				// 6. dK = (dP_b * scale)^T @ Q_b
				dP_b_t := _matrix_transpose(dP_b, allocator)
				dK_b := l.matmul_dyn_simd(&dP_b_t, &q_b, allocator)
				copy(dK[b * seq_k * d_k:(b + 1) * seq_k * d_k], dK_b.data)
				l.matrix_free(&dP_b_t) // ✅ Free immediately
				l.matrix_free(&dK_b) // ✅ Free immediately

				l.matrix_free(&dP_b) // ✅ Free immediately
				l.matrix_free(&s_b) // ✅ Free immediately
			}

			if Q_in.requires_grad && len(Q_in.grad.data) > 0 {
				l.vec_add_simd(Q_in.grad.data, dQ, Q_in.grad.data)
			}
			if K_in.requires_grad && len(K_in.grad.data) > 0 {
				l.vec_add_simd(K_in.grad.data, dK, K_in.grad.data)
			}
			if V_in.requires_grad && len(V_in.grad.data) > 0 {
				l.vec_add_simd(V_in.grad.data, dV, V_in.grad.data)
			}
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
				diff := make([]f64, len(pred_in.data.data), allocator)
				l.vec_sub_simd(pred_in.data.data, target_in.data.data, diff)

				// ✅ SIMD Optimization: grad_pred += scale * diff
				l.axpy_simd(scale, diff, pred_in.grad.data)
				delete(diff, allocator)
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
			centered := make([]f64, d_model, allocator)
			x_hat := make([]f64, d_model, allocator)
			dx_hat := make([]f64, d_model, allocator)
			dx_row := make([]f64, d_model, allocator)
			inv_std_vec := make([]f64, d_model, allocator)
			defer {
				delete(centered, allocator)
				delete(x_hat, allocator)
				delete(dx_hat, allocator)
				delete(dx_row, allocator)
				delete(inv_std_vec, allocator)
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
		case .NormalizeTime:
			input_in := node.inputs[0]
			if input_in.requires_grad && len(input_in.grad.data) > 0 {
				N := node.int_metadata[0]
				C := node.int_metadata[1]
				T := node.int_metadata[2]
				L := node.int_metadata[3]
				eps := f64(node.int_metadata[4]) / 1_000_000_000.0

				T_f := f64(T)
				// Temporary buffers
				grad_buf := make([]f64, T, context.allocator)
				norm_buf := make([]f64, T, context.allocator)
				mean_vec := make([]f64, T, context.allocator)
				defer {
					delete(grad_buf, context.allocator)
					delete(norm_buf, context.allocator)
					delete(mean_vec, context.allocator)
				}

				for n: int = 0; n < N; n += 1 {
					for c: int = 0; c < C; c += 1 {
						for el: int = 0; el < L; el += 1 {
							// Extract gradients and normalized values
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								grad_buf[t] = node.grad.data[idx]
								norm_buf[t] = node.data.data[idx] // This is the normalized input
							}

							// Gradient of Z-score: dx = (dy - mean(dy) - y * dot(dy, y) / T) / std
							sum_grad := l.sum_simd(grad_buf)
							mean_grad := sum_grad / T_f
							dot_grad_y := l.dot_simd(grad_buf, norm_buf)

							for i: int = 0; i < T; i += 1 {mean_vec[i] = mean_grad}

							temp := make([]f64, T, context.allocator)
							l.vec_sub_simd(grad_buf, mean_vec, temp)

							scalar := dot_grad_y / T_f
							for t: int = 0; t < T; t += 1 {
								temp[t] -= norm_buf[t] * scalar
							}

							// Recompute std of input for this slice to divide the gradient
							in_buf := make([]f64, T, context.allocator)
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								in_buf[t] = input_in.data.data[idx]
							}
							in_mean := l.sum_simd(in_buf) / T_f
							for i: int = 0; i < T; i += 1 {mean_vec[i] = in_mean}

							in_centered := make([]f64, T, context.allocator)
							l.vec_sub_simd(in_buf, mean_vec, in_centered)
							in_var := l.dot_simd(in_centered, in_centered) / T_f
							in_std := math.sqrt(in_var + eps)
							inv_std := 1.0 / in_std

							for t: int = 0; t < T; t += 1 {temp[t] *= inv_std}

							// Accumulate to input gradient
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								input_in.grad.data[idx] += temp[t]
							}

							delete(in_buf, context.allocator)
							delete(in_centered, context.allocator)
							delete(temp, context.allocator)
						}
					}
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

			dx := make([]f64, len(x_in.data.data), allocator)
			dw_ih := make([]f64, len(w_ih_in.data.data), allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), allocator)
			dbias := make([]f64, len(bias_in.data.data), allocator)

			h_t := make([]f64, batch * H, allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * H, allocator)
			h_next := make([]f64, batch * H, allocator)
			x_t := make([]f64, batch * in_size, allocator)

			r_buf := make([]f64, batch * H, allocator)
			z_buf := make([]f64, batch * H, allocator)
			n_buf := make([]f64, batch * H, allocator)
			n_hh_buf := make([]f64, batch * H, allocator)

			dh_next := make([]f64, batch * H, allocator)
			dh_prev := make([]f64, batch * H, allocator)
			d_gate_ih := make([]f64, batch * H3, allocator)
			d_gate_hh := make([]f64, batch * H3, allocator)

			defer {
				delete(dx, allocator); delete(dw_ih, allocator)
				delete(dw_hh, allocator); delete(dbias, allocator)
				delete(h_t, allocator); delete(h_prev, allocator)
				delete(h_next, allocator); delete(x_t, allocator)
				delete(r_buf, allocator); delete(z_buf, allocator)
				delete(n_buf, allocator); delete(n_hh_buf, allocator)
				delete(dh_next, allocator); delete(dh_prev, allocator)
				delete(d_gate_ih, allocator); delete(d_gate_hh, allocator)
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
					allocator,
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
				gate_hh := l.matmul_dyn_simd(&h_prev_mat, &w_hh_mat, allocator)
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
				dz := make([]f64, batch * H, allocator)
				dn := make([]f64, batch * H, allocator)
				dr_from_n := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					dh_prev[i] += dh_next[i] * z_buf[i]
					dz[i] = dh_next[i] * (h_prev[i] - n_buf[i])
					dn[i] = dh_next[i] * (1.0 - z_buf[i])
				}

				// Backprop through n = tanh(n_ih + r * n_hh)
				d_n_pre := make([]f64, batch * H, allocator)
				for i in 0 ..< batch * H {
					d_n_pre[i] = dn[i] * (1.0 - n_buf[i] * n_buf[i])
					dr_from_n[i] = d_n_pre[i] * n_hh_buf[i]
				}

				// Backprop through sigmoid gates
				dr_pre := make([]f64, batch * H, allocator)
				dz_pre := make([]f64, batch * H, allocator)
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

				delete(dz, allocator); delete(dn, allocator)
				delete(dr_from_n, allocator); delete(d_n_pre, allocator)
				delete(dr_pre, allocator); delete(dz_pre, allocator)

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

				x_t_t := _matrix_transpose(x_mat, allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &d_gate_ih_mat, allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t); l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat_bwd, allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &d_gate_hh_mat, allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t); l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, allocator)
				dx_t_res := l.matmul_dyn_simd(&d_gate_ih_mat, &w_ih_t, allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {dx[dst + i] += dx_t_res.data[src + i]}
				}
				l.matrix_free(&w_ih_t); l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, allocator)
				dh_prev_res := l.matmul_dyn_simd(&d_gate_hh_mat, &w_hh_t, allocator)
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

			dx := make([]f64, len(x_in.data.data), allocator)
			dw_ih := make([]f64, len(w_ih_in.data.data), allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), allocator)
			dbias := make([]f64, len(bias_in.data.data), allocator)

			h_t := make([]f64, batch * H, allocator)
			c_t := make([]f64, batch * H, allocator)
			copy(h_t, h_0_in.data.data)
			copy(c_t, c_0_in.data.data)

			h_prev := make([]f64, batch * H, allocator)
			c_prev := make([]f64, batch * H, allocator)
			h_next := make([]f64, batch * H, allocator)
			c_next := make([]f64, batch * H, allocator)
			x_t := make([]f64, batch * in_size, allocator)

			i_buf := make([]f64, batch * H, allocator)
			f_buf := make([]f64, batch * H, allocator)
			g_buf := make([]f64, batch * H, allocator)
			o_buf := make([]f64, batch * H, allocator)

			dh_next := make([]f64, batch * H, allocator)
			dc_next := make([]f64, batch * H, allocator)
			dh_prev := make([]f64, batch * H, allocator)
			dc_prev := make([]f64, batch * H, allocator)
			d_gate_ih := make([]f64, batch * H4, allocator)
			d_gate_hh := make([]f64, batch * H4, allocator)

			defer {
				delete(dx, allocator); delete(dw_ih, allocator)
				delete(dw_hh, allocator); delete(dbias, allocator)
				delete(h_t, allocator); delete(c_t, allocator)
				delete(h_prev, allocator); delete(c_prev, allocator)
				delete(h_next, allocator); delete(c_next, allocator)
				delete(x_t, allocator)
				delete(i_buf, allocator); delete(f_buf, allocator)
				delete(g_buf, allocator); delete(o_buf, allocator)
				delete(dh_next, allocator); delete(dc_next, allocator)
				delete(dh_prev, allocator); delete(dc_prev, allocator)
				delete(d_gate_ih, allocator); delete(d_gate_hh, allocator)
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
					allocator,
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
				tanh_c := make([]f64, batch * H, allocator)
				dov := make([]f64, batch * H, allocator)
				dc := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					tanh_c[i] = math.tanh(c_t[i])
					dov[i] = dh_next[i] * tanh_c[i]
					dc[i] = dh_next[i] * o_buf[i] * (1.0 - tanh_c[i] * tanh_c[i]) + dc_next[i]
				}

				// Backprop through c_t = f * c_prev + i * g
				df := make([]f64, batch * H, allocator)
				di := make([]f64, batch * H, allocator)
				dg := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					df[i] = dc[i] * c_prev[i]
					di[i] = dc[i] * g_buf[i]
					dg[i] = dc[i] * i_buf[i]
					dc_prev[i] = dc[i] * f_buf[i]
				}

				// Backprop through activation functions
				di_pre := make([]f64, batch * H, allocator)
				df_pre := make([]f64, batch * H, allocator)
				dg_pre := make([]f64, batch * H, allocator)
				do_pre := make([]f64, batch * H, allocator)

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

				delete(tanh_c, allocator)
				delete(dov, allocator)
				delete(dc, allocator)
				delete(df, allocator)
				delete(di, allocator); delete(dg, allocator)
				delete(di_pre, allocator); delete(df_pre, allocator)
				delete(dg_pre, allocator); delete(do_pre, allocator)

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

				x_t_t := _matrix_transpose(x_mat, allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &d_gate_ih_mat, allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t); l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat_bwd, allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &d_gate_hh_mat, allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t); l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, allocator)
				dx_t_res := l.matmul_dyn_simd(&d_gate_ih_mat, &w_ih_t, allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {dx[dst + i] += dx_t_res.data[src + i]}
				}
				l.matrix_free(&w_ih_t); l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, allocator)
				dh_prev_res := l.matmul_dyn_simd(&d_gate_hh_mat, &w_hh_t, allocator)
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
		case .Scale:
			a_in := node.inputs[0]
			scalar := f64(node.int_metadata[0]) / 1000000.0
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				// a_in.grad += node.grad * scalar
				// Use fma: out += a * b where a=node.grad, b=scalar (broadcast)
				// Since we don't have a vec_fma_scalar, do it in two steps
				temp := make([]f64, len(node.grad.data), allocator)
				l.vec_scale_simd(node.grad.data, scalar, temp)
				l.vec_add_simd(a_in.grad.data, temp, a_in.grad.data)
				delete(temp, allocator)
			}
		case .CrossEntropy:
			// The magical simplified gradient: grad = (softmax_prob - target_one_hot) / N
			logits_in := node.inputs[0]
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: CrossEntropy node has empty gradient, skipping")
				continue
			}
			scalar_grad := node.grad.data[0] // Usually 1.0

			// ✅ FIX: Use stored dimensions
			if len(node.int_metadata) < 2 {
				fmt.println("ERROR: CrossEntropy missing dimension metadata")
				continue
			}
			N := node.int_metadata[0]
			C := node.int_metadata[1]

			if len(node.int_metadata) < 2 + N {
				fmt.printf(
					"ERROR: CrossEntropy int_metadata length %d < %d\n",
					len(node.int_metadata),
					2 + N,
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

				target_class := node.int_metadata[2 + i]

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
		case .BCELoss:
			prediction := node.inputs[0]
			target := node.inputs[1]

			if prediction.requires_grad {
				tensor_ensure_grad(prediction)
				n := len(prediction.data.data)
				grad_scalar := node.grad.data[0]

				// Gradient of BCE loss w.r.t. prediction:
				// dL/dp = -(t/p - (1-t)/(1-p)) / n
				for i in 0 ..< n {
					p := prediction.data.data[i]
					t := target.data.data[i]

					// Clamp prediction for numerical stability
					p = math.max(1e-7, math.min(1.0 - 1e-7, p))

					// Gradient: -(t/p - (1-t)/(1-p))
					grad := -(t / p - (1.0 - t) / (1.0 - p)) / f64(n)
					prediction.grad.data[i] += grad_scalar * grad
				}
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
			x_buf := make([]f64, channel_size, allocator)
			dout_buf := make([]f64, channel_size, allocator)
			defer delete(x_buf, allocator)
			defer delete(dout_buf, allocator)

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
				mean_vec := make([]f64, channel_size, allocator)
				for i in 0 ..< channel_size {mean_vec[i] = mean}
				centered := make([]f64, channel_size, allocator)
				l.vec_sub_simd(x_buf, mean_vec, centered)
				delete(mean_vec, allocator)

				var := l.dot_simd(centered, centered) / N_hw
				std := math.sqrt(var + eps)
				inv_std := 1.0 / std
				gamma := weight_in.data.data[c]

				// 4. Compute x_hat (SIMD mul)
				x_hat := make([]f64, channel_size, allocator)
				inv_std_vec := make([]f64, channel_size, allocator)
				for i in 0 ..< channel_size {inv_std_vec[i] = inv_std}
				l.vec_mul_simd(centered, inv_std_vec, x_hat)
				delete(inv_std_vec, allocator)

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
					dout_gamma := make([]f64, channel_size, allocator)
					gamma_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {gamma_vec[i] = gamma}
					l.vec_mul_simd(dout_buf, gamma_vec, dout_gamma)
					delete(gamma_vec, allocator)

					// Precompute sums (SIMD)
					sum_dout_gamma := l.sum_simd(dout_gamma)
					sum_dout_gamma_x_hat := l.dot_simd(dout_gamma, x_hat)

					// dx = (1 / (N_hw * std)) * (N_hw * dout_gamma - sum_dout_gamma - x_hat * sum_dout_gamma_x_hat)

					// term1 = N_hw * dout_gamma (SIMD)
					term1 := make([]f64, channel_size, allocator)
					n_hw_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {n_hw_vec[i] = N_hw}
					l.vec_mul_simd(dout_gamma, n_hw_vec, term1)
					delete(n_hw_vec, allocator)
					delete(dout_gamma, allocator)

					// term2 = x_hat * sum_dout_gamma_x_hat (SIMD)
					term2 := make([]f64, channel_size, allocator)
					s2_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {s2_vec[i] = sum_dout_gamma_x_hat}
					l.vec_mul_simd(x_hat, s2_vec, term2)
					delete(s2_vec, allocator)
					delete(x_hat, allocator)

					// combined = term1 - sum_dout_gamma (SIMD)
					sum_dg_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {sum_dg_vec[i] = sum_dout_gamma}
					l.vec_sub_simd(term1, sum_dg_vec, term1)
					delete(sum_dg_vec, allocator)

					// combined = combined - term2 (SIMD)
					l.vec_sub_simd(term1, term2, term1)
					delete(term2, allocator)

					// dx = combined / (N_hw * std) (SIMD)
					inv_denom := 1.0 / (N_hw * std)
					inv_denom_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {inv_denom_vec[i] = inv_denom}
					l.vec_mul_simd(term1, inv_denom_vec, term1)
					delete(inv_denom_vec, allocator)

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
					delete(term1, allocator)
				} else {
					delete(x_hat, allocator)
				}

				delete(centered, allocator)
			}
		case .SharpeLoss:
			ret_in := node.inputs[0]
			if ret_in.requires_grad && len(ret_in.grad.data) > 0 {
				n := f64(node.int_metadata[0])
				rf := f64(node.int_metadata[1]) / 1_000_000.0
				mean := f64(node.int_metadata[2]) / 1_000_000.0
				std := f64(node.int_metadata[3]) / 1_000_000.0

				// Analytical gradient of L = -Sharpe w.r.t x_i:
				// dL/dx_i = ( (mean - rf) * (x_i - mean) - std^2 ) / (n * std^3)
				scalar_grad := node.grad.data[0]
				std_sq := std * std
				std_cube := std_sq * std
				denominator := n * std_cube
				excess_mean := mean - rf

				for i in 0 ..< len(ret_in.grad.data) {
					x_i := ret_in.data.data[i]
					grad := ((excess_mean * (x_i - mean)) - std_sq) / denominator
					ret_in.grad.data[i] += scalar_grad * grad
				}
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

			dx := make([]f64, len(x_in.data.data), allocator)
			dh_0 := make([]f64, len(h_0_in.data.data), allocator) // ✅ FIX: typo corrected
			dw_ih := make([]f64, len(w_ih_in.data.data), allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), allocator)
			dbias := make([]f64, len(bias_in.data.data), allocator)

			h_t := make([]f64, batch * hidden_size, allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * hidden_size, allocator)
			h_next := make([]f64, batch * hidden_size, allocator)
			x_t := make([]f64, batch * in_size, allocator)

			dh_next := make([]f64, batch * hidden_size, allocator)
			dx_t := make([]f64, batch * in_size, allocator)
			dh_prev := make([]f64, batch * hidden_size, allocator)
			dw_ih_step := make([]f64, in_size * hidden_size, allocator)
			dw_hh_step := make([]f64, hidden_size * hidden_size, allocator)
			dbias_step := make([]f64, hidden_size, allocator)

			defer {
				delete(dx, allocator)
				delete(dh_0, allocator)
				delete(dw_ih, allocator)
				delete(dw_hh, allocator)
				delete(dbias, allocator)
				delete(h_t, allocator)
				delete(h_prev, allocator)
				delete(h_next, allocator)
				delete(x_t, allocator)
				delete(dh_next, allocator)
				delete(dx_t, allocator)
				delete(dh_prev, allocator)
				delete(dw_ih_step, allocator)
				delete(dw_hh_step, allocator)
				delete(dbias_step, allocator)
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
					allocator,
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

				x_t_t := _matrix_transpose(x_mat, allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &dh_mat, allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t)
				l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat, allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &dh_mat, allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t)
				l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, allocator)
				dx_t_res := l.matmul_dyn_simd(&dh_mat, &w_ih_t, allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {
						dx[dst + i] += dx_t_res.data[src + i]
					}
				}
				l.matrix_free(&w_ih_t)
				l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, allocator)
				dh_prev_res := l.matmul_dyn_simd(&dh_mat, &w_hh_t, allocator)
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
		case .BinaryCrossEntropy:
			pred_in := node.inputs[0]
			target_in := node.inputs[1]

			if pred_in.requires_grad {
				eps := 1e-7
				n := f64(len(pred_in.data.data))
				scalar_grad := node.grad.data[0]

				for i in 0 ..< len(pred_in.data.data) {
					p := math.max(math.min(pred_in.data.data[i], 1.0 - eps), eps)
					t := target_in.data.data[i]
					grad := scalar_grad * (-t / p + (1.0 - t) / (1.0 - p)) / n
					pred_in.grad.data[i] += grad
				}
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
		case .Softmax:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				sum_p_grad := 0.0
				for j in 0 ..< len(node.data.data) {
					sum_p_grad += node.data.data[j] * node.grad.data[j]
				}
				for i in 0 ..< len(a_in.grad.data) {
					a_in.grad.data[i] += node.data.data[i] * (node.grad.data[i] - sum_p_grad)
				}
			}
		case .Entropy:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				scalar_grad := node.grad.data[0]
				for i in 0 ..< len(a_in.grad.data) {
					p := a_in.data.data[i]
					if p > 1e-10 {
						// d(-sum p ln p)/dp = -(1 + ln p)
						a_in.grad.data[i] += scalar_grad * (-(1.0 + math.ln_f64(p)))
					}
				}
			}
		case .Concat:
			dim := node.int_metadata[0]
			num_tensors := len(node.inputs)

			// Calculate total dim size for source indexing
			total_dim_size := 0
			for j in 0 ..< num_tensors {
				total_dim_size += node.int_metadata[1 + j]
			}

			offset := 0
			for i in 0 ..< num_tensors {
				t_in := node.inputs[i]
				// ✅ FIX: Declare t_dim_size outside the if block so it's visible for offset update
				t_dim_size := node.int_metadata[1 + i]

				if t_in.requires_grad && len(t_in.grad.data) > 0 {
					outer_blocks := 1
					for d in 0 ..< dim {
						outer_blocks *= t_in.shape[d]
					}

					inner_block_size := 1
					for d in dim + 1 ..< 4 {
						inner_block_size *= t_in.shape[d]
					}

					for ob in 0 ..< outer_blocks {
						for c in 0 ..< t_dim_size {
							src_start :=
								ob * (total_dim_size * inner_block_size) +
								(offset + c) * inner_block_size
							dst_start :=
								ob * (t_dim_size * inner_block_size) + c * inner_block_size

							for k in 0 ..< inner_block_size {
								t_in.grad.data[dst_start + k] += node.grad.data[src_start + k]
							}
						}
					}
				}
				// ✅ FIX: Now t_dim_size is in scope here
				offset += t_dim_size
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

			col, _, _ := _im2col(input_in.data.data, N, C_in, H, W, kH, kW, stride, pad, allocator)

			// ✅ ADD: Check if col is empty
			if len(col) == 0 {
				fmt.println("WARNING: Conv2d im2col returned empty slice, skipping")
				continue
			}

			if input_in.requires_grad {
				tensor_ensure_grad(input_in)
				if len(input_in.grad.data) > 0 {
					grad_input_col := make([]f64, N * col_w * col_h, allocator)
					weight_2d_t := _matrix_transpose(weight_in.data, allocator)

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

						grad_col_2d := l.matmul_dyn_simd(&weight_2d_t, &grad_out_2d, allocator)

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
						allocator,
					)

					for i in 0 ..< len(input_in.grad.data) {
						if i < len(grad_input) {
							input_in.grad.data[i] += grad_input[i]
						}
					}

					delete(grad_input_col, allocator)
					delete(grad_input, allocator)
				}
			}

			if weight_in.requires_grad {
				tensor_ensure_grad(weight_in)
				if len(weight_in.grad.data) > 0 {
					grad_weight := make([]f64, len(weight_in.data.data), allocator)

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

						col_2d_t := _matrix_transpose(col_2d, allocator)
						grad_weight_2d := l.matmul_dyn_simd(&grad_out_2d, &col_2d_t, allocator)

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

					delete(grad_weight, allocator)
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

			delete(col, allocator)

		// We don't calculate gradients for the target data.
		case .None, .Constant:
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

	nodes := make([dynamic]^Tensor, 0, context.allocator)
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
			if node.dropout_mask != nil {delete(node.dropout_mask, node.allocator)}
			free(node, node.allocator)
			freed_in_graph += 1
			global_tensors_freed += 1
		}
	}

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
	channel_buf := make([]f64, channel_size, context.allocator)
	mean_vec := make([]f64, channel_size, context.allocator)
	centered := make([]f64, channel_size, context.allocator)
	inv_std_vec := make([]f64, channel_size, context.allocator)
	normalized := make([]f64, channel_size, context.allocator)
	gamma_vec := make([]f64, channel_size, context.allocator)
	beta_vec := make([]f64, channel_size, context.allocator)
	scaled := make([]f64, channel_size, context.allocator)

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
	delete(channel_buf, context.allocator)
	delete(mean_vec, context.allocator)
	delete(centered, context.allocator)
	delete(inv_std_vec, context.allocator)
	delete(normalized, context.allocator)
	delete(gamma_vec, context.allocator)
	delete(beta_vec, context.allocator)
	delete(scaled, context.allocator)

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

	h_t := make([]f64, batch * hidden_size, context.allocator)
	copy(h_t, h_0.data.data)

	h_next := make([]f64, batch * hidden_size, context.allocator)
	defer delete(h_t, context.allocator)
	defer delete(h_next, context.allocator)

	for s in 0 ..< seq_len {
		x_t := make([]f64, batch * in_size, context.allocator)
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
			context.allocator,
		)

		for b in 0 ..< batch {
			src := b * hidden_size
			dst := b * seq_len * hidden_size + s * hidden_size
			copy(out_data.data[dst:dst + hidden_size], h_next[src:src + hidden_size])
			copy(h_t[src:src + hidden_size], h_next[src:src + hidden_size])
		}

		delete(x_t, context.allocator)
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

	h_t := make([]f64, batch * hidden_size, context.allocator)
	copy(h_t, h_0.data.data)
	h_next := make([]f64, batch * hidden_size, context.allocator)
	defer delete(h_t, context.allocator)
	defer delete(h_next, context.allocator)

	for s in 0 ..< seq_len {
		x_t := make([]f64, batch * in_size, context.allocator)
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
		delete(x_t, context.allocator)
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

	h_t := make([]f64, batch * hidden_size, context.allocator)
	c_t := make([]f64, batch * hidden_size, context.allocator)
	copy(h_t, h_0.data.data)
	copy(c_t, c_0.data.data)

	h_next := make([]f64, batch * hidden_size, context.allocator)
	c_next := make([]f64, batch * hidden_size, context.allocator)
	defer {
		delete(h_t, context.allocator)
		delete(c_t, context.allocator)
		delete(h_next, context.allocator)
		delete(c_next, context.allocator)
	}

	for s in 0 ..< seq_len {
		x_t := make([]f64, batch * in_size, context.allocator)
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
			context.allocator,
		)

		for b in 0 ..< batch {
			src := b * hidden_size
			dst := b * seq_len * hidden_size + s * hidden_size
			copy(out_data.data[dst:dst + hidden_size], h_next[src:src + hidden_size])
			copy(h_t[src:src + hidden_size], h_next[src:src + hidden_size])
			copy(c_t[src:src + hidden_size], c_next[src:src + hidden_size])
		}
		delete(x_t, context.allocator)
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
		k_b_t := _matrix_transpose(k_b, context.allocator)
		s_b := l.matmul_dyn_simd(&q_b, &k_b_t, context.allocator)
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
		o_b := l.matmul_dyn_simd(&p_b, &v_b, context.allocator)
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
	centered := make([]f64, d_model, context.allocator)
	x_hat := make([]f64, d_model, context.allocator)
	inv_std_vec := make([]f64, d_model, context.allocator)
	defer {
		delete(centered, context.allocator)
		delete(x_hat, context.allocator)
		delete(inv_std_vec, context.allocator)
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
		k_b_t := _matrix_transpose(k_b, context.allocator)
		s_b := l.matmul_dyn_simd(&q_b, &k_b_t, context.allocator)
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
		o_b := l.matmul_dyn_simd(&p_b, &v_b, context.allocator)
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

	out_data := l.matrix_new(f64, 1, 1, pred.allocator)
	out_data.data[0] = loss

	out := tensor_new(out_data, pred.requires_grad, pred.allocator)
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

	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_sub_simd(a.data.data, b.data.data, out_data.data)

	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)
	out.shape = a.shape
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

	out_data := l.matrix_new(f64, 1, 1, a.allocator)
	out_data.data[0] = mean_val

	out := tensor_new(out_data, a.requires_grad, a.allocator)
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

	out_data := l.matrix_new(f64, 1, 1, mu.allocator)
	out_data.data[0] = kl_loss

	out := tensor_new(out_data, mu.requires_grad || log_var.requires_grad, mu.allocator)
	if out.requires_grad {
		out.op = .KLDivergence
		append(&out.inputs, mu)
		append(&out.inputs, log_var)
	}
	return out
}

tensor_sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_sigmoid_simd(a.data.data, out_data.data) // ✅ SIMD
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	if out.requires_grad {
		out.op = .Sigmoid
		append(&out.inputs, a)
	}
	return out
}

tensor_tanh :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_tanh_simd(a.data.data, out_data.data) // ✅ SIMD
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	if out.requires_grad {
		out.op = .Tanh
		append(&out.inputs, a)
	}
	return out
}

tensor_leaky_relu :: proc(a: ^Tensor, alpha: f64 = 0.01) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_leaky_relu_simd(a.data.data, alpha, out_data.data) // ✅ SIMD
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	if out.requires_grad {
		out.op = .LeakyReLU
		append(&out.inputs, a)
	}
	return out
}

tensor_neg :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_neg_simd(a.data.data, out_data.data) // ✅ SIMD
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	if out.requires_grad {
		out.op = .Neg
		append(&out.inputs, a)
	}
	return out
}

tensor_scale :: proc(a: ^Tensor, scalar: f64) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_scale_simd(a.data.data, scalar, out_data.data) // ✅ SIMD
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	if out.requires_grad {
		out.op = .Scale
		append(&out.inputs, a)
		append(&out.int_metadata, int(scalar * 1000000))
	}
	return out
}
// ============================================================================
// Element-wise Exponential: out = exp(a)
// Backward: grad_a = grad_out * exp(a) = grad_out * out
// ============================================================================
tensor_exp :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = math.exp(a.data.data[i])
	}
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
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
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = math.sqrt(a.data.data[i])
	}
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
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
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = math.ln_f64(a.data.data[i])
	}
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
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
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	for i in 0 ..< len(a.data.data) {
		out_data.data[i] = a.data.data[i] / b.data.data[i]
	}
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)
	out.shape = a.shape
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
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)

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

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	if out.requires_grad {
		out.op = .NormCDF
		append(&out.inputs, a)
	}
	return out
}
tensor_sum_dim1 :: proc(a: ^Tensor) -> ^Tensor {
	batch_size := a.shape[0]
	num_assets := a.shape[1]

	out_data := l.matrix_new(f64, batch_size, 1, a.allocator)
	for b in 0 ..< batch_size {
		sum := 0.0
		for asset in 0 ..< num_assets {
			sum += a.data.data[b * num_assets + asset]
		}
		out_data.data[b] = sum
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = [4]int{batch_size, 1, 1, 1}
	if out.requires_grad {
		out.op = .SumDim1
		append(&out.inputs, a)
	}
	return out
}
// ===== add to ./wotan/tensor/tensor.odin =====

// tensor_clamp: element-wise clamp to [lo, hi]
tensor_clamp :: proc(a: ^Tensor, lo: f64, hi: f64) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	for i in 0 ..< len(a.data.data) {
		v := a.data.data[i]
		if v < lo {v = lo}
		if v > hi {v = hi}
		out_data.data[i] = v
	}
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	if out.requires_grad {
		out.op = .Clamp // add .Clamp to the Op enum
		append(&out.inputs, a)
		// store lo/hi for backward
		append(&out.int_metadata, int(lo * 1_000_000))
		append(&out.int_metadata, int(hi * 1_000_000))
	}
	return out
}
tensor_softmax :: proc(logits: ^Tensor) -> ^Tensor {
	max_val := -math.F64_MAX
	for v in logits.data.data {
		if v > max_val {max_val = v}
	}
	exp_sum := 0.0
	n := len(logits.data.data)
	exps := make([]f64, n, context.allocator)
	for i in 0 ..< n {
		exps[i] = math.exp_f64(logits.data.data[i] - max_val)
		exp_sum += exps[i]
	}
	out_data := l.matrix_new(f64, logits.data.rows, logits.data.cols, logits.allocator)
	for i in 0 ..< n {
		out_data.data[i] = exps[i] / exp_sum
	}
	delete(exps, context.allocator)
	out := tensor_new(out_data, logits.requires_grad, logits.allocator)
	out.shape = logits.shape
	if out.requires_grad {
		out.op = .Softmax
		append(&out.inputs, logits)
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
	out_data := l.matrix_new(f64, 1, 1, a.allocator)
	out_data.data[0] = entropy_val
	out := tensor_new(out_data, a.requires_grad, a.allocator)
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
	out_data := l.matrix_new(f64, 1, 1, prediction.allocator)
	out_data.data[0] = avg_loss
	out := tensor_new(
		out_data,
		prediction.requires_grad || target.requires_grad,
		prediction.allocator,
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

	out := tensor_new(out_data, requires_grad, allocator)
	out.shape = out_shape

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
	feat_dim := c_out * l_out
	out_data := l.matrix_new(f64, batch * t_out, feat_dim, alloc)

	for b: int = 0; b < batch; b += 1 {
		for tt: int = 0; tt < t_out; tt += 1 {
			for c: int = 0; c < c_out; c += 1 {
				src_idx := b * (c_out * t_out * l_out) + c * (t_out * l_out) + tt * l_out
				dst_idx := (b * t_out + tt) * feat_dim + c * l_out
				copy(out_data.data[dst_idx:dst_idx + l_out], x.data.data[src_idx:src_idx + l_out])
			}
		}
	}

	out := tensor_new(out_data, x.requires_grad, alloc)
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

	out := tensor_new(out_data, returns.requires_grad, allocator)

	if out.requires_grad {
		out.op = .SharpeLoss
		append(&out.inputs, returns)
		// Store metadata for backward pass (scaled by 1,000,000 to fit in int)
		append(&out.int_metadata, int(n))
		append(&out.int_metadata, int(risk_free_rate * 1_000_000))
		append(&out.int_metadata, int(mean * 1_000_000))
		append(&out.int_metadata, int(std * 1_000_000))
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
	N := input.shape[0]
	C := input.shape[1]
	T := input.shape[2]
	L := input.shape[3]

	out_data := l.matrix_new(f64, 1, N * C * T * L, allocator)

	// Temporary contiguous buffers for SIMD operations (allocated once)
	time_buf := make([]f64, T, context.allocator)
	centered := make([]f64, T, context.allocator)
	mean_vec := make([]f64, T, context.allocator)
	inv_std_vec := make([]f64, T, context.allocator)

	defer {
		delete(time_buf, context.allocator)
		delete(centered, context.allocator)
		delete(mean_vec, context.allocator)
		delete(inv_std_vec, context.allocator)
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

	out := tensor_new(out_data, input.requires_grad, allocator)
	out.shape = input.shape

	if out.requires_grad {
		out.op = .NormalizeTime
		append(&out.inputs, input)
		append(&out.int_metadata, N)
		append(&out.int_metadata, C)
		append(&out.int_metadata, T)
		append(&out.int_metadata, L)
		append(&out.int_metadata, int(eps * 1_000_000_000.0)) // Store eps scaled to fit in int
	}

	return out
}
