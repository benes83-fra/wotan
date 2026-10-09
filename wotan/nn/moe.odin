package nn

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Mixture of Experts (MoE) Layer
// ============================================================================

MoELayer :: struct {
	num_experts:  int,
	top_k:        int,
	d_model:      int,
	d_ff:         int,
	experts:      []FFNLayer,
	router:       LinearLayer, // Maps d_model -> num_experts
	allocator:    mem.Allocator,
	aux_loss_val: f64,
}

moe_layer_new :: proc(
	d_model: int,
	d_ff: int,
	num_experts: int,
	top_k: int = 2,
	allocator: mem.Allocator = context.allocator,
) -> MoELayer {
	layer: MoELayer
	layer.num_experts = num_experts
	layer.top_k = top_k
	layer.d_model = d_model
	layer.d_ff = d_ff
	layer.allocator = allocator

	// Router: d_model -> num_experts
	layer.router = linear_layer_new(d_model, num_experts, allocator)

	// Experts: Array of FFNs
	layer.experts = make([]FFNLayer, num_experts, allocator)
	for i in 0 ..< num_experts {
		layer.experts[i] = ffn_layer_new(d_model, d_ff, allocator)
	}

	return layer
}

moe_layer_free :: proc(layer: ^MoELayer) {
	linear_layer_free(&layer.router)
	for i in 0 ..< layer.num_experts {
		ffn_layer_free(&layer.experts[i])
	}
	if layer.experts != nil {
		delete(layer.experts, layer.allocator) // ✅ USE IT HERE
		layer.experts = nil
	}
	layer.num_experts = 0
}

// moe_layer_forward performs Sparse MoE routing using Gated Dense Routing.
moe_layer_forward :: proc(
	layer: ^MoELayer,
	x: ^t.Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^t.Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	N := batch * seq_len

	// 1. Flatten to [N, d_model] for routing
	// ✅ FIX: Manually create a true 2D matrix (rows=N, cols=d_model).
	// This forces tensor_matmul to use the clean Standard 2D path, bypassing
	// the sequence-model hack that mutates shape[2] and breaks the router dimensions.
	x_flat_data := l.matrix_new(f64, N, layer.d_model, allocator)
	copy(x_flat_data.data, x.data.data)
	x_flat := t.tensor_new(x_flat_data, x.requires_grad, allocator)
	x_flat.shape = [4]int{N, layer.d_model, 1, 1}
	if x_flat.requires_grad {
		x_flat.op = .Reshape
		append(&x_flat.inputs, x)
	}
	// 2. Router Logits & Softmax
	logits := linear_forward(&layer.router, x_flat)
	probs := t.tensor_softmax(logits) // [N, num_experts]

	// 3. Top-K Masking
	mask := t.tensor_top_k_mask(probs, layer.top_k)

	// 4. Gates = probs * mask (Straight-Through Estimator)
	gates := t.tensor_mul(probs, mask)

	// 5. Renormalize gates so they sum to 1 for each token
	for n in 0 ..< N {
		sum_g: f64 = 0.0
		for e in 0 ..< layer.num_experts {
			sum_g += gates.data.data[n * layer.num_experts + e]
		}
		if sum_g > 1e-8 {
			for e in 0 ..< layer.num_experts {
				gates.data.data[n * layer.num_experts + e] /= sum_g
			}
		}
	}
	load := make([]f64, layer.num_experts, allocator)
	importance := make([]f64, layer.num_experts, allocator)

	for n in 0 ..< N {
		for e in 0 ..< layer.num_experts {
			load[e] += mask.data.data[n * layer.num_experts + e]
			importance[e] += probs.data.data[n * layer.num_experts + e]
		}
	}

	aux_loss_val := 0.0
	for e in 0 ..< layer.num_experts {
		f_i := load[e] / f64(N)
		P_i := importance[e] / f64(N)
		aux_loss_val += f_i * P_i
	}
	layer.aux_loss_val = aux_loss_val * f64(layer.num_experts)

	delete(load, allocator)
	delete(importance, allocator)
	// 6. Initialize Accumulator
	//out_data := l.matrix_new(f64, N, layer.d_model, allocator)
	out_flat: ^t.Tensor


	// 7. Route through Experts
	for i in 0 ..< layer.num_experts {
		// ✅ FIX: Use the graph-preserving slice operation.
		// gate_i is now a legitimate child of `gates`. The DAG is unbroken.
		gate_i := t.tensor_slice_gate(gates, i, allocator)

		x_gated := t.tensor_gate_mul(x_flat, gate_i)
		e_out := ffn_layer_forward(&layer.experts[i], x_gated)

		if out_flat == nil {
			out_flat = e_out
		} else {
			if out_flat.data.rows != e_out.data.rows || out_flat.data.cols != e_out.data.cols {
				e_out = t.tensor_reshape(e_out, out_flat.shape)
			}
			out_flat = t.tensor_add(out_flat, e_out)
		}
	}

	// 8. Reshape back
	out := t.tensor_reshape(out_flat, [4]int{batch, seq_len, layer.d_model, 1})
	return out

}
// moe_layer_aux_loss computes the load-balancing loss to prevent expert collapse.
// It encourages the router to distribute tokens evenly across all experts.
moe_layer_aux_loss :: proc(
	layer: ^MoELayer,
	x: ^t.Tensor,
	allocator: mem.Allocator = context.allocator,
) -> ^t.Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	N := batch * seq_len
	E := layer.num_experts


	// 1. Recompute router probabilities (very cheap, just one Linear layer)
	// ✅ FIX: Use true 2D matrix to bypass the sequence hack
	x_flat_data := l.matrix_new(f64, N, layer.d_model, allocator)
	copy(x_flat_data.data, x.data.data)
	x_flat := t.tensor_new(x_flat_data, false, allocator) // Detached from graph
	x_flat.shape = [4]int{N, layer.d_model, 1, 1}
	logits := linear_forward(&layer.router, x_flat)
	probs := t.tensor_softmax(logits)
	mask := t.tensor_top_k_mask(probs, layer.top_k)

	// 2. Compute Load (fraction of tokens routed to each expert)
	// load[e] = sum(mask[:, e]) / N
	load := make([]f64, E, allocator)
	for n in 0 ..< N {
		for e in 0 ..< E {
			load[e] += mask.data.data[n * E + e]
		}
	}
	for e in 0 ..< E {load[e] /= f64(N)}

	// 3. Compute Importance (average router probability for each expert)
	// importance[e] = sum(probs[:, e]) / N
	importance := make([]f64, E, allocator)
	for n in 0 ..< N {
		for e in 0 ..< E {
			importance[e] += probs.data.data[n * E + e]
		}
	}
	for e in 0 ..< E {importance[e] /= f64(N)}

	// 4. Aux Loss = E * sum(load * importance)
	aux_loss_val := 0.0
	for e in 0 ..< E {
		aux_loss_val += load[e] * importance[e]
	}
	aux_loss_val *= f64(E)

	// 5. Wrap in a scalar tensor
	out_data := l.matrix_new(f64, 1, 1, allocator)
	out_data.data[0] = aux_loss_val

	// Note: We don't attach this to the autograd graph because the routing mask
	// is non-differentiable. The gradients for the router flow through the
	// Straight-Through Estimator in the main forward pass.
	out := t.tensor_new(out_data, false, allocator)

	delete(load, allocator)
	delete(importance, allocator)
	t.tensor_free(x_flat)
	t.tensor_free(logits)
	t.tensor_free_graph(probs)
	t.tensor_free_graph(mask)

	return out
}
