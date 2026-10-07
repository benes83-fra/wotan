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
	num_experts: int,
	top_k:       int,
	d_model:     int,
	d_ff:        int,
	experts:     []FFNLayer,
	router:      LinearLayer, // Maps d_model -> num_experts
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
	delete(layer.experts)
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
	x_flat := t.tensor_reshape(x, [4]int{N, layer.d_model, 1, 1})

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

	// 6. Initialize Accumulator
	out_data := l.matrix_new(f64, N, layer.d_model, allocator)
	out_flat: ^t.Tensor


	// 7. Route through Experts
	for i in 0 ..< layer.num_experts {
		// Extract the gate weights for this expert: [N, 1]
		gate_i_data := l.matrix_new(f64, N, 1, allocator)
		for n in 0 ..< N {
			gate_i_data.data[n] = gates.data.data[n * layer.num_experts + i]
		}
		gate_i := t.tensor_new(gate_i_data, true, allocator)
		gate_i.shape = [4]int{N, 1, 1, 1}

		// Apply gates to input: x_gated[n] = x_flat[n] * gate_i[n]
		x_gated := t.tensor_gate_mul(x_flat, gate_i)

		// Pass through expert FFN
		e_out := ffn_layer_forward(&layer.experts[i], x_gated)

		// Accumulate
		if out_flat == nil {
			out_flat = e_out // ✅ First expert becomes the base
		} else {
			out_flat = t.tensor_add(out_flat, e_out) // ✅ Subsequent experts add to the graph
		}
	}

	// 8. Reshape back to [batch, seq_len, d_model, 1]
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
	x_flat := t.tensor_reshape(x, [4]int{N, layer.d_model, 1, 1})
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
