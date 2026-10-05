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
	out_flat := t.tensor_new(out_data, true, allocator)
	out_flat.shape = x_flat.shape

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
		out_flat = t.tensor_add(out_flat, e_out)
	}

	// 8. Reshape back to [batch, seq_len, d_model, 1]
	out := t.tensor_reshape(out_flat, [4]int{batch, seq_len, layer.d_model, 1})
	return out
}
