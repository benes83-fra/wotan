package nn

import l "../linalg"
import t "../tensor"
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
	aux_loss_val: f64, // load-balancing loss of the last forward pass
	aux_loss:     ^t.Tensor, // the same as part of the graph (forward with aux_loss = true), else nil
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
		delete(layer.experts, layer.allocator)
		layer.experts = nil
	}
	layer.num_experts = 0
}

// moe_layer_forward routes every token to its top_k experts:
//   gates = softmax of the router logits over the top_k experts (others 0)
//   out   = Σ_e gate_e · FFN_e(x)
// Experts run on all tokens; one that no token selects is skipped. The
// load-balancing loss E · Σ_e f_e P_e (Switch Transformer; f_e share of the
// top-k assignments, P_e mean router probability) is kept in aux_loss_val,
// and with aux_loss = true also as layer.aux_loss in the graph: add it to the
// training loss so that it trains the router.
moe_layer_forward :: proc(
	layer: ^MoELayer,
	x: ^t.Tensor,
	allocator: mem.Allocator = context.allocator,
	aux_loss: bool = false,
) -> ^t.Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	N := batch * seq_len
	E := layer.num_experts

	// 1. Tokens as a true [N, d_model] matrix (the router softmax works on rows)
	x_flat := as_matrix(x, N, layer.d_model, allocator)
	logits := linear_forward(&layer.router, x_flat) // [N, E]

	// 2. Top-k gates: the other experts get -inf before the softmax
	hide := _constant(N * E, {N, E, 1, 1}, allocator)
	hide.data.rows, hide.data.cols = N, E
	load := make([]f64, E, context.temp_allocator)
	defer delete(load, context.temp_allocator)
	for n in 0 ..< N {
		row := logits.data.data[n * E:(n + 1) * E]
		h := hide.data.data[n * E:(n + 1) * E]
		for &v in h {v = _MASKED}
		for _ in 0 ..< min(layer.top_k, E) {
			best := -1
			for e in 0 ..< E {
				if h[e] != 0 && (best < 0 || row[e] > row[best]) {best = e}
			}
			h[best] = 0
			load[best] += 1
		}
	}
	masked := t.tensor_add(logits, hide)
	_constant_done(hide, masked)
	gates := t.tensor_softmax(masked)

	// 3. Load balancing: E Σ_e f_e P_e = Σ_n Σ_e probs[n, e] · E f_e / N
	probs := t.tensor_softmax(logits)
	aux := 0.0
	for e in 0 ..< E {
		P_e := 0.0
		for n in 0 ..< N {P_e += probs.data.data[n * E + e]}
		aux += (load[e] / f64(N)) * (P_e / f64(N))
	}
	layer.aux_loss_val = f64(E) * aux
	layer.aux_loss = nil
	if aux_loss {
		w := _constant(N * E, {N, E, 1, 1}, allocator)
		w.data.rows, w.data.cols = N, E
		for n in 0 ..< N {
			for e in 0 ..< E {w.data.data[n * E + e] = f64(E) * load[e] / (f64(N) * f64(N))}
		}
		weighted := t.tensor_mul(probs, w)
		_constant_done(w, weighted)
		layer.aux_loss = t.tensor_sum(weighted)
	} else if !probs.requires_grad {
		t.tensor_free(probs)
	}

	// 4. Weighted sum of the experts
	out_flat: ^t.Tensor
	for e in 0 ..< E {
		if load[e] == 0 {continue}
		y := t.tensor_gate_mul(
			ffn_layer_forward(&layer.experts[e], x_flat),
			t.tensor_slice_gate(gates, e),
		)
		out_flat = y if out_flat == nil else t.tensor_add(out_flat, y)
	}
	return t.tensor_reshape(out_flat, [4]int{batch, seq_len, layer.d_model, 1})
}
