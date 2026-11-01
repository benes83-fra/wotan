package nn

import l "../linalg"
import t "../tensor"
import "core:mem"

// ============================================================================
// Graph Attention Network (GAT) Layer
// ============================================================================
// Combines explicit graph structure (Adjacency Matrix) with Multi-Head Attention.
// Perfect for cross-sectional asset pricing where assets influence each other.
GATLayer :: struct {
	linear:    LinearLayer,
	mha:       MultiHeadAttentionLayer,
	num_heads: int,
	head_dim:  int,
	adjacency: ^t.Tensor, // ✅ Added for Sequential compatibility
	allocator: mem.Allocator,
}

// Update the constructor to accept or initialize adjacency
gat_layer_new :: proc(
	in_features: int,
	hidden_dim: int,
	num_heads: int,
	adjacency: ^t.Tensor, // ✅ Pass it in
	allocator: mem.Allocator = context.allocator,
) -> GATLayer {
	layer: GATLayer
	layer.num_heads = num_heads
	layer.head_dim = hidden_dim / num_heads
	layer.allocator = allocator
	layer.adjacency = adjacency // ✅ Store it

	layer.linear = linear_layer_new(in_features, hidden_dim, allocator)
	layer.mha = multi_head_attention_layer_new(hidden_dim, num_heads, allocator = allocator)

	return layer
}

gat_layer_free :: proc(layer: ^GATLayer) {
	linear_layer_free(&layer.linear)
	multi_head_attention_layer_free(&layer.mha)
	if layer.adjacency != nil {
		t.tensor_free(layer.adjacency)
	}
}

// Forward pass:
// x:         [batch, num_assets, feature_dim, 1]
// adjacency: [num_assets, num_assets] (constant)
// 1. h = x W + b                         [batch, num_assets, hidden]
// 2. message passing h_b <- A h_b         (per sequence)
// 3. multi-head self-attention across the assets
gat_layer_forward :: proc(layer: ^GATLayer, x: ^t.Tensor, adjacency: ^t.Tensor) -> ^t.Tensor {
	batch := x.shape[0]
	n := x.shape[1]
	hidden := layer.head_dim * layer.num_heads

	h := linear_forward(&layer.linear, x)

	// (A h_b)ᵀ = h_bᵀ Aᵀ: transpose to [batch * hidden, n] (the MHA permute with
	// `hidden` heads of size 1), one matmul with the constant Aᵀ, transpose back
	a_t := _constant(n * n, {n, n, 1, 1}, x.allocator)
	a_t.data.rows, a_t.data.cols = n, n
	for i in 0 ..< n {
		for k in 0 ..< n {a_t.data.data[k * n + i] = adjacency.data.data[i * n + k]}
	}
	agg_t := t.tensor_matmul(t.tensor_permute_mha(h, batch, n, hidden, 1), a_t)
	_constant_done(a_t, agg_t)
	agg := t.tensor_permute_mha_inverse(agg_t, batch, n, hidden, 1)

	return multi_head_attention_layer_forward(&layer.mha, agg)
}
