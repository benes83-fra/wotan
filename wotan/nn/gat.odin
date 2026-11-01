package nn

import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// Graph Attention Network (GAT) Layer
// ============================================================================
// Multi-head attention restricted to the edges of a graph over the assets,
// weighted by the edge strengths: for cross-sectional models where assets
// influence their neighbours (e.g. a correlation graph).
GATLayer :: struct {
	linear:    LinearLayer,
	mha:       MultiHeadAttentionLayer,
	num_heads: int,
	head_dim:  int,
	adjacency: ^t.Tensor, // [num_assets, num_assets], stored for Sequential and checkpoints
	allocator: mem.Allocator,
}

gat_layer_new :: proc(
	in_features: int,
	hidden_dim: int,
	num_heads: int,
	adjacency: ^t.Tensor,
	allocator: mem.Allocator = context.allocator,
) -> GATLayer {
	layer: GATLayer
	layer.num_heads = num_heads
	layer.head_dim = hidden_dim / num_heads
	layer.allocator = allocator
	layer.adjacency = adjacency

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
// adjacency: [num_assets, num_assets] (constant), A[i, j] >= 0 is the weight
//            of the edge from asset i to asset j, 0 for no edge
// 1. h = x W + b                                     [batch, num_assets, hidden]
// 2. multi-head attention of every asset over its neighbours: the scores get
//    log A[i, j] added, so asset i attends only to j with A[i, j] > 0, with
//    weights ∝ A[i, j] · exp(q_i · k_j / √d). With equal scores this is the
//    A-weighted mean of the neighbours (A row-normalized). Every asset needs at
//    least one edge (e.g. a self-loop, as compute_correlation_adjacency sets).
gat_layer_forward :: proc(layer: ^GATLayer, x: ^t.Tensor, adjacency: ^t.Tensor) -> ^t.Tensor {
	n := x.shape[1]
	assert(
		x.shape[2] * x.shape[3] == layer.linear.in_features,
		"gat_layer_forward: the feature size of x differs from the layer's in_features",
	)
	assert(
		adjacency.data.rows == n && adjacency.data.cols == n,
		"gat_layer_forward: the adjacency must be [num_assets, num_assets]",
	)
	h := linear_forward(&layer.linear, x)

	edges := make([]f64, n * n, x.allocator)
	defer delete(edges, x.allocator)
	for &e, i in edges {
		a := adjacency.data.data[i]
		e = math.ln(a) if a > 0 else _MASKED
	}
	m := &layer.mha
	return _attention(
		&m.q_proj,
		&m.k_proj,
		&m.v_proj,
		&m.out_proj,
		m.num_heads,
		m.head_dim,
		h,
		h,
		m.use_flash,
		m.is_causal,
		mask = edges,
	)
}
