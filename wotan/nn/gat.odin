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
	allocator: mem.Allocator,
}

gat_layer_new :: proc(
	in_features: int,
	hidden_dim: int,
	num_heads: int,
	allocator: mem.Allocator = context.allocator,
) -> GATLayer {
	layer: GATLayer
	layer.num_heads = num_heads
	layer.head_dim = hidden_dim / num_heads
	layer.allocator = allocator

	// 1. Linear projection to hidden dimension
	layer.linear = linear_layer_new(in_features, hidden_dim, allocator)

	// 2. Multi-Head Attention to learn complex relational weights
	layer.mha = multi_head_attention_layer_new(hidden_dim, num_heads, allocator)

	return layer
}

gat_layer_free :: proc(layer: ^GATLayer) {
	linear_layer_free(&layer.linear)
	multi_head_attention_layer_free(&layer.mha)
}

// Forward pass:
// x:         [Batch, Num_Assets, Feature_Dim, 1] -> flattened to [Batch * Num_Assets, Feature_Dim]
// adjacency: [Num_Assets, Num_Assets, 1, 1]
gat_layer_forward :: proc(layer: ^GATLayer, x: ^t.Tensor, adjacency: ^t.Tensor) -> ^t.Tensor {
	batch := x.shape[0]
	num_assets := x.shape[1]
	feature_dim := x.shape[2]
	hidden_dim := layer.head_dim * layer.num_heads

	// 1. Project features: [Batch, Num_Assets, Feature_Dim, 1] -> [Batch * Num_Assets, Hidden_Dim]
	h_proj := linear_forward(&layer.linear, x)

	// 2. Graph Convolution (Message Passing): H_agg = Adjacency @ H_proj
	// Reshape h_proj from [Batch * Num_Assets, Hidden_Dim] to [Batch, Num_Assets, Hidden_Dim]
	// This is just a view change for the matmul logic.
	h_proj_3d := t.tensor_new(h_proj.data, h_proj.requires_grad, h_proj.allocator)
	h_proj_3d.shape = [4]int{batch, num_assets, hidden_dim, 1}

	// Create output tensor for aggregation [Batch, Num_Assets, Hidden_Dim, 1]
	agg_data := l.matrix_new(f64, 1, batch * num_assets * hidden_dim, h_proj.allocator)
	agg_tensor := t.tensor_new(agg_data, h_proj.requires_grad, h_proj.allocator)
	agg_tensor.shape = [4]int{batch, num_assets, hidden_dim, 1}

	// Perform batched matmul: For each batch, do Adjacency (N,N) @ h_proj_b (N, D) -> agg_b (N, D)
	for b in 0 ..< batch {
		// Source slice in h_proj: [b * N * D : (b+1) * N * D]
		// We treat this as a matrix [N, D]
		h_b_rows := num_assets
		h_b_cols := hidden_dim
		h_b_offset := b * h_b_rows * h_b_cols

		// Destination slice in agg_tensor: [b * N * D : (b+1) * N * D]
		agg_b_offset := b * h_b_rows * h_b_cols

		// Perform the matmul using raw linalg on slices
		// adjacency is [N, N], h_b is [N, D] -> result is [N, D]
		for i in 0 ..< num_assets { 	// For each output row
			for j in 0 ..< hidden_dim { 	// For each output column
				sum := 0.0
				for k in 0 ..< num_assets { 	// Dot product over the shared dimension
					a_val := adjacency.data.data[i * num_assets + k] // Adjacency[i, k]
					h_val := h_proj.data.data[h_b_offset + k * hidden_dim + j] // h_b[k, j]
					sum += a_val * h_val
				}
				agg_tensor.data.data[agg_b_offset + i * hidden_dim + j] = sum
			}
		}
	}

	// 3. Apply Multi-Head Attention (Seq_Len = Num_Assets)
	out := multi_head_attention_layer_forward(&layer.mha, agg_tensor)

	return out
}
