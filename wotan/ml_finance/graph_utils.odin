package ml_finance

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// Dynamic Graph Construction for Cross-Sectional Finance
// ============================================================================

// compute_correlation_adjacency: asset graph from a feature tensor [Batch, Num_Assets,
// Time_Steps, 1]. The Pearson correlation of assets i and j is taken over all Batch *
// Time_Steps values of each (all windows pooled, not just the first), then
// A[i, j] = max(0, corr) off the diagonal, A[i, i] = 1 (the self-loop every asset needs in
// gat_layer_forward), and each row is normalised to sum 1. An asset with no dispersion has
// corr 0 with every other asset. Returns a [Num_Assets, Num_Assets] tensor (shape
// {n, n, 1, 1}); give it to gat_layer_new, which owns and frees it.
compute_correlation_adjacency :: proc(
	features: ^t.Tensor, // [Batch, Num_Assets, Time, 1]
	num_assets: int,
	time_steps: int,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	batch := features.shape[0]
	assert(features.shape[1] == num_assets && features.shape[2] * features.shape[3] == time_steps)
	n := num_assets
	m := batch * time_steps // samples per asset
	x := features.data.data
	at :: #force_inline proc(x: []f64, b, a, s, n, ts: int) -> f64 {return x[(b * n + a) * ts + s]}

	mean := make([]f64, n, allocator)
	defer delete(mean, allocator)
	for a in 0 ..< n {
		for b in 0 ..< batch {for s in 0 ..< time_steps {mean[a] += at(x, b, a, s, n, time_steps)}}
		mean[a] /= f64(m)
	}
	cov := make([]f64, n * n, allocator) // centred cross products, upper triangle
	defer delete(cov, allocator)
	for b in 0 ..< batch {
		for s in 0 ..< time_steps {
			for i in 0 ..< n {
				di := at(x, b, i, s, n, time_steps) - mean[i]
				for j in i ..< n {cov[i * n + j] += di * (at(x, b, j, s, n, time_steps) - mean[j])}
			}
		}
	}

	adj_data := l.matrix_new(f64, n, n, allocator)
	A := adj_data.data
	for i in 0 ..< n {
		A[i * n + i] = 1.0
		for j in i + 1 ..< n {
			den := math.sqrt(cov[i * n + i] * cov[j * n + j])
			c := den > 0 ? cov[i * n + j] / den : 0.0
			A[i * n + j] = math.max(0.0, c)
			A[j * n + i] = A[i * n + j]
		}
	}
	for i in 0 ..< n {
		row_sum := 0.0
		for j in 0 ..< n {row_sum += A[i * n + j]}
		for j in 0 ..< n {A[i * n + j] /= row_sum} 	// >= 1 (the self-loop)
	}

	adj := t.tensor_new(adj_data, false, allocator)
	adj.shape = [4]int{n, n, 1, 1}
	return adj
}
