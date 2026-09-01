package ml_finance

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// Dynamic Graph Construction for Cross-Sectional Finance
// ============================================================================

// compute_correlation_adjacency computes a pairwise correlation matrix
// from a feature tensor of shape [Batch, Num_Assets, Time_Steps, 1].
// It returns a normalized adjacency matrix [Num_Assets, Num_Assets, 1, 1] where
// A[i, j] = max(0, correlation(i, j)), representing positive relational strength.
compute_correlation_adjacency :: proc(
	features: ^t.Tensor, // [Batch, Num_Assets, Time, 1]
	num_assets: int,
	time_steps: int,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	adj_data := l.matrix_new(f64, num_assets, num_assets, allocator)

	// Temporary buffers for SIMD correlation computation
	x_i := make([]f64, time_steps, allocator)
	x_j := make([]f64, time_steps, allocator)
	defer {
		delete(x_i, allocator)
		delete(x_j, allocator)
	}

	for i in 0 ..< num_assets {
		for j in 0 ..< num_assets {
			if i == j {
				adj_data.data[i * num_assets + j] = 1.0
				continue
			}

			// Extract time series for asset i and j (using batch 0 for simplicity)
			for t in 0 ..< time_steps {
				x_i[t] = features.data.data[0 * (num_assets * time_steps) + i * time_steps + t]
				x_j[t] = features.data.data[0 * (num_assets * time_steps) + j * time_steps + t]
			}

			// Compute Pearson correlation
			mean_i := l.sum_simd(x_i) / f64(time_steps)
			mean_j := l.sum_simd(x_j) / f64(time_steps)

			cov := 0.0
			var_i := 0.0
			var_j := 0.0

			for t in 0 ..< time_steps {
				diff_i := x_i[t] - mean_i
				diff_j := x_j[t] - mean_j
				cov += diff_i * diff_j
				var_i += diff_i * diff_i
				var_j += diff_j * diff_j
			}

			std_i := math.sqrt(var_i / f64(time_steps))
			std_j := math.sqrt(var_j / f64(time_steps))

			corr := 0.0
			if std_i > 1e-8 && std_j > 1e-8 {
				corr = cov / (f64(time_steps) * std_i * std_j)
			}

			// ReLU the correlation to only keep positive relationships (sector propagation)
			adj_data.data[i * num_assets + j] = math.max(0.0, corr)
		}
	}

	// Row-normalize the adjacency matrix so it acts as a proper aggregation weight
	for i in 0 ..< num_assets {
		row_sum := 0.0
		for j in 0 ..< num_assets {
			row_sum += adj_data.data[i * num_assets + j]
		}
		if row_sum > 1e-8 {
			for j in 0 ..< num_assets {
				adj_data.data[i * num_assets + j] /= row_sum
			}
		}
	}

	adj := t.tensor_new(adj_data, false, allocator)
	adj.shape = [4]int{num_assets, num_assets, 1, 1}
	return adj
}
