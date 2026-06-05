package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Standard Scaler (Z-score normalization)
// Centers to mean=0, scales to unit variance.
// Formula: z = (x - mean) / std
// ============================================================================

StandardScaler :: struct {
	mean:       []f64,
	std:        []f64,
	n_features: int,
	allocator:  mem.Allocator,
}

standard_scaler_fit :: proc(
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> StandardScaler {
	n_samples := X.rows
	n_features := X.cols

	mean := make([]f64, n_features, allocator)
	std := make([]f64, n_features, allocator)

	// Pass 1: Compute means
	for j in 0 ..< n_features {
		sum := 0.0
		for i in 0 ..< n_samples {
			sum += X.data[i * n_features + j]
		}
		mean[j] = sum / f64(n_samples)
	}

	// Pass 2: Compute standard deviations
	for j in 0 ..< n_features {
		sum_sq_diff := 0.0
		for i in 0 ..< n_samples {
			diff := X.data[i * n_features + j] - mean[j]
			sum_sq_diff += diff * diff
		}
		// ✅ CRITICAL: Add epsilon to prevent division by zero for constant features
		std[j] = math.sqrt(sum_sq_diff / f64(n_samples)) + 1e-8
	}

	return StandardScaler{mean = mean, std = std, n_features = n_features, allocator = allocator}
}

standard_scaler_transform :: proc(
	scaler: ^StandardScaler,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols
	if n_features != scaler.n_features {
		panic("standard_scaler_transform: feature mismatch")
	}

	out := l.matrix_new(f64, n_samples, n_features, allocator)

	for i in 0 ..< n_samples {
		for j in 0 ..< n_features {
			idx := i * n_features + j
			out.data[idx] = (X.data[idx] - scaler.mean[j]) / scaler.std[j]
		}
	}
	return out
}

standard_scaler_inverse_transform :: proc(
	scaler: ^StandardScaler,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols
	out := l.matrix_new(f64, n_samples, n_features, allocator)

	for i in 0 ..< n_samples {
		for j in 0 ..< n_features {
			idx := i * n_features + j
			out.data[idx] = X.data[idx] * scaler.std[j] + scaler.mean[j]
		}
	}
	return out
}

standard_scaler_free :: proc(scaler: ^StandardScaler) {
	if len(scaler.mean) > 0 {delete(scaler.mean, scaler.allocator)}
	if len(scaler.std) > 0 {delete(scaler.std, scaler.allocator)}
}


// ============================================================================
// Min-Max Scaler
// Scales features to a given range (default [0, 1]).
// Formula: x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min
// ============================================================================

MinMaxScaler :: struct {
	data_min:   []f64,
	data_max:   []f64,
	scale:      []f64, // Precomputed multiplier
	min_val:    []f64, // Precomputed offset
	n_features: int,
	allocator:  mem.Allocator,
}

minmax_scaler_fit :: proc(
	X: ^l.Matrix(f64),
	feature_range: [2]f64 = [2]f64{0.0, 1.0},
	allocator: mem.Allocator = context.allocator,
) -> MinMaxScaler {
	n_samples := X.rows
	n_features := X.cols

	data_min := make([]f64, n_features, allocator)
	data_max := make([]f64, n_features, allocator)
	scale := make([]f64, n_features, allocator)
	min_val := make([]f64, n_features, allocator)

	// Initialize min/max bounds
	for j in 0 ..< n_features {
		data_min[j] = math.F64_MAX
		data_max[j] = -math.F64_MAX
	}

	// Find actual min/max per column
	for i in 0 ..< n_samples {
		for j in 0 ..< n_features {
			val := X.data[i * n_features + j]
			if val < data_min[j] {data_min[j] = val}
			if val > data_max[j] {data_max[j] = val}
		}
	}

	// Precompute scale and offset for O(1) transformation
	range_min := feature_range[0]
	range_max := feature_range[1]
	for j in 0 ..< n_features {
		data_range := data_max[j] - data_min[j]

		// ✅ CRITICAL: Prevent div by zero for constant features
		if data_range == 0.0 {
			data_range = 1.0
		}

		scale[j] = (range_max - range_min) / data_range
		min_val[j] = range_min - data_min[j] * scale[j]
	}

	return MinMaxScaler {
		data_min = data_min,
		data_max = data_max,
		scale = scale,
		min_val = min_val,
		n_features = n_features,
		allocator = allocator,
	}
}

minmax_scaler_transform :: proc(
	scaler: ^MinMaxScaler,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols
	out := l.matrix_new(f64, n_samples, n_features, allocator)

	for i in 0 ..< n_samples {
		for j in 0 ..< n_features {
			idx := i * n_features + j
			// x * scale + offset is mathematically identical to the full formula, but much faster
			out.data[idx] = X.data[idx] * scaler.scale[j] + scaler.min_val[j]
		}
	}
	return out
}

minmax_scaler_free :: proc(scaler: ^MinMaxScaler) {
	if len(scaler.data_min) > 0 {delete(scaler.data_min, scaler.allocator)}
	if len(scaler.data_max) > 0 {delete(scaler.data_max, scaler.allocator)}
	if len(scaler.scale) > 0 {delete(scaler.scale, scaler.allocator)}
	if len(scaler.min_val) > 0 {delete(scaler.min_val, scaler.allocator)}
}
