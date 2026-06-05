package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Standard Scaler (Z-score normalization)
// ============================================================================

StandardScaler :: struct {
	mean:       []f64,
	std:        []f64,
	inv_std:    []f64, // ✅ Precomputed 1.0 / std to avoid division in hot loops
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
	sum_sq := make([]f64, n_features, allocator)
	std := make([]f64, n_features, allocator)
	inv_std := make([]f64, n_features, allocator)

	temp := make([]f64, n_features, context.temp_allocator)
	defer delete(temp, context.temp_allocator)

	// Initialize mean and sum_sq to 0
	for j in 0 ..< n_features {
		mean[j] = 0.0
		sum_sq[j] = 0.0
	}

	// ✅ Pass 1: Compute means using SIMD row-wise addition (Cache-friendly!)
	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]
		l.vec_add_simd(mean, x_row, mean)
	}
	inv_n := 1.0 / f64(n_samples)
	for j in 0 ..< n_features {
		mean[j] *= inv_n
	}

	// ✅ Pass 2: Compute standard deviations using SIMD
	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]

		// temp = x - mean
		l.vec_sub_simd(x_row, mean, temp)
		// temp = temp^2
		l.vec_mul_simd(temp, temp, temp)
		// sum_sq += temp
		l.vec_add_simd(sum_sq, temp, sum_sq)
	}

	for j in 0 ..< n_features {
		variance := sum_sq[j] * inv_n
		std[j] = math.sqrt(variance) + 1e-8
		inv_std[j] = 1.0 / std[j] // ✅ Precompute inverse for fast transforms
	}

	delete(sum_sq, allocator)

	return StandardScaler {
		mean = mean,
		std = std,
		inv_std = inv_std,
		n_features = n_features,
		allocator = allocator,
	}
}

standard_scaler_transform :: proc(
	scaler: ^StandardScaler,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols
	out := l.matrix_new(f64, n_samples, n_features, allocator)

	temp := make([]f64, n_features, context.temp_allocator)
	defer delete(temp, context.temp_allocator)

	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]
		out_row := out.data[i * n_features:i * n_features + n_features]

		// ✅ SIMD: out = (x - mean) * inv_std
		l.vec_sub_simd(x_row, scaler.mean, temp)
		l.vec_mul_simd(temp, scaler.inv_std, out_row)
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

	temp := make([]f64, n_features, context.temp_allocator)
	defer delete(temp, context.temp_allocator)

	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]
		out_row := out.data[i * n_features:i * n_features + n_features]

		// ✅ SIMD: out = x * std + mean
		l.vec_mul_simd(x_row, scaler.std, temp)
		l.vec_add_simd(temp, scaler.mean, out_row)
	}
	return out
}

standard_scaler_free :: proc(scaler: ^StandardScaler) {
	if len(scaler.mean) > 0 {delete(scaler.mean, scaler.allocator)}
	if len(scaler.std) > 0 {delete(scaler.std, scaler.allocator)}
	if len(scaler.inv_std) > 0 {delete(scaler.inv_std, scaler.allocator)}
}


// ============================================================================
// Min-Max Scaler
// ============================================================================

MinMaxScaler :: struct {
	data_min:   []f64,
	data_max:   []f64,
	scale:      []f64,
	min_val:    []f64,
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

	for j in 0 ..< n_features {
		data_min[j] = math.F64_MAX
		data_max[j] = -math.F64_MAX
	}

	// Note: Min/Max is left as scalar because it requires branch/min-max intrinsics
	// which are highly architecture-specific. It only runs once during fit anyway.
	for i in 0 ..< n_samples {
		for j in 0 ..< n_features {
			val := X.data[i * n_features + j]
			if val < data_min[j] {data_min[j] = val}
			if val > data_max[j] {data_max[j] = val}
		}
	}

	range_min := feature_range[0]
	range_max := feature_range[1]
	for j in 0 ..< n_features {
		data_range := data_max[j] - data_min[j]
		if data_range == 0.0 {data_range = 1.0}

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

	temp := make([]f64, n_features, context.temp_allocator)
	defer delete(temp, context.temp_allocator)

	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]
		out_row := out.data[i * n_features:i * n_features + n_features]

		// ✅ SIMD: out = x * scale + min_val
		l.vec_mul_simd(x_row, scaler.scale, temp)
		l.vec_add_simd(temp, scaler.min_val, out_row)
	}
	return out
}

minmax_scaler_free :: proc(scaler: ^MinMaxScaler) {
	if len(scaler.data_min) > 0 {delete(scaler.data_min, scaler.allocator)}
	if len(scaler.data_max) > 0 {delete(scaler.data_max, scaler.allocator)}
	if len(scaler.scale) > 0 {delete(scaler.scale, scaler.allocator)}
	if len(scaler.min_val) > 0 {delete(scaler.min_val, scaler.allocator)}
}
