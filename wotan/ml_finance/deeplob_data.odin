package ml_finance

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// LOBDataset holds the preprocessed tensors ready for DeepLOB
LOBDataset :: struct {
	features:  ^t.Tensor, // [N, Channels=4, Time=100, Levels=10]
	labels:    []int, // [N]
	count:     int,
	allocator: mem.Allocator,
}

lob_dataset_free :: proc(ds: ^LOBDataset) {
	if ds.features != nil {
		t.tensor_free(ds.features)
	}
	delete(ds.labels, ds.allocator)
	free(ds, ds.allocator)
}

// prepare_lob_data takes raw flattened arrays and builds the 4D tensor + labels
// This assumes you have already parsed the CSV into raw slices using wotan's data module.
prepare_lob_data :: proc(
	raw_prices: []f64, // Flattened: [N * Time * Levels * 2] (Bid then Ask)
	raw_volumes: []f64, // Flattened: [N * Time * Levels * 2]
	raw_labels: []int, // [N]
	time_steps: int,
	price_levels: int,
	allocator: mem.Allocator = context.allocator,
) -> ^LOBDataset {
	n_samples := len(raw_labels)
	channels := 4 // Bid Price, Bid Vol, Ask Price, Ask Vol

	// Allocate flat data for the 4D tensor: [N, C, T, L]
	// Wotan stores this as a 2D matrix under the hood: rows=N, cols=C*T*L
	feat_data := l.matrix_new(f64, n_samples, channels * time_steps * price_levels, allocator)

	stride_t := time_steps * price_levels
	stride_l := price_levels

	for n in 0 ..< n_samples {
		for t_idx in 0 ..< time_steps {
			for l_idx in 0 ..< price_levels {
				// Calculate source indices in the raw flattened arrays
				// Assuming raw data is structured as [sample][time][level]
				src_base := n * (time_steps * price_levels) + t_idx * price_levels + l_idx

				// Channel 0: Bid Price
				feat_data.data[n * (channels * stride_t) + 0 * stride_t + t_idx * stride_l + l_idx] =
					raw_prices[src_base * 2]
				// Channel 1: Bid Volume
				feat_data.data[n * (channels * stride_t) + 1 * stride_t + t_idx * stride_l + l_idx] =
					raw_volumes[src_base * 2]
				// Channel 2: Ask Price
				feat_data.data[n * (channels * stride_t) + 2 * stride_t + t_idx * stride_l + l_idx] =
					raw_prices[src_base * 2 + 1]
				// Channel 3: Ask Volume
				feat_data.data[n * (channels * stride_t) + 3 * stride_t + t_idx * stride_l + l_idx] =
					raw_volumes[src_base * 2 + 1]
			}
		}
	}

	// Create the tensor and set its 4D shape
	features_tensor := t.tensor_new(feat_data, false, allocator)
	features_tensor.shape = [4]int{n_samples, channels, time_steps, price_levels}

	// Copy labels to dataset-owned memory
	labels_out := make([]int, n_samples, allocator)
	copy(labels_out, raw_labels)

	ds := new(LOBDataset, allocator)
	ds.features = features_tensor
	ds.labels = labels_out
	ds.count = n_samples
	ds.allocator = allocator

	return ds
}

// normalize_lob_midprice applies the standard DeepLOB normalization:
// Feature = (Value - Rolling_Mid_Price) / Rolling_Mid_Price
// NOTE: For simplicity in this blueprint, we apply a global mid-price normalization.
// In production, you would use a rolling window (e.g., l.rolling_mean_simd).
normalize_lob_features :: proc(ds: ^LOBDataset) {
	// Implementation would iterate over the time dimension and normalize
	// using wotan's SIMD vector operations (l.vec_sub_simd, l.vec_div_simd)
	fmt.println("Normalization step placeholder (ready for SIMD rolling window)")
}
