package tests

import core "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// 1. Data Ingestion (Reusing net and importer modules)
// ============================================================================

load_lob_data :: proc(url: string, allocator: mem.Allocator) -> core.DataFrame {
	fmt.println("Downloading and parsing LOB data via libcurl...")
	fmt.println("URL:", url)

	// Reuse existing net module to fetch and parse CSV directly.
	df := net.read_csv_from_url(url, allocator)

	if df.rows == 0 {
		fmt.println("Warning: Loaded DataFrame has 0 rows.")
	} else {
		fmt.printf("✓ Successfully loaded %d rows with %d columns.\n", df.rows, len(df.columns))
	}
	return df
}

// ============================================================================
// 2. Data Extraction & Windowing (Reusing core DataFrame API)
// ============================================================================

extract_and_window_lob :: proc(
	df: ^core.DataFrame,
	time_steps: int,
	price_levels: int,
	max_windows: int,
	allocator: mem.Allocator,
) -> (
	^t.Tensor,
	[]int,
) {

	n_rows := df.rows
	total_possible := n_rows - time_steps
	n_windows := max_windows
	if total_possible < n_windows {
		n_windows = total_possible
	}

	if n_windows <= 0 {
		fmt.println("Error: Not enough rows to create windows.")
		return nil, nil
	}

	channels := 4
	feat_data := l.matrix_new(f64, n_windows, channels * time_steps * price_levels, allocator)
	window_labels := make([]int, n_windows, allocator)

	stride_c := time_steps * price_levels
	stride_t := price_levels

	// Extract raw data from DataFrame columns using the existing core API.
	// We assume the first 40 columns are Float features, and the 41st is the Int label.
	feature_cols: [40]^core.Column
	for j in 0 ..< 40 {
		feature_cols[j] = &df.columns[j]
	}
	label_col := &df.columns[40]

	for w in 0 ..< n_windows {
		// 1. Extract raw window data into channels
		for t_idx in 0 ..< time_steps {
			row := w + t_idx
			for l_idx in 0 ..< price_levels {
				// Channel 0: Bid Price (cols 0-9)
				v0, _ := core.column_at_float(feature_cols[l_idx], row)
				feat_data.data[w * (channels * stride_c) + 0 * stride_c + t_idx * stride_t + l_idx] =
					v0

				// Channel 1: Bid Volume (cols 10-19)
				v1, _ := core.column_at_float(feature_cols[10 + l_idx], row)
				feat_data.data[w * (channels * stride_c) + 1 * stride_c + t_idx * stride_t + l_idx] =
					v1

				// Channel 2: Ask Price (cols 20-29)
				v2, _ := core.column_at_float(feature_cols[20 + l_idx], row)
				feat_data.data[w * (channels * stride_c) + 2 * stride_c + t_idx * stride_t + l_idx] =
					v2

				// Channel 3: Ask Volume (cols 30-39)
				v3, _ := core.column_at_float(feature_cols[30 + l_idx], row)
				feat_data.data[w * (channels * stride_c) + 3 * stride_c + t_idx * stride_t + l_idx] =
					v3
			}
		}

		// 2. Per-Window Z-Score Normalization (prevents look-ahead bias)
		for c in 0 ..< channels {
			for l in 0 ..< price_levels {
				// Calculate mean
				mean := 0.0
				for t_idx in 0 ..< time_steps {
					idx := w * (channels * stride_c) + c * stride_c + t_idx * stride_t + l
					mean += feat_data.data[idx]
				}
				mean /= f64(time_steps)

				// Calculate std
				var_sum := 0.0
				for t_idx in 0 ..< time_steps {
					idx := w * (channels * stride_c) + c * stride_c + t_idx * stride_t + l
					diff := feat_data.data[idx] - mean
					var_sum += diff * diff
				}
				std := math.sqrt(var_sum / f64(time_steps)) + 1e-8

				// Normalize
				for t_idx in 0 ..< time_steps {
					idx := w * (channels * stride_c) + c * stride_c + t_idx * stride_t + l
					feat_data.data[idx] = (feat_data.data[idx] - mean) / std
				}
			}
		}

		// 3. Assign Label (FI-2010 uses 1, 2, 3. We convert to 0, 1, 2 for cross-entropy)
		label_val, _ := core.column_at_int(label_col, w + time_steps - 1)
		window_labels[w] = label_val - 1
	}

	features_tensor := t.tensor_new(feat_data, false, allocator)
	features_tensor.shape = [4]int{n_windows, channels, time_steps, price_levels}

	return features_tensor, window_labels
}

// ============================================================================
// 3. Main Test Orchestrator
// ============================================================================

deeplob_real_data_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== DeepLOB Real Data Pipeline Test ===")

	// 1. Setup URL for FI-2010 dataset
	url := "https://raw.githubusercontent.com/zbrent/DeepLOB/master/data/Train_Dst_NoAuction_DecPreProcessing.txt"

	// 2. Load data using existing net/importer modules
	df := load_lob_data(url, allocator)
	defer core.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Aborting test due to empty DataFrame.")
		return
	}

	// 3. Build windows
	time_steps := 100
	price_levels := 10
	num_classes := 3
	max_windows := 2000 // Limit memory usage for the test

	fmt.printf("Creating sliding windows (max %d)...\n", max_windows)
	features_tensor, window_labels := extract_and_window_lob(
		&df,
		time_steps,
		price_levels,
		max_windows,
		allocator,
	)
	defer t.tensor_free(features_tensor)
	defer delete(window_labels, allocator)

	if features_tensor == nil {
		fmt.println("Error: Failed to build windows.")
		return
	}

	batch_size := features_tensor.shape[0]
	fmt.printf("Built tensor of shape [%d, 4, %d, %d]\n", batch_size, time_steps, price_levels)

	// 4. Train Model
	fmt.println("Initializing DeepLOB model...")
	config := ml_fin.DeepLOBConfig {
		time_steps   = time_steps,
		price_levels = price_levels,
		num_classes  = num_classes,
		hidden_dim   = 64,
	}
	model := ml_fin.deeplob_new(config, allocator)
	defer ml_fin.deeplob_free(model)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	ml_fin.deeplob_add_to_adam(model, &opt)

	fmt.println("Starting Training on Real Data...")
	fmt.println("Epoch | Loss    | Accuracy | Status")
	fmt.println("------|---------|----------|-------------------------")

	epochs := 15
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)
		logits := ml_fin.deeplob_forward(model, features_tensor)
		loss := t.tensor_cross_entropy_loss(logits, window_labels)

		t.tensor_backward(loss)
		nn.adam_step(&opt)

		// Calculate accuracy
		correct := 0
		for i in 0 ..< batch_size {
			max_val := -math.F64_MAX
			max_idx := 0
			for c in 0 ..< num_classes {
				val := logits.data.data[i * num_classes + c]
				if val > max_val {
					max_val = val
					max_idx = c
				}
			}
			if max_idx == window_labels[i] {
				correct += 1
			}
		}
		acc := f64(correct) / f64(batch_size) * 100.0

		// FIXED: Odin uses "else if" (two words), not "elseif"
		status := ""
		if epoch == 0 {
			status = "(Initial)"
		} else if acc > 60.0 {
			status = "(Learning!)"
		}

		fmt.printf(" %3d  | %.5f | %6.2f%%  | %s\n", epoch + 1, loss.data.data[0], acc, status)
		t.tensor_free_graph(loss)
	}

	fmt.println("\n✓ Real Data Pipeline Test Complete!")
}
