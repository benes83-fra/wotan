package tests

import w "../wotan/core"
import importer "../wotan/importer"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"

// ============================================================================
// 1. Data Ingestion (Reusing net.http_get and importer.csv_load_from_string)
// ============================================================================

load_real_lob_data :: proc(url: string, allocator: mem.Allocator) -> w.DataFrame {
	fmt.println("Downloading real LOB data via libcurl...")
	fmt.println("URL:", url)

	// 1. Fetch raw text using your existing net module
	text, ok := net.http_get(url, context.temp_allocator)
	if !ok {
		fmt.println("Failed to download data.")
		return w.dataframe_new()
	}

	// 2. The raw TSLA sample has no header. Define it as a constant.
	header := "bid_p1,bid_p2,bid_p3,bid_p4,bid_p5,bid_p6,bid_p7,bid_p8,bid_p9,bid_p10,bid_v1,bid_v2,bid_v3,bid_v4,bid_v5,bid_v6,bid_v7,bid_v8,bid_v9,bid_v10,ask_p1,ask_p2,ask_p3,ask_p4,ask_p5,ask_p6,ask_p7,ask_p8,ask_p9,ask_p10,ask_v1,ask_v2,ask_v3,ask_v4,ask_v5,ask_v6,ask_v7,ask_v8,ask_v9,ask_v10\n"

	// 3. Proper Odin string concatenation for runtime strings
	full_text := strings.join([]string{header, text}, "", allocator)

	// 4. Explicitly define types to bypass inference and ensure robust parsing
	types := make([]w.ColumnType, 40, context.temp_allocator)
	for i in 0 ..< 40 {
		types[i] = .Float
	}

	// 5. Parse using your existing importer
	df := importer.csv_load_from_string(full_text, allocator, types, ',')

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

// ============================================================================
// 2. Data Extraction & Windowing (Reusing core DataFrame API)
// ============================================================================

extract_and_window_lob :: proc(
	df: ^w.DataFrame,
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

	// Extract the first 40 float columns as our LOB features.
	feature_cols: [40]^w.Column
	for j in 0 ..< 40 {
		feature_cols[j] = &df.columns[j]
	}

	for win in 0 ..< n_windows {
		// 1. Extract raw window data into channels
		for t_idx in 0 ..< time_steps {
			row := win + t_idx
			for l_idx in 0 ..< price_levels {
				// Map the 40 columns to the 4 DeepLOB channels sequentially.
				col_idx_0 := l_idx
				col_idx_1 := 10 + l_idx
				col_idx_2 := 20 + l_idx
				col_idx_3 := 30 + l_idx

				v0, _ := w.column_at_float(feature_cols[col_idx_0], row)
				feat_data.data[win * (channels * stride_c) + 0 * stride_c + t_idx * stride_t + l_idx] =
					v0

				v1, _ := w.column_at_float(feature_cols[col_idx_1], row)
				feat_data.data[win * (channels * stride_c) + 1 * stride_c + t_idx * stride_t + l_idx] =
					v1

				v2, _ := w.column_at_float(feature_cols[col_idx_2], row)
				feat_data.data[win * (channels * stride_c) + 2 * stride_c + t_idx * stride_t + l_idx] =
					v2

				v3, _ := w.column_at_float(feature_cols[col_idx_3], row)
				feat_data.data[win * (channels * stride_c) + 3 * stride_c + t_idx * stride_t + l_idx] =
					v3
			}
		}

		// 2. Assign Label (Generate 0, 1, 2 based on mid-price movement)
		price_start, _ := w.column_at_float(feature_cols[0], win)
		price_end, _ := w.column_at_float(feature_cols[0], win + time_steps - 1)
		change := (price_end - price_start) / price_start

		label := 1 // Stationary
		if change > 0.0001 {
			label = 2 // Up
		} else if change < -0.0001 {
			label = 0 // Down
		}
		window_labels[win] = label
	}

	// 3. Create tensor and apply SIMD-optimized Z-score normalization along the time dimension
	features_tensor := t.tensor_new(feat_data, false, allocator)
	features_tensor.shape = [4]int{n_windows, channels, time_steps, price_levels}

	// Apply the new SIMD-optimized normalization
	features_tensor = t.tensor_normalize_time(features_tensor, 1e-8, allocator)

	return features_tensor, window_labels
}

// ============================================================================
// 3. Main Test Orchestrator
// ============================================================================

deeplob_real_data_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== DeepLOB Real Data Pipeline Test ===")

	// Verified public URL for real 10-level Limit Order Book data (TSLA)
	url := "https://raw.githubusercontent.com/thertrader/Using-random-forest-to-model-limit-order-book-dynamic/master/TSLA_2015-01-07_34200000_57600000_orderbook_10_SAMPLE.csv"

	// Load data
	df := load_real_lob_data(url, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Aborting test due to empty DataFrame.")
		return
	}

	// Build windows (Adjusted to 40 steps since the sample file has ~50 rows)
	time_steps := 40
	price_levels := 10
	num_classes := 3
	max_windows := 10

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

	// Train Model
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

	fmt.println("Starting Training on Real LOB Data...")
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

// ============================================================================
// 4. Sharpe Loss Test Orchestrator
// ============================================================================

deeplob_real_data_sharpe_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== DeepLOB Real Data Pipeline Test (Sharpe Loss) ===")

	// Verified public URL for real 10-level Limit Order Book data (TSLA)
	url := "https://raw.githubusercontent.com/thertrader/Using-random-forest-to-model-limit-order-book-dynamic/master/TSLA_2015-01-07_34200000_57600000_orderbook_10_SAMPLE.csv"

	// Load data
	df := load_real_lob_data(url, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Aborting test due to empty DataFrame.")
		return
	}

	// Build windows (Adjusted to 40 steps since the sample file has ~50 rows)
	time_steps := 40
	price_levels := 10
	num_classes := 3
	max_windows := 10

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

	// 1. Create actual returns tensor based on labels: Down=-0.01, Stationary=0.0, Up=0.01
	actual_returns_data := l.matrix_new(f64, batch_size, 1, allocator)
	for i: int = 0; i < batch_size; i += 1 {
		label := window_labels[i]
		ret := 0.0
		if label == 0 {
			ret = -0.01
		} else if label == 2 {
			ret = 0.01
		}
		actual_returns_data.data[i] = ret
	}
	actual_returns := t.tensor_new(actual_returns_data, false, allocator)
	actual_returns.shape = [4]int{batch_size, 1, 1, 1}
	// ✅ FIX: Do NOT set owned_by_graph = true.
	// We explicitly defer cleanup so it survives across all epochs.
	defer t.tensor_free(actual_returns)

	// 2. Create return weights tensor: [Down=-1.0, Stationary=0.0, Up=1.0]
	return_weights_data := l.matrix_new(f64, 3, 1, allocator)
	return_weights_data.data[0] = -1.0
	return_weights_data.data[1] = 0.0
	return_weights_data.data[2] = 1.0
	return_weights := t.tensor_new(return_weights_data, false, allocator)
	return_weights.shape = [4]int{3, 1, 1, 1}
	defer t.tensor_free(return_weights)

	fmt.println("Starting Training with Differentiable Sharpe Loss...")
	fmt.println("Epoch | Loss (Neg Sharpe) | Avg Strategy Ret | Status")
	fmt.println("------|-------------------|------------------|-------------------------")

	epochs := 15
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)
		logits := ml_fin.deeplob_forward(model, features_tensor)

		// 3. Convert logits to probabilities
		probs := t.tensor_softmax(logits)

		// 4. Compute Expected Returns: E[r] = probs @ weights
		expected_returns := t.tensor_matmul(probs, return_weights)

		// 5. Compute Strategy Returns: strategy_r = expected_r * actual_r
		strategy_returns := t.tensor_mul(expected_returns, actual_returns)

		// 6. Calculate Sharpe Loss (Minimizing negative Sharpe = Maximizing Sharpe)
		loss := t.tensor_sharpe_loss(strategy_returns, 0.0, allocator)

		t.tensor_backward(loss)
		nn.adam_step(&opt)

		// Calculate average strategy return for monitoring
		avg_ret := 0.0
		for i: int = 0; i < batch_size; i += 1 {
			avg_ret += strategy_returns.data.data[i]
		}
		avg_ret /= f64(batch_size)

		status := ""
		if epoch == 0 {
			status = "(Initial)"
		} else if avg_ret > 0.0 {
			status = "(Positive Expected Return!)"
		}

		fmt.printf(
			" %3d  | %.5f          | %.6f           | %s\n",
			epoch + 1,
			loss.data.data[0],
			avg_ret,
			status,
		)

		// CRITICAL: Clean up the intermediate graph at the end of the epoch.
		// This safely frees logits, probs, expected_returns, strategy_returns, and loss.
		// It will NOT free actual_returns or return_weights because their op is .None
		// and owned_by_graph is false.
		t.tensor_free_graph(loss)
	}

	fmt.println("\n✓ Sharpe Loss Pipeline Test Complete!")
}
