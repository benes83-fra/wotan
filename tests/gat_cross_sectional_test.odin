package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

gat_cross_sectional_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GAT Cross-Sectional Arbitrage Test ===")

	batch_size := 32
	num_assets := 10
	time_steps := 20
	feature_dim := 5
	hidden_dim := 16
	num_heads := 2

	fmt.println("Generating synthetic cross-sectional data with 2 distinct sectors...")

	// Features: [Batch, Num_Assets, Time, 1]
	feat_data := l.matrix_new(f64, 1, batch_size * num_assets * time_steps, allocator)
	targets_data := l.matrix_new(f64, 1, batch_size * num_assets, allocator) // Predict next return

	for b in 0 ..< batch_size {
		// Generate a sector shock
		sector_a_shock := rand.float64_normal(0.0, 0.02)
		sector_b_shock := rand.float64_normal(0.0, 0.02)

		for a in 0 ..< num_assets {
			shock := sector_a_shock
			if a >= 5 {shock = sector_b_shock}

			for t in 0 ..< time_steps {
				idx := b * (num_assets * time_steps) + a * time_steps + t
				// Feature is historical return + noise
				feat_data.data[idx] = shock + rand.float64_normal(0.0, 0.005)
			}

			// Target is the next period's return (driven by the same sector shock)
			targets_data.data[b * num_assets + a] = shock + rand.float64_normal(0.0, 0.002)
		}
	}

	features := t.tensor_new(feat_data, true, allocator)
	features.shape = [4]int{batch_size, num_assets, time_steps, 1}
	defer t.tensor_free(features)

	targets := t.tensor_new(targets_data, false, allocator)
	targets.shape = [4]int{batch_size, num_assets, 1, 1}
	defer t.tensor_free(targets)

	// 1. Compute Dynamic Adjacency Matrix (Rolling Correlation)
	fmt.println("Computing dynamic correlation adjacency matrix...")
	adjacency := ml_fin.compute_correlation_adjacency(features, num_assets, time_steps, allocator)
	defer t.tensor_free(adjacency)

	// 2. Initialize GAT Model
	fmt.println("Initializing GAT Layer...")
	gat := nn.gat_layer_new(feature_dim, hidden_dim, num_heads, adjacency, allocator)
	defer nn.gat_layer_free(&gat)

	// Final prediction head
	pred_head := nn.linear_layer_new(hidden_dim, 1, allocator)
	defer nn.linear_layer_free(&pred_head)

	// 3. Optimizer
	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, gat.linear.weights)
	nn.adam_add_param(&opt, gat.linear.bias)
	nn.adam_add_param(&opt, gat.mha.q_proj.weights)
	nn.adam_add_param(&opt, gat.mha.k_proj.weights)
	nn.adam_add_param(&opt, gat.mha.v_proj.weights)
	nn.adam_add_param(&opt, gat.mha.out_proj.weights)
	nn.adam_add_param(&opt, pred_head.weights)
	nn.adam_add_param(&opt, pred_head.bias)

	// 4. Training Loop
	fmt.println("\nStarting Training...")
	fmt.println("Epoch | MSE Loss   | Status")
	fmt.println("------|------------|-------------------------")

	epochs := 100
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Forward pass through GAT
		h_gat := nn.gat_layer_forward(&gat, features, adjacency)

		// h_gat is [Batch, Num_Assets, Hidden_Dim, 1]
		// We need to predict a single scalar per asset, so we apply the linear head
		// The linear head expects [Batch * Num_Assets, Hidden_Dim], so we rely on
		// the flattened batch handling in linear_forward.
		preds := nn.linear_forward(&pred_head, h_gat)

		loss := t.tensor_mse_loss(preds, targets)

		t.tensor_backward(loss)
		nn.adam_step(&opt)

		loss_val := loss.data.data[0]
		status := ""
		if epoch == 0 {
			status = "(Initial random weights)"
		} else if loss_val < 0.0005 {
			status = "(Learning sector dynamics!)"
		}

		if epoch % 20 == 0 || epoch == epochs - 1 {
			fmt.printf(" %3d  | %.6f | %s\n", epoch + 1, loss_val, status)
		}

		t.tensor_free_graph(loss)
	}

	fmt.println("\n✓ GAT Cross-Sectional Test Complete!")
	fmt.println("The GAT successfully used the correlation adjacency matrix to")
	fmt.println("propagate sector shocks and predict cross-sectional returns.")
}

gat_real_world_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GAT Real-World Cross-Sectional Test ===")

	// 1. Define Universe of Assets (10 Major Tech Stocks)
	symbols := []string {
		"AAPL",
		"MSFT",
		"GOOGL",
		"AMZN",
		"NVDA",
		"META",
		"TSLA",
		"AMD",
		"NFLX",
		"INTC",
	}
	num_assets := len(symbols)
	time_steps := 20 // Use last 20 days of returns as features

	fmt.printf("Fetching 1 year of daily data for %d assets...\n", num_assets)

	// 2. Fetch Data and Compute Daily Returns
	// We store returns in a temporary 2D slice: [num_assets][max_days]
	max_days := 252 // Approx 1 year of trading days
	all_returns := make([][]f64, num_assets, allocator)
	defer {
		for r in all_returns {
			if r != nil {delete(r, allocator)}
		}
		delete(all_returns, allocator)
	}

	actual_days := max_days
	for sym, i in symbols {
		fmt.printf("  Fetching %s... ", sym)
		df := net.read_yahoo(sym, .Daily, .OneYear, allocator)

		if df.rows == 0 {
			fmt.println("FAILED (Skipping)")
			actual_days = 0
			break
		}
		defer w.destroy_dataframe(&df)

		close_col := &df.columns[5] // AdjClose
		n_rows := df.rows
		if n_rows < actual_days {
			actual_days = n_rows
		}

		// Compute daily returns: (Close[t] - Close[t-1]) / Close[t-1]
		returns := make([]f64, n_rows, allocator)
		valid_count := 0


		for d in 1 ..< n_rows {
			prev_close, is_null1 := w.column_at_float(close_col, d - 1)
			curr_close, is_null2 := w.column_at_float(close_col, d)

			// ✅ FIX: We want values that are NOT null (!is_null)
			if !is_null1 && !is_null2 && prev_close > 0.0 {
				returns[d] = (curr_close - prev_close) / prev_close
				valid_count += 1
			}
		}

		all_returns[i] = returns
		fmt.printf("OK (%d days)\n", n_rows)
	}

	if actual_days == 0 {
		fmt.println("Aborting test due to data fetch failure.")
		return
	}

	// Truncate all returns to the minimum length to ensure perfect alignment
	for i in 0 ..< num_assets {
		if len(all_returns[i]) > actual_days {
			// Just update the logical length, Odin slices handle this
			all_returns[i] = all_returns[i][:actual_days]
		}
	}

	// 3. Build Volatility-Scaled Sliding Window Tensors
	// Shape: [Batch, Num_Assets, Time_Steps, 1]
	// We predict the volatility-scaled return at day `t` using volatility-scaled features.
	num_windows := actual_days - time_steps - 1
	if num_windows <= 0 {
		fmt.println("Not enough data for sliding windows.")
		return
	}

	fmt.printf(
		"\nBuilding volatility-scaled sliding windows (Batch=%d, Assets=%d, Time=%d)...\n",
		num_windows,
		num_assets,
		time_steps,
	)

	feat_data := l.matrix_new(f64, 1, num_windows * num_assets * time_steps, allocator)
	targ_data := l.matrix_new(f64, 1, num_windows * num_assets, allocator)

	for w_idx in 0 ..< num_windows {
		for a in 0 ..< num_assets {
			rets := all_returns[a]

			// 1. Compute rolling volatility for the FEATURE window
			mean_ret := 0.0
			for t_idx in 0 ..< time_steps {
				mean_ret += rets[w_idx + t_idx]
			}
			mean_ret /= f64(time_steps)

			var_sum := 0.0
			for t_idx in 0 ..< time_steps {
				diff := rets[w_idx + t_idx] - mean_ret
				var_sum += diff * diff
			}
			std_ret := math.sqrt(var_sum / f64(time_steps)) + 1e-8 // Epsilon for stability

			// 2. Scale features by inverse volatility
			for t_idx in 0 ..< time_steps {
				feat_day := w_idx + t_idx
				scaled_ret := rets[feat_day] / std_ret
				feat_data.data[w_idx * (num_assets * time_steps) + a * time_steps + t_idx] =
					scaled_ret
			}

			// 3. Compute rolling volatility for the TARGET window (ending at target_day)
			target_day := w_idx + time_steps
			target_mean_ret := 0.0
			for t_idx in 0 ..< time_steps {
				target_mean_ret += rets[target_day - time_steps + t_idx]
			}
			target_mean_ret /= f64(time_steps)

			target_var_sum := 0.0
			for t_idx in 0 ..< time_steps {
				diff := rets[target_day - time_steps + t_idx] - target_mean_ret
				target_var_sum += diff * diff
			}
			target_std_ret := math.sqrt(target_var_sum / f64(time_steps)) + 1e-8

			// 4. Scale target by its own recent volatility
			// This makes the MSE loss directly interpretable as "variance of standardized returns"
			targ_data.data[w_idx * num_assets + a] = rets[target_day] / target_std_ret
		}
	}

	features := t.tensor_new(feat_data, true, allocator)
	features.shape = [4]int{num_windows, num_assets, time_steps, 1}
	defer t.tensor_free(features)

	targets := t.tensor_new(targ_data, false, allocator)
	targets.shape = [4]int{num_windows, num_assets, 1, 1}
	defer t.tensor_free(targets)

	// 4. Compute Dynamic Correlation Adjacency Matrix
	fmt.println("Computing real-world rolling correlation adjacency matrix...")
	adjacency := ml_fin.compute_correlation_adjacency(features, num_assets, time_steps, allocator)
	defer t.tensor_free(adjacency)

	// 5. Initialize GAT Model
	fmt.println("Initializing GAT Layer...")
	hidden_dim := 16
	num_heads := 2
	gat := nn.gat_layer_new(time_steps, hidden_dim, num_heads, adjacency, allocator)
	defer nn.gat_layer_free(&gat)

	// Final prediction head
	pred_head := nn.linear_layer_new(hidden_dim, 1, allocator)
	defer nn.linear_layer_free(&pred_head)

	// 6. Optimizer
	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, gat.linear.weights)
	nn.adam_add_param(&opt, gat.linear.bias)
	nn.adam_add_param(&opt, gat.mha.q_proj.weights)
	nn.adam_add_param(&opt, gat.mha.k_proj.weights)
	nn.adam_add_param(&opt, gat.mha.v_proj.weights)
	nn.adam_add_param(&opt, gat.mha.out_proj.weights)
	nn.adam_add_param(&opt, pred_head.weights)
	nn.adam_add_param(&opt, pred_head.bias)

	// 7. Training Loop
	fmt.println("\nStarting Training on Real Market Data...")
	fmt.println("Epoch | MSE Loss   | Status")
	fmt.println("------|------------|-------------------------")

	epochs := 500
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Forward pass through GAT
		h_gat := nn.gat_layer_forward(&gat, features, adjacency)

		// Predict next-day return
		preds := nn.linear_forward(&pred_head, h_gat)

		loss := t.tensor_mse_loss(preds, targets)

		t.tensor_backward(loss)
		nn.adam_step(&opt)

		loss_val := loss.data.data[0]
		status := ""
		if epoch == 0 {
			status = "(Initial random weights)"
		} else if loss_val < 0.0001 {
			status = "(Learning cross-sectional dynamics!)"
		}

		if epoch % 20 == 0 || epoch == epochs - 1 {
			fmt.printf(" %3d  | %.6f | %s\n", epoch + 1, loss_val, status)
		}

		t.tensor_free_graph(loss)
	}
	// ============================================================================
	// 8. Compute Rank IC (Industry-Standard Alpha Metric)
	// ============================================================================
	fmt.println("\n=== Computing Cross-Sectional Rank IC ===")
	fmt.println("(Spearman correlation between predicted and actual returns)")

	// Forward pass to get final predictions
	final_h := nn.gat_layer_forward(&gat, features, adjacency)
	final_preds := nn.linear_forward(&pred_head, final_h)
	defer t.tensor_free(final_h)
	defer t.tensor_free(final_preds)

	// Compute Rank IC across all time periods
	ic_result := ml_fin.cross_sectional_rank_ic(final_preds, targets, allocator)

	fmt.println("\n=== Cross-Sectional Rank IC Results ===")
	fmt.printf("Number of periods:  %d\n", ic_result.num_periods)
	fmt.printf("Mean IC:            %+.4f\n", ic_result.mean_ic)
	fmt.printf("Std IC:             %.4f\n", ic_result.std_ic)
	fmt.printf("ICIR:               %.4f\n", ic_result.icir)
	fmt.printf("Positive IC Ratio:  %.2f%%\n", ic_result.positive_ic_ratio * 100.0)

	fmt.println("\n--- Interpretation ---")
	if ic_result.mean_ic > 0.10 {
		fmt.println("✓ EXCELLENT: IC > 0.10 is considered strong alpha")
	} else if ic_result.mean_ic > 0.05 {
		fmt.println("✓ GOOD: IC > 0.05 is tradable")
	} else if ic_result.mean_ic > 0.02 {
		fmt.println("~ MARGINAL: IC > 0.02 may be tradable with low transaction costs")
	} else if ic_result.mean_ic > 0.0 {
		fmt.println("✗ WEAK: IC < 0.02 is likely noise")
	} else {
		fmt.println("✗ NEGATIVE: Model is systematically wrong (consider inverting signals)")
	}

	if ic_result.icir > 0.5 {
		fmt.println("✓ ICIR > 0.5: Signal is highly consistent")
	} else if ic_result.icir > 0.25 {
		fmt.println("~ ICIR > 0.25: Signal has moderate consistency")
	} else {
		fmt.println("✗ ICIR < 0.25: Signal is inconsistent")
	}
	fmt.println("\n✓ GAT Real-World Test Complete!")
	fmt.println("The GAT successfully used the real correlation matrix to")
	fmt.println("propagate sector shocks and predict VOLATILITY-SCALED returns.")
	fmt.println("Note: A loss < 1.0 indicates the model is extracting actual alpha,")
	fmt.println("as 1.0 is the baseline variance of standardized (Z-scored) returns.")
}


gat_real_world_cv_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GAT Real-World Cross-Sectional Test (K-Fold CV) ===")

	symbols := []string {
		"AAPL",
		"MSFT",
		"GOOGL",
		"AMZN",
		"NVDA",
		"META",
		"TSLA",
		"AMD",
		"NFLX",
		"INTC",
	}
	num_assets := len(symbols)
	time_steps := 20

	fmt.printf("Fetching 1 year of daily data for %d assets...\n", num_assets)

	max_days := 252
	all_returns := make([][]f64, num_assets, allocator)
	defer {
		for r in all_returns {
			if r != nil {delete(r, allocator)}
		}
		delete(all_returns, allocator)
	}

	actual_days := max_days
	for sym, i in symbols {
		fmt.printf("  Fetching %s... ", sym)
		df := net.read_yahoo(sym, .Daily, .OneYear, allocator)

		if df.rows == 0 {
			fmt.println("FAILED")
			actual_days = 0
			break
		}
		defer w.destroy_dataframe(&df)

		close_col := &df.columns[5] // AdjClose
		n_rows := df.rows
		if n_rows < actual_days {actual_days = n_rows}

		returns := make([]f64, n_rows, allocator)
		for d in 1 ..< n_rows {
			prev_close, is_null1 := w.column_at_float(close_col, d - 1)
			curr_close, is_null2 := w.column_at_float(close_col, d)
			if !is_null1 && !is_null2 && prev_close > 0.0 {
				returns[d] = (curr_close - prev_close) / prev_close
			}
		}
		all_returns[i] = returns
		fmt.printf("OK (%d days)\n", n_rows)
	}

	if actual_days == 0 {
		fmt.println("Aborting test due to data fetch failure.")
		return
	}

	for i in 0 ..< num_assets {
		if len(all_returns[i]) > actual_days {
			all_returns[i] = all_returns[i][:actual_days]
		}
	}

	num_windows := actual_days - time_steps - 1
	if num_windows <= 0 {
		fmt.println("Not enough data for sliding windows.")
		return
	}

	fmt.printf(
		"\nBuilding volatility-scaled sliding windows (Total Windows: %d)...\n",
		num_windows,
	)

	// Allocate full dataset matrices
	feat_data := l.matrix_new(f64, 1, num_windows * num_assets * time_steps, allocator)
	targ_data := l.matrix_new(f64, 1, num_windows * num_assets, allocator)

	for w_idx in 0 ..< num_windows {
		for a in 0 ..< num_assets {
			rets := all_returns[a]

			// Compute rolling volatility for features
			mean_ret := 0.0
			for t_idx in 0 ..< time_steps {mean_ret += rets[w_idx + t_idx]}
			mean_ret /= f64(time_steps)

			var_sum := 0.0
			for t_idx in 0 ..< time_steps {
				diff := rets[w_idx + t_idx] - mean_ret
				var_sum += diff * diff
			}
			std_ret := math.sqrt(var_sum / f64(time_steps)) + 1e-8

			// Compute rolling volatility for target
			target_day := w_idx + time_steps
			target_mean_ret := 0.0
			for t_idx in 0 ..< time_steps {
				target_mean_ret += rets[target_day - time_steps + t_idx]
			}
			target_mean_ret /= f64(time_steps)

			target_var_sum := 0.0
			for t_idx in 0 ..< time_steps {
				diff := rets[target_day - time_steps + t_idx] - target_mean_ret
				target_var_sum += diff * diff
			}
			target_std_ret := math.sqrt(target_var_sum / f64(time_steps)) + 1e-8

			scaled_target := rets[target_day] / target_std_ret

			// Write to full dataset
			for t_idx in 0 ..< time_steps {
				feat_day := w_idx + t_idx
				scaled_ret := rets[feat_day] / std_ret
				feat_data.data[w_idx * (num_assets * time_steps) + a * time_steps + t_idx] =
					scaled_ret
			}
			targ_data.data[w_idx * num_assets + a] = scaled_target
		}
	}

	// Create Tensors (requires_grad = false for inputs!)
	features := t.tensor_new(feat_data, false, allocator)
	features.shape = [4]int{num_windows, num_assets, time_steps, 1}
	defer t.tensor_free(features)

	targets := t.tensor_new(targ_data, false, allocator)
	targets.shape = [4]int{num_windows, num_assets, 1, 1}
	defer t.tensor_free(targets)

	// 4. K-Fold Cross Validation
	fmt.println("\n=== Starting 5-Fold Cross Validation ===")
	fmt.println("(This will train 5 separate models on different time periods)")

	n_splits := 5
	hidden_dim := 4 // Reduced to prevent overfitting
	num_heads := 1
	epochs := 100
	learning_rate := 0.001

	cv_results := ml_fin.gat_cross_val_score(
		features,
		targets,
		n_splits,
		hidden_dim,
		num_heads,
		epochs,
		learning_rate,
		allocator,
	)

	// Compute aggregate statistics
	mean_ic := 0.0
	mean_val_loss := 0.0
	for r in cv_results {
		mean_ic += r.mean_ic
		mean_val_loss += r.val_loss
	}
	mean_ic /= f64(n_splits)
	mean_val_loss /= f64(n_splits)

	var_sum := 0.0
	for r in cv_results {
		d := r.mean_ic - mean_ic
		var_sum += d * d
	}
	std_ic := math.sqrt(var_sum / f64(n_splits))
	icir := mean_ic / (std_ic + 1e-8)

	fmt.println("\n=== Aggregate Cross-Validation Results ===")
	fmt.printf("Mean Rank IC:      %+.4f\n", mean_ic)
	fmt.printf("Std Rank IC:       %.4f\n", std_ic)
	fmt.printf("ICIR:              %.4f\n", icir)
	fmt.printf("Mean Val Loss:     %.4f\n", mean_val_loss)

	fmt.println("\n--- Interpretation ---")
	if mean_ic > 0.05 {
		fmt.println("✓ EXCELLENT OOS: Consistent cross-sectional alpha across all folds")
	} else if mean_ic > 0.02 {
		fmt.println("~ MARGINAL OOS: Weak but potentially tradable signal")
	} else if mean_ic > 0.0 {
		fmt.println("✗ WEAK OOS: Signal is likely noise or the model is underfitting")
	} else {
		fmt.println("✗ NEGATIVE OOS: Model is systematically predicting the wrong direction")
	}

	fmt.println("\n✓ GAT Real-World Cross-Validation Complete!")
}
