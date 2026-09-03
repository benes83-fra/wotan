package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Cross-Sectional Rank IC (Spearman Rank Correlation)
// ============================================================================
// The industry-standard metric for evaluating alpha in quantitative finance.
// Computes the Pearson correlation between ranked predictions and ranked
// actual returns for each cross-section (e.g., each day across all assets).

// RankICResult holds the complete IC statistics
RankICResult :: struct {
	mean_ic:           f64, // Average Spearman correlation across periods
	std_ic:            f64, // Standard deviation of IC
	icir:              f64, // Information Ratio = mean_ic / std_ic
	positive_ic_ratio: f64, // % of periods with positive IC
	num_periods:       int,
}

// compute_ranks converts values to ranks (1-based, ties averaged)
// Uses insertion sort (O(n²) but n is small: typically 10-500 assets)
compute_ranks :: proc(values: []f64, allocator: mem.Allocator) -> []f64 {
	n := len(values)
	ranks := make([]f64, n, allocator)

	// Create index array for sorting
	indices := make([]int, n, allocator)
	for i in 0 ..< n {
		indices[i] = i
	}

	// Insertion sort (stable, handles small n efficiently)
	for i in 1 ..< n {
		key := indices[i]
		key_val := values[key]
		j := i - 1
		for j >= 0 && values[indices[j]] > key_val {
			indices[j + 1] = indices[j]
			j -= 1
		}
		indices[j + 1] = key
	}

	// Assign ranks, handling ties by averaging
	i := 0
	for i < n {
		j := i
		// Find all tied values
		for j < n && values[indices[j]] == values[indices[i]] {
			j += 1
		}
		// Average rank for ties (1-based)
		avg_rank := (f64(i) + f64(j - 1)) / 2.0 + 1.0
		for k in i ..< j {
			ranks[indices[k]] = avg_rank
		}
		i = j
	}

	delete(indices, allocator)
	return ranks
}

// pearson_correlation computes Pearson correlation between two slices
pearson_correlation :: proc(x: []f64, y: []f64) -> f64 {
	n := f64(len(x))
	if n < 2.0 {
		return 0.0
	}

	sum_x := 0.0
	sum_y := 0.0
	for i in 0 ..< len(x) {
		sum_x += x[i]
		sum_y += y[i]
	}
	mean_x := sum_x / n
	mean_y := sum_y / n

	cov := 0.0
	var_x := 0.0
	var_y := 0.0
	for i in 0 ..< len(x) {
		dx := x[i] - mean_x
		dy := y[i] - mean_y
		cov += dx * dy
		var_x += dx * dx
		var_y += dy * dy
	}

	denom := math.sqrt(var_x * var_y)
	if denom < 1e-12 {
		return 0.0
	}
	return cov / denom
}

// cross_sectional_rank_ic computes Rank IC across all cross-sections
// preds:   [Batch, Num_Assets, 1, 1] - model predictions
// targets: [Batch, Num_Assets, 1, 1] - actual returns
//
// For each batch element (time period), computes the Spearman rank correlation
// between predictions and targets across the Num_Assets dimension.
cross_sectional_rank_ic :: proc(
	preds: ^t.Tensor,
	targets: ^t.Tensor,
	allocator: mem.Allocator,
) -> RankICResult {
	batch := preds.shape[0]
	num_assets := preds.shape[1]

	ics := make([]f64, batch, allocator)
	defer delete(ics, allocator)

	pred_row := make([]f64, num_assets, allocator)
	targ_row := make([]f64, num_assets, allocator)
	defer {
		delete(pred_row, allocator)
		delete(targ_row, allocator)
	}

	positive_count := 0

	for b in 0 ..< batch {
		// Extract cross-section (all assets for this time period)
		for a in 0 ..< num_assets {
			idx := b * num_assets + a
			pred_row[a] = preds.data.data[idx]
			targ_row[a] = targets.data.data[idx]
		}

		// Compute ranks
		pred_ranks := compute_ranks(pred_row, allocator)
		targ_ranks := compute_ranks(targ_row, allocator)

		// Compute IC (Pearson correlation of ranks = Spearman)
		ic := pearson_correlation(pred_ranks, targ_ranks)
		ics[b] = ic

		if ic > 0.0 {
			positive_count += 1
		}

		delete(pred_ranks, allocator)
		delete(targ_ranks, allocator)
	}

	// Compute statistics
	sum_ic := 0.0
	for ic in ics {
		sum_ic += ic
	}
	mean_ic := sum_ic / f64(batch)

	var_sum := 0.0
	for ic in ics {
		d := ic - mean_ic
		var_sum += d * d
	}
	std_ic := math.sqrt(var_sum / f64(batch))

	icir := 0.0
	if std_ic > 1e-12 {
		icir = mean_ic / std_ic
	}

	positive_ratio := f64(positive_count) / f64(batch)

	return RankICResult {
		mean_ic = mean_ic,
		std_ic = std_ic,
		icir = icir,
		positive_ic_ratio = positive_ratio,
		num_periods = batch,
	}
}

GATFoldResult :: struct {
	mean_ic:          f64,
	val_loss:         f64,
	early_stop_epoch: int,
}

gat_cross_val_score :: proc(
	features: ^t.Tensor, // [Total_Windows, Num_Assets, Time, 1]
	targets: ^t.Tensor, // [Total_Windows, Num_Assets, 1, 1]
	n_splits: int,
	hidden_dim: int,
	num_heads: int,
	epochs: int,
	learning_rate: f64,
	allocator: mem.Allocator = context.allocator,
) -> []GATFoldResult {
	batch := features.shape[0]
	num_assets := features.shape[1]
	time_steps := features.shape[2]

	// 1. Generate and shuffle indices
	indices := make([]int, batch, context.temp_allocator)
	for i in 0 ..< batch {indices[i] = i}

	// LCG shuffle (deterministic, reproducible)
	state := u32(42) | 1
	n := batch
	for i := n - 1; i > 0; i -= 1 {
		state = u32(u64(state) * 1664525 + 1013904223)
		j := int(state % u32(i + 1))
		indices[i], indices[j] = indices[j], indices[i]
	}

	results := make([]GATFoldResult, n_splits, allocator)
	fold_size := batch / n_splits
	remainder := batch % n_splits

	for k in 0 ..< n_splits {
		val_start := k * fold_size + math.min(k, remainder)
		val_end := val_start + fold_size
		if k < remainder {val_end += 1}

		val_count := val_end - val_start
		train_count := batch - val_count

		train_idx := make([]int, train_count, context.temp_allocator)
		val_idx := make([]int, val_count, context.temp_allocator)

		t_idx, v_idx := 0, 0
		for i in 0 ..< batch {
			orig_idx := indices[i]
			if orig_idx >= val_start && orig_idx < val_end {
				val_idx[v_idx] = orig_idx
				v_idx += 1
			} else {
				train_idx[t_idx] = orig_idx
				t_idx += 1
			}
		}

		// Extract train and val tensors
		train_feat := _extract_tensor_by_indices(features, train_idx, allocator)
		train_targ := _extract_tensor_by_indices(targets, train_idx, allocator)
		val_feat := _extract_tensor_by_indices(features, val_idx, allocator)
		val_targ := _extract_tensor_by_indices(targets, val_idx, allocator)

		// ✅ Compute adjacency matrix STRICTLY on training data to prevent data leakage
		adjacency := compute_correlation_adjacency(train_feat, num_assets, time_steps, allocator)

		// Initialize Model
		gat := nn.gat_layer_new(time_steps, hidden_dim, num_heads, adjacency, allocator)
		pred_head := nn.linear_layer_new(hidden_dim, 1, allocator)
		opt := nn.adam_new(learning_rate, allocator = allocator)

		nn.adam_add_param(&opt, gat.linear.weights)
		nn.adam_add_param(&opt, gat.linear.bias)
		nn.adam_add_param(&opt, gat.mha.q_proj.weights)
		nn.adam_add_param(&opt, gat.mha.k_proj.weights)
		nn.adam_add_param(&opt, gat.mha.v_proj.weights)
		nn.adam_add_param(&opt, gat.mha.out_proj.weights)
		nn.adam_add_param(&opt, pred_head.weights)
		nn.adam_add_param(&opt, pred_head.bias)

		// Training loop with early stopping
		best_val_loss := math.F64_MAX
		patience := 10
		patience_counter := 0
		last_epoch := 0

		for epoch in 0 ..< epochs {
			nn.adam_zero_grad(&opt)
			h_gat := nn.gat_layer_forward(&gat, train_feat, adjacency)
			preds := nn.linear_forward(&pred_head, h_gat)
			loss := t.tensor_mse_loss(preds, train_targ)
			t.tensor_backward(loss)
			nn.adam_step(&opt)
			t.tensor_free_graph(loss)

			// Evaluate on validation set
			val_h := nn.gat_layer_forward(&gat, val_feat, adjacency)
			val_preds := nn.linear_forward(&pred_head, val_h)
			val_loss := t.tensor_mse_loss(val_preds, val_targ)
			val_loss_val := val_loss.data.data[0]
			t.tensor_free_graph(val_loss)
			t.tensor_free(val_h)
			t.tensor_free(val_preds)

			if val_loss_val < best_val_loss {
				best_val_loss = val_loss_val
				patience_counter = 0
			} else {
				patience_counter += 1
			}
			last_epoch = epoch
			if patience_counter >= patience {break}
		}

		// Final evaluation on val set for Rank IC
		final_val_h := nn.gat_layer_forward(&gat, val_feat, adjacency)
		final_val_preds := nn.linear_forward(&pred_head, final_val_h)

		ic_result := cross_sectional_rank_ic(final_val_preds, val_targ, allocator)

		results[k] = GATFoldResult {
			mean_ic          = ic_result.mean_ic,
			val_loss         = best_val_loss,
			early_stop_epoch = last_epoch + 1,
		}

		fmt.printf(
			"  Fold %d/%d: Mean IC = %+.4f, Val Loss = %.4f (stopped at epoch %d)\n",
			k + 1,
			n_splits,
			results[k].mean_ic,
			results[k].val_loss,
			results[k].early_stop_epoch,
		)

		// Cleanup fold
		t.tensor_free(final_val_h)
		t.tensor_free(final_val_preds)
		t.tensor_free(adjacency)
		nn.gat_layer_free(&gat)
		nn.linear_layer_free(&pred_head)
		nn.adam_free(&opt)
		t.tensor_free(train_feat)
		t.tensor_free(train_targ)
		t.tensor_free(val_feat)
		t.tensor_free(val_targ)
		delete(train_idx, context.temp_allocator)
		delete(val_idx, context.temp_allocator)
	}

	delete(indices, context.temp_allocator)
	return results
}

// Helper: Extract rows from a 4D tensor by indices
// ✅ FIX: Renamed parameter 'l' to 'last_dim' to avoid shadowing linalg package
_extract_tensor_by_indices :: proc(
	src: ^t.Tensor,
	indices: []int,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	new_batch := len(indices)
	c := src.shape[1]
	t_steps := src.shape[2]
	last_dim := src.shape[3]

	out_data := l.matrix_new(f64, 1, new_batch * c * t_steps * last_dim, allocator)

	for orig_i, new_i in indices {
		src_offset := orig_i * (c * t_steps * last_dim)
		dst_offset := new_i * (c * t_steps * last_dim)
		copy(
			out_data.data[dst_offset:dst_offset + c * t_steps * last_dim],
			src.data.data[src_offset:src_offset + c * t_steps * last_dim],
		)
	}

	out := t.tensor_new(out_data, false, allocator)
	out.shape = [4]int{new_batch, c, t_steps, last_dim}
	return out
}
