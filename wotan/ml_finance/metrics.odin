package ml_finance

import t "../tensor"
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
