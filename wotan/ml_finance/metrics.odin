package ml_finance

import ml "../analytics/ML"
import fin "../finance"
import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Cross-Sectional Rank IC (Spearman Rank Correlation)
// ============================================================================
// For each cross-section (e.g. each day across all assets) the Spearman correlation
// between predictions and realised returns (fin.spearman_rho: Pearson correlation of
// the average ranks, as scipy.stats.spearmanr). Conventions as alphalens / Qlib:
// assets with a NaN prediction or target are dropped from their cross-section, and a
// cross-section whose IC is undefined (fewer than 2 assets, or constant predictions or
// targets) is left out of the statistics instead of counting as 0.

RankICResult :: struct {
	mean_ic:           f64, // mean IC over the valid periods
	std_ic:            f64, // sample standard deviation of the IC (n - 1)
	icir:              f64, // mean_ic / std_ic (0 when std_ic is 0)
	positive_ic_ratio: f64, // fraction of valid periods with IC > 0
	num_periods:       int, // valid periods
}

// preds, targets: [Batch, Num_Assets, 1, 1] (contiguous); one cross-section per batch row
cross_sectional_rank_ic :: proc(
	preds: ^t.Tensor,
	targets: ^t.Tensor,
	allocator: mem.Allocator,
) -> RankICResult {
	assert(
		preds.shape == targets.shape,
		"cross_sectional_rank_ic: preds and targets differ in shape",
	)
	batch := preds.shape[0]
	num_assets := preds.shape[1]
	assert(
		len(preds.data.data) >= batch * num_assets && len(targets.data.data) >= batch * num_assets,
	)

	ics := make([dynamic]f64, 0, batch, allocator)
	defer delete(ics)
	pred_row := make([]f64, num_assets, allocator)
	targ_row := make([]f64, num_assets, allocator)
	defer {
		delete(pred_row, allocator)
		delete(targ_row, allocator)
	}

	for b in 0 ..< batch {
		m := 0
		for a in 0 ..< num_assets {
			p := preds.data.data[b * num_assets + a]
			y := targets.data.data[b * num_assets + a]
			if math.is_nan(p) || math.is_nan(y) {continue}
			pred_row[m], targ_row[m] = p, y
			m += 1
		}
		if m < 2 || _is_constant(pred_row[:m]) || _is_constant(targ_row[:m]) {continue}
		append(&ics, fin.spearman_rho(pred_row[:m], targ_row[:m], allocator))
	}

	res := RankICResult {
		num_periods = len(ics),
	}
	if len(ics) == 0 {
		res.mean_ic, res.std_ic = math.nan_f64(), math.nan_f64()
		return res
	}
	positive := 0
	for ic in ics {
		res.mean_ic += ic
		if ic > 0 {positive += 1}
	}
	res.mean_ic /= f64(len(ics))
	res.positive_ic_ratio = f64(positive) / f64(len(ics))
	if len(ics) > 1 {
		ss := 0.0
		for ic in ics {ss += (ic - res.mean_ic) * (ic - res.mean_ic)}
		res.std_ic = math.sqrt(ss / f64(len(ics) - 1))
	} else {
		res.std_ic = math.nan_f64()
	}
	if res.std_ic > 0 {res.icir = res.mean_ic / res.std_ic}
	return res
}

@(private = "file")
_is_constant :: proc(x: []f64) -> bool {
	for v in x[1:] {if v != x[0] {return false}}
	return true
}

// ============================================================================
// Blocked K-Fold Cross-Validation of a GAT
// ============================================================================
GATFoldResult :: struct {
	mean_ic:          f64, // mean rank IC on the validation block, model of the best epoch
	val_loss:         f64, // best validation MSE
	early_stop_epoch: int, // epochs trained
}

// The windows are in time order, so the folds are contiguous blocks (no shuffling, which
// would put almost identical overlapping windows into both sets). Training windows within
// `embargo` windows of the validation block are dropped as well (purging, López de Prado
// 2018, ch. 7: ml.CVSplitter's Purged_KFold with label_horizon = embargo); the default -1 means time_steps, enough for windows with stride >= 1 and
// a target just after the window. Full-batch Adam on the training MSE, early stopping on
// the validation MSE with patience 10; the parameters of the best epoch are restored
// before the IC is measured. The adjacency is built from the training windows only.
gat_cross_val_score :: proc(
	features: ^t.Tensor, // [Total_Windows, Num_Assets, Time, 1]
	targets: ^t.Tensor, // [Total_Windows, Num_Assets, 1, 1]
	n_splits: int,
	hidden_dim: int,
	num_heads: int,
	epochs: int,
	learning_rate: f64,
	allocator: mem.Allocator = context.allocator,
	embargo: int = -1,
) -> []GATFoldResult {
	batch := features.shape[0]
	num_assets := features.shape[1]
	time_steps := features.shape[2]
	assert(
		n_splits >= 2 && n_splits <= batch,
		"gat_cross_val_score: need 2 <= n_splits <= windows",
	)
	gap := embargo < 0 ? time_steps : embargo
	PATIENCE :: 10

	results := make([]GATFoldResult, n_splits, allocator)
	// purged K-fold with a label horizon of `gap` windows: training windows within gap of
	// the validation block are dropped
	folds := ml.cv_make_folds(
		ml.CVSplitter{kind = .Purged_KFold, n_splits = n_splits, label_horizon = gap},
		batch,
		allocator = context.temp_allocator,
	)
	defer ml.cv_folds_free(folds, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)

	for fold, k in folds {
		train_idx, val_idx := fold.train, fold.test
		if len(train_idx) == 0 {
			results[k] = {math.nan_f64(), math.nan_f64(), 0}
			continue
		}

		train_feat := _extract_tensor_by_indices(features, train_idx, allocator)
		train_targ := _extract_tensor_by_indices(targets, train_idx, allocator)
		val_feat := _extract_tensor_by_indices(features, val_idx, allocator)
		val_targ := _extract_tensor_by_indices(targets, val_idx, allocator)

		// owned by the layer from here on (gat_layer_free frees it)
		adjacency := compute_correlation_adjacency(train_feat, num_assets, time_steps, allocator)
		gat := nn.gat_layer_new(time_steps, hidden_dim, num_heads, adjacency, allocator)
		pred_head := nn.linear_layer_new(hidden_dim, 1, allocator)
		opt := nn.adam_new(learning_rate, allocator = allocator)

		params := make([dynamic]^t.Tensor, 0, allocator)
		layers := [2]nn.Layer{gat, pred_head}
		for &lay in layers {nn.layer_parameters(&lay, &params)}
		for p in params {nn.adam_add_param(&opt, p)}
		best := make([][]f64, len(params), allocator)
		for p, i in params {best[i] = make([]f64, len(p.data.data), allocator)}

		best_val_loss := math.F64_MAX
		patience_counter := 0
		last_epoch := 0
		for epoch in 0 ..< epochs {
			nn.adam_zero_grad(&opt)
			t.graph_begin(&scope)
			loss := t.tensor_mse_loss(
				nn.linear_forward(&pred_head, nn.gat_layer_forward(&gat, train_feat, adjacency)),
				train_targ,
			)
			t.tensor_backward(loss)
			t.graph_end(&scope)
			nn.adam_step(&opt)

			t.graph_begin(&scope, no_grad = true)
			val_loss :=
				t.tensor_mse_loss(nn.linear_forward(&pred_head, nn.gat_layer_forward(&gat, val_feat, adjacency)), val_targ).data.data[0]
			t.graph_end(&scope)

			last_epoch = epoch
			if val_loss < best_val_loss {
				best_val_loss = val_loss
				patience_counter = 0
				for p, i in params {copy(best[i], p.data.data)}
			} else {
				patience_counter += 1
				if patience_counter >= PATIENCE {break}
			}
		}
		if best_val_loss < math.F64_MAX {
			for p, i in params {copy(p.data.data, best[i])}
		}

		t.graph_begin(&scope, no_grad = true)
		final_preds := nn.linear_forward(
			&pred_head,
			nn.gat_layer_forward(&gat, val_feat, adjacency),
		)
		ic_result := cross_sectional_rank_ic(final_preds, val_targ, allocator)
		t.graph_end(&scope)

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

		for b in best {delete(b, allocator)}
		delete(best, allocator)
		delete(params)
		nn.gat_layer_free(&gat)
		nn.linear_layer_free(&pred_head)
		nn.adam_free(&opt)
		t.tensor_free(train_feat)
		t.tensor_free(train_targ)
		t.tensor_free(val_feat)
		t.tensor_free(val_targ)
	}
	return results
}

// Rows `indices` of a 4D tensor [B, C, T, L], as a new [len(indices), C, T, L] tensor
_extract_tensor_by_indices :: proc(
	src: ^t.Tensor,
	indices: []int,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	new_batch := len(indices)
	c := src.shape[1]
	t_steps := src.shape[2]
	last_dim := src.shape[3]
	row := c * t_steps * last_dim

	out_data := l.matrix_new(f64, 1, new_batch * row, allocator)
	for orig_i, new_i in indices {
		copy(out_data.data[new_i * row:][:row], src.data.data[orig_i * row:][:row])
	}
	out := t.tensor_new(out_data, false, allocator)
	out.shape = [4]int{new_batch, c, t_steps, last_dim}
	return out
}
