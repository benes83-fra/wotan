package ml_finance

import w "../core"
import l "../linalg"
import nn "../nn"
import p "../plot"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// ============================================================================
// Deep Hedging Configuration
// ============================================================================

RiskMeasure :: enum {
	Variance, // Minimizes variance of the terminal PnL
	CVaR, // Conditional Value at Risk (Expected Shortfall)
}

DeepHedgerConfig :: struct {
	state_size:       int, // Number of features in the state (e.g., 3: spot, time, vol)
	hidden_size:      int, // Hidden layer size for the MLP
	num_layers:       int, // Number of hidden layers
	risk_measure:     RiskMeasure,
	cvar_alpha:       f64, // Alpha for CVaR (e.g., 0.05 for 95% CVaR)
	transaction_cost: f64,
	num_assets:       int, // Kept for config, but network outputs 1 for standard training
}

// ✅ REVERTED: Single network field to match all existing tests
DeepHedger :: struct {
	network:   ^nn.Sequential,
	config:    DeepHedgerConfig,
	allocator: mem.Allocator,
}

// ============================================================================
// Initialization
// ============================================================================
deep_hedger_new :: proc(
	config: DeepHedgerConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^DeepHedger {
	hedger := new(DeepHedger, allocator)
	hedger.config = config
	hedger.allocator = allocator

	seq := nn.sequential_new(allocator)
	nn.sequential_add(seq, nn.linear_layer_new(config.state_size, config.hidden_size))
	nn.sequential_add(seq, nn.Activation.ReLU)

	for _ in 1 ..< config.num_layers {
		nn.sequential_add(seq, nn.linear_layer_new(config.hidden_size, config.hidden_size))
		nn.sequential_add(seq, nn.Activation.ReLU)
	}
	nn.sequential_add(seq, nn.linear_layer_new(config.hidden_size, config.num_assets))

	// ✅ FIX: Output layer must predict delta for EACH asset, so size = num_assets
	hedger.network = seq
	return hedger
}

deep_hedger_free :: proc(hedger: ^DeepHedger) {
	if hedger.network != nil {
		nn.sequential_free(hedger.network)
	}
	free(hedger, hedger.allocator)
}

// ============================================================================
// Training Step (Single Asset)
// ============================================================================

// deep_hedger_train_step :: proc(
// 	hedger: ^DeepHedger,
// 	paths: ^t.Tensor,
// 	payoffs: ^t.Tensor,
// 	opt: ^nn.Adam,
// ) -> f64 {
// 	batch_size := paths.shape[0]
// 	seq_len := paths.shape[1]
// 	state_size := paths.shape[2]
// 	alloc := hedger.allocator

// 	pnl_data := l.matrix_new(f64, batch_size, 1, alloc)
// 	for i in 0 ..< batch_size {pnl_data.data[i] = 0.0}
// 	pnl := t.tensor_new(pnl_data, false, alloc)
// 	pnl.shape = [4]int{batch_size, 1, 1, 1}

// 	state_data := l.matrix_new(f64, batch_size, state_size, alloc)
// 	state := t.tensor_new(state_data, false, alloc)

// 	s_t_data := l.matrix_new(f64, batch_size, 1, alloc)
// 	s_t := t.tensor_new(s_t_data, false, alloc)

// 	s_prev_data := l.matrix_new(f64, batch_size, 1, alloc)
// 	s_prev := t.tensor_new(s_prev_data, false, alloc)

// 	prev_delta: ^t.Tensor = nil

// 	for t_step in 0 ..< seq_len {
// 		for b in 0 ..< batch_size {
// 			for s in 0 ..< state_size {
// 				src_idx := b * (seq_len * state_size) + t_step * state_size + s
// 				dst_idx := b * state_size + s
// 				state_data.data[dst_idx] = paths.data.data[src_idx]
// 			}
// 		}

// 		// ✅ Uses hedger.network (single network)
// 		delta := nn.sequential_forward(hedger.network, state)

// 		if t_step > 0 {
// 			for b in 0 ..< batch_size {
// 				s_t_data.data[b] =
// 					paths.data.data[b * (seq_len * state_size) + t_step * state_size + 0]
// 				s_prev_data.data[b] =
// 					paths.data.data[b * (seq_len * state_size) + (t_step - 1) * state_size + 0]
// 			}

// 			ds := t.tensor_sub(s_t, s_prev)
// 			product := t.tensor_mul(delta, ds) // [batch_size, num_assets, 1, 1]

// 			// ✅ FIX: Use tensor_sum_dim1 to preserve the autograd graph!
// 			pnl_increment := t.tensor_sum_dim1(product)

// 			if hedger.config.transaction_cost > 0.0 {
// 				d_delta := t.tensor_sub(delta, prev_delta)
// 				pos_part := t.tensor_relu(d_delta)
// 				neg_part := t.tensor_relu(t.tensor_neg(d_delta))
// 				abs_d_delta := t.tensor_add(pos_part, neg_part)
// 				cost_coeff := t.tensor_scale(abs_d_delta, hedger.config.transaction_cost)
// 				cost := t.tensor_mul(cost_coeff, s_t)
// 				pnl_increment = t.tensor_sub(pnl_increment, cost)
// 			}

// 			pnl = t.tensor_add(pnl, pnl_increment)
// 		}
// 		prev_delta = delta
// 	}

// 	pnl = t.tensor_sub(pnl, payoffs)
// 	pnl_sq := t.tensor_mul(pnl, pnl)
// 	loss := t.tensor_mean(pnl_sq)

// 	t.tensor_backward(loss)
// 	nn.adam_step(opt)
// 	loss_val := loss.data.data[0]

// 	t.tensor_free_graph(loss)
// 	l.matrix_free(&state_data)
// 	t.tensor_free(state)
// 	l.matrix_free(&s_t_data)
// 	t.tensor_free(s_t)
// 	l.matrix_free(&s_prev_data)
// 	t.tensor_free(s_prev)

// 	return loss_val
// }

// ============================================================================
// Model Persistence (Save/Load)
// ============================================================================

deep_hedger_save :: proc(hedger: ^DeepHedger, path: string) -> bool {
	dummy_opt := nn.adam_new(0.001, allocator = hedger.allocator)
	defer nn.adam_free(&dummy_opt)

	// ✅ Uses hedger.network
	nn.sequential_add_to_adam(hedger.network, &dummy_opt)
	return nn.save_checkpoint(hedger.network, &dummy_opt, path, 0, hedger.allocator)
}

deep_hedger_load :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^DeepHedger,
	bool,
) {
	loaded_model, loaded_opt, _, ok := nn.load_checkpoint(path, allocator)
	if !ok {
		return nil, false
	}

	if loaded_opt != nil {
		nn.adam_free(loaded_opt)
	}

	hedger := new(DeepHedger, allocator)
	// ✅ Uses hedger.network
	hedger.network = loaded_model
	hedger.allocator = allocator

	return hedger, true
}

// ============================================================================
// Visualization & Evaluation Helpers
// ============================================================================

deep_hedging_plot_pnl_distribution :: proc(
	pnl_values: []f64,
	title: string,
	output_path: string,
	allocator: mem.Allocator,
) -> bool {
	if len(pnl_values) == 0 {return false}
	df := w.dataframe_new(allocator)
	defer w.destroy_dataframe(&df)
	pnl_col := w.column_new("pnl", .Float, len(pnl_values))
	for i in 0 ..< len(pnl_values) {w.append_float(&pnl_col, pnl_values[i])}
	w.add_column(&df, pnl_col)
	config := p.DEFAULT_PLOT_CONFIG
	config.title = title
	config.x_label = "Terminal PnL ($)"
	config.y_label = "Frequency"
	config.bar_color = p.BLUE
	config.show_grid = true
	return p.histogram_png(&df, "pnl", output_path, 50, config, allocator)
}

deep_hedging_plot_delta_trajectory :: proc(
	deltas: []f64,
	title: string,
	output_path: string,
	allocator: mem.Allocator,
) -> bool {
	if len(deltas) == 0 {return false}
	n_steps := len(deltas)
	times := make([]f64, n_steps, allocator)
	defer delete(times, allocator)
	for i in 0 ..< n_steps {times[i] = f64(i) / f64(n_steps)}
	config := p.DEFAULT_PLOT_CONFIG
	config.title = title
	config.x_label = "Time (fraction of maturity)"
	config.y_label = "Delta (hedge ratio)"
	config.point_color = p.RED
	config.line_style = .Solid
	config.show_grid = true
	return p.line_png(times, deltas, output_path, config, allocator)
}

deep_hedging_plot_pnl_vs_spot :: proc(
	spot_prices: []f64,
	pnl_values: []f64,
	title: string,
	output_path: string,
	allocator: mem.Allocator,
) -> bool {
	if len(spot_prices) != len(pnl_values) || len(spot_prices) == 0 {return false}
	df := w.dataframe_new(allocator)
	defer w.destroy_dataframe(&df)
	spot_col := w.column_new("spot", .Float, len(spot_prices))
	pnl_col := w.column_new("pnl", .Float, len(pnl_values))
	for i in 0 ..< len(spot_prices) {
		w.append_float(&spot_col, spot_prices[i])
		w.append_float(&pnl_col, pnl_values[i])
	}
	w.add_column(&df, spot_col)
	w.add_column(&df, pnl_col)
	config := p.DEFAULT_PLOT_CONFIG
	config.title = title
	config.x_label = "Terminal Spot Price ($)"
	config.y_label = "Terminal PnL ($)"
	config.point_color = p.RED
	config.show_grid = true
	return p.scatter_png(&df, "spot", "pnl", output_path, config, allocator)
}

deep_hedging_evaluate :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	allocator: mem.Allocator,
) -> (
	pnl_values: []f64,
	spot_prices: []f64,
) {
	batch_size := paths.shape[0]
	seq_len := paths.shape[1]
	state_size := paths.shape[2]

	pnl_values = make([]f64, batch_size, allocator)
	spot_prices = make([]f64, batch_size, allocator)

	state_data := l.matrix_new(f64, batch_size, state_size, allocator)
	defer l.matrix_free(&state_data)

	prev_delta_data := make([]f64, batch_size, allocator)
	defer delete(prev_delta_data, allocator)

	for t_step in 0 ..< seq_len {
		for b in 0 ..< batch_size {
			for s in 0 ..< state_size {
				src_idx := b * (seq_len * state_size) + t_step * state_size + s
				dst_idx := b * state_size + s
				state_data.data[dst_idx] = paths.data.data[src_idx]
			}
		}

		state := t.tensor_new(state_data, false, allocator)
		delta := nn.sequential_forward(hedger.network, state)
		t.tensor_free(state)

		if t_step > 0 {
			for b in 0 ..< batch_size {
				s_t := paths.data.data[b * (seq_len * state_size) + t_step * state_size + 0]
				s_prev :=
					paths.data.data[b * (seq_len * state_size) + (t_step - 1) * state_size + 0]
				delta_val := delta.data.data[b]
				pnl_increment := delta_val * (s_t - s_prev)

				if hedger.config.transaction_cost > 0.0 {
					d_delta := delta_val - prev_delta_data[b]
					cost := hedger.config.transaction_cost * math.abs(d_delta) * s_t
					pnl_increment -= cost
				}
				pnl_values[b] += pnl_increment
			}
		}

		for b in 0 ..< batch_size {
			prev_delta_data[b] = delta.data.data[b]
		}
		t.tensor_free(delta)
	}

	for b in 0 ..< batch_size {
		pnl_values[b] -= payoffs.data.data[b]
		spot_prices[b] =
			paths.data.data[b * (seq_len * state_size) + (seq_len - 1) * state_size + 0]
	}

	return pnl_values, spot_prices
}

// compute_hedging_loss computes either Variance or a smooth CVaR proxy
compute_hedging_loss :: proc(
	pnl: ^t.Tensor,
	risk_measure: RiskMeasure,
	cvar_alpha: f64,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	if risk_measure == .Variance {
		// Variance proxy: Mean(PnL^2)
		pnl_sq := t.tensor_mul(pnl, pnl)
		return t.tensor_mean(pnl_sq)
	}

	// ✅ Smooth Differentiable CVaR Proxy
	// CVaR is the average of the worst alpha% losses.
	// Sorting breaks the autograd graph, so we use a smooth proxy:
	// CVaR_proxy = Mean(L) + (1/alpha) * Mean(Relu(L - Mean(L) - 2*Std(L)))
	// where L = -PnL (Loss)

	batch_size_f := f64(pnl.shape[0])

	// 1. L = -PnL
	loss_tensor := t.tensor_scale(pnl, -1.0)

	// 2. Mean(L)
	mean_l := t.tensor_mean(loss_tensor)

	// 3. Std(L) = sqrt(Mean((L - Mean(L))^2))
	centered := t.tensor_sub(loss_tensor, mean_l)
	centered_sq := t.tensor_mul(centered, centered)
	var_l := t.tensor_mean(centered_sq)

	// Smooth sqrt to prevent NaN gradients
	eps := t.tensor_new(l.matrix_new(f64, 1, 1, alloc), false, alloc)
	eps.data.data[0] = 1e-6
	std_l := t.tensor_sqrt(t.tensor_add(var_l, eps))

	// 4. Threshold = Mean(L) + 2 * Std(L)
	two := t.tensor_new(l.matrix_new(f64, 1, 1, alloc), false, alloc)
	two.data.data[0] = 2.0
	two_std := t.tensor_mul(two, std_l)
	threshold := t.tensor_add(mean_l, two_std)

	// 5. Relu(L - Threshold)
	excess_loss := t.tensor_sub(loss_tensor, threshold)
	excess_positive := t.tensor_relu(excess_loss)
	mean_excess := t.tensor_mean(excess_positive)

	// 6. CVaR = Mean(L) + (1/alpha) * Mean(Excess)
	inv_alpha := t.tensor_new(l.matrix_new(f64, 1, 1, alloc), false, alloc)
	inv_alpha.data.data[0] = 1.0 / cvar_alpha

	scaled_excess := t.tensor_mul(mean_excess, inv_alpha)
	cvar_loss := t.tensor_add(mean_l, scaled_excess)

	// Cleanup temporary scalars
	t.tensor_free(eps)
	t.tensor_free(two)
	t.tensor_free(inv_alpha)

	return cvar_loss
}
// ============================================================================
// Training Step (Single Asset)
// ============================================================================

deep_hedger_train_step :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	batch_size := paths.shape[0]
	seq_len := paths.shape[1]
	state_size := paths.shape[2]
	alloc := hedger.allocator

	pnl_data := l.matrix_new(f64, batch_size, 1, alloc)
	for i in 0 ..< batch_size {pnl_data.data[i] = 0.0}
	pnl := t.tensor_new(pnl_data, false, alloc)
	pnl.shape = [4]int{batch_size, 1, 1, 1}
	pnl.owned_by_graph = true // Ensure cleanup

	prev_delta: ^t.Tensor = nil

	for t_step in 0 ..< seq_len {
		// ✅ FIX: Allocate fresh state buffer for this time step to preserve autograd history
		state_data := l.matrix_new(f64, batch_size, state_size, alloc)
		for b in 0 ..< batch_size {
			for s in 0 ..< state_size {
				src_idx := b * (seq_len * state_size) + t_step * state_size + s
				dst_idx := b * state_size + s
				state_data.data[dst_idx] = paths.data.data[src_idx]
			}
		}
		state := t.tensor_new(state_data, false, alloc)
		state.owned_by_graph = true

		delta := nn.sequential_forward(hedger.network, state)

		if t_step > 0 {
			s_t_data := l.matrix_new(f64, batch_size, 1, alloc)
			s_prev_data := l.matrix_new(f64, batch_size, 1, alloc)
			for b in 0 ..< batch_size {
				s_t_data.data[b] =
					paths.data.data[b * (seq_len * state_size) + t_step * state_size + 0]
				s_prev_data.data[b] =
					paths.data.data[b * (seq_len * state_size) + (t_step - 1) * state_size + 0]
			}

			s_t := t.tensor_new(s_t_data, false, alloc)
			s_t.owned_by_graph = true
			s_prev := t.tensor_new(s_prev_data, false, alloc)
			s_prev.owned_by_graph = true

			ds := t.tensor_sub(s_t, s_prev)
			product := t.tensor_mul(delta, ds)
			pnl_increment := t.tensor_sum_dim1(product)

			if hedger.config.transaction_cost > 0.0 && prev_delta != nil {
				d_delta := t.tensor_sub(delta, prev_delta)
				pos_part := t.tensor_relu(d_delta)
				neg_part := t.tensor_relu(t.tensor_neg(d_delta))
				abs_d_delta := t.tensor_add(pos_part, neg_part)
				cost_coeff := t.tensor_scale(abs_d_delta, hedger.config.transaction_cost)
				cost := t.tensor_mul(cost_coeff, s_t)
				pnl_increment = t.tensor_sub(pnl_increment, cost)
			}

			pnl = t.tensor_add(pnl, pnl_increment)
		}

		// Detach delta for next step's transaction cost (prevents infinite graph unrolling)
		prev_delta_data := l.matrix_new(f64, batch_size, 1, alloc)
		for i in 0 ..< batch_size {
			prev_delta_data.data[i] = delta.data.data[i]
		}
		prev_delta = t.tensor_new(prev_delta_data, false, alloc)
		prev_delta.owned_by_graph = true
	}

	pnl = t.tensor_sub(pnl, payoffs)
	pnl_sq := t.tensor_mul(pnl, pnl)
	loss := t.tensor_mean(pnl_sq)

	t.tensor_backward(loss)
	nn.adam_step(opt)
	loss_val := loss.data.data[0]

	// This will now safely free the loss graph AND all the leaf nodes we marked as owned_by_graph
	t.tensor_free_graph(loss)

	return loss_val
}

// ============================================================================
// Enhanced Multi-Asset Training Step (with Transaction Costs)
// ============================================================================

deep_hedger_train_step_multi :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	batch_size := paths.shape[0]
	seq_len := paths.shape[1]
	state_size := paths.shape[2]
	num_assets := hedger.config.num_assets
	alloc := hedger.allocator

	pnl_data := l.matrix_new(f64, batch_size, 1, alloc)
	for i in 0 ..< batch_size {pnl_data.data[i] = 0.0}
	pnl := t.tensor_new(pnl_data, false, alloc)
	pnl.shape = [4]int{batch_size, 1, 1, 1}
	pnl.owned_by_graph = true

	prev_delta: ^t.Tensor = nil

	for t_step in 0 ..< seq_len {
		// ✅ FIX: Allocate fresh state buffer for this time step
		state_data := l.matrix_new(f64, batch_size, state_size, alloc)
		for b in 0 ..< batch_size {
			for s in 0 ..< state_size {
				src_idx := b * (seq_len * state_size) + t_step * state_size + s
				dst_idx := b * state_size + s
				state_data.data[dst_idx] = paths.data.data[src_idx]
			}
		}
		state := t.tensor_new(state_data, false, alloc)
		state.owned_by_graph = true

		delta := nn.sequential_forward(hedger.network, state)

		if t_step > 0 {
			ds_data := l.matrix_new(f64, batch_size, num_assets, alloc)
			s_t_data := l.matrix_new(f64, batch_size, num_assets, alloc)

			for b in 0 ..< batch_size {
				for asset in 0 ..< num_assets {
					s_t_val :=
						paths.data.data[b * (seq_len * state_size) + t_step * state_size + asset]
					s_prev_val :=
						paths.data.data[b * (seq_len * state_size) + (t_step - 1) * state_size + asset]

					ds_data.data[b * num_assets + asset] = s_t_val - s_prev_val
					s_t_data.data[b * num_assets + asset] = s_t_val
				}
			}

			ds := t.tensor_new(ds_data, false, alloc)
			ds.shape = [4]int{batch_size, num_assets, 1, 1}
			ds.owned_by_graph = true

			s_t := t.tensor_new(s_t_data, false, alloc)
			s_t.shape = [4]int{batch_size, num_assets, 1, 1}
			s_t.owned_by_graph = true

			product := t.tensor_mul(delta, ds)
			pnl_increment := t.tensor_sum_dim1(product)

			if hedger.config.transaction_cost > 0.0 && prev_delta != nil {
				d_delta := t.tensor_sub(delta, prev_delta)
				pos_part := t.tensor_relu(d_delta)
				neg_part := t.tensor_relu(t.tensor_neg(d_delta))
				abs_d_delta := t.tensor_add(pos_part, neg_part)

				cost_coeff := t.tensor_scale(abs_d_delta, hedger.config.transaction_cost)
				cost := t.tensor_mul(cost_coeff, s_t)
				cost_sum := t.tensor_sum_dim1(cost)

				pnl_increment = t.tensor_sub(pnl_increment, cost_sum)
			}

			pnl = t.tensor_add(pnl, pnl_increment)
		}

		// Detach delta for next step
		prev_delta_data := l.matrix_new(f64, batch_size, num_assets, alloc)
		for i in 0 ..< batch_size * num_assets {
			prev_delta_data.data[i] = delta.data.data[i]
		}
		prev_delta = t.tensor_new(prev_delta_data, false, alloc)
		prev_delta.shape = [4]int{batch_size, num_assets, 1, 1}
		prev_delta.owned_by_graph = true
	}

	pnl = t.tensor_sub(pnl, payoffs)
	loss := compute_hedging_loss(pnl, hedger.config.risk_measure, hedger.config.cvar_alpha, alloc)

	t.tensor_backward(loss)
	nn.adam_step(opt)
	loss_val := loss.data.data[0]

	t.tensor_free_graph(loss)

	return loss_val
}
