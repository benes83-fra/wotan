package ml_finance

import w "../core"
import fin "../finance"
import l "../linalg"
import nn "../nn"
import p "../plot"
import t "../tensor"
import "core:mem"

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
	num_assets:       int, // hedging instruments = network outputs (0 counts as 1)
}

DeepHedger :: struct {
	network:   ^nn.Sequential,
	config:    DeepHedgerConfig,
	allocator: mem.Allocator,
	// One arena per training step / evaluated time step, reused across them
	scope:     t.Graph_Scope,
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
	nn.sequential_add(seq, nn.linear_layer_new(config.state_size, config.hidden_size, allocator))
	nn.sequential_add(seq, nn.Activation.ReLU)

	for _ in 1 ..< config.num_layers {
		nn.sequential_add(
			seq,
			nn.linear_layer_new(config.hidden_size, config.hidden_size, allocator),
		)
		nn.sequential_add(seq, nn.Activation.ReLU)
	}
	nn.sequential_add(
		seq,
		nn.linear_layer_new(config.hidden_size, max(config.num_assets, 1), allocator),
	)

	hedger.network = seq
	return hedger
}

deep_hedger_free :: proc(hedger: ^DeepHedger) {
	if hedger.network != nil {
		nn.sequential_free(hedger.network)
	}
	t.graph_scope_destroy(&hedger.scope)
	free(hedger, hedger.allocator)
}

// ============================================================================
// Model Persistence (Save/Load)
// ============================================================================

deep_hedger_save :: proc(hedger: ^DeepHedger, path: string) -> bool {
	dummy_opt := nn.adam_new(0.001, allocator = hedger.allocator)
	defer nn.adam_free(&dummy_opt)

	nn.sequential_add_to_adam(hedger.network, &dummy_opt)
	return nn.save_checkpoint(hedger.network, &dummy_opt, path, 0, hedger.allocator)
}

// The checkpoint holds the network only: pass the configuration it was trained with
// (transaction cost, risk measure, number of assets)
deep_hedger_load :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
	config: DeepHedgerConfig = {num_assets = 1},
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
		free(loaded_opt, allocator) // load_checkpoint allocates the Adam struct itself
	}

	hedger := new(DeepHedger, allocator)
	hedger.network = loaded_model
	hedger.config = config
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

// ============================================================================
// Hedging P&L
// ============================================================================
// paths [batch, N + 1, state_size]: the state at the N + 1 dates t_0 .. t_N, whose first
// num_assets columns are the prices of the hedging instruments in units of the numeraire
// (discounted: with a rate r pass e^{-r t_k} S_k and the discounted payoff). payoffs
// [batch, 1]: the liability paid at t_N. At each t_k, k < N, the network sees the state
// at t_k only and sets the holdings delta_k, kept until t_{k+1}. Terminal P&L of the
// hedger who is short the claim (premium not included; it shifts the P&L by a constant):
//   sum_k delta_k . (S_{k+1} - S_k)  -  cost * sum_k |delta_k - delta_{k-1}| . S_k  -  Z
// with delta_{-1} = 0 and delta_N = 0: positions are opened at t_0 and closed at t_N, and
// every trade pays the proportional cost (Buehler, Gonon, Teichmann & Wood 2019).
@(private = "file")
_hedging_pnl :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	B, S, F := paths.shape[0], paths.shape[1], paths.shape[2]
	A := max(hedger.config.num_assets, 1)
	assert(
		S >= 2 && F >= A,
		"deep hedging: paths need 2+ dates and the prices in the first num_assets columns",
	)
	assert(len(payoffs.data.data) == B, "deep hedging: one payoff per path")
	c := hedger.config.transaction_cost

	pnl := t.tensor_new(l.matrix_new(f64, B, 1, alloc), false, alloc)
	prev: ^t.Tensor = nil
	for k in 0 ..< S - 1 {
		sd := l.matrix_new(f64, B, F, alloc)
		for b in 0 ..< B {copy(sd.data[b * F:][:F], paths.data.data[b * S * F + k * F:][:F])}
		delta := nn.sequential_forward(hedger.network, t.tensor_new(sd, false, alloc))
		assert(
			len(delta.data.data) == B * A,
			"deep hedging: the network must output num_assets holdings",
		)
		ds := _asset_prices(paths, k + 1, A, alloc)
		p_k := _asset_prices(paths, k, A, alloc)
		for &v, i in ds.data {v -= p_k.data[i]}
		gain := t.tensor_mul(delta, t.tensor_new(ds, false, alloc))
		gain.shape = {B, A, 1, 1}
		pnl = t.tensor_add(pnl, t.tensor_sum_dim1(gain))
		if c > 0 {
			trade := delta if prev == nil else t.tensor_sub(delta, prev)
			pnl = t.tensor_sub(pnl, _trade_cost(trade, p_k, c, alloc))
		}
		prev = delta
	}
	if c > 0 { 	// close the positions at t_N
		pnl = t.tensor_sub(pnl, _trade_cost(prev, _asset_prices(paths, S - 1, A, alloc), c, alloc))
	}
	return t.tensor_sub(pnl, payoffs)
}

// Prices of the A hedging instruments at date k, [batch, A]
@(private = "file")
_asset_prices :: proc(paths: ^t.Tensor, k, A: int, alloc: mem.Allocator) -> l.Matrix(f64) {
	B, S, F := paths.shape[0], paths.shape[1], paths.shape[2]
	m := l.matrix_new(f64, B, A, alloc)
	for b in 0 ..< B {copy(m.data[b * A:][:A], paths.data.data[b * S * F + k * F:][:A])}
	return m
}

// cost * sum_a |trade_a| * price_a, [batch, 1]
@(private = "file")
_trade_cost :: proc(
	trade: ^t.Tensor,
	prices: l.Matrix(f64),
	c: f64,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	abs_trade := t.tensor_add(t.tensor_relu(trade), t.tensor_relu(t.tensor_neg(trade)))
	w := t.tensor_mul(abs_trade, t.tensor_new(prices, false, alloc))
	w.shape = {prices.rows, prices.cols, 1, 1}
	return t.tensor_scale(t.tensor_sum_dim1(w), c)
}

// Terminal P&L per path (see _hedging_pnl) and the terminal price of the first asset
deep_hedging_evaluate :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	allocator: mem.Allocator,
) -> (
	pnl_values: []f64,
	spot_prices: []f64,
) {
	B, S, F := paths.shape[0], paths.shape[1], paths.shape[2]
	pnl_values = make([]f64, B, allocator)
	spot_prices = make([]f64, B, allocator)
	t.graph_begin(&hedger.scope, no_grad = true)
	defer t.graph_end(&hedger.scope)
	pnl := _hedging_pnl(hedger, paths, payoffs, t.graph_allocator(&hedger.scope))
	copy(pnl_values, pnl.data.data)
	for b in 0 ..< B {spot_prices[b] = paths.data.data[b * S * F + (S - 1) * F]}
	return
}

// compute_hedging_loss is the training loss of the terminal PnL [batch, 1]:
//   Variance: the variance of the PnL, Mean(PnL^2) - Mean(PnL)^2 (the premium does
//             not change it)
//   CVaR:     the empirical CVaR at level cvar_alpha of the loss -PnL (mean of
//             the worst cvar_alpha * batch losses, Acerbi-Tasche), as
//             -sum c_b PnL_b with the tail weights c of fin.cvar_scenario_weights.
//             The weights are constants of the graph (the sort is not
//             differentiated), so the gradient is exactly the CVaR gradient, -c. CVaR
//             is translation equivariant: the premium lowers it by its amount
compute_hedging_loss :: proc(
	pnl: ^t.Tensor,
	risk_measure: RiskMeasure,
	cvar_alpha: f64,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	if risk_measure == .Variance {
		m := t.tensor_mean(pnl)
		return t.tensor_sub(t.tensor_mean(t.tensor_mul(pnl, pnl)), t.tensor_mul(m, m))
	}

	assert(cvar_alpha > 0 && cvar_alpha <= 1, "compute_hedging_loss: cvar_alpha must be in (0, 1]")
	tail := fin.cvar_scenario_weights(pnl.data.data, 1.0 - cvar_alpha, context.temp_allocator)
	defer delete(tail, context.temp_allocator)
	// A graph input: it must live until backward, so the graph owns it
	neg_c := t.tensor_new(l.matrix_new(f64, pnl.data.rows, pnl.data.cols, alloc), false, alloc)
	neg_c.owned_by_graph = true
	neg_c.shape = pnl.shape
	for c, b in tail {neg_c.data.data[b] = -c}
	return t.tensor_sum(t.tensor_mul(pnl, neg_c))
}

// ============================================================================
// Training Step
// ============================================================================
// One full-batch step on the configured risk measure of the P&L; returns the loss before
// the update. One or several hedging instruments (num_assets holdings per date).
deep_hedger_train_step :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	// The whole step (graph, helper leaves, backward scratch) lives in the hedger's scope
	t.graph_begin(&hedger.scope)
	defer t.graph_end(&hedger.scope)
	alloc := t.graph_allocator(&hedger.scope)
	pnl := _hedging_pnl(hedger, paths, payoffs, alloc)
	loss := compute_hedging_loss(pnl, hedger.config.risk_measure, hedger.config.cvar_alpha, alloc)
	nn.adam_zero_grad(opt)
	t.tensor_backward(loss)
	nn.adam_step(opt)
	return loss.data.data[0]
}

// The same step (kept for existing multi-asset callers)
deep_hedger_train_step_multi :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	return deep_hedger_train_step(hedger, paths, payoffs, opt)
}
