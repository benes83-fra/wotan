package ml_finance

import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Trading Orchestrator: Closed-Loop Trading Engine
// ============================================================================
// Integrates: HMM Regime → Ensemble Vol → VRP Signal → HRP Portfolio →
//             Almgren-Chriss Execution → Avellaneda-Stoikov Quotes

RegimeAction :: enum {
	Hold, // Regime is stable, maintain current positions
	Reduce, // Regime is deteriorating, reduce exposure
	Aggressive, // Regime is favorable, increase exposure
	Liquidate, // Regime is in crisis, flatten everything
}

OrchestratorState :: struct {
	// Regime tracking
	current_regime:     int,
	regime_history:     [dynamic]int,

	// Volatility tracking
	index_vol_forecast: f64, // Ensemble forecast (annualized)
	vol_lower_bound:    f64, // Conformal lower bound
	vol_upper_bound:    f64, // Conformal upper bound

	// Signal tracking
	vrp_signal:         f64, // [-1, 1]
	dispersion_signal:  f64, // [-1, 1]

	// Portfolio tracking
	target_weights:     []f64, // HRP target weights
	current_weights:    []f64, // Current portfolio weights
	n_assets:           int,

	// Execution tracking
	shares_to_trade:    []f64, // Shares to execute per asset
	execution_complete: []bool,

	// Market making
	bid_quotes:         []f64,
	ask_quotes:         []f64,

	// PnL tracking
	cumulative_pnl:     f64,
	daily_pnl:          f64,
	day:                int,
	allocator:          mem.Allocator,
}

orchestrator_init :: proc(
	n_assets: int,
	allocator: mem.Allocator = context.allocator,
) -> OrchestratorState {
	state: OrchestratorState
	state.n_assets = n_assets
	state.allocator = allocator

	state.regime_history = make([dynamic]int, 0, allocator)
	state.target_weights = make([]f64, n_assets, allocator)
	state.current_weights = make([]f64, n_assets, allocator)
	state.shares_to_trade = make([]f64, n_assets, allocator)
	state.execution_complete = make([]bool, n_assets, allocator)
	state.bid_quotes = make([]f64, n_assets, allocator)
	state.ask_quotes = make([]f64, n_assets, allocator)

	// Initialize equal weights
	equal_w := 1.0 / f64(n_assets)
	for i in 0 ..< n_assets {
		state.target_weights[i] = equal_w
		state.current_weights[i] = equal_w
	}

	state.current_regime = 1 // Assume Bull regime initially
	state.day = 0

	return state
}

orchestrator_free :: proc(state: ^OrchestratorState) {
	delete(state.regime_history)
	if state.target_weights != nil {delete(state.target_weights, state.allocator)}
	if state.current_weights != nil {delete(state.current_weights, state.allocator)}
	if state.shares_to_trade != nil {delete(state.shares_to_trade, state.allocator)}
	if state.execution_complete != nil {delete(state.execution_complete, state.allocator)}
	if state.bid_quotes != nil {delete(state.bid_quotes, state.allocator)}
	if state.ask_quotes != nil {delete(state.ask_quotes, state.allocator)}
}

// ============================================================================
// Step 1: Regime Detection → Action Mapping
// ============================================================================

regime_to_action :: proc(
	regime: int,
	regime_mean: f64, // Annualized mean of current regime
	regime_vol: f64, // Annualized vol of current regime
) -> RegimeAction {
	// Heuristic mapping based on regime characteristics
	if regime_vol > 0.30 && regime_mean < 0.0 {
		return .Liquidate // High vol + negative mean = crisis
	}
	if regime_vol > 0.25 {
		return .Reduce // High vol regardless of mean = reduce
	}
	if regime_mean > 0.10 && regime_vol < 0.15 {
		return .Aggressive // Strong positive + low vol = go hard
	}
	return .Hold // Default: maintain positions
}

// ============================================================================
// Step 2: Position Sizing based on Regime + Signal
// ============================================================================

compute_position_multiplier :: proc(
	action: RegimeAction,
	vrp_signal: f64,
	confidence: f64, // Conformal confidence (0 to 1)
) -> f64 {
	base := 1.0
	switch action {
	case .Liquidate:
		return 0.0
	case .Reduce:
		base = 0.5
	case .Aggressive:
		base = 1.5
	case .Hold:
		base = 1.0
	}

	// Modulate by VRP signal strength
	// Strong positive VRP signal → increase exposure
	// Strong negative VRP signal → decrease exposure
	signal_adj := 1.0 + 0.3 * vrp_signal

	// Modulate by confidence (Conformal)
	// Lower confidence → reduce exposure
	conf_adj := 0.5 + 0.5 * confidence

	return base * signal_adj * conf_adj
}

// ============================================================================
// Step 3: Compute Required Trades
// ============================================================================

compute_rebalance_trades :: proc(
	state: ^OrchestratorState,
	target_weights: []f64,
	total_capital: f64,
	prices: []f64,
) {
	n := state.n_assets
	current_value := make([]f64, n, state.allocator)
	target_value := make([]f64, n, state.allocator)
	defer {
		delete(current_value, state.allocator)
		delete(target_value, state.allocator)
	}

	for i in 0 ..< n {
		current_value[i] = state.current_weights[i] * total_capital
		target_value[i] = target_weights[i] * total_capital
		state.shares_to_trade[i] = (target_value[i] - current_value[i]) / prices[i]
		state.execution_complete[i] = false
	}
}

// ============================================================================
// Step 4: Generate Market Making Quotes
// ============================================================================

generate_quotes :: proc(
	state: ^OrchestratorState,
	prices: []f64,
	as_params: AvellanedaStoikovParams,
) {
	for i in 0 ..< state.n_assets {
		// Use current portfolio weight as "inventory" proxy
		inventory := state.current_weights[i] * 100.0 // Scale to share-equivalent
		mid := prices[i]
		time_fraction := 0.5 // Mid-day

		quotes := avellaneda_stoikov_quote(mid, inventory, time_fraction, as_params)
		state.bid_quotes[i] = quotes.bid_quote
		state.ask_quotes[i] = quotes.ask_quote
	}
}

// ============================================================================
// Full Trading Cycle
// ============================================================================

OrchestratorDecision :: struct {
	action:         RegimeAction,
	regime:         int,
	vol_forecast:   f64,
	vol_lower:      f64,
	vol_upper:      f64,
	vrp_signal:     f64,
	position_mult:  f64,
	target_weights: []f64,
	trades:         []f64,
	bid_quotes:     []f64,
	ask_quotes:     []f64,
}

orchestrator_step :: proc(
	state: ^OrchestratorState,
	// Inputs from sub-systems
	regime: int,
	regime_mean: f64,
	regime_vol: f64,
	vol_forecast: f64,
	vol_lower: f64,
	vol_upper: f64,
	vrp_signal: f64,
	hrp_weights: []f64,
	prices: []f64,
	total_capital: f64,
	as_params: AvellanedaStoikovParams,
) -> OrchestratorDecision {
	state.day += 1

	// Step 1: Regime → Action
	action := regime_to_action(regime, regime_mean, regime_vol)

	// Step 2: Confidence from Conformal bounds
	// Confidence is inversely proportional to interval width relative to forecast
	interval_width := vol_upper - vol_lower
	confidence := 1.0
	if vol_forecast > 1e-6 {
		confidence = math.max(
			0.2,
			math.min(1.0, 1.0 - interval_width / (2.0 * vol_forecast + 1e-8)),
		)
	}

	// Step 3: Position multiplier
	position_mult := compute_position_multiplier(action, vrp_signal, confidence)

	// Step 4: Adjust target weights by position multiplier
	n := state.n_assets
	// In orchestrator.odin, replace the weight adjustment section:

	// ✅ FIX: Regime-conditional asset allocation
	// Layer 1: Determine equity/bond split based on regime action
	equity_fraction := 0.6 // Default
	switch action {
	case .Aggressive:
		equity_fraction = 0.85 // 85% equities in bull regime
	case .Hold:
		equity_fraction = 0.65 // 65% equities in neutral
	case .Reduce:
		equity_fraction = 0.40 // 40% equities in high-vol
	case .Liquidate:
		equity_fraction = 0.10 // 10% equities in crisis
	}

	// Layer 2: Apply HRP weights within each asset class
	// Identify equity vs bond assets (by convention: last asset is bond proxy)
	n_equity := n - 1 // All except TLT
	n_bond := 1

	// Renormalize HRP weights within equities
	equity_hrp_sum := 0.0
	for i in 0 ..< n_equity {
		equity_hrp_sum += hrp_weights[i]
	}

	adjusted_weights := make([]f64, n, state.allocator)
	defer delete(adjusted_weights, state.allocator)

	for i in 0 ..< n_equity {
		// Equity allocation = equity_fraction * (HRP weight within equities)
		if equity_hrp_sum > 1e-8 {
			adjusted_weights[i] = equity_fraction * (hrp_weights[i] / equity_hrp_sum)
		}
	}
	// Bond allocation = remainder
	adjusted_weights[n - 1] = 1.0 - equity_fraction

	// Apply position multiplier and renormalize
	weight_sum := 0.0
	for i in 0 ..< n {
		adjusted_weights[i] *= position_mult
		weight_sum += adjusted_weights[i]
	}
	if weight_sum > 1e-8 {
		for i in 0 ..< n {
			adjusted_weights[i] /= weight_sum
		}
	}

	// Step 5: Compute trades
	compute_rebalance_trades(state, adjusted_weights, total_capital, prices)

	// Step 6: Generate market making quotes
	generate_quotes(state, prices, as_params)

	// Update state
	state.current_regime = regime
	append(&state.regime_history, regime)
	state.index_vol_forecast = vol_forecast
	state.vol_lower_bound = vol_lower
	state.vol_upper_bound = vol_upper
	state.vrp_signal = vrp_signal

	decision := OrchestratorDecision {
		action         = action,
		regime         = regime,
		vol_forecast   = vol_forecast,
		vol_lower      = vol_lower,
		vol_upper      = vol_upper,
		vrp_signal     = vrp_signal,
		position_mult  = position_mult,
		target_weights = make([]f64, n, state.allocator),
		trades         = make([]f64, n, state.allocator),
		bid_quotes     = make([]f64, n, state.allocator),
		ask_quotes     = make([]f64, n, state.allocator),
	}
	copy(decision.target_weights, adjusted_weights)
	copy(decision.trades, state.shares_to_trade)
	copy(decision.bid_quotes, state.bid_quotes)
	copy(decision.ask_quotes, state.ask_quotes)

	return decision
}

decision_free :: proc(d: ^OrchestratorDecision, allocator: mem.Allocator) {
	if d.target_weights != nil {delete(d.target_weights, allocator)}
	if d.trades != nil {delete(d.trades, allocator)}
	if d.bid_quotes != nil {delete(d.bid_quotes, allocator)}
	if d.ask_quotes != nil {delete(d.ask_quotes, allocator)}
}

// ============================================================================
// Pretty Print
// ============================================================================

print_decision :: proc(d: ^OrchestratorDecision, tickers: []string, day: int) {
	fmt.println(
		"\n╔══════════════════════════════════════════════════════════════╗",
	)
	fmt.printf("║  ORCHESTRATOR DECISION — Day %3d                              ║\n", day)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)

	action_str := "HOLD"
	switch d.action {
	case .Liquidate:
		action_str = "LIQUIDATE"
	case .Reduce:
		action_str = "REDUCE"
	case .Aggressive:
		action_str = "AGGRESSIVE"
	case .Hold:
		action_str = "HOLD"
	}

	fmt.printf(
		"║  Regime: %d | Action: %-10s | Pos. Mult: %.2f            ║\n",
		d.regime,
		action_str,
		d.position_mult,
	)
	fmt.printf(
		"║  Vol Forecast: %.2f%% | CI: [%.2f%%, %.2f%%]                ║\n",
		d.vol_forecast * 100.0,
		d.vol_lower * 100.0,
		d.vol_upper * 100.0,
	)
	fmt.printf(
		"║  VRP Signal: %+.3f | Pos. Multiplier: %.2fx                ║\n",
		d.vrp_signal,
		d.position_mult,
	)

	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.println("║  Asset  | Target Wt | Trade (sh) | Bid      | Ask      ║")
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)

	for i in 0 ..< len(d.target_weights) {
		fmt.printf(
			"║  %-5s | %7.1f%% | %+10.0f | %8.2f | %8.2f ║\n",
			tickers[i],
			d.target_weights[i] * 100.0,
			d.trades[i],
			d.bid_quotes[i],
			d.ask_quotes[i],
		)
	}
	fmt.println(
		"╚══════════════════════════════════════════════════════════════╝",
	)
}
