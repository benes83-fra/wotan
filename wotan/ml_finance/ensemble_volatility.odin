package ml_finance

import fin "../finance"
import nn "../nn"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// EnsembleVolatilityForecaster combines GARCH and sequence-model forecasts with a convex
// weight w (ensemble_predict), e.g. the MSE-optimal one from compute_optimal_weight.
EnsembleVolatilityForecaster :: struct {
	lstm:            LSTMVolatilityForecaster,
	garch_omega:     f64,
	garch_alpha:     f64,
	garch_beta:      f64,
	ensemble_weight: f64,
	allocator:       mem.Allocator,
}

ensemble_volatility_new :: proc(
	input_size: int,
	hidden_size: int,
	seq_len: int,
	allocator: mem.Allocator = context.allocator,
	seq_backend: nn.SeqBackend = .LSTM, // .LSTM, .GRU or .Mamba
) -> EnsembleVolatilityForecaster {
	e: EnsembleVolatilityForecaster
	e.allocator = allocator

	config := LSTMVolaConfig {
		input_size  = input_size,
		hidden_size = hidden_size,
		seq_len     = seq_len,
		seq_backend = seq_backend,
	}
	e.lstm = lstm_volatility_forecaster_new(config, allocator)
	e.ensemble_weight = 0.5
	return e
}

ensemble_volatility_free :: proc(e: ^EnsembleVolatilityForecaster) {
	lstm_volatility_forecaster_free(&e.lstm)
}

ensemble_add_to_optimizer :: proc(e: ^EnsembleVolatilityForecaster, opt: ^nn.Adam) {
	lstm_volatility_add_to_optimizer(&e.lstm, opt)
}

// One-step-ahead GARCH(1,1) volatilities: out[t] = sigma_t with
//   sigma_0^2 = mean(r^2) (the start ts.garch_fit uses), sigma_{t+1}^2 = omega + alpha r_t^2 + beta sigma_t^2,
// so sigma_t uses returns before t only. With the residuals a fit was made on, this is the
// square root of its conditional_var. Pass demeaned returns (ts.extract_residuals).
garch_recursive_forecast :: proc(
	returns: []f64,
	omega, alpha, beta: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	vol_series := make([]f64, n, allocator)
	if n == 0 {return vol_series}

	var_t := 0.0
	for r in returns {var_t += r * r}
	var_t /= f64(n)
	for i in 0 ..< n {
		vol_series[i] = math.sqrt(var_t)
		var_t = math.max(omega + alpha * returns[i] * returns[i] + beta * var_t, 1e-8) // floor as ts
	}
	return vol_series
}

// w * garch_vol + (1 - w) * lstm_vol, w = ensemble_weight clamped to [0, 1]
ensemble_predict :: proc(e: ^EnsembleVolatilityForecaster, garch_vol, lstm_vol: f64) -> f64 {
	w := clamp(e.ensemble_weight, 0.0, 1.0)
	return w * garch_vol + (1.0 - w) * lstm_vol
}

// argmin over w in [0, 1] of sum (w g + (1 - w) l - y)^2, in closed form:
//   w* = sum (g - l)(y - l) / sum (g - l)^2, clamped to [0, 1] (the objective is a convex
// parabola in w). 0.5 when the inputs are empty, of different length, or g = l everywhere.
compute_optimal_weight :: proc(
	garch_forecasts: []f64,
	lstm_forecasts: []f64,
	actual_vols: []f64,
) -> f64 {
	n := len(actual_vols)
	if n == 0 || len(garch_forecasts) != n || len(lstm_forecasts) != n {
		return 0.5
	}
	num, den := 0.0, 0.0
	for i in 0 ..< n {
		d := garch_forecasts[i] - lstm_forecasts[i]
		num += d * (actual_vols[i] - lstm_forecasts[i])
		den += d * d
	}
	if den == 0 {return 0.5}
	return clamp(num / den, 0.0, 1.0)
}

VRPSignal :: struct {
	vix_annualized: f64, // Current Implied Volatility (VIX) in %
	rv_annualized:  f64, // Forecasted Realized Volatility in %
	vrp:            f64, // IV - RV in vol points (the volatility spread, a proxy for IV^2 - RV^2)
	z_score:        f64, // Historical Z-score of the VRP
	signal:         f64, // Normalized trading signal [-1.0, 1.0]
	recommendation: string, // Actionable advice
}

// compute_vrp calculates the volatility risk premium IV - RV and a trading signal. The
// horizons should match: VIX is a 30-day implied vol, so the forecast should be the
// average daily vol over the next ~21 trading days.
// vix_level: Current VIX close (e.g., 15.5 for 15.5%)
// forecast_rv_daily: The model's forecast for next-day realized volatility (decimal, e.g., 0.008)
// hist_vrp_mean: Historical mean of the VRP (for Z-score calculation)
// hist_vrp_std: Historical standard deviation of the VRP
compute_vrp :: proc(
	vix_level: f64,
	forecast_rv_daily: f64,
	hist_vrp_mean: f64,
	hist_vrp_std: f64,
) -> VRPSignal {
	sig: VRPSignal
	sig.vix_annualized = vix_level

	// Annualize daily RV forecast and convert to percentage
	// daily_vol * sqrt(252) * 100
	sig.rv_annualized = forecast_rv_daily * math.sqrt_f64(252.0) * 100.0

	// Variance Risk Premium = Implied Vol - Realized Vol
	sig.vrp = sig.vix_annualized - sig.rv_annualized

	// Calculate Z-score relative to historical VRP
	if hist_vrp_std > 1e-6 {
		sig.z_score = (sig.vrp - hist_vrp_mean) / hist_vrp_std
	} else {
		sig.z_score = 0.0
	}

	// Map Z-score to a [-1, 1] signal using tanh
	// Z > 1.5 implies IV is significantly overpriced -> Sell Vol (Signal > 0)
	// Z < -1.5 implies IV is significantly underpriced -> Buy Vol (Signal < 0)
	sig.signal = math.tanh(sig.z_score / 1.5)

	// Generate recommendation
	if sig.z_score > 1.5 {
		sig.recommendation = "SELL PREMIUM (Short Volatility / Iron Condor)"
	} else if sig.z_score < -1.5 {
		sig.recommendation = "BUY PREMIUM (Long Volatility / Straddle)"
	} else if sig.z_score > 0.5 {
		sig.recommendation = "LEAN SELL (Slight overpricing)"
	} else if sig.z_score < -0.5 {
		sig.recommendation = "LEAN BUY (Slight underpricing)"
	} else {
		sig.recommendation = "NEUTRAL (Fairly priced)"
	}

	return sig
}

print_vrp_signal :: proc(sig: VRPSignal) {
	fmt.println("\n========================================")
	fmt.println("   VOLATILITY RISK PREMIUM (VRP) SIGNAL   ")
	fmt.println("========================================")
	fmt.printf("Implied Volatility (VIX):   %6.2f%%\n", sig.vix_annualized)
	fmt.printf("Forecast Realized Vol (RV): %6.2f%%\n", sig.rv_annualized)
	fmt.println("----------------------------------------")
	fmt.printf("Variance Risk Premium (VRP):%+6.2f%%\n", sig.vrp)
	fmt.printf("Historical VRP Z-Score:     %+6.2f\n", sig.z_score)
	fmt.printf("Normalized Signal:          %+6.3f  (Range: -1.0 to +1.0)\n", sig.signal)
	fmt.println("----------------------------------------")
	fmt.printf("RECOMMENDATION: %s\n", sig.recommendation)

	if sig.signal > 0.5 {
		fmt.println(
			"  -> Strategy: Sell Strangle / Iron Condor to capture the volatility risk premium.",
		)
	} else if sig.signal < -0.5 {
		fmt.println(
			"  -> Strategy: Buy Straddle / Strangle to profit from an expected volatility expansion.",
		)
	} else {
		fmt.println("  -> Strategy: Delta-neutral theta decay or stay on the sidelines.")
	}
	fmt.println("========================================\n")
}
// VRPBacktestResult holds the complete output of a walk-forward backtest.
VRPBacktestResult :: struct {
	daily_pnl:    []f64,
	equity_curve: []f64,
	positions:    []f64,
	signals:      []f64,
	forecast_rv:  []f64,
	implied_vol:  []f64,
	actual_vol:   []f64,
	total_pnl:    f64,
	sharpe_ratio: f64,
	max_drawdown: f64,
	win_rate:     f64,
	num_trades:   int,
	num_days:     int,
	allocator:    mem.Allocator,
}

vrp_backtest_result_free :: proc(r: ^VRPBacktestResult) {
	if r.daily_pnl != nil {delete(r.daily_pnl, r.allocator)}
	if r.equity_curve != nil {delete(r.equity_curve, r.allocator)}
	if r.positions != nil {delete(r.positions, r.allocator)}
	if r.signals != nil {delete(r.signals, r.allocator)}
	if r.forecast_rv != nil {delete(r.forecast_rv, r.allocator)}
	if r.implied_vol != nil {delete(r.implied_vol, r.allocator)}
	if r.actual_vol != nil {delete(r.actual_vol, r.allocator)}
}

// compute_backtest_metrics from daily_pnl, equity_curve (cumulative P&L) and positions:
// - total_pnl: sum of the daily P&L
// - sharpe_ratio: mean / std (n - 1) * sqrt(252) of the daily P&L (fin.sharpe_ratio_from_returns)
// - max_drawdown: largest fall of the equity from its running peak, in P&L units; the
//   peak starts at the equity before day 0 (equity_curve[0] - daily_pnl[0])
// - num_trades: number of times a new non-zero position is taken (entries and flips)
// - win_rate: share of the days with non-zero P&L that were positive
compute_backtest_metrics :: proc(result: ^VRPBacktestResult) {
	n := len(result.daily_pnl)
	if n == 0 {return}
	assert(
		len(result.equity_curve) == n && len(result.positions) == n,
		"compute_backtest_metrics: series differ in length",
	)

	total := 0.0
	for p in result.daily_pnl {total += p}
	result.total_pnl = total
	result.sharpe_ratio = fin.sharpe_ratio_from_returns(result.daily_pnl)

	peak := result.equity_curve[0] - result.daily_pnl[0]
	max_dd := 0.0
	for e in result.equity_curve {
		peak = math.max(peak, e)
		max_dd = math.max(max_dd, peak - e)
	}
	result.max_drawdown = max_dd

	trades, prev := 0, 0.0
	for x in result.positions {
		if x != 0 && x != prev {trades += 1}
		prev = x
	}
	result.num_trades = trades
	wins, active := 0, 0
	for p in result.daily_pnl {
		if p != 0 {
			active += 1
			if p > 0 {wins += 1}
		}
	}
	result.win_rate = active > 0 ? f64(wins) / f64(active) : 0.0
	result.num_days = n
}

// A boxed summary of the backtest metrics; every row is padded to the box. The edge
// labels only restate the Sharpe ratio (no significance test: see probabilistic_sharpe_ratio)
print_backtest_result :: proc(r: ^VRPBacktestResult) {
	W :: 48 // inner width after the two-space margin
	row :: proc(format: string, args: ..any) {
		fmt.printf("║  %-48s║\n", fmt.tprintf(format, ..args))
	}
	rule :: proc(left, right: string) {
		fmt.print(left)
		for _ in 0 ..< W + 2 {fmt.print("═")}
		fmt.println(right)
	}
	fmt.println()
	rule("╔", "╗")
	row("   WALK-FORWARD VRP BACKTEST RESULTS")
	rule("╠", "╣")
	row("Trading days:        %6d", r.num_days)
	row("Trades (entries):    %6d", r.num_trades)
	row("Total P&L:           %+8.4f (daily vol units)", r.total_pnl)
	row("Sharpe ratio:        %+8.2f (annualized)", r.sharpe_ratio)
	row("Max drawdown:        %8.4f", r.max_drawdown)
	row("Win rate (days):     %6.1f%%", r.win_rate * 100)
	rule("╠", "╣")
	if r.sharpe_ratio > 1.5 {
		row("STRONG EDGE: Sharpe ratio above 1.5")
	} else if r.sharpe_ratio > 0.5 {
		row("MODERATE EDGE: Sharpe ratio above 0.5")
	} else if r.sharpe_ratio > 0.0 {
		row("WEAK EDGE: positive Sharpe ratio")
	} else {
		row("NO EDGE: Sharpe ratio not positive")
	}
	rule("╚", "╝")
}
