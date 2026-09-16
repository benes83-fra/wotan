package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// EnsembleVolatilityForecaster combines GARCH and LSTM forecasts
// using learned inverse-MSE weights.
EnsembleVolatilityForecaster :: struct {
	lstm:            LSTMVolatilityForecaster, // ✅ Stored by value
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
) -> EnsembleVolatilityForecaster {
	e: EnsembleVolatilityForecaster
	e.allocator = allocator
	e.lstm = lstm_volatility_forecaster_new(input_size, hidden_size, seq_len, allocator)
	e.ensemble_weight = 0.5 // Start with equal weight
	return e
}

ensemble_volatility_free :: proc(e: ^EnsembleVolatilityForecaster) {
	lstm_volatility_forecaster_free(&e.lstm)
}

ensemble_add_to_optimizer :: proc(e: ^EnsembleVolatilityForecaster, opt: ^nn.Adam) {
	nn.adam_add_param(opt, e.lstm.lstm.w_ih)
	nn.adam_add_param(opt, e.lstm.lstm.w_hh)
	nn.adam_add_param(opt, e.lstm.lstm.bias)
	nn.adam_add_param(opt, e.lstm.fc1.weights)
	nn.adam_add_param(opt, e.lstm.fc1.bias)
	nn.adam_add_param(opt, e.lstm.fc2.weights)
	nn.adam_add_param(opt, e.lstm.fc2.bias)
}

garch_recursive_forecast :: proc(
	returns: []f64,
	omega, alpha, beta: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	vol_series := make([]f64, n, allocator)

	unconditional_var := omega / (1.0 - alpha - beta)
	if unconditional_var <= 0.0 || unconditional_var > 1.0 {
		unconditional_var = 0.0001 // Fallback
	}

	var_t := unconditional_var
	for i in 0 ..< n {
		vol_series[i] = math.sqrt_f64(var_t)
		var_t = omega + alpha * returns[i] * returns[i] + beta * var_t
		if var_t < 0.0 {var_t = unconditional_var}
	}
	return vol_series
}

ensemble_predict :: proc(e: ^EnsembleVolatilityForecaster, garch_vol, lstm_vol: f64) -> f64 {
	w := e.ensemble_weight
	if w < 0.0 {w = 0.0}
	if w > 1.0 {w = 1.0}
	return w * garch_vol + (1.0 - w) * lstm_vol
}

compute_optimal_weight :: proc(
	garch_forecasts: []f64,
	lstm_forecasts: []f64,
	actual_vols: []f64,
) -> f64 {
	n := len(actual_vols)
	if n == 0 || len(garch_forecasts) != n || len(lstm_forecasts) != n {
		return 0.5
	}

	best_weight := 0.5
	best_mse := math.F64_MAX

	for step in 0 ..= 100 {
		w := f64(step) / 100.0
		mse := 0.0
		for i in 0 ..< n {
			pred := w * garch_forecasts[i] + (1.0 - w) * lstm_forecasts[i]
			diff := pred - actual_vols[i]
			mse += diff * diff
		}
		mse /= f64(n)
		if mse < best_mse {
			best_mse = mse
			best_weight = w
		}
	}
	return best_weight
}
VRPSignal :: struct {
	vix_annualized: f64, // Current Implied Volatility (VIX) in %
	rv_annualized:  f64, // Forecasted Realized Volatility in %
	vrp:            f64, // Variance Risk Premium (IV - RV) in %
	z_score:        f64, // Historical Z-score of the VRP
	signal:         f64, // Normalized trading signal [-1.0, 1.0]
	recommendation: string, // Actionable advice
}

// compute_vrp calculates the Volatility Risk Premium and generates a trading signal.
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

// compute_backtest_metrics calculates Sharpe, Max Drawdown, and Win Rate
// from a daily PnL series.
compute_backtest_metrics :: proc(result: ^VRPBacktestResult) {
	n := len(result.daily_pnl)
	if n == 0 {return}

	// Total PnL
	total := 0.0
	for p in result.daily_pnl {total += p}
	result.total_pnl = total

	// Sharpe Ratio (annualized, assuming 252 trading days)
	mean_pnl := total / f64(n)
	var_sum := 0.0
	for p in result.daily_pnl {
		d := p - mean_pnl
		var_sum += d * d
	}
	std_pnl := math.sqrt(var_sum / f64(n))
	if std_pnl > 1e-10 {
		result.sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt_f64(252.0)
	} else {
		result.sharpe_ratio = 0.0
	}

	// Max Drawdown
	peak := result.equity_curve[0]
	max_dd := 0.0
	for i in 0 ..< n {
		if result.equity_curve[i] > peak {
			peak = result.equity_curve[i]
		}
		dd := peak - result.equity_curve[i]
		if dd > max_dd {max_dd = dd}
	}
	result.max_drawdown = max_dd

	// Win Rate & Trade Count
	wins := 0
	trades := 0
	for i in 0 ..< n {
		if math.abs(result.positions[i]) > 0.1 {
			trades += 1
			if result.daily_pnl[i] > 0.0 {wins += 1}
		}
	}
	result.num_trades = trades
	if trades > 0 {
		result.win_rate = f64(wins) / f64(trades)
	} else {
		result.win_rate = 0.0
	}
	result.num_days = n
}

print_backtest_result :: proc(r: ^VRPBacktestResult) {
	fmt.println(
		"\n╔══════════════════════════════════════════════════╗",
	)
	fmt.println("║     WALK-FORWARD VRP BACKTEST RESULTS           ║")
	fmt.println(
		"╠══════════════════════════════════════════════════╣",
	)
	fmt.printf("║  Trading Days:        %6d                     ║\n", r.num_days)
	fmt.printf("║  Active Trades:       %6d                     ║\n", r.num_trades)
	fmt.printf("║  Total PnL:           %+8.4f (daily vol units) ║\n", r.total_pnl)
	fmt.printf("║  Sharpe Ratio:        %+8.2f (annualized)     ║\n", r.sharpe_ratio)
	fmt.printf("║  Max Drawdown:        %8.4f                  ║\n", r.max_drawdown)
	fmt.printf("║  Win Rate:            %6.1f%%                   ║\n", r.win_rate * 100)
	fmt.println(
		"╠══════════════════════════════════════════════════╣",
	)

	if r.sharpe_ratio > 1.5 {
		fmt.println("║  ★ STRONG EDGE: Strategy shows significant α   ║")
	} else if r.sharpe_ratio > 0.5 {
		fmt.println("║  ● MODERATE EDGE: Positive expectancy           ║")
	} else if r.sharpe_ratio > 0.0 {
		fmt.println("║  ○ WEAK EDGE: Marginal positive expectancy      ║")
	} else {
		fmt.println("║  ✗ NO EDGE: Strategy needs refinement           ║")
	}
	fmt.println(
		"╚══════════════════════════════════════════════════╝",
	)
}
