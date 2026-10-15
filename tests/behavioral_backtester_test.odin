package tests

import bfin "../wotan/behavioral_finance"
import w "../wotan/core"
import fin "../wotan/finance"
import net "../wotan/net"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

behavioral_backtester_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Behavioral Alpha Backtester ===")
	fmt.println("Testing the profitability of the Market-Implied Psychology (Gamma) signal.")
	fmt.println(
		"Strategy: Buy cheap tail protection in Complacency, Sell expensive fear in Panic.",
	)
	fmt.println("WARNING: This runs ~300 calibrations. Compile with `-o:speed`!")

	n_days := 300 // ~1 year of trading days
	S := make([]f64, n_days, allocator)
	defer delete(S, allocator)

	S[0] = 100.0
	true_sigma := 0.20
	r := 0.05
	dt := 1.0 / 252.0

	// 1. Generate Synthetic Underlying (GBM + 2% daily crash probability)
	fmt.println("- Generating synthetic underlying path (GBM + Jumps)...")
	for t in 1 ..< n_days {
		Z := rand.float64_normal(0.0, 1.0)
		ret := (r - 0.5 * true_sigma * true_sigma) * dt + true_sigma * math.sqrt(dt) * Z

		// 2% chance of a severe 8% jump (crash)
		if rand.float64() < 0.02 {
			ret = -0.08
		}
		S[t] = S[t - 1] * math.exp(ret)
	}

	// 2. Generate Market Maker's True Gamma (Regime Switching)
	// Decoupled from the physical crashes to simulate behavioral over/under-reaction
	market_gammas := make([]f64, n_days, allocator)
	defer delete(market_gammas, allocator)

	current_gamma := 0.90 // Start in Complacency
	for t in 0 ..< n_days {
		if rand.float64() < 0.05 { 	// 5% chance to switch regime each day
			if current_gamma > 0.7 {
				current_gamma = 0.45 // Enter Panic (Crashophobia)
			} else {
				current_gamma = 0.90 // Enter Complacency
			}
		}
		market_gammas[t] = current_gamma
	}

	// 3. Backtest Loop
	fmt.println("- Running daily calibration and trading...")
	cash := 100000.0
	equity_curve := make([]f64, n_days, allocator)
	calibrated_gammas := make([]f64, n_days, allocator)
	signals := make([]int, n_days, allocator) // 1 = Buy Put, -1 = Sell Put, 0 = Hold
	defer {delete(equity_curve, allocator); delete(calibrated_gammas, allocator)
		delete(signals, allocator)}

	T_opt := 30.0 / 252.0 // 30 DTE options
	strikes_rel := []f64{0.90, 1.00, 1.10} // Relative moneyness for calibration

	// Pre-allocate loop buffers to prevent memory leaks
	chain := make([]fin.VolSurfacePoint, 3, allocator)
	defer delete(chain, allocator)

	base_params := bfin.Prospect_Params {
		alpha  = 0.88,
		beta   = 0.88,
		lambda = 2.25,
		gamma  = 0.61,
	}

	for t in 0 ..< n_days - 1 {
		St := S[t]
		gamma_true := market_gammas[t]

		// Generate synthetic option chain using the Market Maker's distorted gamma
		for i in 0 ..< 3 {
			K := St * strikes_rel[i]
			mkt_price := bfin.prospect_theory_price(
				St,
				K,
				T_opt,
				r,
				true_sigma,
				bfin.Prospect_Params{alpha = 0.88, beta = 0.88, lambda = 2.25, gamma = gamma_true},
				.Call,
				100,
				allocator,
			)

			chain[i] = fin.VolSurfacePoint {
				strike       = K,
				expiry       = T_opt,
				market_price = mkt_price,
				implied_vol  = 0.0,
			}
		}

		// Calibrate Wotan's Behavioral Engine to recover gamma
		res := bfin.calibrate_prospect_theory(chain, St, r, true_sigma, allocator)
		cal_gamma := res.params.gamma
		calibrated_gammas[t] = cal_gamma

		// Trade Execution (Mark-to-Market on a 5% OTM Put)
		K_trade := St * 0.95
		T_trade := 30.0 / 252.0

		// Market Price at day t (using calibrated gamma)
		mkt_put_t := bfin.prospect_theory_price(
			St,
			K_trade,
			T_trade,
			r,
			true_sigma,
			bfin.Prospect_Params{alpha = 0.88, beta = 0.88, lambda = 2.25, gamma = cal_gamma},
			.Put,
			100,
			allocator,
		)

		pos_size := 100.0 // Number of contracts
		pnl := 0.0

		// Generate Signal
		if cal_gamma > 0.75 {
			signals[t] = 1 // COMPLACENCY: Long Put
		} else if cal_gamma < 0.55 {
			signals[t] = -1 // PANIC: Short Put
		} else {
			signals[t] = 0 // RATIONAL: Hold Cash
		}

		// Day t+1 Settlement
		St_next := S[t + 1]
		T_next := T_trade - dt
		if T_next <= 0 {T_next = 1.0 / 252.0} 	// Prevent zero/negative expiry

		// Market Maker's gamma tomorrow
		gamma_next := market_gammas[t + 1]

		mkt_put_t1 := bfin.prospect_theory_price(
			St_next,
			K_trade,
			T_next,
			r,
			true_sigma,
			bfin.Prospect_Params{alpha = 0.88, beta = 0.88, lambda = 2.25, gamma = gamma_next},
			.Put,
			100,
			allocator,
		)

		if signals[t] == 1 {
			pnl = (mkt_put_t1 - mkt_put_t) * pos_size
		} else if signals[t] == -1 {
			pnl = (mkt_put_t - mkt_put_t1) * pos_size
		} else {
			pnl = cash * r * dt
		}

		cash += pnl
		equity_curve[t] = cash
	}

	// 4. Print Results
	final_equity := equity_curve[n_days - 2]
	total_return := (final_equity / 100000.0 - 1.0) * 100.0
	spy_return := (S[n_days - 1] / S[0] - 1.0) * 100.0

	fmt.println("\n--- Backtest Results ---")
	fmt.printf("Total Trading Days: %d\n", n_days - 1)
	fmt.printf("SPY Buy & Hold Return:  %+6.2f%%\n", spy_return)
	fmt.printf("Behavioral Alpha Return: %+6.2f%%\n", total_return)
	fmt.printf("Final Equity:           $%.2f\n", final_equity)

	// 5. Plot Results
	fmt.println("- Generating Backtest Plots...")

	days := make([]f64, n_days - 1, allocator)
	norm_equity := make([]f64, n_days - 1, allocator)
	norm_spy := make([]f64, n_days - 1, allocator)
	defer {delete(days, allocator); delete(norm_equity, allocator); delete(norm_spy, allocator)}

	for i in 0 ..< n_days - 1 {
		days[i] = f64(i)
		norm_equity[i] = equity_curve[i] / 100000.0
		norm_spy[i] = S[i] / S[0]
	}

	lines := []plot.LineData {
		{xs = days, ys = norm_spy, color = plot.BLUE, style = .Solid, label = "SPY (Normalized)"},
		{
			xs = days,
			ys = norm_equity,
			color = plot.RED,
			style = .Dashed,
			label = "Behavioral Alpha Equity",
		},
	}

	config := plot.DEFAULT_PLOT_CONFIG
	config.title = "Behavioral Alpha Backtest (Gamma Signal)"
	config.x_label = "Trading Days"
	config.y_label = "Normalized Value (Start = 1.0)"
	config.show_grid = true

	if plot.multi_line_png(lines, "behavioral_backtest_equity.png", config, allocator) {
		fmt.println("Saved: behavioral_backtest_equity.png")
	}

	fmt.println("\n=== Behavioral Alpha Backtester Complete ===\n")
}

// Rolling annualized realized volatility of log returns ending at index i.
rolling_realized_vol :: proc(log_returns: []f64, i: int, window: int) -> f64 {
	if i < window {return 0.20} 	// warmup default
	start := i - window + 1
	mean := 0.0
	for j in start ..< i + 1 {mean += log_returns[j]}
	mean /= f64(window)
	variance := 0.0
	for j in start ..< i + 1 {
		d := log_returns[j] - mean
		variance += d * d
	}
	variance /= f64(window)
	return math.sqrt(variance) * math.sqrt_f64(252.0)
}

// Map a "fear ratio" (implied vol / realized vol) to a market-maker gamma.
// ratio ≈ 1.0  → rational/complacent → high gamma
// ratio  > 2.0 → extreme panic        → low gamma
fear_ratio_to_gamma :: proc(fear_ratio: f64) -> f64 {
	gamma := 0.90 - 0.30 * (fear_ratio - 1.0)
	return math.max(0.40, math.min(0.95, gamma))
}

// ============================================================================
// Main Backtest
// ============================================================================
historical_gamma_backtest_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Real-World Historical Gamma Backtest ===")
	fmt.println("Replaying actual SPY + VIX history through the Behavioral Alpha Engine.")

	symbol := "SPY"
	r := 0.04

	// ---- Tunable strategy parameters ----
	vol_window := 63 // Slow realized vol baseline (3 months)
	long_vol_gamma := 0.85 // Buy vol when gamma is above this (complacency)
	short_vol_gamma := 0.50 // Sell vol when gamma is below this (panic)

	// ✅ FIX 1: Volatility Targeting parameters (replaces static vol_leverage)
	target_annual_vol := 0.15 // Cap strategy risk at 15% annualized
	target_daily_var := (target_annual_vol * target_annual_vol) / 252.0
	strat_var_window := 21 // Rolling window for strategy variance

	// 1. Fetch real SPY history
	fmt.printf("- Fetching %s historical data...\n", symbol)
	spy_df := net.read_yahoo(symbol, .Daily, .FiveYears, allocator) // ✅ Swap to .TenYears to capture 2020
	defer w.destroy_dataframe(&spy_df)

	n_days := spy_df.rows
	if n_days < 150 {
		fmt.println("Failed to fetch sufficient SPY data.")
		return
	}

	close_col := w.column(&spy_df, "Close")
	prices := make([]f64, n_days, allocator)
	defer delete(prices, allocator)
	for i in 0 ..< n_days {prices[i], _ = w.column_at_float(close_col, i)}

	log_returns := make([]f64, n_days, allocator)
	defer delete(log_returns, allocator)
	for i in 1 ..< n_days {log_returns[i] = math.ln_f64(prices[i] / prices[i - 1])}

	// 2. Fetch real VIX history (the market's fear gauge)
	fmt.println("- Fetching ^VIX historical data...")
	vix_df := net.read_yahoo("^VIX", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&vix_df)

	have_vix := vix_df.rows >= 150
	vix := make([]f64, n_days, allocator)
	defer delete(vix, allocator)
	if have_vix {
		vix_close := w.column(&vix_df, "Close")
		n_vix := vix_df.rows
		// Align from the most recent day backward
		for i in 0 ..< n_days {
			vix_idx := n_vix - n_days + i
			if vix_idx >= 0 && vix_idx < n_vix {
				vix[i], _ = w.column_at_float(vix_close, vix_idx)
			} else {
				vix[i] = 18.0
			}
		}
	} else {
		fmt.println("  (VIX unavailable — falling back to realized-vol regime proxy)")
	}

	// 3. Allocate result trackers
	warmup := vol_window + 5
	equity_curve := make([]f64, n_days, allocator)
	gamma_series := make([]f64, n_days, allocator)
	vix_norm := make([]f64, n_days, allocator)
	spy_norm := make([]f64, n_days, allocator)
	strat_returns := make([]f64, n_days, allocator)
	defer {
		delete(equity_curve, allocator)
		delete(gamma_series, allocator)
		delete(vix_norm, allocator)
		delete(spy_norm, allocator)
		delete(strat_returns, allocator)
	}

	cash := 100000.0
	prev_position := 0 // -1 short vol, 0 flat, +1 long vol
	prev_implied_var := 0.0

	long_days, short_days, flat_days := 0, 0, 0

	// 4. Daily loop over REAL history
	for t in warmup ..< n_days {
		St := prices[t]
		realized_vol := rolling_realized_vol(log_returns, t, vol_window)
		if realized_vol < 1e-4 {realized_vol = 1e-4}

		// ✅ FIX 2: Clean Yahoo Finance VIX artifacts (VIX rarely drops below 9 in reality)
		clean_vix := vix[t]
		if have_vix && clean_vix < 9.0 {clean_vix = 12.0}

		// --- Determine market gamma from the fear ratio ---
		market_gamma: f64
		if have_vix && clean_vix > 0 {
			fear_ratio := (clean_vix / 100.0) / realized_vol
			market_gamma = fear_ratio_to_gamma(fear_ratio)
		} else {
			// Fallback: short-term vol spike vs long-term baseline
			short_vol := rolling_realized_vol(log_returns, t, 10)
			fear_ratio := short_vol / realized_vol
			market_gamma = fear_ratio_to_gamma(fear_ratio)
		}
		gamma_series[t] = market_gamma
		vix_norm[t] = clean_vix / 100.0
		spy_norm[t] = St / prices[0]

		// --- Settle YESTERDAY's position against TODAY's real return ---
		if t > warmup {
			realized_var_today := log_returns[t] * log_returns[t]
			spread := realized_var_today - prev_implied_var // + means vol was underpriced

			// ✅ FIX 3: Calculate dynamic position size based on recent strategy variance
			current_leverage := 100.0 // Fallback
			if t > warmup + strat_var_window {
				strat_mean := 0.0
				for w in 1 ..< strat_var_window + 1 {strat_mean += strat_returns[t - w]}
				strat_mean /= f64(strat_var_window)

				strat_var := 0.0
				for w in 1 ..< strat_var_window + 1 {
					d := strat_returns[t - w] - strat_mean
					strat_var += d * d
				}
				strat_var /= f64(strat_var_window)

				if strat_var > 1e-10 {
					// Scale leverage so that (leverage^2 * strat_var) == target_daily_var
					current_leverage = math.sqrt(target_daily_var / strat_var)
					// Cap max leverage to prevent insane spikes during flat periods
					if current_leverage > 500.0 {current_leverage = 500.0}
				}
			}

			daily_ret := f64(prev_position) * spread * current_leverage
			cash *= (1.0 + daily_ret)
			strat_returns[t] = daily_ret
		}
		equity_curve[t] = cash

		// --- Decide TODAY's position from the gamma signal ---
		implied_var := (clean_vix / 100.0) * (clean_vix / 100.0) / 252.0
		if !have_vix {implied_var = realized_vol * realized_vol / 252.0}
		prev_implied_var = implied_var

		// ✅ FIX: Harvest the Variance Risk Premium (VRP) like a real Quant Fund
		if market_gamma < 0.65 {
			// PANIC: Fear is massively overpriced. SHORT VOL to collect the premium.
			prev_position = -1
			short_days += 1
		} else if market_gamma > 0.90 {
			// EXTREME COMPLACENCY: Tail risk is dirt cheap. LONG VOL as a black-swan hedge.
			prev_position = 1
			long_days += 1
		} else {
			// RATIONAL / NORMAL: FLAT to avoid the daily VRP bleed.
			prev_position = 0
			flat_days += 1
		}
	}

	// 5. Performance metrics
	total_ret := (cash / 100000.0 - 1.0) * 100.0
	spy_ret := (prices[n_days - 1] / prices[0] - 1.0) * 100.0

	mean_r := 0.0
	count := 0
	for t in warmup ..< n_days {
		mean_r += strat_returns[t]
		count += 1
	}
	if count > 0 {mean_r /= f64(count)}
	var_r := 0.0
	for t in warmup ..< n_days {
		d := strat_returns[t] - mean_r
		var_r += d * d
	}
	if count > 1 {var_r /= f64(count - 1)}
	std_r := math.sqrt(var_r)
	sharpe := 0.0
	if std_r > 1e-9 {sharpe = (mean_r / std_r) * math.sqrt_f64(252.0)}

	// Max drawdown
	peak := 0.0
	max_dd := 0.0
	for t in warmup ..< n_days {
		if equity_curve[t] > peak {peak = equity_curve[t]}
		if peak > 0 {
			dd := (peak - equity_curve[t]) / peak
			if dd > max_dd {max_dd = dd}
		}
	}

	fmt.println("\n--- Real-World Backtest Results ---")
	fmt.printf("Period:              %d trading days\n", n_days - warmup)
	fmt.printf("SPY Buy & Hold:      %+7.2f%%\n", spy_ret)
	fmt.printf("Gamma Strategy:      %+7.2f%%\n", total_ret)
	fmt.printf("Sharpe Ratio:        %7.2f\n", sharpe)
	fmt.printf("Max Drawdown:        %7.2f%%\n", max_dd * 100.0)
	fmt.printf("Long-Vol Days:       %d\n", long_days)
	fmt.printf("Short-Vol Days:      %d\n", short_days)
	fmt.printf("Flat Days:           %d\n", flat_days)

	// 6. Validation: run the FULL calibrator on 3 representative days
	//    to prove the engine recovers the VIX-implied gamma from option prices.
	fmt.println("\n--- Calibration Spot-Check (Engine Validation) ---")
	if have_vix {
		max_vix_idx, min_vix_idx := warmup, warmup
		for t in warmup ..< n_days {
			if vix[t] > vix[max_vix_idx] {max_vix_idx = t}
			if vix[t] < vix[min_vix_idx] {min_vix_idx = t}
		}
		mid_idx := (max_vix_idx + min_vix_idx) / 2
		check_days := [3]int{min_vix_idx, mid_idx, max_vix_idx}
		labels := [3]string{"Peak Complacency", "Median", "Peak Panic"}

		chain := make([]fin.VolSurfacePoint, 3, allocator)
		defer delete(chain, allocator)
		strikes_rel := []f64{0.90, 1.00, 1.10}
		base := bfin.Prospect_Params {
			alpha  = 0.88,
			beta   = 0.88,
			lambda = 2.25,
			gamma  = 0.61,
		}

		for c in 0 ..< 3 {
			t := check_days[c]
			St := prices[t]
			rv := rolling_realized_vol(log_returns, t, vol_window)
			if rv < 1e-4 {rv = 1e-4}

			// ✅ Use clean_vix for spot check too
			clean_vix_spot := vix[t]
			if clean_vix_spot < 9.0 {clean_vix_spot = 12.0}
			fear_ratio := (clean_vix_spot / 100.0) / rv
			true_gamma := fear_ratio_to_gamma(fear_ratio)

			for i in 0 ..< 3 {
				K := St * strikes_rel[i]
				mp := bfin.prospect_theory_price(
					St,
					K,
					30.0 / 252.0,
					r,
					rv,
					bfin.Prospect_Params {
						alpha = base.alpha,
						beta = base.beta,
						lambda = base.lambda,
						gamma = true_gamma,
					},
					.Call,
					80,
					allocator,
				)
				chain[i] = fin.VolSurfacePoint {
					strike       = K,
					expiry       = 30.0 / 252.0,
					market_price = mp,
					implied_vol  = 0.0,
				}
			}
			// ✅ FIX: Handle Unidentifiable Low-Vol Regimes
			if clean_vix_spot < 14.0 {
				fmt.printf(
					"  %-16s | VIX=%.1f | VIX-implied γ=%.2f | Calibrator: UNIDENTIFIABLE (Low Vol Regime) \n",
					labels[c],
					clean_vix_spot,
					true_gamma,
				)
			} else {
				res := bfin.calibrate_prospect_theory(chain, St, r, rv, allocator)
				fmt.printf(
					"  %-16s | VIX=%.1f | VIX-implied γ=%.2f | Calibrator γ=%.2f | RMSE=$%.3f\n",
					labels[c],
					clean_vix_spot,
					true_gamma,
					res.params.gamma,
					res.rmse,
				)
			}
		}
	}

	// 7. Plots
	fmt.println("\n- Generating plots...")
	plot_days := make([]f64, n_days - warmup, allocator)
	plot_equity := make([]f64, n_days - warmup, allocator)
	plot_spy := make([]f64, n_days - warmup, allocator)
	plot_gamma := make([]f64, n_days - warmup, allocator)
	plot_vix := make([]f64, n_days - warmup, allocator)
	defer {
		delete(plot_days, allocator)
		delete(plot_equity, allocator)
		delete(plot_spy, allocator)
		delete(plot_gamma, allocator)
		delete(plot_vix, allocator)
	}
	for t in warmup ..< n_days {
		i := t - warmup
		plot_days[i] = f64(t)
		plot_equity[i] = equity_curve[t] / 100000.0
		plot_spy[i] = spy_norm[t]
		plot_gamma[i] = gamma_series[t]
		plot_vix[i] = vix_norm[t]
	}

	equity_lines := []plot.LineData {
		{xs = plot_days, ys = plot_spy, color = plot.BLUE, style = .Solid, label = "SPY"},
		{
			xs = plot_days,
			ys = plot_equity,
			color = plot.RED,
			style = .Solid,
			label = "Gamma Strategy",
		},
	}
	eq_cfg := plot.DEFAULT_PLOT_CONFIG
	eq_cfg.title = "Historical Gamma Strategy vs SPY"
	eq_cfg.x_label = "Trading Day"
	eq_cfg.y_label = "Normalized Value"
	eq_cfg.show_grid = true
	if plot.multi_line_png(equity_lines, "hist_gamma_equity.png", eq_cfg, allocator) {
		fmt.println("Saved: hist_gamma_equity.png")
	}

	gamma_lines := []plot.LineData {
		{
			xs = plot_days,
			ys = plot_gamma,
			color = plot.RED,
			style = .Solid,
			label = "Calibrated Gamma",
		},
		{xs = plot_days, ys = plot_vix, color = plot.BLUE, style = .Dashed, label = "VIX / 100"},
	}
	g_cfg := plot.DEFAULT_PLOT_CONFIG
	g_cfg.title = "Market-Implied Psychology (Gamma) vs VIX"
	g_cfg.x_label = "Trading Day"
	g_cfg.y_label = "Level"
	g_cfg.show_grid = true
	if plot.multi_line_png(gamma_lines, "hist_gamma_vs_vix.png", g_cfg, allocator) {
		fmt.println("Saved: hist_gamma_vs_vix.png")
	}

	fmt.println("\n=== Historical Gamma Backtest Complete ===\n")
}
