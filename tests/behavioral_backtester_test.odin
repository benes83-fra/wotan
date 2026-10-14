package tests

import bfin "../wotan/behavioral_finance"
import fin "../wotan/finance"
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
