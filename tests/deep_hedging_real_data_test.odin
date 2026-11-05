// tests/deep_hedging_real_data_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

deep_hedging_real_data_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n======================================================================")
	fmt.println("    DEEP HEDGING: REAL MARKET DATA BACKTEST")
	fmt.println("    Training on Live Options Data & Comparing to Market Prices")
	fmt.println("======================================================================\n")

	// ========================================================================
	// 1. Fetch Real Market Data
	// ========================================================================
	symbol := "SPY"
	fmt.printf("Fetching real market data for %s...\n", symbol)

	// Fetch historical price data
	df := net.read_yahoo(symbol, .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("ERROR: Failed to fetch historical data")
		return
	}

	// Get current spot price
	last_idx := df.rows - 1
	spot, _ := w.column_at_float(w.column(&df, "Close"), last_idx)
	fmt.printf("Current Spot Price: $%.2f\n", spot)

	// Fetch options chain
	fmt.println("Fetching options chain...")
	chain := fin.fetch_yahoo_options(symbol, allocator)

	if chain.n_options == 0 {
		fmt.println("ERROR: Failed to fetch options chain")
		return
	}

	fmt.printf("Found %d options contracts\n", chain.n_options)

	// ========================================================================
	// 2. Select ATM Call Option for Training
	// ========================================================================
	fmt.println("\nSelecting ATM call option for training...")

	// Find ATM call option (closest strike to spot, call type)
	best_idx := -1
	best_diff := math.F64_MAX

	for i in 0 ..< chain.n_options {
		if chain.option_types[i] == "call" {
			diff := math.abs(chain.strikes[i] - spot)
			if diff < best_diff {
				best_diff = diff
				best_idx = i
			}
		}
	}

	if best_idx < 0 {
		fmt.println("ERROR: No call options found")
		return
	}

	strike := chain.strikes[best_idx]
	market_price := chain.market_prices[best_idx]
	expiry := chain.expiries[best_idx]
	R :: 0.04
	// inverted from the quoted price (Yahoo's impliedVolatility runs high)
	implied_vol, iv_ok, _ := fin.implied_volatility(market_price, spot, strike, expiry, R, .Call)
	if !iv_ok {
		fmt.println("ERROR: the quoted price has no implied volatility")
		return
	}

	fmt.println("\nSelected ATM Call Option:")
	fmt.printf("  Strike:        $%.2f\n", strike)
	fmt.printf("  Market Price:  $%.2f\n", market_price)
	fmt.printf("  Implied Vol:   %.2f%%\n", implied_vol * 100)
	fmt.printf("  Time to Expiry: %.2f years\n", expiry)

	// ========================================================================
	// 3. Calculate Historical Volatility
	// ========================================================================
	fmt.println("\nCalculating historical volatility...")

	// Extract closing prices
	n_days := df.rows
	prices := make([]f64, n_days, allocator)
	defer delete(prices, allocator)

	for i in 0 ..< n_days {
		prices[i], _ = w.column_at_float(w.column(&df, "AdjClose"), i)
	}

	// Calculate log returns
	returns := make([]f64, n_days - 1, allocator)
	defer delete(returns, allocator)

	for i in 1 ..< n_days {
		// a missing (null or 0, as Yahoo sends for some rows) close counts as no move
		returns[i - 1] =
			math.ln(prices[i] / prices[i - 1]) if prices[i] > 0 && prices[i - 1] > 0 else 0
	}

	// Calculate historical volatility (annualized)
	mean_ret := 0.0
	for r in returns {
		mean_ret += r
	}
	mean_ret /= f64(len(returns))

	var_sum := 0.0
	for r in returns {
		diff := r - mean_ret
		var_sum += diff * diff
	}
	hist_var := var_sum / f64(len(returns) - 1)
	hist_vol := math.sqrt(hist_var * 252.0) // Annualized

	fmt.printf("Historical Volatility: %.2f%%\n", hist_vol * 100)
	fmt.printf("Implied Volatility:    %.2f%%\n", implied_vol * 100)

	// Use implied volatility for training (more realistic)
	training_vol := implied_vol

	// ========================================================================
	// 4. Train Deep Hedger
	// ========================================================================
	fmt.println("\nTraining Deep Hedger...")

	// Training parameters
	n_paths := 2048
	n_steps := 50
	epochs := 300

	config := ml_fin.DeepHedgerConfig {
		state_size       = 4,
		hidden_size      = 64,
		num_layers       = 2,
		risk_measure     = .Variance,
		cvar_alpha       = 0.05,
		transaction_cost = 0.001, // 10 bps
		num_assets       = 1, // Single asset (SPY)
	}

	hedger := ml_fin.deep_hedger_new(config, allocator)
	defer ml_fin.deep_hedger_free(hedger)

	opt := nn.adam_new(3e-3, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(hedger.network, &opt)
	schedule := optim.lr_cosine(3e-3, epochs, warmup_steps = 10)

	// GBM paths at the implied volatility, in units of the spot, discounted
	paths, payoffs := _generate_real_paths(
		spot,
		strike,
		expiry,
		R,
		training_vol,
		n_paths,
		n_steps,
		allocator,
	)
	defer t.tensor_free(paths)
	defer t.tensor_free(payoffs)

	for epoch in 0 ..< epochs {
		opt.learning_rate = optim.lr_schedule_at(schedule, epoch)
		loss := ml_fin.deep_hedger_train_step(hedger, paths, payoffs, &opt)
		if epoch % 60 == 0 || epoch == epochs - 1 {
			fmt.printf("  Epoch %3d | P&L variance (units of spot^2): %.3e\n", epoch, loss)
		}
	}

	// ========================================================================
	// 5. Price implied by the hedge
	// ========================================================================
	// The variance-minimising hedger needs the premium p that makes its expected P&L
	// zero, p = -E[P&L] (a mean-variance hedging price; it includes the trading costs)
	pnl_values, spot_prices := ml_fin.deep_hedging_evaluate(hedger, paths, payoffs, allocator)
	mean_pnl := 0.0
	for &v in pnl_values {mean_pnl += v / f64(n_paths); v *= spot}
	for &v in spot_prices {v *= spot}
	learned_price := -mean_pnl * spot
	bs_price, _ := fin.price_and_greeks(spot, strike, expiry, R, training_vol, .Call, allocator)
	// the Black-Scholes delta hedge on the same paths with the same costs: its price
	// -E[P&L] (premium 0) and P&L std
	bs_pnl := bs_delta_hedge_pnl(
		strike / spot,
		expiry,
		R,
		training_vol,
		config.transaction_cost,
		paths,
		payoffs,
		0,
		allocator,
	)
	bs_mean, bs_m2 := 0.0, 0.0
	for v in bs_pnl {bs_mean += v / f64(n_paths); bs_m2 += v * v / f64(n_paths)}
	bs_hedge_price := -bs_mean * spot
	bs_std := math.sqrt(bs_m2 - bs_mean * bs_mean) * spot
	deep_std := 0.0
	for v in pnl_values {deep_std += (v + learned_price) * (v + learned_price) / f64(n_paths)}
	deep_std = math.sqrt(deep_std)

	// ========================================================================
	// 6. Results Summary
	// ========================================================================
	fmt.println("\n======================================================================")
	fmt.println("    RESULTS SUMMARY")
	fmt.println("======================================================================")
	fmt.printf("  Market Price:                    $%.2f\n", market_price)
	fmt.printf("  Black-Scholes at the implied vol: $%.2f\n", bs_price)
	fmt.printf(
		"  Price -E[P&L], 10 bp costs:      deep hedge $%.2f, BS delta hedge $%.2f\n",
		learned_price,
		bs_hedge_price,
	)
	fmt.printf(
		"  Hedged P&L std:                  deep hedge $%.2f, BS delta hedge $%.2f\n",
		deep_std,
		bs_std,
	)
	rel := (learned_price - bs_hedge_price) / bs_hedge_price
	ok := abs(rel) < 0.05 && deep_std < 2 * bs_std
	fmt.printf(
		"%s deep hedge price within 5%% of the BS delta hedge's with the same costs (%+.2f%%), P&L std below twice its (%.2f vs %.2f)\n",
		"OK  " if ok else "FAIL",
		100 * rel,
		deep_std,
		bs_std,
	)
	record_failures(int(!ok))
	fmt.println(
		"======================================================================",
	); fmt.println("======================================================================")


	// ========================================================================
	// 7. Generate Visualizations
	// ========================================================================
	fmt.println("\n--- Generating Visualizations ---")

	// 2. Plot 1: PnL Distribution
	fmt.println("  Generating PnL distribution plot...")
	ok1 := ml_fin.deep_hedging_plot_pnl_distribution(
		pnl_values,
		"Deep Hedging: Terminal PnL Distribution",
		"plots/deep_hedging_pnl_dist.png",
		allocator,
	)
	if ok1 {
		fmt.println("  [OK] Saved: plots/deep_hedging_pnl_dist.png")
	} else {
		fmt.println("  [FAIL] Could not save PnL distribution plot.")
	}

	// 3. Plot 2: PnL vs Spot Price
	fmt.println("  Generating PnL vs Spot Price plot...")
	ok2 := ml_fin.deep_hedging_plot_pnl_vs_spot(
		spot_prices,
		pnl_values,
		"Deep Hedging: PnL vs Terminal Spot Price",
		"plots/deep_hedging_pnl_vs_spot.png",
		allocator,
	)
	if ok2 {
		fmt.println("  [OK] Saved: plots/deep_hedging_pnl_vs_spot.png")
	} else {
		fmt.println("  [FAIL] Could not save PnL vs Spot plot.")
	}

	// 4. Cleanup evaluation slices
	delete(pnl_values, allocator)
	delete(spot_prices, allocator)

}
_generate_real_paths :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator,
) -> (
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
) {
	dt := T / f64(n_steps)
	sqrt_dt := math.sqrt_f64(dt)
	drift := (r - 0.5 * sigma * sigma) * dt

	// ✅ FIX: Generate 3 features: [Spot, Time Remaining, Volatility]
	// features: discounted spot / S_0, time to maturity, sigma, ln(S / K) / (sigma sqrt(T))
	// (the moneyness in standard deviations: O(1) even for short expiries)
	F :: 4
	paths_data := l.matrix_new(f64, n_paths * (n_steps + 1), F, allocator)
	payoffs_data := l.matrix_new(f64, n_paths, 1, allocator)

	rand_count := n_paths * n_steps
	norm_data := make([]f64, rand_count, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	rand_idx := 0
	for path in 0 ..< n_paths {
		S := S_0
		for step in 0 ..< n_steps + 1 {
			idx := (path * (n_steps + 1) + step) * F
			time_remaining := T - f64(step) * dt

			// Feature 0: discounted spot in units of S_0
			paths_data.data[idx + 0] = S / S_0 * math.exp(-r * f64(step) * dt)
			// Feature 1: Time Remaining
			paths_data.data[idx + 1] = time_remaining
			// Feature 2: Volatility
			paths_data.data[idx + 2] = sigma
			paths_data.data[idx + 3] = math.ln(S / K) / (sigma * math.sqrt(T))

			if step < n_steps {
				Z := norm_data[rand_idx]
				rand_idx += 1
				S = S * math.exp_f64(drift + sigma * sqrt_dt * Z)
			}
		}
		payoffs_data.data[path] = math.exp(-r * T) * math.max(S - K, 0.0) / S_0
	}

	paths = t.tensor_new(paths_data, false, allocator)
	// ✅ FIX: shape[2] must be 3 to match state_size
	paths.shape = [4]int{n_paths, n_steps + 1, F, 1}

	payoffs = t.tensor_new(payoffs_data, false, allocator)
	payoffs.shape = [4]int{n_paths, 1, 1, 1}

	return paths, payoffs
}
