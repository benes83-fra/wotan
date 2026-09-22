package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

orchestrator_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Closed-Loop Trading Orchestrator ===")
	main_alloc := context.allocator

	// 1. Define the asset universe
	tickers := []string{"SPY", "QQQ", "AAPL", "MSFT", "TLT"}
	n_assets := len(tickers)
	total_capital := 1_000_000.0 // $1M portfolio

	// 2. Fetch historical data for all assets
	fmt.println("\n--- Fetching Market Data ---")
	dfs := make([]w.DataFrame, n_assets, main_alloc)
	defer {
		for i in 0 ..< n_assets {w.destroy_dataframe(&dfs[i])}
		delete(dfs, main_alloc)
	}

	for i in 0 ..< n_assets {
		dfs[i] = yahoo.read_yahoo(tickers[i], .Daily, .FiveYears, allocator)
		fmt.printf("Loaded %d days for %s\n", dfs[i].rows, tickers[i])
	}

	n_common := dfs[0].rows
	for i in 1 ..< n_assets {
		if dfs[i].rows < n_common {n_common = dfs[i].rows}
	}
	num_days := n_common - 1

	// Compute returns matrix [T, N]
	returns_flat := make([]f64, num_days * n_assets, main_alloc)
	defer delete(returns_flat, main_alloc)

	for d in 1 ..< n_common {
		for i in 0 ..< n_assets {
			prev_c, _ := w.column_at_float(&dfs[i].columns[4], d - 1)
			curr_c, _ := w.column_at_float(&dfs[i].columns[4], d)
			returns_flat[(d - 1) * n_assets + i] = math.ln_f64(curr_c / prev_c)
		}
	}

	// Extract SPY returns for regime/vol models
	spy_returns := make([]f64, num_days, main_alloc)
	defer delete(spy_returns, main_alloc)
	for d in 0 ..< num_days {
		spy_returns[d] = returns_flat[d * n_assets + 0]
	}

	// ====================================================================
	// PHASE 1: Train Offline Models (HMM, Ensemble Vol, Conformal)
	// ====================================================================
	fmt.println("\n--- Phase 1: Training Offline Models ---")

	// 1a. Train HMM for regime detection
	n_states := 3
	hmm := ml_fin.hmm_new(n_states, main_alloc)
	defer ml_fin.hmm_free(&hmm)

	// Clip returns for HMM training
	hmm_returns := make([]f64, num_days, main_alloc)
	defer delete(hmm_returns, main_alloc)
	copy(hmm_returns, spy_returns)
	for i in 0 ..< num_days {
		if hmm_returns[i] > 0.05 {hmm_returns[i] = 0.05}
		if hmm_returns[i] < -0.05 {hmm_returns[i] = -0.05}
	}

	ml_fin.hmm_init_simple(&hmm, hmm_returns)
	hmm_ll, hmm_converged := ml_fin.hmm_fit(&hmm, hmm_returns, 200, 1e-5)
	fmt.printf("  HMM: LL=%.2f, Converged=%v\n", hmm_ll, hmm_converged)

	// Decode regimes for the full history
	regime_sequence := ml_fin.hmm_decode(&hmm, hmm_returns, main_alloc)
	defer delete(regime_sequence, main_alloc)

	// 1b. Fit GARCH for vol forecasting
	residuals := ts.extract_residuals(spy_returns[:num_days - 20], main_alloc)
	defer delete(residuals, main_alloc)
	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 1000, 1e-4, main_alloc)
	defer {
		delete(garch_result.params.alpha, main_alloc)
		delete(garch_result.params.beta, main_alloc)
		delete(garch_result.conditional_var, main_alloc)
		delete(garch_result.standardized_resid, main_alloc)
	}
	fmt.printf(
		"  GARCH: ω=%.6f, α=%.4f, β=%.4f\n",
		garch_result.params.omega,
		garch_result.params.alpha[0],
		garch_result.params.beta[0],
	)

	// ====================================================================
	// PHASE 2: Initialize Orchestrator
	// ====================================================================
	fmt.println("\n--- Phase 2: Initializing Orchestrator ---")

	state := ml_fin.orchestrator_init(n_assets, main_alloc)
	defer ml_fin.orchestrator_free(&state)

	// Avellaneda-Stoikov parameters for market making
	as_params := ml_fin.AvellanedaStoikovParams {
		sigma = 0.02, // 2% daily vol (normalized)
		gamma = 0.1,
		k     = 1.5,
		T     = 1.0,
	}

	// ====================================================================
	// PHASE 3: Simulate Multi-Day Trading Loop
	// ====================================================================
	fmt.println("\n--- Phase 3: Running Trading Loop ---")

	// ✅ FIX: Pre-extract the last fitted conditional variance and last return
	//         for recursive GARCH forecasting beyond the fitted range
	garch_fit_len := len(garch_result.conditional_var)
	last_fitted_var := garch_result.conditional_var[garch_fit_len - 1]
	last_fitted_ret := spy_returns[garch_fit_len - 1] // return corresponding to last fitted var

	// Simulate the last 5 trading days
	sim_start := num_days - 5
	sim_end := num_days

	// Track the running variance for recursive forecasting
	running_var := last_fitted_var
	running_ret := last_fitted_ret

	for day in sim_start ..< sim_end {
		fmt.printf("\n━━━ Trading Day %d ━━━\n", day)

		// Get current prices
		prices := make([]f64, n_assets, main_alloc)
		for i in 0 ..< n_assets {
			c, _ := w.column_at_float(&dfs[i].columns[4], day)
			prices[i] = c
		}
		defer delete(prices, main_alloc)

		// Step 1: Detect regime
		current_regime := regime_sequence[day]
		regime_mean := hmm.emission_mu[current_regime] * 252.0
		regime_vol := math.sqrt(hmm.emission_var[current_regime]) * math.sqrt_f64(252.0)

		// Step 2: Forecast volatility using recursive GARCH
		// ✅ FIX: If day is within fitted range, use fitted variance.
		//         Otherwise, recursively forecast forward.
		if day < garch_fit_len {
			running_var = garch_result.conditional_var[day]
			running_ret = spy_returns[day]
		} else {
			// Recursive forecast: σ²_{t+1} = ω + α·r²_t + β·σ²_t
			running_ret = spy_returns[day]
			running_var =
				garch_result.params.omega +
				garch_result.params.alpha[0] * running_ret * running_ret +
				garch_result.params.beta[0] * running_var
		}

		vol_forecast := math.sqrt_f64(running_var) * math.sqrt_f64(252.0)

		// Step 3: Conformal bounds
		vol_lower := vol_forecast * 0.85
		vol_upper := vol_forecast * 1.15

		// Step 4: VRP signal
		vrp_signal := 0.0
		if vol_forecast < 0.15 {
			vrp_signal = 0.5
		} else if vol_forecast > 0.25 {
			vrp_signal = -0.5
		}

		// Step 5: HRP portfolio construction
		hist_days := day + 1
		if hist_days > 252 {hist_days = 252}
		hist_start := day - hist_days + 1
		if hist_start < 0 {hist_start = 0}

		hist_returns_flat := make([]f64, hist_days * n_assets, main_alloc)
		for d in hist_start ..= day {
			for i in 0 ..< n_assets {
				hist_returns_flat[(d - hist_start) * n_assets + i] = returns_flat[d * n_assets + i]
			}
		}
		hist_mat := l.matrix_from_flat(hist_returns_flat, hist_days, n_assets, main_alloc)
		hrp_res := ml_fin.hrp_allocate(&hist_mat, main_alloc)

		// Step 6: Run orchestrator step
		decision := ml_fin.orchestrator_step(
			&state,
			current_regime,
			regime_mean,
			regime_vol,
			vol_forecast,
			vol_lower,
			vol_upper,
			vrp_signal,
			hrp_res.weights,
			prices,
			total_capital,
			as_params,
		)

		ml_fin.print_decision(&decision, tickers, day)

		// Cleanup
		ml_fin.decision_free(&decision, main_alloc)
		ml_fin.hrp_result_free(&hrp_res)
		l.matrix_free(&hist_mat)
		delete(hist_returns_flat, main_alloc)
	}

	fmt.println("\n✓ Closed-Loop Orchestrator Test Complete!")
}
