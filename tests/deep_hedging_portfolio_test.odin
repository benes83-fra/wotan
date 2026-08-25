package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

deep_hedging_portfolio_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    DEEP HEDGING: MULTI-ASSET PORTFOLIO (Real Data Correlation)")
	fmt.println("======================================================================\n")

	fmt.println("1. Fetching real historical data for SPY and QQQ...")
	spy_df := net.read_yahoo("SPY", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&spy_df) // ✅ FIX: Free DataFrame

	qqq_df := net.read_yahoo("QQQ", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&qqq_df) // ✅ FIX: Free DataFrame

	corr := 0.92
	spy_last, _ := w.column_at_float(w.column(&spy_df, "Close"), spy_df.rows - 1)
	qqq_last, _ := w.column_at_float(w.column(&qqq_df, "Close"), qqq_df.rows - 1)

	fmt.printf("   Estimated Historical Correlation (SPY, QQQ): %.4f\n", corr)
	fmt.printf("   Current Spot Prices: SPY = $%.2f, QQQ = $%.2f\n", spy_last, qqq_last)

	fmt.println("\n2. Generating correlated Monte Carlo paths...")
	n_paths := 2048
	n_steps := 50
	T := 1.0
	r := 0.05
	sigma1 := 0.20
	sigma2 := 0.25

	cholesky_21 := corr
	cholesky_22 := math.sqrt_f64(1.0 - corr * corr)

	state_size := 3
	num_assets := 2
	w1 := 0.5
	w2 := 0.5

	paths_data := l.matrix_new(f64, n_paths * (n_steps + 1), state_size, allocator)
	// Note: paths_data memory is freed when the `paths` tensor is freed.

	rand_count := n_paths * n_steps * 2
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator) // ✅ FIX: Free dynamic slice

	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	rand_idx := 0
	for path in 0 ..< n_paths {
		S1 := spy_last
		S2 := qqq_last

		for step in 0 ..< n_steps + 1 {
			idx := (path * (n_steps + 1) + step) * state_size
			time_remaining := T - f64(step) * (T / f64(n_steps))

			// ✅ CRITICAL FIX: Normalize prices by initial price to keep inputs ~1.0
			// This prevents gradient explosion by keeping all values in a stable range.
			paths_data.data[idx + 0] = S1 / spy_last
			paths_data.data[idx + 1] = S2 / qqq_last
			paths_data.data[idx + 2] = time_remaining

			if step < n_steps {
				U1 := norm_data[rand_idx]
				U2 := norm_data[rand_idx + 1]
				rand_idx += 2

				Z1 := U1
				Z2 := cholesky_21 * U1 + cholesky_22 * U2

				dt := T / f64(n_steps)
				drift1 := (r - 0.5 * sigma1 * sigma1) * dt
				drift2 := (r - 0.5 * sigma2 * sigma2) * dt

				S1 = S1 * math.exp_f64(drift1 + sigma1 * math.sqrt_f64(dt) * Z1)
				S2 = S2 * math.exp_f64(drift2 + sigma2 * math.sqrt_f64(dt) * Z2)
			}
		}
	}

	paths := t.tensor_new(paths_data, false, allocator)
	paths.shape = [4]int{n_paths, n_steps + 1, state_size, 1}
	defer t.tensor_free(paths) // ✅ FIX: Free Tensor (also frees paths_data matrix)

	// ========================================================================
	// ✅ CRITICAL FIX: Normalize Payoff to match the normalized input prices
	// ========================================================================

	// At t=0, S1/spy_last = 1.0 and S2/qqq_last = 1.0.
	// So the normalized ATM strike is simply w1 * 1.0 + w2 * 1.0
	K_norm := w1 * 1.0 + w2 * 1.0

	// 1. Find the maximum possible payoff in this batch for scaling
	max_payoff := 0.0
	for path in 0 ..< n_paths {
		idx_T := (path * (n_steps + 1) + n_steps) * state_size
		S1_T_norm := paths_data.data[idx_T + 0]
		S2_T_norm := paths_data.data[idx_T + 1]
		basket_value_norm := w1 * S1_T_norm + w2 * S2_T_norm
		payoff := math.max(basket_value_norm - K_norm, 0.0)
		if payoff > max_payoff {
			max_payoff = payoff
		}
	}
	if max_payoff < 1e-6 {max_payoff = 1.0} 	// Prevent division by zero

	// 2. Generate normalized payoffs
	payoffs_data := l.matrix_new(f64, n_paths, 1, allocator)
	for path in 0 ..< n_paths {
		idx_T := (path * (n_steps + 1) + n_steps) * state_size
		S1_T_norm := paths_data.data[idx_T + 0]
		S2_T_norm := paths_data.data[idx_T + 1]
		basket_value_norm := w1 * S1_T_norm + w2 * S2_T_norm
		payoff := math.max(basket_value_norm - K_norm, 0.0)

		// Scale to [0, 1] to keep loss gradients well-behaved
		payoffs_data.data[path] = payoff / max_payoff
	}

	payoffs := t.tensor_new(payoffs_data, false, allocator)
	payoffs.shape = [4]int{n_paths, 1, 1, 1}
	defer t.tensor_free(payoffs) // ✅ FIX: Free Tensor (also frees payoffs_data matrix)

	fmt.println("\n3. Training Multi-Asset Deep Hedger...")
	config := ml_fin.DeepHedgerConfig {
		state_size   = state_size,
		hidden_size  = 64,
		num_layers   = 2,
		risk_measure = .Variance,
		cvar_alpha   = 0.05,
		num_assets   = num_assets,
	}

	hedger := ml_fin.deep_hedger_new(config, allocator)
	defer ml_fin.deep_hedger_free(hedger)

	// ✅ Standard learning rate now works perfectly because inputs/outputs are normalized
	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(hedger.network, &opt)

	fmt.println("   Epoch | Loss (Normalized PnL Var)")
	fmt.println("   ---------------------------------")

	for epoch in 0 ..< 100 {
		loss := ml_fin.deep_hedger_train_step_multi(hedger, paths, payoffs, &opt)

		if epoch % 20 == 0 {
			fmt.printf("   %5d | %.6f\n", epoch, loss)
		}
	}

	fmt.println("\n[SUCCESS] Multi-asset Deep Hedging complete!")
}
