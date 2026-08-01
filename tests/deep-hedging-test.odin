package tests

import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Generate Geometric Brownian Motion paths for testing
_generate_gbm_paths :: proc(
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

	paths_data := l.matrix_new(f64, n_paths * (n_steps + 1), 3, allocator)
	payoffs_data := l.matrix_new(f64, n_paths, 1, allocator)

	for path in 0 ..< n_paths {
		S := S_0
		for step in 0 ..< n_steps + 1 {
			idx := (path * (n_steps + 1) + step) * 3
			time_remaining := T - f64(step) * dt

			paths_data.data[idx + 0] = S
			paths_data.data[idx + 1] = time_remaining
			paths_data.data[idx + 2] = sigma

			if step < n_steps {
				Z := rand.float64_normal(0.0, 1.0)
				S = S * math.exp_f64(drift + sigma * sqrt_dt * Z)
			}
		}

		S_T := paths_data.data[(path * (n_steps + 1) + n_steps) * 3 + 0]
		payoffs_data.data[path] = math.max(S_T - K, 0.0)
	}

	paths = t.tensor_new(paths_data, false, allocator)
	paths.shape = [4]int{n_paths, n_steps + 1, 3, 1}

	payoffs = t.tensor_new(payoffs_data, false, allocator)
	payoffs.shape = [4]int{n_paths, 1, 1, 1}

	return paths, payoffs
}

// Main test procedure
deep_hedging_test :: proc(allocator: mem.Allocator) {
	fmt.println("")
	fmt.println("======================================================================")
	fmt.println("    DEEP HEDGING: NEURAL NETWORK OPTION PRICING & HEDGING")
	fmt.println("======================================================================")
	fmt.println("")

	// ========================================================================
	// Test Parameters
	// ========================================================================
	S_0 := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20
	n_paths := 1024
	n_steps := 50
	epochs := 100

	fmt.printf("Option Parameters:\n")
	fmt.printf("  S_0 = %.2f, K = %.2f, T = %.2f\n", S_0, K, T)
	fmt.printf("  r = %.2f%%, sigma = %.2f%%\n", r * 100, sigma * 100)
	fmt.printf("  Paths: %d, Steps: %d, Epochs: %d\n\n", n_paths, n_steps, epochs)

	// ========================================================================
	// 1. Black-Scholes Benchmark
	// ========================================================================
	fmt.println("1. Black-Scholes Benchmark (Analytical Solution)")
	fmt.println("   ----------------------------------------------------------------------")
	bs_price, bs_greeks := fin.price_and_greeks(S_0, K, T, r, sigma, .Call, allocator)
	fmt.printf("   BS Call Price: $%.4f\n", bs_price)
	fmt.printf("   BS Delta:      %.4f\n", bs_greeks.delta)
	fmt.printf("   BS Gamma:      %.4f\n", bs_greeks.gamma)
	fmt.printf("   BS Vega:       %.4f\n", bs_greeks.vega)

	// ========================================================================
	// 2. Generate Training Data
	// ========================================================================
	fmt.println("\n2. Generating GBM Training Data...")
	paths, payoffs := _generate_gbm_paths(S_0, K, T, r, sigma, n_paths, n_steps, allocator)
	defer t.tensor_free(paths)
	defer t.tensor_free(payoffs)
	fmt.printf("   [OK] Generated %d paths with %d time steps\n", n_paths, n_steps)

	// ========================================================================
	// 3. Create Deep Hedger
	// ========================================================================
	fmt.println("\n3. Initializing Deep Hedger...")
	config := ml_fin.DeepHedgerConfig {
		state_size   = 3,
		hidden_size  = 64,
		num_layers   = 2,
		risk_measure = .CVaR,
		cvar_alpha   = 0.05,
	}
	hedger := ml_fin.deep_hedger_new(config, allocator)
	defer ml_fin.deep_hedger_free(hedger)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.sequential_add_to_adam(hedger.network, &opt)

	fmt.println("   [OK] Created MLP with 2 hidden layers (64 units each)")

	// ========================================================================
	// 4. Training Loop
	// ========================================================================
	fmt.println("\n4. Training Deep Hedger...")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.println("   Epoch    | Loss (PnL Var)")
	fmt.println("   ----------------------------------------------------------------------")

	initial_loss := 0.0
	final_loss := 0.0

	for epoch in 0 ..< epochs {
		loss := ml_fin.deep_hedger_train_step(hedger, paths, payoffs, &opt)

		if epoch == 0 {
			initial_loss = loss
		}
		final_loss = loss

		// Print progress every 10 epochs (Fixed Odin formatting)
		if epoch % 10 == 0 || epoch == epochs - 1 {
			fmt.printf("   %3d      | $%.4f\n", epoch, loss)
		}
	}

	fmt.println("   ----------------------------------------------------------------------")
	loss_reduction := (1.0 - final_loss / initial_loss) * 100.0
	fmt.printf("   Loss Reduction: %.2f%%\n", loss_reduction)

	// ========================================================================
	// 5. Compute True Baselines
	// ========================================================================
	fmt.println("\n5. Computing True Baselines...")
	fmt.println("   ----------------------------------------------------------------------")

	// True unhedged variance
	true_unhedged_var := _unhedged_variance(payoffs, n_paths)
	fmt.printf("   True Unhedged Variance:  $%.4f\n", true_unhedged_var)

	// Static delta baseline
	static_var, delta_bs := _static_delta_variance(
		S_0,
		K,
		T,
		r,
		sigma,
		paths,
		payoffs,
		n_paths,
		n_steps,
		allocator,
	)
	fmt.printf("   BS Delta (t=0):          %.4f\n", delta_bs)
	fmt.printf("   Static Hedge Variance:   $%.4f\n", static_var)

	// ========================================================================
	// 6. Results Summary
	// ========================================================================
	fmt.println("\n6. Results Summary")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-30s | $%.4f\n", "Black-Scholes Price", bs_price)
	fmt.printf("   %-30s | $%.4f\n", "True Unhedged Variance", true_unhedged_var)
	fmt.printf("   %-30s | $%.4f\n", "Static Delta Variance", static_var)
	fmt.printf("   %-30s | $%.4f\n", "Deep Hedge Variance (epoch 0)", initial_loss)
	fmt.printf("   %-30s | $%.4f\n", "Deep Hedge Variance (final)", final_loss)
	fmt.println("   ----------------------------------------------------------------------")

	// Improvement metrics
	static_improvement := (1.0 - static_var / true_unhedged_var) * 100.0
	deep_improvement := (1.0 - final_loss / true_unhedged_var) * 100.0
	deep_vs_static := 0.0
	if static_var > 1e-10 {
		deep_vs_static = (1.0 - final_loss / static_var) * 100.0
	}

	fmt.printf("   %-30s | %.2f%%\n", "Static vs Unhedged", static_improvement)
	fmt.printf("   %-30s | %.2f%%\n", "Deep vs Unhedged", deep_improvement)
	fmt.printf("   %-30s | %.2f%%\n", "Deep vs Static", deep_vs_static)

	fmt.println("\n[*] Key Insights:")
	fmt.println("   - Static Delta removes first-order (linear) risk")
	fmt.println("   - Deep Hedging additionally removes Gamma and higher-order risk")
	fmt.println("   - The gap between Static and Deep = value of dynamic rebalancing")
	if deep_vs_static > 5.0 {
		fmt.println("   - The NN significantly outperforms static hedging!")
	} else {
		fmt.println("   - For vanilla European calls, static Delta is already strong.")
		fmt.println("   - Try exotic payoffs (Asians, Barriers) to see bigger gaps.")
	}
	fmt.println("======================================================================")
}
// Compute TRUE unhedged variance (variance of -payoff)
_unhedged_variance :: proc(payoffs: ^t.Tensor, n_paths: int) -> f64 {
	sum := 0.0
	sq_sum := 0.0
	for i in 0 ..< n_paths {
		p := -payoffs.data.data[i] // Loss = -payoff
		sum += p
		sq_sum += p * p
	}
	mean := sum / f64(n_paths)
	return (sq_sum / f64(n_paths)) - mean * mean
}

// Compute static delta variance (corrected)
_static_delta_variance :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator,
) -> (
	static_var: f64,
	static_delta: f64,
) {
	// 1. Compute Black-Scholes Delta at t=0
	d1 := (math.ln(S_0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt_f64(T))
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	delta_bs := N_d1

	// 2. Compute BS price (premium received)
	d2 := d1 - sigma * math.sqrt_f64(T)
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
	call_price := S_0 * N_d1 - K * math.exp_f64(-r * T) * N_d2

	// 3. Compute static hedge PnL for each path
	//    PnL = call_price + delta * (S_T - S_0) - payoff
	pnl_sum := 0.0
	pnl_sq_sum := 0.0

	for p in 0 ..< n_paths {
		s_t_idx := p * (n_steps + 1) * 3 + n_steps * 3 + 0
		S_T := paths.data.data[s_t_idx]
		payoff := payoffs.data.data[p]

		// Static hedge PnL (includes premium received)
		hedge_pnl := call_price + delta_bs * (S_T - S_0) - payoff

		pnl_sum += hedge_pnl
		pnl_sq_sum += hedge_pnl * hedge_pnl
	}

	mean_pnl := pnl_sum / f64(n_paths)
	variance := (pnl_sq_sum / f64(n_paths)) - mean_pnl * mean_pnl

	return variance, delta_bs
}
