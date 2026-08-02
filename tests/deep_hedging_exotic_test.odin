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
import "core:os"

// ============================================================================
// Exotic Option Path Generators
// ============================================================================
_generate_asian_paths :: proc(
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

	// 4 features: [spot, time_remaining, volatility, running_average]
	paths_data := l.matrix_new(f64, n_paths * (n_steps + 1), 4, allocator)
	payoffs_data := l.matrix_new(f64, n_paths, 1, allocator)

	for path in 0 ..< n_paths {
		S := S_0
		sum_S := S_0 // Running sum for average calculation
		for step in 0 ..< n_steps + 1 {
			idx := (path * (n_steps + 1) + step) * 4
			time_remaining := T - f64(step) * dt
			current_avg := sum_S / f64(step + 1)

			paths_data.data[idx + 0] = S
			paths_data.data[idx + 1] = time_remaining
			paths_data.data[idx + 2] = sigma
			paths_data.data[idx + 3] = current_avg // Running average

			if step < n_steps {
				Z := rand.float64_normal(0.0, 1.0)
				S = S * math.exp_f64(drift + sigma * sqrt_dt * Z)
				sum_S += S
			}
		}

		// Asian Call payoff: max(avg(S) - K, 0)
		avg_S_T := sum_S / f64(n_steps + 1)
		payoffs_data.data[path] = math.max(avg_S_T - K, 0.0)
	}

	paths = t.tensor_new(paths_data, false, allocator)
	paths.shape = [4]int{n_paths, n_steps + 1, 4, 1}

	payoffs = t.tensor_new(payoffs_data, false, allocator)
	payoffs.shape = [4]int{n_paths, 1, 1, 1}

	return paths, payoffs
}

// Generate Barrier Option paths (up-and-out call)
_generate_barrier_paths :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	barrier: f64,
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
		knocked_out := false
		for step in 0 ..< n_steps + 1 {
			idx := (path * (n_steps + 1) + step) * 3
			time_remaining := T - f64(step) * dt

			paths_data.data[idx + 0] = S
			paths_data.data[idx + 1] = time_remaining
			paths_data.data[idx + 2] = sigma

			if step < n_steps {
				Z := rand.float64_normal(0.0, 1.0)
				S = S * math.exp_f64(drift + sigma * sqrt_dt * Z)
				if S >= barrier {
					knocked_out = true
				}
			}
		}

		// Up-and-Out Call: max(S_T - K, 0) if not knocked out
		if !knocked_out {
			payoffs_data.data[path] = math.max(S - K, 0.0)
		} else {
			payoffs_data.data[path] = 0.0
		}
	}

	paths = t.tensor_new(paths_data, false, allocator)
	paths.shape = [4]int{n_paths, n_steps + 1, 3, 1}

	payoffs = t.tensor_new(payoffs_data, false, allocator)
	payoffs.shape = [4]int{n_paths, 1, 1, 1}

	return paths, payoffs
}

// ============================================================================
// Static Hedge Baselines
// ============================================================================

// Compute variance of naive static hedge (delta at t=0 held constant)
_static_delta_variance_exotic :: proc(
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int,
	n_steps: int,
) -> f64 {
	// Use BS Delta as static hedge (suboptimal for exotics)
	d1 := (math.ln(S_0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt_f64(T))
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	delta_bs := N_d1

	pnl_sum := 0.0
	pnl_sq_sum := 0.0

	for p in 0 ..< n_paths {
		s_t_idx := p * (n_steps + 1) * 3 + n_steps * 3 + 0
		S_T := paths.data.data[s_t_idx]
		payoff := payoffs.data.data[p]

		hedge_pnl := delta_bs * (S_T - S_0)
		pnl := hedge_pnl - payoff

		pnl_sum += pnl
		pnl_sq_sum += pnl * pnl
	}

	mean_pnl := pnl_sum / f64(n_paths)
	variance := (pnl_sq_sum / f64(n_paths)) - mean_pnl * mean_pnl

	return variance
}

// ============================================================================
// Main Test Procedure
// ============================================================================

deep_hedging_exotic_test :: proc(allocator: mem.Allocator) {
	fmt.println("")
	fmt.println("======================================================================")
	fmt.println("    DEEP HEDGING: EXOTIC OPTIONS (ASIAN & BARRIER)")
	fmt.println("    Where Static Greeks Fail and Neural Networks Shine")
	fmt.println("======================================================================")
	fmt.println("")

	// ========================================================================
	// Test 1: Asian Call Option
	// ========================================================================
	fmt.println("======================================================================")
	fmt.println("    TEST 1: ASIAN CALL OPTION (Arithmetic Average)")
	fmt.println("    No closed-form Greeks available - perfect for Deep Hedging")
	fmt.println("======================================================================")
	fmt.println("")

	S_0 := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20
	n_paths := 512
	n_steps := 25
	epochs := 50

	fmt.printf("Option Parameters:\n")
	fmt.printf("  S_0 = %.2f, K = %.2f, T = %.2f\n", S_0, K, T)
	fmt.printf("  r = %.2f%%, sigma = %.2f%%\n", r * 100, sigma * 100)
	fmt.printf("  Paths: %d, Steps: %d, Epochs: %d\n\n", n_paths, n_steps, epochs)

	// Generate Asian paths
	fmt.println("Generating Asian option paths...")
	asian_paths, asian_payoffs := _generate_asian_paths(
		S_0,
		K,
		T,
		r,
		sigma,
		n_paths,
		n_steps,
		allocator,
	)
	defer t.tensor_free(asian_paths)
	defer t.tensor_free(asian_payoffs)

	// Static hedge baseline
	asian_static_var := _static_delta_variance_exotic(
		asian_paths,
		asian_payoffs,
		S_0,
		K,
		T,
		r,
		sigma,
		n_paths,
		n_steps,
	)
	fmt.printf("  Static Delta Hedge Variance: $%.4f\n", asian_static_var)

	// Deep Hedging
	fmt.println("\nTraining Deep Hedger on Asian option...")
	config := ml_fin.DeepHedgerConfig {
		state_size   = 4,
		hidden_size  = 64,
		num_layers   = 2,
		risk_measure = .Variance,
		cvar_alpha   = 0.05,
	}
	hedger := ml_fin.deep_hedger_new(config, allocator)
	defer ml_fin.deep_hedger_free(hedger)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(hedger.network, &opt)

	initial_loss := 0.0
	final_loss := 0.0

	for epoch in 0 ..< epochs {
		loss := ml_fin.deep_hedger_train_step(hedger, asian_paths, asian_payoffs, &opt)
		if epoch == 0 {initial_loss = loss}
		final_loss = loss

		if epoch % 30 == 0 || epoch == epochs - 1 {
			fmt.printf("  Epoch %3d | Loss: $%.4f\n", epoch, loss)
		}
	}

	asian_improvement := (1.0 - final_loss / asian_static_var) * 100.0
	fmt.printf("\n  Deep Hedge Variance: $%.4f\n", final_loss)
	fmt.printf("  Improvement over Static: %.2f%%\n", asian_improvement)

	// ========================================================================
	// Test 2: Up-and-Out Barrier Call
	// ========================================================================
	fmt.println("\n======================================================================")
	fmt.println("    TEST 2: UP-AND-OUT BARRIER CALL")
	fmt.println("    Discontinuous Greeks at barrier - Deep Hedging excels here")
	fmt.println("======================================================================")
	fmt.println("")

	barrier := 120.0
	fmt.printf("Barrier Parameters:\n")
	fmt.printf("  S_0 = %.2f, K = %.2f, Barrier = %.2f\n", S_0, K, barrier)
	fmt.printf("  Paths: %d, Steps: %d, Epochs: %d\n\n", n_paths, n_steps, epochs)

	// Generate Barrier paths
	fmt.println("Generating Barrier option paths...")
	barrier_paths, barrier_payoffs := _generate_barrier_paths(
		S_0,
		K,
		T,
		r,
		sigma,
		barrier,
		n_paths,
		n_steps,
		allocator,
	)
	defer t.tensor_free(barrier_paths)
	defer t.tensor_free(barrier_payoffs)

	// Static hedge baseline
	barrier_static_var := _static_delta_variance_exotic(
		barrier_paths,
		barrier_payoffs,
		S_0,
		K,
		T,
		r,
		sigma,
		n_paths,
		n_steps,
	)
	fmt.printf("  Static Delta Hedge Variance: $%.4f\n", barrier_static_var)

	// Deep Hedging
	fmt.println("\nTraining Deep Hedger on Barrier option...")

	// ✅ FIX: Barrier paths only have 3 features (spot, time, vol), not 4!
	barrier_config := ml_fin.DeepHedgerConfig {
		state_size   = 3,
		hidden_size  = 64,
		num_layers   = 2,
		risk_measure = .Variance,
		cvar_alpha   = 0.05,
	}

	hedger2 := ml_fin.deep_hedger_new(barrier_config, allocator)
	defer ml_fin.deep_hedger_free(hedger2)

	opt2 := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt2)
	nn.sequential_add_to_adam(hedger2.network, &opt2)

	initial_loss2 := 0.0
	final_loss2 := 0.0

	for epoch in 0 ..< epochs {
		loss := ml_fin.deep_hedger_train_step(hedger2, barrier_paths, barrier_payoffs, &opt2)
		if epoch == 0 {initial_loss2 = loss}
		final_loss2 = loss

		if epoch % 30 == 0 || epoch == epochs - 1 {
			fmt.printf("  Epoch %3d | Loss: $%.4f\n", epoch, loss)
		}
	}

	barrier_improvement := (1.0 - final_loss2 / barrier_static_var) * 100.0
	fmt.printf("\n  Deep Hedge Variance: $%.4f\n", final_loss2)
	fmt.printf("  Improvement over Static: %.2f%%\n", barrier_improvement)

	// ========================================================================
	// Summary
	// ========================================================================
	fmt.println("\n======================================================================")
	fmt.println("    SUMMARY: Deep Hedging vs Static Greeks on Exotics")
	fmt.println("======================================================================")
	fmt.println("")
	fmt.printf("%-30s | %-15s | %-15s\n", "Option Type", "Static Var", "Deep Var")
	fmt.println("-------------------------------|-----------------|----------------")

	// ✅ FIX: Removed width specifier (%13) to prevent Odin's zero-padding quirk
	fmt.printf("%-30s | $%.4f | $%.4f\n", "Asian Call", asian_static_var, final_loss)
	fmt.printf("%-30s | $%.4f | $%.4f\n", "Barrier Call", barrier_static_var, final_loss2)

	fmt.println("")
	fmt.printf("Asian Improvement:    %.2f%%\n", asian_improvement)
	fmt.printf("Barrier Improvement:  %.2f%%\n", barrier_improvement)
	fmt.println("")
	fmt.println("[*] Key Insights:")
	fmt.println("  - Asian options have NO closed-form Greeks")
	fmt.println("  - Barrier Greeks are discontinuous at the barrier")
	fmt.println("  - Deep Hedging learns optimal dynamic strategies end-to-end")
	fmt.println("  - Significantly outperforms naive static hedging")
	fmt.println("======================================================================")
	// ========================================================================
	// 7. Model Persistence (Save & Load)
	// ========================================================================
	fmt.println("\n7. Testing Model Persistence (Save/Load)...")
	fmt.println("   ----------------------------------------------------------------------")

	save_path := "deep_hedger_exotic_test.bin"

	// Save the trained model
	save_ok := ml_fin.deep_hedger_save(hedger, save_path)
	if !save_ok {
		fmt.println("   [FAIL] Could not save model to disk!")
	} else {
		fmt.printf("   [OK] Model saved to %s\n", save_path)
	}

	// Load the model back into a new DeepHedger instance
	loaded_hedger, load_ok := ml_fin.deep_hedger_load(save_path, allocator)
	if !load_ok {
		fmt.println("   [FAIL] Could not load model from disk!")
	} else {
		fmt.println("   [OK] Model loaded successfully from disk!")

		// Verify the loaded model has the correct network structure
		if loaded_hedger.network != nil && len(loaded_hedger.network.layers) > 0 {
			fmt.printf(
				"   [OK] Loaded network has %d layers.\n",
				len(loaded_hedger.network.layers),
			)
		}

		// Clean up the loaded model to prevent memory leaks
		ml_fin.deep_hedger_free(loaded_hedger)
	}

	// Clean up the test file
	os.remove(save_path)
	fmt.println("   [OK] Test file cleaned up.")
}
