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
import "core:mem"

// ============================================================================
// Neural SDF Calibration Test
// Calibrate a neural SDF to real multi-asset return data (SPY, QQQ, IWM)
// ============================================================================

neural_sdf_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    NEURAL SDF CALIBRATION: LEARNING THE PRICING KERNEL")
	fmt.println("    E[M_{t+1} · R_{i,t+1}] = 1  for all assets i")
	fmt.println("======================================================================\n")

	// ========================================================================
	// 1. Fetch Real Multi-Asset Data
	// ========================================================================
	fmt.println("1. Fetching Real Market Data (SPY, QQQ, IWM)...")

	symbols := []string{"SPY", "QQQ", "IWM"}
	num_assets := len(symbols)

	// Fetch 1 year of daily data for each asset
	dfs: [3]w.DataFrame
	for sym, i in symbols {
		dfs[i] = net.read_yahoo(sym, .Daily, .OneYear, allocator)
		fmt.printf("   %s: %d trading days\n", sym, dfs[i].rows)
	}
	defer {
		for i in 0 ..< num_assets {
			w.destroy_dataframe(&dfs[i])
		}
	}

	// Use the minimum number of rows across all assets
	T := dfs[0].rows
	for i in 1 ..< num_assets {
		if dfs[i].rows < T {
			T = dfs[i].rows
		}
	}
	T -= 1 // Need T-1 returns from T prices

	fmt.printf("   Using %d return observations across %d assets\n", T, num_assets)

	// ========================================================================
	// 2. Compute Gross Returns and Conditioning Variables
	// ========================================================================
	fmt.println("\n2. Computing Gross Returns & Conditioning Variables...")

	// Gross returns R_{i,t} = Close_t / Close_{t-1}  (already 1 + net return)
	r_data := l.matrix_new(f64, T, num_assets, allocator)

	// Conditioning variables Z_t: [R_market, R_market², |R_market|, 1.0]
	// Using SPY as the market factor
	input_size := 4
	z_data := l.matrix_new(f64, T, input_size, allocator)

	for row in 0 ..< T {
		// Get SPY return for conditioning
		spy_prev, _ := w.column_at_float(w.column(&dfs[0], "Close"), row)
		spy_curr, _ := w.column_at_float(w.column(&dfs[0], "Close"), row + 1)
		r_spy := spy_curr / spy_prev // Gross return

		// Fill gross returns for all assets
		for j in 0 ..< num_assets {
			prev, _ := w.column_at_float(w.column(&dfs[j], "Close"), row)
			curr, _ := w.column_at_float(w.column(&dfs[j], "Close"), row + 1)
			if prev > 0 {
				r_data.data[row * num_assets + j] = curr / prev
			} else {
				r_data.data[row * num_assets + j] = 1.0
			}
		}

		// Conditioning variables (polynomial + absolute value basis)
		net_ret := r_spy - 1.0
		z_data.data[row * input_size + 0] = net_ret
		z_data.data[row * input_size + 1] = net_ret * net_ret
		z_data.data[row * input_size + 2] = math.abs(net_ret)
		z_data.data[row * input_size + 3] = 1.0 // Bias feature
	}

	r := t.tensor_new(r_data, false, allocator)
	r.shape = [4]int{T, num_assets, 1, 1}
	defer t.tensor_free(r)

	z := t.tensor_new(z_data, false, allocator)
	z.shape = [4]int{T, input_size, 1, 1}
	defer t.tensor_free(z)

	// Print summary statistics
	mean_ret := [3]f64{}
	for j in 0 ..< num_assets {
		sum := 0.0
		for row in 0 ..< T {
			sum += r_data.data[row * num_assets + j] - 1.0
		}
		mean_ret[j] = sum / f64(T) * 252.0 // Annualized
	}
	fmt.printf(
		"   Annualized mean returns: SPY=%.2f%%, QQQ=%.2f%%, IWM=%.2f%%\n",
		mean_ret[0] * 100,
		mean_ret[1] * 100,
		mean_ret[2] * 100,
	)

	// ========================================================================
	// 3. Initialize Neural SDF
	// ========================================================================
	fmt.println("\n3. Initializing Neural SDF Model...")

	config := ml_fin.NeuralSDFConfig {
		input_size  = input_size,
		hidden_size = 64,
		num_layers  = 3,
		num_assets  = num_assets,
		epsilon     = 0.01,
	}

	model := ml_fin.neural_sdf_new(config, allocator)
	defer ml_fin.neural_sdf_free(model)

	opt := nn.adam_new(0.0005, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(model.network, &opt)

	fmt.printf("   Architecture: %d -> 64 -> 64 -> 64 -> 1 (ReLU)\n", input_size)
	fmt.printf("   Positivity constraint: M = f(Z)² + %.3f\n", config.epsilon)
	fmt.printf("   Calibrating to %d assets simultaneously\n", num_assets)

	// ========================================================================
	// 4. Training Loop
	// ========================================================================
	fmt.println("\n4. Training Neural SDF (Euler Equation Loss)...")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.println("   Epoch   | Loss (Σ pricing errors²)")
	fmt.println("   ----------------------------------------------------------------------")

	epochs := 500
	initial_loss := 0.0
	final_loss := 0.0

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)
		loss := ml_fin.neural_sdf_train_step(model, z, r, &opt)

		if epoch == 0 {
			initial_loss = loss
		}
		final_loss = loss

		if epoch % 50 == 0 || epoch == epochs - 1 {
			fmt.printf("   %5d   | %.8f\n", epoch, loss)
		}
	}

	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   Loss reduction: %.2f%%\n", (1.0 - final_loss / initial_loss) * 100.0)

	// ========================================================================
	// 5. Evaluate Pricing Errors
	// ========================================================================
	fmt.println("\n5. Evaluating Pricing Errors (|E[M·R] - 1| per asset)...")
	fmt.println("   ----------------------------------------------------------------------")

	pricing_errors := ml_fin.neural_sdf_evaluate(model, z, r, allocator)
	defer delete(pricing_errors, allocator)

	for j in 0 ..< num_assets {
		fmt.printf("   %s: |E[M·R] - 1| = %.6f\n", symbols[j], pricing_errors[j])
	}

	avg_error := 0.0
	for j in 0 ..< num_assets {
		avg_error += pricing_errors[j]
	}
	avg_error /= f64(num_assets)
	fmt.printf("   Average pricing error: %.6f\n", avg_error)

	// ========================================================================
	// 6. Compare Against Trivial SDF (M = constant)
	// ========================================================================
	fmt.println("\n6. Baseline Comparison (Trivial Constant SDF)...")
	fmt.println("   ----------------------------------------------------------------------")

	// Trivial SDF: M = 1/E[R] for each asset independently
	for j in 0 ..< num_assets {
		sum_r := 0.0
		for row in 0 ..< T {
			sum_r += r_data.data[row * num_assets + j]
		}
		mean_r := sum_r / f64(T)
		// If M = c (constant), then c * E[R] = 1 => c = 1/E[R]
		// Pricing error for OTHER assets with this constant SDF:
		trivial_m := 1.0 / mean_r
		sum_mr_other := 0.0
		for row in 0 ..< T {
			sum_mr_other += trivial_m * r_data.data[row * num_assets + j]
		}
		trivial_error := math.abs(sum_mr_other / f64(T) - 1.0)
		fmt.printf("   %s trivial SDF error: %.6f\n", symbols[j], trivial_error)
	}

	// ========================================================================
	// 7. Interpretation
	// ========================================================================
	fmt.println("\n======================================================================")
	fmt.println("    INTERPRETATION")
	fmt.println("======================================================================")
	fmt.println("  • The Neural SDF learns M_{t+1} = f_θ(Z_t)² + ε")
	fmt.println("  • Z_t = [R_mkt, R_mkt², |R_mkt|, 1] captures non-linear risk pricing")
	fmt.println("  • The squared term ensures M > 0 (no-arbitrage requirement)")
	fmt.println("  • E[M·R] = 1 means the SDF correctly prices each asset")
	fmt.println("  • Low pricing error => the learned kernel is economically meaningful")
	fmt.println("  • This replaces grid-search SDF calibration with gradient-based learning")
	fmt.println("======================================================================\n")

	// ========================================================================
	// 8. Model Persistence
	// ========================================================================
	fmt.println("7. Testing Model Persistence (Save/Load)...")
	save_path := "neural_sdf_model.bin"
	save_ok := ml_fin.neural_sdf_save(model, save_path)
	if save_ok {
		fmt.printf("   [OK] Model saved to %s\n", save_path)
	} else {
		fmt.println("   [FAIL] Could not save model")
	}

	loaded, load_ok := ml_fin.neural_sdf_load(save_path, config, allocator)
	if load_ok {
		fmt.println("   [OK] Model loaded successfully")
		ml_fin.neural_sdf_free(loaded)
	} else {
		fmt.println("   [FAIL] Could not load model")
	}
}
