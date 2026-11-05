package tests

import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:slice"
// GBM paths in units of the numeraire, as deep hedging expects: state (e^{-rt} S_t, time
// to maturity, sigma) and the discounted call payoff e^{-rT} (S_T - K)^+
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

			paths_data.data[idx + 0] = S * math.exp(-r * f64(step) * dt)
			paths_data.data[idx + 1] = time_remaining
			paths_data.data[idx + 2] = sigma

			if step < n_steps {
				Z := rand.float64_normal(0.0, 1.0)
				S = S * math.exp_f64(drift + sigma * sqrt_dt * Z)
			}
		}

		payoffs_data.data[path] = math.exp(-r * T) * math.max(S - K, 0.0)
	}

	paths = t.tensor_new(paths_data, false, allocator)
	paths.shape = [4]int{n_paths, n_steps + 1, 3, 1}

	payoffs = t.tensor_new(payoffs_data, false, allocator)
	payoffs.shape = [4]int{n_paths, 1, 1, 1}

	return paths, payoffs
}

// Test parameters: prices in units of S_0 (S_0 = 1, as in Buehler et al.), so the network
// inputs are O(1); the P&L figures are fractions of the notional
@(private = "file")
DH_S_0 :: 1.0
@(private = "file")
DH_K :: 1.0
@(private = "file")
DH_T :: 1.0
@(private = "file")
DH_R :: 0.05
@(private = "file")
DH_SIGMA :: 0.20
@(private = "file")
DH_PATHS :: 1024
@(private = "file")
DH_STEPS :: 50
@(private = "file")
DH_EPOCHS :: 300

// One training run: the trained hedger (free with deep_hedger_free), its
// training loss (the configured risk measure of the PnL without the premium) at
// the first and last epoch, and on the training paths the terminal PnL variance
// of the deep hedge, of no hedge and of the static delta hedge, and the CVaR at
// level cvar_alpha (positive loss, the Black-Scholes premium received) of the
// deep and the static hedge
@(private = "file")
_Deep_Hedging_Run :: struct {
	hedger:                             ^ml_fin.DeepHedger,
	initial_loss, final_loss:           f64,
	deep_var, unhedged_var, static_var: f64,
	cvar_alpha:                         f64,
	deep_cvar, static_cvar:             f64,
	delta_bs:                           f64,
	bs_var, bs_cvar:                    f64, // Black-Scholes delta rebalanced every step
}

@(private = "file")
_deep_hedging_run :: proc(allocator: mem.Allocator, verbose: bool) -> (run: _Deep_Hedging_Run) {
	S_0, K, T, r, sigma := DH_S_0, DH_K, DH_T, DH_R, DH_SIGMA
	n_paths, n_steps, epochs := DH_PATHS, DH_STEPS, DH_EPOCHS

	// ========================================================================
	// 2. Generate Training Data
	// ========================================================================
	if verbose {fmt.println("\n2. Generating GBM Training Data...")}
	paths, payoffs := _generate_gbm_paths(S_0, K, T, r, sigma, n_paths, n_steps, allocator)
	defer t.tensor_free(paths)
	defer t.tensor_free(payoffs)
	if verbose {fmt.printf("   [OK] Generated %d paths with %d time steps\n", n_paths, n_steps)}

	// ========================================================================
	// 3. Create Deep Hedger
	// ========================================================================
	if verbose {fmt.println("\n3. Initializing Deep Hedger...")}
	config := ml_fin.DeepHedgerConfig {
		state_size       = 3,
		hidden_size      = 64,
		num_layers       = 2,
		risk_measure     = .CVaR,
		cvar_alpha       = 0.05,
		transaction_cost = 0.001,
		num_assets       = 1,
	}
	run.hedger = ml_fin.deep_hedger_new(config, allocator)

	// Warmup + cosine decay from 3e-3 over the epochs. Over 5 seeds (300 epochs) the CVaR
	// 95% is ~74% below the static delta hedge's and ~1.5x the Black-Scholes delta
	// hedge's (rebalanced every step); 600 epochs bring it to ~1.3x
	schedule := optim.lr_cosine(0.003, epochs, warmup_steps = 10)
	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.sequential_add_to_adam(run.hedger.network, &opt)

	if verbose {
		fmt.println("   [OK] Created MLP with 2 hidden layers (64 units each)")
		fmt.printf("   [OK] Transaction Cost: %.2f%% per trade\n", config.transaction_cost * 100)
		fmt.println("   [OK] Adam, learning rate 3e-3, 10 warmup epochs, then cosine decay to 0")
	}
	// ========================================================================
	// 4. Training Loop
	// ========================================================================
	if verbose {
		fmt.println("\n4. Training Deep Hedger...")
		fmt.println("   ----------------------------------------------------------------------")
		fmt.println("   Epoch    | Loss CVaR 95% (without premium)")
		fmt.println("   ----------------------------------------------------------------------")
	}

	for epoch in 0 ..< epochs {
		opt.learning_rate = optim.lr_schedule_at(schedule, epoch)
		loss := ml_fin.deep_hedger_train_step(run.hedger, paths, payoffs, &opt)

		if epoch == 0 {
			run.initial_loss = loss
		}
		run.final_loss = loss

		// Print progress every 10 epochs (Fixed Odin formatting)
		if verbose && (epoch % 10 == 0 || epoch == epochs - 1) {
			fmt.printf("   %3d      | $%.4f\n", epoch, loss)
		}
	}

	// PnL of the trained hedge (plus the premium, as the static hedge's) and of
	// the static delta hedge
	pnl, spots := ml_fin.deep_hedging_evaluate(run.hedger, paths, payoffs, allocator)
	defer delete(pnl, allocator)
	defer delete(spots, allocator)
	static_pnl, delta_bs, premium := _static_delta_pnl(
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
	defer delete(static_pnl, allocator)
	run.delta_bs = delta_bs
	for &v in pnl {v += premium}

	run.deep_var, run.static_var = _variance(pnl), _variance(static_pnl)
	run.unhedged_var = _unhedged_variance(payoffs, n_paths)
	run.cvar_alpha = config.cvar_alpha
	run.deep_cvar = fin.portfolio_cvar_historical(pnl, 1 - config.cvar_alpha)
	run.static_cvar = fin.portfolio_cvar_historical(static_pnl, 1 - config.cvar_alpha)
	bs_pnl := bs_delta_hedge_pnl(
		K,
		T,
		r,
		sigma,
		config.transaction_cost,
		paths,
		payoffs,
		premium,
		allocator,
	)
	defer delete(bs_pnl, allocator)
	run.bs_var = _variance(bs_pnl)
	run.bs_cvar = fin.portfolio_cvar_historical(bs_pnl, 1 - config.cvar_alpha)
	return run
}

// Deep vs static hedge: PnL variance reduction of the deep hedge relative to
// the static delta hedge in percent
@(private = "file")
_deep_vs_static :: proc(run: _Deep_Hedging_Run) -> f64 {
	return (1.0 - run.deep_var / run.static_var) * 100.0 if run.static_var > 1e-10 else 0.0
}

// Deep vs static hedge: CVaR reduction of the deep hedge relative to the static
// delta hedge in percent
@(private = "file")
_deep_vs_static_cvar :: proc(run: _Deep_Hedging_Run) -> f64 {
	return (1.0 - run.deep_cvar / run.static_cvar) * 100.0
}

// Main test procedure
deep_hedging_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("")
	fmt.println("======================================================================")
	fmt.println("    DEEP HEDGING: NEURAL NETWORK OPTION PRICING & HEDGING")
	fmt.println("======================================================================")
	fmt.println("")

	S_0, K, T, r, sigma := DH_S_0, DH_K, DH_T, DH_R, DH_SIGMA
	n_paths, n_steps, epochs := DH_PATHS, DH_STEPS, DH_EPOCHS

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

	run := _deep_hedging_run(allocator, true)
	hedger := run.hedger
	defer ml_fin.deep_hedger_free(hedger)
	initial_loss, final_loss := run.initial_loss, run.final_loss

	fmt.println("   ----------------------------------------------------------------------")
	loss_reduction := (1.0 - final_loss / initial_loss) * 100.0
	fmt.printf("   Loss Reduction: %.2f%%\n", loss_reduction)

	// ========================================================================
	// 5. Compute True Baselines
	// ========================================================================
	fmt.println("\n5. Computing True Baselines...")
	fmt.println("   ----------------------------------------------------------------------")

	true_unhedged_var, static_var, delta_bs := run.unhedged_var, run.static_var, run.delta_bs
	fmt.printf("   True Unhedged Variance:  $%.4f\n", true_unhedged_var)
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
	fmt.printf("   %-30s | $%.4f\n", "Loss CVaR, epoch 0", initial_loss)
	fmt.printf("   %-30s | $%.4f\n", "Loss CVaR, final", final_loss)
	fmt.printf("   %-30s | $%.4f\n", "Deep Hedge Variance", run.deep_var)
	fmt.printf("   %-30s | $%.4f\n", "BS Delta (every step) Variance", run.bs_var)
	cvar_label := fmt.tprintf("CVaR %.0f%%", (1 - run.cvar_alpha) * 100)
	fmt.printf("   %-30s | $%.4f\n", fmt.tprintf("Static Delta %s", cvar_label), run.static_cvar)
	fmt.printf("   %-30s | $%.4f\n", fmt.tprintf("Deep Hedge %s", cvar_label), run.deep_cvar)
	fmt.printf("   %-30s | $%.4f\n", fmt.tprintf("BS Delta %s", cvar_label), run.bs_cvar)
	fmt.println("   ----------------------------------------------------------------------")

	// Improvement metrics
	static_improvement := (1.0 - static_var / true_unhedged_var) * 100.0
	deep_improvement := (1.0 - run.deep_var / true_unhedged_var) * 100.0
	deep_vs_static := _deep_vs_static(run)

	fmt.printf("   %-30s | %.2f%%\n", "Static vs Unhedged", static_improvement)
	fmt.printf("   %-30s | %.2f%%\n", "Deep vs Unhedged", deep_improvement)
	fmt.printf("   %-30s | %.2f%%\n", "Deep vs Static", deep_vs_static)
	fmt.printf(
		"   %-30s | %.2f%%\n",
		fmt.tprintf("Deep vs Static, %s", cvar_label),
		_deep_vs_static_cvar(run),
	)

	fmt.printf(
		"   %-30s | %.2f%%\n",
		fmt.tprintf("Deep vs BS Delta, %s", cvar_label),
		(1 - run.deep_cvar / run.bs_cvar) * 100,
	)
	fmt.println(
		"\n   The Black-Scholes delta rebalanced every step is the benchmark: in this model it",
	)
	fmt.println(
		"   is the variance-optimal hedge as the steps shrink; the deep hedge optimises the",
	)
	fmt.println("   CVaR with the trading cost, which the delta hedge ignores.")
	fmt.println("======================================================================")
	// ========================================================================
	// 7. Model Persistence (Save & Load)
	// ========================================================================
	fmt.println("\n7. Testing Model Persistence (Save/Load)...")
	fmt.println("   ----------------------------------------------------------------------")

	save_path := "deep_hedger_test.bin"

	// Save the trained model
	save_ok := ml_fin.deep_hedger_save(hedger, save_path)
	if !save_ok {
		fmt.println("   [FAIL] Could not save model to disk!")
	} else {
		fmt.printf("   [OK] Model saved to %s\n", save_path)
	}

	// Load the model back into a new DeepHedger instance
	loaded_hedger, load_ok := ml_fin.deep_hedger_load(save_path, allocator, hedger.config)
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
// Compute TRUE unhedged variance (variance of -payoff)
_unhedged_variance :: proc(payoffs: ^t.Tensor, n_paths: int) -> f64 {
	return _variance(payoffs.data.data[:n_paths])
}

// Population variance
@(private = "file")
_variance :: proc(xs: []f64) -> f64 {
	mean := math.sum(xs) / f64(len(xs))
	ss := 0.0
	for x in xs {ss += (x - mean) * (x - mean)}
	return ss / f64(len(xs))
}

// deep_hedging_test over several seeds: mean ± std of its results, to tell a
// change of the training from seed noise (run 0 is deep_hedging_test itself)
deep_hedging_seed_sweep_test :: proc(allocator: mem.Allocator, n_seeds := 5) {
	fmt.printf("\n=== Deep hedging over %d seeds ===\n", n_seeds)
	stats := seed_sweep(
		"deep_hedging_test",
		n_seeds,
		{
			"deep vs static (%)",
			"deep vs unhedged (%)",
			"deep PnL variance",
			"static PnL variance",
			"deep vs static CVaR (%)",
			"deep CVaR",
			"static CVaR",
			"BS delta CVaR",
			"deep vs BS delta CVaR (%)",
		},
		proc(allocator: mem.Allocator) -> []f64 {
			run := _deep_hedging_run(allocator, false)
			ml_fin.deep_hedger_free(run.hedger)
			return slice.clone(
				[]f64 {
					_deep_vs_static(run),
					(1.0 - run.deep_var / run.unhedged_var) * 100.0,
					run.deep_var,
					run.static_var,
					_deep_vs_static_cvar(run),
					run.deep_cvar,
					run.static_cvar,
					run.bs_cvar,
					(1 - run.deep_cvar / run.bs_cvar) * 100,
				},
				allocator,
			)
		},
		allocator,
	)
	seed_sweep_print(stats)
	// Dynamic hedging must beat the static delta hedge with every seed, in the
	// trained risk measure (CVaR) and in variance
	ok_cvar, ok_var := stats[4].min > 0, stats[0].min > 0
	// ... and stay near the Black-Scholes delta hedge: its CVaR below twice the delta's
	ok_bs := stats[8].min > -100
	fmt.printf(
		"%s deep hedge CVaR below the static delta hedge's with every seed (worst seed %.1f%% lower)\n",
		"OK  " if ok_cvar else "FAIL",
		stats[4].min,
	)
	fmt.printf(
		"%s deep hedge PnL variance below the static delta hedge's with every seed (worst seed %.1f%% lower)\n",
		"OK  " if ok_var else "FAIL",
		stats[0].min,
	)
	fmt.printf(
		"%s deep hedge CVaR within 2x the Black-Scholes delta hedge's with every seed (worst seed %.0f%% higher)\n",
		"OK  " if ok_bs else "FAIL",
		-stats[8].min,
	)
	record_failures(int(!ok_cvar) + int(!ok_var) + int(!ok_bs))
}

// Terminal PnL per path of the static delta hedge (premium received, Black-Scholes
// delta at t = 0 held to maturity), its delta and the premium
@(private = "file")
_static_delta_pnl :: proc(
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
	pnl: []f64,
	static_delta: f64,
	premium: f64,
) {
	// 1. Compute Black-Scholes Delta at t=0
	d1 := (math.ln(S_0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt_f64(T))
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	delta_bs := N_d1

	// 2. Compute BS price (premium received)
	d2 := d1 - sigma * math.sqrt_f64(T)
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
	call_price := S_0 * N_d1 - K * math.exp_f64(-r * T) * N_d2

	// 3. Static hedge PnL for each path, discounted: call_price + delta * (e^{-rT} S_T - S_0)
	// - e^{-rT} payoff (the paths and payoffs are discounted)
	pnl = make([]f64, n_paths, allocator)
	for p in 0 ..< n_paths {
		s_t_idx := p * (n_steps + 1) * 3 + n_steps * 3 + 0
		S_T := paths.data.data[s_t_idx]
		pnl[p] = call_price + delta_bs * (S_T - S_0) - payoffs.data.data[p]
	}
	return pnl, delta_bs, call_price
}

// Terminal P&L per path (discounted, premium received) of the Black-Scholes delta hedge
// rebalanced at every step, with the same proportional cost as the deep hedge on every
// trade (opening at t_0, closing at T)
// (paths: column 0 the discounted spot; K in the same units)
bs_delta_hedge_pnl :: proc(
	K, T, r, sigma, cost: f64,
	paths, payoffs: ^t.Tensor,
	premium: f64,
	allocator: mem.Allocator,
) -> []f64 {
	B, S, F := paths.shape[0], paths.shape[1], paths.shape[2]
	N := S - 1
	dt := T / f64(N)
	pnl := make([]f64, B, allocator)
	for b in 0 ..< B {
		x := paths.data.data[b * S * F:][:S * F]
		v := premium
		prev := 0.0
		for k in 0 ..< N {
			tau := T - f64(k) * dt
			spot := x[k * F] * math.exp(r * f64(k) * dt) // undiscounted
			d1 := (math.ln(spot / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
			delta := 0.5 * (1 + math.erf(d1 / math.SQRT_TWO))
			v += delta * (x[(k + 1) * F] - x[k * F]) - cost * abs(delta - prev) * x[k * F]
			prev = delta
		}
		pnl[b] = v - cost * abs(prev) * x[N * F] - payoffs.data.data[b]
	}
	return pnl
}
