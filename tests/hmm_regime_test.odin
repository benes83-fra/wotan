package tests

import w "../wotan/core"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

hmm_regime_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Hidden Markov Model (HMM) Regime Switching ===")
	main_alloc := context.allocator

	fmt.println("\n--- Fetching Market Data ---")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	num_days := spy_df.rows - 1
	returns := make([]f64, num_days, main_alloc)
	defer delete(returns, main_alloc)

	for i in 1 ..< spy_df.rows {
		prev_c, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_c, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_c / prev_c)
	}
	fmt.printf("Loaded %d daily returns for SPY.\n", num_days)

	// 1. Initialize and Train HMM (3 States: e.g., Low-Vol Bull, High-Vol Bear, Choppy)
	n_states := 3
	hmm := ml_fin.hmm_new(n_states, main_alloc)
	defer ml_fin.hmm_free(&hmm)

	ml_fin.hmm_init_simple(&hmm, returns)

	fmt.println("\n--- Training HMM (Baum-Welch) ---")
	ll, converged := ml_fin.hmm_fit(&hmm, returns, 100, 1e-5)
	if converged {
		fmt.println("  ✅ HMM converged successfully.")
	} else {
		fmt.println("  ⚠️  HMM reached max iterations without full convergence.")
	}

	// 2. Decode the most likely regime sequence (Viterbi)
	fmt.println("\n--- Decoding Regimes (Viterbi) ---")
	states := ml_fin.hmm_decode(&hmm, returns, main_alloc)
	defer delete(states, main_alloc)

	// 3. Output Dashboard
	fmt.println(
		"\n╔══════════════════════════════════════════════════════════════╗",
	)
	fmt.println("║               HMM REGIME SWITCHING DASHBOARD                 ║")
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf("║  Log-Likelihood: %+10.4f   Converged: %-5v              ║\n", ll, converged)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║ %-10s | %-12s | %-12s | %-10s ║\n",
		"Regime",
		"Mean (Ann.)",
		"Vol (Ann.)",
		"Frequency",
	)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)

	for i in 0 ..< n_states {
		count := 0
		for t in 0 ..< num_days {
			if states[t] == i {count += 1}
		}
		freq := f64(count) / f64(num_days) * 100.0

		// Annualize daily returns
		mean_ann := hmm.emission_mu[i] * 252.0 * 100.0
		vol_ann := math.sqrt(hmm.emission_var[i]) * math.sqrt_f64(252.0) * 100.0

		fmt.printf("║ %-10d | %+10.2f%% | %10.2f%% | %8.1f%%  ║\n", i, mean_ann, vol_ann, freq)
	}
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.println("║  Transition Matrix (Probability of moving from Row to Col)   ║")
	for i in 0 ..< n_states {
		fmt.print("║ ")
		for j in 0 ..< n_states {
			fmt.printf("%6.3f  ", hmm.transition[i * n_states + j])
		}
		fmt.println("   ║")
	}
	fmt.println(
		"╚══════════════════════════════════════════════════════════════╝",
	)

	// Print the last 20 days of decoded regimes
	fmt.println("\n--- Last 20 Trading Days Regime Sequence ---")
	fmt.print("  ")
	start_day := num_days - 20
	if start_day < 0 {start_day = 0}
	for t in start_day ..< num_days {
		fmt.printf("%d ", states[t])
	}
	fmt.println("\n  (0=Low Vol, 1=Choppy, 2=High Vol typically)")

	fmt.println("\n✓ Hidden Markov Model Regime Test Complete!")
}
