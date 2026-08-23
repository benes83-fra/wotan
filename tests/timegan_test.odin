package tests

import w "../wotan/core"
import ml_finance "../wotan/ml_finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

timegan_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    TimeGAN: Synthetic Market Data Generation Test")
	fmt.println("======================================================================")

	// 1. Fetch Data
	fmt.println("\n[1/5] Fetching real market data (SPY)...")
	spy_df := net.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	if spy_df.rows < 500 {
		fmt.println("ERROR: Not enough data.")
		return
	}

	n_days := spy_df.rows
	close_col := w.column(&spy_df, "Close")
	volume_col := w.column(&spy_df, "Volume")

	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	defer {delete(prices, allocator); delete(volumes, allocator)}

	for i in 0 ..< n_days {
		prices[i], _ = w.column_at_float(close_col, i)
		volumes[i], _ = w.column_at_float(volume_col, i)
	}

	// 2. Compute Features
	fmt.println("\n[2/5] Computing features...")
	indicators, n_indicators := ml_finance.compute_trading_features(prices, volumes, allocator)
	defer delete(indicators, allocator)

	seq_len := 20 // 20-day sequences
	sequences, n_sequences, feature_dim := ml_finance.prepare_market_data(
		prices,
		volumes,
		indicators,
		n_indicators,
		seq_len,
		allocator,
	)
	defer delete(sequences, allocator)

	fmt.printf(
		"  Prepared %d sequences of length %d (feature dim: %d)\n",
		n_sequences,
		seq_len,
		feature_dim,
	)

	// 3. Initialize TimeGAN
	fmt.println("\n[3/5] Initializing TimeGAN...")
	latent_dim := 16
	hidden_dim := 32
	lr := 1e-3

	tg := ml_finance.new_timegan(seq_len, feature_dim, latent_dim, hidden_dim, lr, allocator)
	defer ml_finance.timegan_free(tg)

	fmt.printf(
		"  Latent Dim: %d | Hidden Dim: %d | Learning Rate: %.0e\n",
		latent_dim,
		hidden_dim,
		lr,
	)

	// 4. Training Loop
	fmt.println("\n[4/5] Training TimeGAN...")
	batch_size := 64

	// Phase 1: Embedder & Recovery
	fmt.println("  Phase 1: Training Embedder & Recovery...")
	epochs_e := 100
	for epoch in 0 ..< epochs_e {
		loss := ml_finance.train_embedder_recovery(tg, sequences, batch_size, allocator)
		if epoch % 20 == 0 || epoch == epochs_e - 1 {
			fmt.printf("    Epoch %3d | E&R Loss: %.4f\n", epoch, loss)
		}
	}

	// Phase 2: Supervisor Pre-training
	fmt.println("  Phase 2: Pre-training Supervisor...")
	epochs_s := 100
	for epoch in 0 ..< epochs_s {
		loss := ml_finance.train_supervisor(tg, sequences, batch_size, allocator)
		if epoch % 20 == 0 || epoch == epochs_s - 1 {
			fmt.printf("    Epoch %3d | Supervisor Loss: %.4f\n", epoch, loss)
		}
	}

	// Phase 3: Joint Generator & Discriminator Training
	fmt.println("  Phase 3: Joint Generator & Discriminator Training...")
	epochs_gd := 200
	for epoch in 0 ..< epochs_gd {
		d_loss := ml_finance.train_discriminator(tg, sequences, batch_size, allocator)
		g_loss := ml_finance.train_generator(tg, sequences, batch_size, allocator)

		if epoch % 40 == 0 || epoch == epochs_gd - 1 {
			fmt.printf("    Epoch %3d | D Loss: %.4f | G Loss: %.4f\n", epoch, d_loss, g_loss)
		}
	}

	// 5. Generate and Validate Synthetic Data
	fmt.println("\n[5/5] Generating and validating synthetic data...")
	n_synth_sequences := 100
	synth_sequences := ml_finance.generate_synthetic_data(tg, n_synth_sequences, allocator)
	defer delete(synth_sequences, allocator)

	// Compute basic statistics for comparison (flattened across all features)
	real_mean := 0.0
	real_std := 0.0
	for i in 0 ..< len(sequences) {
		real_mean += sequences[i]
	}
	real_mean /= f64(len(sequences))
	for i in 0 ..< len(sequences) {
		diff := sequences[i] - real_mean
		real_std += diff * diff
	}
	real_std = math.sqrt(real_std / f64(len(sequences)))

	synth_mean := 0.0
	synth_std := 0.0
	for i in 0 ..< len(synth_sequences) {
		synth_mean += synth_sequences[i]
	}
	synth_mean /= f64(len(synth_sequences))
	for i in 0 ..< len(synth_sequences) {
		diff := synth_sequences[i] - synth_mean
		synth_std += diff * diff
	}
	synth_std = math.sqrt(synth_std / f64(len(synth_sequences)))

	fmt.println("\n--- Statistical Comparison (All Features Flattened) ---")
	fmt.printf("  Real Data    -> Mean: %+8.5f | Std: %+8.5f\n", real_mean, real_std)
	fmt.printf("  Synthetic    -> Mean: %+8.5f | Std: %+8.5f\n", synth_mean, synth_std)

	// Check if means and stds are within a reasonable tolerance
	mean_diff := math.abs(real_mean - synth_mean)
	std_diff := math.abs(real_std - synth_std)

	fmt.println("\n--- Validation Result ---")
	if mean_diff < 0.01 && std_diff < 0.01 {
		fmt.println("✅ SUCCESS: Synthetic data closely matches real data statistics!")
	} else {
		fmt.println(
			"⚠️  WARNING: Synthetic data statistics diverge from real data. More training may be needed.",
		)
	}
	fmt.println("======================================================================")
}
