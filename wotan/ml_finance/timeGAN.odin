package ml_finance

import l "../linalg"
import nn "../nn"
import tensor "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// TimeGAN Architecture
// ============================================================================

TimeGAN :: struct {
	embedder:      ^nn.Sequential,
	recovery:      ^nn.Sequential,
	generator:     ^nn.Sequential,
	discriminator: ^nn.Sequential,
	supervisor:    ^nn.Sequential,
	optimizer_e:   nn.Adam,
	optimizer_r:   nn.Adam,
	optimizer_g:   nn.Adam,
	optimizer_d:   nn.Adam,
	optimizer_s:   nn.Adam,
	seq_len:       int,
	feature_dim:   int,
	latent_dim:    int,
	allocator:     mem.Allocator,
}

new_timegan :: proc(
	seq_len: int,
	feature_dim: int,
	latent_dim: int = 32,
	hidden_dim: int = 64,
	lr: f64 = 1e-3,
	alloc: mem.Allocator = context.allocator,
) -> ^TimeGAN {
	tg := new(TimeGAN, alloc)
	tg.seq_len = seq_len
	tg.feature_dim = feature_dim
	tg.latent_dim = latent_dim
	tg.allocator = alloc

	// Embedder: feature_dim -> latent_dim
	tg.embedder = nn.sequential_new(alloc)
	nn.sequential_add(tg.embedder, nn.lstm_layer_new(feature_dim, hidden_dim, alloc))
	nn.sequential_add(tg.embedder, nn.linear_layer_new(hidden_dim, latent_dim, alloc))

	// Recovery: latent_dim -> feature_dim
	tg.recovery = nn.sequential_new(alloc)
	nn.sequential_add(tg.recovery, nn.lstm_layer_new(latent_dim, hidden_dim, alloc))
	nn.sequential_add(tg.recovery, nn.linear_layer_new(hidden_dim, feature_dim, alloc))

	// Generator: latent_dim -> latent_dim (generates in latent space)
	tg.generator = nn.sequential_new(alloc)
	nn.sequential_add(tg.generator, nn.lstm_layer_new(latent_dim, hidden_dim, alloc))
	nn.sequential_add(tg.generator, nn.linear_layer_new(hidden_dim, latent_dim, alloc))
	nn.sequential_add(tg.generator, nn.Activation.Sigmoid)

	// Discriminator: latent_dim -> 1 (real vs fake)
	tg.discriminator = nn.sequential_new(alloc)
	nn.sequential_add(tg.discriminator, nn.lstm_layer_new(latent_dim, hidden_dim, alloc))
	nn.sequential_add(tg.discriminator, nn.linear_layer_new(hidden_dim, 1, alloc))
	nn.sequential_add(tg.discriminator, nn.Activation.Sigmoid)

	// Supervisor: latent_dim -> feature_dim (helps generator learn temporal dynamics)
	tg.supervisor = nn.sequential_new(alloc)
	nn.sequential_add(tg.supervisor, nn.lstm_layer_new(latent_dim, hidden_dim, alloc))
	nn.sequential_add(tg.supervisor, nn.linear_layer_new(hidden_dim, feature_dim, alloc))

	// Optimizers
	tg.optimizer_e = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_r = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_g = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_d = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_s = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)

	nn.sequential_add_to_adam(tg.embedder, &tg.optimizer_e)
	nn.sequential_add_to_adam(tg.recovery, &tg.optimizer_r)
	nn.sequential_add_to_adam(tg.generator, &tg.optimizer_g)
	nn.sequential_add_to_adam(tg.discriminator, &tg.optimizer_d)
	nn.sequential_add_to_adam(tg.supervisor, &tg.optimizer_s)

	return tg
}

timegan_free :: proc(tg: ^TimeGAN) {
	nn.sequential_free(tg.embedder)
	nn.sequential_free(tg.recovery)
	nn.sequential_free(tg.generator)
	nn.sequential_free(tg.discriminator)
	nn.sequential_free(tg.supervisor)
	nn.adam_free(&tg.optimizer_e)
	nn.adam_free(&tg.optimizer_r)
	nn.adam_free(&tg.optimizer_g)
	nn.adam_free(&tg.optimizer_d)
	nn.adam_free(&tg.optimizer_s)
	free(tg, tg.allocator)
}

// ============================================================================
// Data Preparation
// ============================================================================

// prepare_market_data normalizes and sequences market data for TimeGAN
prepare_market_data :: proc(
	prices: []f64,
	volumes: []f64,
	indicators: []f64,
	n_indicators: int,
	seq_len: int,
	alloc: mem.Allocator,
) -> (
	sequences: []f64,
	n_sequences: int,
	feature_dim: int,
) {
	n_days := len(prices)
	feature_dim = 2 + n_indicators // price_return, volume_change, indicators

	// Normalize data
	log_returns := make([]f64, n_days, alloc)
	defer delete(log_returns, alloc)
	for i in 1 ..< n_days {
		if prices[i - 1] > 0 {
			log_returns[i] = math.ln_f64(prices[i] / prices[i - 1])
		}
	}

	vol_changes := make([]f64, n_days, alloc)
	defer delete(vol_changes, alloc)
	for i in 1 ..< n_days {
		if volumes[i - 1] > 0 {
			vol_changes[i] = math.ln_f64(volumes[i] / volumes[i - 1])
		}
	}

	// Clip outliers
	for i in 0 ..< n_days {
		log_returns[i] = math.max(-0.1, math.min(0.1, log_returns[i]))
		vol_changes[i] = math.max(-2.0, math.min(2.0, vol_changes[i]))
	}

	// Create sequences
	n_sequences = n_days - seq_len
	sequences = make([]f64, n_sequences * seq_len * feature_dim, alloc)

	for s in 0 ..< n_sequences {
		for t in 0 ..< seq_len {
			idx := (s * seq_len + t) * feature_dim
			day := s + t
			sequences[idx + 0] = log_returns[day]
			sequences[idx + 1] = vol_changes[day]
			for ind in 0 ..< n_indicators {
				sequences[idx + 2 + ind] = indicators[day * n_indicators + ind]
			}
		}
	}

	return sequences, n_sequences, feature_dim
}

// ============================================================================
// Forward Passes
// ============================================================================

// embedder_forward processes a sequence through the embedder
embedder_forward :: proc(tg: ^TimeGAN, sequence: []f64, alloc: mem.Allocator) -> ^tensor.Tensor {
	input := tensor.tensor_new(l.matrix_new(f64, tg.seq_len, tg.feature_dim, alloc), false, alloc)
	copy(input.data.data, sequence)
	input.shape = [4]int{1, tg.seq_len, tg.feature_dim, 1}

	h := nn.sequential_forward(tg.embedder, input)
	tensor.tensor_free_graph(input)
	return h
}

// recovery_forward processes latent sequence back to feature space
recovery_forward :: proc(
	tg: ^TimeGAN,
	latent_seq: ^tensor.Tensor,
	alloc: mem.Allocator,
) -> ^tensor.Tensor {
	x_hat := nn.sequential_forward(tg.recovery, latent_seq)
	return x_hat
}

// generator_forward generates synthetic latent sequences
generator_forward :: proc(
	tg: ^TimeGAN,
	noise: ^tensor.Tensor,
	alloc: mem.Allocator,
) -> ^tensor.Tensor {
	e_hat := nn.sequential_forward(tg.generator, noise)
	return e_hat
}

// discriminator_forward classifies sequences as real or fake
discriminator_forward :: proc(
	tg: ^TimeGAN,
	latent_seq: ^tensor.Tensor,
	alloc: mem.Allocator,
) -> ^tensor.Tensor {
	y_hat := nn.sequential_forward(tg.discriminator, latent_seq)
	return y_hat
}

// supervisor_forward predicts next feature from latent
supervisor_forward :: proc(
	tg: ^TimeGAN,
	latent_seq: ^tensor.Tensor,
	alloc: mem.Allocator,
) -> ^tensor.Tensor {
	y := nn.sequential_forward(tg.supervisor, latent_seq)
	return y
}

// ============================================================================
// Training Steps
// ============================================================================

// train_embedder_recovery trains the embedding and recovery networks
train_embedder_recovery :: proc(
	tg: ^TimeGAN,
	real_data: []f64,
	batch_size: int,
	alloc: mem.Allocator,
) -> f64 {
	nn.adam_zero_grad(&tg.optimizer_e)
	nn.adam_zero_grad(&tg.optimizer_r)

	total_loss := 0.0
	n_seqs := len(real_data) / (tg.seq_len * tg.feature_dim)

	if n_seqs == 0 {
		fmt.printf(
			"ERROR: real_data is empty or seq_len/feature_dim is invalid! (seq_len=%d, feature_dim=%d)\n",
			tg.seq_len,
			tg.feature_dim,
		)
		return 0.0
	}

	for b in 0 ..< batch_size {
		// Sample a random sequence
		seq_idx := rand.int_range(0, n_seqs - 1)
		seq_start := seq_idx * tg.seq_len * tg.feature_dim
		seq := real_data[seq_start:seq_start + tg.seq_len * tg.feature_dim]

		// Embed
		h := embedder_forward(tg, seq, alloc)

		// Recover
		x_hat := recovery_forward(tg, h, alloc)

		// ✅ DYNAMIC FIX: Create target with the EXACT same matrix dimensions as x_hat
		target := tensor.tensor_new(
			l.matrix_new(f64, x_hat.data.rows, x_hat.data.cols, alloc),
			false,
			alloc,
		)
		copy(target.data.data, seq)
		target.shape = x_hat.shape // Copy the shape metadata too

		loss := tensor.tensor_mse_loss(x_hat, target)
		total_loss += loss.data.data[0]

		tensor.tensor_backward(loss)

		tensor.tensor_free_graph(h)
		tensor.tensor_free_graph(x_hat)
		tensor.tensor_free_graph(target)
		tensor.tensor_free_graph(loss)
	}

	nn.adam_step(&tg.optimizer_e)
	nn.adam_step(&tg.optimizer_r)

	return total_loss / f64(batch_size)
}

// train_supervisor pre-trains the supervisor to predict features from latent space
train_supervisor :: proc(
	tg: ^TimeGAN,
	real_data: []f64,
	batch_size: int,
	alloc: mem.Allocator,
) -> f64 {
	nn.adam_zero_grad(&tg.optimizer_s)

	total_loss := 0.0

	for b in 0 ..< batch_size {
		// Sample a random sequence
		seq_idx := rand.int_range(0, len(real_data) / (tg.seq_len * tg.feature_dim) - 1)
		seq_start := seq_idx * tg.seq_len * tg.feature_dim
		seq := real_data[seq_start:seq_start + tg.seq_len * tg.feature_dim]

		// Embed real data
		h_real := embedder_forward(tg, seq, alloc)

		// Supervisor predicts features from latent representation
		y_pred := supervisor_forward(tg, h_real, alloc)

		// ✅ DYNAMIC FIX: Match y_pred's exact matrix dimensions
		target_feat := tensor.tensor_new(
			l.matrix_new(f64, y_pred.data.rows, y_pred.data.cols, alloc),
			false,
			alloc,
		)
		copy(target_feat.data.data, seq)
		target_feat.shape = y_pred.shape

		// Compute MSE loss
		loss := tensor.tensor_mse_loss(y_pred, target_feat)
		total_loss += loss.data.data[0]

		tensor.tensor_backward(loss)

		tensor.tensor_free_graph(h_real)
		tensor.tensor_free_graph(y_pred)
		tensor.tensor_free_graph(target_feat)
		tensor.tensor_free_graph(loss)
	}

	nn.adam_step(&tg.optimizer_s)
	return total_loss / f64(batch_size)
}

// train_generator trains the generator with adversarial + supervised loss
train_generator :: proc(
	tg: ^TimeGAN,
	real_data: []f64,
	batch_size: int,
	alloc: mem.Allocator,
) -> f64 {
	nn.adam_zero_grad(&tg.optimizer_g)

	total_loss := 0.0

	for b in 0 ..< batch_size {
		// Generate noise
		noise := tensor.tensor_new(
			l.matrix_new(f64, tg.seq_len, tg.latent_dim, alloc),
			false,
			alloc,
		)
		for i in 0 ..< tg.seq_len * tg.latent_dim {
			noise.data.data[i] = rand.float64()
		}
		noise.shape = [4]int{1, tg.seq_len, tg.latent_dim, 1}

		// Generate synthetic latent sequence
		e_hat := generator_forward(tg, noise, alloc)

		// Adversarial loss (fool discriminator)
		y_hat_fake := discriminator_forward(tg, e_hat, alloc)

		// ✅ DYNAMIC FIX: Match y_hat_fake's exact matrix dimensions for BCE
		target_real := tensor.tensor_new(
			l.matrix_new(f64, y_hat_fake.data.rows, y_hat_fake.data.cols, alloc),
			false,
			alloc,
		)
		for i in 0 ..< len(target_real.data.data) {
			target_real.data.data[i] = 1.0
		}
		target_real.shape = y_hat_fake.shape

		adv_loss := tensor.tensor_bce_loss(y_hat_fake, target_real)

		// Supervised loss (predict next feature)
		seq_idx := rand.int_range(0, len(real_data) / (tg.seq_len * tg.feature_dim) - 1)
		seq_start := seq_idx * tg.seq_len * tg.feature_dim
		seq := real_data[seq_start:seq_start + tg.seq_len * tg.feature_dim]

		h_real := embedder_forward(tg, seq, alloc)
		y_pred := supervisor_forward(tg, h_real, alloc)

		// ✅ DYNAMIC FIX: Match y_pred's exact matrix dimensions
		target_feat := tensor.tensor_new(
			l.matrix_new(f64, y_pred.data.rows, y_pred.data.cols, alloc),
			false,
			alloc,
		)
		copy(target_feat.data.data, seq)
		target_feat.shape = y_pred.shape

		sup_loss := tensor.tensor_mse_loss(y_pred, target_feat)

		// Combined loss
		combined := tensor.tensor_add(adv_loss, sup_loss)
		total_loss += combined.data.data[0]

		tensor.tensor_backward(combined)

		tensor.tensor_free_graph(noise)
		tensor.tensor_free_graph(e_hat)
		tensor.tensor_free_graph(y_hat_fake)
		tensor.tensor_free_graph(target_real)
		tensor.tensor_free_graph(adv_loss)
		tensor.tensor_free_graph(h_real)
		tensor.tensor_free_graph(y_pred)
		tensor.tensor_free_graph(target_feat)
		tensor.tensor_free_graph(sup_loss)
		tensor.tensor_free_graph(combined)
	}

	nn.adam_step(&tg.optimizer_g)

	return total_loss / f64(batch_size)
}

// train_discriminator trains the discriminator to distinguish real from fake
train_discriminator :: proc(
	tg: ^TimeGAN,
	real_data: []f64,
	batch_size: int,
	alloc: mem.Allocator,
) -> f64 {
	nn.adam_zero_grad(&tg.optimizer_d)

	total_loss := 0.0

	for b in 0 ..< batch_size {
		// Real sequence
		seq_idx := rand.int_range(0, len(real_data) / (tg.seq_len * tg.feature_dim) - 1)
		seq_start := seq_idx * tg.seq_len * tg.feature_dim
		seq := real_data[seq_start:seq_start + tg.seq_len * tg.feature_dim]

		h_real := embedder_forward(tg, seq, alloc)
		y_hat_real := discriminator_forward(tg, h_real, alloc)

		// ✅ DYNAMIC FIX: Match y_hat_real's exact matrix dimensions for BCE
		target_real := tensor.tensor_new(
			l.matrix_new(f64, y_hat_real.data.rows, y_hat_real.data.cols, alloc),
			false,
			alloc,
		)
		for i in 0 ..< len(target_real.data.data) {
			target_real.data.data[i] = 1.0
		}
		target_real.shape = y_hat_real.shape

		loss_real := tensor.tensor_bce_loss(y_hat_real, target_real)

		// Fake sequence
		noise := tensor.tensor_new(
			l.matrix_new(f64, tg.seq_len, tg.latent_dim, alloc),
			false,
			alloc,
		)
		for i in 0 ..< tg.seq_len * tg.latent_dim {
			noise.data.data[i] = rand.float64()
		}
		noise.shape = [4]int{1, tg.seq_len, tg.latent_dim, 1}

		e_hat := generator_forward(tg, noise, alloc)
		y_hat_fake := discriminator_forward(tg, e_hat, alloc)

		// ✅ DYNAMIC FIX: Match y_hat_fake's exact matrix dimensions for BCE
		target_fake := tensor.tensor_new(
			l.matrix_new(f64, y_hat_fake.data.rows, y_hat_fake.data.cols, alloc),
			false,
			alloc,
		)
		for i in 0 ..< len(target_fake.data.data) {
			target_fake.data.data[i] = 0.0
		}
		target_fake.shape = y_hat_fake.shape

		loss_fake := tensor.tensor_bce_loss(y_hat_fake, target_fake)

		// Combined loss
		combined := tensor.tensor_add(loss_real, loss_fake)
		total_loss += combined.data.data[0]

		tensor.tensor_backward(combined)

		tensor.tensor_free_graph(h_real)
		tensor.tensor_free_graph(y_hat_real)
		tensor.tensor_free_graph(target_real)
		tensor.tensor_free_graph(loss_real)
		tensor.tensor_free_graph(noise)
		tensor.tensor_free_graph(e_hat)
		tensor.tensor_free_graph(y_hat_fake)
		tensor.tensor_free_graph(target_fake)
		tensor.tensor_free_graph(loss_fake)
		tensor.tensor_free_graph(combined)
	}

	nn.adam_step(&tg.optimizer_d)

	return total_loss / f64(batch_size)
}
// ============================================================================
// Generation
// ============================================================================

// generate_synthetic_data creates synthetic market sequences
generate_synthetic_data :: proc(tg: ^TimeGAN, n_sequences: int, alloc: mem.Allocator) -> []f64 {
	output := make([]f64, n_sequences * tg.seq_len * tg.feature_dim, alloc)

	for s in 0 ..< n_sequences {
		// Generate noise
		noise := tensor.tensor_new(
			l.matrix_new(f64, tg.seq_len, tg.latent_dim, alloc),
			false,
			alloc,
		)
		for i in 0 ..< tg.seq_len * tg.latent_dim {
			noise.data.data[i] = rand.float64()
		}

		// ✅ CRITICAL FIX: Set the shape so the LSTM knows the sequence length!
		noise.shape = [4]int{1, tg.seq_len, tg.latent_dim, 1}

		// Generate latent sequence
		e_hat := generator_forward(tg, noise, alloc)

		// Recover to feature space
		x_hat := recovery_forward(tg, e_hat, alloc)

		// Copy to output
		copy(
			output[s * tg.seq_len * tg.feature_dim:(s + 1) * tg.seq_len * tg.feature_dim],
			x_hat.data.data,
		)

		tensor.tensor_free_graph(noise)
		tensor.tensor_free_graph(e_hat)
		tensor.tensor_free_graph(x_hat)
	}

	return output
}
