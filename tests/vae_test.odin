// wotan/tests/vae_test.odin
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

vae_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== VAE Test: Sine Wave Generation ===")

	// Hyperparameters
	input_dim := 64 // 64-point signals
	hidden_dim := 128
	latent_dim := 16 // Latent space dimension
	batch_size := 16
	epochs := 3000
	learning_rate := 0.001

	// Create training data: sine waves
	num_samples := 100
	real_data := make([][]f64, num_samples, allocator)
	defer {
		for sample in real_data {
			delete(sample, allocator)
		}
		delete(real_data, allocator)
	}

	for i in 0 ..< num_samples {
		sample := make([]f64, input_dim, allocator)
		phase := rand.float64() * 2.0 * math.PI
		amplitude := 0.5 + rand.float64() * 0.5

		for j in 0 ..< input_dim {
			x := f64(j) / f64(input_dim) * 2.0 * math.PI
			sample[j] = amplitude * math.sin(x + phase)
		}
		real_data[i] = sample
	}

	fmt.printf("Training data: %d sine wave samples\n", num_samples)
	fmt.printf("Input dimension: %d points\n", input_dim)
	fmt.printf("Latent dimension: %d\n", latent_dim)

	// Create VAE
	encoder := nn.vae_encoder_new(input_dim, hidden_dim, latent_dim, allocator)
	defer nn.vae_encoder_free(&encoder)

	decoder := nn.vae_decoder_new(latent_dim, hidden_dim, input_dim, allocator)
	defer nn.vae_decoder_free(&decoder)

	// Single optimizer for entire VAE
	opt := nn.adam_new(learning_rate, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.vae_encoder_add_to_optimizer(&encoder, &opt)
	nn.vae_decoder_add_to_optimizer(&decoder, &opt)

	fmt.printf("\nTraining VAE for %d epochs...\n", epochs)
	fmt.println("Epoch | Recon_loss | KL_loss | Total_loss")

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Sample batch
		batch_data := l.matrix_new(f64, batch_size, input_dim, allocator)
		for b in 0 ..< batch_size {
			sample_idx := int(rand.int31()) % num_samples
			for j in 0 ..< input_dim {
				batch_data.data[b * input_dim + j] = real_data[sample_idx][j]
			}
		}
		x := t.tensor_new(batch_data, false, allocator)

		// Encode
		mu, log_var := nn.vae_encoder_forward(&encoder, x)

		// Reparameterize
		z := nn.reparameterize(mu, log_var, allocator)

		// Decode
		x_recon := nn.vae_decoder_forward(&decoder, z)

		// Reconstruction loss (MSE)
		recon_loss := t.tensor_mse_loss(x_recon, x)

		// KL divergence loss
		kl_loss := t.tensor_kl_divergence(mu, log_var)

		// Total loss = reconstruction + KL
		total_loss := t.tensor_add(recon_loss, kl_loss)

		// Backward and step
		t.tensor_backward(total_loss)
		nn.adam_step(&opt)

		// Save values before freeing
		recon_loss_val := recon_loss.data.data[0]
		kl_loss_val := kl_loss.data.data[0]
		total_loss_val := total_loss.data.data[0]

		// Cleanup
		t.tensor_free_graph(total_loss)
		t.tensor_free(x)
		t.tensor_free(mu)
		t.tensor_free(log_var)
		t.tensor_free(z)
		t.tensor_free(x_recon)

		// Print progress
		if epoch % 200 == 0 {
			fmt.printf(
				"Epoch %d | Recon: %.4f | KL: %.4f | Total: %.4f\n",
				epoch,
				recon_loss_val,
				kl_loss_val,
				total_loss_val,
			)
		}
	}

	// Generate samples
	fmt.println("\n=== Generated Samples ===")
	fmt.println("Generating 5 sine wave samples from random latent vectors:\n")

	for sample in 0 ..< 5 {
		// Sample from standard normal in latent space
		z_data := l.matrix_new(f64, 1, latent_dim, allocator)
		for i in 0 ..< latent_dim {
			z_data.data[i] = rand.norm_float64()
		}
		z := t.tensor_new(z_data, false, allocator)

		// Decode
		x_gen := nn.vae_decoder_forward(&decoder, z)

		// Visualize
		fmt.printf("Sample %d: ", sample + 1)
		for i in 0 ..< 32 {
			val := x_gen.data.data[i]
			bar_len := int((val + 1.0) * 10.0)
			bar_len = max(0, min(20, bar_len))

			for j in 0 ..< bar_len {
				fmt.print("*")
			}
			fmt.print("\n          ")
		}
		fmt.println()

		t.tensor_free(z)
		t.tensor_free(x_gen)
	}

	// Test interpolation
	fmt.println("\n=== Latent Space Interpolation ===")
	fmt.println("Interpolating between two random latent vectors:\n")

	// Encode two real samples
	sample1_idx := int(rand.int31()) % num_samples
	sample2_idx := int(rand.int31()) % num_samples

	x1_data := l.matrix_new(f64, 1, input_dim, allocator)
	x2_data := l.matrix_new(f64, 1, input_dim, allocator)
	for j in 0 ..< input_dim {
		x1_data.data[j] = real_data[sample1_idx][j]
		x2_data.data[j] = real_data[sample2_idx][j]
	}

	x1 := t.tensor_new(x1_data, false, allocator)
	x2 := t.tensor_new(x2_data, false, allocator)

	mu1, _ := nn.vae_encoder_forward(&encoder, x1)
	mu2, _ := nn.vae_encoder_forward(&encoder, x2)

	// Interpolate in latent space
	for step in 0 ..< 5 {
		alpha := f64(step) / 4.0 // 0.0, 0.25, 0.5, 0.75, 1.0

		z_interp_data := l.matrix_new(f64, 1, latent_dim, allocator)
		for d in 0 ..< latent_dim {
			z_interp_data.data[d] = (1.0 - alpha) * mu1.data.data[d] + alpha * mu2.data.data[d]
		}
		z_interp := t.tensor_new(z_interp_data, false, allocator)

		x_interp := nn.vae_decoder_forward(&decoder, z_interp)

		fmt.printf("Step %d (α=%.2f): ", step, alpha)
		for i in 0 ..< 32 {
			val := x_interp.data.data[i]
			bar_len := int((val + 1.0) * 10.0)
			bar_len = max(0, min(20, bar_len))

			for j in 0 ..< bar_len {
				fmt.print("*")
			}
			fmt.print("\n                    ")
		}
		fmt.println()

		t.tensor_free(z_interp)
		t.tensor_free(x_interp)
	}

	t.tensor_free(x1)
	t.tensor_free(x2)
	t.tensor_free(mu1)
	t.tensor_free(mu2)

	fmt.println("\n✓ VAE test completed!")
}
