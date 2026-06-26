// wotan/tests/wgan_test.odin
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

wgan_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== WGAN Test: Wasserstein GAN ===")

	// Hyperparameters
	noise_dim := 64
	hidden_dim_gen := 64
	hidden_dim_critic := 24
	data_dim := 64
	batch_size := 16
	epochs := 5000
	critic_iterations := 2
	clip_value := 0.03
	learning_rate := 0.0002

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
		sample := make([]f64, data_dim, allocator)
		phase := rand.float64() * 2.0 * math.PI
		amplitude := 0.5 + rand.float64() * 0.5

		for j in 0 ..< data_dim {
			x := f64(j) / f64(data_dim) * 2.0 * math.PI
			sample[j] = amplitude * math.sin(x + phase)
		}
		real_data[i] = sample
	}

	fmt.printf("Training data: %d sine wave samples\n", num_samples)

	// Create Generator and Critic
	gen := nn.generator_new_custom(noise_dim, hidden_dim_gen, data_dim, allocator)
	defer nn.generator_free(&gen)

	critic := nn.discriminator_new_custom(data_dim, hidden_dim_critic, allocator)
	defer nn.discriminator_free(&critic)

	// Initialize weights
	nn.initialize_generator_weights(&gen, allocator)
	nn.initialize_discriminator_weights(&critic, allocator)

	opt_g := nn.adam_new(learning_rate, allocator = allocator)
	defer nn.adam_free(&opt_g)
	nn.generator_add_to_optimizer(&gen, &opt_g)

	opt_c := nn.adam_new(learning_rate, allocator = allocator)
	defer nn.adam_free(&opt_c)
	nn.discriminator_add_to_optimizer(&critic, &opt_c)

	fmt.printf("\nTraining WGAN for %d epochs...\n", epochs)
	fmt.println("Epoch | Critic_loss | G_loss | Wasserstein_dist")

	critic_loss_val := 0.0
	g_loss_val := 0.0
	wasserstein_dist := 0.0

	for epoch in 0 ..< epochs {
		// Train Critic
		for _ in 0 ..< critic_iterations {
			nn.adam_zero_grad(&opt_c)

			// Real data
			real_batch_data := l.matrix_new(f64, batch_size, data_dim, allocator)
			for b in 0 ..< batch_size {
				sample_idx := int(rand.int31()) % num_samples
				for j in 0 ..< data_dim {
					noise := rand.float64() * 0.02
					real_batch_data.data[b * data_dim + j] = real_data[sample_idx][j] + noise
				}
			}
			real_batch := t.tensor_new(real_batch_data, false, allocator)

			// Fake data
			noise_data := l.matrix_new(f64, batch_size, noise_dim, allocator)
			for i in 0 ..< batch_size * noise_dim {
				noise_data.data[i] = rand.float64() * 2.0 - 1.0
			}
			noise := t.tensor_new(noise_data, false, allocator)
			fake_data := nn.generator_forward(&gen, noise)

			critic_real := nn.discriminator_forward_no_sigmoid(&critic, real_batch)
			critic_fake := nn.discriminator_forward_no_sigmoid(&critic, fake_data)

			critic_real_mean := t.tensor_mean(critic_real)
			critic_fake_mean := t.tensor_mean(critic_fake)
			critic_loss := t.tensor_sub(critic_fake_mean, critic_real_mean)

			t.tensor_backward(critic_loss)
			nn.adam_step(&opt_c)

			// Weight clipping
			t.tensor_clip_weights(critic.fc1.weights, -clip_value, clip_value)
			t.tensor_clip_weights(critic.fc1.bias, -clip_value, clip_value)
			t.tensor_clip_weights(critic.fc2.weights, -clip_value, clip_value)
			t.tensor_clip_weights(critic.fc2.bias, -clip_value, clip_value)
			t.tensor_clip_weights(critic.fc3.weights, -clip_value, clip_value)
			t.tensor_clip_weights(critic.fc3.bias, -clip_value, clip_value)

			critic_loss_val = critic_loss.data.data[0]
			wasserstein_dist = critic_real_mean.data.data[0] - critic_fake_mean.data.data[0]

			t.tensor_free_graph(critic_loss)
			t.tensor_free(real_batch)
			t.tensor_free(noise)
			t.tensor_free(fake_data)
		}

		// Train Generator
		nn.adam_zero_grad(&opt_g)

		noise_data2 := l.matrix_new(f64, batch_size, noise_dim, allocator)
		for i in 0 ..< batch_size * noise_dim {
			noise_data2.data[i] = rand.float64() * 2.0 - 1.0
		}
		noise2 := t.tensor_new(noise_data2, false, allocator)
		fake_data2 := nn.generator_forward(&gen, noise2)

		critic_on_fake := nn.discriminator_forward_no_sigmoid(&critic, fake_data2)
		critic_fake_mean2 := t.tensor_mean(critic_on_fake)
		g_loss := t.tensor_neg(critic_fake_mean2)

		t.tensor_backward(g_loss)
		nn.adam_step(&opt_g)

		g_loss_val = g_loss.data.data[0]

		t.tensor_free_graph(g_loss)
		t.tensor_free(noise2)
		t.tensor_free(fake_data2)

		// Print progress in original format
		if epoch % 200 == 0 {
			fmt.printf(
				"Epoch %d | Critic_loss: %.4f | G_loss: %.4f | Wasserstein: %.4f\n",
				epoch,
				critic_loss_val,
				g_loss_val,
				wasserstein_dist,
			)
		}
	}

	// Generate samples with bar visualization
	fmt.println("\n=== Generated Samples ===")
	fmt.println("Generating 5 sine wave samples:\n")

	for sample in 0 ..< 5 {
		noise_data := l.matrix_new(f64, 1, noise_dim, allocator)
		for i in 0 ..< noise_dim {
			noise_data.data[i] = rand.float64() * 2.0 - 1.0
		}
		noise := t.tensor_new(noise_data, false, allocator)

		fake := nn.generator_forward(&gen, noise)

		fmt.printf("Sample %d: ", sample + 1)
		for i in 0 ..< 32 {
			val := fake.data.data[i]
			bar_len := int((val + 1.0) * 10.0)
			bar_len = max(0, min(20, bar_len))

			for j in 0 ..< bar_len {
				fmt.print("*")
			}
			fmt.print("\n          ")
		}
		fmt.println()

		t.tensor_free(noise)
		t.tensor_free(fake)
	}

	fmt.println("\n✓ WGAN test completed!")
}
