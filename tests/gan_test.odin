// wotan/tests/gan_test.odin
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

gan_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GAN Test: Sine Wave Generation ===")

	// Hyperparameters
	noise_dim := 32
	hidden_dim := 128
	data_dim := 64 // Generate 64-point signals
	batch_size := 32
	epochs := 2000

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
		// Random phase and amplitude
		phase := rand.float64() * 2.0 * math.PI
		amplitude := 0.5 + rand.float64() * 0.5

		for j in 0 ..< data_dim {
			x := f64(j) / f64(data_dim) * 2.0 * math.PI
			sample[j] = amplitude * math.sin(x + phase)
		}
		real_data[i] = sample
	}

	fmt.printf("Training data: %d sine wave samples\n", num_samples)
	fmt.printf("Data dimension: %d points per sample\n", data_dim)

	// Create Generator and Discriminator
	gen := nn.generator_new(noise_dim, hidden_dim, data_dim, allocator)
	defer nn.generator_free(&gen)

	disc := nn.discriminator_new(data_dim, hidden_dim, allocator)
	defer nn.discriminator_free(&disc)

	// Separate optimizers for G and D
	opt_g := nn.adam_new(0.0002, allocator = allocator)
	defer nn.adam_free(&opt_g)
	nn.generator_add_to_optimizer(&gen, &opt_g)

	opt_d := nn.adam_new(0.0002, allocator = allocator)
	defer nn.adam_free(&opt_d)
	nn.discriminator_add_to_optimizer(&disc, &opt_d)

	fmt.printf("\nTraining GAN for %d epochs...\n", epochs)
	fmt.println("Epoch | D_loss  | G_loss  | D_real  | D_fake")

	for epoch in 0 ..< epochs {
		// ==================== Train Discriminator ====================
		nn.adam_zero_grad(&opt_d)

		// Real data
		real_batch_data := l.matrix_new(f64, batch_size, data_dim, allocator)
		real_labels_data := l.matrix_new(f64, batch_size, 1, allocator)

		for b in 0 ..< batch_size {
			sample_idx := int(rand.int31()) % num_samples
			for j in 0 ..< data_dim {
				real_batch_data.data[b * data_dim + j] = real_data[sample_idx][j]
			}
			real_labels_data.data[b] = 1.0 // Label: real
		}

		real_batch := t.tensor_new(real_batch_data, false, allocator)
		real_labels := t.tensor_new(real_labels_data, false, allocator)

		// Fake data
		noise_data := l.matrix_new(f64, batch_size, noise_dim, allocator)
		for i in 0 ..< batch_size * noise_dim {
			noise_data.data[i] = rand.float64() * 2.0 - 1.0 // Uniform [-1, 1]
		}
		noise := t.tensor_new(noise_data, false, allocator)

		fake_data := nn.generator_forward(&gen, noise)
		fake_labels_data := l.matrix_new(f64, batch_size, 1, allocator)
		for i in 0 ..< batch_size {
			fake_labels_data.data[i] = 0.0 // Label: fake
		}
		fake_labels := t.tensor_new(fake_labels_data, false, allocator)

		// Discriminator predictions
		d_real := nn.discriminator_forward(&disc, real_batch)
		d_fake := nn.discriminator_forward(&disc, fake_data)

		// D loss: BCE for real + BCE for fake
		d_loss_real := t.tensor_binary_cross_entropy(d_real, real_labels)
		d_loss_fake := t.tensor_binary_cross_entropy(d_fake, fake_labels)
		d_loss := t.tensor_add(d_loss_real, d_loss_fake)

		// Backward and step
		t.tensor_backward(d_loss)
		nn.adam_step(&opt_d)

		// ✅ Save values BEFORE freeing (tensor_free_graph will free d_real, d_fake too)
		d_loss_val := d_loss.data.data[0]

		// ✅ Compute averages BEFORE freeing
		avg_d_real := 0.0
		avg_d_fake := 0.0
		for i in 0 ..< batch_size {
			avg_d_real += d_real.data.data[i]
			avg_d_fake += d_fake.data.data[i]
		}
		avg_d_real /= f64(batch_size)
		avg_d_fake /= f64(batch_size)

		// Cleanup D training tensors
		t.tensor_free_graph(d_loss)
		t.tensor_free(real_batch)
		t.tensor_free(real_labels)
		t.tensor_free(noise)
		t.tensor_free(fake_data)
		t.tensor_free(fake_labels)

		// ==================== Train Generator ====================
		nn.adam_zero_grad(&opt_g)

		// Generate fake data
		noise_data2 := l.matrix_new(f64, batch_size, noise_dim, allocator)
		for i in 0 ..< batch_size * noise_dim {
			noise_data2.data[i] = rand.float64() * 2.0 - 1.0
		}
		noise2 := t.tensor_new(noise_data2, false, allocator)

		fake_data2 := nn.generator_forward(&gen, noise2)

		// We want D to classify fake as real (label = 1)
		g_labels_data := l.matrix_new(f64, batch_size, 1, allocator)
		for i in 0 ..< batch_size {
			g_labels_data.data[i] = 1.0
		}
		g_labels := t.tensor_new(g_labels_data, false, allocator)

		// Forward through D (gradients flow to G, not D since we don't step opt_d)
		d_on_fake := nn.discriminator_forward(&disc, fake_data2)

		// G loss: BCE with label=1 (want D to think it's real)
		g_loss := t.tensor_binary_cross_entropy(d_on_fake, g_labels)

		// Backward and step (only G's params updated)
		t.tensor_backward(g_loss)
		nn.adam_step(&opt_g)

		// ✅ Save loss value before freeing
		g_loss_val := g_loss.data.data[0]

		// Cleanup G training tensors
		t.tensor_free_graph(g_loss)
		t.tensor_free(noise2)
		t.tensor_free(fake_data2)
		t.tensor_free(g_labels)

		// Print progress
		// In gan_test, change the print statement to:
		if epoch % 200 == 0 {
			fmt.printf(
				"Epoch %d | D_loss: %.4f | G_loss: %.4f | D_real: %.3f | D_fake: %.3f\n",
				epoch,
				d_loss_val,
				g_loss_val,
				avg_d_real,
				avg_d_fake,
			)
		}
	}

	// ==================== Generate samples ====================
	fmt.println("\n=== Generated Samples ===")
	fmt.println("Generating 5 sine wave samples:\n")

	for sample in 0 ..< 5 {
		// Generate noise
		noise_data := l.matrix_new(f64, 1, noise_dim, allocator)
		for i in 0 ..< noise_dim {
			noise_data.data[i] = rand.float64() * 2.0 - 1.0
		}
		noise := t.tensor_new(noise_data, false, allocator)

		// Generate fake data
		fake := nn.generator_forward(&gen, noise)

		// Print as simple visualization
		fmt.printf("Sample %d: ", sample + 1)
		for i in 0 ..< 32 { 	// Print first 32 points
			val := fake.data.data[i]
			// Map [-1, 1] to [0, 20] for visualization
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

	fmt.println("\n✓ GAN test completed!")
}

gan_test_v2 :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GAN Test V2: Improved Training ===")

	// Hyperparameters - adjusted for stability
	noise_dim := 32
	hidden_dim := 64 // Reduced to prevent D from being too strong
	data_dim := 64
	batch_size := 16 // Smaller batch for stability
	epochs := 3000

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

	// Create Generator and Discriminator
	gen := nn.generator_new(noise_dim, hidden_dim, data_dim, allocator)
	defer nn.generator_free(&gen)

	disc := nn.discriminator_new(data_dim, hidden_dim, allocator)
	defer nn.discriminator_free(&disc)

	// ✅ Use different learning rates - G learns slower than D
	opt_g := nn.adam_new(0.0001, allocator = allocator) // Slower for G
	defer nn.adam_free(&opt_g)
	nn.generator_add_to_optimizer(&gen, &opt_g)

	opt_d := nn.adam_new(0.0002, allocator = allocator) // Faster for D
	defer nn.adam_free(&opt_d)
	nn.discriminator_add_to_optimizer(&disc, &opt_d)

	fmt.printf("\nTraining GAN for %d epochs...\n", epochs)
	fmt.println("Epoch | D_loss  | G_loss  | D_real  | D_fake")

	// ✅ Declare variables outside the loop for consistent formatting
	d_loss_val := 0.0
	g_loss_val := 0.0
	avg_d_real := 0.0
	avg_d_fake := 0.0

	for epoch in 0 ..< epochs {
		// ✅ Train D multiple times per G update (1:5 ratio)
		d_updates := 5
		for _ in 0 ..< d_updates {
			nn.adam_zero_grad(&opt_d)

			// Real data
			real_batch_data := l.matrix_new(f64, batch_size, data_dim, allocator)
			real_labels_data := l.matrix_new(f64, batch_size, 1, allocator)

			for b in 0 ..< batch_size {
				sample_idx := int(rand.int31()) % num_samples
				for j in 0 ..< data_dim {
					real_batch_data.data[b * data_dim + j] = real_data[sample_idx][j]
				}
				// ✅ Label smoothing: use 0.9 instead of 1.0
				real_labels_data.data[b] = 0.9
			}

			real_batch := t.tensor_new(real_batch_data, false, allocator)
			real_labels := t.tensor_new(real_labels_data, false, allocator)

			// Fake data
			noise_data := l.matrix_new(f64, batch_size, noise_dim, allocator)
			for i in 0 ..< batch_size * noise_dim {
				noise_data.data[i] = rand.float64() * 2.0 - 1.0
			}
			noise := t.tensor_new(noise_data, false, allocator)

			fake_data := nn.generator_forward(&gen, noise)
			fake_labels_data := l.matrix_new(f64, batch_size, 1, allocator)
			for i in 0 ..< batch_size {
				// ✅ Label smoothing: use 0.1 instead of 0.0
				fake_labels_data.data[i] = 0.1
			}
			fake_labels := t.tensor_new(fake_labels_data, false, allocator)

			// Discriminator predictions
			d_real := nn.discriminator_forward(&disc, real_batch)
			d_fake := nn.discriminator_forward(&disc, fake_data)

			// D loss
			d_loss_real := t.tensor_binary_cross_entropy(d_real, real_labels)
			d_loss_fake := t.tensor_binary_cross_entropy(d_fake, fake_labels)
			d_loss := t.tensor_add(d_loss_real, d_loss_fake)

			t.tensor_backward(d_loss)
			nn.adam_step(&opt_d)

			// Save values before freeing
			d_loss_val = d_loss.data.data[0]
			avg_d_real = 0.0
			avg_d_fake = 0.0
			for i in 0 ..< batch_size {
				avg_d_real += d_real.data.data[i]
				avg_d_fake += d_fake.data.data[i]
			}
			avg_d_real /= f64(batch_size)
			avg_d_fake /= f64(batch_size)

			// Cleanup
			t.tensor_free_graph(d_loss)
			t.tensor_free(real_batch)
			t.tensor_free(real_labels)
			t.tensor_free(noise)
			t.tensor_free(fake_data)
			t.tensor_free(fake_labels)
		}

		// Train G once
		nn.adam_zero_grad(&opt_g)

		noise_data2 := l.matrix_new(f64, batch_size, noise_dim, allocator)
		for i in 0 ..< batch_size * noise_dim {
			noise_data2.data[i] = rand.float64() * 2.0 - 1.0
		}
		noise2 := t.tensor_new(noise_data2, false, allocator)

		fake_data2 := nn.generator_forward(&gen, noise2)

		g_labels_data := l.matrix_new(f64, batch_size, 1, allocator)
		for i in 0 ..< batch_size {
			// ✅ Use 0.9 for G's target (label smoothing)
			g_labels_data.data[i] = 0.9
		}
		g_labels := t.tensor_new(g_labels_data, false, allocator)

		d_on_fake := nn.discriminator_forward(&disc, fake_data2)
		g_loss := t.tensor_binary_cross_entropy(d_on_fake, g_labels)

		t.tensor_backward(g_loss)
		nn.adam_step(&opt_g)

		g_loss_val = g_loss.data.data[0]

		t.tensor_free_graph(g_loss)
		t.tensor_free(noise2)
		t.tensor_free(fake_data2)
		t.tensor_free(g_labels)

		// Print progress - ✅ Consistent format with other tests
		if epoch % 200 == 0 {
			fmt.printf(
				"Epoch %d | D_loss: %.4f | G_loss: %.4f | D_real: %.3f | D_fake: %.3f\n",
				epoch,
				d_loss_val,
				g_loss_val,
				avg_d_real,
				avg_d_fake,
			)
		}
	}

	// Generate samples
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

	fmt.println("\n✓ GAN V2 test completed!")
}
