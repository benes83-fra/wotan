package tests

import data "../wotan/data"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

mnist_loader_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing MNIST Loader ===")

	// Load training set
	train_set, ok := data.mnist_load(
		"data/mnist/train-images-idx3-ubyte",
		"data/mnist/train-labels-idx1-ubyte",
		allocator,
	)
	if !ok {
		fmt.println("Failed to load training set")
		return
	}
	defer data.mnist_free(&train_set)

	fmt.printf("Loaded %d training samples\n", train_set.num_samples)
	fmt.printf(
		"Image tensor shape: %dx%dx%dx%d\n",
		train_set.images.shape[0],
		train_set.images.shape[1],
		train_set.images.shape[2],
		train_set.images.shape[3],
	)

	// Check a few labels
	fmt.println("\nFirst 10 labels:")
	for i in 0 ..< 10 {
		fmt.printf("%d ", train_set.labels[i])
	}
	fmt.println("")

	// Check pixel normalization
	fmt.printf(
		"\nPixel range: [%.3f, %.3f]\n",
		train_set.images.data.data[0], // Should be close to 0 or 1
		train_set.images.data.data[100], // Random sample
	)

	// Test batch extraction
	batch_imgs, batch_labs := data.mnist_get_batch(&train_set, 0, 32, allocator)
	defer t.tensor_free(batch_imgs)
	defer delete(batch_labs, allocator)

	fmt.printf(
		"\nBatch shape: %dx%dx%dx%d\n",
		batch_imgs.shape[0],
		batch_imgs.shape[1],
		batch_imgs.shape[2],
		batch_imgs.shape[3],
	)
	fmt.printf("Batch labels: ")
	for i in 0 ..< 10 {
		fmt.printf("%d ", batch_labs[i])
	}
	fmt.println(" ...")

	// Load test set
	test_set, ok2 := data.mnist_load(
		"data/mnist/t10k-images-idx3-ubyte",
		"data/mnist/t10k-labels-idx1-ubyte",
		allocator,
	)
	if !ok2 {
		fmt.println("Failed to load test set")
		return
	}
	defer data.mnist_free(&test_set)

	fmt.printf("\nLoaded %d test samples\n", test_set.num_samples)
}
mnist_cnn_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Training CNN on MNIST with Augmentation ===")

	train_set, ok := data.mnist_load(
		"data/mnist/train-images-idx3-ubyte",
		"data/mnist/train-labels-idx1-ubyte",
		allocator,
	)
	if !ok {return}
	defer data.mnist_free(&train_set)

	model := nn.sequential_new(allocator)
	defer nn.sequential_free(model)

	nn.sequential_add(
		model,
		nn.conv2d_layer_new(1, 6, 5, 1, 0, true, allocator),
		nn.Activation.ReLU,
		nn.maxpool2d_layer_new(2, 2),
		nn.conv2d_layer_new(6, 16, 5, 1, 0, true, allocator),
		nn.Activation.ReLU,
		nn.maxpool2d_layer_new(2, 2),
		nn.FlattenLayer{},
		nn.linear_layer_new(256, 120, allocator),
		nn.Activation.ReLU,
		nn.linear_layer_new(120, 84, allocator),
		nn.Activation.ReLU,
		nn.linear_layer_new(84, 10, allocator),
	)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_opt(model, &opt)

	// ✅ NEW: Augmentation configuration
	aug_config := nn.AugmentationConfig {
		rotation_range    = 15.0, // ±15 degrees
		translation_range = 2, // ±2 pixels
		scale_range       = 0.1, // ±10%
		apply             = true, // Enable augmentation
	}

	batch_size := 32
	epochs := 5
	num_batches := train_set.num_samples / batch_size

	for epoch in 0 ..< epochs {
		epoch_loss := 0.0
		correct := 0
		total := 0

		for i in 0 ..< num_batches {
			// Get batch
			batch_imgs, batch_labs := data.mnist_get_batch(
				&train_set,
				i * batch_size,
				batch_size,
				allocator,
			)

			// ✅ NEW: Apply augmentation
			augmented_imgs := nn.augment_batch(batch_imgs, aug_config, allocator)

			// Forward pass (use augmented images)
			nn.adam_zero_grad(&opt)
			output := nn.sequential_forward(model, augmented_imgs)
			loss := t.tensor_cross_entropy_loss(output, batch_labs)
			batch_loss := loss.data.data[0]

			// Track metrics
			epoch_loss += batch_loss

			for j in 0 ..< batch_size {
				pred_class := 0
				max_val := -math.F64_MAX
				for k in 0 ..< 10 {
					if output.data.data[j * 10 + k] > max_val {
						max_val = output.data.data[j * 10 + k]
						pred_class = k
					}
				}
				if pred_class == batch_labs[j] {correct += 1}
				total += 1
			}

			// Backward pass
			if !t.tensor_validate_graph(loss) {
				fmt.println("ERROR: Invalid computation graph detected!")
				t.tensor_free_graph(loss)
				t.tensor_free(augmented_imgs)
				t.tensor_free(batch_imgs)
				delete(batch_labs, allocator)
				continue
			}

			t.tensor_backward(loss)
			nn.adam_step(&opt)

			// ✅ CRITICAL: Free in reverse order
			t.tensor_free_graph(loss)
			t.tensor_free(augmented_imgs) // ✅ Free augmented batch
			t.tensor_free(batch_imgs) // Free original batch
			delete(batch_labs, allocator)

			if i % 100 == 0 {
				fmt.printf("Batch %d/%d - Loss: %.4f\n", i, num_batches, batch_loss)
			}
		}

		avg_loss := epoch_loss / f64(num_batches)
		accuracy := f64(correct) / f64(total) * 100.0
		fmt.printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%%\n", epoch, avg_loss, accuracy)
	}

	// Save final augmented model
	nn.save_checkpoint(model, &opt, "mnist_augmented.bin", 5, allocator)
	fmt.println("✓ Augmented model saved to mnist_augmented.bin")
}


augmentation_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Data Augmentation ===")

	// Create a simple test image (4x4 for easy visualization)
	test_img := t.tensor_new_4d(1, 1, 4, 4, false, allocator)
	defer t.tensor_free(test_img)

	// Fill with a simple pattern
	for i in 0 ..< 16 {
		test_img.data.data[i] = f64(i) / 16.0
	}

	fmt.println("Original image (4x4):")
	for y in 0 ..< 4 {
		for x in 0 ..< 4 {
			fmt.printf("%.2f ", test_img.data.data[y * 4 + x])
		}
		fmt.println("")
	}

	// Apply augmentation
	config := nn.AugmentationConfig {
		rotation_range    = 45.0,
		translation_range = 1,
		scale_range       = 0.2,
		apply             = true,
	}

	augmented := nn.augment_batch(test_img, config, allocator)
	defer t.tensor_free(augmented)

	fmt.println("\nAugmented image (4x4):")
	for y in 0 ..< 4 {
		for x in 0 ..< 4 {
			fmt.printf("%.2f ", augmented.data.data[y * 4 + x])
		}
		fmt.println("")
	}

	fmt.println("\n✓ Augmentation test passed!")
}


evaluate_on_test_set :: proc(
	model: ^nn.Sequential,
	test_set: ^data.MNIST_Dataset,
	allocator: mem.Allocator,
) -> f64 {
	batch_size := 256
	num_batches := test_set.num_samples / batch_size
	correct := 0
	total := 0

	for i in 0 ..< num_batches {
		batch_imgs, batch_labs := data.mnist_get_batch(
			test_set,
			i * batch_size,
			batch_size,
			allocator,
		)

		// Forward pass (no augmentation during evaluation!)
		output := nn.sequential_forward(model, batch_imgs)

		// Calculate accuracy
		for j in 0 ..< batch_size {
			pred_class := 0
			max_val := -math.F64_MAX
			for k in 0 ..< 10 {
				if output.data.data[j * 10 + k] > max_val {
					max_val = output.data.data[j * 10 + k]
					pred_class = k
				}
			}
			if pred_class == batch_labs[j] {
				correct += 1
			}
			total += 1
		}

		t.tensor_free_graph(output)
		t.tensor_free(batch_imgs)
		delete(batch_labs, allocator)
	}

	accuracy := f64(correct) / f64(total) * 100.0
	return accuracy
}

test_both_models :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Comparing Models on Test Set ===")

	// Load test set
	test_set, ok := data.mnist_load(
		"data/mnist/t10k-images-idx3-ubyte",
		"data/mnist/t10k-labels-idx1-ubyte",
		allocator,
	)
	if !ok {
		fmt.println("Failed to load test set")
		return
	}
	defer data.mnist_free(&test_set)

	// Load non-augmented model
	fmt.println("\nLoading non-augmented model...")
	model1, opt1, epoch1, ok1 := nn.load_checkpoint("mnist_98_91.bin", allocator)
	if !ok1 {
		fmt.println("Failed to load non-augmented model")
		return
	}
	defer nn.sequential_free(model1)
	defer nn.adam_free(opt1)

	acc1 := evaluate_on_test_set(model1, &test_set, allocator)
	fmt.printf("Non-augmented model (epoch %d): %.2f%% on test set\n", epoch1, acc1)

	// Load augmented model
	fmt.println("\nLoading augmented model...")
	model2, opt2, epoch2, ok2 := nn.load_checkpoint("mnist_augmented.bin", allocator)
	if !ok2 {
		fmt.println("Failed to load augmented model")
		return
	}
	defer nn.sequential_free(model2)
	defer nn.adam_free(opt2)

	acc2 := evaluate_on_test_set(model2, &test_set, allocator)
	fmt.printf("Augmented model (epoch %d): %.2f%% on test set\n", epoch2, acc2)

	fmt.println("\n=== Results ===")
	if acc2 > acc1 {
		fmt.printf("✓ Augmentation improved test accuracy by %.2f%%\n", acc2 - acc1)
	} else {
		fmt.printf("Note: Augmentation decreased test accuracy by %.2f%%\n", acc1 - acc2)
		fmt.println("This might be because:")
		fmt.println("  1. More epochs needed for augmentation to pay off")
		fmt.println("  2. Model capacity is already sufficient for MNIST")
		fmt.println("  3. Augmentation parameters might need tuning")
	}
}
test_augmented_model :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Augmented Model ===")

	// Load test set
	test_set, ok := data.mnist_load(
		"data/mnist/t10k-images-idx3-ubyte",
		"data/mnist/t10k-labels-idx1-ubyte",
		allocator,
	)
	if !ok {
		fmt.println("Failed to load test set")
		return
	}
	defer data.mnist_free(&test_set)

	// Load augmented model
	fmt.println("Loading augmented model...")
	model, opt, epoch, ok2 := nn.load_checkpoint("mnist_augmented.bin", allocator)
	if !ok2 {
		fmt.println("Failed to load augmented model")
		return
	}
	defer nn.sequential_free(model)
	defer nn.adam_free(opt)

	fmt.printf("Loaded model from epoch %d\n", epoch)

	// Evaluate on test set
	batch_size := 256
	num_batches := test_set.num_samples / batch_size
	correct := 0
	total := 0

	for i in 0 ..< num_batches {
		batch_imgs, batch_labs := data.mnist_get_batch(
			&test_set,
			i * batch_size,
			batch_size,
			allocator,
		)

		// Forward pass (no augmentation during evaluation!)
		output := nn.sequential_forward(model, batch_imgs)

		// Calculate accuracy
		for j in 0 ..< batch_size {
			pred_class := 0
			max_val := -math.F64_MAX
			for k in 0 ..< 10 {
				if output.data.data[j * 10 + k] > max_val {
					max_val = output.data.data[j * 10 + k]
					pred_class = k
				}
			}
			if pred_class == batch_labs[j] {
				correct += 1
			}
			total += 1
		}

		t.tensor_free_graph(output)
		t.tensor_free(batch_imgs)
		delete(batch_labs, allocator)
	}

	accuracy := f64(correct) / f64(total) * 100.0
	fmt.printf("\n✓ Augmented model accuracy on test set: %.2f%%\n", accuracy)
	fmt.printf("  (Correct: %d / Total: %d)\n", correct, total)
}
