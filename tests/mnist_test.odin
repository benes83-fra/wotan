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
	fmt.println("\n=== Training CNN on MNIST ===")

	train_set, ok := data.mnist_load(
		"data/mnist/train-images-idx3-ubyte",
		"data/mnist/train-labels-idx1-ubyte",
		allocator,
	)
	if !ok {return}
	defer data.mnist_free(&train_set)

	model := nn.sequential_new(allocator)
	defer nn.sequential_free(&model)

	nn.sequential_add(
		&model,
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
	nn.sequential_add_to_opt(&model, &opt)

	batch_size := 64
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

			// Forward pass
			nn.adam_zero_grad(&opt)
			output := nn.sequential_forward(&model, batch_imgs)
			loss := t.tensor_cross_entropy_loss(output, batch_labs)

			// Track metrics
			epoch_loss += loss.data.data[0]

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
			t.tensor_backward(loss)
			nn.adam_step(&opt)

			// ✅ Explicit cleanup in correct order
			t.tensor_free_graph(loss) // Free computation graph
			t.tensor_free(batch_imgs) // Free batch images
			delete(batch_labs, allocator) // Free batch labels
		}

		avg_loss := epoch_loss / f64(num_batches)
		accuracy := f64(correct) / f64(total) * 100.0
		fmt.printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%%\n", epoch, avg_loss, accuracy)
	}
}
