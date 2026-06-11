package tests

import data "../wotan/data"
import t "../wotan/tensor"
import "core:fmt"
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
