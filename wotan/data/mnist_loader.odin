package data

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:mem"
import "core:os"

// ============================================================================
// MNIST Dataset
// ============================================================================

MNIST_Dataset :: struct {
	images:      ^t.Tensor, // (N, 1, 28, 28) normalized to [0, 1]
	labels:      []int, // (N,) integer labels 0-9
	num_samples: int,
	allocator:   mem.Allocator,
}

// Helper to read big-endian uint32 (since Odin doesn't have encoding/binary)
read_big_endian_u32 :: proc(data: []u8) -> u32 {
	return (u32(data[0]) << 24) | (u32(data[1]) << 16) | (u32(data[2]) << 8) | u32(data[3])
}

// mnist_load loads the MNIST dataset from IDX files
mnist_load :: proc(
	images_path: string,
	labels_path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	dataset: MNIST_Dataset,
	ok: bool,
) {
	// 1. Read image file
	img_data, img_err := os.read_entire_file(images_path, allocator)
	if img_err != nil {
		fmt.printf("Error reading %s: %v\n", images_path, img_err)
		return dataset, false
	}
	defer delete(img_data, allocator)

	// 2. Read label file
	lab_data, lab_err := os.read_entire_file(labels_path, allocator)
	if lab_err != nil {
		fmt.printf("Error reading %s: %v\n", labels_path, lab_err)
		return dataset, false
	}
	defer delete(lab_data, allocator)

	// 3. Parse image header
	if len(img_data) < 16 {
		fmt.println("Image file too small")
		return dataset, false
	}

	img_magic := read_big_endian_u32(img_data[0:4])
	num_images := int(read_big_endian_u32(img_data[4:8]))
	rows := int(read_big_endian_u32(img_data[8:12]))
	cols := int(read_big_endian_u32(img_data[12:16]))

	if img_magic != 0x00000803 {
		fmt.printf("Invalid image magic: 0x%x\n", img_magic)
		return dataset, false
	}

	fmt.printf("MNIST Images: %d samples, %dx%d\n", num_images, rows, cols)

	// 4. Parse label header
	if len(lab_data) < 8 {
		fmt.println("Label file too small")
		return dataset, false
	}

	lab_magic := read_big_endian_u32(lab_data[0:4])
	num_labels := int(read_big_endian_u32(lab_data[4:8]))

	if lab_magic != 0x00000801 {
		fmt.printf("Invalid label magic: 0x%x\n", lab_magic)
		return dataset, false
	}

	if num_images != num_labels {
		fmt.printf("Mismatch: %d images vs %d labels\n", num_images, num_labels)
		return dataset, false
	}

	fmt.printf("MNIST Labels: %d samples\n", num_labels)

	// 5. Allocate tensor for images: (N, 1, 28, 28)
	img_tensor := t.tensor_new_4d(num_images, 1, rows, cols, false, allocator)

	// 6. Copy and normalize pixel data (ubyte -> f64, scale to [0, 1])
	pixel_offset := 16 // Skip header
	for i in 0 ..< num_images {
		for j in 0 ..< rows * cols {
			idx := pixel_offset + i * (rows * cols) + j
			pixel_val := f64(img_data[idx]) / 255.0 // Normalize to [0, 1]
			img_tensor.data.data[i * (rows * cols) + j] = pixel_val
		}
	}

	// 7. Extract labels
	labels := make([]int, num_labels, allocator)
	label_offset := 8 // Skip header
	for i in 0 ..< num_labels {
		labels[i] = int(lab_data[label_offset + i])
	}

	// 8. Build dataset
	dataset.images = img_tensor
	dataset.labels = labels
	dataset.num_samples = num_images
	dataset.allocator = allocator

	return dataset, true
}

// mnist_free cleans up the dataset
mnist_free :: proc(dataset: ^MNIST_Dataset) {
	if dataset.images != nil {
		t.tensor_free(dataset.images)
	}
	if dataset.labels != nil {
		delete(dataset.labels, dataset.allocator)
	}
}

// mnist_get_batch returns a batch of images and labels
mnist_get_batch :: proc(
	dataset: ^MNIST_Dataset,
	start_idx: int,
	batch_size: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	batch_images: ^t.Tensor,
	batch_labels: []int,
) {
	N := dataset.num_samples
	actual_size := batch_size
	if start_idx + batch_size > N {
		actual_size = N - start_idx
	}

	// Create batch tensor
	batch_images = t.tensor_new_4d(actual_size, 1, 28, 28, false, allocator)

	// Copy images
	for i in 0 ..< actual_size {
		src_offset := (start_idx + i) * 784 // 28*28
		dst_offset := i * 784
		copy(
			batch_images.data.data[dst_offset:dst_offset + 784],
			dataset.images.data.data[src_offset:src_offset + 784],
		)
	}

	// Copy labels
	batch_labels = make([]int, actual_size, allocator)
	for i in 0 ..< actual_size {
		batch_labels[i] = dataset.labels[start_idx + i]
	}

	return batch_images, batch_labels
}
