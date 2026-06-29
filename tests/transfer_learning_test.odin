// wotan/tests/transfer_learning_test.odin
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Task A: Learn to classify sine wave frequency (low vs high)
// ============================================================================
generate_task_a_data :: proc(
	num_samples: int,
	allocator: mem.Allocator,
) -> (
	inputs: ^t.Tensor,
	labels: []int,
) {
	seq_len := 32
	data := l.matrix_new(f64, num_samples * seq_len, 1, allocator)
	labels = make([]int, num_samples, allocator)

	for i in 0 ..< num_samples {
		is_high_freq := i < num_samples / 2
		labels[i] = bool_to_int(is_high_freq)

		freq := is_high_freq ? 4.0 : 1.0
		phase := rand.float64() * 2.0 * math.PI

		for j in 0 ..< seq_len {
			x := f64(j) / f64(seq_len) * 2.0 * math.PI
			data.data[i * seq_len + j] = math.sin(freq * x + phase)
		}
	}

	inputs = t.tensor_new(data, false, allocator)
	inputs.shape = [4]int{num_samples, 1, 1, seq_len}
	return inputs, labels
}

// ============================================================================
// Task B: Learn to classify sine wave amplitude (small vs large)
// ============================================================================
generate_task_b_data :: proc(
	num_samples: int,
	allocator: mem.Allocator,
) -> (
	inputs: ^t.Tensor,
	labels: []int,
) {
	seq_len := 32
	data := l.matrix_new(f64, num_samples * seq_len, 1, allocator)
	labels = make([]int, num_samples, allocator)

	for i in 0 ..< num_samples {
		is_large_amp := i < num_samples / 2
		labels[i] = bool_to_int(is_large_amp)

		amp := is_large_amp ? 1.0 : 0.2
		freq := 2.0
		phase := rand.float64() * 2.0 * math.PI

		for j in 0 ..< seq_len {
			x := f64(j) / f64(seq_len) * 2.0 * math.PI
			data.data[i * seq_len + j] = amp * math.sin(freq * x + phase)
		}
	}

	inputs = t.tensor_new(data, false, allocator)
	inputs.shape = [4]int{num_samples, 1, 1, seq_len}
	return inputs, labels
}

bool_to_int :: proc(b: bool) -> int {
	return b ? 1 : 0
}

// ============================================================================
// Create model architecture using Conv2d
// ============================================================================
create_model :: proc(allocator: mem.Allocator, num_classes: int) -> ^nn.Sequential {
	model := nn.sequential_new(allocator)

	// Feature extractor (shared between tasks)
	// Input: [batch, 1, 1, 32]
	nn.sequential_add(model, nn.conv2d_layer_new(1, 16, 3, 1, 1, true, allocator))
	nn.sequential_add(model, nn.Activation.ReLU)
	nn.sequential_add(model, nn.conv2d_layer_new(16, 32, 3, 1, 1, true, allocator))
	nn.sequential_add(model, nn.Activation.ReLU)
	nn.sequential_add(model, nn.FlattenLayer{})
	// After flatten: [batch, 32*1*32] = [batch, 1024]
	nn.sequential_add(model, nn.linear_layer_new(1024, 64, allocator))
	nn.sequential_add(model, nn.Activation.ReLU)

	// Classification head (task-specific)
	nn.sequential_add(model, nn.linear_layer_new(64, num_classes, allocator))

	return model
}

// ============================================================================
// Training loop
// ============================================================================
train_model :: proc(
	model: ^nn.Sequential,
	inputs: ^t.Tensor,
	labels: []int,
	epochs: int,
	learning_rate: f64,
	allocator: mem.Allocator,
) -> f64 {
	opt := nn.adam_new(learning_rate, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(model, &opt)

	batch_size := 32
	num_samples := len(labels)

	final_loss := 0.0

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		total_loss := 0.0
		num_batches := 0

		for batch_start := 0; batch_start < num_samples; batch_start += batch_size {
			batch_end := min(batch_start + batch_size, num_samples)
			batch_size_actual := batch_end - batch_start

			// Extract batch - reshape to [batch, 1, 1, seq_len]
			batch_data := l.matrix_new(f64, batch_size_actual, 32, allocator)
			for i in 0 ..< batch_size_actual {
				src_offset := (batch_start + i) * 32
				for j in 0 ..< 32 {
					batch_data.data[i * 32 + j] = inputs.data.data[src_offset + j]
				}
			}

			batch_inputs := t.tensor_new(batch_data, false, allocator)
			batch_inputs.shape = [4]int{batch_size_actual, 1, 1, 32}

			// Forward pass
			outputs := nn.sequential_forward(model, batch_inputs)

			// Compute loss
			batch_labels := labels[batch_start:batch_end]
			loss := t.tensor_cross_entropy_loss(outputs, batch_labels)

			// Backward pass
			t.tensor_backward(loss)
			nn.adam_step(&opt)

			total_loss += loss.data.data[0]
			num_batches += 1

			t.tensor_free_graph(loss)
			t.tensor_free(batch_inputs)
		}

		final_loss = total_loss / f64(num_batches)

		if epoch % 10 == 0 || epoch == epochs - 1 {
			fmt.printf("  Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, final_loss)
		}
	}

	return final_loss
}

// ============================================================================
// Evaluate model accuracy
// ============================================================================
evaluate_model :: proc(model: ^nn.Sequential, inputs: ^t.Tensor, labels: []int) -> f64 {
	num_samples := len(labels)
	correct := 0

	for i in 0 ..< num_samples {
		// Extract single sample - reshape to [1, 1, 1, 32]
		sample_data := l.matrix_new(f64, 1, 32, inputs.allocator)
		for j in 0 ..< 32 {
			sample_data.data[j] = inputs.data.data[i * 32 + j]
		}

		sample_input := t.tensor_new(sample_data, false, inputs.allocator)
		sample_input.shape = [4]int{1, 1, 1, 32}

		// Forward pass
		output := nn.sequential_forward(model, sample_input)

		// Get prediction (argmax)
		pred := 0
		max_val := output.data.data[0]
		for j in 1 ..< output.data.cols {
			if output.data.data[j] > max_val {
				max_val = output.data.data[j]
				pred = j
			}
		}

		if pred == labels[i] {
			correct += 1
		}

		t.tensor_free(sample_input)
	}

	accuracy := f64(correct) / f64(num_samples)
	return accuracy
}

// ============================================================================
// Save weights snapshot for comparison
// ============================================================================
// ============================================================================
// Save weights snapshot for comparison - ONLY frozen layers
// ============================================================================
save_weights_snapshot :: proc(model: ^nn.Sequential, allocator: mem.Allocator) -> [dynamic][]f64 {
	snapshots: [dynamic][]f64

	// ✅ Only snapshot first 5 layers (frozen feature extractor)
	for i in 0 ..< 5 {
		layer := &model.layers[i]
		switch l in layer {
		case nn.Conv2dLayer:
			snapshot := make([]f64, len(l.weight.data.data), allocator)
			copy(snapshot, l.weight.data.data)
			append(&snapshots, snapshot)
		case nn.LinearLayer:
			snapshot := make([]f64, len(l.weights.data.data), allocator)
			copy(snapshot, l.weights.data.data)
			append(&snapshots, snapshot)
		case nn.MaxPool2dLayer,
		     nn.AvgPool2dLayer,
		     nn.DropoutLayer,
		     nn.BatchNorm2dLayer,
		     nn.Activation,
		     nn.FlattenLayer,
		     nn.RNNLayer,
		     nn.GRULayer,
		     nn.LSTMLayer,
		     nn.EmbeddingLayer,
		     nn.MultiHeadAttentionLayer,
		     nn.LayerNormLayer,
		     nn.FFNLayer,
		     nn.TransformerEncoderBlock,
		     nn.TransformerEncoder:
		// Skip non-trainable or unhandled layers
		}
	}

	return snapshots
}

// ============================================================================
// Verify weights haven't changed - ONLY frozen layers
// ============================================================================
verify_weights_unchanged :: proc(model: ^nn.Sequential, snapshots: [dynamic][]f64) -> bool {
	idx := 0

	// ✅ Only verify first 5 layers (frozen feature extractor)
	for i in 0 ..< 5 {
		layer := &model.layers[i]
		switch l in layer {
		case nn.Conv2dLayer:
			if idx >= len(snapshots) {
				return false
			}
			snapshot := snapshots[idx]
			if len(snapshot) != len(l.weight.data.data) {
				return false
			}
			for j in 0 ..< len(snapshot) {
				if math.abs(snapshot[j] - l.weight.data.data[j]) > 1e-6 {
					return false
				}
			}
			idx += 1
		case nn.LinearLayer:
			if idx >= len(snapshots) {
				return false
			}
			snapshot := snapshots[idx]
			if len(snapshot) != len(l.weights.data.data) {
				return false
			}
			for j in 0 ..< len(snapshot) {
				if math.abs(snapshot[j] - l.weights.data.data[j]) > 1e-6 {
					return false
				}
			}
			idx += 1
		case nn.MaxPool2dLayer,
		     nn.AvgPool2dLayer,
		     nn.DropoutLayer,
		     nn.BatchNorm2dLayer,
		     nn.Activation,
		     nn.FlattenLayer,
		     nn.RNNLayer,
		     nn.GRULayer,
		     nn.LSTMLayer,
		     nn.EmbeddingLayer,
		     nn.MultiHeadAttentionLayer,
		     nn.LayerNormLayer,
		     nn.FFNLayer,
		     nn.TransformerEncoderBlock,
		     nn.TransformerEncoder:
		// Skip
		}
	}

	return true
}

// ============================================================================
// Main test
// ============================================================================
transfer_learning_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Transfer Learning Test ===\n")

	// ========================================================================
	// Phase 1: Pretrain on Task A (frequency classification)
	// ========================================================================
	fmt.println("Phase 1: Pretraining on Task A (frequency classification)")
	fmt.println("-----------------------------------------------------------")

	train_inputs_a, train_labels_a := generate_task_a_data(1000, allocator)
	defer {
		t.tensor_free(train_inputs_a)
		delete(train_labels_a, allocator)
	}

	test_inputs_a, test_labels_a := generate_task_a_data(200, allocator)
	defer {
		t.tensor_free(test_inputs_a)
		delete(test_labels_a, allocator)
	}

	model_a := create_model(allocator, 2)
	defer nn.sequential_free(model_a)

	fmt.println("Training model on Task A...")
	loss_a := train_model(model_a, train_inputs_a, train_labels_a, 30, 0.001, allocator)
	fmt.printf("Final loss: %.4f\n\n", loss_a)

	train_acc_a := evaluate_model(model_a, train_inputs_a, train_labels_a)
	test_acc_a := evaluate_model(model_a, test_inputs_a, test_labels_a)
	fmt.printf("Task A - Train accuracy: %.2f%%\n", train_acc_a * 100)
	fmt.printf("Task A - Test accuracy: %.2f%%\n\n", test_acc_a * 100)

	// ========================================================================
	// Phase 2: Save checkpoint
	// ========================================================================
	fmt.println("Phase 2: Saving checkpoint")
	fmt.println("--------------------------")

	checkpoint_path := "task_a_checkpoint.bin"
	dummy_opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&dummy_opt)
	save_ok := nn.save_checkpoint(model_a, &dummy_opt, checkpoint_path, 30, allocator)
	if !save_ok {
		fmt.println("ERROR: Failed to save checkpoint")
		return
	}
	fmt.printf("Checkpoint saved to: %s\n\n", checkpoint_path)

	// ========================================================================
	// Phase 3: Load checkpoint into new model for Task B
	// ========================================================================
	fmt.println("Phase 3: Loading checkpoint for Task B")
	fmt.println("---------------------------------------")

	model_b := create_model(allocator, 2)
	defer nn.sequential_free(model_b)

	loaded, skipped, load_ok := nn.sequential_load_partial(model_b, checkpoint_path, allocator)
	if !load_ok {
		fmt.println("ERROR: Failed to load checkpoint")
		return
	}
	fmt.printf("Loaded %d layers, skipped %d layers\n\n", loaded, skipped)

	// ========================================================================
	// Phase 4: Freeze feature extractor, keep classification head trainable
	// ========================================================================
	fmt.println("Phase 4: Freezing feature extractor")
	fmt.println("------------------------------------")

	// Freeze first 5 layers (feature extractor)
	nn.sequential_freeze_range(model_b, 0, 5)

	// Take snapshot of frozen weights
	frozen_snapshot := save_weights_snapshot(model_b, allocator)
	defer {
		for snapshot in frozen_snapshot {
			delete(snapshot, allocator)
		}
		delete(frozen_snapshot)
	}

	fmt.println("Frozen layers 0-4 (feature extractor)")
	fmt.println("Trainable layers 5-6 (classification head)\n")

	// ========================================================================
	// Phase 5: Train on Task B (amplitude classification)
	// ========================================================================
	fmt.println("Phase 5: Fine-tuning on Task B (amplitude classification)")
	fmt.println("----------------------------------------------------------")

	train_inputs_b, train_labels_b := generate_task_b_data(1000, allocator)
	defer {
		t.tensor_free(train_inputs_b)
		delete(train_labels_b, allocator)
	}

	test_inputs_b, test_labels_b := generate_task_b_data(200, allocator)
	defer {
		t.tensor_free(test_inputs_b)
		delete(test_labels_b, allocator)
	}

	// Use per-parameter learning rates
	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)

	// Add frozen params with tiny LR
	// for i in 0 ..< 5 {
	// 	layer := &model_b.layers[i]
	// 	switch &l in layer {
	// 	case nn.Conv2dLayer:
	// 		nn.adam_add_param_with_lr(&opt, l.weight, 1e-6)
	// 		if l.bias != nil {
	// 			nn.adam_add_param_with_lr(&opt, l.bias, 1e-6)
	// 		}
	// 	case nn.LinearLayer:
	// 		nn.adam_add_param_with_lr(&opt, l.weights, 1e-6)
	// 		if l.bias != nil {
	// 			nn.adam_add_param_with_lr(&opt, l.bias, 1e-6)
	// 		}
	// 	case nn.MaxPool2dLayer,
	// 	     nn.AvgPool2dLayer,
	// 	     nn.DropoutLayer,
	// 	     nn.BatchNorm2dLayer,
	// 	     nn.Activation,
	// 	     nn.FlattenLayer,
	// 	     nn.RNNLayer,
	// 	     nn.GRULayer,
	// 	     nn.LSTMLayer,
	// 	     nn.EmbeddingLayer,
	// 	     nn.MultiHeadAttentionLayer,
	// 	     nn.LayerNormLayer,
	// 	     nn.FFNLayer,
	// 	     nn.TransformerEncoderBlock,
	// 	     nn.TransformerEncoder:
	// 	// Skip
	// 	}
	// }

	// Add trainable head with normal LR
	for i in 5 ..< len(model_b.layers) {
		layer := &model_b.layers[i]
		switch &l in layer {
		case nn.LinearLayer:
			nn.adam_add_param(&opt, l.weights)
			if l.bias != nil {
				nn.adam_add_param(&opt, l.bias)
			}
		case nn.Conv2dLayer,
		     nn.MaxPool2dLayer,
		     nn.AvgPool2dLayer,
		     nn.DropoutLayer,
		     nn.BatchNorm2dLayer,
		     nn.Activation,
		     nn.FlattenLayer,
		     nn.RNNLayer,
		     nn.GRULayer,
		     nn.LSTMLayer,
		     nn.EmbeddingLayer,
		     nn.MultiHeadAttentionLayer,
		     nn.LayerNormLayer,
		     nn.FFNLayer,
		     nn.TransformerEncoderBlock,
		     nn.TransformerEncoder:
		// Skip
		}
	}

	fmt.println("Training model on Task B with frozen feature extractor...")
	batch_size := 32
	num_samples := len(train_labels_b)

	for epoch in 0 ..< 30 {
		nn.adam_zero_grad(&opt)

		total_loss := 0.0
		num_batches := 0

		for batch_start := 0; batch_start < num_samples; batch_start += batch_size {
			batch_end := min(batch_start + batch_size, num_samples)
			batch_size_actual := batch_end - batch_start

			batch_data := l.matrix_new(f64, batch_size_actual, 32, allocator)
			for i in 0 ..< batch_size_actual {
				src_offset := (batch_start + i) * 32
				for j in 0 ..< 32 {
					batch_data.data[i * 32 + j] = train_inputs_b.data.data[src_offset + j]
				}
			}

			batch_inputs := t.tensor_new(batch_data, false, allocator)
			batch_inputs.shape = [4]int{batch_size_actual, 1, 1, 32}

			outputs := nn.sequential_forward(model_b, batch_inputs)
			batch_labels := train_labels_b[batch_start:batch_end]
			loss := t.tensor_cross_entropy_loss(outputs, batch_labels)

			t.tensor_backward(loss)
			nn.adam_step(&opt)

			total_loss += loss.data.data[0]
			num_batches += 1

			t.tensor_free_graph(loss)
			t.tensor_free(batch_inputs)
		}

		avg_loss := total_loss / f64(num_batches)

		if epoch % 10 == 0 || epoch == 29 {
			fmt.printf("  Epoch %d/%d - Loss: %.4f\n", epoch + 1, 30, avg_loss)
		}
	}

	fmt.println()

	// ========================================================================
	// Phase 6: Verify frozen weights didn't change
	// ========================================================================
	fmt.println("Phase 6: Verifying frozen weights")
	fmt.println("----------------------------------")

	weights_unchanged := verify_weights_unchanged(model_b, frozen_snapshot)
	if weights_unchanged {
		fmt.println("✓ Frozen weights remained unchanged")
	} else {
		fmt.println("✗ ERROR: Frozen weights changed!")
	}
	fmt.println()

	// ========================================================================
	// Phase 7: Evaluate on Task B
	// ========================================================================
	fmt.println("Phase 7: Evaluating on Task B")
	fmt.println("------------------------------")

	train_acc_b := evaluate_model(model_b, train_inputs_b, train_labels_b)
	test_acc_b := evaluate_model(model_b, test_inputs_b, test_labels_b)
	fmt.printf("Task B - Train accuracy: %.2f%%\n", train_acc_b * 100)
	fmt.printf("Task B - Test accuracy: %.2f%%\n\n", test_acc_b * 100)

	// ========================================================================
	// Phase 8: Verify transfer learning benefit
	// ========================================================================
	fmt.println("Phase 8: Comparing with training from scratch")
	fmt.println("----------------------------------------------")

	model_scratch := create_model(allocator, 2)
	defer nn.sequential_free(model_scratch)

	fmt.println("Training new model on Task B from scratch...")
	loss_scratch := train_model(
		model_scratch,
		train_inputs_b,
		train_labels_b,
		30,
		0.001,
		allocator,
	)
	fmt.printf("Final loss: %.4f\n\n", loss_scratch)

	test_acc_scratch := evaluate_model(model_scratch, test_inputs_b, test_labels_b)
	fmt.printf("From scratch - Test accuracy: %.2f%%\n", test_acc_scratch * 100)
	fmt.printf("Transfer learning - Test accuracy: %.2f%%\n", test_acc_b * 100)

	if test_acc_b > test_acc_scratch {
		fmt.printf(
			"\n✓ Transfer learning improved accuracy by %.2f%%\n",
			(test_acc_b - test_acc_scratch) * 100,
		)
	} else {
		fmt.println("\n✗ Transfer learning did not improve accuracy")
	}

	fmt.println("\n=== Transfer Learning Test Complete ===\n")
}
