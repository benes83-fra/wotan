package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import mem_virtual "core:mem/virtual"

// Special token IDs
CLS_TOKEN :: 0
SEP_TOKEN :: 1
MASK_TOKEN :: 2

bert_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== BERT Character-Level Test (Arena Allocation) ===")

	// Training corpus - two related sentence groups
	text_a := "the cat sat on the mat the dog ran in the park the bird flew over the tree"
	text_b := "fish swim in the deep blue ocean waves crash upon the sandy shore"

	// Build vocabulary from both texts plus special tokens
	vocab := make(map[u8]int, allocator)
	inv_vocab := make(map[int]u8, allocator)
	vocab_size := 3 // Start with special tokens

	// Reserve IDs for special tokens
	inv_vocab[CLS_TOKEN] = u8('[')
	inv_vocab[SEP_TOKEN] = u8(']')
	inv_vocab[MASK_TOKEN] = u8('_')

	// Add characters from text
	for c in text_a {
		c_u8 := u8(c)
		if !(c_u8 in vocab) {
			vocab[c_u8] = vocab_size
			inv_vocab[vocab_size] = c_u8
			vocab_size += 1
		}
	}
	for c in text_b {
		c_u8 := u8(c)
		if !(c_u8 in vocab) {
			vocab[c_u8] = vocab_size
			inv_vocab[vocab_size] = c_u8
			vocab_size += 1
		}
	}

	fmt.printf("Vocabulary size: %d tokens\n", vocab_size)
	fmt.printf("Text A length: %d characters\n", len(text_a))
	fmt.printf("Text B length: %d characters\n", len(text_b))

	// Model hyperparameters
	seq_len := 32
	d_model := 64
	num_heads := 4
	d_ff := 256
	num_layers := 2
	batch_size := 4
	max_seq_len := seq_len

	// Create BERT model
	model := nn.bert_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator,
	)
	defer nn.bert_model_free(&model)

	// Create optimizer
	opt := nn.adam_new(0.0003, allocator = allocator)
	defer nn.adam_free(&opt)

	// Register all parameters
	nn.bert_model_add_to_optimizer(&model, &opt)

	// ✅ Create arena allocator for backward pass
	arena: mem_virtual.Arena
	err := mem_virtual.arena_init_growing(&arena)
	if err != nil {
		fmt.printf("Failed to initialize arena: %v\n", err)
		return
	}
	defer mem_virtual.arena_destroy(&arena)

	arena_alloc := mem_virtual.arena_allocator(&arena)

	// Convert texts to token indices
	text_a_indices := make([]int, len(text_a), allocator)
	defer delete(text_a_indices, allocator)
	for i in 0 ..< len(text_a) {
		text_a_indices[i] = vocab[u8(text_a[i])]
	}

	text_b_indices := make([]int, len(text_b), allocator)
	defer delete(text_b_indices, allocator)
	for i in 0 ..< len(text_b) {
		text_b_indices[i] = vocab[u8(text_b[i])]
	}

	// Training loop
	epochs := 2500 // Increased to test arena stability
	fmt.printf(
		"Training BERT (layers=%d, d_model=%d, heads=%d)...\n",
		num_layers,
		d_model,
		num_heads,
	)

	initial_loss := 0.0

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Create training batch
		input_ids_data := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
		segment_ids_data := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
		mlm_labels := make([]int, batch_size * seq_len, allocator)
		nsp_labels := make([]int, batch_size, allocator)

		// Initialize
		for i in 0 ..< batch_size * seq_len {
			input_ids_data.data[i] = f64(SEP_TOKEN) // Pad with SEP
			segment_ids_data.data[i] = 0
			mlm_labels[i] = -1 // -1 means ignore
		}

		for b in 0 ..< batch_size {
			// Decide if sentence B follows sentence A (50% chance)
			is_next := rand.float64() > 0.5
			nsp_labels[b] = is_next ? 1 : 0

			pos := 0

			// [CLS] token
			input_ids_data.data[b * seq_len + pos] = f64(CLS_TOKEN)
			segment_ids_data.data[b * seq_len + pos] = 0
			pos += 1

			// Sentence A (random substring from text_a)
			start_a := int(rand.int31()) % max(len(text_a) - 10, 1)
			len_a := min(10, seq_len / 2 - 2)
			for s in 0 ..< len_a {
				if pos >= seq_len - 1 {break}
				token_id := text_a_indices[start_a + s]

				// 15% chance to mask
				if rand.float64() < 0.15 {
					mlm_labels[b * seq_len + pos] = token_id
					input_ids_data.data[b * seq_len + pos] = f64(MASK_TOKEN)
				} else {
					input_ids_data.data[b * seq_len + pos] = f64(token_id)
				}
				segment_ids_data.data[b * seq_len + pos] = 0
				pos += 1
			}

			// [SEP]
			if pos < seq_len {
				input_ids_data.data[b * seq_len + pos] = f64(SEP_TOKEN)
				segment_ids_data.data[b * seq_len + pos] = 0
				pos += 1
			}

			// Sentence B
			start_b := 0
			len_b := 0
			if is_next {
				start_b = int(rand.int31()) % max(len(text_b) - 8, 1)
				len_b = min(8, seq_len - pos - 1)
			} else {
				// Random sentence from text_a instead
				start_b = int(rand.int31()) % max(len(text_a) - 8, 1)
				len_b = min(8, seq_len - pos - 1)
			}

			src_indices := text_b_indices if is_next else text_a_indices
			for s in 0 ..< len_b {
				if pos >= seq_len - 1 {break}
				token_id := src_indices[start_b + s]

				if rand.float64() < 0.15 {
					mlm_labels[b * seq_len + pos] = token_id
					input_ids_data.data[b * seq_len + pos] = f64(MASK_TOKEN)
				} else {
					input_ids_data.data[b * seq_len + pos] = f64(token_id)
				}
				segment_ids_data.data[b * seq_len + pos] = 1
				pos += 1
			}

			// Final [SEP]
			if pos < seq_len {
				input_ids_data.data[b * seq_len + pos] = f64(SEP_TOKEN)
				segment_ids_data.data[b * seq_len + pos] = 1
				pos += 1
			}
		}

		// Create tensors
		input_ids := t.tensor_new(input_ids_data, false, allocator)
		input_ids.shape = [4]int{batch_size, seq_len, 1, 1}

		segment_ids := t.tensor_new(segment_ids_data, false, allocator)
		segment_ids.shape = [4]int{batch_size, seq_len, 1, 1}

		// Forward pass
		mlm_logits, nsp_logits := nn.bert_model_forward(&model, input_ids, segment_ids, true)

		// Compute MLM loss
		num_masked := 0
		for i in 0 ..< batch_size * seq_len {
			if mlm_labels[i] != -1 {
				num_masked += 1
			}
		}

		mlm_loss: ^t.Tensor
		if num_masked > 0 {
			masked_logits_data := l.matrix_new(f64, num_masked, vocab_size, allocator)
			masked_labels := make([]int, num_masked, allocator)

			masked_idx := 0
			for i in 0 ..< batch_size * seq_len {
				if mlm_labels[i] != -1 {
					for v in 0 ..< vocab_size {
						masked_logits_data.data[masked_idx * vocab_size + v] =
							mlm_logits.data.data[i * vocab_size + v]
					}
					masked_labels[masked_idx] = mlm_labels[i]
					masked_idx += 1
				}
			}

			masked_logits := t.tensor_new(masked_logits_data, true, allocator)
			mlm_loss = t.tensor_cross_entropy_loss(masked_logits, masked_labels)
			delete(masked_labels, allocator)
		} else {
			zero_data := l.matrix_new(f64, 1, 1, allocator)
			zero_data.data[0] = 0.0
			mlm_loss = t.tensor_new(zero_data, true, allocator)
		}

		// Compute NSP loss
		nsp_logits_2d := l.matrix_new(f64, batch_size, 2, allocator)
		for b in 0 ..< batch_size {
			nsp_logits_2d.data[b * 2 + 0] = nsp_logits.data.data[b * 2 + 0]
			nsp_logits_2d.data[b * 2 + 1] = nsp_logits.data.data[b * 2 + 1]
		}
		nsp_logits_tensor := t.tensor_new(nsp_logits_2d, true, allocator)
		nsp_loss := t.tensor_cross_entropy_loss(nsp_logits_tensor, nsp_labels)

		// Combined loss
		total_loss := t.tensor_add(mlm_loss, nsp_loss)

		if epoch == 0 {
			initial_loss = total_loss.data.data[0]
		}

		if epoch % 100 == 0 {
			fmt.printf(
				"Epoch %d | MLM: %.4f | NSP: %.4f | Total: %.4f\n",
				epoch,
				mlm_loss.data.data[0],
				nsp_loss.data.data[0],
				total_loss.data.data[0],
			)
		}

		// ✅ Backward pass with arena allocator
		t.tensor_backward(total_loss, arena_alloc)

		// ✅ Reset arena (frees all temporaries from this backward pass)
		mem_virtual.arena_free_all(&arena)

		// Optimizer step
		nn.adam_step(&opt)

		// Cleanup
		t.tensor_free_graph(total_loss)
		t.tensor_free(input_ids)
		t.tensor_free(segment_ids)
		delete(mlm_labels, allocator)
		delete(nsp_labels, allocator)
	}

	final_loss := 0.0
	// Get final loss from last epoch
	// (we'd need to track this properly, but for now just report initial)
	reduction := 0.0
	fmt.printf("\n✓ BERT training complete! Initial Loss: %.4f\n", initial_loss)

	// Test MLM: Fill in the blank
	fmt.println("\n=== MLM Test: Fill in the Blank ===")

	test_input := "the cat _ on the mat"
	fmt.printf("Input: %s\n", test_input)

	// Create input tensor
	test_ids_data := l.matrix_new(f64, 1, seq_len, allocator)
	test_seg_data := l.matrix_new(f64, 1, seq_len, allocator)
	for i in 0 ..< seq_len {
		test_ids_data.data[i] = f64(SEP_TOKEN)
		test_seg_data.data[i] = 0
	}

	test_ids_data.data[0] = f64(CLS_TOKEN)
	pos := 1
	for c in test_input {
		if pos >= seq_len - 1 {break}
		if c == '_' {
			test_ids_data.data[pos] = f64(MASK_TOKEN)
		} else {
			c_u8 := u8(c)
			if c_u8 in vocab {
				test_ids_data.data[pos] = f64(vocab[c_u8])
			}
		}
		pos += 1
	}
	test_ids_data.data[pos] = f64(SEP_TOKEN)

	test_ids := t.tensor_new(test_ids_data, false, allocator)
	test_ids.shape = [4]int{1, seq_len, 1, 1}

	test_seg := t.tensor_new(test_seg_data, false, allocator)
	test_seg.shape = [4]int{1, seq_len, 1, 1}

	mlm_out, _ := nn.bert_model_forward(&model, test_ids, test_seg, false)

	// Find the MASK position and get predictions
	for i in 1 ..< pos {
		if int(test_ids_data.data[i]) == MASK_TOKEN {
			// Get logits for this position
			best_idx := 0
			best_val := mlm_out.data.data[i * vocab_size]
			for v in 1 ..< vocab_size {
				if mlm_out.data.data[i * vocab_size + v] > best_val {
					best_val = mlm_out.data.data[i * vocab_size + v]
					best_idx = v
				}
			}
			predicted_char := inv_vocab[best_idx]
			fmt.printf("Predicted token at position %d: '%c'\n", i, predicted_char)
		}
	}

	// Cleanup test tensors
	t.tensor_free(test_ids)
	t.tensor_free(test_seg)

	fmt.println("\n✓ BERT test completed!")
}
