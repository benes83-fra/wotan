package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ✅ Moved to package level to avoid closure capture issues
generate_batch_moe :: proc(
	batch_size: int,
	seq_len: int,
	pad_tok: int,
	query_tok: int,
	alloc: mem.Allocator,
) -> (
	x: ^t.Tensor,
	y: []int,
) {
	x_data := l.matrix_new(f64, 1, batch_size * seq_len, alloc)
	y_data := make([]int, batch_size * seq_len, alloc)

	for b in 0 ..< batch_size {
		// ✅ FIX: Use rand.int31() and cast to int
		key := int(rand.int31()) % 5 // 0 to 4
		value := key + 5 // 5 to 9

		// Full sequence (length seq_len + 1)
		seq := make([]int, seq_len + 1, alloc)
		for i in 0 ..< seq_len + 1 {seq[i] = pad_tok}
		seq[0] = key
		seq[seq_len - 1] = query_tok
		seq[seq_len] = value

		// x is seq[0 .. seq_len-1]
		// y is seq[1 .. seq_len]
		for i in 0 ..< seq_len {
			idx := b * seq_len + i
			x_data.data[idx] = f64(seq[i])
			y_data[idx] = seq[i + 1]
		}
		delete(seq, alloc)
	}

	x_tensor := t.tensor_new(x_data, false, alloc)
	x_tensor.shape = [4]int{batch_size, seq_len, 1, 1}
	return x_tensor, y_data
}

gpt_moe_flash_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GPT MoE + Flash Attention Test (Associative Recall) ===")

	// 1. Hyperparameters (Kept small for fast execution)
	vocab_size := 16
	d_model := 32
	num_heads := 4
	d_ff := 64
	num_layers := 2
	max_seq_len := 16
	batch_size := 64
	seq_len := 16 // Length of input/output sequences
	num_experts := 4
	top_k := 2

	PAD_TOKEN := 15
	QUERY_TOKEN := 14

	// 2. Initialize Model
	fmt.println("Initializing GPT-MoE Model with Flash Attention...")
	model := nn.gpt_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		use_flash = true, // ✅ Enable Flash Attention
		use_moe = true, // ✅ Enable Mixture of Experts
		num_experts = num_experts,
		top_k = top_k,
		allocator = allocator,
	)
	defer nn.gpt_model_free(&model)

	opt := nn.adam_new(0.005, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.gpt_model_add_to_optimizer(&model, &opt)

	// 3. Training Loop
	fmt.println("Training on Associative Recall Task...")
	fmt.println(
		"Task: Predict the associated Value (5-9) given a Key (0-4) at the start of the sequence.",
	)

	epochs := 150
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// ✅ Pass the special tokens explicitly
		x_batch, y_batch := generate_batch_moe(
			batch_size,
			seq_len,
			PAD_TOKEN,
			QUERY_TOKEN,
			allocator,
		)
		mask := nn.create_causal_mask(seq_len, allocator)

		// Forward pass (uses Flash Attention and MoE internally)
		logits := nn.gpt_model_forward(&model, x_batch, mask, true)
		ce_loss := t.tensor_cross_entropy_loss(logits, y_batch)

		aux_loss_data := l.matrix_new(f64, 1, 1, allocator)
		aux_loss_data.data[0] = model.total_aux_loss * 0.01 // 0.01 is standard weight
		aux_loss_tensor := t.tensor_new(aux_loss_data, false, allocator)

		total_loss := t.tensor_add(ce_loss, aux_loss_tensor)

		// Backward pass
		t.tensor_backward(total_loss)
		nn.clip_grad_norm(&opt, 1.0)
		nn.adam_step(&opt)

		if epoch % 20 == 0 || epoch == epochs - 1 {
			fmt.printf("Epoch %3d | CE Loss: %.4f\n", epoch, ce_loss.data.data[0])
		}

		t.tensor_free_graph(total_loss)
		t.tensor_free(aux_loss_tensor)
		t.tensor_free(x_batch)
		delete(y_batch, allocator)
		delete(mask, allocator)
	}

	// 4. Evaluation
	fmt.println("\n--- Evaluation (Inference) ---")
	fmt.println("Testing if the model learned the Key -> Value mapping.")

	correct := 0
	total := 5

	for key in 0 ..< 5 {
		value := key + 5

		seq := make([]int, seq_len + 1, allocator)
		for i in 0 ..< seq_len + 1 {seq[i] = PAD_TOKEN}
		seq[0] = key
		seq[seq_len - 1] = QUERY_TOKEN

		x_data := l.matrix_new(f64, 1, seq_len, allocator)
		for i in 0 ..< seq_len {
			x_data.data[i] = f64(seq[i])
		}
		x_tensor := t.tensor_new(x_data, false, allocator)
		x_tensor.shape = [4]int{1, seq_len, 1, 1}

		mask := nn.create_causal_mask(seq_len, allocator)
		logits := nn.gpt_model_forward(&model, x_tensor, mask, false)

		// Get prediction for the last token (which should be the Value)
		last_token_logits := logits.data.data[(seq_len - 1) * vocab_size:seq_len * vocab_size]

		pred := 0
		max_logit := -math.F64_MAX
		for v in 0 ..< vocab_size {
			if last_token_logits[v] > max_logit {
				max_logit = last_token_logits[v]
				pred = v
			}
		}

		status := "✗"
		if pred == value {
			status = "✓"
			correct += 1
		}

		fmt.printf(
			"  %s Key: %d | Expected Value: %d | Predicted: %d (Logit: %.2f)\n",
			status,
			key,
			value,
			pred,
			max_logit,
		)

		t.tensor_free_graph(logits)
		t.tensor_free(x_tensor)
		delete(seq, allocator)
		delete(mask, allocator)
	}

	fmt.printf("\nAccuracy: %d / %d\n", correct, total)
	if correct == total {
		fmt.println("✓ GPT-MoE with Flash Attention successfully learned the recall task!")
	} else {
		fmt.println("✗ Model struggled with the task. (May need more epochs or larger d_model).")
	}

	fmt.println("=== GPT MoE + Flash Attention Test Complete ===\n")
}
