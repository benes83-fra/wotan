package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

gpt_full_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Full GPT Implementation Test ===")

	// Larger training corpus (multiple Shakespeare excerpts)
	text := "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles and by opposing end them to die to sleep no more and by a sleep to say we end the heartache and the thousand natural shocks that flesh is heir to tis a consummation devoutly to be wished to die to sleep to sleep perchance to dream ay theres the rub for in that sleep of death what dreams may come when we have shuffled off this mortal coil must give us pause theres the respect that makes calamity of so long life for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely the pangs of despised love the laws delay the insolence of office and the spurns that patient merit of the unworthy takes when he himself might his quietus make with a bare bodkin who would fardels bear to grunt and sweat under a weary life but that the dread of something after death the undiscovered country from whose bourn no traveller returns puzzles the will and makes us rather bear those ills we have than fly to others that we know not of thus conscience does make cowards of us all and thus the native hue of resolution is sicklied oer with the pale cast of thought and enterprises of great pith and moment with this regard their currents turn awry and lose the name of action"

	// Build vocabulary
	vocab := make(map[u8]int, allocator)
	inv_vocab := make(map[int]u8, allocator)
	vocab_size := 0

	for c in text {
		c_u8 := u8(c)
		if !(c_u8 in vocab) {
			vocab[c_u8] = vocab_size
			inv_vocab[vocab_size] = c_u8
			vocab_size += 1
		}
	}

	fmt.printf("Vocabulary size: %d characters\n", vocab_size)
	fmt.printf("Text length: %d characters\n", len(text))

	// Hyperparameters
	seq_len := 64
	d_model := 128
	num_heads := 4
	d_ff := 512
	num_layers := 3
	batch_size := 16
	max_seq_len := seq_len + 1

	// Create GPT model
	model := nn.gpt_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator,
	)
	defer nn.gpt_model_free(&model)

	// Create optimizer
	opt := nn.adam_new(0.0003, allocator = allocator)
	defer nn.adam_free(&opt)

	// Register all parameters
	nn.gpt_model_add_to_optimizer(&model, &opt)

	// Create causal mask
	mask := nn.create_causal_mask(max_seq_len, allocator)
	defer delete(mask, allocator)

	// Convert text to indices
	text_indices := make([]int, len(text), allocator)
	defer delete(text_indices, allocator)

	for i in 0 ..< len(text) {
		text_indices[i] = vocab[u8(text[i])]
	}

	// Training loop
	epochs := 200
	fmt.printf(
		"Training GPT (layers=%d, d_model=%d, heads=%d)...\n",
		num_layers,
		d_model,
		num_heads,
	)

	initial_loss := 0.0
	final_loss := 0.0
	best_loss := math.F64_MAX

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Sample random sequences from text
		input_ids_data := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
		target_indices := make([]int, batch_size * seq_len, allocator)

		for b in 0 ..< batch_size {
			start := int(rand.int31()) % (len(text) - seq_len - 1)

			for s in 0 ..< seq_len {
				input_ids_data.data[b * seq_len + s] = f64(text_indices[start + s])
				target_indices[b * seq_len + s] = text_indices[start + s + 1]
			}
		}

		input_ids := t.tensor_new(input_ids_data, false, allocator)
		input_ids.shape = [4]int{batch_size, seq_len, 1, 1}

		// Forward pass
		logits := nn.gpt_model_forward(&model, input_ids, mask)

		// Compute cross-entropy loss
		loss := t.tensor_cross_entropy_loss(logits, target_indices)

		if epoch == 0 {
			initial_loss = loss.data.data[0]
		}

		// Backward pass
		t.tensor_backward(loss)

		// Optimizer step
		nn.adam_step(&opt)

		// Track best loss
		if loss.data.data[0] < best_loss {
			best_loss = loss.data.data[0]
		}

		if epoch % 40 == 0 {
			perplexity := math.exp(loss.data.data[0])
			fmt.printf(
				"Epoch %d | Loss: %.4f | Perplexity: %.2f\n",
				epoch,
				loss.data.data[0],
				perplexity,
			)
		}

		if epoch == epochs - 1 {
			final_loss = loss.data.data[0]
		}

		// Clean up
		t.tensor_free_graph(loss)
		l.matrix_free(&input_ids_data)
		delete(target_indices, allocator)
		t.tensor_free(input_ids)
		t.tensor_free(logits)
	}

	reduction := (1.0 - final_loss / initial_loss) * 100.0
	final_perplexity := math.exp(final_loss)
	fmt.printf(
		"\n✓ Training complete! Loss: %.4f → %.4f (%.1f%% reduction)\n",
		initial_loss,
		final_loss,
		reduction,
	)
	fmt.printf("Final perplexity: %.2f\n", final_perplexity)

	// Generate text with different sampling strategies
	// Generate text with different sampling strategies
	fmt.println("\n=== Generating Text ===")

	// Define sampling config struct
	SamplingConfig :: struct {
		name:   string,
		method: string,
		param:  f64,
	}

	// Test different sampling methods
	sampling_configs := []SamplingConfig {
		{"Temperature (0.8)", "temperature", 0.8},
		{"Temperature (1.2)", "temperature", 1.2},
		{"Top-k (k=10)", "top_k", 10.0},
		{"Top-p (p=0.9)", "top_p", 0.9},
	}

	for config in sampling_configs {
		fmt.printf("\n%s:\n", config.name)

		// Start with "to be"
		seed := "to be"
		generated := make([dynamic]u8, 0, allocator)

		for c in seed {
			append(&generated, u8(c))
		}

		// Generate 100 characters
		for i in 0 ..< 100 {
			// Prepare input sequence
			gen_input := l.matrix_new(f64, 1, seq_len, allocator)

			gen_len := len(generated)
			if gen_len > seq_len {
				gen_len = seq_len
			}

			for s in 0 ..< gen_len {
				offset := len(generated) - gen_len + s
				gen_input.data[s] = f64(vocab[generated[offset]])
			}

			gen_ids := t.tensor_new(gen_input, false, allocator)
			gen_ids.shape = [4]int{1, seq_len, 1, 1}

			// Forward pass
			gen_logits := nn.gpt_model_forward(&model, gen_ids, mask)

			// Get logits for last position
			last_logits := make([]f64, vocab_size, allocator)
			for v in 0 ..< vocab_size {
				last_logits[v] = gen_logits.data.data[(seq_len - 1) * vocab_size + v]
			}

			// Sample based on method
			next_char_idx := 0
			if config.method == "temperature" {
				next_char_idx = nn.gpt_sample_temperature(last_logits, config.param)
			} else if config.method == "top_k" {
				next_char_idx = nn.gpt_sample_top_k(last_logits, int(config.param), 1.0)
			} else if config.method == "top_p" {
				next_char_idx = nn.gpt_sample_top_p(last_logits, config.param, 1.0)
			}

			append(&generated, inv_vocab[next_char_idx])

			// Clean up
			l.matrix_free(&gen_input)
			delete(last_logits, allocator)
			t.tensor_free(gen_ids)
			t.tensor_free(gen_logits)
		}

		fmt.printf("%.*s\n", len(generated), generated[:])
		delete(generated)
	}

	fmt.println("\n✓ Full GPT test completed!")
}
