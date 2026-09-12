package tests

import l "../wotan/linalg"
import net "../wotan/net"
import nn "../wotan/nn"
import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

gpt_completion_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GPT Financial Text Completion Test ===")

	// 1. Load the converted DistilGPT-2 model
	fmt.println("\nLoading DistilGPT-2 Model...")
	checkpoint_path := "distilgpt2_checkpoint.bin"

	gpt_model: ^nn.GPTModel
	ok_load := false
	gpt_model, ok_load = nn.load_gpt_model(checkpoint_path, allocator)

	if !ok_load {
		fmt.println("⚠ Checkpoint not found. Please ensure it was generated.")
		return
	}
	fmt.println("✓ Pre-trained DistilGPT-2 model loaded successfully!")

	// 2. Download or Load GPT-2 Tokenizer Files
	vocab_path := "vocab.json"
	merges_path := "merges.txt"

	_, err_vocab := os.read_entire_file(vocab_path, allocator)
	if err_vocab != nil {
		fmt.println("vocab.json not found locally. Downloading from HuggingFace (gpt2)...")
		url := "https://huggingface.co/gpt2/resolve/main/vocab.json"
		vocab_data_str, ok := net.http_get(url, allocator)
		if !ok {
			fmt.println("Failed to download vocab.json. Aborting.")
			return
		}
		file, create_err := os.create(vocab_path)
		if create_err != nil {
			fmt.printf("Failed to create vocab file: %v\n", create_err)
			delete(vocab_data_str, allocator)
			return
		}
		os.write(file, transmute([]u8)vocab_data_str)
		os.close(file)
		delete(vocab_data_str, allocator)
		fmt.println("✓ vocab.json downloaded and cached successfully.")
	} else {
		fmt.println("✓ Found local vocab.json")
	}

	_, err_merges := os.read_entire_file(merges_path, allocator)
	if err_merges != nil {
		fmt.println("merges.txt not found locally. Downloading from HuggingFace (gpt2)...")
		url := "https://huggingface.co/gpt2/resolve/main/merges.txt"
		merges_data_str, ok := net.http_get(url, allocator)
		if !ok {
			fmt.println("Failed to download merges.txt. Aborting.")
			return
		}
		file, create_err := os.create(merges_path)
		if create_err != nil {
			fmt.printf("Failed to create merges file: %v\n", create_err)
			delete(merges_data_str, allocator)
			return
		}
		os.write(file, transmute([]u8)merges_data_str)
		os.close(file)
		delete(merges_data_str, allocator)
		fmt.println("✓ merges.txt downloaded and cached successfully.")
	} else {
		fmt.println("✓ Found local merges.txt")
	}

	// 3. Initialize BPE Tokenizer
	fmt.println("\nLoading BPE Tokenizer...")
	tokenizer, ok_tok := tok.bpe_tokenizer_new(vocab_path, merges_path, 512, allocator)
	if !ok_tok {
		fmt.println("⚠ Failed to load BPE tokenizer. Aborting.")
		return
	}
	defer tok.bpe_tokenizer_free(&tokenizer)
	fmt.println("✓ BPE Tokenizer loaded successfully!")

	// 4. Tokenize the prompt (GPT-2 style, stripped of padding)
	prompt_text := "The Federal Reserve announced a"
	fmt.printf("\nPrompt: \"%s\"\n", prompt_text)

	input_ids_slice := gpt_encode(&tokenizer, prompt_text, allocator)
	defer delete(input_ids_slice, allocator)

	seq_len := len(input_ids_slice)
	batch := 1

	ids_data := l.matrix_new(f64, 1, batch * seq_len, allocator)
	for i in 0 ..< len(input_ids_slice) {
		ids_data.data[i] = f64(input_ids_slice[i])
	}

	input_ids := t.tensor_new(ids_data, false, allocator)
	input_ids.shape = [4]int{batch, seq_len, 1, 1}
	defer t.tensor_free(input_ids)

	// 5. Create Causal Mask
	causal_mask := nn.create_causal_mask(seq_len, allocator)
	defer delete(causal_mask, allocator)

	// 6. Run Forward Pass
	fmt.println("\nRunning Forward Pass...")
	logits := nn.gpt_model_forward(gpt_model, input_ids, causal_mask, false)
	defer t.tensor_free(logits)

	// 7. Extract Top-5 Predictions for the Next Token
	last_token_idx := seq_len - 1
	vocab_size := gpt_model.vocab_size
	offset := last_token_idx * vocab_size

	fmt.println("\n--- Top 5 Predictions for Next Token ---")

	top_k := 5
	top_ids := make([]int, top_k, allocator)
	top_probs := make([]f64, top_k, allocator)
	defer {
		delete(top_ids, allocator)
		delete(top_probs, allocator)
	}

	for k in 0 ..< top_k {
		top_ids[k] = -1
		top_probs[k] = -math.F64_MAX
	}

	// Simple Top-K extraction
	for v in 0 ..< vocab_size {
		logit := logits.data.data[offset + v]
		for k in 0 ..< top_k {
			if logit > top_probs[k] {
				for j := top_k - 1; j > k; j -= 1 {
					top_ids[j] = top_ids[j - 1]
					top_probs[j] = top_probs[j - 1]
				}
				top_ids[k] = v
				top_probs[k] = logit
				break
			}
		}
	}

	for k in 0 ..< top_k {
		decoded_token := ""
		if top_ids[k] >= 0 && top_ids[k] < len(tokenizer.ids_to_tokens) {
			decoded_token = tokenizer.ids_to_tokens[top_ids[k]]
		}
		fmt.printf(
			"  Token ID: %-6d | Logit: %8.4f | Text: \"%s\"\n",
			top_ids[k],
			top_probs[k],
			decoded_token,
		)
	}
	fmt.println("\n✓ GPT Completion Test Complete!")
}

// GPT-2 does not use [CLS] or [SEP] tokens, and we want to strip padding.
// This helper returns ONLY the actual text tokens.
gpt_encode :: proc(toki: ^tok.BPETokenizer, text: string, allocator: mem.Allocator) -> []int {
	ids, _ := tok.bpe_tokenize(toki, text, allocator, false)

	start := 0
	for start < len(ids) {
		if ids[start] == 0 || ids[start] == 50256 {
			start += 1
		} else {
			break
		}
	}

	end := len(ids) - 1
	for end >= 0 {
		if ids[end] == 0 || ids[end] == 50256 {
			end -= 1
		} else {
			break
		}
	}

	if start > end {
		return make([]int, 0, allocator)
	}

	result := make([]int, end - start + 1, allocator)
	for i in 0 ..< len(result) {
		result[i] = ids[start + i]
	}
	return result
}
