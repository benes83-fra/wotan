package tests

import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
import "core:os"

tokenizer_bpe_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== BPE Tokenizer Isolated Test ===")

	vocab_json := `{"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "[PAD]": 100, "[UNK]": 101, "[CLS]": 102, "[SEP]": 103}`
	vocab_path := "temp_bpe_vocab.json"

	file, err := os.create(vocab_path)
	if err != nil {
		fmt.printf("Failed to create vocab file: %v\n", err)
		return
	}
	os.write(file, transmute([]u8)vocab_json)
	os.close(file)
	defer os.remove(vocab_path)

	merges_txt := `#version: 0.2
a b
ab c`
	merges_path := "temp_bpe_merges.txt"

	file2, err2 := os.create(merges_path)
	if err2 != nil {
		fmt.printf("Failed to create merges file: %v\n", err2)
		return
	}
	os.write(file2, transmute([]u8)merges_txt)
	os.close(file2)
	defer os.remove(merges_path)

	fmt.println("✓ Temporary vocab and merges files created.")

	max_len := 16
	tok_inst, ok := tok.bpe_tokenizer_new(vocab_path, merges_path, max_len, allocator)
	if !ok {
		fmt.println("Failed to load BPE tokenizer.")
		return
	}
	defer tok.bpe_tokenizer_free(&tok_inst)
	fmt.println("✓ BPE Tokenizer loaded successfully.")

	test_text := "abc"
	fmt.printf("\nTest 1: Tokenizing '%s'\n", test_text)

	input_ids, attention_mask := tok.bpe_tokenize(&tok_inst, test_text, allocator)
	defer {
		delete(input_ids, allocator)
		delete(attention_mask, allocator)
	}

	fmt.printf("Input IDs: %v\n", input_ids)

	expected_1 := []int {
		102,
		4,
		103,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
		100,
	}
	match_1 := true
	for i in 0 ..< len(expected_1) {
		if input_ids[i] != expected_1[i] {
			match_1 = false
			fmt.printf(
				"  ❌ Mismatch at index %d: got %d, expected %d\n",
				i,
				input_ids[i],
				expected_1[i],
			)
		}
	}

	if match_1 {
		fmt.println("✓ Test 1 passed: 'abc' correctly merged to single token")
	}

	test_text_2 := "ac"
	fmt.printf("\nTest 2: Tokenizing '%s'\n", test_text_2)

	input_ids_2, _ := tok.bpe_tokenize(&tok_inst, test_text_2, allocator)
	defer delete(input_ids_2, allocator)

	fmt.printf("Input IDs: %v\n", input_ids_2)

	expected_2 := []int{102, 0, 2, 103, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100}
	match_2 := true
	for i in 0 ..< len(expected_2) {
		if input_ids_2[i] != expected_2[i] {
			match_2 = false
			fmt.printf(
				"  ❌ Mismatch at index %d: got %d, expected %d\n",
				i,
				input_ids_2[i],
				expected_2[i],
			)
		}
	}

	if match_2 {
		fmt.println("✓ Test 2 passed: 'ac' correctly kept as separate tokens")
	}

	fmt.println("\nTest 3: Tensor conversion")
	input_tensor, segment_tensor := tok.bpe_tokenize_to_tensors(&tok_inst, "abc", allocator)
	defer {
		t.tensor_free(input_tensor)
		t.tensor_free(segment_tensor)
	}

	fmt.printf("Input Tensor Shape: %v\n", input_tensor.shape)
	fmt.printf("Segment Tensor Shape: %v\n", segment_tensor.shape)

	if input_tensor.shape[0] == 1 && input_tensor.shape[1] == 16 {
		fmt.println("✓ Test 3 passed: Tensor shapes correct")
	}

	fmt.println("\n✓ BPE Tokenizer Test Complete!")
}
