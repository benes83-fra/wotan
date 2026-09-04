package tests

import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

tokenizer_wordpiece_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== WordPiece Tokenizer Isolated Test ===")

	vocab_content := "[PAD]\n[UNK]\n[CLS]\n[SEP]\napple\n##s\nplay\n##ing\nun\n##want\n##ed\n"
	temp_vocab_path := "temp_test_vocab.txt"

	file, err := os.create(temp_vocab_path)
	if err != nil {
		fmt.printf("Failed to create temp vocab file: %v\n", err)
		return
	}

	// Odin transmutation for zero-allocation string to []u8
	os.write(file, transmute([]u8)vocab_content)
	os.close(file)
	defer os.remove(temp_vocab_path)

	fmt.println("✓ Temporary vocab file created.")

	max_len := 16
	tok_inst, ok := tok.wordpiece_tokenizer_new(temp_vocab_path, max_len, allocator)
	if !ok {
		fmt.println("Failed to load tokenizer.")
		return
	}
	defer tok.wordpiece_tokenizer_free(&tok_inst)
	fmt.println("✓ Tokenizer loaded successfully.")

	test_text := "apples playing unwanted"
	fmt.printf("\nTokenizing: '%s'\n", test_text)

	input_ids, attention_mask := tok.tokenize(&tok_inst, test_text, allocator)
	defer {
		delete(input_ids, allocator)
		delete(attention_mask, allocator)
	}

	fmt.printf("Input IDs:    %v\n", input_ids)
	fmt.printf("Attention:    %v\n", attention_mask)

	input_tensor, segment_tensor := tok.tokenize_to_tensors(&tok_inst, test_text, allocator)
	defer {
		t.tensor_free(input_tensor)
		t.tensor_free(segment_tensor)
	}

	fmt.printf("\nInput Tensor Shape:  %v\n", input_tensor.shape)
	fmt.printf("Segment Tensor Shape:%v\n", segment_tensor.shape)

	expected_ids := []int{2, 4, 5, 6, 7, 8, 9, 10, 3, 0, 0, 0, 0, 0, 0, 0}
	all_match := true
	for i in 0 ..< len(expected_ids) {
		if input_ids[i] != expected_ids[i] {
			all_match = false
			fmt.printf(
				"  ❌ Mismatch at index %d: got %d, expected %d\n",
				i,
				input_ids[i],
				expected_ids[i],
			)
		}
	}

	if all_match {
		fmt.println("\n✓ Tokenization logic perfectly matches expected WordPiece behavior!")
	} else {
		fmt.println("\n✗ Tokenization output did not match expectations.")
	}

	fmt.println("\n✓ WordPiece Tokenizer Test Complete!")
}
