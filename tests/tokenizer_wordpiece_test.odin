package tests

import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"

// WordPiece on a toy vocabulary: longest-match-first pieces with ##
// continuations, punctuation split off, a word that cannot be matched
// completely becomes one [UNK], the [CLS] … [SEP] input with padding, the
// tensors. tokenizer_reference_test compares with Hugging Face on the
// bert-base-uncased vocabulary.
tokenizer_wordpiece_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== WordPiece Tokenizer Isolated Test ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	vocab_path := "temp_test_vocab.txt"
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\napple\n##s\nplay\n##ing\nun\n##want\n##ed\n,\n"
	_ = os.write_entire_file(vocab_path, transmute([]u8)vocab)
	defer os.remove(vocab_path)

	tok_inst, ok := tok.wordpiece_tokenizer_new(vocab_path, 16, allocator)
	expect(&passed, &total, ok, "WordPiece tokenizer loads a vocab.txt")
	if !ok {
		record_failures(total - passed)
		return
	}
	defer tok.wordpiece_tokenizer_free(&tok_inst)

	input_ids, attention_mask := tok.tokenize(&tok_inst, "Apples playing unwanted")
	defer delete(input_ids)
	defer delete(attention_mask)
	want := []int{2, 4, 5, 6, 7, 8, 9, 10, 3, 0, 0, 0, 0, 0, 0, 0}
	n_mask := 0
	for m in attention_mask {n_mask += m}
	expect(
		&passed,
		&total,
		slice.equal(input_ids, want) && n_mask == 9,
		fmt.tprintf(
			"'Apples playing unwanted' → [CLS] apple ##s play ##ing un ##want ##ed [SEP] + padding: %v",
			input_ids,
		),
	)
	raw := tok.wordpiece_encode(&tok_inst, "apples,playz")
	defer delete(raw)
	expect(
		&passed,
		&total,
		slice.equal(raw, []int{4, 5, 11, 1}),
		fmt.tprintf(
			"'apples,playz': punctuation split off, 'playz' one [UNK] (not play [UNK]): %v",
			raw,
		),
	)

	input_tensor, segment_tensor := tok.tokenize_to_tensors(&tok_inst, "apples", allocator)
	defer t.tensor_free(input_tensor)
	defer t.tensor_free(segment_tensor)
	expect(
		&passed,
		&total,
		input_tensor.shape == {1, 16, 1, 1} &&
		segment_tensor.shape == {1, 16, 1, 1} &&
		input_tensor.data.data[1] == 4,
		fmt.tprintf("tensors [1, 16]: shape %v", input_tensor.shape),
	)

	fmt.printf("WordPiece tokenizer: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
