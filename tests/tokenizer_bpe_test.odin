package tests

import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"

// BPE on a toy vocabulary (merges a b → ab, ab c → abc): merging by priority,
// the [CLS] … [SEP] input with padding, the tensors. tokenizer_reference_test
// compares with Hugging Face on the GPT-2 vocabulary.
tokenizer_bpe_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== BPE Tokenizer Isolated Test ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	vocab_path, merges_path := "temp_bpe_vocab.json", "temp_bpe_merges.txt"
	vocab_json := `{"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "[PAD]": 100, "[UNK]": 101, "[CLS]": 102, "[SEP]": 103}`
	merges_txt := "#version: 0.2\na b\nab c"
	_ = os.write_entire_file(vocab_path, transmute([]u8)vocab_json)
	_ = os.write_entire_file(merges_path, transmute([]u8)merges_txt)
	defer os.remove(vocab_path)
	defer os.remove(merges_path)

	tok_inst, ok := tok.bpe_tokenizer_new(vocab_path, merges_path, 16, allocator)
	expect(&passed, &total, ok, "BPE tokenizer loads a vocab.json / merges.txt")
	if !ok {
		record_failures(total - passed)
		return
	}
	defer tok.bpe_tokenizer_free(&tok_inst)

	PAD :: 100
	check :: proc(tk: ^tok.BPETokenizer, text: string, want: []int) -> bool {
		ids, mask := tok.bpe_tokenize(tk, text)
		defer delete(ids)
		defer delete(mask)
		for v, i in ids {
			if v != (want[i] if i < len(want) else PAD) || mask[i] != (1 if i < len(want) else 0) {return false}
		}
		return true
	}
	expect(&passed, &total, check(&tok_inst, "abc", {102, 4, 103}), "'abc' merges to one token: [CLS] abc [SEP] + padding")
	expect(&passed, &total, check(&tok_inst, "ac", {102, 0, 2, 103}), "'ac' stays two tokens (no merge a c)")
	raw := tok.bpe_encode(&tok_inst, "abcab")
	defer delete(raw)
	expect(&passed, &total, slice.equal(raw, []int{4, 3}), fmt.tprintf("'abcab' encodes to abc ab: %v", raw))

	input_tensor, segment_tensor := tok.bpe_tokenize_to_tensors(&tok_inst, "abc", allocator)
	defer t.tensor_free(input_tensor)
	defer t.tensor_free(segment_tensor)
	expect(
		&passed,
		&total,
		input_tensor.shape == {1, 16, 1, 1} && segment_tensor.shape == {1, 16, 1, 1} && input_tensor.data.data[0] == 4,
		fmt.tprintf("tensors [1, 16] (no special tokens): shape %v", input_tensor.shape),
	)

	fmt.printf("BPE tokenizer: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
