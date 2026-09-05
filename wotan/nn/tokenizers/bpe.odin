package tokenizers

import l "../../linalg"
import t "../../tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

BPETokenizer :: struct {
	vocab:         map[string]int,
	merges:        map[string]int,
	ids_to_tokens: []string,
	unk_token:     string,
	unk_id:        int,
	cls_id:        int,
	sep_id:        int,
	pad_id:        int,
	max_len:       int,
	allocator:     mem.Allocator,
}

bpe_tokenizer_new :: proc(
	vocab_path: string,
	merges_path: string,
	max_len: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	BPETokenizer,
	bool,
) {
	tok: BPETokenizer
	tok.allocator = allocator
	tok.max_len = max_len
	tok.unk_token = "[UNK]"

	tok.vocab = make(map[string]int, allocator)
	tok.merges = make(map[string]int, allocator)

	vocab_data, err := os.read_entire_file(vocab_path, allocator)
	if err != nil {
		fmt.printf("Error reading vocab file: %v\n", err)
		return tok, false
	}
	defer delete(vocab_data, allocator)

	vocab_str := string(vocab_data)
	pos := strings.index(vocab_str, "{")
	if pos < 0 {
		fmt.println("Invalid vocab.json format")
		return tok, false
	}

	for pos < len(vocab_str) {
		key_start := strings.index(vocab_str[pos:], "\"")
		if key_start < 0 {break}
		key_start += pos

		key_end := strings.index(vocab_str[key_start + 1:], "\"")
		if key_end < 0 {break}
		key_end += key_start + 1

		key := vocab_str[key_start + 1:key_end]

		colon_pos := strings.index(vocab_str[key_end:], ":")
		if colon_pos < 0 {break}
		colon_pos += key_end

		val_start := colon_pos + 1
		val_end := val_start
		for val_end < len(vocab_str) && (vocab_str[val_end] == ' ' || vocab_str[val_end] == '\t') {
			val_start += 1
			val_end += 1
		}
		for val_end < len(vocab_str) && vocab_str[val_end] >= '0' && vocab_str[val_end] <= '9' {
			val_end += 1
		}

		if val_end > val_start {
			val_str := vocab_str[val_start:val_end]
			val := 0
			// ✅ ODIN: for value, index in string
			for ch, _ in val_str {
				val = val * 10 + int(ch - '0')
			}

			owned_key := strings.clone(key, allocator)
			tok.vocab[owned_key] = val
		}

		pos = val_end
	}

	max_id := 0
	// ✅ ODIN: for key, value in map
	for token, id in tok.vocab {
		if id > max_id {
			max_id = id
		}
	}
	tok.ids_to_tokens = make([]string, max_id + 1, allocator)
	for token, id in tok.vocab {
		tok.ids_to_tokens[id] = token
	}

	merges_data, err2 := os.read_entire_file(merges_path, allocator)
	if err2 != nil {
		fmt.printf("Error reading merges file: %v\n", err2)
		return tok, false
	}
	defer delete(merges_data, allocator)

	merge_lines := strings.split(string(merges_data), "\n", allocator)
	defer delete(merge_lines, allocator)

	rank := 0
	// ✅ ODIN: for value, index in slice
	for line, i in merge_lines {
		_ = i
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 || trimmed[0] == '#' {
			continue
		}

		space_pos := strings.index(trimmed, " ")
		if space_pos < 0 {
			continue
		}

		first := trimmed[:space_pos]
		second := trimmed[space_pos + 1:]

		merge_key := fmt.aprintf("%s %s", first, second, allocator = allocator)
		tok.merges[merge_key] = rank
		rank += 1
	}

	if id, ok := tok.vocab["[PAD]"]; ok {tok.pad_id = id}
	if id, ok := tok.vocab["[UNK]"]; ok {tok.unk_id = id}
	if id, ok := tok.vocab["[CLS]"]; ok {tok.cls_id = id}
	if id, ok := tok.vocab["[SEP]"]; ok {tok.sep_id = id}

	return tok, true
}

bpe_tokenizer_free :: proc(tok: ^BPETokenizer) {
	// ✅ ODIN: for value, index in slice
	for token, i in tok.ids_to_tokens {
		_ = i
		if len(token) > 0 {
			delete(token, tok.allocator)
		}
	}
	delete(tok.ids_to_tokens, tok.allocator)
	delete(tok.vocab)

	// ✅ ODIN: for key, value in map
	for key, _ in tok.merges {
		delete(key, tok.allocator)
	}
	delete(tok.merges)
}

_bpe :: proc(tok: ^BPETokenizer, word: []string, allocator: mem.Allocator) -> [dynamic]string {
	if len(word) < 2 {
		res := make([dynamic]string, allocator)
		// ✅ ODIN: for value, index in slice
		for s, _ in word {
			append(&res, s)
		}
		return res
	}

	symbols := make([dynamic]string, allocator)
	for s, _ in word {
		append(&symbols, s)
	}

	for {
		best_pair_key := ""
		best_rank := 999999999

		for i in 0 ..< len(symbols) - 1 {
			pair_key := fmt.aprintf("%s %s", symbols[i], symbols[i + 1], allocator = allocator)
			if rank, ok := tok.merges[pair_key]; ok {
				if rank < best_rank {
					best_rank = rank
					best_pair_key = pair_key
				}
			}
			delete(pair_key, allocator)
		}

		if len(best_pair_key) == 0 {
			break
		}

		space_pos := strings.index(best_pair_key, " ")
		first := best_pair_key[:space_pos]
		second := best_pair_key[space_pos + 1:]
		merged := fmt.aprintf("%s%s", first, second, allocator = allocator)

		new_symbols := make([dynamic]string, allocator)
		i := 0
		for i < len(symbols) {
			if i < len(symbols) - 1 && symbols[i] == first && symbols[i + 1] == second {
				append(&new_symbols, merged)
				i += 2
			} else {
				append(&new_symbols, symbols[i])
				i += 1
			}
		}

		delete(symbols)
		symbols = new_symbols
		delete(best_pair_key, allocator)
	}

	return symbols
}

bpe_tokenize :: proc(
	tok: ^BPETokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids: []int,
	attention_mask: []int,
) {
	words := strings.split(text, " ", allocator)
	defer delete(words, allocator)

	all_tokens := make([dynamic]string, allocator)

	// ✅ ODIN: for value, index in slice
	for word, _ in words {
		if len(word) == 0 {continue}

		chars := make([dynamic]string, allocator)
		// ✅ ODIN: for value, index in string
		for ch, _ in word {
			char_str := fmt.aprintf("%c", ch, allocator = allocator)
			append(&chars, char_str)
		}

		bpe_tokens := _bpe(tok, chars[:], allocator)

		for token, _ in bpe_tokens {
			append(&all_tokens, token)
		}

		delete(chars)
		delete(bpe_tokens)
	}

	actual_len := len(all_tokens)
	total_len := actual_len + 2

	if total_len > tok.max_len {
		total_len = tok.max_len
		actual_len = total_len - 2
	}

	input_ids = make([]int, tok.max_len, allocator)
	attention_mask = make([]int, tok.max_len, allocator)

	input_ids[0] = tok.cls_id
	attention_mask[0] = 1

	for i in 0 ..< actual_len {
		token_str := all_tokens[i]
		if id, ok := tok.vocab[token_str]; ok {
			input_ids[i + 1] = id
		} else {
			input_ids[i + 1] = tok.unk_id
		}
		attention_mask[i + 1] = 1
	}

	if total_len > 1 {
		input_ids[actual_len + 1] = tok.sep_id
		attention_mask[actual_len + 1] = 1
	}

	for i in total_len ..< tok.max_len {
		input_ids[i] = tok.pad_id
		attention_mask[i] = 0
	}

	for token, _ in all_tokens {
		delete(token, allocator)
	}
	delete(all_tokens)

	return input_ids, attention_mask
}

bpe_tokenize_to_tensors :: proc(
	tok: ^BPETokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
) {
	ids, mask := bpe_tokenize(tok, text, allocator)
	defer {
		delete(ids, allocator)
		delete(mask, allocator)
	}

	batch := 1
	seq_len := tok.max_len

	ids_data := l.matrix_new(f64, 1, batch * seq_len, allocator)
	seg_data := l.matrix_new(f64, 1, batch * seq_len, allocator)

	for i in 0 ..< seq_len {
		ids_data.data[i] = f64(ids[i])
		seg_data.data[i] = f64(mask[i])
	}

	in_tensor := t.tensor_new(ids_data, false, allocator)
	in_tensor.shape = [4]int{batch, seq_len, 1, 1}

	seg_tensor := t.tensor_new(seg_data, false, allocator)
	seg_tensor.shape = [4]int{batch, seq_len, 1, 1}

	return in_tensor, seg_tensor
}
