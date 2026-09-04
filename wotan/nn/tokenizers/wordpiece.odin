package tokenizers

import l "../../linalg"
import t "../../tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

WordPieceTokenizer :: struct {
	vocab:         map[string]int,
	ids_to_tokens: []string,
	unk_token:     string,
	unk_id:        int,
	cls_id:        int,
	sep_id:        int,
	pad_id:        int,
	max_len:       int,
	allocator:     mem.Allocator,
}

wordpiece_tokenizer_new :: proc(
	vocab_path: string,
	max_len: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	WordPieceTokenizer,
	bool,
) {
	tok: WordPieceTokenizer
	tok.allocator = allocator
	tok.max_len = max_len
	tok.unk_token = "[UNK]"
	tok.unk_id = 100
	tok.cls_id = 101
	tok.sep_id = 102
	tok.pad_id = 0

	tok.vocab = make(map[string]int, allocator)

	data, err := os.read_entire_file(vocab_path, allocator)
	if err != nil {
		fmt.printf("Error reading vocab file: %v\n", err)
		return tok, false
	}
	defer delete(data, allocator)

	lines := strings.split(string(data), "\n", allocator)
	defer delete(lines, allocator)

	tok.ids_to_tokens = make([]string, len(lines), allocator)

	// Odin loop is strictly: for index, value in collection
	for line, i in lines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 {
			continue
		}

		owned_token := strings.clone(trimmed, allocator)
		tok.vocab[owned_token] = i
		tok.ids_to_tokens[i] = owned_token
	}

	return tok, true
}

wordpiece_tokenizer_free :: proc(tok: ^WordPieceTokenizer) {
	// '_' discards the int index, 'token' is the string value
	for token, _ in tok.ids_to_tokens {
		if len(token) > 0 {
			delete(token)
		}
	}
	delete(tok.ids_to_tokens, tok.allocator)
	delete(tok.vocab)
}

tokenize :: proc(
	tok: ^WordPieceTokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids: []int,
	attention_mask: []int,
) {
	words := strings.split(text, " ", allocator)
	defer delete(words, allocator)

	tokens := make([dynamic]string, allocator)
	defer delete(tokens)

	for word in words {
		if len(word) == 0 {
			continue
		}

		start := 0
		word_len := len(word)

		for start < word_len {
			end := word_len
			found := false

			for end > start {
				sub := word[start:end]

				if start > 0 {
					sub = fmt.aprintf("##%s", sub, allocator = allocator)
				}

				if id, ok := tok.vocab[sub]; ok {
					append(&tokens, sub)
					start = end
					found = true
					break
				}
				end -= 1
			}

			if !found {
				append(&tokens, tok.unk_token)
				break
			}
		}
	}

	actual_len := len(tokens)
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
		token_str := tokens[i]
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

	return input_ids, attention_mask
}

tokenize_to_tensors :: proc(
	tok: ^WordPieceTokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
) {
	ids, mask := tokenize(tok, text, allocator)
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
