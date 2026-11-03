package tokenizers

import t "../../tensor"
import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import "core:unicode"
import "core:unicode/utf8"

// ============================================================================
// Byte-level BPE (GPT-2), as Hugging Face's GPT2Tokenizer
// ============================================================================
// vocab.json maps tokens to ids, merges.txt lists the merges by priority.
// Encoding: the text is split at special tokens (<|endoftext|>, [CLS], … if
// in the vocabulary) and pre-tokenized with GPT-2's pattern
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// (a space stays with the word after it); each piece's UTF-8 bytes become the
// printable characters GPT-2 assigns to bytes (space → Ġ, newline → Ċ), and
// the highest-priority merge of adjacent symbols is applied until none is
// left. Merges whose parts or result are not in the vocabulary are ignored.

BPETokenizer :: struct {
	vocab:         map[string]int,
	merges:        map[[2]int][2]int, // pair of token ids → {rank, merged id}
	ids_to_tokens: []string,
	specials:      map[string]int, // never split: <|…|> and [CLS] / [SEP] / [PAD] / [UNK] / [MASK]
	byte_ids:      [256]int, // id of each byte's character, -1 if not in the vocabulary
	byte_runes:    [256]rune, // GPT-2's character for each byte
	unk_token:     string,
	unk_id:        int,
	cls_id:        int,
	sep_id:        int,
	pad_id:        int,
	max_len:       int,
	allocator:     mem.Allocator,
}

// GPT-2's bytes_to_unicode: printable Latin-1 bytes stand for themselves, the
// others (control characters, space, …) are mapped to 256, 257, … in order
_gpt2_byte_runes :: proc() -> (runes: [256]rune) {
	n: rune = 0
	for b in 0 ..< 256 {
		printable := (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
		if printable {
			runes[b] = rune(b)
		} else {
			runes[b] = 256 + n
			n += 1
		}
	}
	return
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
	tok.byte_runes = _gpt2_byte_runes()

	vocab_data, err := os.read_entire_file(vocab_path, allocator)
	if err != nil {
		fmt.printf("Error reading vocab file: %v\n", err)
		return tok, false
	}
	defer delete(vocab_data, allocator)
	tok.vocab = make(map[string]int, allocator)
	if jerr := json.unmarshal(vocab_data, &tok.vocab, allocator = allocator); jerr != nil {
		fmt.printf("Invalid vocab.json: %v\n", jerr)
		bpe_tokenizer_free(&tok)
		return tok, false
	}

	max_id := -1
	for _, id in tok.vocab {
		if id < 0 {
			fmt.println("Invalid vocab.json: negative token id")
			bpe_tokenizer_free(&tok)
			return tok, false
		}
		max_id = max(max_id, id)
	}
	tok.ids_to_tokens = make([]string, max_id + 1, allocator)
	tok.specials = make(map[string]int, allocator)
	for token, id in tok.vocab {
		tok.ids_to_tokens[id] = token
		is_special := strings.has_prefix(token, "<|") && strings.has_suffix(token, "|>")
		switch token {
		case "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]":
			is_special = true
		}
		if is_special {tok.specials[token] = id}
	}
	for r, b in tok.byte_runes {
		buf, n := utf8.encode_rune(r)
		id, ok := tok.vocab[string(buf[:n])]
		tok.byte_ids[b] = id if ok else -1
	}

	merges_data, err2 := os.read_entire_file(merges_path, allocator)
	if err2 != nil {
		fmt.printf("Error reading merges file: %v\n", err2)
		bpe_tokenizer_free(&tok)
		return tok, false
	}
	defer delete(merges_data, allocator)
	tok.merges = make(map[[2]int][2]int, allocator)
	rank := 0
	text := string(merges_data)
	for line in strings.split_lines_iterator(&text) {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 || strings.has_prefix(trimmed, "#version") {continue}
		space := strings.index_byte(trimmed, ' ')
		if space < 0 {continue}
		first, second := trimmed[:space], trimmed[space + 1:]
		merged := strings.concatenate({first, second}, context.temp_allocator)
		a, ok_a := tok.vocab[first]
		b, ok_b := tok.vocab[second]
		m, ok_m := tok.vocab[merged]
		delete(merged, context.temp_allocator)
		if ok_a && ok_b && ok_m {tok.merges[{a, b}] = {rank, m}} 	// a repeated merge: the later rank, as HF
		rank += 1
	}

	if id, ok := tok.vocab["[PAD]"]; ok {tok.pad_id = id}
	if id, ok := tok.vocab["[UNK]"]; ok {
		tok.unk_id = id
	} else if id, ok := tok.vocab["<|endoftext|>"]; ok {
		tok.unk_token, tok.unk_id = "<|endoftext|>", id // GPT-2's unknown token
	}
	if id, ok := tok.vocab["[CLS]"]; ok {tok.cls_id = id}
	if id, ok := tok.vocab["[SEP]"]; ok {tok.sep_id = id}

	return tok, true
}

bpe_tokenizer_free :: proc(tok: ^BPETokenizer) {
	for token in tok.vocab {delete(token, tok.allocator)}
	delete(tok.vocab)
	delete(tok.merges)
	delete(tok.specials)
	delete(tok.ids_to_tokens, tok.allocator)
}

// One pre-tokenized piece: its bytes as symbols, merged by priority
@(private = "file")
_bpe_piece :: proc(tok: ^BPETokenizer, piece: string, out: ^[dynamic]int) {
	start := len(out)
	for i in 0 ..< len(piece) {append(out, tok.byte_ids[piece[i]])}
	for {
		symbols := out[start:]
		best_rank, best := max(int), -1
		for i in 0 ..< len(symbols) - 1 {
			if m, ok := tok.merges[{symbols[i], symbols[i + 1]}]; ok && m[0] < best_rank {
				best_rank, best = m[0], i
			}
		}
		if best < 0 {break}
		// merge every occurrence of the pair, left to right
		a, b := symbols[best], symbols[best + 1]
		merged := tok.merges[{a, b}][1]
		w := 0
		for r := 0; r < len(symbols); r += 1 {
			if r + 1 < len(symbols) && symbols[r] == a && symbols[r + 1] == b {
				symbols[w] = merged
				r += 1
			} else {
				symbols[w] = symbols[r]
			}
			w += 1
		}
		resize(out, start + w)
	}
	for &id in out[start:] {if id < 0 {id = tok.unk_id}}
}

// GPT-2's pre-tokenizer: the end of the piece starting at text[i:]
@(private = "file")
_gpt2_piece_end :: proc(text: string, i: int) -> int {
	Class :: enum {
		Letter,
		Number,
		Space,
		Other,
	}
	class :: proc(r: rune) -> Class {
		if unicode.is_letter(r) {return .Letter}
		if unicode.is_number(r) {return .Number}
		// Python's \s: also the separators 0x1C - 0x1F
		if unicode.is_space(r) || (r >= 0x1C && r <= 0x1F) {return .Space}
		return .Other
	}
	run_end :: proc(text: string, i: int, c: Class) -> int {
		j := i
		for j < len(text) {
			r, n := utf8.decode_rune_in_string(text[j:])
			if class(r) != c {break}
			j += n
		}
		return j
	}
	rest := text[i:]
	// contractions
	if rest[0] == '\'' {
		for s in ([]string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}) {
			if strings.has_prefix(rest, s) {return i + len(s)}
		}
	}
	r, n := utf8.decode_rune_in_string(rest)
	c := class(r)
	// " ?" before letters, numbers and others
	if r == ' ' && n < len(rest) {
		r2, n2 := utf8.decode_rune_in_string(rest[n:])
		if c2 := class(r2); c2 != .Space {return run_end(text, i + n + n2, c2)}
	}
	if c != .Space {return run_end(text, i + n, c)}
	// whitespace: \s+(?!\S) leaves the last one to the word after it
	j := run_end(text, i + n, .Space)
	if j < len(text) {
		last_start := j - 1
		for last_start > i && !utf8.rune_start(text[last_start]) {last_start -= 1}
		if last_start > i {return last_start}
	}
	return j
}

@(private = "file")
_bpe_text :: proc(text: string, out: ^[dynamic]int, data: rawptr) {
	tok := (^BPETokenizer)(data)
	for i := 0; i < len(text); {
		end := _gpt2_piece_end(text, i)
		_bpe_piece(tok, text[i:end], out)
		i = end
	}
}

// Token ids of text (no special tokens added, no padding)
bpe_encode :: proc(tok: ^BPETokenizer, text: string, allocator := context.allocator) -> []int {
	out := make([dynamic]int, 0, len(text) / 3 + 1, allocator)
	_split_special(text, tok.specials, &out, tok, _bpe_text)
	return out[:]
}

// Text of token ids: GPT-2's characters back to their bytes (the result may
// end in an incomplete UTF-8 sequence if the ids do)
bpe_decode :: proc(tok: ^BPETokenizer, ids: []int, allocator := context.allocator) -> string {
	byte_of: map[rune]u8
	defer delete(byte_of)
	for r, b in tok.byte_runes {byte_of[r] = u8(b)}
	out := make([dynamic]u8, 0, 4 * len(ids), allocator)
	for id in ids {
		if id < 0 || id >= len(tok.ids_to_tokens) {continue}
		for r in tok.ids_to_tokens[id] {
			if b, ok := byte_of[r]; ok {
				append(&out, b)
			} else {
				buf, n := utf8.encode_rune(r)
				append(&out, ..buf[:n])
			}
		}
	}
	return string(out[:])
}

// bpe_tokenize tokenizes text to max_len ids and the attention mask.
// add_special_tokens: If true, adds [CLS] and [SEP] (for BERT). If false, returns raw tokens (for GPT-2).
bpe_tokenize :: proc(
	tok: ^BPETokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
	add_special_tokens: bool = true,
) -> (
	input_ids: []int,
	attention_mask: []int,
) {
	raw := bpe_encode(tok, text, allocator)
	defer delete(raw, allocator)
	return _pack(
		raw,
		tok.max_len,
		tok.cls_id,
		tok.sep_id,
		tok.pad_id,
		add_special_tokens,
		allocator,
	)
}

// max_len ids without special tokens (padded) and segment ids as [1, max_len] tensors
bpe_tokenize_to_tensors :: proc(
	tok: ^BPETokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
) {
	ids, mask := bpe_tokenize(tok, text, allocator, false)
	defer delete(ids, allocator)
	defer delete(mask, allocator)
	return _to_tensors(ids, allocator)
}
