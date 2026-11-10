package tokenizers

import t "../../tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:unicode"
import "core:unicode/utf8"

// ============================================================================
// WordPiece (BERT), as Hugging Face's BertTokenizer
// ============================================================================
// vocab.txt: one token per line, the id is the line number. Encoding: the text
// is split at special tokens ([CLS], [SEP], [PAD], [UNK], [MASK]); control,
// format, private-use and unassigned characters (Unicode C*) are removed, whitespace becomes spaces and CJK
// ideographs get spaces around them; each word is lowercased and its accents
// stripped (NFD without nonspacing marks, unicode_tables.odin) unless the
// tokenizer is cased (lowercase = false, for cased models such as
// bert-base-cased: do_lower_case = False), then its punctuation is split off;
// each piece is matched greedily longest-first against the vocabulary
// (continuations as ##…). A piece that cannot be matched completely, or has
// more than 100 characters, becomes [UNK].
// Model input: tokenize ([CLS] text [SEP]), tokenize_pair ([CLS] a [SEP] b
// [SEP] with segment ids 0 / 1), tokenize_batch_to_tensors (several texts or
// pairs as one padded batch).

WordPieceTokenizer :: struct {
	vocab:         map[string]int,
	ids_to_tokens: []string,
	specials:      map[string]int,
	unk_token:     string,
	unk_id:        int,
	cls_id:        int,
	sep_id:        int,
	pad_id:        int,
	max_len:       int,
	lowercase:     bool, // uncased model: lowercase and strip accents
	allocator:     mem.Allocator,
}

WORDPIECE_MAX_CHARS :: 100 // longer words become [UNK]

wordpiece_tokenizer_new :: proc(
	vocab_path: string,
	max_len: int,
	allocator: mem.Allocator = context.allocator,
	lowercase := true,
) -> (
	WordPieceTokenizer,
	bool,
) {
	tok: WordPieceTokenizer
	tok.allocator = allocator
	tok.max_len = max_len
	tok.lowercase = lowercase
	tok.unk_token = "[UNK]"

	data, err := os.read_entire_file(vocab_path, allocator)
	if err != nil {
		fmt.printf("Error reading vocab file: %v\n", err)
		return tok, false
	}
	defer delete(data, allocator)

	tok.vocab = make(map[string]int, allocator)
	tok.specials = make(map[string]int, allocator)
	lines := strings.split_lines(string(data), allocator)
	defer delete(lines, allocator)
	tok.ids_to_tokens = make([]string, len(lines), allocator)
	for line, i in lines {
		token := strings.trim_right(line, "\r")
		if len(token) == 0 {continue}
		// a repeated token: the later line's id (as HF), both ids decode to it
		key, id, just_inserted, _ := map_entry(&tok.vocab, token)
		if just_inserted {key^ = strings.clone(token, allocator)}
		id^ = i
		tok.ids_to_tokens[i] = key^
	}
	for s in ([]string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}) {
		if id, ok := tok.vocab[s]; ok {tok.specials[s] = id}
	}

	if id, ok := tok.vocab["[PAD]"]; ok {tok.pad_id = id}
	if id, ok := tok.vocab["[UNK]"]; ok {tok.unk_id = id}
	if id, ok := tok.vocab["[CLS]"]; ok {tok.cls_id = id}
	if id, ok := tok.vocab["[SEP]"]; ok {tok.sep_id = id}

	return tok, true
}

wordpiece_tokenizer_free :: proc(tok: ^WordPieceTokenizer) {
	for token in tok.vocab {delete(token, tok.allocator)}
	delete(tok.ids_to_tokens, tok.allocator)
	delete(tok.vocab)
	delete(tok.specials)
}

// BERT's character classes (BasicTokenizer)
@(private = "file")
_is_bert_whitespace :: proc(r: rune) -> bool {
	switch r {
	case ' ', '\t', '\n', '\r', 0xA0, 0x1680, 0x2000 ..= 0x200A, 0x202F, 0x205F, 0x3000:
		return true // and the space separators (Zs)
	}
	return false
}

// The C* categories but tab, newline and carriage return: removed by the text
// cleaner (surrogates, Cs, cannot occur in decoded UTF-8)
@(private = "file")
_is_bert_control :: proc(r: rune) -> bool {
	if r == '\t' || r == '\n' || r == '\r' {return false}
	if r < 0x20 || (r >= 0x7F && r <= 0x9F) {return true} 	// Cc
	if r < 0xAD {return false} 	// the first Cf, Co and Cn code points are above
	for rg in _FORMAT_RANGES {if r >= rg[0] && r <= rg[1] {return true}} 	// Cf
	if (r >= 0xE000 && r <= 0xF8FF) ||
	   (r >= 0xF0000 && r <= 0xFFFFD) ||
	   (r >= 0x100000 && r <= 0x10FFFD) {
		return true // Co (private use)
	}
	// Cn (unassigned): the last range starting at or before r
	lo, hi := 0, len(_UNASSIGNED_RANGES)
	for lo < hi {
		mid := (lo + hi) / 2
		if _UNASSIGNED_RANGES[mid][0] <= r {lo = mid + 1} else {hi = mid}
	}
	return lo > 0 && r <= _UNASSIGNED_RANGES[lo - 1][1]
}

@(private = "file")
_is_bert_punct :: proc(r: rune) -> bool {
	// all non-alphanumeric ASCII counts, also $ + < = > ^ ` | ~
	if (r >= 33 && r <= 47) ||
	   (r >= 58 && r <= 64) ||
	   (r >= 91 && r <= 96) ||
	   (r >= 123 && r <= 126) {
		return true
	}
	return unicode.is_punct(r)
}

@(private = "file")
_is_cjk :: proc(r: rune) -> bool {
	switch r {
	case 0x4E00 ..=
	     0x9FFF,
	     0x3400 ..=
	     0x4DBF,
	     0x20000 ..=
	     0x2A6DF,
	     0x2A700 ..=
	     0x2B73F,
	     0x2B740 ..=
	     0x2B81F,
	     0x2B820 ..=
	     0x2CEAF,
	     0xF900 ..=
	     0xFAFF,
	     0x2F800 ..=
	     0x2FA1F:
		return true
	}
	return false
}

// r without accents (NFD without nonspacing marks) appended to b
@(private = "file")
_write_stripped :: proc(b: ^strings.Builder, r: rune) {
	if unicode.is_nonspacing_mark(r) {return}
	if r >= 0xAC00 && r <= 0xD7A3 { 	// Hangul syllable → its jamo
		s := r - 0xAC00
		strings.write_rune(b, 0x1100 + s / 588)
		strings.write_rune(b, 0x1161 + (s % 588) / 28)
		if s % 28 != 0 {strings.write_rune(b, 0x11A7 + s % 28)}
		return
	}
	i, found := slice.binary_search_by(_ACCENT_STRIP[:], r, proc(e: struct {
			cp:   rune,
			base: [3]rune,
		}, r: rune) -> slice.Ordering {return slice.cmp(e.cp, r)})
	if !found {
		strings.write_rune(b, r)
		return
	}
	for base in _ACCENT_STRIP[i].base {if base != 0 {strings.write_rune(b, base)}}
}

// Greedy longest-match-first WordPiece of one piece (no whitespace, no punctuation)
@(private = "file")
_wordpiece :: proc(
	tok: ^WordPieceTokenizer,
	piece: string,
	key: ^strings.Builder,
	out: ^[dynamic]int,
) {
	if utf8.rune_count_in_string(piece) > WORDPIECE_MAX_CHARS {
		append(out, tok.unk_id)
		return
	}
	first := len(out)
	start := 0
	for start < len(piece) {
		end := len(piece)
		found := false
		for end > start {
			strings.builder_reset(key)
			if start > 0 {strings.write_string(key, "##")}
			strings.write_string(key, piece[start:end])
			if id, ok := tok.vocab[strings.to_string(key^)]; ok {
				append(out, id)
				found = true
				break
			}
			end -= 1
			for end > start && !utf8.rune_start(piece[end]) {end -= 1} 	// whole characters only
		}
		if !found {
			resize(out, first)
			append(out, tok.unk_id) // the whole piece is unknown
			return
		}
		start = end
	}
}

@(private = "file")
_wordpiece_text :: proc(text: string, out: ^[dynamic]int, data: rawptr) {
	tok := (^WordPieceTokenizer)(data)
	clean := strings.builder_make(0, len(text) + 16, context.temp_allocator)
	word := strings.builder_make(context.temp_allocator)
	key := strings.builder_make(context.temp_allocator)
	defer strings.builder_destroy(&clean)
	defer strings.builder_destroy(&word)
	defer strings.builder_destroy(&key)
	for r in text {
		switch {
		case r == 0 || r == 0xFFFD || _is_bert_control(r):
		case _is_bert_whitespace(r):
			strings.write_byte(&clean, ' ')
		case _is_cjk(r):
			strings.write_byte(&clean, ' ')
			strings.write_rune(&clean, r)
			strings.write_byte(&clean, ' ')
		case:
			strings.write_rune(&clean, r)
		}
	}
	rest := strings.to_string(clean)
	for w in strings.fields_iterator(&rest) {
		strings.builder_reset(&word)
		if tok.lowercase {
			for r in w {_write_stripped(&word, unicode.to_lower(r))}
		} else {
			strings.write_string(&word, w)
		}
		// split off punctuation
		normalized := strings.to_string(word)
		start := 0
		for r, i in normalized {
			if !_is_bert_punct(r) {continue}
			if i > start {_wordpiece(tok, normalized[start:i], &key, out)}
			n := utf8.rune_size(r)
			_wordpiece(tok, normalized[i:i + n], &key, out)
			start = i + n
		}
		if start < len(normalized) {_wordpiece(tok, normalized[start:], &key, out)}
	}
}

// Token ids of text (no special tokens added, no padding)
wordpiece_encode :: proc(
	tok: ^WordPieceTokenizer,
	text: string,
	allocator := context.allocator,
) -> []int {
	out := make([dynamic]int, 0, len(text) / 3 + 1, allocator)
	_split_special(text, tok.specials, &out, tok, _wordpiece_text)
	return out[:]
}

// Text of token ids as Hugging Face's decode: tokens joined by spaces, ##
// continuations attached, spaces before . , ! ? and English contractions removed
wordpiece_decode :: proc(
	tok: ^WordPieceTokenizer,
	ids: []int,
	allocator := context.allocator,
	skip_special_tokens := false,
) -> string {
	b := strings.builder_make(allocator)
	for id in ids {
		if id < 0 || id >= len(tok.ids_to_tokens) {continue}
		token := tok.ids_to_tokens[id]
		if skip_special_tokens && token in tok.specials {continue}
		if strings.has_prefix(token, "##") {
			strings.write_string(&b, token[2:])
		} else {
			if strings.builder_len(b) > 0 {strings.write_byte(&b, ' ')}
			strings.write_string(&b, token)
		}
	}
	s := strings.to_string(b)
	for pair in ([][2]string{{" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","}, {" ' ", "'"}, {" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"}, {" 've", "'ve"}, {" 're", "'re"}}) {
		replaced, was_alloc := strings.replace_all(s, pair[0], pair[1], allocator)
		if was_alloc {
			delete(s, allocator)
			s = replaced
		}
	}
	return s
}

// max_len ids: [CLS] tokens [SEP], truncated or padded, and the attention mask
tokenize :: proc(
	tok: ^WordPieceTokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids: []int,
	attention_mask: []int,
) {
	raw := wordpiece_encode(tok, text, allocator)
	defer delete(raw, allocator)
	segments: []int
	input_ids, attention_mask, segments = _pack(
		raw,
		nil,
		tok.max_len,
		tok.cls_id,
		tok.sep_id,
		tok.pad_id,
		true,
		allocator,
	)
	delete(segments, allocator)
	return
}

// tokenize's ids and segment ids as [1, max_len] tensors
tokenize_to_tensors :: proc(
	tok: ^WordPieceTokenizer,
	text: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
) {
	ids, mask := tokenize(tok, text, allocator)
	defer delete(ids, allocator)
	defer delete(mask, allocator)
	input_ids_tensor, segment_ids_tensor, _ = _to_tensors(ids, nil, nil, 1, allocator)
	return
}

// max_len ids of a sentence pair: [CLS] a [SEP] b [SEP], segment ids 0 for
// [CLS] a [SEP] and 1 for b [SEP] (question answering, entailment, next
// sentence prediction); too long: the longer text is cut first (HF's
// longest_first), then padding
tokenize_pair :: proc(
	tok: ^WordPieceTokenizer,
	text_a, text_b: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids: []int,
	attention_mask: []int,
	segment_ids: []int,
) {
	a := wordpiece_encode(tok, text_a, allocator)
	b := wordpiece_encode(tok, text_b, allocator)
	defer delete(a, allocator)
	defer delete(b, allocator)
	return _pack(a, b, tok.max_len, tok.cls_id, tok.sep_id, tok.pad_id, true, allocator)
}

// tokenize_pair's ids and segment ids as [1, max_len] tensors
tokenize_pair_to_tensors :: proc(
	tok: ^WordPieceTokenizer,
	text_a, text_b: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
) {
	ids, mask, segments := tokenize_pair(tok, text_a, text_b, allocator)
	defer delete(ids, allocator)
	defer delete(mask, allocator)
	defer delete(segments, allocator)
	input_ids_tensor, segment_ids_tensor, _ = _to_tensors(ids, segments, nil, 1, allocator)
	return
}

// Several texts (or pairs texts[i], pairs[i]) as one batch [len(texts), L] of
// ids, segment ids and attention mask, each as tokenize / tokenize_pair;
// L = the longest input (pad_to_longest) or max_len
tokenize_batch_to_tensors :: proc(
	tok: ^WordPieceTokenizer,
	texts: []string,
	allocator: mem.Allocator = context.allocator,
	pairs: []string = nil,
	pad_to_longest := true,
) -> (
	input_ids: ^t.Tensor,
	segment_ids: ^t.Tensor,
	attention_mask: ^t.Tensor,
) {
	assert(
		pairs == nil || len(pairs) == len(texts),
		"tokenize_batch_to_tensors: one pair text per text",
	)
	Batch :: struct {
		tok:          ^WordPieceTokenizer,
		texts, pairs: []string,
	}
	batch := Batch{tok, texts, pairs}
	return _batch_to_tensors(
		len(texts),
		tok.max_len,
		pad_to_longest,
		&batch,
		proc(i: int, data: rawptr, allocator: mem.Allocator) -> (ids, mask, segments: []int) {
			b := (^Batch)(data)
			if b.pairs != nil {return tokenize_pair(b.tok, b.texts[i], b.pairs[i], allocator)}
			raw := wordpiece_encode(b.tok, b.texts[i], allocator)
			defer delete(raw, allocator)
			return _pack(
				raw,
				nil,
				b.tok.max_len,
				b.tok.cls_id,
				b.tok.sep_id,
				b.tok.pad_id,
				true,
				allocator,
			)
		},
		allocator,
	)
}
