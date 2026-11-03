package tokenizers

import pq "core:container/priority_queue"
import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os"
import "core:slice"
import "core:strings"
import "core:unicode/utf8"

// ============================================================================
// Training a byte-level BPE vocabulary, as Hugging Face's BpeTrainer
// ============================================================================
// As tokenizers' BpeTrainer with the ByteLevel pre-tokenizer and its whole
// alphabet (ByteLevelBPETokenizer.train): the texts are pre-tokenized as
// bpe_encode does (GPT-2's pattern) and the distinct pieces counted. The
// vocabulary starts with the special tokens (ids 0, 1, …), then GPT-2's 256
// byte characters in code point order; then, until it has vocab_size tokens,
// the most frequent pair of adjacent symbols over all pieces (ties: the pair
// of smaller ids) is merged into a new token, everywhere. Training stops early
// when no pair occurs at least max(min_frequency, 1) times.
// Unlike BpeTrainer, the special tokens are cut out of the texts first (as
// bpe_encode does): "<|endoftext|>" between documents adds no merges.
// The result encodes like a tokenizer loaded with bpe_tokenizer_new;
// bpe_save writes its vocab.json / merges.txt (readable by GPT2Tokenizer).

@(private = "file")
Pair :: [2]int

@(private = "file")
Merge_Entry :: struct {
	count: int,
	pair:  Pair,
	pos:   []int, // the pieces the pair (may) occur in
}

@(private = "file")
Train_State :: struct {
	pair_counts: map[Pair]int, // over all pieces, weighted by their counts
	where_new:   map[Pair][dynamic]int, // pairs that just gained occurrences, and where
	piece_count: []int,
	scratch:     mem.Allocator,
}

// a pair's count changes by delta occurrences in piece iw
@(private = "file")
_train_change :: proc(st: ^Train_State, p: Pair, delta, iw: int) {
	st.pair_counts[p] += delta * st.piece_count[iw]
	if delta <= 0 {return}
	if p not_in st.where_new {st.where_new[p] = make([dynamic]int, st.scratch)}
	pos := &st.where_new[p]
	if len(pos) == 0 || pos[len(pos) - 1] != iw {append(pos, iw)} 	// pieces come in order
}

// the pairs that gained occurrences into the queue (with their current counts)
@(private = "file")
_train_flush :: proc(st: ^Train_State, queue: ^pq.Priority_Queue(Merge_Entry)) {
	for p, pos in st.where_new {
		if c := st.pair_counts[p]; c > 0 {pq.push(queue, Merge_Entry{c, p, pos[:]})}
	}
	clear(&st.where_new)
}

bpe_train :: proc(
	texts: []string,
	vocab_size: int,
	special_tokens: []string = {"<|endoftext|>"},
	min_frequency := 0,
	max_len := 1024,
	allocator := context.allocator,
) -> BPETokenizer {
	arena: vmem.Arena
	if err := vmem.arena_init_growing(&arena); err != nil {panic("bpe_train: cannot allocate scratch memory")}
	defer vmem.arena_destroy(&arena)
	scratch := vmem.arena_allocator(&arena)

	tok: BPETokenizer
	tok.allocator = allocator
	tok.max_len = max_len
	tok.unk_token = "[UNK]"
	tok.byte_runes = _gpt2_byte_runes()
	tok.vocab = make(map[string]int, allocator)
	tokens := make([dynamic]string, scratch) // by id (the strings are the vocabulary's keys)
	add :: proc(tok: ^BPETokenizer, tokens: ^[dynamic]string, s: string) -> int {
		if id, ok := tok.vocab[s]; ok {return id}
		owned := strings.clone(s, tok.allocator)
		tok.vocab[owned] = len(tokens)
		append(tokens, owned)
		return len(tokens) - 1
	}

	// 1. the special tokens, then the byte characters in code point order
	for s in special_tokens {add(&tok, &tokens, s)}
	sorted := tok.byte_runes
	slice.sort(sorted[:])
	for r in sorted {
		buf, n := utf8.encode_rune(r)
		add(&tok, &tokens, string(buf[:n]))
	}
	byte_id: [256]int
	for r, b in tok.byte_runes {
		buf, n := utf8.encode_rune(r)
		byte_id[b] = tok.vocab[string(buf[:n])]
	}

	// 2. the distinct pieces of the texts (outside special tokens), counted
	specials := make(map[string]int, scratch)
	for s in special_tokens {specials[s] = tok.vocab[s]}
	piece_counts := make(map[string]int, scratch)
	ignored := make([dynamic]int, scratch)
	for text in texts {
		_split_special(text, specials, &ignored, &piece_counts, proc(text: string, _: ^[dynamic]int, data: rawptr) {
			counts := (^map[string]int)(data)
			for i := 0; i < len(text); {
				end := _gpt2_piece_end(text, i)
				counts[text[i:end]] += 1
				i = end
			}
		})
		clear(&ignored)
	}
	pieces := make([][dynamic]int, len(piece_counts), scratch) // their symbols
	st := Train_State {
		pair_counts = make(map[Pair]int, scratch),
		where_new   = make(map[Pair][dynamic]int, scratch),
		piece_count = make([]int, len(piece_counts), scratch),
		scratch     = scratch,
	}
	{
		k := 0
		for piece, c in piece_counts {
			pieces[k] = make([dynamic]int, len(piece), scratch)
			for i in 0 ..< len(piece) {pieces[k][i] = byte_id[piece[i]]}
			st.piece_count[k] = c
			k += 1
		}
	}

	// 3. the pairs' counts, in a queue: most frequent first, ties the smaller pair
	queue: pq.Priority_Queue(Merge_Entry)
	pq.init(&queue, proc(a, b: Merge_Entry) -> bool {
			if a.count != b.count {return a.count > b.count}
			return a.pair[0] < b.pair[0] || (a.pair[0] == b.pair[0] && a.pair[1] < b.pair[1])
		}, pq.default_swap_proc(Merge_Entry), allocator = scratch)
	for symbols, iw in pieces {
		for i in 0 ..< len(symbols) - 1 {_train_change(&st, {symbols[i], symbols[i + 1]}, 1, iw)}
	}
	_train_flush(&st, &queue)

	// 4. merges (an entry whose count is out of date goes back with the current one)
	merges := make([dynamic][3]int, scratch) // a, b, merged
	for len(tokens) < vocab_size && pq.len(queue) > 0 {
		top := pq.pop(&queue)
		if c := st.pair_counts[top.pair]; top.count != c {
			top.count = c
			pq.push(&queue, top)
			continue
		}
		if top.count < max(min_frequency, 1) {break}
		a, b := top.pair[0], top.pair[1]
		joined := strings.concatenate({tokens[a], tokens[b]}, scratch)
		merged := add(&tok, &tokens, joined)
		append(&merges, [3]int{a, b, merged})
		for iw in top.pos {
			s := pieces[iw][:]
			w := 0
			for r := 0; r < len(s); {
				if r + 1 < len(s) && s[r] == a && s[r + 1] == b {
					if w > 0 {
						_train_change(&st, {s[w - 1], a}, -1, iw)
						_train_change(&st, {s[w - 1], merged}, 1, iw)
					}
					s[w] = merged
					w += 1
					r += 2
					if r < len(s) {
						_train_change(&st, {b, s[r]}, -1, iw)
						_train_change(&st, {merged, s[r]}, 1, iw)
					}
				} else {
					s[w] = s[r]
					w += 1
					r += 1
				}
			}
			resize(&pieces[iw], w)
		}
		_train_flush(&st, &queue)
	}

	// 5. the tokenizer (a merge learned twice keeps its later rank, as HF)
	tok.merges = make(map[[2]int][2]int, allocator)
	for m, rank in merges {tok.merges[{m[0], m[1]}] = {rank, m[2]}}
	_bpe_index(&tok, special_tokens)
	return tok
}

// vocab.json (tokens by id, as JSON) and merges.txt (by rank) of tok, as
// Hugging Face writes them: bpe_tokenizer_new and GPT2Tokenizer read them back
bpe_save :: proc(tok: ^BPETokenizer, vocab_path, merges_path: string) -> bool {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	json_string :: proc(sb: ^strings.Builder, s: string) {
		strings.write_byte(sb, '"')
		for r in s {
			switch r {
			case '"':
				strings.write_string(sb, `\"`)
			case '\\':
				strings.write_string(sb, `\\`)
			case '\n':
				strings.write_string(sb, `\n`)
			case '\r':
				strings.write_string(sb, `\r`)
			case '\t':
				strings.write_string(sb, `\t`)
			case '\b':
				strings.write_string(sb, `\b`)
			case '\f':
				strings.write_string(sb, `\f`)
			case 0 ..< 0x20:
				fmt.sbprintf(sb, `\u%04x`, r)
			case:
				strings.write_rune(sb, r)
			}
		}
		strings.write_byte(sb, '"')
	}
	strings.write_byte(&sb, '{')
	first := true
	for token, id in tok.ids_to_tokens {
		if token not_in tok.vocab || tok.vocab[token] != id {continue} 	// an unused id
		if !first {strings.write_byte(&sb, ',')}
		first = false
		json_string(&sb, token)
		fmt.sbprintf(&sb, ":%d", id)
	}
	strings.write_byte(&sb, '}')
	if err := os.write_entire_file(vocab_path, strings.to_string(sb)); err != nil {
		fmt.printf("Error writing %s: %v\n", vocab_path, err)
		return false
	}

	ranked := make([][3]int, len(tok.merges)) // rank, a, b
	defer delete(ranked)
	k := 0
	for p, m in tok.merges {
		ranked[k] = {m[0], p[0], p[1]}
		k += 1
	}
	slice.sort_by(ranked, proc(x, y: [3]int) -> bool {return x[0] < y[0]})
	strings.builder_reset(&sb)
	strings.write_string(&sb, "#version: 0.2\n")
	for m in ranked {
		fmt.sbprintf(&sb, "%s %s\n", tok.ids_to_tokens[m[1]], tok.ids_to_tokens[m[2]])
	}
	if err := os.write_entire_file(merges_path, strings.to_string(sb)); err != nil {
		fmt.printf("Error writing %s: %v\n", merges_path, err)
		return false
	}
	return true
}
