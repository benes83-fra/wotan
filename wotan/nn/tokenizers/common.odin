package tokenizers

import l "../../linalg"
import t "../../tensor"
import "core:mem"
import "core:strings"

// ============================================================================
// Shared by the BPE and WordPiece tokenizers
// ============================================================================

// Fixed-length model input from token ids: optionally [CLS] ids [SEP],
// truncated to max_len (the [SEP] kept), then padded with pad_id; the
// attention mask is 1 on the real tokens
_pack :: proc(
	raw: []int,
	max_len, cls_id, sep_id, pad_id: int,
	add_special: bool,
	allocator: mem.Allocator,
) -> (
	ids: []int,
	mask: []int,
) {
	ids = make([]int, max_len, allocator)
	mask = make([]int, max_len, allocator)
	n := 0
	put :: proc(ids, mask: []int, n: ^int, id: int) {
		ids[n^], mask[n^] = id, 1
		n^ += 1
	}
	if add_special && max_len >= 2 {
		put(ids, mask, &n, cls_id)
		for id in raw[:min(len(raw), max_len - 2)] {put(ids, mask, &n, id)}
		put(ids, mask, &n, sep_id)
	} else {
		for id in raw[:min(len(raw), max_len)] {put(ids, mask, &n, id)}
	}
	for i in n ..< max_len {ids[i] = pad_id}
	return
}

// ids as model input [1, len(ids)], with segment ids 0 (one sentence)
_to_tensors :: proc(
	ids: []int,
	allocator: mem.Allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
) {
	n := len(ids)
	ids_data := l.matrix_new(f64, 1, n, allocator)
	for id, i in ids {ids_data.data[i] = f64(id)}
	input_ids_tensor = t.tensor_new(ids_data, false, allocator)
	input_ids_tensor.shape = {1, n, 1, 1}
	segment_ids_tensor = t.tensor_new(l.matrix_new(f64, 1, n, allocator), false, allocator)
	segment_ids_tensor.shape = {1, n, 1, 1}
	return
}

// Splits text at the special tokens (they are never broken up, as in
// Hugging Face's tokenizers): calls piece for each stretch of ordinary text
// and appends the special tokens' ids to out. At a position the longest
// special token wins.
_split_special :: proc(
	text: string,
	specials: map[string]int,
	out: ^[dynamic]int,
	data: rawptr,
	piece: proc(text: string, out: ^[dynamic]int, data: rawptr),
) {
	start, i := 0, 0
	for i < len(text) {
		best, best_id := 0, 0
		if text[i] == '<' || text[i] == '[' {
			for s, id in specials {
				if len(s) > best && strings.has_prefix(text[i:], s) {best, best_id = len(s), id}
			}
		}
		if best == 0 {
			i += 1
			continue
		}
		if i > start {piece(text[start:i], out, data)}
		append(out, best_id)
		i += best
		start = i
	}
	if start < len(text) {piece(text[start:], out, data)}
}
