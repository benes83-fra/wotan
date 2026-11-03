package tokenizers

import l "../../linalg"
import t "../../tensor"
import "core:mem"
import "core:strings"

// ============================================================================
// Shared by the BPE and WordPiece tokenizers
// ============================================================================

// Fixed-length model input from token ids a, or the pair a, b (b != nil):
//   one text:  [CLS] a [SEP]          (add_special; else a alone)
//   pair:      [CLS] a [SEP] b [SEP]  segment ids 0 … 0 1 … 1
// truncated to max_len (one text: its end; a pair as HF's longest_first: the
// last token of the longer one, of b on a tie, until both fit), then padded
// with pad_id (segment 0). The attention mask is 1 on the real tokens.
_pack :: proc(
	a, b: []int,
	max_len, cls_id, sep_id, pad_id: int,
	add_special: bool,
	allocator: mem.Allocator,
) -> (
	ids: []int,
	mask: []int,
	segments: []int,
) {
	ids = make([]int, max_len, allocator)
	mask = make([]int, max_len, allocator)
	segments = make([]int, max_len, allocator)
	n := 0
	put :: proc(ids, mask, segments: []int, n: ^int, id, segment: int) {
		ids[n^], mask[n^], segments[n^] = id, 1, segment
		n^ += 1
	}
	switch {
	case b != nil:
		assert(
			add_special && max_len >= 3,
			"tokenizers: a sentence pair needs [CLS] / [SEP] and max_len ≥ 3",
		)
		la, lb := len(a), len(b)
		for la + lb > max_len - 3 {
			if la > lb {la -= 1} else {lb -= 1}
		}
		put(ids, mask, segments, &n, cls_id, 0)
		for id in a[:la] {put(ids, mask, segments, &n, id, 0)}
		put(ids, mask, segments, &n, sep_id, 0)
		for id in b[:lb] {put(ids, mask, segments, &n, id, 1)}
		put(ids, mask, segments, &n, sep_id, 1)
	case add_special && max_len >= 2:
		put(ids, mask, segments, &n, cls_id, 0)
		for id in a[:min(len(a), max_len - 2)] {put(ids, mask, segments, &n, id, 0)}
		put(ids, mask, segments, &n, sep_id, 0)
	case:
		for id in a[:min(len(a), max_len)] {put(ids, mask, segments, &n, id, 0)}
	}
	for i in n ..< max_len {ids[i] = pad_id}
	return
}

// [batch, L] tensors (L = len(ids) / batch) of row-major ids, segment ids
// (nil: all 0) and attention mask (nil: no mask tensor)
_to_tensors :: proc(
	ids, segments, mask: []int,
	batch: int,
	allocator: mem.Allocator,
) -> (
	input_ids_tensor: ^t.Tensor,
	segment_ids_tensor: ^t.Tensor,
	attention_mask_tensor: ^t.Tensor,
) {
	L := len(ids) / batch
	tensor :: proc(v: []int, batch, L: int, allocator: mem.Allocator) -> ^t.Tensor {
		m := l.matrix_new(f64, 1, batch * L, allocator)
		for x, i in v {m.data[i] = f64(x)}
		x := t.tensor_new(m, false, allocator)
		x.shape = {batch, L, 1, 1}
		return x
	}
	input_ids_tensor = tensor(ids, batch, L, allocator)
	segment_ids_tensor = tensor(segments, batch, L, allocator)
	if mask != nil {attention_mask_tensor = tensor(mask, batch, L, allocator)}
	return
}

// A batch of n inputs packed by pack(i) (max_len each) as [n, L] tensors of
// ids, segment ids and attention mask: L is the longest input's length
// (pad_to_longest) or max_len
_batch_to_tensors :: proc(
	n, max_len: int,
	pad_to_longest: bool,
	data: rawptr,
	pack: proc(i: int, data: rawptr, allocator: mem.Allocator) -> (ids, mask, segments: []int),
	allocator: mem.Allocator,
) -> (
	input_ids: ^t.Tensor,
	segment_ids: ^t.Tensor,
	attention_mask: ^t.Tensor,
) {
	assert(n > 0, "tokenizers: empty batch")
	rows := make([][3][]int, n, allocator)
	defer delete(rows, allocator)
	L := 0 if pad_to_longest else max_len
	for &row, i in rows {
		row[0], row[1], row[2] = pack(i, data, allocator)
		if pad_to_longest {
			real := 0
			for m in row[1] {real += m}
			L = max(L, real)
		}
	}
	L = max(L, 1)
	flat: [3][]int
	for &f in flat {f = make([]int, n * L, allocator)}
	defer for f in flat {delete(f, allocator)}
	for row, i in rows {
		for k in 0 ..< 3 {
			copy(flat[k][i * L:(i + 1) * L], row[k][:L])
			delete(row[k], allocator)
		}
	}
	return _to_tensors(flat[0], flat[2], flat[1], n, allocator)
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
	first: [256]bool // the bytes a special token can start with
	for s in specials {if len(s) > 0 {first[s[0]] = true}}
	start, i := 0, 0
	for i < len(text) {
		best, best_id := 0, 0
		if first[text[i]] {
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
