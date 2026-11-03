package tokenizers

import t "../../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// Time-series tokenizer: mean scaling + uniform bins (Chronos)
// ============================================================================
// As Chronos' MeanScaleUniformBins (Ansari et al. 2024, "Chronos: Learning the
// Language of Time Series"): a series is divided by its scale s = the mean
// |value| of its context (missing values, NaN, left out; s = 1 if not > 0),
// and each scaled value becomes the token of its bin among
// n_tokens - n_special - 1 bins whose centers are spaced evenly on
// [low, high] (values outside fall into the outer bins):
//   token = n_special + 1 + bin,   value = center[bin] · s
// Missing values become pad_id (attention mask 0); eos_id can end a sequence.
// A language model over these tokens (nn.GPT) forecasts by sampling the next
// tokens: each sample path is one draw of the future (ml_finance/token_forecast.odin).

TimeSeriesTokenizer :: struct {
	n_tokens:       int, // vocabulary size, special tokens included
	n_special:      int, // ids 0 ..< n_special are special
	pad_id:         int,
	eos_id:         int,
	low, high:      f64, // range of the bin centers (scaled values)
	context_length: int, // longer contexts keep their last context_length values
	use_eos:        bool, // context and labels end with eos_id (encoder-decoder models)
	centers:        []f64, // n_tokens - n_special - 1 bin centers
	edges:          []f64, // midpoints between neighboring centers
	allocator:      mem.Allocator,
}

// Chronos' defaults (chronos-t5: 4096 tokens, centers on [-15, 15], 512
// context values), but without EOS: chronos-t5 (encoder-decoder) appends it,
// a decoder-only model such as nn.GPT continues the context directly
ts_tokenizer_new :: proc(
	n_tokens := 4096,
	low := -15.0,
	high := 15.0,
	context_length := 512,
	use_eos := false,
	n_special := 2,
	pad_id := 0,
	eos_id := 1,
	allocator := context.allocator,
) -> TimeSeriesTokenizer {
	n_bins := n_tokens - n_special - 1
	assert(n_bins >= 2 && high > low, "ts_tokenizer_new: need at least 2 bins on low < high")
	tok := TimeSeriesTokenizer {
		n_tokens       = n_tokens,
		n_special      = n_special,
		pad_id         = pad_id,
		eos_id         = eos_id,
		low            = low,
		high           = high,
		context_length = context_length,
		use_eos        = use_eos,
		allocator      = allocator,
	}
	tok.centers = make([]f64, n_bins, allocator)
	tok.edges = make([]f64, n_bins - 1, allocator)
	for &c, i in tok.centers {c = low + (high - low) * f64(i) / f64(n_bins - 1)} 	// as torch.linspace
	for &e, i in tok.edges {e = 0.5 * (tok.centers[i] + tok.centers[i + 1])}
	return tok
}

ts_tokenizer_free :: proc(tok: ^TimeSeriesTokenizer) {
	delete(tok.centers, tok.allocator)
	delete(tok.edges, tok.allocator)
}

// Mean |value| of the non-missing values, 1 if that is not > 0
ts_scale :: proc(values: []f64) -> f64 {
	sum, n := 0.0, 0
	for v in values {
		if math.is_nan(v) {continue}
		sum += abs(v)
		n += 1
	}
	s := sum / f64(n) if n > 0 else 0
	return s if s > 0 else 1
}

// Token of one value at scale s (NaN: pad_id)
ts_token :: proc(tok: ^TimeSeriesTokenizer, value, scale: f64) -> int {
	if math.is_nan(value) {return tok.pad_id}
	x := value / scale
	if x < -1e20 {return tok.n_special} 	// below Chronos' outer boundary (-inf)
	// bin = how many edges are ≤ x (torch.bucketize(right = True))
	lo, hi := 0, len(tok.edges)
	for lo < hi {
		mid := (lo + hi) / 2
		if tok.edges[mid] <= x {lo = mid + 1} else {hi = mid}
	}
	return tok.n_special + 1 + lo
}

// Tokens and attention mask of values at scale s, eos_id appended if use_eos
ts_encode :: proc(
	tok: ^TimeSeriesTokenizer,
	values: []f64,
	scale: f64,
	allocator := context.allocator,
) -> (
	ids: []int,
	mask: []int,
) {
	n := len(values) + (1 if tok.use_eos else 0)
	ids = make([]int, n, allocator)
	mask = make([]int, n, allocator)
	for v, i in values {
		ids[i] = ts_token(tok, v, scale)
		mask[i] = 0 if math.is_nan(v) else 1
	}
	if tok.use_eos {ids[n - 1], mask[n - 1] = tok.eos_id, 1}
	return
}

// A context: its last context_length values, their scale, their tokens
ts_encode_context :: proc(
	tok: ^TimeSeriesTokenizer,
	context_values: []f64,
	allocator := context.allocator,
) -> (
	ids: []int,
	mask: []int,
	scale: f64,
) {
	c := context_values[max(0, len(context_values) - tok.context_length):]
	scale = ts_scale(c)
	ids, mask = ts_encode(tok, c, scale, allocator)
	return
}

// Values of tokens at scale s: the bin centers (special tokens: the lowest bin, as Chronos)
ts_decode :: proc(
	tok: ^TimeSeriesTokenizer,
	ids: []int,
	scale: f64,
	allocator := context.allocator,
) -> []f64 {
	out := make([]f64, len(ids), allocator)
	for id, i in ids {
		bin := clamp(id - tok.n_special - 1, 0, len(tok.centers) - 1)
		out[i] = tok.centers[bin] * scale
	}
	return out
}

// Contexts of different lengths as one batch [len(series), L] of tokens and
// attention mask (L = the longest context after truncation; shorter ones
// left-padded with pad_id, as Chronos) and each series' scale
ts_encode_batch_to_tensors :: proc(
	tok: ^TimeSeriesTokenizer,
	series: [][]f64,
	allocator := context.allocator,
) -> (
	input_ids: ^t.Tensor,
	attention_mask: ^t.Tensor,
	scales: []f64,
) {
	B := len(series)
	assert(B > 0, "ts_encode_batch_to_tensors: empty batch")
	L := 0
	for s in series {L = max(L, min(len(s), tok.context_length) + (1 if tok.use_eos else 0))}
	ids := make([]int, B * L, allocator)
	mask := make([]int, B * L, allocator)
	defer delete(ids, allocator)
	defer delete(mask, allocator)
	scales = make([]f64, B, allocator)
	for s, b in series {
		row_ids, row_mask, scale := ts_encode_context(tok, s, allocator)
		scales[b] = scale
		off := b * L + L - len(row_ids) // left padding
		for i in b * L ..< off {ids[i] = tok.pad_id}
		copy(ids[off:(b + 1) * L], row_ids)
		copy(mask[off:(b + 1) * L], row_mask)
		delete(row_ids, allocator)
		delete(row_mask, allocator)
	}
	segments: ^t.Tensor
	input_ids, segments, attention_mask = _to_tensors(ids, nil, mask, B, allocator)
	t.tensor_free(segments)
	return
}
