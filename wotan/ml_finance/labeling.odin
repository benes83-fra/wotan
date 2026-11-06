package ml_finance

import ana "../analytics"
import w "../core"
import fin "../finance"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Financial labels, sample weights and stationary features
// ============================================================================
// After López de Prado (2018), "Advances in Financial Machine Learning". Bars are
// integer indices into the price series; an event is the bar a label starts at.
//   events       cusum_filter (ch. 2.5), daily_volatility (snippet 3.1)
//   labels       triple_barrier_labels (ch. 3: profit taking, stop loss, maximum
//                holding period; meta-labels when the side is given), bet sizes from
//                a meta-model's probability (ch. 10)
//   weights      label concurrency, average uniqueness, return attribution, time
//                decay, sequential bootstrap (ch. 4)
//   features     fixed-width-window fractional differencing and the smallest d that
//                passes the ADF test (ch. 5)
// The label ends (TripleBarrierLabel.end) are the `label_end` of ml.CVSplitter's
// Purged_KFold / Combinatorial_Purged.

// ----------------------------------------------------------------------------
// Events
// ----------------------------------------------------------------------------

// Symmetric CUSUM filter on the increments of x (snippet 2.4, usually on log prices):
// s+ = max(0, s+ + dx), s- = min(0, s- + dx); an event at bar i when s- < -threshold or
// s+ > threshold, which resets that sum
cusum_filter :: proc(x: []f64, threshold: f64, allocator := context.allocator) -> []int {
	events := make([dynamic]int, 0, 16, allocator)
	s_pos, s_neg := 0.0, 0.0
	for i in 1 ..< len(x) {
		d := x[i] - x[i - 1]
		s_pos = max(0, s_pos + d)
		s_neg = min(0, s_neg + d)
		if s_neg < -threshold {
			s_neg = 0
			append(&events, i)
		} else if s_pos > threshold {
			s_pos = 0
			append(&events, i)
		}
	}
	return events[:]
}

// EWM standard deviation of the simple returns (pandas
// close.pct_change().ewm(span=span).std(): adjust, bias-corrected), the usual barrier
// width. NaN where pandas gives NaN (bars 0 and 1).
daily_volatility :: proc(close: []f64, span := 100, allocator := context.allocator) -> []f64 {
	n := len(close)
	out := make([]f64, n, allocator)
	if n < 2 {
		for &v in out {v = math.nan_f64()}
		return out
	}
	ret := make([]f64, n, context.temp_allocator)
	defer delete(ret, context.temp_allocator)
	ret[0] = math.nan_f64() // null for the EWM
	for i in 1 ..< n {ret[i] = close[i] / close[i - 1] - 1}
	df := w.dataframe_new(context.temp_allocator)
	defer w.destroy_dataframe(&df)
	w.add_column(&df, w.column_from_floats("r", ret))
	df.rows = n
	col := ana.ewm_std_column(
		ana.rolling_window(&df, "r", 0, 0),
		2.0 / (f64(span) + 1),
		adjust = true,
		bias = false,
	)
	defer w.destroy_column(&col)
	for i in 0 ..< n {
		v, null := w.column_at_float(&col, i)
		out[i] = math.nan_f64() if null else v
	}
	return out
}

// ----------------------------------------------------------------------------
// Triple-barrier labels
// ----------------------------------------------------------------------------

Barrier :: enum {
	Profit_Taking,
	Stop_Loss,
	Vertical, // maximum holding period or the end of the data
}

TripleBarrierParams :: struct {
	profit_taking:      f64, // upper barrier at +profit_taking x target (0: none)
	stop_loss:          f64, // lower barrier at -stop_loss x target (0: none)
	max_holding:        int, // vertical barrier after this many bars (0: the last bar)
	min_target:         f64, // events with target <= min_target are dropped
	zero_on_vertical:   bool, // label 0 when the vertical barrier is hit first
}

TripleBarrierLabel :: struct {
	start, end: int, // the event bar and the bar of the first touch (the label's span)
	ret:        f64, // close[end] / close[start] - 1, times the side when given
	label:      int, // sign of ret (-1, 0, 1); with a side, 1 if ret > 0 else 0 (meta-label)
	barrier:    Barrier,
	target:     f64,
	side:       f64, // the primary model's side (+1 / -1), 1 when none was given
}

// Triple-barrier method (snippets 3.2 - 3.5, 3.7): from each event bar t0, the return
// path (close[t] / close[t0] - 1) x side over t0 < t <= t1 (t1 the vertical barrier)
// touches the upper barrier when it exceeds profit_taking x target[t0], the lower one
// when it falls below -stop_loss x target[t0] (strict, as the book); the label ends at
// the first touch, else at t1. Without `side` (one per event) the labels are -1 / 0 / 1
// by the sign of the return; with it they are meta-labels: 1 if the primary model's bet
// made money, else 0. Events with a NaN target, a target <= min_target, or no bar after
// them are dropped.
triple_barrier_labels :: proc(
	close: []f64,
	events: []int,
	target: []f64, // per bar (e.g. daily_volatility), read at the event bar
	params: TripleBarrierParams,
	side: []f64 = nil, // per event
	allocator := context.allocator,
) -> []TripleBarrierLabel {
	assert(side == nil || len(side) == len(events), "triple_barrier_labels: one side per event")
	n := len(close)
	out := make([dynamic]TripleBarrierLabel, 0, len(events), allocator)
	for t0, e in events {
		trgt := target[t0]
		if math.is_nan(trgt) || trgt <= params.min_target || t0 >= n - 1 {continue}
		s := side[e] if side != nil else 1
		t1 := n - 1 if params.max_holding <= 0 else min(t0 + params.max_holding, n - 1)
		lab := TripleBarrierLabel {
			start   = t0,
			end     = t1,
			barrier = .Vertical,
			target  = trgt,
			side    = s,
		}
		for t in t0 + 1 ..= t1 {
			r := (close[t] / close[t0] - 1) * s
			if params.profit_taking > 0 && r > params.profit_taking * trgt {
				lab.end, lab.barrier = t, .Profit_Taking
				break
			}
			if params.stop_loss > 0 && r < -params.stop_loss * trgt {
				lab.end, lab.barrier = t, .Stop_Loss
				break
			}
		}
		lab.ret = (close[lab.end] / close[t0] - 1) * s
		if side != nil {
			lab.label = 1 if lab.ret > 0 else 0
		} else if params.zero_on_vertical && lab.barrier == .Vertical {
			lab.label = 0
		} else {
			lab.label = 1 if lab.ret > 0 else (-1 if lab.ret < 0 else 0)
		}
		append(&out, lab)
	}
	return out[:]
}

// The label ends in sample units, for ml.CVSplitter.label_end when the samples are the
// labels (in order of start): out[i] is the index of the last label starting at or
// before labels[i].end, so the splitter's overlap test on sample indices is the overlap
// of the bar spans. (An embargo is then counted in labels, not bars.)
triple_barrier_sample_ends :: proc(
	labels: []TripleBarrierLabel,
	allocator := context.allocator,
) -> []int {
	out := make([]int, len(labels), allocator)
	j := 0
	for lab, i in labels {
		j = max(j, i)
		for j + 1 < len(labels) && labels[j + 1].start <= lab.end {j += 1}
		out[i] = j
	}
	return out
}

// Bet size from a classifier's probability p that the bet pays (two classes, snippet
// 10.1): z = (p - 1/2) / sqrt(p (1 - p)), size = side (2 Phi(z) - 1), in [-1, 1]
bet_size_from_probability :: proc(p: f64, side := 1.0) -> f64 {
	if p <= 0 || p >= 1 {return side * (1 if p >= 1 else -1)}
	z := (p - 0.5) / math.sqrt(p * (1 - p))
	return side * (2 * 0.5 * math.erfc(-z / math.SQRT_TWO) - 1)
}

// ----------------------------------------------------------------------------
// Sample weights (ch. 4)
// ----------------------------------------------------------------------------

// Number of labels alive at each bar: labels span start ..= end, inclusive (snippet 4.1)
label_concurrency :: proc(n_bars: int, labels: []TripleBarrierLabel, allocator := context.allocator) -> []int {
	c := make([]int, n_bars, allocator)
	for lab in labels {
		for t in lab.start ..= lab.end {c[t] += 1}
	}
	return c
}

// Average uniqueness of each label: the mean of 1 / concurrency over its span (snippet 4.2)
average_uniqueness :: proc(
	labels: []TripleBarrierLabel,
	concurrency: []int,
	allocator := context.allocator,
) -> []f64 {
	u := make([]f64, len(labels), allocator)
	for lab, i in labels {
		for t in lab.start ..= lab.end {u[i] += 1 / f64(concurrency[t])}
		u[i] /= f64(lab.end - lab.start + 1)
	}
	return u
}

// Return attribution (snippet 4.10): |sum over start < t <= end of log(close[t] /
// close[t-1]) / concurrency[t]|, scaled to sum to the number of labels. The book's code
// sums from the start bar, whose return happened before the label existed; here only
// the returns the label earns.
return_attribution_weights :: proc(
	close: []f64,
	labels: []TripleBarrierLabel,
	concurrency: []int,
	allocator := context.allocator,
) -> []f64 {
	wts := make([]f64, len(labels), allocator)
	total := 0.0
	for lab, i in labels {
		s := 0.0
		for t in lab.start + 1 ..= lab.end {s += math.ln(close[t] / close[t - 1]) / f64(concurrency[t])}
		wts[i] = abs(s)
		total += wts[i]
	}
	if total > 0 {for &v in wts {v *= f64(len(labels)) / total}}
	return wts
}

// Time decay of weights (snippet 4.11) along the cumulative uniqueness x (labels in time
// order): the newest label gets 1, the oldest c (c in [0, 1]); c < 0 zeroes the oldest
// fraction -c of the cumulative uniqueness. Multiplies `weights` in place.
time_decay_weights :: proc(weights: []f64, uniqueness: []f64, c := 1.0) {
	total := 0.0
	for u in uniqueness {total += u}
	slope := (1 - c) / total if c >= 0 else 1 / ((c + 1) * total)
	constant := 1 - slope * total
	x := 0.0
	for u, i in uniqueness {
		x += u
		weights[i] *= max(constant + slope * x, 0)
	}
}

// Sequential bootstrap probabilities (snippet 4.5): for each label i, the mean over its
// span of 1 / (number of drawn labels alive + 1), normalised; `drawn_count` holds the
// number of drawn labels alive at each bar
sequential_bootstrap_probabilities :: proc(
	labels: []TripleBarrierLabel,
	drawn_count: []int,
	allocator := context.allocator,
) -> []f64 {
	p := make([]f64, len(labels), allocator)
	total := 0.0
	for lab, i in labels {
		for t in lab.start ..= lab.end {p[i] += 1 / f64(drawn_count[t] + 1)}
		p[i] /= f64(lab.end - lab.start + 1)
		total += p[i]
	}
	for &v in p {v /= total}
	return p
}

// Sequential bootstrap (snippets 4.5, 4.6): n_draws labels (default as many as labels),
// each drawn with sequential_bootstrap_probabilities given the draws so far, so that
// overlapping labels are drawn less often than by the standard bootstrap. Uses
// context.random_generator.
sequential_bootstrap :: proc(
	labels: []TripleBarrierLabel,
	n_bars: int,
	n_draws := 0,
	allocator := context.allocator,
) -> []int {
	m := n_draws if n_draws > 0 else len(labels)
	out := make([]int, m, allocator)
	count := make([]int, n_bars, context.temp_allocator)
	defer delete(count, context.temp_allocator)
	for k in 0 ..< m {
		p := sequential_bootstrap_probabilities(labels, count, context.temp_allocator)
		defer delete(p, context.temp_allocator)
		u := rand.float64()
		i := 0
		for acc := p[0]; acc < u && i < len(p) - 1; {i += 1; acc += p[i]}
		out[k] = i
		for t in labels[i].start ..= labels[i].end {count[t] += 1}
	}
	return out
}

// ----------------------------------------------------------------------------
// Fractional differencing (ch. 5)
// ----------------------------------------------------------------------------

// Weights of (1 - B)^d: w_0 = 1, w_k = -w_{k-1} (d - k + 1) / k, up to the last with
// |w_k| >= threshold (snippet 5.3, fixed-width window)
frac_diff_weights :: proc(d: f64, threshold := 1e-5, allocator := context.allocator) -> []f64 {
	assert(threshold > 0, "frac_diff_weights: threshold must be > 0")
	wts := make([dynamic]f64, 0, 64, allocator)
	append(&wts, 1)
	for k := 1;; k += 1 {
		v := -wts[k - 1] * (d - f64(k) + 1) / f64(k)
		if abs(v) < threshold || k > 100_000 {break}
		append(&wts, v)
	}
	return wts[:]
}

// Fixed-width-window fractionally differenced series: y_t = sum_k w_k x_{t-k} with the
// weights of frac_diff_weights; NaN for the first len(weights) - 1 bars (and where the
// window holds a NaN)
frac_diff_ffd :: proc(
	x: []f64,
	d: f64,
	threshold := 1e-5,
	allocator := context.allocator,
) -> []f64 {
	wts := frac_diff_weights(d, threshold, context.temp_allocator)
	defer delete(wts, context.temp_allocator)
	K := len(wts) - 1
	out := make([]f64, len(x), allocator)
	for t in 0 ..< len(x) {
		if t < K {out[t] = math.nan_f64(); continue}
		s := 0.0
		for wk, k in wts {s += wk * x[t - k]}
		out[t] = s
	}
	return out
}

FracDiffResult :: struct {
	d:           f64, // the smallest d on the grid whose FFD series passes the ADF test
	adf_stat:    f64,
	p_value:     f64,
	correlation: f64, // with the original series, over the bars both are defined
	found:       bool, // false: no d on the grid passed (d is the grid's last)
}

// Smallest d in 0, step, 2 step, ... <= max_d for which the FFD series rejects a unit
// root at `significance` (ADF with a constant and `adf_lags` fixed lags, as the book's
// adfuller(maxlag=1, regression='c', autolag=None)): the most memory a stationary
// transform can keep
frac_diff_min_d :: proc(
	x: []f64,
	step := 0.05,
	max_d := 1.0,
	threshold := 1e-4,
	significance := 0.05,
	adf_lags := 1,
) -> (
	res: FracDiffResult,
) {
	steps := int(math.round(max_d / step))
	for s in 0 ..= steps {
		d := f64(s) * step
		y := frac_diff_ffd(x, d, threshold, context.temp_allocator)
		defer delete(y, context.temp_allocator)
		first := 0
		for first < len(y) && math.is_nan(y[first]) {first += 1}
		if len(y) - first < 20 {continue}
		stat, p, _, _, _, _, _ := ana.adf_test(y[first:], adf_lags, .Constant, .Fixed, context.temp_allocator)
		res.d, res.adf_stat, res.p_value = d, stat, p
		res.correlation = fin.correlation(x[first:], y[first:])
		if p < significance {res.found = true; return}
	}
	return
}
