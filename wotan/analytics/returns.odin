package analytics

import "core:math"
import "core:mem"

// Returns from prices.
//
// `prices` holds T observations of n_series series, row-major
// [t * n_series + series] (n_series = 1 for a single price path, n_series = the
// number of tickers for an aligned panel such as net.read_yahoo_panel's
// adj_close). The result has T - 1 rows in the same layout: row t is the return
// from t to t + 1. Prices must be > 0 for log returns and != 0 for simple returns;
// a non-positive price gives NaN or an infinite return, it is not skipped.

// ln(p[t + 1] / p[t]) per series; empty when T < 2. The caller owns the result.
log_returns :: proc(
	prices: []f64,
	n_series := 1,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	T := len(prices) / n_series
	if T < 2 {return make([]f64, 0, allocator)}
	out := make([]f64, (T - 1) * n_series, allocator)
	for k in 0 ..< len(out) {
		out[k] = math.ln(prices[k + n_series] / prices[k])
	}
	return out
}

// p[t + 1] / p[t] - 1 per series; empty when T < 2. The caller owns the result.
simple_returns :: proc(
	prices: []f64,
	n_series := 1,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	T := len(prices) / n_series
	if T < 2 {return make([]f64, 0, allocator)}
	out := make([]f64, (T - 1) * n_series, allocator)
	for k in 0 ..< len(out) {
		out[k] = prices[k + n_series] / prices[k] - 1
	}
	return out
}
