package behavioral_finance

import "core:math"

// ============================================================================
// The prospect theory value of a stock (Barberis, Mukherjee & Wang 2016)
// ============================================================================
// Investors who think of a stock as the distribution of its past returns and
// evaluate that distribution with cumulative prospect theory tilt toward stocks with
// a high value, which then earn lower returns. BMW's TK: the CPT value (TK92
// parameters) of the stock's monthly returns in excess of the market over the
// previous 60 months, each with probability 1/60 (their eq. 8: cpt_value). TK
// predicts the next month's return with a negative sign: the lowest-TK decile beats
// the highest.

// TK at every period t from the `window` returns before t (t - window .. t - 1) in
// excess of the benchmark (e.g. the market; nil = raw returns); NaN for t < window and
// while the window holds a NaN return. TK[t] predicts the return of period t.
tk_value :: proc(
	returns: []f64,
	benchmark: []f64 = nil,
	window := 60,
	params := TK92_PARAMS,
	allocator := context.allocator,
) -> []f64 {
	assert(window > 0, "tk_value: window must be positive")
	assert(
		benchmark == nil || len(benchmark) == len(returns),
		"tk_value: one benchmark return per return",
	)
	T := len(returns)
	tk := make([]f64, T, allocator)
	x := make([]f64, window, context.temp_allocator)
	defer delete(x, context.temp_allocator)
	for t in 0 ..< T {
		tk[t] = math.nan_f64()
		if t < window {continue}
		valid := true
		for k in 0 ..< window {
			s := t - window + k
			x[k] = returns[s] - (benchmark[s] if benchmark != nil else 0)
			valid &&= !math.is_nan(x[k])
		}
		if valid {tk[t] = cpt_value(x, params)}
	}
	return tk
}
