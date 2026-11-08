package behavioral_finance

import "core:math"

// ============================================================================
// Under- and overreaction: momentum, long-term reversal, frog in the pan
// ============================================================================
//   Jegadeesh & Titman (1993): each month buy the past winners and sell the past
//     losers (deciles of the J-month return), hold K months with overlapping
//     portfolios: past_return + long_short_portfolio(fraction 0.1) +
//     overlapping_strategy_returns (anchoring.odin). The common 12-1 variant skips the
//     most recent month (past_return lookback 11, skip 1).
//   De Bondt & Thaler (1985): the past 36 months' losers beat the winners over the
//     next 36 months. Their ranking variable is the cumulative (summed) market-
//     adjusted return: past_return(lookback 36, market, kind .Sum).
//   Da, Gurun & Warachka (2014), frog in the pan: investors underreact more to
//     information that arrives continuously in small amounts. Information
//     discreteness over the formation period,
//       ID = sgn(PRET) (%neg - %pos)                                     (DGW eq. 1)
//     %pos / %neg the shares of days with positive / negative returns, PRET the
//     formation-period return (the past twelve months skipping the most recent: months
//     t-12 .. t-2); ID_Z = sgn(PRET) (%neg - %pos) / (%neg + %pos) ignores zero-return
//     days (eq. 3). Low ID (continuous) means stronger momentum: sort sequentially on
//     PRET quintiles then ID quintiles, momentum (winner - loser) within each ID
//     quintile (double_sort_portfolio). DGW 1927-2007: 5.94% over six months for
//     continuous, -2.07% for discrete information (Table 2A).

Past_Return_Kind :: enum {
	Compound, // prod (1 + r) - 1
	Sum, // sum r (De Bondt & Thaler's cumulative abnormal return)
}

// Formation-period returns [asset][period]: value t covers periods
// t - skip - lookback + 1 .. t - skip of returns [asset][period] (market: subtracted
// period by period first, nil = raw returns). NaN when the window starts before 0 or
// holds a non-finite return.
past_return :: proc(
	returns: [][]f64,
	lookback: int,
	skip := 0,
	market: []f64 = nil,
	kind := Past_Return_Kind.Compound,
	allocator := context.allocator,
) -> [][]f64 {
	assert(lookback > 0 && skip >= 0, "past_return: lookback > 0, skip >= 0")
	out := make([][]f64, len(returns), allocator)
	for r, i in returns {
		assert(market == nil || len(market) == len(r), "past_return: one market return per period")
		out[i] = make([]f64, len(r), allocator)
		for t in 0 ..< len(r) {
			a := t - skip - lookback + 1
			g := math.nan_f64()
			if a >= 0 {
				g = 1 if kind == .Compound else 0
				for s in a ..= t - skip {
					x := r[s] - (market[s] if market != nil else 0)
					if !_finite(x) {g = math.nan_f64(); break}
					if kind == .Compound {g *= 1 + x} else {g += x}
				}
				if kind == .Compound {g -= 1}
			}
			out[i][t] = g
		}
	}
	return out
}

Discreteness_Kind :: enum {
	ID, // sgn(PRET) (%neg - %pos), shares of all days
	ID_Z, // sgn(PRET) (%neg - %pos) / (%neg + %pos), zero-return days left out
}

// ID of one formation period: its daily returns r and its return pret (sgn(0) = 0).
// NaN when r is empty or a return or pret is not finite; ID_Z = 0 when every day is 0.
information_discreteness_value :: proc(r: []f64, pret: f64, kind := Discreteness_Kind.ID) -> f64 {
	if len(r) == 0 || !_finite(pret) {return math.nan_f64()}
	pos, neg := 0, 0
	for v in r {
		if !_finite(v) {return math.nan_f64()}
		if v > 0 {pos += 1} else if v < 0 {neg += 1}
	}
	sg := f64(int(pret > 0) - int(pret < 0))
	switch kind {
	case .ID:
		return sg * f64(neg - pos) / f64(len(r))
	case .ID_Z:
		return 0 if pos + neg == 0 else sg * f64(neg - pos) / f64(pos + neg)
	}
	unreachable()
}

// ID and PRET at the end of every month from daily returns [asset][day]:
// months[k] .. months[k+1] - 1 are the days of month k (len(months) = months + 1);
// month k's formation period is months k - skip - lookback + 1 .. k - skip (DGW:
// lookback 11, skip 1), PRET its compounded daily return. NaN when the period starts
// before month 0 or holds a non-finite return. Result [asset][month].
information_discreteness :: proc(
	daily: [][]f64,
	months: []int,
	lookback := 11,
	skip := 1,
	kind := Discreteness_Kind.ID,
	allocator := context.allocator,
) -> (
	id, pret: [][]f64,
) {
	_check_months(daily, months)
	assert(lookback > 0 && skip >= 0, "information_discreteness: lookback > 0, skip >= 0")
	M := len(months) - 1
	id = make([][]f64, len(daily), allocator)
	pret = make([][]f64, len(daily), allocator)
	for r, i in daily {
		id[i] = make([]f64, M, allocator)
		pret[i] = make([]f64, M, allocator)
		for k in 0 ..< M {
			a := k - skip - lookback + 1
			id[i][k], pret[i][k] = math.nan_f64(), math.nan_f64()
			if a < 0 {continue}
			days := r[months[a]:months[k - skip + 1]]
			g := 1.0
			for v in days {g *= 1 + v}
			if !_finite(g) {continue}
			pret[i][k] = g - 1
			id[i][k] = information_discreteness_value(days, g - 1, kind)
		}
	}
	return
}

// Quantile groups 0 .. n_groups - 1 by ascending value: the finite values ranked
// 0 .. n - 1 (ties by index), rank r in group floor(r n_groups / n); -1 for the rest
quantile_groups :: proc(values: []f64, n_groups: int, out: []int) {
	assert(n_groups > 0 && len(out) == len(values), "quantile_groups: one group per value")
	for &g in out {g = -1}
	idx := _ascending_order(values)
	defer delete(idx)
	n := len(idx)
	for i, r in idx {out[i] = r * n_groups / n}
}

Double_Sort :: enum {
	Sequential, // second groups within each first group (DGW Table 2A)
	Independent, // both on all assets (Table 2B)
}

// Groups g1 (on first) and g2 (on second) of the assets where both are finite (-1
// elsewhere)
double_sort :: proc(
	first, second: []f64,
	n1, n2: int,
	g1, g2: []int,
	mode := Double_Sort.Sequential,
) {
	n := len(first)
	assert(
		len(second) == n && len(g1) == n && len(g2) == n,
		"double_sort: one value and group per asset",
	)
	a := make([]f64, n, context.temp_allocator)
	b := make([]f64, n, context.temp_allocator)
	defer {delete(a, context.temp_allocator); delete(b, context.temp_allocator)}
	for i in 0 ..< n {
		ok := _finite(first[i]) && _finite(second[i])
		a[i] = first[i] if ok else math.nan_f64()
		b[i] = second[i] if ok else math.nan_f64()
	}
	quantile_groups(a, n1, g1)
	if mode == .Independent {
		quantile_groups(b, n2, g2)
		return
	}
	c := make([]f64, n, context.temp_allocator)
	h := make([]int, n, context.temp_allocator)
	defer {delete(c, context.temp_allocator); delete(h, context.temp_allocator)}
	for &g in g2 {g = -1}
	for q in 0 ..< n1 {
		for i in 0 ..< n {c[i] = b[i] if g1[i] == q else math.nan_f64()}
		quantile_groups(c, n2, h)
		for i in 0 ..< n {if g1[i] == q {g2[i] = h[i]}}
	}
}

// Within second-group g: equally weighted long the top first-group and short the
// bottom one (weights +1 / -1 in total; zero on a side without assets). DGW:
// first = PRET, second = ID, n1 = n2 = 5, g = 0 continuous .. 4 discrete.
double_sort_portfolio :: proc(
	first, second: []f64,
	g: int,
	n1 := 5,
	n2 := 5,
	mode := Double_Sort.Sequential,
	allocator := context.allocator,
) -> []f64 {
	assert(g >= 0 && g < n2 && n1 >= 2, "double_sort_portfolio: group out of range")
	n := len(first)
	g1 := make([]int, n, context.temp_allocator)
	g2 := make([]int, n, context.temp_allocator)
	defer {delete(g1, context.temp_allocator); delete(g2, context.temp_allocator)}
	double_sort(first, second, n1, n2, g1, g2, mode)
	nl, ns := 0, 0
	for i in 0 ..< n {
		if g2[i] != g {continue}
		if g1[i] == n1 - 1 {nl += 1} else if g1[i] == 0 {ns += 1}
	}
	w := make([]f64, n, allocator)
	for i in 0 ..< n {
		if g2[i] != g {continue}
		if g1[i] == n1 - 1 {w[i] = 1 / f64(nl)} else if g1[i] == 0 {w[i] = -1 / f64(ns)}
	}
	return w
}
