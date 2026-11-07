package behavioral_finance

import "core:math"
import "core:slice"

// ============================================================================
// The disposition effect
// ============================================================================
// Investors sell winners too early and ride losers too long (Shefrin & Statman 1985),
// as prospect theory with the purchase price as the reference point predicts.
//   Odean (1998): the proportion of gains realized (PGR) against the proportion of
//     losses realized (PLR), from trading records
//   Grinblatt & Han (2005): the aggregate unrealized gain of a stock's holders, the
//     capital gains overhang, predicts its return

// ----------------------------------------------------------------------------
// Grinblatt & Han (2005): reference price and capital gains overhang
// ----------------------------------------------------------------------------
// The aggregate cost basis at t from the prices and turnovers (volume / shares
// outstanding, per period) of the `window` periods before t (their eq. 25):
//   R_t = (1/k) sum_{n=1}^{window} V_{t-n} prod_{tau=1}^{n-1} (1 - V_{t-n+tau}) P_{t-n}
// k makes the weights sum to one: the probability that a share last traded at t - n
// (traded then, not since). NaN for t < window, or when the window holds a price that
// is not positive or a turnover outside [0, 1] (or NaN), or all weights are zero.
// Grinblatt & Han: weekly data, window 260 (five years).
reference_price :: proc(
	prices, turnover: []f64,
	window := 260,
	allocator := context.allocator,
) -> []f64 {
	assert(len(prices) == len(turnover) && window > 0, "reference_price: one turnover per price")
	T := len(prices)
	R := make([]f64, T, allocator)
	for t in 0 ..< T {
		R[t] = math.nan_f64()
		if t < window {continue}
		num, k, surv := 0.0, 0.0, 1.0
		valid := true
		for n in 1 ..= window {
			p, v := prices[t - n], turnover[t - n]
			if !(p > 0) || !(v >= 0 && v <= 1) || math.is_inf(p) {
				valid = false
				break
			}
			wn := v * surv // traded at t - n, not between t - n + 1 and t - 1
			num += wn * p
			k += wn
			surv *= 1 - v
		}
		if valid && k > 0 {R[t] = num / k}
	}
	return R
}

// The capital gains overhang g_s = (P_{s-1} - R_s) / P_{s-1}: the percentage
// unrealized gain of the stock's holders, from data up to s - 1 (the price lagged one
// period against bid-ask bounce). It is Grinblatt & Han's g_{t-1} for t = s + 1: g_s
// predicts the return over period s + 1, P_{s+1} / P_s - 1.
capital_gains_overhang :: proc(
	prices, turnover: []f64,
	window := 260,
	allocator := context.allocator,
) -> []f64 {
	g := reference_price(prices, turnover, window, allocator)
	for &v, t in g {
		if math.is_nan(v) {continue}
		p := prices[t - 1]
		v = (p - v) / p
	}
	return g
}

// ----------------------------------------------------------------------------
// Odean (1998): PGR and PLR
// ----------------------------------------------------------------------------

Trade :: struct {
	account:  int,
	asset:    int,
	day:      int,
	quantity: f64, // > 0 buy, < 0 sell (long positions only)
	price:    f64,
}

Disposition_Counts :: struct {
	realized_gains:  int,
	realized_losses: int,
	paper_gains:     int,
	paper_losses:    int,
	pgr:             f64, // realized gains / (realized gains + paper gains)
	plr:             f64,
	difference:      f64, // PGR - PLR: > 0 is the disposition effect
	std_error:       f64, // sqrt(PGR (1 - PGR) / (RG + PG) + PLR (1 - PLR) / (RL + PL))
	t_stat:          f64,
}

// Counts, for every account and every day on which the account sells something
// (sale_days_only, Odean's definition; otherwise every day it holds a position), each
// position held at the start of that day: sold that day above / below its average
// purchase price is a realized gain / loss (the sale price quantity-weighted over the
// day's sales of the asset), not sold with the day's close (prices[asset][day]) above
// / below it a paper gain / loss. Equal prices count as neither. Positions bought the
// same day do not count. Purchases update the average purchase price; sales keep it.
odean_disposition :: proc(
	trades: []Trade,
	prices: [][]f64, // [asset][day] closing prices
	sale_days_only := true,
) -> Disposition_Counts {
	n_days := len(prices[0]) if len(prices) > 0 else 0
	n_assets := len(prices)
	order := make([]int, len(trades), context.temp_allocator)
	defer delete(order, context.temp_allocator)
	for i in 0 ..< len(trades) {
		order[i] = i
		assert(
			trades[i].day >= 0 && trades[i].day < n_days,
			"odean_disposition: trade day outside the prices",
		)
	}
	context.user_ptr = raw_data(trades)
	slice.sort_by(order, proc(a, b: int) -> bool {
		tr := ([^]Trade)(context.user_ptr)
		x, y := tr[a], tr[b]
		if x.account != y.account {return x.account < y.account}
		if x.day != y.day {return x.day < y.day}
		return a < b
	})
	qty := make([]f64, n_assets, context.temp_allocator)
	cost := make([]f64, n_assets, context.temp_allocator)
	sold_q := make([]f64, n_assets, context.temp_allocator) // today's sales
	sold_v := make([]f64, n_assets, context.temp_allocator)
	defer {
		delete(qty, context.temp_allocator)
		delete(cost, context.temp_allocator)
		delete(sold_q, context.temp_allocator)
		delete(sold_v, context.temp_allocator)
	}
	c := Disposition_Counts{}
	i := 0
	for i < len(order) {
		acct := trades[order[i]].account
		for &q in qty {q = 0}
		// the account's days from its first trade to the last price
		first := trades[order[i]].day
		j := i
		for j < len(order) && trades[order[j]].account == acct {j += 1}
		k := i
		for d in first ..< n_days {
			// today's trades: [k, e)
			e := k
			for e < j && trades[order[e]].day == d {e += 1}
			for &q in sold_q {q = 0}
			for &v in sold_v {v = 0}
			any_sale := false
			for x in k ..< e {
				tr := trades[order[x]]
				assert(
					tr.asset >= 0 && tr.asset < n_assets,
					"odean_disposition: asset out of range",
				)
				if tr.quantity < 0 && qty[tr.asset] > 0 {
					sold_q[tr.asset] -= tr.quantity
					sold_v[tr.asset] -= tr.quantity * tr.price
					any_sale = true
				}
			}
			holding := false
			for q in qty {holding ||= q > 0}
			if holding && (any_sale || !sale_days_only) {
				for a in 0 ..< n_assets {
					if qty[a] <= 0 {continue}
					if sold_q[a] > 0 {
						p := sold_v[a] / sold_q[a]
						if p >
						   cost[a] {c.realized_gains += 1} else if p < cost[a] {c.realized_losses += 1}
					} else {
						p := prices[a][d]
						if p >
						   cost[a] {c.paper_gains += 1} else if p < cost[a] {c.paper_losses += 1}
					}
				}
			}
			// apply today's trades
			for x in k ..< e {
				tr := trades[order[x]]
				a := tr.asset
				if tr.quantity > 0 {
					cost[a] = (qty[a] * cost[a] + tr.quantity * tr.price) / (qty[a] + tr.quantity)
					qty[a] += tr.quantity
				} else {
					assert(
						-tr.quantity <= qty[a] * (1 + 1e-12),
						"odean_disposition: sale larger than the position (long only)",
					)
					qty[a] = max(0, qty[a] + tr.quantity)
					if qty[a] == 0 {cost[a] = 0}
				}
			}
			k = e
		}
		i = j
	}
	nG := c.realized_gains + c.paper_gains
	nL := c.realized_losses + c.paper_losses
	c.pgr = f64(c.realized_gains) / f64(nG) if nG > 0 else math.nan_f64()
	c.plr = f64(c.realized_losses) / f64(nL) if nL > 0 else math.nan_f64()
	c.difference = c.pgr - c.plr
	if nG > 0 && nL > 0 {
		c.std_error = math.sqrt(c.pgr * (1 - c.pgr) / f64(nG) + c.plr * (1 - c.plr) / f64(nL))
		c.t_stat = c.difference / c.std_error
	} else {
		c.std_error, c.t_stat = math.nan_f64(), math.nan_f64()
	}
	return c
}
