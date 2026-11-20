package main

// Stage 4: walk-forward forecasts. Every RETRAIN_EVERY days the models are
// fitted again on all the history up to then and forecast the next
// RETRAIN_EVERY days, which they have never seen.
// Stage 5: portfolios. Every HORIZON days, at the close, the strategy buys the
// TOP_K sectors with the highest forecast and sells short the TOP_K with the
// lowest, in equal parts (dollar neutral). Between rebalances the positions
// drift with the prices. Each rebalance pays the costs of `ml_finance`'s
// default cost model: a 1 bp half spread plus square-root market impact from
// the sector's 20-day volatility and dollar volume, for a CAPITAL book.

import fin "../../wotan/finance"
import l "../../wotan/linalg"
import mlf "../../wotan/ml_finance"
import nn "../../wotan/nn"
import t "../../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:time"

TRAIN_MIN_DAYS :: 756 // three years of history before the first forecast
RETRAIN_EVERY :: 126 // a new model every six months
TOP_K :: 3 // sectors held long, and sectors held short
CAPITAL :: 10_000_000.0 // book size for the market-impact cost
TRADING_DAYS :: 252

Forecasts :: struct {
	lstm, ridge, momentum: []f64, // [day][asset], NaN outside the test period
	test_lo, test_hi:      int, // forecasts for days test_lo ..< test_hi
	train_seconds:         f64, // LSTM training, all folds
}

walk_forward :: proc(m: ^Market, ds: ^Dataset, on_gpu: bool) -> (f: Forecasts) {
	T := m.n_days
	f.lstm = make([]f64, T * N_ASSETS)
	f.ridge = make([]f64, T * N_ASSETS)
	f.momentum = make([]f64, T * N_ASSETS)
	for i in 0 ..< T * N_ASSETS {f.lstm[i], f.ridge[i], f.momentum[i] = math.nan_f64(), math.nan_f64(), math.nan_f64()}
	first := ds.first_day + WINDOW - 1 // the first day with a full window
	f.test_lo = first + TRAIN_MIN_DAYS
	f.test_hi = T - 1 // the last forecast needs a next day to trade into
	fold := 0
	for s := f.test_lo; s < f.test_hi; s += RETRAIN_EVERY {
		e := min(s + RETRAIN_EVERY, f.test_hi)
		// Purging: the target of day d covers days d+1 ..= d+HORIZON, so the
		// training days stop HORIZON days before the test period begins
		samples := training_samples(ds, first, s - HORIZON - 1)
		defer delete(samples)

		start := time.tick_now()
		model, loss := train_lstm(ds, samples, on_gpu, u64(100 + fold))
		predict_lstm(model, ds, s, e, f.lstm)
		secs := time.duration_seconds(time.tick_since(start))
		f.train_seconds += secs
		nn.sequential_free(model)

		beta := fit_ridge(ds, samples)
		defer delete(beta)
		predict_ridge(beta, ds, s, e, f.ridge)
		predict_momentum(ds, s, e, f.momentum)
		fmt.printfln(
			"  fold %d: trained on %s .. %s (%5d samples, loss %.3f, %4.1f s), forecasts %s .. %s",
			fold + 1,
			date_string(m.dates[first]),
			date_string(m.dates[s - HORIZON - 1]),
			len(samples),
			loss,
			secs,
			date_string(m.dates[s]),
			date_string(m.dates[e - 1]),
		)
		fold += 1
	}
	return
}

// rank_ic: the daily Spearman correlation across sectors between forecast and
// realised target, over the test days whose target is known
rank_ic :: proc(pred: []f64, ds: ^Dataset, lo, hi: int) -> mlf.RankICResult {
	n := 0
	for d in lo ..< hi {if !math.is_nan(ds.target[d * N_ASSETS]) {n += 1}}
	p := t.tensor_new(l.matrix_new(f64, n, N_ASSETS))
	y := t.tensor_new(l.matrix_new(f64, n, N_ASSETS))
	defer {t.tensor_free(p); t.tensor_free(y)}
	p.shape, y.shape = {n, N_ASSETS, 1, 1}, {n, N_ASSETS, 1, 1}
	copy(p.data.data, pred[lo * N_ASSETS:(lo + n) * N_ASSETS])
	copy(y.data.data, ds.target[lo * N_ASSETS:(lo + n) * N_ASSETS])
	return mlf.cross_sectional_rank_ic(p, y, context.allocator)
}

Strategy :: struct {
	name:      string,
	daily:     []f64, // net daily returns, entry i for day lo + 1 + i
	gross:     []f64, // the same before costs
	turnover:  f64, // mean one-way turnover per rebalance
	cost_drag: f64, // costs per year, as a fraction of the capital
}

// Liquidity: what the cost model needs, known at each day's close
Liquidity :: struct {
	sigma, adv: []f64, // [day][asset]: 20-day volatility of daily returns, dollar volume
}

liquidity_new :: proc(m: ^Market) -> (liq: Liquidity) {
	liq.sigma = make([]f64, m.n_days * N_ASSETS)
	liq.adv = make([]f64, m.n_days * N_ASSETS)
	for a in 0 ..< N_ASSETS {
		close := asset_series(m.close, a, context.temp_allocator)
		volume := asset_series(m.volume, a, context.temp_allocator)
		sigma, adv := mlf.liquidity_estimates(close, volume, 20, context.temp_allocator)
		for d in 0 ..< m.n_days {
			liq.sigma[d * N_ASSETS + a] = sigma[d]
			liq.adv[d * N_ASSETS + a] = adv[d]
		}
		free_all(context.temp_allocator)
	}
	return
}

// target_weights: long/short TOP_K sectors by forecast; nil forecasts: equal
// weights, long only
target_weights :: proc(pred: []f64, d: int) -> (w: [N_ASSETS]f64) {
	if pred == nil {
		for &v in w {v = 1.0 / N_ASSETS}
		return
	}
	order: [N_ASSETS]int
	for &v, i in order {v = i}
	row := pred[d * N_ASSETS:][:N_ASSETS]
	// sort the sector indices by forecast, highest first (insertion sort)
	for i in 1 ..< N_ASSETS {
		for k := i; k > 0 && row[order[k]] > row[order[k - 1]]; k -= 1 {
			order[k], order[k - 1] = order[k - 1], order[k]
		}
	}
	for k in 0 ..< TOP_K {
		w[order[k]] = 1.0 / TOP_K
		w[order[N_ASSETS - 1 - k]] = -1.0 / TOP_K
	}
	return
}

// simulate runs a strategy over the test days lo ..< hi: rebalance at the
// close of every HORIZON-th day, hold until the next rebalance
simulate :: proc(
	m: ^Market,
	liq: ^Liquidity,
	name: string,
	pred: []f64,
	lo, hi: int,
) -> (
	s: Strategy,
) {
	s.name = name
	s.daily = make([]f64, hi - lo)
	s.gross = make([]f64, hi - lo)
	w: [N_ASSETS]f64 // positions as fractions of the equity
	equity, total_cost, turnover := 1.0, 0.0, 0.0
	rebalances := 0
	for d in lo ..< hi {
		cost := 0.0
		if (d - lo) % HORIZON == 0 {
			target := target_weights(pred, d)
			cost = mlf.rebalance_cost(
				mlf.DEFAULT_COST_MODEL,
				w[:],
				target[:],
				CAPITAL * equity,
				liq.sigma[d * N_ASSETS:][:N_ASSETS],
				liq.adv[d * N_ASSETS:][:N_ASSETS],
			)
			for a in 0 ..< N_ASSETS {turnover += abs(target[a] - w[a]) / 2}
			rebalances += 1
			w = target
		}
		// the next day's return, then the positions drift with the prices
		rp := 0.0
		r: [N_ASSETS]f64
		for a in 0 ..< N_ASSETS {
			r[a] = m.close[(d + 1) * N_ASSETS + a] / m.close[d * N_ASSETS + a] - 1
			rp += w[a] * r[a]
		}
		for a in 0 ..< N_ASSETS {w[a] *= (1 + r[a]) / (1 + rp)}
		net := (1 - cost) * (1 + rp) - 1
		s.daily[d - lo] = net
		s.gross[d - lo] = rp
		equity *= 1 + net
		total_cost += cost
	}
	s.turnover = turnover / f64(max(rebalances, 1))
	s.cost_drag = total_cost / f64(hi - lo) * TRADING_DAYS
	return
}

// buy_and_hold: SPY's daily returns over the same days
buy_and_hold :: proc(m: ^Market, lo, hi: int) -> (s: Strategy) {
	s.name = "SPY (buy and hold)"
	s.daily = make([]f64, hi - lo)
	for d in lo ..< hi {s.daily[d - lo] = m.spy[d + 1] / m.spy[d] - 1}
	s.gross = s.daily
	return
}

Stats :: struct {
	ann_return, ann_vol, sharpe, psr: f64,
	sharpe_gross:                     f64, // before costs
	max_drawdown, var95, es95:        f64,
}

// stats_of: annualised return and volatility, the annualised Sharpe ratio,
// the probabilistic Sharpe ratio (P[true Sharpe > 0]), the maximum drawdown
// and the 1-day historical VaR and expected shortfall at 95%
stats_of :: proc(r: []f64) -> (s: Stats) {
	n := f64(len(r))
	growth, mean := 1.0, 0.0
	for x in r {growth *= 1 + x; mean += x / n}
	sq := 0.0
	for x in r {sq += (x - mean) * (x - mean) / (n - 1)}
	s.ann_return = math.pow(growth, TRADING_DAYS / n) - 1
	s.ann_vol = math.sqrt(sq * TRADING_DAYS)
	s.sharpe = mlf.sharpe_stats(r).sharpe * math.sqrt(f64(TRADING_DAYS))
	s.psr = mlf.probabilistic_sharpe_ratio_returns(r, 0)
	s.max_drawdown = fin.max_drawdown(r)
	s.var95 = fin.value_at_risk(r, 0.95)
	s.es95 = fin.expected_shortfall(r, 0.95)
	return
}

// deflated_sharpe: the deflated Sharpe ratio of the best of the strategies
// tried, which accounts for having picked the best of several
deflated_sharpe :: proc(strategies: []Strategy) -> mlf.DeflatedSharpeResult {
	n := len(strategies[0].daily)
	R := l.matrix_new(f64, n, len(strategies))
	defer l.matrix_free(&R)
	for s, j in strategies {
		for x, i in s.daily {R.data[i * len(strategies) + j] = x}
	}
	return mlf.deflated_sharpe_trials(&R)
}
