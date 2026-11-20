package main

// End-to-end example: a sector-rotation strategy on real market data.
//
//     odin run examples/sector_rotation -o:speed      (from the repository root)
//     odin run examples/sector_rotation -o:speed -- -cpu   (train on the CPU)
//
// 1. data       Yahoo Finance: the 11 sector ETFs and SPY, daily     (data.odin)
// 2. features   8 per sector and day, z-scored across sectors        (features.odin)
// 3. models     LSTM (GPU if there is one), ridge, momentum          (models.odin)
// 4. forecasts  walk-forward: refit every six months, purged         (backtest.odin)
// 5. portfolio  weekly long/short top 3 vs bottom 3, with costs      (backtest.odin)
// 6. report     statistics, charts, Excel and HTML                   (report.odin)
//
// Without a network connection the program stops at step 1: it never makes
// up data. Without a GPU it trains on the CPU.

import gpu "../../wotan/gpu"
import mlf "../../wotan/ml_finance"
import "core:fmt"
import "core:math"
import vmem "core:mem/virtual"
import "core:os"
import "core:time"

main :: proc() {
	// Everything the run keeps (prices, features, forecasts, returns) lives in
	// one arena, released at the end
	arena: vmem.Arena
	_ = vmem.arena_init_growing(&arena)
	defer vmem.arena_destroy(&arena)
	context.allocator = vmem.arena_allocator(&arena)
	run_start := time.tick_now()

	fmt.println("1. Market data")
	m, ok := load_market()
	if !ok {os.exit(1)}
	fmt.printfln(
		"  %d sector ETFs and %s, %d common trading days: %s .. %s",
		N_ASSETS,
		BENCHMARK,
		m.n_days,
		date_string(m.dates[0]),
		date_string(m.dates[m.n_days - 1]),
	)

	fmt.println("2. Features")
	ds := build_dataset(&m, TRAIN_MIN_DAYS)
	print_dataset(&m, &ds)

	fmt.println("3./4. Models and walk-forward forecasts")
	use_cpu := len(os.args) > 1 && os.args[1] == "-cpu"
	on_gpu := !use_cpu && gpu.init()
	if on_gpu {
		info := gpu.device_info()
		fmt.printfln("  LSTM on the GPU: %s", info.name)
	} else if use_cpu {
		fmt.println("  LSTM on the CPU (-cpu)")
	} else {
		fmt.printfln("  LSTM on the CPU (no GPU: %s)", gpu.init_error())
	}
	f := walk_forward(&m, &ds, on_gpu)
	if on_gpu {gpu.shutdown()}
	fmt.printfln("  LSTM training and forecasting: %.1f s in total", f.train_seconds)

	fmt.println("5. Portfolios (weekly rebalancing, costs included)")
	liq := liquidity_new(&m)
	lo, hi := f.test_lo, f.test_hi
	results := []Result {
		{strategy = simulate(&m, &liq, "LSTM long/short", f.lstm, lo, hi)},
		{strategy = simulate(&m, &liq, "Ridge long/short", f.ridge, lo, hi)},
		{strategy = simulate(&m, &liq, "Momentum long/short", f.momentum, lo, hi)},
		{strategy = simulate(&m, &liq, "Equal weight (long)", nil, lo, hi)},
		{strategy = buy_and_hold(&m, lo, hi)},
	}
	for &r, i in results {
		r.stats = stats_of(r.strategy.daily)
		r.stats.sharpe_gross =
			mlf.sharpe_stats(r.strategy.gross).sharpe * math.sqrt(f64(TRADING_DAYS))
		if i < 3 {
			preds := [3][]f64{f.lstm, f.ridge, f.momentum}
			r.ic_daily, r.ic = rank_ic(preds[i], &ds, lo, hi)
			r.has_ic = true
		}
	}
	fmt.printfln(
		"  test period %s .. %s (%d days), annualised; VaR and ES of one day",
		date_string(m.dates[lo + 1]),
		date_string(m.dates[hi]),
		hi - lo,
	)
	print_summary(results)

	// three long/short strategies were tried: is the best one still good
	// once that selection is accounted for?
	ls := []Strategy{results[0].strategy, results[1].strategy, results[2].strategy}
	dsr := deflated_sharpe(ls)
	fmt.printfln(
		"  best of the %d long/short strategies: %s, deflated Sharpe ratio %.2f (P[skill] after picking the best)",
		len(ls),
		ls[dsr.best].name,
		dsr.dsr,
	)
	// even the benchmark's Sharpe ratio takes years to tell from luck
	fmt.printfln(
		"  track record the equal-weight portfolio's Sharpe ratio needs to be positive with 95%% confidence: %.1f years",
		_min_track_months(results[3].strategy.daily) / 12,
	)

	fmt.println("6. Report")
	ensure_out_dir()
	write_charts(&m, &ds, &f, results)
	write_tables(&m, &f, results)
	fmt.printfln(
		"  written to %s: equity.png, drawdown.png, rank_ic.png, correlation.png, daily_returns.xlsx, summary.html",
		OUT_DIR,
	)
	fmt.printfln("done in %.1f s", time.duration_seconds(time.tick_since(run_start)))
}

// _min_track_months: the minimum track record length (Bailey & López de
// Prado) for PSR(0) = 95% at the observed daily Sharpe ratio, in months
_min_track_months :: proc(daily: []f64) -> f64 {
	s := mlf.sharpe_stats(daily)
	n := mlf.min_track_record_length(s.sharpe, 0, s.skewness, s.kurtosis, 0.95)
	return n / 21
}
