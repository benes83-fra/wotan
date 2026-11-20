package main

// End-to-end example: forecasting volatility, and using the forecasts for
// options and risk, on real market data.
//
//     odin run examples/volatility -o:speed            (from the repository root)
//     odin run examples/volatility -o:speed -- -cpu    (train on the CPU)
//
// 1. data       Yahoo Finance: SPY and the VIX since 1993             (data.odin)
// 2. models     VIX, GJR-GARCH, HAR + VIX, LSTM and Mamba (GPU)       (models.odin, neural.odin)
// 3. regimes    a two-state hidden Markov model, filtered             (regimes.odin)
// 4. forecasts  walk-forward, refit every two years; RMSE, QLIKE,
//               Diebold-Mariano tests, errors by regime              (forecasts.odin)
// 5. use        short straddles priced at the VIX, sold on signals;
//               one-day 99% VaR and its backtest                     (options.odin, risk.odin)
// 6. report     charts, Excel and HTML                               (report.odin)
//
// Without a network connection the program stops at step 1: it never makes
// up data. Without a GPU it trains on the CPU.

import gpu "../../wotan/gpu"
import "core:fmt"
import vmem "core:mem/virtual"
import "core:os"
import "core:time"

main :: proc() {
	// Everything the run keeps lives in one arena, released at the end
	arena: vmem.Arena
	_ = vmem.arena_init_growing(&arena)
	defer vmem.arena_destroy(&arena)
	context.allocator = vmem.arena_allocator(&arena)
	run_start := time.tick_now()

	fmt.println("1. Market data")
	m, ok := load_market()
	if !ok {os.exit(1)}
	fmt.printfln(
		"  SPY and the VIX, %d common trading days: %s .. %s",
		m.n_days,
		date_string(m.dates[0]),
		date_string(m.dates[m.n_days - 1]),
	)

	fmt.println("2./3./4. Walk-forward: models, regimes and forecasts")
	use_cpu := len(os.args) > 1 && os.args[1] == "-cpu"
	on_gpu := !use_cpu && gpu.init()
	if on_gpu {
		fmt.printfln("  LSTM and Mamba on the GPU: %s", gpu.device_info().name)
	} else if use_cpu {
		fmt.println("  LSTM and Mamba on the CPU (-cpu)")
	} else {
		fmt.printfln("  LSTM and Mamba on the CPU (no GPU: %s)", gpu.init_error())
	}
	f := walk_forward(&m, on_gpu)
	if on_gpu {gpu.shutdown()}

	acc := evaluate(&m, &f)
	last := m.n_days - HORIZON - 1
	fmt.printfln(
		"  forecasts made %s .. %s (%d days); volatility in annualised points",
		date_string(m.dates[f.test_lo]),
		date_string(m.dates[last]),
		acc[.VIX].n,
	)
	fmt.printfln(
		"  %d of them (%.0f%%) in the turbulent regime",
		acc[.VIX].n_turb,
		100 * f64(acc[.VIX].n_turb) / f64(acc[.VIX].n),
	)
	print_accuracy(acc, &f)

	fmt.println("5. Using the forecasts")
	trades := sell_straddles(&m, &f)
	opts := option_strategies(&m, &f, trades)
	fmt.printfln(
		"  one-month ATM straddles sold at the VIX, delta-hedged daily, %d periods; P&L per period",
		len(trades),
	)
	print_option_stats(opts, trades)
	v := value_at_risk(&m, &f)
	fmt.printfln("  one-day 99%% VaR of SPY, %d test days", len(v.returns))
	print_var(&v)

	fmt.println("6. Report")
	ensure_out_dir()
	write_charts(&m, &f, &v, opts, trades)
	write_tables(&m, &f, acc, opts, trades)
	fmt.printfln(
		"  written to %s: forecasts.png, var.png, regimes.png, straddles.png, forecasts.xlsx, accuracy.html, straddles.html",
		OUT_DIR,
	)
	fmt.printfln("done in %.1f s", time.duration_seconds(time.tick_since(run_start)))
}
