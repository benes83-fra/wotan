package main

// Stage 6: the report. A summary table on the console, four charts as PNG
// files, the daily returns of every strategy as an Excel workbook and the
// summary as an HTML table.

import w "../../wotan/core"
import ex "../../wotan/exporter"
import fin "../../wotan/finance"
import mlf "../../wotan/ml_finance"
import plot "../../wotan/plot"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"

OUT_DIR :: "examples/sector_rotation/output"

out_path :: proc(name: string) -> string {
	return strings.concatenate({OUT_DIR, "/", name}, context.temp_allocator)
}

// Result: one strategy with its statistics (and its forecasts' rank IC)
Result :: struct {
	strategy: Strategy,
	stats:    Stats,
	ic:       mlf.RankICResult,
	has_ic:   bool,
}

print_summary :: proc(results: []Result) {
	fmt.printfln(
		"  %-22s %8s %7s %7s %7s %6s %8s %7s %7s %9s %8s %8s %6s",
		"",
		"return",
		"vol",
		"Sharpe",
		"(gross)",
		"PSR",
		"max DD",
		"VaR 95",
		"ES 95",
		"turnover",
		"costs",
		"rank IC",
		"ICIR",
	)
	for r in results {
		s := r.stats
		fmt.printf(
			"  %-22s %7.1f%% %6.1f%% %7.2f %7.2f %6.2f %7.1f%% %6.2f%% %6.2f%%",
			r.strategy.name,
			100 * s.ann_return,
			100 * s.ann_vol,
			s.sharpe,
			s.sharpe_gross,
			s.psr,
			100 * s.max_drawdown,
			100 * s.var95,
			100 * s.es95,
		)
		if r.strategy.turnover > 0 {
			fmt.printf(" %8.0f%% %7.2f%%", 100 * r.strategy.turnover, 100 * r.strategy.cost_drag)
		} else {
			fmt.printf(" %9s %8s", "-", "-")
		}
		if r.has_ic {
			fmt.printfln(" %8.3f %6.2f", r.ic.mean_ic, r.ic.icir)
		} else {
			fmt.printfln(" %8s %6s", "-", "-")
		}
	}
}

// write_charts: growth of 1 per strategy, drawdowns, the rolling rank IC of
// the forecasts and the correlations of the sectors' daily returns
write_charts :: proc(m: ^Market, ds: ^Dataset, f: ^Forecasts, results: []Result) {
	lo, hi := f.test_lo, f.test_hi
	n := hi - lo
	xs := make([]f64, n, context.temp_allocator)
	for &x, i in xs {x = year_fraction(m.dates[lo + 1 + i])}

	growth := make([]plot.LineData, len(results), context.temp_allocator)
	drawdown := make([]plot.LineData, len(results), context.temp_allocator)
	for r, j in results {
		g := make([]f64, n, context.temp_allocator)
		v := 1.0
		for x, i in r.strategy.daily {v *= 1 + x; g[i] = v}
		growth[j] = {
			xs    = xs,
			ys    = g,
			label = r.strategy.name,
		}
		drawdown[j] = {
			xs    = xs,
			ys    = fin.drawdown_series(r.strategy.daily),
			label = r.strategy.name,
		}
	}
	cfg := plot.DEFAULT_PLOT_CONFIG
	cfg.width, cfg.height, cfg.show_grid = 1000, 560, true
	cfg.title, cfg.x_label, cfg.y_label = "Growth of 1 (after costs)", "year", "value"
	plot.multi_line_png(growth, out_path("equity.png"), cfg)
	cfg.title, cfg.y_label = "Drawdown", "fraction of the peak"
	plot.multi_line_png(drawdown, out_path("drawdown.png"), cfg)

	// rolling one-year mean of the daily rank IC
	ROLL :: TRADING_DAYS
	ic_lines := make([dynamic]plot.LineData, context.temp_allocator)
	for pred, k in ([][]f64{f.lstm, f.ridge, f.momentum}) {
		daily := make([dynamic]f64, context.temp_allocator)
		ic_x := make([dynamic]f64, context.temp_allocator)
		for d in lo ..< hi {
			if math.is_nan(ds.target[d * N_ASSETS]) {break}
			append(
				&daily,
				fin.spearman_rho(
					pred[d * N_ASSETS:][:N_ASSETS],
					ds.target[d * N_ASSETS:][:N_ASSETS],
					context.temp_allocator,
				),
			)
			append(&ic_x, year_fraction(m.dates[d]))
		}
		roll := make([]f64, len(daily) - ROLL + 1, context.temp_allocator)
		for i in 0 ..< len(roll) {
			for v in daily[i:i + ROLL] {roll[i] += v / ROLL}
		}
		append(
			&ic_lines,
			plot.LineData{xs = ic_x[ROLL - 1:], ys = roll, label = results[k].strategy.name},
		)
	}
	cfg.title, cfg.y_label = "Rank IC, rolling one-year mean", "Spearman correlation"
	plot.multi_line_png(ic_lines[:], out_path("rank_ic.png"), cfg)

	// correlations of the sectors' daily log returns, whole history
	T := m.n_days
	corr := make([]f64, N_ASSETS * N_ASSETS, context.temp_allocator)
	ret := make([][]f64, N_ASSETS, context.temp_allocator)
	for a in 0 ..< N_ASSETS {
		ret[a] = make([]f64, T - 1, context.temp_allocator)
		for d in 0 ..< T - 1 {ret[a][d] = math.ln(m.close[(d + 1) * N_ASSETS + a] / m.close[d * N_ASSETS + a])}
	}
	for a in 0 ..< N_ASSETS {
		for b in 0 ..< N_ASSETS {corr[a * N_ASSETS + b] = fin.correlation(ret[a], ret[b])}
	}
	hcfg := plot.DEFAULT_PLOT_CONFIG
	hcfg.width, hcfg.height = 820, 760
	hcfg.title = "Correlation of daily sector returns"
	plot.heatmap_png(
		corr,
		N_ASSETS,
		N_ASSETS,
		out_path("correlation.png"),
		SECTORS[:],
		SECTORS[:],
		hcfg,
	)
}

// write_tables: the daily net returns (Excel) and the summary (HTML)
write_tables :: proc(m: ^Market, f: ^Forecasts, results: []Result) {
	lo, hi := f.test_lo, f.test_hi
	cols := make([dynamic]w.Column, context.temp_allocator)
	append(&cols, w.column_from_dates("Date", m.dates[lo + 1:hi + 1]))
	for r in results {append(&cols, w.column_from_floats(r.strategy.name, r.strategy.daily))}
	daily := w.df_from(..cols[:])
	defer w.destroy_dataframe(&daily)
	ex.xlsx_save(&daily, out_path("daily_returns.xlsx"), context.temp_allocator)

	k := len(results)
	names := make([]string, k, context.temp_allocator)
	cols2: [8][]f64
	for &c in cols2 {c = make([]f64, k, context.temp_allocator)}
	for r, i in results {
		names[i] = r.strategy.name
		s := r.stats
		v := [8]f64 {
			s.ann_return,
			s.ann_vol,
			s.sharpe,
			s.sharpe_gross,
			s.psr,
			s.max_drawdown,
			s.var95,
			s.es95,
		}
		for c in 0 ..< 8 {cols2[c][i] = v[c]}
	}
	summary := w.df_from(
		w.column_from_strings("Strategy", names),
		w.column_from_floats("Annual return", cols2[0]),
		w.column_from_floats("Volatility", cols2[1]),
		w.column_from_floats("Sharpe", cols2[2]),
		w.column_from_floats("Sharpe before costs", cols2[3]),
		w.column_from_floats("PSR", cols2[4]),
		w.column_from_floats("Max drawdown", cols2[5]),
		w.column_from_floats("VaR 95%", cols2[6]),
		w.column_from_floats("ES 95%", cols2[7]),
	)
	defer w.destroy_dataframe(&summary)
	ex.html_write(&summary, out_path("summary.html"), context.temp_allocator)
}

ensure_out_dir :: proc() {
	if !os.exists(OUT_DIR) {_ = os.make_directory(OUT_DIR)}
}
