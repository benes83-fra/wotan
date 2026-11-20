package main

// Stage 6: the report. Four charts as PNG files, the daily forecasts as an
// Excel workbook, the forecast accuracy and the option strategies as HTML
// tables.

import w "../../wotan/core"
import ex "../../wotan/exporter"
import plot "../../wotan/plot"
import "core:os"
import "core:strings"

OUT_DIR :: "examples/volatility/output"
CHART_FROM :: 2019.5 // the window of the forecast and VaR charts: COVID-19
CHART_TO :: 2021.5

out_path :: proc(name: string) -> string {
	return strings.concatenate({OUT_DIR, "/", name}, context.temp_allocator)
}

ensure_out_dir :: proc() {
	if !os.exists(OUT_DIR) {_ = os.make_directory(OUT_DIR)}
}

write_charts :: proc(
	m: ^Market,
	f: ^Forecasts,
	v: ^VaR_Result,
	opts: []Option_Strategy,
	trades: []Straddle,
) {
	ta := context.temp_allocator
	cfg := plot.DEFAULT_PLOT_CONFIG
	cfg.width, cfg.height, cfg.show_grid = 1000, 560, true

	// the forecasts around the COVID-19 crash
	{
		days := make([dynamic]int, ta)
		for d in f.test_lo ..< m.n_days {
			x := year_fraction(m.dates[d])
			if x >= CHART_FROM && x < CHART_TO {append(&days, d)}
		}
		xs := make([]f64, len(days), ta)
		for d, i in days {xs[i] = year_fraction(m.dates[d])}
		series := make([dynamic]plot.LineData, ta)
		real := make([]f64, len(days), ta)
		for d, i in days {real[i] = m.rv_fwd[d]}
		append(&series, plot.LineData{xs = xs, ys = real, label = "realised (next 21 days)"})
		for model in Model {
			ys := make([]f64, len(days), ta)
			for d, i in days {ys[i] = f.vol[model][d]}
			append(&series, plot.LineData{xs = xs, ys = ys, label = MODEL_NAMES[model]})
		}
		cfg.title, cfg.x_label, cfg.y_label =
			"Volatility of the next 21 days: forecasts and outcome", "year", "annualised %"
		plot.multi_line_png(series[:], out_path("forecasts.png"), cfg)

		// daily losses against the VaR of the same days
		loss := make([]f64, len(days), ta)
		gv := make([]f64, len(days), ta)
		hv := make([]f64, len(days), ta)
		for d, i in days {
			k := d - v.lo
			loss[i] = -100 * m.ret[d]
			gv[i] = 100 * v.series[.GARCH][k]
			hv[i] = 100 * v.series[.Historical][k]
		}
		var_lines := []plot.LineData {
			{xs = xs, ys = loss, label = "daily loss"},
			{xs = xs, ys = gv, label = "99% VaR, GJR-GARCH-t"},
			{xs = xs, ys = hv, label = "99% VaR, historical 250d"},
		}
		cfg.title, cfg.y_label = "SPY daily loss and the one-day 99% VaR", "% of the position"
		plot.multi_line_png(var_lines, out_path("var.png"), cfg)
	}

	// the regime probability over the whole test period
	{
		n := m.n_days - f.test_lo
		xs := make([]f64, n, ta)
		ys := make([]f64, n, ta)
		for i in 0 ..< n {
			xs[i] = year_fraction(m.dates[f.test_lo + i])
			ys[i] = f.p_turbulent[f.test_lo + i]
		}
		rcfg := cfg
		rcfg.title, rcfg.y_label = "Filtered probability of the turbulent regime", "probability"
		plot.line_png(xs, ys, out_path("regimes.png"), rcfg)
	}

	// cumulative P&L of the option strategies
	{
		xs := make([]f64, len(trades), ta)
		for tr, i in trades {xs[i] = year_fraction(m.dates[tr.day + HORIZON])}
		lines := make([]plot.LineData, len(opts), ta)
		for s, j in opts {
			cum := make([]f64, len(trades), ta)
			acc := 0.0
			for p, i in s.monthly {acc += p; cum[i] = acc}
			lines[j] = {
				xs    = xs,
				ys    = cum,
				label = s.name,
			}
		}
		cfg.title, cfg.y_label =
			"Short straddles, delta-hedged: cumulative P&L", "% of spot, summed"
		plot.multi_line_png(lines, out_path("straddles.png"), cfg)
	}
}

write_tables :: proc(
	m: ^Market,
	f: ^Forecasts,
	acc: [Model]Accuracy,
	opts: []Option_Strategy,
	trades: []Straddle,
) {
	ta := context.temp_allocator
	lo := f.test_lo
	cols := make([dynamic]w.Column, ta)
	append(&cols, w.column_from_dates("Date", m.dates[lo:]))
	append(&cols, w.column_from_floats("Realised next 21 days", m.rv_fwd[lo:]))
	for model in Model {append(&cols, w.column_from_floats(MODEL_NAMES[model], f.vol[model][lo:]))}
	append(&cols, w.column_from_floats("P(turbulent)", f.p_turbulent[lo:]))
	append(&cols, w.column_from_floats("GJR-GARCH 99% VaR, next day", f.var99[lo:]))
	daily := w.df_from(..cols[:])
	defer w.destroy_dataframe(&daily)
	ex.xlsx_save(&daily, out_path("forecasts.xlsx"), ta)

	names := make([]string, len(Model), ta)
	vals: [7][]f64
	for &c in vals {c = make([]f64, len(Model), ta)}
	for model, i in Model {
		a := acc[model]
		names[i] = MODEL_NAMES[model]
		row := [7]f64{a.rmse, a.bias, a.qlike, a.rmse_calm, a.rmse_turb, a.dm_t, a.dm_p}
		for c in 0 ..< 7 {vals[c][i] = row[c]}
	}
	summary := w.df_from(
		w.column_from_strings("Model", names),
		w.column_from_floats("RMSE", vals[0]),
		w.column_from_floats("Bias", vals[1]),
		w.column_from_floats("QLIKE", vals[2]),
		w.column_from_floats("RMSE calm", vals[3]),
		w.column_from_floats("RMSE turbulent", vals[4]),
		w.column_from_floats("DM t vs GARCH", vals[5]),
		w.column_from_floats("DM p", vals[6]),
	)
	defer w.destroy_dataframe(&summary)
	ex.html_write(&summary, out_path("accuracy.html"), ta)

	k := len(opts)
	onames := make([]string, k, ta)
	ov: [6][]f64
	for &c in ov {c = make([]f64, k, ta)}
	for &s, i in opts {
		st := option_stats(&s, trades)
		onames[i] = s.name
		row := [6]f64{f64(s.sold), st.mean, st.sharpe, st.psr, st.worst, st.break_even}
		for c in 0 ..< 6 {ov[c][i] = row[c]}
	}
	options := w.df_from(
		w.column_from_strings("Strategy", onames),
		w.column_from_floats("Straddles sold", ov[0]),
		w.column_from_floats("Mean P&L per month (% of spot)", ov[1]),
		w.column_from_floats("Sharpe", ov[2]),
		w.column_from_floats("PSR", ov[3]),
		w.column_from_floats("Worst month (% of spot)", ov[4]),
		w.column_from_floats("Break-even (vol points below the VIX)", ov[5]),
	)
	defer w.destroy_dataframe(&options)
	ex.html_write(&options, out_path("straddles.html"), ta)
}
