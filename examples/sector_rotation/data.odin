package main

// Stage 1: market data. Daily adjusted closes and volumes of the eleven
// Select Sector SPDR ETFs and of SPY, from Yahoo Finance, on the days all
// twelve traded.

import w "../../wotan/core"
import net "../../wotan/net"
import "core:fmt"
import "core:time"

SECTORS := [?]string{"XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"}
SECTOR_NAMES := [?]string {
	"Materials",
	"Communication",
	"Energy",
	"Financials",
	"Industrials",
	"Technology",
	"Staples",
	"Real Estate",
	"Utilities",
	"Health Care",
	"Discretionary",
}
N_ASSETS :: len(SECTORS)
BENCHMARK :: "SPY"

// Market: day-major arrays, [day * N_ASSETS + asset]
Market :: struct {
	dates:         []w.Date,
	n_days:        int,
	close:         []f64, // adjusted close
	volume:        []f64, // shares
	dollar_volume: []f64, // close · volume
	spy:           []f64, // SPY's adjusted close, [day]
}

// load_market reads ten years of daily data (XLC, the youngest fund, started
// in June 2018, so the common history begins there). false if Yahoo cannot be
// reached or a ticker cannot be loaded.
load_market :: proc() -> (m: Market, ok: bool) {
	tickers: [N_ASSETS + 1]string
	copy(tickers[:], SECTORS[:])
	tickers[N_ASSETS] = BENCHMARK
	panel, loaded, failed := net.read_yahoo_panel(tickers[:], .Daily, .TenYears)
	if !loaded {
		if failed >= 0 {
			fmt.printfln("could not load %s from Yahoo Finance (no network?)", tickers[failed])
		} else {
			fmt.println("the tickers share no trading day")
		}
		return
	}
	T, N := len(panel.dates), panel.n_tickers
	// While the US market is open, Yahoo's row for today holds the current
	// price, not a close: leave today out
	y, mo, d := time.date(time.now())
	last := panel.dates[T - 1]
	if int(last.year) == y && int(last.month) == int(mo) && int(last.day) == d {T -= 1}
	m.dates = panel.dates[:T]
	m.n_days = T
	m.close = make([]f64, T * N_ASSETS)
	m.volume = make([]f64, T * N_ASSETS)
	m.dollar_volume = make([]f64, T * N_ASSETS)
	m.spy = make([]f64, T)
	for d in 0 ..< T {
		for a in 0 ..< N_ASSETS {
			m.close[d * N_ASSETS + a] = panel.adj_close[d * N + a]
			m.volume[d * N_ASSETS + a] = panel.volume[d * N + a]
			m.dollar_volume[d * N_ASSETS + a] = panel.dollar_volume[d * N + a]
		}
		m.spy[d] = panel.adj_close[d * N + N_ASSETS]
	}
	return m, true
}

// asset_series copies one asset's column out of a day-major array
asset_series :: proc(x: []f64, a: int, allocator := context.allocator) -> []f64 {
	out := make([]f64, len(x) / N_ASSETS, allocator)
	for &v, d in out {v = x[d * N_ASSETS + a]}
	return out
}

date_string :: proc(d: w.Date) -> string {
	return w.date_to_string(d, context.temp_allocator)
}

// year_fraction: a date as a number for chart axes (2021.5 = mid-2021)
year_fraction :: proc(d: w.Date) -> f64 {
	return f64(d.year) + (f64(d.month) - 1) / 12 + (f64(d.day) - 1) / 365.25
}
