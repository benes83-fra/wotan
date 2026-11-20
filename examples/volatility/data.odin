package main

// Stage 1: market data. SPY's adjusted closes and the VIX, daily, from Yahoo
// Finance, on the days both have a value: all of SPY's history (since 1993).

import w "../../wotan/core"
import net "../../wotan/net"
import "core:fmt"
import "core:math"

HORIZON :: 21 // the forecast horizon in trading days: one month, as the VIX
TRADING_DAYS :: 252

// Market: one entry per day
Market :: struct {
	dates:  []w.Date,
	n_days: int,
	spy:    []f64, // SPY's adjusted close
	vix:    []f64, // the VIX at the close, in % (20 = 20% annualised)
	ret:    []f64, // SPY's log return from the previous close; ret[0] = 0
	rv_fwd: []f64, // realised volatility over the next HORIZON days, annualised %; NaN at the end
}

// load_market: false if Yahoo cannot be reached
load_market :: proc() -> (m: Market, ok: bool) {
	tickers := []string{"SPY", "^VIX"}
	panel, loaded, failed := net.read_yahoo_panel(tickers, .Daily, .Max)
	if !loaded {
		if failed >= 0 {
			fmt.printfln("could not load %s from Yahoo Finance (no network?)", tickers[failed])
		} else {
			fmt.println("SPY and the VIX share no trading day")
		}
		return
	}
	T := len(panel.dates)
	// While the US market is open, the last row holds current prices, not closes
	if panel.last_is_live {T -= 1}
	m.n_days = T
	m.dates = panel.dates[:T]
	m.spy = make([]f64, T)
	m.vix = make([]f64, T)
	m.ret = make([]f64, T)
	for d in 0 ..< T {
		m.spy[d] = panel.adj_close[d * 2]
		m.vix[d] = panel.adj_close[d * 2 + 1]
		if d > 0 {m.ret[d] = math.ln(m.spy[d] / m.spy[d - 1])}
	}
	m.rv_fwd = make([]f64, T)
	for d in 0 ..< T {
		m.rv_fwd[d] = d + HORIZON < T ? realised_vol(&m, d + HORIZON, HORIZON) : math.nan_f64()
	}
	return m, true
}

// realised_vol: the annualised volatility (in %) of the k returns up to and
// including day d, sqrt(252 · mean r²), as option traders measure it (no
// demeaning)
realised_vol :: proc(m: ^Market, d, k: int) -> f64 {
	s := 0.0
	for j in d - k + 1 ..= d {s += m.ret[j] * m.ret[j]}
	return 100 * math.sqrt(f64(TRADING_DAYS) * s / f64(k))
}

date_string :: proc(d: w.Date) -> string {
	return w.date_to_string(d, context.temp_allocator)
}

// year_fraction: a date as a number for chart axes (2021.5 = mid-2021)
year_fraction :: proc(d: w.Date) -> f64 {
	return f64(d.year) + (f64(d.month) - 1) / 12 + (f64(d.day) - 1) / 365.25
}
