package tests

import w "../wotan/core"
import yahoo "../wotan/net"

// 24 US large caps used by the real-data panel tests
LARGE_CAP_TICKERS := []string {
	"AAPL",
	"MSFT",
	"AMZN",
	"GOOG",
	"META",
	"NVDA",
	"JPM",
	"BAC",
	"XOM",
	"CVX",
	"JNJ",
	"PFE",
	"KO",
	"PEP",
	"WMT",
	"HD",
	"DIS",
	"INTC",
	"CSCO",
	"ORCL",
	"VZ",
	"MRK",
	"UNH",
	"PG",
}

// Daily adjusted closes (dividends and splits) and dollar volumes from Yahoo, on the
// dates every ticker has: P, V [date * len(tickers) + ticker]. ok = false when a
// ticker has fewer than min_rows rows (offline). Allocates with context.allocator.
yahoo_adjclose_panel :: proc(
	tickers: []string,
	rng: yahoo.range,
	min_rows: int,
) -> (
	P, V: []f64,
	dates: []w.Date,
	ok: bool,
) {
	NA := len(tickers)
	price_maps := make([]map[w.Date]f64, NA)
	dvol_maps := make([]map[w.Date]f64, NA)
	defer {
		for k in 0 ..< NA {delete(price_maps[k]); delete(dvol_maps[k])}
		delete(price_maps)
		delete(dvol_maps)
	}
	first := make([dynamic]w.Date)
	defer delete(first)
	for sym, k in tickers {
		df := yahoo.read_yahoo(sym, .Daily, rng, context.allocator)
		defer w.destroy_dataframe(&df)
		if df.rows < min_rows {return}
		price_maps[k] = make(map[w.Date]f64)
		dvol_maps[k] = make(map[w.Date]f64)
		dc, pc, vc := w.column(&df, "Date"), w.column(&df, "AdjClose"), w.column(&df, "Volume")
		for i in 0 ..< df.rows {
			d, dn := w.column_at_date(dc, i)
			p, pn := w.column_at_float(pc, i)
			v, vn := w.column_at_numeric(vc, i)
			if dn || pn || vn || p <= 0 {continue}
			price_maps[k][d] = p
			dvol_maps[k][d] = p * v
			if k == 0 {append(&first, d)}
		}
	}
	common := make([dynamic]w.Date)
	for d in first {
		all := true
		for m in price_maps {all &&= d in m}
		if all {append(&common, d)}
	}
	D := len(common)
	P = make([]f64, D * NA)
	V = make([]f64, D * NA)
	for d, t in common {
		for k in 0 ..< NA {P[t * NA + k], V[t * NA + k] = price_maps[k][d], dvol_maps[k][d]}
	}
	return P, V, common[:], true
}
