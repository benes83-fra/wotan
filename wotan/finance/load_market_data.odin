package finance


import w "../core"
import csv "../importer"
import "core:fmt"
import "core:mem"

// Historical prices from a CSV file with Date, Close and Volume columns
// (others are ignored): dates, closing prices, volumes and simple returns
// (the first return is 0). Every Date must be a date and every Close and
// Volume a number. Panics when the file cannot be read or does not fit;
// try_load_market_data returns ok = false instead. Free with
// free_market_data.
load_market_data :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> MarketData {
	df := csv.csv_load(path)
	defer w.destroy_dataframe(&df)
	md, err := _market_data(&df, allocator)
	if err != "" {panic(fmt.tprintf("load_market_data: %q: %s", path, err))}
	return md
}

// load_market_data, with ok = false (and the reason on stderr) instead of a
// panic
try_load_market_data :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	MarketData,
	bool,
) {
	df, ok := csv.try_csv_load(path)
	if !ok {return {}, false}
	defer w.destroy_dataframe(&df)
	md, err := _market_data(&df, allocator)
	if err != "" {
		fmt.eprintfln("try_load_market_data: %q: %s", path, err)
		return {}, false
	}
	return md, true
}

// Option chain from a CSV file with Type, Strike, Expiry, ImpliedVol and
// Price columns (others, such as Symbol, Bid or Volume, are ignored), with
// expiries in years from `current_date`. Expiry must be a date and the other
// three numbers in every row. The Type column is read case-insensitively by
// its first letter into the literals "call" / "put" ("" for anything else).
// Panics when the file cannot be read or does not fit; try_load_option_chain
// returns ok = false instead. Free with free_option_chain.
load_option_chain :: proc(
	path: string,
	current_date: w.Date,
	allocator: mem.Allocator = context.allocator,
) -> OptionChain {
	df := csv.csv_load(path)
	defer w.destroy_dataframe(&df)
	chain, err := _option_chain(&df, current_date, allocator)
	if err != "" {panic(fmt.tprintf("load_option_chain: %q: %s", path, err))}
	return chain
}

// load_option_chain, with ok = false (and the reason on stderr) instead of a
// panic
try_load_option_chain :: proc(
	path: string,
	current_date: w.Date,
	allocator: mem.Allocator = context.allocator,
) -> (
	OptionChain,
	bool,
) {
	df, ok := csv.try_csv_load(path)
	if !ok {return {}, false}
	defer w.destroy_dataframe(&df)
	chain, err := _option_chain(&df, current_date, allocator)
	if err != "" {
		fmt.eprintfln("try_load_option_chain: %q: %s", path, err)
		return {}, false
	}
	return chain, true
}

// The column `name` of df if it has one of the types `want`; the reason
// otherwise (temp memory)
@(private = "file")
_need :: proc(df: ^w.DataFrame, name: string, want: ..w.ColumnType) -> (^w.Column, string) {
	col, found := w.try_column(df, name)
	if !found {return nil, fmt.tprintf("no %s column", name)}
	for t in want {
		if col.type == t {return col, ""}
	}
	return nil, fmt.tprintf(
		"the %s column holds %v values, not %s",
		name,
		col.type,
		"dates" if want[0] == .Date else "numbers",
	)
}

@(private = "file")
_market_data :: proc(df: ^w.DataFrame, allocator: mem.Allocator) -> (MarketData, string) {
	date_col, e1 := _need(df, "Date", .Date)
	if e1 != "" {return {}, e1}
	close_col, e2 := _need(df, "Close", .Float, .Int)
	if e2 != "" {return {}, e2}
	volume_col, e3 := _need(df, "Volume", .Float, .Int)
	if e3 != "" {return {}, e3}

	n := df.rows
	for i in 0 ..< n {
		if w.is_null(date_col, i) {return {}, fmt.tprintf("Date is missing in data row %d", i + 1)}
		if _, missing := w.column_at_numeric(close_col, i); missing {
			return {}, fmt.tprintf("Close is missing in data row %d", i + 1)
		}
		if _, missing := w.column_at_numeric(volume_col, i); missing {
			return {}, fmt.tprintf("Volume is missing in data row %d", i + 1)
		}
	}

	dates := make([]w.Date, n, allocator)
	prices := make([]f64, n, allocator)
	volumes := make([]f64, n, allocator)
	returns := make([]f64, n, allocator)
	for i in 0 ..< n {
		dates[i], _ = w.column_at_date(date_col, i)
		prices[i], _ = w.column_at_numeric(close_col, i)
		volumes[i], _ = w.column_at_numeric(volume_col, i)
		returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1] if i > 0 else 0
	}
	return MarketData {
			dates = dates,
			prices = prices,
			volumes = volumes,
			returns = returns,
			n_obs = n,
		},
		""
}

@(private = "file")
_option_chain :: proc(
	df: ^w.DataFrame,
	current_date: w.Date,
	allocator: mem.Allocator,
) -> (
	OptionChain,
	string,
) {
	type_col, e1 := _need(df, "Type", .String)
	if e1 != "" {return {}, e1}
	strike_col, e2 := _need(df, "Strike", .Float, .Int)
	if e2 != "" {return {}, e2}
	expiry_col, e3 := _need(df, "Expiry", .Date)
	if e3 != "" {return {}, e3}
	iv_col, e4 := _need(df, "ImpliedVol", .Float, .Int)
	if e4 != "" {return {}, e4}
	price_col, e5 := _need(df, "Price", .Float, .Int)
	if e5 != "" {return {}, e5}

	n := df.rows
	for i in 0 ..< n {
		if w.is_null(
			expiry_col,
			i,
		) {return {}, fmt.tprintf("Expiry is missing in data row %d", i + 1)}
		for col in ([3]^w.Column{strike_col, iv_col, price_col}) {
			if _, missing := w.column_at_numeric(col, i); missing {
				return {}, fmt.tprintf("%s is missing in data row %d", col.name, i + 1)
			}
		}
	}

	strikes := make([]f64, n, allocator)
	expiries := make([]f64, n, allocator)
	implied_vols := make([]f64, n, allocator)
	market_prices := make([]f64, n, allocator)
	option_types := make([]string, n, allocator)
	for i in 0 ..< n {
		// literals, not views into the DataFrame destroyed by the caller
		type_str, _ := w.column_at_string(type_col, i)
		switch {
		case len(type_str) > 0 && (type_str[0] == 'c' || type_str[0] == 'C'):
			option_types[i] = "call"
		case len(type_str) > 0 && (type_str[0] == 'p' || type_str[0] == 'P'):
			option_types[i] = "put"
		case:
			option_types[i] = ""
		}
		strikes[i], _ = w.column_at_numeric(strike_col, i)
		expiry, _ := w.column_at_date(expiry_col, i)
		expiries[i] = f64(w.date_diff(current_date, expiry)) / 365.0 // years to expiry
		implied_vols[i], _ = w.column_at_numeric(iv_col, i)
		market_prices[i], _ = w.column_at_numeric(price_col, i)
	}
	return OptionChain {
			strikes = strikes,
			expiries = expiries,
			implied_vols = implied_vols,
			market_prices = market_prices,
			option_types = option_types,
			n_options = n,
		},
		""
}
