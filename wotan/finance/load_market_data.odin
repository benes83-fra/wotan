package finance


import w "../core"
import csv "../importer"
import "core:fmt"
import "core:mem"

// Load historical price data (free with free_market_data)
load_market_data :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> MarketData {
	df := csv.csv_load(path)

	n := df.rows
	dates := make([]w.Date, n, allocator)
	prices := make([]f64, n, allocator)
	volumes := make([]f64, n, allocator)
	returns := make([]f64, n, allocator)

	for i in 0 ..< n {
		date_col := w.column(&df, "Date")
		close_col := w.column(&df, "Close")
		volume_col := w.column(&df, "Volume")

		dates[i], _ = w.column_at_date(date_col, i)
		prices[i], _ = w.column_at_float(close_col, i)
		volumes[i], _ = w.column_at_numeric(volume_col, i)

		if i > 0 {
			returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
		} else {
			returns[i] = 0.0
		}
	}

	w.destroy_dataframe(&df)

	return MarketData {
		dates = dates,
		prices = prices,
		volumes = volumes,
		returns = returns,
		n_obs = n,
	}
}
// Load option chain data. The Type column is read case-insensitively by its
// first letter into the literals "call" / "put" ("" for anything else).
// Free with free_option_chain.
load_option_chain :: proc(
	path: string,
	current_date: w.Date,
	allocator: mem.Allocator = context.allocator,
) -> OptionChain {
	// Force specific types for option data
	types := []w.ColumnType {
		.String, // Symbol
		.String, // Type
		.Float, // Strike (force to Float)
		.Date, // Expiry
		.Float, // ImpliedVol
		.Float, // Price
		.Float, // Bid
		.Float, // Ask
		.Float, // Volume
		.Float, // OpenInterest
	}

	df := csv.csv_load(path, types)

	n := df.rows
	strikes := make([]f64, n, allocator)
	expiries := make([]f64, n, allocator)
	implied_vols := make([]f64, n, allocator)
	market_prices := make([]f64, n, allocator)
	option_types := make([]string, n, allocator)

	for i in 0 ..< n {
		type_col := w.column(&df, "Type")
		strike_col := w.column(&df, "Strike")
		expiry_col := w.column(&df, "Expiry")
		iv_col := w.column(&df, "ImpliedVol")
		price_col := w.column(&df, "Price")

		// Literals, not views into the DataFrame destroyed below
		type_str, _ := w.column_at_string(type_col, i)
		switch {
		case len(type_str) > 0 && (type_str[0] == 'c' || type_str[0] == 'C'):
			option_types[i] = "call"
		case len(type_str) > 0 && (type_str[0] == 'p' || type_str[0] == 'P'):
			option_types[i] = "put"
		case:
			option_types[i] = ""
		}
		strikes[i], _ = w.column_at_float(strike_col, i)

		// Use column_at_date for already-parsed Date
		expiry_date, _ := w.column_at_date(expiry_col, i)

		// Calculate time to expiry in years
		days_to_expiry := w.date_diff(current_date, expiry_date)
		expiries[i] = f64(days_to_expiry) / 365.0

		implied_vols[i], _ = w.column_at_float(iv_col, i)
		market_prices[i], _ = w.column_at_float(price_col, i)
	}

	w.destroy_dataframe(&df)

	return OptionChain {
		strikes = strikes,
		expiries = expiries,
		implied_vols = implied_vols,
		market_prices = market_prices,
		option_types = option_types,
		n_options = n,
	}
}
