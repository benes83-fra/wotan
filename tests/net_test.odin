package tests

import w "../wotan/core" // your http_get wrapper
import importer "../wotan/importer"
import net "../wotan/net"
import "core:fmt"
import "core:mem"
import "core:strings"

http_get_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	seed_test(#procedure)
	fmt.println("=== HTTP GET TEST ===")

	url := "https://example.com"

	text, ok := net.http_get(url, allocator)
	if !ok {fail_missing_data(#procedure, url); return}

	fmt.println("Received", len(text), "bytes")

	// Basic sanity check
	if !strings.contains(text, "Example Domain") {
		fmt.println("FAIL http_get_test: response does not contain 'Example Domain'")
		record_failures(1)
		return
	}

	fmt.println("OK   http_get_test: response contains 'Example Domain'")

	fmt.println("=== END HTTP GET TEST ===")
}
// yahoo_download_csv: the CSV text reads back (csv_load_from_string) into the
// DataFrame read_yahoo gives for the same request: same columns and types, same
// dates, prices and volumes. The newest day is skipped (its quote moves between
// the two requests while the market is open), and adjusted closes are compared
// to 1e-6 relative (Yahoo recomputes them per request, with float32 noise).
yahoo_finance_csv_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	seed_test(#procedure)
	fmt.println("=== YAHOO FINANCE CSV TEST ===")

	text, ok := net.yahoo_download_csv("MSFT", allocator, .Daily, .OneYear)
	if !ok {fail_missing_data(#procedure, "Yahoo CSV for MSFT"); return}
	defer delete(text, allocator)
	ref, ref_ok := net.read_yahoo("MSFT", .Daily, .OneYear, allocator)
	if !ref_ok {fail_missing_data(#procedure, "Yahoo data for MSFT"); return}
	defer w.destroy_dataframe(&ref)

	df := importer.csv_load_from_string(text, allocator)
	defer w.destroy_dataframe(&df)
	fmt.println(
		"CSV:",
		len(text),
		"bytes,",
		df.rows,
		"rows; header:",
		text[:strings.index_byte(text, '\n')],
	)

	same_shape := len(df.columns) == len(ref.columns) && abs(df.rows - ref.rows) <= 1
	for c in 0 ..< min(len(df.columns), len(ref.columns)) {
		same_shape &&=
			df.columns[c].name == ref.columns[c].name && df.columns[c].type == ref.columns[c].type
	}
	mismatches := 0
	n := min(df.rows, ref.rows) - 1
	if same_shape {
		for r in 0 ..< n {
			for c in 0 ..< len(df.columns) {
				a, b := &df.columns[c], &ref.columns[c]
				if w.is_null(a, r) != w.is_null(b, r) {mismatches += 1; continue}
				if w.is_null(a, r) {continue}
				#partial switch a.type {
				case .Date:
					da, _ := w.column_at_date(a, r)
					db, _ := w.column_at_date(b, r)
					if da != db {mismatches += 1}
				case:
					va, _ := w.column_at_numeric(a, r)
					vb, _ := w.column_at_numeric(b, r)
					// Yahoo recomputes adjusted closes per request (float32 noise, ~1e-7)
					tol := 1e-6 * abs(vb) if a.name == "AdjClose" else 0
					if abs(va - vb) > tol {
						if mismatches <
						   5 {fmt.printf("  row %d %s: %v vs %v\n", r, a.name, va, vb)}
						mismatches += 1
					}
				}
			}
		}
	}
	ok_all := same_shape && n > 200 && mismatches == 0
	fmt.printf(
		"%s yahoo_download_csv reads back as read_yahoo: %d rows x %d columns compared, %d mismatches\n",
		"OK  " if ok_all else "FAIL",
		n,
		len(df.columns),
		mismatches,
	)
	if !ok_all {record_failures(1)}

	fmt.println("=== END YAHOO FINANCE CSV TEST ===")
}

yahoo_finance_json_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	seed_test(#procedure)
	fmt.println("=== YAHOO FINANCE JSON TEST ===")

	df, df_ok := net.read_yahoo("GOOG", .Daily, .OneMonth, allocator)
	if !df_ok {fail_missing_data(#procedure, "Yahoo data for GOOG"); return}
	defer w.destroy_dataframe(&df)

	fmt.println("Rows:", df.rows)
	fmt.println("Columns:", len(df.columns))
	w.df_head(&df, 20)

	fmt.println("=== END YAHOO FINANCE JSON TEST ===")
}

yahoo_finance_events_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	seed_test(#procedure)
	fmt.println("=== YAHOO FINANCE DIVIDENDS & SPLITS TEST ===")

	df, df_ok := net.read_yahoo("AAPL", .Daily, .TenYears, allocator)
	if !df_ok {fail_missing_data(#procedure, "Yahoo data for AAPL"); return}
	defer w.destroy_dataframe(&df)

	fmt.println("Rows:", df.rows)
	fmt.println("Columns:", len(df.columns))
	w.df_head(&df, 50)

	// ------------------------------------------------------------
	// Locate Dividend and Split columns by name
	// ------------------------------------------------------------
	div_col_idx := -1
	split_col_idx := -1

	for i in 0 ..< len(df.columns) {
		if df.columns[i].name == "Dividend" {
			div_col_idx = i
		}
		if df.columns[i].name == "Split" {
			split_col_idx = i
		}
	}

	if div_col_idx < 0 {
		panic("Dividend column missing")
	}
	if split_col_idx < 0 {
		panic("Split column missing")
	}

	div_col := &df.columns[div_col_idx]
	split_col := &df.columns[split_col_idx]

	// ------------------------------------------------------------
	// Scan for actual dividend and split events
	// ------------------------------------------------------------
	div_count := 0
	split_count := 0
	div_data := cast([^]f64)div_col.data
	spl_data := cast([^]f64)split_col.data
	for i in 0 ..< df.rows {
		div := div_data[i]
		spl := spl_data[i]

		if div != 0.0 {
			fmt.println("Dividend event at row", i, "=", div)
			div_count += 1
		}
		if spl != 0.0 {
			fmt.println("Split event at row", i, "=", spl)
			split_count += 1
		}
	}

	fmt.println("Total dividends found:", div_count)
	fmt.println("Total splits found:", split_count)

	if div_count == 0 {
		panic("No dividends detected - expected at least one for AAPL")
	}

	if split_count == 0 {
		fmt.println("Warning: No splits detected in last 5 years (this may be correct)")
	}

	fmt.println("=== END YAHOO FINANCE DIVIDENDS & SPLITS TEST ===")
}
