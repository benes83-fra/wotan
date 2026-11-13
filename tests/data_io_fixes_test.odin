package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import ex "../wotan/exporter"
import io "../wotan/importer"
import net "../wotan/net"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

// Offline regression checks for the data I/O fixes (docs/data_io_guide.md, section 10):
// Yahoo chart parsing (nested adjclose, nulls, several dividends), Yahoo_Series,
// returns from prices, column names that outlive the importer's temp buffers, and
// exporters that free their cell texts.

// The shape of a Yahoo chart answer: the adjusted closes are nested in an object
// with the same key, day 3 (2024-01-04) has no quote, there are two dividends.
@(private = "file")
_YAHOO_CHART :: `{"chart":{"result":[{"meta":{"currency":"USD","symbol":"TEST","regularMarketPrice":103.0,"chartPreviousClose":99.0},"timestamp":[1704205800,1704292200,1704378600,1704465000],"events":{"dividends":{"1704292200":{"amount":0.5,"date":1704292200},"1704465000":{"amount":0.25,"date":1704465000}},"splits":{"1704378600":{"date":1704378600,"numerator":2.0,"denominator":1.0,"splitRatio":"2:1"}}},"indicators":{"quote":[{"open":[100.0,101.0,null,103.5],"close":[100.5,101.5,null,103.0],"low":[99.5,100.5,null,102.5],"high":[101.0,102.0,null,104.0],"volume":[1000,1100,null,1300]}],"adjclose":[{"adjclose":[99.5,100.6,null,103.0]}]}}],"error":null}}`

data_io_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Data I/O fixes ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	near :: proc(a, b: f64) -> bool {return abs(a - b) <= 1e-12 * max(1, abs(b))}

	// 1. Yahoo chart JSON -> DataFrame
	{
		df := net.yahoo_json_to_dataframe(_YAHOO_CHART, allocator)
		defer w.destroy_dataframe(&df)
		d0, _ := w.column_at_date(w.column(&df, "Date"), 0)
		d3, _ := w.column_at_date(w.column(&df, "Date"), 3)
		expect(
			&passed,
			&total,
			df.rows == 4 && d0 == w.Date{2024, 1, 2} && d3 == w.Date{2024, 1, 5},
			"Yahoo chart: 4 rows from 2024-01-02 to 2024-01-05",
		)

		adj0, adj0_null := w.column_at_float(w.column(&df, "AdjClose"), 0)
		adj3, _ := w.column_at_float(w.column(&df, "AdjClose"), 3)
		expect(
			&passed,
			&total,
			!adj0_null && adj0 == 99.5 && adj3 == 103.0,
			fmt.tprintf(
				"nested adjclose read from the first day on (first %v, last %v)",
				adj0,
				adj3,
			),
		)

		all_null := true
		for name in ([]string{"Open", "High", "Low", "Close", "AdjClose", "Volume"}) {
			_, null := w.column_at_numeric(w.column(&df, name), 2)
			all_null &&= null
		}
		c1, _ := w.column_at_float(w.column(&df, "Close"), 1)
		v3, v3_null := w.column_at_int(w.column(&df, "Volume"), 3)
		expect(
			&passed,
			&total,
			all_null && c1 == 101.5 && !v3_null && v3 == 1300,
			"a day of JSON nulls is a row of nulls; the other days keep their values",
		)

		div1, _ := w.column_at_float(w.column(&df, "Dividend"), 1)
		div3, _ := w.column_at_float(w.column(&df, "Dividend"), 3)
		div0, _ := w.column_at_float(w.column(&df, "Dividend"), 0)
		split2, _ := w.column_at_float(w.column(&df, "Split"), 2)
		expect(
			&passed,
			&total,
			div0 == 0 && div1 == 0.5 && div3 == 0.25 && split2 == 2,
			fmt.tprintf(
				"both dividends (%v, %v) and the 2:1 split (%v) on their days",
				div1,
				div3,
				split2,
			),
		)

		s, ok := net.yahoo_series_from_dataframe(&df, "TEST", allocator)
		defer net.yahoo_series_free(&s)
		expect(
			&passed,
			&total,
			ok &&
			len(s.dates) == 3 &&
			s.dates[2] == w.Date{2024, 1, 5} &&
			s.close[1] == 101.5 &&
			s.adj_close[0] == 99.5 &&
			s.volume[2] == 1300 &&
			s.symbol == "TEST",
			"Yahoo_Series drops the day without a price, arrays aligned",
		)
	}

	// 2. JSON array extraction: null is NaN, not 0
	{
		v := io.extract_json_array_f64(`{"x":[1.5,null,-2]}`, `"x"`)
		expect(
			&passed,
			&total,
			len(v) == 3 && v[0] == 1.5 && math.is_nan(v[1]) && v[2] == -2,
			"extract_json_array_f64: a JSON null becomes NaN",
		)
	}

	// 3. Returns from prices, single series and a row-major panel of two
	{
		p := []f64{100, 50, 110, 40, 121, 60} // [day][series], 3 days x 2 series
		lr := ts.log_returns(p, 2, allocator)
		defer delete(lr, allocator)
		sr := ts.simple_returns(p, 2, allocator)
		defer delete(sr, allocator)
		single := ts.log_returns([]f64{100, 110, 121}, allocator = allocator)
		defer delete(single, allocator)
		short := ts.simple_returns([]f64{100}, allocator = allocator)
		defer delete(short, allocator)
		expect(
			&passed,
			&total,
			len(lr) == 4 &&
			near(lr[0], math.ln(1.1)) &&
			near(lr[1], math.ln(0.8)) &&
			near(lr[2], math.ln(1.1)) &&
			near(lr[3], math.ln(1.5)) &&
			len(sr) == 4 &&
			near(sr[0], 0.1) &&
			near(sr[1], -0.2) &&
			near(sr[2], 0.1) &&
			near(sr[3], 0.5) &&
			len(single) == 2 &&
			near(single[1], math.ln(1.1)) &&
			len(short) == 0,
			"log_returns / simple_returns: per series in a row-major panel, empty for one price",
		)
	}

	// 3b. A panel as a DataFrame: Date plus one adjusted-close column per ticker, with
	// column names that are copies (the names passed in are freed before the lookup)
	{
		panel := net.Yahoo_Panel {
			dates     = []w.Date{{2026, 11, 12}, {2026, 11, 13}},
			adj_close = []f64{100, 50, 101, 49}, // [day][ticker]
			n_tickers = 2,
		}
		a := fmt.aprintf("A%d", 1, allocator = allocator)
		b := fmt.aprintf("B%d", 2, allocator = allocator)
		df := net.yahoo_panel_to_dataframe(&panel, []string{a, b}, allocator)
		defer w.destroy_dataframe(&df)
		delete(a, allocator)
		delete(b, allocator)
		d1, _ := w.column_at_date(w.column(&df, "Date"), 1)
		a1, _ := w.column_at_float(w.column(&df, "A1"), 1)
		b0, _ := w.column_at_float(w.column(&df, "B2"), 0)
		expect(
			&passed,
			&total,
			df.rows == 2 &&
			len(df.columns) == 3 &&
			d1 == w.Date{2026, 11, 13} &&
			a1 == 101 &&
			b0 == 50,
			"yahoo_panel_to_dataframe: Date and one column per ticker under its own name",
		)
	}

	// 4. Column names outlive the importer's temporary buffers. The importers are
	// run with a temp allocator on a local buffer, which is then overwritten.
	{
		buf := make([]u8, 1 << 20, allocator)
		defer delete(buf, allocator)
		scratch: mem.Arena
		mem.arena_init(&scratch, buf)

		csv_df, html_df: w.DataFrame
		{
			context.temp_allocator = mem.arena_allocator(&scratch)
			csv_df = io.csv_load_from_string("ticker,close\nAAPL,185.5\n", allocator)
			html_df = io.html_load_from_string(
				"<table><tr><th>name</th><th>age</th></tr><tr><td>Ann</td><td>30</td></tr></table>",
				allocator,
			)
		}
		defer w.destroy_dataframe(&csv_df)
		defer w.destroy_dataframe(&html_df)
		mem.set(raw_data(buf), '#', len(buf))
		expect(
			&passed,
			&total,
			csv_df.columns[0].name == "ticker" &&
			csv_df.columns[1].name == "close" &&
			html_df.columns[0].name == "name" &&
			html_df.columns[1].name == "age" &&
			w.column(&csv_df, "close") != nil,
			fmt.tprintf(
				"column names survive the release of the temp allocator (%q, %q)",
				csv_df.columns[0].name,
				html_df.columns[1].name,
			),
		)
	}

	// 5. Writers free the text of every cell (tracked: nothing left after the calls)
	{
		df := io.csv_load_from_string("d,x,n,s\n2024-01-02,1.25,7,a\n2024-01-03,,8,b\n", allocator)
		defer w.destroy_dataframe(&df)
		track: mem.Tracking_Allocator
		// on the heap: an arena (the temp allocator) does not implement free, so the
		// tracker would never see an allocation released
		mem.tracking_allocator_init(&track, runtime.heap_allocator())
		defer mem.tracking_allocator_destroy(&track)
		files := []string {
			"io_fix_test.csv",
			"io_fix_test.json",
			"io_fix_test.jsonl",
			"io_fix_test.html",
			"io_fix_test.xlsx",
		}
		{
			context.allocator = mem.tracking_allocator(&track)
			ta := context.allocator
			ok_csv := ex.csv_save(&df, files[0], allocator = ta)
			ex.json_write(&df, files[1], ta)
			ex.jsonl_write(&df, files[2], ta)
			ex.html_write(&df, files[3], ta)
			ok_xlsx := ex.xlsx_save(&df, files[4], ta)
			if !ok_csv || !ok_xlsx {fmt.println("  (a writer reported failure)")}
		}
		for f in files {_ = os.remove(f)}
		expect(
			&passed,
			&total,
			len(track.allocation_map) == 0,
			fmt.tprintf(
				"csv / json / jsonl / html / xlsx writers leave %d allocations",
				len(track.allocation_map),
			),
		)
	}

	fmt.printf("Data I/O fixes: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
