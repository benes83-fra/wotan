package tests

import w "../wotan/core"
import fin "../wotan/finance"
import imp "../wotan/importer"
import util "../wotan/util"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strings"

// The checked loaders and the parallel CSV reader:
// - every try_ loader gives the same frame as its panicking twin, and ok =
//   false (no panic) for a missing file, text that is not the format, a
//   missing table / sheet, and (CSV) types that do not fit;
// - header names: repeats numbered and blanks named as pandas 1.5.3 does;
// - numbers: what counts as an Int or a Float, against pandas 1.5.3 read_csv
//   (no hex, no "_", "1e400" stays text; an integer beyond i64 becomes Float
//   where pandas uses uint64, and "NaN" in an integer column is a null where
//   pandas makes the column float64);
// - a 2 MB CSV with quoted line breaks, escaped quotes and CRLF: the same
//   frame from one thread and from all (the text is cut into chunks whose
//   boundaries fall inside quoted fields);
// - the finance loaders: try_load_market_data and try_load_option_chain
//   against pandas 1.5.3 and their panicking twins, and ok = false for files
//   that do not fit.

loaders_checked_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Checked Loaders and the Parallel CSV Reader ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	same :: proc(a, b: ^w.DataFrame) -> bool {
		if a.rows != b.rows || len(a.columns) != len(b.columns) {return false}
		for ci in 0 ..< len(a.columns) {
			x, y := &a.columns[ci], &b.columns[ci]
			if x.name != y.name || x.type != y.type {return false}
			for r in 0 ..< a.rows {
				if w.is_null(x, r) != w.is_null(y, r) {return false}
				if w.is_null(x, r) {continue}
				if x.type == .Float {
					u, _ := w.column_at_float(x, r)
					v, _ := w.column_at_float(y, r)
					if transmute(u64)u != transmute(u64)v &&
					   !(math.is_nan(u) && math.is_nan(v)) {return false}
					continue
				}
				if w.column_value_string(x, r) != w.column_value_string(y, r) {return false}
			}
		}
		return true
	}
	texts :: proc(c: ^w.Column) -> string {
		out := ""
		for r in 0 ..< c.len {
			s := "<null>" if w.is_null(c, r) else w.column_value_string(c, r)
			out = fmt.tprintf("%s%s%s", out, "|" if r > 0 else "", s)
		}
		return out
	}
	fails :: proc(ok: bool, df: ^w.DataFrame) -> bool {
		if ok {w.destroy_dataframe(df)}
		return !ok
	}

	// ---------------- try_ twins: same frames
	{
		a := imp.csv_load("demo_files/people.csv")
		defer w.destroy_dataframe(&a)
		b, ok := imp.try_csv_load("demo_files/people.csv")
		defer if ok {w.destroy_dataframe(&b)}
		j1 := imp.json_load("demo_files/example_data.json", allocator = context.allocator)
		defer w.destroy_dataframe(&j1)
		j2, jok := imp.try_json_load("demo_files/example_data.json")
		defer if jok {w.destroy_dataframe(&j2)}
		l1 := imp.jsonl_load("demo_files/example_data2.jsonl", allocator = context.allocator)
		defer w.destroy_dataframe(&l1)
		l2, lok := imp.try_jsonl_load("demo_files/example_data2.jsonl")
		defer if lok {w.destroy_dataframe(&l2)}
		h1 := imp.html_load_table_id("demo_files/test_multi.html", "second", context.allocator)
		defer w.destroy_dataframe(&h1)
		h2, hok := imp.try_html_load_table_id("demo_files/test_multi.html", "second")
		defer if hok {w.destroy_dataframe(&h2)}
		x1 := imp.xlsx_load("demo_files/example.xlsx", context.allocator)
		defer w.destroy_dataframe(&x1)
		x2, xok := imp.try_xlsx_load("demo_files/example.xlsx")
		defer if xok {w.destroy_dataframe(&x2)}
		names, nok := imp.try_xlsx_sheet_names("demo_files/example.xlsx")
		defer if nok {
			for n in names {delete(n)}
			delete(names)
		}
		good :=
			ok &&
			same(&a, &b) &&
			jok &&
			same(&j1, &j2) &&
			lok &&
			same(&l1, &l2) &&
			hok &&
			same(&h1, &h2) &&
			xok &&
			same(&x1, &x2) &&
			nok &&
			len(names) == 1
		expect(
			&passed,
			&total,
			good,
			"try_csv_load, try_json_load, try_jsonl_load, try_html_load_table_id, try_xlsx_load, try_xlsx_sheet_names: the same frames as their panicking twins",
		)
	}

	// ---------------- ok = false instead of a panic
	{
		_ = os.write_entire_file(
			"demo_files/loader_bad.json",
			transmute([]u8)string(`[{"a": 1}, 2]`),
		)
		_ = os.write_entire_file(
			"demo_files/loader_notarray.json",
			transmute([]u8)string(`{"a": 1}`),
		)
		_ = os.write_entire_file("demo_files/loader_empty.json", transmute([]u8)string(`[]`))
		_ = os.write_entire_file(
			"demo_files/loader_broken.json",
			transmute([]u8)string(`[{"a": 1},`),
		)
		_ = os.write_entire_file(
			"demo_files/loader_bad.jsonl",
			transmute([]u8)string("{\"a\": 1}\n[1, 2]\n"),
		)
		df: w.DataFrame
		ok: bool
		c := 0
		n := 0
		check :: proc(c, n: ^int, failed: bool) {
			n^ += 1
			if failed {c^ += 1}
		}
		df, ok = imp.try_csv_load("demo_files/no_such.csv"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_csv_load_from_string(""); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_csv_load_from_string("\r\n\n"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_csv_load_from_string(
			"x,y\n1,2\n",
			types = {.Int},
		); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_csv_load_from_string(
			"x\n1\nabc\n",
			types = {.Int},
		); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_json_load("demo_files/no_such.json"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_json_load("demo_files/loader_broken.json"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_json_load(
			"demo_files/loader_notarray.json",
		); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_json_load("demo_files/loader_empty.json"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_json_load("demo_files/loader_bad.json"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_jsonl_load("demo_files/loader_bad.jsonl"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_jsonl_load("demo_files/loader_empty.json"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_html_load("demo_files/no_such.html"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_html_load_from_string("<p>no table</p>"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_html_load_table_index(
			"demo_files/test_multi.html",
			9,
		); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_html_load_table_id(
			"demo_files/test_multi.html",
			"nope",
		); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_xlsx_load("demo_files/no_such.xlsx"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_xlsx_load("demo_files/people.csv"); check(&c, &n, fails(ok, &df))
		df, ok = imp.try_xlsx_load_sheet(
			"demo_files/example.xlsx",
			"nope",
		); check(&c, &n, fails(ok, &df))
		_, nok := imp.try_xlsx_sheet_names("demo_files/people.csv")
		check(&c, &n, !nok)
		expect(
			&passed,
			&total,
			c == n,
			fmt.tprintf(
				"%d of %d failure cases give ok = false: missing files, empty or non-CSV text, types that do not fit, bad JSON / JSON Lines, no table, no sheet, not a workbook",
				c,
				n,
			),
		)
	}

	// ---------------- header names as pandas makes them
	{
		df, ok := imp.try_csv_load_from_string("a,a,,b,a.1,\n1,2,3,4,5,6\n")
		defer if ok {w.destroy_dataframe(&df)}
		got := ""
		if ok {
			for col, i in df.columns {got = fmt.tprintf("%s%s%s", got, "," if i > 0 else "", col.name)}
		}
		h, hok := imp.try_html_load_from_string(
			"<table><tr><th>x</th><th>x</th><th></th></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>",
		)
		defer if hok {w.destroy_dataframe(&h)}
		hn := ""
		if hok {
			for col, i in h.columns {hn = fmt.tprintf("%s%s%s", hn, "," if i > 0 else "", col.name)}
		}
		expect(
			&passed,
			&total,
			got == "a,a.2,Unnamed: 2,b,a.1,Unnamed: 5" && hn == "x,x.1,Unnamed: 2",
			fmt.tprintf(
				"repeated and blank header names as pandas 1.5.3 (CSV %s; HTML %s)",
				got,
				hn,
			),
		)
	}

	// ---------------- numbers as pandas reads them
	{
		Case :: struct {
			name:  string,
			lines: string,
			type:  w.ColumnType,
			cells: string,
		}
		cases := []Case {
			{"hex", "0x10\n1", .String, "0x10|1"},
			{"underscore", "1_000\n2", .String, "1_000|2"},
			{"signs", "+5\n-3", .Int, "5|-3"},
			{"beyond i64", "9223372036854775808\n1", .Float, "9.223372036854776e+18|1"},
			{"beyond f64", "1e400\n1.5", .String, "1e400|1.5"},
			{"points", ".5\n5.", .Float, "0.5|5"},
			{"infinities", "inf\n-Infinity", .Float, "+Inf|-Inf"},
			{"exponents", "1e5\n2E-3", .Float, "100000|0.002"},
			{"NaN among ints", "NaN\n4", .Int, "<null>|4"},
			{"spaces", " 7 \n8", .Int, "7|8"},
		}
		bad := ""
		for c in cases {
			df, ok := imp.try_csv_load_from_string(fmt.tprintf("x\n%s\n", c.lines))
			if !ok {
				bad = fmt.tprintf("%s %s(failed)", bad, c.name)
				continue
			}
			col := &df.columns[0]
			got := texts(col)
			if col.type != c.type ||
			   got != c.cells {bad = fmt.tprintf("%s %s(%v %s)", bad, c.name, col.type, got)}
			w.destroy_dataframe(&df)
		}
		expect(
			&passed,
			&total,
			bad == "",
			fmt.tprintf(
				"Int / Float / text as pandas 1.5.3 reads them (hex, _, signs, beyond i64 and f64, points, infinities, exponents, NaN, spaces)%s",
				bad,
			),
		)
	}

	// ---------------- chunks: one thread and all threads
	{
		b := strings.builder_make()
		defer strings.builder_destroy(&b)
		strings.write_string(&b, "id,text,x,when\r\n")
		N :: 40_000
		for i in 0 ..< N {
			// every third text has a quoted line break or an escaped quote
			switch i % 3 {
			case 0:
				fmt.sbprintf(
					&b,
					"%d,\"line one, %d\r\nline two\",%d.5,2024-03-01 09:30:%02d.%d\r\n",
					i,
					i,
					i,
					i % 60,
					i % 1000,
				)
			case 1:
				fmt.sbprintf(&b, "%d,\"say \"\"%d\"\"\",,2024-03-02\r\n", i, i)
			case 2:
				fmt.sbprintf(&b, "%d,plain %d,%d,\r\n", i, i, -i)
			}
		}
		text := strings.to_string(b)
		util.parallel_max_threads = 1
		one, ok1 := imp.try_csv_load_from_string(text)
		util.parallel_max_threads = 0
		all, ok2 := imp.try_csv_load_from_string(text)
		defer if ok1 {w.destroy_dataframe(&one)}
		defer if ok2 {w.destroy_dataframe(&all)}
		good := ok1 && ok2 && same(&one, &all) && one.rows == N
		if good {
			t0, _ := w.column_at_string(w.column(&all, "text"), 3)
			t1, _ := w.column_at_string(w.column(&all, "text"), 4)
			good =
				t0 == "line one, 3\r\nline two" &&
				t1 == "say \"4\"" &&
				w.column(&all, "when").type == .String
		}
		expect(
			&passed,
			&total,
			good,
			fmt.tprintf(
				"%d KB of CSV with quoted line breaks: the same %d rows from one thread and from all",
				len(text) / 1000,
				one.rows,
			),
		)
	}

	// ---------------- finance loaders
	{
		_ = os.write_entire_file(
			"demo_files/loader_prices_gap.csv",
			transmute([]u8)string("Date,Close,Volume\n2024-01-02,10.5,100\n2024-01-03,,200\n"),
		)
		_ = os.write_entire_file(
			"demo_files/loader_prices_text.csv",
			transmute([]u8)string("Date,Close,Volume\n2024-01-02,10.5,100\nsoon,11,200\n"),
		)
		_ = os.write_entire_file(
			"demo_files/loader_options_gap.csv",
			transmute([]u8)string(
				"Type,Strike,Expiry,ImpliedVol,Price\ncall,100,2024-03-15,0.2,5\nput,100,2024-03-15,,4\n",
			),
		)
		md1 := fin.load_market_data("data/spy_prices.csv")
		defer fin.free_market_data(&md1)
		md2, mok := fin.try_load_market_data("data/spy_prices.csv")
		defer fin.free_market_data(&md2)
		vol, ret := 0.0, 0.0
		for i in 0 ..< md2.n_obs {
			vol += md2.volumes[i]
			ret += md2.returns[i]
		}
		// pandas 1.5.3: read_csv(parse_dates=["Date"]), Close.pct_change().fillna(0)
		market_ok :=
			mok &&
			md2.n_obs == 30 &&
			md2.dates[0] == w.Date{2023, 1, 3} &&
			md2.dates[29] == w.Date{2023, 2, 14} &&
			abs(md2.returns[29] - 0.003866167388027142) < 1e-15 &&
			abs(ret - 0.07394841593815948) < 1e-14 &&
			vol == 1979398800 &&
			md1.n_obs == 30 &&
			md1.returns[29] == md2.returns[29]
		oc1 := fin.load_option_chain("data/spy_options.csv", md2.dates[29] if mok else {})
		defer fin.free_option_chain(&oc1)
		oc2, cok := fin.try_load_option_chain("data/spy_options.csv", md2.dates[29] if mok else {})
		defer fin.free_option_chain(&oc2)
		strikes, years, ivs, prices := 0.0, 0.0, 0.0, 0.0
		for i in 0 ..< oc2.n_options {
			strikes += oc2.strikes[i]
			years += oc2.expiries[i]
			ivs += oc2.implied_vols[i]
			prices += oc2.market_prices[i]
		}
		// pandas: (Expiry - last Date).days / 365 and the column sums
		chain_ok :=
			cok &&
			oc2.n_options == 12 &&
			oc2.option_types[0] == "call" &&
			oc2.option_types[11] == "put" &&
			abs(oc2.expiries[0] - 0.00821917808219178) < 1e-15 &&
			abs(years - 0.09863013698630134) < 1e-14 &&
			strikes == 4830 &&
			abs(ivs - 1.86) < 1e-12 &&
			abs(prices - 72.66) < 1e-12 &&
			oc1.n_options == 12 &&
			oc1.expiries[5] == oc2.expiries[5]
		failed, n := 0, 0
		for path in ([]string{"data/no_such.csv", "data/spy_options.csv", "demo_files/loader_prices_gap.csv", "demo_files/loader_prices_text.csv"}) {
			n += 1
			md, ok := fin.try_load_market_data(path)
			if ok {fin.free_market_data(&md)} else {failed += 1}
		}
		for path in ([]string{"data/no_such.csv", "data/spy_prices.csv", "demo_files/loader_options_gap.csv"}) {
			n += 1
			oc, ok := fin.try_load_option_chain(path, {2024, 1, 2})
			if ok {fin.free_option_chain(&oc)} else {failed += 1}
		}
		expect(
			&passed,
			&total,
			market_ok && chain_ok && failed == n,
			fmt.tprintf(
				"try_load_market_data / try_load_option_chain: dates, returns, volumes, expiries in years and sums as pandas 1.5.3 computes them, the same as the panicking twins; %d of %d bad files (missing, no Close / Type column, a missing Close / ImpliedVol, a Date that is text) give ok = false",
				failed,
				n,
			),
		)
	}

	fmt.printf("Checked loaders: %d/%d passed\n", passed, total)
}
