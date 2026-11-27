package tests

import w "../wotan/core"
import ex "../wotan/exporter"
import imp "../wotan/importer"
import util "../wotan/util"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"

// parquet_info and filtered parquet_load:
// - parquet_info against pyarrow 14's metadata: rows, row groups, column
//   types, and the statistics (null count, min, max) of files written by
//   pyarrow and DuckDB, with nil where they cannot be used (UUID, INT96,
//   nested columns);
// - filters: a 600-row frame written in 10 row groups and read with 16
//   filters (dates, strings, floats, ints against f64 bounds, open and
//   exclusive ends, two filters, a filter column that is not loaded); each
//   result equals pyarrow 14 read_table(filters=...) on the same file;
// - row groups are skipped: a corrupt row group outside the filter's range is
//   never read;
// - bad filters and unreadable files give ok = false;
// - 200,000 rows give the same result on one thread and on all.

parquet_filter_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Parquet Info and Filters ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	F :: imp.Parquet_Filter
	D :: proc(y, m, d: i32) -> w.Date {return {y, m, d}}

	// ---------------- parquet_info against pyarrow
	{
		info, ok := imp.parquet_info("demo_files/parquet/types_delta_pages.parquet")
		defer imp.destroy_parquet_info(&info)
		fails := 0
		col :: proc(info: ^imp.Parquet_Info, name: string) -> int {
			for c, i in info.columns {if c.name == name {return i}}
			return -1
		}
		st :: proc(info: ^imp.Parquet_Info, g: int, name: string) -> imp.Parquet_Column_Stats {
			return info.row_groups[g].stats[col(info, name)]
		}
		check :: proc(fails: ^int, ok: bool, what: string) {
			if !ok {
				fails^ += 1
				fmt.println("    mismatch:", what)
			}
		}
		check(&fails, ok && info.rows == 12 && len(info.row_groups) == 3, "rows / row groups")
		if ok && len(info.row_groups) == 3 {
			g := info.row_groups
			check(&fails, g[0].rows == 5 && g[1].rows == 5 && g[2].rows == 2, "group rows")
			check(&fails, info.created_by == "parquet-cpp-arrow version 14.0.2", "created_by")
			c := info.columns
			check(&fails, len(c) == 21, "21 columns")
			check(
				&fails,
				c[col(&info, "s")].type == .String &&
				c[col(&info, "s")].parquet_type == "BYTE_ARRAY STRING",
				"s type",
			)
			check(
				&fails,
				c[col(&info, "u64")].type == .Int &&
				c[col(&info, "u64")].parquet_type == "INT64 INT(64, unsigned)",
				"u64 type",
			)
			check(
				&fails,
				c[col(&info, "dec_big")].type == .Float &&
				c[col(&info, "dec_big")].parquet_type == "FIXED_LEN_BYTE_ARRAY(13) DECIMAL(30, 4)",
				"dec_big type",
			)
			check(
				&fails,
				c[col(&info, "ts_ns_utc")].type == .Datetime &&
				c[col(&info, "ts_ns_utc")].parquet_type == "INT64 TIMESTAMP(NANOS, UTC)",
				"ts_ns_utc type",
			)
			check(
				&fails,
				c[col(&info, "t_ms")].type == .Time &&
				c[col(&info, "t_ms")].parquet_type == "INT32 TIME(MILLIS, UTC)",
				"t_ms type",
			)

			s := st(&info, 0, "i64")
			check(
				&fails,
				s.null_count == 0 &&
				s.min == imp.Parquet_Value(min(int)) &&
				s.max == imp.Parquet_Value(4611686018427387904),
				"i64 group 0",
			)
			s = st(&info, 1, "i64")
			check(
				&fails,
				s.null_count == 2 &&
				s.min == imp.Parquet_Value(7) &&
				s.max == imp.Parquet_Value(max(int)),
				"i64 group 1",
			)
			s = st(&info, 0, "u32")
			check(
				&fails,
				s.null_count == 1 &&
				s.min == imp.Parquet_Value(0) &&
				s.max == imp.Parquet_Value(4294967295),
				"u32 group 0",
			)
			s = st(&info, 0, "f64")
			lo, _ := s.min.(f64)
			check(
				&fails,
				lo == 0 && math.sign_bit(lo) && s.max == imp.Parquet_Value(math.inf_f64(1)),
				"f64 group 0: -0 .. +inf",
			)
			s = st(&info, 0, "s")
			check(
				&fails,
				s.min == imp.Parquet_Value("") && s.max == imp.Parquet_Value("日本語"),
				"s group 0",
			)
			s = st(&info, 1, "d")
			check(
				&fails,
				s.min == imp.Parquet_Value(D(1, 1, 1)) &&
				s.max == imp.Parquet_Value(D(9999, 12, 31)),
				"d group 1",
			)
			s = st(&info, 0, "ts_ns_utc")
			check(
				&fails,
				s.min == imp.Parquet_Value(w.Datetime{1969, 12, 31, 23, 59, 59, 500_000_000}) &&
				s.max == imp.Parquet_Value(w.Datetime{2024, 3, 1, 9, 30, 15, 999_999_000}),
				"ts_ns_utc group 0",
			)
			s = st(&info, 2, "t_ms")
			check(
				&fails,
				s.null_count == 1 &&
				s.min == imp.Parquet_Value(w.Time{16, 17, 18, 0}) &&
				s.max == s.min,
				"t_ms group 2",
			)
			s = st(&info, 0, "dec")
			check(
				&fails,
				s.min == imp.Parquet_Value(-99999999.99) &&
				s.max == imp.Parquet_Value(99999999.99),
				"dec group 0",
			)
			s = st(&info, 1, "b")
			check(
				&fails,
				s.min == imp.Parquet_Value(false) && s.max == imp.Parquet_Value(true),
				"b group 1",
			)
		}

		duck, dok := imp.parquet_info("demo_files/parquet/duckdb_zstd.parquet")
		defer imp.destroy_parquet_info(&duck)
		check(&fails, dok && duck.rows == 500, "duckdb rows")
		if dok {
			check(&fails, st(&duck, 0, "uid").min == nil, "UUID: no min / max")
			s := st(&duck, 0, "amount")
			check(
				&fails,
				s.min == imp.Parquet_Value(0.0) && s.max == imp.Parquet_Value(623.75),
				"amount (DECIMAL on INT64)",
			)
			s = st(&duck, 0, "half")
			check(
				&fails,
				s.null_count == 100 &&
				s.min == imp.Parquet_Value(0.5) &&
				s.max == imp.Parquet_Value(249.5),
				"half (DECIMAL on 16 bytes)",
			)
			s = st(&duck, 0, "ts")
			check(
				&fails,
				s.min == imp.Parquet_Value(w.Datetime{2024, 1, 1, 9, 30, 0, 0}) &&
				s.max == imp.Parquet_Value(w.Datetime{2024, 1, 1, 17, 57, 19, 0}),
				"ts",
			)
		}
		dec, cok := imp.parquet_info("demo_files/parquet/decimals_duckdb.parquet")
		defer imp.destroy_parquet_info(&dec)
		if cok {
			s := st(&dec, 0, "d9")
			check(
				&fails,
				s.null_count == 1 &&
				s.min == imp.Parquet_Value(-1234567.89) &&
				s.max == imp.Parquet_Value(1.25),
				"d9 (DECIMAL on INT32)",
			)
		} else {check(&fails, false, "decimals_duckdb")}
		i96, iok := imp.parquet_info("demo_files/parquet/types_int96.parquet")
		defer imp.destroy_parquet_info(&i96)
		check(
			&fails,
			iok &&
			len(i96.row_groups) == 1 &&
			i96.columns[1].type == .Datetime &&
			i96.columns[1].parquet_type == "INT96" &&
			i96.row_groups[0].stats[1].min == nil &&
			i96.row_groups[0].stats[0].min == imp.Parquet_Value(int(min(i32))),
			"INT96: a Datetime column without min / max",
		)
		nest, nok := imp.parquet_info("demo_files/parquet/nested.parquet")
		defer imp.destroy_parquet_info(&nest)
		if nok && len(nest.columns) == 3 {
			t := nest.columns[1]
			check(
				&fails,
				t.name == "tags" &&
				t.type == .Invalid &&
				t.parquet_type == "LIST" &&
				t.reason != "" &&
				nest.row_groups[0].stats[1].null_count == -1,
				"nested column listed, not loadable",
			)
			check(
				&fails,
				nest.columns[2].type == .Float && nest.columns[2].reason == "",
				"the column after it",
			)
		} else {check(&fails, false, "nested.parquet")}
		br, bok := imp.parquet_info("demo_files/parquet/brotli.parquet")
		defer imp.destroy_parquet_info(&br)
		check(&fails, bok && br.rows == 12, "a Brotli file: the footer still reads")
		expect(
			&passed,
			&total,
			fails == 0,
			"parquet_info as pyarrow 14 reads the metadata: rows, row groups, column types; null count / min / max per row group for ints (signed, unsigned, extremes), -0 / inf, strings, dates, timestamps, times, decimals, bools; none for UUID, INT96 and nested columns",
		)
	}

	// ---------------- filters against pyarrow
	frame := _pqf_frame(600)
	defer w.destroy_dataframe(&frame)
	data, wok := ex.parquet_bytes(&frame, row_group_rows = 60)
	defer delete(data)
	{
		Case :: struct {
			name:    string,
			columns: []string,
			filters: []F,
			rows:    int,
			digest:  string,
		}
		cases := []Case {
			{
				"all",
				nil,
				nil,
				600,
				"date:2024-01-01..2024-07-18:0;ticker:AAPL..NVDA:0;close:67397.5000:35:25;volume:1793829:21;ts:2024-01-01T09:30:00.000000000..2024-07-18T09:50:00.000599000:0;flag:120:0",
			},
			{
				"date_range",
				nil,
				{{column = "date", lower = D(2024, 3, 1), upper = D(2024, 3, 31)}},
				93,
				"date:2024-03-01..2024-03-31:0;ticker:AAPL..NVDA:0;close:10647.0000:5:3;volume:232191:3;ts:2024-03-01T09:30:00.000180000..2024-03-31T09:50:00.000272000:0;flag:19:0",
			},
			{
				"ticker_eq",
				{"date", "close"},
				{{column = "ticker", lower = "MSFT", upper = "MSFT"}},
				200,
				"date:2024-01-01..2024-07-18:0;close:22379.5000:12:9",
			},
			{
				"close_range_int",
				{"close"},
				{{column = "close", lower = 120, upper = 130}},
				112,
				"close:13996.5000:0:0",
			},
			{
				"close_lower_only",
				{"close"},
				{{column = "close", lower = 149.5}},
				10,
				"close:1497.5000:0:0",
			},
			{"close_open", {"close"}, {{column = "close"}}, 540, "close:67397.5000:0:0"},
			{
				"vol_f64",
				{"volume"},
				{{column = "volume", lower = 4000.5, upper = 4100.0}},
				13,
				"volume:52613:0",
			},
			{
				"vol_excl",
				{"volume"},
				{{column = "volume", lower = 1007, upper = 1070, inclusive = .Neither}},
				7,
				"volume:7287:0",
			},
			{
				"vol_left",
				{"volume"},
				{{column = "volume", lower = 1007, upper = 1070, inclusive = .Left}},
				8,
				"volume:8294:0",
			},
			{
				"vol_right",
				{"volume"},
				{{column = "volume", lower = 1007, upper = 1070, inclusive = .Right}},
				8,
				"volume:8357:0",
			},
			{
				"ts_lt",
				{"ts"},
				{
					{
						column = "ts",
						upper = w.Datetime{2024, 1, 3, 9, 40, 0, 7000},
						inclusive = .Left,
					},
				},
				7,
				"ts:2024-01-01T09:30:00.000000000..2024-01-03T09:30:00.000006000:0",
			},
			{
				"two",
				{"ticker", "date", "close"},
				{
					{column = "date", lower = D(2024, 5, 1), upper = D(2024, 5, 10)},
					{column = "ticker", lower = "AAPL", upper = "MSFT"},
				},
				20,
				"ticker:AAPL..MSFT:0;date:2024-05-01..2024-05-10:0;close:2252.5000:1:1",
			},
			{
				"flag",
				{"date"},
				{
					{column = "flag", lower = true, upper = true},
					{column = "date", upper = D(2024, 1, 20)},
				},
				12,
				"date:2024-01-01..2024-01-19:0",
			},
			{
				"empty",
				nil,
				{{column = "date", lower = D(2030, 1, 1)}},
				0,
				"date:-:0;ticker:-:0;close:-:0;volume:-:0;ts:-:0;flag:-:0",
			},
			{
				"reversed",
				nil,
				{{column = "date", lower = D(2024, 3, 1), upper = D(2024, 2, 1)}},
				0,
				"date:-:0;ticker:-:0;close:-:0;volume:-:0;ts:-:0;flag:-:0",
			},
			{
				"str_range",
				{"ticker"},
				{{column = "ticker", lower = "B", upper = "N"}},
				200,
				"ticker:MSFT..MSFT:0",
			},
		}
		good := 0
		for c in cases {
			df, ok := imp.parquet_load_from_bytes(data, c.columns, filters = c.filters)
			got := _pqf_digest(&df) if ok else "<failed>"
			if ok && df.rows == c.rows && got == c.digest {
				good += 1
			} else {
				fmt.printfln("    %s: %d rows %s", c.name, df.rows, got)
			}
			if ok {w.destroy_dataframe(&df)}
		}
		expect(
			&passed,
			&total,
			wok && good == len(cases),
			fmt.tprintf(
				"%d of %d filters give the rows pyarrow 14 read_table(filters=...) gives: date range, string equality and range, float range with int bounds, int column with f64 bounds, open / exclusive ends, two filters, a filter column not loaded, nulls and NaN dropped, empty results",
				good,
				len(cases),
			),
		)
	}

	// ---------------- row groups outside the range are not read
	{
		info, iok := imp.parquet_info_from_bytes(data)
		defer imp.destroy_parquet_info(&info)
		g0 :=
			info.row_groups[0] if iok && len(info.row_groups) == 10 else imp.Parquet_Row_Group_Info{}
		date_ok :=
			iok &&
			len(info.row_groups) == 10 &&
			g0.stats[0].min == imp.Parquet_Value(D(2024, 1, 1)) &&
			g0.stats[0].max == imp.Parquet_Value(D(2024, 1, 20)) &&
			info.row_groups[1].stats[0].min == imp.Parquet_Value(D(2024, 1, 21))
		broken := make([]u8, len(data))
		defer delete(broken)
		copy(broken, data)
		for k in g0.offset ..< g0.offset + g0.compressed_bytes {broken[k] = 0xff}
		fmt.println("    (a corrupt row group; the parquet error below is expected)")
		full, full_ok := imp.parquet_load_from_bytes(broken)
		if full_ok {w.destroy_dataframe(&full)}
		f := []F{{column = "date", lower = D(2024, 1, 21)}}
		got, got_ok := imp.parquet_load_from_bytes(broken, filters = f)
		want, want_ok := imp.parquet_load_from_bytes(data, filters = f)
		same := got_ok && want_ok && got.rows == 540 && _pqf_digest(&got) == _pqf_digest(&want)
		if got_ok {w.destroy_dataframe(&got)}
		if want_ok {w.destroy_dataframe(&want)}
		expect(
			&passed,
			&total,
			date_ok && !full_ok && same,
			"row groups are skipped by their statistics: with row group 0 (2024-01-01 .. 2024-01-20) destroyed, the full read fails and date >= 2024-01-21 still reads its 540 rows",
		)
	}

	// ---------------- ok = false
	{
		failed, n := 0, 0
		try :: proc(failed, n: ^int, data: []u8, cols: []string, f: []F) {
			n^ += 1
			df, ok := imp.parquet_load_from_bytes(data, cols, filters = f)
			if ok {w.destroy_dataframe(&df)} else {failed^ += 1}
		}
		try(&failed, &n, data, nil, {{column = "date", lower = "2024-01-01"}})
		try(&failed, &n, data, nil, {{column = "nope", lower = 1}})
		try(&failed, &n, data, nil, {{column = "close", lower = math.nan_f64()}})
		try(&failed, &n, data, nil, {{column = "flag", lower = 1}})
		try(&failed, &n, data, {"close"}, {{column = "volume", upper = "x"}})
		n += 1
		if df, ok := imp.parquet_load(
			"demo_files/parquet/nested.parquet",
			{"id"},
			filters = {{column = "tags", lower = 1}},
		); ok {w.destroy_dataframe(&df)} else {failed += 1}
		n += 1
		if i, ok := imp.parquet_info("demo_files/parquet/no_such.parquet");
		   ok {imp.destroy_parquet_info(&i)} else {failed += 1}
		n += 1
		if i, ok := imp.parquet_info_from_bytes(transmute([]u8)string("PAR1 not really PAR1"));
		   ok {imp.destroy_parquet_info(&i)} else {failed += 1}
		expect(
			&passed,
			&total,
			failed == n,
			fmt.tprintf(
				"%d of %d give ok = false: a bound of another type (string for a Date, int for a Bool, string for an Int), a NaN bound, a missing or nested filter column, a missing file, bytes that are not Parquet",
				failed,
				n,
			),
		)
	}

	// ---------------- one thread and all threads
	{
		big := _pqf_frame(200_000)
		defer w.destroy_dataframe(&big)
		bytes, bok := ex.parquet_bytes(&big, row_group_rows = 1 << 14)
		defer delete(bytes)
		f := []F {
			{column = "ticker", lower = "MSFT", upper = "NVDA"},
			{column = "close", lower = 110, upper = 140, inclusive = .Right},
		}
		saved := util.parallel_max_threads
		util.parallel_max_threads = 1
		one, ok1 := imp.parquet_load_from_bytes(bytes, filters = f)
		util.parallel_max_threads = saved
		all, ok2 := imp.parquet_load_from_bytes(bytes, filters = f)
		// expected: every row of the frame that matches, in order
		want := 0
		for r in 0 ..< big.rows {
			t, _ := w.column_at_string(&big.columns[1], r)
			c, missing := w.column_at_numeric(&big.columns[2], r)
			if (t == "MSFT" || t == "NVDA") && !missing && c > 110 && c <= 140 {want += 1}
		}
		same :=
			ok1 &&
			ok2 &&
			one.rows == want &&
			all.rows == want &&
			_pqf_digest(&one) == _pqf_digest(&all)
		if ok1 {w.destroy_dataframe(&one)}
		if ok2 {w.destroy_dataframe(&all)}
		expect(
			&passed,
			&total,
			bok && same,
			fmt.tprintf(
				"200,000 rows in 13 row groups, two filters: the same %d rows on one thread and on all",
				want,
			),
		)
	}

	fmt.printf("Parquet info and filters: %d/%d passed\n", passed, total)
}

// n rows of three tickers on consecutive days (sorted by date): close with
// nulls and NaN, volume with nulls, a timestamp and a flag
@(private = "file")
_pqf_frame :: proc(n: int) -> w.DataFrame {
	df := w.dataframe_new()
	date := w.column_new("date", .Date, n)
	ticker := w.column_new("ticker", .String, n)
	close := w.column_new("close", .Float, n)
	volume := w.column_new("volume", .Int, n)
	ts := w.column_new("ts", .Datetime, n)
	flag := w.column_new("flag", .Bool, n)
	TICKERS := [3]string{"AAPL", "MSFT", "NVDA"}
	base := w.days_from_civil(2024, 1, 1)
	for i in 0 ..< n {
		d := w.civil_from_days(base + i64(i / 3))
		w.append_date(&date, d)
		w.append_string(&ticker, TICKERS[i % 3])
		switch {
		case i % 17 == 5:
			w.append_null(&close)
		case i % 23 == 7:
			w.append_float(&close, math.nan_f64())
		case:
			w.append_float(&close, 100 + f64(i * 37 % 101) * 0.5)
		}
		if i % 29 == 3 {w.append_null(&volume)} else {w.append_int(&volume, 1000 + i * 7)}
		w.append_datetime(
			&ts,
			w.Datetime{d.year, d.month, d.day, 9, i32(30 + (i % 3) * 10), 0, i32(i * 1000)},
		)
		w.append_bool(&flag, i % 5 == 0)
	}
	w.add_column(&df, date)
	w.add_column(&df, ticker)
	w.add_column(&df, close)
	w.add_column(&df, volume)
	w.add_column(&df, ts)
	w.add_column(&df, flag)
	df.rows = n
	return df
}

// Per column: sum (numbers, true count for Bool) or first..last present
// value, then the null count (and NaN count for Float); as the pyarrow
// reference prints it
@(private = "file")
_pqf_digest :: proc(df: ^w.DataFrame) -> string {
	sb := strings.builder_make(context.temp_allocator)
	for &c, ci in df.columns {
		if ci > 0 {strings.write_byte(&sb, ';')}
		nulls, nans, present := 0, 0, 0
		isum, fsum := 0, 0.0
		first, last := -1, -1
		for r in 0 ..< df.rows {
			if w.is_null(&c, r) {
				nulls += 1
				continue
			}
			present += 1
			if first < 0 {first = r}
			last = r
			#partial switch c.type {
			case .Int:
				v, _ := w.column_at_int(&c, r)
				isum += v
			case .Bool:
				v, _ := w.column_at_bool(&c, r)
				isum += int(v)
			case .Float:
				v, _ := w.column_at_float(&c, r)
				if v != v {nans += 1} else {fsum += v}
			}
		}
		text :: proc(c: ^w.Column, r: int) -> string {
			#partial switch c.type {
			case .Date:
				v, _ := w.column_at_date(c, r)
				return fmt.tprintf("%04d-%02d-%02d", v.year, v.month, v.day)
			case .Datetime:
				v, _ := w.column_at_datetime(c, r)
				return fmt.tprintf(
					"%04d-%02d-%02dT%02d:%02d:%02d.%09d",
					v.year,
					v.month,
					v.day,
					v.hour,
					v.minute,
					v.second,
					v.nanosecond,
				)
			case .String:
				v, _ := w.column_at_string(c, r)
				return v
			}
			return "?"
		}
		switch {
		case present == 0:
			fmt.sbprintf(&sb, "%s:-:%d", c.name, nulls)
		case c.type == .Int || c.type == .Bool:
			fmt.sbprintf(&sb, "%s:%d:%d", c.name, isum, nulls)
		case c.type == .Float:
			fmt.sbprintf(&sb, "%s:%.4f:%d:%d", c.name, fsum, nulls, nans)
		case:
			fmt.sbprintf(&sb, "%s:%s..%s:%d", c.name, text(&c, first), text(&c, last), nulls)
		}
	}
	return strings.to_string(sb)
}
