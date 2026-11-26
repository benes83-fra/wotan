package tests

import w "../wotan/core"
import ex "../wotan/exporter"
import imp "../wotan/importer"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

// Parquet reader and writer.
//
// Reader, against pyarrow 14.0.2 on the fixtures in demo_files/parquet
// (written by scripts/make_parquet_fixtures.py):
// - one 12-row table with every mapped type (ints of all widths, unsigned,
//   floats with NaN / -0 / inf, bools, UTF-8 and binary strings, fixed-length
//   binary, dates back to year 1, timestamps in ms / us / ns (UTC) before and
//   after 1970, times in ms / us / ns, decimals up to 30 digits) and nulls,
//   written five ways: PLAIN v1 uncompressed, dictionary v1 Snappy,
//   dictionary v2 gzip, PLAIN v2 LZ4 with RLE booleans, and three row groups
//   of many small pages whose dictionaries fall back to PLAIN mid-chunk.
//   Every cell must equal pyarrow's (sub-second parts dropped);
// - INT96 timestamps; decimals stored as INT32 / INT64 / FIXED_LEN_BYTE_ARRAY
//   (DuckDB); a pandas file with its index; column selection and order;
// - ok = false (no crash) for nested columns, Zstandard, a missing file, a
//   CSV file, unknown or repeated column names, truncated and corrupted bytes.
//
// Writer: the same table written with every codec, dictionary on and off, in
// one or three row groups, read back cell for cell; a file on disk (which
// pyarrow also reads: scripts/check_parquet_export.py); a view; an empty
// frame; a hidden index; 70 000 rows over several pages and Snappy fragments,
// with an all-null column.

parquet_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Parquet Reader and Writer ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	NEG_ZERO := math.copy_sign(f64(0), -1)
	Expect :: struct {
		name:   string,
		type:   w.ColumnType,
		texts:  []string, // non-float columns: column_value_string of each cell
		nulls:  []bool, // float columns
		floats: []f64,
		rel:    f64, // relative tolerance (0: bit-exact, NaN equal to NaN)
	}
	// every column of `df`, in order, as `exp` says; the first mismatch is printed
	matches :: proc(df: ^w.DataFrame, exp: []Expect) -> bool {
		if len(df.columns) != len(exp) {
			fmt.printf("    %d columns, expected %d\n", len(df.columns), len(exp))
			return false
		}
		for &e, ci in exp {
			c := &df.columns[ci]
			if c.name != e.name || c.type != e.type {
				fmt.printf(
					"    column %d: %s %v, expected %s %v\n",
					ci,
					c.name,
					c.type,
					e.name,
					e.type,
				)
				return false
			}
			n := len(e.texts) if e.type != .Float else len(e.floats)
			if c.len != n || df.rows != n {
				fmt.printf("    %s: %d rows, expected %d\n", e.name, c.len, n)
				return false
			}
			for r in 0 ..< n {
				if e.type != .Float {
					got := w.column_value_string(c, r)
					if got != e.texts[r] {
						fmt.printf("    %s[%d] = %q, expected %q\n", e.name, r, got, e.texts[r])
						return false
					}
					continue
				}
				null := w.is_null(c, r)
				if null != e.nulls[r] {
					fmt.printf("    %s[%d] null = %v, expected %v\n", e.name, r, null, e.nulls[r])
					return false
				}
				if null {continue}
				got, _ := w.column_at_float(c, r)
				want := e.floats[r]
				same: bool
				switch {
				case math.is_nan(want):
					same = math.is_nan(got)
				case e.rel == 0:
					same = transmute(u64)got == transmute(u64)want
				case:
					same = abs(got - want) <= e.rel * abs(want)
				}
				if !same {
					fmt.printf("    %s[%d] = %v, expected %v\n", e.name, r, got, want)
					return false
				}
			}
		}
		return true
	}
	path :: proc(name: string) -> string {
		return fmt.tprintf("demo_files/parquet/%s.parquet", name)
	}

	// ---------------- one table, five layouts
	types_exp := []Expect {
		{
			name = "i64",
			type = .Int,
			texts = {
				"0",
				"1",
				"-1",
				"4611686018427387904",
				"-9223372036854775808",
				"9223372036854775807",
				"NULL",
				"42",
				"7",
				"NULL",
				"-5",
				"123456789012",
			},
		},
		{
			name = "i32",
			type = .Int,
			texts = {
				"0",
				"1",
				"-1",
				"2147483647",
				"-2147483648",
				"5",
				"6",
				"7",
				"8",
				"9",
				"10",
				"11",
			},
		},
		{
			name = "i8",
			type = .Int,
			texts = {"-128", "127", "0", "NULL", "1", "2", "3", "4", "5", "6", "7", "8"},
		},
		{
			name = "u32",
			type = .Int,
			texts = {
				"0",
				"4294967295",
				"2147483648",
				"1",
				"NULL",
				"3",
				"4",
				"5",
				"6",
				"7",
				"8",
				"9",
			},
		},
		{
			name = "u64",
			type = .Int,
			texts = {
				"0",
				"4611686018427387904",
				"1",
				"NULL",
				"3",
				"4",
				"5",
				"6",
				"7",
				"8",
				"9",
				"10",
			},
		},
		{
			name = "f64",
			type = .Float,
			nulls = {
				false,
				false,
				true,
				false,
				false,
				false,
				false,
				false,
				false,
				true,
				false,
				false,
			},
			floats = {
				1.5,
				math.nan_f64(),
				0,
				NEG_ZERO,
				math.inf_f64(1),
				math.inf_f64(-1),
				1e-300,
				3.141592653589793,
				0.1,
				0,
				2.5,
				-7.25,
			},
			rel = 0,
		},
		{
			name = "f32",
			type = .Float,
			nulls = {
				false,
				true,
				false,
				false,
				false,
				false,
				false,
				false,
				true,
				false,
				false,
				false,
			},
			floats = {
				1.5,
				0,
				0.10000000149011612,
				-2.25,
				3.3999999521443642e+38,
				9.99994610111476e-41,
				0.0,
				7.0,
				0,
				-1.0,
				2.0,
				10000000000.0,
			},
			rel = 0,
		},
		{
			name = "b",
			type = .Bool,
			texts = {
				"true",
				"false",
				"NULL",
				"true",
				"true",
				"false",
				"NULL",
				"false",
				"true",
				"true",
				"false",
				"true",
			},
		},
		{
			name = "s",
			type = .String,
			texts = {
				"Apple",
				"",
				"NULL",
				"Zürich",
				"日本語",
				"a,b",
				"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
				"NVDA",
				"NULL",
				"MSFT",
				"msft",
				"last",
			},
		},
		{
			name = "cat",
			type = .String,
			texts = {"EU", "US", "EU", "US", "ASIA", "EU", "NULL", "US", "EU", "EU", "US", "ASIA"},
		},
		{
			name = "raw",
			type = .String,
			texts = {"raw1", "", "NULL", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11"},
		},
		{
			name = "fixed",
			type = .String,
			texts = {
				"abc",
				"def",
				"NULL",
				"ghi",
				"jkl",
				"mno",
				"pqr",
				"stu",
				"vwx",
				"yz0",
				"123",
				"456",
			},
		},
		{
			name = "d",
			type = .Date,
			texts = {
				"2024-02-29",
				"1969-12-31",
				"1900-01-01",
				"NULL",
				"1970-01-01",
				"9999-12-31",
				"0001-01-01",
				"2000-03-01",
				"2024-12-31",
				"NULL",
				"2023-07-04",
				"2024-01-01",
			},
		},
		{
			name = "ts_us",
			type = .Datetime,
			texts = {
				"2024-03-01 09:30:15",
				"1969-12-31 23:59:59",
				"NULL",
				"1970-01-01 00:00:00",
				"2000-02-29 23:59:59",
				"1900-01-01 00:00:01",
				"2262-04-11 23:47:16",
				"2024-12-31 12:00:00",
				"NULL",
				"2021-06-15 18:45:30",
				"1999-12-31 23:59:59",
				"2024-01-02 03:04:05",
			},
		},
		{
			name = "ts_ms",
			type = .Datetime,
			texts = {
				"2024-03-01 09:30:15",
				"1969-12-31 23:59:59",
				"NULL",
				"1970-01-01 00:00:00",
				"2000-02-29 23:59:59",
				"1900-01-01 00:00:01",
				"2262-04-11 23:47:16",
				"2024-12-31 12:00:00",
				"NULL",
				"2021-06-15 18:45:30",
				"1999-12-31 23:59:59",
				"2024-01-02 03:04:05",
			},
		},
		{
			name = "ts_ns_utc",
			type = .Datetime,
			texts = {
				"2024-03-01 09:30:15",
				"1969-12-31 23:59:59",
				"NULL",
				"1970-01-01 00:00:00",
				"2000-02-29 23:59:59",
				"1900-01-01 00:00:01",
				"NULL",
				"2024-12-31 12:00:00",
				"NULL",
				"2021-06-15 18:45:30",
				"1999-12-31 23:59:59",
				"2024-01-02 03:04:05",
			},
		},
		{
			name = "t_ms",
			type = .Time,
			texts = {
				"09:30:15",
				"00:00:00",
				"NULL",
				"23:59:59",
				"12:00:00",
				"01:02:03",
				"04:05:06",
				"07:08:09",
				"10:11:12",
				"13:14:15",
				"NULL",
				"16:17:18",
			},
		},
		{
			name = "t_us",
			type = .Time,
			texts = {
				"09:30:15",
				"NULL",
				"00:00:00",
				"23:59:59",
				"12:00:00",
				"01:02:03",
				"04:05:06",
				"07:08:09",
				"10:11:12",
				"13:14:15",
				"14:00:00",
				"16:17:18",
			},
		},
		{
			name = "t_ns",
			type = .Time,
			texts = {
				"09:30:15",
				"NULL",
				"00:00:00",
				"23:59:59",
				"12:00:00",
				"01:02:03",
				"04:05:06",
				"07:08:09",
				"10:11:12",
				"13:14:15",
				"14:00:00",
				"16:17:18",
			},
		},
		{
			name = "dec",
			type = .Float,
			nulls = {
				false,
				false,
				true,
				false,
				false,
				false,
				false,
				false,
				false,
				false,
				false,
				false,
			},
			floats = {
				123.45,
				-0.01,
				0,
				99999999.99,
				-99999999.99,
				0.0,
				1.1,
				2.2,
				3.3,
				4.4,
				5.5,
				6.6,
			},
			rel = 0,
		},
		{
			name = "dec_big",
			type = .Float,
			nulls = {
				false,
				false,
				false,
				true,
				false,
				false,
				false,
				false,
				false,
				false,
				false,
				false,
			},
			floats = {
				1.2345678901234567e+19,
				-0.0001,
				1.5,
				0,
				-1e+26,
				0.0,
				7.0,
				8.0,
				9.0,
				10.0,
				11.0,
				12.0,
			},
			rel = 1e-15,
		},
	}
	for f in ([]string{"types_plain_v1", "types_dict_snappy", "types_dict_v2_gzip", "types_plain_v2_lz4", "types_pages"}) {
		df, ok := imp.parquet_load(path(f))
		expect(
			&passed,
			&total,
			ok && matches(&df, types_exp),
			fmt.tprintf("%s: all 21 columns as pyarrow reads them", f),
		)
		if ok {w.destroy_dataframe(&df)}
	}

	// ---------------- INT96 timestamps
	{
		int96_exp := []Expect {
			{
				name = "i32",
				type = .Int,
				texts = {
					"0",
					"1",
					"-1",
					"2147483647",
					"-2147483648",
					"5",
					"6",
					"7",
					"8",
					"9",
					"10",
					"11",
				},
			},
			{
				name = "ts_us",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15",
					"1969-12-31 23:59:59",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"2262-04-11 23:47:16",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30",
					"1999-12-31 23:59:59",
					"2024-01-02 03:04:05",
				},
			},
			{
				name = "ts_ms",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15",
					"1969-12-31 23:59:59",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"2262-04-11 23:47:16",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30",
					"1999-12-31 23:59:59",
					"2024-01-02 03:04:05",
				},
			},
			{
				name = "ts_ns_utc",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15",
					"1969-12-31 23:59:59",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"NULL",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30",
					"1999-12-31 23:59:59",
					"2024-01-02 03:04:05",
				},
			},
		}
		df, ok := imp.parquet_load(path("types_int96"))
		expect(
			&passed,
			&total,
			ok && matches(&df, int96_exp),
			"INT96 timestamps (Spark / Impala layout)",
		)
		if ok {w.destroy_dataframe(&df)}
	}

	// ---------------- integer-stored decimals
	{
		dec_exp := []Expect {
			{name = "id", type = .Int, texts = {"1", "2", "3", "4"}},
			{
				name = "d9",
				type = .Float,
				nulls = {false, false, true, false},
				floats = {1.25, -1234567.89, 0, 0.0},
				rel = 0,
			},
			{
				name = "d18",
				type = .Float,
				nulls = {false, false, false, true},
				floats = {12345.678, -1000000000000000.0, 0.001, 0},
				rel = 0,
			},
			{
				name = "d38",
				type = .Float,
				nulls = {false, false, true, false},
				floats = {-1e-06, 1.2345678901234567e+19, 0, 1.5},
				rel = 1e-15,
			},
		}
		df, ok := imp.parquet_load(path("decimals_duckdb"))
		expect(
			&passed,
			&total,
			ok && matches(&df, dec_exp),
			"DECIMAL on INT32 / INT64 / FIXED_LEN_BYTE_ARRAY (DuckDB)",
		)
		if ok {w.destroy_dataframe(&df)}
	}

	// ---------------- pandas file: the index is an ordinary column
	pandas_exp := []Expect {
		{
			name = "close",
			type = .Float,
			nulls = {false, false, true, false},
			floats = {101.5, 102.25, 0, 99.0},
			rel = 0,
		},
		{name = "volume", type = .Int, texts = {"1200", "900", "0", "4500"}},
		{
			name = "date",
			type = .Datetime,
			texts = {
				"2024-01-02 00:00:00",
				"2024-01-03 00:00:00",
				"2024-01-04 00:00:00",
				"2024-01-05 00:00:00",
			},
		},
	}
	{
		df, ok := imp.parquet_load(path("pandas_index"))
		expect(
			&passed,
			&total,
			ok && matches(&df, pandas_exp),
			"pandas to_parquet: values, NaN as null, index as a column",
		)
		if ok {w.destroy_dataframe(&df)}
	}

	// ---------------- column selection
	{
		proj_exp := []Expect {
			{
				name = "s",
				type = .String,
				texts = {
					"Apple",
					"",
					"NULL",
					"Zürich",
					"日本語",
					"a,b",
					"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
					"NVDA",
					"NULL",
					"MSFT",
					"msft",
					"last",
				},
			},
			{
				name = "i64",
				type = .Int,
				texts = {
					"0",
					"1",
					"-1",
					"4611686018427387904",
					"-9223372036854775808",
					"9223372036854775807",
					"NULL",
					"42",
					"7",
					"NULL",
					"-5",
					"123456789012",
				},
			},
			{
				name = "ts_ns_utc",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15",
					"1969-12-31 23:59:59",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"NULL",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30",
					"1999-12-31 23:59:59",
					"2024-01-02 03:04:05",
				},
			},
		}
		df, ok := imp.parquet_load(path("types_dict_snappy"), {"s", "i64", "ts_ns_utc"})
		expect(
			&passed,
			&total,
			ok && matches(&df, proj_exp),
			"columns: only the named ones, in the given order",
		)
		if ok {w.destroy_dataframe(&df)}

		nested_exp := []Expect {
			{
				name = "price",
				type = .Float,
				nulls = {false, true, false},
				floats = {1.5, 0, 3.25},
				rel = 0,
			},
			{name = "id", type = .Int, texts = {"1", "2", "3"}},
		}
		df2, ok2 := imp.parquet_load(path("nested"), {"price", "id"})
		expect(
			&passed,
			&total,
			ok2 && matches(&df2, nested_exp),
			"columns: flat columns of a file with a nested one",
		)
		if ok2 {w.destroy_dataframe(&df2)}
	}

	// ---------------- from memory
	{
		data, rerr := os.read_entire_file(path("types_dict_v2_gzip"), context.allocator)
		defer delete(data)
		df, ok := imp.parquet_load_from_bytes(data)
		expect(
			&passed,
			&total,
			rerr == nil && ok && matches(&df, types_exp),
			"parquet_load_from_bytes: same as from the file",
		)
		if ok {w.destroy_dataframe(&df)}
	}

	// ---------------- ok = false contracts (the reasons go to stderr)
	{
		fails :: proc(p: string, cols: []string = nil) -> bool {
			df, ok := imp.parquet_load(p, cols)
			if ok {w.destroy_dataframe(&df)}
			return !ok
		}
		expect(&passed, &total, fails(path("nested")), "nested column: ok = false")
		expect(&passed, &total, fails(path("zstd")), "Zstandard codec: ok = false")
		expect(&passed, &total, fails(path("no_such_file")), "missing file: ok = false")
		expect(&passed, &total, fails("demo_files/people.csv"), "not a Parquet file: ok = false")
		expect(
			&passed,
			&total,
			fails(path("types_plain_v1"), {"i64", "nope"}),
			"unknown column: ok = false",
		)
		expect(
			&passed,
			&total,
			fails(path("types_plain_v1"), {"i64", "i64"}),
			"column named twice: ok = false",
		)
	}

	// ---------------- damaged files: never a crash, never a wrong shape
	{
		data, _ := os.read_entire_file(path("types_dict_snappy"), context.allocator)
		defer delete(data)
		cut_ok := true
		for cut := 0; cut < len(data); cut += 97 {
			df, ok := imp.parquet_load_from_bytes(data[:cut])
			if ok {
				w.destroy_dataframe(&df)
				cut_ok = false
			}
		}
		expect(&passed, &total, cut_ok, "every truncated copy: ok = false")

		bad := make([]u8, len(data))
		defer delete(bad)
		survived := true
		for pos := 0; pos < len(data); pos += 7 {
			copy(bad, data)
			bad[pos] ~= 0x5a
			df, ok := imp.parquet_load_from_bytes(bad)
			if ok {
				if df.rows != 12 || len(df.columns) != 21 {survived = false}
				w.destroy_dataframe(&df)
			}
		}
		expect(
			&passed,
			&total,
			survived,
			"a flipped byte anywhere: no crash, and a loaded frame keeps its shape",
		)
	}

	// ================ writer: every file read back as written
	// (pyarrow and DuckDB read the same bytes identically; see the guide)
	{
		src, ok := imp.parquet_load(path("types_plain_v1"))
		defer if ok {w.destroy_dataframe(&src)}
		for comp in ex.Parquet_Compression {
			all_ok := ok
			for dict in ([]bool{true, false}) {
				for rg in ([]int{1 << 20, 5}) { 	// one row group, or three
					data, wok := ex.parquet_bytes(&src, comp, rg, dict)
					back, rok := imp.parquet_load_from_bytes(data)
					all_ok = all_ok && wok && rok && matches(&back, types_exp)
					if rok {w.destroy_dataframe(&back)}
					delete(data)
				}
			}
			expect(
				&passed,
				&total,
				all_ok,
				fmt.tprintf(
					"write %v: all column types read back (dictionary on / off, 1 or 3 row groups)",
					comp,
				),
			)
		}

		// to a file (scripts/check_parquet_export.py checks it with pyarrow)
		saved := ex.parquet_save(&src, "demo_files/export_test.parquet")
		back, rok := imp.parquet_load("demo_files/export_test.parquet")
		expect(
			&passed,
			&total,
			saved && rok && matches(&back, types_exp),
			"parquet_save: the file reads back",
		)
		if rok {w.destroy_dataframe(&back)}
		expect(
			&passed,
			&total,
			!ex.parquet_save(&src, "demo_files/no_such_dir/x.parquet"),
			"parquet_save into a missing folder: false",
		)

		// views: rows 3 ..< 9 of three columns
		rows := w.dataframe_slice_rows(&src, 3, 9, false)
		defer w.destroy_dataframe(&rows)
		names := []string{"s", "f64", "ts_us"}
		view := w.select_columns(&rows, names)
		defer w.destroy_dataframe(&view)
		sub := make([]Expect, len(names), context.temp_allocator)
		for n, i in names {
			for e in types_exp {
				if e.name != n {continue}
				sub[i] = e
				if e.type == .Float {
					sub[i].nulls, sub[i].floats = e.nulls[3:9], e.floats[3:9]
				} else {
					sub[i].texts = e.texts[3:9]
				}
			}
		}
		data, wok := ex.parquet_bytes(&view)
		vb, vok := imp.parquet_load_from_bytes(data)
		expect(
			&passed,
			&total,
			wok && vok && matches(&vb, sub),
			"a view (row slice of a column selection) writes its own rows",
		)
		if vok {w.destroy_dataframe(&vb)}
		delete(data)

		// no rows: the columns and their types survive
		empty := w.head(&src, 0)
		defer w.destroy_dataframe(&empty)
		none := make([]Expect, len(types_exp), context.temp_allocator)
		for e, i in types_exp {none[i] = Expect {
				name = e.name,
				type = e.type,
			}}
		data2, wok2 := ex.parquet_bytes(&empty)
		eb, eok := imp.parquet_load_from_bytes(data2)
		expect(
			&passed,
			&total,
			wok2 && eok && matches(&eb, none),
			"an empty frame keeps its columns and types",
		)
		if eok {w.destroy_dataframe(&eb)}
		delete(data2)
	}

	// hidden index (set_index with drop): written as the first column
	{
		px, ok := imp.parquet_load(path("pandas_index"))
		defer if ok {w.destroy_dataframe(&px)}
		if ok {w.set_index(&px, "date", true)}
		data, wok := ex.parquet_bytes(&px)
		back, rok := imp.parquet_load_from_bytes(data)
		good := ok && wok && rok && len(back.columns) == 3 && back.columns[0].name == "date"
		if good {
			want := []Expect{pandas_exp[2], pandas_exp[0], pandas_exp[1]}
			good = matches(&back, want)
		}
		expect(
			&passed,
			&total,
			good,
			"a hidden index is written as the first column, under its name",
		)
		if rok {w.destroy_dataframe(&back)}
		delete(data)
	}

	// 70 000 rows: several pages and 64 KiB Snappy fragments, long matches
	// (a constant text), incompressible values, an all-null column
	{
		N :: 70_000
		big := w.dataframe_new()
		defer w.destroy_dataframe(&big)
		text := w.column_new("text", .String, N)
		noise := w.column_new("noise", .Int, N)
		seq := w.column_new("seq", .Float, N)
		empty := w.column_new("empty", .Float, N)
		flag := w.column_new("flag", .Bool, N)
		when_ := w.column_new("when", .Datetime, N)
		x := u64(88172645463325252)
		for i in 0 ..< N {
			x ~= x << 13
			x ~= x >> 7
			x ~= x << 17
			w.append_string(
				&text,
				"the same long text in every row of the column" if i % 1000 != 999 else "an odd one",
			)
			w.append_int(&noise, int(x))
			w.append_float(&seq, f64(i) * 0.25)
			w.append_null(&empty)
			if i % 3 == 0 {w.append_null(&flag)} else {w.append_bool(&flag, i % 2 == 0)}
			w.append_datetime(&when_, w.datetime_from_unix(1_700_000_000 + i64(i) * 61))
		}
		w.add_column(&big, text)
		w.add_column(&big, noise)
		w.add_column(&big, seq)
		w.add_column(&big, empty)
		w.add_column(&big, flag)
		w.add_column(&big, when_)
		for comp in ex.Parquet_Compression {
			data, wok := ex.parquet_bytes(&big, comp, row_group_rows = 30_000)
			back, rok := imp.parquet_load_from_bytes(data)
			same := wok && rok && back.rows == N && len(back.columns) == len(big.columns)
			for ci in 0 ..< len(big.columns) {
				if !same {break}
				a, b := &big.columns[ci], &back.columns[ci]
				same = a.name == b.name && a.type == b.type
				for r := 0; same && r < N; r += 1 {
					same =
						w.is_null(a, r) == w.is_null(b, r) &&
						w.column_value_string(a, r) == w.column_value_string(b, r)
				}
			}
			expect(
				&passed,
				&total,
				same,
				fmt.tprintf("write %v: 70 000 rows in 3 row groups read back", comp),
			)
			if rok {w.destroy_dataframe(&back)}
			delete(data)
		}
	}

	fmt.printf("Parquet: %d/%d passed\n", passed, total)
}
