package tests

import w "../wotan/core"
import ex "../wotan/exporter"
import imp "../wotan/importer"
import util "../wotan/util"
import zs "../wotan/zstd"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

// Parquet reader and writer.
//
// Reader, against pyarrow on the fixtures in demo_files/parquet (written by
// pyarrow 14.0.2 and 17.0.0 and by DuckDB 1.1.3; the expected cells are
// pyarrow's reading of them):
// - one 12-row table with every mapped type (ints of all widths, unsigned,
//   floats with NaN / -0 / inf, bools, UTF-8 and binary strings, fixed-length
//   binary, dates back to year 1, timestamps in ms / us / ns (UTC) before and
//   after 1970, times in ms / us / ns, decimals up to 30 digits) and nulls,
//   written eight ways: PLAIN v1 uncompressed, dictionary v1 Snappy,
//   dictionary v2 gzip, dictionary Zstandard, PLAIN v2 LZ4 with RLE booleans,
//   three row groups of many small pages whose dictionaries fall back to
//   PLAIN mid-chunk, and the delta / byte-stream-split encodings (v2
//   Zstandard; v1 Snappy in many small pages). Every cell must equal
//   pyarrow's (sub-second parts dropped);
// - FLOAT16 and byte stream split on integers and fixed-length binary
//   (pyarrow 17); Zstandard pages of several blocks; DuckDB's Zstandard file
//   with UUIDs; the Zstandard decoder on its own (frames, skippable frames,
//   checksum); the Zstandard compressor's frames read back (libzstd reads
//   them too, checked outside the suite); one thread and all threads give
//   the same frame;
// - INT96 timestamps; decimals stored as INT32 / INT64 / FIXED_LEN_BYTE_ARRAY
//   (DuckDB); a pandas file with its index; column selection and order;
// - ok = false (no crash) for Brotli, a missing file, a CSV file, unknown or
//   repeated column names, truncated and corrupted bytes; a nested column
//   loads (as JSON text, see parquet_nested_test).
//
// Writer: the same table written with every codec, dictionary on and off, in
// one or three row groups, read back cell for cell; a file on disk
// (demo_files/export_test.parquet); a view; an empty
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
				"2024-03-01 09:30:15.999999",
				"1969-12-31 23:59:59.500",
				"NULL",
				"1970-01-01 00:00:00",
				"2000-02-29 23:59:59",
				"1900-01-01 00:00:01",
				"2262-04-11 23:47:16",
				"2024-12-31 12:00:00",
				"NULL",
				"2021-06-15 18:45:30.000001",
				"1999-12-31 23:59:59.999999",
				"2024-01-02 03:04:05",
			},
		},
		{
			name = "ts_ms",
			type = .Datetime,
			texts = {
				"2024-03-01 09:30:15.999",
				"1969-12-31 23:59:59.500",
				"NULL",
				"1970-01-01 00:00:00",
				"2000-02-29 23:59:59",
				"1900-01-01 00:00:01",
				"2262-04-11 23:47:16",
				"2024-12-31 12:00:00",
				"NULL",
				"2021-06-15 18:45:30",
				"1999-12-31 23:59:59.999",
				"2024-01-02 03:04:05",
			},
		},
		{
			name = "ts_ns_utc",
			type = .Datetime,
			texts = {
				"2024-03-01 09:30:15.999999+00:00",
				"1969-12-31 23:59:59.500+00:00",
				"NULL",
				"1970-01-01 00:00:00+00:00",
				"2000-02-29 23:59:59+00:00",
				"1900-01-01 00:00:01+00:00",
				"NULL",
				"2024-12-31 12:00:00+00:00",
				"NULL",
				"2021-06-15 18:45:30.000001+00:00",
				"1999-12-31 23:59:59.999999+00:00",
				"2024-01-02 03:04:05+00:00",
			},
		},
		{
			name = "t_ms",
			type = .Time,
			texts = {
				"09:30:15.250",
				"00:00:00",
				"NULL",
				"23:59:59.999",
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
				"09:30:15.250001",
				"NULL",
				"00:00:00.000001",
				"23:59:59.999999",
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
				"09:30:15.250001",
				"NULL",
				"00:00:00.000001",
				"23:59:59.999999",
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
	for f in ([]string{"types_plain_v1", "types_dict_snappy", "types_dict_v2_gzip", "types_plain_v2_lz4", "types_pages", "types_delta_v2_zstd", "types_delta_pages", "types_dict_zstd"}) {
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
					"2024-03-01 09:30:15.999999",
					"1969-12-31 23:59:59.500",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"2262-04-11 23:47:16",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30.000001",
					"1999-12-31 23:59:59.999999",
					"2024-01-02 03:04:05",
				},
			},
			{
				name = "ts_ms",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15.999",
					"1969-12-31 23:59:59.500",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"2262-04-11 23:47:16",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30",
					"1999-12-31 23:59:59.999",
					"2024-01-02 03:04:05",
				},
			},
			{
				name = "ts_ns_utc",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15.999999",
					"1969-12-31 23:59:59.500",
					"NULL",
					"1970-01-01 00:00:00",
					"2000-02-29 23:59:59",
					"1900-01-01 00:00:01",
					"NULL",
					"2024-12-31 12:00:00",
					"NULL",
					"2021-06-15 18:45:30.000001",
					"1999-12-31 23:59:59.999999",
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
	f16_exp := []Expect {
		{
			name = "h",
			type = .Float,
			nulls = {false, false, false, false, false, false, false, false},
			floats = {
				1.5,
				-2.0,
				0.0999755859375,
				65504.0,
				math.inf_f64(1),
				0.0,
				NEG_ZERO,
				5.960464477539063e-08,
			},
			rel = 0,
		},
		{
			name = "h_null",
			type = .Float,
			nulls = {false, true, false, true, false, true, false, false},
			floats = {1.0, 0, 3.0, 0, 5.0, 0, 7.0, 8.0},
			rel = 0,
		},
		{
			name = "i32",
			type = .Int,
			texts = {"0", "-1", "2147483647", "-2147483648", "5", "NULL", "7", "8"},
		},
		{
			name = "i64",
			type = .Int,
			texts = {
				"0",
				"-1",
				"9223372036854775807",
				"-9223372036854775808",
				"5",
				"NULL",
				"7",
				"8",
			},
		},
		{
			name = "fixed",
			type = .String,
			texts = {"ab", "cd", "NULL", "gh", "ij", "kl", "mn", "op"},
		},
	}
	duck_uids := []string {
		"cfcd2084-95d5-65ef-66e7-dff9f98764da",
		"c4ca4238-a0b9-2382-0dcc-509a6f75849b",
		"c81e728d-9d4c-2f63-6f06-7f89cc14862c",
		"eccbc87e-4b5c-e2fe-2830-8fd9f2a7baf3",
		"a87ff679-a2f3-e71d-9181-a67b7542122c",
		"e4da3b7f-bbce-2345-d777-2b0674a318d5",
		"1679091c-5a88-0faf-6fb5-e6087eb1b2dc",
		"8f14e45f-ceea-167a-5a36-dedd4bea2543",
		"c9f0f895-fb98-ab91-59f5-1fd0297e236d",
		"45c48cce-2e2d-7fbd-ea1a-fc51c7c6ad26",
		"d3d94468-02a4-4259-755d-38e6d163e820",
		"6512bd43-d9ca-a6e0-2c99-0b0a82652dca",
		"c20ad4d7-6fe9-7759-aa27-a0c99bff6710",
		"c51ce410-c124-a10e-0db5-e4b97fc2af39",
		"aab32389-22bc-c25a-6f60-6eb525ffdc56",
		"9bf31c7f-f062-936a-96d3-c8bd1f8f2ff3",
		"c74d97b0-1eae-257e-44aa-9d5bade97baf",
		"70efdf2e-c9b0-8607-9795-c442636b55fb",
		"6f4922f4-5568-161a-8cdf-4ad2299f6d23",
		"1f0e3dad-9990-8345-f743-9f8ffabdffc4",
	}
	zstd_plain := "line 0: the quick brown fox jumps over the lazy dog 0. line 1: the quick brown fox jumps over the lazy dog 1. line 2: the quick brown fox jumps over the lazy dog 2. line 3: the quick brown fox jumps over the lazy dog 3. line 4: the quick brown fox jumps over the lazy dog 4. line 5: the quick brown fox jumps over the lazy dog 5. line 6: the quick brown fox jumps over the lazy dog 6. line 7: the quick brown fox jumps over the lazy dog 7. line 8: the quick brown fox jumps over the lazy dog 8. line 9: the quick brown fox jumps over the lazy dog 9. line 10: the quick brown fox jumps over the lazy dog 10. line 11: the quick brown fox jumps over the lazy dog 11. line 12: the quick brown fox jumps over the lazy dog 12. line 13: the quick brown fox jumps over the lazy dog 0. line 14: the quick brown fox jumps over the lazy dog 1. line 15: the quick brown fox jumps over the lazy dog 2. line 16: the quick brown fox jumps over the lazy dog 3. line 17: the quick brown fox jumps over the lazy dog 4. line 18: the quick brown fox jumps over the lazy dog 5. line 19: the quick brown fox jumps over the lazy dog 6. line 20: the quick brown fox jumps over the lazy dog 7. line 21: the quick brown fox jumps over the lazy dog 8. line 22: the quick brown fox jumps over the lazy dog 9. line 23: the quick brown fox jumps over the lazy dog 10. line 24: the quick brown fox jumps over the lazy dog 11. line 25: the quick brown fox jumps over the lazy dog 12. line 26: the quick brown fox jumps over the lazy dog 0. line 27: the quick brown fox jumps over the lazy dog 1. line 28: the quick brown fox jumps over the lazy dog 2. line 29: the quick brown fox jumps over the lazy dog 3. line 30: the quick brown fox jumps over the lazy dog 4. line 31: the quick brown fox jumps over the lazy dog 5. line 32: the quick brown fox jumps over the lazy dog 6. line 33: the quick brown fox jumps over the lazy dog 7. line 34: the quick brown fox jumps over the lazy dog 8. line 35: the quick brown fox jumps over the lazy dog 9. line 36: the quick brown fox jumps over the lazy dog 10. line 37: the quick brown fox jumps over the lazy dog 11. line 38: the quick brown fox jumps over the lazy dog 12. line 39: the quick brown fox jumps over the lazy dog 0. line 40: the quick brown fox jumps over the lazy dog 1. line 41: the quick brown fox jumps over the lazy dog 2. line 42: the quick brown fox jumps over the lazy dog 3. line 43: the quick brown fox jumps over the lazy dog 4. line 44: the quick brown fox jumps over the lazy dog 5. line 45: the quick brown fox jumps over the lazy dog 6. line 46: the quick brown fox jumps over the lazy dog 7. line 47: the quick brown fox jumps over the lazy dog 8. line 48: the quick brown fox jumps over the lazy dog 9. line 49: the quick brown fox jumps over the lazy dog 10. line 50: the quick brown fox jumps over the lazy dog 11. line 51: the quick brown fox jumps over the lazy dog 12. line 52: the quick brown fox jumps over the lazy dog 0. line 53: the quick brown fox jumps over the lazy dog 1. line 54: the quick brown fox jumps over the lazy dog 2. line 55: the quick brown fox jumps over the lazy dog 3. line 56: the quick brown fox jumps over the lazy dog 4. line 57: the quick brown fox jumps over the lazy dog 5. line 58: the quick brown fox jumps over the lazy dog 6. line 59: the quick brown fox jumps over the lazy dog 7. "
	zstd_frames := []u8 {
		40,
		181,
		47,
		253,
		96,
		220,
		4,
		21,
		6,
		0,
		18,
		134,
		21,
		27,
		112,
		109,
		58,
		0,
		2,
		60,
		246,
		215,
		38,
		94,
		163,
		44,
		30,
		0,
		0,
		69,
		73,
		219,
		91,
		74,
		153,
		86,
		244,
		170,
		254,
		223,
		11,
		234,
		121,
		252,
		253,
		191,
		114,
		26,
		125,
		190,
		237,
		100,
		219,
		102,
		152,
		101,
		89,
		75,
		169,
		76,
		196,
		187,
		223,
		174,
		146,
		106,
		104,
		132,
		74,
		38,
		1,
		7,
		98,
		143,
		214,
		198,
		66,
		17,
		33,
		26,
		105,
		133,
		62,
		144,
		6,
		9,
		154,
		24,
		52,
		17,
		163,
		236,
		224,
		40,
		32,
		149,
		61,
		141,
		19,
		50,
		32,
		80,
		51,
		18,
		31,
		32,
		76,
		39,
		166,
		19,
		243,
		65,
		111,
		189,
		1,
		13,
		241,
		186,
		73,
		4,
		53,
		196,
		227,
		150,
		8,
		198,
		155,
		49,
		108,
		136,
		50,
		59,
		106,
		140,
		236,
		76,
		160,
		17,
		175,
		140,
		49,
		99,
		20,
		217,
		33,
		99,
		100,
		50,
		33,
		134,
		120,
		103,
		12,
		24,
		163,
		188,
		144,
		56,
		92,
		136,
		220,
		66,
		194,
		98,
		33,
		198,
		138,
		139,
		67,
		197,
		200,
		50,
		153,
		57,
		46,
		219,
		72,
		247,
		102,
		152,
		43,
		55,
		199,
		101,
		27,
		233,
		222,
		12,
		115,
		229,
		230,
		184,
		108,
		35,
		221,
		91,
		25,
		166,
		153,
		80,
		139,
		139,
		145,
		1,
		83,
		42,
		77,
		24,
		5,
		0,
		0,
		0,
		115,
		107,
		105,
		112,
		33,
		40,
		181,
		47,
		253,
		100,
		70,
		6,
		21,
		5,
		0,
		114,
		199,
		24,
		27,
		96,
		109,
		218,
		175,
		114,
		192,
		3,
		233,
		55,
		182,
		80,
		192,
		39,
		8,
		130,
		134,
		145,
		66,
		43,
		187,
		55,
		185,
		2,
		10,
		130,
		208,
		18,
		189,
		172,
		41,
		23,
		245,
		242,
		136,
		238,
		254,
		235,
		187,
		153,
		83,
		218,
		85,
		205,
		68,
		188,
		251,
		81,
		162,
		190,
		243,
		241,
		219,
		202,
		153,
		112,
		79,
		159,
		8,
		177,
		143,
		47,
		153,
		4,
		24,
		71,
		61,
		150,
		53,
		22,
		138,
		7,
		97,
		145,
		85,
		176,
		6,
		178,
		28,
		129,
		37,
		6,
		203,
		195,
		48,
		117,
		112,
		24,
		32,
		157,
		169,
		102,
		97,
		76,
		65,
		102,
		88,
		4,
		11,
		66,
		168,
		17,
		176,
		55,
		236,
		127,
		7,
		192,
		151,
		203,
		1,
		18,
		168,
		16,
		80,
		224,
		5,
		255,
		255,
		187,
		31,
		122,
		123,
		222,
		103,
		198,
		55,
		239,
		118,
		117,
		246,
		249,
		255,
		57,
		198,
		185,
		183,
		183,
		225,
		109,
		59,
		158,
		38,
		210,
		104,
		222,
		55,
		198,
		185,
		151,
		123,
		184,
		160,
		189,
		89,
		189,
		200,
		239,
		42,
		81,
		216,
		139,
		137,
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
					"2024-03-01 09:30:15.999999+00:00",
					"1969-12-31 23:59:59.500+00:00",
					"NULL",
					"1970-01-01 00:00:00+00:00",
					"2000-02-29 23:59:59+00:00",
					"1900-01-01 00:00:01+00:00",
					"NULL",
					"2024-12-31 12:00:00+00:00",
					"NULL",
					"2021-06-15 18:45:30.000001+00:00",
					"1999-12-31 23:59:59.999999+00:00",
					"2024-01-02 03:04:05+00:00",
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
		expect(&passed, &total, !fails(path("nested")), "nested column: loads (as JSON text)")
		expect(&passed, &total, fails(path("brotli")), "Brotli codec: ok = false")
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

	// ================ Zstandard, delta and byte-stream-split encodings
	// float16, byte stream split on integers and fixed-length binary (pyarrow 17)
	{
		df, ok := imp.parquet_load(path("float16_split"))
		expect(
			&passed,
			&total,
			ok && matches(&df, f16_exp),
			"FLOAT16 and BYTE_STREAM_SPLIT on INT32 / INT64 / FIXED_LEN_BYTE_ARRAY (pyarrow 17)",
		)
		if ok {w.destroy_dataframe(&df)}
	}
	// Zstandard pages of 1.6 MB, several Zstandard blocks each, two row groups
	{
		df, ok := imp.parquet_load(path("zstd_big"))
		good := ok && df.rows == 300_000
		if good {
			ic, sc := w.column(&df, "i"), w.column(&df, "s")
			for k in 0 ..< df.rows {
				v, _ := w.column_at_int(ic, k)
				s, _ := w.column_at_string(sc, k)
				if v != (k * 7919) % 1009 || s != fmt.tprintf("id-%d", k % 5000) {
					good = false
					fmt.printf("    row %d: %d %q\n", k, v, s)
					break
				}
			}
		}
		expect(
			&passed,
			&total,
			good,
			"Zstandard: 300 000 rows in pages larger than a Zstandard block",
		)
		if ok {w.destroy_dataframe(&df)}
	}
	// DuckDB with Zstandard: UUIDs as text, timestamps, decimals, nulls
	{
		df, ok := imp.parquet_load(path("duckdb_zstd"))
		good := ok && df.rows == 500 && len(df.columns) == 6
		if good {
			for k in 0 ..< 500 {
				id, _ := w.column_at_int(w.column(&df, "id"), k)
				uid, _ := w.column_at_string(w.column(&df, "uid"), k)
				ts, _ := w.column_at_datetime(w.column(&df, "ts"), k)
				label, _ := w.column_at_string(w.column(&df, "label"), k)
				amount, _ := w.column_at_float(w.column(&df, "amount"), k)
				half, null := w.column_at_float(w.column(&df, "half"), k)
				want_ts := w.datetime_from_unix(
					w.datetime_to_unix(w.Datetime{2024, 1, 1, 9, 30, 0, 0}) + i64(k) * 61,
				)
				ok_row :=
					id == k &&
					ts == want_ts &&
					label == fmt.tprintf("row %d", k % 7) &&
					amount == f64(k) * 1.25 &&
					null == (k % 5 == 0) &&
					(null || half == f64(k) * 0.5) &&
					(k >= len(duck_uids) || uid == duck_uids[k])
				if !ok_row {
					good = false
					fmt.printf(
						"    row %d: %v %q %v %q %v %v\n",
						k,
						id,
						uid,
						ts,
						label,
						amount,
						half,
					)
					break
				}
			}
		}
		expect(
			&passed,
			&total,
			good,
			"DuckDB with Zstandard: UUIDs as text, timestamps, decimals, nulls",
		)
		if ok {w.destroy_dataframe(&df)}
	}
	// the decoder on its own: two frames, a skippable frame, a checksum
	{
		out := make([]u8, len(zstd_plain) + 16)
		defer delete(out)
		n, ok := zs.decompress(out, zstd_frames)
		size, sok := zs.decompressed_size(zstd_frames)
		expect(
			&passed,
			&total,
			ok && n == len(zstd_plain) && string(out[:n]) == zstd_plain && sok && size == n,
			"zstd.decompress: two frames with a skippable frame between them",
		)
		bad := make([]u8, len(zstd_frames))
		defer delete(bad)
		copy(bad, zstd_frames)
		bad[len(bad) - 6] ~= 0x10 // inside the checksummed frame's last block
		_, bok := zs.decompress(out, bad)
		expect(&passed, &total, !bok, "zstd.decompress: a changed byte fails the content checksum")
		_, sok2 := zs.decompress(out[:n - 1], zstd_frames)
		expect(&passed, &total, !sok2, "zstd.decompress: an output buffer one byte short fails")
	}
	// the compressor: empty input, one byte, one repeated byte (RLE blocks),
	// random bytes (stored blocks), text, the tz database twice in a row (its
	// second copy matches the first, three blocks back) and a CSV file
	{
		tzdata, _ := os.read_entire_file("wotan/core/tzdata.bin", context.allocator)
		defer delete(tzdata)
		csv, _ := os.read_entire_file("demo_files/timezones.csv", context.allocator)
		defer delete(csv)
		same := make([]u8, 300_000)
		defer delete(same)
		for &b in same {b = 'a'}
		noise := make([]u8, 200_000)
		defer delete(noise)
		x := u64(88172645463325252)
		for &b in noise {
			x ~= x << 13
			x ~= x >> 7
			x ~= x << 17
			b = u8(x >> 32)
		}
		twice := make([]u8, 2 * len(tzdata))
		defer delete(twice)
		copy(twice, tzdata)
		copy(twice[len(tzdata):], tzdata)
		inputs := [][]u8{{}, {'x'}, same, noise, transmute([]u8)zstd_plain, twice, csv}
		sizes: [7]int
		all_ok := len(tzdata) > 0 && len(csv) > 0
		for src, i in inputs {
			comp, ok := zs.compress(src)
			back := make([]u8, len(src) + 1)
			n, dok := zs.decompress(back, comp)
			size, sok := zs.decompressed_size(comp)
			all_ok =
				all_ok &&
				ok &&
				dok &&
				n == len(src) &&
				string(back[:n]) == string(src) &&
				sok &&
				size == len(src)
			sizes[i] = len(comp)
			delete(back)
			delete(comp)
		}
		expect(
			&passed,
			&total,
			all_ok,
			"zstd.compress: seven inputs read back, with their sizes declared",
		)
		expect(
			&passed,
			&total,
			sizes[2] < 30 &&
			sizes[3] <= len(noise) + 32 &&
			sizes[4] * 10 < len(zstd_plain) &&
			sizes[5] * 8 < len(twice) &&
			sizes[6] * 3 < len(csv),
			fmt.tprintf(
				"zstd.compress sizes: repeated byte %d, random %d of %d, text %d of %d, tz database twice %d of %d, CSV %d of %d",
				sizes[2],
				sizes[3],
				len(noise),
				sizes[4],
				len(zstd_plain),
				sizes[5],
				len(twice),
				sizes[6],
				len(csv),
			),
		)
	}
	// the same frames, decoded on one thread and on all
	{
		util.parallel_max_threads = 1
		one, ok1 := imp.parquet_load(path("types_delta_pages"))
		util.parallel_max_threads = 0
		all, ok2 := imp.parquet_load(path("types_delta_pages"))
		expect(
			&passed,
			&total,
			ok1 && ok2 && matches(&one, types_exp) && matches(&all, types_exp),
			"one thread and all threads read the same frame",
		)
		if ok1 {w.destroy_dataframe(&one)}
		if ok2 {w.destroy_dataframe(&all)}
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

		// the same bytes from one thread and from all
		util.parallel_max_threads = 1
		one, ok1 := ex.parquet_bytes(&src, .Snappy, 5)
		util.parallel_max_threads = 0
		all, ok2 := ex.parquet_bytes(&src, .Snappy, 5)
		expect(
			&passed,
			&total,
			ok1 && ok2 && string(one) == string(all),
			"one thread and all threads write the same bytes",
		)
		delete(one)
		delete(all)

		// to a file
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

	// nanoseconds: kept exactly (TIMESTAMP / TIME in NANOS); a column that
	// also holds the year 9999 does not fit nanoseconds and keeps microseconds
	{
		ns := w.column_new("ns", .Datetime, 4)
		far := w.column_new("far", .Datetime, 4)
		tod := w.column_new("tod", .Time, 4)
		w.append_datetime(&ns, w.Datetime{2024, 3, 1, 9, 30, 15, 123_456_789})
		w.append_datetime(&ns, w.Datetime{1969, 12, 31, 23, 59, 59, 999_999_999})
		w.append_null(&ns)
		w.append_datetime(&ns, w.Datetime{2262, 4, 11, 23, 47, 16, 854_775_807})
		w.append_datetime(&far, w.Datetime{9999, 12, 31, 23, 59, 59, 999_999_999})
		w.append_datetime(&far, w.Datetime{1, 1, 1, 0, 0, 0, 1_000})
		w.append_datetime(&far, w.Datetime{2024, 3, 1, 9, 30, 15, 123_456_789})
		w.append_null(&far)
		w.append_time(&tod, w.Time{23, 59, 59, 999_999_999})
		w.append_time(&tod, w.Time{0, 0, 0, 1})
		w.append_time(&tod, w.Time{12, 0, 0, 500_000_000})
		w.append_null(&tod)
		fr := w.dataframe_new()
		defer w.destroy_dataframe(&fr)
		w.add_column(&fr, ns)
		w.add_column(&fr, far)
		w.add_column(&fr, tod)
		want := []Expect {
			{
				name = "ns",
				type = .Datetime,
				texts = {
					"2024-03-01 09:30:15.123456789",
					"1969-12-31 23:59:59.999999999",
					"NULL",
					"2262-04-11 23:47:16.854775807",
				},
			},
			{
				name = "far",
				type = .Datetime,
				texts = {
					"9999-12-31 23:59:59.999999",
					"0001-01-01 00:00:00.000001",
					"2024-03-01 09:30:15.123456",
					"NULL",
				},
			},
			{
				name = "tod",
				type = .Time,
				texts = {"23:59:59.999999999", "00:00:00.000000001", "12:00:00.500", "NULL"},
			},
		}
		for comp in ([]ex.Parquet_Compression{.None, .Snappy}) {
			data, wok := ex.parquet_bytes(&fr, comp)
			back, rok := imp.parquet_load_from_bytes(data)
			expect(
				&passed,
				&total,
				wok && rok && matches(&back, want),
				fmt.tprintf(
					"write %v: nanoseconds kept; the year 9999 falls back to microseconds",
					comp,
				),
			)
			if rok {w.destroy_dataframe(&back)}
			delete(data)
		}
		ex.parquet_save(&fr, "demo_files/export_nanos.parquet")
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
