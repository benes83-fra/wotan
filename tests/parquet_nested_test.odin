package tests

import w "../wotan/core"
import imp "../wotan/importer"
import util "../wotan/util"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// Nested Parquet columns (lists, maps, structs) read as String columns of
// JSON text:
// - files written by pyarrow 14 (Snappy, version 1 pages of 512 bytes, row
//   groups of 97 rows; and version 2 pages, Zstandard, no dictionary) and by
//   DuckDB 1.1: every cell of every nested column equals the JSON text of
//   pyarrow's value (an FNV-1a digest per column, plus whole cells). The
//   columns cover lists, lists of lists, a struct holding a list and a
//   struct, maps with string and integer keys, lists of structs, decimals,
//   binary, times, timestamps and bools; nulls and empty lists at every
//   level, a 6000-element list across pages, and a required struct;
// - the same text on one thread and on all;
// - parquet_info lists nested columns as String columns with their stored
//   type;
// - column selection; a filter on a flat column keeps the nested cells of
//   its rows; a filter on a nested column gives ok = false;
// - damaged copies never crash and never give a frame of the wrong shape.

parquet_nested_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Parquet Nested Columns ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	path :: proc(name: string) -> string {
		return fmt.tprintf("demo_files/parquet/%s.parquet", name)
	}
	Digest :: struct {
		file, column: string,
		fnv:          u64,
	}
	Cell :: struct {
		file, column: string,
		row:          int,
		text:         string,
	}
	// as pyarrow 14 / DuckDB 1.1 values written as JSON text
	DIGESTS := []Digest {
		{"nested_types", "ints", 0x7c2f05ef0cc10b4b},
		{"nested_types", "strs", 0x84c5aca20038c9c8},
		{"nested_types", "grid", 0xfa0e98651eb48b6e},
		{"nested_types", "st", 0x145a245b6a3da615},
		{"nested_types", "m", 0x3d60d736662b290e},
		{"nested_types", "mi", 0x48d77f5352967711},
		{"nested_types", "los", 0x8250e35c5056e44f},
		{"nested_types", "dec", 0x39ef98d6b083b948},
		{"nested_types", "bin", 0x329783495e6802db},
		{"nested_types", "tm", 0xa4715123911e8552},
		{"nested_types", "flags", 0xbb7c9f98a00e7780},
		{"nested_types", "req", 0x218ebda6b79a7fdb},
		{"nested_duckdb", "ints", 0xa2726dc9af549ab6},
		{"nested_duckdb", "st", 0xe23822c215c65b60},
		{"nested_duckdb", "m", 0x405da12aca28fa53},
		{"nested_duckdb", "grid", 0x62c8de0b46af8761},
	}
	CELLS := []Cell {
		{"nested_types", "ints", 1, `[]`},
		{"nested_types", "ints", 6, `[null,null]`},
		{"nested_types", "ints", 77, `[null,554714602269,-67732232470]`},
		{"nested_types", "strs", 1, `["Zürich"]`},
		{"nested_types", "strs", 6, `["quote\"d","back\\slash","日本"]`},
		{"nested_types", "strs", 77, `[]`},
		{"nested_types", "grid", 1, `[[null,41]]`},
		{"nested_types", "grid", 6, `[[],null,[null],[1,2]]`},
		{"nested_types", "grid", 77, `[[79,8,29],[13]]`},
		{
			"nested_types",
			"st",
			1,
			`{"a":null,"b":"quote\"d","c":[-0.25,-0.25,null],"d":{"x":"2020-03-10","y":null}}`,
		},
		{
			"nested_types",
			"st",
			6,
			`{"a":1,"b":"beta","c":[1e+300,null],"d":{"x":"1941-03-16","y":"2024-02-29 22:05:19.000147"}}`,
		},
		{
			"nested_types",
			"st",
			77,
			`{"a":0,"b":null,"c":[1e+300,null,1e+300],"d":{"x":null,"y":"2024-03-05 17:25:15.390904"}}`,
		},
		{"nested_types", "m", 1, `{"w":9}`},
		{"nested_types", "m", 77, `{"x":null,"y":2}`},
		{"nested_types", "mi", 6, `{"32":"beta","10":"alpha","43":"tab\there"}`},
		{
			"nested_types",
			"los",
			1,
			`[{"k":"tab\there","v":7.0},{"k":null,"v":null},{"k":null,"v":2.5}]`,
		},
		{
			"nested_types",
			"los",
			6,
			`[{"k":"","v":null},{"k":"quote\"d","v":-1e-300},{"k":"日本","v":7.0}]`,
		},
		{"nested_types", "dec", 77, `[202459.03,-6.67636966e+06,9.39571639e+06]`},
		{"nested_types", "bin", 1, `<null>`},
		{"nested_types", "bin", 6, `["\u00daGb","\u00af"]`},
		{"nested_types", "tm", 6, `["18:12:56","13:24:26.123456"]`},
		{"nested_types", "flags", 1, `[true,false,true,false]`},
		{"nested_types", "req", 1, `{"n":3,"s":null}`},
		{"nested_types", "req", 6, `{"n":18,"s":"Zürich"}`},
		{"nested_duckdb", "ints", 0, `<null>`},
		{"nested_duckdb", "ints", 4, `[4,null,8]`},
		{"nested_duckdb", "ints", 5, `[]`},
		{"nested_duckdb", "st", 5, `{"a":5,"b":"r5","l":[2.5,null]}`},
		{"nested_duckdb", "m", 7, `{"k7":7}`},
		{"nested_duckdb", "grid", 4, `[[4],[],null]`},
	}
	text_at :: proc(df: ^w.DataFrame, column: string, row: int) -> string {
		c, found := w.try_column(df, column)
		if !found || row >= df.rows {return "<missing>"}
		if w.is_null(c, row) {return "<null>"}
		v, _ := w.column_at_string(c, row)
		return v
	}
	// FNV-1a over the cells' text, each followed by a line break
	digest :: proc(df: ^w.DataFrame, column: string, rows: []int = nil) -> u64 {
		h: u64 = 0xcbf29ce484222325
		c, found := w.try_column(df, column)
		if !found {return 0}
		n := len(rows) if rows != nil else df.rows
		for i in 0 ..< n {
			r := rows[i] if rows != nil else i
			t := text_at(df, column, r)
			for b in transmute([]u8)t {
				h ~= u64(b)
				h *= 0x100000001b3
			}
			h ~= '\n'
			h *= 0x100000001b3
		}
		_ = c
		return h
	}
	check_file :: proc(
		df: ^w.DataFrame,
		file: string,
		digests: []Digest,
		cells: []Cell,
	) -> (
		good, n: int,
	) {
		for d in digests {
			if d.file != file {continue}
			n += 1
			got := digest(df, d.column)
			if got ==
			   d.fnv {good += 1} else {fmt.printfln("    %s.%s: digest %16x", file, d.column, got)}
		}
		for c in cells {
			if c.file != file {continue}
			n += 1
			got := text_at(df, c.column, c.row)
			if got ==
			   c.text {good += 1} else {fmt.printfln("    %s.%s[%d] = %s", file, c.column, c.row, got)}
		}
		return
	}

	// ---------------- every cell as pyarrow's value in JSON
	types, tok := imp.parquet_load(path("nested_types"))
	defer if tok {w.destroy_dataframe(&types)}
	{
		g1, n1 := check_file(&types, "nested_types", DIGESTS, CELLS)
		v2, vok := imp.parquet_load(path("nested_types_v2_zstd"))
		g2, n2 := 0, 1
		if vok {g2, n2 = check_file(&v2, "nested_types", DIGESTS, nil)}
		if vok {w.destroy_dataframe(&v2)}
		duck, dok := imp.parquet_load(path("nested_duckdb"))
		g3, n3 := 0, 1
		if dok {g3, n3 = check_file(&duck, "nested_duckdb", DIGESTS, CELLS)}
		if dok {w.destroy_dataframe(&duck)}
		small, sok := imp.parquet_load(path("nested"))
		small_ok :=
			sok &&
			text_at(&small, "tags", 0) == `["a","b"]` &&
			text_at(&small, "tags", 1) == `[]` &&
			text_at(&small, "tags", 2) == "<null>"
		if sok {w.destroy_dataframe(&small)}
		long := text_at(&types, "ints", 5)
		long_ok := len(long) > 2 && long[:8] == "[0,1,2,3" && long[len(long) - 6:] == ",5999]"
		shape_ok :=
			tok &&
			types.rows == 400 &&
			len(types.columns) == 14 &&
			w.column(&types, "ints").type == .String &&
			w.column(&types, "id").type == .Int &&
			w.column(&types, "price").type == .Float
		expect(
			&passed,
			&total,
			shape_ok && g1 == n1 && g2 == n2 && g3 == n3 && small_ok && long_ok,
			fmt.tprintf(
				"%d of %d digests and cells as pyarrow 14 / DuckDB 1.1 values in JSON (lists, lists of lists, structs, maps with string and int keys, lists of structs, decimals, binary, times, timestamps, bools; nulls and empty lists at every level); both pyarrow layouts; a 6000-element list across pages",
				g1 + g2 + g3,
				n1 + n2 + n3,
			),
		)
	}

	// ---------------- one thread and all threads
	{
		saved := util.parallel_max_threads
		util.parallel_max_threads = 1
		one, ok1 := imp.parquet_load(path("nested_types"))
		util.parallel_max_threads = saved
		same := ok1 && tok && one.rows == types.rows
		if same {
			for c in types.columns {
				if c.type != .String {continue}
				if digest(&one, c.name) != digest(&types, c.name) {same = false}
			}
		}
		if ok1 {w.destroy_dataframe(&one)}
		expect(&passed, &total, same, "the same JSON text on one thread and on all")
	}

	// ---------------- parquet_info
	{
		info, ok := imp.parquet_info(path("nested_types"))
		defer if ok {imp.destroy_parquet_info(&info)}
		kind :: proc(info: ^imp.Parquet_Info, name: string) -> string {
			for c in info.columns {
				if c.name == name {
					return fmt.tprintf("%v %s %v %s", c.type, c.parquet_type, c.nullable, c.reason)
				}
			}
			return "?"
		}
		good :=
			ok &&
			len(info.columns) == 14 &&
			kind(&info, "ints") == "String LIST true " &&
			kind(&info, "st") == "String struct true " &&
			kind(&info, "m") == "String MAP true " &&
			kind(&info, "req") == "String struct false " &&
			kind(&info, "price") == "Float DOUBLE true " &&
			info.row_groups[0].stats[1].min == nil
		expect(
			&passed,
			&total,
			good,
			"parquet_info: nested columns are String columns with their stored type (LIST, MAP, struct) and no statistics",
		)
	}

	// ---------------- selection and filters
	{
		sel, sok := imp.parquet_load(path("nested_types"), {"st", "id"})
		sel_ok :=
			sok &&
			len(sel.columns) == 2 &&
			sel.columns[0].name == "st" &&
			digest(&sel, "st") == digest(&types, "st")
		if sok {w.destroy_dataframe(&sel)}
		f := []imp.Parquet_Filter{{column = "id", lower = 100, upper = 199}}
		part, pok := imp.parquet_load(path("nested_types"), {"id", "los", "m"}, filters = f)
		rows := make([]int, 100)
		defer delete(rows)
		for &r, i in rows {r = 100 + i}
		part_ok :=
			pok &&
			part.rows == 100 &&
			digest(&part, "los") == digest(&types, "los", rows) &&
			digest(&part, "m") == digest(&types, "m", rows)
		if pok {w.destroy_dataframe(&part)}
		bad, bok := imp.parquet_load(
			path("nested_types"),
			filters = {{column = "ints", lower = 1}},
		)
		if bok {w.destroy_dataframe(&bad)}
		expect(
			&passed,
			&total,
			sel_ok && part_ok && !bok,
			"column selection keeps the JSON text; a filter on id keeps the nested cells of rows 100..199; a filter on a nested column: ok = false",
		)
	}

	// ---------------- damaged copies
	{
		data, err := os.read_entire_file(path("nested_types"), context.allocator)
		defer delete(data)
		broken := make([]u8, len(data))
		defer delete(broken)
		runs, loaded, wrong := 0, 0, 0
		fmt.println("    (damaged copies; the parquet errors below are expected)")
		for _ in 0 ..< 300 {
			if err != nil {break}
			copy(broken, data)
			for _ in 0 ..< 1 + rand.int_max(3) {
				k := rand.int_max(len(data))
				broken[k] ~= u8(1 + rand.int_max(255))
			}
			runs += 1
			df, ok := imp.parquet_load_from_bytes(broken)
			if ok {
				loaded += 1
				for &c in df.columns {if c.len != df.rows {wrong += 1}}
				w.destroy_dataframe(&df)
			}
		}
		expect(
			&passed,
			&total,
			runs == 300 && wrong == 0,
			fmt.tprintf(
				"300 damaged copies: no crash, no frame of the wrong shape (%d still load: the damage hit values)",
				loaded,
			),
		)
	}

	fmt.printf("Parquet nested columns: %d/%d passed\n", passed, total)
}
