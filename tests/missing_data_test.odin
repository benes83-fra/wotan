package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

// Missing-data helpers of core (na_at, count_na, isna / notna, dropna,
// fillna, ffill / bfill) against pandas 1.5.3 on the same frame:
// a Float column mixing nulls and NaN (leading, inner and trailing gaps), an
// Int, a String and a Bool column with nulls. Pandas sees nulls and NaN alike
// as NA, so cells are compared as text with "NA" for both; row labels are
// compared with the pandas index. Also: views as input, index metadata kept,
// fillna by type, and the ok = false contracts.

missing_data_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DataFrame Missing Data ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	NA :: -999 // marks a null cell in the builders below
	floats :: proc(name: string, vals: []f64, null_at: []int) -> w.Column {
		c := w.column_new(name, .Float, len(vals))
		outer: for v, i in vals {
			for n in null_at {if n == i {w.append_null(&c); continue outer}}
			w.append_float(&c, v)
		}
		return c
	}
	ints :: proc(name: string, vals: []int) -> w.Column {
		c := w.column_new(name, .Int, len(vals))
		for v in vals {if v == NA {w.append_null(&c)} else {w.append_int(&c, v)}}
		return c
	}
	strs :: proc(name: string, vals: []string) -> w.Column {
		c := w.column_new(name, .String, len(vals))
		for v in vals {if v == "" {w.append_null(&c)} else {w.append_string(&c, v)}}
		return c
	}
	bools :: proc(name: string, vals: []int) -> w.Column {
		c := w.column_new(name, .Bool, len(vals))
		for v in vals {if v == NA {w.append_null(&c)} else {w.append_bool(&c, v == 1)}}
		return c
	}
	cell :: proc(c: ^w.Column, r: int) -> string {
		if w.na_at(c, r) {return "NA"}
		if c.type == .Float {
			v, _ := w.column_at_float(c, r)
			return fmt.tprintf("%.3f", v)
		}
		return w.column_value_string(c, r)
	}
	col_text :: proc(c: ^w.Column) -> string {
		out := ""
		for i in 0 ..< c.len {out = fmt.tprintf("%s%s%s", out, " " if i > 0 else "", cell(c, i))}
		return out
	}
	// "<col> | <col> | ... # <row labels>", the format of the pandas reference
	frame_text :: proc(df: ^w.DataFrame) -> string {
		out := ""
		for &c, i in df.columns {
			out = fmt.tprintf("%s%s%s", out, " | " if i > 0 else "", col_text(&c))
		}
		out = fmt.tprintf("%s #", out)
		for r in 0 ..< df.rows {out = fmt.tprintf("%s %d", out, w.pretty_row_label(df, r))}
		return out
	}
	check :: proc(passed, total: ^int, df: ^w.DataFrame, ok: bool, want, name: string) {
		got := frame_text(df) if ok else "<not ok>"
		expect(passed, total, ok && got == want, fmt.tprintf("%s: %s", name, got))
		if ok && got != want {fmt.printf("     want: %s\n", want)}
	}

	nan := math.nan_f64()
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	// px: NaN at 0, 3, 5, 6, 7 and a null at 2
	w.add_column(&df, floats("px", {nan, 2.5, 0, nan, 5.5, nan, nan, nan}, {2}))
	w.add_column(&df, ints("vol", {10, NA, 30, NA, NA, 60, 70, NA}))
	w.add_column(&df, strs("tk", {"a", "", "c", "d", "", "", "g", "h"}))
	w.add_column(&df, bools("flag", {1, NA, 0, NA, 1, NA, NA, 0}))

	// ---------------- counting and masks
	{
		counts := w.count_na(&df)
		defer w.destroy_dataframe(&counts)
		expect(
			&passed,
			&total,
			col_text(w.column(&counts, "column")) == "px vol tk flag" &&
			col_text(w.column(&counts, "na")) == "6 4 3 4" &&
			w.count_na(w.column(&df, "px")) == 6,
			fmt.tprintf("count_na per column: %s", col_text(w.column(&counts, "na"))),
		)
		m := w.isna(w.column(&df, "px"))
		nm := w.notna(w.column(&df, "px"))
		defer delete(m)
		defer delete(nm)
		bits := ""
		for v, i in m {
			bits = fmt.tprintf("%s%s%d", bits, " " if i > 0 else "", 1 if v else 0)
			if nm[i] == v {bits = "notna is not the complement"; break}
		}
		expect(&passed, &total, bits == "1 0 1 1 0 1 1 1", fmt.tprintf("isna(px) = %s, notna its complement", bits))
	}

	// ---------------- dropna
	{
		r, ok := w.dropna(&df)
		check(&passed, &total, &r, ok && len(r.columns) == 4, " |  |  |  #", "dropna(): every row has an NA, columns kept")
		w.destroy_dataframe(&r)

		r, ok = w.dropna(&df, {"px", "vol"}, .All)
		check(
			&passed,
			&total,
			&r,
			ok,
			"NA 2.500 NA 5.500 NA NA | 10 NA 30 NA 60 70 | a NA c NA NA g | true NA false true NA NA # 0 1 2 4 5 6",
			"dropna(subset px vol, how all)",
		)
		w.destroy_dataframe(&r)

		r, ok = w.dropna(&df, {"vol", "tk"})
		check(&passed, &total, &r, ok, "NA NA NA | 10 30 70 | a c g | true false NA # 0 2 6", "dropna(subset vol tk)")
		w.destroy_dataframe(&r)

		r, ok = w.dropna(&df, thresh = 3)
		check(&passed, &total, &r, ok, "NA NA | 10 30 | a c | true false # 0 2", "dropna(thresh 3)")
		w.destroy_dataframe(&r)

		view := w.iloc_range(&df, 2, 8)
		defer w.destroy_dataframe(&view)
		r, ok = w.dropna(&view, {"vol"})
		check(&passed, &total, &r, ok, "NA NA NA | 30 60 70 | c NA g | false NA NA # 2 5 6", "dropna on a view (rows 2..7)")
		w.destroy_dataframe(&r)
	}

	// ---------------- fillna
	{
		r, ok := w.fillna(&df, 0)
		check(
			&passed,
			&total,
			&r,
			ok,
			"0.000 2.500 0.000 0.000 5.500 0.000 0.000 0.000 | 10 0 30 0 0 60 70 0 | a NA c d NA NA g h | true NA false NA true NA NA false # 0 1 2 3 4 5 6 7",
			"fillna(0): fills the Int and Float columns, others unchanged",
		)
		w.destroy_dataframe(&r)

		tk, tok := w.fillna(w.column(&df, "tk"), "?")
		expect(&passed, &total, tok && col_text(&tk) == "a ? c d ? ? g h" && tk.name == "tk", fmt.tprintf("fillna(tk, \"?\") = %s", col_text(&tk)))
		w.destroy_column(&tk)

		dates := w.column_new("d", .Date, 3)
		w.append_date(&dates, w.Date{2024, 1, 2})
		w.append_null(&dates)
		w.append_date(&dates, w.Date{2024, 1, 4})
		dfill, dok := w.fillna(&dates, w.Date{2024, 1, 3})
		expect(&passed, &total, dok && col_text(&dfill) == "2024-01-02 2024-01-03 2024-01-04", fmt.tprintf("fillna on a Date column: %s", col_text(&dfill)))
		w.destroy_column(&dfill)
		w.destroy_column(&dates)

		_, bad1 := w.fillna(&df, "x", {"vol"})
		_, bad2 := w.fillna(&df, 0, {"nope"})
		_, bad3 := w.fillna(w.column(&df, "vol"), 1.5)
		expect(&passed, &total, !bad1 && !bad2 && !bad3, "fillna: ok = false for a mismatched type or unknown column")
	}

	// ---------------- ffill / bfill
	{
		r, ok := w.ffill(&df)
		check(
			&passed,
			&total,
			&r,
			ok,
			"NA 2.500 2.500 2.500 5.500 5.500 5.500 5.500 | 10 10 30 30 30 60 70 70 | a a c d d d g h | true true false false true true true false # 0 1 2 3 4 5 6 7",
			"ffill()",
		)
		w.destroy_dataframe(&r)

		r, ok = w.ffill(&df, limit = 1)
		check(
			&passed,
			&total,
			&r,
			ok,
			"NA 2.500 2.500 NA 5.500 5.500 NA NA | 10 10 30 30 NA 60 70 70 | a a c d d NA g h | true true false false true true NA false # 0 1 2 3 4 5 6 7",
			"ffill(limit 1)",
		)
		w.destroy_dataframe(&r)

		r, ok = w.bfill(&df)
		check(
			&passed,
			&total,
			&r,
			ok,
			"2.500 2.500 5.500 5.500 5.500 NA NA NA | 10 30 30 60 60 60 70 NA | a c c d g g g h | true false false true true false false false # 0 1 2 3 4 5 6 7",
			"bfill()",
		)
		w.destroy_dataframe(&r)

		r, ok = w.bfill(&df, limit = 1)
		check(
			&passed,
			&total,
			&r,
			ok,
			"2.500 2.500 NA 5.500 5.500 NA NA NA | 10 30 30 NA 60 60 70 NA | a c c d NA g g h | true false false true true NA false false # 0 1 2 3 4 5 6 7",
			"bfill(limit 1)",
		)
		w.destroy_dataframe(&r)

		view := w.iloc_range(&df, 3, 8)
		defer w.destroy_dataframe(&view)
		r, ok = w.ffill(&view)
		check(
			&passed,
			&total,
			&r,
			ok,
			"NA 5.500 5.500 5.500 5.500 | NA NA 60 70 70 | d d d g h | NA true true true false # 3 4 5 6 7",
			"ffill on a view (rows 3..7)",
		)
		w.destroy_dataframe(&r)

		// only the named column is filled
		r, ok = w.ffill(&df, {"vol"})
		vol_ok := ok && col_text(w.column(&r, "vol")) == "10 10 30 30 30 60 70 70" && col_text(w.column(&r, "tk")) == col_text(w.column(&df, "tk"))
		if ok {w.destroy_dataframe(&r)}
		_, bad1 := w.ffill(&df, limit = -1)
		_, bad2 := w.bfill(&df, {"nope"})
		_, bad3 := w.ffill(w.column(&df, "px"), -2)
		expect(&passed, &total, vol_ok && !bad1 && !bad2 && !bad3, "ffill(columns) fills only those; ok = false for a negative limit or unknown column")
	}

	// ---------------- index metadata survives
	{
		idx := w.dataframe_new()
		defer w.destroy_dataframe(&idx)
		w.add_column(&idx, ints("key", {1, 2, 3, 4}))
		w.add_column(&idx, floats("v", {1.5, nan, 3.5, nan}, {}))
		w.set_index(&idx, "key", drop = true)
		f, fok := w.ffill(&idx)
		d, dok := w.dropna(&idx)
		defer if fok {w.destroy_dataframe(&f)}
		defer if dok {w.destroy_dataframe(&d)}
		hit := w.loc(&d, 3) if dok else w.DataFrame{}
		defer w.destroy_dataframe(&hit)
		expect(
			&passed,
			&total,
			fok && dok && f.has_index && f.index_column == "key" && d.has_index &&
			col_text(w.column(&f, "v")) == "1.500 1.500 3.500 3.500" &&
			hit.rows == 1 && col_text(w.column(&hit, "v")) == "3.500",
			"ffill / dropna keep a dropped index: loc(3) on the dropna result",
		)
	}

	fmt.printf("Missing data summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
