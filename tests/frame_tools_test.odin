package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

// DataFrame housekeeping in core (head / tail, drop_columns, rename_column,
// concat_rows, unique / nunique / value_counts, duplicated / drop_duplicates,
// cast_column / cast_columns) against pandas 1.5.3 on the same frame: a String,
// an Int and a Float column with nulls, NaN and both signs of zero, so that
// missing cells and -0.0 / 0.0 must compare equal. Cells are compared as text
// with "NA" for missing; row labels with the pandas index. Also: views as
// input, the index surviving each operation, and the ok = false contracts.

frame_tools_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DataFrame Housekeeping ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	NA :: -999 // marks a null cell in the builders below
	ints :: proc(name: string, vals: []int) -> w.Column {
		c := w.column_new(name, .Int, len(vals))
		for v in vals {if v == NA {w.append_null(&c)} else {w.append_int(&c, v)}}
		return c
	}
	strs :: proc(name: string, vals: []string) -> w.Column {
		c := w.column_new(name, .String, len(vals))
		for v in vals {if v == "<NA>" {w.append_null(&c)} else {w.append_string(&c, v)}}
		return c
	}
	cell :: proc(c: ^w.Column, r: int) -> string {
		if w.na_at(c, r) {return "NA"}
		if c.type == .Float {
			v, _ := w.column_at_float(c, r)
			return fmt.tprintf("%.3f", v + 0.0) // -0.0 prints as 0.000, as in the reference
		}
		return w.column_value_string(c, r)
	}
	col_text :: proc(c: ^w.Column) -> string {
		out := ""
		for i in 0 ..< c.len {out = fmt.tprintf("%s%s%s", out, " " if i > 0 else "", cell(c, i))}
		return out
	}
	frame_text :: proc(df: ^w.DataFrame) -> string {
		out := ""
		for &c, i in df.columns {
			out = fmt.tprintf("%s%s%s", out, " | " if i > 0 else "", col_text(&c))
		}
		out = fmt.tprintf("%s #", out)
		for r in 0 ..< df.rows {out = fmt.tprintf("%s %d", out, w.pretty_row_label(df, r))}
		return out
	}
	bits :: proc(m: []bool) -> string {
		out := ""
		for v, i in m {out = fmt.tprintf("%s%s%d", out, " " if i > 0 else "", 1 if v else 0)}
		return out
	}
	check :: proc(passed, total: ^int, got, want, name: string) {
		expect(passed, total, got == want, fmt.tprintf("%s: %s", name, got))
		if got != want {fmt.printf("     want: %s\n", want)}
	}

	nan := math.nan_f64()
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	w.add_column(&df, strs("key", {"b", "a", "b", "c", "a", "b", "<NA>", "c", "<NA>"}))
	w.add_column(&df, ints("x", {1, 2, 1, 3, 2, 1, NA, 3, NA}))
	y := w.column_new("y", .Float, 9)
	for v, i in ([]f64{0.5, nan, 0.5, 0.0, 0, 2.5, -0.0, -0.0, 0.0}) {
		if i == 4 {w.append_null(&y)} else {w.append_float(&y, v)} 	// row 4: null, row 1: NaN
	}
	w.add_column(&df, y)

	// ---------------- head / tail
	{
		cases := []struct {
			n:          int,
			head, tail: string,
		} {
			{
				3,
				"b a b | 1 2 1 | 0.500 NA 0.500 # 0 1 2",
				"NA c NA | NA 3 NA | 0.000 0.000 0.000 # 6 7 8",
			},
			{
				-3,
				"b a b c a b | 1 2 1 3 2 1 | 0.500 NA 0.500 0.000 NA 2.500 # 0 1 2 3 4 5",
				"c a b NA c NA | 3 2 1 NA 3 NA | 0.000 NA 2.500 0.000 0.000 0.000 # 3 4 5 6 7 8",
			},
			{0, " |  |  #", " |  |  #"},
		}
		got, want := "", ""
		for c in cases {
			h := w.head(&df, c.n)
			t := w.tail(&df, c.n)
			got = fmt.tprintf("%s[%s][%s]", got, frame_text(&h), frame_text(&t))
			want = fmt.tprintf("%s[%s][%s]", want, c.head, c.tail)
			w.destroy_dataframe(&h)
			w.destroy_dataframe(&t)
		}
		big := w.tail(&df, 20)
		all := frame_text(&big) == frame_text(&df)
		w.destroy_dataframe(&big)
		expect(
			&passed,
			&total,
			got == want && all,
			"head / tail with n = 3, -3, 0 and n beyond the row count",
		)
		if got != want {fmt.printf("     got:  %s\n     want: %s\n", got, want)}
	}

	// ---------------- duplicated / drop_duplicates
	{
		Case :: struct {
			cols: []string,
			keep: w.Keep,
			want: string,
		}
		cases := []Case {
			{nil, .First, "0 0 1 0 1 0 0 1 1"},
			{{"key", "x"}, .First, "0 0 1 0 1 1 0 1 1"},
			{nil, .Last, "1 1 0 1 0 0 1 0 0"},
			{{"key", "x"}, .Last, "1 1 1 1 0 0 1 0 0"},
			{nil, .None, "1 1 1 1 1 0 1 1 1"},
			{{"key", "x"}, .None, "1 1 1 1 1 1 1 1 1"},
		}
		got, want := "", ""
		for c in cases {
			m, ok := w.duplicated(&df, c.cols, c.keep)
			got = fmt.tprintf("%s[%s]", got, bits(m) if ok else "not ok")
			want = fmt.tprintf("%s[%s]", want, c.want)
			delete(m)
		}
		expect(
			&passed,
			&total,
			got == want,
			"duplicated: all columns and subset key,x, keep first / last / none (null = NaN, -0.0 = 0.0)",
		)
		if got != want {fmt.printf("     got:  %s\n     want: %s\n", got, want)}

		r, ok := w.drop_duplicates(&df)
		check(
			&passed,
			&total,
			frame_text(&r) if ok else "not ok",
			"b a c b NA | 1 2 3 1 NA | 0.500 NA 0.000 2.500 0.000 # 0 1 3 5 6",
			"drop_duplicates()",
		)
		w.destroy_dataframe(&r)
		r, ok = w.drop_duplicates(&df, {"key", "x"}, .Last)
		check(
			&passed,
			&total,
			frame_text(&r) if ok else "not ok",
			"a b c NA | 2 1 3 NA | NA 2.500 0.000 0.000 # 4 5 7 8",
			"drop_duplicates(subset key x, keep last)",
		)
		w.destroy_dataframe(&r)
		r, ok = w.drop_duplicates(&df, {"y"}, .None)
		check(
			&passed,
			&total,
			frame_text(&r) if ok else "not ok",
			"b | 1 | 2.500 # 5",
			"drop_duplicates(subset y, keep none)",
		)
		w.destroy_dataframe(&r)

		view := w.iloc_range(&df, 3, 9)
		defer w.destroy_dataframe(&view)
		r, ok = w.drop_duplicates(&view, {"key"})
		check(
			&passed,
			&total,
			frame_text(&r) if ok else "not ok",
			"c a b NA | 3 2 1 NA | 0.000 NA 2.500 0.000 # 3 4 5 6",
			"drop_duplicates on a view (rows 3..8)",
		)
		w.destroy_dataframe(&r)

		_, bad := w.duplicated(&df, {"nope"})
		expect(&passed, &total, !bad, "duplicated: ok = false for an unknown column")
	}

	// ---------------- unique / nunique / value_counts
	{
		key := w.column(&df, "key")
		yc := w.column(&df, "y")
		uk := w.unique(key)
		uy := w.unique(yc)
		defer w.destroy_column(&uk)
		defer w.destroy_column(&uy)
		expect(
			&passed,
			&total,
			col_text(&uk) == "b a c NA" &&
			w.nunique(key) == 3 &&
			w.nunique(key, dropna = false) == 4 &&
			col_text(&uy) == "0.500 NA 0.000 2.500" &&
			w.nunique(yc) == 3 &&
			w.nunique(yc, dropna = false) == 4 &&
			uk.name == "key",
			fmt.tprintf("unique / nunique: key %q, y %q", col_text(&uk), col_text(&uy)),
		)

		vk := w.value_counts(key)
		vn := w.value_counts(key, dropna = false)
		vy := w.value_counts(yc)
		defer w.destroy_dataframe(&vk)
		defer w.destroy_dataframe(&vn)
		defer w.destroy_dataframe(&vy)
		got := fmt.tprintf(
			"%s / %s; %s / %s; %s / %s",
			col_text(w.column(&vk, "key")),
			col_text(w.column(&vk, "count")),
			col_text(w.column(&vn, "key")),
			col_text(w.column(&vn, "count")),
			col_text(w.column(&vy, "y")),
			col_text(w.column(&vy, "count")),
		)
		check(
			&passed,
			&total,
			got,
			"b a c / 3 2 2; b a c NA / 3 2 2 2; 0.000 0.500 2.500 / 4 2 1",
			"value_counts (key, key with NA, y)",
		)
	}

	// ---------------- drop_columns / rename_column
	{
		idx := w.dataframe_new()
		defer w.destroy_dataframe(&idx)
		w.add_column(&idx, ints("id", {10, 20, 30}))
		w.add_column(&idx, ints("a", {1, 2, 3}))
		w.add_column(&idx, ints("b", {4, 5, 6}))
		w.set_index(&idx, "id")

		d, ok := w.drop_columns(&idx, {"b"})
		hit := w.loc(&d, 20) if ok else w.DataFrame{}
		names_ok :=
			ok && len(d.columns) == 2 && d.columns[0].name == "id" && d.columns[1].name == "a"
		di, iok := w.drop_columns(&idx, {"id"})
		expect(
			&passed,
			&total,
			names_ok &&
			d.has_index &&
			hit.rows == 1 &&
			col_text(w.column(&hit, "a")) == "2" &&
			iok &&
			!di.has_index &&
			len(di.columns) == 2,
			"drop_columns keeps the index (loc 20 -> a = 2); dropping the index column removes the index",
		)
		w.destroy_dataframe(&hit)
		if ok {w.destroy_dataframe(&d)}
		if iok {w.destroy_dataframe(&di)}
		_, bad := w.drop_columns(&idx, {"a", "nope"})

		r1 := w.rename_column(&idx, "a", "alpha")
		r2 := w.rename_column(&idx, "id", "key")
		r3 := w.rename_column(&idx, "alpha", "b") // taken
		r4 := w.rename_column(&idx, "nope", "c")
		hit2 := w.loc(&idx, 30)
		defer w.destroy_dataframe(&hit2)
		view := w.head(&idx, 2)
		defer w.destroy_dataframe(&view)
		r5 := w.rename_column(&view, "b", "beta")
		expect(
			&passed,
			&total,
			!bad &&
			r1 &&
			r2 &&
			!r3 &&
			!r4 &&
			!r5 &&
			idx.columns[1].name == "alpha" &&
			idx.index_column == "key" &&
			w.column(&idx, "alpha") == &idx.columns[1] &&
			col_text(w.column(&hit2, "alpha")) == "3",
			"rename_column renames in place (also the index column); ok = false for a taken or unknown name or a view",
		)
	}

	// ---------------- concat_rows
	{
		p1 := w.iloc_range(&df, 5, 9) // views as input
		p2 := w.head(&df, 2)
		defer w.destroy_dataframe(&p1)
		defer w.destroy_dataframe(&p2)
		// p3 has the same columns in a different order
		p3 := w.dataframe_new()
		defer w.destroy_dataframe(&p3)
		w.add_column(&p3, w.column_from_floats("y", {9.5}))
		w.add_column(&p3, ints("x", {7}))
		w.add_column(&p3, strs("key", {"z"}))
		r, ok := w.concat_rows({&p1, &p2, &p3})
		check(
			&passed,
			&total,
			frame_text(&r) if ok else "not ok",
			"b NA c NA b a z | 1 NA 3 NA 1 2 7 | 2.500 0.000 0.000 0.000 0.500 NA 9.500 # 0 1 2 3 4 5 6",
			"concat_rows of two views and a frame with reordered columns",
		)
		if ok {w.destroy_dataframe(&r)}

		wrong := w.dataframe_new()
		defer w.destroy_dataframe(&wrong)
		w.add_column(&wrong, strs("key", {"z"}))
		w.add_column(&wrong, w.column_from_floats("x", {7}))
		w.add_column(&wrong, w.column_from_floats("y", {1}))
		_, bad1 := w.concat_rows({&df, &wrong}) // x is Float here
		_, bad2 := w.concat_rows({})

		a := w.dataframe_new()
		b := w.dataframe_new()
		defer w.destroy_dataframe(&a)
		defer w.destroy_dataframe(&b)
		w.add_column(&a, ints("k", {1, 2}))
		w.add_column(&a, ints("v", {10, 20}))
		w.add_column(&b, ints("k", {3}))
		w.add_column(&b, ints("v", {30}))
		w.set_index(&a, "k", drop = true)
		w.set_index(&b, "k", drop = true)
		c, cok := w.concat_rows({&a, &b})
		hit := w.loc(&c, 3) if cok else w.DataFrame{}
		expect(
			&passed,
			&total,
			!bad1 &&
			!bad2 &&
			cok &&
			c.has_index &&
			hit.rows == 1 &&
			col_text(w.column(&hit, "v")) == "30",
			"concat_rows keeps a shared dropped index (loc 3 -> 30); ok = false for a type mismatch or no frames",
		)
		w.destroy_dataframe(&hit)
		if cok {w.destroy_dataframe(&c)}
	}

	// ---------------- cast
	{
		f := w.column_from_floats("f", {1.9, -1.9, 2.0, -0.5})
		defer w.destroy_column(&f)
		fi, ok1 := w.cast_column(&f, .Int)
		defer w.destroy_column(&fi)

		s := strs("s", {"1.5", " 2 ", "x", "", "-3e2", "<NA>"})
		defer w.destroy_column(&s)
		sf, ok2 := w.cast_column(&s, .Float, strict = false)
		defer w.destroy_column(&sf)
		_, strict_fail := w.cast_column(&s, .Float)

		si_src := strs("s", {"12", " -7 ", "1.5", "abc"})
		defer w.destroy_column(&si_src)
		si, ok3 := w.cast_column(&si_src, .Int, strict = false)
		defer w.destroy_column(&si)

		big := w.column_from_floats("big", {1e300, math.inf_f64(1), 3})
		defer w.destroy_column(&big)
		bi, ok4 := w.cast_column(&big, .Int, strict = false)
		defer w.destroy_column(&bi)
		_, big_fail := w.cast_column(&big, .Int)

		check(
			&passed,
			&total,
			fmt.tprintf(
				"%s; %s; %s; %s",
				col_text(&fi),
				col_text(&sf),
				col_text(&si),
				col_text(&bi),
			),
			"1 -1 2 0; 1.500 2.000 NA NA -300.000 NA; 12 -7 NA NA; NA NA 3",
			"cast Float->Int truncates, String->Float / Int parses (strict = false: failures are null)",
		)
		expect(
			&passed,
			&total,
			ok1 && ok2 && ok3 && ok4 && !strict_fail && !big_fail,
			"cast: strict fails on unparseable text and on floats out of Int range",
		)

		bs := strs("b", {"true", "FALSE", "1", "0", "<NA>"})
		defer w.destroy_column(&bs)
		bb, _ := w.cast_column(&bs, .Bool)
		defer w.destroy_column(&bb)
		bf, _ := w.cast_column(&bb, .Float)
		defer w.destroy_column(&bf)
		ds := strs("d", {"2024-02-29", "2024-03-01 13:45:00", "2024-03-02T08:00:05"})
		defer w.destroy_column(&ds)
		dt, ok5 := w.cast_column(&ds, .Datetime)
		defer w.destroy_column(&dt)
		dd, _ := w.cast_column(&dt, .Date)
		defer w.destroy_column(&dd)
		tt, _ := w.cast_column(&dt, .Time)
		defer w.destroy_column(&tt)
		back, _ := w.cast_column(&dd, .String)
		defer w.destroy_column(&back)
		fs, _ := w.cast_column(&f, .String)
		defer w.destroy_column(&fs)
		_, unsupported := w.cast_column(&dd, .Int)
		check(
			&passed,
			&total,
			fmt.tprintf(
				"%s; %s; %s; %s; %s; %s",
				col_text(&bb),
				col_text(&bf),
				col_text(&dt),
				col_text(&tt),
				col_text(&back),
				col_text(&fs),
			),
			"true false true false NA; 1.000 0.000 1.000 0.000 NA; 2024-02-29 00:00:00 2024-03-01 13:45:00 2024-03-02 08:00:05; 00:00:00 13:45:00 08:00:05; 2024-02-29 2024-03-01 2024-03-02; 1.9 -1.9 2.0 -0.5",
			"cast Bool / dates / to String",
		)

		cf, ok6 := w.cast_columns(&df, {"x"}, .Float)
		defer if ok6 {w.destroy_dataframe(&cf)}
		_, bad := w.cast_columns(&df, {"key"}, .Int) // "b" is not an int
		expect(
			&passed,
			&total,
			ok5 &&
			!unsupported &&
			ok6 &&
			w.column(&cf, "x").type == .Float &&
			frame_text(&cf) ==
				"b a b c a b NA c NA | 1.000 2.000 1.000 3.000 2.000 1.000 NA 3.000 NA | 0.500 NA 0.500 0.000 NA 2.500 0.000 0.000 0.000 # 0 1 2 3 4 5 6 7 8" &&
			!bad,
			"cast_columns converts the named columns only; ok = false for Date->Int and unparseable text",
		)
	}

	fmt.printf("Housekeeping summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
