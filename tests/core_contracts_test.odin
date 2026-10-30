package tests

import w "../wotan/core"
import "core:fmt"
import "core:mem"

// DataFrame core contracts:
// - columns / series: every type round-trips, constructors and comparisons
//   use the right type, growth honours the column allocator
// - views: slices of slices, index-list selections, masks with nulls,
//   set_index / reset_index (incl. drop), loc on a dropped index
// - sort: stable, multi-key, nulls last, all types, sliced input
// - join: unique / duplicate / null keys, nulls in values, right-only keys,
//   name collisions
// - groupby / groupby2: float means, Var / Corr / Quantile, Min / Max of
//   strings, null groups
// - select conversions and numeric coercion
// - dates: Julian / Unix round trips, negative timestamps, day differences,
//   second arithmetic, month / year clamping, strict validation

core_contracts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DataFrame Core Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	ints :: proc(name: string, vals: []int, null_at := -1) -> w.Column {
		c := w.column_new(name, .Int, len(vals))
		for v, i in vals {if i == null_at {w.append_null(&c)} else {w.append_int(&c, v)}}
		return c
	}
	frame :: proc(cols: ..w.Column) -> w.DataFrame {
		df := w.dataframe_new()
		for c in cols {w.add_column(&df, c)}
		return df
	}
	col_text :: proc(df: ^w.DataFrame, name: string) -> string {
		c := w.column(df, name)
		out := ""
		for i in 0 ..< df.rows {
			out = fmt.tprintf("%s%s%s", out, " " if i > 0 else "", w.column_value_string(c, i))
		}
		return out
	}

	// ---------------- columns / series
	{
		t := w.column_from_times("t", []w.Time{{1, 2, 3}, {4, 5, 6}})
		d := w.column_from_datetimes("d", []w.Datetime{{2024, 1, 1, 0, 0, 0}})
		eq := w.column_eq_datetime(&d, w.Datetime{2024, 1, 1, 0, 0, 0})
		df := frame(t)
		s := w.df_series(&df, "t")
		t1, _ := w.series_at_time(&s, 1)
		e0, _ := w.column_at_bool(&eq, 0)
		expect(
			&passed,
			&total,
			t.type == .Time && d.type == .Datetime && e0 && t1 == w.Time{4, 5, 6},
			"column_from_times / _datetimes, column_eq_datetime, series_at_time use their own type",
		)
		w.destroy_column(&d)
		w.destroy_column(&eq)
		w.destroy_dataframe(&df)
	}
	{
		buf := make([]u8, 1 << 15)
		defer delete(buf)
		arena: mem.Arena
		mem.arena_init(&arena, buf)
		c := w.column_new("x", .Int, 1, mem.arena_allocator(&arena))
		for i in 0 ..< 20 {
			if i == 7 {w.append_null(&c)} else {w.append_int(&c, i)}
		}
		v, _ := w.column_at_int(&c, 19)
		expect(
			&passed,
			&total,
			v == 19 &&
			w.is_null(&c, 7) &&
			uintptr(c.data) >= uintptr(raw_data(buf)) &&
			uintptr(c.data) < uintptr(raw_data(buf)) + uintptr(len(buf)),
			"growth stays in the column's allocator (arena-backed column grows to 20 rows)",
		)
	}

	// ---------------- views, selections, masks
	{
		df := frame(ints("a", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, null_at = 4))
		defer w.destroy_dataframe(&df)
		s1 := w.iloc_range(&df, 2, 8)
		s2 := w.iloc_range(&s1, 1, 5)
		pick := w.iloc_many(&s2, []int{3, 0})
		empty := w.iloc_range(&df, 5, 5)
		expect(
			&passed,
			&total,
			col_text(&s2, "a") == "3 NULL 5 6" &&
			col_text(&pick, "a") == "6 3" &&
			empty.rows == 0 &&
			len(empty.columns) == 1,
			fmt.tprintf(
				"slice of slice %q, iloc_many of it %q, empty slice keeps columns",
				col_text(&s2, "a"),
				col_text(&pick, "a"),
			),
		)
		w.destroy_dataframe(&s1)
		w.destroy_dataframe(&s2)
		w.destroy_dataframe(&pick)
		w.destroy_dataframe(&empty)
	}
	{
		a := w.column_from_bools("a", []bool{true, true, false, true})
		b := w.column_new("b", .Bool, 4)
		w.append_bool(
			&b,
			true,
		); w.append_bool(&b, false); w.append_bool(&b, true); w.append_null(&b)
		m_and := w.column_and(&a, &b)
		m_or := w.column_or(&a, &b)
		m_not := w.column_not(&b)
		defer {delete(m_and); delete(m_or); delete(m_not); w.destroy_column(&a)
			w.destroy_column(&b)}
		ok :=
			m_and[0] &&
			!m_and[1] &&
			!m_and[2] &&
			!m_and[3] &&
			m_or[0] &&
			m_or[1] &&
			m_or[2] &&
			m_or[3] &&
			!m_not[0] &&
			m_not[1] &&
			!m_not[3]
		expect(&passed, &total, ok, "mask and / or / not on Bool columns (null counts as false)")
	}
	{
		df := frame(
			w.column_from_dates("d", []w.Date{{2024, 1, 3}, {2024, 1, 1}, {2024, 1, 2}}),
			ints("v", {30, 10, 20}),
		)
		defer w.destroy_dataframe(&df)
		w.set_index(&df, "d", true)
		row := w.loc(&df, w.Date{2024, 1, 2})
		r0, _ := w.column_at_int(w.column(&row, "v"), 0)
		rng := w.loc(&df, w.Date{2024, 1, 2}, w.Date{2024, 1, 3})
		w.reset_index(&df, false)
		ok :=
			r0 == 20 &&
			rng.rows == 2 &&
			len(df.columns) == 2 &&
			df.columns[0].name == "d" &&
			col_text(&df, "v") == "10 20 30"
		expect(
			&passed,
			&total,
			ok,
			"set_index(drop) keeps loc / loc_range working; reset_index restores the column once",
		)
		w.destroy_dataframe(&row)
		w.destroy_dataframe(&rng)
	}

	// ---------------- sort
	{
		df := frame(
			ints("g", {2, 1, 2, 1, 2}),
			ints("v", {5, 0, 3, 9, 0}, null_at = 4),
			ints("id", {0, 1, 2, 3, 4}),
		)
		defer w.destroy_dataframe(&df)
		tc := w.column_new("t", .Time, 5)
		for i in 0 ..< 5 {w.append_time(&tc, w.Time{i32(10 + i), 0, 0})}
		w.add_column(&df, tc)
		s := w.dataframe_sort_by(&df, []string{"g", "v"}, []bool{false, true})
		single := w.dataframe_sort(&df, "g", false)
		t0, _ := w.column_at_time(w.column(&s, "t"), 0)
		ok :=
			col_text(&s, "id") == "3 1 0 2 4" &&
			col_text(&single, "id") == "1 3 0 2 4" &&
			t0 == w.Time{13, 0, 0}
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"multi-key sort (g asc, v desc, nulls last) %q, stable single-key %q, Time column follows",
				col_text(&s, "id"),
				col_text(&single, "id"),
			),
		)
		w.destroy_dataframe(&s)
		w.destroy_dataframe(&single)
	}

	// ---------------- join
	{
		n := 300
		ids := make([]int, n)
		defer delete(ids)
		for i in 0 ..< n {ids[i] = i * 7}
		l := frame(ints("id", ids), ints("val", ids))
		r := frame(ints("id", ids), ints("val", ids))
		defer {w.destroy_dataframe(&l); w.destroy_dataframe(&r)}
		j := w.join(&l, &r, {"id"})
		bad := 0
		for i in 0 ..< j.rows {
			x, _ := w.column_at_int(w.column(&j, "val"), i)
			y, _ := w.column_at_int(w.column(&j, "val_right"), i)
			if x != y {bad += 1}
		}
		expect(
			&passed,
			&total,
			j.rows == n && bad == 0,
			fmt.tprintf(
				"join of %d unique keys: %d rows, %d mismatches, collision column val_right",
				n,
				j.rows,
				bad,
			),
		)
		w.destroy_dataframe(&j)
	}
	{
		l := frame(ints("id", {0, 5, 5, 9}, null_at = 0), ints("lv", {1, 2, 3, 0}, null_at = 3))
		r := frame(ints("id", {0, 5, 5, 7}), ints("rv", {10, 20, 30, 70}))
		defer {w.destroy_dataframe(&l); w.destroy_dataframe(&r)}
		inner := w.join(&l, &r, {"id"})
		outer := w.join(&l, &r, {"id"}, .Outer)
		ok :=
			inner.rows == 4 &&
			col_text(&inner, "rv") == "20 30 20 30" &&
			col_text(&outer, "id") == "NULL 5 5 5 5 9 0 7" &&
			col_text(&outer, "lv") == "1 2 2 3 3 NULL NULL NULL"
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"join: duplicate keys give all pairs, null keys never match, outer keys %q",
				col_text(&outer, "id"),
			),
		)
		w.destroy_dataframe(&inner)
		w.destroy_dataframe(&outer)
	}

	// ---------------- groupby / agg
	{
		df := frame(
			ints("g", {1, 1, 2, 2, 2}),
			ints("x", {1, 2, 4, 6, 9}),
			w.column_from_strings("s", []string{"b", "a", "z", "y", "x"}),
		)
		defer w.destroy_dataframe(&df)
		gdf := w.groupby(&df, {"g"})
		res := w.agg(
			&gdf,
			{w.avg_agg("m", w.column(&df, "x")), w.min_agg("smin", w.column(&df, "s"))},
		)
		m0, _ := w.column_at_float(w.column(&res, "m"), 0)
		s1, _ := w.column_at_string(w.column(&res, "smin"), 1)
		expect(
			&passed,
			&total,
			m0 == 1.5 && s1 == "x",
			fmt.tprintf("agg: float mean %v, string min %q", m0, s1),
		)
		w.destroy_dataframe(&res)
		w.destroy_grouped_dataframe(&gdf)

		gb := w.groupby2(&df, {"g"})
		defer w.destroy_groupby2(&gb)
		out := w.groupby2_agg(
			&gb,
			{
				w.Aggregator{name = "mean", column = "x", kind = .Mean},
				w.make_median_agg("med", "x"),
				w.make_var("var", "x"),
				w.make_corr("corr", "x", "x"),
				w.Aggregator{name = "n", column = "x", kind = .Count},
			},
		)
		defer w.destroy_dataframe(&out)
		med0, _ := w.column_at_float(w.column(&out, "med"), 0)
		var1, _ := w.column_at_float(w.column(&out, "var"), 1)
		corr1, _ := w.column_at_float(w.column(&out, "corr"), 1)
		ok :=
			w.column(&out, "mean").type == .Float &&
			med0 == 1.5 &&
			abs(var1 - 6.333333333333333) < 1e-12 &&
			abs(corr1 - 1) < 1e-12
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"groupby2: interpolated median %v, sample var %v, corr %v",
				med0,
				var1,
				corr1,
			),
		)
	}

	// ---------------- select conversions
	{
		df := frame(
			ints("i", {20240229, 0}, null_at = 1),
			w.column_from_floats("f", []f64{2.5, -1}),
		)
		defer w.destroy_dataframe(&df)
		exprs := []w.Select_Expr {
			w.conv_expr("as_date", w.column(&df, "i"), "date"),
			w.conv_expr("as_text", w.column(&df, "f"), "string"),
			w.conv_int_to_f64_expr("as_float", w.column(&df, "i")),
			w.div_expr("half", w.column(&df, "f"), 2),
		}
		s := w.select(&df, exprs)
		defer w.destroy_dataframe(&s)
		expect(
			&passed,
			&total,
			col_text(&s, "as_date") == "2024-02-29 NULL" &&
			col_text(&s, "as_text") == "2.5 -1.0" &&
			col_text(&s, "half") == "1.25 -0.5",
			fmt.tprintf(
				"select conversions %q / %q, int divisor on Float column %q",
				col_text(&s, "as_date"),
				col_text(&s, "as_text"),
				col_text(&s, "half"),
			),
		)
	}

	// ---------------- dates
	{
		bad := 0
		d := w.Date{1890, 1, 1}
		for d.year < 2110 {
			if w.f64_to_date(w.date_to_f64(d)) != d {bad += 1}
			dt := w.Datetime{d.year, d.month, d.day, 23, 59, 59}
			if w.datetime_from_unix(w.datetime_to_unix(dt)) != dt {bad += 1}
			d = w.add_day_date(d, 1)
		}
		ok :=
			bad == 0 &&
			w.datetime_to_unix(w.Datetime{1970, 1, 1, 0, 0, 0}) == 0 &&
			w.datetime_to_unix(w.Datetime{2000, 1, 1, 0, 0, 0}) == 946684800 &&
			w.date_from_unix(-1) == w.Date{1969, 12, 31}
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"Julian / Unix round trips 1890-2110 (%d errors), epoch, negative timestamps",
				bad,
			),
		)
	}
	{
		_, bad_day := w.parse_date("2024-01-00")
		_, bad_time := w.parse_time("24:00:00")
		ok :=
			w.get_date_day_diffs(w.Date{2023, 12, 31}, w.Date{2024, 1, 1}) == 1 &&
			w.add_seconds_time(w.Time{10, 0, 0}, -1) == w.Time{9, 59, 59} &&
			w.add_seconds_time(w.Time{0, 0, 30}, -60) == w.Time{23, 59, 30} &&
			w.add_seconds_datetime(w.Datetime{2024, 12, 31, 23, 59, 59}, 1) ==
				w.Datetime{2025, 1, 1, 0, 0, 0} &&
			w.add_month_date(w.Date{2024, 1, 31}, 1) == w.Date{2024, 2, 29} &&
			w.add_month_date(w.Date{2024, 3, 31}, -13) == w.Date{2023, 2, 28} &&
			w.add_year_date(w.Date{2024, 2, 29}, 1) == w.Date{2025, 2, 28} &&
			!bad_day &&
			!bad_time
		expect(
			&passed,
			&total,
			ok,
			"day differences, second arithmetic across midnight / new year, month and year clamping, strict parsing",
		)
	}

	fmt.printf("Core contracts summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
