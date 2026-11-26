package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"

// resample against pandas 1.5.3 (df.resample(freq, on="date").agg(...)) on
// twelve daily rows in shuffled order, with a NaN price and gaps of whole
// weeks and months: day, week (ending Sunday, and Friday as "W-FRI"), month,
// quarter and year, each with first / max / min / last (OHLC), sum, count and
// mean. Empty periods must be present (sum and count 0, the rest null), labels
// are period ends. Also: groups with `by` (each from its own first to its own
// last period, as groupby("tk").resample("M")), a Datetime key, and the
// ok = false contracts.

resample_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Resample ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	// Does `c` hold `want` (space-separated; "NA" = missing; numbers to 1e-12)?
	col_is :: proc(c: ^w.Column, want: string) -> bool {
		toks := strings.fields(want, context.temp_allocator)
		if c == nil || len(toks) != c.len {return false}
		for t, i in toks {
			if t == "NA" || w.na_at(c, i) {
				if !(t == "NA" && w.na_at(c, i)) {return false}
				continue
			}
			if c.type == .Int || c.type == .Float {
				x, _ := strconv.parse_f64(t)
				v, _ := w.column_at_numeric(c, i)
				if abs(v - x) > 1e-12 * max(1, abs(x)) {return false}
			} else if w.column_value_string(c, i) != t {
				return false
			}
		}
		return true
	}
	// want: "name: v v v | name: v v v", the format of the pandas reference;
	// `first` > 0 compares only the first rows of every column
	frame_is :: proc(df: ^w.DataFrame, want: string, first := 0) -> bool {
		parts := strings.split(want, " | ", context.temp_allocator)
		if len(parts) != len(df.columns) {return false}
		for p, k in parts {
			colon := strings.index(p, ": ")
			if colon < 0 || df.columns[k].name != p[:colon] {return false}
			c := &df.columns[k]
			if first > 0 {
				view := w.column_slice_view(c, 0, min(first, c.len))
				if !col_is(&view, p[colon + 2:]) {return false}
			} else if !col_is(c, p[colon + 2:]) {
				return false
			}
		}
		return true
	}
	show :: proc(df: ^w.DataFrame) -> string {
		out := ""
		for &c, k in df.columns {
			out = fmt.tprintf("%s%s%s:", out, " | " if k > 0 else "", c.name)
			for i in 0 ..< c.len {
				out = fmt.tprintf(
					"%s %s",
					out,
					"NA" if w.na_at(&c, i) else w.column_value_string(&c, i),
				)
			}
		}
		return out
	}

	// the reference rows, in pandas' shuffled order
	days := []w.Date {
		{2024, 2, 6},
		{2024, 2, 5},
		{2024, 1, 31},
		{2024, 2, 1},
		{2025, 1, 2},
		{2024, 2, 20},
		{2024, 2, 21},
		{2024, 1, 30},
		{2024, 2, 2},
		{2024, 4, 2},
		{2024, 3, 28},
		{2024, 12, 31},
	}
	nan := math.nan_f64()
	df := w.df_from(
		w.column_from_dates("date", days),
		w.column_from_floats("px", {13, 11.5, 11, nan, 18, 14, 13.5, 10, 12, 16, 15, 17}),
		w.column_from_ints("vol", {600, 500, 200, 300, 1200, 700, 800, 100, 400, 1000, 900, 1100}),
	)
	defer w.destroy_dataframe(&df)
	specs := []w.Resample_Spec {
		{"px", .First, "open"},
		{"px", .Max, "high"},
		{"px", .Min, "low"},
		{"px", .Last, "close"},
		{"vol", .Sum, ""},
		{"px", .Count, "n"},
		{"px", .Mean, "mean"},
	}

	// ---------------- every frequency
	{
		Case :: struct {
			freq:     w.Resample_Freq,
			week_end: w.Days,
			want:     string,
		}
		cases := []Case {
			{
				.Week,
				.Sunday,
				"date: 2024-02-04 2024-02-11 2024-02-18 2024-02-25 2024-03-03 2024-03-10 2024-03-17 2024-03-24 2024-03-31 2024-04-07 2024-04-14 2024-04-21 2024-04-28 2024-05-05 2024-05-12 2024-05-19 2024-05-26 2024-06-02 2024-06-09 2024-06-16 2024-06-23 2024-06-30 2024-07-07 2024-07-14 2024-07-21 2024-07-28 2024-08-04 2024-08-11 2024-08-18 2024-08-25 2024-09-01 2024-09-08 2024-09-15 2024-09-22 2024-09-29 2024-10-06 2024-10-13 2024-10-20 2024-10-27 2024-11-03 2024-11-10 2024-11-17 2024-11-24 2024-12-01 2024-12-08 2024-12-15 2024-12-22 2024-12-29 2025-01-05 | open: 10.0 11.5 NA 14.0 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 17.0 | high: 12.0 13.0 NA 14.0 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 18.0 | low: 10.0 11.5 NA 13.5 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 17.0 | close: 12.0 13.0 NA 13.5 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 18.0 | vol: 1000 1100 0 1500 0 0 0 0 900 1000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2300 | n: 3 2 0 2 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 | mean: 11.0 12.25 NA 13.75 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 17.5",
			},
			{
				.Week,
				.Friday,
				"date: 2024-02-02 2024-02-09 2024-02-16 2024-02-23 2024-03-01 2024-03-08 2024-03-15 2024-03-22 2024-03-29 2024-04-05 2024-04-12 2024-04-19 2024-04-26 2024-05-03 2024-05-10 2024-05-17 2024-05-24 2024-05-31 2024-06-07 2024-06-14 2024-06-21 2024-06-28 2024-07-05 2024-07-12 2024-07-19 2024-07-26 2024-08-02 2024-08-09 2024-08-16 2024-08-23 2024-08-30 2024-09-06 2024-09-13 2024-09-20 2024-09-27 2024-10-04 2024-10-11 2024-10-18 2024-10-25 2024-11-01 2024-11-08 2024-11-15 2024-11-22 2024-11-29 2024-12-06 2024-12-13 2024-12-20 2024-12-27 2025-01-03 | open: 10.0 11.5 NA 14.0 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 17.0 | high: 12.0 13.0 NA 14.0 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 18.0 | low: 10.0 11.5 NA 13.5 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 17.0 | close: 12.0 13.0 NA 13.5 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 18.0 | vol: 1000 1100 0 1500 0 0 0 0 900 1000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2300 | n: 3 2 0 2 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 | mean: 11.0 12.25 NA 13.75 NA NA NA NA 15.0 16.0 NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA 17.5",
			},
			{
				.Month,
				.Sunday,
				"date: 2024-01-31 2024-02-29 2024-03-31 2024-04-30 2024-05-31 2024-06-30 2024-07-31 2024-08-31 2024-09-30 2024-10-31 2024-11-30 2024-12-31 2025-01-31 | open: 10.0 12.0 15.0 16.0 NA NA NA NA NA NA NA 17.0 18.0 | high: 11.0 14.0 15.0 16.0 NA NA NA NA NA NA NA 17.0 18.0 | low: 10.0 11.5 15.0 16.0 NA NA NA NA NA NA NA 17.0 18.0 | close: 11.0 13.5 15.0 16.0 NA NA NA NA NA NA NA 17.0 18.0 | vol: 300 3300 900 1000 0 0 0 0 0 0 0 1100 1200 | n: 2 5 1 1 0 0 0 0 0 0 0 1 1 | mean: 10.5 12.8 15.0 16.0 NA NA NA NA NA NA NA 17.0 18.0",
			},
			{
				.Quarter,
				.Sunday,
				"date: 2024-03-31 2024-06-30 2024-09-30 2024-12-31 2025-03-31 | open: 10.0 16.0 NA 17.0 18.0 | high: 15.0 16.0 NA 17.0 18.0 | low: 10.0 16.0 NA 17.0 18.0 | close: 15.0 16.0 NA 17.0 18.0 | vol: 4500 1000 0 1100 1200 | n: 8 1 0 1 1 | mean: 12.5 16.0 NA 17.0 18.0",
			},
			{
				.Year,
				.Sunday,
				"date: 2024-12-31 2025-12-31 | open: 10.0 18.0 | high: 17.0 18.0 | low: 10.0 18.0 | close: 17.0 18.0 | vol: 6600 1200 | n: 10 1 | mean: 13.3 18.0",
			},
		}
		bad := 0
		for c in cases {
			r, ok := w.resample(&df, "date", c.freq, specs, week_end = c.week_end)
			if !(ok && frame_is(&r, c.want)) {
				bad += 1
				fmt.printf(
					"     %v (%v): got %s\n     want: %s\n",
					c.freq,
					c.week_end,
					show(&r) if ok else "not ok",
					c.want,
				)
			}
			if ok {w.destroy_dataframe(&r)}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			"resample by week (Sunday, Friday), month, quarter, year: OHLC, sum, count, mean, empty periods",
		)

		d, ok := w.resample(&df, "date", .Day, specs)
		good :=
			ok &&
			d.rows == 339 &&
			frame_is(
				&d,
				"date: 2024-01-30 2024-01-31 2024-02-01 2024-02-02 2024-02-03 2024-02-04 2024-02-05 2024-02-06 | open: 10.0 11.0 NA 12.0 NA NA 11.5 13.0 | high: 10.0 11.0 NA 12.0 NA NA 11.5 13.0 | low: 10.0 11.0 NA 12.0 NA NA 11.5 13.0 | close: 10.0 11.0 NA 12.0 NA NA 11.5 13.0 | vol: 100 200 300 400 0 0 500 600 | n: 1 1 0 1 0 0 1 1 | mean: 10.0 11.0 NA 12.0 NA NA 11.5 13.0",
				first = 8,
			)
		if ok {w.destroy_dataframe(&d)}
		expect(
			&passed,
			&total,
			good,
			"resample by day: 339 days from 2024-01-30 to 2025-01-02, the first week matches",
		)
	}

	// ---------------- groups, Datetime key
	{
		t := w.df_from(
			w.column_from_strings("tk", {"A", "B", "A", "B", "A", "B"}),
			w.column_from_dates(
				"date",
				{
					{2024, 1, 31},
					{2024, 1, 15},
					{2024, 3, 1},
					{2024, 2, 10},
					{2024, 3, 15},
					{2024, 2, 11},
				},
			),
			w.column_from_floats("px", {1, 2, 3, 4, 5, 6}),
		)
		defer w.destroy_dataframe(&t)
		r, ok := w.resample(&t, "date", .Month, {{"px", .Last, ""}}, by = {"tk"})
		good :=
			ok &&
			frame_is(
				&r,
				"tk: A A A B B | date: 2024-01-31 2024-02-29 2024-03-31 2024-01-31 2024-02-29 | px: 1.0 NA 5.0 2.0 6.0",
			)
		if !good {fmt.println("     by:", show(&r) if ok else "not ok")}
		if ok {w.destroy_dataframe(&r)}

		dt := w.df_from(
			w.column_from_datetimes(
				"ts",
				{{2024, 3, 4, 9, 30, 0, 0}, {2024, 3, 4, 16, 0, 0, 0}, {2024, 3, 6, 10, 0, 0, 0}},
			),
			w.column_from_floats("px", {1, 2, 3}),
		)
		defer w.destroy_dataframe(&dt)
		rd, dok := w.resample(&dt, "ts", .Day, {{"px", .Last, "close"}})
		dgood :=
			dok &&
			rd.rows == 3 &&
			rd.columns[0].type == .Datetime &&
			w.column_value_string(&rd.columns[0], 0) == "2024-03-04 00:00:00" &&
			w.column_value_string(&rd.columns[0], 2) == "2024-03-06 00:00:00" &&
			col_is(w.column(&rd, "close"), "2.0 NA 3.0")
		if dok {w.destroy_dataframe(&rd)}
		expect(
			&passed,
			&total,
			good && dgood,
			"resample by ticker (each from its own first to last month); a Datetime key gives midnight labels",
		)
	}

	// ---------------- contracts
	{
		_, bad1 := w.resample(&df, "nope", .Month, specs)
		_, bad2 := w.resample(&df, "px", .Month, specs) // not a date
		s := w.df_from(
			w.column_from_dates("date", {{2024, 1, 1}}),
			w.column_from_strings("tk", {"A"}),
		)
		defer w.destroy_dataframe(&s)
		_, bad3 := w.resample(&s, "date", .Month, {{"tk", .Mean, ""}}) // mean of text
		_, bad4 := w.resample(&df, "date", .Month, {{"px", .First, "x"}, {"px", .Last, "x"}}) // same name
		_, bad5 := w.resample(&df, "date", .Month, {{"px", .First, "date"}}) // clashes with the label
		r, ok := w.resample(&s, "date", .Month, {{"tk", .Last, ""}})
		text := ok && frame_is(&r, "date: 2024-01-31 | tk: A")
		if ok {w.destroy_dataframe(&r)}
		expect(
			&passed,
			&total,
			!bad1 && !bad2 && !bad3 && !bad4 && !bad5 && text,
			"resample: Last of text works; ok = false for bad columns, Mean of text, duplicate names",
		)
	}

	fmt.printf("Resample summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
