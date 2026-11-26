package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

// Date and string parts and the checked core entry points:
// - date_part (all 17 parts on a Datetime column with a null, leap days,
//   ISO weeks that belong to the neighbouring year) and the str_ procedures
//   (upper, lower, strip, len, replace, slice, split_part, contains,
//   startswith, endswith, with non-ASCII text) against pandas 1.5.3 .dt and
//   .str; compare against pandas' comparison operators (except that a missing
//   cell gives false for every operator, where pandas' != gives true)
// - every try_ procedure: the same result as the original on good input,
//   ok = false (and no panic) for each condition under which the original
//   panics

parts_checked_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Date / String Parts and Checked Entry Points ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	// cell texts: "<NA>" for missing, true / false, integers, strings as is
	texts :: proc(c: ^w.Column) -> []string {
		out := make([]string, c.len, context.temp_allocator)
		for i in 0 ..< c.len {
			if w.is_null(c, i) {
				out[i] = "<NA>"
				continue
			}
			out[i] = w.column_value_string(c, i)
		}
		return out
	}
	same :: proc(a, b: []string) -> bool {
		if len(a) != len(b) {return false}
		for x, i in a {if x != b[i] {return false}}
		return true
	}
	bits :: proc(m: []bool) -> string {
		out := ""
		for v, i in m {out = fmt.tprintf("%s%s%d", out, " " if i > 0 else "", 1 if v else 0)}
		return out
	}

	// ---------------- date_part
	{
		ts := w.column_new("ts", .Datetime, 8)
		for v, i in ([]w.Datetime{{2024, 1, 1, 9, 30, 15, 0}, {2024, 2, 29, 23, 59, 59, 0}, {2024, 3, 31, 0, 0, 0, 0}, {2024, 12, 30, 12, 0, 0, 0}, {2021, 1, 3, 7, 5, 9, 0}, {}, {2026, 12, 31, 18, 45, 0, 0}, {2027, 1, 1, 0, 0, 1, 0}}) {
			if i == 5 {w.append_null(&ts)} else {w.append_datetime(&ts, v)}
		}
		defer w.destroy_column(&ts)
		Case :: struct {
			part: w.Date_Part,
			want: []string,
		}
		cases := []Case {
			{.Year, {"2024", "2024", "2024", "2024", "2021", "<NA>", "2026", "2027"}},
			{.Quarter, {"1", "1", "1", "4", "1", "<NA>", "4", "1"}},
			{.Month, {"1", "2", "3", "12", "1", "<NA>", "12", "1"}},
			{.Day, {"1", "29", "31", "30", "3", "<NA>", "31", "1"}},
			{.Weekday, {"0", "3", "6", "0", "6", "<NA>", "3", "4"}},
			{.Day_Of_Year, {"1", "60", "91", "365", "3", "<NA>", "365", "1"}},
			{.Week, {"1", "9", "13", "1", "53", "<NA>", "53", "53"}},
			{.Hour, {"9", "23", "0", "12", "7", "<NA>", "18", "0"}},
			{.Minute, {"30", "59", "0", "0", "5", "<NA>", "45", "0"}},
			{.Second, {"15", "59", "0", "0", "9", "<NA>", "0", "1"}},
			{
				.Is_Month_Start,
				{"true", "false", "false", "false", "false", "<NA>", "false", "true"},
			},
			{.Is_Month_End, {"false", "true", "true", "false", "false", "<NA>", "true", "false"}},
			{
				.Is_Quarter_Start,
				{"true", "false", "false", "false", "false", "<NA>", "false", "true"},
			},
			{
				.Is_Quarter_End,
				{"false", "false", "true", "false", "false", "<NA>", "true", "false"},
			},
			{
				.Is_Year_Start,
				{"true", "false", "false", "false", "false", "<NA>", "false", "true"},
			},
			{.Is_Year_End, {"false", "false", "false", "false", "false", "<NA>", "true", "false"}},
			{.Is_Leap_Year, {"true", "true", "true", "true", "false", "<NA>", "false", "false"}},
		}
		bad := 0
		for c in cases {
			r, ok := w.date_part(&ts, c.part)
			if !(ok && same(texts(&r), c.want)) {
				bad += 1
				fmt.printf("     %v: got %v, want %v\n", c.part, texts(&r) if ok else nil, c.want)
			}
			if ok {w.destroy_column(&r)}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			fmt.tprintf("date_part: %d parts of a Datetime column match pandas .dt", len(cases)),
		)

		d := w.column_from_dates("d", {{2024, 12, 30}, {2021, 1, 3}})
		defer w.destroy_column(&d)
		t := w.column_from_times("t", {{9, 30, 15, 0}})
		defer w.destroy_column(&t)
		wk, wok := w.date_part(&d, .Week, name = "week")
		hr, hok := w.date_part(&t, .Hour)
		_, bad1 := w.date_part(&d, .Hour)
		_, bad2 := w.date_part(&t, .Month)
		s := w.column_from_strings("s", {"2024-01-01"})
		defer w.destroy_column(&s)
		_, bad3 := w.date_part(&s, .Year)
		good :=
			wok &&
			same(texts(&wk), {"1", "53"}) &&
			wk.name == "week" &&
			hok &&
			same(texts(&hr), {"9"})
		if wok {w.destroy_column(&wk)}
		if hok {w.destroy_column(&hr)}
		expect(
			&passed,
			&total,
			good && !bad1 && !bad2 && !bad3,
			"date_part on Date and Time columns; ok = false for a time part of a Date, a date part of a Time, a String",
		)
	}

	// ---------------- strings
	{
		s := w.column_new("s", .String, 7)
		for v, i in ([]string{"  Apple Inc ", "msft", "", "Ünïcode é", "a,b,,c", "BRK.B", "x"}) {
			if i == 2 {w.append_null(&s)} else {w.append_string(&s, v)}
		}
		defer w.destroy_column(&s)
		Case :: struct {
			label: string,
			want:  []string,
		}
		results: [dynamic]w.Column
		oks: [dynamic]bool
		defer {
			for &r, i in results {if oks[i] {w.destroy_column(&r)}}
			delete(results)
			delete(oks)
		}
		c, ok := w.str_upper(&s); append(&results, c); append(&oks, ok)
		c, ok = w.str_lower(&s); append(&results, c); append(&oks, ok)
		c, ok = w.str_strip(&s); append(&results, c); append(&oks, ok)
		c, ok = w.str_strip(&s, " xa"); append(&results, c); append(&oks, ok)
		c, ok = w.str_len(&s); append(&results, c); append(&oks, ok)
		c, ok = w.str_replace(&s, ",", ";"); append(&results, c); append(&oks, ok)
		c, ok = w.str_slice(&s, 1, 4); append(&results, c); append(&oks, ok)
		c, ok = w.str_slice(&s, -3); append(&results, c); append(&oks, ok)
		c, ok = w.str_split_part(&s, ",", 1); append(&results, c); append(&oks, ok)
		c, ok = w.str_split_part(&s, ",", -1); append(&results, c); append(&oks, ok)
		c, ok = w.str_split_part(&s, "", 0); append(&results, c); append(&oks, ok)
		cases := []Case {
			{
				"upper",
				[]string{"  APPLE INC ", "MSFT", "<NA>", "ÜNÏCODE É", "A,B,,C", "BRK.B", "X"},
			},
			{
				"lower",
				[]string{"  apple inc ", "msft", "<NA>", "ünïcode é", "a,b,,c", "brk.b", "x"},
			},
			{
				"strip",
				[]string{"Apple Inc", "msft", "<NA>", "Ünïcode é", "a,b,,c", "BRK.B", "x"},
			},
			{
				"strip ' xa'",
				[]string{"Apple Inc", "msft", "<NA>", "Ünïcode é", ",b,,c", "BRK.B", ""},
			},
			{"len", []string{"12", "4", "<NA>", "9", "6", "5", "1"}},
			{
				"replace , ;",
				[]string{"  Apple Inc ", "msft", "<NA>", "Ünïcode é", "a;b;;c", "BRK.B", "x"},
			},
			{"slice 1:4", []string{" Ap", "sft", "<NA>", "nïc", ",b,", "RK.", ""}},
			{"slice -3:", []string{"nc ", "sft", "<NA>", "e é", ",,c", "K.B", "x"}},
			{"split , [1]", []string{"<NA>", "<NA>", "<NA>", "<NA>", "b", "<NA>", "<NA>"}},
			{
				"split , [-1]",
				[]string{"  Apple Inc ", "msft", "<NA>", "Ünïcode é", "c", "BRK.B", "x"},
			},
			{
				"split whitespace [0]",
				[]string{"Apple", "msft", "<NA>", "Ünïcode", "a,b,,c", "BRK.B", "x"},
			},
		}
		bad := 0
		for cs, i in cases {
			if !(oks[i] && same(texts(&results[i]), cs.want)) {
				bad += 1
				fmt.printf(
					"     %s: got %v, want %v\n",
					cs.label,
					texts(&results[i]) if oks[i] else nil,
					cs.want,
				)
			}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			fmt.tprintf(
				"str_*: %d transformations match pandas .str (non-ASCII text, null kept)",
				len(cases),
			),
		)

		m1, k1 := w.str_contains(&s, "c")
		m2, k2 := w.str_contains(&s, "C", case_sensitive = false)
		m3, k3 := w.str_startswith(&s, "m")
		m4, k4 := w.str_endswith(&s, "B")
		n := w.column_from_ints("n", {1})
		defer w.destroy_column(&n)
		_, bad1 := w.str_upper(&n)
		_, bad2 := w.str_contains(&n, "1")
		_, bad3 := w.str_replace(&s, "", "x")
		expect(
			&passed,
			&total,
			k1 &&
			k2 &&
			k3 &&
			k4 &&
			!bad1 &&
			!bad2 &&
			!bad3 &&
			bits(m1) == "1 0 0 1 1 0 0" &&
			bits(m2) == "1 0 0 1 1 0 0" &&
			bits(m3) == "0 1 0 0 0 0 0" &&
			bits(m4) == "0 0 0 0 0 1 0",
			fmt.tprintf(
				"str_contains %s, case-insensitive %s, startswith %s, endswith %s; ok = false for an Int column or an empty old text",
				bits(m1),
				bits(m2),
				bits(m3),
				bits(m4),
			),
		)
	}

	// ---------------- compare
	{
		x := w.column_from_floats("x", {1.5, math.nan_f64(), -0.0, 3.0, 2.0})
		defer w.destroy_column(&x)
		wants := [w.Compare_Op]string {
			.Eq = "0 0 0 0 1",
			.Ne = "1 0 1 1 0",
			.Lt = "1 0 1 0 0",
			.Le = "1 0 1 0 1",
			.Gt = "0 0 0 1 0",
			.Ge = "0 0 0 1 1",
		}
		good := true
		got := ""
		for want, op in wants {
			m, ok := w.compare(&x, op, 2.0)
			good = good && ok && bits(m) == want
			got = fmt.tprintf("%s [%v %s]", got, op, bits(m))
		}
		xi := w.column_from_ints("xi", {1, 2, 3})
		defer w.destroy_column(&xi)
		mi, iok := w.compare(&xi, .Ge, 2)
		b := w.column_from_bools("b", {true, false})
		defer w.destroy_column(&b)
		mb, bok := w.compare(&b, .Eq, true)
		s := w.column_from_strings("s", {"AAPL", "MSFT"})
		defer w.destroy_column(&s)
		ms, sok := w.compare(&s, .Lt, "B")
		_, bad1 := w.compare(&s, .Eq, 1)
		_, bad2 := w.compare(&xi, .Eq, "1")
		expect(
			&passed,
			&total,
			good &&
			iok &&
			bits(mi) == "0 1 1" &&
			bok &&
			bits(mb) == "1 0" &&
			sok &&
			bits(ms) == "1 0" &&
			!bad1 &&
			!bad2,
			fmt.tprintf(
				"compare: six operators on Float%s; Int, Bool, String; ok = false on a type mismatch",
				got,
			),
		)
	}

	// ---------------- try_ entry points
	{
		df := w.df_from(
			w.column_from_ints("id", {30, 10, 20}),
			w.column_from_strings("tk", {"C", "A", "B"}),
			w.column_from_floats("px", {3.5, 1.5, 2.5}),
		)
		defer w.destroy_dataframe(&df)

		c, cok := w.try_column(&df, "tk")
		_, cbad := w.try_column(&df, "nope")
		good_col := cok && c == w.column(&df, "tk") && !cbad

		extra := w.column_from_ints("qty", {1, 2, 3})
		short := w.column_from_ints("bad", {1})
		dup := w.column_from_ints("px", {1, 2, 3})
		a1 := w.try_add_column(&df, extra)
		a2 := w.try_add_column(&df, short)
		a3 := w.try_add_column(&df, dup)
		w.destroy_column(&short) // still ours after a refusal
		w.destroy_column(&dup)
		good_add := a1 && !a2 && !a3 && len(df.columns) == 4

		sel, selok := w.try_select(&df, {"tk", "px"})
		_, selbad := w.try_select(&df, {"tk", "nope"})
		good_sel := selok && len(sel.columns) == 2 && !selbad
		w.destroy_dataframe(&sel)

		srt, sok := w.try_sort(&df, {"px"}, {true})
		_, sbad := w.try_sort(&df, {"nope"})
		good_sort := sok && w.column_value_string(w.column(&srt, "tk"), 0) == "C" && !sbad
		w.destroy_dataframe(&srt)

		r1, iok := w.try_iloc(&df, 2)
		_, ibad1 := w.try_iloc(&df, 3)
		_, ibad2 := w.try_iloc(&df, -1)
		r2, mok := w.try_iloc_many(&df, {2, 0})
		_, mbad := w.try_iloc_many(&df, {0, 5})
		good_iloc := iok && r1.rows == 1 && !ibad1 && !ibad2 && mok && r2.rows == 2 && !mbad
		w.destroy_dataframe(&r1)
		w.destroy_dataframe(&r2)

		_, lbad1 := w.try_loc(&df, 20) // no index yet
		ixbad := w.try_set_index(&df, "nope")
		ixok := w.try_set_index(&df, "id")
		h, lok := w.try_loc(&df, 20)
		_, lbad2 := w.try_loc(&df, 25) // not a key
		_, lbad3 := w.try_loc(&df, 20.0) // Float key on an Int index
		_, lbad4 := w.try_loc(&df, "B")
		good_loc :=
			!lbad1 &&
			!ixbad &&
			ixok &&
			lok &&
			w.column_value_string(w.column(&h, "tk"), 0) == "B" &&
			!lbad2 &&
			!lbad3 &&
			!lbad4
		w.destroy_dataframe(&h)

		// a String index: loc, loc_range and try_loc by ticker
		by_tk := w.df_from(
			w.column_from_strings("tk", {"MSFT", "AAPL", "NVDA"}),
			w.column_from_floats("px", {410, 185, 875}),
		)
		defer w.destroy_dataframe(&by_tk)
		w.set_index(&by_tk, "tk")
		hit := w.loc(&by_tk, "MSFT")
		rng := w.loc_range(&by_tk, "AAPL", "MSFT")
		th, tok := w.try_loc(&by_tk, "NVDA")
		_, tbad := w.try_loc(&by_tk, "TSLA")
		good_loc &&=
			w.column_value_string(w.column(&hit, "px"), 0) == "410" &&
			rng.rows == 2 &&
			tok &&
			w.column_value_string(w.column(&th, "px"), 0) == "875" &&
			!tbad
		w.destroy_dataframe(&hit)
		w.destroy_dataframe(&rng)
		w.destroy_dataframe(&th)

		right := w.df_from(
			w.column_from_strings("sym", {"A", "B"}),
			w.column_from_strings("sector", {"Tech", "Bank"}),
		)
		defer w.destroy_dataframe(&right)
		j, jok := w.try_join_on(&df, &right, {"tk"}, {"sym"}, .Left)
		_, jbad1 := w.try_join(&df, &right, {"tk"}) // right has no "tk"
		_, jbad2 := w.try_join_on(&df, &right, {"tk", "px"}, {"sym"})
		_, jbad3 := w.try_join(&df, &right, {})
		good_join := jok && j.rows == 3 && !jbad1 && !jbad2 && !jbad3
		w.destroy_dataframe(&j)

		long := w.df_from(
			w.column_from_strings("tk", {"A", "B", "A"}),
			w.column_from_floats("px", {1, 2, 3}),
		)
		defer w.destroy_dataframe(&long)
		g, gok := w.try_groupby(&long, {"tk"}, {{name = "avg", column = "px", kind = .Mean}})
		_, gbad1 := w.try_groupby(&long, {"nope"}, {{name = "avg", column = "px", kind = .Mean}})
		_, gbad2 := w.try_groupby(&long, {"tk"}, {{name = "avg", column = "nope", kind = .Mean}})
		_, gbad3 := w.try_groupby(&long, {"tk"}, {{name = "c", column = "px", kind = .Corr}}) // no other
		_, gbad4 := w.try_groupby(&long, {"tk"}, {w.make_quantile_agg("q", "px", 1.5)})
		good_group :=
			gok &&
			g.rows == 2 &&
			w.column_value_string(w.column(&g, "tk"), 1) == "B" &&
			w.column_value_string(w.column(&g, "avg"), 0) == "2" &&
			!gbad1 &&
			!gbad2 &&
			!gbad3 &&
			!gbad4
		if gok {w.destroy_dataframe(&g)}

		expect(
			&passed,
			&total,
			good_col &&
			good_add &&
			good_sel &&
			good_sort &&
			good_iloc &&
			good_loc &&
			good_join &&
			good_group,
			fmt.tprintf(
				"try_ variants: column %v, add_column %v, select %v, sort %v, iloc %v, set_index / loc (Int and String index) %v, join %v, groupby %v",
				good_col,
				good_add,
				good_sel,
				good_sort,
				good_iloc,
				good_loc,
				good_join,
				good_group,
			),
		)
	}

	fmt.printf("Parts / checked summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
