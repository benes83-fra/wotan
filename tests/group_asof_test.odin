package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"

// Group-wise column calculations (the `by` option of shift, diff, pct_change,
// cumsum, cummax, rank, ffill / bfill, interpolate) and merge_asof, against
// pandas 1.5.3 on the same data: a long table of two tickers, interleaved,
// with NaN prices and a row with a missing ticker (pandas' groupby gives NA
// there), and a trades / quotes pair for merge_asof (by ticker, all three
// directions, tolerance, no exact matches, no by key, a nearest tie, Date keys
// with a tolerance in days). pct_change is compared with x / groupby.shift(1)
// - 1 (no forward fill). merge_asof is also run on shuffled inputs, which
// pandas would reject, and must give the same rows.

group_asof_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Group-wise Calculations and merge_asof ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
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
	show :: proc(c: ^w.Column) -> string {
		out := ""
		for i in 0 ..< c.len {
			out = fmt.tprintf("%s %s", out, "NA" if w.na_at(c, i) else w.column_value_string(c, i))
		}
		return out
	}

	nan := math.nan_f64()
	tk := w.column_new("tk", .String, 10)
	for v in ([]string{"A", "B", "A", "B", "", "A", "B", "A", "B", "A"}) {
		if v == "" {w.append_null(&tk)} else {w.append_string(&tk, v)}
	}
	df := w.df_from(
		tk,
		w.column_from_floats("px", {10, 20, 11, nan, 99, nan, 22, 12.1, 21, nan}),
		w.column_from_strings("side", {"x", "x", "y", "y", "x", "x", "y", "y", "x", "x"}),
	)
	defer w.destroy_dataframe(&df)
	px := w.column(&df, "px")
	by := []^w.Column{w.column(&df, "tk")}

	// ---------------- by: lags, cumulative, rank, fills
	{
		Case :: struct {
			label: string,
			want:  string,
		}
		Result :: struct {
			c:  w.Column,
			ok: bool,
		}
		results: [dynamic]Result
		defer {
			for &r in results {if r.ok {w.destroy_column(&r.c)}}
			delete(results)
		}
		c, ok := w.shift(px, 1, by = by)
		append(&results, Result{c, ok})
		c, ok = w.shift(px, -1, by = by)
		append(&results, Result{c, ok})
		c, ok = w.diff(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.pct_change(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.cumsum(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.cummax(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.rank(px, .Average, false, by = by)
		append(&results, Result{c, ok})
		c, ok = w.rank(px, .Dense, false, by = by)
		append(&results, Result{c, ok})
		c, ok = w.ffill(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.bfill(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.interpolate(px, by = by)
		append(&results, Result{c, ok})
		c, ok = w.shift(px, 1, by = {w.column(&df, "tk"), w.column(&df, "side")})
		append(&results, Result{c, ok})
		cases := []Case {
			{"shift 1", "NA NA 10.0 20.0 NA 11.0 NA NA 22.0 12.1"},
			{"shift -1", "11.0 NA NA 22.0 NA 12.1 21.0 NA NA NA"},
			{"diff", "NA NA 1.0 NA NA NA NA NA -1.0 NA"},
			{"pct_change", "NA NA 0.10000000000000009 NA NA NA NA NA -0.045454545454545414 NA"},
			{"cumsum", "10.0 20.0 21.0 NA NA NA 42.0 33.1 63.0 NA"},
			{"cummax", "10.0 20.0 11.0 NA NA NA 22.0 12.1 22.0 NA"},
			{"rank average desc", "3.0 3.0 2.0 NA NA NA 1.0 1.0 2.0 NA"},
			{"rank dense desc", "3.0 3.0 2.0 NA NA NA 1.0 1.0 2.0 NA"},
			{"ffill", "10.0 20.0 11.0 20.0 NA 11.0 22.0 12.1 21.0 12.1"},
			{"bfill", "10.0 20.0 11.0 22.0 NA 12.1 22.0 12.1 21.0 NA"},
			{"interpolate", "10.0 20.0 11.0 21.0 NA 11.55 22.0 12.1 21.0 NA"},
			{"shift 1 by tk, side", "NA NA NA NA NA 10.0 NA 11.0 20.0 NA"},
		}
		bad := 0
		for cs, i in cases {
			r := &results[i]
			if !(r.ok && col_is(&r.c, cs.want)) {
				bad += 1
				fmt.printf(
					"     %s: got%s, want %s\n",
					cs.label,
					show(&r.c) if r.ok else " not ok",
					cs.want,
				)
			}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			fmt.tprintf(
				"by ticker: %d procedures match pandas groupby (missing key -> NA)",
				len(cases),
			),
		)

		f, fok := w.ffill(&df, {"px"}, by = {"tk"})
		ffill_ok :=
			fok &&
			col_is(w.column(&f, "px"), "10.0 20.0 11.0 20.0 NA 11.0 22.0 12.1 21.0 12.1") &&
			col_is(w.column(&f, "tk"), "A B A B NA A B A B A")
		if fok {w.destroy_dataframe(&f)}
		short := w.column_from_strings("k", {"a"})
		defer w.destroy_column(&short)
		_, bad1 := w.pct_change(px, by = {&short})
		_, bad2 := w.rank(px, by = {&short})
		_, bad3 := w.ffill(&df, by = {"nope"})
		expect(
			&passed,
			&total,
			ffill_ok && !bad1 && !bad2 && !bad3,
			"ffill(df, by) keeps the key column; ok = false for a key of another length or an unknown key",
		)
	}

	// ---------------- merge_asof
	{
		trades := w.df_from(
			w.column_from_ints("time", {1, 2, 3, 5, 5, 8, 10}),
			w.column_from_strings("tk", {"A", "B", "A", "A", "B", "B", "C"}),
			w.column_from_ints("qty", {100, 200, 300, 400, 500, 600, 700}),
		)
		defer w.destroy_dataframe(&trades)
		quotes := w.df_from(
			w.column_from_ints("time", {1, 1, 2, 3, 4, 5, 7, 9}),
			w.column_from_strings("tk", {"A", "B", "A", "B", "A", "B", "A", "A"}),
			w.column_from_floats("bid", {10, 20, 10.5, 20.5, 11, 21, 11.5, 12}),
		)
		defer w.destroy_dataframe(&quotes)
		Case :: struct {
			label: string,
			by:    []string,
			dir:   w.Asof_Direction,
			tol:   Maybe(f64),
			exact: bool,
			bid:   string,
		}
		cases := []Case {
			{"backward", {"tk"}, .Backward, nil, true, "10.0 20.0 10.5 11.0 21.0 21.0 NA"},
			{"forward", {"tk"}, .Forward, nil, true, "10.0 20.5 11.0 11.5 21.0 NA NA"},
			{
				"nearest (tie at 3 -> backward)",
				{"tk"},
				.Nearest,
				nil,
				true,
				"10.0 20.0 10.5 11.0 21.0 21.0 NA",
			},
			{"tolerance 1", {"tk"}, .Backward, 1.0, true, "10.0 20.0 10.5 11.0 21.0 NA NA"},
			{"no exact matches", {"tk"}, .Backward, nil, false, "NA 20.0 10.5 11.0 20.5 21.0 NA"},
		}
		bad := 0
		for cs in cases {
			r, ok := w.merge_asof(&trades, &quotes, "time", cs.by, cs.dir, cs.tol, cs.exact)
			good :=
				ok &&
				len(r.columns) == 4 &&
				col_is(w.column(&r, "qty"), "100 200 300 400 500 600 700") &&
				col_is(w.column(&r, "bid"), cs.bid)
			if !good {
				bad += 1
				fmt.printf(
					"     %s: got%s, want %s\n",
					cs.label,
					show(w.column(&r, "bid")) if ok else " not ok",
					cs.bid,
				)
			}
			if ok {w.destroy_dataframe(&r)}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			"merge_asof by ticker: backward, forward, nearest, tolerance, no exact matches",
		)

		// without by, the right ticker column comes along as tk_right
		r, ok := w.merge_asof(&trades, &quotes, "time")
		noby :=
			ok &&
			col_is(w.column(&r, "bid"), "20.0 10.5 20.5 21.0 21.0 11.5 12.0") &&
			col_is(w.column(&r, "tk_right"), "B A B B B A A")
		if ok {w.destroy_dataframe(&r)}
		expect(
			&passed,
			&total,
			noby,
			"merge_asof without by: equal right keys -> last row; clashing name -> tk_right",
		)

		// shuffled inputs (pandas requires sorted ones): same rows, left order kept
		st := w.iloc_many(&trades, []int{5, 0, 6, 3, 1, 4, 2})
		sq := w.iloc_many(&quotes, []int{7, 3, 0, 5, 2, 6, 4, 1})
		defer w.destroy_dataframe(&st)
		defer w.destroy_dataframe(&sq)
		rs, sok := w.merge_asof(&st, &sq, "time", {"tk"})
		shuffled :=
			sok &&
			col_is(w.column(&rs, "qty"), "600 100 700 400 200 500 300") &&
			col_is(w.column(&rs, "bid"), "21.0 10.0 NA 11.0 20.0 21.0 10.5")
		labels := sok && w.pretty_row_label(&rs, 0) == 5 && w.pretty_row_label(&rs, 6) == 2
		if sok {w.destroy_dataframe(&rs)}
		expect(
			&passed,
			&total,
			shuffled && labels,
			"merge_asof on shuffled inputs: same matches, left order and labels kept",
		)

		// Date keys, tolerance in days
		L := w.df_from(w.column_from_dates("d", {{2024, 1, 5}, {2024, 1, 20}, {2024, 2, 10}}))
		R := w.df_from(
			w.column_from_dates("d", {{2024, 1, 1}, {2024, 1, 15}, {2024, 2, 1}}),
			w.column_from_floats("rate", {5, 5.25, 5.5}),
		)
		defer w.destroy_dataframe(&L)
		defer w.destroy_dataframe(&R)
		rd, dok := w.merge_asof(&L, &R, "d", tolerance = 7.0)
		dates := dok && col_is(w.column(&rd, "rate"), "5.0 5.25 NA")
		if dok {w.destroy_dataframe(&rd)}

		// nearest tie in isolation
		l1 := w.df_from(w.column_from_ints("t", {3}))
		r1 := w.df_from(w.column_from_ints("t", {2, 4}), w.column_from_ints("v", {1, 2}))
		defer w.destroy_dataframe(&l1)
		defer w.destroy_dataframe(&r1)
		rt, tok := w.merge_asof(&l1, &r1, "t", direction = .Nearest)
		tie := tok && col_is(w.column(&rt, "v"), "1")
		if tok {w.destroy_dataframe(&rt)}

		_, bad1 := w.merge_asof(&trades, &quotes, "nope")
		_, bad2 := w.merge_asof(&trades, &L, "time", right_on = "d") // Int vs Date
		_, bad3 := w.merge_asof(&trades, &quotes, "time", tolerance = -1.0)
		_, bad4 := w.merge_asof(&trades, &quotes, "tk") // String key
		_, bad5 := w.merge_asof(&trades, &quotes, "time", {"qty"}) // not in right
		expect(
			&passed,
			&total,
			dates && tie && !bad1 && !bad2 && !bad3 && !bad4 && !bad5,
			"merge_asof: Date keys with a 7-day tolerance, nearest tie; ok = false for bad keys and a negative tolerance",
		)
	}

	fmt.printf("Group-wise / merge_asof summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
