package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"

// Column calculations in core (shift, diff, pct_change, cumsum / cumprod /
// cummax / cummin, clip, rank, nlargest / nsmallest, describe) against pandas
// 1.5.3 on the same data: a Float column with NaN, a null, zeros (for x / 0
// and 0 / 0 in pct_change) and ties; an Int column with a null; a String
// column with a null. References for Int cumsum / cumprod come from pandas'
// float path, because pandas 1.5.3's nullable Int64 cumsum / cumprod turn
// every row after an NA into NA even with skipna=True. pct_change is compared
// with fill_method=None (no forward fill). Values match to 1e-12 relative.

column_ops_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Column Calculations ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// Does `c` hold the numbers of `want` ("NA" = missing, "inf" / "-inf")?
	matches :: proc(c: ^w.Column, want: string) -> bool {
		toks := strings.fields(want, context.temp_allocator)
		if len(toks) != c.len {return false}
		for t, i in toks {
			if t == "NA" {
				if !w.na_at(c, i) {return false}
				continue
			}
			if w.na_at(c, i) {return false}
			x: f64
			switch t {
			case "inf":
				x = math.inf_f64(1)
			case "-inf":
				x = math.inf_f64(-1)
			case:
				x, _ = strconv.parse_f64(t)
			}
			v, _ := w.column_at_numeric(c, i)
			if math.is_inf(x) {
				if v != x {return false}
			} else if abs(v - x) > 1e-12 * max(1, abs(x)) {
				return false
			}
		}
		return true
	}
	show :: proc(c: ^w.Column) -> string {
		out := ""
		for i in 0 ..< c.len {
			s := "NA" if w.na_at(c, i) else fmt.tprintf("%v", _numeric(c, i))
			out = fmt.tprintf("%s%s%s", out, " " if i > 0 else "", s)
		}
		return out
	}
	check :: proc(passed, total: ^int, c: ^w.Column, ok: bool, want, name: string) {
		good := ok && matches(c, want)
		expect(passed, total, good, name)
		if !good {fmt.printf("     got:  %s\n     want: %s\n", show(c) if ok else "not ok", want)}
	}

	nan := math.nan_f64()
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	px := w.column_new("px", .Float, 11)
	for v, i in ([]f64{10, 11, nan, 12.1, 0, 9, 0, 3, 3, 0, 0}) {
		if i == 4 {w.append_null(&px)} else {w.append_float(&px, v)}
	}
	w.add_column(&df, px)
	q := w.column_new("q", .Int, 11)
	for v, i in ([]int{3, 1, 4, 1, 5, 0, 9, 2, 6, 5, 3}) {
		if i == 5 {w.append_null(&q)} else {w.append_int(&q, v)}
	}
	w.add_column(&df, q)
	s := w.column_new("s", .String, 11)
	for v in ([]string{"b", "a", "c", "a", "", "d", "b", "e", "a", "c", "b"}) {
		if v == "" {w.append_null(&s)} else {w.append_string(&s, v)}
	}
	w.add_column(&df, s)
	pxc, qc, sc := w.column(&df, "px"), w.column(&df, "q"), w.column(&df, "s")

	// ---------------- shift / diff / pct_change
	{
		a, aok := w.shift(pxc, 2)
		b, bok := w.shift(qc, -1, name = "q_next")
		check(&passed, &total, &a, aok, "NA NA 10 11 NA 12.1 NA 9 0 3 3", "shift(px, 2)")
		check(
			&passed,
			&total,
			&b,
			bok && b.name == "q_next" && b.type == .Int,
			"1 4 1 5 NA 9 2 6 5 3 NA",
			"shift(q, -1) named q_next",
		)
		ss, _ := w.shift(sc, 1)
		expect(
			&passed,
			&total,
			ss.type == .String && w.na_at(&ss, 0) && w.column_value_string(&ss, 1) == "b",
			"shift works on a String column",
		)
		w.destroy_column(&a)
		w.destroy_column(&b)
		w.destroy_column(&ss)

		d1, ok1 := w.diff(pxc)
		d2, ok2 := w.diff(qc, 2)
		d3, ok3 := w.diff(qc, -1)
		check(&passed, &total, &d1, ok1, "NA 1 NA NA NA NA -9 3 0 -3 0", "diff(px)")
		check(
			&passed,
			&total,
			&d2,
			ok2 && d2.type == .Int,
			"NA NA 1 0 1 NA 4 NA -3 3 -3",
			"diff(q, 2) stays Int",
		)
		check(&passed, &total, &d3, ok3, "2 -3 3 -4 NA NA 7 -4 1 2 NA", "diff(q, -1)")
		w.destroy_column(&d1)
		w.destroy_column(&d2)
		w.destroy_column(&d3)

		p1, pok1 := w.pct_change(pxc)
		p2, pok2 := w.pct_change(qc, 2)
		check(
			&passed,
			&total,
			&p1,
			pok1,
			"NA 0.1 NA NA NA NA -1 inf 0 -1 NA",
			"pct_change(px): x / 0 = inf, 0 / 0 missing",
		)
		check(
			&passed,
			&total,
			&p2,
			pok2 && p2.type == .Float,
			"NA NA 0.333333333333333 0 0.25 NA 0.8 NA -0.333333333333333 1.5 -0.5",
			"pct_change(q, 2) is Float",
		)
		w.destroy_column(&p1)
		w.destroy_column(&p2)
		_, bad1 := w.diff(sc)
		_, bad2 := w.pct_change(sc)
		expect(
			&passed,
			&total,
			!bad1 && !bad2,
			"diff / pct_change: ok = false for a String column",
		)
	}

	// ---------------- cumulative
	{
		Case :: struct {
			f:      proc(
				col: ^w.Column,
				skipna := true,
				name := "",
				by: []^w.Column = nil,
				allocator := context.allocator,
			) -> (
				w.Column,
				bool,
			),
			label:  string,
			col:    ^w.Column,
			skipna: bool,
			want:   string,
		}
		cases := []Case {
			{w.cumsum, "cumsum px", pxc, true, "10 21 NA 33.1 NA 42.1 42.1 45.1 48.1 48.1 48.1"},
			{w.cumsum, "cumsum q", qc, true, "3 4 8 9 14 NA 23 25 31 36 39"},
			{w.cumsum, "cumsum px skipna=false", pxc, false, "10 21 NA NA NA NA NA NA NA NA NA"},
			{w.cumsum, "cumsum q skipna=false", qc, false, "3 4 8 9 14 NA NA NA NA NA NA"},
			{w.cumprod, "cumprod px", pxc, true, "10 110 NA 1331 NA 11979 0 0 0 0 0"},
			{w.cumprod, "cumprod q", qc, true, "3 3 12 12 60 NA 540 1080 6480 32400 97200"},
			{w.cumprod, "cumprod q skipna=false", qc, false, "3 3 12 12 60 NA NA NA NA NA NA"},
			{w.cummax, "cummax px", pxc, true, "10 11 NA 12.1 NA 12.1 12.1 12.1 12.1 12.1 12.1"},
			{w.cummax, "cummax q", qc, true, "3 3 4 4 5 NA 9 9 9 9 9"},
			{w.cummax, "cummax px skipna=false", pxc, false, "10 11 NA NA NA NA NA NA NA NA NA"},
			{w.cummin, "cummin px", pxc, true, "10 10 NA 10 NA 9 0 0 0 0 0"},
			{w.cummin, "cummin q", qc, true, "3 1 1 1 1 NA 1 1 1 1 1"},
		}
		bad := 0
		for c in cases {
			r, ok := c.f(c.col, c.skipna)
			if !(ok && matches(&r, c.want) && r.type == c.col.type) {
				bad += 1
				fmt.printf(
					"     %s: got %s, want %s\n",
					c.label,
					show(&r) if ok else "not ok",
					c.want,
				)
			}
			if ok {w.destroy_column(&r)}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			fmt.tprintf(
				"cumsum / cumprod / cummax / cummin, with and without skipna (%d cases)",
				len(cases),
			),
		)

		sm, _ := w.cummax(sc)
		defer w.destroy_column(&sm)
		text := ""
		for i in 0 ..< sm.len {text = fmt.tprintf("%s %s", text, "NA" if w.na_at(&sm, i) else w.column_value_string(&sm, i))}
		_, cbad := w.cumsum(sc)
		expect(
			&passed,
			&total,
			text == " b b c c NA d d e e e e" && !cbad,
			fmt.tprintf("cummax on strings:%s; cumsum: ok = false for strings", text),
		)
	}

	// ---------------- clip
	{
		c1, ok1 := w.clip(pxc, 1.0, 10.0)
		c2, ok2 := w.clip(qc, lower = 2.0)
		c3, ok3 := w.clip(qc, upper = 4.0)
		check(&passed, &total, &c1, ok1, "10 10 NA 10 NA 9 1 3 3 1 1", "clip(px, 1, 10)")
		check(
			&passed,
			&total,
			&c2,
			ok2 && c2.type == .Int,
			"3 2 4 2 5 NA 9 2 6 5 3",
			"clip(q, lower 2) stays Int",
		)
		check(&passed, &total, &c3, ok3, "3 1 4 1 4 NA 4 2 4 4 3", "clip(q, upper 4)")
		w.destroy_column(&c1)
		w.destroy_column(&c2)
		w.destroy_column(&c3)
		_, bad1 := w.clip(qc, lower = 2.5)
		_, bad2 := w.clip(pxc, 5.0, 1.0)
		_, bad3 := w.clip(sc, lower = 1.0)
		c4, ok4 := w.clip(pxc, lower = nan)
		defer w.destroy_column(&c4)
		expect(
			&passed,
			&total,
			!bad1 && !bad2 && !bad3 && ok4 && matches(&c4, "10 11 NA 12.1 NA 9 0 3 3 0 0"),
			"clip: a NaN bound is ignored; ok = false for a fractional Int bound, lower > upper, a String column",
		)
	}

	// ---------------- rank
	{
		Case :: struct {
			method:    w.Rank_Method,
			col:       ^w.Column,
			ascending: bool,
			want:      string,
		}
		cases := []Case {
			{.Average, pxc, true, "7 8 NA 9 NA 6 2 4.5 4.5 2 2"},
			{.Average, qc, false, "6.5 9.5 5 9.5 3.5 NA 1 8 2 3.5 6.5"},
			{.Average, sc, true, "5 2 7.5 2 NA 9 5 10 2 7.5 5"},
			{.Min, pxc, true, "7 8 NA 9 NA 6 1 4 4 1 1"},
			{.Min, qc, false, "6 9 5 9 3 NA 1 8 2 3 6"},
			{.Min, sc, true, "4 1 7 1 NA 9 4 10 1 7 4"},
			{.Max, pxc, true, "7 8 NA 9 NA 6 3 5 5 3 3"},
			{.Max, qc, false, "7 10 5 10 4 NA 1 8 2 4 7"},
			{.Max, sc, true, "6 3 8 3 NA 9 6 10 3 8 6"},
			{.First, pxc, true, "7 8 NA 9 NA 6 1 4 5 2 3"},
			{.First, qc, false, "6 9 5 10 3 NA 1 8 2 4 7"},
			{.First, sc, true, "4 1 7 2 NA 9 5 10 3 8 6"},
			{.Dense, pxc, true, "4 5 NA 6 NA 3 1 2 2 1 1"},
			{.Dense, qc, false, "5 7 4 7 3 NA 1 6 2 3 5"},
			{.Dense, sc, true, "2 1 3 1 NA 4 2 5 1 3 2"},
		}
		bad := 0
		for c in cases {
			r, ok := w.rank(c.col, c.method, c.ascending)
			if !(ok && r.type == .Float && matches(&r, c.want)) {
				bad += 1
				fmt.printf(
					"     rank %v %s: got %s, want %s\n",
					c.method,
					c.col.name,
					show(&r) if ok else "not ok",
					c.want,
				)
			}
			if ok {w.destroy_column(&r)}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			"rank: 5 methods on Float, Int (descending) and String columns, ties and missing values",
		)
	}

	// ---------------- nlargest / nsmallest
	{
		Case :: struct {
			largest: bool,
			n:       int,
			col:     string,
			px, q:   string,
			s:       string,
			labels:  []int,
		}
		cases := []Case {
			{true, 3, "q", "0 3 NA", "9 6 5", "b a NA", {6, 8, 4}},
			{true, 4, "q", "0 3 NA 0", "9 6 5 5", "b a NA c", {6, 8, 4, 9}},
			{false, 4, "px", "0 0 0 3", "9 5 3 2", "b c b e", {6, 9, 10, 7}},
		}
		bad := 0
		for c in cases {
			r: w.DataFrame
			ok: bool
			if c.largest {r, ok = w.nlargest(&df, c.n, c.col)} else {r, ok = w.nsmallest(&df, c.n, c.col)}
			good := ok && r.rows == len(c.labels)
			if good {
				good = matches(w.column(&r, "px"), c.px) && matches(w.column(&r, "q"), c.q)
				for l, i in c.labels {
					sv := w.column(&r, "s")
					cell := "NA" if w.na_at(sv, i) else w.column_value_string(sv, i)
					good =
						good &&
						w.pretty_row_label(&r, i) == l &&
						cell == strings.fields(c.s, context.temp_allocator)[i]
				}
			}
			if !good {bad += 1}
			if ok {w.destroy_dataframe(&r)}
		}
		all, aok := w.nlargest(&df, 50, "px") // more than there are present values
		_, bad1 := w.nlargest(&df, 2, "nope")
		_, bad2 := w.nsmallest(&df, -1, "q")
		expect(
			&passed,
			&total,
			bad == 0 && aok && all.rows == 9 && !bad1 && !bad2,
			"nlargest / nsmallest: order, ties by appearance, labels, missing skipped; ok = false for an unknown column or negative n",
		)
		if aok {w.destroy_dataframe(&all)}
	}

	// ---------------- describe
	{
		d := w.describe(&df)
		defer w.destroy_dataframe(&d)
		lab := w.column(&d, "stat")
		labels := ""
		for i in 0 ..< lab.len {labels = fmt.tprintf("%s %s", labels, w.column_value_string(lab, i))}
		expect(
			&passed,
			&total,
			len(d.columns) == 3 &&
			labels == " count mean std min 25% 50% 75% max" &&
			matches(
				w.column(&d, "px"),
				"9.0 5.344444444444445 5.115445022456774 0.0 0.0 3.0 10.0 12.1",
			) &&
			matches(w.column(&d, "q"), "10.0 3.9 2.4698178070456938 1.0 2.25 3.5 5.0 9.0"),
			fmt.tprintf(
				"describe: px %s / q %s",
				show(w.column(&d, "px")),
				show(w.column(&d, "q")),
			),
		)

		one := w.dataframe_new()
		defer w.destroy_dataframe(&one)
		w.add_column(&one, w.column_from_floats("stat", {2.5}))
		d1 := w.describe(&one)
		defer w.destroy_dataframe(&d1)
		st := w.column(&d1, "stat")
		std, _ := w.column_at_float(st, 2)
		expect(
			&passed,
			&total,
			d1.columns[0].name == "statistic" &&
			matches(st, "1 2.5 NA 2.5 2.5 2.5 2.5 2.5") &&
			std != std,
			"describe of one value: std is NaN; a column named stat moves the labels to \"statistic\"",
		)
	}

	fmt.printf("Column calculations summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

@(private = "file")
_numeric :: proc(c: ^w.Column, i: int) -> f64 {
	v, _ := w.column_at_numeric(c, i)
	return v
}
