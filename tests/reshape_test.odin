package tests

import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:strconv"
import "core:strings"

// Reshaping and selection helpers in core (melt, pivot, pivot_table, isin,
// between, interpolate, sample). melt / pivot / pivot_table / isin / between /
// interpolate are compared with pandas 1.5.3 on the same data (interpolate
// with limit_area='inside'; method 'index' and 'time' for the x column).
// Numbers match to 1e-12 relative. sample has no pandas counterpart with the
// same generator, so it is checked by its contract (unique_rows rows, labels,
// repeatable after rand.reset) and by the frequencies of 20000 draws.

reshape_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Reshaping and Sampling ===")
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
	// want: "name: v v v | name: v v v", the format of the pandas reference
	frame_is :: proc(df: ^w.DataFrame, want: string) -> bool {
		parts := strings.split(want, " | ", context.temp_allocator)
		if len(parts) != len(df.columns) {return false}
		for p, k in parts {
			colon := strings.index(p, ": ")
			if colon < 0 || df.columns[k].name != p[:colon] {return false}
			if !col_is(&df.columns[k], p[colon + 2:]) {return false}
		}
		return true
	}
	check :: proc(passed, total: ^int, df: ^w.DataFrame, ok: bool, want, name: string) {
		good := ok && frame_is(df, want)
		expect(passed, total, good, name)
		if !good {fmt.printf("     got:  %s\n     want: %s\n", show(df) if ok else "not ok", want)}
	}
	bits :: proc(m: []bool) -> string {
		out := ""
		for v, i in m {out = fmt.tprintf("%s%s%d", out, " " if i > 0 else "", 1 if v else 0)}
		return out
	}

	nan := math.nan_f64()

	// ---------------- melt
	{
		msft := w.column_new("MSFT", .Int, 3)
		w.append_int(&msft, 410)
		w.append_null(&msft)
		w.append_int(&msft, 415)
		wide := w.df_from(
			w.column_from_strings("date", {"2024-01-02", "2024-01-03", "2024-01-04"}),
			msft,
			w.column_from_floats("AAPL", {185.5, 186.25, nan}),
		)
		defer w.destroy_dataframe(&wide)
		m, ok := w.melt(&wide, {"date"}, {"AAPL", "MSFT"}, "ticker", "close")
		check(
			&passed,
			&total,
			&m,
			ok && w.column(&m, "close").type == .Float,
			"date: 2024-01-02 2024-01-03 2024-01-04 2024-01-02 2024-01-03 2024-01-04 | ticker: AAPL AAPL AAPL MSFT MSFT MSFT | close: 185.5 186.25 NA 410 NA 415",
			"melt(id date, values AAPL MSFT): Int and Float mix into Float",
		)
		if ok {w.destroy_dataframe(&m)}
		m2, ok2 := w.melt(&wide, {"date"})
		check(
			&passed,
			&total,
			&m2,
			ok2,
			"date: 2024-01-02 2024-01-03 2024-01-04 2024-01-02 2024-01-03 2024-01-04 | variable: MSFT MSFT MSFT AAPL AAPL AAPL | value: 410 NA 415 185.5 186.25 NA",
			"melt with default value columns and names",
		)
		if ok2 {w.destroy_dataframe(&m2)}
		_, bad1 := w.melt(&wide, {"nope"})
		_, bad2 := w.melt(&wide, {"date", "MSFT", "AAPL"}) // no value columns left
		_, bad3 := w.melt(&wide, {"date"}, {"date", "AAPL"}) // String and Float
		_, bad4 := w.melt(&wide, {"date"}, var_name = "date")
		expect(
			&passed,
			&total,
			!bad1 && !bad2 && !bad3 && !bad4,
			"melt: ok = false for an unknown column, no value columns, mixed types, a clashing name",
		)
	}

	// ---------------- pivot / pivot_table
	{
		long := w.df_from(
			w.column_from_strings(
				"date",
				{"2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-04"},
			),
			w.column_from_strings("ticker", {"MSFT", "MSFT", "AAPL", "AAPL", "AAPL"}),
			w.column_from_floats("close", {411, 410, 186.25, nan, 187}),
		)
		defer w.destroy_dataframe(&long)
		p, ok := w.pivot(&long, "date", "ticker", "close")
		check(
			&passed,
			&total,
			&p,
			ok,
			"date: 2024-01-02 2024-01-03 2024-01-04 | AAPL: NA 186.25 187.0 | MSFT: 410.0 411.0 NA",
			"pivot(date, ticker, close): keys sorted, absent pairs null",
		)
		if ok {w.destroy_dataframe(&p)}

		qty := w.column_new("qty", .Int, 7)
		for v, i in ([]int{5, 3, 7, 0, 2, 4, 1}) {if i == 3 {w.append_null(&qty)} else {w.append_int(&qty, v)}}
		tr := w.df_from(
			w.column_from_ints("day", {2, 1, 2, 1, 2, 1, 3}),
			w.column_from_strings("side", {"buy", "sell", "buy", "buy", "sell", "sell", "buy"}),
			qty,
		)
		defer w.destroy_dataframe(&tr)
		Case :: struct {
			agg:  w.Agg_Func,
			want: string,
		}
		cases := []Case {
			{.Mean, "day: 1 2 3 | buy: NA 6.0 1.0 | sell: 3.5 2.0 NA"},
			{.Sum, "day: 1 2 3 | buy: 0 12 1 | sell: 7 2 NA"},
			{.Count, "day: 1 2 3 | buy: 0 2 1 | sell: 2 1 NA"},
			{.Min, "day: 1 2 3 | buy: NA 5 1 | sell: 3 2 NA"},
			{.Max, "day: 1 2 3 | buy: NA 7 1 | sell: 4 2 NA"},
			{.First, "day: 1 2 3 | buy: NA 5 1 | sell: 3 2 NA"},
			{.Last, "day: 1 2 3 | buy: NA 7 1 | sell: 4 2 NA"},
		}
		bad := 0
		for c in cases {
			t, tok := w.pivot_table(&tr, "day", "side", "qty", c.agg)
			if !(tok && frame_is(&t, c.want)) {
				bad += 1
				fmt.printf(
					"     %v: got %s, want %s\n",
					c.agg,
					show(&t) if tok else "not ok",
					c.want,
				)
			}
			if tok {w.destroy_dataframe(&t)}
		}
		expect(
			&passed,
			&total,
			bad == 0,
			"pivot_table with mean / sum / count / min / max / first / last (Int values with a null)",
		)

		_, dup := w.pivot(&tr, "day", "side", "qty") // (2, buy) occurs twice
		_, bad1 := w.pivot(&tr, "day", "nope", "qty")
		_, bad2 := w.pivot_table(&tr, "day", "qty", "side", .Mean) // mean of strings
		clash := w.df_from(
			w.column_from_strings("k", {"a", "b"}),
			w.column_from_strings("c", {"k", "x"}), // key text "k" = index name
			w.column_from_ints("v", {1, 2}),
		)
		defer w.destroy_dataframe(&clash)
		_, bad3 := w.pivot(&clash, "k", "c", "v")
		expect(
			&passed,
			&total,
			!dup && !bad1 && !bad2 && !bad3,
			"pivot: ok = false for a repeated pair, an unknown column, a non-numeric mean, a clashing key name",
		)
	}

	// ---------------- isin / between
	{
		x := w.column_new("x", .Float, 7)
		for v, i in ([]f64{1.5, nan, -0.0, 3.0, 2.0, 0, 7.0}) {if i == 5 {w.append_null(&x)} else {w.append_float(&x, v)}}
		defer w.destroy_column(&x)
		m1, ok1 := w.isin(&x, []f64{0.0, 3.0, nan})
		m2, ok2 := w.isin(&x, []f64{0.0, 3.0})
		xi := w.column_from_ints("xi", {1, 5, 3, 9, 4})
		defer w.destroy_column(&xi)
		m3, ok3 := w.isin(&xi, []int{3, 4, 10})
		s := w.column_from_strings("s", {"AAPL", "SAP", "MSFT"})
		defer w.destroy_column(&s)
		m4, ok4 := w.isin(&s, []string{"SAP", "NVDA"})
		_, bad := w.isin(&s, []int{1})
		expect(
			&passed,
			&total,
			ok1 &&
			ok2 &&
			ok3 &&
			ok4 &&
			!bad &&
			bits(m1) == "0 1 1 1 0 1 0" &&
			bits(m2) == "0 0 1 1 0 0 0" &&
			bits(m3) == "0 0 1 0 1" &&
			bits(m4) == "0 1 0",
			fmt.tprintf(
				"isin: %s (NaN in values matches missing), %s, ints %s, strings %s; ok = false on a type mismatch",
				bits(m1),
				bits(m2),
				bits(m3),
				bits(m4),
			),
		)

		wants := [w.Inclusive]string {
			.Both    = "1 0 0 1 1 0 0",
			.Neither = "0 0 0 0 1 0 0",
			.Left    = "1 0 0 0 1 0 0",
			.Right   = "0 0 0 1 1 0 0",
		}
		got := ""
		good := true
		for want, inc in wants {
			m, ok := w.between(&x, 1.5, 3.0, inc)
			good = good && ok && bits(m) == want
			got = fmt.tprintf("%s [%v %s]", got, inc, bits(m))
		}
		bi, iok := w.between(&xi, 3, 5)
		d := w.column_from_dates("d", {{2024, 1, 5}, {2024, 2, 1}, {2024, 3, 1}})
		defer w.destroy_column(&d)
		bd, dok := w.between(&d, w.Date{2024, 1, 31}, w.Date{2024, 2, 29})
		_, dbad := w.between(&d, 1, 2)
		expect(
			&passed,
			&total,
			good && iok && bits(bi) == "0 1 1 0 1" && dok && bits(bd) == "0 1 0" && !dbad,
			fmt.tprintf(
				"between, four inclusive modes:%s; ints, dates; ok = false on a type mismatch",
				got,
			),
		)
	}

	// ---------------- interpolate
	{
		y := w.column_new("y", .Float, 8)
		for v, i in ([]f64{nan, 1, 0, nan, 4, nan, 10, nan}) {if i == 2 {w.append_null(&y)} else {w.append_float(&y, v)}}
		defer w.destroy_column(&y)
		r1, ok1 := w.interpolate(&y)
		r2, ok2 := w.interpolate(&y, limit = 1)
		xs := w.column_from_ints("x", {0, 1, 2, 10, 11, 12, 20, 21})
		defer w.destroy_column(&xs)
		r3, ok3 := w.interpolate(&y, &xs)
		t := w.column_new("t", .Float, 4)
		w.append_float(&t, 1)
		w.append_null(&t)
		w.append_null(&t)
		w.append_float(&t, 4)
		defer w.destroy_column(&t)
		days := w.column_from_dates(
			"day",
			{{2024, 1, 1}, {2024, 1, 2}, {2024, 1, 5}, {2024, 1, 7}},
		)
		defer w.destroy_column(&days)
		r4, ok4 := w.interpolate(&t, &days)
		defer w.destroy_column(&r1)
		defer w.destroy_column(&r2)
		defer w.destroy_column(&r3)
		defer w.destroy_column(&r4)
		good :=
			ok1 &&
			col_is(&r1, "NA 1.0 2.0 3.0 4.0 7.0 10.0 NA") &&
			ok2 &&
			col_is(&r2, "NA 1.0 2.0 NA 4.0 7.0 10.0 NA") &&
			ok3 &&
			col_is(&r3, "NA 1.0 1.3 3.6999999999999997 4.0 4.666666666666667 10.0 NA") &&
			ok4 &&
			col_is(&r4, "1.0 1.5 3.0 4.0")
		expect(
			&passed,
			&total,
			good,
			"interpolate: by row, limit 1, by an Int x column, by a Date column (pandas 'time')",
		)
		if !good {
			fmt.println("     ", r1, r2)
		}
		short := w.column_from_ints("x", {1, 2})
		defer w.destroy_column(&short)
		_, bad1 := w.interpolate(&y, &short)
		_, bad2 := w.interpolate(&y, limit = -1)
		_, bad3 := w.interpolate(&days)
		expect(
			&passed,
			&total,
			!bad1 && !bad2 && !bad3,
			"interpolate: ok = false for an x of the wrong length, a negative limit, a Date column",
		)
	}

	// ---------------- sample
	{
		df := w.df_from(w.column_from_ints("id", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
		defer w.destroy_dataframe(&df)
		rand.reset(42)
		a, aok := w.sample(&df, 10)
		rand.reset(42)
		b, bok := w.sample(&df, 10)
		defer w.destroy_dataframe(&a)
		defer w.destroy_dataframe(&b)
		seen: [10]bool
		unique_rows, same := aok && bok, true
		for i in 0 ..< 10 {
			v, _ := w.column_at_int(w.column(&a, "id"), i)
			vb, _ := w.column_at_int(w.column(&b, "id"), i)
			unique_rows = unique_rows && !seen[v] && w.pretty_row_label(&a, i) == v
			seen[v] = true
			same = same && v == vb
		}
		_, bad1 := w.sample(&df, 11)
		_, bad2 := w.sample(&df, -1)
		big, bigok := w.sample(&df, 25, replace = true)
		defer if bigok {w.destroy_dataframe(&big)}
		expect(
			&passed,
			&total,
			unique_rows && same && !bad1 && !bad2 && bigok && big.rows == 25,
			"sample: a full draw is a permutation with labels, repeatable after rand.reset; n > rows needs replace",
		)

		// 20000 single draws: each row ~2000 times (sd ~42); 5 sd band
		counts: [10]int
		counts_r: [10]int
		for _ in 0 ..< 20000 {
			s1, _ := w.sample(&df, 1)
			v, _ := w.column_at_int(w.column(&s1, "id"), 0)
			counts[v] += 1
			w.destroy_dataframe(&s1)
			s2, _ := w.sample(&df, 1, replace = true)
			v2, _ := w.column_at_int(w.column(&s2, "id"), 0)
			counts_r[v2] += 1
			w.destroy_dataframe(&s2)
		}
		uniform := true
		for k in 0 ..< 10 {uniform = uniform && abs(counts[k] - 2000) < 212 && abs(counts_r[k] - 2000) < 212}
		expect(
			&passed,
			&total,
			uniform,
			fmt.tprintf("sample frequencies over 20000 draws: %v / %v", counts, counts_r),
		)
	}

	fmt.printf("Reshaping summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
