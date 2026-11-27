package tests

import w "../wotan/core"
import ex "../wotan/exporter"
import imp "../wotan/importer"
import "core:fmt"
import "core:mem"

// Durations (pandas' Timedelta), against pandas 1.5.3:
// - text: parse_duration reads pandas' forms ("0 days 00:05:00",
//   "-1 days +23:55:00", "-00:05:00", "2 days", fractions) and
//   duration_to_string writes the form pandas reads back; out of range and
//   malformed text fail;
// - values: datetime_sub / datetime_add, duration_components, the units;
// - columns, on demo_files/durations.csv (written by pandas, read with a
//   Duration column inferred): diff of Datetimes per ticker, duration_between,
//   add_duration, add_durations, datetime_floor, diff of Durations,
//   duration_total (total_seconds), cast to and from Int;
// - aggregation: groupby2 Sum / Mean / Min / Max / Count / Median / Quantile,
//   resample_every 15-minute OHLC bars with a Duration sum, and 25-minute
//   bins per ticker with a Duration mean, as pandas computes them;
// - files: Parquet from pyarrow (durations in s, ms, us, ns) read as
//   Duration; a frame with durations through Parquet, CSV, JSON, HTML and
//   Excel (Excel to the microsecond) and back;
// - ok = false for unsuitable columns and arguments.

duration_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Durations ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	D :: w.Duration

	// ---------------- text
	{
		Case :: struct {
			text: string,
			ns:   i64,
		}
		// as pandas.Timedelta(text).value gives them
		cases := []Case {
			{"0 days 00:05:00", 300_000_000_000},
			{"-1 days +23:55:00", -300_000_000_000},
			{"-1 days 23:55:00", -300_000_000_000},
			{"-00:05:00", -300_000_000_000},
			{"00:05:00", 300_000_000_000},
			{"2 days", 172_800_000_000_000},
			{"1 day 00:00:01", 86_401_000_000_000},
			{"-2 days 01:00:00", -169_200_000_000_000},
			{"0 days 00:00:00.5", 500_000_000},
			{"1 days 02:03:04.123456789", 93_784_123_456_789},
			{"106751 days 23:47:16.854775807", 9_223_372_036_854_775_807},
		}
		good := 0
		for c in cases {
			d, ok := w.parse_duration(c.text)
			if ok &&
			   i64(d) ==
				   c.ns {good += 1} else {fmt.printfln("    parse %q = %v %v", c.text, i64(d), ok)}
		}
		Out :: struct {
			d:    D,
			text: string,
		}
		outs := []Out {
			{5 * w.MINUTES, "0 days 00:05:00"},
			{-5 * w.MINUTES, "-1 days +23:55:00"},
			{-3 * w.DAYS + w.MINUTES, "-3 days +00:01:00"},
			{1500 * w.NANOSECONDS, "0 days 00:00:00.000001500"},
			{500 * w.MILLISECONDS, "0 days 00:00:00.500"},
			{0, "0 days 00:00:00"},
		}
		for o in outs {
			t := w.duration_to_string(o.d, context.temp_allocator)
			back, _ := w.parse_duration(t)
			if t == o.text &&
			   back == o.d {good += 1} else {fmt.printfln("    format %v = %q", i64(o.d), t)}
		}
		bad := []string {
			"",
			"5",
			"1:2:3",
			"00:60:00",
			"00:00:60",
			"x days",
			"1 days 00:05",
			"00:00:00.",
			"00:00:00.1234567890",
			"106752 days",
		}
		rejected := 0
		for s in bad {
			if _, ok := w.parse_duration(s); !ok {rejected += 1}
		}
		expect(
			&passed,
			&total,
			good == len(cases) + len(outs) && rejected == len(bad),
			fmt.tprintf(
				"text: %d pandas forms parsed, %d written as pandas writes them (it reads them back), %d of %d malformed or out-of-range texts rejected",
				len(cases),
				len(outs),
				rejected,
				len(bad),
			),
		)
	}

	// ---------------- values
	{
		a := w.Datetime{2024, 3, 1, 9, 30, 0, 5}
		b := w.Datetime{2024, 2, 28, 23, 59, 59, 999_999_999}
		d, ok := w.datetime_sub(a, b)
		back := w.datetime_add(b, d)
		_, far := w.datetime_sub(
			w.Datetime{2400, 1, 1, 0, 0, 0, 0},
			w.Datetime{1900, 1, 1, 0, 0, 0, 0},
		)
		days, rest := w.duration_components(-5 * w.MINUTES)
		units :=
			w.MINUTES == 60 * w.SECONDS &&
			w.WEEKS == 7 * 24 * w.HOURS &&
			w.MILLISECONDS == 1000 * w.MICROSECONDS &&
			w.duration_to_seconds(90 * w.SECONDS) == 90
		expect(
			&passed,
			&total,
			ok &&
			i64(d) == 120_600_000_000_006 && // pandas: 1 days 09:30:00.000000006
			back == a && !far && days == -1 && rest == w.Time{23, 55, 0, 0} && units,
			"values: datetime_sub over a leap day to the nanosecond and datetime_add back; 500 years apart is out of range; -5 minutes is -1 days +23:55:00; the units",
		)
	}

	// ---------------- columns against pandas
	df, dok := imp.try_csv_load("demo_files/durations.csv")
	defer if dok {w.destroy_dataframe(&df)}
	if !dok || w.column(&df, "dur").type != .Duration || w.column(&df, "t").type != .Datetime {
		expect(
			&passed,
			&total,
			false,
			"demo_files/durations.csv: a Duration and a Datetime column",
		)
		fmt.printf("Durations: %d/%d passed\n", passed, total)
		return
	}
	t := w.column(&df, "t")
	dur := w.column(&df, "dur")
	{
		Ref :: struct {
			name: string,
			fnv:  u64,
		}
		// pandas: groupby("tk").t.diff(), end - t, t + Timedelta(15 min 1 ns),
		// t + dur, t.dt.floor("15min"), dur.diff()
		refs := []Ref {
			{"diff_by", 0x87f506a2e93ea10c},
			{"between", 0x6960447fdad0ab83},
			{"add", 0xeaa27c02ac656046},
			{"adds", 0x532a0713875191b8},
			{"floor", 0xe99c255481b5856e},
			{"diff_dur", 0xb4aa7c3fa4ad1e0e},
		}
		cols: [6]w.Column
		oks: [6]bool
		cols[0], oks[0] = w.diff(t, by = {w.column(&df, "tk")})
		cols[1], oks[1] = w.duration_between(w.column(&df, "end"), t)
		cols[2], oks[2] = w.add_duration(t, 15 * w.MINUTES + 1)
		cols[3], oks[3] = w.add_durations(t, dur)
		cols[4], oks[4] = w.datetime_floor(t, 15 * w.MINUTES)
		cols[5], oks[5] = w.diff(dur)
		good := 0
		for &c, k in cols {
			got := _dur_digest(&c) if oks[k] else 0
			if got ==
			   refs[k].fnv {good += 1} else {fmt.printfln("    %s: %16x", refs[k].name, got)}
			if oks[k] {w.destroy_column(&c)}
		}
		secs, sok := w.duration_total(dur, w.SECONDS)
		s0, _ := w.column_at_float(&secs, 0)
		s1, _ := w.column_at_float(&secs, 1)
		if sok {w.destroy_column(&secs)}
		as_int, iok := w.cast_column(dur, .Int)
		again, aok := w.cast_column(&as_int, .Duration)
		round := iok && aok && _dur_digest(&again) == _dur_digest(dur)
		if iok {w.destroy_column(&as_int)}
		if aok {w.destroy_column(&again)}
		expect(
			&passed,
			&total,
			good == len(refs) && sok && s0 == 66562.371174488 && s1 == 37088.15205139 && round,
			fmt.tprintf(
				"%d of %d columns as pandas computes them: diff of Datetimes per ticker, duration_between, add_duration, add_durations, datetime_floor, diff of Durations; duration_total(SECONDS) = total_seconds(); Duration -> Int -> Duration",
				good,
				len(refs),
			),
		)
	}

	// ---------------- aggregation against pandas
	{
		gb := w.groupby2(&df, {"tk"})
		defer w.destroy_groupby2(&gb)
		g := w.groupby2_agg(
			&gb,
			{
				{name = "sum", column = "dur", kind = .Sum},
				{name = "mean", column = "dur", kind = .Mean},
				{name = "min", column = "dur", kind = .Min},
				{name = "max", column = "dur", kind = .Max},
				{name = "count", column = "dur", kind = .Count},
				w.make_median_agg("median", "dur"),
				w.make_quantile_agg("q30", "dur", 0.3),
			},
		)
		defer w.destroy_dataframe(&g)
		// pandas, per ticker: sum mean min max count median quantile(0.3)
		Want :: struct {
			key:  string,
			vals: [7]i64,
		}
		want := []Want {
			{
				"AAPL",
				{
					742265469643674,
					33739339529257,
					-7778719278056,
					68015481187348,
					22,
					37031037537715,
					26703170432080,
				},
			},
			{
				"MSFT",
				{
					992304244168213,
					33076808138940,
					-3566726934167,
					71500479084245,
					30,
					32487813148577,
					15038790139923,
				},
			},
		}
		group_ok := g.rows == 2
		for r in 0 ..< g.rows {
			k, _ := w.column_at_string(w.column(&g, "tk"), r)
			exp: [7]i64
			for x in want {if x.key == k {exp = x.vals}}
			for name, j in ([7]string{"sum", "mean", "min", "max", "count", "median", "q30"}) {
				c := w.column(&g, name)
				v: i64
				if c.type ==
				   .Int {x, _ := w.column_at_int(c, r); v = i64(x)} else {x, _ := w.column_at_duration(c, r); v = i64(x)}
				if v != exp[j] {
					group_ok = false
					fmt.printfln("    %s %s = %d, pandas %d", k, name, v, exp[j])
				}
			}
		}
		Ref :: struct {
			name: string,
			fnv:  u64,
		}
		bars, bok := w.resample_every(
			&df,
			"t",
			15 * w.MINUTES,
			{
				{"px", .First, "open"},
				{"px", .Max, "high"},
				{"px", .Min, "low"},
				{"px", .Last, "close"},
				{"px", .Count, "n"},
				{"dur", .Sum, "dsum"},
			},
		)
		defer if bok {w.destroy_dataframe(&bars)}
		// pandas: df.resample("15min", on="t").px.first() / max / min / last / count, .dur.sum()
		refs15 := []Ref {
			{"t", 0x9cbb3b817dd8bef5},
			{"open", 0xb3addaa244c7f699},
			{"high", 0x0743601871f8020a},
			{"low", 0x9851d42a77edb625},
			{"close", 0x8364221e44e66280},
			{"n", 0xfa9a9cdede73766d},
			{"dsum", 0x74223ce971fc18ca},
		}
		bars_ok := bok && bars.rows == 189
		for r in refs15 {
			if !bok {break}
			if _dur_digest(w.column(&bars, r.name)) != r.fnv {
				bars_ok = false
				fmt.printfln("    every15.%s differs", r.name)
			}
		}
		by, byok := w.resample_every(
			&df,
			"t",
			25 * w.MINUTES,
			{{"dur", .Mean, "dmean"}, {"px", .Count, "n"}},
			by = {"tk"},
		)
		defer if byok {w.destroy_dataframe(&by)}
		// pandas: df.groupby("tk").resample("25min", on="t").agg(...)
		refs25 := []Ref {
			{"tk", 0x873b93955d009c37},
			{"t", 0x590e01010f9a40a2},
			{"dmean", 0xd1443994c0b70da2},
			{"n", 0x3861dde343636063},
		}
		by_ok := byok && by.rows == 219
		for r in refs25 {
			if !byok {break}
			if _dur_digest(w.column(&by, r.name)) != r.fnv {
				by_ok = false
				fmt.printfln("    every25.%s differs", r.name)
			}
		}
		expect(
			&passed,
			&total,
			group_ok && bars_ok && by_ok,
			"groupby2 Sum / Mean / Min / Max / Count / Median / Quantile of durations per ticker; resample_every: 189 15-minute OHLC bars with a Duration sum, 219 25-minute bins per ticker from each ticker's first day, as pandas computes them",
		)
	}

	// ---------------- files
	{
		pq, pok := imp.parquet_load("demo_files/parquet/durations.parquet")
		defer if pok {w.destroy_dataframe(&pq)}
		// pyarrow: duration[s], [ms], [us], [ns] with nulls, in two row groups
		want := [4][4]i64 {
			{1_000_000_000, -90061_000_000_000, -1, 0},
			{1_500_000_000, -1, -1_000_000, 86400_000_000_000},
			{-1, 1000, 2000, 3000},
			{1, 2, 3, -1},
		}
		read_ok := pok && pq.rows == 4 && w.column(&pq, "n").type == .Int
		for name, k in ([4]string{"sec", "ms", "us", "ns"}) {
			if !read_ok {break}
			c := w.column(&pq, name)
			if c.type != .Duration {read_ok = false; break}
			for r in 0 ..< 4 {
				v, missing := w.column_at_duration(c, r)
				if (want[k][r] == -1 && k != 1) || (k == 1 && r == 1) {
					if !missing {read_ok = false}
				} else if missing || i64(v) != want[k][r] {read_ok = false}
			}
		}
		// a frame with durations through every format
		src := w.dataframe_new()
		defer w.destroy_dataframe(&src)
		d := w.column_new("d", .Duration, 4)
		w.append_duration(&d, 5 * w.MINUTES)
		w.append_duration(&d, -90 * w.SECONDS + 250 * w.MICROSECONDS)
		w.append_null(&d)
		w.append_duration(&d, 3 * w.DAYS + 7)
		w.add_column(&src, d)
		x := w.column_new("x", .Int, 4)
		for i in 0 ..< 4 {w.append_int(&x, i)}
		w.add_column(&src, x)
		src.rows = 4
		same :: proc(a, b: ^w.DataFrame, micro: bool) -> bool {
			c, found := w.try_column(b, "d")
			if !found || c.type != .Duration || b.rows != a.rows {return false}
			for r in 0 ..< a.rows {
				u, um := w.column_at_duration(w.column(a, "d"), r)
				v, vm := w.column_at_duration(c, r)
				if um != vm {return false}
				if micro {u = u / 1000 * 1000} 	// Excel keeps microseconds
				if u != v {return false}
			}
			return true
		}
		bytes, wok := ex.parquet_bytes(&src)
		defer delete(bytes)
		pb, pbok := imp.parquet_load_from_bytes(bytes)
		defer if pbok {w.destroy_dataframe(&pb)}
		ex.csv_save(&src, "demo_files/export_durations.csv")
		cb, cbok := imp.try_csv_load("demo_files/export_durations.csv")
		defer if cbok {w.destroy_dataframe(&cb)}
		ex.json_write(&src, "demo_files/export_durations.json", context.allocator)
		jb, jbok := imp.try_json_load("demo_files/export_durations.json")
		defer if jbok {w.destroy_dataframe(&jb)}
		ex.html_write(&src, "demo_files/export_durations.html", context.allocator)
		hb, hbok := imp.try_html_load("demo_files/export_durations.html")
		defer if hbok {w.destroy_dataframe(&hb)}
		ex.xlsx_save(&src, "demo_files/export_durations.xlsx", context.allocator)
		xb, xbok := imp.try_xlsx_load("demo_files/export_durations.xlsx")
		defer if xbok {w.destroy_dataframe(&xb)}
		trips :=
			wok &&
			pbok &&
			same(&src, &pb, false) &&
			cbok &&
			same(&src, &cb, false) &&
			jbok &&
			same(&src, &jb, false) &&
			hbok &&
			same(&src, &hb, false) &&
			xbok &&
			same(&src, &xb, true)
		expect(
			&passed,
			&total,
			read_ok && trips,
			"files: pyarrow's duration[s / ms / us / ns] read as Duration; a frame with durations through Parquet, CSV, JSON, HTML and Excel (to the microsecond) and back",
		)
	}

	// ---------------- ok = false
	{
		date := w.column_new("day", .Date, 1)
		defer w.destroy_column(&date)
		w.append_date(&date, w.Date{2024, 1, 1})
		failed, n := 0, 0
		fails :: proc(failed, n: ^int, c: w.Column, ok: bool) {
			n^ += 1
			if ok {
				cc := c
				w.destroy_column(&cc)
			} else {failed^ += 1}
		}
		c: w.Column
		ok: bool
		c, ok = w.duration_between(t, &date); fails(&failed, &n, c, ok)
		c, ok = w.duration_between(
			w.column(&df, "px"),
			w.column(&df, "px"),
		); fails(&failed, &n, c, ok)
		c, ok = w.add_duration(w.column(&df, "px"), w.MINUTES); fails(&failed, &n, c, ok)
		c, ok = w.add_durations(t, t); fails(&failed, &n, c, ok)
		c, ok = w.duration_total(t, w.SECONDS); fails(&failed, &n, c, ok)
		c, ok = w.duration_total(dur, 0); fails(&failed, &n, c, ok)
		c, ok = w.datetime_floor(t, -w.MINUTES); fails(&failed, &n, c, ok)
		c, ok = w.datetime_floor(dur, w.MINUTES); fails(&failed, &n, c, ok)
		c, ok = w.cast_column(dur, .Float); fails(&failed, &n, c, ok)
		n += 1
		if r, rok := w.resample_every(&df, "t", 0, {{"px", .Sum, ""}});
		   rok {w.destroy_dataframe(&r)} else {failed += 1}
		n += 1
		if r, rok := w.resample_every(&df, "dur", w.HOURS, {{"px", .Sum, ""}});
		   rok {w.destroy_dataframe(&r)} else {failed += 1}
		expect(
			&passed,
			&total,
			failed == n,
			fmt.tprintf(
				"%d of %d give ok = false: duration_between of different types or of numbers, add_duration(s) on numbers or with a non-Duration column, duration_total of a non-Duration or with a zero unit, datetime_floor with a negative step or of durations, cast Duration -> Float, resample_every with a zero step or on a Duration column",
				failed,
				n,
			),
		)
	}

	fmt.printf("Durations: %d/%d passed\n", passed, total)
}

// FNV-1a over a column's cells as text, each followed by a line break:
// durations and datetimes as nanosecond counts, floats as their shortest
// text, missing cells (and NaN) as NA
@(private = "file")
_dur_digest :: proc(c: ^w.Column) -> u64 {
	h: u64 = 0xcbf29ce484222325
	for r in 0 ..< c.len {
		text := "NA"
		if !w.is_null(c, r) {
			#partial switch c.type {
			case .Duration:
				v, _ := w.column_at_duration(c, r)
				text = fmt.tprint(i64(v))
			case .Datetime:
				v, _ := w.column_at_datetime(c, r)
				ns, _ := w.datetime_to_unix_nanos(v)
				text = fmt.tprint(ns)
			case .Float:
				v, _ := w.column_at_float(c, r)
				if v == v {text = w.float_to_string(v, context.temp_allocator)}
			case .Int:
				v, _ := w.column_at_int(c, r)
				text = fmt.tprint(v)
			case .String:
				v, _ := w.column_at_string(c, r)
				text = v
			}
		}
		for b in transmute([]u8)text {
			h ~= u64(b)
			h *= 0x100000001b3
		}
		h ~= '\n'
		h *= 0x100000001b3
	}
	return h
}
