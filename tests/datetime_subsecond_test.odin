package tests

import w "../wotan/core"
import ex "../wotan/exporter"
import imp "../wotan/importer"
import "core:fmt"
import "core:mem"

// Datetime and Time to the nanosecond:
// - parse_datetime / parse_time with 1 to 9 fraction digits against pandas
//   1.5.3 Timestamp (nanoseconds since 1970, .microsecond, .nanosecond),
//   including the ends of the i64 nanosecond range; malformed fractions
//   rejected;
// - the text form (no fraction, or 3, 6 or 9 digits) and its round trip;
// - comparison, sorting, unique, join keys and merge_asof see nanoseconds;
// - arithmetic keeps them (days, months, seconds) or moves them
//   (add_nanoseconds_datetime across a second, a day and a year, backwards);
//   Unix nanoseconds both ways, ok = false outside the i64 range;
// - date_part .Microsecond / .Nanosecond; casts to Float keep the fraction;
// - CSV and JSON round trips keep nanoseconds; XLSX keeps milliseconds.

datetime_subsecond_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Datetime and Time to the Nanosecond ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// ---------------- parsing against pandas
	{
		Case :: struct {
			text:        string,
			unix_ns:     i64, // pandas Timestamp(text).value
			microsecond: int,
			nanosecond:  int,
		}
		cases := []Case {
			{"2024-03-01 09:30:15.5", 1709285415500000000, 500000, 0},
			{"2024-03-01T09:30:15.123", 1709285415123000000, 123000, 0},
			{"2024-03-01 09:30:15.123456", 1709285415123456000, 123456, 0},
			{"2024-03-01 09:30:15.123456789", 1709285415123456789, 123456, 789},
			{"1969-12-31 23:59:59.999999999", -1, 999999, 999},
			{"1970-01-01 00:00:00.000000001", 1, 0, 1},
			{"2262-04-11 23:47:16.854775807", 9223372036854775807, 854775, 807},
		}
		col := w.column_new("t", .Datetime, len(cases))
		defer w.destroy_column(&col)
		good := true
		for c in cases {
			dt, ok := w.parse_datetime(c.text)
			ns, nok := w.datetime_to_unix_nanos(dt)
			if !ok || !nok || ns != c.unix_ns {
				good = false
				fmt.printf("    %s: %v %v %d\n", c.text, ok, nok, ns)
			}
			w.append_datetime(&col, dt)
		}
		us, _ := w.date_part(&col, .Microsecond)
		nsp, _ := w.date_part(&col, .Nanosecond)
		defer w.destroy_column(&us)
		defer w.destroy_column(&nsp)
		for c, i in cases {
			u, _ := w.column_at_int(&us, i)
			n, _ := w.column_at_int(&nsp, i)
			if u != c.microsecond || n != c.nanosecond {
				good = false
				fmt.printf("    %s: microsecond %d nanosecond %d\n", c.text, u, n)
			}
		}
		expect(
			&passed,
			&total,
			good,
			"parse_datetime with 1 to 9 fraction digits: Unix nanoseconds, microsecond and nanosecond parts as pandas",
		)

		bad_ok := false
		for s in ([]string{"2024-03-01 09:30:15.", "2024-03-01 09:30:15.1234567890", "2024-03-01 09:30:15.12a", "2024-03-01 09:30:15.-1"}) {
			if _, ok := w.parse_datetime(s); ok {bad_ok = true}
		}
		for s in ([]string{"09:30:15.", "09:30:15.x", "24:00:00.5"}) {
			if _, ok := w.parse_time(s); ok {bad_ok = true}
		}
		t, tok := w.parse_time("23:59:59.999999999")
		expect(
			&passed,
			&total,
			!bad_ok && tok && t == w.Time{23, 59, 59, 999_999_999},
			"malformed fractions are rejected; parse_time reads nine digits",
		)
	}

	// ---------------- text form
	{
		Case :: struct {
			dt:   w.Datetime,
			text: string,
		}
		cases := []Case {
			{{2024, 3, 1, 9, 30, 15, 0}, "2024-03-01 09:30:15"},
			{{2024, 3, 1, 9, 30, 15, 500_000_000}, "2024-03-01 09:30:15.500"},
			{{2024, 3, 1, 9, 30, 15, 123_000}, "2024-03-01 09:30:15.000123"},
			{{2024, 3, 1, 9, 30, 15, 1}, "2024-03-01 09:30:15.000000001"},
			{{2024, 3, 1, 9, 30, 15, 120_000_000}, "2024-03-01 09:30:15.120"},
		}
		good := true
		for c in cases {
			s := w.datetime_to_string(c.dt, context.temp_allocator)
			back, ok := w.parse_datetime(s)
			if s != c.text || !ok || back != c.dt {
				good = false
				fmt.printf("    %v -> %q -> %v\n", c.dt, s, back)
			}
		}
		ts := w.time_to_string(w.Time{0, 0, 0, 1_500}, context.temp_allocator)
		expect(
			&passed,
			&total,
			good && ts == "00:00:00.000001500",
			"text: no fraction, or 3, 6 or 9 digits (the fewest that are exact); it parses back",
		)
	}

	// ---------------- order, identity, keys
	{
		base := w.Datetime{2024, 3, 1, 9, 30, 15, 0}
		vals := []i32{300, 0, 999_999_999, 299, 300}
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		c := w.column_new("t", .Datetime, len(vals))
		id := w.column_new("id", .Int, len(vals))
		for v, i in vals {
			d := base
			d.nanosecond = v
			w.append_datetime(&c, d)
			w.append_int(&id, i)
		}
		w.add_column(&df, c)
		w.add_column(&df, id)
		sorted := w.dataframe_sort_by(&df, {"t"}, {false})
		defer w.destroy_dataframe(&sorted)
		order := ""
		for r in 0 ..< sorted.rows {
			v, _ := w.column_at_int(w.column(&sorted, "id"), r)
			order = fmt.tprintf("%s%d", order, v)
		}
		n := w.nunique(w.column(&df, "t"))
		probe := base
		probe.nanosecond = 300
		m, _ := w.compare(w.column(&df, "t"), .Eq, probe)
		defer delete(m)
		hits := 0
		for b in m {hits += int(b)}
		expect(
			&passed,
			&total,
			order == "13042" && n == 4 && hits == 2,
			fmt.tprintf(
				"sort (order %s), nunique (%d) and compare (%d equal) tell nanoseconds apart",
				order,
				n,
				hits,
			),
		)

		right := w.dataframe_new()
		defer w.destroy_dataframe(&right)
		rc := w.column_new("t", .Datetime, 2)
		rv := w.column_new("label", .String, 2)
		w.append_datetime(&rc, probe)
		w.append_string(&rv, "at 300 ns")
		p2 := base
		p2.nanosecond = 301
		w.append_datetime(&rc, p2)
		w.append_string(&rv, "at 301 ns")
		w.add_column(&right, rc)
		w.add_column(&right, rv)
		j := w.join(&df, &right, {"t"})
		defer w.destroy_dataframe(&j)
		expect(
			&passed,
			&total,
			j.rows == 2,
			fmt.tprintf("join on a Datetime key matches to the nanosecond (%d rows)", j.rows),
		)

		// merge_asof: the nearest earlier time, 0.25 s apart
		l := w.dataframe_new()
		defer w.destroy_dataframe(&l)
		lt := w.column_new("t", .Datetime, 2)
		w.append_datetime(&lt, w.Datetime{2024, 3, 1, 9, 30, 15, 600_000_000})
		w.append_datetime(&lt, w.Datetime{2024, 3, 1, 9, 30, 16, 100_000_000})
		w.add_column(&l, lt)
		r := w.dataframe_new()
		defer w.destroy_dataframe(&r)
		rt := w.column_new("t", .Datetime, 3)
		rq := w.column_new("quote", .Float, 3)
		for k in 0 ..< 3 {
			w.append_datetime(
				&rt,
				w.Datetime{2024, 3, 1, 9, 30, 15, i32(k) * 250_000_000 + 250_000_000},
			)
			w.append_float(&rq, f64(k + 1))
		}
		w.add_column(&r, rt)
		w.add_column(&r, rq)
		a, aok := w.merge_asof(&l, &r, "t")
		defer if aok {w.destroy_dataframe(&a)}
		q0, _ := w.column_at_float(w.column(&a, "quote"), 0)
		q1, _ := w.column_at_float(w.column(&a, "quote"), 1)
		expect(
			&passed,
			&total,
			aok && q0 == 2 && q1 == 3,
			fmt.tprintf("merge_asof on sub-second times (quotes %v, %v)", q0, q1),
		)
	}

	// ---------------- arithmetic and Unix nanoseconds
	{
		d := w.Datetime{2024, 12, 31, 23, 59, 59, 999_999_999}
		good :=
			w.add_nanoseconds_datetime(d, 1) == w.Datetime{2025, 1, 1, 0, 0, 0, 0} &&
			w.add_nanoseconds_datetime(w.Datetime{2025, 1, 1, 0, 0, 0, 0}, -1) == d &&
			w.add_nanoseconds_datetime(d, 2_500_000_001) ==
				w.Datetime{2025, 1, 1, 0, 0, 2, 500_000_000} &&
			w.add_seconds_datetime(d, 1).nanosecond == 999_999_999 &&
			w.add_day_datetime(d, 1) == w.Datetime{2025, 1, 1, 23, 59, 59, 999_999_999} &&
			w.add_month_datetime(d, 2).nanosecond == 999_999_999
		_, hi := w.datetime_to_unix_nanos(w.Datetime{2262, 4, 11, 23, 47, 16, 854_775_808})
		_, lo := w.datetime_to_unix_nanos(w.Datetime{1677, 9, 21, 0, 12, 43, 145_224_192})
		lo_ns, lo_ok := w.datetime_to_unix_nanos(w.Datetime{1677, 9, 21, 0, 12, 43, 145_224_193})
		rt_ok := true
		for ns in ([]i64{-1, 0, 1, 1_709_285_415_123_456_789, -86_400_000_000_001, lo_ns}) {
			got, ok := w.datetime_to_unix_nanos(w.datetime_from_unix_nanos(ns))
			if !ok || got != ns {rt_ok = false}
		}
		expect(
			&passed,
			&total,
			good && !hi && !lo && lo_ok && rt_ok,
			"arithmetic keeps nanoseconds; Unix nanoseconds round trip; ok = false past 2262-04-11 23:47:16.854775807 and at pandas' NaT",
		)
	}

	// ---------------- casts keep the nanoseconds
	{
		c := w.column_new("t", .Datetime, 1)
		defer w.destroy_column(&c)
		w.append_datetime(&c, w.Datetime{1970, 1, 1, 0, 0, 1, 250_000_001})
		s, sok := w.cast_column(&c, .String)
		defer if sok {w.destroy_column(&s)}
		back, bok := w.cast_column(&s, .Datetime)
		defer if bok {w.destroy_column(&back)}
		text, _ := w.column_at_string(&s, 0)
		bv, _ := w.column_at_datetime(&back, 0)
		dtt, dok := w.cast_column(&c, .Time)
		defer if dok {w.destroy_column(&dtt)}
		tt, _ := w.column_at_time(&dtt, 0)
		good :=
			sok &&
			bok &&
			dok &&
			text == "1970-01-01 00:00:01.250000001" &&
			bv == w.Datetime{1970, 1, 1, 0, 0, 1, 250_000_001} &&
			tt == w.Time{0, 0, 1, 250_000_001}
		expect(
			&passed,
			&total,
			good,
			fmt.tprintf(
				"cast_column: Datetime to String (%q) and back, Datetime to Time keep the nanoseconds",
				text,
			),
		)
	}

	// ---------------- files
	{
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		dt := w.column_new("when", .Datetime, 3)
		tm := w.column_new("at", .Time, 3)
		w.append_datetime(&dt, w.Datetime{2024, 3, 1, 9, 30, 15, 123_456_789})
		w.append_datetime(&dt, w.Datetime{1969, 12, 31, 23, 59, 59, 500_000_000})
		w.append_datetime(&dt, w.Datetime{2024, 3, 1, 9, 30, 15, 0})
		w.append_time(&tm, w.Time{9, 30, 15, 1})
		w.append_time(&tm, w.Time{23, 59, 59, 999_000_000})
		w.append_null(&tm)
		w.add_column(&df, dt)
		w.add_column(&df, tm)
		same :: proc(a, b: ^w.DataFrame, ms_only := false) -> bool {
			if a.rows != b.rows || len(a.columns) != len(b.columns) {return false}
			for ci in 0 ..< len(a.columns) {
				x, y := &a.columns[ci], &b.columns[ci]
				if x.type != y.type {return false}
				for r in 0 ..< a.rows {
					if w.is_null(x, r) != w.is_null(y, r) {return false}
					if w.is_null(x, r) {continue}
					if x.type == .Datetime {
						p, _ := w.column_at_datetime(x, r)
						q, _ := w.column_at_datetime(y, r)
						if ms_only {p.nanosecond = (p.nanosecond + 500_000) / 1_000_000 * 1_000_000}
						if p != q {return false}
					} else {
						p, _ := w.column_at_time(x, r)
						q, _ := w.column_at_time(y, r)
						if ms_only {p.nanosecond = (p.nanosecond + 500_000) / 1_000_000 * 1_000_000}
						if p != q {return false}
					}
				}
			}
			return true
		}
		text := ex.csv_string(&df)
		defer delete(text)
		csv := imp.csv_load_from_string(text)
		defer w.destroy_dataframe(&csv)
		expect(
			&passed,
			&total,
			same(&df, &csv),
			"CSV round trip keeps nanoseconds (types inferred from the text)",
		)

		ex.json_write(&df, "demo_files/export_subsecond.json", context.allocator)
		js := imp.json_load("demo_files/export_subsecond.json", allocator = context.allocator)
		defer w.destroy_dataframe(&js)
		expect(&passed, &total, same(&df, &js), "JSON round trip keeps nanoseconds")

		ok := ex.xlsx_save(&df, "demo_files/export_subsecond.xlsx", context.allocator)
		xl := imp.xlsx_load("demo_files/export_subsecond.xlsx", context.allocator)
		defer w.destroy_dataframe(&xl)
		expect(
			&passed,
			&total,
			ok && same(&df, &xl, ms_only = true),
			"XLSX round trip keeps milliseconds (Excel's resolution), rounded",
		)
	}

	fmt.printf("Datetime to the nanosecond: %d/%d passed\n", passed, total)
}
