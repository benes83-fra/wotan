package tests

import w "../wotan/core"
import ex "../wotan/exporter"
import imp "../wotan/importer"
import "core:fmt"
import "core:mem"

// Time zones, against pandas 1.5.3 (pytz) and Python's zoneinfo on the same
// IANA data (2026f):
// - the database: zone names (IANA, letter case forgiven, UTC, fixed
//   offsets), UTC offsets in winter and summer, far beyond the last listed
//   switch (2099), before standard time (1883), negative summer time
//   (Dublin) and a skipped day (Samoa 2011);
// - single times: datetime_from_local with each choice for a skipped and a
//   repeated wall-clock time, datetime_to_local back;
// - columns, on demo_files/timezones.csv (written by pandas): tz_localize
//   with the ambiguous / nonexistent choices, tz_convert, to_local_time,
//   date_part in local time, datetime_floor in local time, diff; the text
//   of every cell as pandas prints it ("2024-03-10 03:30:00-04:00");
// - aggregation: daily bars per ticker, 25-minute bins per ticker and weekly
//   bars of a New York column, binned by local time as pandas does;
// - files: CSV text with offsets read as zoned columns (one shared offset or
//   UTC), Parquet from pandas with its zones, a zoned frame through Parquet,
//   CSV, JSON, HTML (instants kept) and Excel (local time);
// - the zone kept by sort, filter, slices, take, groupby keys, join, concat,
//   fillna, shift, drop_duplicates and melt;
// - ok = false for unknown zones and mixed zoned / plain columns.

timezone_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Time zones ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	zone :: proc(name: string) -> w.Time_Zone {
		tz, _ := w.time_zone(name)
		return tz
	}

	// ---------------- the database
	{
		names := w.time_zone_names(context.temp_allocator)
		ny, nyok := w.time_zone("America/New_York")
		folded, fok := w.time_zone("america/NEW_YORK")
		z, zok := w.time_zone("Z")
		f1, f1ok := w.time_zone("+05:30")
		f2, f2ok := w.time_zone("-0800")
		f3, f3ok := w.time_zone("+05")
		rejected := 0
		for bad in ([]string{"", "Mars/Base", "+25:00", "+05:60", "05:00", "+5", "UTC+1"}) {
			if _, ok := w.time_zone(bad); !ok {rejected += 1}
		}
		lookups :=
			nyok &&
			fok &&
			folded == ny &&
			w.time_zone_name(folded) == "America/New_York" &&
			zok &&
			w.time_zone_name(z) == "UTC" &&
			f1ok &&
			w.time_zone_name(f1) == "+05:30" &&
			f2ok &&
			w.time_zone_name(f2) == "-08:00" &&
			f3ok &&
			w.time_zone_name(f3) == "+05:00" &&
			rejected == 7 &&
			w.time_zone_name({}) == ""
		Off :: struct {
			zone: string,
			utc:  w.Datetime,
			secs: i64,
		}
		// zoneinfo: ZoneInfo(zone).utcoffset at the UTC instant
		offs := []Off {
			{"America/New_York", {2024, 1, 15, 12, 0, 0, 0}, -18000},
			{"America/New_York", {2024, 7, 1, 12, 0, 0, 0}, -14400},
			{"Australia/Sydney", {2024, 1, 15, 12, 0, 0, 0}, 39600},
			{"Australia/Sydney", {2024, 7, 1, 12, 0, 0, 0}, 36000},
			{"Asia/Kolkata", {2024, 7, 1, 12, 0, 0, 0}, 19800},
			{"America/New_York", {2099, 7, 1, 12, 0, 0, 0}, -14400},
			{"America/New_York", {1883, 1, 1, 12, 0, 0, 0}, -17762},
			{"Europe/Dublin", {2024, 1, 15, 12, 0, 0, 0}, 0},
			{"Europe/Dublin", {2024, 7, 1, 12, 0, 0, 0}, 3600},
			{"Pacific/Apia", {2011, 12, 30, 9, 59, 59, 0}, -36000},
			{"Pacific/Apia", {2011, 12, 30, 10, 0, 0, 0}, 50400},
			{"+05:30", {2024, 7, 1, 12, 0, 0, 0}, 19800},
		}
		good := 0
		for o in offs {
			got := w.utc_offset(zone(o.zone), o.utc)
			if got ==
			   w.Duration(o.secs) *
				   w.SECONDS {good += 1} else {fmt.printfln("    %s %v: %v", o.zone, o.utc, got / w.SECONDS)}
		}
		expect(
			&passed,
			&total,
			w.tz_data_version() == "2026f" && len(names) == 598 && lookups && good == len(offs),
			fmt.tprintf(
				"database %s: %d zone names; lookups by IANA name (any case), Z, +05:30 / -0800 / +05; %d bad names rejected; %d of %d UTC offsets as zoneinfo gives them (summer and winter, 2099, 1883, Dublin, Samoa's skipped day)",
				w.tz_data_version(),
				len(names),
				rejected,
				good,
				len(offs),
			),
		)
	}

	// ---------------- single times
	{
		ny := zone("America/New_York")
		gap := w.Datetime{2024, 3, 10, 2, 30, 0, 0}
		twice := w.Datetime{2024, 11, 3, 1, 30, 0, 0}
		fwd, fok := w.datetime_from_local(gap, ny, nonexistent = .Shift_Forward)
		back, bok := w.datetime_from_local(gap, ny, nonexistent = .Shift_Backward)
		_, gap_raise := w.datetime_from_local(gap, ny)
		early, eok := w.datetime_from_local(twice, ny, ambiguous = .Earliest)
		late, lok := w.datetime_from_local(twice, ny, ambiguous = .Latest)
		_, twice_raise := w.datetime_from_local(twice, ny)
		plain, pok := w.datetime_from_local({2024, 7, 4, 9, 30, 0, 5}, ny)
		// pandas: Timestamp(t).tz_localize("America/New_York", ...).tz_convert("UTC")
		expect(
			&passed,
			&total,
			fok &&
			fwd == w.Datetime{2024, 3, 10, 7, 0, 0, 0} &&
			bok &&
			back == w.Datetime{2024, 3, 10, 6, 59, 59, 999_999_999} &&
			!gap_raise &&
			eok &&
			early == w.Datetime{2024, 11, 3, 5, 30, 0, 0} &&
			lok &&
			late == w.Datetime{2024, 11, 3, 6, 30, 0, 0} &&
			!twice_raise &&
			pok &&
			plain == w.Datetime{2024, 7, 4, 13, 30, 0, 5} &&
			w.datetime_to_local(plain, ny) == w.Datetime{2024, 7, 4, 9, 30, 0, 5} &&
			w.datetime_to_local(late, ny) == twice &&
			w.datetime_zoned_string(late, ny, context.temp_allocator) ==
				"2024-11-03 01:30:00-05:00",
			"single times: 02:30 on the spring-forward day shifts to 07:00 UTC or 06:59:59.999999999 UTC (Raise refuses); 01:30 on the fall-back day is 05:30 or 06:30 UTC (Raise refuses); back to local time; text with the offset",
		)
	}

	// ---------------- columns against pandas
	df, dok := imp.try_csv_load("demo_files/timezones.csv")
	defer if dok {w.destroy_dataframe(&df)}
	if !dok || w.column(&df, "local").type != .Datetime || w.column(&df, "utc").tz.id == 0 {
		expect(
			&passed,
			&total,
			false,
			"demo_files/timezones.csv: a plain and a zoned Datetime column",
		)
		fmt.printf("Time zones: %d/%d passed\n", passed, total)
		return
	}
	local := w.column(&df, "local")
	utc := w.column(&df, "utc")
	ny, nyok := w.tz_convert(utc, "America/New_York", "t")
	defer if nyok {w.destroy_column(&ny)}
	{
		Ref :: struct {
			name: string,
			fnv:  u64,
		}
		// pandas: local.dt.tz_localize(zone, ambiguous = all True / "NaT" /
		// all False, nonexistent = ...), utc.dt.tz_convert(...),
		// .dt.tz_localize(None), .dt.hour, .dt.dayofweek,
		// .dt.floor("15min", ambiguous = all True, nonexistent =
		// "shift_forward"), .diff()
		refs := []Ref {
			{"loc_ny_early_fwd", 0x72dacdb02dbfcbe6},
			{"loc_ny_missing", 0x56b84f4f01018b1d},
			{"loc_berlin_late_back", 0x1b7ee1390e3f7669},
			{"conv_sydney", 0x4940a14c2b0cf91a},
			{"conv_berlin", 0xaea523c2724c4c02},
			{"naive_ny", 0x1f7aea2c06801f7e},
			{"hour_ny", 0x8d15538e8e6643a4},
			{"weekday_ny", 0x64ea57fdf8eca402},
			{"floor15_ny", 0xf22ccfe63c71c976},
			{"diff_ny", 0xb970b83ac6adb7ef},
		}
		cols: [10]w.Column
		oks: [10]bool
		cols[0], oks[0] = w.tz_localize(local, "America/New_York", .Earliest, .Shift_Forward)
		cols[1], oks[1] = w.tz_localize(local, "America/New_York", .Missing, .Missing)
		cols[2], oks[2] = w.tz_localize(local, "Europe/Berlin", .Latest, .Shift_Backward)
		cols[3], oks[3] = w.tz_convert(utc, "Australia/Sydney")
		cols[4], oks[4] = w.tz_convert(utc, "Europe/Berlin")
		cols[5], oks[5] = w.to_local_time(&ny)
		cols[6], oks[6] = w.date_part(&ny, .Hour)
		cols[7], oks[7] = w.date_part(&ny, .Weekday)
		cols[8], oks[8] = w.datetime_floor(&ny, 15 * w.MINUTES, .Earliest, .Shift_Forward)
		cols[9], oks[9] = w.diff(&ny)
		good := 0
		for &c, k in cols {
			got := _tz_digest(&c) if oks[k] else 0
			if got ==
			   refs[k].fnv {good += 1} else {fmt.printfln("    %s: %16x", refs[k].name, got)}
			if oks[k] {w.destroy_column(&c)}
		}
		// pandas raises NonExistentTimeError / AmbiguousTimeError
		_, raise_loc := w.tz_localize(local, "America/New_York")
		_, raise_floor := w.datetime_floor(&ny, w.HOURS)
		zones_ok :=
			w.time_zone_name(utc.tz) == "+00:00" &&
			w.time_zone_name(w.column(&df, "ny").tz) == "UTC" &&
			w.time_zone_name(w.column(&df, "kol").tz) == "+05:30"
		expect(
			&passed,
			&total,
			good == len(refs) && !raise_loc && !raise_floor && zones_ok,
			fmt.tprintf(
				"%d of %d columns as pandas prints them: tz_localize (Earliest + Shift_Forward, Missing, Latest + Shift_Backward), tz_convert to Sydney and Berlin, to_local_time, local hour and weekday, datetime_floor in local time, diff; Raise refuses as pandas does; CSV offsets give the zones +00:00, UTC (New York's two offsets) and +05:30",
				good,
				len(refs),
			),
		)
	}

	// ---------------- aggregation against pandas
	{
		fr := w.dataframe_new()
		defer w.destroy_dataframe(&fr)
		w.add_column(&fr, w.column_clone(w.column(&df, "k")))
		w.add_column(&fr, w.column_clone(&ny))
		w.add_column(&fr, w.column_clone(w.column(&df, "px")))
		specs := []w.Resample_Spec{{"px", .Sum, "s"}, {"px", .Count, "n"}, {"px", .First, "f"}}
		Ref :: struct {
			name: string,
			fnv:  u64,
		}
		check :: proc(out: ^w.DataFrame, ok: bool, rows: int, refs: []Ref) -> bool {
			if !ok || out.rows != rows {return false}
			good := true
			for r in refs {
				got := _tz_digest(w.column(out, r.name))
				if got != r.fnv {
					fmt.printfln("    %s: %16x", r.name, got)
					good = false
				}
			}
			return good
		}
		// pandas: fr.groupby("k").resample("D", on="t").px.sum() / count() / first()
		day, dok2 := w.resample(&fr, "t", .Day, specs, {"k"})
		day_ok := check(
			&day,
			dok2,
			483,
			{
				{"t", 0x12c9f24f165dc35d},
				{"s", 0x9363ee8f1e7c9cf8},
				{"n", 0xceb86bbb480425bc},
				{"f", 0x46a8aa4b7a221470},
			},
		)
		if dok2 {w.destroy_dataframe(&day)}
		// pandas: the rows before April, groupby("k").resample("25min", on="t")
		before := w.Datetime{2024, 4, 1, 4, 0, 0, 0} // midnight in New York, in UTC
		mask := make([]bool, fr.rows, context.temp_allocator)
		for i in 0 ..< fr.rows {
			v, missing := w.column_at_datetime(w.column(&fr, "t"), i)
			mask[i] = !missing && w.datetime_compare(v, before) < 0
		}
		march := w.loc_mask(&fr, mask)
		defer w.destroy_dataframe(&march)
		every, eok := w.resample_every(&march, "t", 25 * w.MINUTES, specs, {"k"})
		every_ok := check(
			&every,
			eok,
			654,
			{
				{"t", 0xa306c700f5aa64fc},
				{"s", 0x7ed54d094a760433},
				{"n", 0x6cf19da66469dd7a},
				{"f", 0x748dff6be653b429},
			},
		)
		if eok {w.destroy_dataframe(&every)}
		// pandas: fr.resample("W", on="t")
		week, wok := w.resample(&fr, "t", .Week, specs)
		week_ok := check(
			&week,
			wok,
			35,
			{
				{"t", 0xe9f37b9aac6f7acd},
				{"s", 0x6ea8da40a2f9c6b9},
				{"n", 0xebb60f10281cdddd},
				{"f", 0x64764a243114d5e8},
			},
		)
		if wok {w.destroy_dataframe(&week)}
		expect(
			&passed,
			&total,
			day_ok && every_ok && week_ok,
			"resample of a New York column as pandas bins it in local time: daily bars per ticker (483 rows over two clock changes), 25-minute bins per ticker from local midnight (654 rows), weekly bars (35 rows); labels in the zone",
		)
	}

	// ---------------- files
	{
		// the same instants in three zones: pandas' UTC nanoseconds
		INSTANTS :: 0x8bf2887caf1ffd6b
		csv_ok :=
			_tz_ns_digest(utc) == INSTANTS &&
			_tz_ns_digest(w.column(&df, "ny")) == INSTANTS &&
			_tz_ns_digest(w.column(&df, "kol")) == INSTANTS
		pq, pok := imp.parquet_load("demo_files/parquet/timezones.parquet")
		defer if pok {w.destroy_dataframe(&pq)}
		pq_ok := pok
		if pok {
			Z :: struct {
				col, zone: string,
			}
			for z in ([]Z{{"ny", "America/New_York"}, {"utc", "UTC"}, {"kol", "Asia/Kolkata"}, {"fixed", "+05:30"}, {"naive", ""}}) {
				c := w.column(&pq, z.col)
				if w.time_zone_name(c.tz) != z.zone || _tz_ns_digest(c) != INSTANTS {
					fmt.printfln(
						"    parquet %s: %q %16x",
						z.col,
						w.time_zone_name(c.tz),
						_tz_ns_digest(c),
					)
					pq_ok = false
				}
			}
		}
		// a zoned frame out and back: Parquet keeps the zones, text formats
		// the instants, Excel the local time
		round_ok := pok
		if pok {
			same :: proc(a, b: ^w.Column, zone: bool) -> bool {
				if b == nil || b.type != .Datetime || b.len != a.len {return false}
				if zone && a.tz != b.tz {return false}
				return _tz_ns_digest(a) == _tz_ns_digest(b)
			}
			bytes, bok := ex.parquet_bytes(&pq)
			back, backok := imp.parquet_load_from_bytes(bytes)
			delete(bytes)
			if !bok || !backok {
				round_ok = false
			} else {
				for name in ([]string{"ny", "utc", "kol", "fixed", "naive"}) {
					if !same(w.column(&pq, name), w.column(&back, name), true) {round_ok = false}
				}
				w.destroy_dataframe(&back)
			}
			_ = ex.csv_save(&pq, "demo_files/export_timezones.csv")
			ex.json_write(&pq, "demo_files/export_timezones.json", context.allocator)
			ex.html_write(&pq, "demo_files/export_timezones.html", context.allocator)
			_ = ex.xlsx_save(&pq, "demo_files/export_timezones.xlsx", context.allocator)
			c, cok := imp.try_csv_load("demo_files/export_timezones.csv")
			j, jok := imp.try_json_load("demo_files/export_timezones.json")
			h, hok := imp.try_html_load("demo_files/export_timezones.html")
			x, xok := imp.try_xlsx_load("demo_files/export_timezones.xlsx")
			round_ok &&= cok && jok && hok && xok
			if round_ok {
				for name in ([]string{"ny", "kol", "fixed"}) {
					src := w.column(&pq, name)
					if !same(src, w.column(&c, name), false) ||
					   !same(src, w.column(&j, name), false) ||
					   !same(src, w.column(&h, name), false) {
						fmt.printfln("    text formats: %s", name)
						round_ok = false
					}
				}
				// Excel holds local time to the millisecond
				src := w.column(&pq, "ny")
				xc := w.column(&x, "ny")
				for i in 0 ..< src.len {
					a, am := w.column_at_datetime(src, i)
					b, bm := w.column_at_datetime(xc, i)
					if am != bm {round_ok = false; break}
					if am {continue}
					l := w.datetime_to_local(a, src.tz)
					d, _ := w.datetime_sub(b, l)
					if xc.tz.id != 0 || d < -w.MILLISECONDS || d > w.MILLISECONDS {
						round_ok = false
						break
					}
				}
			}
			if cok {w.destroy_dataframe(&c)}
			if jok {w.destroy_dataframe(&j)}
			if hok {w.destroy_dataframe(&h)}
			if xok {w.destroy_dataframe(&x)}
		}
		expect(
			&passed,
			&total,
			csv_ok && pq_ok && round_ok,
			"files: CSV text with offsets read as pandas reads the instants; Parquet from pandas with the zones America/New_York, UTC, Asia/Kolkata, +05:30 and none; out and back through Parquet (zones kept), CSV, JSON and HTML (instants kept) and Excel (local time)",
		)
	}

	// ---------------- the zone survives frame operations
	{
		fr := w.dataframe_new()
		defer w.destroy_dataframe(&fr)
		w.add_column(&fr, w.column_clone(w.column(&df, "k")))
		w.add_column(&fr, w.column_clone(&ny))
		w.add_column(&fr, w.column_clone(w.column(&df, "px")))
		tz := ny.tz
		kept := 0
		check :: proc(kept: ^int, c: ^w.Column, tz: w.Time_Zone, what: string) {
			if c != nil && c.tz == tz {kept^ += 1} else {fmt.printfln("    zone lost: %s", what)}
		}
		sorted, sok := w.try_sort(&fr, {"t"})
		if sok {check(&kept, w.column(&sorted, "t"), tz, "sort"); w.destroy_dataframe(&sorted)}
		mask := make([]bool, fr.rows, context.temp_allocator)
		for i in 0 ..< fr.rows {mask[i] = i % 3 == 0}
		filtered := w.loc_mask(&fr, mask)
		check(&kept, w.column(&filtered, "t"), tz, "loc_mask")
		w.destroy_dataframe(&filtered)
		view := w.head(&fr, 5)
		check(&kept, w.column(&view, "t"), tz, "head (a view)")
		w.destroy_dataframe(&view)
		taken := w.df_take(&fr, {3, 1, 2})
		check(&kept, w.column(&taken, "t"), tz, "df_take")
		w.destroy_dataframe(&taken)
		g, gok := w.try_groupby(&fr, {"t"}, {{name = "n", column = "px", kind = .Count}})
		if gok {check(&kept, w.column(&g, "t"), tz, "groupby key"); w.destroy_dataframe(&g)}
		g2, g2ok := w.try_groupby(&fr, {"k"}, {{name = "last", column = "t", kind = .Max}})
		if g2ok {check(&kept, w.column(&g2, "last"), tz, "groupby Max"); w.destroy_dataframe(&g2)}
		j, jok := w.try_join(&fr, &fr, {"k"}, .Inner)
		if jok {
			check(&kept, w.column(&j, "t"), tz, "join left")
			check(&kept, w.column(&j, "t_right"), tz, "join right")
			w.destroy_dataframe(&j)
		}
		frames := []^w.DataFrame{&fr, &fr}
		cat, cok := w.concat_rows(frames)
		if cok {check(&kept, w.column(&cat, "t"), tz, "concat_rows"); w.destroy_dataframe(&cat)}
		filled, fok := w.fillna_column(&ny, w.Datetime{2024, 1, 1, 0, 0, 0, 0})
		if fok {check(&kept, &filled, tz, "fillna"); w.destroy_column(&filled)}
		shifted, shok := w.shift(&ny)
		if shok {check(&kept, &shifted, tz, "shift"); w.destroy_column(&shifted)}
		dd, ddok := w.drop_duplicates(&fr, {"k"})
		if ddok {check(&kept, w.column(&dd, "t"), tz, "drop_duplicates"); w.destroy_dataframe(&dd)}
		m, mok := w.melt(&fr, {"t"}, {"px"})
		if mok {check(&kept, w.column(&m, "t"), tz, "melt"); w.destroy_dataframe(&m)}
		moved, mvok := w.add_duration(&ny, w.HOURS)
		if mvok {check(&kept, &moved, tz, "add_duration"); w.destroy_column(&moved)}

		// concat of New York with Berlin gives UTC; zoned with plain fails
		ber, _ := w.tz_convert(&ny, "Europe/Berlin", "t")
		fb := w.dataframe_new()
		w.add_column(&fb, w.column_clone(w.column(&df, "k")))
		w.add_column(&fb, ber)
		w.add_column(&fb, w.column_clone(w.column(&df, "px")))
		mixed, mixok := w.concat_rows([]^w.DataFrame{&fr, &fb})
		mixed_utc := mixok && w.time_zone_name(w.column(&mixed, "t").tz) == "UTC"
		if mixok {w.destroy_dataframe(&mixed)}
		w.destroy_dataframe(&fb)
		expect(
			&passed,
			&total,
			kept == 14 && mixed_utc,
			fmt.tprintf(
				"the zone survives %d of 14 operations: sort, loc_mask, head, df_take, groupby key and Max, join (both sides), concat_rows, fillna, shift, drop_duplicates, melt, add_duration; concat of New York with Berlin gives UTC",
				kept,
			),
		)
	}

	// ---------------- ok = false
	{
		n := 0
		failed := 0
		fails :: proc(failed, n: ^int, c: w.Column, ok: bool) {
			n^ += 1
			if !ok {failed^ += 1} else {c := c; w.destroy_column(&c)}
		}
		c, ok := w.tz_localize(&ny, "Europe/Berlin"); fails(&failed, &n, c, ok) // already zoned
		c, ok = w.tz_convert(local, "Europe/Berlin"); fails(&failed, &n, c, ok) // no zone
		c, ok = w.tz_localize(local, "Mars/Base"); fails(&failed, &n, c, ok)
		c, ok = w.tz_convert(&ny, "nowhere"); fails(&failed, &n, c, ok)
		c, ok = w.tz_localize(w.column(&df, "px"), "UTC"); fails(&failed, &n, c, ok)
		c, ok = w.to_local_time(w.column(&df, "px")); fails(&failed, &n, c, ok)
		plain, _ := w.to_local_time(&ny)
		c, ok = w.duration_between(&ny, &plain); fails(&failed, &n, c, ok) // zoned - plain
		w.destroy_column(&plain)
		// merge_asof of a zoned column with a plain one
		a := w.dataframe_new()
		b := w.dataframe_new()
		w.add_column(&a, w.column_clone(&ny))
		lt := w.column_clone(local)
		w.column_own_name(&lt, "t")
		w.add_column(&b, lt)
		_, asof_ok := w.merge_asof(&a, &b, "t")
		w.destroy_dataframe(&a)
		w.destroy_dataframe(&b)
		n += 1
		if !asof_ok {failed += 1}
		expect(
			&passed,
			&total,
			failed == n,
			fmt.tprintf(
				"%d of %d misuses give ok = false: tz_localize of a zoned column, tz_convert of a plain one, unknown zones, a column that is not Datetime, duration_between and merge_asof of zoned with plain",
				failed,
				n,
			),
		)
	}

	fmt.printf("Time zones: %d/%d passed\n", passed, total)
}

// FNV-1a of every cell's text as pandas prints it ("NA" for a missing
// cell): Datetimes with their offset, Durations as Timedelta text, floats
// with six decimals (pandas sums with compensation, which can change the
// last bit)
_tz_digest :: proc(c: ^w.Column) -> u64 {
	h: u64 = 0xcbf29ce484222325
	for r in 0 ..< c.len {
		text := "NA" if w.is_null(c, r) else w.value_as_string(c, r, context.temp_allocator)
		if c.type == .Float && !w.is_null(c, r) {
			v, _ := w.column_at_float(c, r)
			text = fmt.tprintf("%.6f", v)
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

// FNV-1a of a Datetime column's UTC nanoseconds ("NA" for a missing cell)
_tz_ns_digest :: proc(c: ^w.Column) -> u64 {
	h: u64 = 0xcbf29ce484222325
	for r in 0 ..< c.len {
		text := "NA"
		if v, missing := w.column_at_datetime(c, r); !missing {
			ns, _ := w.datetime_to_unix_nanos(v)
			text = fmt.tprint(ns)
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
