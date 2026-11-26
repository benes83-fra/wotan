package core

// Calendar periods for resample
Resample_Freq :: enum {
	Day,
	Week, // ends on `week_end` (Sunday by default, as pandas' "W")
	Month,
	Quarter,
	Year,
}

// One output column of resample: `agg` of `column` per period, named `name`
// (or like `column` when empty)
Resample_Spec :: struct {
	column: string,
	agg:    Agg_Func,
	name:   string,
}

// One row per calendar period of the Date or Datetime column `on`, as pandas'
// df.resample(freq, on=...).agg(...) gives it: daily data to weekly or
// monthly bars, for example. The first column holds each period's label, its
// last day (for Datetime: midnight of that day), named like `on`; then one
// column per spec. Every period from the first to the last one in the data is
// present; a period without rows has a Sum and Count of 0 and nulls
// otherwise. Missing values are left out of every aggregate, and rows with a
// missing `on` are left out entirely. The rows need not be sorted: First and
// Last are the earliest and latest values of a period by `on` (rows with the
// same time in row order), so that First / Max / Min / Last of a price give
// open / high / low / close bars.
//
// With `by` (key column names such as a ticker), every group gets its own
// run of periods, from its own first to its own last period; the key columns
// come first, groups in order of first appearance.
//
// ok = false for an unknown column, an `on` column that is not Date or
// Datetime, an aggregation that does not suit its column (Mean or Sum of
// text), or duplicate output names.
resample :: proc(
	df: ^DataFrame,
	on: string,
	freq: Resample_Freq,
	specs: []Resample_Spec,
	by: []string = nil,
	week_end := Days.Sunday,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	oi, found := df.name_to_index[on]
	if !found {return {}, false}
	oncol := &df.columns[oi]
	if oncol.type != .Date && oncol.type != .Datetime {return {}, false}

	vals := make([]^Column, len(specs), context.temp_allocator)
	defer delete(vals, context.temp_allocator)
	names := make(map[string]bool, allocator = context.temp_allocator)
	defer delete(names)
	names[oncol.name] = true
	keys := make([]^Column, len(by), context.temp_allocator)
	defer delete(keys, context.temp_allocator)
	for name, k in by {
		idx, kfound := df.name_to_index[name]
		if !kfound || name in names {return {}, false}
		keys[k] = &df.columns[idx]
		names[name] = true
	}
	for sp, k in specs {
		idx, vfound := df.name_to_index[sp.column]
		if !vfound || !_agg_supported(df.columns[idx].type, sp.agg) {return {}, false}
		vals[k] = &df.columns[idx]
		out_name := sp.name if sp.name != "" else sp.column
		if out_name in names {return {}, false}
		names[out_name] = true
	}

	g, gok := _groups(df.rows, keys)
	if !gok {return {}, false}
	defer _groups_free(&g)

	// period ordinal of every row; per group the lowest and highest one
	ord := make([]int, df.rows, context.temp_allocator)
	defer delete(ord, context.temp_allocator)
	lo := make([]int, len(g.lists), context.temp_allocator)
	hi := make([]int, len(g.lists), context.temp_allocator)
	defer delete(lo, context.temp_allocator)
	defer delete(hi, context.temp_allocator)
	for k in 0 ..< len(g.lists) {lo[k], hi[k] = max(int), min(int)}
	for r in 0 ..< df.rows {
		if na_at(oncol, r) || g.of[r] < 0 {
			ord[r] = min(int)
			continue
		}
		ord[r] = _period_of(_day_of(oncol, r), freq, week_end)
		k := g.of[r]
		lo[k] = min(lo[k], ord[r])
		hi[k] = max(hi[k], ord[r])
	}

	// output rows: per group (in order of first appearance) its periods
	base := make([]int, len(g.lists) + 1, context.temp_allocator)
	defer delete(base, context.temp_allocator)
	for k in 0 ..< len(g.lists) {
		base[k + 1] = base[k] + (hi[k] - lo[k] + 1 if hi[k] >= lo[k] else 0)
	}
	n := base[len(g.lists)]
	cell_of := make([]int, df.rows, context.temp_allocator)
	defer delete(cell_of, context.temp_allocator)
	for r in 0 ..< df.rows {
		cell_of[r] = -1 if ord[r] == min(int) else base[g.of[r]] + ord[r] - lo[g.of[r]]
	}
	// rows in time order (stable), so that First / Last are the earliest and
	// latest values of a period
	by_time := column_sort_order(oncol, false, context.temp_allocator)
	defer delete(by_time, context.temp_allocator)
	cells := _cells_build(cell_of, n, by_time)
	defer _cells_free(&cells)

	out := dataframe_new(allocator)
	// key columns: one source row of the group per output row
	group_rows := make([]int, n, context.temp_allocator)
	defer delete(group_rows, context.temp_allocator)
	for k in 0 ..< len(g.lists) {
		for j in base[k] ..< base[k + 1] {group_rows[j] = g.lists[k][0]}
	}
	for c in keys {add_column(&out, column_take(c, group_rows, allocator))}
	labels := column_new("", oncol.type, max(n, 1), allocator)
	column_own_name(&labels, oncol.name)
	for k in 0 ..< len(g.lists) {
		for p in lo[k] ..= hi[k] { 	// no periods if the group has no dates
			d := civil_from_days(_period_last_day(p, freq, week_end))
			if oncol.type ==
			   .Date {append_date(&labels, d)} else {append_datetime(&labels, Datetime{d.year, d.month, d.day, 0, 0, 0, 0})}
		}
	}
	add_column(&out, labels)
	for sp, k in specs {
		col := _agg_cells(vals[k], sp.agg, &cells, 0, n, true, allocator)
		column_own_name(&col, sp.name if sp.name != "" else sp.column)
		add_column(&out, col)
	}
	out.rows = n
	return out, true
}

// Day number (days since 1970-01-01) of a present Date or Datetime cell
@(private = "file")
_day_of :: proc(col: ^Column, r: int) -> i64 {
	if col.type == .Date {
		d, _ := column_at_date(col, r)
		return days_from_civil(i64(d.year), i64(d.month), i64(d.day))
	}
	d, _ := column_at_datetime(col, r)
	return days_from_civil(i64(d.year), i64(d.month), i64(d.day))
}

// Consecutive number of the period that contains day n
@(private = "file")
_period_of :: proc(n: i64, freq: Resample_Freq, week_end: Days) -> int {
	switch freq {
	case .Day:
		return int(n)
	case .Week:
		// weeks end on days with n ≡ res (mod 7); day 0 was a Thursday
		res := (i64(week_end) - 3) %% 7
		end := n + (res - n) %% 7
		return int((end - res) / 7)
	case .Month, .Quarter, .Year:
		d := civil_from_days(n)
		m := int(d.month) - 1
		switch freq {
		case .Month:
			return int(d.year) * 12 + m
		case .Quarter:
			return int(d.year) * 4 + m / 3
		case .Day, .Week, .Year:
		}
		return int(d.year)
	}
	return 0
}

// Day number of the last day of period p
@(private = "file")
_period_last_day :: proc(p: int, freq: Resample_Freq, week_end: Days) -> i64 {
	month_end :: proc(year, month: int) -> i64 { 	// month 1..12
		next_y, next_m := year, month + 1
		if next_m > 12 {next_y, next_m = year + 1, 1}
		return days_from_civil(i64(next_y), i64(next_m), 1) - 1
	}
	switch freq {
	case .Day:
		return i64(p)
	case .Week:
		res := (i64(week_end) - 3) %% 7
		return i64(p) * 7 + res
	case .Month:
		return month_end((p - p %% 12) / 12, p %% 12 + 1)
	case .Quarter:
		return month_end((p - p %% 4) / 4, (p %% 4) * 3 + 3)
	case .Year:
		return month_end(p, 12)
	}
	return 0
}
