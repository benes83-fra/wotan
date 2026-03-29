package core


import "core:mem"
import "core:slice"

RollingWindow :: struct {
	df:          ^DataFrame,
	column:      string,
	window:      int,
	min_periods: int,
	centered:    bool,
}


rolling_window :: proc(
	df: ^DataFrame,
	column: string,
	window: int,
	min_periods: int,
) -> RollingWindow {
	return RollingWindow{df = df, column = column, window = window, min_periods = min_periods}
}


rolling_apply :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> Column {
	src := column(r.df, r.column)
	switch src.type {
	case .Int:
		return rolling_apply_int(r, agg, allocator)
	case .Float:
		return rolling_apply_float(r, agg, allocator)
	case .Bool:
		return rolling_apply_bool(r, agg, allocator)
	case .Date:
		return rolling_apply_date(r, agg, allocator)
	case .Time:
		return rolling_apply_time(r, agg, allocator)
	case .Datetime:
		return rolling_apply_datetime(r, agg, allocator)
	case .String:
		return rolling_apply_string(r, agg, allocator)
	case .Invalid:
		panic("Invalid column type for rolling apply")
	}
	return column_new("invalid", .Invalid, 0)
}


rolling_apply_int :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]int, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_int(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_int_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_float :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]f64, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_float(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_float_into(&out, values[:], agg)
	}
	return out
}


rolling_apply_bool :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]bool, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_bool(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_bool_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_date :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]Date, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_date(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_date_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_time :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]Time, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_time(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_time_into(&out, values[:], agg)
	}
	return out
}
rolling_apply_datetime :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]Datetime, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_datetime(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_datetime_into(&out, values[:], agg)
	}
	return out
}
rolling_apply_string :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]string, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_string(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_string_into(&out, values[:], agg)
	}
	return out
}


rolling_apply_many :: proc(
	r: RollingWindow,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := dataframe_new()

	for agg in aggs {
		col := rolling_apply(r, agg, allocator)
		add_column(&out, col)
	}

	out.rows = out.columns[0].len
	return out
}

rolling_agg_int_into :: proc(out: ^Column, values: []int, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Sum:
		total := 0
		for v in values do total += v
		append_int(out, total)

	case .Mean:
		total := 0
		for v in values do total += v
		append_int(out, total / len(values))

	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		append_int(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		append_int(out, m)

	case .Count:
		append_int(out, len(values))

	case .Median, .Quantile:
		tmp := make([dynamic]int, len(values))
		copy(tmp[:], values)
		slice.sort(tmp[:])

		q := agg.quantile
		if kind == .Median do q = 0.5

		idx := int(q * f64(len(tmp) - 1))
		append_int(out, tmp[idx])
		delete(tmp)
	}
}


rolling_agg_float_into :: proc(out: ^Column, values: []f64, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Sum:
		total: f64 = 0
		for v in values do total += v
		append_float(out, total)

	case .Mean:
		total: f64 = 0
		for v in values do total += v
		append_float(out, total / f64(len(values)))

	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		append_float(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		append_float(out, m)

	case .Count:
		append_float(out, f64(len(values)))

	case .Median, .Quantile:
		tmp := make([dynamic]f64, len(values))
		copy(tmp[:], values)
		slice.sort(tmp[:])

		q := agg.quantile
		if kind == .Median do q = 0.5

		idx := int(q * f64(len(tmp) - 1))
		append_float(out, tmp[idx])
		delete(tmp)
	}
}


rolling_agg_bool_into :: proc(out: ^Column, values: []bool, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	count_true := 0
	for v in values do if v do count_true += 1

	switch kind {
	case .Sum:
		// Sum = any true?
		append_bool(out, count_true > 0)

	case .Mean:
		// Mean = ratio of true
		append_float(out, f64(count_true) / f64(len(values)))

	case .Min:
		// Min = false if any false exists
		m := true
		for v in values do if !v do m = false
		append_bool(out, m)

	case .Max:
		// Max = true if any true exists
		m := false
		for v in values do if v do m = true
		append_bool(out, m)

	case .Count:
		append_int(out, len(values))

	case .Median, .Quantile:
		append_null(out)
	}
}

rolling_agg_string_into :: proc(out: ^Column, values: []string, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		append_string(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		append_string(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile:
		append_null(out)
	}
}

rolling_agg_date_into :: proc(out: ^Column, values: []Date, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if date_less(v, m) do m = v
		append_date(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if date_less(m, v) do m = v
		append_date(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile:
		append_null(out)
	}
}

rolling_agg_time_into :: proc(out: ^Column, values: []Time, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if time_less(v, m) do m = v
		append_time(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if time_less(m, v) do m = v
		append_time(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile:
		append_null(out)
	}
}

rolling_agg_datetime_into :: proc(out: ^Column, values: []Datetime, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if datetime_less(v, m) do m = v
		append_datetime(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if datetime_less(m, v) do m = v
		append_datetime(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile:
		append_null(out)
	}
}
