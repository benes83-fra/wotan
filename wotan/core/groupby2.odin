package core

import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"


GroupBy2 :: struct {
	df:    ^DataFrame,
	keys:  []string,
	index: Join_Multi_Index,
}

AggregationKind :: enum {
	Sum,
	Mean,
	Min,
	Max,
	Count,
	Median,
	Quantile,
	EWM_Mean, // new
	EWM_Var,
	EWM_Std,
	EWM_Cov,
	EWM_Corr,
}


Aggregator :: struct {
	name:     string,
	column:   string,
	other:    string, // ← ADD THIS
	kind:     AggregationKind,
	quantile: f64,
	alpha:    f64,
	bias:     bool,
}


make_median_agg :: proc(name, column: string) -> Aggregator {
	return Aggregator{name = name, column = column, kind = .Median, quantile = 0.5}
}

make_quantile_agg :: proc(name, column: string, q: f64) -> Aggregator {
	return Aggregator{name = name, column = column, kind = .Quantile, quantile = q, bias = false}
}


groupby2 :: proc(
	df: ^DataFrame,
	keys: []string,
	allocator: mem.Allocator = context.allocator,
) -> GroupBy2 {
	// 1. Extract key columns
	key_cols := make([]^Column, len(keys), allocator)
	for i in 0 ..< len(keys) {
		key_cols[i] = column(df, keys[i])
	}

	// 2. Build multi-key index (same as join)
	idx := build_join_multi_index(df, key_cols, allocator)

	// 3. Return GroupBy2 struct
	return GroupBy2{df = df, keys = keys, index = idx}
}


groupby2_debug_print :: proc(gb: ^GroupBy2) {
	fmt.printf("GroupBy2 keys = %v\n", gb.keys)

	for key, head in gb.index.bucket_head {
		fmt.printf("Group key: %s\n", key)

		i := head
		fmt.print("  rows: ")
		for i != -1 {
			fmt.printf("%d ", gb.index.rows[i])
			i = gb.index.next[i]
		}
		fmt.println()
	}
}
build_groupby2_schema :: proc(
	gb: ^GroupBy2,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := dataframe_new()

	// 1. Add key columns (same type as original)
	for key in gb.keys {
		src := column(gb.df, key)
		dst := column_new(key, src.type, 0)
		add_column(&out, dst)
	}

	// 2. Add aggregated columns
	for agg in aggs {
		src := column(gb.df, agg.column)

		col_type := src.type
		if agg.kind == .Count {
			col_type = .Int
		}

		new_col := column_new(agg.name, col_type, 0)
		add_column(&out, new_col)
	}

	return out
}


agg_compute_into :: proc(out_col: ^Column, src_col: ^Column, rows: []int, agg: Aggregator) {
	#partial switch src_col.type {
	case .Int:
		agg_int_into(out_col, src_col, rows, agg)
	case .Float:
		agg_float_into(out_col, src_col, rows, agg)
	case .Bool:
		agg_bool_into(out_col, src_col, rows, agg)
	case .String:
		agg_string_into(out_col, src_col, rows, agg)
	case .Date:
		agg_date_into(out_col, src_col, rows, agg)
	case .Time:
		agg_time_into(out_col, src_col, rows, agg)
	case .Datetime:
		agg_datetime_into(out_col, src_col, rows, agg)
	}
}


emit_groupby2_row :: proc(
	out: ^DataFrame,
	gb: ^GroupBy2,
	group_head: int,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) {
	// 1. Collect row indices for this group
	rows := make([dynamic]int, 0, 16, allocator)
	i := group_head
	for i != -1 {
		append(&rows, gb.index.rows[i])
		i = gb.index.next[i]
	}


	// 2. Emit key values (use first row of group)
	first := rows[0]
	for key, ki in gb.keys {
		src := column(gb.df, key)
		dst := &out.columns[ki]
		copy_cell(dst, src, first)
	}


	// 3. Emit aggregated values
	key_count := len(gb.keys)
	for agg, ai in aggs {
		src := column(gb.df, agg.column)
		out_col := &out.columns[key_count + ai]
		agg_compute_into(out_col, src, rows[:], agg)
	}

}


groupby2_agg :: proc(
	gb: ^GroupBy2,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := build_groupby2_schema(gb, aggs, allocator)

	// Iterate groups in stable order
	for key in gb.index.ordered_keys {
		head := gb.index.bucket_head[key]
		emit_groupby2_row(&out, gb, head, aggs, allocator)
	}

	out.rows = out.columns[0].len
	return out
}

agg_int_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind
	sum := 0
	count := 0
	min := int(1 << 62)
	max := -min

	for r in rows {
		v, is_null := column_at_int(src, r)
		if is_null do continue

		count += 1
		sum += v
		if v < min do min = v
		if v > max do max = v
	}

	if count == 0 {
		append_null(out)
		return
	}

	#partial switch kind {
	case .Sum:
		append_int(out, sum)
	case .Mean:
		append_int(out, sum / count)
	case .Min:
		append_int(out, min)
	case .Max:
		append_int(out, max)
	case .Count:
		append_int(out, count)
	case .Median, .Quantile:
		values := make([dynamic]int, 0, len(rows))
		defer delete(values)
		for r in rows {
			v, is_null := column_at_int(src, r)
			if !is_null do append(&values, v)
		}

		if len(values) == 0 {
			append_null(out)
			return
		}
		q: f64
		if kind == .Median {
			q = 0.5
		} else {
			q = agg.quantile
		}

		slice.sort(values[:])
		idx := int(q * f64(len(values) - 1))
		append_int(out, values[idx])
	}
}


agg_float_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind

	sum := 0.0
	count := 0
	min := math.INF_F64
	max := -math.INF_F64

	for r in rows {
		v, is_null := column_at_float(src, r)
		if is_null do continue

		count += 1
		sum += v
		if v < min do min = v
		if v > max do max = v
	}

	if count == 0 {
		append_null(out)
		return
	}

	#partial switch kind {
	case .Sum:
		append_float(out, sum)
	case .Mean:
		append_float(out, sum / f64(count))
	case .Min:
		append_float(out, min)
	case .Max:
		append_float(out, max)
	case .Count:
		append_float(out, f64(count))
	case .Median, .Quantile:
		values := make([dynamic]f64, 0, len(rows))
		defer delete(values)
		for r in rows {
			v, is_null := column_at_float(src, r)
			if !is_null do append(&values, v)
		}

		if len(values) == 0 {
			append_null(out)
			return
		}
		q: f64
		if kind == .Median {
			q = 0.5
		} else {
			q = agg.quantile
		}

		slice.sort(values[:])
		idx := int(q * f64(len(values) - 1))
		append_float(out, values[idx])
	}
}


agg_bool_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind
	count_true := 0
	count := 0
	min := true
	max := false

	for r in rows {
		v, is_null := column_at_bool(src, r)
		if is_null do continue

		count += 1
		if v do count_true += 1
		if !v do min = false
		if v do max = true
	}

	if count == 0 {
		append_null(out)
		return
	}
	switch kind {
	case .Sum:
		append_bool(out, count_true > 0)
	case .Mean:
		append_float(out, f64(count_true) / f64(count))
	case .Min:
		append_bool(out, min)
	case .Max:
		append_bool(out, max)
	case .Count:
		append_int(out, count)
	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr:
		append_null(out)

	}
}


agg_string_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind
	count := 0
	min := ""
	max := ""

	first := true

	for r in rows {
		v, is_null := column_at_string(src, r)
		if is_null do continue

		count += 1

		if first {
			min = v
			max = v
			first = false
		} else {
			if v < min do min = v
			if v > max do max = v
		}
	}

	if count == 0 {
		append_null(out)
		return
	}

	switch kind {
	case .Min:
		append_string(out, min)
	case .Max:
		append_string(out, max)
	case .Count:
		append_int(out, count)
	case .Sum, .Mean:
		append_null(out)
	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr:
		append_null(out)
	}
}


agg_date_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind
	count := 0
	min := Date {
		year  = 9999,
		month = 12,
		day   = 31,
	}
	max := Date {
		year  = 0,
		month = 1,
		day   = 1,
	}

	for r in rows {
		v, is_null := column_at_date(src, r)
		if is_null do continue

		count += 1
		if date_less(v, min) do min = v
		if date_less(max, v) do max = v
	}

	if count == 0 {
		append_null(out)
		return
	}

	switch kind {
	case .Min:
		append_date(out, min)
	case .Max:
		append_date(out, max)
	case .Count:
		append_int(out, count)
	case .Sum, .Mean:
		append_null(out)
	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr:
		append_null(out)

	}
}


agg_time_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind
	count := 0
	min := Time {
		hour   = 23,
		minute = 59,
		second = 59,
	}
	max := Time {
		hour   = 0,
		minute = 0,
		second = 0,
	}

	for r in rows {
		v, is_null := column_at_time(src, r)
		if is_null do continue

		count += 1
		if time_less(v, min) do min = v
		if time_less(max, v) do max = v
	}

	if count == 0 {
		append_null(out)
		return
	}

	switch kind {
	case .Min:
		append_time(out, min)
	case .Max:
		append_time(out, max)
	case .Count:
		append_int(out, count)
	case .Sum, .Mean:
		append_null(out)
	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr:
		append_null(out)
	}
}


agg_datetime_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	kind := agg.kind
	count := 0
	min := Datetime {
		year  = 9999,
		month = 12,
		day   = 31,
	}
	max := Datetime {
		year  = 0,
		month = 1,
		day   = 1,
	}

	for r in rows {
		v, is_null := column_at_datetime(src, r)
		if is_null do continue

		count += 1
		if datetime_less(v, min) do min = v
		if datetime_less(max, v) do max = v
	}

	if count == 0 {
		append_null(out)
		return
	}

	switch kind {
	case .Min:
		append_datetime(out, min)
	case .Max:
		append_datetime(out, max)
	case .Count:
		append_int(out, count)
	case .Sum, .Mean:
		append_null(out)
	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr:
		append_null(out)
	}
}
copy_cell :: proc(dst: ^Column, src: ^Column, row: int) {
	#partial switch src.type {
	case .Int:
		v, is_null := column_at_int(src, row)
		if is_null {append_null(dst)
		} else {
			append_int(dst, v)
		}
	case .Float:
		v, is_null := column_at_float(src, row)
		if is_null {
			append_null(dst)
		} else {
			append_float(dst, v)
		}
	case .Bool:
		v, is_null := column_at_bool(src, row)
		if is_null {
			append_null(dst)} else {
			append_bool(dst, v)
		}
	case .String:
		v, is_null := column_at_string(src, row)
		if is_null {
			append_null(dst)
		} else {

			append_string(dst, v)
		}
	case .Date:
		v, is_null := column_at_date(src, row)
		if is_null {
			append_null(dst)
		} else {
			append_date(dst, v)
		}
	case .Time:
		v, is_null := column_at_time(src, row)
		if is_null {
			append_null(dst)
		} else {
			append_time(dst, v)
		}
	case .Datetime:
		v, is_null := column_at_datetime(src, row)
		if is_null {
			append_null(dst)
		} else {
			append_datetime(dst, v)
		}
	}
}
