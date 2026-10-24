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
	Corr,
	Var,
	Cov,
	EWM_Mean,
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
make_corr :: proc(name, col, other: string) -> Aggregator {
	return Aggregator{name = name, column = col, other = other, kind = .Corr}
}
make_var :: proc(name, col: string) -> Aggregator {
	return Aggregator{name = name, column = col, kind = .Var}
}

make_cov :: proc(name, col, other: string) -> Aggregator {
	return Aggregator{name = name, column = col, other = other, kind = .Cov}
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
// Output column type of an aggregation over a column of type `src`
groupby2_output_type :: proc(src: ColumnType, kind: AggregationKind) -> ColumnType {
	numeric := src == .Int || src == .Float || src == .Bool
	switch kind {
	case .Count:
		return .Int
	case .Min, .Max:
		return src
	case .Sum:
		if src == .Float {return .Float}
		return .Int if numeric else src
	case .Mean,
	     .Median,
	     .Quantile,
	     .Var,
	     .Cov,
	     .Corr,
	     .EWM_Mean,
	     .EWM_Var,
	     .EWM_Std,
	     .EWM_Cov,
	     .EWM_Corr:
		return .Float if numeric else src
	}
	return src
}

build_groupby2_schema :: proc(
	gb: ^GroupBy2,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := dataframe_new(allocator)

	// 1. Key columns (same type as original)
	for key in gb.keys {
		src := column(gb.df, key)
		add_column(&out, column_new(key, src.type, 0, allocator))
	}

	// 2. Aggregated columns
	for agg in aggs {
		src := column(gb.df, agg.column)
		if (agg.kind == .Cov ||
			   agg.kind == .Corr ||
			   agg.kind == .EWM_Cov ||
			   agg.kind == .EWM_Corr) &&
		   agg.other == "" {
			panic(
				fmt.tprintf(
					"groupby2: %v aggregation %q needs a second column (other)",
					agg.kind,
					agg.name,
				),
			)
		}
		if agg.kind == .Quantile && !(agg.quantile >= 0 && agg.quantile <= 1) {
			panic(
				fmt.tprintf(
					"groupby2: quantile %v of %q is outside [0, 1]",
					agg.quantile,
					agg.name,
				),
			)
		}
		if (agg.kind == .EWM_Mean ||
			   agg.kind == .EWM_Var ||
			   agg.kind == .EWM_Std ||
			   agg.kind == .EWM_Cov ||
			   agg.kind == .EWM_Corr) &&
		   !(agg.alpha > 0 && agg.alpha <= 1) {
			panic(
				fmt.tprintf("groupby2: EWM alpha %v of %q must be in (0, 1]", agg.alpha, agg.name),
			)
		}
		add_column(
			&out,
			column_new(agg.name, groupby2_output_type(src.type, agg.kind), 0, allocator),
		)
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
	case:
		append_null(out_col)
	}
}


emit_groupby2_row :: proc(
	out: ^DataFrame,
	gb: ^GroupBy2,
	group_head: int,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) {
	// 1. Row indices of this group (in row order)
	rows := make([dynamic]int, 0, 16, context.temp_allocator)
	defer delete(rows)
	for i := group_head; i != -1; i = gb.index.next[i] {
		append(&rows, gb.index.rows[i])
	}

	// 2. Key values (first row of the group; a null key stays null)
	for key, ki in gb.keys {
		copy_cell(&out.columns[ki], column(gb.df, key), rows[0])
	}

	// 3. Aggregated values
	key_count := len(gb.keys)
	for agg, ai in aggs {
		out_col := &out.columns[key_count + ai]
		src := column(gb.df, agg.column)
		#partial switch agg.kind {
		case .Cov, .Corr, .EWM_Cov, .EWM_Corr:
			_agg_pair_into(out_col, src, column(gb.df, agg.other), rows[:], agg)
		case:
			agg_compute_into(out_col, src, rows[:], agg)
		}
	}
}


groupby2_agg :: proc(
	gb: ^GroupBy2,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := build_groupby2_schema(gb, aggs, allocator)

	// Groups in order of first appearance
	for key in gb.index.ordered_keys {
		emit_groupby2_row(&out, gb, gb.index.bucket_head[key], aggs, allocator)
	}

	out.rows = len(gb.index.ordered_keys)
	for c in out.columns {
		fmt.assertf(
			c.len == out.rows,
			"groupby2_agg: column %q has %d rows, expected %d",
			c.name,
			c.len,
			out.rows,
		)
	}
	return out
}

destroy_groupby2 :: proc(gb: ^GroupBy2) {
	join_multi_index_destroy(&gb.index)
}


// --- numeric aggregations ----------------------------------------------------

// Non-null values of a numeric column (Int / Float / Bool as f64), in row order
@(private = "file")
_numeric_values :: proc(src: ^Column, rows: []int) -> [dynamic]f64 {
	vals := make([dynamic]f64, 0, len(rows), context.temp_allocator)
	for r in rows {
		if is_null(src, r) {continue}
		v: f64
		#partial switch src.type {
		case .Int:
			x, _ := column_at_int(src, r)
			v = f64(x)
		case .Float:
			v, _ = column_at_float(src, r)
			if v != v {continue} 	// NaN is skipped like a null
		case .Bool:
			x, _ := column_at_bool(src, r)
			v = 1 if x else 0
		}
		append(&vals, v)
	}
	return vals
}

// Linear-interpolated quantile of sorted values (pandas default)
@(private = "file")
_quantile :: proc(sorted: []f64, q: f64) -> f64 {
	pos := q * f64(len(sorted) - 1)
	lo := int(math.floor(pos))
	hi := min(lo + 1, len(sorted) - 1)
	return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - f64(lo))
}

// EWM weights (adjust=True): w_i = (1 - alpha)^(n - 1 - i), newest weight 1
@(private = "file")
_ewm_weights :: proc(n: int, alpha: f64) -> []f64 {
	w := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {w[i] = math.pow(1 - alpha, f64(n - 1 - i))}
	return w
}

// Weighted (co)variance with pandas' bias correction (bias=false)
@(private = "file")
_ewm_cov :: proc(x, y, w: []f64, bias: bool) -> f64 {
	sw, sw2, mx, my := 0.0, 0.0, 0.0, 0.0
	for i in 0 ..< len(x) {
		sw += w[i]
		sw2 += w[i] * w[i]
		mx += w[i] * x[i]
		my += w[i] * y[i]
	}
	mx /= sw
	my /= sw
	c := 0.0
	for i in 0 ..< len(x) {c += w[i] * (x[i] - mx) * (y[i] - my)}
	c /= sw
	if !bias {
		denom := sw * sw - sw2
		if denom <= 0 {return math.nan_f64()}
		c *= sw * sw / denom
	}
	return c
}

// Aggregate the numeric values of a group into `out` (type from groupby2_output_type)
@(private = "file")
_agg_numeric :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	vals := _numeric_values(src, rows)
	defer delete(vals)
	n := len(vals)

	put :: proc(out: ^Column, v: f64) {
		if v != v {
			append_null(out)
			return
		}
		#partial switch out.type {
		case .Int:
			append_int(out, int(v))
		case .Bool:
			append_bool(out, v != 0)
		case:
			append_float(out, v)
		}
	}

	if agg.kind == .Count {
		append_int(out, n)
		return
	}
	if n == 0 {
		if agg.kind == .Sum { 	// sum of nothing is 0 (pandas min_count=0)
			put(out, 0)
		} else {
			append_null(out)
		}
		return
	}

	#partial switch agg.kind {
	case .Sum:
		if src.type == .Int { 	// exact integer sum
			total := 0
			for r in rows {
				if v, null := column_at_int(src, r); !null {total += v}
			}
			append_int(out, total)
			return
		}
		s := 0.0
		for v in vals {s += v}
		put(out, s)
	case .Mean:
		s := 0.0
		for v in vals {s += v}
		put(out, s / f64(n))
	case .Min, .Max:
		best := -1
		for r in rows { 	// keep the original value / type (exact ints)
			if is_null(src, r) {continue}
			if src.type == .Float {
				if v, _ := column_at_float(src, r); v != v {continue}
			}
			if best < 0 {best = r; continue}
			c := column_compare_rows(src, r, best)
			if (agg.kind == .Min && c < 0) || (agg.kind == .Max && c > 0) {best = r}
		}
		copy_cell(out, src, best)
	case .Median, .Quantile:
		sorted := slice_clone_sorted(vals[:])
		defer delete(sorted, context.temp_allocator)
		put(out, _quantile(sorted, 0.5 if agg.kind == .Median else agg.quantile))
	case .Var:
		if n < 2 {
			append_null(out)
			return
		}
		m := 0.0
		for v in vals {m += v}
		m /= f64(n)
		ss := 0.0
		for v in vals {ss += (v - m) * (v - m)}
		put(out, ss / f64(n - 1))
	case .EWM_Mean, .EWM_Var, .EWM_Std:
		w := _ewm_weights(n, agg.alpha)
		defer delete(w, context.temp_allocator)
		if agg.kind == .EWM_Mean {
			sw, s := 0.0, 0.0
			for i in 0 ..< n {sw += w[i]; s += w[i] * vals[i]}
			put(out, s / sw)
			return
		}
		v := _ewm_cov(vals[:], vals[:], w, agg.bias)
		put(out, math.sqrt(v) if agg.kind == .EWM_Std else v)
	case:
		append_null(out)
	}
}

// Sorted copy (temp allocator)
slice_clone_sorted :: proc(v: []f64) -> []f64 {
	out := make([]f64, len(v), context.temp_allocator)
	copy(out, v)
	slice.sort(out)
	return out
}

// Pairwise statistics over rows where both columns are present
@(private = "file")
_agg_pair_into :: proc(out: ^Column, a: ^Column, b: ^Column, rows: []int, agg: Aggregator) {
	numeric :: proc(t: ColumnType) -> bool {return t == .Int || t == .Float || t == .Bool}
	if !numeric(a.type) || !numeric(b.type) {
		append_null(out)
		return
	}
	xs := make([dynamic]f64, 0, len(rows), context.temp_allocator)
	ys := make([dynamic]f64, 0, len(rows), context.temp_allocator)
	defer {delete(xs); delete(ys)}
	for r in rows {
		one := []int{r}
		x := _numeric_values(a, one)
		y := _numeric_values(b, one)
		if len(x) == 1 && len(y) == 1 {
			append(&xs, x[0])
			append(&ys, y[0])
		}
		delete(x)
		delete(y)
	}
	n := len(xs)
	if n < 2 {
		append_null(out)
		return
	}
	w: []f64
	bias := agg.bias
	if agg.kind == .Cov || agg.kind == .Corr { 	// sample statistics: equal weights, n - 1
		w = make([]f64, n, context.temp_allocator)
		for &v in w {v = 1}
		bias = false
	} else {
		w = _ewm_weights(n, agg.alpha)
	}
	defer delete(w, context.temp_allocator)
	cxy := _ewm_cov(xs[:], ys[:], w, bias)
	if agg.kind == .Cov || agg.kind == .EWM_Cov {
		if cxy != cxy {append_null(out)} else {append_float(out, cxy)}
		return
	}
	sx := _ewm_cov(xs[:], xs[:], w, bias)
	sy := _ewm_cov(ys[:], ys[:], w, bias)
	r := cxy / math.sqrt(sx * sy)
	if r != r || sx <= 0 || sy <= 0 {append_null(out)} else {append_float(out, r)}
}

agg_int_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_numeric(out, src, rows, agg)
}

agg_float_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_numeric(out, src, rows, agg)
}

// Bool: Sum = number of trues, Mean = fraction of trues, Min / Max = all / any
agg_bool_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_numeric(out, src, rows, agg)
}

// Non-numeric columns: Count, Min, Max; other aggregations give null
@(private = "file")
_agg_ordered :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	#partial switch agg.kind {
	case .Count:
		n := 0
		for r in rows {
			if !is_null(src, r) {n += 1}
		}
		append_int(out, n)
	case .Min, .Max:
		best := -1
		for r in rows {
			if is_null(src, r) {continue}
			if best < 0 {best = r; continue}
			c := column_compare_rows(src, r, best)
			if (agg.kind == .Min && c < 0) || (agg.kind == .Max && c > 0) {best = r}
		}
		if best < 0 {append_null(out)} else {copy_cell(out, src, best)}
	case:
		append_null(out)
	}
}

agg_string_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_ordered(out, src, rows, agg)
}

agg_date_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_ordered(out, src, rows, agg)
}

agg_time_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_ordered(out, src, rows, agg)
}

agg_datetime_into :: proc(out: ^Column, src: ^Column, rows: []int, agg: Aggregator) {
	_agg_ordered(out, src, rows, agg)
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
