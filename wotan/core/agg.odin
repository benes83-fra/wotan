package core


import "core:fmt"
import "core:mem"

import vmem "core:mem/virtual"
// --- Aggregation kinds -------------------------------------------------------

Agg_Kind :: enum {
	Count,
	Sum,
	Avg,
	Min,
	Max,
}

Agg_Expr :: struct {
	name: string,
	kind: Agg_Kind,
	col:  ^Column, //
}

// --- Constructors ------------------------------------------------------------

count :: proc(name: string) -> Agg_Expr {
	return Agg_Expr{name = name, kind = .Count}
}

sum_agg :: proc(name: string, col: ^Column) -> Agg_Expr {
	return Agg_Expr{name = name, kind = .Sum, col = col}
}

avg_agg :: proc(name: string, col: ^Column) -> Agg_Expr {
	return Agg_Expr{name = name, kind = .Avg, col = col}
}

min_agg :: proc(name: string, col: ^Column) -> Agg_Expr {
	return Agg_Expr{name = name, kind = .Min, col = col}
}

max_agg :: proc(name: string, col: ^Column) -> Agg_Expr {
	return Agg_Expr{name = name, kind = .Max, col = col}
}

// --- Type inference ----------------------------------------------------------

infer_agg_type :: proc(expr: Agg_Expr) -> ColumnType {
	switch expr.kind {
	case .Count:
		return .Int
	case .Sum, .Min, .Max:
		return expr.col.type
	case .Avg:
		if expr.col.type == .Int {
			return .Float
		}
		return expr.col.type // <-- FIXED

	}

	return .Int
}

// --- Helper: copy single value ----------------------------------------------

copy_value_safe :: proc(dst: ^Column, src: ^Column, row: int) {
	// if src is a view, compute the real index into the origin
	src := src
	real_index := row
	if src.is_view {
		real_index = src.offset + row
		src = src.orig
	}

	if real_index < 0 || real_index >= src.len {
		// out of range — append null to dst
		append_null(dst)
		return
	}

	#partial switch src.type {
	case .Int:
		v, n := column_at_int(src, real_index)
		if n {append_null(dst)} else {append_int(dst, v)}

	case .Float:
		v, n := column_at_float(src, real_index)
		if n {append_null(dst)} else {append_float(dst, v)}

	case .String:
		v, n := column_at_string(src, real_index)
		if n {append_null(dst)} else {append_string(dst, v)}

	case .Bool:
		v, n := column_at_bool(src, real_index)
		if n {append_null(dst)} else {append_bool(dst, v)}

	case .Date:
		v, n := column_at_date(src, real_index)
		if n {append_null(dst)} else {append_date(dst, v)}

	case .Time:
		v, n := column_at_time(src, real_index)
		if n {append_null(dst)} else {append_time(dst, v)}

	case .Datetime:
		v, n := column_at_datetime(src, real_index)
		if n {append_null(dst)} else {append_datetime(dst, v)}
	}
}


// --- Aggregation implementations ---------------------------------------------

sum_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	col := expr.col

	#partial switch col.type {
	case .Int:
		total := 0
		for row in g.row_indices {
			v, n := column_at_int(col, row)
			if !n {total += v}
		}
		append_int(outcol, total)

	case .Float:
		total := 0.0
		for row in g.row_indices {
			v, n := column_at_float(col, row)
			if !n {total += v}
		}
		append_float(outcol, total) //experimental attempt since even if Int it should make sense calculate average, thus column type check disabled int his call
	}
}

avg_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	col := expr.col

	#partial switch col.type {
	case .Int:
		total := 0
		count := 0
		for row in g.row_indices {
			v, n := column_at_int(col, row)
			if !n {total += v; count += 1}
		}
		if count == 0 {
			append_null(outcol)
		} else {
			append_fake_float(outcol, f64(total / count))
		}

	case .Float:
		total := 0.0
		count := 0
		for row in g.row_indices {
			v, n := column_at_float(col, row)
			if !n {total += v; count += 1}
		}
		if count == 0 {
			append_null(outcol)
		} else {
			append_float(outcol, total / f64(count))
		}
	}
}

min_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	col := expr.col

	#partial switch col.type {
	case .Int:
		first := true
		minv := 0
		for row in g.row_indices {
			fmt.println(row)
			v, n := column_at_int(col, row)
			fmt.println(v)
			if n {continue}
			if first || v < minv {minv = v; first = false}
		}
		if first {append_null(outcol)} else {append_int(outcol, minv)}

	case .Float:
		first := true
		minv := 0.0
		for row in g.row_indices {
			fmt.println(row)
			v, n := column_at_float(col, row)
			fmt.println(v)
			if n {continue}
			if first || v < minv {minv = v; first = false}
		}
		if first {append_null(outcol)} else {append_float(outcol, minv)}
	}
}

max_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	col := expr.col

	#partial switch col.type {
	case .Int:
		first := true
		maxv := 0
		for row in g.row_indices {
			v, n := column_at_int(col, row)
			if n {continue}
			if first || v > maxv {maxv = v; first = false}
		}
		if first {append_null(outcol)} else {append_int(outcol, maxv)}

	case .Float:
		first := true
		maxv := 0.0
		for row in g.row_indices {
			v, n := column_at_float(col, row)
			if n {continue}
			if first || v > maxv {maxv = v; first = false}
		}
		if first {append_null(outcol)} else {append_float(outcol, maxv)}
	}
}

// --- Aggregation driver ------------------------------------------------------

agg_with_allocator :: proc(
	gdf: ^GroupedDataFrame,
	exprs: []Agg_Expr,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := dataframe_new()

	// 1) key columns in output
	for ki in 0 ..< len(gdf.keys) {
		key_name := gdf.keys[ki]
		src_col := column(gdf.df, key_name)
		out_col := column_new(key_name, src_col.type, len(gdf.groups))
		add_column(&out, out_col)
	}

	// 2) agg columns in output
	for ei in 0 ..< len(exprs) {
		expr := exprs[ei]
		t := infer_agg_type(expr)
		out_col := column_new(expr.name, t, len(gdf.groups))
		add_column(&out, out_col)
	}


	// 3) fill rows: one per group
	for gi in 0 ..< len(gdf.groups) {
		g := gdf.groups[gi]

		// 3a) write key columns: take first row of each group
		first_row := g.row_indices[0]
		for ki in 0 ..< len(gdf.keys) {
			key_name := gdf.keys[ki]
			src_col := column(gdf.df, key_name)
			out_col := &out.columns[ki]

			copy_value_safe(out_col, src_col, first_row) // your existing helper
		}


		// 3b) write agg columns
		for ei in 0 ..< len(exprs) {
			expr := exprs[ei]
			out_col := &out.columns[len(gdf.keys) + ei]

			#partial switch expr.kind {
			case .Count:
				append_int(out_col, len(g.row_indices))

			case .Sum:
				sum_value(expr, g, out_col)

			case .Avg:
				avg_value(expr, g, out_col)

			case .Min:
				min_value(expr, g, out_col)

			case .Max:
				max_value(expr, g, out_col)
			}
		}

	}


	// set DataFrame row count so printers see the rows we appended
	// after filling all groups
	expected := len(gdf.groups)
	for c in out.columns {
		if c.len != expected {
			fmt.printf(
				"agg invariant violated: column %s len=%d, expected=%d\n",
				c.name,
				c.len,
				expected,
			)
			panic("agg: inconsistent column lengths")
		}
	}

	out.rows = expected


	return out
}

agg :: proc(gdf: ^GroupedDataFrame, exprs: []Agg_Expr) -> DataFrame {
	allocator := vmem.arena_allocator(&gdf.arena)
	return agg_with_allocator(gdf, exprs, allocator)
}
