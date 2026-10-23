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
	case .Min, .Max:
		return expr.col.type
	case .Sum:
		#partial switch expr.col.type {
		case .Float:
			return .Float
		case .Int, .Bool:
			return .Int
		}
		return expr.col.type // unsupported: nulls of the column's type
	case .Avg:
		#partial switch expr.col.type {
		case .Int, .Float, .Bool:
			return .Float
		}
		return expr.col.type
	}
	return .Int
}

// --- Helper: copy single value ----------------------------------------------

// Append src[row] to dst (view-aware, nulls kept; out-of-range rows give null)
copy_value_safe :: proc(dst: ^Column, src: ^Column, row: int) {
	if row < 0 || row >= src.len {
		append_null(dst)
		return
	}
	copy_cell(dst, src, row)
}


// --- Aggregation implementations ---------------------------------------------

// Sum of Int / Bool (number of trues) / Float; 0 for a group without values;
// other types give null
sum_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	col := expr.col
	#partial switch col.type {
	case .Int:
		total := 0
		for row in g.row_indices {
			if v, n := column_at_int(col, row); !n {total += v}
		}
		append_int(outcol, total)
	case .Bool:
		total := 0
		for row in g.row_indices {
			if v, n := column_at_bool(col, row); !n && v {total += 1}
		}
		append_int(outcol, total)
	case .Float:
		total := 0.0
		for row in g.row_indices {
			if v, n := column_at_float(col, row); !n && v == v {total += v}
		}
		append_float(outcol, total)
	case:
		append_null(outcol)
	}
}

// Mean of Int / Float / Bool as Float (nulls and NaN skipped); other types give null
avg_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	col := expr.col
	total := 0.0
	count := 0
	for row in g.row_indices {
		if is_null(col, row) {continue}
		#partial switch col.type {
		case .Int:
			v, _ := column_at_int(col, row)
			total += f64(v)
		case .Float:
			v, _ := column_at_float(col, row)
			if v != v {continue}
			total += v
		case .Bool:
			v, _ := column_at_bool(col, row)
			total += 1 if v else 0
		case:
			continue
		}
		count += 1
	}
	if count == 0 {
		append_null(outcol)
	} else {
		append_float(outcol, total / f64(count))
	}
}

// Smallest / largest value of any type (nulls and NaN skipped)
@(private = "file")
_extreme_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column, want_max: bool) {
	col := expr.col
	best := -1
	for row in g.row_indices {
		if is_null(col, row) {continue}
		if col.type == .Float {
			if v, _ := column_at_float(col, row); v != v {continue}
		}
		if best < 0 {
			best = row
			continue
		}
		c := column_compare_rows(col, row, best)
		if (want_max && c > 0) || (!want_max && c < 0) {best = row}
	}
	if best < 0 {
		append_null(outcol)
	} else {
		copy_cell(outcol, col, best)
	}
}

min_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	_extreme_value(expr, g, outcol, false)
}

max_value :: proc(expr: Agg_Expr, g: Group, outcol: ^Column) {
	_extreme_value(expr, g, outcol, true)
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
