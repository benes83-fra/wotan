package core

import "core:mem"

// Missing data: a cell is missing ("NA") when it is null, or when it is NaN in
// a Float column. Every procedure here treats the two the same way, like
// pandas does with None and NaN.
//
//   na_at / count_na      test or count missing cells
//   isna / notna          row masks ([]bool), usable with loc_mask, mask_and, ...
//   dropna                drop rows with missing cells
//   fillna                replace missing cells with a constant
//   ffill / bfill         carry the last / next present value into the gaps
//
// The procedures never change their input. Column results are new owned
// columns with the name of the source column; DataFrame results are new owned
// frames that keep the index metadata and row labels of the source, like
// df_take does. Inputs may be views (slices of a frame).

// How dropna combines the missing cells of a row
NA_How :: enum {
	Any, // drop the row if any checked cell is missing
	All, // drop the row only if every checked cell is missing
}

// True when the cell is null, or NaN in a Float column
na_at :: proc(col: ^Column, row: int) -> bool {
	if is_null(col, row) {return true}
	if col.type == .Float {
		v, _ := column_at_float(col, row)
		return v != v
	}
	return false
}

count_na :: proc {
	count_na_column,
	count_na_dataframe,
}

// Number of missing cells in a column
count_na_column :: proc(col: ^Column) -> int {
	n := 0
	for i in 0 ..< col.len {
		if na_at(col, i) {n += 1}
	}
	return n
}

// Missing cells per column, as a frame with the columns "column" (String,
// the column names in frame order) and "na" (Int, the counts)
count_na_dataframe :: proc(df: ^DataFrame, allocator := context.allocator) -> DataFrame {
	n := max(len(df.columns), 1)
	names := column_new("column", .String, n, allocator)
	counts := column_new("na", .Int, n, allocator)
	for &col in df.columns {
		append_string(&names, col.name)
		append_int(&counts, count_na_column(&col))
	}
	out := dataframe_new(allocator)
	add_column(&out, names)
	add_column(&out, counts)
	return out
}

// Mask that is true where the cell is missing
isna :: proc(col: ^Column, allocator := context.allocator) -> []bool {
	mask := make([]bool, col.len, allocator)
	for i in 0 ..< col.len {mask[i] = na_at(col, i)}
	return mask
}

// Mask that is true where the cell is present
notna :: proc(col: ^Column, allocator := context.allocator) -> []bool {
	mask := make([]bool, col.len, allocator)
	for i in 0 ..< col.len {mask[i] = !na_at(col, i)}
	return mask
}

// Rows of `df` without missing cells in `columns` (all columns when empty).
// how = .Any drops a row with any missing cell, .All only a row whose checked
// cells are all missing. thresh >= 0 replaces `how`: a row is kept when at
// least `thresh` of its checked cells are present.
// ok = false for an unknown column name.
dropna :: proc(
	df: ^DataFrame,
	columns: []string = nil,
	how := NA_How.Any,
	thresh := -1,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	cols, ok := _na_columns(df, columns)
	if !ok {return {}, false}
	defer delete(cols, context.temp_allocator)

	keep := make([dynamic]int, 0, df.rows, context.temp_allocator)
	defer delete(keep)
	for r in 0 ..< df.rows {
		missing := 0
		for c in cols {
			if na_at(c, r) {missing += 1}
		}
		present := len(cols) - missing
		keep_row: bool
		if thresh >= 0 {
			keep_row = present >= thresh
		} else if how == .Any {
			keep_row = missing == 0
		} else {
			keep_row = len(cols) == 0 || present > 0
		}
		if keep_row {append(&keep, r)}
	}
	return df_take(df, keep[:], allocator), true
}

fillna :: proc {
	fillna_column,
	fillna_dataframe,
}

// Copy of `col` with every missing cell set to `value`. The value's type must
// match the column (int, f64, bool, string, Date, Time, Datetime); an int value
// also fills a Float column. ok = false on a type mismatch.
fillna_column :: proc(col: ^Column, value: $T, allocator := context.allocator) -> (Column, bool) {
	if !_fill_type_ok(col.type, T) {return {}, false}
	out := column_new("", col.type, max(col.len, 1), allocator)
	column_own_name(&out, col.name)
	for i in 0 ..< col.len {
		if !na_at(col, i) {
			copy_cell(&out, col, i)
			continue
		}
		when T == int {
			if col.type == .Float {append_float(&out, f64(value))} else {append_int(&out, value)}
		} else when T == f64 {
			append_float(&out, value)
		} else when T == bool {
			append_bool(&out, value)
		} else when T == string {
			append_string(&out, value)
		} else when T == Date {
			append_date(&out, value)
		} else when T == Time {
			append_time(&out, value)
		} else when T == Datetime {
			append_datetime(&out, value)
		}
	}
	return out, true
}

// Copy of `df` with the missing cells of `columns` set to `value`. With no
// columns given, every column whose type takes the value is filled and the
// others are copied unchanged. ok = false for an unknown column name, or a
// named column whose type does not take the value.
fillna_dataframe :: proc(
	df: ^DataFrame,
	value: $T,
	columns: []string = nil,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	targets, ok := _na_targets(df, columns)
	if !ok {return {}, false}
	defer delete(targets, context.temp_allocator)
	for &col, i in df.columns {
		if !targets[i] {continue}
		if _fill_type_ok(col.type, T) {continue}
		if len(columns) > 0 {return {}, false}
		targets[i] = false
	}
	out_cols := make([]Column, len(df.columns), context.temp_allocator)
	defer delete(out_cols, context.temp_allocator)
	for &col, i in df.columns {
		if targets[i] {
			out_cols[i], _ = fillna_column(&col, value, allocator)
		} else {
			out_cols[i] = column_clone(&col, allocator)
		}
	}
	return _frame_like(df, out_cols, allocator), true
}

ffill :: proc {
	ffill_column,
	ffill_dataframe,
}

bfill :: proc {
	bfill_column,
	bfill_dataframe,
}

// Copy of `col` where each missing cell takes the last present value above it.
// limit > 0 fills at most `limit` cells of each gap (the rest stay missing);
// limit = 0 fills whole gaps. Cells before the first present value stay
// missing. ok = false for a negative limit.
ffill_column :: proc(col: ^Column, limit := 0, allocator := context.allocator) -> (Column, bool) {
	return _fill_direction(col, limit, false, allocator)
}

// Copy of `col` where each missing cell takes the next present value below it.
// limit as for ffill_column; cells after the last present value stay missing.
bfill_column :: proc(col: ^Column, limit := 0, allocator := context.allocator) -> (Column, bool) {
	return _fill_direction(col, limit, true, allocator)
}

// ffill_column applied to `columns` of `df` (all columns when empty); the
// other columns are copied unchanged. ok = false for an unknown column name or
// a negative limit.
ffill_dataframe :: proc(
	df: ^DataFrame,
	columns: []string = nil,
	limit := 0,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return _fill_frame(df, columns, limit, false, allocator)
}

// bfill_column applied to `columns` of `df` (all columns when empty)
bfill_dataframe :: proc(
	df: ^DataFrame,
	columns: []string = nil,
	limit := 0,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return _fill_frame(df, columns, limit, true, allocator)
}

// --- helpers ---

@(private = "file")
_fill_type_ok :: proc(type: ColumnType, $T: typeid) -> bool {
	when T == int {
		return type == .Int || type == .Float
	} else when T == f64 {
		return type == .Float
	} else when T == bool {
		return type == .Bool
	} else when T == string {
		return type == .String
	} else when T == Date {
		return type == .Date
	} else when T == Time {
		return type == .Time
	} else when T == Datetime {
		return type == .Datetime
	} else {
		return false
	}
}

// Columns named in `names` (all columns when empty), temp-allocated
@(private = "file")
_na_columns :: proc(df: ^DataFrame, names: []string) -> ([]^Column, bool) {
	if len(names) == 0 {
		cols := make([]^Column, len(df.columns), context.temp_allocator)
		for &c, i in df.columns {cols[i] = &c}
		return cols, true
	}
	cols := make([]^Column, len(names), context.temp_allocator)
	for name, i in names {
		idx, found := df.name_to_index[name]
		if !found {
			delete(cols, context.temp_allocator)
			return nil, false
		}
		cols[i] = &df.columns[idx]
	}
	return cols, true
}

// Per column of `df`: is it named in `names` (all true when empty)
@(private = "file")
_na_targets :: proc(df: ^DataFrame, names: []string) -> ([]bool, bool) {
	targets := make([]bool, len(df.columns), context.temp_allocator)
	if len(names) == 0 {
		for &t in targets {t = true}
		return targets, true
	}
	for name in names {
		idx, found := df.name_to_index[name]
		if !found {
			delete(targets, context.temp_allocator)
			return nil, false
		}
		targets[idx] = true
	}
	return targets, true
}

@(private = "file")
_fill_direction :: proc(
	col: ^Column,
	limit: int,
	backward: bool,
	allocator: mem.Allocator,
) -> (
	Column,
	bool,
) {
	if limit < 0 {return {}, false}
	// source row for every row: itself when present (or unfillable), else
	// the nearest present row in the fill direction
	rows := make([]int, col.len, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	last, gap := -1, 0
	for k in 0 ..< col.len {
		i := col.len - 1 - k if backward else k
		rows[i] = i
		if !na_at(col, i) {
			last, gap = i, 0
			continue
		}
		gap += 1
		if last >= 0 && (limit == 0 || gap <= limit) {rows[i] = last}
	}
	return column_take(col, rows, allocator), true
}

@(private = "file")
_fill_frame :: proc(
	df: ^DataFrame,
	columns: []string,
	limit: int,
	backward: bool,
	allocator: mem.Allocator,
) -> (
	DataFrame,
	bool,
) {
	if limit < 0 {return {}, false}
	targets, ok := _na_targets(df, columns)
	if !ok {return {}, false}
	defer delete(targets, context.temp_allocator)
	out_cols := make([]Column, len(df.columns), context.temp_allocator)
	defer delete(out_cols, context.temp_allocator)
	for &col, i in df.columns {
		if targets[i] {
			out_cols[i], _ = _fill_direction(&col, limit, backward, allocator)
		} else {
			out_cols[i] = column_clone(&col, allocator)
		}
	}
	return _frame_like(df, out_cols, allocator), true
}

// Frame of all rows of `src` built from `cols` (same row count), with the
// index metadata and row labels of `src`
@(private)
_frame_like :: proc(src: ^DataFrame, cols: []Column, allocator: mem.Allocator) -> DataFrame {
	out := dataframe_new(allocator)
	for c in cols {add_column(&out, c)}
	out.rows = src.rows
	out.index_column = src.index_column
	out.has_index = src.has_index
	if src.index_values.type != .Invalid {
		out.index_values = column_clone(&src.index_values, allocator)
	}
	if src.rows > 0 {
		labels := make([]int, src.rows, allocator)
		for r in 0 ..< src.rows {labels[r] = pretty_row_label(src, r)}
		out.index = labels
	}
	return out
}
