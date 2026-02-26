package wotan

import "core:fmt"

// Copy-based row slice: safe and immediate
dataframe_slice_rows_copy :: proc(df: ^DataFrame, start: int, end: int) -> DataFrame {
	if start < 0 || end < start || end > dataframe_row_count(df) {
		panic(fmt.tprintf("slice_rows: invalid range %d..%d", start, end))
	}

	n := end - start
	out := dataframe_new()

	col_count := dataframe_column_count(df)
	for ci in 0 ..< col_count {
		name := dataframe_column_name(df, ci)
		col_type := dataframe_column_type(df, ci)
		new_col := column_new(name, col_type, n)

		orig := dataframe_column_ptr(df, ci)

		for i in start ..< end {
			#partial switch col_type {
			case .Int:
				v, is_null := column_at_int(orig, i)
				if is_null {append_null(&new_col)} else {append_int(&new_col, v)}
			case .Float:
				v, is_null := column_at_float(orig, i)
				if is_null {append_null(&new_col)} else {append_float(&new_col, v)}
			case .Bool:
				v, is_null := column_at_bool(orig, i)
				if is_null {append_null(&new_col)} else {append_bool(&new_col, v)}
			case .String:
				v, is_null := column_at_string(orig, i)
				if is_null {append_null(&new_col)} else {append_string(&new_col, v)}
			case .Date:
				v, is_null := column_at_date(orig, i)
				if is_null {append_null(&new_col)} else {append_date(&new_col, v)}
			}
		}

		add_column(&out, new_col)
	}

	return out
}


dataframe_slice_rows :: proc(df: ^DataFrame, start: int, end: int, copy: bool) -> DataFrame {
	if copy {
		return dataframe_slice_rows_copy(df, start, end)
	}
	out := dataframe_new()
	for ci in 0 ..< dataframe_column_count(df) {
		orig := dataframe_column_ptr(df, ci)
		new_col := column_slice_view(orig, start, end)
		add_column(&out, new_col)
	}
	return out
}
