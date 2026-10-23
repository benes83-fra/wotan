package core

import "core:fmt"

// Copy-based row slice: safe and immediate
dataframe_slice_rows_copy :: proc(df: ^DataFrame, start: int, end: int) -> DataFrame {
	if start < 0 || end < start || end > dataframe_row_count(df) {
		panic(fmt.tprintf("slice_rows: invalid range %d..%d", start, end))
	}

	rows := make([]int, end - start, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	for i in 0 ..< len(rows) {rows[i] = start + i}
	out := dataframe_new()
	for &col in df.columns {
		add_column(&out, column_take(&col, rows))
	}
	out.rows = len(rows)
	return out
}


dataframe_slice_rows :: proc(df: ^DataFrame, start: int, end: int, copy: bool) -> DataFrame {
	if copy {
		return dataframe_slice_rows_copy(df, start, end)
	}
	if start < 0 || end < start || end > df.rows {
		panic(fmt.tprintf("slice_rows: invalid range %d..%d", start, end))
	}
	out := dataframe_new()
	for ci in 0 ..< dataframe_column_count(df) {
		orig := dataframe_column_ptr(df, ci)
		new_col := column_slice_view(orig, start, end)
		add_column(&out, new_col)
	}
	out.rows = end - start
	return out
}
