package core

import "core:fmt"
import "core:mem"

dataframe_filter_bool_column :: proc(df: ^DataFrame, mask_col_name: string) -> DataFrame {
	// 1) Lookup mask column
	mask_col := column(df, mask_col_name)
	if mask_col.type != .Bool {
		panic("dataframe_filter_bool_column: mask column must be Bool")
	}

	// 2) Collect matching row indices
	idxs := make([dynamic]int, 0, df.rows)
	for i in 0 ..< df.rows {
		v, is_null := column_at_bool(mask_col, i)
		if !is_null && v {
			_, err := append(&idxs, i)
			if err != nil {
				panic("dataframe_filter_bool_column: append idx failed")
			}
		}
	}

	// 3) Build output DataFrame with copied rows
	out := dataframe_new()

	for &col in df.columns {
		new_col := column_new(col.name, col.type, len(idxs))

		for idx in idxs {
			#partial switch col.type {
			case .Int:
				v, is_null := column_at_int(&col, idx)
				if is_null {append_null(&new_col)} else {append_int(&new_col, v)}

			case .Float:
				v, is_null := column_at_float(&col, idx)
				if is_null {append_null(&new_col)} else {append_float(&new_col, v)}

			case .Bool:
				v, is_null := column_at_bool(&col, idx)
				if is_null {append_null(&new_col)} else {append_bool(&new_col, v)}

			case .String:
				v, is_null := column_at_string(&col, idx)
				if is_null {append_null(&new_col)} else {append_string(&new_col, v)}

			case .Date:
				v, is_null := column_at_date(&col, idx)
				if is_null {append_null(&new_col)} else {append_date(&new_col, v)}
			case .Time:
				v, is_null := column_at_time(&col, idx)
				if is_null {append_null(&new_col)} else {append_time(&new_col, v)}
			case .Datetime:
				v, is_null := column_at_datetime(&col, idx)
				if is_null {append_null(&new_col)} else {append_datetime(&new_col, v)}
			}
		}

		add_column(&out, new_col)
	}

	delete(idxs)
	return out
}
filter :: proc {
	dataframe_filter_by_bool_mask,
	dataframe_filter_column_by_col_name,
	dataframe_filter_column,
}

dataframe_filter_by_bool_mask :: proc(df: ^DataFrame, mask: []bool) -> DataFrame {
	if len(mask) != df.rows {
		panic("dataframe_filter: mask length mismatch")
	}

	// Collect indices of rows to keep
	idxs := make([dynamic]int, 0, df.rows)
	for i in 0 ..< df.rows {
		if mask[i] {
			_, err := append(&idxs, i)
			if err != nil {
				panic("dataframe_filter: append failed")
			}
		}
	}
	out := dataframe_new()

	// Build filtered columns
	for &col in df.columns {
		new_col := column_new(col.name, col.type, len(idxs))

		for idx in idxs {
			#partial switch col.type {
			case .Int:
				v, is_null := column_at_int(&col, idx)
				if is_null {append_null(&new_col)} else {append_int(&new_col, v)}

			case .Float:
				v, is_null := column_at_float(&col, idx)
				if is_null {append_null(&new_col)} else {append_float(&new_col, v)}

			case .Bool:
				v, is_null := column_at_bool(&col, idx)
				if is_null {append_null(&new_col)} else {append_bool(&new_col, v)}

			case .String:
				v, is_null := column_at_string(&col, idx)
				if is_null {append_null(&new_col)} else {append_string(&new_col, v)}

			case .Date:
				v, is_null := column_at_date(&col, idx)
				if is_null {append_null(&new_col)} else {append_date(&new_col, v)}
			case .Time:
				v, is_null := column_at_time(&col, idx)
				if is_null {append_null(&new_col)} else {append_time(&new_col, v)}
			case .Datetime:
				v, is_null := column_at_datetime(&col, idx)
				if is_null {append_null(&new_col)} else {append_datetime(&new_col, v)}
			}
		}

		add_column(&out, new_col)
	}
	delete(idxs)
	return out
}


dataframe_filter_column_by_col_name :: proc(df: ^DataFrame, col_name: string) -> DataFrame {
	col := column(df, col_name)
	if col.type != .Bool {
		panic("dataframe_filter_column: column must be Bool")
	}

	mask := make([]bool, df.rows)
	for i in 0 ..< df.rows {
		v, is_null := column_at_bool(col, i)
		mask[i] = (!is_null && v)
	}
	out := dataframe_filter_by_bool_mask(df, mask)
	delete(mask)
	return out
}

dataframe_filter_column :: proc(df: ^DataFrame, mask_col: Column) -> DataFrame {
	col := mask_col
	defer destroy_column(&col)
	if col.type != .Bool {
		panic("dataframe_filter_column: column must be Bool")
	}

	mask := make([]bool, df.rows)
	for i in 0 ..< df.rows {
		v, is_null := column_at_bool(&col, i)
		mask[i] = (!is_null && v)
	}
	out := dataframe_filter_by_bool_mask(df, mask)
	delete(mask)
	return out
}


dataframe_filter_series :: proc(df: ^DataFrame, s: Series) -> DataFrame {
	s := s
	if s.type != .Bool {
		panic("dataframe_filter_series: series must be Bool")
	}

	mask := make([]bool, s.len)
	for i in 0 ..< s.len {
		v, is_null := series_at_bool(&s, i)
		mask[i] = (!is_null && v)
	}

	out := filter(df, mask)
	delete(mask)
	return out
}
