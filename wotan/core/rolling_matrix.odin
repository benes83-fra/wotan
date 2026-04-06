package core

import "core:fmt"
import "core:mem"

rolling_corr_matrix :: proc(
	df: ^DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	out := dataframe_new()

	// output schema
	add_column(&out, column_new("row", .Int, 0))
	add_column(&out, column_new("col_i", .String, 0))
	add_column(&out, column_new("col_j", .String, 0))
	add_column(&out, column_new("corr", .Float, 0))

	for i_idx in 0 ..< len(cols) {
		for j_idx in 0 ..< len(cols) {

			col_i := cols[i_idx]
			col_j := cols[j_idx]

			// rolling window on col_i
			rw := rolling_window(df, col_i, window, min_periods)

			// compute corr(col_i, col_j)
			agg := make_corr("corr_tmp", col_i, col_j)
			corr_col := rolling_apply(rw, agg, allocator)
			defer destroy_column(&corr_col)

			// append results
			for r in 0 ..< corr_col.len {
				append_int(&out.columns[0], r)
				append_string(&out.columns[1], col_i)
				append_string(&out.columns[2], col_j)

				v, is_null := column_at_float(&corr_col, r)
				if is_null {
					append_null(&out.columns[3])
				} else {
					append_float(&out.columns[3], v)
				}
			}
		}
	}

	out.rows = out.columns[0].len
	return out
}

rolling_var_matrix :: proc(
	df: ^DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	out := dataframe_new()

	add_column(&out, column_new("row", .Int, 0))
	add_column(&out, column_new("col", .String, 0))
	add_column(&out, column_new("var", .Float, 0))

	for col_name in cols {
		rw := rolling_window(df, col_name, window, min_periods)
		agg := make_var("tmp_var", col_name)
		var_col := rolling_apply(rw, agg, allocator)
		defer destroy_column(&var_col)

		for r in 0 ..< var_col.len {
			append_int(&out.columns[0], r)
			append_string(&out.columns[1], col_name)

			v, is_null := column_at_float(&var_col, r)
			if is_null {
				append_null(&out.columns[2])
			} else {
				append_float(&out.columns[2], v)
			}
		}
	}

	out.rows = out.columns[0].len
	return out
}
rolling_cov_matrix :: proc(
	df: ^DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	out := dataframe_new()

	add_column(&out, column_new("row", .Int, 0))
	add_column(&out, column_new("col_i", .String, 0))
	add_column(&out, column_new("col_j", .String, 0))
	add_column(&out, column_new("cov", .Float, 0))

	for i_idx in 0 ..< len(cols) {
		for j_idx in 0 ..< len(cols) {

			col_i := cols[i_idx]
			col_j := cols[j_idx]

			rw := rolling_window(df, col_i, window, min_periods)
			agg := make_cov("tmp_cov", col_i, col_j)
			cov_col := rolling_apply(rw, agg, allocator)
			defer destroy_column(&cov_col)

			for r in 0 ..< cov_col.len {
				append_int(&out.columns[0], r)
				append_string(&out.columns[1], col_i)
				append_string(&out.columns[2], col_j)

				v, is_null := column_at_float(&cov_col, r)
				if is_null {
					append_null(&out.columns[3])
				} else {
					append_float(&out.columns[3], v)
				}
			}
		}
	}

	out.rows = out.columns[0].len
	return out
}
