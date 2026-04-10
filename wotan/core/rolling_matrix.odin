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


ewm_cov_matrix :: proc(
	df: ^DataFrame,
	cols: []string,
	alpha: f64,
	min_periods: int,
	bias: bool,
	adjust: bool,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := dataframe_new()

	// schema: row, col_i, col_j, cov
	col_row := column_new("row", .Int, 0)
	col_i := column_new("col_i", .String, 0)
	col_j := column_new("col_j", .String, 0)
	col_cov := column_new("cov", .Float, 0)

	add_column(&out, col_row)
	add_column(&out, col_i)
	add_column(&out, col_j)
	add_column(&out, col_cov)

	n_cols := len(cols)
	if n_cols == 0 {
		out.rows = 0
		return out
	}

	// For each pair (i, j) with i <= j
	for i in 0 ..< n_cols {
		name_i := cols[i]
		col_x := column(df, name_i)

		for j in i ..< n_cols {
			name_j := cols[j]
			col_y := column(df, name_j)

			// Build a RollingWindow on X; EWM ignores window size
			rw := rolling_window(df, name_i, 0, min_periods)

			agg := make_ewm_cov("ewm_cov_tmp", name_i, alpha, bias)

			// Choose implementation based on type + adjust flag
			cov_series: Column
			switch col_x.type {
			case .Float:
				if adjust {
					cov_series = rolling_apply_float_ewm_cov(rw, col_y, agg, allocator)
				} else {
					cov_series = rolling_apply_float_ewm_cov_adjust_false(
						rw,
						col_y,
						agg,
						allocator,
					)
				}
			case .Int:
				if adjust {
					cov_series = rolling_apply_int_ewm_cov(rw, col_y, agg, allocator)
				} else {
					cov_series = rolling_apply_int_ewm_cov_adjust_false(rw, col_y, agg, allocator)
				}
			case .Invalid, .Bool, .String, .Date, .Time, .Datetime:
				// For now: skip unsupported types
				continue
			}

			// Emit long-form rows
			for r in 0 ..< cov_series.len {
				v, is_null := column_at_float(&cov_series, r)

				append_int(&out.columns[0], r) // "row"
				append_string(&out.columns[1], name_i) // "col_i"
				append_string(&out.columns[2], name_j) // "col_j"

				if is_null {
					append_null(&out.columns[3])
				} else {
					append_float(&out.columns[3], v)
				}
			}

			destroy_column(&cov_series)
		}
	}

	out.rows = out.columns[0].len
	return out
}


ewm_pca :: proc(
	df: ^DataFrame,
	cols: []string,
	alpha: f64,
	min_periods: int,
	bias: bool,
	adjust: bool,
	allocator: mem.Allocator = context.allocator,
) -> []PCAResult {

	cov_df := ewm_cov_matrix(df, cols, alpha, min_periods, bias, adjust, allocator)
	defer destroy_dataframe(&cov_df)

	// We have one covariance matrix per *row index* in the original df,
	// not per long-form row in cov_df.
	n_rows := df.rows
	results := make([]PCAResult, n_rows, allocator)

	for r in 0 ..< n_rows {
		cov := extract_cov_matrix_row(&cov_df, cols, r, allocator)
		results[r] = pca_from_cov(cov, allocator)
		destroy_matrix(cov)
	}

	return results
}


ewm_pca_last :: proc(
	df: ^DataFrame,
	cols: []string,
	alpha: f64,
	min_periods: int,
	bias: bool,
	adjust: bool,
	allocator: mem.Allocator = context.allocator,
) -> PCAResult {

	cov_df := ewm_cov_matrix(df, cols, alpha, min_periods, bias, adjust, allocator)
	defer destroy_dataframe(&cov_df)

	last_row := df.rows - 1
	cov := extract_cov_matrix_row(&cov_df, cols, last_row, allocator)
	defer destroy_matrix(cov)

	return pca_from_cov(cov, allocator)
}
