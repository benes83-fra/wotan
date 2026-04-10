package core

import "core:mem"

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
