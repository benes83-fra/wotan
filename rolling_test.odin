package main


import w "./wotan/core"
import "core:fmt"


rolling_test :: proc() {
	df := w.dataframe_new()

	col := w.column_new("value", .Int, 0)
	w.append_int(&col, 1)
	w.append_int(&col, 2)
	w.append_int(&col, 3)
	w.append_int(&col, 4)
	w.append_int(&col, 5)

	w.add_column(&df, col)
	df.rows = 5
	rw := w.rolling_window(&df, "value", 3, 1)
	result := w.rolling_apply(
		rw,
		w.Aggregator{name = "rolling_mean", column = "value", kind = .Mean},
	)
	defer w.destroy_column(&result)
	print_result_int(&result)
	colf := w.column_new("value_f", .Float, 0)

	w.append_float(&colf, 1.0)
	w.append_float(&colf, 2.0)
	w.append_float(&colf, 3.0)
	w.append_float(&colf, 4.0)
	w.append_float(&colf, 5.0)
	w.add_column(&df, colf)
	df.rows = 5
	rwf := w.rolling_window(&df, "value_f", 3, 1)
	resultf := w.rolling_apply(
		rwf,
		w.Aggregator{name = "rolling_mean", column = "value_f", kind = .Mean},
	)
	defer w.destroy_column(&resultf)
	print_result_float(&resultf)

	rwm := w.rolling_window(&df, "value_f", 3, 1)
	resultm := w.rolling_apply(rwm, w.make_median_agg("rolling_median", "value_f"))
	defer w.destroy_column(&resultm)
	print_result_float(&resultm)
	rwq := w.rolling_window(&df, "value_f", 3, 1)
	resultq := w.rolling_apply(rwq, w.make_quantile_agg("q25", "value_f", 0.25))
	defer w.destroy_column(&resultq)


	print_result_float(&resultq)
	rwe := w.rolling_window(&df, "value_f", 3, 1)
	resulte := w.rolling_apply(rwe, w.make_ewm_mean("ewm", "value_f", 0.5))
	print_result_float(&resulte)
	defer w.destroy_column(&resulte)
	w.destroy_dataframe(&df)
}


print_result_float :: proc(result: ^w.Column) {
	for i in 0 ..< result.len {
		v, is_null := w.column_at_float(result, i)
		if is_null {
			fmt.println("null")
		} else {
			fmt.println(v)
		}
	}
}

print_result_int :: proc(result: ^w.Column) {
	for i in 0 ..< result.len {
		v, is_null := w.column_at_int(result, i)
		if is_null {
			fmt.println("null")
		} else {
			fmt.println(v)
		}
	}
}
