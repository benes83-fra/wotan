package main


import w "./wotan/core"
import "core:fmt"
import "core:mem"


rolling_test :: proc(allocator: mem.Allocator) {
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
	rwev := w.rolling_window(&df, "value_f", 3, 1)
	resultv := w.rolling_apply(rwev, w.make_ewm_var("ewm_var", "value_f", 0.5))


	print_result_float(&resultv)

	// --- EWM STD (adjust=False) ---
	fmt.println("EWM STD (alpha=0.5, adjust=False)")
	rws := w.rolling_window(&df, "value_f", 3, 1)
	results := w.rolling_apply(
	rws,
	w.Aggregator {
		name   = "ewm_std",
		column = "value_f",
		kind   = .EWM_Std, // assuming you add this enum
		alpha  = 0.5,
	},
	)
	print_result_float(&results)
	w.destroy_column(&results)

	// --- EWM VAR (adjust=True) ---
	fmt.println("EWM VAR (alpha=0.5, adjust=True)")
	rwvat := w.rolling_window(&df, "value_f", 3, 1)
	resultvat := w.rolling_apply_float_ewm_var_adjust_true(
		rwvat,
		w.Aggregator{name = "ewm_var_adj", column = "value_f", kind = .EWM_Var, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultvat)
	w.destroy_column(&resultvat)

	// --- EWM STD (adjust=True) ---
	fmt.println("EWM STD (alpha=0.5, adjust=True)")
	rwstat := w.rolling_window(&df, "value_f", 3, 1)
	resultstat := w.rolling_apply_float_ewm_std(
		rwstat,
		w.Aggregator{name = "ewm_std_adj", column = "value_f", kind = .EWM_Std, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultstat)
	w.destroy_column(&resultstat)

	// --- EWM COV float–float (adjust=False) ---
	fmt.println("EWM COV float - float (alpha=0.5, adjust=False)")
	covf1 := w.column_new("cov_a", .Float, 0)
	covf2 := w.column_new("cov_b", .Float, 0)
	range := []f64{1.0, 2.0, 3.0, 4.0, 5.0}
	for v in range do w.append_float(&covf1, v)
	for v in range do w.append_float(&covf2, v)
	w.add_column(&df, covf1)
	w.add_column(&df, covf2)
	df.rows = 5

	rwcovf := w.rolling_window(&df, "cov_a", 3, 1)
	resultcovf := w.rolling_apply_float_ewm_cov_adjust_false(
		rwcovf,
		&covf2,
		w.Aggregator{name = "ewm_cov_f", column = "cov_a", kind = .EWM_Cov, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultcovf)
	w.destroy_column(&resultcovf)

	// --- EWM COV int–int (adjust=False) ---
	fmt.println("EWM COV int - int (alpha=0.5, adjust=False)")
	covi1 := w.column_new("cov_i_a", .Int, 0)
	covi2 := w.column_new("cov_i_b", .Int, 0)
	rangei := []int{1, 2, 3, 4, 5}
	for v in rangei do w.append_int(&covi1, v)
	for v in rangei do w.append_int(&covi2, v)
	w.add_column(&df, covi1)
	w.add_column(&df, covi2)
	df.rows = 5

	rwcovi := w.rolling_window(&df, "cov_i_a", 3, 1)
	resultcovi := w.rolling_apply_int_ewm_cov_adjust_false(
		rwcovi,
		&covi2,
		w.Aggregator{name = "ewm_cov_i", column = "cov_i_a", kind = .EWM_Cov, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultcovi)
	w.destroy_column(&resultcovi)

	// --- EWM COV float–float (adjust=True) ---
	fmt.println("EWM COV float - float (alpha=0.5, adjust=True)")
	rwcovfat := w.rolling_window(&df, "cov_a", 3, 1)
	resultcovfat := w.rolling_apply_float_ewm_cov(
		rwcovfat,
		&covf2,
		w.Aggregator{name = "ewm_cov_f_adj", column = "cov_a", kind = .EWM_Cov, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultcovfat)
	w.destroy_column(&resultcovfat)

	// --- EWM COV int–int (adjust=True) ---
	fmt.println("EWM COV int - int (alpha=0.5, adjust=True)")
	rwcoviat := w.rolling_window(&df, "cov_i_a", 3, 1)
	resultcoviat := w.rolling_apply_int_ewm_cov(
		rwcoviat,
		&covi2,
		w.Aggregator{name = "ewm_cov_i_adj", column = "cov_i_a", kind = .EWM_Cov, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultcoviat)
	w.destroy_column(&resultcoviat)


	defer w.destroy_column(&resultv)
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
