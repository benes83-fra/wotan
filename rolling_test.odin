package main


import w "./wotan/core"
import "core:fmt"
import "core:mem"


rolling_test :: proc(allocator: mem.Allocator) {
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
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


	fmt.println("CONST SERIES - VAR/COV (bias=false/true)")

	constf := w.column_new("const_f", .Float, 0)
	for _ in 0 ..< 5 do w.append_float(&constf, 7.0)
	w.add_column(&df, constf)
	df.rows = 5

	rw_const := w.rolling_window(&df, "const_f", 5, 1)

	res_const_unbiased := w.rolling_apply(
		rw_const,
		w.make_ewm_var("const_var_unbiased", "const_f", 0.5, false),
	)
	print_result_float(&res_const_unbiased)
	w.destroy_column(&res_const_unbiased)

	res_const_biased := w.rolling_apply(
		rw_const,
		w.make_ewm_var("const_var_biased", "const_f", 0.5, true),
	)
	print_result_float(&res_const_biased)

	fmt.println("ALTERNATING SERIES -  VAR (bias=false/true)")

	alt := w.column_new("alt", .Float, 0)
	rangeu := []f64{1, -1, 1, -1, 1}
	for v in rangeu do w.append_float(&alt, v)
	w.add_column(&df, alt)
	df.rows = 5

	rw_alt := w.rolling_window(&df, "alt", 5, 1)

	res_alt_unbiased := w.rolling_apply(
		rw_alt,
		w.make_ewm_var("alt_var_unbiased", "alt", 0.5, false),
	)
	print_result_float(&res_alt_unbiased)
	w.destroy_column(&res_alt_unbiased)

	res_alt_biased := w.rolling_apply(rw_alt, w.make_ewm_var("alt_var_biased", "alt", 0.5, true))
	print_result_float(&res_alt_biased)
	w.destroy_column(&res_alt_biased)
	fmt.println("INDEPENDENT SERIES -  COV (bias=false/true)")

	cov_x := w.column_new("cov_x", .Float, 0)
	cov_y := w.column_new("cov_y", .Float, 0)
	range_in := []f64{1, 2, 3, 4, 5}
	range_y := []f64{10, 20, 30, 40, 50}
	for v in range_in do w.append_float(&cov_x, v)
	for v in range_y do w.append_float(&cov_y, v)

	w.add_column(&df, cov_x)
	w.add_column(&df, cov_y)
	df.rows = 5

	rw_cov := w.rolling_window(&df, "cov_x", 5, 1)

	res_cov_unbiased := w.rolling_apply_float_ewm_cov(
		rw_cov,
		&cov_y,
		w.make_ewm_cov("cov_unbiased", "cov_x", 0.5, false),
		allocator,
	)
	print_result_float(&res_cov_unbiased)
	w.destroy_column(&res_cov_unbiased)

	res_cov_biased := w.rolling_apply_float_ewm_cov(
		rw_cov,
		&cov_y,
		w.make_ewm_cov("cov_biased", "cov_x", 0.5, true),
		allocator,
	)
	print_result_float(&res_cov_biased)

	fmt.println("PERFECT CORRELATION - COV (bias=false/true)")

	pc_x := w.column_new("pc_x", .Float, 0)
	pc_y := w.column_new("pc_y", .Float, 0)
	range_pc := []f64{1, 2, 3, 4, 5}
	for v in range_pc {
		w.append_float(&pc_x, v)
		w.append_float(&pc_y, v * 2.0)
	}

	w.add_column(&df, pc_x)
	w.add_column(&df, pc_y)
	df.rows = 5

	rw_pc := w.rolling_window(&df, "pc_x", 5, 1)

	res_pc_unbiased := w.rolling_apply_float_ewm_cov(
		rw_pc,
		&pc_y,
		w.make_ewm_cov("pc_unbiased", "pc_x", 0.5, false),
		allocator,
	)
	print_result_float(&res_pc_unbiased)

	res_pc_biased := w.rolling_apply_float_ewm_cov(
		rw_pc,
		&pc_y,
		w.make_ewm_cov("pc_biased", "pc_x", 0.5, true),
		allocator,
	)
	print_result_float(&res_pc_biased)

	// Reset DF for short-series test
	w.destroy_dataframe(&df)
	df = w.dataframe_new()

	fmt.println("SHORT SERIES - VAR/COV min_periods test")

	short := w.column_new("short", .Float, 0)
	range_short := []f64{1, 2}
	for v in range_short do w.append_float(&short, v)
	w.add_column(&df, short)
	df.rows = 2

	rw_short := w.rolling_window(&df, "short", 5, 3)

	res_short := w.rolling_apply(rw_short, w.make_ewm_var("short_var", "short", 0.5, false))
	print_result_float(&res_short)

	// --- EWM CORRELATION TESTS ---
	fmt.println("=== EWM CORRELATION TESTS ===")
	w.destroy_dataframe(&df)
	// Build DF fresh
	df = w.dataframe_new()
	range_corr_a_i := []int{1, 2, 3, 4, 5}
	range_corr_b_i := []int{2, 4, 6, 8, 10}
	range_corr_a_f := []f64{1, 2, 3, 4, 5}
	range_corr_b_f := []f64{2, 4, 6, 8, 10}
	// FLOAT columns: perfect correlation
	cov_a := w.column_new("cov_a", .Float, 0)
	cov_b := w.column_new("cov_b", .Float, 0)
	for v in range_corr_a_f do w.append_float(&cov_a, v)
	for v in range_corr_b_f do w.append_float(&cov_b, v)

	w.add_column(&df, cov_a)
	w.add_column(&df, cov_b)
	df.rows = 5

	fmt.println("EWM CORR float - float (alpha=0.5)")
	rw_corr_f := w.rolling_window(&df, "cov_a", 3, 1)
	res_corr_f := w.rolling_apply(
		rw_corr_f,
		w.make_ewm_corr("ewm_corr_f", "cov_a", "cov_b", 0.5, false),
		allocator,
	)
	print_result_float(&res_corr_f)
	w.destroy_column(&res_corr_f)
	w.destroy_dataframe(&df)

	// INT columns: perfect correlation
	df = w.dataframe_new()

	cov_i_a := w.column_new("cov_i_a", .Int, 0)
	cov_i_b := w.column_new("cov_i_b", .Int, 0)
	for v in range_corr_a_i do w.append_int(&cov_i_a, v)
	for v in range_corr_b_i do w.append_int(&cov_i_b, v)

	w.add_column(&df, cov_i_a)
	w.add_column(&df, cov_i_b)
	df.rows = 5

	fmt.println("EWM CORR int - int (alpha=0.5)")
	rw_corr_i := w.rolling_window(&df, "cov_i_a", 3, 1)
	res_corr_i := w.rolling_apply(
		rw_corr_i,
		w.make_ewm_corr("ewm_corr_i", "cov_i_a", "cov_i_b", 0.5, false),
		allocator,
	)
	print_result_float(&res_corr_i)
	defer w.destroy_column(&res_corr_i)
	w.destroy_dataframe(&df)
	// --- ROLLING CORRELATION TESTS ---
	fmt.println("=== ROLLING CORRELATION TESTS ===")

	// FLOAT perfect correlation
	df = w.dataframe_new()
	cov_a2 := w.column_new("cov_a", .Float, 0)
	cov_b2 := w.column_new("cov_b", .Float, 0)
	range_f_a := []f64{1, 2, 3, 4, 5}
	range_f_b := []f64{2, 4, 6, 8, 10}
	for v in range_f_a do w.append_float(&cov_a2, v)
	for v in range_f_b do w.append_float(&cov_b2, v)
	w.add_column(&df, cov_a2)
	w.add_column(&df, cov_b2)
	df.rows = 5

	fmt.println("ROLLING CORR float - float (perfect corr)")
	rw_corr_f2 := w.rolling_window(&df, "cov_a", 3, 1)
	res_corr_f2 := w.rolling_apply(
		rw_corr_f2,
		w.make_corr("roll_corr_f", "cov_a", "cov_b"),
		allocator,
	)
	print_result_float(&res_corr_f2)
	defer w.destroy_column(&res_corr_f2)
	w.destroy_dataframe(&df)


	// INT perfect correlation
	df = w.dataframe_new()
	cov_i_a2 := w.column_new("cov_i_a", .Int, 0)
	cov_i_b2 := w.column_new("cov_i_b", .Int, 0)
	range_i_a := []int{1, 2, 3, 4, 5}
	range_i_b := []int{2, 4, 6, 8, 10}
	for v in range_i_a do w.append_int(&cov_i_a2, v)
	for v in range_i_b do w.append_int(&cov_i_b2, v)
	w.add_column(&df, cov_i_a2)
	w.add_column(&df, cov_i_b2)
	df.rows = 5

	fmt.println("ROLLING CORR int - int (perfect corr)")
	rw_corr_i2 := w.rolling_window(&df, "cov_i_a", 3, 1)
	res_corr_i2 := w.rolling_apply(
		rw_corr_i2,
		w.make_corr("roll_corr_i", "cov_i_a", "cov_i_b"),
		allocator,
	)
	print_result_float(&res_corr_i2)
	defer w.destroy_column(&res_corr_i2)
	w.destroy_dataframe(&df)


	// FLOAT negative correlation
	df = w.dataframe_new()
	neg_a := w.column_new("neg_a", .Float, 0)
	neg_b := w.column_new("neg_b", .Float, 0)
	range_neg_a := []f64{1, 2, 3, 4, 5}
	range_neg_b := []f64{10, 8, 6, 4, 2}
	for v in range_neg_a do w.append_float(&neg_a, v)
	for v in range_neg_b do w.append_float(&neg_b, v)
	w.add_column(&df, neg_a)
	w.add_column(&df, neg_b)
	df.rows = 5

	fmt.println("ROLLING CORR float - float (negative corr)")
	rw_corr_neg := w.rolling_window(&df, "neg_a", 3, 1)
	res_corr_neg := w.rolling_apply(
		rw_corr_neg,
		w.make_corr("roll_corr_neg", "neg_a", "neg_b"),
		allocator,
	)
	print_result_float(&res_corr_neg)
	defer w.destroy_column(&res_corr_neg)
	w.destroy_dataframe(&df)


	// FLOAT zero correlation
	df = w.dataframe_new()
	zero_a := w.column_new("zero_a", .Float, 0)
	zero_b := w.column_new("zero_b", .Float, 0)
	range_zero_a := []f64{1, 2, 3, 4, 5}
	range_zero_b := []f64{5, 1, 5, 1, 5}
	for v in range_zero_a do w.append_float(&zero_a, v)
	for v in range_zero_b do w.append_float(&zero_b, v)
	w.add_column(&df, zero_a)
	w.add_column(&df, zero_b)
	df.rows = 5

	fmt.println("ROLLING CORR float - float (zero corr)")
	rw_corr_zero := w.rolling_window(&df, "zero_a", 3, 1)
	res_corr_zero := w.rolling_apply(
		rw_corr_zero,
		w.make_corr("roll_corr_zero", "zero_a", "zero_b"),
		allocator,
	)
	print_result_float(&res_corr_zero)
	defer w.destroy_column(&res_corr_zero)
	w.destroy_dataframe(&df)


	// SHORT SERIES (min_periods test)
	df = w.dataframe_new()
	short_a := w.column_new("short_a", .Float, 0)
	short_b := w.column_new("short_b", .Float, 0)
	range_short_a := []f64{1, 2}
	range_short_b := []f64{2, 4}
	for v in range_short_a do w.append_float(&short_a, v)
	for v in range_short_b do w.append_float(&short_b, v)
	w.add_column(&df, short_a)
	w.add_column(&df, short_b)
	df.rows = 2

	fmt.println("ROLLING CORR short series (min_periods=3)")
	rw_corr_short := w.rolling_window(&df, "short_a", 5, 3)
	res_corr_short := w.rolling_apply(
		rw_corr_short,
		w.make_corr("roll_corr_short", "short_a", "short_b"),
		allocator,
	)
	print_result_float(&res_corr_short)
	w.destroy_dataframe(&df)

	w.destroy_column(&res_corr_i)
	w.destroy_column(&res_corr_i2)

	w.destroy_dataframe(&df)
	w.destroy_column(&res_corr_short)

	w.destroy_column(&res_short)


	w.destroy_column(&res_pc_biased)

	w.destroy_column(&res_pc_unbiased)

	w.destroy_column(&res_cov_biased)


	w.destroy_column(&res_const_biased)


	w.destroy_column(&resultcoviat)


	defer w.destroy_column(&resultv)
	// w.destroy_dataframe(&df)
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
