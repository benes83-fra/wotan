package tests


import analytic "../wotan/analytics"
import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

rolling_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
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
	rw := analytic.rolling_window(&df, "value", 3, 1)
	result := analytic.rolling_apply(
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
	rwf := analytic.rolling_window(&df, "value_f", 3, 1)
	resultf := analytic.rolling_apply(
		rwf,
		w.Aggregator{name = "rolling_mean", column = "value_f", kind = .Mean},
	)
	defer w.destroy_column(&resultf)
	print_result_float(&resultf)

	rwm := analytic.rolling_window(&df, "value_f", 3, 1)
	resultm := analytic.rolling_apply(rwm, w.make_median_agg("rolling_median", "value_f"))
	defer w.destroy_column(&resultm)
	print_result_float(&resultm)
	rwq := analytic.rolling_window(&df, "value_f", 3, 1)
	resultq := analytic.rolling_apply(rwq, w.make_quantile_agg("q25", "value_f", 0.25))
	defer w.destroy_column(&resultq)


	print_result_float(&resultq)
	rwe := analytic.rolling_window(&df, "value_f", 3, 1)
	resulte := analytic.rolling_apply(rwe, analytic.make_ewm_mean("ewm", "value_f", 0.5))
	print_result_float(&resulte)
	defer w.destroy_column(&resulte)
	rwev := analytic.rolling_window(&df, "value_f", 3, 1)
	resultv := analytic.rolling_apply(rwev, analytic.make_ewm_var("ewm_var", "value_f", 0.5))


	print_result_float(&resultv)

	// --- EWM STD (adjust=False) ---
	fmt.println("EWM STD (alpha=0.5, adjust=False)")
	rws := analytic.rolling_window(&df, "value_f", 3, 1)
	results := analytic.rolling_apply(
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
	rwvat := analytic.rolling_window(&df, "value_f", 3, 1)
	resultvat := analytic.rolling_apply_float_ewm_var_adjust_true(
		rwvat,
		w.Aggregator{name = "ewm_var_adj", column = "value_f", kind = .EWM_Var, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultvat)
	w.destroy_column(&resultvat)

	// --- EWM STD (adjust=True) ---
	fmt.println("EWM STD (alpha=0.5, adjust=True)")
	rwstat := analytic.rolling_window(&df, "value_f", 3, 1)
	resultstat := analytic.ewm_std_column(rwstat, 0.5, true, false, name = "ewm_std_adj")
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

	rwcovf := analytic.rolling_window(&df, "cov_a", 3, 1)
	resultcovf := analytic.rolling_apply_float_ewm_cov_adjust_false(
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

	rwcovi := analytic.rolling_window(&df, "cov_i_a", 3, 1)
	resultcovi := analytic.rolling_apply_int_ewm_cov_adjust_false(
		rwcovi,
		&covi2,
		w.Aggregator{name = "ewm_cov_i", column = "cov_i_a", kind = .EWM_Cov, alpha = 0.5},
		allocator,
	)
	print_result_float(&resultcovi)
	w.destroy_column(&resultcovi)

	// --- EWM COV float–float (adjust=True) ---
	fmt.println("EWM COV float - float (alpha=0.5, adjust=True)")
	rwcovfat := analytic.rolling_window(&df, "cov_a", 3, 1)
	resultcovfat := analytic.ewm_cov_column(rwcovfat, &covf2, 0.5, true, false, name = "ewm_cov_f_adj")
	print_result_float(&resultcovfat)
	w.destroy_column(&resultcovfat)

	// --- EWM COV int–int (adjust=True) ---
	fmt.println("EWM COV int - int (alpha=0.5, adjust=True)")
	rwcoviat := analytic.rolling_window(&df, "cov_i_a", 3, 1)
	resultcoviat := analytic.ewm_cov_column(rwcoviat, &covi2, 0.5, true, false, name = "ewm_cov_i_adj")
	print_result_float(&resultcoviat)


	fmt.println("CONST SERIES - VAR/COV (bias=false/true)")

	constf := w.column_new("const_f", .Float, 0)
	for _ in 0 ..< 5 do w.append_float(&constf, 7.0)
	w.add_column(&df, constf)
	df.rows = 5

	rw_const := analytic.rolling_window(&df, "const_f", 5, 1)

	res_const_unbiased := analytic.rolling_apply(
		rw_const,
		analytic.make_ewm_var("const_var_unbiased", "const_f", 0.5, false),
	)
	print_result_float(&res_const_unbiased)
	w.destroy_column(&res_const_unbiased)

	res_const_biased := analytic.rolling_apply(
		rw_const,
		analytic.make_ewm_var("const_var_biased", "const_f", 0.5, true),
	)
	print_result_float(&res_const_biased)

	fmt.println("ALTERNATING SERIES -  VAR (bias=false/true)")

	alt := w.column_new("alt", .Float, 0)
	rangeu := []f64{1, -1, 1, -1, 1}
	for v in rangeu do w.append_float(&alt, v)
	w.add_column(&df, alt)
	df.rows = 5

	rw_alt := analytic.rolling_window(&df, "alt", 5, 1)

	res_alt_unbiased := analytic.rolling_apply(
		rw_alt,
		analytic.make_ewm_var("alt_var_unbiased", "alt", 0.5, false),
	)
	print_result_float(&res_alt_unbiased)
	w.destroy_column(&res_alt_unbiased)

	res_alt_biased := analytic.rolling_apply(
		rw_alt,
		analytic.make_ewm_var("alt_var_biased", "alt", 0.5, true),
	)
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

	rw_cov := analytic.rolling_window(&df, "cov_x", 5, 1)

	res_cov_unbiased := analytic.rolling_apply_float_ewm_cov(
		rw_cov,
		&cov_y,
		analytic.make_ewm_cov("cov_unbiased", "cov_x", 0.5, false),
		allocator,
	)
	print_result_float(&res_cov_unbiased)
	w.destroy_column(&res_cov_unbiased)

	res_cov_biased := analytic.rolling_apply_float_ewm_cov(
		rw_cov,
		&cov_y,
		analytic.make_ewm_cov("cov_biased", "cov_x", 0.5, true),
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

	rw_pc := analytic.rolling_window(&df, "pc_x", 5, 1)

	res_pc_unbiased := analytic.rolling_apply_float_ewm_cov(
		rw_pc,
		&pc_y,
		analytic.make_ewm_cov("pc_unbiased", "pc_x", 0.5, false),
		allocator,
	)
	print_result_float(&res_pc_unbiased)

	res_pc_biased := analytic.rolling_apply_float_ewm_cov(
		rw_pc,
		&pc_y,
		analytic.make_ewm_cov("pc_biased", "pc_x", 0.5, true),
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

	rw_short := analytic.rolling_window(&df, "short", 5, 3)

	res_short := analytic.rolling_apply(
		rw_short,
		analytic.make_ewm_var("short_var", "short", 0.5, false),
	)
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
	rw_corr_f := analytic.rolling_window(&df, "cov_a", 3, 1)
	res_corr_f := analytic.rolling_apply(
		rw_corr_f,
		analytic.make_ewm_corr("ewm_corr_f", "cov_a", "cov_b", 0.5, false),
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
	rw_corr_i := analytic.rolling_window(&df, "cov_i_a", 3, 1)
	res_corr_i := analytic.rolling_apply(
		rw_corr_i,
		analytic.make_ewm_corr("ewm_corr_i", "cov_i_a", "cov_i_b", 0.5, false),
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
	rw_corr_f2 := analytic.rolling_window(&df, "cov_a", 3, 1)
	res_corr_f2 := analytic.rolling_apply(
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
	rw_corr_i2 := analytic.rolling_window(&df, "cov_i_a", 3, 1)
	res_corr_i2 := analytic.rolling_apply(
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
	rw_corr_neg := analytic.rolling_window(&df, "neg_a", 3, 1)
	res_corr_neg := analytic.rolling_apply(
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
	rw_corr_zero := analytic.rolling_window(&df, "zero_a", 3, 1)
	res_corr_zero := analytic.rolling_apply(
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
	rw_corr_short := analytic.rolling_window(&df, "short_a", 5, 3)
	res_corr_short := analytic.rolling_apply(
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

	rolling_pandas_reference_checks()
}

// Regression checks against pandas 1.5.3 (values computed with
// Series.rolling / Series.ewm on the same data, NaN = null):
//   x = [1, 2, NaN, 4, 5, 3, NaN, NaN, 8, 2.5], y = [2, 1, 3, NaN, 6, 4, 5, 1, 7, 3]
//   xi = Int [3, 1, 4, 1, 5, null, 9, 2, 6, 5], b = Bool [T, F, T, T, F, T, T, F, F, T]
rolling_pandas_reference_checks :: proc() {
	fmt.println("\n=== Rolling / EWM vs pandas ===")
	passed, total := 0, 0
	N := math.nan_f64()

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	xs := []f64{1, 2, N, 4, 5, 3, N, N, 8, 2.5}
	ys := []f64{2, 1, 3, N, 6, 4, 5, 1, 7, 3}
	xis := []f64{3, 1, 4, 1, 5, N, 9, 2, 6, 5}
	bs := []bool{true, false, true, true, false, true, true, false, false, true}
	cx := w.column_new("x", .Float, 0)
	cy := w.column_new("y", .Float, 0)
	cxi := w.column_new("xi", .Int, 0)
	cb := w.column_new("b", .Bool, 0)
	for i in 0 ..< len(xs) {
		if math.is_nan(xs[i]) {w.append_null(&cx)} else {w.append_float(&cx, xs[i])}
		if math.is_nan(ys[i]) {w.append_null(&cy)} else {w.append_float(&cy, ys[i])}
		if math.is_nan(xis[i]) {w.append_null(&cxi)} else {w.append_int(&cxi, int(xis[i]))}
		w.append_bool(&cb, bs[i])
	}
	w.add_column(&df, cx)
	w.add_column(&df, cy)
	w.add_column(&df, cxi)
	w.add_column(&df, cb)
	df.rows = len(xs)

	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	// compares (and frees) a result column; NaN in `want` means null
	check :: proc(passed, total: ^int, c: w.Column, want: []f64, name: string) {
		c := c
		defer w.destroy_column(&c)
		ok := c.len == len(want)
		for i in 0 ..< len(want) {
			if !ok do break
			v: f64
			is_null: bool
			if c.type == .Int {
				iv, n := w.column_at_int(&c, i)
				v, is_null = f64(iv), n
			} else {
				v, is_null = w.column_at_float(&c, i)
			}
			if math.is_nan(want[i]) {
				ok = is_null
			} else {
				ok = !is_null && abs(v - want[i]) <= 1e-9 * max(1, abs(want[i]))
			}
		}
		expect(passed, total, ok, name)
	}
	A :: proc(kind: w.AggregationKind) -> w.Aggregator {return w.Aggregator{name = "r", kind = kind}}

	// rolling(3, min_periods=2)
	r3 := analytic.rolling_window(&df, "x", 3, 2)
	check(&passed, &total, analytic.rolling_apply(r3, A(.Sum)), {N, 3.0, 3.0, 6.0, 9.0, 12.0, 8.0, N, N, 10.5}, "rolling(3, 2).sum: window counts rows, nulls skipped")
	check(&passed, &total, analytic.rolling_apply(r3, A(.Mean)), {N, 1.5, 1.5, 3.0, 4.5, 4.0, 4.0, N, N, 5.25}, "rolling(3, 2).mean")
	check(&passed, &total, analytic.rolling_apply(r3, w.make_median_agg("r", "x")), {N, 1.5, 1.5, 3.0, 4.5, 4.0, 4.0, N, N, 5.25}, "rolling(3, 2).median (interpolated)")
	check(&passed, &total, analytic.rolling_apply(r3, w.make_quantile_agg("r", "x", 0.3)), {N, 1.3, 1.3, 2.6, 4.3, 3.6, 3.6, N, N, 4.15}, "rolling(3, 2).quantile(0.3), linear")
	check(&passed, &total, analytic.rolling_apply(r3, A(.Count)), {N, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0}, "rolling(3, 2).count: min_periods vs rows")
	check(&passed, &total, analytic.rolling_apply(r3, w.make_var("r", "x")), {N, 0.5, 0.5, 2.0, 0.5, 1.0, 2.0, N, N, 15.125}, "rolling(3, 2).var, ddof=1")
	check(&passed, &total, analytic.rolling_apply(r3, w.make_cov("r", "x", "y")), {N, -0.5, -0.5, N, N, 2.0, 2.0, N, N, 11.0}, "rolling(3, 2).cov: pairwise, ddof=1")
	check(&passed, &total, analytic.rolling_apply(r3, w.make_corr("r", "x", "y")), {N, -1.0, -1.0, N, N, 1.0, 1.0, N, N, 1.0}, "rolling(3, 2).corr: pairwise")

	// rolling(4, min_periods=2, center=True)
	c4 := analytic.rolling_window(&df, "x", 4, 2)
	c4.centered = true
	check(&passed, &total, analytic.rolling_apply(c4, A(.Mean)), {1.5, 1.5, 2.3333333333333335, 3.6666666666666665, 4.0, 4.0, 4.0, 5.5, 5.25, 5.25}, "rolling(4, 2, center=True).mean")
	check(&passed, &total, analytic.rolling_apply(c4, w.make_var("r", "x")), {0.5, 0.5, 2.333333333333333, 2.333333333333333, 1.0, 1.0, 2.0, 12.5, 15.125, 15.125}, "rolling(4, 2, center=True).var")

	// Int and Bool columns
	ri := analytic.rolling_window(&df, "xi", 3, 2)
	imean := analytic.rolling_apply(ri, A(.Mean))
	expect(&passed, &total, imean.type == .Float, "rolling Int mean returns a Float column")
	check(&passed, &total, imean, {N, 2.0, 2.6666666666666665, 2.0, 3.3333333333333335, 3.0, 7.0, 5.5, 5.666666666666667, 4.333333333333333}, "rolling(3, 2).mean of an Int column")
	check(&passed, &total, analytic.rolling_apply(ri, w.make_median_agg("r", "xi")), {N, 2.0, 3.0, 1.0, 4.0, 3.0, 7.0, 5.5, 6.0, 5.0}, "rolling(3, 2).median of an Int column")
	rb := analytic.rolling_window(&df, "b", 3, 1)
	check(&passed, &total, analytic.rolling_apply(rb, A(.Sum)), {1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0}, "rolling(3, 1).sum of a Bool column counts trues")

	// ewm(alpha=0.3, min_periods=2): rolling_apply is adjust=False, ignore_na=False
	e2 := analytic.rolling_window(&df, "x", 0, 2)
	EA :: proc(kind: w.AggregationKind, bias := false) -> w.Aggregator {
		return w.Aggregator{name = "e", other = "y", kind = kind, alpha = 0.3, bias = bias}
	}
	check(&passed, &total, analytic.rolling_apply(e2, EA(.EWM_Mean)), {N, 1.2999999999999998, 1.2999999999999998, 2.3253164556962025, 3.1277215189873417, 3.089405063291139, 3.089405063291139, 3.089405063291139, 5.38050690001378, 4.5163548300096465}, "ewm(adjust=False).mean")
	check(&passed, &total, analytic.rolling_apply(e2, EA(.EWM_Var)), {N, 0.5, 0.5, 2.919955323901713, 3.8293831107472784, 2.52056591771769, 2.52056591771769, 2.5205659177176902, 9.800527043601834, 8.643170725290451}, "ewm(adjust=False).var()")
	check(&passed, &total, analytic.rolling_apply(e2, EA(.EWM_Var, true)), {N, 0.21000000000000002, 0.21000000000000002, 1.8473337606152862, 2.7954593654863, 1.9602472409870213, 1.9602472409870213, 1.9602472409870213, 7.04719388758847, 6.675472921527598}, "ewm(adjust=False).var(bias=True)")
	check(&passed, &total, analytic.rolling_apply(e2, EA(.EWM_Std)), {N, 0.7071067811865476, 0.7071067811865476, 1.7087876766590147, 1.9568809648896068, 1.587629023959215, 1.587629023959215, 1.5876290239592152, 3.130579346319437, 2.939926993190554}, "ewm(adjust=False).std()")
	check(&passed, &total, analytic.rolling_apply(e2, EA(.EWM_Cov)), {N, -0.4999999999999999, -0.4999999999999999, -0.5, 6.233327957422791, 3.7257983480554855, 3.7257983480554855, 3.725798348055486, 7.692301571844134, 6.740767662921542}, "ewm(adjust=False).cov(y)")
	check(&passed, &total, analytic.rolling_apply(e2, EA(.EWM_Corr)), {N, -0.9999999999999999, -0.9999999999999999, -0.9999999999999999, 0.94468997391713, 0.9415112567866573, 0.9415112567866573, 0.9415112567866573, 0.9502884912026933, 0.9626753784080091}, "ewm(adjust=False).corr(y)")
	ei := analytic.rolling_window(&df, "xi", 0, 2)
	check(&passed, &total, analytic.rolling_apply(ei, EA(.EWM_Var)), {N, 2.0, 1.7986577181208059, 2.1057742327033115, 3.329547465591918, 3.329547465591918, 12.65918307218241, 11.518881871317657, 8.636658410072428, 6.003150525898552}, "ewm(adjust=False).var() of an Int column")

	// adjust=True through the explicit API
	y_col := w.column(&df, "y")
	check(&passed, &total, analytic.ewm_mean_column(e2, 0.3, true), {N, 1.5882352941176472, 1.5882352941176472, 2.903982542280415, 3.822040208488459, 3.5056482062374674, 3.5056482062374674, 3.5056482062374674, 5.882138053300919, 4.426716740416973}, "ewm(adjust=True).mean")
	check(&passed, &total, analytic.ewm_mean_column(e2, 0.3, true, ignore_na = true), {N, 1.5882352941176472, 1.5882352941176472, 2.6894977168949774, 3.6016581129095937, 3.384695827773971, 3.384695827773971, 3.384695827773971, 4.953902698585937, 4.151664089964133}, "ewm(adjust=True, ignore_na=True).mean")
	check(&passed, &total, analytic.ewm_var_column(e2, 0.3, true, false), {N, 0.5, 0.49999999999999994, 2.6047479197258934, 2.8709530817772393, 1.8629136805778475, 1.8629136805778475, 1.862913680577847, 8.589516515135355, 8.566765552054225}, "ewm(adjust=True).var()")
	check(&passed, &total, analytic.rolling_apply_float_ewm_var_adjust_true(e2, EA(.EWM_Var), context.allocator), {N, 0.5, 0.49999999999999994, 2.6047479197258934, 2.8709530817772393, 1.8629136805778475, 1.8629136805778475, 1.862913680577847, 8.589516515135355, 8.566765552054225}, "rolling_apply_float_ewm_var_adjust_true")
	check(&passed, &total, analytic.ewm_cov_column(e2, y_col, 0.3, true, true), {N, -0.2422145328719723, -0.2422145328719723, -0.2422145328719723, 3.552879167004977, 1.9250511772911343, 1.9250511772911343, 1.9250511772911343, 3.9925814623786535, 4.665391220310252}, "ewm(adjust=True).cov(y, bias=True)")
	check(&passed, &total, analytic.ewm_corr_column(e2, y_col, 0.3, true), {N, -1.0, -1.0, -1.0, 0.9511997490693292, 0.9332722310150257, 0.9332722310150257, 0.9332722310150257, 0.9421320321459189, 0.9712117975142762}, "ewm(adjust=True).corr(y)")

	fmt.printf("Rolling vs pandas summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
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
