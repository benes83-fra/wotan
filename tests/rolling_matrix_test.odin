package tests

import analytic "../wotan/analytics"
import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

rolling_matrix_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== ROLLING MATRIX TESTS ===")

	// ------------------------------------------------------------
	// 1. PERFECT CORRELATION (float)
	// ------------------------------------------------------------
	df := w.dataframe_new()

	a := w.column_new("a", .Float, 0)
	b := w.column_new("b", .Float, 0)
	range := []f64{1, 2, 3, 4, 5}
	range2 := []f64{2, 4, 6, 8, 10}
	for v in range do w.append_float(&a, v)
	for v in range2 do w.append_float(&b, v)


	w.add_column(&df, a)
	w.add_column(&df, b)
	df.rows = 5

	fmt.println("Rolling CORR matrix (perfect corr)")
	corr_mat := analytic.rolling_corr_matrix(&df, []string{"a", "b"}, 3, 1, allocator)
	w.dataframe_pretty_print(&corr_mat)
	w.destroy_dataframe(&corr_mat)
	// ------------------------------------------------------------
	// 2. NEGATIVE CORRELATION
	// ------------------------------------------------------------
	df2 := w.dataframe_new()

	na := w.column_new("na", .Float, 0)
	nb := w.column_new("nb", .Float, 0)
	range3 := []f64{10, 8, 6, 4, 2}
	for v in range do w.append_float(&na, v)
	for v in range3 do w.append_float(&nb, v)

	w.add_column(&df2, na)
	w.add_column(&df2, nb)
	df2.rows = 5

	fmt.println("Rolling CORR matrix (negative corr)")
	corr_neg := analytic.rolling_corr_matrix(&df2, []string{"na", "nb"}, 3, 1, allocator)
	w.dataframe_pretty_print(&corr_neg)
	w.destroy_dataframe(&corr_neg)

	// ------------------------------------------------------------
	// 3. ZERO CORRELATION
	// ------------------------------------------------------------
	df3 := w.dataframe_new()

	za := w.column_new("za", .Float, 0)
	zb := w.column_new("zb", .Float, 0)
	range4 := []f64{5, 1, 5, 1, 5}
	for v in range do w.append_float(&za, v)
	for v in range4 do w.append_float(&zb, v)

	w.add_column(&df3, za)
	w.add_column(&df3, zb)
	df3.rows = 5

	fmt.println("Rolling CORR matrix (zero corr)")
	corr_zero := analytic.rolling_corr_matrix(&df3, []string{"za", "zb"}, 3, 1, allocator)
	w.dataframe_pretty_print(&corr_zero)
	w.destroy_dataframe(&corr_zero)

	// ------------------------------------------------------------
	// 4. SHORT SERIES (min_periods)
	// ------------------------------------------------------------
	df4 := w.dataframe_new()

	sa := w.column_new("sa", .Float, 0)
	sb := w.column_new("sb", .Float, 0)
	short_range := []f64{1, 2}
	short_range2 := []f64{2, 4}
	for v in short_range do w.append_float(&sa, v)
	for v in short_range2 do w.append_float(&sb, v)

	w.add_column(&df4, sa)
	w.add_column(&df4, sb)
	df4.rows = 2

	fmt.println("Rolling CORR matrix (short series, min_periods=3)")
	corr_short := analytic.rolling_corr_matrix(&df4, []string{"sa", "sb"}, 5, 3, allocator)
	w.dataframe_pretty_print(&corr_short)
	w.destroy_dataframe(&corr_short)

	// ------------------------------------------------------------
	// 5. VARIANCE MATRIX
	// ------------------------------------------------------------
	fmt.println("Rolling VAR matrix")

	df5 := w.dataframe_new()
	va := w.column_new("va", .Float, 0)
	vb := w.column_new("vb", .Float, 0)
	range5 := []f64{10, 20, 30, 40, 50}
	for v in range do w.append_float(&va, v)
	for v in range5 do w.append_float(&vb, v)

	w.add_column(&df5, va)
	w.add_column(&df5, vb)
	df5.rows = 5

	var_mat := analytic.rolling_var_matrix(&df5, []string{"va", "vb"}, 3, 1, allocator)
	w.dataframe_pretty_print(&var_mat)
	w.destroy_dataframe(&var_mat)

	// ------------------------------------------------------------
	// 6. COVARIANCE MATRIX
	// ------------------------------------------------------------
	fmt.println("Rolling COV matrix")

	cov_mat := analytic.rolling_cov_matrix(&df5, []string{"va", "vb"}, 3, 1, allocator)
	w.dataframe_pretty_print(&cov_mat)
	w.destroy_dataframe(&cov_mat)

	// Cleanup
	w.destroy_dataframe(&df)
	w.destroy_dataframe(&df2)
	w.destroy_dataframe(&df3)
	w.destroy_dataframe(&df4)
	w.destroy_dataframe(&df5)

	rolling_matrix_pandas_reference_checks(allocator)
}


ewm_cov_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	x := w.column_from_f64("x", []f64{1, 2, 3, 4})
	y := w.column_from_f64("y", []f64{2, 4, 6, 8})
	z := w.column_from_f64("z", []f64{1, 0, 1, 0})

	w.add_column(&df, x)
	w.add_column(&df, y)
	w.add_column(&df, z)
	df.rows = 4

	alpha := 0.5
	minp := 1
	bias := false
	adjust := false

	ewm_cov_df := analytic.ewm_cov_matrix(
		&df,
		[]string{"x", "y", "z"},
		alpha,
		minp,
		bias,
		adjust,
		allocator,
	)
	fmt.println("EWM Cov Matrix:")
	w.dataframe_pretty_print(&ewm_cov_df) // or your own printer

	// pandas: df.ewm(alpha=0.5, adjust=False).cov().loc[3]
	passed, total := 0, 0
	got := analytic.extract_cov_matrix_row(&ewm_cov_df, []string{"x", "y", "z"}, 3, allocator)
	want := [3][3]f64 {
		{1.6904761904761905, 3.380952380952381, -0.45238095238095233},
		{3.380952380952381, 6.761904761904762, -0.9047619047619047},
		{-0.45238095238095233, -0.9047619047619047, 0.3571428571428571},
	}
	ok := true
	for i in 0 ..< 3 do for j in 0 ..< 3 do if abs(got[i][j] - want[i][j]) > 1e-12 do ok = false
	expect(&passed, &total, ok, "ewm_cov_matrix(alpha=0.5, adjust=False) row 3 matches pandas")
	expect(&passed, &total, ewm_cov_df.rows == 9 * df.rows, "ewm_cov_matrix has all 9 (i, j) pairs per row")
	record_failures(total - passed)
	w.destroy_dataframe(&ewm_cov_df)
}


ewm_pca_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== EWM PCA Test ===")

	// Build a tiny test DataFrame
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	x := w.column_from_f64("x", []f64{1, 2, 3, 4})
	y := w.column_from_f64("y", []f64{2, 4, 6, 8})
	z := w.column_from_f64("z", []f64{1, 0, 1, 0})

	w.add_column(&df, x)
	w.add_column(&df, y)
	w.add_column(&df, z)
	df.rows = 4

	alpha := 0.5
	minp := 1
	bias := false
	adjust := false

	// --- EWM Covariance Matrix ---
	fmt.println("\nEWM Covariance Matrix:")
	cov_df := analytic.ewm_cov_matrix(
		&df,
		[]string{"x", "y", "z"},
		alpha,
		minp,
		bias,
		adjust,
		allocator,
	)
	defer w.destroy_dataframe(&cov_df)
	w.dataframe_pretty_print(&cov_df)

	// --- EWM PCA (full time series) ---
	fmt.println("\nEWM PCA (all rows):")
	pca_series := analytic.ewm_pca(
		&df,
		[]string{"x", "y", "z"},
		alpha,
		minp,
		bias,
		adjust,
		allocator,
	)
	for i in 0 ..< len(pca_series) {
		fmt.printf("Row %d:\n", i)
		fmt.println("  Eigenvalues:  ", pca_series[i].eigenvalues)
		fmt.println("  Eigenvectors: ", pca_series[i].eigenvectors)

	}

	// --- EWM PCA (last row only) ---
	fmt.println("\nEWM PCA (last row):")
	last := analytic.ewm_pca_last(
		&df,
		[]string{"x", "y", "z"},
		alpha,
		minp,
		bias,
		adjust,
		allocator,
	)
	fmt.println("Eigenvalues:", last.eigenvalues)
	fmt.println("Eigenvectors:", last.eigenvectors)

	// numpy.linalg.eigh of pandas' df.ewm(alpha=0.5, adjust=False).cov() at rows 3 and 1
	passed, total := 0, 0
	want_last := []f64{8.576867198859116, 0.23265661066468918, 0}
	want_row1 := []f64{3, 0, 0}
	ok_last := len(last.eigenvalues) == 3
	ok_row1 := len(pca_series) == 4 && len(pca_series[1].eigenvalues) == 3
	for i in 0 ..< 3 {
		if ok_last && abs(last.eigenvalues[i] - want_last[i]) > 1e-9 do ok_last = false
		if ok_row1 && abs(pca_series[1].eigenvalues[i] - want_row1[i]) > 1e-9 do ok_row1 = false
	}
	expect(&passed, &total, ok_last, "ewm_pca_last eigenvalues match numpy eigh of the pandas ewm cov")
	expect(&passed, &total, ok_row1, "ewm_pca row 1 eigenvalues match numpy eigh of the pandas ewm cov")
	record_failures(total - passed)


	fmt.println("\n=== END EWM PCA TEST ===")
}


// Reference checks for the matrix procs (pandas 1.5.3 / numpy 1.23):
//   x = [1, 2, NaN, 4, 5, 3, NaN, NaN, 8, 2.5], y = [2, 1, 3, NaN, 6, 4, 5, 1, 7, 3]
rolling_matrix_pandas_reference_checks :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Rolling / EWM matrices vs pandas ===")
	passed, total := 0, 0
	N := math.nan_f64()
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	xs := []f64{1, 2, N, 4, 5, 3, N, N, 8, 2.5}
	ys := []f64{2, 1, 3, N, 6, 4, 5, 1, 7, 3}
	cx := w.column_new("x", .Float, 0)
	cy := w.column_new("y", .Float, 0)
	for i in 0 ..< len(xs) {
		if math.is_nan(xs[i]) {w.append_null(&cx)} else {w.append_float(&cx, xs[i])}
		if math.is_nan(ys[i]) {w.append_null(&cy)} else {w.append_float(&cy, ys[i])}
	}
	w.add_column(&df, cx)
	w.add_column(&df, cy)
	df.rows = len(xs)
	cols := []string{"x", "y"}

	rcov := analytic.rolling_cov_matrix(&df, cols, 4, 2, allocator)
	defer w.destroy_dataframe(&rcov)
	expect_near4(&passed, &total, _matrix_row(&rcov, 5, "cov"), {1.0000000000000018, 2.0, 2.0, 2.3333333333333357}, "rolling_cov_matrix(4, 2) row 5 (ddof=1, pairwise)")
	rcorr := analytic.rolling_corr_matrix(&df, cols, 4, 2, allocator)
	defer w.destroy_dataframe(&rcorr)
	expect_near4(&passed, &total, _matrix_row(&rcorr, 5, "corr"), {1, 1, 1, 1}, "rolling_corr_matrix(4, 2) row 5")
	rvar := analytic.rolling_var_matrix(&df, cols, 4, 2, allocator)
	defer w.destroy_dataframe(&rvar)
	v5 := _var_row(&rvar, 5)
	expect_near4(&passed, &total, {v5[0], 0, 0, v5[1]}, {1.0, 0, 0, 2.333333333333334}, "rolling_var_matrix(4, 2) row 5")

	for adjust in ([]bool{false, true}) {
		m := analytic.ewm_cov_matrix(&df, cols, 0.3, 1, false, adjust, allocator)
		defer w.destroy_dataframe(&m)
		expect(&passed, &total, m.rows == 4 * df.rows, "ewm_cov_matrix emits the full matrix (both triangles)")
		want9 := [4]f64{8.643170725290451, 6.740767662921542, 6.740767662921542, 5.238446784044843}
		want2 := [4]f64{0.5, -0.4999999999999999, -0.4999999999999999, 0.802013422818792}
		eig := [2]f64{13.89321732756607, -0.011599818230776826}
		if adjust {
			want9 = {8.566765552054225, 6.9096074186211265, 6.9096074186211265, 5.3987050994389465}
			want2 = {0.49999999999999994, -0.49999999999999994, -0.49999999999999994, 1.1849315068493147}
			eig = {14.071587590910555, -0.10611693941738354}
		}
		expect_near4(&passed, &total, _matrix_row(&m, 9, "cov"), want9, fmt.tprintf("ewm_cov_matrix(alpha=0.3, adjust=%v) row 9", adjust))
		expect_near4(&passed, &total, _matrix_row(&m, 2, "cov"), want2, fmt.tprintf("ewm_cov_matrix(alpha=0.3, adjust=%v) row 2 (null row)", adjust))
		last := analytic.ewm_pca_last(&df, cols, 0.3, 1, false, adjust, allocator)
		ok := len(last.eigenvalues) == 2
		if ok {
			ok = abs(last.eigenvalues[0] - eig[0]) < 1e-9 && abs(last.eigenvalues[1] - eig[1]) < 1e-9
		}
		expect(&passed, &total, ok, fmt.tprintf("ewm_pca_last eigenvalues = eigh(ewm cov), adjust=%v", adjust))
	}

	fmt.printf("Rolling matrix vs pandas summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

@(private = "file")
expect :: proc(passed, total: ^int, ok: bool, name: string) {
	total^ += 1
	if ok {passed^ += 1}
	fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
}

@(private = "file")
expect_near4 :: proc(passed, total: ^int, got, want: [4]f64, name: string) {
	ok := true
	for i in 0 ..< 4 {
		if math.is_nan(got[i]) || abs(got[i] - want[i]) > 1e-9 * max(1, abs(want[i])) do ok = false
	}
	if !ok do fmt.println("  got", got, "want", want)
	expect(passed, total, ok, name)
}

// [xx, xy, yx, yy] at `row` of a long-form (row, col_i, col_j, value) matrix over {x, y}
@(private = "file")
_matrix_row :: proc(m: ^w.DataFrame, row: int, value_col: string) -> [4]f64 {
	out := [4]f64{math.nan_f64(), math.nan_f64(), math.nan_f64(), math.nan_f64()}
	for k in 0 ..< m.rows {
		r, _ := w.column_at_int(w.column(m, "row"), k)
		if r != row do continue
		ci, _ := w.column_at_string(w.column(m, "col_i"), k)
		cj, _ := w.column_at_string(w.column(m, "col_j"), k)
		v, is_null := w.column_at_float(w.column(m, value_col), k)
		if is_null do continue
		idx := (0 if ci == "x" else 2) + (0 if cj == "x" else 1)
		out[idx] = v
	}
	return out
}

@(private = "file")
_var_row :: proc(m: ^w.DataFrame, row: int) -> [2]f64 {
	out := [2]f64{math.nan_f64(), math.nan_f64()}
	for k in 0 ..< m.rows {
		r, _ := w.column_at_int(w.column(m, "row"), k)
		if r != row do continue
		c, _ := w.column_at_string(w.column(m, "col"), k)
		v, is_null := w.column_at_float(w.column(m, "var"), k)
		if !is_null do out[0 if c == "x" else 1] = v
	}
	return out
}
