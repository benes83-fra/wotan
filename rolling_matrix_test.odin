package main

import w "./wotan/core"
import "core:fmt"
import "core:mem"

rolling_matrix_test :: proc(allocator: mem.Allocator) {
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
	corr_mat := w.rolling_corr_matrix(&df, []string{"a", "b"}, 3, 1, allocator)
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
	corr_neg := w.rolling_corr_matrix(&df2, []string{"na", "nb"}, 3, 1, allocator)
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
	corr_zero := w.rolling_corr_matrix(&df3, []string{"za", "zb"}, 3, 1, allocator)
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
	corr_short := w.rolling_corr_matrix(&df4, []string{"sa", "sb"}, 5, 3, allocator)
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

	var_mat := w.rolling_var_matrix(&df5, []string{"va", "vb"}, 3, 1, allocator)
	w.dataframe_pretty_print(&var_mat)
	w.destroy_dataframe(&var_mat)

	// ------------------------------------------------------------
	// 6. COVARIANCE MATRIX
	// ------------------------------------------------------------
	fmt.println("Rolling COV matrix")

	cov_mat := w.rolling_cov_matrix(&df5, []string{"va", "vb"}, 3, 1, allocator)
	w.dataframe_pretty_print(&cov_mat)
	w.destroy_dataframe(&cov_mat)

	// Cleanup
	w.destroy_dataframe(&df)
	w.destroy_dataframe(&df2)
	w.destroy_dataframe(&df3)
	w.destroy_dataframe(&df4)
	w.destroy_dataframe(&df5)
}


ewm_cov_test :: proc(allocator: mem.Allocator) {
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

	ewm_cov_df := w.ewm_cov_matrix(
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
	w.destroy_dataframe(&ewm_cov_df)
}


ewm_pca_test :: proc(allocator: mem.Allocator) {
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
	cov_df := w.ewm_cov_matrix(&df, []string{"x", "y", "z"}, alpha, minp, bias, adjust, allocator)
	defer w.destroy_dataframe(&cov_df)
	w.dataframe_pretty_print(&cov_df)

	// --- EWM PCA (full time series) ---
	fmt.println("\nEWM PCA (all rows):")
	pca_series := w.ewm_pca(&df, []string{"x", "y", "z"}, alpha, minp, bias, adjust, allocator)
	for i in 0 ..< len(pca_series) {
		fmt.printf("Row %d:\n", i)
		fmt.println("  Eigenvalues:  ", pca_series[i].eigenvalues)
		fmt.println("  Eigenvectors: ", pca_series[i].eigenvectors)
		w.destroy_pca_result(pca_series[i])
	}

	// --- EWM PCA (last row only) ---
	fmt.println("\nEWM PCA (last row):")
	last := w.ewm_pca_last(&df, []string{"x", "y", "z"}, alpha, minp, bias, adjust, allocator)
	fmt.println("Eigenvalues:", last.eigenvalues)
	fmt.println("Eigenvectors:", last.eigenvectors)
	w.destroy_pca_result(last)

	fmt.println("\n=== END EWM PCA TEST ===")
}
