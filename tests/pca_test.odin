package tests

import "core:fmt"
import "core:mem"

import analytic "../wotan/analytics"
import w "../wotan/core"

pca_test :: proc(allocator: mem.Allocator) {
	fmt.println("PCA Test...........")
	df := w.dataframe_new()
	x := w.column_from_f64("x", []f64{1, 2, 3, 4})
	y := w.column_from_f64("y", []f64{2, 4, 6, 8})
	z := w.column_from_f64("z", []f64{1, 0, 1, 0})
	// defer
	w.add_column(&df, x)
	w.add_column(&df, y)
	w.add_column(&df, z)

	df.rows = 4
	data := analytic.extract_numeric_matrix(&df, []string{"x", "y", "z"}, allocator)
	// defer w.destroy_matrix(data)
	cov := analytic.covariance_matrix(data, allocator)
	// defer w.destroy_matrix(data)
	analytic.print_matrix(cov)
	pca := analytic.pca_dataframe(&df, []string{"x", "y", "z"}, allocator)
	// defer w.destroy_matrix(data)
	fmt.println("Eigenvalues:", pca.eigenvalues)
	fmt.println("Eigenvectors:", pca.eigenvectors)
	scores := analytic.pca_transform(data, pca)
	defer analytic.destroy_matrix(scores)
	fmt.println("PCA - Transform Score: ", scores)
	rp := analytic.rolling_pca(&df, []string{"x", "y"}, 3, 2, allocator)
	fmt.println("Rolling PCA: ", rp)


	w.destroy_dataframe(&df)


}
