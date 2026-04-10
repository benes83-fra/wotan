package main

import "core:fmt"
import "core:mem"

import w "./wotan/core"

pca_test :: proc(allocator: mem.Allocator) {
	fmt.println("PCA Test...........")
	df := w.dataframe_new()

	w.add_column(&df, w.column_from_f64("x", []f64{1, 2, 3, 4}))
	w.add_column(&df, w.column_from_f64("y", []f64{2, 4, 6, 8}))
	w.add_column(&df, w.column_from_f64("z", []f64{1, 0, 1, 0}))

	df.rows = 4
	data := w.extract_numeric_matrix(&df, []string{"x", "y", "z"})
	defer w.destroy_matrix(data)
	cov := w.covariance_matrix(data, allocator)
	// defer w.destroy_matrix(data)
	w.print_matrix(cov)
	pca := w.pca_dataframe(&df, []string{"x", "y", "z"}, allocator)
	// defer w.destroy_matrix(data)
	fmt.println("Eigenvalues:", pca.eigenvalues)
	fmt.println("Eigenvectors:", pca.eigenvectors)
	scores := w.pca_transform(data, pca)
	defer w.destroy_matrix(scores)
	fmt.println("PCA - Transform Score: ", scores)
	rp := w.rolling_pca(&df, []string{"x", "y"}, 3, 2,allocator)
	fmt.println("Rolling PCA: ", rp)
	// for res in rp {
	// 	w.destroy_pca_result(res)
	// }

	w.destroy_dataframe(&df)


}
