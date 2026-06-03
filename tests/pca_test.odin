package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
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
pcr_test :: proc(allocator: mem.Allocator) {
	// Generate highly collinear data (5 features, but really only 1 underlying signal)
	n := 200
	X := l.matrix_new(f64, n, 5, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		base := rand.float64_normal(0, 1)
		// Features 0, 1, 2 are highly correlated
		X.data[i * 5 + 0] = base + rand.float64_normal(0, 0.1)
		X.data[i * 5 + 1] = base + rand.float64_normal(0, 0.1)
		X.data[i * 5 + 2] = base + rand.float64_normal(0, 0.1)
		// Features 3, 4 are pure noise
		X.data[i * 5 + 3] = rand.float64_normal(0, 1)
		X.data[i * 5 + 4] = rand.float64_normal(0, 1)

		// Target depends heavily on the underlying 'base' signal
		y[i] = 3.0 * base + rand.float64_normal(0, 0.5)
	}

	// Fit PCR: Automatically select components that explain 95% of variance
	params := ml.PCRParams {
		n_components       = 0, // Auto-select
		variance_threshold = 0.95,
	}

	model := ml.pcr_fit(&X, y, params, allocator)
	defer ml.pcr_free(&model)

	fmt.printf("PCR selected %v components (out of 5)\n", model.n_components)

	preds := ml.pcr_predict(&model, &X, allocator)
	defer delete(preds, allocator)

	// Compute MSE
	mse := 0.0
	for i in 0 ..< n {
		err := y[i] - preds[i]
		mse += err * err
	}
	mse /= f64(n)
	fmt.printf("PCR Training MSE: %.4f\n", mse)

	l.matrix_free(&X)
	delete(y, allocator)
}
