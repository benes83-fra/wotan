package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"

import analytic "../wotan/analytics"
import w "../wotan/core"

pca_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
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

	pca_reference_checks(allocator)
}

// PCA reference checks (numpy.cov / numpy.linalg.eigh / sklearn 1.2 PCA):
// ddof-1 eigenvalues in descending order, components with the largest loading
// positive, transform centres the data, inverse transform adds the mean back,
// rolling_pca returns one result per row (empty during warm-up)
pca_reference_checks :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== PCA Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	close :: proc(a: []f64, b: []f64, tol: f64) -> bool {
		if len(a) != len(b) {return false}
		for v, i in a {if abs(v - b[i]) > tol {return false}}
		return true
	}

	xs := []f64{2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1}
	ys := []f64{2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9}
	zs := []f64{1.0, 2.1, 0.3, 1.7, 0.2, 1.1, 2.5, 1.9, 0.8, 2.2}
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	w.add_column(&df, w.column_from_f64("x", xs))
	w.add_column(&df, w.column_from_f64("y", ys))
	w.add_column(&df, w.column_from_f64("z", zs))
	df.rows = 10
	names := []string{"x", "y", "z"}

	pca := analytic.pca_dataframe(&df, names, allocator)
	defer analytic.destroy_pca_result(pca, allocator)
	expect(&passed, &total,
		close(pca.eigenvalues, []f64{1.7355064520482997, 0.22205886370958336, 0.035101350908784482}, 1e-13),
		fmt.tprintf("eigenvalues = numpy.cov (ddof 1) eigenvalues, descending: %v", pca.eigenvalues))
	expect(&passed, &total,
		close(pca.eigenvectors[0], []f64{0.55505889085569649, 0.62932593193315201, -0.54392876287112668}, 1e-12) &&
		close(pca.eigenvectors[1], []f64{0.55494302052655076, 0.20694705799706375, 0.80573640798664903}, 1e-12),
		"components (rows) match numpy, largest loading positive")

	data := analytic.extract_numeric_matrix(&df, names, allocator)
	defer analytic.destroy_matrix(data, allocator)
	scores := analytic.pca_transform(data, pca)
	defer analytic.destroy_matrix(scores)
	expect(&passed, &total,
		close(scores[0], []f64{0.89805327122870338, 0.17813490754695449, -0.14956027797735477}, 1e-12) &&
		close(scores[9], []f64{-1.4757325893143518, 0.057677781398166816, -0.12444516209377093}, 1e-12),
		"pca_transform centres the data (scores = (X - mean) Vᵀ)")

	top1 := make([][]f64, 10)
	defer delete(top1)
	for r, i in scores {top1[i] = r[:1]}
	rec := analytic.pca_inverse_transform(top1, pca)
	defer analytic.destroy_matrix(rec)
	full := analytic.pca_inverse_transform(scores, pca)
	defer analytic.destroy_matrix(full)
	round_trip := true
	for r, i in full {round_trip &= close(r, data[i], 1e-12)}
	expect(&passed, &total,
		close(rec[0], []f64{2.3084724526575342, 2.4751682118416189, 0.8915229951882031}, 1e-12) && round_trip,
		"pca_inverse_transform adds the mean back (rank-1 reconstruction, exact round trip)")

	rp := analytic.rolling_pca(&df, names[:2], 5, 4, allocator)
	defer {
		for r in rp {analytic.destroy_pca_result(r, allocator)}
		delete(rp, allocator)
	}
	warm_empty := len(rp) == 10
	if warm_empty {for i in 0 ..< 3 {warm_empty &= len(rp[i].eigenvalues) == 0}}
	expect(&passed, &total,
		warm_empty &&
		close(rp[3].eigenvalues, []f64{1.6266560789829616, 0.052510587683704868}, 1e-13) &&
		close(rp[9].eigenvalues, []f64{0.76654800781241406, 0.037451992187585853}, 1e-13),
		fmt.tprintf("rolling_pca: one result per row (%d), empty before min_periods, windowed eigenvalues", len(rp)))

	fmt.printf("PCA reference summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
pcr_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
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
