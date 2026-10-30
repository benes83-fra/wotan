package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// SVM / SVR / k-means / KNN / naive Bayes against scikit-learn 1.2.1 on a fixed
// 12-point dataset (reference values hardcoded):
// - SVC(kernel="linear", C=1) and SVC(rbf, C=3, γ=0.7) decision values (exact SMO)
// - SVR(rbf, C=2, ε=0.2, γ=0.5) predictions
// - KMeans(init=array, lloyd): centres and inertia; seeded k-means++ is reproducible
// - KNeighborsClassifier(3, weights="distance"), GaussianNB predictions and means
// - degenerate inputs: n_init = 0, identical points, empty clusters, tiny-scale NB

ml_svm_cluster_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML SVM / Clustering / KNN / NB vs sklearn ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	close :: proc(got, want: []f64, rtol: f64) -> bool {
		if len(got) != len(want) do return false
		for g, i in got {
			if !(math.abs(g - want[i]) <= rtol * math.max(math.abs(want[i]), 1e-12)) do return false
		}
		return true
	}

	X := l.matrix_new(f64, 12, 2)
	X.data = {
		0.2,
		1.1,
		1.5,
		-0.3,
		-0.7,
		0.4,
		2.1,
		1.8,
		-1.2,
		-1.5,
		0.9,
		0.6,
		-0.4,
		2.2,
		1.7,
		-1.4,
		0.1,
		-0.8,
		-1.8,
		0.9,
		2.4,
		0.2,
		-0.3,
		-0.2,
	}
	yc := []f64{1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1}
	yr := []f64{0.5, 1.2, -0.3, 2.0, -1.1, 0.9, 0.4, 0.6, -0.5, -0.9, 1.6, -0.1}
	yl := make([]f64, 12)
	for v, i in yc do yl[i] = 1 if v > 0 else 0
	T := l.matrix_new(f64, 3, 2)
	T.data = {0.5, 0.5, -1.0, 1.0, 1.0, -1.0}

	// --- SVM / SVR ---
	lin := ml.svm_fit_linear(&X, yc, {C = 1.0, tol = 1e-10, fit_intercept = true})
	expect(
		&passed,
		&total,
		close(
			ml.svm_predict_linear(&lin, &T),
			{0.1400000065565109, -1.8099999934434892, 0.390000006556511},
			1e-6,
		) &&
		len(lin.support_vectors) == 4,
		"Linear SVM = sklearn SVC(kernel=linear) (decision values, 4 SVs)",
	)
	expect(&passed, &total, lin.converged, "Linear SVM SMO converged")
	rbf := ml.kernel_svm_fit(&X, yc, {C = 3.0, kernel_type = .RBF, gamma = 0.7, tol = 1e-10})
	expect(
		&passed,
		&total,
		close(
			ml.kernel_svm_predict(&rbf, &T),
			{0.8765499166714906, -0.9767444241888926, 0.3474732057997667},
			1e-6,
		),
		"RBF SVM decision values = sklearn SVC(rbf)",
	)
	expect(
		&passed,
		&total,
		math.abs(rbf.bias - -0.042193106545379645) < 1e-6,
		"RBF SVM intercept = sklearn",
	)
	sa := 0.0
	for a, i in rbf.alpha do sa += a * rbf.sv_labels[i]
	expect(&passed, &total, math.abs(sa) < 1e-10, "SVM dual satisfies Σ α_i y_i = 0")
	svr := ml.svr_fit(
		&X,
		yr,
		{C = 2.0, epsilon = 0.2, kernel_type = .RBF, gamma = 0.5, tol = 1e-10},
	)
	expect(
		&passed,
		&total,
		close(
			ml.svr_predict(&svr, &T),
			{0.5112407110801622, -0.4207234889666947, 0.33607475748117965},
			1e-6,
		),
		"SVR predictions = sklearn SVR(rbf)",
	)

	// --- k-means ---
	km := ml.km_fit(
		&X,
		{n_clusters = 2, max_iter = 300, tol = 1e-4, init_centers = {0.2, 1.1, 1.5, -0.3}},
	)
	expect(
		&passed,
		&total,
		close(
			km.centroids.data,
			{-0.5857142857142856, 0.30000000000000004, 1.72, 0.18000000000000005},
			1e-12,
		),
		"KMeans centres = sklearn KMeans(init=array)",
	)
	expect(
		&passed,
		&total,
		math.abs(km.inertia - 19.24457142857143) < 1e-10,
		"KMeans inertia (labels from the final centres)",
	)
	pa := ml.km_fit(
		&X,
		{
			n_clusters = 3,
			max_iter = 100,
			tol = 1e-4,
			init = .KMeansPlusPlus,
			n_init = 0,
			random_state = 3,
		},
	)
	pb := ml.km_fit(
		&X,
		{
			n_clusters = 3,
			max_iter = 100,
			tol = 1e-4,
			init = .KMeansPlusPlus,
			n_init = 0,
			random_state = 3,
		},
	)
	expect(
		&passed,
		&total,
		pa.inertia == pb.inertia && pa.centroids.data[0] == pb.centroids.data[0],
		"k-means++ with random_state is reproducible (n_init = 0 → 1)",
	)
	Xs := l.matrix_new(f64, 6, 1)
	Xs.data = {1, 1, 1, 1, 1, 1}
	same := ml.km_fit(
		&Xs,
		{n_clusters = 2, max_iter = 10, init = .KMeansPlusPlus, n_init = 2, random_state = 1},
	)
	expect(&passed, &total, same.inertia == 0, "k-means++ on identical points")
	ke := ml.km_fit(
		&X,
		{n_clusters = 2, max_iter = 50, tol = 1e-4, init_centers = {50, 50, 0.2, 1.1}},
	)
	expect(
		&passed,
		&total,
		close(ke.centroids.data, {-1.2, -1.5, 0.5181818181818182, 0.40909090909090917}, 1e-12) &&
		math.abs(ke.inertia - 28.74545454545455) < 1e-10,
		"Empty cluster relocated to the farthest point = sklearn",
	)

	// --- KNN / NB ---
	knn := ml.knn_fit(&X, yl, {k = 3, weights = .Distance})
	expect(
		&passed,
		&total,
		close(ml.knn_predict(&knn, &T), {1, 0, 1}, 0),
		"KNN (k=3, distance) = sklearn",
	)
	g := ml.gnb_fit(&X, yl)
	expect(
		&passed,
		&total,
		close(ml.gnb_predict(&g, &T), {1, 0, 1}, 0),
		"GaussianNB predictions = sklearn",
	)
	expect(
		&passed,
		&total,
		close(
			g.theta.data,
			{-0.7166666666666667, 0.1666666666666667, 1.4666666666666668, 0.3333333333333334},
			1e-12,
		),
		"GaussianNB class means, classes sorted",
	)
	Xt := l.matrix_new(f64, 12, 2)
	for v, i in X.data do Xt.data[i] = v * 1e-5
	gt := ml.gnb_fit(&Xt, yl)
	pt := ml.gnb_predict(&gt, &Xt)
	pr := ml.gnb_predict(&g, &X)
	expect(
		&passed,
		&total,
		close(pt, pr, 0),
		"GaussianNB var_smoothing is scale-relative (tiny-scale X)",
	)

	fmt.printf("ML SVM/cluster summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
