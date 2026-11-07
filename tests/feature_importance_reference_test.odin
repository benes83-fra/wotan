package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:mem"

// Feature importance (ml_finance/feature_importance.odin) against López de Prado's
// snippets in pandas / sklearn:
//   A. MDI aggregation of per-tree importances with unused features (AFML 8.2) and
//      clustered (MLAM 6.4), on an LCG matrix
//   B. MDA from fold scores, negative log loss and accuracy (AFML 8.3)
//   C. SFI end to end: sklearn DecisionTreeClassifier(max_depth=3), KFold(5), sample
//      weights, -log_loss(sample_weight=) per feature (AFML 8.4)
//   D. MDA end to end with the same tree and the LCG permutations (seed): single
//      features with log loss, clusters with accuracy
//   E. cluster_features = clusterKMeansBase (MLAM 4.1) on three blocks of four
//      correlated features: same partition, same silhouettes

@(private = "file")
_FI_MDI_MEAN := [5]f64 {
	0.22583206754627372,
	0.18951466083530605,
	0.17112355565259876,
	0.21191761858547647,
	0.20161209738034502,
}
@(private = "file")
_FI_MDI_STD := [5]f64 {
	0.019928032944955289,
	0.02055632118682918,
	0.026101000689117176,
	0.016544080445434241,
	0.022646586905437232,
}
@(private = "file")
_FI_CMDI_MEAN := [3]f64{0.41534672838157966, 0.17112355565259876, 0.41352971596582155}
@(private = "file")
_FI_CMDI_STD := [3]f64{0.0351910822866092, 0.033116138094239161, 0.039859036900687372}
@(private = "file")
_FI_MDA_LL := [6]f64 {
	0.13298295906991559,
	0.027724733071441153,
	-0.0002407318247472319,
	0.01686751880202399,
	0.0080812882072118716,
	0.0049123450902974364,
}
@(private = "file")
_FI_MDA_ACC := [6]f64 {
	0.30287818383167214,
	0.16841024128949816,
	-0.00074074074074074179,
	0.044127408739278909,
	0.024072840115967657,
	0.011135775095091052,
}
@(private = "file")
_FI_SFI := [8]f64 {
	-1.5877457685507057,
	-2.7147450820247299,
	-2.2813827625276462,
	-1.6272480775225351,
	0.50139612971063763,
	0.91499847561093117,
	0.36062796990945023,
	0.40808000203680606,
}
@(private = "file")
_FI_MDA1 := [9]f64 {
	0.70791802708461071,
	0.0044096708630659617,
	0,
	0.15663642174782688,
	0.093234812778058909,
	0.35119477248962455,
	0,
	0.088048914383492105,
	-1.0793763034407156,
}
@(private = "file")
_FI_MDA2 := [5]f64 {
	0.56957972689066549,
	-0.034163979691279246,
	0.052674295366257645,
	0.046363385644886256,
	0.79257274660592658,
}
@(private = "file")
_FI_SILH := [12]f64 {
	0.6093448763480479,
	0.61097844369375265,
	0.60932961118447959,
	0.60570567753613658,
	0.61502286731050337,
	0.6104844125691351,
	0.59917780183547498,
	0.60195824687845123,
	0.60300031784161534,
	0.59827574679302842,
	0.6125455791392862,
	0.60431734529129522,
}

@(private = "file")
_fi_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

// max error: relative where |want| > 1e-3, absolute otherwise
@(private = "file")
_fi_err :: proc(got, want: []f64) -> f64 {
	e := 0.0
	for v, i in want {
		d := abs(got[i] - v)
		e = max(e, d / abs(v) if abs(v) > 1e-3 else d)
	}
	return e
}

feature_importance_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Feature Importance Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. MDI aggregation
	{
		T, F :: 30, 5
		imp := l.matrix_new(f64, T, F)
		defer l.matrix_free(&imp)
		x: u64 = 11
		for t in 0 ..< T {
			s := 0.0
			for j in 0 ..< F {
				v := _fi_u(&x)
				imp.data[t * F + j] = 0 if (t * 7 + j * 3) % 5 == 0 else v
				s += imp.data[t * F + j]
			}
			for j in 0 ..< F {imp.data[t * F + j] /= s}
		}
		r := ml_fin.mdi_from_tree_importances(&imp)
		defer ml_fin.importance_result_free(&r)
		groups := [][]int{{0, 1}, {2}, {3, 4}}
		c := ml_fin.mdi_from_tree_importances(&imp, groups)
		defer ml_fin.importance_result_free(&c)
		err := max(
			_fi_err(r.mean, _FI_MDI_MEAN[:]),
			_fi_err(r.std_err, _FI_MDI_STD[:]),
			_fi_err(c.mean, _FI_CMDI_MEAN[:]),
			_fi_err(c.std_err, _FI_CMDI_STD[:]),
		)
		expect(
			&passed,
			&total,
			err < 1e-13,
			fmt.tprintf(
				"MDI of 30 trees x 5 features with unused features (zeros skipped) and clustered (group sums): means and standard errors = pandas, error %.1e",
				err,
			),
		)
	}
	// B. MDA from scores
	{
		base := []f64{-0.61, -0.58, -0.66, -0.6, -0.63}
		pd := []f64 {
			-0.7,
			-0.62,
			-0.61,
			-0.69,
			-0.6,
			-0.58,
			-0.8,
			-0.7,
			-0.67,
			-0.65,
			-0.61,
			-0.6,
			-0.72,
			-0.64,
			-0.62,
		}
		perm := l.Matrix(f64) {
			data = pd,
			rows = 5,
			cols = 3,
		}
		r := ml_fin.mda_from_scores(base, &perm, .Neg_Log_Loss)
		defer ml_fin.importance_result_free(&r)
		base_a := []f64{0.7, 0.66, 0.72, 0.69, 0.71}
		pa := []f64 {
			0.55,
			0.6,
			0.7,
			0.58,
			0.62,
			0.66,
			0.5,
			0.66,
			0.73,
			0.6,
			0.64,
			0.69,
			0.57,
			0.65,
			0.7,
		}
		perm_a := l.Matrix(f64) {
			data = pa,
			rows = 5,
			cols = 3,
		}
		a := ml_fin.mda_from_scores(base_a, &perm_a, .Accuracy)
		defer ml_fin.importance_result_free(&a)
		err := max(
			_fi_err(r.mean, _FI_MDA_LL[:3]),
			_fi_err(r.std_err, _FI_MDA_LL[3:]),
			_fi_err(a.mean, _FI_MDA_ACC[:3]),
			_fi_err(a.std_err, _FI_MDA_ACC[3:]),
		)
		expect(
			&passed,
			&total,
			err < 1e-13,
			fmt.tprintf(
				"MDA from 5 folds x 3 features, log loss ((base - perm) / -perm) and accuracy (/ (1 - perm)) = pandas, error %.1e",
				err,
			),
		)
	}

	// data for C / D
	N, F :: 200, 4
	X := l.matrix_new(f64, N, F)
	defer l.matrix_free(&X)
	y := make([]f64, N)
	defer delete(y)
	w := make([]f64, N)
	defer delete(w)
	{
		x: u64 = 2025
		for i in 0 ..< N {
			for j in 0 ..< F {X.data[i * F + j] = _fi_u(&x) - 0.5}
			e := _fi_u(&x) - 0.5
			y[i] = 1 if X.data[i * F] + 0.6 * X.data[i * F + 1] + 0.5 * e > 0 else 0
			w[i] = 0.5 + _fi_u(&x)
		}
	}
	tp := ml.TreeParams {
		max_depth   = 3,
		min_samples = 2,
		criterion   = .Gini,
	}
	model := ml_fin.dt_importance_model(&tp)
	cv := ml.CVSplitter {
		kind     = .KFold,
		n_splits = 5,
	}
	// C. SFI
	{
		r := ml_fin.sfi_importance(model, &X, y, cv, w)
		defer ml_fin.importance_result_free(&r)
		err := max(_fi_err(r.mean, _FI_SFI[:4]), _fi_err(r.std_err, _FI_SFI[4:]))
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf(
				"SFI end to end (tree of depth 3, KFold(5), sample weights, weighted -log loss) = sklearn per feature, error %.1e",
				err,
			),
		)
	}
	// D. MDA end to end
	{
		r, oos := ml_fin.mda_importance(model, &X, y, cv, w, seed = 99)
		defer ml_fin.importance_result_free(&r)
		groups := [][]int{{0, 1}, {2, 3}}
		g, oos2 := ml_fin.mda_importance(model, &X, y, cv, w, .Accuracy, groups, seed = 5)
		defer ml_fin.importance_result_free(&g)
		err := max(
			_fi_err(r.mean, _FI_MDA1[:4]),
			_fi_err(r.std_err, _FI_MDA1[4:8]),
			_fi_err([]f64{oos}, _FI_MDA1[8:]),
			_fi_err(g.mean, _FI_MDA2[:2]),
			_fi_err(g.std_err, _FI_MDA2[2:4]),
			_fi_err([]f64{oos2}, _FI_MDA2[4:]),
		)
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf(
				"MDA end to end with the LCG permutations: 4 features (log loss) and 2 clusters (accuracy), means, standard errors and OOS scores = sklearn, error %.1e",
				err,
			),
		)
	}
	// E. clusters
	{
		NZ, FZ :: 300, 12
		Z := l.matrix_new(f64, NZ, FZ)
		defer l.matrix_free(&Z)
		x: u64 = 77
		for i in 0 ..< NZ {
			f := [3]f64{_fi_u(&x) - 0.5, _fi_u(&x) - 0.5, _fi_u(&x) - 0.5}
			for j in 0 ..< FZ {Z.data[i * FZ + j] = f[j % 3] + 0.6 * (_fi_u(&x) - 0.5)}
		}
		c := ml_fin.cluster_features(&Z)
		defer ml_fin.feature_clusters_free(&c)
		ok := len(c.groups) == 3
		for g in c.groups {
			ok &&= len(g) == 4
			for f in g {ok &&= f % 3 == g[0] % 3}
		}
		err := _fi_err(c.silhouette, _FI_SILH[:])
		expect(
			&passed,
			&total,
			ok && err < 1e-12,
			fmt.tprintf(
				"cluster_features = clusterKMeansBase on 12 features in 3 blocks: %d clusters, the blocks recovered, silhouettes = sklearn (error %.1e)",
				len(c.groups),
				err,
			),
		)
	}
	fmt.printf("Feature importance reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
