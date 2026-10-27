package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Trees and ensembles against scikit-learn 1.2.1 on a fixed 30-point dataset
// (reference values hardcoded):
// - DecisionTreeRegressor(max_depth=3): predictions, node count, importances
// - GradientBoostingRegressor(20 trees, lr 0.1, depth 2): predictions
// - a forest without bootstrap and with all features equals the single tree
// - bootstrap forests on 1 sample, constant X and tiny-scale y (used to panic /
//   never split), OOB R² is reported, stochastic GB runs on a row subsample

ml_tree_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML Trees / Ensembles vs sklearn ===")
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

	xd := []f64{-0.802, -1.324, -0.248, 0.42, 1.136, 0.11, -0.553, -0.785, 0.749, 1.635, 0.273, -1.233, -0.958, 1.6, 0.203, -1.732, -0.084, -1.163, -0.629, -0.488, -0.713, 0.553, -0.063, -0.589, 0.41, 0.83, -1.643, -0.257, -0.981, -0.173, -1.289, 0.021, -0.038, -0.304, -1.048, -0.396, -1.091, -1.355, 0.225, -1.109, 1.17, 0.717, -1.998, 0.272, -1.102, 0.033, 0.044, -1.988, -0.233, -0.256, 0.962, -1.181, 0.738, -1.099, -0.331, -0.84, 1.449, 0.568, 2.432, 0.642}
	y := []f64{-0.038, -0.304, 0.709, -0.593, 2.469, 1.24, 0.358, 1.893, 0.57, -0.869, -0.852, 0.072, 1.086, 0.09, -0.82, -0.664, -0.15, -0.915, 0.196, 1.014, 0.878, 0.678, -0.763, 1.958, -0.544, 1.697, 1.48, -0.294, 0.402, -0.827}
	n := len(y)
	X := l.matrix_new(f64, n, 2)
	copy(X.data, xd)
	T := l.matrix_new(f64, 4, 2)
	T.data = {0.1, -0.5, -1.2, 0.8, 0.9, 1.5, -0.3, -1.1}

	// --- Decision tree ---
	dt := ml.dt_fit(&X, y, {max_depth = 3, min_samples = 2})
	expect(&passed, &total, close(ml.dt_predict(&dt, &T), {1.4811428571428575, -0.48753846153846153, 2.469, 0.24266666666666667}, 1e-12), "DT predictions = sklearn DecisionTreeRegressor(max_depth=3)")
	expect(&passed, &total, len(dt.nodes) == 15, "DT node count = sklearn (15)")
	st := ml.dt_compute_stats(&dt, &X, y)
	expect(&passed, &total, close(st.feature_importance, {0.9117198407615832, 0.08828015923841682}, 1e-10), "DT impurity-decrease feature importances = sklearn")

	// --- Gradient boosting ---
	gb := ml.gb_fit(&X, y, {n_estimators = 20, learning_rate = 0.1, max_depth = 2, min_samples = 2, subsample = 1.0})
	expect(&passed, &total, close(ml.gb_predict(&gb, &T), {1.0604137803394664, -0.3912832469595344, 1.6892224669157268, -0.013910510822891353}, 1e-12), "GB predictions = sklearn GradientBoostingRegressor")

	// --- Random forest ---
	rf := ml.rf_fit(&X, y, {n_trees = 4, max_features = 2, min_samples = 2, max_depth = 3, bootstrap = false})
	expect(&passed, &total, close(ml.rf_predict(&rf, &T), ml.dt_predict(&dt, &T), 1e-12), "RF without bootstrap, all features = single tree")
	rand.reset(7)
	rfb := ml.rf_fit(&X, y, {n_trees = 50, max_features = 1, min_samples = 2, max_depth = 6, bootstrap = true, oob_score = true})
	expect(&passed, &total, !math.is_nan(rfb.oob_score) && rfb.oob_score < 1.0, fmt.tprintf("RF OOB R² reported (%.3f)", rfb.oob_score))
	rs := ml.rf_compute_stats(&rfb, &X, y)
	sum_imp := 0.0
	for v in rs.feature_importance do sum_imp += v
	splits := 0
	for v in rs.splits_per_feature do splits += v
	expect(&passed, &total, math.abs(sum_imp - 1) < 1e-12 && splits > 0, "RF importances sum to 1, splits_per_feature filled")

	// --- Degenerate inputs ---
	X1 := l.matrix_new(f64, 1, 2)
	X1.data = {0.5, 0.5}
	r1 := ml.rf_fit(&X1, []f64{2.0}, {n_trees = 3, max_depth = 3, min_samples = 2, bootstrap = true})
	expect(&passed, &total, ml.rf_predict(&r1, &X1)[0] == 2.0, "RF bootstrap on a single sample")
	Xc := l.matrix_new(f64, 10, 2)
	yc := make([]f64, 10)
	for i in 0 ..< 10 do yc[i] = f64(i)
	dc := ml.dt_fit(&Xc, yc, {max_depth = 5, min_samples = 2})
	expect(&passed, &total, len(dc.nodes) == 1, "DT on constant X is a single leaf")
	ys := make([]f64, n)
	for v, i in y do ys[i] = v * 1e-7
	dsm := ml.dt_fit(&X, ys, {max_depth = 3, min_samples = 2})
	expect(&passed, &total, len(dsm.nodes) == 15, "DT splits tiny-scale y (variance ~1e-14) like sklearn")
	gbs := ml.gb_fit(&X, y, {n_estimators = 30, learning_rate = 0.1, max_depth = 2, min_samples = 2, subsample = 0.5})
	expect(&passed, &total, len(gbs.trees) == 30 && gbs.trees[0].nodes[0].sample_count == 15, "Stochastic GB fits each tree on int(0.5·n) rows")

	fmt.printf("ML trees summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
