package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// Cross-validation splitters vs sklearn 1.2.1 (exact train / test indices):
// KFold (no shuffle), StratifiedKFold (imbalanced, classes not in sorted order),
// TimeSeriesSplit (default, test_size + gap, rolling max_train_size + gap),
// and a walk-forward ridge cross_val_score; plus KFold-compatibility of the
// old cross_val_score and the no-look-ahead property of Time_Series.

@(private = "file")
CV_XR := [?]f64{1.0, 0.0, 2.0409191213851825, 1.0, 0.19866933079506122, -2.5556650313141818, 1.0, 0.3894183423086505, 0.41809884672577885, 1.0, 0.5646424733950354, -0.5677696061279298, 1.0, 0.7173560908995228, -0.45264929211044586, 1.0, 0.8414709848078965, -0.2155971630897659, 1.0, 0.9320390859672263, -2.019986129147251, 1.0, 0.9854497299884601, -0.23193237764418947, 1.0, 0.9995736030415051, -0.8652130762749417, 1.0, 0.9738476308781951, 3.3229995166448827, 1.0, 0.9092974268256817, 0.22578661322792176, 1.0, 0.8084964038195901, -0.3526307943415954, 1.0, 0.675463180551151, -0.2812874181513504, 1.0, 0.5155013718214642, -0.6680463461089501, 1.0, 0.3349881501559051, -1.0551505512051214, 1.0, 0.1411200080598672, -0.39080097723465473, 1.0, -0.058374143427580086, 0.48194538850678587, 1.0, -0.2555411020268312, -0.2385536065733667, 1.0, -0.44252044329485246, 0.9577587029597641, 1.0, -0.6118578909427189, -0.19980212906658, 1.0, -0.7568024953079282, 0.024259565076664623, 1.0, -0.8715757724135882, 1.545820851212812, 1.0, -0.9516020738895161, 0.5451055226876446, 1.0, -0.9936910036334644, -0.505228735614018, 1.0, -0.9961646088358407, -0.1828389745977349, 1.0, -0.9589242746631385, 0.5405251317548021, 1.0, -0.8834546557201531, 1.9350880340988528, 1.0, -0.7727644875559871, -0.2696203273419135, 1.0, -0.6312666378723216, -0.24355867907910456, 1.0, -0.46460217941375737, 1.0023136012756912, 1.0, -0.27941549819892586, -0.8864599431605871, 1.0, -0.0830894028174964, -0.291720232439864, 1.0, 0.11654920485049364, 0.8825389674564839, 1.0, 0.31154136351337786, 0.5803500161908991, 1.0, 0.49411335113860816, 0.09151670328235219, 1.0, 0.6569865987187891, 0.6701043548284794, 1.0, 0.7936678638491531, -2.8281623068437627, 1.0, 0.8987080958116269, 1.02130681750008, 1.0, 0.9679196720314863, -0.9596447598081417, 1.0, 0.998543345374605, -1.6686198426559695}
@(private = "file")
CV_YR := [?]f64{0.06247416716370864, 2.90533464285203, 1.4763570245495765, 2.1502479978237936, 2.748874277914479, 2.874916358687812, 4.415750686313506, 3.45108804504282, 3.649898438099867, 1.7996854650011465, 2.844044632065665, 2.7355383327154676, 2.906787563485552, 2.799787442294111, 2.4131029025763144, 1.542797931347872, 1.0310251904554257, 0.20002631569562374, 0.20311719297925665, 0.4035957608713495, -0.6173919163213352, -1.0776559174821312, -1.0249868081852267, -0.047601326032474744, -1.0311599124058424, -0.9624594766020544, -1.0015793321671123, 0.4762015030886296, -0.2281555502429533, 0.0002268871092485636, 1.5828050029452148, 1.416916469199978, 1.9090209953696888, 1.6355397144459722, 2.7288279345221182, 2.3643993644599854, 5.143205764060653, 3.020267414156546, 4.063986554766858, 4.09594112715741}

ml_cv_splitter_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML CV Splitters (KFold / Stratified / TimeSeries) ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	same_folds :: proc(folds: []ml.CVFold, tr_len, te_len, flat: []int) -> bool {
		if len(folds) != len(tr_len) do return false
		p := 0
		for f, i in folds {
			if len(f.train) != tr_len[i] || len(f.test) != te_len[i] do return false
			for v in f.train {if v != flat[p] do return false; p += 1}
			for v in f.test {if v != flat[p] do return false; p += 1}
		}
		return true
	}
	y_cls := []f64{2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}

	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .KFold, n_splits = 4}, 10, nil)
		expect(&passed, &total, same_folds(folds, {7, 7, 8, 8}, {3, 3, 2, 2}, {3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 0, 1, 2, 6, 7, 8, 9, 3, 4, 5, 0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), "KFold {kind = .KFold, n_splits = 4} = sklearn")
		ml.cv_folds_free(folds)
	}
	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .Stratified_KFold, n_splits = 3}, 23, y_cls)
		expect(&passed, &total, same_folds(folds, {15, 15, 16}, {8, 8, 7}, {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}), "StratifiedKFold {kind = .Stratified_KFold, n_splits = 3} = sklearn")
		ml.cv_folds_free(folds)
	}
	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .Stratified_KFold, n_splits = 5}, 23, y_cls)
		expect(&passed, &total, same_folds(folds, {18, 18, 18, 19, 19}, {5, 5, 5, 4, 4}, {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}), "StratifiedKFold {kind = .Stratified_KFold, n_splits = 5} = sklearn")
		ml.cv_folds_free(folds)
	}
	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .Time_Series, n_splits = 5}, 23, nil)
		expect(&passed, &total, same_folds(folds, {8, 11, 14, 17, 20}, {3, 3, 3, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}), "TimeSeriesSplit {kind = .Time_Series, n_splits = 5} = sklearn")
		ml.cv_folds_free(folds)
	}
	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .Time_Series, n_splits = 3, test_size = 4, gap = 2}, 23, nil)
		expect(&passed, &total, same_folds(folds, {9, 13, 17}, {4, 4, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}), "TimeSeriesSplit {kind = .Time_Series, n_splits = 3, test_size = 4, gap = 2} = sklearn")
		ml.cv_folds_free(folds)
	}
	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .Time_Series, n_splits = 4, max_train_size = 6, gap = 1}, 23, nil)
		expect(&passed, &total, same_folds(folds, {6, 6, 6, 6}, {4, 4, 4, 4}, {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22}), "TimeSeriesSplit {kind = .Time_Series, n_splits = 4, max_train_size = 6, gap = 1} = sklearn")
		ml.cv_folds_free(folds)
	}

	// stratified + shuffle: still exact class counts per fold, different assignment
	{
		plain := ml.cv_make_folds(ml.CVSplitter{kind = .Stratified_KFold, n_splits = 3}, len(y_cls), y_cls)
		shuf := ml.cv_make_folds(ml.CVSplitter{kind = .Stratified_KFold, n_splits = 3, shuffle = true, random_state = 7}, len(y_cls), y_cls)
		counts_ok, differs := true, false
		for f, i in shuf {
			for c in ([]f64{0, 1, 2}) {
				a, b := 0, 0
				for v in f.test {if y_cls[v] == c do a += 1}
				for v in plain[i].test {if y_cls[v] == c do b += 1}
				if a != b do counts_ok = false
			}
			if len(f.test) != len(plain[i].test) {differs = true; continue}
			for v, j in f.test {if v != plain[i].test[j] do differs = true}
		}
		expect(&passed, &total, counts_ok && differs, "Stratified shuffle: same class counts per fold, shuffled membership")
		ml.cv_folds_free(plain)
		ml.cv_folds_free(shuf)
	}

	// Time_Series: every training index lies at least gap + 1 steps before every test index
	{
		folds := ml.cv_make_folds(ml.CVSplitter{kind = .Time_Series, n_splits = 4, gap = 3}, 50)
		ok := true
		for f in folds {
			if f.train[len(f.train) - 1] + 3 >= f.test[0] do ok = false
		}
		expect(&passed, &total, ok, "Time_Series: no look-ahead (train ends gap samples before test)")
		ml.cv_folds_free(folds)
	}

	// walk-forward ridge = sklearn cross_val_score(Ridge(0.5), X, y, TimeSeriesSplit(4, gap=1), "r2")
	X := l.Matrix(f64){data = CV_XR[:], rows = 40, cols = 3}
	params := ml.RidgeParams{lambda = 0.5}
	wf := ml.cross_val_score_cv(&X, CV_YR[:], ml.CVSplitter{kind = .Time_Series, n_splits = 4, gap = 1}, ml.ridge_cv_evaluator, &params)
	want := []f64{0.12772602230324237, 0.4526197723261094, 0.8538239222271244, 0.7233396937576954}
	wf_ok := len(wf) == len(want)
	for v, i in wf {if wf_ok && !(math.abs(v - want[i]) <= 1e-10) do wf_ok = false}
	expect(&passed, &total, wf_ok, "Walk-forward ridge R² per split = sklearn cross_val_score")

	// the old API is the KFold splitter: same scores
	old := ml.cross_val_score(&X, CV_YR[:], 4, ml.ridge_cv_evaluator, &params)
	new := ml.cross_val_score_cv(&X, CV_YR[:], ml.CVSplitter{kind = .KFold, n_splits = 4, shuffle = true, random_state = 42}, ml.ridge_cv_evaluator, &params)
	same := true
	for v, i in old {if v != new[i] do same = false}
	expect(&passed, &total, same, "cross_val_score(n_splits) = cross_val_score_cv(KFold, shuffle, 42)")

	// grid search with walk-forward validation picks the best lambda
	grid := []ml.RidgeParams{{lambda = 1000}, {lambda = 0.5}, {lambda = 50}}
	best, score := ml.grid_search_cv(&X, CV_YR[:], ml.CVSplitter{kind = .Time_Series, n_splits = 4, gap = 1}, ml.ridge_cv_evaluator, grid)
	mean_wf := 0.0
	for v in wf {mean_wf += v}
	mean_wf /= f64(len(wf))
	expect(&passed, &total, best.lambda == 0.5 && math.abs(score - mean_wf) <= 1e-12, "grid_search_cv (Time_Series) selects λ = 0.5 with the walk-forward mean R²")

	fmt.printf("ML CV splitter summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
