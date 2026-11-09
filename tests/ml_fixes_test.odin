package tests

import ml "../wotan/analytics/ML"
import w "../wotan/core"
import l "../wotan/linalg"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"

// Regression checks for fixes in analytics/ML:
// - transform_dataset: a failed call returns outputs that are safe to free; a missing
//   target column fails (ok = false) instead of giving y = 0
// - prepare_dataset: a String target fails instead of becoming all zeros
// - LassoParams max_iter / tol ≤ 0 default to 1000 / 1e-4 in pipelines and grid search
// - ridge / lasso CV evaluators fit an intercept (as elastic net and the pipelines)
// - pipeline files (format 6) store the hyper-parameters: load -> refit works;
//   format-5 files still load
// - tree / forest / boosting evaluators score regressors with R², classifiers with accuracy

ml_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println(
		"\n=== ML fixes (bridge, lasso defaults, CV intercept, model params, tree scoring) ===",
	)
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// --- DataFrame bridge ---
	{
		track: mem.Tracking_Allocator
		// heap backing: an arena backing (the suite allocator) does not support free
		mem.tracking_allocator_init(&track, runtime.heap_allocator())
		track.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
		empty_outputs := _fixes_failed_transform(mem.tracking_allocator(&track))
		expect(
			&passed,
			&total,
			empty_outputs,
			"transform_dataset: missing feature -> ok = false, empty outputs",
		)
		expect(
			&passed,
			&total,
			len(track.bad_free_array) == 0 && len(track.allocation_map) == 0,
			"transform_dataset failure + the usual defers: no bad free, no leak",
		)
		mem.tracking_allocator_destroy(&track)

		train := w.df_from(
			w.column_from_floats("x", []f64{1, 2, 3}),
			w.column_from_ints("target", []int{0, 1, 0}),
		)
		defer w.destroy_dataframe(&train)
		ds, _ := ml.prepare_dataset(&train, "target")
		defer ml.dataset_free(&ds)

		Xm, ym, okm := ml.transform_dataset(&train, &ds, "targt") // misspelled
		defer l.matrix_free(&Xm)
		defer delete(ym)
		expect(
			&passed,
			&total,
			!okm && ym == nil,
			"transform_dataset: unknown target column -> ok = false",
		)

		Xt, yt, okt := ml.transform_dataset(&train, &ds, "target")
		defer l.matrix_free(&Xt)
		defer delete(yt)
		expect(
			&passed,
			&total,
			okt && len(yt) == 3 && yt[1] == 1 && Xt.data[2] == 3,
			"transform_dataset: existing target is read",
		)

		labels := w.df_from(
			w.column_from_floats("x", []f64{1, 2}),
			w.column_from_strings("label", []string{"up", "down"}),
		)
		defer w.destroy_dataframe(&labels)
		ds2, ok2 := ml.prepare_dataset(&labels, "label")
		defer ml.dataset_free(&ds2)
		expect(&passed, &total, !ok2, "prepare_dataset: String target -> ok = false")
	}

	// regression data with a large intercept: y = 5 + 2 x0 - x1 + noise
	n := 60
	X := l.matrix_new(f64, n, 2)
	defer l.matrix_free(&X)
	XA := l.matrix_new(f64, n, 3) // [1, x0, x1]
	defer l.matrix_free(&XA)
	y := make([]f64, n)
	defer delete(y)
	for i in 0 ..< n {
		a, b := rand.float64_normal(0, 1), rand.float64_normal(0, 1)
		X.data[i * 2], X.data[i * 2 + 1] = a, b
		XA.data[i * 3], XA.data[i * 3 + 1], XA.data[i * 3 + 2] = 1, a, b
		y[i] = 5 + 2 * a - b + rand.float64_normal(0, 0.5)
	}

	// --- Lasso defaults ---
	{
		pipe := ml.pipeline_new()
		defer ml.pipeline_free(&pipe)
		ml.pipeline_set_lasso(&pipe, {lambda = 0.5}) // max_iter, tol = 0
		ml.pipeline_fit(&pipe, &X, y)
		ref := ml.lasso_fit(&XA, y, 0.5, 1000, 1e-4)
		defer ml._ols_result_free(&ref, context.allocator)
		same := len(pipe.model.lasso.beta) == 3
		for v, i in ref.beta {if same && pipe.model.lasso.beta[i] != v do same = false}
		expect(
			&passed,
			&total,
			same && ref.beta[1] > 1.5,
			"Lasso pipeline with max_iter = tol = 0 = lasso_fit(1000, 1e-4)",
		)

		p0 := ml.LassoParams {
			lambda = 0.5,
		}
		p1 := ml.LassoParams {
			lambda   = 0.5,
			max_iter = 1000,
			tol      = 1e-4,
		}
		s0 := ml.cross_val_score(&X, y, 3, ml.lasso_cv_evaluator, &p0)
		defer delete(s0)
		s1 := ml.cross_val_score(&X, y, 3, ml.lasso_cv_evaluator, &p1)
		defer delete(s1)
		eq := true
		for v, i in s0 {if v != s1[i] do eq = false}
		expect(
			&passed,
			&total,
			eq && s0[0] > 0.8,
			"lasso_cv_evaluator with max_iter = tol = 0 uses 1000 / 1e-4",
		)
	}

	// --- intercept in the linear CV evaluators ---
	{
		rg := []ml.RidgeParams{{lambda = 0.1}, {lambda = 1}}
		lg := []ml.LassoParams {
			{lambda = 0.1, max_iter = 1000, tol = 1e-6},
			{lambda = 1, max_iter = 1000, tol = 1e-6},
		}
		eg := []ml.ElasticNetParams{{lambda = 0.1, l1_ratio = 0.5}, {lambda = 1, l1_ratio = 0.5}}
		_, rs := ml.grid_search_ridge(&X, y, 3, rg)
		_, ls := ml.grid_search_lasso(&X, y, 3, lg)
		_, es := ml.grid_search_elastic_net(&X, y, 3, eg)
		fmt.printf(
			"   CV R² without a ones column: ridge %.4f, lasso %.4f, elastic net %.4f\n",
			rs,
			ls,
			es,
		)
		expect(
			&passed,
			&total,
			rs > 0.9 &&
			ls > 0.9 &&
			es > 0.9 &&
			math.abs(rs - es) < 0.05 &&
			math.abs(ls - es) < 0.05,
			"Ridge / lasso / elastic net grid search: comparable R² on data with an intercept",
		)
		_, rsa := ml.grid_search_ridge(&XA, y, 3, rg)
		expect(
			&passed,
			&total,
			math.abs(rsa - rs) < 1e-10,
			"Ridge grid search: a ones column in X gives the same score (no second intercept)",
		)
	}

	// --- model files keep the hyper-parameters ---
	{
		dir, terr := os.temp_directory(context.allocator)
		path := fmt.aprintf("%s/wotan_ml_fixes_test.wotan", dir if terr == nil else ".")
		defer delete(path)
		defer delete(dir)
		defer os.remove(path)

		yb := make([]f64, n)
		defer delete(yb)
		for v, i in y do yb[i] = 1 if v > 5 else 0
		lp := ml.LogisticParams {
			C              = 0.5,
			max_iter       = 300,
			tol            = 1e-8,
			learning_rate  = 1,
			fit_intercept  = true,
			optimizer_type = .LBFGS,
			class_weight   = .Balanced,
		}
		pipe := ml.pipeline_new()
		defer ml.pipeline_free(&pipe)
		ml.pipeline_add_standard_scaler(&pipe)
		ml.pipeline_set_logistic(&pipe, lp)
		ml.pipeline_fit(&pipe, &X, yb)
		before := ml.pipeline_predict(&pipe, &X)
		defer delete(before)

		saved := ml.pipeline_save(&pipe, path)
		loaded, lok := ml.pipeline_load(path)
		defer ml.pipeline_free(&loaded)
		expect(
			&passed,
			&total,
			saved && lok && loaded.model.logistic_params == lp,
			"Pipeline file stores the logistic parameters",
		)
		ml.pipeline_fit(&loaded, &X, yb) // refit without setting the model again
		after := ml.pipeline_predict(&loaded, &X)
		defer delete(after)
		same := true
		for v, i in before {if v != after[i] do same = false}
		same = same && loaded.model.logistic.weights[0] == pipe.model.logistic.weights[0]
		expect(&passed, &total, same, "Loaded pipeline refits to the same model")

		// a format-5 file: the same bytes without the parameter block, version 5
		data, _ := os.read_entire_file(path, context.allocator)
		defer delete(data)
		LOGISTIC_PARAMS_BYTES :: 8 + 8 + 8 + 8 + 4 + 4 + 4
		old := make([]u8, len(data) - LOGISTIC_PARAMS_BYTES)
		defer delete(old)
		copy(old, data)
		old[5], old[6], old[7], old[8] = 5, 0, 0, 0
		_ = os.write_entire_file(path, old)
		v5, v5ok := ml.pipeline_load(path)
		defer ml.pipeline_free(&v5)
		p5 := ml.pipeline_predict(&v5, &X)
		defer delete(p5)
		same5 := v5ok && v5.model.logistic_params == ml.LogisticParams{}
		for v, i in before {if same5 && v != p5[i] do same5 = false}
		expect(&passed, &total, same5, "Format-5 file still loads and predicts (parameters zero)")

		// MLP: hidden_layers come back owned by the loaded pipeline
		rand.reset(3)
		mp := ml.pipeline_new()
		defer ml.pipeline_free(&mp)
		hidden := make([]int, 2)
		defer delete(hidden)
		hidden[0], hidden[1] = 6, 3
		ml.pipeline_set_mlp(
			&mp,
			{
				hidden_layers = hidden,
				activation = .Tanh,
				output_activation = .Linear,
				task = .Regression,
				learning_rate = 0.01,
				max_iter = 50,
				optimizer_type = .Adam,
			},
		)
		ml.pipeline_fit(&mp, &X, y)
		_ = ml.pipeline_save(&mp, path)
		ml_, mok := ml.pipeline_load(path)
		defer ml.pipeline_free(&ml_)
		hl := ml_.model.mlp_params.hidden_layers
		expect(
			&passed,
			&total,
			mok &&
			len(hl) == 2 &&
			hl[0] == 6 &&
			hl[1] == 3 &&
			ml_.model.mlp_params.max_iter == 50 &&
			ml_.model.mlp_params.task == .Regression,
			"MLP pipeline file stores hidden_layers and the training parameters",
		)

		// an unfitted pipeline keeps its parameters too
		up := ml.pipeline_new()
		defer ml.pipeline_free(&up)
		ml.pipeline_set_random_forest(
			&up,
			{n_trees = 7, max_depth = 4, min_samples = 2, bootstrap = true, criterion = .Entropy},
		)
		_ = ml.pipeline_save(&up, path)
		ul, uok := ml.pipeline_load(path)
		defer ml.pipeline_free(&ul)
		expect(
			&passed,
			&total,
			uok && !ul.is_fitted && ul.model.rf_params == up.model.rf_params,
			"Unfitted pipeline file stores the forest parameters",
		)
	}

	// --- tree / forest / boosting evaluators ---
	{
		cv := ml.CVSplitter {
			kind     = .KFold,
			n_splits = 3,
		}
		folds := ml.cv_make_folds(cv, n)
		defer ml.cv_folds_free(folds)
		manual := 0.0
		tp := ml.TreeParams {
			max_depth   = 3,
			min_samples = 2,
		}
		for f in folds {
			Xtr := ml.subset_matrix(&X, f.train)
			defer l.matrix_free(&Xtr)
			ytr := ml.subset_slice(y, f.train)
			defer delete(ytr)
			Xte := ml.subset_matrix(&X, f.test)
			defer l.matrix_free(&Xte)
			yte := ml.subset_slice(y, f.test)
			defer delete(yte)
			tree := ml.dt_fit(&Xtr, ytr, tp)
			defer ml.dt_free(&tree)
			p := ml.dt_predict(&tree, &Xte)
			defer delete(p)
			manual += ml.metrics_r2(yte, p) / 3
		}
		_, dscore := ml.grid_search_cv(&X, y, cv, ml.dt_cv_evaluator, []ml.TreeParams{tp})
		expect(
			&passed,
			&total,
			math.abs(dscore - manual) < 1e-12 && dscore > 0.3,
			"Regression tree grid search scores the mean R²",
		)

		_, gscore := ml.grid_search_cv(
			&X,
			y,
			cv,
			ml.gb_cv_evaluator,
			[]ml.GBParams {
				{n_estimators = 30, learning_rate = 0.1, max_depth = 2, min_samples = 2},
			},
		)
		rand.reset(5)
		_, fscore := ml.grid_search_cv(
			&X,
			y,
			cv,
			ml.rf_cv_evaluator,
			[]ml.RFParams{{n_trees = 20, max_depth = 4, min_samples = 2, bootstrap = true}},
		)
		fmt.printf("   CV R²: tree %.4f, boosting %.4f, forest %.4f\n", dscore, gscore, fscore)
		expect(
			&passed,
			&total,
			gscore > 0.5 && gscore <= 1 && fscore > 0.5 && fscore <= 1,
			"Boosting / forest regressors are scored with R²",
		)

		yb := make([]f64, n)
		defer delete(yb)
		for v, i in y do yb[i] = 1 if v > 5 else 0
		ctp := ml.TreeParams {
			max_depth   = 3,
			min_samples = 2,
			criterion   = .Gini,
		}
		cs := ml.cross_val_score_cv(&X, yb, cv, ml.dt_cv_evaluator, &ctp)
		defer delete(cs)
		acc_like := true // accuracies on 20 test rows are multiples of 1/20
		for s in cs {if math.abs(s * 20 - math.round(s * 20)) > 1e-9 || s < 0 || s > 1 do acc_like = false}
		expect(&passed, &total, acc_like, "Classification tree evaluator still scores accuracy")
	}

	fmt.printf("ML fixes summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// A failed transform_dataset (feature missing) followed by the usual defers, all on
// `allocator`; true if it failed with empty outputs
@(private = "file")
_fixes_failed_transform :: proc(allocator: mem.Allocator) -> bool {
	context.allocator = allocator
	train := w.df_from(
		w.column_from_floats("x", []f64{1, 2, 3}),
		w.column_from_floats("target", []f64{0, 1, 0}),
	)
	defer w.destroy_dataframe(&train)
	ds, _ := ml.prepare_dataset(&train, "target")
	defer ml.dataset_free(&ds)
	test := w.df_from(w.column_from_floats("other", []f64{5, 6}))
	defer w.destroy_dataframe(&test)

	X, y, ok := ml.transform_dataset(&test, &ds) // feature "x" is missing
	defer l.matrix_free(&X)
	defer delete(y)
	return !ok && X.data == nil && y == nil
}
