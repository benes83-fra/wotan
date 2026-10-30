package tests

import ml "../wotan/analytics/ML"
import w "../wotan/core"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

// Model selection, pipelines, serialization and the DataFrame bridge:
// - KFold / train_test_split sizes follow sklearn (first n % k folds larger, ceil test size)
// - pipeline OLS / Ridge = sklearn make_pipeline(Scaler, LinearRegression / Ridge)
//   (intercept included); refitting a pipeline gives the same model
// - save → load → predict is bit-exact (incl. gradient boosting, which used to
//   lose F₀ and the learning rate); truncated / corrupt files fail cleanly
// - prepared datasets do not borrow DataFrame strings; unseen categories → -1

ml_selection_pipeline_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML Model Selection / Pipeline / Serialization ===")
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

	X := l.matrix_new(f64, 8, 2)
	X.data = {5.1, 2.0, 6.3, 1.1, 4.8, 3.3, 7.2, 0.4, 5.9, 2.8, 6.6, 1.9, 5.4, 0.7, 6.9, 2.5}
	y := []f64{12.0, 14.1, 11.2, 16.3, 13.5, 15.2, 12.1, 15.9}

	// --- folds / splits ---
	f := ml.cv_folds(10, 3, 7)
	cnt := [3]int{}
	for v in f do cnt[v] += 1
	expect(&passed, &total, cnt == {4, 3, 3}, "cv_folds sizes = sklearn KFold (4, 3, 3)")
	sp := ml.train_test_split(&X, y, 0.3)
	expect(
		&passed,
		&total,
		sp.X_train.rows == 5 && sp.X_test.rows == 3,
		"train_test_split: n_test = ceil(0.3 · 8) = 3",
	)

	// --- pipelines (intercept) ---
	p := ml.pipeline_new()
	ml.pipeline_add_standard_scaler(&p)
	ml.pipeline_set_ols(&p)
	ml.pipeline_fit(&p, &X, y)
	expect(
		&passed,
		&total,
		close(
			ml.pipeline_predict(&p, &X)[:3],
			{11.720685643761568, 14.294821876187441, 11.244263434171335},
			1e-12,
		),
		"Pipeline(StandardScaler, OLS) = sklearn, with intercept",
	)
	ml.pipeline_fit(&p, &X, y)
	expect(
		&passed,
		&total,
		close(
			ml.pipeline_predict(&p, &X)[:3],
			{11.720685643761568, 14.294821876187441, 11.244263434171335},
			1e-12,
		),
		"Pipeline refit gives the same model",
	)
	ml.pipeline_free(&p)
	pr := ml.pipeline_new()
	ml.pipeline_add_minmax_scaler(&pr)
	ml.pipeline_set_ridge(&pr, {lambda = 1.5})
	ml.pipeline_fit(&pr, &X, y)
	expect(
		&passed,
		&total,
		close(
			ml.pipeline_predict(&pr, &X)[:3],
			{13.019321596471395, 14.107485020177503, 12.60439075796313},
			1e-12,
		),
		"Pipeline(MinMaxScaler, Ridge(1.5)) = sklearn",
	)
	ml.pipeline_free(&pr)

	// --- serialization ---
	path := "ml_selection_test_model.wotan"
	defer os.remove(path)
	roundtrip :: proc(pipe: ^ml.Pipeline, X: ^l.Matrix(f64), y: []f64, path: string) -> bool {
		ml.pipeline_fit(pipe, X, y)
		before := ml.pipeline_predict(pipe, X)
		if !ml.pipeline_save(pipe, path) do return false
		loaded, ok := ml.pipeline_load(path)
		defer ml.pipeline_free(&loaded)
		if !ok do return false
		after := ml.pipeline_predict(&loaded, X)
		for v, i in before {if transmute(u64)v != transmute(u64)after[i] do return false}
		return true
	}
	gb := ml.pipeline_new()
	ml.pipeline_set_gradient_boosting(
		&gb,
		{n_estimators = 10, learning_rate = 0.1, max_depth = 2, min_samples = 2, subsample = 1},
	)
	expect(
		&passed,
		&total,
		roundtrip(&gb, &X, y, path),
		"Gradient boosting save/load is bit-exact (F₀ and learning rate stored)",
	)
	ml.pipeline_free(&gb)
	ol := ml.pipeline_new()
	ml.pipeline_add_standard_scaler(&ol)
	ml.pipeline_set_ols(&ol)
	expect(
		&passed,
		&total,
		roundtrip(&ol, &X, y, path),
		"OLS pipeline (with intercept) save/load is bit-exact",
	)
	data, _ := os.read_entire_file(path, context.allocator)
	_ = os.write_entire_file(path, data[:len(data) / 2])
	bad, bad_ok := ml.pipeline_load(path)
	expect(&passed, &total, !bad_ok, "Truncated model file is rejected")
	ml.pipeline_free(&bad)
	huge := make([]u8, len(data))
	copy(huge, data)
	for k in 17 ..< 25 do huge[k] = 0x7f // absurd step count
	_ = os.write_entire_file(path, huge)
	hb, hb_ok := ml.pipeline_load(path)
	expect(&passed, &total, !hb_ok, "Model file with an absurd length field is rejected")
	ml.pipeline_free(&hb)
	ml.pipeline_free(&ol)

	// --- DataFrame bridge ---
	df := w.df_from(
		w.column_from_floats("x", []f64{1, 2, 3, 4}),
		w.column_from_strings("c", []string{"red", "blue", "red", "green"}),
		w.column_from_floats("target", []f64{0, 1, 0, 1}),
	)
	ds, ok := ml.prepare_dataset(&df, "target")
	w.destroy_dataframe(&df)
	expect(
		&passed,
		&total,
		ok && ds.feature_names[1] == "c" && ds.encoders[0].classes[2] == "green",
		"prepare_dataset owns its strings (valid after the DataFrame is freed)",
	)
	df2 := w.df_from(
		w.column_from_floats("x", []f64{5, 6}),
		w.column_from_strings("c", []string{"blue", "purple"}),
	)
	Xt, yt, tok := ml.transform_dataset(&df2, &ds)
	expect(
		&passed,
		&total,
		tok && close(Xt.data, {5, 1, 6, -1}, 0),
		"transform_dataset: unseen category → -1",
	)
	l.matrix_free(&Xt)
	delete(yt)
	w.destroy_dataframe(&df2)
	ml.dataset_free(&ds)

	fmt.printf("ML selection/pipeline summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
