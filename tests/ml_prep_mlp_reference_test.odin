package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Preprocessing / PCR / metrics / MLP against scikit-learn 1.2.1 (reference
// values hardcoded) and a finite-difference check:
// - StandardScaler (incl. a 1e-6-scale feature and a constant feature), MinMaxScaler
//   and both inverse transforms
// - PCR = sklearn make_pipeline(PCA(k), LinearRegression())
// - r2_score for constant y_true, OneHotEncoder(drop_first) with an unknown category
// - MLP backprop gradients = central differences of its loss; mini-batch training fits

ml_prep_mlp_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML Preprocessing / PCR / Metrics / MLP vs sklearn ===")
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

	X := l.matrix_new(f64, 6, 3)
	X.data = {
		1.0,
		200.0,
		3e-6,
		2.0,
		180.0,
		1e-6,
		0.5,
		220.0,
		4e-6,
		3.0,
		150.0,
		2e-6,
		1.5,
		210.0,
		5e-6,
		2.5,
		170.0,
		2.5e-6,
	}
	y := []f64{3.1, 4.0, 2.2, 6.5, 3.0, 5.1}

	// --- Scalers ---
	ss := ml.standard_scaler_fit(&X)
	Xs := ml.standard_scaler_transform(&ss, &X)
	expect(
		&passed,
		&total,
		close(ss.std, {0.8539125638299665, 24.094720491334932, 1.3043729868748775e-06}, 1e-12),
		"StandardScaler std (no +1e-8 bias on a 1e-6-scale feature)",
	)
	expect(
		&passed,
		&total,
		close(Xs.data[:3], {-0.8783100656536799, 0.4842001247062519, 0.06388765649999413}, 1e-12),
		"StandardScaler transform = sklearn",
	)
	Xb := ml.standard_scaler_inverse_transform(&ss, &Xs)
	expect(&passed, &total, close(Xb.data, X.data, 1e-12), "StandardScaler inverse round trip")
	Cm := l.matrix_new(f64, 3, 2)
	Cm.data = {1, 2, 1, 3, 1, 4}
	sc := ml.standard_scaler_fit(&Cm)
	cs := ml.standard_scaler_transform(&sc, &Cm)
	expect(
		&passed,
		&total,
		cs.data[0] == 0 && cs.data[2] == 0 && sc.std[0] == 1,
		"StandardScaler: constant feature gets scale 1",
	)
	mm := ml.minmax_scaler_fit(&X, {-1, 2})
	Xm := ml.minmax_scaler_transform(&mm, &X)
	expect(
		&passed,
		&total,
		close(Xm.data[3:6], {0.7999999999999998, 0.2857142857142856, -1.0}, 1e-12),
		"MinMaxScaler(-1, 2) = sklearn",
	)
	Xmb := ml.minmax_scaler_inverse_transform(&mm, &Xm)
	expect(&passed, &total, close(Xmb.data, X.data, 1e-12), "MinMaxScaler inverse round trip")

	// --- PCR ---
	p1 := ml.pcr_fit(&X, y, {n_components = 1})
	expect(
		&passed,
		&total,
		close(
			ml.pcr_predict(&p1, &X)[:3],
			{3.2954417661236484, 4.474117354429239, 2.1177585999123365},
			1e-10,
		),
		"PCR(k=1) = PCA + LinearRegression",
	)
	p2 := ml.pcr_fit(&X, y, {n_components = 2})
	expect(
		&passed,
		&total,
		close(
			ml.pcr_predict(&p2, &X)[:3],
			{3.1191860465103405, 4.458720930230919, 2.0279069767435445},
			1e-10,
		),
		"PCR(k=2) = PCA + LinearRegression",
	)

	// --- Metrics / encoders ---
	cst := []f64{2, 2, 2}
	expect(
		&passed,
		&total,
		ml.metrics_r2(cst, cst) == 1 && ml.metrics_r2(cst, []f64{2, 2, 3}) == 0,
		"r2 with constant y_true = sklearn (1 if perfect, else 0)",
	)
	expect(
		&passed,
		&total,
		math.is_nan(ml.metrics_mse([]f64{}, []f64{})),
		"MSE of empty input is NaN, not 0",
	)
	ohe := ml.ohe_fit([]string{"b", "a", "c", "a"}, true)
	expect(
		&passed,
		&total,
		close(ml.ohe_transform(&ohe, []string{"a", "c", "z", "b"}), {0, 0, 0, 1, 0, 0, 1, 0}, 0),
		"OneHotEncoder(drop_first): sorted categories, unknown → zeros",
	)

	// --- MLP: backprop = finite differences ---
	rand.reset(9)
	Xg := l.matrix_new(f64, 7, 3)
	for &v in Xg.data do v = rand.float64_normal(0, 1)
	yreg := []f64{0.3, -1.2, 0.8, 2.0, -0.5, 0.1, 1.1}
	ybin := []f64{0, 1, 1, 0, 1, 0, 1}
	worst := 0.0
	for cfg in ([][2]int{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {1, 1}, {2, 1}}) {
		act := ml.Activation(cfg[0])
		task := ml.MLPTask(cfg[1])
		outa := ml.Activation.Linear if task == .Regression else ml.Activation.Sigmoid
		yy := yreg if task == .Regression else ybin
		m := ml.mlp_fit(
			&Xg,
			yy,
			{
				hidden_layers = {4, 3},
				activation = act,
				output_activation = outa,
				task = task,
				learning_rate = 0.01,
				max_iter = 3,
				optimizer_type = .Adam,
			},
		)
		dW := make([]l.Matrix(f64), m.n_layers)
		db := make([][]f64, m.n_layers)
		for i in 0 ..< m.n_layers {dW[i] = l.matrix_new(f64, m.weights[i].rows, m.weights[i].cols); db[i] = make([]f64, m.weights[i].cols)}
		ml._mlp_forward_backward(
			m.weights,
			m.biases,
			m.activations,
			task,
			outa,
			Xg.data,
			yy,
			dW,
			db,
		)
		g := make([]f64, len(dW[0].data))
		copy(g, dW[0].data) // first layer: the gradient that passes through every later layer
		h := 1e-6
		for idx in 0 ..< len(g) {
			w0 := m.weights[0].data[idx]
			m.weights[0].data[idx] = w0 + h
			lp := ml._mlp_forward_backward(
				m.weights,
				m.biases,
				m.activations,
				task,
				outa,
				Xg.data,
				yy,
				dW,
				db,
			)
			m.weights[0].data[idx] = w0 - h
			lm := ml._mlp_forward_backward(
				m.weights,
				m.biases,
				m.activations,
				task,
				outa,
				Xg.data,
				yy,
				dW,
				db,
			)
			m.weights[0].data[idx] = w0
			fd := (lp - lm) / (2 * h)
			worst = max(worst, abs(fd - g[idx]) / max(abs(fd) + abs(g[idx]), 1e-8))
		}
	}
	expect(
		&passed,
		&total,
		worst < 1e-5,
		fmt.tprintf("MLP backprop = finite differences, all activations (rel err %.1e)", worst),
	)

	rand.reset(2)
	Xt := l.matrix_new(f64, 200, 1)
	yt := make([]f64, 200)
	for i in 0 ..< 200 {Xt.data[i] = -3 + 6 * f64(i) / 199; yt[i] = math.sin(Xt.data[i])}
	mt := ml.mlp_fit(
		&Xt,
		yt,
		{
			hidden_layers = {16, 16},
			activation = .Tanh,
			output_activation = .Linear,
			task = .Regression,
			learning_rate = 0.01,
			max_iter = 300,
			batch_size = 32,
			optimizer_type = .Adam,
		},
	)
	mse := ml.metrics_mse(yt, ml.mlp_predict(&mt, &Xt))
	expect(
		&passed,
		&total,
		mse < 0.01,
		fmt.tprintf("MLP mini-batch Adam fits sin(x) (MSE %.1e)", mse),
	)

	fmt.printf("ML prep/MLP summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
