package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// N-BEATS (generic, interpretable) and PatchTST out of sample:
//   A. 600 steps of 0.02 t + 2 sin(2π t / 12) + 0.5 sin(2π t / 6) + N(0, 0.3²); windows of
//      36 -> 12, trained on the first 450 steps, tested on the windows after: every model
//      beats the naive (last value) and the seasonal naive (the last 12 values) forecasts.
//      (The interpretable model, two weight-shared blocks, trains 200 epochs at lr 3e-3;
//      the others 60 at 1e-3.)
//   B. PatchTST on two channels of the same pattern at scales 1 and 100 (RevIN): one model
//      beats the seasonal naive forecast on both.
//   C. SPY log prices, 60 days -> 5: no model beats the random walk (MSE ratio > 0.95);
//      trained without early stopping both fit noise and lose to it.
// N-BEATS runs with relative = true: the trending series of A and the price levels of C
// otherwise reach inputs far outside the training range (generic 0.23 instead of 0.14
// on A, 1.7x the random walk's MSE on C).

// a tensor holding a copy of X (free with tensor_free)
@(private = "file")
_fm_tensor :: proc(X: ^l.Matrix(f64)) -> ^t.Tensor {
	m := l.matrix_new(f64, X.rows, X.cols)
	copy(m.data, X.data)
	return t.tensor_new(m)
}

@(private = "file")
_fm_mse :: proc(a, b: []f64) -> f64 {
	s := 0.0
	for v, i in a {s += (v - b[i]) * (v - b[i])}
	return s / f64(len(a))
}

// naive and seasonal-naive forecasts of windows X [n, I] for horizon H (season s)
@(private = "file")
_fm_baselines :: proc(X: ^l.Matrix(f64), H, season: int) -> (naive, snaive: []f64) {
	n, I := X.rows, X.cols
	naive = make([]f64, n * H, context.temp_allocator)
	snaive = make([]f64, n * H, context.temp_allocator)
	for r in 0 ..< n {
		for j in 0 ..< H {
			naive[r * H + j] = X.data[r * I + I - 1]
			snaive[r * H + j] = X.data[r * I + I - season + j % season]
		}
	}
	return
}

// Mini-batch Adam on (X, Y) with batches of `batch` windows in random order
@(private = "file")
_fm_train_nbeats :: proc(m: ^ml_fin.NBeats, X, Y: ^l.Matrix(f64), epochs, batch: int, lr: f64) {
	opt := nn.adam_new(lr)
	defer nn.adam_free(&opt)
	params := ml_fin.nbeats_parameters(m)
	defer delete(params)
	for q in params {nn.adam_add_param(&opt, q)}
	n, I, H := X.rows, X.cols, Y.cols
	order := make([]int, n)
	defer delete(order)
	for &o, i in order {o = i}
	for _ in 0 ..< epochs {
		rand.shuffle(order)
		for s := 0; s < n; s += batch {
			b := min(batch, n - s)
			xb := t.tensor_new(l.matrix_new(f64, b, I))
			yb := t.tensor_new(l.matrix_new(f64, b, H))
			for k in 0 ..< b {
				copy(xb.data.data[k * I:][:I], X.data[order[s + k] * I:][:I])
				copy(yb.data.data[k * H:][:H], Y.data[order[s + k] * H:][:H])
			}
			ml_fin.nbeats_train_step(m, xb, yb, &opt)
			t.tensor_free(xb)
			t.tensor_free(yb)
		}
	}
}

@(private = "file")
_fm_train_patchtst :: proc(
	m: ^ml_fin.PatchTST,
	X, Y: ^l.Matrix(f64),
	channels: []int,
	epochs, batch: int,
	lr: f64,
) {
	opt := nn.adam_new(lr)
	defer nn.adam_free(&opt)
	params := ml_fin.patchtst_parameters(m)
	defer delete(params)
	for q in params {nn.adam_add_param(&opt, q)}
	n, I, H := X.rows, X.cols, Y.cols
	order := make([]int, n)
	defer delete(order)
	for &o, i in order {o = i}
	ch := make([]int, batch)
	defer delete(ch)
	for _ in 0 ..< epochs {
		rand.shuffle(order)
		for s := 0; s < n; s += batch {
			b := min(batch, n - s)
			xb := t.tensor_new(l.matrix_new(f64, b, I))
			yb := t.tensor_new(l.matrix_new(f64, b, H))
			for k in 0 ..< b {
				copy(xb.data.data[k * I:][:I], X.data[order[s + k] * I:][:I])
				copy(yb.data.data[k * H:][:H], Y.data[order[s + k] * H:][:H])
				ch[k] = channels[order[s + k]] if channels != nil else 0
			}
			ml_fin.patchtst_train_step(m, xb, yb, ch[:b], &opt)
			t.tensor_free(xb)
			t.tensor_free(yb)
		}
	}
}

@(private = "file")
_fm_pattern :: proc(n: int, scale: f64) -> []f64 {
	s := make([]f64, n)
	for i in 0 ..< n {
		ti := f64(i)
		s[i] =
			scale *
			(0.02 * ti +
					2 * math.sin(2 * math.PI * ti / 12) +
					0.5 * math.sin(2 * math.PI * ti / 6) +
					rand.float64_normal(0, 0.3))
	}
	return s
}

forecasting_models_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== N-BEATS and PatchTST, out of sample ===")
	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}
	I, H :: 36, 12

	// A. trend + seasonality
	{
		s := _fm_pattern(600, 1)
		defer delete(s)
		Xa, Ya := ml_fin.forecast_windows(s[:450], I, H)
		defer {l.matrix_free(&Xa); l.matrix_free(&Ya)}
		Xt, Yt := ml_fin.forecast_windows(s[450 - I:], I, H)
		defer {l.matrix_free(&Xt); l.matrix_free(&Yt)}
		naive, snaive := _fm_baselines(&Xt, H, 12)
		mse_n, mse_s := _fm_mse(naive, Yt.data), _fm_mse(snaive, Yt.data)
		xt := _fm_tensor(&Xt)
		models := [3]string{"N-BEATS generic", "N-BEATS interpretable", "PatchTST"}
		mse: [3]f64
		{
			m := ml_fin.nbeats_new(
				{
					input_size = I,
					horizon = H,
					blocks = 3,
					layers = 2,
					layer_size = 64,
					relative = true,
				},
			)
			defer ml_fin.nbeats_free(m)
			_fm_train_nbeats(m, &Xa, &Ya, 60, 32, 1e-3)
			f := ml_fin.nbeats_predict(m, xt)
			defer delete(f)
			mse[0] = _fm_mse(f, Yt.data)
		}
		{
			m := ml_fin.nbeats_new(
				{
					input_size = I,
					horizon = H,
					interpretable = true,
					blocks = 2,
					layers = 2,
					layer_size = 32,
					seasonality_layer_size = 64,
					degree = 2,
					harmonics = 1,
					relative = true,
				},
			)
			defer ml_fin.nbeats_free(m)
			_fm_train_nbeats(m, &Xa, &Ya, 200, 32, 3e-3)
			f := ml_fin.nbeats_predict(m, xt)
			defer delete(f)
			mse[1] = _fm_mse(f, Yt.data)
		}
		{
			m := ml_fin.patchtst_new(
				{
					input_size = I,
					horizon = H,
					patch_len = 6,
					stride = 3,
					d_model = 16,
					n_heads = 2,
					d_ff = 32,
					n_layers = 2,
					channels = 1,
					revin = true,
				},
			)
			defer ml_fin.patchtst_free(m)
			_fm_train_patchtst(m, &Xa, &Ya, nil, 60, 32, 1e-3)
			f := ml_fin.patchtst_predict(m, xt, nil)
			defer delete(f)
			mse[2] = _fm_mse(f, Yt.data)
		}
		t.tensor_free(xt)
		ok := true
		for v in mse {ok &&= v < mse_s && v < mse_n}
		check(
			&failures,
			ok,
			fmt.tprintf(
				"trend + seasonality, %d test windows: MSE %s %.3f, %s %.3f, %s %.3f vs naive %.3f, seasonal naive %.3f (noise alone 0.090)",
				Xt.rows,
				models[0],
				mse[0],
				models[1],
				mse[1],
				models[2],
				mse[2],
				mse_n,
				mse_s,
			),
		)
	}

	// B. two channels at scales 1 and 100
	{
		X := [2]l.Matrix(f64){}
		Y := [2]l.Matrix(f64){}
		Xt := [2]l.Matrix(f64){}
		Yt := [2]l.Matrix(f64){}
		for c in 0 ..< 2 {
			s := _fm_pattern(600, 1 if c == 0 else 100)
			defer delete(s)
			X[c], Y[c] = ml_fin.forecast_windows(s[:450], I, H)
			Xt[c], Yt[c] = ml_fin.forecast_windows(s[450 - I:], I, H)
		}
		defer for c in 0 ..< 2 {l.matrix_free(&X[c]); l.matrix_free(&Y[c]); l.matrix_free(&Xt[c]); l.matrix_free(&Yt[c])}
		n := X[0].rows
		Xa := l.matrix_new(f64, 2 * n, I)
		defer l.matrix_free(&Xa)
		Ya := l.matrix_new(f64, 2 * n, H)
		defer l.matrix_free(&Ya)
		ch := make([]int, 2 * n)
		defer delete(ch)
		for c in 0 ..< 2 {
			copy(Xa.data[c * n * I:][:n * I], X[c].data)
			copy(Ya.data[c * n * H:][:n * H], Y[c].data)
			for k in 0 ..< n {ch[c * n + k] = c}
		}
		m := ml_fin.patchtst_new(
			{
				input_size = I,
				horizon = H,
				patch_len = 6,
				stride = 3,
				d_model = 16,
				n_heads = 2,
				d_ff = 32,
				n_layers = 2,
				channels = 2,
				revin = true,
			},
		)
		defer ml_fin.patchtst_free(m)
		_fm_train_patchtst(m, &Xa, &Ya, ch, 40, 32, 1e-3)
		ratio: [2]f64
		for c in 0 ..< 2 {
			_, sn := _fm_baselines(&Xt[c], H, 12)
			xt := _fm_tensor(&Xt[c])
			cc := make([]int, Xt[c].rows, context.temp_allocator)
			for &v in cc {v = c}
			f := ml_fin.patchtst_predict(m, xt, cc)
			ratio[c] = _fm_mse(f, Yt[c].data) / _fm_mse(sn, Yt[c].data)
			delete(f)
			t.tensor_free(xt)
		}
		check(
			&failures,
			ratio[0] < 1 && ratio[1] < 1,
			fmt.tprintf(
				"PatchTST, one model for channels at scales 1 and 100 (RevIN): MSE / seasonal naive %.2f and %.2f",
				ratio[0],
				ratio[1],
			),
		)
	}

	// C. SPY log prices
	{
		df, df_ok := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
		if !df_ok {fail_missing_data(#procedure, "Yahoo data for SPY"); return}
		defer w.destroy_dataframe(&df)
		if df.rows < 600 {
			fail_missing_data(
				#procedure,
				fmt.tprintf("600 days of Yahoo data for SPY (got %d)", df.rows),
			)
		} else {
			pc := w.column(&df, "AdjClose")
			lp := make([dynamic]f64)
			defer delete(lp)
			for i in 0 ..< df.rows {
				if v, null := w.column_at_float(pc, i);
				   !null && v > 0 {append(&lp, 100 * math.ln(v))}
			}
			IS, HS :: 60, 5
			cut := len(lp) * 3 / 4
			Xa, Ya := ml_fin.forecast_windows(lp[:cut], IS, HS)
			defer {l.matrix_free(&Xa); l.matrix_free(&Ya)}
			Xt, Yt := ml_fin.forecast_windows(lp[cut - IS:], IS, HS)
			defer {l.matrix_free(&Xt); l.matrix_free(&Yt)}
			naive, _ := _fm_baselines(&Xt, HS, 1)
			mse_n := _fm_mse(naive, Yt.data)
			xt := _fm_tensor(&Xt)
			defer t.tensor_free(xt)
			nb := ml_fin.nbeats_new(
				{
					input_size = IS,
					horizon = HS,
					blocks = 3,
					layers = 2,
					layer_size = 64,
					relative = true,
				},
			)
			defer ml_fin.nbeats_free(nb)
			_fm_train_nbeats(nb, &Xa, &Ya, 20, 32, 1e-3)
			f1 := ml_fin.nbeats_predict(nb, xt)
			defer delete(f1)
			pt := ml_fin.patchtst_new(
				{
					input_size = IS,
					horizon = HS,
					patch_len = 10,
					stride = 5,
					d_model = 16,
					n_heads = 2,
					d_ff = 32,
					n_layers = 2,
					channels = 1,
					revin = true,
				},
			)
			defer ml_fin.patchtst_free(pt)
			_fm_train_patchtst(pt, &Xa, &Ya, nil, 20, 32, 1e-3)
			f2 := ml_fin.patchtst_predict(pt, xt, nil)
			defer delete(f2)
			r1, r2 := _fm_mse(f1, Yt.data) / mse_n, _fm_mse(f2, Yt.data) / mse_n
			check(
				&failures,
				r1 > 0.95 && r2 > 0.95,
				fmt.tprintf(
					"SPY log prices, %d test windows of 60 -> 5 days: no model beats the random walk (MSE relative to it: N-BEATS %.3f, PatchTST %.3f; fitted noise without early stopping)",
					Xt.rows,
					r1,
					r2,
				),
			)
		}
	}
	fmt.printf("Forecasting models: %d failure(s)\n", failures)
	record_failures(failures)
}
