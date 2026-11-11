package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"
import "core:time"

// Deep Heston calibration: an MLP trained on 4000 random Heston surfaces (implied vols on
// 5 forward moneyness x 4 expiries) maps a surface to its parameters. Scored on 200
// held-out surfaces by the parameter error and, what matters for pricing, the implied-vol
// error of the surface its parameters reproduce, next to Levenberg-Marquardt
// (fin.calibrate_heston) on the same grids. Then AAPL quotes are put on the grid and
// calibrated both ways.
deep_calibration_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep Heston Calibration Test ===")
	context.allocator = allocator
	G :: ml_fin.DEEP_HESTON_N_GRID
	N_TRAIN, N_TEST :: 4000, 200

	t0 := time.now()
	X, Y := ml_fin.generate_heston_calibration_dataset(N_TRAIN + N_TEST, allocator)
	fmt.printf(
		"Dataset: %d surfaces in %.1f s (%d train, %d held out)\n",
		N_TRAIN + N_TEST,
		time.duration_seconds(time.since(t0)),
		N_TRAIN,
		N_TEST,
	)

	model := ml_fin.deep_heston_calibrator_new(128, allocator)
	defer ml_fin.deep_heston_calibrator_free(&model)
	opt := nn.adam_new(1e-3, allocator = allocator)
	defer nn.adam_free(&opt)
	ml_fin.deep_heston_calibrator_add_to_adam(&model, &opt)

	EPOCHS, BATCH :: 150, 128
	steps_per_epoch := N_TRAIN / BATCH
	schedule := optim.lr_step_decay(1e-3, EPOCHS * steps_per_epoch / 3, 0.2)
	order := make([]int, N_TRAIN)
	for &v, i in order {v = i}
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	step := 0
	t0 = time.now()
	for epoch in 0 ..< EPOCHS {
		rand.shuffle(order)
		epoch_loss := 0.0
		for b in 0 ..< steps_per_epoch {
			t.graph_begin(&scope)
			alloc := t._scope_or(allocator)
			xb := l.matrix_new(f64, BATCH, G, alloc)
			yb := l.matrix_new(f64, BATCH, 5, alloc)
			for k in 0 ..< BATCH {
				row := order[b * BATCH + k]
				copy(xb.data[k * G:][:G], X.data.data[row * G:][:G])
				copy(yb.data[k * 5:][:5], Y.data.data[row * 5:][:5])
			}
			nn.adam_zero_grad(&opt)
			pred := ml_fin.deep_heston_calibrator_forward(&model, t.tensor_new(xb, false, alloc))
			loss := t.tensor_mse_loss(pred, t.tensor_new(yb, false, alloc))
			t.tensor_backward(loss)
			nn.adam_step_scheduled(&opt, schedule, step)
			epoch_loss += loss.data.data[0] / f64(steps_per_epoch)
			t.graph_end(&scope)
			step += 1
		}
		if epoch % 30 == 0 || epoch == EPOCHS - 1 {
			fmt.printf("  epoch %3d | scaled-parameter MSE %.5f\n", epoch + 1, epoch_loss)
		}
	}
	fmt.printf("Training: %.1f s\n", time.duration_seconds(time.since(t0)))

	// --- held-out surfaces
	names := [5]string{"v0", "kappa", "theta", "sigma", "rho"}
	se: [5]f64
	iv_errs := make([]f64, N_TEST)
	lm_errs := make([dynamic]f64)
	N_LM :: 10
	nn_time, lm_time: time.Duration
	for s in 0 ..< N_TEST {
		row := N_TRAIN + s
		x := X.data.data[row * G:][:G]
		ivs: [G]f64
		for v, i in x {ivs[i] = 0.25 + 0.1 * v} 	// invert deep_heston_features
		t1 := time.now()
		p := ml_fin.deep_heston_calibrate(&model, ivs[:])
		nn_time += time.since(t1)
		enc := ml_fin.deep_heston_encode(p)
		for k in 0 ..< 5 {
			d := enc[k] - Y.data.data[row * 5 + k]
			se[k] += d * d / N_TEST
		}
		iv_errs[s] = surface_iv_rmse(p, ivs[:])
		if s < N_LM {
			pts := grid_points(ivs[:], 100, 0)
			t1 = time.now()
			res := fin.calibrate_heston(pts, 100, 0)
			lm_time += time.since(t1)
			append(&lm_errs, surface_iv_rmse(res.params, ivs[:]))
		}
	}
	fmt.println("Held-out parameter RMSE, scaled to [-1, 1] (predicting the box centre: 0.577):")
	for k in 0 ..< 5 {fmt.printf("  %-5s %.3f\n", names[k], math.sqrt(se[k]))}
	med := median_of(iv_errs)
	p90 := quantile_of(iv_errs, 0.9)
	fmt.printf(
		"Surface implied-vol RMSE of the predicted parameters: median %.2f vol pts, 90%% %.2f (%.0f us per calibration)\n",
		100 * med,
		100 * p90,
		1e6 * time.duration_seconds(nn_time) / N_TEST,
	)
	fmt.printf(
		"Levenberg-Marquardt on the first %d grids: median %.4f vol pts (%.0f ms per calibration)\n",
		N_LM,
		100 * median_of(lm_errs[:]),
		1e3 * time.duration_seconds(lm_time) / N_LM,
	)
	ok := med < 0.005 && math.sqrt(se[0]) < 0.1 && math.sqrt(se[4]) < 0.2
	fmt.printf(
		"%s deep calibration: median surface error %.2f vol pts (< 0.5), v0 RMSE %.3f (< 0.1), rho RMSE %.3f (< 0.2)\n",
		"OK  " if ok else "FAIL",
		100 * med,
		math.sqrt(se[0]),
		math.sqrt(se[4]),
	)

	// --- AAPL: quotes onto the grid, then both calibrations
	fmt.println("\n=== AAPL option chain ===")
	chain, chain_ok := fin.fetch_yahoo_option_surface("AAPL", allocator = allocator)
	if !chain_ok {fail_missing_data(#procedure, "Yahoo option chain for AAPL"); return}
	df, df_ok := net.read_yahoo("AAPL", .Daily, .OneMonth, allocator)
	if !df_ok {fail_missing_data(#procedure, "Yahoo data for AAPL"); return}
	defer w.destroy_dataframe(&df)
	if chain.n_options == 0 || df.rows == 0 {
		fail_missing_data(#procedure, "Yahoo option chain or spot for AAPL")
		return
	}
	spot, _ := w.column_at_float(w.column(&df, "Close"), df.rows - 1)
	r := 0.04
	// out-of-the-money quotes (puts below the forward, calls above: the liquid side); the
	// implied vols are inverted from the quoted prices at the same r (Yahoo's own
	// impliedVolatility runs several vol points above them on the call side)
	quotes := make([dynamic]fin.VolSurfacePoint)
	for i in 0 ..< chain.n_options {
		K, T := chain.strikes[i], chain.expiries[i]
		F := spot * math.exp(r * T)
		is_put := chain.option_types[i] == "put"
		if is_put != (K < F) || T < 0.05 || K < 0.6 * F || K > 1.5 * F {continue}
		iv, ok, _ := fin.implied_volatility(
			chain.market_prices[i],
			spot,
			K,
			T,
			r,
			.Put if is_put else .Call,
		)
		if !ok || iv < 0.05 || iv > 1.5 {continue}
		append(&quotes, fin.VolSurfacePoint{strike = K, expiry = T, implied_vol = iv})
	}
	grid: [G]f64
	if !ml_fin.deep_heston_grid_from_surface(quotes[:], spot, r, grid[:]) {
		fmt.println("Too few quotes: skipped")
		return
	}
	fmt.printf(
		"Spot %.2f, %d OTM quotes; grid implied vols (rows: expiry 0.25, 0.5, 1, 2):\n",
		spot,
		len(quotes),
	)
	for j in 0 ..< 4 {
		fmt.print("  ")
		for i in 0 ..< 5 {fmt.printf("%.3f ", grid[j * 5 + i])}
		fmt.println()
	}
	p := ml_fin.deep_heston_calibrate(&model, grid[:])
	res := fin.calibrate_heston(grid_points(grid[:], spot, r), spot, r)
	print_p :: proc(label: string, p: fin.Heston_Params, err: f64) {
		fmt.printf(
			"  %-20s v0 %.4f kappa %.2f theta %.4f sigma %.2f rho %+.2f | surface RMSE %.2f vol pts\n",
			label,
			p.v0,
			p.kappa,
			p.theta,
			p.sigma,
			p.rho,
			100 * err,
		)
	}
	print_p("deep calibrator", p, surface_iv_rmse(p, grid[:]))
	print_p("Levenberg-Marquardt", res.params, surface_iv_rmse(res.params, grid[:]))
}

// RMSE between the grid vols a parameter set gives and the target vols (1 when a vol does
// not exist)
@(private = "file")
surface_iv_rmse :: proc(p: fin.Heston_Params, target: []f64) -> f64 {
	model: [ml_fin.DEEP_HESTON_N_GRID]f64
	if !ml_fin.deep_heston_surface(p, model[:]) {return 1}
	s := 0.0
	for v, i in model {s += (v - target[i]) * (v - target[i])}
	return math.sqrt(s / f64(len(target)))
}

// The grid as quotes at spot S and rate r (strikes K = m S e^{rT})
@(private = "file")
grid_points :: proc(ivs: []f64, S, r: f64) -> []fin.VolSurfacePoint {
	M := ml_fin.DEEP_HESTON_MONEYNESS
	E := ml_fin.DEEP_HESTON_EXPIRIES
	pts := make([]fin.VolSurfacePoint, ml_fin.DEEP_HESTON_N_GRID)
	for T, j in E {
		for m, i in M {
			K := m * S * math.exp(r * T)
			iv := ivs[j * len(M) + i]
			pts[j * len(M) + i] = {
				strike       = K,
				expiry       = T,
				implied_vol  = iv,
				market_price = fin.compute_greeks(S, K, T, r, iv, .Call).price,
			}
		}
	}
	return pts
}

@(private = "file")
median_of :: proc(v: []f64) -> f64 {return quantile_of(v, 0.5)}

@(private = "file")
quantile_of :: proc(v: []f64, q: f64) -> f64 {
	s := make([]f64, len(v), context.temp_allocator)
	copy(s, v)
	slice.sort(s)
	return s[int(q * f64(len(s) - 1))]
}
