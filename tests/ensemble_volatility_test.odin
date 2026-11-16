package tests

import ts "../wotan/analytics"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// GARCH(1,1) and a sequence model (LSTM / GRU / Mamba) forecasting SPY's realised
// volatility, their ensemble, the volatility risk premium (VIX against the forecast)
// and a walk-forward VRP backtest, on Yahoo data for SPY and the VIX.
//
// Shared pipeline (the _ev_* helpers below):
// - day i carries the information through SPY's return r_i (from close i to close
//   i + 1) and the VIX at close i + 1;
// - the features of day i are r_i, |r_i|, the standard deviation of r_{i-19..i} and
//   the VIX (as a decimal);
// - a sample ending on day e is the features of days e - 19 .. e, and its target is
//   the realised volatility of the next h returns, sqrt(mean(r²_{e+1..e+h})), in
//   daily units;
// - the model learns the logarithm of the target, standardised with the training
//   rows' mean and standard deviation (as are the features); its forecast is the
//   exponential of the scaled-back output, so it is always positive. Raw targets
//   (~0.01) against network outputs of order 1 would leave the model far from the
//   targets' scale, and an unconstrained output can go negative;
// - GARCH(1,1) runs its conditional variance over every return, so the forecast on
//   day e uses r_0..r_e, and averages the variance path over the next h days
//   (fin.garch_term_structure_vol).
// The model trains with early stopping on validation days (_ev_fit).
// Each test checks that the forecasts are plausible: annualised between 3 % and
// 150 %, and the sequence model's out-of-sample error within 1.5 times GARCH's.

@(private = "file")
EV_WINDOW :: 20 // days in the trailing standard deviation feature
@(private = "file")
EV_SEQ :: 20 // days in a sample
@(private = "file")
EV_NF :: 4 // features per day

@(private = "file")
_EV :: struct {
	returns: []f64, // r_i: SPY log return from close i to close i + 1
	vix:     []f64, // VIX at close i + 1, as a decimal
	n:       int,
}

// First day whose sample is complete (the trailing deviation needs EV_WINDOW returns)
@(private = "file")
EV_FIRST_END :: EV_WINDOW - 1 + EV_SEQ - 1

@(private = "file")
_ev_load :: proc(r: yahoo.range, allocator: mem.Allocator) -> (d: _EV, ok: bool) {
	panel, panel_ok, _ := yahoo.read_yahoo_panel({"SPY", "^VIX"}, .Daily, r, allocator)
	defer yahoo.yahoo_panel_free(&panel)
	if !panel_ok {return}
	all_returns := ts.log_returns(panel.adj_close, 2, allocator) // [day][SPY, VIX]
	defer delete(all_returns, allocator)
	d.n = len(panel.dates) - 1
	d.returns = make([]f64, d.n, allocator)
	d.vix = make([]f64, d.n, allocator)
	for i in 0 ..< d.n {
		d.returns[i] = all_returns[i * 2]
		d.vix[i] = panel.adj_close[(i + 1) * 2 + 1] / 100
	}
	return d, true
}

@(private = "file")
_ev_free :: proc(d: ^_EV, allocator: mem.Allocator) {
	delete(d.returns, allocator)
	delete(d.vix, allocator)
}

@(private = "file")
_ev_feature :: proc(d: ^_EV, i: int, out: []f64) {
	r := d.returns[i]
	s, s2 := 0.0, 0.0
	for j in i - EV_WINDOW + 1 ..= i {s += d.returns[j]; s2 += d.returns[j] * d.returns[j]}
	m := s / EV_WINDOW
	out[0], out[1], out[2], out[3] = r, abs(r), math.sqrt(max(s2 / EV_WINDOW - m * m, 0)), d.vix[i]
}

// Realised volatility of the next h returns after day e (daily units)
@(private = "file")
_ev_target :: proc(d: ^_EV, e, h: int) -> f64 {
	s := 0.0
	for j in e + 1 ..= e + h {s += d.returns[j] * d.returns[j]}
	return math.sqrt(s / f64(h))
}

// Trailing realised volatility of r_{e-h+1..e} (daily units)
@(private = "file")
_ev_trailing :: proc(d: ^_EV, e, h: int) -> f64 {
	s := 0.0
	for j in e - h + 1 ..= e {s += d.returns[j] * d.returns[j]}
	return math.sqrt(s / f64(h))
}

@(private = "file")
_EV_Garch :: struct {
	omega, alpha, beta: f64,
	var_next:           []f64, // σ²_{t+1} after observing r_t
}

// Fits GARCH(1,1) (Student t) to the demeaned returns r_0..r_last, then runs the
// conditional variance over all returns (demeaned with the same mean)
@(private = "file")
_ev_garch :: proc(d: ^_EV, last: int, max_iter: int, allocator: mem.Allocator) -> _EV_Garch {
	resid := ts.extract_residuals(d.returns[:last + 1], allocator)
	defer delete(resid, allocator)
	res := ts.garch_fit(resid, .StudentT, 1, 1, max_iter, 1e-4, allocator)
	defer ts.garch_result_free(&res, allocator)
	mu := d.returns[0] - resid[0]
	g := _EV_Garch {
		omega = res.params.omega,
		alpha = res.params.alpha[0],
		beta  = res.params.beta[0],
	}
	g.var_next = make([]f64, d.n, allocator)
	v := res.conditional_var[0]
	for i in 0 ..< d.n {
		e := d.returns[i] - mu
		v = max(g.omega + g.alpha * e * e + g.beta * v, 1e-12)
		g.var_next[i] = v
	}
	return g
}

// GARCH's forecast on day e of the mean daily volatility over the next h days
@(private = "file")
_ev_garch_vol :: proc(g: ^_EV_Garch, e, h: int) -> f64 {
	return(
		fin.garch_term_structure_vol(g.omega, g.alpha, g.beta, g.var_next[e], h) /
		math.sqrt_f64(252) \
	)
}

@(private = "file")
_EV_Model :: struct {
	ens:        ml_fin.EnsembleVolatilityForecaster,
	opt:        nn.Adam,
	scope:      t.Graph_Scope,
	fmean, fsd: [EV_NF]f64,
	ymean, ysd: f64,
	allocator:  mem.Allocator,
}

@(private = "file")
_ev_model_new :: proc(
	backend: nn.SeqBackend,
	hidden: int,
	lr: f64,
	allocator: mem.Allocator,
) -> ^_EV_Model {
	md := new(_EV_Model, allocator)
	md.allocator = allocator
	md.ens = ml_fin.ensemble_volatility_new(EV_NF, hidden, EV_SEQ, allocator, backend)
	md.opt = nn.adam_new(lr, 0.9, 0.999, 1e-8, allocator)
	ml_fin.ensemble_add_to_optimizer(&md.ens, &md.opt)
	return md
}

@(private = "file")
_ev_model_free :: proc(md: ^_EV_Model) {
	t.graph_scope_destroy(&md.scope)
	nn.adam_free(&md.opt)
	ml_fin.ensemble_volatility_free(&md.ens)
	free(md, md.allocator)
}

// Standardised input window of the sample ending on day e, written to x
@(private = "file")
_ev_window :: proc(md: ^_EV_Model, d: ^_EV, e: int, x: []f64) {
	for k in 0 ..< EV_SEQ {
		row := x[k * EV_NF:(k + 1) * EV_NF]
		_ev_feature(d, e - EV_SEQ + 1 + k, row)
		for f in 0 ..< EV_NF {row[f] = (row[f] - md.fmean[f]) / md.fsd[f]}
	}
}

// Sets the standardisation from the samples ending on days ends[...] (features of
// every day they cover, and their h-day targets), then trains for `epochs` passes in
// shuffled mini-batches. Returns the last epoch's mean squared error, in units of the
// variance of ln(target).
@(private = "file")
_ev_train :: proc(md: ^_EV_Model, d: ^_EV, ends: []int, h, epochs, batch: int) -> f64 {
	ns := len(ends)
	feat: [EV_NF]f64
	md.fmean, md.fsd = {}, {}
	rows := 0
	for day in ends[0] - EV_SEQ + 1 ..= ends[ns - 1] {
		_ev_feature(d, day, feat[:])
		for f in 0 ..< EV_NF {md.fmean[f] += feat[f]}
		rows += 1
	}
	for f in 0 ..< EV_NF {md.fmean[f] /= f64(rows)}
	for day in ends[0] - EV_SEQ + 1 ..= ends[ns - 1] {
		_ev_feature(d, day, feat[:])
		for f in 0 ..< EV_NF {md.fsd[f] += (feat[f] - md.fmean[f]) * (feat[f] - md.fmean[f])}
	}
	for f in 0 ..< EV_NF {md.fsd[f] = max(math.sqrt(md.fsd[f] / f64(rows)), 1e-8)}

	Y := make([]f64, ns, context.temp_allocator)
	md.ymean, md.ysd = 0, 0
	for e, i in ends {Y[i] = math.ln(_ev_target(d, e, h)); md.ymean += Y[i]}
	md.ymean /= f64(ns)
	for y in Y {md.ysd += (y - md.ymean) * (y - md.ymean)}
	md.ysd = max(math.sqrt(md.ysd / f64(ns)), 1e-12)
	X := make([]f64, ns * EV_SEQ * EV_NF, context.temp_allocator)
	for e, i in ends {
		_ev_window(md, d, e, X[i * EV_SEQ * EV_NF:(i + 1) * EV_SEQ * EV_NF])
		Y[i] = (Y[i] - md.ymean) / md.ysd
	}

	order := make([]int, ns, context.temp_allocator)
	for i in 0 ..< ns {order[i] = i}
	loss_sum := 0.0
	for _ in 0 ..< epochs {
		rand.shuffle(order)
		loss_sum = 0
		nb := ns / batch
		for b in 0 ..< nb {
			xm := l.matrix_new(f64, 1, batch * EV_SEQ * EV_NF, md.allocator)
			ym := l.matrix_new(f64, batch, 1, md.allocator)
			for k in 0 ..< batch {
				s := order[b * batch + k]
				copy(xm.data[k * EV_SEQ * EV_NF:], X[s * EV_SEQ * EV_NF:(s + 1) * EV_SEQ * EV_NF])
				ym.data[k] = Y[s]
			}
			x := t.tensor_new(xm, false, md.allocator)
			x.shape = [4]int{batch, EV_SEQ, EV_NF, 1}
			y := t.tensor_new(ym, false, md.allocator)
			y.shape = [4]int{batch, 1, 1, 1}
			t.graph_begin(&md.scope)
			pred := ml_fin.lstm_volatility_forecaster_forward(&md.ens.lstm, x)
			loss := t.tensor_mse_loss(pred, y)
			t.tensor_backward(loss, md.allocator)
			nn.adam_step(&md.opt)
			nn.adam_zero_grad(&md.opt)
			loss_sum += loss.data.data[0]
			t.tensor_free_graph(loss)
			t.graph_end(&md.scope)
			t.tensor_free(x)
			t.tensor_free(y)
		}
		loss_sum /= f64(max(nb, 1))
	}
	return loss_sum
}

// Trains in rounds of 5 epochs on the samples ending on days tr, up to 60 epochs, and
// keeps the parameters with the lowest RMSE on the samples ending on days va; stops
// after 3 rounds without improvement (early stopping). Returns the epochs of the kept
// parameters and their validation RMSE (daily units).
@(private = "file")
_ev_fit :: proc(
	md: ^_EV_Model,
	d: ^_EV,
	tr, va: []int,
	h: int,
) -> (
	best_epochs: int,
	best_rmse: f64,
) {
	ROUND :: 5
	MAX_ROUNDS :: 12
	PATIENCE :: 3
	snap := make([][]f64, len(md.opt.parameters), context.temp_allocator)
	for p, i in md.opt.parameters {snap[i] = make([]f64, len(p.data.data), context.temp_allocator)}
	fmean, fsd, ymean, ysd := md.fmean, md.fsd, md.ymean, md.ysd
	best_rmse = math.INF_F64
	since := 0
	for round in 1 ..= MAX_ROUNDS {
		_ev_train(md, d, tr, h, ROUND, 32)
		s := 0.0
		for e in va {
			y := _ev_target(d, e, h)
			f := _ev_predict(md, d, e)
			s += (f - y) * (f - y)
		}
		rmse := math.sqrt(s / f64(len(va)))
		if rmse < best_rmse {
			best_rmse, best_epochs, since = rmse, ROUND * round, 0
			for p, i in md.opt.parameters {copy(snap[i], p.data.data)}
			fmean, fsd, ymean, ysd = md.fmean, md.fsd, md.ymean, md.ysd
		} else {
			since += 1
			if since >= PATIENCE {break}
		}
	}
	for p, i in md.opt.parameters {copy(p.data.data, snap[i])}
	md.fmean, md.fsd, md.ymean, md.ysd = fmean, fsd, ymean, ysd
	return
}

// The model's forecast on day e of the realised daily volatility over the next h days
@(private = "file")
_ev_predict :: proc(md: ^_EV_Model, d: ^_EV, e: int) -> f64 {
	xm := l.matrix_new(f64, 1, EV_SEQ * EV_NF, md.allocator)
	_ev_window(md, d, e, xm.data)
	x := t.tensor_new(xm, false, md.allocator)
	x.shape = [4]int{1, EV_SEQ, EV_NF, 1}
	t.graph_begin(&md.scope)
	pred := ml_fin.lstm_volatility_forecaster_forward(&md.ens.lstm, x)
	out := math.exp(md.ymean + md.ysd * pred.data.data[0])
	t.tensor_free_graph(pred)
	t.graph_end(&md.scope)
	t.tensor_free(x)
	return out
}

@(private = "file")
_ev_rmse :: proc(f, y: []f64) -> f64 {
	s := 0.0
	for v, i in f {s += (v - y[i]) * (v - y[i])}
	return math.sqrt(s / f64(len(f)))
}

// Annualised volatility in %, from daily units
@(private = "file")
_ev_ann :: proc(v: f64) -> f64 {return v * math.sqrt_f64(252) * 100}

@(private = "file")
_ev_plausible :: proc(xs: []f64) -> (ok: bool, lo, hi: f64) {
	lo, hi = math.INF_F64, -math.INF_F64
	for x in xs {lo = min(lo, _ev_ann(x)); hi = max(hi, _ev_ann(x))}
	return lo >= 3 && hi <= 150, lo, hi
}

@(private = "file")
_ev_expect :: proc(passed, total: ^int, ok: bool, name: string) {
	total^ += 1
	if ok {passed^ += 1}
	fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
}

ensemble_volatility_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Ensemble Volatility Forecasting (GARCH + Sequence Model) ===")
	context.allocator = allocator
	passed, total := 0, 0

	backend: nn.SeqBackend = .Mamba // .LSTM, .GRU or .Mamba
	H :: 5 // forecast the realised volatility of the next 5 days
	fmt.printf("--- Sequence backend: %v, horizon %d days ---\n", backend, H)

	d, ok := _ev_load(.FiveYears, allocator)
	if !ok {fail_missing_data(#procedure, "Yahoo data for SPY and ^VIX"); return}
	defer _ev_free(&d, allocator)
	fmt.printf("Aligned %d days of SPY and VIX data\n", d.n)

	// samples: train 60 %, validation 20 % (ensemble weight, conformal), test 20 %
	ends := make([]int, d.n - H - EV_FIRST_END, allocator)
	defer delete(ends, allocator)
	for &e, i in ends {e = EV_FIRST_END + i}
	n_tr := len(ends) * 6 / 10
	n_va := len(ends) * 2 / 10
	tr, va, te := ends[:n_tr], ends[n_tr:n_tr + n_va], ends[n_tr + n_va:]

	// GARCH on the returns the training targets cover
	g := _ev_garch(&d, tr[len(tr) - 1] + H, 2000, allocator)
	defer delete(g.var_next, allocator)
	fmt.printf("GARCH(1,1): ω %.3e  α %.4f  β %.4f\n", g.omega, g.alpha, g.beta)

	md := _ev_model_new(backend, 32, 1e-3, allocator)
	defer _ev_model_free(md)
	fmt.printf("\n--- Training %v on %d samples ---\n", backend, len(tr))
	ep, vr := _ev_fit(md, &d, tr, va, H)
	fmt.printf("  early stopping: %d epochs, validation RMSE %.2f%% annualised\n", ep, _ev_ann(vr))

	forecast :: proc(
		md: ^_EV_Model,
		d: ^_EV,
		g: ^_EV_Garch,
		days: []int,
		h: int,
	) -> (
		gv, mv, y: []f64,
	) {
		gv = make([]f64, len(days), context.temp_allocator)
		mv = make([]f64, len(days), context.temp_allocator)
		y = make([]f64, len(days), context.temp_allocator)
		for e, i in days {
			gv[i], mv[i], y[i] = _ev_garch_vol(g, e, h), _ev_predict(md, d, e), _ev_target(d, e, h)
		}
		return
	}
	g_va, m_va, y_va := forecast(md, &d, &g, va, H)
	g_te, m_te, y_te := forecast(md, &d, &g, te, H)

	md.ens.ensemble_weight = ml_fin.compute_optimal_weight(g_va, m_va, y_va)
	e_va := make([]f64, len(va), context.temp_allocator)
	e_te := make([]f64, len(te), context.temp_allocator)
	for i in 0 ..< len(va) {e_va[i] = ml_fin.ensemble_predict(&md.ens, g_va[i], m_va[i])}
	for i in 0 ..< len(te) {e_te[i] = ml_fin.ensemble_predict(&md.ens, g_te[i], m_te[i])}
	fmt.printf(
		"\nGARCH weight chosen on the validation days: %.2f (%v %.2f)\n",
		md.ens.ensemble_weight,
		backend,
		1 - md.ens.ensemble_weight,
	)

	rg, rm, re := _ev_rmse(g_te, y_te), _ev_rmse(m_te, y_te), _ev_rmse(e_te, y_te)
	fmt.printf(
		"\nTest RMSE of the %d-day realised volatility (annualised %%, %d days):\n",
		H,
		len(te),
	)
	fmt.printf(
		"  GARCH(1,1) %.2f   %v %.2f   ensemble %.2f\n",
		_ev_ann(rg),
		backend,
		_ev_ann(rm),
		_ev_ann(re),
	)

	last := len(te) - 1
	fmt.printf(
		"\nLast test day: realised %.2f%%, GARCH %.2f%%, %v %.2f%%, ensemble %.2f%% (annualised)\n",
		_ev_ann(y_te[last]),
		_ev_ann(g_te[last]),
		backend,
		_ev_ann(m_te[last]),
		_ev_ann(e_te[last]),
	)

	// conformal interval from the validation residuals
	cp := ml_fin.conformal_new(allocator)
	defer ml_fin.conformal_free(&cp)
	ml_fin.conformal_calibrate(&cp, y_va, e_va, 0.05)
	lo, hi := ml_fin.conformal_predict_interval(&cp, e_te[last])
	fmt.printf(
		"95%% conformal interval: [%.2f%%, %.2f%%] (annualised)\n",
		_ev_ann(max(lo, 0)),
		_ev_ann(hi),
	)
	covered := 0
	for i in 0 ..< len(te) {
		a, b := ml_fin.conformal_predict_interval(&cp, e_te[i])
		if y_te[i] >= a && y_te[i] <= b {covered += 1}
	}

	pm, mlo, mhi := _ev_plausible(m_te)
	pg, glo, ghi := _ev_plausible(g_te)
	_ev_expect(
		&passed,
		&total,
		pm && pg,
		fmt.tprintf(
			"test forecasts between 3%% and 150%% annualised: %v %.1f%%..%.1f%%, GARCH %.1f%%..%.1f%%",
			backend,
			mlo,
			mhi,
			glo,
			ghi,
		),
	)
	_ev_expect(
		&passed,
		&total,
		rm <= 1.5 * rg,
		fmt.tprintf(
			"%v test RMSE %.2f%% within 1.5 x GARCH's %.2f%%",
			backend,
			_ev_ann(rm),
			_ev_ann(rg),
		),
	)
	cov := f64(covered) / f64(len(te))
	_ev_expect(
		&passed,
		&total,
		cov >= 0.85,
		fmt.tprintf(
			"95%% conformal interval covers %.1f%% of the test days (at least 85%%)",
			100 * cov,
		),
	)

	fmt.printf("Ensemble volatility summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

vrp_signal_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Live Volatility Risk Premium (VRP) Signal ===")
	context.allocator = allocator
	passed, total := 0, 0

	backend: nn.SeqBackend = .Mamba
	H :: 21 // the VIX is a 30-day (about 21 trading days) implied volatility

	d, ok := _ev_load(.TwoYears, allocator)
	if !ok {fail_missing_data(#procedure, "Yahoo data for SPY and ^VIX"); return}
	defer _ev_free(&d, allocator)

	// Historical VRP: VIX minus the realised volatility of the following 21 days, both
	// annualised in %, over every day whose 21 days are known
	hist := make([]f64, d.n - H - (EV_WINDOW - 1), allocator)
	defer delete(hist, allocator)
	for &v, k in hist {
		e := EV_WINDOW - 1 + k
		v = 100 * d.vix[e] - _ev_ann(_ev_target(&d, e, H))
	}
	vrp_mean, vrp_sd := 0.0, 0.0
	for v in hist {vrp_mean += v}
	vrp_mean /= f64(len(hist))
	for v in hist {vrp_sd += (v - vrp_mean) * (v - vrp_mean)}
	vrp_sd = math.sqrt(vrp_sd / f64(len(hist) - 1))
	fmt.printf("Historical VRP: mean %.2f, std %.2f volatility points\n", vrp_mean, vrp_sd)

	// samples with a known 21-day target: train 80 %, validation 20 % (ensemble weight)
	ends := make([]int, d.n - H - EV_FIRST_END, allocator)
	defer delete(ends, allocator)
	for &e, i in ends {e = EV_FIRST_END + i}
	n_tr := len(ends) * 8 / 10
	tr, va := ends[:n_tr], ends[n_tr:]

	g := _ev_garch(&d, tr[len(tr) - 1] + H, 1000, allocator)
	defer delete(g.var_next, allocator)
	md := _ev_model_new(backend, 32, 1e-3, allocator)
	defer _ev_model_free(md)
	ep, _ := _ev_fit(md, &d, tr, va, H)
	fmt.printf("Trained the ensemble: %d epochs (early stopping)\n", ep)

	g_va := make([]f64, len(va), context.temp_allocator)
	m_va := make([]f64, len(va), context.temp_allocator)
	y_va := make([]f64, len(va), context.temp_allocator)
	for e, i in va {g_va[i], m_va[i], y_va[i] = _ev_garch_vol(&g, e, H), _ev_predict(md, &d, e), _ev_target(&d, e, H)}
	md.ens.ensemble_weight = ml_fin.compute_optimal_weight(g_va, m_va, y_va)

	// today's forecast: the last day, with every return known
	today := d.n - 1
	gv, mv := _ev_garch_vol(&g, today, H), _ev_predict(md, &d, today)
	ev := ml_fin.ensemble_predict(&md.ens, gv, mv)
	fmt.printf(
		"21-day volatility forecast: GARCH %.2f%%, %v %.2f%%, ensemble %.2f%% (GARCH weight %.2f); trailing 21 days %.2f%%\n",
		_ev_ann(gv),
		backend,
		_ev_ann(mv),
		_ev_ann(ev),
		md.ens.ensemble_weight,
		_ev_ann(_ev_trailing(&d, today, H)),
	)

	sig := ml_fin.compute_vrp(100 * d.vix[today], ev, vrp_mean, vrp_sd)
	ml_fin.print_vrp_signal(sig)

	all := []f64{gv, mv, ev}
	p, lo, hi := _ev_plausible(all)
	_ev_expect(
		&passed,
		&total,
		p,
		fmt.tprintf("today's forecasts between 3%% and 150%% annualised (%.1f%%..%.1f%%)", lo, hi),
	)
	rg, rm := _ev_rmse(g_va, y_va), _ev_rmse(m_va, y_va)
	_ev_expect(
		&passed,
		&total,
		rm <= 1.5 * rg,
		fmt.tprintf(
			"%v validation RMSE %.2f%% within 1.5 x GARCH's %.2f%%",
			backend,
			_ev_ann(rm),
			_ev_ann(rg),
		),
	)
	_ev_expect(
		&passed,
		&total,
		abs(sig.z_score) < 6,
		fmt.tprintf("VRP z-score %.2f within ±6", sig.z_score),
	)

	fmt.printf("VRP signal summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// Walk-forward VRP strategy: every RETRAIN days the model trains further (with early
// stopping on the latest known samples) and GARCH is refitted on the data known by then; the ensemble weight is chosen on the last 120
// forecasts whose 20-day outcome is known (0.5 until there are 60). The z-score
// compares VIX − forecast with the history of VIX − realised volatility known so far.
// Position: short volatility (+1) above z = 1.2, long (−1) below −1.2; the day's P&L
// is the position times the fall of VIX − forecast.
vrp_backtest_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Walk-Forward VRP Backtest ===")
	context.allocator = allocator
	passed, total := 0, 0

	H :: 20
	INITIAL :: 500 // days of the initial training
	RETRAIN :: 60
	d, ok := _ev_load(.FiveYears, allocator)
	if !ok {fail_missing_data(#procedure, "Yahoo data for SPY and ^VIX"); return}
	defer _ev_free(&d, allocator)

	start := INITIAL
	days := d.n - 1 - start
	if days <= H {fmt.println("FAIL not enough data for the backtest"); record_failures(1); return}
	fmt.printf("Backtest: days %d to %d (%d days)\n", start, d.n - 1, days)

	result := ml_fin.VRPBacktestResult {
		allocator = allocator,
	}
	result.daily_pnl = make([]f64, days, allocator)
	result.equity_curve = make([]f64, days, allocator)
	result.positions = make([]f64, days, allocator)
	result.signals = make([]f64, days, allocator)
	result.forecast_rv = make([]f64, days, allocator)
	result.implied_vol = make([]f64, days, allocator)
	result.actual_vol = make([]f64, days, allocator)
	defer ml_fin.vrp_backtest_result_free(&result)
	g_fc := make([]f64, days, allocator)
	m_fc := make([]f64, days, allocator)
	defer {delete(g_fc, allocator); delete(m_fc, allocator)}

	md := _ev_model_new(.Mamba, 32, 1e-3, allocator)
	defer _ev_model_free(md)
	g: _EV_Garch
	defer delete(g.var_next, allocator)

	// train on the samples whose target is known on day `day` (e + H <= day): the first
	// 80 % for training, the last 20 % for early stopping
	retrain :: proc(md: ^_EV_Model, d: ^_EV, g: ^_EV_Garch, day: int, allocator: mem.Allocator) {
		ends := make([]int, day - H - EV_FIRST_END + 1, context.temp_allocator)
		for &e, i in ends {e = EV_FIRST_END + i}
		n_tr := len(ends) * 8 / 10
		_ev_fit(md, d, ends[:n_tr], ends[n_tr:], H)
		delete(g.var_next, allocator)
		g^ = _ev_garch(d, day, 500, allocator)
	}
	retrain(md, &d, &g, start, allocator)
	fmt.println("Initial training complete.")

	equity, prev_vrp, prev_pos := 0.0, 0.0, 0.0
	weight := 0.5
	for day in start ..< d.n - 1 {
		k := day - start
		if k > 0 && k % RETRAIN == 0 {
			retrain(md, &d, &g, day, allocator)
			// weight from the forecasts made on days day - H - 119 .. day - H
			hi := k - H
			if hi >= 59 {
				lo := max(0, hi - 119)
				y := make([]f64, hi - lo + 1, context.temp_allocator)
				for j in lo ..= hi {y[j - lo] = _ev_target(&d, start + j, H)}
				weight = ml_fin.compute_optimal_weight(g_fc[lo:hi + 1], m_fc[lo:hi + 1], y)
			}
		}
		g_fc[k], m_fc[k] = _ev_garch_vol(&g, day, H), _ev_predict(md, &d, day)
		fc := _ev_ann(weight * g_fc[k] + (1 - weight) * m_fc[k])

		// VRP history known on this day: VIX_e − realised over e+1..e+H, e + H <= day
		s, s2, n := 0.0, 0.0, 0
		for e in EV_WINDOW - 1 ..= day - H {
			v := 100 * d.vix[e] - _ev_ann(_ev_target(&d, e, H))
			s += v; s2 += v * v; n += 1
		}
		mean := s / f64(n)
		sd := math.sqrt(max((s2 - f64(n) * mean * mean) / f64(n - 1), 1e-12))

		vix := 100 * d.vix[day]
		vrp := vix - fc
		z := (vrp - mean) / sd
		pos := 1.0 if z > 1.2 else (-1.0 if z < -1.2 else 0.0)
		pnl := prev_pos * (prev_vrp - vrp)
		equity += pnl
		result.daily_pnl[k], result.equity_curve[k], result.positions[k] = pnl, equity, pos
		result.signals[k], result.forecast_rv[k], result.implied_vol[k] = z, fc, vix
		result.actual_vol[k] = _ev_ann(_ev_target(&d, day, H)) if day + H < d.n else math.nan_f64()
		prev_vrp, prev_pos = vrp, pos
		if k % 100 == 0 {
			fmt.printf(
				"  day %4d | VIX %5.2f | forecast %5.2f | VRP %+6.2f | z %+5.2f | pos %+.0f | equity %+8.2f\n",
				day,
				vix,
				fc,
				vrp,
				z,
				pos,
				equity,
			)
		}
	}
	ml_fin.compute_backtest_metrics(&result)
	ml_fin.print_backtest_result(&result)

	// forecast quality over the days whose 20-day outcome is known
	nk := days - H
	f_ens := result.forecast_rv[:nk]
	f_g := make([]f64, nk, context.temp_allocator)
	for i in 0 ..< nk {f_g[i] = _ev_ann(g_fc[i])}
	r_ens, r_g := _ev_rmse(f_ens, result.actual_vol[:nk]), _ev_rmse(f_g, result.actual_vol[:nk])
	fmt.printf(
		"20-day volatility forecast RMSE over %d days: ensemble %.2f, GARCH %.2f (annualised %%)\n",
		nk,
		r_ens,
		r_g,
	)
	lo, hi := math.INF_F64, -math.INF_F64
	for f in result.forecast_rv {lo, hi = min(lo, f), max(hi, f)}
	_ev_expect(
		&passed,
		&total,
		lo >= 3 && hi <= 150,
		fmt.tprintf(
			"walk-forward forecasts between 3%% and 150%% annualised (%.1f%%..%.1f%%)",
			lo,
			hi,
		),
	)
	_ev_expect(
		&passed,
		&total,
		r_ens <= 1.5 * r_g,
		fmt.tprintf("ensemble forecast RMSE %.2f within 1.5 x GARCH's %.2f", r_ens, r_g),
	)

	fmt.printf("VRP backtest summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
