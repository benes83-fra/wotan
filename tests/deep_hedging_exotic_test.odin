package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"

// Deep hedging of path-dependent calls (arithmetic Asian, up-and-out barrier) under GBM,
// prices in units of S_0 and discounted (the numeraire deep hedging expects). The state
// carries what the payoff depends on: the running average for the Asian, a knocked-out
// flag for the barrier. Compared by P&L variance with the static Black-Scholes delta of
// the vanilla call held to maturity. Then a save / load round trip.

@(private = "file")
EX_S0, EX_K, EX_T, EX_R, EX_SIGMA :: 1.0, 1.0, 1.0, 0.05, 0.2
@(private = "file")
EX_PATHS, EX_STEPS, EX_EPOCHS :: 512, 25, 200

// GBM paths, state (e^{-rt} S, time to maturity, sigma, extra), extra = running average
// of S (Asian) or 1 once S has reached the barrier (barrier); discounted payoffs
@(private = "file")
_exotic_paths :: proc(asian: bool, barrier: f64) -> (paths, payoffs: ^t.Tensor) {
	N, F := EX_STEPS, 4
	dt := EX_T / f64(N)
	pd := l.matrix_new(f64, EX_PATHS * (N + 1), F)
	yd := l.matrix_new(f64, EX_PATHS, 1)
	for p in 0 ..< EX_PATHS {
		S, sum := EX_S0, 0.0
		knocked := false
		for k in 0 ..= N {
			if k > 0 {
				S *= math.exp(
					(EX_R - 0.5 * EX_SIGMA * EX_SIGMA) * dt +
					EX_SIGMA * math.sqrt(dt) * rand.float64_normal(0, 1),
				)
			}
			sum += S
			knocked ||= S >= barrier
			row := pd.data[(p * (N + 1) + k) * F:][:F]
			row[0] = S * math.exp(-EX_R * f64(k) * dt)
			row[1] = EX_T - f64(k) * dt
			row[2] = EX_SIGMA
			row[3] = sum / f64(k + 1) if asian else (1 if knocked else 0)
		}
		under := sum / f64(N + 1) if asian else (0 if knocked else S)
		yd.data[p] = math.exp(-EX_R * EX_T) * max(under - EX_K, 0)
	}
	paths = t.tensor_new(pd)
	paths.shape = {EX_PATHS, N + 1, F, 1}
	payoffs = t.tensor_new(yd)
	return
}

@(private = "file")
_ex_variance :: proc(xs: []f64) -> f64 {
	m, m2 := 0.0, 0.0
	for x in xs {m += x / f64(len(xs)); m2 += x * x / f64(len(xs))}
	return m2 - m * m
}

// P&L variance of the static Black-Scholes delta (vanilla call at t = 0) held to maturity
@(private = "file")
_ex_static_variance :: proc(paths, payoffs: ^t.Tensor) -> f64 {
	d1 :=
		(math.ln(EX_S0 / EX_K) + (EX_R + 0.5 * EX_SIGMA * EX_SIGMA) * EX_T) /
		(EX_SIGMA * math.sqrt(EX_T))
	delta := 0.5 * (1 + math.erf(d1 / math.SQRT_TWO))
	S, F := paths.shape[1], paths.shape[2]
	pnl := make([]f64, EX_PATHS, context.temp_allocator)
	for p in 0 ..< EX_PATHS {
		pnl[p] = delta * (paths.data.data[p * S * F + (S - 1) * F] - EX_S0) - payoffs.data.data[p]
	}
	return _ex_variance(pnl)
}

// Trains a variance-minimising hedger; returns it (free with deep_hedger_free) and its
// P&L variance on the paths
@(private = "file")
_ex_train :: proc(paths, payoffs: ^t.Tensor) -> (^ml_fin.DeepHedger, f64) {
	h := ml_fin.deep_hedger_new(
		{
			state_size = 4,
			hidden_size = 64,
			num_layers = 2,
			risk_measure = .Variance,
			num_assets = 1,
		},
	)
	opt := nn.adam_new(3e-3)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(h.network, &opt)
	schedule := optim.lr_cosine(3e-3, EX_EPOCHS, warmup_steps = 10)
	for epoch in 0 ..< EX_EPOCHS {
		opt.learning_rate = optim.lr_schedule_at(schedule, epoch)
		loss := ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt)
		if epoch % 50 == 0 || epoch == EX_EPOCHS - 1 {
			fmt.printf("  epoch %3d | P&L variance %.6f\n", epoch, loss)
		}
	}
	pnl, spots := ml_fin.deep_hedging_evaluate(h, paths, payoffs, context.allocator)
	defer {delete(pnl); delete(spots)}
	return h, _ex_variance(pnl)
}

deep_hedging_exotic_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Deep hedging: Asian and up-and-out barrier calls ===")
	fmt.printf(
		"S_0 = K = 1, T = %.0f, r = %.0f%%, sigma = %.0f%%; %d paths, %d steps, %d epochs\n",
		EX_T,
		100 * EX_R,
		100 * EX_SIGMA,
		EX_PATHS,
		EX_STEPS,
		EX_EPOCHS,
	)
	names := [2]string{"Asian call (arithmetic average)", "Up-and-out call, barrier 1.2"}
	hedgers: [2]^ml_fin.DeepHedger
	defer for h in hedgers {if h != nil {ml_fin.deep_hedger_free(h)}}
	all_ok := true
	paths0, payoffs0: ^t.Tensor
	for k in 0 ..< 2 {
		fmt.printf("\n%s\n", names[k])
		paths, payoffs := _exotic_paths(k == 0, 1.2)
		if k == 0 {paths0, payoffs0 = paths, payoffs}
		unhedged := _ex_variance(payoffs.data.data)
		static_var := _ex_static_variance(paths, payoffs)
		deep_var: f64
		hedgers[k], deep_var = _ex_train(paths, payoffs)
		ok := deep_var < static_var
		all_ok &&= ok
		fmt.printf(
			"%s %s: P&L variance unhedged %.5f, static delta %.5f, deep hedge %.5f (%.0f%% below static)\n",
			"OK  " if ok else "FAIL",
			names[k],
			unhedged,
			static_var,
			deep_var,
			100 * (1 - deep_var / static_var),
		)
	}

	// save / load: the loaded hedger (given its configuration) reproduces the P&L
	save_path := "deep_hedger_exotic_test.bin"
	defer os.remove(save_path)
	same := ml_fin.deep_hedger_save(hedgers[0], save_path)
	loaded, load_ok := ml_fin.deep_hedger_load(save_path, allocator, hedgers[0].config)
	if load_ok {
		a, _ := ml_fin.deep_hedging_evaluate(hedgers[0], paths0, payoffs0, allocator)
		b, _ := ml_fin.deep_hedging_evaluate(loaded, paths0, payoffs0, allocator)
		for i in 0 ..< len(a) {same &&= a[i] == b[i]}
		ml_fin.deep_hedger_free(loaded)
	}
	same &&= load_ok
	fmt.printf(
		"%s saved and loaded Asian hedger gives the same P&L on every path\n",
		"OK  " if same else "FAIL",
	)
	record_failures(int(!all_ok) + int(!same))
}
