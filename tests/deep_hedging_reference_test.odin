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

// Deep hedging P&L and training against closed forms:
// - a constant hedge (zero weights, output bias b) gives
//     b . (S_N - S_0) - cost |b| . (S_0 + S_N) - Z
//   (opened at t_0, closed at t_N), for one and for two assets
// - the gradient of the variance loss with costs, two assets, vs finite differences
// - one period with the same state at t_0 on every path: the network can only choose
//   one holding, and training must reach the minimum-variance hedge
//     delta* = Cov(Z, X) / Var(X),  X = S_1 - S_0 - cost S_1  (delta > 0),
//   and its residual variance Var(Z) - Cov(Z, X)^2 / Var(X). The holding is decided before
//   S_1 is known; when the network saw S_1 (as it did) it could push the variance to ~0.
deep_hedging_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep Hedging Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	// paths [n, steps + 1, F]: the first A columns prices, the rest features in [-1, 1]
	make_paths :: proc(n, steps, F, A: int, seed: u64) -> (paths, payoffs: ^t.Tensor) {
		pd := l.matrix_new(f64, n * (steps + 1), F)
		gcl_fill(pd.data, seed)
		for i in 0 ..< n * (steps + 1) {for a in 0 ..< A {pd.data[i * F + a] = 100 + 5 * pd.data[i * F + a]}}
		paths = t.tensor_new(pd)
		paths.shape = {n, steps + 1, F, 1}
		yd := l.matrix_new(f64, n, 1)
		gcl_fill(yd.data, seed + 1)
		payoffs = t.tensor_new(yd)
		return
	}
	set_constant :: proc(h: ^ml_fin.DeepHedger, b: []f64) {
		params := nn.sequential_parameters(h.network)
		defer delete(params)
		for p in params {for &v in p.data.data {v = 0}}
		copy(params[len(params) - 1].data.data, b) // the output bias
	}

	{ 	// --- constant hedges
		err := 0.0
		for A in 1 ..= 2 {
			F := A + 2
			h := ml_fin.deep_hedger_new(
				{
					state_size = F,
					hidden_size = 6,
					num_layers = 2,
					transaction_cost = 0.002,
					num_assets = A,
				},
			)
			defer ml_fin.deep_hedger_free(h)
			b := []f64{0.6, -0.3}
			set_constant(h, b[:A])
			N :: 4
			paths, payoffs := make_paths(7, N, F, A, 77 + u64(A))
			pnl, _ := ml_fin.deep_hedging_evaluate(h, paths, payoffs, allocator)
			for p in 0 ..< 7 {
				want := -payoffs.data.data[p]
				for a in 0 ..< A {
					s0 := paths.data.data[p * (N + 1) * F + a]
					sN := paths.data.data[p * (N + 1) * F + N * F + a]
					want += b[a] * (sN - s0) - 0.002 * abs(b[a]) * (s0 + sN)
				}
				err = max(err, abs(pnl[p] - want))
			}
		}
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf(
				"constant hedge (1 and 2 assets, cost 0.2%%: opened at t_0, closed at t_N) vs the closed form: max err %.1e",
				err,
			),
		)
	}

	{ 	// --- gradient of the variance loss, 2 assets, costs, vs central differences
		h := ml_fin.deep_hedger_new(
			{
				state_size = 4,
				hidden_size = 5,
				num_layers = 2,
				transaction_cost = 0.003,
				num_assets = 2,
				risk_measure = .Variance,
			},
		)
		defer ml_fin.deep_hedger_free(h)
		params := nn.sequential_parameters(h.network)
		defer delete(params)
		for p, i in params {if i % 2 == 1 {for &v in p.data.data {v = 0.3}}} 	// hidden units away from the kink
		paths, payoffs := make_paths(9, 3, 4, 2, 501)
		opt := nn.adam_new(0) // lr 0: the step leaves the parameters, the gradients stay
		defer nn.adam_free(&opt)
		nn.sequential_add_to_adam(h.network, &opt)
		ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt)
		objective :: proc(h: ^ml_fin.DeepHedger, paths, payoffs: ^t.Tensor) -> f64 {
			pnl, spots := ml_fin.deep_hedging_evaluate(h, paths, payoffs, context.allocator)
			defer {delete(pnl); delete(spots)}
			m, m2 := 0.0, 0.0
			for v in pnl {m += v / f64(len(pnl)); m2 += v * v / f64(len(pnl))}
			return m2 - m * m
		}
		err, gmax := 0.0, 0.0
		for p in params {
			for j in 0 ..< len(p.data.data) {
				x0 := p.data.data[j]
				p.data.data[j] = x0 + 1e-6
				fp := objective(h, paths, payoffs)
				p.data.data[j] = x0 - 1e-6
				fm := objective(h, paths, payoffs)
				p.data.data[j] = x0
				fd := (fp - fm) / 2e-6
				err = max(err, abs(fd - p.grad.data[j]))
				gmax = max(gmax, abs(fd))
			}
		}
		expect(
			&passed,
			&total,
			err < 1e-6 * max(gmax, 1) && gmax > 0,
			fmt.tprintf(
				"variance loss gradient (2 assets, costs) vs finite differences: max err %.1e (largest %.1f)",
				err,
				gmax,
			),
		)
	}

	{ 	// --- one period: training reaches the minimum-variance hedge
		n :: 2000
		c :: 0.01
		pd := l.matrix_new(f64, n * 2, 2)
		yd := l.matrix_new(f64, n, 1)
		for p in 0 ..< n {
			s1 := 100 * math.exp(-0.02 + 0.2 * rand.float64_normal(0, 1))
			pd.data[p * 4 + 0], pd.data[p * 4 + 1] = 100, 1 // t_0: same state everywhere
			pd.data[p * 4 + 2], pd.data[p * 4 + 3] = s1, 0
			yd.data[p] = max(s1 - 100, 0)
		}
		paths := t.tensor_new(pd)
		paths.shape = {n, 2, 2, 1}
		payoffs := t.tensor_new(yd)
		// regression of Z on X = S_1 - S_0 - c S_1 (the P&L is delta X - c delta S_0 - Z)
		mx, mz := 0.0, 0.0
		for p in 0 ..< n {mx += (pd.data[p * 4 + 2] * (1 - c) - 100) / n; mz += yd.data[p] / n}
		sxx, sxz, szz := 0.0, 0.0, 0.0
		for p in 0 ..< n {
			x := pd.data[p * 4 + 2] * (1 - c) - 100 - mx
			z := yd.data[p] - mz
			sxx += x * x / n; sxz += x * z / n; szz += z * z / n
		}
		d_star := sxz / sxx
		v_star := szz - sxz * sxz / sxx

		h := ml_fin.deep_hedger_new(
			{
				state_size = 2,
				hidden_size = 8,
				num_layers = 2,
				transaction_cost = c,
				num_assets = 1,
				risk_measure = .Variance,
			},
		)
		defer ml_fin.deep_hedger_free(h)
		STEPS :: 3000
		opt := nn.adam_new(0.01)
		defer nn.adam_free(&opt)
		nn.sequential_add_to_adam(h.network, &opt)
		schedule := optim.lr_step_decay(0.01, STEPS / 3, 0.1)
		loss := 0.0
		for s in 0 ..< STEPS {
			opt.learning_rate = optim.lr_schedule_at(schedule, s)
			loss = ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt)
		}
		x := l.matrix_new(f64, 1, 2)
		x.data[0], x.data[1] = 100, 1
		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		t.graph_begin(&scope, no_grad = true)
		delta :=
			nn.sequential_forward(h.network, t.tensor_new(x, false, t._scope_or(allocator))).data.data[0]
		t.graph_end(&scope)
		pnl, spots := ml_fin.deep_hedging_evaluate(h, paths, payoffs, allocator)
		m, m2 := 0.0, 0.0
		for v in pnl {m += v / n; m2 += v * v / n}
		var := m2 - m * m
		expect(
			&passed,
			&total,
			abs(delta - d_star) < 1e-5 &&
			abs(var - v_star) < 1e-8 * v_star &&
			var >= v_star * (1 - 1e-12),
			fmt.tprintf(
				"one period, cost 1%%: trained holding %.6f vs minimum-variance %.6f, P&L variance %.4f vs %.4f (unhedged %.1f; no hedge can do better)",
				delta,
				d_star,
				var,
				v_star,
				szz,
			),
		)
		_ = spots
	}
	fmt.printf("Deep hedging reference checks: %d/%d passed\n", passed, total)
}
