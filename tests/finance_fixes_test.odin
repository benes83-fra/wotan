package tests

import fin "../wotan/finance"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"

// Regression tests for wotan/finance issues found while writing the finance
// guides. Options and simulation: the binomial and trinomial trees read the
// level-2 node values for gamma and theta inside the backward induction loop,
// which starts at level n − 1, so with n_steps ≤ 2 (n = 2) gamma came out as 0
// and theta as −price / (2Δt). black_scholes_price multiplied by a 1 × 1
// constant, so a batch of options (n × 1 inputs) panicked in tensor_mul; and
// its intermediates computed from inputs without gradient (e^{−rT}, √T for
// fixed r, T), or all of them without any gradient, were not linked into the
// graph and leaked. Volatility and exotics: the FX normal CDF was
// ½(1 + erf(x/√2)), which cancels in the left tail, so far out-of-the-money
// Garman-Kohlhagen prices came out as 0 or negative; fx_digital_call_price
// divided by σ, so σ = 0 with the forward at the strike gave NaN;
// calibrate_sabr said β could be fixed by equal bounds but hard-coded its
// bounds (now fixed_beta).

@(private = "file")
ff_vec :: proc(v: []f64, grad: bool) -> ^t.Tensor {
	m := l.matrix_new(f64, len(v), 1)
	copy(m.data, v)
	return t.tensor_new(m, grad)
}

finance_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== finance regression tests ===")
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	{ 	// 2-step trees: Greeks from the expiry level
		// S = 100, K = 105, T = 1, r = 5%, σ = 30%, q = 4%; level 2 is the expiry, so
		// the middle node (price S) is worth the payoff 5 and theta = (5 − V)/(2Δt)
		b := fin.american_option_binomial(100, 105, 1, 0.05, 0.3, 0.04, .Put, 2, allocator)
		tr := fin.american_option_trinomial(100, 105, 1, 0.05, 0.3, 0.04, .Put, 2, allocator)
		// QuantLib 1.44 BinomialCRRVanillaEngine, 2 steps: gamma 0.0207131506156
		good :=
			math.abs(b.gamma - 0.0207131506156) < 1e-12 &&
			math.abs(b.theta - (5 - b.price)) < 1e-12 &&
			tr.gamma > 0 &&
			math.abs(tr.theta - (5 - tr.price)) < 1e-12
		report(
			fmt.tprintf(
				"2-step trees: binomial gamma %.10f theta %.6f, trinomial gamma %.6f theta %.6f",
				b.gamma,
				b.theta,
				tr.gamma,
				tr.theta,
			),
			good,
			&passed,
			&total,
		)
	}

	{ 	// FX: far out-of-the-money Garman-Kohlhagen prices keep their precision
		// S = 1.1, T = 0.5, r_d = 3%, r_f = 1%, σ = 10%; exact values (mpmath, 40 digits)
		call := fin.fx_gk_price(1.1, 2.0, 0.5, 0.03, 0.01, 0.1, .Call)
		put := fin.fx_gk_price(1.1, 0.605, 0.5, 0.03, 0.01, 0.1, .Put)
		far := fin.fx_gk_price(1.1, 1.21 / 3.0, 0.5, 0.03, 0.01, 0.1, .Put)
		good :=
			math.abs(call / 5.65366489261909e-19 - 1) < 1e-9 &&
			math.abs(put / 2.66807118060116e-20 - 1) < 1e-9 &&
			math.abs(far / 2.28047185159941e-49 - 1) < 1e-9
		report(
			fmt.tprintf("FX far out of the money: call %.6e, put %.6e, %.6e", call, put, far),
			good,
			&passed,
			&total,
		)
	}
	{ 	// FX digital without volatility: S_T is the forward
		at_fwd := fin.fx_digital_call_price(1.1, 1.1, 0.5, 0.02, 0.02, 0) // forward = strike
		above := fin.fx_digital_call_price(1.1, 1.0, 0.5, 0.03, 0.01, 0)
		good := at_fwd == 0 && math.abs(above - math.exp(-0.015)) < 1e-15
		report(
			fmt.tprintf(
				"FX digital at sigma = 0: %v at the forward, %.12f in the money",
				at_fwd,
				above,
			),
			good,
			&passed,
			&total,
		)
	}
	{ 	// SABR with β held fixed (calibrate_sabr's comment promised it; no way to do it)
		p := fin.SABR_Params {
			alpha = 1.3,
			beta  = 0.6,
			rho   = -0.35,
			nu    = 0.5,
		}
		F := 100 * math.exp(0.03)
		quotes: [7]fin.VolSurfacePoint
		for &q, i in quotes {
			K := 70 + 10 * f64(i)
			q = {
				strike      = K,
				expiry      = 1,
				implied_vol = fin.sabr_implied_vol(F, K, 1, p),
			}
		}
		r := fin.calibrate_sabr(quotes[:], 100, 0.03, allocator, fixed_beta = 0.6)
		good :=
			r.params.beta == 0.6 &&
			math.abs(r.params.alpha - 1.3) < 1e-8 &&
			math.abs(r.params.rho + 0.35) < 1e-8 &&
			math.abs(r.params.nu - 0.5) < 1e-8
		report(
			fmt.tprintf("calibrate_sabr with fixed_beta = 0.6: %v", r.params),
			good,
			&passed,
			&total,
		)
	}
	{ 	// black_scholes_price leaves nothing behind, with and without gradients
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		context.allocator = mem.tracking_allocator(&track)

		S, K, T, r, sigma :=
			ff_vec({100}, false),
			ff_vec({110}, false),
			ff_vec({0.5}, false),
			ff_vec({0.03}, false),
			ff_vec({0.25}, true)
		p := fin.black_scholes_price(S, K, T, r, sigma, .Put, context.allocator)
		t.tensor_backward(p)
		vega := sigma.grad.data[0]
		t.tensor_free_graph(p)
		with_grad := len(track.allocation_map) // the 5 inputs and sigma's gradient
		sigma.requires_grad = false
		q := fin.black_scholes_price(S, K, T, r, sigma, .Call, context.allocator)
		call := q.data.data[0]
		t.tensor_free_graph(q)
		without := len(track.allocation_map)
		for x in ([]^t.Tensor{S, K, T, r, sigma}) {t.tensor_free(x)}
		g := fin.compute_greeks(100, 110, 0.5, 0.03, 0.25, .Put)
		good :=
			with_grad == 11 &&
			without == 11 &&
			len(track.allocation_map) == 0 &&
			math.abs(vega - 100 * g.vega) < 1e-10 &&
			math.abs(call - fin.compute_greeks(100, 110, 0.5, 0.03, 0.25, .Call).price) < 1e-10
		report(
			fmt.tprintf(
				"black_scholes_price with constant K, T, r: live allocations %d, %d, then %d",
				with_grad,
				without,
				len(track.allocation_map),
			),
			good,
			&passed,
			&total,
		)
	}
	{ 	// a batch of options (last: it used to panic)
		S, K := ff_vec({100, 100, 100}, true), ff_vec({90, 100, 110}, false)
		T, r, sigma :=
			ff_vec({0.5, 1, 2}, false),
			ff_vec({0.03, 0.03, 0.03}, false),
			ff_vec({0.25, 0.25, 0.25}, false)
		p := fin.black_scholes_price(S, K, T, r, sigma, .Call, allocator)
		s := t.tensor_sum(p)
		t.tensor_backward(s)
		good := true
		for k in 0 ..< 3 {
			g := fin.compute_greeks(100, K.data.data[k], T.data.data[k], 0.03, 0.25, .Call)
			good =
				good &&
				math.abs(p.data.data[k] - g.price) < 1e-10 &&
				math.abs(S.grad.data[k] - g.delta) < 1e-12
		}
		report(
			fmt.tprintf("black_scholes_price on 3 options: %.6f", p.data.data),
			good,
			&passed,
			&total,
		)
		t.tensor_free_graph(s)
		for x in ([]^t.Tensor{S, K, T, r, sigma}) {t.tensor_free(x)}
	}

	fmt.printf("finance regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
