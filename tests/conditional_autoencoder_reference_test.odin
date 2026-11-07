package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import "core:fmt"
import "core:mem"

// Conditional autoencoder (ml_finance/conditional_autoencoder.odin) against torch
// (float64) on an unbalanced LCG panel (12 assets, 6 dates, 3 characteristics, 61
// observations), beta network 3 -> 4 ReLU -> 2, linear factor network on the managed
// portfolios, L1 1e-4, weights from the LCG in both: the loss (MSE + L1) and every
// parameter gradient (torch differentiates |w|), the parameters and last loss after 25
// Adam steps (lr 0.005), the total / predictive R².

@(private = "file")
_CA_GRADS := [6][]f64 {
	{
		-0.00010389255549561167,
		0.00064698312968477763,
		-3.9199324928405947e-06,
		-8.9373420252145258e-05,
		9.5522677739798186e-05,
		-0.00021605602944289632,
		5.2320514994928783e-05,
		-0.00010794226335609182,
		-0.00012210753815337149,
		-0.00020921054543320613,
		-5.3769659672916796e-05,
		0.00014590503877493337,
	},
	{
		-3.8925554956116364e-06,
		0.00054698312968477769,
		-0.00010391993249284064,
		1.0626579747854754e-05,
	},
	{
		-0.00010513630965691062,
		-8.3565110315734971e-05,
		-0.00069314229384970639,
		0.00033442059134496099,
		-0.0004641421275349718,
		0.00017540089246426718,
		-0.00012096115131007303,
		-0.00012127896397577694,
	},
	{-0.0012565843020720527, 0.00078911879707044947},
	{
		0.00027892807698736604,
		-0.00017540236985944647,
		-0.00035027092971027564,
		0.00038907288913863735,
		-0.00041948577848501656,
		0.0001903888705130466,
	},
	{-0.00011800926267877638, 0.000826974175459257},
}
@(private = "file")
_CA_PARAMS := [6][]f64 {
	{
		-0.016446056057032782,
		0.19899831563528275,
		0.3271993986138001,
		-0.097092487858038867,
		0.073494378856706427,
		-0.33143752282328737,
		0.09248313793362184,
		-0.34061661365331836,
		0.033557657471506519,
		-0.059151073545261001,
		0.010912206446893641,
		0.085986272530047092,
	},
	{0.024591764154841482, -0.02056367716444896, 0.011719425422794348, -0.021632898823359337},
	{
		-0.26121231540642015,
		-0.00038834759264730822,
		-0.18970735804693981,
		0.27805769112209067,
		0.0027406759236845361,
		-0.20853641050692681,
		-0.33119507100550666,
		0.042759605095627734,
	},
	{0.033569743174781438, -0.0411586390585469},
	{
		0.22026383961251061,
		-0.15520387391965967,
		-0.18989549141569009,
		-0.077635750190078651,
		-0.11343078609611638,
		0.1248564174330717,
	},
	{0.15129598317886167, -0.085209574670873284},
}

@(private = "file")
_ca_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_ca_err :: proc(got, want: []f64, floor := 1e-6) -> f64 {
	e := 0.0
	for v, i in want {e = max(e, abs(got[i] - v) / max(abs(v), floor))}
	return e
}

conditional_autoencoder_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Conditional Autoencoder Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	N, T, L :: 12, 6, 3
	x: u64 = 404
	rows := make([dynamic]f64)
	defer delete(rows)
	y := make([dynamic]f64)
	defer delete(y)
	times := make([dynamic]int)
	defer delete(times)
	for d in 0 ..< T {
		for i in 0 ..< N {
			z := [L]f64{1, 2 * (_ca_u(&x) - 0.5), 2 * (_ca_u(&x) - 0.5)}
			r := 0.5 * (_ca_u(&x) - 0.5)
			r += 0.2 * z[1] * (_ca_u(&x) - 0.3)
			if (i + 2 * d) % 7 == 3 {continue}
			append(&rows, ..z[:])
			append(&y, r)
			append(&times, d)
		}
	}
	X := l.Matrix(f64) {
		data = rows[:],
		rows = len(y),
		cols = L,
	}
	panel := ml_fin.ca_panel_new(&X, y[:], times[:])
	defer ml_fin.ca_panel_free(&panel)
	m := ml_fin.ca_new({n_chars = L, n_factors = 2, hidden = []int{4}, l1 = 1e-4})
	defer ml_fin.ca_free(m)
	params := ml_fin.ca_parameters(m)
	defer delete(params)
	for q, k in params {
		scale := 0.8 if k % 2 == 0 else 0.1 // weights, biases
		for &v in q.data.data {v = scale * (_ca_u(&x) - 0.5)}
	}

	loss0 := ml_fin.ca_gradients(m, &panel)
	eg := abs(loss0 - 0.023559631096265278) / 0.023559631096265278
	for q, k in params {eg = max(eg, _ca_err(q.grad.data, _CA_GRADS[k]))}
	expect(
		&passed,
		&total,
		len(y) == 61 && eg < 1e-10,
		fmt.tprintf(
			"loss (MSE + L1) and the gradients of all 6 parameter tensors = torch on %d observations (relative error %.1e)",
			len(y),
			eg,
		),
	)

	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	for q in params {nn.adam_add_param(&opt, q)}
	last := 0.0
	for _ in 0 ..< 25 {last = ml_fin.ca_train_step(m, &panel, &opt)}
	ep := abs(last - 0.022878427048847055) / 0.022878427048847055
	for q, k in params {ep = max(ep, _ca_err(q.data.data, _CA_PARAMS[k]))}
	tot, pred := ml_fin.ca_r2(m, &panel)
	er := max(abs(tot - 0.016954896250170215) / 0.017, abs(pred - 0.016383654757394117) / 0.016)
	expect(
		&passed,
		&total,
		ep < 1e-9 && er < 1e-9,
		fmt.tprintf(
			"after 25 Adam steps: parameters and loss = torch (relative error %.1e); total / predictive R² %.4f / %.4f (error %.1e)",
			ep,
			tot,
			pred,
			er,
		),
	)
	fmt.printf("Conditional autoencoder reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
