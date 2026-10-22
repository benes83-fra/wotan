package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

// Hull-White model contracts:
// - the 2-factor (G2++) caplet price reduces to the 1-factor price when the
//   second factor is switched off (eta = 0, rho = 0)
// - 1-factor calibration recovers known parameters from noise-free caplets

hull_white_contracts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Hull-White Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	n := 10
	Ts, Te, F, K, P := make([]f64, n), make([]f64, n), make([]f64, n), make([]f64, n), make([]f64, n)
	defer {delete(Ts); delete(Te); delete(F); delete(K); delete(P)}
	r0 := 0.04
	for i in 0 ..< n {
		Ts[i], Te[i] = f64(i + 1), f64(i + 2)
		F[i], K[i] = r0, r0
		P[i] = math.exp_f64(-r0 * Te[i])
	}
	ts, te := fin._make_const_tensor(Ts, n, allocator), fin._make_const_tensor(Te, n, allocator)
	f, k, p := fin._make_const_tensor(F, n, allocator), fin._make_const_tensor(K, n, allocator), fin._make_const_tensor(P, n, allocator)

	// 1. G2++ with eta = 0 is the 1-factor model
	true_a, true_sigma := 0.08, 0.012
	one_f := fin.hull_white_caplet_price_tensor(true_a, true_sigma, ts, te, f, k, p, allocator)
	two_f := fin.hull_white_2f_caplet_price_tensor(
		fin.HW2_Params{a = true_a, b = 0.5, sigma = true_sigma, eta = 1e-12, rho = 0},
		ts, te, f, k, p, allocator,
	)
	worst := 0.0
	for i in 0 ..< n {worst = max(worst, abs(two_f.data.data[i] / one_f.data.data[i] - 1.0))}
	expect(&passed, &total, worst < 1e-12, fmt.tprintf("2F caplet (eta = 0) == 1F caplet (max rel dev %.1e)", worst))

	// 2. 1F calibration recovers (a, sigma) from its own noise-free prices
	market := make([]f64, n)
	defer delete(market)
	copy(market, one_f.data.data)
	res := fin.calibrate_hull_white(market, Ts, Te, F, K, P, n, allocator)
	expect(&passed, &total, abs(res.params.a / true_a - 1) < 0.05 && abs(res.params.sigma / true_sigma - 1) < 0.02,
		fmt.tprintf("HW1F calibration recovers a=%.4f sigma=%.5f (truth 0.08 / 0.012)", res.params.a, res.params.sigma))

	fmt.printf("Hull-White summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
