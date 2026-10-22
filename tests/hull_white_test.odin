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

	// 3. G2++ swaption pricer vs the exact price (2D Gaussian expectation under
	//    the T-forward measure), co-terminal ATM swaptions on a flat curve
	g2 := fin.HW2_Params{a = 0.15, b = 0.02, sigma = 0.008, eta = 0.006, rho = -0.4}
	worst_swpt, worst_mart := 0.0, 0.0
	for T in ([]int{1, 5}) {
		tenor := 10 - T
		A0 := 0.0
		for i in 1 ..= tenor {A0 += math.exp_f64(-r0 * f64(T + i))}
		S0 := (math.exp_f64(-r0 * f64(T)) - math.exp_f64(-r0 * f64(T + tenor))) / A0
		spec := []fin.SwaptionSpec{{T_exp = f64(T), swap_length = tenor, F = S0, K = S0, annuity = A0}}
		lib := fin.hw2f_swaption_price_strip(g2, spec, 1, r0, allocator)
		mart := 0.0
		exact := _g2pp_payer_swaption_exact(g2, r0, f64(T), tenor, S0, &mart)
		worst_swpt = max(worst_swpt, abs(lib[0] / exact - 1.0))
		worst_mart = max(worst_mart, abs(mart))
		delete(lib, allocator)
	}
	expect(&passed, &total, worst_mart < 1e-9, fmt.tprintf("exact reference is a martingale (E^T[P(T,Tn)] err %.1e)", worst_mart))
	expect(&passed, &total, worst_swpt < 2e-3, fmt.tprintf("G2++ swaption pricer vs exact (max rel err %.2e)", worst_swpt))

	fmt.printf("Hull-White summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// Exact G2++ payer swaption (annual payments T+1..T+n, strike K, flat curve r0):
// P(0,T) * E^T[(1 - P(T,Tn) - K sum_i P(T,Ti))^+] with (x(T), y(T)) jointly
// Gaussian under the T-forward measure (Brigo & Mercurio, ch. 4), integrated on
// a grid over +-8 standard deviations. `mart` returns E^T[P(T,Tn)]/(P(0,Tn)/P(0,T)) - 1.
_g2pp_payer_swaption_exact :: proc(p: fin.HW2_Params, r0, T: f64, n: int, K: f64, mart: ^f64) -> f64 {
	a, b, s, e, r := p.a, p.b, p.sigma, p.eta, p.rho
	P0 :: proc(r0, t: f64) -> f64 {return math.exp_f64(-r0 * t)}
	B :: proc(k, tau: f64) -> f64 {return (1.0 - math.exp_f64(-k * tau)) / k}
	V :: proc(p: fin.HW2_Params, tau: f64) -> f64 {
		a, b, s, e, r := p.a, p.b, p.sigma, p.eta, p.rho
		va := s * s / (a * a) * (tau + 2.0 / a * math.exp_f64(-a * tau) - 1.0 / (2.0 * a) * math.exp_f64(-2.0 * a * tau) - 3.0 / (2.0 * a))
		vb := e * e / (b * b) * (tau + 2.0 / b * math.exp_f64(-b * tau) - 1.0 / (2.0 * b) * math.exp_f64(-2.0 * b * tau) - 3.0 / (2.0 * b))
		vab := 2.0 * r * s * e / (a * b) * (tau + (math.exp_f64(-a * tau) - 1.0) / a + (math.exp_f64(-b * tau) - 1.0) / b - (math.exp_f64(-(a + b) * tau) - 1.0) / (a + b))
		return va + vb + vab
	}
	sx := math.sqrt_f64(s * s / (2 * a) * (1 - math.exp_f64(-2 * a * T)))
	sy := math.sqrt_f64(e * e / (2 * b) * (1 - math.exp_f64(-2 * b * T)))
	rxy := r * s * e / ((a + b) * sx * sy) * (1 - math.exp_f64(-(a + b) * T))
	Mx := (s * s / (a * a) + r * s * e / (a * b)) * (1 - math.exp_f64(-a * T)) - s * s / (2 * a * a) * (1 - math.exp_f64(-2 * a * T)) - r * s * e / (b * (a + b)) * (1 - math.exp_f64(-(a + b) * T))
	My := (e * e / (b * b) + r * s * e / (a * b)) * (1 - math.exp_f64(-b * T)) - e * e / (2 * b * b) * (1 - math.exp_f64(-2 * b * T)) - r * s * e / (a * (a + b)) * (1 - math.exp_f64(-(a + b) * T))

	lnA := make([]f64, n + 1)
	Ba := make([]f64, n + 1)
	Bb := make([]f64, n + 1)
	defer {delete(lnA); delete(Ba); delete(Bb)}
	for i in 1 ..= n {
		Ti := T + f64(i)
		lnA[i] = math.ln_f64(P0(r0, Ti) / P0(r0, T)) + 0.5 * (V(p, Ti - T) - V(p, Ti) + V(p, T))
		Ba[i], Bb[i] = B(a, Ti - T), B(b, Ti - T)
	}
	N, L := 400, 8.0
	h := 2 * L / f64(N)
	val, mart_sum := 0.0, 0.0
	for i in 0 ..< N {
		z1 := -L + (f64(i) + 0.5) * h
		w1 := math.exp_f64(-0.5 * z1 * z1) / math.sqrt_f64(2 * math.PI) * h
		for j in 0 ..< N {
			z2 := -L + (f64(j) + 0.5) * h
			w := w1 * math.exp_f64(-0.5 * z2 * z2) / math.sqrt_f64(2 * math.PI) * h
			x := -Mx + sx * z1
			y := -My + sy * (rxy * z1 + math.sqrt_f64(1 - rxy * rxy) * z2)
			sum_P, P_n := 0.0, 0.0
			for k in 1 ..= n {
				P_k := math.exp_f64(lnA[k] - Ba[k] * x - Bb[k] * y)
				sum_P += P_k
				if k == n {P_n = P_k}
			}
			val += w * math.max(1 - P_n - K * sum_P, 0)
			mart_sum += w * P_n
		}
	}
	mart^ = mart_sum / (P0(r0, T + f64(n)) / P0(r0, T)) - 1
	return P0(r0, T) * val
}
