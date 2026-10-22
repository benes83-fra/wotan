package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// GPD fit contracts, on exceedances drawn from known GPDs:
// - the fit is the MLE: no neighbouring (xi, beta) has a lower NLL
// - parameters are close to the truth
// - `converged` reports a likelihood stall, not an exhausted budget

gpd_fit_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== GPD Fit Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	cases := [][2]f64{{0.25, 0.010}, {0.10, 0.020}, {0.40, 0.005}}
	n := 2000
	ex := make([]f64, n)
	defer delete(ex)
	for c in cases {
		xi_t, beta_t := c[0], c[1]
		for i in 0 ..< n {
			u := rand.float64()
			ex[i] = beta_t / xi_t * (math.pow(1.0 - u, -xi_t) - 1.0) // inverse CDF
		}
		xi, beta, converged := fin.gpd_fit(ex, allocator)
		nll := fin._gpd_neg_log_likelihood([]f64{xi, beta}, ex)

		// Local optimality on a ring of relative steps in both directions
		worst := math.inf_f64(1)
		for d in ([]f64{1e-3, 1e-2}) {
			for k in 0 ..< 8 {
				ang := f64(k) * math.PI / 4
				x := []f64{xi + d * math.cos(ang), beta * (1.0 + d * math.sin(ang))}
				worst = min(worst, fin._gpd_neg_log_likelihood(x, ex) - nll)
			}
		}
		expect(&passed, &total, worst > -1e-12 && converged,
			fmt.tprintf("xi=%.2f: fit is the MLE (min ring NLL gain %.1e, converged=%v)", xi_t, worst, converged))
		expect(&passed, &total, abs(xi - xi_t) < 0.08 && abs(beta / beta_t - 1.0) < 0.1,
			fmt.tprintf("xi=%.2f: recovers xi=%.4f beta=%.5f (truth beta %.3f)", xi_t, xi, beta, beta_t))
	}

	fmt.printf("GPD fit summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
