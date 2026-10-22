package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// GPD fit contracts, on exceedances drawn from known GPDs:
// - the fit is the MLE: no neighbouring (xi, beta) has a lower NLL
// - parameters are close to the truth
// - `converged` reports a likelihood stall, not an exhausted budget
// EVT (evt_fit) contracts:
// - on a large sample with an exact GPD left tail, VaR99 and CVaR99 match the
//   empirical 1% quantile and expected shortfall, and CVaR lies beyond VaR
// - the exponential (xi -> 0) VaR branch is the continuous limit of the general one

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

	// EVT: returns whose left tail beyond -0.02 is exactly -0.02 - GPD(0.2, 0.01)
	m := 100_000
	rets := make([]f64, m)
	defer delete(rets)
	for i in 0 ..< m {
		if rand.float64() < 0.1 {
			u := rand.float64()
			rets[i] = -0.02 - 0.01 / 0.2 * (math.pow(1.0 - u, -0.2) - 1.0)
		} else {
			rets[i] = -0.02 + 0.05 * rand.float64()
		}
	}
	evt := fin.evt_fit(rets, 0.95, allocator)
	sorted := slice.clone(rets)
	defer delete(sorted)
	slice.sort(sorted)
	k := m / 100
	emp_var := sorted[k]
	emp_es := 0.0
	for i in 0 ..< k {emp_es += sorted[i]}
	emp_es /= f64(k)
	expect(&passed, &total, abs(evt.var_99 / emp_var - 1.0) < 0.03 && abs(evt.cvar_99 / emp_es - 1.0) < 0.05 && evt.cvar_99 < evt.var_99,
		fmt.tprintf("evt_fit VaR99=%.5f CVaR99=%.5f vs empirical %.5f / %.5f", evt.var_99, evt.cvar_99, emp_var, emp_es))

	v0 := fin._evt_tail_var(-0.02, 0.0, 0.01, 1000, 50, 0.99)
	vp := fin._evt_tail_var(-0.02, 1e-7, 0.01, 1000, 50, 0.99)
	vm := fin._evt_tail_var(-0.02, -1e-7, 0.01, 1000, 50, 0.99)
	expect(&passed, &total, abs(vp - v0) < 1e-8 && abs(vm - v0) < 1e-8 && v0 < -0.02,
		fmt.tprintf("exponential VaR branch is the ξ → 0 limit (%.8f vs %.8f / %.8f)", v0, vm, vp))

	fmt.printf("GPD fit summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
