package tests

import bfin "../wotan/behavioral_finance"
import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

// Contracts behind the fast Heston calibration paths:
// - heston_prices_same_expiry (one characteristic-function pass per expiry)
//   equals heston_price strike by strike, exactly
// - behavioral_heston_price_precomputed (cached Heston CDF and premium) equals
//   behavioral_heston_price exactly
// - put-call parity holds for the Heston pricer

heston_pricing_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Heston Pricing Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	S, r, T := 100.0, 0.03, 0.75
	params := fin.Heston_Params{v0 = 0.05, kappa = 1.8, theta = 0.06, sigma = 0.5, rho = -0.6}
	strikes := []f64{70, 85, 95, 100, 105, 120, 140}

	batch := make([]f64, len(strikes), allocator)
	defer delete(batch, allocator)
	fin.heston_prices_same_expiry(S, strikes, T, r, params, .Call, batch, 1000)
	same := true
	for K, k in strikes {same &&= batch[k] == fin.heston_price(S, K, T, r, params, .Call, 1000)}
	expect(&passed, &total, same, "batch pricer == heston_price for every strike (exact)")

	parity_err := 0.0
	for K in strikes {
		c := fin.heston_price(S, K, T, r, params, .Call, 1000)
		p := fin.heston_price(S, K, T, r, params, .Put, 1000)
		// The pricer floors prices at intrinsic value; parity holds where neither floor is active
		if c > math.max(S - K * math.exp(-r * T), 0.0) + 1e-6 && p > math.max(K * math.exp(-r * T) - S, 0.0) + 1e-6 {
			parity_err = math.max(parity_err, abs((c - p) - (S - K * math.exp(-r * T))))
		}
	}
	expect(&passed, &total, parity_err < 1e-8, fmt.tprintf("put-call parity (max err %.1e)", parity_err))

	prospect := bfin.Prospect_Params{alpha = 0.9, beta = 0.85, lambda = 2.1, gamma = 0.65}
	same = true
	for K in strikes {
		direct := bfin.behavioral_heston_price(S, K, T, r, params, prospect, .Call)
		pre := bfin.behavioral_heston_precompute(S, K, T, r, params, .Call)
		cached := bfin.behavioral_heston_price_precomputed(&pre, K, T, r, prospect, .Call)
		delete(pre.heston_cdf)
		same &&= direct == cached
	}
	expect(&passed, &total, same, "precomputed behavioral price == direct price (exact)")

	fmt.printf("Heston pricing summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
