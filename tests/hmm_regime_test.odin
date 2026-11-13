package tests

import ts "../wotan/analytics"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

// Gaussian HMM reference checks against hmmlearn 0.3.3 (GaussianHMM, diag, no priors,
// min_covar 0, the same initial parameters) and numpy/scipy, on a deterministic 3-regime
// series (calm / normal / stress, sticky Markov chain): the forward log-likelihood, one
// Baum-Welch step, the fit to convergence and its Viterbi path; a bad tick (-200%) far
// from every state (the log-likelihood and the Viterbi path at fixed parameters); returns
// scaled by 1e-3 (the same fit, scaled); and a single state (the sample mean and variance)
hmm_reference_checks :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== HMM Reference Checks (hmmlearn) ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}
	// largest relative difference of the parameters from hmmlearn's
	param_err :: proc(h: ^ml_fin.HMM, pi, A, mu, var: []f64, mu_scale := 1.0) -> f64 {
		e := 0.0
		for i in 0 ..< h.n_states {
			e = max(
				e,
				abs(h.initial[i] - pi[i]),
				rel(h.emission_mu[i], mu[i] * mu_scale),
				rel(h.emission_var[i], var[i] * mu_scale * mu_scale),
			)
			for j in 0 ..< h.n_states {e = max(e, abs(h.transition[i * h.n_states + j] - A[i * h.n_states + j]))}
		}
		return e
	}
	path_hash :: proc(p: []int) -> int {
		s := 0
		for v, t in p {s += t * v}
		return s
	}

	T := 1000
	MU := [3]f64{0.0006, 0.0, -0.003}
	SD := [3]f64{0.006, 0.012, 0.03}
	P := [3][3]f64{{0.98, 0.015, 0.005}, {0.02, 0.96, 0.02}, {0.02, 0.08, 0.90}}
	y := make([]f64, T)
	S := make([]int, T)
	defer {delete(y); delete(S)}
	state: u64 = 11
	s := 0
	for t in 0 ..< T {
		if t > 0 {
			u, c, nxt := garch_ref_uniform(&state), 0.0, 2
			for j in 0 ..< 3 {
				c += P[s][j]
				if u < c {nxt = j; break}
			}
			s = nxt
		}
		S[t] = s
		y[t] = MU[s] + SD[s] * garch_ref_normal(&state)
	}
	expect(
		&passed,
		&total,
		rel(y[999], 0.007493756490277018) < 1e-12,
		"reference series reproduced",
	)

	h := ml_fin.hmm_new(3)
	defer ml_fin.hmm_free(&h)
	ml_fin.hmm_init_simple(&h, y)
	ll0 := ml_fin.hmm_log_likelihood(&h, y)
	ordered := h.emission_var[0] < h.emission_var[1] && h.emission_var[1] < h.emission_var[2]
	expect(
		&passed,
		&total,
		rel(ll0, 2891.447943728515) < 1e-12 && ordered,
		fmt.tprintf(
			"log-likelihood at the initial parameters %.10f (numpy 2891.4479437285), states ordered calm to turbulent",
			ll0,
		),
	)

	ll1, _ := ml_fin.hmm_fit(&h, y, 1)
	e1 := param_err(
		&h,
		{0.56334547740212, 0.42808007569637274, 0.008574446901507275},
		{
			0.7187750193066759,
			0.18297626828134972,
			0.09824871241197444,
			0.004114140280387112,
			0.9664303383561529,
			0.029455521363459938,
			0.008103740188857102,
			0.04751861315764422,
			0.9443776466534988,
		},
		{-0.00019362864849571004, 6.636612563685753e-05, -0.002872198999513615},
		{1.1288932239754314e-05, 7.104568704759728e-05, 0.0005076224940115977},
	)
	expect(
		&passed,
		&total,
		e1 < 1e-10 && ll1 == ll0,
		fmt.tprintf("one Baum-Welch step: parameters as hmmlearn's (largest difference %.1e)", e1),
	)

	FIT_PI := []f64{1.0, 2.3531765719453375e-67, 2.1586651054623808e-126}
	FIT_A := []f64 {
		0.9690433432581083,
		0.00887289030331759,
		0.022083766438574248,
		0.01779206595489581,
		0.9730065321244702,
		0.00920140192063406,
		7.2259321771919e-20,
		0.08440802746965234,
		0.9155919725303476,
	}
	FIT_MU := []f64{0.0006871273326813659, -0.0002561128723230155, -0.007649505600525067}
	FIT_VAR := []f64{4.005247476584993e-05, 0.00015459152338728724, 0.000892069297410845}
	ml_fin.hmm_init_simple(&h, y)
	_, converged := ml_fin.hmm_fit(&h, y, 2000, 1e-8)
	ef := param_err(&h, FIT_PI, FIT_A, FIT_MU, FIT_VAR)
	llf := ml_fin.hmm_log_likelihood(&h, y)
	expect(
		&passed,
		&total,
		converged && ef < 1e-6 && rel(llf, 2983.917698216192) < 1e-11,
		fmt.tprintf(
			"fit converges to hmmlearn's (40 iterations): LL %.9f vs 2983.917698216, largest parameter difference %.1e",
			llf,
			ef,
		),
	)

	path := ml_fin.hmm_decode(&h, y)
	defer delete(path)
	counts: [3]int
	agree := 0
	for v, t in path {
		counts[v] += 1
		if v == S[t] {agree += 1}
	}
	expect(
		&passed,
		&total,
		path_hash(path) == 421111 && counts == {314, 566, 120} && agree == 909,
		fmt.tprintf(
			"Viterbi path as hmmlearn's (%v days per state; %d/1000 on the simulated regimes)",
			counts,
			agree,
		),
	)

	// a bad tick at -200%: every state's density underflows in linear space
	y2 := make([]f64, T)
	defer delete(y2)
	copy(y2, y)
	y2[500] = -2.0
	ml_fin.hmm_init_simple(&h, y)
	llo := ml_fin.hmm_log_likelihood(&h, y2)
	po := ml_fin.hmm_decode(&h, y2)
	defer delete(po)
	expect(
		&passed,
		&total,
		rel(llo, -330.57355936627545) < 1e-10 && path_hash(po) == 701690 && po[500] == 2,
		fmt.tprintf(
			"bad tick: log-likelihood %.8f (numpy -330.57355937), Viterbi path as hmmlearn's, the tick in the turbulent state",
			llo,
		),
	)

	// returns in other units: the same fit, scaled (no absolute variance floor)
	ys := make([]f64, T)
	defer delete(ys)
	for v, t in y {ys[t] = v * 1e-3}
	ml_fin.hmm_init_simple(&h, ys)
	ml_fin.hmm_fit(&h, ys, 2000, 1e-8)
	es := param_err(&h, FIT_PI, FIT_A, FIT_MU, FIT_VAR, 1e-3)
	lls := ml_fin.hmm_log_likelihood(&h, ys)
	ps := ml_fin.hmm_decode(&h, ys)
	defer delete(ps)
	expect(
		&passed,
		&total,
		es < 1e-6 &&
		rel(lls, 2983.917698216192 + 6907.755278982137) < 1e-11 &&
		path_hash(ps) == 421111,
		fmt.tprintf(
			"returns x 1e-3: the same fit scaled (largest difference %.1e), LL + 1000 ln(1000), the same Viterbi path",
			es,
		),
	)

	// one state: the Gaussian MLE
	h1 := ml_fin.hmm_new(1)
	defer ml_fin.hmm_free(&h1)
	ml_fin.hmm_init_simple(&h1, y)
	ml_fin.hmm_fit(&h1, y, 10, 1e-12)
	ll_1 := ml_fin.hmm_log_likelihood(&h1, y)
	expect(
		&passed,
		&total,
		rel(h1.emission_mu[0], -0.0010054743170224702) < 1e-12 &&
		rel(h1.emission_var[0], 0.00023016263053018943) < 1e-12 &&
		h1.transition[0] == 1 &&
		rel(ll_1, 2769.4236716199357) < 1e-12,
		fmt.tprintf("one state: sample mean and variance, LL %.10f (scipy 2769.4236716199)", ll_1),
	)

	fmt.printf("HMM reference summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

hmm_regime_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	hmm_reference_checks(context.allocator)
	fmt.println("\n=== Hidden Markov Model (HMM) Regime Switching ===")
	main_alloc := context.allocator

	fmt.println("\n--- Fetching Market Data ---")
	spy, spy_ok := yahoo.read_yahoo_series("SPY", .Daily, .FiveYears, allocator)
	if !spy_ok {fail_missing_data(#procedure, "Yahoo data for SPY"); return}
	defer yahoo.yahoo_series_free(&spy)

	returns := ts.log_returns(spy.adj_close, 1, main_alloc)
	defer delete(returns, main_alloc)
	num_days := len(returns)
	fmt.printf("Loaded %d daily returns for SPY.\n", num_days)

	// 1. Initialize and Train HMM (3 States: e.g., Low-Vol Bull, High-Vol Bear, Choppy)
	n_states := 3
	hmm := ml_fin.hmm_new(n_states, main_alloc)
	defer ml_fin.hmm_free(&hmm)
	// ✅ Clip extreme outliers to prevent degenerate HMM states
	// Daily SPY returns beyond ±5% are typically black swan events
	for i in 0 ..< num_days {
		if returns[i] > 0.05 {returns[i] = 0.05}
		if returns[i] < -0.05 {returns[i] = -0.05}
	}
	ml_fin.hmm_init_simple(&hmm, returns)

	fmt.println("\n--- Training HMM (Baum-Welch) ---")
	ll, converged := ml_fin.hmm_fit(&hmm, returns, 1000, 1e-5)
	if converged {
		fmt.println("  ✅ HMM converged successfully.")
	} else {
		fmt.println("  ⚠️  HMM reached max iterations without full convergence.")
	}

	// 2. Decode the most likely regime sequence (Viterbi)
	fmt.println("\n--- Decoding Regimes (Viterbi) ---")
	states := ml_fin.hmm_decode(&hmm, returns, main_alloc)
	defer delete(states, main_alloc)

	// 3. Output Dashboard
	fmt.println(
		"\n╔══════════════════════════════════════════════════════════════╗",
	)
	fmt.println("║               HMM REGIME SWITCHING DASHBOARD                 ║")
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf("║  Log-Likelihood: %+10.4f   Converged: %-5v              ║\n", ll, converged)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║ %-10s | %-12s | %-12s | %-10s ║\n",
		"Regime",
		"Mean (Ann.)",
		"Vol (Ann.)",
		"Frequency",
	)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)

	for i in 0 ..< n_states {
		count := 0
		for t in 0 ..< num_days {
			if states[t] == i {count += 1}
		}
		freq := f64(count) / f64(num_days) * 100.0

		// Annualize daily returns
		mean_ann := hmm.emission_mu[i] * 252.0 * 100.0
		vol_ann := math.sqrt(hmm.emission_var[i]) * math.sqrt_f64(252.0) * 100.0

		fmt.printf("║ %-10d | %+10.2f%% | %10.2f%% | %8.1f%%  ║\n", i, mean_ann, vol_ann, freq)
	}
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.println("║  Transition Matrix (Probability of moving from Row to Col)   ║")
	for i in 0 ..< n_states {
		fmt.print("║ ")
		for j in 0 ..< n_states {
			fmt.printf("%6.3f  ", hmm.transition[i * n_states + j])
		}
		fmt.println("   ║")
	}
	fmt.println(
		"╚══════════════════════════════════════════════════════════════╝",
	)

	// Print the last 20 days of decoded regimes
	fmt.println("\n--- Last 20 Trading Days Regime Sequence ---")
	fmt.print("  ")
	start_day := num_days - 20
	if start_day < 0 {start_day = 0}
	for t in start_day ..< num_days {
		fmt.printf("%d ", states[t])
	}
	fmt.println("\n  (0=Low Vol, 1=Choppy, 2=High Vol typically)")

	fmt.println("\n✓ Hidden Markov Model Regime Test Complete!")
}
