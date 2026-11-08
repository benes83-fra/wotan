package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Extrapolative expectations: the X-CAPM of Barberis, Greenwood, Jin & Shleifer (2015),
// behavioral_finance/extrapolation.odin
//   xcapm_reference_test: eq. 28 (all 13 derived parameters for mu 0.25, beta 0.5) to
//     the printed decimals and to a separate Python solution of Appendix A; the
//     residuals of all of Appendix A's conditions; the rational benchmark (Corollary
//     1); Tables 3 - 6 (D/r - P's predictive coefficient, autocorrelation of P - D/r,
//     sd of annual price changes and of P - D/r, autocorrelation of quarterly price
//     changes) on 9 (mu, beta) pairs to the printed decimals; Corollary 2's signs;
//     Proposition 2 (extrapolators expect S t, the rational expectation starts at the
//     true drift, the population-average expectation is extrapolative)
//   xcapm_test: 10,000 simulated years of quarters (exact transitions) in the base
//     economy and the rational one: D/r - P predicts annual price changes with
//     Proposition 3's slope, quarterly price changes have Proposition 6's negative
//     autocorrelation, annual changes Proposition 5's sd; the survey fact
//     (Greenwood-Shleifer 2014): extrapolators' expectations rise with past price
//     changes while realized future changes fall with those expectations; in the
//     rational economy sentiment predicts nothing and the sd is sigma_D / r. On 9
//     seeds: slope within 1.2 se, autocorrelation within 1.6 se, sd within 2.7 se,
//     rational |t| <= 2.2.

// Tables 3 - 6: [beta 0.05, 0.5, 0.75][...][mu 0.75, 0.5, 0.25]
@(private = "file")
X_T3 := [3][5][3]f64 {
	{
		{.014, .016, .022},
		{.055, .064, .085},
		{.106, .124, .162},
		{.155, .180, .233},
		{.201, .233, .298},
	},
	{
		{.134, .161, .219},
		{.438, .504, .628},
		{.684, .754, .861},
		{.822, .878, .948},
		{.900, .940, .981},
	},
	{
		{.194, .232, .311},
		{.579, .652, .774},
		{.822, .879, .949},
		{.925, .958, .988},
		{.968, .985, .997},
	},
}
@(private = "file")
X_T4 := [3][5][3]f64 {
	{
		{.986, .984, .978},
		{.945, .936, .915},
		{.894, .876, .838},
		{.845, .820, .767},
		{.799, .767, .702},
	},
	{
		{.866, .839, .781},
		{.562, .496, .372},
		{.316, .246, .139},
		{.178, .122, .052},
		{.100, .060, .019},
	},
	{
		{.806, .768, .689},
		{.421, .348, .226},
		{.178, .121, .051},
		{.075, .042, .012},
		{.032, .015, .003},
	},
}
@(private = "file")
X_T5A := [3][3]f64{{11.20, 13.15, 17.43}, {11.17, 13.03, 16.86}, {11.04, 12.67, 15.90}}
@(private = "file")
X_T5B := [3][3]f64{{3.66, 8.92, 18.29}, {1.41, 3.41, 6.94}, {1.16, 2.80, 5.70}}
@(private = "file")
X_T6 := [3][6][3]f64 {
	{
		{-.001, -.004, -.007},
		{-.001, -.003, -.007},
		{-.001, -.003, -.007},
		{-.001, -.003, -.007},
		{-.001, -.003, -.006},
		{-.001, -.003, -.006},
	},
	{
		{-.016, -.038, -.079},
		{-.013, -.032, -.062},
		{-.012, -.027, -.048},
		{-.010, -.022, -.038},
		{-.006, -.011, -.014},
		{-.003, -.006, -.005},
	},
	{
		{-.022, -.054, -.110},
		{-.018, -.041, -.076},
		{-.014, -.032, -.052},
		{-.012, -.024, -.036},
		{-.005, -.008, -.008},
		{-.002, -.003, -.002},
	},
}

xcapm_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== X-CAPM (BGJS 2015) Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	base := bfin.XCAPM_2015
	s, ok0 := bfin.xcapm_solve(base)

	// 1. eq. 28: eta_e0, eta_r0, eta_e1, eta_r1, a_e, a_r, b_e, b_r, c_e, c_r, A, B,
	// sigma_P as printed, and = a separate Python solution (scratch xcapm.py)
	{
		got := [13]f64 {
			s.eta_e0,
			s.eta_r0,
			s.eta_e1,
			s.eta_r1,
			s.a_e,
			s.a_r,
			s.b_e,
			s.b_r,
			s.c_e,
			s.c_r,
			s.A,
			s.B,
			s.sigma_P,
		}
		PRINTED := [13]f64 {
			1.54,
			15.39,
			0.51,
			-1.54,
			-1.22e-3,
			-1.28e-3,
			-7.31e-3,
			0.042,
			1.63,
			-3.47,
			-117.04,
			0.99,
			19.75,
		}
		HALF := [13]f64 {
			5e-3,
			5e-3,
			5e-3,
			5e-3,
			5e-6,
			5e-6,
			5e-6,
			5e-4,
			5e-3,
			5e-3,
			5e-3,
			5e-3,
			5e-3,
		}
		PY := [13]f64 {
			1.5374882135929369,
			15.38753535922114,
			0.5124960711976474,
			-1.5374882135929429,
			-0.0012187598220058816,
			-0.001280839932707138,
			-0.0073125589320352504,
			0.0416861543910049,
			1.625799675878637,
			-3.4688359675845795,
			-117.0377283607646,
			0.9874238797449562,
			19.751601484501208,
		}
		ok := ok0
		rel := 0.0
		for i in 0 ..< 13 {
			ok &&= abs(got[i] - PRINTED[i]) <= HALF[i]
			rel = max(rel, abs(got[i] - PY[i]) / abs(PY[i]))
		}
		expect(
			&passed,
			&total,
			ok && rel < 1e-9,
			fmt.tprintf(
				"eq. 28 (mu 0.25, beta 0.5): A %.2f, B %.4f, sigma_P %.2f, demands %.2f + %.2f S / %.2f %.2f S and value-function coefficients as printed; = Python within %.1e",
				s.A,
				s.B,
				s.sigma_P,
				s.eta_e0,
				s.eta_e1,
				s.eta_r0,
				s.eta_r1,
				rel,
			),
		)
	}

	// 2. all of Appendix A's conditions hold (residuals relative to their largest term
	// scale ~1) on 9 (mu, beta) pairs
	{
		worst := 0.0
		for beta in ([3]f64{0.05, 0.5, 0.75}) {
			for mu in ([3]f64{0.75, 0.5, 0.25}) {
				p := base
				p.beta, p.mu = beta, mu
				q, ok := bfin.xcapm_solve(p)
				if !ok {worst = math.inf_f64(1); continue}
				for v in bfin.xcapm_residuals(q, p) {worst = max(worst, abs(v))}
			}
		}
		expect(
			&passed,
			&total,
			worst < 1e-10,
			fmt.tprintf(
				"Appendix A (A24)-(A26), (A28)-(A30), (A33), (A34) hold on 9 (mu, beta) pairs, largest residual %.1e",
				worst,
			),
		)
	}

	// 3. rational benchmark (mu = 1, Corollary 1): P = (g_D - gamma sigma_D^2 Q) / r^2 +
	// D / r (A = 30), B = 0, everyone holds Q, sd of annual price changes sigma_D / r = 10
	{
		p := base
		p.mu = 1
		q, ok := bfin.xcapm_solve(p)
		A := (p.g_D - p.gamma * p.sigma_D * p.sigma_D * p.Q) / (p.r * p.r)
		ok &&= q.B == 0 && abs(q.A - A) < 1e-9 && abs(q.eta_r0 - p.Q) < 1e-12
		ok &&= abs(math.sqrt(bfin.xcapm_price_change_variance(q, p, 1)) - 10) < 1e-12
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"rational benchmark: A = %.4f (Corollary 1: 30), B = 0, demand = Q, sd of annual price changes 10",
				q.A,
			),
		)
	}

	// 4. Tables 3 - 6 at their printed decimals (half a unit in the last digit)
	{
		w3, w4, w5a, w5b, w6 := 0.0, 0.0, 0.0, 0.0, 0.0
		ok := true
		for beta, bi in ([3]f64{0.05, 0.5, 0.75}) {
			for mu, mi in ([3]f64{0.75, 0.5, 0.25}) {
				p := base
				p.beta, p.mu = beta, mu
				q, found := bfin.xcapm_solve(p)
				ok &&= found
				for kq, i in ([5]f64{1, 4, 8, 12, 16}) {
					w3 = max(w3, abs(bfin.xcapm_dp_coefficient(q, kq / 4) - X_T3[bi][i][mi]))
					w4 = max(w4, abs(bfin.xcapm_pd_autocorrelation(q, kq / 4) - X_T4[bi][i][mi]))
				}
				w5a = max(
					w5a,
					abs(math.sqrt(bfin.xcapm_price_change_variance(q, p, 1)) - X_T5A[bi][mi]),
				)
				w5b = max(w5b, abs(bfin.xcapm_pd_sd(q) - X_T5B[bi][mi]))
				v := bfin.xcapm_price_change_variance(q, p, 0.25)
				for L, i in ([6]f64{1, 2, 3, 4, 8, 12}) {
					c := bfin.xcapm_price_change_covariance(q, p, 0.25, L / 4, L / 4 + 0.25) / v
					w6 = max(w6, abs(c - X_T6[bi][i][mi]))
				}
			}
		}
		expect(
			&passed,
			&total,
			ok && w3 <= 5e-4 && w4 <= 5e-4 && w5a <= 5e-3 && w5b <= 5e-3 && w6 <= 5e-4,
			fmt.tprintf(
				"Tables 3-6 on 9 (mu, beta) pairs: D/r - P coefficient within %.1e, P - D/r autocorrelation %.1e, sd of annual price changes %.1e, of P - D/r %.1e, price-change autocorrelations %.1e",
				w3,
				w4,
				w5a,
				w5b,
				w6,
			),
		)
	}

	// 5. Corollary 2: B in (0, 1 / beta), eta_e1 > 0, eta_r1 < 0, a_r < 0, a_e <= 0;
	// Proposition 2: with lambda = (0, 1) extrapolators expect S t; the rational
	// expectation's slope at t = 0 is the true drift -beta B / (1 - beta B) S + g_D /
	// ((1 - beta B) r); the population-weighted expectation rises with S (mu < 1)
	{
		ok := true
		for beta in ([3]f64{0.05, 0.5, 0.75}) {
			for mu in ([3]f64{0.75, 0.5, 0.25}) {
				p := base
				p.beta, p.mu = beta, mu
				q, _ := bfin.xcapm_solve(p)
				ok &&=
					q.B > 0 &&
					q.B < 1 / beta &&
					q.eta_e1 > 0 &&
					q.eta_r1 < 0 &&
					q.a_r < 0 &&
					q.a_e <= 0
				for S in ([3]f64{-5, 2, 9}) {
					ok &&=
						abs(bfin.xcapm_expected_change_extrapolator(p, S, 1.5) - 1.5 * S) < 1e-12
					h := 1e-6
					drift := -beta * q.B / (1 - beta * q.B) * S + p.g_D / ((1 - beta * q.B) * p.r)
					ok &&=
						abs(bfin.xcapm_expected_change_rational(q, p, S, h) / h - drift) <
						1e-4 * (1 + abs(drift))
				}
				avg :: proc(q: bfin.XCAPM_Solution, p: bfin.XCAPM_Params, S: f64) -> f64 {
					return(
						p.mu * bfin.xcapm_expected_change_rational(q, p, S, 1) +
						(1 - p.mu) * bfin.xcapm_expected_change_extrapolator(p, S, 1) \
					)
				}
				ok &&= avg(q, p, 3) > avg(q, p, 2)
			}
		}
		// a general lambda: eq. 26 against its own ODE, d/dt E = lambda0 + lambda1 E[S_t]
		// with dE[S]/dt = beta lambda0 + beta (lambda1 - 1) E[S] (eq. 17), by RK4
		p := base
		p.lambda0, p.lambda1 = 0.3, 0.6
		E, m := 0.0, 4.0 // E[P_t - P_0], E[S_t] from S_0 = 4
		N :: 2000
		hstep := 2.0 / N
		f :: proc(m: f64, p: bfin.XCAPM_Params) -> f64 {return(
				p.beta * p.lambda0 +
				p.beta * (p.lambda1 - 1) * m \
			)}
		for _ in 0 ..< N {
			k1 := f(m, p)
			k2 := f(m + 0.5 * hstep * k1, p)
			k3 := f(m + 0.5 * hstep * k2, p)
			k4 := f(m + hstep * k3, p)
			// E' = lambda0 + lambda1 m, integrated with the same stages
			E +=
				hstep /
				6 *
				(6 * p.lambda0 +
						p.lambda1 *
							(m +
									2 * (m + 0.5 * hstep * k1) +
									2 * (m + 0.5 * hstep * k2) +
									(m + hstep * k3)))
			m += hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
		}
		eq26 := bfin.xcapm_expected_change_extrapolator(p, 4, 2)
		expect(
			&passed,
			&total,
			ok && abs(E - eq26) < 1e-9,
			fmt.tprintf(
				"Corollary 2 signs on 9 pairs; Proposition 2: extrapolators expect S t, the rational expectation starts at the true drift, the average investor is extrapolative; eq. 26 with lambda (0.3, 0.6) = its ODE (%.10f vs %.10f)",
				eq26,
				E,
			),
		)
	}

	fmt.printf("X-CAPM reference: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// OLS slope of y on a constant and x with its Newey-West se
@(private = "file")
_x_slope :: proc(x, y: []f64, lags: int) -> (slope, se: f64) {
	n := len(x)
	X := l.matrix_new(f64, n, 2)
	defer l.matrix_free(&X)
	for i in 0 ..< n {X.data[2 * i], X.data[2 * i + 1] = 1, x[i]}
	fit := ml.ols_fit_robust(&X, y, {type = .HAC, maxlags = lags}, .QR, context.temp_allocator)
	defer ml._ols_result_free(&fit, context.temp_allocator)
	return fit.beta[1], fit.stderr[1]
}

xcapm_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== X-CAPM: Simulated Economies ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// 10,000 years of quarters (exact transitions), sentiment started from its
	// stationary distribution, in the base economy (mu 0.25, beta 0.5) and the
	// rational one (mu 1)
	YEARS :: 10_000
	N :: 4 * YEARS
	res: [2][8]f64
	for mu, k in ([2]f64{0.25, 1}) {
		p := bfin.XCAPM_2015
		p.mu = mu
		s, _ := bfin.xcapm_solve(p)
		z := make([]f64, 2 * N)
		defer delete(z)
		for &v in z {v = rand.float64_normal(0, 1)}
		S0 := p.g_D / p.r + rand.float64_normal(0, s.sigma_S / math.sqrt(2 * s.k))
		x := bfin.xcapm_simulate(s, p, 10, S0, 0.25, z)
		defer bfin.xcapm_path_free(&x)
		// annual (4-quarter) price changes on D/r - P at the start (overlapping,
		// Newey-West 8 lags) - Proposition 3
		n4 := N - 4
		dp := make([]f64, n4)
		ch := make([]f64, n4)
		defer {delete(dp); delete(ch)}
		for t in 0 ..< n4 {
			dp[t] = x.dividend[t] / p.r - x.price[t]
			ch[t] = x.price[t + 4] - x.price[t]
		}
		if k == 0 {res[k][0], res[k][1] = _x_slope(dp, ch, 8)}
		// survey fact: extrapolators' expected annual change (S, lambda = (0, 1)) rises
		// with the past annual change, and the realized next annual change falls with it
		past := make([]f64, N - 8)
		expd := make([]f64, N - 8)
		next := make([]f64, N - 8)
		defer {delete(past); delete(expd); delete(next)}
		for t in 4 ..< N - 4 {
			past[t - 4] = x.price[t] - x.price[t - 4]
			expd[t - 4] = bfin.xcapm_expected_change_extrapolator(p, x.sentiment[t], 1)
			next[t - 4] = x.price[t + 4] - x.price[t]
		}
		res[k][2], res[k][3] = _x_slope(past, expd, 8)
		res[k][4], res[k][5] = _x_slope(expd, next, 8)
		// lag-1 autocorrelation of quarterly price changes (Proposition 6) and the sd of
		// annual changes (Proposition 5, non-overlapping years)
		q := make([]f64, N)
		defer delete(q)
		for t in 0 ..< N {q[t] = x.price[t + 1] - x.price[t]}
		res[k][6], _ = _x_slope(q[:N - 1], q[1:], 4)
		ss, m := 0.0, 0.0
		for y in 0 ..< YEARS {m += (x.price[4 * y + 4] - x.price[4 * y]) / YEARS}
		for y in 0 ..< YEARS {
			d := x.price[4 * y + 4] - x.price[4 * y] - m
			ss += d * d / (YEARS - 1)
		}
		res[k][7] = math.sqrt(ss)
	}
	p := bfin.XCAPM_2015
	s, _ := bfin.xcapm_solve(p)
	want_b := bfin.xcapm_dp_coefficient(s, 1)
	want_ac :=
		bfin.xcapm_price_change_covariance(s, p, 0.25, 0.25, 0.5) /
		bfin.xcapm_price_change_variance(s, p, 0.25)
	want_sd := math.sqrt(bfin.xcapm_price_change_variance(s, p, 1))
	// se of a lag-1 autocorrelation ~ 1 / sqrt(N); of an sd from YEARS iid ~normal
	// annual changes ~ sd / sqrt(2 YEARS) (annual changes are close to uncorrelated)
	ac_se := 1 / math.sqrt(f64(N))
	sd_se := want_sd / math.sqrt(2 * f64(YEARS))
	expect(
		&passed,
		&total,
		abs(res[0][0] - want_b) < 4 * res[0][1] &&
		abs(res[0][6] - want_ac) < 4 * ac_se &&
		abs(res[0][7] - want_sd) < 4 * sd_se,
		fmt.tprintf(
			"extrapolators' economy (mu 0.25, beta 0.5), %d years: D/r - P predicts annual price changes, slope %.3f (se %.3f; Prop. 3 %.3f); quarterly autocorrelation %.3f (Prop. 6 %.3f); sd of annual changes %.2f (Prop. 5 %.2f)",
			YEARS,
			res[0][0],
			res[0][1],
			want_b,
			res[0][6],
			want_ac,
			res[0][7],
			want_sd,
		),
	)
	expect(
		&passed,
		&total,
		res[0][2] / res[0][3] > 3 && res[0][4] / res[0][5] < -3,
		fmt.tprintf(
			"survey fact (Greenwood-Shleifer): extrapolators' expected annual change rises with the past change (slope %.3f, t %.1f) and the realized next change falls with their expectation (slope %.3f, t %.1f)",
			res[0][2],
			res[0][2] / res[0][3],
			res[0][4],
			res[0][4] / res[0][5],
		),
	)
	expect(
		&passed,
		&total,
		abs(res[1][4] / res[1][5]) < 3 &&
		abs(res[1][6]) < 4 * ac_se &&
		abs(res[1][7] - 10) < 4 * 10 / math.sqrt(2 * f64(YEARS)),
		fmt.tprintf(
			"rational economy (mu 1): sentiment does not predict price changes (t %.1f), quarterly autocorrelation %.3f, sd of annual changes %.2f (sigma_D / r = 10)",
			res[1][4] / res[1][5],
			res[1][6],
			res[1][7],
		),
	)

	fmt.printf("X-CAPM: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
