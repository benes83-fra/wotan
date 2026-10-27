package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// Linear models against statsmodels 0.13.5 / scikit-learn 1.2.1 on a fixed
// 15-point dataset (reference values hardcoded):
// - OLS: β, standard errors, p-values, t-interval, R², adjusted R², F; uncentered
//   R²/F without an intercept; rank-deficient X does not panic (pinv, df = n - rank)
// - WLS / GLS: β, standard errors and the weighted / whitened R²
// - Ridge(α=2) and Lasso(α=3/n) with an unpenalised intercept; ridge sandwich SEs
// - LogisticRegression(C=1) objective: weights and intercept at its exact optimum
//   (scipy BFGS, gtol 1e-15; sklearn's own lbfgs stops ~1e-6 away)
// - degenerate inputs (n < p, λ = 0 on singular X'X) return instead of panicking

ml_linear_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML Linear Models vs statsmodels / sklearn ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	close :: proc(got, want: []f64, rtol: f64) -> bool {
		if len(got) != len(want) do return false
		for g, i in got {
			if !(math.abs(g - want[i]) <= rtol * math.max(math.abs(want[i]), 1e-12)) do return false
		}
		return true
	}

	x1 := []f64{0.5, -1.2, 0.3, 1.8, -0.7, 2.1, -0.4, 0.9, -1.6, 1.1, 0.0, -0.9, 1.4, -0.2, 0.6}
	x2 := []f64{1.0, 0.4, -0.8, 0.2, 1.5, -1.1, 0.7, -0.3, 0.9, -1.4, 0.5, -0.6, 1.2, 0.1, -0.9}
	y := []f64{2.9, -0.8, 1.4, 5.0, 0.2, 4.1, 0.9, 2.8, -1.7, 3.6, 1.2, -0.1, 3.9, 1.6, 2.5}
	wts := []f64{1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.8, 1.2, 2.5, 0.6, 1.0, 1.8, 0.9, 1.3, 2.2}
	n := len(y)

	XA := l.matrix_new(f64, n, 3) // [1, x1, x2]
	XF := l.matrix_new(f64, n, 2) // [x1, x2]
	for i in 0 ..< n {
		XA.data[i * 3 + 0], XA.data[i * 3 + 1], XA.data[i * 3 + 2] = 1, x1[i], x2[i]
		XF.data[i * 2 + 0], XF.data[i * 2 + 1] = x1[i], x2[i]
	}

	// --- OLS ---
	o := ml.ols_fit(&XA, y)
	expect(&passed, &total, close(o.beta, {1.3824096056643713, 1.7692692053929386, 0.15539989648610547}, 1e-10), "OLS beta")
	expect(&passed, &total, close(o.stderr, {0.11518646662089421, 0.11237556148530507, 0.13695840074860047}, 1e-10), "OLS standard errors")
	expect(&passed, &total, close(o.pvalues, {4.8292282626047534e-08, 2.2311165342008323e-09, 0.2786788673244642}, 1e-8), "OLS p-values")
	expect(&passed, &total, close(o.ci_low, {1.13143985438717, 1.5244238902881257, -0.14300682419512906}, 1e-10), "OLS 95% t-interval")
	expect(&passed, &total, close({o.r2, o.r2_adj, o.fstat}, {0.9585876195580318, 0.9516855561510371, 138.8842094070854}, 1e-10), "OLS R², adj R², F (df_model = p - 1)")
	oq := ml.ols_fit(&XA, y, .QR)
	expect(&passed, &total, close(oq.beta, o.beta, 1e-12), "OLS QR path = Cholesky path")
	on := ml.ols_fit(&XF, y)
	expect(&passed, &total, close({on.r2, on.fstat}, {0.7255297613723737, 17.181984729931127}, 1e-10), "OLS without intercept: uncentered R², F")

	XC := l.matrix_new(f64, n, 4) // [1, x1, x2, x1 + x2]: rank 3
	for i in 0 ..< n {
		XC.data[i * 4 + 0], XC.data[i * 4 + 1], XC.data[i * 4 + 2], XC.data[i * 4 + 3] = 1, x1[i], x2[i], x1[i] + x2[i]
	}
	oc := ml.ols_fit(&XC, y)
	expect(&passed, &total, oc.rank == 3 && oc.df_resid == n - 3 && math.abs(oc.r2 - o.r2) < 1e-10, "OLS rank-deficient X: pinv, rank 3, same fit")
	Xs := l.matrix_new(f64, 2, 3) // n < p
	Xs.data = {1, 0.5, 1.0, 1, -1.2, 0.4}
	os := ml.ols_fit(&Xs, []f64{1, 2})
	expect(&passed, &total, os.df_resid == 0 && math.is_nan(os.sigma2), "OLS with n < p returns (σ² = NaN)")

	// --- WLS / GLS ---
	ow := ml.wls_fit(&XA, y, wts)
	expect(&passed, &total, close(ow.beta, {1.3144646221661127, 1.728119336357687, 0.1974140441960714}, 1e-10), "WLS beta")
	expect(&passed, &total, math.abs(ow.r2 - 0.9618583057724291) < 1e-10, "WLS weighted R²")
	expect(&passed, &total, math.abs(ow.residuals[0] - (y[0] - ow.fitted[0])) < 1e-14 && math.abs(ow.fitted[0] - (ow.beta[0] + 0.5 * ow.beta[1] + ow.beta[2])) < 1e-12, "WLS residuals on the original scale")
	Om := l.matrix_new(f64, n, n)
	for i in 0 ..< n do for j in 0 ..< n do Om.data[i * n + j] = math.pow(0.5, f64(abs(i - j)))
	og := ml.gls_fit(&XA, y, &Om)
	expect(&passed, &total, close(og.beta, {1.4270333948140017, 1.7600053753209752, 0.12827099108084555}, 1e-10), "GLS beta")
	expect(&passed, &total, close(og.stderr, {0.25140943181340764, 0.10491907829562884, 0.13215003181198354}, 1e-10), "GLS standard errors")
	expect(&passed, &total, math.abs(og.r2 - 0.9665059329614065) < 1e-10, "GLS whitened R²")

	// --- Ridge / Lasso ---
	rr := ml.ridge_fit(&XA, y, 2.0)
	expect(&passed, &total, close(rr.beta, {1.4469791337830165, 1.5499686924062805, 0.04316345096536588}, 1e-10), "Ridge beta = sklearn Ridge(alpha=2), intercept unpenalised")
	expect(&passed, &total, close(rr.stderr, {0.1287257945464131, 0.10947966397991803, 0.1267888527386127}, 1e-9), "Ridge sandwich standard errors")
	rq := ml.ridge_fit(&XA, y, 2.0, .QR)
	expect(&passed, &total, close(rq.beta, rr.beta, 1e-10), "Ridge QR path applies the penalty")
	r0 := ml.ridge_fit(&XC, y, 0.0)
	expect(&passed, &total, math.abs(r0.r2 - o.r2) < 1e-10, "Ridge λ = 0 on singular X'X does not panic")
	la := ml.lasso_fit(&XA, y, 3.0, 100000, 1e-13)
	expect(&passed, &total, close(la.beta[:2], {1.4529713114754097, 1.5420081967213113}, 1e-9) && la.beta[2] == 0, "Lasso beta = sklearn Lasso(alpha=3/n)")

	// --- Logistic ---
	yb := make([]f64, n)
	for v, i in y do yb[i] = 1.0 if v > 1.5 else 0.0
	lp := ml.LogisticParams{C = 1.0, max_iter = 5000, tol = 1e-10, learning_rate = 1.0, fit_intercept = true, optimizer_type = .LBFGS}
	lg := ml.logistic_fit(&XF, yb, lp)
	expect(&passed, &total, close({lg.weights[0], lg.weights[1], lg.bias}, {1.510946052228875, -0.15795052552912978, -0.1445721264385722}, 1e-6), "Logistic C=1 = exact optimum of sklearn's objective")
	lp.fit_intercept = false
	lg0 := ml.logistic_fit(&XF, yb, lp)
	expect(&passed, &total, lg0.bias == 0, "Logistic fit_intercept=false keeps bias at 0")

	fmt.printf("ML linear models summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
