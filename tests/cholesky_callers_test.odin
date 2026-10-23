package tests

import ml "../wotan/analytics/ML"
import fin "../wotan/finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Callers that solve with a Cholesky factor L (non-unit diagonal) agree with
// reference inverses from the LU solver:
// - min-variance / max-Sharpe portfolio weights (Σ⁻¹1, Σ⁻¹(μ - rf))
// - OLS and ridge coefficient covariance σ²(XᵀX [+ λI])⁻¹
// - GLS β = (XᵀΩ⁻¹X)⁻¹XᵀΩ⁻¹y on noisy AR(1) errors (noise-free data cannot
//   tell a wrong whitening apart)

cholesky_callers_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Cholesky Callers Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	M :: l.Matrix(f64)
	at :: proc(a: M, i, j: int) -> f64 {return a.data[i * a.cols + j]}
	inverse :: proc(a: ^M) -> M {r, ok := l.mat_inverse_lu(a); assert(ok); return r}
	matvec :: proc(a: M, x: []f64) -> []f64 {
		o := make([]f64, a.rows)
		for i in 0 ..< a.rows {for j in 0 ..< a.cols {o[i] += at(a, i, j) * x[j]}}
		return o
	}
	vrel :: proc(a, b: []f64) -> f64 {
		s, n := 0.0, 0.0
		for v, i in a {s += (v - b[i]) * (v - b[i]); n += b[i] * b[i]}
		return math.sqrt(s / max(n, 1e-300))
	}
	gram :: proc(X: M) -> M {
		G := l.matrix_new(f64, X.cols, X.cols)
		for i in 0 ..< X.cols {for j in 0 ..< X.cols {s := 0.0; for k in 0 ..< X.rows {s += at(X, k, i) * at(X, k, j)}; G.data[i * X.cols + j] = s}}
		return G
	}
	normalise :: proc(x: []f64) {
		s := 0.0
		for v in x {s += v}
		for &v in x {v /= s}
	}

	// 1. portfolios on a correlated 5-asset covariance
	{
		n := 5
		B := l.matrix_new(f64, n, n)
		defer l.matrix_free(&B)
		for &v in B.data {v = 0.1 * rand.norm_float64()}
		S := gram(B)
		defer l.matrix_free(&S)
		for i in 0 ..< n {S.data[i * n + i] += 0.01}
		Si := inverse(&S)
		defer l.matrix_free(&Si)
		ones := []f64{1, 1, 1, 1, 1}
		ref_mv := matvec(Si, ones)
		defer delete(ref_mv)
		normalise(ref_mv)
		w_mv := fin.min_variance_portfolio(&S, allocator)
		defer delete(w_mv, allocator)
		mu := []f64{0.08, 0.05, 0.12, 0.07, 0.10}
		ex := []f64{0.06, 0.03, 0.10, 0.05, 0.08} // mu - rf, rf = 0.02
		ref_ms := matvec(Si, ex)
		defer delete(ref_ms)
		normalise(ref_ms)
		w_ms := fin.max_sharpe_portfolio(mu, &S, 0.02, allocator)
		defer delete(w_ms, allocator)
		expect(&passed, &total, vrel(w_mv, ref_mv) < 1e-12 && vrel(w_ms, ref_ms) < 1e-12,
			fmt.tprintf("portfolio weights match Σ⁻¹ references (min-var %.1e, max-Sharpe %.1e)", vrel(w_mv, ref_mv), vrel(w_ms, ref_ms)))
	}

	// 2. OLS and ridge covariance of beta
	{
		N, p := 200, 4
		X := l.matrix_new(f64, N, p)
		defer l.matrix_free(&X)
		y := make([]f64, N)
		defer delete(y)
		for i in 0 ..< N {
			X.data[i * p] = 1
			for j in 1 ..< p {X.data[i * p + j] = rand.norm_float64() * f64(j)}
			y[i] = 1 + 0.5 * at(X, i, 1) - 0.2 * at(X, i, 2) + 0.1 * at(X, i, 3) + 0.3 * rand.norm_float64()
		}
		XtX := gram(X)
		defer l.matrix_free(&XtX)
		o := ml.ols_fit_full(&X, y, .Cholesky, allocator)
		XtXi := inverse(&XtX)
		defer l.matrix_free(&XtXi)
		ref := make([]f64, p * p)
		defer delete(ref)
		for v, i in XtXi.data {ref[i] = v * o.sigma2}
		err_ols := vrel(o.vcov.data, ref)

		lam := 5.0
		r := ml.ridge_fit(&X, y, lam, .Cholesky, allocator)
		for i in 0 ..< p {XtX.data[i * p + i] += lam}
		XtXli := inverse(&XtX)
		defer l.matrix_free(&XtXli)
		for v, i in XtXli.data {ref[i] = v * r.sigma2}
		err_ridge := vrel(r.vcov.data, ref)
		expect(&passed, &total, err_ols < 1e-12 && err_ridge < 1e-12,
			fmt.tprintf("OLS / ridge vcov match σ²(XᵀX [+λI])⁻¹ (%.1e / %.1e)", err_ols, err_ridge))
	}

	// 3. GLS with AR(1) errors
	{
		n := 60
		rho := 0.7
		X := l.matrix_new(f64, n, 2)
		defer l.matrix_free(&X)
		y := make([]f64, n)
		defer delete(y)
		Om := l.matrix_new(f64, n, n)
		defer l.matrix_free(&Om)
		e := 0.0
		for i in 0 ..< n {
			X.data[i * 2], X.data[i * 2 + 1] = 1, f64(i) / 10
			e = rho * e + rand.norm_float64()
			y[i] = 3 + 2 * at(X, i, 1) + e
			for j in 0 ..< n {Om.data[i * n + j] = math.pow(rho, f64(abs(i - j))) / (1 - rho * rho)}
		}
		Oi := inverse(&Om)
		defer l.matrix_free(&Oi)
		A := l.matrix_new(f64, 2, 2)
		defer l.matrix_free(&A)
		bv := make([]f64, 2)
		defer delete(bv)
		for a in 0 ..< 2 {
			for b in 0 ..< 2 {
				s := 0.0
				for i in 0 ..< n {for k in 0 ..< n {s += at(X, i, a) * at(Oi, i, k) * at(X, k, b)}}
				A.data[a * 2 + b] = s
			}
			s := 0.0
			for i in 0 ..< n {for k in 0 ..< n {s += at(X, i, a) * at(Oi, i, k) * y[k]}}
			bv[a] = s
		}
		Ai := inverse(&A)
		defer l.matrix_free(&Ai)
		ref := matvec(Ai, bv)
		defer delete(ref)
		g := ml.gls_fit(&X, y, &Om, .Cholesky, allocator)
		expect(&passed, &total, vrel(g.beta, ref) < 1e-10,
			fmt.tprintf("GLS beta %v matches (XᵀΩ⁻¹X)⁻¹XᵀΩ⁻¹y (rel err %.1e)", g.beta, vrel(g.beta, ref)))
	}

	fmt.printf("Cholesky callers summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
