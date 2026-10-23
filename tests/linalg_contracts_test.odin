package tests

import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Linear-algebra contracts at sizes beyond one SIMD tile / one Jacobi budget
// (the sizes where the blocked LU, blocked QR, eigh and the AᵀA-SVD used to fail):
// - LU solve / inverse, eigh, QR (full and thin), SVD (both methods) reconstruct
//   their input to machine precision with orthonormal factors
// - lstsq agrees across QR / Cholesky / SVD, and the Cholesky path does not
//   panic on rank-deficient data (minimum-norm solution instead)

linalg_contracts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Linear Algebra Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	M :: l.Matrix(f64)
	at :: proc(a: M, i, j: int) -> f64 {return a.data[i * a.cols + j]}
	randn :: proc(r, c: int) -> M {
		o := l.matrix_new(f64, r, c)
		for &v in o.data {v = rand.norm_float64()}
		return o
	}
	mm :: proc(a, b: M) -> M { 	// reference matmul
		o := l.matrix_new(f64, a.rows, b.cols)
		for i in 0 ..< a.rows {
			for k in 0 ..< a.cols {
				aik := at(a, i, k)
				for j in 0 ..< b.cols {o.data[i * o.cols + j] += aik * at(b, k, j)}
			}
		}
		return o
	}
	rel :: proc(a, b: M) -> f64 {
		num, den := 0.0, 0.0
		for v, i in a.data {num += (v - b.data[i]) * (v - b.data[i]); den += b.data[i] * b.data[i]}
		return math.sqrt(num / max(den, 1e-300))
	}
	orth :: proc(q: M, cols: int) -> f64 { 	// ||Q[:, :cols]ᵀ Q[:, :cols] - I||_max
		worst := 0.0
		for a in 0 ..< cols {
			for b in 0 ..< cols {
				s := 0.0
				for i in 0 ..< q.rows {s += at(q, i, a) * at(q, i, b)}
				worst = max(worst, abs(s - (1 if a == b else 0)))
			}
		}
		return worst
	}
	vrel :: proc(a, b: []f64) -> f64 {
		num, den := 0.0, 0.0
		for v, i in a {num += (v - b[i]) * (v - b[i]); den += b[i] * b[i]}
		return math.sqrt(num / max(den, 1e-300))
	}
	matvec :: proc(a: M, x: []f64) -> []f64 {
		o := make([]f64, a.rows)
		for i in 0 ..< a.rows {for j in 0 ..< a.cols {o[i] += at(a, i, j) * x[j]}}
		return o
	}

	// 1. LU (n = 64: four 16-wide panels)
	{
		n := 64
		A := randn(n, n)
		defer l.matrix_free(&A)
		x := make([]f64, n)
		defer delete(x)
		for &v in x {v = rand.norm_float64()}
		b := matvec(A, x)
		defer delete(b)
		LU, piv, _, ok := l.lu_decompose(&A)
		defer {l.matrix_free(&LU); delete(piv)}
		xs := l.lu_solve(&LU, piv, b)
		defer delete(xs)
		Ainv, ok_inv := l.mat_inverse_lu(&A)
		defer l.matrix_free(&Ainv)
		I := l.matrix_new(f64, n, n)
		defer l.matrix_free(&I)
		for i in 0 ..< n {I.data[i * n + i] = 1}
		AAi := mm(A, Ainv)
		defer l.matrix_free(&AAi)
		inv_err := 0.0
		for v, i in AAi.data {inv_err = max(inv_err, abs(v - I.data[i]))}
		expect(&passed, &total, ok && ok_inv && vrel(xs, x) < 1e-10 && inv_err < 1e-10,
			fmt.tprintf("LU n=64: solve rel err %.1e, max |A A⁻¹ - I| %.1e", vrel(xs, x), inv_err))
	}

	// 2. eigh (n = 80: thousands of rotations)
	{
		n := 80
		B := randn(n, n)
		defer l.matrix_free(&B)
		S := l.matrix_new(f64, n, n)
		defer l.matrix_free(&S)
		for i in 0 ..< n {for j in 0 ..< n {s := 0.0; for k in 0 ..< n {s += at(B, k, i) * at(B, k, j)}; S.data[i * n + j] = s}}
		W, V := l.eigh(&S)
		defer {delete(W); l.matrix_free(&V)}
		SV := mm(S, V)
		defer l.matrix_free(&SV)
		VW := l.matrix_new(f64, n, n)
		defer l.matrix_free(&VW)
		for i in 0 ..< n {for j in 0 ..< n {VW.data[i * n + j] = at(V, i, j) * W[j]}}
		ascending := true
		for i in 1 ..< n {if W[i] < W[i - 1] {ascending = false}}
		expect(&passed, &total, rel(SV, VW) < 1e-12 && orth(V, n) < 1e-11 && ascending,
			fmt.tprintf("eigh n=80: ||SV - VW||/||VW|| %.1e, ||VᵀV - I|| %.1e, ascending %v", rel(SV, VW), orth(V, n), ascending))
	}

	// 3. QR (64×48 = three 16-wide blocks) and thin R
	{
		A := randn(64, 48)
		defer l.matrix_free(&A)
		Q, R := l.qr_decompose(&A)
		defer {l.matrix_free(&Q); l.matrix_free(&R)}
		QR := mm(Q, R)
		defer l.matrix_free(&QR)
		below := 0.0
		for i in 0 ..< R.rows {for j in 0 ..< min(i, R.cols) {below = max(below, abs(at(R, i, j)))}}
		Rt := l.qr_decompose_thin(&A)
		defer l.matrix_free(&Rt)
		thin_same := rel(Rt, R) < 1e-14
		expect(&passed, &total, rel(QR, A) < 1e-13 && orth(Q, Q.cols) < 1e-12 && below < 1e-12 && thin_same,
			fmt.tprintf("QR 64x48: ||QR - A|| %.1e, ||QᵀQ - I|| %.1e, below-diag %.1e, thin R matches %v", rel(QR, A), orth(Q, Q.cols), below, thin_same))
	}

	// 4. SVD: both methods on a square and a tall matrix
	{
		worst_recon, worst_orth := 0.0, 0.0
		for shape in ([][2]int{{50, 50}, {120, 40}}) {
			A := randn(shape[0], shape[1])
			defer l.matrix_free(&A)
			for method in ([]l.SVDMethod{.Jacobi, .GolubReinsch}) {
				U, S, V := l.svd(&A, method, allocator)
				defer {l.matrix_free(&U); delete(S); l.matrix_free(&V)}
				k := len(S)
				Rc := l.matrix_new(f64, A.rows, A.cols)
				defer l.matrix_free(&Rc)
				for i in 0 ..< A.rows {
					for j in 0 ..< A.cols {
						s := 0.0
						for p in 0 ..< k {s += at(U, i, p) * S[p] * at(V, j, p)}
						Rc.data[i * A.cols + j] = s
					}
				}
				worst_recon = max(worst_recon, rel(Rc, A))
				worst_orth = max(worst_orth, orth(U, k), orth(V, k))
			}
		}
		expect(&passed, &total, worst_recon < 1e-13 && worst_orth < 1e-10,
			fmt.tprintf("SVD (Jacobi + GolubReinsch, 50x50 / 120x40): recon %.1e, orthogonality %.1e", worst_recon, worst_orth))
	}

	// 5. lstsq: methods agree (120×40 is where Auto selects Cholesky)
	{
		X := randn(120, 40)
		defer l.matrix_free(&X)
		beta := make([]f64, 40)
		defer delete(beta)
		for &v in beta {v = rand.norm_float64()}
		y := matvec(X, beta)
		defer delete(y)
		worst := 0.0
		auto_used: l.LstsqMethod
		for method in ([]l.LstsqMethod{.Auto, .QR, .Cholesky, .SVD}) {
			r := l.lstsq(&X, y, method)
			defer l.lstsq_result_free(&r, allocator)
			worst = max(worst, vrel(r.beta, beta))
			if method == .Auto {auto_used = r.method_used}
		}
		expect(&passed, &total, worst < 1e-10,
			fmt.tprintf("lstsq 120x40 noise-free: all methods recover beta (worst rel err %.1e, Auto used %v)", worst, auto_used))
	}

	// 6. lstsq Cholesky on rank-deficient X: no panic, minimum-norm solution
	{
		X := l.matrix_new(f64, 200, 4)
		defer l.matrix_free(&X)
		y := make([]f64, 200)
		defer delete(y)
		for i in 0 ..< 200 {
			a, b, c := rand.norm_float64(), rand.norm_float64(), rand.norm_float64()
			X.data[i * 4], X.data[i * 4 + 1], X.data[i * 4 + 2], X.data[i * 4 + 3] = a, b, c, a + b
			y[i] = 2 * a + b - c
		}
		r := l.lstsq(&X, y, .Cholesky)
		defer l.lstsq_result_free(&r, allocator)
		min_norm := []f64{1, 0, -1, 1} // argmin ||beta|| over all exact fits
		expect(&passed, &total, r.success && r.rank == 3 && vrel(r.beta, min_norm) < 1e-10,
			fmt.tprintf("lstsq Cholesky, rank-deficient: rank %d via %v, beta %v (minimum norm 1, 0, -1, 1)", r.rank, r.method_used, r.beta))
	}

	fmt.printf("Linear algebra summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
