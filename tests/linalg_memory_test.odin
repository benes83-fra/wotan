package tests

import l "../wotan/linalg"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"

// Regression tests for wotan/linalg issues found while writing the linalg guide:
// leaks in lu_solve, mat_inverse_lu, svd_thin, cond2_svd / cond2_svd_thin /
// cond2_sym and the SVD least-squares path; lstsq on an underdetermined system
// (fewer rows than columns) read out of bounds; lstsq with .Auto / .QR on a
// rank-deficient X returned ~1e13 coefficients instead of falling back to the
// SVD as the Cholesky path does; svd_jacobi defaulted to the temp allocator.
// Each area runs under its own Tracking_Allocator wrapping the default heap
// allocator, frees exactly the documented outputs, and must end with no live
// allocations and no bad frees. Reference values: numpy 1.23.5.

@(private = "file")
_lm_begin :: proc(track: ^mem.Tracking_Allocator) -> mem.Allocator {
	mem.tracking_allocator_init(track, runtime.default_allocator())
	track.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
	return mem.tracking_allocator(track)
}

@(private = "file")
_lm_end :: proc(track: ^mem.Tracking_Allocator, name: string) -> bool {
	live, bad := len(track.allocation_map), len(track.bad_free_array)
	if live > 0 || bad > 0 {
		fmt.printf("     %s: %d live allocation(s), %d bad free(s)\n", name, live, bad)
		for _, e in track.allocation_map {fmt.printf("       allocated at %v\n", e.location)}
	}
	mem.tracking_allocator_destroy(track)
	return live == 0 && bad == 0
}

@(private = "file")
_lm_close :: proc(a, b: []f64, tol: f64) -> bool {
	if len(a) != len(b) do return false
	for i in 0 ..< len(a) {if math.abs(a[i] - b[i]) > tol do return false}
	return true
}

// SPD test matrix and its numpy results
@(private = "file")
LM_A := [9]f64{4, 1, 2, 1, 3, 0, 2, 0, 5}
@(private = "file")
LM_SOLVE := [3]f64{-0.302325581395, 0.767441860465, 0.720930232558}
@(private = "file")
LM_INV := [9]f64 {
	0.348837209302,
	-0.116279069767,
	-0.139534883721,
	-0.116279069767,
	0.372093023256,
	0.046511627907,
	-0.139534883721,
	0.046511627907,
	0.255813953488,
}
@(private = "file")
LM_SV := [3]f64{6.669079088282, 3.476023602918, 1.8548973088}
@(private = "file")
LM_COND :: 3.595389920856738

linalg_memory_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== linalg memory and lstsq regression tests ===")
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("  %-58s %s\n", name, "OK" if ok else "FAIL")
	}

	{ 	// lu_solve: the forward-substitution vector leaked
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		LU, piv, _, ok := l.lu_decompose(&A, a)
		x := l.lu_solve(&LU, piv, []f64{1, 2, 3}, a)
		good := ok && _lm_close(x, LM_SOLVE[:], 1e-10)
		delete(x, a)
		delete(piv, a)
		l.matrix_free(&LU)
		l.matrix_free(&A)
		report(
			"lu_solve: numpy solve, no leak",
			_lm_end(&track, "lu_solve") && good,
			&passed,
			&total,
		)
	}
	{ 	// mat_inverse_lu: LU factor and pivots leaked, also on the singular path
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		inv, ok := l.mat_inverse_lu(&A, a)
		good := ok && _lm_close(inv.data, LM_INV[:], 1e-10)
		l.matrix_free(&inv)
		l.matrix_free(&A)
		S := l.matrix_from_flat([]f64{1, 2, 2, 4}, 2, 2, a)
		sinv, sok := l.mat_inverse_lu(&S, a)
		good = good && !sok && sinv.rows == 0
		l.matrix_free(&sinv)
		l.matrix_free(&S)
		report(
			"mat_inverse_lu: numpy inverse, singular -> ok=false, no leak",
			_lm_end(&track, "mat_inverse_lu") && good,
			&passed,
			&total,
		)
	}
	{ 	// svd_thin: the full U, S, V leaked
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		U, S, V := l.svd_thin(&A, .GolubReinsch, a)
		good := _lm_close(S, LM_SV[:], 1e-10) && U.cols == 3 && V.cols == 3
		l.matrix_free(&U)
		delete(S, a)
		l.matrix_free(&V)
		l.matrix_free(&A)
		report(
			"svd_thin: numpy singular values, no leak",
			_lm_end(&track, "svd_thin") && good,
			&passed,
			&total,
		)
	}
	{ 	// cond2_*: the decompositions they computed leaked
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		c1 := l.cond2_svd(&A, a)
		c2 := l.cond2_svd_thin(&A, a)
		c3 := l.cond2_sym(&A, a)
		r1 := l.rcond2_svd(&A, a)
		r2 := l.rcond2_sym(&A, a)
		l.matrix_free(&A)
		good :=
			math.abs(c1 - LM_COND) < 1e-10 &&
			math.abs(c2 - LM_COND) < 1e-10 &&
			math.abs(c3 - LM_COND) < 1e-10 &&
			math.abs(r1 - 1 / LM_COND) < 1e-10 &&
			math.abs(r2 - 1 / LM_COND) < 1e-10
		report(
			"cond2_svd / _thin / _sym, rcond2_*: numpy cond, no leak",
			_lm_end(&track, "cond2") && good,
			&passed,
			&total,
		)
	}
	{ 	// lstsq .SVD: two work vectors leaked
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		X := l.matrix_from_flat([]f64{1, 1, 1, 2, 1, 3, 1, 4}, 4, 2, a)
		r := l.lstsq(&X, []f64{1, 2, 2, 4}, .SVD, a)
		good := r.method_used == .SVD && r.rank == 2 && _lm_close(r.beta, []f64{0, 0.9}, 1e-12)
		l.lstsq_result_free(&r, a)
		l.matrix_free(&X)
		report(
			"lstsq .SVD: numpy beta, no leak",
			_lm_end(&track, "lstsq svd") && good,
			&passed,
			&total,
		)
	}
	{ 	// lstsq on 2 x 3: used to index past Q (bounds-check panic)
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		X := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 6}, 2, 3, a)
		good := true
		for m in ([]l.LstsqMethod{.Auto, .QR, .SVD}) {
			r := l.lstsq(&X, []f64{1, 2}, m, a)
			good =
				good &&
				r.method_used == .SVD &&
				r.rank == 2 &&
				_lm_close(r.beta, []f64{-0.055555555556, 0.111111111111, 0.277777777778}, 1e-10) &&
				_lm_close(r.residuals, []f64{0, 0}, 1e-12)
			l.lstsq_result_free(&r, a)
		}
		l.matrix_free(&X)
		report(
			"lstsq 2x3 (Auto, QR, SVD): numpy minimum-norm solution",
			_lm_end(&track, "lstsq wide") && good,
			&passed,
			&total,
		)
	}
	{ 	// rank-deficient X: .Auto / .QR returned ~1e13 coefficients
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		X := l.matrix_from_flat([]f64{1, 2, 2, 4, 3, 6}, 3, 2, a)
		good := true
		for m in ([]l.LstsqMethod{.Auto, .QR, .Cholesky, .SVD}) {
			r := l.lstsq(&X, []f64{1, 2, 3.1}, m, a)
			good =
				good &&
				r.method_used == .SVD &&
				r.rank == 1 &&
				r.success &&
				_lm_close(r.beta, []f64{0.204285714286, 0.408571428571}, 1e-10)
			l.lstsq_result_free(&r, a)
		}
		l.matrix_free(&X)
		report(
			"lstsq rank 1 (all methods): SVD fallback, numpy beta",
			_lm_end(&track, "lstsq rank-deficient") && good,
			&passed,
			&total,
		)
	}
	{ 	// full-rank tall X through Auto still uses Householder QR
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		X := l.matrix_from_flat([]f64{1, 1, 1, 2, 1, 3, 1, 4}, 4, 2, a)
		r := l.lstsq(&X, []f64{1, 2, 2, 4}, .Auto, a)
		good := r.method_used == .QR && r.success && _lm_close(r.beta, []f64{0, 0.9}, 1e-12)
		l.lstsq_result_free(&r, a)
		l.matrix_free(&X)
		report(
			"lstsq full rank .Auto: still QR, numpy beta",
			_lm_end(&track, "lstsq qr") && good,
			&passed,
			&total,
		)
	}
	{ 	// svd_jacobi now defaults to context.allocator (was context.temp_allocator)
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		context.allocator = a
		A := l.matrix_from_flat(LM_A[:], 3, 3)
		U, S, V := l.svd_jacobi(&A)
		good := _lm_close(S, LM_SV[:], 1e-10) && U.allocator == a && V.allocator == a
		l.matrix_free(&U)
		delete(S)
		l.matrix_free(&V)
		l.matrix_free(&A)
		report(
			"svd_jacobi default allocator is context.allocator",
			_lm_end(&track, "svd_jacobi") && good,
			&passed,
			&total,
		)
	}
	{ 	// blocked Cholesky (frees its scratch now) still equals the unblocked one
		n :: 40
		A := l.matrix_new(f64, n, n, allocator)
		for i in 0 ..< n {
			for j in 0 ..< n {A.data[i * n + j] = 1.0 / f64(1 + abs(i - j))}
			A.data[i * n + i] += f64(n)
		}
		B := l.matrix_new(f64, n, n, allocator)
		copy(B.data, A.data)
		l.cholesky(&A, .Unblocked)
		l.cholesky(&B, .Blocked)
		report(
			"cholesky .Blocked == .Unblocked (40 x 40)",
			_lm_close(A.data, B.data, 1e-12),
			&passed,
			&total,
		)
	}

	fmt.printf("  linalg memory tests: %d/%d passed\n", passed, total)
}
