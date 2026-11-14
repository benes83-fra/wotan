package tests

import w "../wotan/core"
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
@(private = "file")
LM_EIGVALSH := [3]f64{1.854897308799577, 3.476023602918134, 6.669079088282288}

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

	{ 	// matrix_inverse: ok return; singular -> ok = false (no panic), no leak
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		inv, ok := l.matrix_inverse(&A, a)
		good := ok && _lm_close(inv.data, LM_INV[:], 1e-10)
		l.matrix_free(&inv)
		l.matrix_free(&A)
		S := l.matrix_from_flat([]f64{1, 2, 2, 4}, 2, 2, a)
		sinv, sok := l.matrix_inverse(&S, a)
		good = good && !sok && sinv.rows == 0 && sinv.cols == 0
		l.matrix_free(&sinv)
		l.matrix_free(&S)
		report(
			"matrix_inverse: numpy inverse, singular -> ok=false, no leak",
			_lm_end(&track, "matrix_inverse") && good,
			&passed,
			&total,
		)
	}
	{ 	// apply_pivots_vec: result from the given allocator (default context.allocator)
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		piv := []int{2, 0, 1}
		b := []f64{10, 20, 30}
		x := l.apply_pivots_vec(piv, b, a)
		good := _lm_close(x, []f64{30, 10, 20}, 0) && len(track.allocation_map) == 1
		delete(x, a)
		{
			context.allocator = a
			y := l.apply_pivots_vec(piv, b)
			good = good && len(track.allocation_map) == 1
			delete(y)
		}
		report(
			"apply_pivots_vec: caller's allocator, freed with delete",
			_lm_end(&track, "apply_pivots_vec") && good,
			&passed,
			&total,
		)
	}
	{ 	// jacobi_eigen_symmetric works on a copy: the input is unchanged
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		vals, V := l.jacobi_eigen_symmetric(&A, a)
		sorted := [3]f64{vals[0], vals[1], vals[2]}
		for i in 0 ..< 3 {for j in i + 1 ..< 3 {if sorted[j] < sorted[i] {sorted[i], sorted[j] = sorted[j], sorted[i]}}}
		good := _lm_close(A.data, LM_A[:], 0) && _lm_close(sorted[:], LM_EIGVALSH[:], 1e-10)
		delete(vals, a)
		l.matrix_free(&V)
		l.matrix_free(&A)
		report(
			"jacobi_eigen_symmetric: input untouched, numpy eigvalsh",
			_lm_end(&track, "jacobi_eigen_symmetric") && good,
			&passed,
			&total,
		)
	}
	{ 	// eigh: symmetry checked in every build, tolerance relative to the scale
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		A := l.matrix_from_flat(LM_A[:], 3, 3, a)
		W, V, ok := l.eigh(&A, .Ascending, a)
		good := ok && _lm_close(W, LM_EIGVALSH[:], 1e-10) && _lm_close(A.data, LM_A[:], 0)
		delete(W, a)
		l.matrix_free(&V)
		// clearly asymmetric: ok = false, empty outputs
		A.data[1] += 0.5
		W, V, ok = l.eigh(&A, .Ascending, a)
		good = good && !ok && len(W) == 0 && V.rows == 0 && V.cols == 0
		delete(W, a)
		l.matrix_free(&V)
		// scale 1e10, asymmetry 1e-2 (1.6e-13 relative): accepted, numpy eigvalsh of (A + Aᵀ)/2
		for v, i in LM_A {A.data[i] = v * 1e10}
		A.data[1] += 1e-2
		W, V, ok = l.eigh(&A, .Ascending, a)
		big := []f64{1.854897308799173e+10, 3.476023602918429e+10, 6.669079088282396e+10}
		good = good && ok && _lm_close(W, big, 1e-3)
		delete(W, a)
		l.matrix_free(&V)
		// scale 1e-6, asymmetry 1e-15 (1.6e-10 relative): rejected
		for v, i in LM_A {A.data[i] = v * 1e-6}
		A.data[1] += 1e-15
		W, V, ok = l.eigh(&A, .Ascending, a)
		good = good && !ok
		delete(W, a)
		l.matrix_free(&V)
		l.matrix_free(&A)
		report(
			"eigh: asymmetric -> ok=false (relative tolerance), no leak",
			_lm_end(&track, "eigh") && good,
			&passed,
			&total,
		)
	}
	{ 	// pinv: numpy.linalg.pinv on a rank-2 square and a wide matrix
		track: mem.Tracking_Allocator
		a := _lm_begin(&track)
		R := l.matrix_from_flat([]f64{1, 2, 3, 2, 4, 6, 1, 0, 1}, 3, 3, a)
		P := l.pinv(&R, allocator = a)
		ref_r := []f64 {
			-0.033333333333334,
			-0.066666666666667,
			0.833333333333334,
			0.066666666666667,
			0.133333333333333,
			-0.666666666666667,
			0.033333333333333,
			0.066666666666667,
			0.166666666666667,
		}
		good := _lm_close(P.data, ref_r, 1e-10)
		Wd := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 6}, 2, 3, a)
		Pw := l.pinv(&Wd, allocator = a)
		ref_w := []f64 {
			-0.944444444444445,
			0.444444444444445,
			-0.111111111111112,
			0.111111111111111,
			0.722222222222223,
			-0.222222222222223,
		}
		good = good && Pw.rows == 3 && Pw.cols == 2 && _lm_close(Pw.data, ref_w, 1e-10)
		l.matrix_free(&P)
		l.matrix_free(&R)
		l.matrix_free(&Pw)
		l.matrix_free(&Wd)
		report(
			"pinv: numpy pinv (rank-deficient 3x3, wide 2x3), no leak",
			_lm_end(&track, "pinv") && good,
			&passed,
			&total,
		)
	}
	{ 	// matrix_from_df / vector_from_df: null -> NaN (pandas to_numpy), views read
		x := w.column_new("x", .Float, 4)
		w.append_float(&x, 1.5)
		w.append_null(&x)
		w.append_float(&x, 2.5)
		w.append_float(&x, 3.5)
		k := w.column_new("k", .Int, 4)
		w.append_int(&k, 7)
		w.append_int(&k, 8)
		w.append_null(&k)
		w.append_int(&k, 9)
		df := w.df_from(x, k)
		defer w.destroy_dataframe(&df)
		M := l.matrix_from_df(&df, []string{"x", "k"}, allocator)
		v := l.vector_from_df(&df, "k", allocator)
		good :=
			M.data[0] == 1.5 &&
			M.data[1] == 7 &&
			math.is_nan(M.data[2]) &&
			M.data[3] == 8 &&
			M.data[4] == 2.5 &&
			math.is_nan(M.data[5]) &&
			M.data[6] == 3.5 &&
			M.data[7] == 9 &&
			v[0] == 7 &&
			v[1] == 8 &&
			math.is_nan(v[2]) &&
			v[3] == 9
		view := w.df_slice(&df, 1, 4) // rows 1..3 as views
		defer w.destroy_dataframe(&view)
		Mv := l.matrix_from_df(&view, []string{"x", "k"}, allocator)
		good =
			good &&
			Mv.rows == 3 &&
			math.is_nan(Mv.data[0]) &&
			Mv.data[1] == 8 &&
			Mv.data[2] == 2.5 &&
			math.is_nan(Mv.data[3]) &&
			Mv.data[4] == 3.5 &&
			Mv.data[5] == 9
		report("matrix_from_df / vector_from_df: null -> NaN, views", good, &passed, &total)
	}

	fmt.printf("  linalg memory tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
