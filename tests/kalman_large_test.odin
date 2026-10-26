package tests

import "../wotan/analytics"
import "core:fmt"
import "core:math"
import "core:mem"

// Large-state (N > 8) Kalman predict step on flat row-major arrays.
//
// The ARIMA/SARIMA forecasts run x = F x, P = F P Fᵀ + Q through
// analytics._kalman_predict_flat for state sizes above 8 (seasonal models).
// It is checked against a naive triple loop, with a non-symmetric F so a
// missing transpose (F P F instead of F P Fᵀ) is caught.

kl_passed, kl_total: int

kalman_large_n_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Large-State Kalman Predict (flat arrays) ===")
	context.allocator = allocator
	kl_passed, kl_total = 0, 0

	for N in ([]int{3, 10, 14}) {
		F := make([]f64, N * N, allocator)
		Q := make([]f64, N * N, allocator)
		P := make([]f64, N * N, allocator)
		x := make([]f64, N, allocator)
		x_ref := make([]f64, N, allocator)
		P_ref := make([]f64, N * N, allocator)
		FP := make([]f64, N * N, allocator)
		defer {
			delete(F, allocator); delete(Q, allocator); delete(P, allocator)
			delete(x, allocator); delete(x_ref, allocator)
			delete(P_ref, allocator); delete(FP, allocator)
		}

		gcl_fill(F, u64(300 + N)) // non-symmetric
		gcl_fill(x, u64(400 + N))
		// Symmetric positive semi-definite P and Q: A Aᵀ
		A := make([]f64, N * N, allocator)
		defer delete(A, allocator)
		gcl_fill(A, u64(500 + N))
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {s += A[i * N + k] * A[j * N + k]}
				P[i * N + j] = s
				Q[i * N + j] = 0.1 * s
			}
		}

		// Reference: naive loops
		for i in 0 ..< N {
			s := 0.0
			for j in 0 ..< N {s += F[i * N + j] * x[j]}
			x_ref[i] = s
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {s += F[i * N + k] * P[k * N + j]}
				FP[i * N + j] = s
			}
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := Q[i * N + j]
				for k in 0 ..< N {s += FP[i * N + k] * F[j * N + k]} // F[j, k] = Fᵀ[k, j]
				P_ref[i * N + j] = s
			}
		}

		analytics._kalman_predict_flat(F, Q, x, P, N)

		err := 0.0
		for i in 0 ..< N {err = max(err, abs(x[i] - x_ref[i]))}
		for i in 0 ..< N * N {err = max(err, abs(P[i] - P_ref[i]))}
		asym := 0.0 // F P Fᵀ + Q must stay symmetric
		for i in 0 ..< N {for j in 0 ..< N {asym = max(asym, abs(P[i * N + j] - P[j * N + i]))}}

		kl_total += 1
		if err < 1e-10 && asym < 1e-10 {
			kl_passed += 1
			fmt.printf("OK   _kalman_predict_flat N=%d max_err=%.2e asym=%.2e\n", N, err, asym)
		} else {
			fmt.printf("FAIL _kalman_predict_flat N=%d max_err=%.2e asym=%.2e\n", N, err, asym)
		}
	}

	// Flat-array filter procs used by ARIMA/SARIMA (kalman_loglik_scalar, kalman_filter_residuals,
	// kalman_filter_last_scalar), checked against numpy for N = 3 (small) and N = 10, 14 (> 8).
	// Convention: x0/P0 is the state BEFORE the first observation (predict first, then update);
	// the log-likelihood sums all T innovations (exact Gaussian likelihood, no burn-in).
	// Reference per N: {loglik, v[T-1], S[T-1], sum(v), sum(S), sum(xT), trace(PT)}.
	flat_ref := [3][7]f64 {
		{-30.94613812211873, 0.7267334240223875, 0.58662092575814995, 5.4326099349481698, 17.94627660248587, 0.22516357805631121, 0.39470572211219979},
		{-41.411454075020615, 0.62632030349703238, 1.9799598036302695, -4.7191207204110812, 60.253531744925588, 0.49519690225036511, 2.3734706558279353},
		{-44.83581304072319, 0.51769421867681442, 2.768826603735012, 2.6311420170089961, 84.706855615373527, 0.20125946299226577, 6.1353884332573259},
	}
	for N, ni in ([]int{3, 10, 14}) {
		T :: 30
		F := make([]f64, N * N, allocator)
		A := make([]f64, N * N, allocator)
		P0 := make([]f64, N * N, allocator)
		Q := make([]f64, N * N, allocator)
		H := make([]f64, N, allocator)
		x0 := make([]f64, N, allocator)
		y := make([]f64, T, allocator)
		defer {
			delete(F, allocator); delete(A, allocator); delete(P0, allocator); delete(Q, allocator)
			delete(H, allocator); delete(x0, allocator); delete(y, allocator)
		}
		gcl_fill(F, u64(700 + N))
		for &v in F {v *= 0.5 / math.sqrt(f64(N))} // spectral radius < 0.4
		gcl_fill(A, u64(800 + N))
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {s += A[i * N + k] * A[j * N + k]}
				P0[i * N + j] = s + (1.0 if i == j else 0.0)
				Q[i * N + j] = 0.1 * s
			}
		}
		gcl_fill(H, u64(900 + N))
		gcl_fill(x0, u64(1000 + N))
		gcl_fill(y, u64(1100 + N))
		R := []f64{0.5}

		ll := analytics.kalman_loglik_scalar(y, F, Q, P0, H, R, x0, N)
		v, S := analytics.kalman_filter_residuals(y, F, Q, P0, H, R, x0, N, allocator)
		xT, PT := analytics.kalman_filter_last_scalar(y, F, Q, P0, H, R, x0, N, allocator)
		defer {delete(v, allocator); delete(S, allocator); delete(xT, allocator); delete(PT, allocator)}

		sv, sS, sx, trP, ll_from_res := 0.0, 0.0, 0.0, 0.0, 0.0
		for t in 0 ..< T {
			sv += v[t]; sS += S[t]
			ll_from_res += -0.5 * (math.ln_f64(2.0 * math.PI) + math.ln_f64(S[t]) + v[t] * v[t] / S[t])
		}
		for i in 0 ..< N {sx += xT[i]; trP += PT[i * N + i]}
		got := [7]f64{ll, v[T - 1], S[T - 1], sv, sS, sx, trP}
		want := flat_ref[ni]
		err := 0.0
		for i in 0 ..< 7 {err = max(err, abs(got[i] - want[i]) / max(1.0, abs(want[i])))}
		err = max(err, abs(ll_from_res - ll) / max(1.0, abs(ll)))

		kl_total += 1
		if err < 1e-10 {
			kl_passed += 1
			fmt.printf("OK   flat filter procs vs numpy N=%d (loglik %.10f) max_rel_err=%.2e\n", N, ll, err)
		} else {
			fmt.printf("FAIL flat filter procs vs numpy N=%d (loglik %.10f) max_rel_err=%.2e\n", N, ll, err)
		}
	}

	fmt.printf("Large-state Kalman summary: %d/%d passed\n", kl_passed, kl_total)
	record_failures(kl_total - kl_passed)
}
