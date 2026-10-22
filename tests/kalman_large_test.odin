package tests

import "../wotan/analytics"
import "core:fmt"
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

	fmt.printf("Large-state Kalman summary: %d/%d passed\n", kl_passed, kl_total)
	record_failures(kl_total - kl_passed)
}
