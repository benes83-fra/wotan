package tests

import l "../wotan/linalg"
import "core:fmt"
import "core:mem"

// Checks the accumulating matmul kernels (out += A@B, A@B^T, A^T@B) against a
// naive triple loop. Sizes are deliberately not multiples of the SIMD width or
// tile size, and `out` starts non-zero to verify accumulation.

mk_naive_acc :: proc(a_t, b_t: bool, A, B, expected: ^l.Matrix(f64), M, N, K: int) {
	for i in 0 ..< M {
		for j in 0 ..< N {
			sum := 0.0
			for k in 0 ..< K {
				av := a_t ? A.data[k * A.cols + i] : A.data[i * A.cols + k]
				bv := b_t ? B.data[j * B.cols + k] : B.data[k * B.cols + j]
				sum += av * bv
			}
			expected.data[i * N + j] += sum
		}
	}
}

matmul_kernels_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Accumulating MatMul Kernel Tests ===")
	// The last two shapes exceed MATMUL_PARALLEL_MIN_FLOPS, so they run multi-threaded
	shapes := [][3]int{{1, 1, 1}, {3, 5, 7}, {17, 9, 13}, {64, 33, 65}, {5, 130, 3}, {300, 200, 250}, {7, 600, 900}}
	kinds := []string{"nn  A@B", "nt  A@B^T", "tn  A^T@B"}
	passed, total := 0, 0

	for sh in shapes {
		M, N, K := sh[0], sh[1], sh[2]
		for kind, ki in kinds {
			a_t := ki == 2
			b_t := ki == 1
			A := a_t ? l.matrix_new(f64, K, M, allocator) : l.matrix_new(f64, M, K, allocator)
			B := b_t ? l.matrix_new(f64, N, K, allocator) : l.matrix_new(f64, K, N, allocator)
			out := l.matrix_new(f64, M, N, allocator)
			expected := l.matrix_new(f64, M, N, allocator)
			defer {
				l.matrix_free(&A); l.matrix_free(&B)
				l.matrix_free(&out); l.matrix_free(&expected)
			}
			gcl_fill(A.data, u64(M * 31 + ki))
			gcl_fill(B.data, u64(N * 17 + ki))
			gcl_fill(out.data, u64(K * 7 + ki))
			copy(expected.data, out.data)

			switch ki {
			case 0: l.matmul_nn_acc_simd(&A, &B, &out)
			case 1: l.matmul_nt_acc_simd(&A, &B, &out)
			case 2: l.matmul_tn_acc_simd(&A, &B, &out)
			}
			mk_naive_acc(a_t, b_t, &A, &B, &expected, M, N, K)

			max_err := 0.0
			for i in 0 ..< len(out.data) {
				max_err = max(max_err, abs(out.data[i] - expected.data[i]))
			}
			total += 1
			if max_err < 1e-10 {
				passed += 1
			} else {
				fmt.printf("FAIL %s M=%d N=%d K=%d max_abs_err=%.2e\n", kind, M, N, K, max_err)
			}
		}
	}
	// Threading must not change results: compare single-threaded and threaded
	// runs bit for bit (each output row is computed by one thread, same order).
	saved_threads := l.matmul_max_threads
	defer l.matmul_max_threads = saved_threads
	for kind, ki in kinds {
		M, N, K := 300, 200, 250
		a_t := ki == 2
		b_t := ki == 1
		A := a_t ? l.matrix_new(f64, K, M, allocator) : l.matrix_new(f64, M, K, allocator)
		B := b_t ? l.matrix_new(f64, N, K, allocator) : l.matrix_new(f64, K, N, allocator)
		single := l.matrix_new(f64, M, N, allocator)
		threaded := l.matrix_new(f64, M, N, allocator)
		defer {
			l.matrix_free(&A); l.matrix_free(&B)
			l.matrix_free(&single); l.matrix_free(&threaded)
		}
		gcl_fill(A.data, u64(900 + ki))
		gcl_fill(B.data, u64(950 + ki))
		for run in 0 ..< 2 {
			l.matmul_max_threads = 1 if run == 0 else 0
			dst := &single if run == 0 else &threaded
			switch ki {
			case 0: l.matmul_nn_acc_simd(&A, &B, dst)
			case 1: l.matmul_nt_acc_simd(&A, &B, dst)
			case 2: l.matmul_tn_acc_simd(&A, &B, dst)
			}
		}
		identical := true
		for i in 0 ..< len(single.data) {identical &&= single.data[i] == threaded.data[i]}
		total += 1
		if identical {
			passed += 1
		} else {
			fmt.printf("FAIL %s threaded result differs from single-threaded\n", kind)
		}
	}

	fmt.printf("MatMul kernel summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
