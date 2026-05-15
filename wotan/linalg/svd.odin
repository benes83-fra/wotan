package wotan_linalg

import "core:math"
import "core:mem"

// SVD (Jacobi) for general m x n matrix A
// A ≈ U * diag(S) * Vᵀ
// U: m x n (columns = left singular vectors)
// S: length n (singular values, descending)
// V: n x n (columns = right singular vectors)
svd_jacobi :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	m := A.rows
	n := A.cols
	if m == 0 || n == 0 {
		return
	}

	// U starts as a copy of A (we'll orthogonalize its columns)
	U = matrix_new(f64, m, n, allocator)
	copy(U.data, A.data)

	// V starts as identity (n x n)
	V = matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			if i == j {
				V.data[i * n + j] = 1.0
			} else {
				V.data[i * n + j] = 0.0
			}
		}
	}

	max_iter := 50
	eps := 1e-12

	for iter := 0; iter < max_iter; iter += 1 {
		changed := false

		for p := 0; p < n; p += 1 {
			for q := p + 1; q < n; q += 1 {
				// Compute alpha = ||u_p||^2, beta = ||u_q||^2, gamma = <u_p, u_q>
				alpha := 0.0
				beta := 0.0
				gamma := 0.0

				for i := 0; i < m; i += 1 {
					up := U.data[i * U.cols + p]
					uq := U.data[i * U.cols + q]
					alpha += up * up
					beta += uq * uq
					gamma += up * uq
				}

				if math.abs(gamma) <= eps * math.sqrt(alpha * beta) {
					continue
				}

				changed = true

				// Jacobi rotation parameters
				tau := (beta - alpha) / (2.0 * gamma)
				t := 0.0
				if tau >= 0 {
					t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
				} else {
					t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
				}
				c := 1.0 / math.sqrt(1.0 + t * t)
				s := c * t

				// Rotate columns p and q of U
				for i := 0; i < m; i += 1 {
					up := U.data[i * U.cols + p]
					uq := U.data[i * U.cols + q]
					U.data[i * U.cols + p] = c * up - s * uq
					U.data[i * U.cols + q] = s * up + c * uq
				}

				// Rotate columns p and q of V
				for i := 0; i < n; i += 1 {
					vp := V.data[i * V.cols + p]
					vq := V.data[i * V.cols + q]
					V.data[i * V.cols + p] = c * vp - s * vq
					V.data[i * V.cols + q] = s * vp + c * vq
				}
			}
		}

		if !changed {
			break
		}
	}

	// Singular values = norms of columns of U
	S = make([]f64, n, allocator)
	for j := 0; j < n; j += 1 {
		norm2 := 0.0
		for i := 0; i < m; i += 1 {
			uij := U.data[i * U.cols + j]
			norm2 += uij * uij
		}
		sigma := math.sqrt(norm2)
		S[j] = sigma

		// Normalize column j of U
		if sigma > 0 {
			inv := 1.0 / sigma
			for i := 0; i < m; i += 1 {
				U.data[i * U.cols + j] *= inv
			}
		}
	}

	// Sort singular values descending, permute U and V accordingly
	// Simple selection sort (n is usually small-ish)
	for i := 0; i < n; i += 1 {
		max_i := i
		for j := i + 1; j < n; j += 1 {
			if S[j] > S[max_i] {
				max_i = j
			}
		}
		if max_i == i {
			continue
		}

		// swap S
		S[i], S[max_i] = S[max_i], S[i]

		// swap columns i and max_i in U
		for r := 0; r < m; r += 1 {
			U.data[r * U.cols + i], U.data[r * U.cols + max_i] =
				U.data[r * U.cols + max_i], U.data[r * U.cols + i]
		}

		// swap columns i and max_i in V
		for r := 0; r < n; r += 1 {
			V.data[r * V.cols + i], V.data[r * V.cols + max_i] =
				V.data[r * V.cols + max_i], V.data[r * V.cols + i]
		}
	}

	return
}
