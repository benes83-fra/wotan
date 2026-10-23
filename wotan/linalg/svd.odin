package wotan_linalg

import "core:math"
import "core:mem"

SVDMethod :: enum {
	Jacobi,
	GolubReinsch,
}

// ============================================================================
// SVD (Jacobi) - Optimized: dot_simd + workspace reuse (SAFE)
// ============================================================================
svd_jacobi :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.temp_allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	m := A.rows
	n := A.cols
	if m == 0 || n == 0 {return}

	// Initialize U = A, V = I
	U = matrix_new(f64, m, n, allocator)
	copy(U.data, A.data)

	V = matrix_new(f64, n, n, allocator)
	for i in 0 ..< n {V.data[i * n + i] = 1.0}

	// Pre-allocate workspace ONCE (reused every iteration)
	col_p := make([]f64, max(m, n), context.temp_allocator)
	col_q := make([]f64, max(m, n), context.temp_allocator)

	max_iter := 100
	eps := 1e-12

	for iter in 0 ..< max_iter {
		changed := false

		for p in 0 ..< n {
			for q in p + 1 ..< n {
				// Gather columns p and q of U into temp buffers
				for i in 0 ..< m {
					col_p[i] = U.data[i * U.cols + p]
					col_q[i] = U.data[i * U.cols + q]
				}

				// Compute alpha, beta, gamma using SIMD dot (SAFE: read-only)
				alpha := dot_simd(col_p[0:m], col_p[0:m])
				beta := dot_simd(col_q[0:m], col_q[0:m])
				gamma := dot_simd(col_p[0:m], col_q[0:m])

				if math.abs(gamma) <= eps * math.sqrt(alpha * beta) {
					continue
				}

				changed = true

				// Jacobi rotation parameters (idiomatic Odin)
				tau := (beta - alpha) / (2.0 * gamma)
				t: f64
				if tau >= 0.0 {
					t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
				} else {
					t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
				}
				c := 1.0 / math.sqrt(1.0 + t * t)
				s := c * t

				// Rotate columns p,q of U (m rows) - SCALAR for now (safe)
				rotate_pair_simd(c, s, col_p, col_q, m) // Pass full arrays + explicit len
				for i in 0 ..< m {
					U.data[i * U.cols + p] = col_p[i]
					U.data[i * U.cols + q] = col_q[i]
				}

				// Rotate columns p,q of V (n rows) - SCALAR for now (safe)
				for i in 0 ..< n {
					vp := V.data[i * V.cols + p]
					vq := V.data[i * V.cols + q]
					V.data[i * V.cols + p] = c * vp - s * vq
					V.data[i * V.cols + q] = s * vp + c * vq
				}
			}
		}

		if !changed {break}
	}

	// Compute singular values = column norms of U, normalize columns
	S = make([]f64, n, allocator)
	for j in 0 ..< n {
		norm2 := 0.0
		for i in 0 ..< m {
			uij := U.data[i * U.cols + j]
			norm2 += uij * uij
		}
		sigma := math.sqrt(norm2)
		S[j] = sigma
		if sigma > 1e-15 {
			inv := 1.0 / sigma
			for i in 0 ..< m {
				U.data[i * U.cols + j] *= inv
			}
		}
	}

	// Sort descending (selection sort)
	for i in 0 ..< n {
		max_i := i
		for j in i + 1 ..< n {
			if S[j] > S[max_i] {max_i = j}
		}
		if max_i == i {continue}
		S[i], S[max_i] = S[max_i], S[i]
		// Swap columns in U
		for r in 0 ..< m {
			U.data[r * U.cols + i], U.data[r * U.cols + max_i] =
				U.data[r * U.cols + max_i], U.data[r * U.cols + i]
		}
		// Swap columns in V
		for r in 0 ..< n {
			V.data[r * V.cols + i], V.data[r * V.cols + max_i] =
				V.data[r * V.cols + max_i], V.data[r * V.cols + i]
		}
	}

	// Columns of U for (numerically) zero singular values carry no direction:
	// complete them to an orthonormal set (Gram-Schmidt on unit vectors), so the
	// first min(m, n) columns of U stay orthonormal for rank-deficient A
	tol := f64(max(m, n)) * 2.2e-16 * (S[0] if n > 0 else 0)
	cand := make([]f64, m, context.temp_allocator)
	defer delete(cand, context.temp_allocator)
	for j in 0 ..< min(m, n) {
		if S[j] > tol && S[j] > 1e-15 {continue}
		S[j] = 0
		for e in 0 ..< m {
			for i in 0 ..< m {cand[i] = 1.0 if i == e else 0.0}
			for _ in 0 ..< 2 { 	// re-orthogonalise once for stability
				for k in 0 ..< j {
					d := 0.0
					for i in 0 ..< m {d += U.data[i * U.cols + k] * cand[i]}
					for i in 0 ..< m {cand[i] -= d * U.data[i * U.cols + k]}
				}
			}
			nrm := math.sqrt(dot_simd(cand, cand))
			if nrm > 0.5 {
				for i in 0 ..< m {U.data[i * U.cols + j] = cand[i] / nrm}
				break
			}
		}
	}

	return
}

// ============================================================================
// Symmetric Jacobi eigen - original working version + idiomatic Odin
// ============================================================================
// ============================================================================
// Symmetric Jacobi eigen - SIMD-optimized version
// ============================================================================
jacobi_eigen_symmetric :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	eigenvalues: []f64,
	eigenvectors: Matrix(f64),
) {
	n := A.rows
	if n == 0 || A.cols != n {
		panic("jacobi_eigen_symmetric: matrix must be square")
	}

	// V = I
	eigenvectors = matrix_new(f64, n, n, allocator)
	for i in 0 ..< n {eigenvectors.data[i * n + i] = 1.0}

	col_p := make([]f64, n, context.temp_allocator)
	col_q := make([]f64, n, context.temp_allocator)
	defer {
		delete(col_p, context.temp_allocator)
		delete(col_q, context.temp_allocator)
	}

	// Cyclic Jacobi: sweep all (p, q) pairs; an off-diagonal entry below rounding
	// level relative to its diagonal pair is set to zero (scale-free). Converges
	// quadratically; stops after a sweep without rotations.
	MAX_SWEEPS :: 60
	for _ in 0 ..< MAX_SWEEPS {
		rotations := 0
		for p in 0 ..< n - 1 {
			for q in p + 1 ..< n {
				apq := A.data[p * n + q]
				if apq == 0 {continue}
				app := A.data[p * n + p]
				aqq := A.data[q * n + q]
				// negligible next to the diagonal: drop it
				if abs(apq) <= 1e-17 * (abs(app) + abs(aqq)) || abs(apq) < 1e-300 {
					A.data[p * n + q] = 0
					A.data[q * n + p] = 0
					continue
				}
				rotations += 1

				tau := (aqq - app) / (2.0 * apq)
				t: f64
				if tau >= 0.0 {
					t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
				} else {
					t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
				}
				c := 1.0 / math.sqrt(1.0 + t * t)
				s := c * t

				// rows / columns p, q of A (symmetric)
				for k in 0 ..< n {
					if k == p || k == q {continue}
					up := A.data[p * n + k]
					uq := A.data[q * n + k]
					np := c * up - s * uq
					nq := s * up + c * uq
					A.data[p * n + k] = np
					A.data[k * n + p] = np
					A.data[q * n + k] = nq
					A.data[k * n + q] = nq
				}
				A.data[p * n + p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
				A.data[q * n + q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
				A.data[p * n + q] = 0.0
				A.data[q * n + p] = 0.0

				// eigenvector columns p, q
				for k in 0 ..< n {
					col_p[k] = eigenvectors.data[k * n + p]
					col_q[k] = eigenvectors.data[k * n + q]
				}
				rotate_pair_simd(c, s, col_p, col_q, n)
				for k in 0 ..< n {
					eigenvectors.data[k * n + p] = col_p[k]
					eigenvectors.data[k * n + q] = col_q[k]
				}
			}
		}
		if rotations == 0 {break}
	}

	eigenvalues = make([]f64, n, allocator)
	for i in 0 ..< n {eigenvalues[i] = A.data[i * n + i]}
	return
}

// ============================================================================
// Golub-Reinsch SVD - unchanged (already efficient)
// ============================================================================
// Kept for API compatibility. Formerly the SVD from eig(AᵀA), which squares the
// condition number (small singular values lose all accuracy); now the
// one-sided Jacobi SVD, which is accurate to machine precision.
svd_golub_reinsch :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	return svd_jacobi(A, allocator)
}

// ============================================================================
// Thin SVD wrapper (unchanged)
// ============================================================================
svd_thin :: proc(
	A: ^Matrix(f64),
	method: SVDMethod = .GolubReinsch,
	allocator: mem.Allocator = context.allocator,
) -> (
	U_t: Matrix(f64),
	S_t: []f64,
	V_t: Matrix(f64),
) {
	U_full, S_full, V_full := svd(A, method, allocator)
	m, n := U_full.rows, U_full.cols

	S_max := 0.0
	for i in 0 ..< n {if S_full[i] > S_max {S_max = S_full[i]}}
	tol := f64(max(m, n)) * 1e-12 * S_max
	r := 0
	for i in 0 ..< n {if S_full[i] > tol {r += 1} else {break}}

	if r == 0 {
		U_t = matrix_new(f64, m, 0, allocator)
		S_t = make([]f64, 0, allocator)
		V_t = matrix_new(f64, n, 0, allocator)
		return
	}

	U_t = matrix_new(f64, m, r, allocator)
	V_t = matrix_new(f64, n, r, allocator)
	S_t = make([]f64, r, allocator)

	for i in 0 ..< r {S_t[i] = S_full[i]}
	for i in 0 ..< m {
		for j in 0 ..< r {U_t.data[i * U_t.cols + j] = U_full.data[i * U_full.cols + j]}
	}
	for i in 0 ..< n {
		for j in 0 ..< r {V_t.data[i * V_t.cols + j] = V_full.data[i * V_full.cols + j]}
	}
	return
}

// ============================================================================
// Dispatcher (unchanged)
// ============================================================================
svd :: proc(
	A: ^Matrix(f64),
	method: SVDMethod = .Jacobi,
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	switch method {
	case .Jacobi:
		return svd_jacobi(A, allocator)
	case .GolubReinsch:
		return svd_golub_reinsch(A, allocator)
	}
	return svd_jacobi(A, allocator)
}
