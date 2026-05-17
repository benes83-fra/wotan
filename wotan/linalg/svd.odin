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
				for i in 0 ..< m {
					up := U.data[i * U.cols + p]
					uq := U.data[i * U.cols + q]
					U.data[i * U.cols + p] = c * up - s * uq
					U.data[i * U.cols + q] = s * up + c * uq
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

	return
}

// ============================================================================
// Symmetric Jacobi eigen - original working version + idiomatic Odin
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

	max_iter := 100
	eps := 1e-12

	for iter in 0 ..< max_iter {
		// Find largest off-diagonal |A[p,q]|
		p, q := 0, 1
		max_val := 0.0
		for i in 0 ..< n {
			for j in i + 1 ..< n {
				aij := math.abs(A.data[i * A.cols + j])
				if aij > max_val {
					max_val = aij
					p, q = i, j
				}
			}
		}
		if max_val < eps {break}

		app := A.data[p * A.cols + p]
		aqq := A.data[q * A.cols + q]
		apq := A.data[p * A.cols + q]

		// Jacobi rotation parameters (idiomatic Odin)
		tau := (aqq - app) / (2.0 * apq)
		t: f64
		if tau >= 0.0 {
			t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
		} else {
			t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
		}
		c := 1.0 / math.sqrt(1.0 + t * t)
		s := c * t

		// Rotate rows/columns p,q of A
		for k in 0 ..< n {
			if k != p && k != q {
				aik := A.data[p * A.cols + k]
				akq := A.data[q * A.cols + k]
				A.data[p * A.cols + k] = c * aik - s * akq
				A.data[q * A.cols + k] = s * aik + c * akq
				A.data[k * A.cols + p] = A.data[p * A.cols + k]
				A.data[k * A.cols + q] = A.data[q * A.cols + k]
			}
		}
		A.data[p * A.cols + p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
		A.data[q * A.cols + q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
		A.data[p * A.cols + q] = 0.0
		A.data[q * A.cols + p] = 0.0

		// Rotate eigenvectors columns p,q
		for k in 0 ..< n {
			vip := eigenvectors.data[k * eigenvectors.cols + p]
			viq := eigenvectors.data[k * eigenvectors.cols + q]
			eigenvectors.data[k * eigenvectors.cols + p] = c * vip - s * viq
			eigenvectors.data[k * eigenvectors.cols + q] = s * vip + c * viq
		}
	}

	eigenvalues = make([]f64, n, allocator)
	for i in 0 ..< n {eigenvalues[i] = A.data[i * A.cols + i]}
	return
}

// ============================================================================
// Golub-Reinsch SVD - unchanged (already efficient)
// ============================================================================
svd_golub_reinsch :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	m, n := A.rows, A.cols
	if m == 0 || n == 0 {return}

	AtA := xtx(A, allocator)
	evals, evecs := jacobi_eigen_symmetric(&AtA, allocator)

	S = make([]f64, n, allocator)
	max_abs := 0.0
	for i in 0 ..< n {
		ev := math.abs(evals[i])
		if ev > max_abs {max_abs = ev}
	}
	if max_abs == 0.0 {max_abs = 1.0}
	eps_rel := 1e-10

	for i in 0 ..< n {
		lam := evals[i]
		if lam < 0.0 {
			if -lam <= eps_rel * max_abs {
				lam = 0.0
			} else {
				panic("svd_golub_reinsch: negative eigenvalue")
			}
		}
		S[i] = math.sqrt(lam)
	}

	V = matrix_new(f64, n, n, allocator)
	copy(V.data, evecs.data)
	for i in 0 ..< n {
		max_i := i
		for j in i + 1 ..< n {
			if S[j] > S[max_i] {max_i = j}
		}
		if max_i == i {continue}
		S[i], S[max_i] = S[max_i], S[i]
		for r in 0 ..< n {
			V.data[r * V.cols + i], V.data[r * V.cols + max_i] =
				V.data[r * V.cols + max_i], V.data[r * V.cols + i]
		}
	}

	U = matrix_new(f64, m, n, allocator)
	temp_v := make([]f64, n, context.temp_allocator)

	for j in 0 ..< n {
		sigma := S[j]
		if sigma == 0.0 {
			for i in 0 ..< m {U.data[i * U.cols + j] = 0.0}
			U.data[j % m * U.cols + j] = 1.0
			continue
		}
		for k in 0 ..< n {temp_v[k] = V.data[k * V.cols + j]}
		w := matvec_dyn_simd(A, temp_v, allocator)
		inv := 1.0 / sigma
		for i in 0 ..< m {U.data[i * U.cols + j] = w[i] * inv}
	}
	return
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
