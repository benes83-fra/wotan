package wotan_linalg

import "core:math"
import "core:mem"

SVDMethod :: enum {
	Jacobi,
	GolubReinsch,
}
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

// Dispatcher
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

// ------------------------------------------------------------
// Symmetric Jacobi eigen for dense n×n (A is modified in-place)
// ------------------------------------------------------------
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
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			eigenvectors.data[i * n + j] = (i == j) ? 1.0 : 0.0
		}
	}

	max_iter := 100
	eps := 1e-12

	for iter := 0; iter < max_iter; iter += 1 {
		// find largest off-diagonal |A[p,q]|
		p := 0
		q := 1
		max_val := 0.0
		for i := 0; i < n; i += 1 {
			for j := i + 1; j < n; j += 1 {
				aij := math.abs(A.data[i * A.cols + j])
				if aij > max_val {
					max_val = aij
					p = i
					q = j
				}
			}
		}

		if max_val < eps {
			break
		}

		app := A.data[p * A.cols + p]
		aqq := A.data[q * A.cols + q]
		apq := A.data[p * A.cols + q]

		tau := (aqq - app) / (2.0 * apq)
		t: f64
		if tau >= 0 {
			t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
		} else {
			t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
		}
		c := 1.0 / math.sqrt(1.0 + t * t)
		s := c * t

		// rotate A
		for k := 0; k < n; k += 1 {
			if k != p && k != q {
				aik := A.data[p * A.cols + k]
				akq := A.data[q * A.cols + k]
				A.data[p * A.cols + k] = c * aik - s * akq
				A.data[q * A.cols + k] = s * aik + c * akq
				A.data[k * A.cols + p] = A.data[p * A.cols + k]
				A.data[k * A.cols + q] = A.data[q * A.cols + k]
			}
		}

		app_new := c * c * app - 2.0 * c * s * apq + s * s * aqq
		aqq_new := s * s * app + 2.0 * c * s * apq + c * c * aqq
		A.data[p * A.cols + p] = app_new
		A.data[q * A.cols + q] = aqq_new
		A.data[p * A.cols + q] = 0.0
		A.data[q * A.cols + p] = 0.0

		// rotate eigenvectors
		for k := 0; k < n; k += 1 {
			vip := eigenvectors.data[k * eigenvectors.cols + p]
			viq := eigenvectors.data[k * eigenvectors.cols + q]
			eigenvectors.data[k * eigenvectors.cols + p] = c * vip - s * viq
			eigenvectors.data[k * eigenvectors.cols + q] = s * vip + c * viq
		}
	}

	eigenvalues = make([]f64, n, allocator)
	for i := 0; i < n; i += 1 {
		eigenvalues[i] = A.data[i * A.cols + i]
	}

	return
}

// ------------------------------------------------------------
// "Golub–Reinsch style" SVD via AᵀA eigen-decomposition
// A: m×n, U: m×n, S: n, V: n×n
// ------------------------------------------------------------
svd_golub_reinsch :: proc(
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

	// 1) Build AtA = Aᵀ A (n×n)
	AtA := xtx(A, allocator) // you already have xtx(X: ^Matrix(f64)) -> Matrix(f64)

	// 2) Eigen-decompose AtA = V Λ Vᵀ
	evals, evecs := jacobi_eigen_symmetric(&AtA, allocator)

	// 3) Singular values = sqrt(max(λ, 0)), sort descending
	// 3) Singular values = sqrt(max(λ, 0)), sort descending
	S = make([]f64, n, allocator)
	eps_rel := 1e-10

	// Find a global scale for AtA
	max_abs_eval := 0.0
	for i := 0; i < n; i += 1 {
		ev := math.abs(evals[i])
		if ev > max_abs_eval {
			max_abs_eval = ev
		}
	}
	if max_abs_eval == 0.0 {
		max_abs_eval = 1.0 // avoid zero scale
	}

	for i := 0; i < n; i += 1 {
		lam := evals[i]

		if lam < 0 {
			// Treat small negative eigenvalues as zero
			if -lam <= eps_rel * max_abs_eval {
				lam = 0.0
			} else {
				panic("svd_golub_reinsch: negative eigenvalue in AtA")
			}
		}

		S[i] = math.sqrt(lam)
	}


	// V = evecs (n×n)
	V = matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			V.data[i * V.cols + j] = evecs.data[i * evecs.cols + j]
		}
	}

	// 4) Sort S descending and permute V accordingly
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
		S[i], S[max_i] = S[max_i], S[i]
		for r := 0; r < n; r += 1 {
			V.data[r * V.cols + i], V.data[r * V.cols + max_i] =
				V.data[r * V.cols + max_i], V.data[r * V.cols + i]
		}
	}

	// 5) Build U: columns u_j = (1/σ_j) * A * v_j
	U = matrix_new(f64, m, n, allocator)
	temp_v := make([]f64, n, context.temp_allocator)

	for j := 0; j < n; j += 1 {
		sigma := S[j]
		if sigma == 0 {
			// Generate orthonormal filler column
			// Start with a basis vector
			for i := 0; i < m; i += 1 {
				U.data[i * U.cols + j] = 0.0
			}
			U.data[j % m * U.cols + j] = 1.0
			continue
		}


		// v_j column
		for k := 0; k < n; k += 1 {
			temp_v[k] = V.data[k * V.cols + j]
		}

		// w = A * v_j  (m×n * n×1 = m×1)
		w := matvec_dyn_simd(A, temp_v, allocator)

		inv_sigma := 1.0 / sigma
		for i := 0; i < m; i += 1 {
			U.data[i * U.cols + j] = w[i] * inv_sigma
		}
	}

	return
}
// Thin SVD wrapper around svd_golub_reinsch
// A: m×n, returns U_t: m×r, S_t: r, V_t: n×r, where r = numerical rank
svd_thin :: proc(
	A: ^Matrix(f64),
	method: SVDMethod = .GolubReinsch,
	allocator: mem.Allocator = context.allocator,
) -> (
	U_t: Matrix(f64),
	S_t: []f64,
	V_t: Matrix(f64),
) {
	// Use your existing dispatcher
	U_full, S_full, V_full := svd(A, method, allocator)
	m := U_full.rows
	n := U_full.cols // for Golub–Reinsch: m×n

	// --- 1) Determine numerical rank r
	// Simple heuristic: tol = max(m,n) * eps * S_max
	S_max := 0.0
	for i := 0; i < n; i += 1 {
		if S_full[i] > S_max {
			S_max = S_full[i]
		}
	}
	eps := 1e-12
	tol := f64(max(m, n)) * eps * S_max

	r := 0
	for i := 0; i < n; i += 1 {
		if S_full[i] > tol {
			r += 1
		} else {
			break // S is sorted descending
		}
	}

	if r == 0 {
		// Completely rank-deficient: return zeros of minimal shape
		U_t = matrix_new(f64, m, 0, allocator)
		S_t = make([]f64, 0, allocator)
		V_t = matrix_new(f64, n, 0, allocator)
		return
	}

	// --- 2) Allocate thin factors
	U_t = matrix_new(f64, m, r, allocator)
	V_t = matrix_new(f64, n, r, allocator)
	S_t = make([]f64, r, allocator)

	// Copy leading r singular values
	for i := 0; i < r; i += 1 {
		S_t[i] = S_full[i]
	}

	// Copy first r columns of U_full into U_t
	for i := 0; i < m; i += 1 {
		for j := 0; j < r; j += 1 {
			U_t.data[i * U_t.cols + j] = U_full.data[i * U_full.cols + j]
		}
	}

	// Copy first r columns of V_full into V_t
	for i := 0; i < n; i += 1 {
		for j := 0; j < r; j += 1 {
			V_t.data[i * V_t.cols + j] = V_full.data[i * V_full.cols + j]
		}
	}

	return
}
