package wotan_linalg

import "core:math"
import "core:mem"


EighOrder :: enum {
	Ascending,
	Descending,
}

// Symmetric eigen-decomposition: A = V diag(W) Vᵀ (W sorted by `order`,
// eigenvectors in the columns of V). A must be square and non-empty (panics
// otherwise). A is checked for symmetry in every build: each pair must satisfy
// |a_ij - a_ji| <= EIGH_SYM_RTOL · max|a_kl| (largest finite entry); if not,
// ok = false and W / V are empty (W has length 0, V is 0 × 0; freeing them is
// still fine). Within the tolerance the decomposition is of (A + Aᵀ) / 2.
// NaN entries are not reported by the check; they give NaN results. A is not
// modified.
EIGH_SYM_RTOL :: 1e-10

eigh :: proc(
	A: ^Matrix(f64),
	order: EighOrder = .Ascending,
	allocator: mem.Allocator = context.allocator,
) -> (
	W: []f64,
	V: Matrix(f64),
	ok: bool, // eigenvalues // eigenvectors (columns)
) {
	n := A.rows
	if n == 0 || A.cols != n {
		panic("eigh: matrix must be square and non-empty")
	}

	scale := 0.0
	for v in A.data[:n * n] {
		if !math.is_inf(v) && !math.is_nan(v) {scale = max(scale, math.abs(v))}
	}
	tol := EIGH_SYM_RTOL * scale
	for i in 0 ..< n {
		for j in i + 1 ..< n {
			if math.abs(A.data[i * n + j] - A.data[j * n + i]) > tol {
				W = make([]f64, 0, allocator)
				V = matrix_new(f64, 0, 0, allocator)
				return
			}
		}
	}

	// Work on the symmetric part (A + Aᵀ) / 2, so the result does not depend on
	// which triangle carries the rounding
	A_sym := matrix_new(f64, n, n, allocator)
	defer matrix_free(&A_sym)
	for i in 0 ..< n {
		A_sym.data[i * n + i] = A.data[i * n + i]
		for j in i + 1 ..< n {
			s := 0.5 * (A.data[i * n + j] + A.data[j * n + i])
			A_sym.data[i * n + j], A_sym.data[j * n + i] = s, s
		}
	}

	W, V = _jacobi_eigen_inplace(&A_sym, allocator)

	// Sort eigenvalues + eigenvectors
	en := len(W)
	for i := 0; i < en; i += 1 {
		best := i
		for j := i + 1; j < en; j += 1 {
			cond := (order == .Ascending) ? (W[j] < W[best]) : (W[j] > W[best])
			if cond {
				best = j
			}
		}
		if best == i {
			continue
		}
		// swap eigenvalues
		W[i], W[best] = W[best], W[i]
		// swap columns in V
		for r := 0; r < V.rows; r += 1 {
			V.data[r * V.cols + i], V.data[r * V.cols + best] =
				V.data[r * V.cols + best], V.data[r * V.cols + i]
		}
	}

	ok = true
	return
}
// κ₂(A) = σ_max / σ_min (non-zero σ_min), returns +Inf if rank-deficient
cond2_svd :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> f64 {
	U, S, V := svd_jacobi(A, allocator)
	defer matrix_free(&U)
	defer delete(S, allocator)
	defer matrix_free(&V)
	n := len(S)
	if n == 0 {
		return math.INF_F64
	}

	sigma_max := S[0]
	sigma_min := S[0]
	for i := 1; i < n; i += 1 {
		if S[i] < sigma_min {
			sigma_min = S[i]
		}
	}

	if sigma_min == 0.0 {
		return math.INF_F64
	}
	return sigma_max / sigma_min
}
cond2_svd_thin :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> f64 {
	U_t, S_t, V_t := svd_thin(A, .Jacobi, allocator)
	defer matrix_free(&U_t)
	defer delete(S_t, allocator)
	defer matrix_free(&V_t)
	r := len(S_t)
	if r == 0 {
		return math.INF_F64
	}
	sigma_max := S_t[0]
	sigma_min := S_t[r - 1]
	if sigma_min == 0.0 {
		return math.INF_F64
	}
	return sigma_max / sigma_min
}
// κ₂(A) for symmetric (ideally SPD) matrix via eigenvalues
cond2_sym :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> f64 {
	W, V, ok := eigh(A, .Ascending, allocator)
	defer delete(W, allocator)
	defer matrix_free(&V)
	n := len(W)
	if !ok || n == 0 {
		return math.INF_F64
	}

	lam_min := W[0]
	lam_max := W[n - 1]

	// If not strictly SPD, guard small/negative eigenvalues
	if lam_min <= 0.0 {
		return math.INF_F64
	}
	return lam_max / lam_min
}
// Reciprocal condition number (useful for quick diagnostics)
rcond2_svd :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> f64 {
	c := cond2_svd(A, allocator)
	if math.is_inf(c) || c == 0.0 {
		return 0.0
	}
	return 1.0 / c
}

rcond2_sym :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> f64 {
	c := cond2_sym(A, allocator)
	if math.is_inf(c) || c == 0.0 {
		return 0.0
	}
	return 1.0 / c
}
