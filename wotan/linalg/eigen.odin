package wotan_linalg

import "core:math"
import "core:mem"


EighOrder :: enum {
	Ascending,
	Descending,
}

// Symmetric eigen-decomposition: A = V diag(W) Vᵀ
eigh :: proc(
	A: ^Matrix(f64),
	order: EighOrder = .Ascending,
	allocator: mem.Allocator = context.allocator,
) -> (
	W: []f64,
	V: Matrix(f64), // eigenvalues// eigenvectors (columns)
) {
	// Optionally: debug symmetry check
	when ODIN_DEBUG {
		if A.rows != A.cols {
			panic("eigh: matrix must be square")
		}
		n := A.rows
		for i := 0; i < n; i += 1 {
			for j := i + 1; j < n; j += 1 {
				aij := A.data[i * A.cols + j]
				aji := A.data[j * A.cols + i]
				if math.abs(aij - aji) > 1e-12 {
					panic("eigh: matrix not symmetric within tolerance")
				}
			}
		}
	}

	// Work on a copy, since jacobi_eigen_symmetric modifies in-place
	A_copy := matrix_new(f64, A.rows, A.cols, allocator)
	copy(A_copy.data, A.data)

	W, V = jacobi_eigen_symmetric(&A_copy, allocator)

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

	return
}
// κ₂(A) = σ_max / σ_min (non-zero σ_min), returns +Inf if rank-deficient
cond2_svd :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> f64 {
	_, S, _ := svd_golub_reinsch(A, allocator)
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
	_, S_t, _ := svd_thin(A, .GolubReinsch, allocator)
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
	W, _ := eigh(A, .Ascending, allocator)
	n := len(W)
	if n == 0 {
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
