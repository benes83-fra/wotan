package wotan_linalg

import "core:math"
import "core:mem"


// Cholesky decomposition: A = L * Lᵀ
// A is symmetric positive definite, stored as full matrix.
// L is written in-place into A (lower triangle), upper triangle is left unspecified.
cholesky_decompose :: proc(A: ^Matrix(f64)) {
	n := A.rows
	if A.cols != n do panic("cholesky_decompose: non-square matrix")

	for j := 0; j < n; j += 1 {
		// Diagonal element
		sum := A.data[j * n + j]
		for k := 0; k < j; k += 1 {
			l_jk := A.data[j * n + k]
			sum -= l_jk * l_jk
		}
		if sum <= 0 {
			panic("cholesky_decompose: matrix not positive definite")
		}
		A.data[j * n + j] = math.sqrt(sum)

		// Off-diagonal elements in column j (rows i > j)
		for i := j + 1; i < n; i += 1 {
			sum = A.data[i * n + j]
			for k := 0; k < j; k += 1 {
				sum -= A.data[i * n + k] * A.data[j * n + k]
			}
			A.data[i * n + j] = sum / A.data[j * n + j]
		}
	}

	// Optionally zero upper triangle for clarity
	for i := 0; i < n; i += 1 {
		for j := i + 1; j < n; j += 1 {
			A.data[i * n + j] = 0.0
		}
	}
}

// Forward solve: L z = b, L lower-triangular (from Cholesky)
forward_substitute :: proc(L: ^Matrix(f64), b: []f64) -> []f64 {
	n := L.rows
	if L.cols != n || len(b) != n {
		panic("forward_substitute: dimension mismatch")
	}

	z := make([]f64, n, context.allocator)
	for i := 0; i < n; i += 1 {
		sum := b[i]
		for j := 0; j < i; j += 1 {
			sum -= L.data[i * n + j] * z[j]
		}
		sum /= L.data[i * n + i]
		z[i] = sum
	}
	return z
}

// Backward solve: Lᵀ x = z, L lower-triangular
backward_substitute :: proc(L: ^Matrix(f64), z: []f64) -> []f64 {
	n := L.rows
	if L.cols != n || len(z) != n {
		panic("backward_substitute: dimension mismatch")
	}

	x := make([]f64, n, context.allocator)
	for i := n - 1; i >= 0; i -= 1 {
		sum := z[i]
		for j := i + 1; j < n; j += 1 {
			sum -= L.data[j * n + i] * x[j]
		}
		sum /= L.data[i * n + i]
		x[i] = sum
	}
	return x
}

// Solve (A) x = b where A is SPD, via Cholesky
solve_spd_cholesky :: proc(
	A: ^Matrix(f64),
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	// Make a working copy of A so we don't destroy the input
	n := A.rows
	if A.cols != n || len(b) != n {
		panic("solve_spd_cholesky: dimension mismatch")
	}

	L := matrix_new(f64, n, n, allocator)
	L.data = A.data // if you want a deep copy, copy elements instead

	cholesky_decompose(&L)
	z := forward_substitute(&L, b)
	x := backward_substitute(&L, z)
	return x
}
