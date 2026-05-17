package wotan_linalg

import "core:math"
import "core:mem"


Cholesky_Mode :: enum {
	Unblocked,
	Blocked,
}

cholesky :: proc(A: ^Matrix(f64), mode: Cholesky_Mode) {
	switch mode {
	case .Unblocked:
		cholesky_decompose(A)
	case .Blocked:
		cholesky_decompose_blocked(A)
	}
}


// Cholesky decomposition: A = L * Lᵀ
// A is symmetric positive definite, stored as full matrix.
// L is written in-place into A (lower triangle), upper triangle is left unspecified.
cholesky_decompose :: proc(A: ^Matrix(f64)) {
	n := A.rows
	if A.cols != n do panic("cholesky_decompose: non-square matrix")

	for j := 0; j < n; j += 1 {
		// Diagonal
		sum := A.data[j * n + j]
		if j > 0 {
			row_j := A.data[j * n:j * n + j]
			sum -= dot_simd(row_j, row_j)
		}
		if sum <= 0 {
			panic("cholesky_decompose: matrix not positive definite")
		}
		A.data[j * n + j] = math.sqrt(sum)

		// Off-diagonal
		for i := j + 1; i < n; i += 1 {
			sum = A.data[i * n + j]
			if j > 0 {
				row_i := A.data[i * n:i * n + j]
				row_j := A.data[j * n:j * n + j]
				sum -= dot_simd(row_i, row_j)
			}
			A.data[i * n + j] = sum / A.data[j * n + j]
		}
	}

	// Optional: zero upper triangle
	for i := 0; i < n; i += 1 {
		for j := i + 1; j < n; j += 1 {
			A.data[i * n + j] = 0.0
		}
	}
}

// Forward solve: L z = b, L lower-triangular (from Cholesky)
forward_substitute_simd :: proc(
	L: ^Matrix(f64),
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := L.rows
	if L.cols != n || len(b) != n {
		panic("forward_substitute_simd: dimension mismatch")
	}

	z := make([]f64, n, allocator)

	for i := 0; i < n; i += 1 {
		if i == 0 {
			z[0] = b[0] / L.data[0]
			continue
		}

		row := L.data[i * n:i * n + i] // L[i,0:i]
		sum := dot_simd(row, z[0:i])
		z[i] = (b[i] - sum) / L.data[i * n + i]
	}

	return z
}


// Backward solve: Lᵀ x = z, L lower-triangular
backward_substitute_simd :: proc(
	L: ^Matrix(f64),
	z: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := L.rows
	if L.cols != n || len(z) != n {
		panic("backward_substitute_simd: dimension mismatch")
	}

	x := make([]f64, n, allocator)

	for ii := n - 1; ii >= 0; ii -= 1 {
		i := ii

		if i == n - 1 {
			x[i] = z[i] / L.data[i * n + i]
			continue
		}

		// length of the tail (rows i+1 .. n-1)
		tail_len := n - (i + 1)

		// gather column i into a contiguous temp buffer
		tmp := make([]f64, tail_len, context.temp_allocator)
		for k := 0; k < tail_len; k += 1 {
			tmp[k] = L.data[(i + 1 + k) * n + i]
		}

		// SIMD dot
		sum := dot_simd(tmp, x[i + 1:n])

		x[i] = (z[i] - sum) / L.data[i * n + i]
	}

	return x
}


// Solve (A) x = b where A is SPD, via Cholesky
solve_spd_cholesky :: proc(
	A: ^Matrix(f64),
	b: []f64,
	mode: Cholesky_Mode = .Blocked, // default to fast version
	allocator: mem.Allocator = context.allocator,
) -> []f64 {

	n := A.rows
	if A.cols != n || len(b) != n {
		panic("solve_spd_cholesky: dimension mismatch")
	}

	L := matrix_new(f64, n, n, allocator)
	copy(L.data, A.data)

	cholesky(&L, mode)

	z := forward_substitute_simd(&L, b, context.temp_allocator)
	x := backward_substitute_simd(&L, z, allocator)
	return x
}


spd_inverse :: proc(
	A: ^Matrix(f64),
	mode: Cholesky_Mode = .Blocked,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {

	n := A.rows
	if A.cols != n do panic("spd_inverse: non-square")

	Acopy := matrix_new(f64, n, n, allocator)
	copy(Acopy.data, A.data)

	cholesky(&Acopy, mode)

	inv := matrix_new(f64, n, n, allocator)
	e := make([]f64, n, context.temp_allocator)

	for k := 0; k < n; k += 1 {
		for i := 0; i < n; i += 1 do e[i] = 0
		e[k] = 1

		z := forward_substitute_simd(&Acopy, e, context.temp_allocator)
		x := backward_substitute_simd(&Acopy, z, context.temp_allocator)

		for i := 0; i < n; i += 1 {
			inv.data[i * n + k] = x[i]
		}
	}

	return inv
}

// Blocked Cholesky: A = L Lᵀ, in-place on A (lower triangle filled)
cholesky_decompose_blocked :: proc(A: ^Matrix(f64)) {
	n := A.rows
	if A.cols != n do panic("cholesky_decompose_blocked: non-square matrix")

	block := tile_for_matmul()

	for k := 0; k < n; k += block {
		b := min(block, n - k)

		// ----------------------------------------------------
		// 1) Factor diagonal block A11 = A[k:k+b, k:k+b]
		//    Use existing unblocked Cholesky as panel kernel
		// ----------------------------------------------------
		A11 := matrix_new(f64, b, b, context.temp_allocator)
		for i := 0; i < b; i += 1 {
			for j := 0; j < b; j += 1 {
				A11.data[i * A11.cols + j] = A.data[(k + i) * A.cols + (k + j)]
			}
		}

		cholesky_decompose(&A11)

		// write back L11 into A (lower triangle)
		for i := 0; i < b; i += 1 {
			for j := 0; j <= i; j += 1 {
				A.data[(k + i) * A.cols + (k + j)] = A11.data[i * A11.cols + j]
			}
		}

		// ----------------------------------------------------
		// 2) Compute L21: A21 := A21 * inv(L11ᵀ)
		//    For rows i = k+b .. n-1, columns j = k .. k+b-1
		// ----------------------------------------------------
		rows_A21 := n - (k + b)
		if rows_A21 > 0 {
			// L11 view is A11 (b×b, lower-triangular)
			for j := 0; j < b; j += 1 {
				// RHS: column (k+j) of A21 (rows k+b..n-1)
				rhs := make([]f64, rows_A21, context.temp_allocator)
				for i := 0; i < rows_A21; i += 1 {
					rhs[i] = A.data[(k + b + i) * A.cols + (k + j)]
				}

				// Solve L11 * x = rhs  (forward, lower-triangular)
				x := forward_substitute_simd(&A11, rhs, context.temp_allocator)

				// Write back into A21 as column j
				for i := 0; i < rows_A21; i += 1 {
					A.data[(k + b + i) * A.cols + (k + j)] = x[i]
				}
			}
		}

		// ----------------------------------------------------
		// 3) Trailing update: A22 := A22 - L21 * L21ᵀ
		//    Use rank-1 updates with axpy_simd
		// ----------------------------------------------------
		if rows_A21 > 0 {
			m := rows_A21 // size of A22: m×m
			for j := 0; j < b; j += 1 {
				// v = column j of L21 (length m)
				v := make([]f64, m, context.temp_allocator)
				for i := 0; i < m; i += 1 {
					v[i] = A.data[(k + b + i) * A.cols + (k + j)]
				}

				// A22 -= v vᵀ
				for i := 0; i < m; i += 1 {
					row_start := (k + b + i) * A.cols + (k + b)
					row := A.data[row_start:row_start + m]
					axpy_simd(-v[i], v, row)
				}
			}
		}
	}

	// Optional: zero upper triangle
	for i := 0; i < n; i += 1 {
		for j := i + 1; j < n; j += 1 {
			A.data[i * A.cols + j] = 0.0
		}
	}
}
