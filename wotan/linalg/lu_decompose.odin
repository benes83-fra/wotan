package wotan_linalg

import "core:math"
import "core:mem"

// LU factorization with partial pivoting, blocked and SIMD-updated trailing part.
// A: n×n (square), returns LU (combined L,U), piv (permutation), sign (det sign), ok (non-singular)
lu_decompose :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	LU: Matrix(f64),
	piv: []int,
	sign: int,
	ok: bool,
) {
	n := A.rows
	if n == 0 || A.cols != n {
		panic("lu_decompose: matrix must be non-empty and square")
	}

	LU = matrix_new(f64, n, n, allocator)
	copy(LU.data, A.data)

	piv = make([]int, n, allocator)
	for i := 0; i < n; i += 1 {
		piv[i] = i
	}
	sign = 1
	ok = true

	block := tile_for_matmul() // reuse SIMD tiling heuristic

	// Blocked LU: factor panels, then update trailing submatrix via SIMD matmul
	for k := 0; k < n; k += block {
		panel_width := min(block, n - k)

		// ------------------------------------------------------------
		// 1) Panel factorization (columns k .. k+panel_width-1)
		//    Standard Doolittle with partial pivoting, in-place in LU
		// ------------------------------------------------------------
		for kk := k; kk < k + panel_width; kk += 1 {
			// Pivot search in column kk
			max_row := kk
			max_val := math.abs(LU.data[kk * LU.cols + kk])
			for i := kk + 1; i < n; i += 1 {
				v := math.abs(LU.data[i * LU.cols + kk])
				if v > max_val {
					max_val = v
					max_row = i
				}
			}

			if max_val == 0.0 {
				ok = false // singular
				return
			}

			// Row swap if needed
			if max_row != kk {
				for j := 0; j < n; j += 1 {
					LU.data[kk * LU.cols + j], LU.data[max_row * LU.cols + j] =
						LU.data[max_row * LU.cols + j], LU.data[kk * LU.cols + j]
				}
				piv[kk], piv[max_row] = piv[max_row], piv[kk]
				sign = -sign
			}

			// Factorization step for this column
			pivot := LU.data[kk * LU.cols + kk]
			for i := kk + 1; i < n; i += 1 {
				LU.data[i * LU.cols + kk] /= pivot
				lik := LU.data[i * LU.cols + kk]
				for j := kk + 1; j < k + panel_width; j += 1 {
					LU.data[i * LU.cols + j] -= lik * LU.data[kk * LU.cols + j]
				}
			}
		}

		// ------------------------------------------------------------
		// 2) Trailing update: A22 := A22 - L21 * U12
		//    Use SIMD matmul_dyn_simd for the rank-k update
		// ------------------------------------------------------------
		k2 := k + panel_width
		if k2 < n {
			// Dimensions:
			// L21: (n-k2) × panel_width
			// U12: panel_width × (n-k2)
			// A22: (n-k2) × (n-k2)

			rows_L21 := n - k2
			cols_L21 := panel_width
			rows_U12 := panel_width
			cols_U12 := n - k2

			// Compact copies for SIMD matmul (contiguous row-major)
			L21 := matrix_new(f64, rows_L21, cols_L21, context.temp_allocator)
			U12 := matrix_new(f64, rows_U12, cols_U12, context.temp_allocator)

			// Copy L21 from LU (strictly lower part, but we just take stored multipliers)
			for i := 0; i < rows_L21; i += 1 {
				for j := 0; j < cols_L21; j += 1 {
					src_row := k2 + i
					src_col := k + j
					L21.data[i * L21.cols + j] = LU.data[src_row * LU.cols + src_col]
				}
			}

			// Copy U12 from LU (upper part)
			for i := 0; i < rows_U12; i += 1 {
				for j := 0; j < cols_U12; j += 1 {
					src_row := k + i
					src_col := k2 + j
					U12.data[i * U12.cols + j] = LU.data[src_row * LU.cols + src_col]
				}
			}

			// T = L21 * U12  (SIMD-accelerated)
			T := matmul_dyn_simd(&L21, &U12, context.temp_allocator)

			// A22 := A22 - T, in-place in LU
			for i := 0; i < rows_L21; i += 1 {
				for j := 0; j < cols_U12; j += 1 {
					dst_row := k2 + i
					dst_col := k2 + j
					LU.data[dst_row * LU.cols + dst_col] -= T.data[i * T.cols + j]
				}
			}
		}
	}

	return
}

// Apply permutation P to b: Pb
apply_pivots_vec :: proc(piv: []int, b: []f64) -> []f64 {
	n := len(piv)
	if len(b) != n {
		panic("apply_pivots_vec: dimension mismatch")
	}
	x := make([]f64, n, context.temp_allocator)
	for i := 0; i < n; i += 1 {
		x[i] = b[piv[i]]
	}
	return x
}

// Solve LU x = Pb (LU from lu_decompose, piv permutation)
lu_solve :: proc(
	LU: ^Matrix(f64),
	piv: []int,
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := LU.rows
	if n == 0 || LU.cols != n {
		panic("lu_solve: LU must be square")
	}
	if len(b) != n {
		panic("lu_solve: dimension mismatch")
	}

	// y = P b
	y := apply_pivots_vec(piv, b)

	// Forward solve L y = P b (L has unit diagonal, stored in lower part of LU)
	for i := 0; i < n; i += 1 {
		sum := y[i]
		for j := 0; j < i; j += 1 {
			sum -= LU.data[i * LU.cols + j] * y[j]
		}
		y[i] = sum
	}

	// Backward solve U x = y (U is upper part of LU)
	x := make([]f64, n, allocator)
	for i := n - 1; i >= 0; i -= 1 {
		sum := y[i]
		for j := i + 1; j < n; j += 1 {
			sum -= LU.data[i * LU.cols + j] * x[j]
		}
		piv_ii := LU.data[i * LU.cols + i]
		if piv_ii == 0.0 {
			panic("lu_solve: singular U")
		}
		x[i] = sum / piv_ii
	}

	return x
}

lu_det :: proc(LU: ^Matrix(f64), sign: int) -> f64 {
	n := LU.rows
	if n == 0 || LU.cols != n {
		panic("lu_det: LU must be square")
	}
	det := f64(sign)
	for i := 0; i < n; i += 1 {
		det *= LU.data[i * LU.cols + i]
	}
	return det
}

lu_extract_LU :: proc(
	LU: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	L: Matrix(f64),
	U: Matrix(f64),
) {
	n := LU.rows
	if n == 0 || LU.cols != n {
		panic("lu_extract_LU: LU must be square")
	}

	L = matrix_new(f64, n, n, allocator)
	U = matrix_new(f64, n, n, allocator)

	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			if j < i {
				L.data[i * L.cols + j] = LU.data[i * LU.cols + j]
				U.data[i * U.cols + j] = 0.0
			} else if j == i {
				L.data[i * L.cols + j] = 1.0
				U.data[i * U.cols + j] = LU.data[i * LU.cols + j]
			} else {
				L.data[i * L.cols + j] = 0.0
				U.data[i * U.cols + j] = LU.data[i * LU.cols + j]
			}
		}
	}

	return
}
