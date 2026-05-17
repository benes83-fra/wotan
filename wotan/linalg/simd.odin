package wotan_linalg

import "base:builtin"
import "base:intrinsics"
import "core:fmt"
import "core:mem"
import simd "core:simd"

TILE_BASE :: 8
TILE_AVX :: 16

tile_for_matmul :: proc() -> int {
	// Prefer AVX/AVX2/AVX-512 tile size
	if intrinsics.has_target_feature("avx") {
		return TILE_AVX
	}
	return TILE_BASE
}
// SIMD dot product for f64
dot_simd :: proc(a, b: []f64) -> f64 {
	n := len(a)
	if n != len(b) do panic("dot_simd: length mismatch")

	if !simd.HAS_HARDWARE_SIMD {
		return dot(a, b)
	}

	// ============================================================
	// 1. Try widest SIMD first (f64x8) — requires AVX/AVX2/AVX512
	// ============================================================
	if intrinsics.has_target_feature("avx") {
		acc8: simd.f64x8 = simd.f64x8{0, 0, 0, 0, 0, 0, 0, 0}
		i := 0
		for ; i + 8 <= n; i += 8 {
			va := simd.f64x8 {
				a[i],
				a[i + 1],
				a[i + 2],
				a[i + 3],
				a[i + 4],
				a[i + 5],
				a[i + 6],
				a[i + 7],
			}
			vb := simd.f64x8 {
				b[i],
				b[i + 1],
				b[i + 2],
				b[i + 3],
				b[i + 4],
				b[i + 5],
				b[i + 6],
				b[i + 7],
			}
			acc8 = intrinsics.simd_add(acc8, intrinsics.simd_mul(va, vb))
		}
		sum := intrinsics.simd_reduce_add_pairs(acc8)
		for ; i < n; i += 1 do sum += a[i] * b[i]
		return sum
	}

	// ==========================================
	// 2. Next: f64x4 (SSE2/SSE4, always available)
	// ==========================================
	if intrinsics.has_target_feature("sse2") {
		acc4: simd.f64x4 = simd.f64x4{0, 0, 0, 0}
		i := 0
		for ; i + 4 <= n; i += 4 {
			va := simd.f64x4{a[i], a[i + 1], a[i + 2], a[i + 3]}
			vb := simd.f64x4{b[i], b[i + 1], b[i + 2], b[i + 3]}
			acc4 = intrinsics.simd_add(acc4, intrinsics.simd_mul(va, vb))
		}
		sum := intrinsics.simd_reduce_add_pairs(acc4)
		for ; i < n; i += 1 do sum += a[i] * b[i]
		return sum
	}

	// ===========================
	// 3. Fallback: f64x2 (baseline)
	// ===========================
	acc2: simd.f64x2 = simd.f64x2{0, 0}
	i := 0
	for ; i + 2 <= n; i += 2 {
		va := simd.f64x2{a[i], a[i + 1]}
		vb := simd.f64x2{b[i], b[i + 1]}
		acc2 = intrinsics.simd_add(acc2, intrinsics.simd_mul(va, vb))
	}
	sum := intrinsics.simd_reduce_add_pairs(acc2)
	for ; i < n; i += 1 do sum += a[i] * b[i]
	return sum
}

matmul_dyn_simd :: proc(
	a, b: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	if a.cols != b.rows do panic("Dimension mismatch")

	out := matrix_new(f64, a.rows, b.cols, allocator)

	// Transpose B once
	bt := matrix_new(f64, b.cols, b.rows, context.temp_allocator)
	for r in 0 ..< b.rows {
		for c in 0 ..< b.cols {
			bt.data[c * bt.cols + r] = b.data[r * b.cols + c]
		}
	}

	tile := tile_for_matmul()

	for ii := 0; ii < a.rows; ii += tile {
		for jj := 0; jj < b.cols; jj += tile {
			for kk := 0; kk < a.cols; kk += tile {

				i_end := min(ii + tile, a.rows)
				j_end := min(jj + tile, b.cols)
				k_end := min(kk + tile, a.cols)

				for i := ii; i < i_end; i += 1 {
					row_a := a.data[i * a.cols:i * a.cols + a.cols]

					for j := jj; j < j_end; j += 1 {
						row_bt := bt.data[j * bt.cols:j * bt.cols + bt.cols]

						sum := dot_simd(row_a[kk:k_end], row_bt[kk:k_end])

						out.data[i * out.cols + j] += sum
					}
				}
			}
		}
	}

	return out
}


matvec_dyn_simd :: proc(
	m: ^Matrix(f64),
	x: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	if m.cols != len(x) {
		panic("matvec_dyn: dimension mismatch")
	}

	y := make([]f64, m.rows, allocator)

	for r in 0 ..< m.rows {
		row := m.data[r * m.cols:r * m.cols + m.cols]
		y[r] = dot_simd(row, x)
	}

	return y
}

forward_subst_unit_lower_simd :: proc(
	L: ^Matrix(f64),
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := L.rows
	if n == 0 || L.cols != n {
		panic("forward_subst_unit_lower_simd: L must be square")
	}
	if len(b) != n {
		panic("forward_subst_unit_lower_simd: dimension mismatch")
	}

	y := make([]f64, n, allocator)

	for i := 0; i < n; i += 1 {
		// sum = L[i,0:i] · y[0:i]
		if i == 0 {
			y[0] = b[0] // unit diagonal
			continue
		}
		row := L.data[i * L.cols:i * L.cols + i] // 0 .. i-1
		sum := dot_simd(row, y[0:i])
		y[i] = b[i] - sum // L[i,i] = 1
	}

	return y
}

// Solve U x = y, U upper-triangular with non-unit diagonal
// Uses SIMD dot on the trailing part of x.
back_subst_upper_simd :: proc(
	U: ^Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := U.rows
	if n == 0 || U.cols != n {
		panic("back_subst_upper_simd: U must be square")
	}
	if len(y) != n {
		panic("back_subst_upper_simd: dimension mismatch")
	}

	x := make([]f64, n, allocator)

	for ii := n - 1; ii >= 0; ii -= 1 {
		i := ii

		// sum = U[i,i+1:n] · x[i+1:n]
		if i == n - 1 {
			diag := U.data[i * U.cols + i]
			if diag == 0.0 {
				panic("back_subst_upper_simd: zero diagonal")
			}
			x[i] = y[i] / diag
			continue
		}

		row_tail := U.data[i * U.cols + i + 1:i * U.cols + n]
		sum := dot_simd(row_tail, x[i + 1:n])

		diag := U.data[i * U.cols + i]
		if diag == 0.0 {
			panic("back_subst_upper_simd: zero diagonal")
		}
		x[i] = (y[i] - sum) / diag
	}

	return x
}
// Solve LU x = Pb using SIMD triangular solves
lu_solve_simd :: proc(
	LU: ^Matrix(f64),
	piv: []int,
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := LU.rows
	if n == 0 || LU.cols != n {
		panic("lu_solve_simd: LU must be square")
	}
	if len(b) != n {
		panic("lu_solve_simd: dimension mismatch")
	}

	// y = P b
	y_perm := apply_pivots_vec(piv, b)

	// L y = P b  (L unit lower in LU)
	y := forward_subst_unit_lower_simd(LU, y_perm, allocator)

	// U x = y  (U upper in LU)
	x := back_subst_upper_simd(LU, y, allocator)

	return x
}
// y += alpha * x  (SIMD, f64)
// y += alpha * x  (SIMD, f64)
axpy_simd :: proc(alpha: f64, x, y: []f64) {
	n := len(x)
	if n != len(y) do panic("axpy_simd: length mismatch")

	if alpha == 0.0 {
		return
	}

	if !simd.HAS_HARDWARE_SIMD {
		for i := 0; i < n; i += 1 {
			y[i] += alpha * x[i]
		}
		return
	}

	// AVX: f64x8
	if intrinsics.has_target_feature("avx") {
		a8: simd.f64x8 = simd.f64x8{alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha}
		i := 0
		for ; i + 8 <= n; i += 8 {
			vx := simd.f64x8 {
				x[i + 0],
				x[i + 1],
				x[i + 2],
				x[i + 3],
				x[i + 4],
				x[i + 5],
				x[i + 6],
				x[i + 7],
			}
			vy := simd.f64x8 {
				y[i + 0],
				y[i + 1],
				y[i + 2],
				y[i + 3],
				y[i + 4],
				y[i + 5],
				y[i + 6],
				y[i + 7],
			}
			vy = intrinsics.simd_add(vy, intrinsics.simd_mul(a8, vx))

			// spill SIMD back to scalars
			vy_arr := transmute([8]f64)vy
			y[i + 0] = vy_arr[0]
			y[i + 1] = vy_arr[1]
			y[i + 2] = vy_arr[2]
			y[i + 3] = vy_arr[3]
			y[i + 4] = vy_arr[4]
			y[i + 5] = vy_arr[5]
			y[i + 6] = vy_arr[6]
			y[i + 7] = vy_arr[7]
		}
		for ; i < n; i += 1 {
			y[i] += alpha * x[i]
		}
		return
	}

	// SSE2: f64x4
	if intrinsics.has_target_feature("sse2") {
		a4: simd.f64x4 = simd.f64x4{alpha, alpha, alpha, alpha}
		i := 0
		for ; i + 4 <= n; i += 4 {
			vx := simd.f64x4{x[i + 0], x[i + 1], x[i + 2], x[i + 3]}
			vy := simd.f64x4{y[i + 0], y[i + 1], y[i + 2], y[i + 3]}
			vy = intrinsics.simd_add(vy, intrinsics.simd_mul(a4, vx))

			vy_arr := transmute([4]f64)vy
			y[i + 0] = vy_arr[0]
			y[i + 1] = vy_arr[1]
			y[i + 2] = vy_arr[2]
			y[i + 3] = vy_arr[3]
		}
		for ; i < n; i += 1 {
			y[i] += alpha * x[i]
		}
		return
	}

	// Baseline: f64x2
	a2: simd.f64x2 = simd.f64x2{alpha, alpha}
	i := 0
	for ; i + 2 <= n; i += 2 {
		vx := simd.f64x2{x[i + 0], x[i + 1]}
		vy := simd.f64x2{y[i + 0], y[i + 1]}
		vy = intrinsics.simd_add(vy, intrinsics.simd_mul(a2, vx))

		vy_arr := transmute([2]f64)vy
		y[i + 0] = vy_arr[0]
		y[i + 1] = vy_arr[1]
	}
	for ; i < n; i += 1 {
		y[i] += alpha * x[i]
	}
}
ger_simd :: proc(alpha: f64, x: []f64, y: []f64, A: ^Matrix(f64)) {
	m := A.rows
	n := A.cols

	if len(x) != m {
		panic("ger_simd: x length mismatch with A.rows")
	}
	if len(y) != n {
		panic("ger_simd: y length mismatch with A.cols")
	}
	if alpha == 0.0 {
		return
	}

	for i := 0; i < m; i += 1 {
		scalar := alpha * x[i]
		if scalar == 0.0 {
			continue
		}
		row := A.data[i * A.cols:i * A.cols + n]
		axpy_simd(scalar, y, row)
	}
}


rotate_pair_simd :: proc(c, s: f64, col_p, col_q: []f64, length: int) {
	// [new_p; new_q] = [c, -s; s, c] * [old_p; old_q]
	// len: explicit number of elements to process (avoids slice aliasing issues)

	if length > len(col_p) || length > len(col_q) {
		panic("rotate_pair_simd: length exceeds array bounds")
	}

	if !simd.HAS_HARDWARE_SIMD {
		for i := 0; i < length; i += 1 {
			up := col_p[i]
			uq := col_q[i]
			col_p[i] = c * up - s * uq
			col_q[i] = s * up + c * uq
		}
		return
	}

	// AVX: f64x8
	if intrinsics.has_target_feature("avx") {
		cc := simd.f64x8{c, c, c, c, c, c, c, c}
		ss := simd.f64x8{s, s, s, s, s, s, s, s}
		i := 0
		for ; i + 8 <= length; i += 8 {
			vp := simd.f64x8 {
				col_p[i + 0],
				col_p[i + 1],
				col_p[i + 2],
				col_p[i + 3],
				col_p[i + 4],
				col_p[i + 5],
				col_p[i + 6],
				col_p[i + 7],
			}
			vq := simd.f64x8 {
				col_q[i + 0],
				col_q[i + 1],
				col_q[i + 2],
				col_q[i + 3],
				col_q[i + 4],
				col_q[i + 5],
				col_q[i + 6],
				col_q[i + 7],
			}
			// new_p = c*old_p - s*old_q
			new_p := intrinsics.simd_sub(intrinsics.simd_mul(cc, vp), intrinsics.simd_mul(ss, vq))
			// new_q = s*old_p + c*old_q
			new_q := intrinsics.simd_add(intrinsics.simd_mul(ss, vp), intrinsics.simd_mul(cc, vq))
			// Store back via transmute (ensures memory visibility)
			new_p_arr := transmute([8]f64)new_p
			new_q_arr := transmute([8]f64)new_q
			for j := 0; j < 8; j += 1 {
				col_p[i + j] = new_p_arr[j]
				col_q[i + j] = new_q_arr[j]
			}
		}
		// Tail: scalar fallback
		for ; i < length; i += 1 {
			up := col_p[i]; uq := col_q[i]
			col_p[i] = c * up - s * uq
			col_q[i] = s * up + c * uq
		}
		return
	}

	// SSE2/fallback: similar pattern (omitted for brevity - add if needed)
	// For now, scalar fallback handles all non-AVX cases
	for i := 0; i < length; i += 1 {
		up := col_p[i]; uq := col_q[i]
		col_p[i] = c * up - s * uq
		col_q[i] = s * up + c * uq
	}
}

xtx_simd :: proc(A: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> Matrix(f64) {
	m, n := A.rows, A.cols
	// AtA is n×n symmetric
	AtA := matrix_new(f64, n, n, allocator)

	// Pre-allocate column buffers ONCE (reused for every dot product)
	// This avoids O(n²) allocations in the nested loops
	col_i := make([]f64, m, context.temp_allocator)
	col_j := make([]f64, m, context.temp_allocator)

	for ii in 0 ..< n {
		// Extract column ii of A into col_i (strided access in row-major)
		for k in 0 ..< m {
			col_i[k] = A.data[k * A.cols + ii]
		}

		for jj in 0 ..= ii { 	// Only compute lower triangle (symmetric)
			if ii == jj {
				// Diagonal: norm squared of column ii
				AtA.data[ii * n + ii] = dot_simd(col_i, col_i)
			} else {
				// Extract column jj of A into col_j
				for k in 0 ..< m {
					col_j[k] = A.data[k * A.cols + jj]
				}
				// Off-diagonal: dot product of columns ii and jj
				sum := dot_simd(col_i, col_j)
				AtA.data[ii * n + jj] = sum
				AtA.data[jj * n + ii] = sum // Symmetric
			}
		}
	}
	return AtA
}

// Optional: Add this for large-m matrices
xtx_blocked_simd :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	m, n := A.rows, A.cols
	AtA := matrix_new(f64, n, n, allocator)

	// Reusable column buffers
	col_i := make([]f64, m, context.temp_allocator)
	col_j := make([]f64, m, context.temp_allocator)

	// Block size for cache efficiency
	tile := tile_for_matmul() // 16 for AVX

	for ii in 0 ..< n {
		// Extract column ii
		for k in 0 ..< m {col_i[k] = A.data[k * A.cols + ii]}

		for jj in 0 ..= ii {
			if ii == jj {
				AtA.data[ii * n + ii] = dot_simd(col_i, col_i)
			} else {
				// Extract column jj (could also block this if m is huge)
				for k in 0 ..< m {col_j[k] = A.data[k * A.cols + jj]}
				sum := dot_simd(col_i, col_j)
				AtA.data[ii * n + jj] = sum
				AtA.data[jj * n + ii] = sum
			}
		}
	}
	return AtA
}
