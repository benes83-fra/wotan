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
