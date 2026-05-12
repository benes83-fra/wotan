package wotan_linalg

import "base:builtin"
import "base:intrinsics"
import "core:mem"
import simd "core:simd"


// SIMD dot product for f64
dot_simd :: proc(a, b: []f64) -> f64 {
	n := len(a)
	if n != len(b) {
		panic("dot_simd: length mismatch")
	}

	if !simd.HAS_HARDWARE_SIMD {
		return dot(a, b)
	}

	acc: simd.f64x2 = simd.f64x2{0.0, 0.0}

	i := 0
	for ; i + 2 <= n; i += 2 {
		va := simd.f64x2{a[i], a[i + 1]}
		vb := simd.f64x2{b[i], b[i + 1]}
		acc = intrinsics.simd_add(acc, intrinsics.simd_mul(va, vb))
	}

	// Horizontal reduction via correct transmute syntax
	sum := intrinsics.simd_reduce_add_pairs(acc)


	for ; i < n; i += 1 {
		sum += a[i] * b[i]
	}

	return sum
}

matmul_dyn_simd :: proc(
	a, b: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	if a.cols != b.rows do panic("Dimension mismatch")

	out := matrix_new(f64, a.rows, b.cols, allocator)
	
	// 1. Transpose B once so columns are contiguous
	bt := matrix_new(f64, b.cols, b.rows, context.temp_allocator)
	for r in 0 ..< b.rows {
		for c in 0 ..< b.cols {
			bt.data[c * bt.cols + r] = b.data[r * b.cols + c]
		}
	}

	// 2. Perform multiplication
	for i in 0 ..< a.rows {
		row_a := a.data[i * a.cols : (i+1) * a.cols]
		for j in 0 ..< b.cols {
			// Now we can slice B directly because it's transposed!
			row_bt := bt.data[j * bt.cols : (j+1) * bt.cols]
			out.data[i * out.cols + j] = dot_simd(row_a, row_bt)
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
