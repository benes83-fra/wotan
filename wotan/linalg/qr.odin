package wotan_linalg

import "core:math"
import "core:mem"

// ============================================================================
// QR Mode enum - matches Cholesky_Mode pattern exactly
// ============================================================================
QR_Mode :: enum {
	Unblocked,
	Blocked,
}
// R factor only (m×n, upper triangular), via Householder reflections
qr_decompose_thin :: proc(
	A: ^Matrix(f64),
	mode: QR_Mode = .Blocked,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	Q, R := qr_decompose_unblocked(A, allocator)
	matrix_free(&Q)
	return R
}
// ============================================================================
// Main QR entry point - maintains ORIGINAL signature for compatibility
// ============================================================================
// Both modes use the Householder QR below: the former blocked variant produced
// wrong factors once min(m, n) reached two blocks (32 columns with AVX).
qr_decompose :: proc(
	A: ^Matrix(f64),
	mode: QR_Mode = .Blocked,
	allocator: mem.Allocator = context.allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	return qr_decompose_unblocked(A, allocator)
}

// ============================================================================
// Unblocked QR (SIMD-optimized, row-major) - your working version
// ============================================================================
qr_decompose_unblocked :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	m, n := A.rows, A.cols
	kmax := min(m, n)

	R = matrix_new(f64, m, n, allocator)
	copy(R.data, A.data)

	Q = matrix_new(f64, m, m, allocator)
	for i in 0 ..< m {Q.data[i * m + i] = 1.0}

	v := make([]f64, m, context.temp_allocator)
	w := make([]f64, n, context.temp_allocator)
	u := make([]f64, m, context.temp_allocator)
	col_buf := make([]f64, m, context.temp_allocator)

	for k in 0 ..< kmax {
		xlen := m - k
		for i in 0 ..< xlen {col_buf[i] = R.data[(k + i) * n + k]}
		x := col_buf[0:xlen]

		normx := math.sqrt(dot_simd(x, x))
		if normx == 0 {continue}

		for i in k ..< m {v[i] = 0.0}
		value: f64
		if x[0] < 0 {
			value = -1.0
		} else {
			value = 1.0
		}
		sign := value
		v[k] = x[0] + sign * normx
		for i in 1 ..< xlen {v[k + i] = x[i]}

		vk := v[k:k + xlen]
		vnorm := math.sqrt(dot_simd(vk, vk))
		if vnorm == 0 {continue}
		inv := 1.0 / vnorm
		for i in 0 ..< xlen {v[k + i] *= inv}

		w_len := n - k
		for j in 0 ..< w_len {
			for i in 0 ..< xlen {col_buf[i] = R.data[(k + i) * n + (k + j)]}
			w[j] = dot_simd(vk, col_buf[0:xlen])
		}
		for i in 0 ..< xlen {
			row_start := (k + i) * n + k
			row_seg := R.data[row_start:row_start + w_len]
			scalar := 2.0 * v[k + i]
			if scalar != 0.0 {axpy_simd(-scalar, w[0:w_len], row_seg)}
		}

		for i in 0 ..< m {
			row_seg := Q.data[i * m + k:i * m + m]
			u[i] = dot_simd(row_seg[0:xlen], vk)
		}
		for i in 0 ..< m {
			row_seg := Q.data[i * m + k:i * m + k + xlen]
			scalar := 2.0 * u[i]
			if scalar != 0.0 {axpy_simd(-scalar, vk, row_seg)}
		}
	}
	return
}

// ============================================================================
// Upper triangular solve (unchanged)
// ============================================================================
upper_tri_solve :: proc(
	R: ^Matrix(f64),
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(b)
	x := make([]f64, n, allocator)
	for i := n - 1; i >= 0; i -= 1 {
		sum := b[i]
		for j := i + 1; j < n; j += 1 {
			sum -= R.data[i * R.cols + j] * x[j]
		}
		sum /= R.data[i * R.cols + i]
		x[i] = sum
	}
	return x
}
