package wotan_linalg

import "core:math"
import "core:mem"

// ============================================================================
// QR Configuration (optional, for advanced usage)
// ============================================================================
QR_Config :: struct {
	compute_q: bool, // Compute full Q matrix (default: true)
}

// ============================================================================
// Main QR entry point - maintains ORIGINAL signature for full compatibility
// ============================================================================
qr_decompose :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	return qr_decompose_impl(A, QR_Config{compute_q = true}, allocator)
}

// Internal implementation with config support
qr_decompose_impl :: proc(
	A: ^Matrix(f64),
	config: QR_Config,
	allocator: mem.Allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	m, n := A.rows, A.cols
	kmax := min(m, n)

	// R = copy of A
	R = matrix_new(f64, m, n, allocator)
	copy(R.data, A.data)

	// Q = identity (if requested)
	if config.compute_q {
		Q = matrix_new(f64, m, m, allocator)
		for i in 0 ..< m {
			Q.data[i * m + i] = 1.0
		}
	}

	// Pre-allocate workspace ONCE (reused each iteration)
	v := make([]f64, m, context.temp_allocator) // Householder vector
	w := make([]f64, n, context.temp_allocator) // for R update
	u := make([]f64, m, context.temp_allocator) // for Q update
	col_buf := make([]f64, m, context.temp_allocator) // reusable gather buffer

	for k in 0 ..< kmax {
		xlen := m - k // active column length

		// ================================================================
		// 1. Build Householder vector v from column k of R
		// ================================================================

		// Gather column k segment (strided access in row-major)
		for i in 0 ..< xlen {
			col_buf[i] = R.data[(k + i) * n + k]
		}
		x := col_buf[0:xlen]

		normx := math.sqrt(dot_simd(x, x))
		if normx == 0 {continue}

		// Reset active portion of v
		for i in k ..< m {v[i] = 0.0}

		// Compute v0 with numerical stability trick
		sign := 1.0
		if x[0] < 0 {sign = -1.0}
		v[k] = x[0] + sign * normx
		for i in 1 ..< xlen {v[k + i] = x[i]}

		// Normalize v[k:k+xlen]
		vk := v[k:k + xlen]
		vnorm := math.sqrt(dot_simd(vk, vk))
		if vnorm == 0 {continue}
		inv_vnorm := 1.0 / vnorm
		for i in 0 ..< xlen {v[k + i] *= inv_vnorm}

		// ================================================================
		// 2. Apply H_k to R: R = (I - 2*v*vᵀ) * R
		//    Reformulated as row-wise rank-1 update (row-major friendly)
		// ================================================================

		w_len := n - k
		// Compute w = R[k:m, k:n]ᵀ * v[k:m]
		for j in 0 ..< w_len {
			for i in 0 ..< xlen {
				col_buf[i] = R.data[(k + i) * n + (k + j)]
			}
			w[j] = dot_simd(vk, col_buf[0:xlen])
		}

		// Apply: R[k:m, k:n] -= 2 * outer(v[k:m], w[0:w_len])
		// KEY: updating ROWS → contiguous memory in row-major!
		for i in 0 ..< xlen {
			row_start := (k + i) * n + k
			row_seg := R.data[row_start:row_start + w_len]
			scalar := 2.0 * v[k + i]
			if scalar != 0.0 {
				axpy_simd(-scalar, w[0:w_len], row_seg)
			}
		}

		// ================================================================
		// 3. Apply H_kᵀ to Q: Q = Q * (I - 2*v*vᵀ)  (if computing Q)
		// ================================================================

		if config.compute_q {
			// Compute u = Q * v[k:m] (only columns k:m matter)
			for i in 0 ..< m {
				row_seg := Q.data[i * m + k:i * m + m]
				u[i] = dot_simd(row_seg[0:xlen], vk)
			}
			// Apply: Q -= 2 * outer(u, v[k:m])
			for i in 0 ..< m {
				row_seg := Q.data[i * m + k:i * m + k + xlen]
				scalar := 2.0 * u[i]
				if scalar != 0.0 {
					axpy_simd(-scalar, vk, row_seg)
				}
			}
		}
	}

	return
}

// ============================================================================
// Upper triangular solve (unchanged from your original)
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
