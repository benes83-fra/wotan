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

// ============================================================================
// Main QR entry point - maintains ORIGINAL signature for compatibility
// ============================================================================
qr_decompose :: proc(
	A: ^Matrix(f64),
	mode: QR_Mode = .Blocked,
	allocator: mem.Allocator = context.allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	m, n := A.rows, A.cols
	kmax := min(m, n)
	nb := tile_for_matmul() // 16 for AVX, 8 baseline

	// Auto-fallback to unblocked for small matrices (matches Cholesky logic)
	if mode == .Blocked && kmax >= nb * 2 {
		return qr_decompose_blocked(A, nb, allocator)
	}
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
// Blocked QR - simple, safe, matches Cholesky pattern
// ============================================================================
qr_decompose_blocked :: proc(
	A: ^Matrix(f64),
	block_size: int,
	allocator: mem.Allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	m, n := A.rows, A.cols
	kmax := min(m, n)

	// Initialize outputs
	R = matrix_new(f64, m, n, allocator)
	copy(R.data, A.data)

	Q = matrix_new(f64, m, m, allocator)
	for i in 0 ..< m {Q.data[i * m + i] = 1.0}

	// Workspace for panel (reused)
	panel := matrix_new(f64, m, block_size, context.temp_allocator)
	taus := make([]f64, block_size, context.temp_allocator)
	col_buf := make([]f64, m, context.temp_allocator)

	for k in 0 ..< kmax {
		nb_cur := min(block_size, kmax - k)

		// ================================================================
		// 1. Panel factorization: copy panel + unblocked QR on it
		// ================================================================

		// Copy R[k:m, k:k+nb_cur] into panel
		for i in 0 ..< m - k {
			for j in 0 ..< nb_cur {
				panel.data[i * panel.cols + j] = R.data[(k + i) * n + (k + j)]
			}
		}

		// Factor panel in-place (unblocked Householder)
		panel_qr_unblocked(&panel, m - k, nb_cur, taus, context.temp_allocator)

		// Write upper triangle back to R
		for j in 0 ..< nb_cur {
			for i in 0 ..< min(m - k, j + 1) {
				R.data[(k + i) * n + (k + j)] = panel.data[i * panel.cols + j]
			}
		}

		// ================================================================
		// 2. Update trailing matrix: R[k:m, k+nb_cur:n] -= V * (Vᵀ * R_trailing)
		//    Using rank-1 updates with axpy_simd (safe, row-major friendly)
		// ================================================================

		if k + nb_cur < n {
			trailing_cols := n - (k + nb_cur)
			trailing_rows := m - k

			// For each column in trailing matrix
			for col in 0 ..< trailing_cols {
				global_col := k + nb_cur + col

				// Compute y = Vᵀ * R[k:m, global_col]
				y := make([]f64, nb_cur, context.temp_allocator)
				for i in 0 ..< nb_cur {
					v_len := trailing_rows - i // V column i length
					sum := 0.0
					for ii in 0 ..< v_len {
						// V stored in lower triangle of panel
						v_idx := (i + ii) * panel.cols + i
						r_idx := (k + ii) * n + global_col
						sum += panel.data[v_idx] * R.data[r_idx]
					}
					y[i] = sum
				}

				// Apply: R[k:m, global_col] -= 2 * V * y
				for i in 0 ..< trailing_rows {
					sum := 0.0
					for j in 0 ..< nb_cur {
						v_idx := i * panel.cols + j
						sum += panel.data[v_idx] * y[j]
					}
					R.data[(k + i) * n + global_col] -= 2.0 * sum
				}
			}
		}

		// ================================================================
		// 3. Update Q: apply each Householder from panel to Q
		// ================================================================

		for p in 0 ..< nb_cur {
			vk_len := m - (k + p)
			if vk_len <= 0 {continue}

			// Gather Householder vector from panel column p
			for i in 0 ..< vk_len {
				col_buf[i] = panel.data[(p + i) * panel.cols + p]
			}
			vk := col_buf[0:vk_len]

			// Apply H_p to Q[:, k+p:m]
			for i in 0 ..< m {
				row_start := i * m + (k + p)
				row_end := row_start + vk_len
				if row_end > i * m + m {row_end = i * m + m}
				row_seg := Q.data[row_start:row_end]
				if len(row_seg) < vk_len {continue}

				tau := 2.0 * dot_simd(row_seg[0:vk_len], vk)
				if tau != 0.0 {
					axpy_simd(-tau, vk, row_seg[0:vk_len])
				}
			}
		}
	}
	return
}

// ============================================================================
// Helper: Unblocked QR on a panel (FIXED indexing)
// ============================================================================
panel_qr_unblocked :: proc(
	panel: ^Matrix(f64),
	panel_rows: int,
	nb: int,
	taus: []f64,
	allocator: mem.Allocator,
) {
	col_buf := make([]f64, panel_rows, allocator)
	v_work := make([]f64, panel_rows, allocator)

	for k in 0 ..< nb {
		xlen := panel_rows - k
		if xlen <= 0 {taus[k] = 0.0; continue}

		// Gather column k
		for i in 0 ..< xlen {
			col_buf[i] = panel.data[(k + i) * panel.cols + k]
		}

		normx := math.sqrt(dot_simd(col_buf[0:xlen], col_buf[0:xlen]))
		if normx == 0 {taus[k] = 0.0; continue}

		// Build Householder vector
		value: f64
		if col_buf[0] < 0 {
			value = -1.0
		} else {
			value = 1.0
		}
		sign := value
		v0 := col_buf[0] + sign * normx
		v_work[k] = v0
		for i in 1 ..< xlen {v_work[k + i] = col_buf[i]}

		// Store in panel (lower triangle, unit diag implicit)
		panel.data[k * panel.cols + k] = 1.0 // implicit
		for i in 1 ..< xlen {
			panel.data[(k + i) * panel.cols + k] = col_buf[i]
		}

		// Normalize v_work[k:k+xlen]
		vk := v_work[k:k + xlen]
		vnorm := math.sqrt(dot_simd(vk, vk))
		if vnorm == 0 {taus[k] = 0.0; continue}
		inv := 1.0 / vnorm
		for i in 0 ..< xlen {
			v_work[k + i] *= inv
			// FIX: Correct indexing for panel update
			if k + i > k {
				panel.data[(k + i) * panel.cols + k] *= inv
			}
		}

		// Apply to trailing panel columns
		for j in k + 1 ..< nb {
			for i in 0 ..< xlen {
				col_buf[i] = panel.data[(k + i) * panel.cols + j]
			}
			alpha := 2.0 * dot_simd(vk, col_buf[0:xlen])
			for i in 0 ..< xlen {
				panel.data[(k + i) * panel.cols + j] -= alpha * v_work[k + i]
			}
		}

		taus[k] = 2.0 // simplified
	}
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
