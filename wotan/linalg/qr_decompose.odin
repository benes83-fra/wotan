package wotan_linalg

import "core:math"
import "core:mem"

qr_decompose :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	m := A.rows
	n := A.cols
	kmax := min(m, n)

	// R = copy(A)
	R = matrix_new(f64, m, n, allocator)
	copy(R.data, A.data)

	// Q = I_m
	Q = matrix_new(f64, m, m, allocator)
	for i := 0; i < m; i += 1 {
		for j := 0; j < m; j += 1 {
			if i == j {
				Q.data[i * m + j] = 1.0
			} else {
				Q.data[i * m + j] = 0.0
			}
		}
	}

	v := make([]f64, m, context.temp_allocator)

	for k := 0; k < kmax; k += 1 {
		xlen := m - k

		// 1. Build Householder vector v for column k
		x := make([]f64, xlen, context.temp_allocator)
		for i := 0; i < xlen; i += 1 {
			x[i] = R.data[(k + i) * n + k]
		}

		normx := 0.0
		for i := 0; i < xlen; i += 1 {
			normx += x[i] * x[i]
		}
		normx = math.sqrt(normx)
		if normx == 0 {
			continue
		}

		// reset v
		for i := 0; i < m; i += 1 {
			v[i] = 0.0
		}

		sign := 1.0
		if x[0] < 0 {
			sign = -1.0
		}
		v0 := x[0] + sign * normx

		v[k] = v0
		for i := 1; i < xlen; i += 1 {
			v[k + i] = x[i]
		}

		// normalize v
		vnorm := 0.0
		for i := 0; i < xlen; i += 1 {
			vi := v[k + i]
			vnorm += vi * vi
		}
		vnorm = math.sqrt(vnorm)
		if vnorm == 0 {
			continue
		}
		for i := 0; i < xlen; i += 1 {
			v[k + i] /= vnorm
		}

		vk := v[k:k + xlen]

		// 2. Apply H_k to R: R = (I - 2 v vᵀ) R
		for j := k; j < n; j += 1 {
			col := make([]f64, xlen, context.temp_allocator)
			for i := 0; i < xlen; i += 1 {
				col[i] = R.data[(k + i) * n + j]
			}

			alpha := 2.0 * dot_simd(vk, col)

			for i := 0; i < xlen; i += 1 {
				R.data[(k + i) * n + j] -= alpha * v[k + i]
			}
		}

		// 3. Apply H_kᵀ on the RIGHT of Q: Q = Q (I - 2 v vᵀ)
		//    i.e. operate on rows of Q
		for i := 0; i < m; i += 1 {
			// row segment Q[i, k:m]
			row := make([]f64, xlen, context.temp_allocator)
			for j := 0; j < xlen; j += 1 {
				row[j] = Q.data[i * m + (k + j)]
			}

			tau := 2.0 * dot_simd(row, vk)

			for j := 0; j < xlen; j += 1 {
				Q.data[i * m + (k + j)] -= tau * v[k + j]
			}
		}
	}

	return
}


upper_tri_solve :: proc(
	R: ^Matrix(f64),
	b: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(b)
	x := make([]f64, n, allocator)

	// R is m x n, but we only use the leading n x n upper triangle
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
