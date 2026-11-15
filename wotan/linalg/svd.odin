package wotan_linalg

import "core:math"
import "core:mem"

// .Jacobi: one-sided Jacobi, accurate to high relative precision also for small
// singular values. .GolubReinsch: bidiagonalization + implicit QR, faster for
// larger matrices, singular values accurate to about ε·σ_max.
SVDMethod :: enum {
	Jacobi,
	GolubReinsch,
}

// ============================================================================
// SVD (Jacobi) - Optimized: dot_simd + workspace reuse (SAFE)
// ============================================================================
svd_jacobi :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	m := A.rows
	n := A.cols
	if m == 0 || n == 0 {return}

	// Initialize U = A, V = I
	U = matrix_new(f64, m, n, allocator)
	copy(U.data, A.data)

	V = matrix_new(f64, n, n, allocator)
	for i in 0 ..< n {V.data[i * n + i] = 1.0}

	// Pre-allocate workspace ONCE (reused every iteration)
	col_p := make([]f64, max(m, n), context.temp_allocator)
	col_q := make([]f64, max(m, n), context.temp_allocator)
	defer delete(col_q, context.temp_allocator)
	defer delete(col_p, context.temp_allocator)

	// Rotate a pair until its columns are orthogonal to working precision:
	// |u_p·u_q| ≤ √m·ε·‖u_p‖‖u_q‖ (the tolerance of LAPACK's dgesvj). Columns
	// of norm below ε·‖A‖_F are rounding noise of a rank-deficient A: they never
	// meet a relative test, and _svd_sort_and_complete replaces them anyway.
	max_iter := 100
	eps := math.sqrt(f64(m)) * 2.220446049250313e-16
	fro2 := dot_simd(A.data[:m * n], A.data[:m * n])
	tiny2 := 2.220446049250313e-16 * 2.220446049250313e-16 * fro2

	for iter in 0 ..< max_iter {
		changed := false

		for p in 0 ..< n {
			for q in p + 1 ..< n {
				// Gather columns p and q of U into temp buffers
				for i in 0 ..< m {
					col_p[i] = U.data[i * U.cols + p]
					col_q[i] = U.data[i * U.cols + q]
				}

				// Compute alpha, beta, gamma using SIMD dot (SAFE: read-only)
				alpha := dot_simd(col_p[0:m], col_p[0:m])
				beta := dot_simd(col_q[0:m], col_q[0:m])
				gamma := dot_simd(col_p[0:m], col_q[0:m])

				if alpha <= tiny2 || beta <= tiny2 {continue}
				if math.abs(gamma) <= eps * math.sqrt(alpha) * math.sqrt(beta) {
					continue
				}

				changed = true

				// Jacobi rotation parameters (idiomatic Odin)
				tau := (beta - alpha) / (2.0 * gamma)
				t: f64
				if tau >= 0.0 {
					t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
				} else {
					t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
				}
				c := 1.0 / math.sqrt(1.0 + t * t)
				s := c * t

				// Rotate columns p,q of U (m rows) - SCALAR for now (safe)
				rotate_pair_simd(c, s, col_p, col_q, m) // Pass full arrays + explicit len
				for i in 0 ..< m {
					U.data[i * U.cols + p] = col_p[i]
					U.data[i * U.cols + q] = col_q[i]
				}

				// Rotate columns p,q of V (n rows) - SCALAR for now (safe)
				for i in 0 ..< n {
					vp := V.data[i * V.cols + p]
					vq := V.data[i * V.cols + q]
					V.data[i * V.cols + p] = c * vp - s * vq
					V.data[i * V.cols + q] = s * vp + c * vq
				}
			}
		}

		if !changed {break}
	}

	// Compute singular values = column norms of U, normalize columns
	S = make([]f64, n, allocator)
	for j in 0 ..< n {
		norm2 := 0.0
		for i in 0 ..< m {
			uij := U.data[i * U.cols + j]
			norm2 += uij * uij
		}
		sigma := math.sqrt(norm2)
		S[j] = sigma
		if sigma > 1e-15 {
			inv := 1.0 / sigma
			for i in 0 ..< m {
				U.data[i * U.cols + j] *= inv
			}
		}
	}

	_svd_sort_and_complete(&U, S, &V)
	return
}

// Sorts the singular values descending (with their columns of U and V), and
// completes the columns of U for (numerically) zero singular values to an
// orthonormal set, so the first min(m, n) columns of U stay orthonormal for a
// rank-deficient A
@(private = "file")
_svd_sort_and_complete :: proc(U: ^Matrix(f64), S: []f64, V: ^Matrix(f64)) {
	m, n := U.rows, V.rows
	// Sort descending (selection sort)
	for i in 0 ..< n {
		max_i := i
		for j in i + 1 ..< n {
			if S[j] > S[max_i] {max_i = j}
		}
		if max_i == i {continue}
		S[i], S[max_i] = S[max_i], S[i]
		// Swap columns in U
		for r in 0 ..< m {
			U.data[r * U.cols + i], U.data[r * U.cols + max_i] =
				U.data[r * U.cols + max_i], U.data[r * U.cols + i]
		}
		// Swap columns in V
		for r in 0 ..< n {
			V.data[r * V.cols + i], V.data[r * V.cols + max_i] =
				V.data[r * V.cols + max_i], V.data[r * V.cols + i]
		}
	}

	// Columns of U for (numerically) zero singular values carry no direction:
	// complete them to an orthonormal set (Gram-Schmidt on unit vectors), so the
	// first min(m, n) columns of U stay orthonormal for rank-deficient A
	tol := f64(max(m, n)) * 2.2e-16 * (S[0] if n > 0 else 0)
	cand := make([]f64, m, context.temp_allocator)
	defer delete(cand, context.temp_allocator)
	for j in 0 ..< min(m, n) {
		if S[j] > tol && S[j] > 1e-15 {continue}
		S[j] = 0
		for e in 0 ..< m {
			for i in 0 ..< m {cand[i] = 1.0 if i == e else 0.0}
			for _ in 0 ..< 2 { 	// re-orthogonalise once for stability
				for k in 0 ..< j {
					d := 0.0
					for i in 0 ..< m {d += U.data[i * U.cols + k] * cand[i]}
					for i in 0 ..< m {cand[i] -= d * U.data[i * U.cols + k]}
				}
			}
			nrm := math.sqrt(dot_simd(cand, cand))
			if nrm > 0.5 {
				for i in 0 ..< m {U.data[i * U.cols + j] = cand[i] / nrm}
				break
			}
		}
	}
	// A wide A (m < n) has rank ≤ m: its last n − m singular values are zero
	// (left as rounding noise they would break the descending order)
	for j in min(m, n) ..< n {S[j] = 0}
}

// ============================================================================
// Symmetric Jacobi eigen (cyclic Jacobi, SIMD rotations of the eigenvectors)
// ============================================================================
// Eigenvalues (unsorted, in diagonal order) and eigenvectors (columns) of a
// symmetric matrix. Works on a copy: A is not modified. A is assumed symmetric
// (not checked); use eigh for a checked, sorted decomposition.
jacobi_eigen_symmetric :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	eigenvalues: []f64,
	eigenvectors: Matrix(f64),
) {
	n := A.rows
	if n == 0 || A.cols != n {
		panic("jacobi_eigen_symmetric: matrix must be square")
	}
	work := matrix_new(f64, n, n, allocator)
	defer matrix_free(&work)
	copy(work.data, A.data[:n * n])
	return _jacobi_eigen_inplace(&work, allocator)
}

// Jacobi on A itself: A ends up (numerically) diagonal
@(private)
_jacobi_eigen_inplace :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator,
) -> (
	eigenvalues: []f64,
	eigenvectors: Matrix(f64),
) {
	n := A.rows

	// V = I
	eigenvectors = matrix_new(f64, n, n, allocator)
	for i in 0 ..< n {eigenvectors.data[i * n + i] = 1.0}

	col_p := make([]f64, n, context.temp_allocator)
	col_q := make([]f64, n, context.temp_allocator)
	defer {
		delete(col_p, context.temp_allocator)
		delete(col_q, context.temp_allocator)
	}

	// Cyclic Jacobi: sweep all (p, q) pairs; an off-diagonal entry below rounding
	// level relative to its diagonal pair is set to zero (scale-free). Converges
	// quadratically; stops after a sweep without rotations.
	MAX_SWEEPS :: 60
	for _ in 0 ..< MAX_SWEEPS {
		rotations := 0
		for p in 0 ..< n - 1 {
			for q in p + 1 ..< n {
				apq := A.data[p * n + q]
				if apq == 0 {continue}
				app := A.data[p * n + p]
				aqq := A.data[q * n + q]
				// negligible next to the diagonal: drop it
				if abs(apq) <= 1e-17 * (abs(app) + abs(aqq)) || abs(apq) < 1e-300 {
					A.data[p * n + q] = 0
					A.data[q * n + p] = 0
					continue
				}
				rotations += 1

				tau := (aqq - app) / (2.0 * apq)
				t: f64
				if tau >= 0.0 {
					t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
				} else {
					t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
				}
				c := 1.0 / math.sqrt(1.0 + t * t)
				s := c * t

				// rows / columns p, q of A (symmetric)
				for k in 0 ..< n {
					if k == p || k == q {continue}
					up := A.data[p * n + k]
					uq := A.data[q * n + k]
					np := c * up - s * uq
					nq := s * up + c * uq
					A.data[p * n + k] = np
					A.data[k * n + p] = np
					A.data[q * n + k] = nq
					A.data[k * n + q] = nq
				}
				A.data[p * n + p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
				A.data[q * n + q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
				A.data[p * n + q] = 0.0
				A.data[q * n + p] = 0.0

				// eigenvector columns p, q
				for k in 0 ..< n {
					col_p[k] = eigenvectors.data[k * n + p]
					col_q[k] = eigenvectors.data[k * n + q]
				}
				rotate_pair_simd(c, s, col_p, col_q, n)
				for k in 0 ..< n {
					eigenvectors.data[k * n + p] = col_p[k]
					eigenvectors.data[k * n + q] = col_q[k]
				}
			}
		}
		if rotations == 0 {break}
	}

	eigenvalues = make([]f64, n, allocator)
	for i in 0 ..< n {eigenvalues[i] = A.data[i * n + i]}
	return
}

// ============================================================================
// Golub-Kahan-Reinsch SVD
// ============================================================================
// Householder bidiagonalization followed by implicitly shifted QR sweeps on the
// bidiagonal (Golub & Reinsch 1970, the EISPACK / LINPACK algorithm). Same
// shapes and ordering as svd_jacobi. Faster than the Jacobi SVD for larger
// matrices; singular values are accurate to about ε·σ_max (absolute), while
// svd_jacobi also gets small singular values to high relative accuracy.
svd_golub_reinsch :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	m := A.rows
	n := A.cols
	if m == 0 || n == 0 {return}

	U = matrix_new(f64, m, n, allocator)
	copy(U.data, A.data)
	V = matrix_new(f64, n, n, allocator)
	S = make([]f64, n, allocator)
	rv1 := make([]f64, n, context.temp_allocator)
	defer delete(rv1, context.temp_allocator)
	u := U.data
	v := V.data
	w := S

	// 1. Householder reduction to bidiagonal form: diagonal in w, superdiagonal in rv1
	g, scale, anorm := 0.0, 0.0, 0.0
	l := 0
	for i in 0 ..< n {
		l = i + 1
		rv1[i] = scale * g
		g, scale = 0, 0
		s := 0.0
		if i < m {
			for k in i ..< m {scale += abs(u[k * n + i])}
			if scale != 0 {
				for k in i ..< m {
					u[k * n + i] /= scale
					s += u[k * n + i] * u[k * n + i]
				}
				f := u[i * n + i]
				g = -math.copy_sign(math.sqrt(s), f)
				h := f * g - s
				u[i * n + i] = f - g
				for j in l ..< n {
					s = 0
					for k in i ..< m {s += u[k * n + i] * u[k * n + j]}
					f = s / h
					for k in i ..< m {u[k * n + j] += f * u[k * n + i]}
				}
				for k in i ..< m {u[k * n + i] *= scale}
			}
		}
		w[i] = scale * g
		g, scale, s = 0, 0, 0
		if i < m && i != n - 1 {
			for k in l ..< n {scale += abs(u[i * n + k])}
			if scale != 0 {
				for k in l ..< n {
					u[i * n + k] /= scale
					s += u[i * n + k] * u[i * n + k]
				}
				f := u[i * n + l]
				g = -math.copy_sign(math.sqrt(s), f)
				h := f * g - s
				u[i * n + l] = f - g
				for k in l ..< n {rv1[k] = u[i * n + k] / h}
				for j in l ..< m {
					s = 0
					for k in l ..< n {s += u[j * n + k] * u[i * n + k]}
					for k in l ..< n {u[j * n + k] += s * rv1[k]}
				}
				for k in l ..< n {u[i * n + k] *= scale}
			}
		}
		anorm = max(anorm, abs(w[i]) + abs(rv1[i]))
	}

	// 2. accumulate the right-hand transformations in V
	for i := n - 1; i >= 0; i -= 1 {
		if i < n - 1 {
			if g != 0 {
				// double division avoids a possible underflow
				for j in l ..< n {v[j * n + i] = (u[i * n + j] / u[i * n + l]) / g}
				for j in l ..< n {
					s := 0.0
					for k in l ..< n {s += u[i * n + k] * v[k * n + j]}
					for k in l ..< n {v[k * n + j] += s * v[k * n + i]}
				}
			}
			for j in l ..< n {
				v[i * n + j] = 0
				v[j * n + i] = 0
			}
		}
		v[i * n + i] = 1
		g = rv1[i]
		l = i
	}

	// 3. accumulate the left-hand transformations in U
	for i := min(m, n) - 1; i >= 0; i -= 1 {
		l = i + 1
		g = w[i]
		for j in l ..< n {u[i * n + j] = 0}
		if g != 0 {
			g = 1 / g
			for j in l ..< n {
				s := 0.0
				for k in l ..< m {s += u[k * n + i] * u[k * n + j]}
				f := (s / u[i * n + i]) * g
				for k in i ..< m {u[k * n + j] += f * u[k * n + i]}
			}
			for j in i ..< m {u[j * n + i] *= g}
		} else {
			for j in i ..< m {u[j * n + i] = 0}
		}
		u[i * n + i] += 1
	}

	// 4. diagonalize the bidiagonal form: implicit-shift QR sweeps, one
	// singular value at a time from the bottom
	MAX_ITS :: 75
	for k := n - 1; k >= 0; k -= 1 {
		for its in 1 ..= MAX_ITS {
			// test for splitting (rv1[0] is always zero)
			split := false
			nm := 0
			for l = k; l >= 0; l -= 1 {
				nm = l - 1
				if abs(rv1[l]) + anorm == anorm {
					split = true
					break
				}
				if abs(w[nm]) + anorm == anorm {break}
			}
			if !split {
				// cancel rv1[l] when w[l - 1] is negligible
				c, s := 0.0, 1.0
				for i in l ..= k {
					f := s * rv1[i]
					rv1[i] = c * rv1[i]
					if abs(f) + anorm == anorm {break}
					g = w[i]
					h := math.hypot(f, g)
					w[i] = h
					h = 1 / h
					c = g * h
					s = -f * h
					for j in 0 ..< m {
						y := u[j * n + nm]
						z := u[j * n + i]
						u[j * n + nm] = y * c + z * s
						u[j * n + i] = z * c - y * s
					}
				}
			}
			z := w[k]
			if l == k { 	// converged: make the singular value non-negative
				if z < 0 {
					w[k] = -z
					for j in 0 ..< n {v[j * n + k] = -v[j * n + k]}
				}
				break
			}
			if its == MAX_ITS {
				// no convergence (not seen in practice): use the Jacobi SVD
				matrix_free(&U)
				matrix_free(&V)
				delete(S, allocator)
				return svd_jacobi(A, allocator)
			}
			// shift from the bottom 2×2 minor
			x := w[l]
			nm = k - 1
			y := w[nm]
			g = rv1[nm]
			h := rv1[k]
			f := ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
			g = math.hypot(f, 1)
			f = ((x - z) * (x + z) + h * ((y / (f + math.copy_sign(g, f))) - h)) / x
			// next QR transformation
			c, s := 1.0, 1.0
			for j in l ..= nm {
				i := j + 1
				g = rv1[i]
				y = w[i]
				h = s * g
				g = c * g
				z = math.hypot(f, h)
				rv1[j] = z
				c = f / z
				s = h / z
				f = x * c + g * s
				g = g * c - x * s
				h = y * s
				y *= c
				for jj in 0 ..< n {
					xv := v[jj * n + j]
					zv := v[jj * n + i]
					v[jj * n + j] = xv * c + zv * s
					v[jj * n + i] = zv * c - xv * s
				}
				z = math.hypot(f, h)
				w[j] = z
				if z != 0 {
					z = 1 / z
					c = f * z
					s = h * z
				}
				f = c * g + s * y
				x = c * y - s * g
				for jj in 0 ..< m {
					yu := u[jj * n + j]
					zu := u[jj * n + i]
					u[jj * n + j] = yu * c + zu * s
					u[jj * n + i] = zu * c - yu * s
				}
			}
			rv1[l] = 0
			rv1[k] = f
			w[k] = x
		}
	}

	_svd_sort_and_complete(&U, S, &V)
	return
}

// ============================================================================
// Thin SVD
// ============================================================================
svd_thin :: proc(
	A: ^Matrix(f64),
	method: SVDMethod = .Jacobi,
	allocator: mem.Allocator = context.allocator,
) -> (
	U_t: Matrix(f64),
	S_t: []f64,
	V_t: Matrix(f64),
) {
	U_full, S_full, V_full := svd(A, method, allocator)
	defer matrix_free(&U_full) // full factors are scratch; the truncated copies are returned
	defer delete(S_full, allocator)
	defer matrix_free(&V_full)
	m, n := U_full.rows, U_full.cols

	S_max := 0.0
	for i in 0 ..< n {if S_full[i] > S_max {S_max = S_full[i]}}
	tol := f64(max(m, n)) * 1e-12 * S_max
	r := 0
	for i in 0 ..< n {if S_full[i] > tol {r += 1} else {break}}

	if r == 0 {
		U_t = matrix_new(f64, m, 0, allocator)
		S_t = make([]f64, 0, allocator)
		V_t = matrix_new(f64, n, 0, allocator)
		return
	}

	U_t = matrix_new(f64, m, r, allocator)
	V_t = matrix_new(f64, n, r, allocator)
	S_t = make([]f64, r, allocator)

	for i in 0 ..< r {S_t[i] = S_full[i]}
	for i in 0 ..< m {
		for j in 0 ..< r {U_t.data[i * U_t.cols + j] = U_full.data[i * U_full.cols + j]}
	}
	for i in 0 ..< n {
		for j in 0 ..< r {V_t.data[i * V_t.cols + j] = V_full.data[i * V_full.cols + j]}
	}
	return
}

// ============================================================================
// Dispatcher
// ============================================================================
svd :: proc(
	A: ^Matrix(f64),
	method: SVDMethod = .Jacobi,
	allocator: mem.Allocator = context.allocator,
) -> (
	U: Matrix(f64),
	S: []f64,
	V: Matrix(f64),
) {
	switch method {
	case .Jacobi:
		return svd_jacobi(A, allocator)
	case .GolubReinsch:
		return svd_golub_reinsch(A, allocator)
	}
	return svd_jacobi(A, allocator)
}

// Moore-Penrose pseudo-inverse A⁺ (n × m for an m × n A) from the Jacobi SVD:
// A⁺ = V diag(1/σ) Uᵀ over the singular values σ > rtol · σ_max; the others
// count as zero. rtol < 0 (default) uses max(m, n) · 1e-12, the cutoff of
// lstsq(.SVD). For a non-singular square A this is A⁻¹. An empty A gives an
// empty n × m result.
pinv :: proc(
	A: ^Matrix(f64),
	rtol: f64 = -1,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	m, n := A.rows, A.cols
	if m == 0 || n == 0 {return matrix_new(f64, n, m, allocator)}
	if m < n { 	// wide: pinv(A) = pinv(Aᵀ)ᵀ
		At := matrix_transpose(A, allocator)
		defer matrix_free(&At)
		P := pinv(&At, rtol, allocator)
		defer matrix_free(&P)
		return matrix_transpose(&P, allocator)
	}

	U, S, V := svd_jacobi(A, allocator) // U m × n, V n × n, S descending
	defer matrix_free(&U)
	defer delete(S, allocator)
	defer matrix_free(&V)
	tol := (rtol if rtol >= 0 else f64(max(m, n)) * 1e-12) * S[0]

	out := matrix_new(f64, n, m, allocator)
	for k in 0 ..< n {
		if !(S[k] > tol) {continue}
		inv := 1.0 / S[k]
		for i in 0 ..< n {
			vik := V.data[i * V.cols + k] * inv
			if vik == 0 {continue}
			for j in 0 ..< m {out.data[i * m + j] += vik * U.data[j * U.cols + k]}
		}
	}
	return out
}
