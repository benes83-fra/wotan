package wotan_linalg

import "core:math"
import "core:mem"

// ============================================================================
// QR decomposition (Householder)
// ============================================================================
// .Unblocked applies one reflection at a time (rank-1 updates). .Blocked groups
// QR_BLOCK reflections into the compact WY form H₁⋯H_b = I − V·T·Vᵀ (LAPACK's
// dgeqrf / dlarft / dlarfb) and applies them to the trailing columns and to Q
// with matrix products. Both give the same factors up to rounding.
QR_Mode :: enum {
	Unblocked,
	Blocked,
}

// Columns per panel of the blocked QR
QR_BLOCK :: 32

// R factor only (m×n, upper triangular), via Householder reflections; Q is not
// formed
qr_decompose_thin :: proc(
	A: ^Matrix(f64),
	mode: QR_Mode = .Blocked,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	_, R := _qr_householder(A, mode, false, allocator)
	return R
}

// A = Q·R with Q m×m orthogonal and R m×n upper triangular
qr_decompose :: proc(
	A: ^Matrix(f64),
	mode: QR_Mode = .Blocked,
	allocator: mem.Allocator = context.allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	return _qr_householder(A, mode, true, allocator)
}

// qr_decompose(A, .Unblocked)
qr_decompose_unblocked :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	return _qr_householder(A, .Unblocked, true, allocator)
}

@(private = "file")
_qr_householder :: proc(
	A: ^Matrix(f64),
	mode: QR_Mode,
	want_q: bool,
	allocator: mem.Allocator,
) -> (
	Q: Matrix(f64),
	R: Matrix(f64),
) {
	m, n := A.rows, A.cols
	kmax := min(m, n)

	R = matrix_new(f64, m, n, allocator)
	copy(R.data, A.data)
	if want_q {
		Q = matrix_new(f64, m, m, allocator)
		for i in 0 ..< m {Q.data[i * m + i] = 1.0}
	}
	nb := QR_BLOCK if mode == .Blocked else 1
	for k0 := 0; k0 < kmax; k0 += nb {
		b := min(nb, kmax - k0)
		if b == 1 {
			_qr_step_unblocked(&R, &Q, k0, want_q)
		} else {
			_qr_panel_blocked(&R, &Q, k0, b, want_q)
		}
	}
	return
}

// Householder vector of column k of R (rows k..m): v (unit, length m − k) with
// (I − 2vvᵀ)x = ∓‖x‖e₁. ok = false for a zero column (no reflection).
@(private = "file")
_householder_vector :: proc(R: ^Matrix(f64), k: int, v: []f64) -> (ok: bool) {
	m, n := R.rows, R.cols
	xlen := m - k
	for i in 0 ..< xlen {v[i] = R.data[(k + i) * n + k]}
	normx := math.sqrt(dot_simd(v[:xlen], v[:xlen]))
	if normx == 0 {return false}
	v[0] += (-1.0 if v[0] < 0 else 1.0) * normx
	vnorm := math.sqrt(dot_simd(v[:xlen], v[:xlen]))
	if vnorm == 0 {return false}
	inv := 1.0 / vnorm
	for i in 0 ..< xlen {v[i] *= inv}
	return true
}

// Reflection of column k applied to columns k..c_end of R: R ← (I − 2vvᵀ)R
@(private = "file")
_apply_reflection_cols :: proc(R: ^Matrix(f64), k, c_end: int, v, w, col_buf: []f64) {
	m, n := R.rows, R.cols
	xlen := m - k
	w_len := c_end - k
	for j in 0 ..< w_len {
		for i in 0 ..< xlen {col_buf[i] = R.data[(k + i) * n + (k + j)]}
		w[j] = dot_simd(v[:xlen], col_buf[:xlen])
	}
	for i in 0 ..< xlen {
		scalar := 2.0 * v[i]
		if scalar == 0 {continue}
		row_start := (k + i) * n + k
		axpy_simd(-scalar, w[:w_len], R.data[row_start:row_start + w_len])
	}
}

// One reflection on all trailing columns and on Q (rank-1 updates)
@(private = "file")
_qr_step_unblocked :: proc(R, Q: ^Matrix(f64), k: int, want_q: bool) {
	m, n := R.rows, R.cols
	xlen := m - k
	scratch := context.temp_allocator
	v := make([]f64, m, scratch)
	w := make([]f64, n, scratch)
	col_buf := make([]f64, m, scratch)
	defer {
		delete(col_buf, scratch)
		delete(w, scratch)
		delete(v, scratch)
	}
	if !_householder_vector(R, k, v) {return}
	_apply_reflection_cols(R, k, n, v, w, col_buf)
	if !want_q {return}
	// Q ← Q(I − 2vvᵀ) on columns k..m
	for i in 0 ..< m {
		row := Q.data[i * m + k:i * m + m]
		u := dot_simd(row[:xlen], v[:xlen])
		if u != 0 {axpy_simd(-2.0 * u, v[:xlen], row)}
	}
}

// Panel of b reflections from column k0: factor the panel column by column,
// form T of the compact WY form, then update the trailing columns
// (C ← (I − V·Tᵀ·Vᵀ)·C) and Q (Q ← Q − (Q·V)·T·Vᵀ) with matrix products
@(private = "file")
_qr_panel_blocked :: proc(R, Q: ^Matrix(f64), k0, b: int, want_q: bool) {
	m, n := R.rows, R.cols
	mk := m - k0
	scratch := context.temp_allocator

	V := matrix_new(f64, mk, b, scratch) // reflector j in column j, rows k0..m
	defer matrix_free(&V)
	tau := make([]f64, b, scratch)
	v := make([]f64, mk, scratch)
	w := make([]f64, n, scratch)
	col_buf := make([]f64, mk, scratch)
	z := make([]f64, b, scratch)
	defer {
		delete(z, scratch)
		delete(col_buf, scratch)
		delete(w, scratch)
		delete(v, scratch)
		delete(tau, scratch)
	}

	// 1. panel factorization (reflections applied to the panel columns only)
	for j in 0 ..< b {
		k := k0 + j
		if !_householder_vector(R, k, v) {continue}
		tau[j] = 2
		for i in 0 ..< m - k {V.data[(j + i) * b + j] = v[i]}
		_apply_reflection_cols(R, k, k0 + b, v, w, col_buf)
	}

	// 2. T (b×b upper triangular): T[j,j] = τⱼ, T[0:j, j] = −τⱼ·T[0:j, 0:j]·(V[:, 0:j]ᵀ·vⱼ)
	T := matrix_new(f64, b, b, scratch)
	defer matrix_free(&T)
	for j in 0 ..< b {
		T.data[j * b + j] = tau[j]
		if tau[j] == 0 {continue}
		for p in 0 ..< j {
			s := 0.0
			for r in j ..< mk {s += V.data[r * b + p] * V.data[r * b + j]}
			z[p] = s
		}
		for p in 0 ..< j {
			s := 0.0
			for q in p ..< j {s += T.data[p * b + q] * z[q]}
			T.data[p * b + j] = -tau[j] * s
		}
	}

	// 3. trailing columns: C ← C − V·(Tᵀ·(Vᵀ·C))
	c0 := k0 + b
	nc := n - c0
	if nc > 0 {
		C := matrix_new(f64, mk, nc, scratch)
		defer matrix_free(&C)
		for r in 0 ..< mk {
			src := (k0 + r) * n
			copy(C.data[r * nc:(r + 1) * nc], R.data[src + c0:src + n])
		}
		W := matrix_new(f64, b, nc, scratch)
		defer matrix_free(&W)
		matmul_tn_acc_simd(&V, &C, &W)
		W2 := matrix_new(f64, b, nc, scratch) // −Tᵀ·W
		defer matrix_free(&W2)
		for i in 0 ..< b {
			row := W2.data[i * nc:(i + 1) * nc]
			for p in 0 ..< i + 1 {
				t := T.data[p * b + i]
				if t != 0 {axpy_simd(-t, W.data[p * nc:(p + 1) * nc], row)}
			}
		}
		matmul_nn_acc_simd(&V, &W2, &C)
		for r in 0 ..< mk {
			dst := (k0 + r) * n
			copy(R.data[dst + c0:dst + n], C.data[r * nc:(r + 1) * nc])
		}
	}

	// 4. Q ← Q − (Q[:, k0:m]·V)·T·Vᵀ on columns k0..m
	if want_q {
		Qs := matrix_new(f64, m, mk, scratch)
		defer matrix_free(&Qs)
		for r in 0 ..< m {copy(Qs.data[r * mk:(r + 1) * mk], Q.data[r * m + k0:r * m + m])}
		Y := matrix_new(f64, m, b, scratch)
		defer matrix_free(&Y)
		matmul_nn_acc_simd(&Qs, &V, &Y)
		Y2 := matrix_new(f64, m, b, scratch) // −Y·T
		defer matrix_free(&Y2)
		for r in 0 ..< m {
			for j in 0 ..< b {
				s := 0.0
				for p in 0 ..< j + 1 {s += Y.data[r * b + p] * T.data[p * b + j]}
				Y2.data[r * b + j] = -s
			}
		}
		matmul_nt_acc_simd(&Y2, &V, &Qs)
		for r in 0 ..< m {copy(Q.data[r * m + k0:r * m + m], Qs.data[r * mk:(r + 1) * mk])}
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
