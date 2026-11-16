package wotan_linalg

import w "../core"
import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"
// ------------------------------------------------------------
// Generic dynamic matrix
// ------------------------------------------------------------

// Contiguous row-major storage: element (i, j) is data[i * cols + j]
Matrix :: struct($T: typeid) {
	rows:      int,
	cols:      int,
	data:      []T,
	allocator: mem.Allocator,
}

// ------------------------------------------------------------
// Fixed-size matrices (Odin auto-SIMDs these)
// ------------------------------------------------------------

matmul :: proc(a, b: $M/matrix[$R, $C]f64) -> M {
	return a * b
}

matvec :: proc(a: $M/matrix[$R, $C]f64, x: [C]f64) -> [R]f64 {
	return a * x
}

transpose :: proc(m: $M/matrix[$R, $C]f64) -> matrix[C, R]f64 {
	return intrinsics.transpose(m)
}

// ------------------------------------------------------------
// Dynamic matrix core
// ------------------------------------------------------------

matrix_new :: proc(
	$T: typeid,
	rows, cols: int,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(T) {
	m: Matrix(T)
	m.rows = rows
	m.cols = cols
	m.allocator = allocator

	if rows * cols > 0 {
		err: mem.Allocator_Error
		m.data, err = make([]T, rows * cols, allocator)
		// Fail here instead of returning an empty matrix that crashes far away
		// with a misleading length/slice error (e.g. when the machine runs out of memory)
		if err != nil {
			panic(fmt.tprintf("matrix_new: failed to allocate %d x %d (%v)", rows, cols, err))
		}
	}

	return m
}

matrix_free :: proc(m: ^Matrix($T)) {
	if m.data != nil {
		delete(m.data, m.allocator)
	}
	m.data = nil
	m.rows, m.cols = 0, 0
}

matrix_at :: proc(m: ^Matrix($T), r, c: int) -> ^T {
	return &m.data[r * m.cols + c]
}

// ------------------------------------------------------------
// Scalar dot + dynamic matvec/matmul (f64)
// ------------------------------------------------------------

dot :: proc(a, b: []f64) -> f64 {
	if len(a) != len(b) {
		panic("dot: length mismatch")
	}

	s: f64 = 0.0
	for i in 0 ..< len(a) {
		s += a[i] * b[i]
	}
	return s
}

matvec_dyn :: proc(
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
		y[r] = dot(row, x)
	}

	return y
}

matmul_dyn :: proc(
	a, b: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	if a.cols != b.rows {
		panic("matmul_dyn: dimension mismatch")
	}

	out := matrix_new(f64, a.rows, b.cols, allocator)

	for i in 0 ..< a.rows {
		for j in 0 ..< b.cols {
			acc := 0.0
			for k in 0 ..< a.cols {
				acc += a.data[i * a.cols + k] * b.data[k * b.cols + j]
			}
			out.data[i * out.cols + j] = acc
		}
	}

	return out
}
// Physical storage and row offset of a numeric DataFrame column (views
// resolved); panics on an unknown name or a column that is not Int or Float
@(private = "file")
_df_numeric_source :: proc(
	df: ^w.DataFrame,
	name: string,
	who: string,
) -> (
	base: ^w.Column,
	offset: int,
) {
	idx, ok := df.name_to_index[name]
	if !ok {
		panic(strings.concatenate({who, ": unknown column name: ", name}))
	}
	base = &df.columns[idx]
	for base.is_view {
		offset += base.offset
		base = base.orig
	}
	if base.type != .Float && base.type != .Int {
		panic(strings.concatenate({who, ": column is not Int or Float: ", name}))
	}
	return
}

// Cell r of a physical numeric column as f64; a null cell is NaN
@(private = "file")
_df_numeric_at :: #force_inline proc(base: ^w.Column, r: int) -> f64 {
	if base.nulls != nil && r < len(base.nulls) && base.nulls[r] {
		return math.nan_f64()
	}
	if base.type == .Float {
		return (cast([^]f64)base.data)[r]
	}
	return f64((cast([^]int)base.data)[r])
}

// Named Int / Float columns of df as an n × len(cols) matrix (row-major, in
// the order given). A null cell becomes NaN, like pandas' df.to_numpy().
matrix_from_df :: proc(
	df: ^w.DataFrame,
	cols: []string,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	m := matrix_new(f64, df.rows, len(cols), allocator)
	for name, j in cols {
		base, offset := _df_numeric_source(df, name, "matrix_from_df")
		for i in 0 ..< df.rows {m.data[i * m.cols + j] = _df_numeric_at(base, offset + i)}
	}
	return m
}

// One Int / Float column of df as a []f64; a null cell becomes NaN
vector_from_df :: proc(
	df: ^w.DataFrame,
	colname: string,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	base, offset := _df_numeric_source(df, colname, "vector_from_df")
	out := make([]f64, df.rows, allocator)
	for i in 0 ..< df.rows {out[i] = _df_numeric_at(base, offset + i)}
	return out
}


covariance :: proc(X: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> Matrix(f64) {
	n := X.rows
	p := X.cols
	if n <= 1 do panic("covariance: need at least 2 rows")

	// 1. Compute column means
	means := make([]f64, p, context.temp_allocator)
	defer delete(means, context.temp_allocator)
	for j := 0; j < p; j += 1 {
		sum := 0.0
		for i := 0; i < n; i += 1 {
			sum += X.data[i * X.cols + j]
		}
		means[j] = sum / f64(n)
	}

	// 2. Build centered copy (or do it in-place if allowed)
	Xc := matrix_new(f64, n, p, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < p; j += 1 {
			Xc.data[i * Xc.cols + j] = X.data[i * X.cols + j] - means[j]
		}
	}

	// 3. Sigma = (1/(n-1)) * Xcᵀ Xc
	S := xtx_simd(&Xc, allocator)
	scale := 1.0 / f64(n - 1)
	for i := 0; i < S.rows * S.cols; i += 1 {
		S.data[i] *= scale
	}
	matrix_free(&Xc)
	return S
}


xtx :: proc(X: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> Matrix(f64) {
	p := X.cols
	n := X.rows
	XtX := matrix_new(f64, p, p, allocator)

	for j := 0; j < p; j += 1 {
		col_j := make([]f64, n, context.temp_allocator)
		for i := 0; i < n; i += 1 {
			col_j[i] = X.data[i * X.cols + j]
		}

		for k := j; k < p; k += 1 {
			col_k := make([]f64, n, context.temp_allocator)
			for i := 0; i < n; i += 1 {
				col_k[i] = X.data[i * X.cols + k]
			}

			v := dot_simd(col_j, col_k)
			XtX.data[j * XtX.cols + k] = v
			XtX.data[k * XtX.cols + j] = v

			delete(col_k, context.temp_allocator)
		}

		delete(col_j, context.temp_allocator)
	}

	return XtX
}

xty :: proc(X: ^Matrix(f64), y: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	if X.rows != len(y) do panic("xty: dimension mismatch")

	p := X.cols
	n := X.rows
	Xty := make([]f64, p, allocator)

	for j := 0; j < p; j += 1 {
		col_j := make([]f64, n, context.temp_allocator)
		for i := 0; i < n; i += 1 {
			col_j[i] = X.data[i * X.cols + j]
		}
		Xty[j] = dot_simd(col_j, y)
		delete(col_j, context.temp_allocator)
	}

	return Xty
}

correlation :: proc(X: ^Matrix(f64), allocator: mem.Allocator = context.allocator) -> Matrix(f64) {
	// 1. Compute covariance matrix
	C := covariance(X, allocator)
	p := C.cols

	// 2. Allocate correlation matrix
	R := matrix_new(f64, p, p, allocator)

	// 3. Precompute standard deviations
	sd := make([]f64, p, context.temp_allocator)
	defer delete(sd, context.temp_allocator)
	for i := 0; i < p; i += 1 {
		v := C.data[i * p + i]
		if v <= 0 {
			panic("correlation: non-positive variance")
		}
		sd[i] = math.sqrt(v)
	}

	// 4. Compute correlation entries
	for i := 0; i < p; i += 1 {
		for j := 0; j < p; j += 1 {
			R.data[i * p + j] = C.data[i * p + j] / (sd[i] * sd[j])
		}
	}
	matrix_free(&C)
	return R
}
// ============================================================================
// Kronecker Product: C = A ⊗ B
// If A is (m×n) and B is (p×q), then C is (m*p × n*q)
// C[i*p + ii, j*q + jj] = A[i,j] * B[ii,jj]
// ============================================================================
kron :: proc(
	A: ^Matrix(f64),
	B: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	m, n := A.rows, A.cols
	p, q := B.rows, B.cols

	// Result dimensions: (m*p) × (n*q)
	C := matrix_new(f64, m * p, n * q, allocator)

	// Compute C block-by-block: C[i,j] block = A[i,j] * B
	for i in 0 ..< m {
		for j in 0 ..< n {
			a_ij := A.data[i * A.cols + j]
			if a_ij == 0.0 {continue} 	// Skip zero blocks (sparse-friendly)

			// Block position in C
			row_start := i * p
			col_start := j * q

			// Fill block: C[row_start:row_start+p, col_start:col_start+q] = a_ij * B
			for ii in 0 ..< p {
				row_C := row_start + ii
				row_B := ii * B.cols
				row_C_start := row_C * C.cols + col_start

				// SIMD-friendly: scale entire row of B by a_ij
				for jj in 0 ..< q {
					C.data[row_C_start + jj] = a_ij * B.data[row_B + jj]
				}
			}
		}
	}

	return C
}
// ============================================================================
// Matrix Transpose: out = inᵀ
// out[i,j] = in[j,i]
// ============================================================================
matrix_transpose :: proc(
	input: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	if input.rows * input.cols > 4096 {
		return matrix_transpose_blocked(input, allocator)
	}
	rows, cols := input.rows, input.cols
	out := matrix_new(f64, cols, rows, allocator)

	// Simple transpose: out[j,i] = in[i,j]
	// For better cache performance on large matrices, consider blocking:
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			out.data[j * rows + i] = input.data[i * cols + j]
		}
	}

	return out
}


// ============================================================================
// Matrix Transpose (blocked version for better cache performance)
// Uses TILE_BASE/TILE_AVX blocking for cache efficiency
// ============================================================================
matrix_transpose_blocked :: proc(
	input: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	rows, cols := input.rows, input.cols
	out := matrix_new(f64, cols, rows, allocator)

	tile := tile_for_matmul() // 8 or 16 depending on AVX

	// Blocked transpose: process tile×tile blocks
	for ii := 0; ii < rows; ii += tile {
		for jj := 0; jj < cols; jj += tile {
			i_end := min(ii + tile, rows)
			j_end := min(jj + tile, cols)

			for i in ii ..< i_end {
				for j in jj ..< j_end {
					out.data[j * rows + i] = input.data[i * cols + j]
				}
			}
		}
	}

	return out
}


matrix_from_flat :: proc(
	data: []f64,
	rows, cols: int,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	m := matrix_new(f64, rows, cols, allocator)
	expected_len := rows * cols
	if len(data) < expected_len {
		panic(fmt.aprintf("matrix_from_flat: data too short (%v < %v)", len(data), expected_len))
	}
	copy(m.data, data[0:expected_len])
	return m
}


// ============================================================================
// SIMD-Optimized Matrix Inversion
// Computes A^-1 by solving A * x_i = e_i using LU decomposition.
// A must be square and non-empty (panics otherwise). A singular or numerically
// singular matrix (an exactly zero pivot, or rcond1 < ε) gives ok = false and an
// empty 0 × 0 matrix, which is safe to free. Same contract as mat_inverse_lu.
// ============================================================================
matrix_inverse :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	Ainv: Matrix(f64),
	ok: bool,
) {
	n := A.rows
	if n == 0 || A.cols != n {
		panic("matrix_inverse: matrix must be square and non-empty")
	}

	// 1. LU decomposition (A = P * L * U); factor and pivots are scratch
	LU, piv, _, lu_ok := lu_decompose(A, allocator)
	defer matrix_free(&LU)
	defer delete(piv, allocator)
	if !lu_ok {
		Ainv = matrix_new(f64, 0, 0, allocator)
		return
	}

	// 2. Solve A * x_i = e_i for each column i
	Ainv = matrix_new(f64, n, n, allocator)
	e := make([]f64, n, context.temp_allocator)
	defer delete(e, context.temp_allocator)
	for i in 0 ..< n {
		for j in 0 ..< n {e[j] = 0.0}
		e[i] = 1.0
		x := lu_solve_simd(&LU, piv, e, allocator)
		// column i of Ainv (row-major: indices j * n + i)
		for j in 0 ..< n {Ainv.data[j * n + i] = x[j]}
		delete(x, allocator)
	}

	ok = _inverse_accept(A, &Ainv, allocator)
	return
}
