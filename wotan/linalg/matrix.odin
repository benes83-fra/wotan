package wotan_linalg

import w "../core"
import "base:intrinsics"
import "core:mem"
import "core:strings"
// ------------------------------------------------------------
// Generic dynamic matrix
// ------------------------------------------------------------

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
		m.data = make([]T, rows * cols, allocator)
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
matrix_from_df :: proc(
	df: ^w.DataFrame,
	cols: []string,
	allocator: mem.Allocator = context.allocator,
) -> Matrix(f64) {
	m := matrix_new(f64, df.rows, len(cols), allocator)

	for name, j in cols {
		idx, ok := df.name_to_index[name]
		if !ok {
			res := strings.concatenate({"matrix_from_df: unknown column name: ", name})
			panic(res)
		}

		col := &df.columns[idx]
		// assume f64 backing for .Float
		base := cast([^]f64)col.data

		// respect view offset
		start := col.offset
		for i in 0 ..< df.rows {
			m.data[i * m.cols + j] = base[start + i]
		}
	}

	return m
}

vector_from_df :: proc(
	df: ^w.DataFrame,
	colname: string,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	idx, ok := df.name_to_index[colname]
	if !ok {
		res := strings.concatenate({"vector_from_df: unknown column name: ", colname})
		panic(res)
	}

	col := &df.columns[idx]
	base := cast([^]f64)col.data

	out := make([]f64, df.rows, allocator)
	start := col.offset

	for i in 0 ..< df.rows {
		out[i] = base[start + i]
	}

	return out
}
