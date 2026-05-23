package analytics

import w "../core"
import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"


// ============================================================================
// Helper: Convert [][]f64 (row-major) → l.Matrix(f64) (contiguous, row-major)
// ============================================================================
_to_linalg_matrix :: proc(
	data: [][]f64,
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	if len(data) == 0 {
		return l.Matrix(f64){}
	}
	rows := len(data)
	cols := len(data[0])

	m := l.matrix_new(f64, rows, cols, allocator)
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			m.data[i * cols + j] = data[i][j]
		}
	}
	return m
}

// ============================================================================
// Helper: Convert l.Matrix(f64) → [][]f64 (row-major, for API compatibility)
// ============================================================================
_from_linalg_matrix :: proc(
	m: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	if m.rows == 0 || m.cols == 0 {
		return [][]f64{}
	}
	rows := m.rows
	cols := m.cols

	out := make([][]f64, rows, allocator)
	for i in 0 ..< rows {
		out[i] = make([]f64, cols, allocator)
		for j in 0 ..< cols {
			out[i][j] = m.data[i * cols + j]
		}
	}
	return out
}
PCAResult :: struct {
	eigenvalues:  []f64,
	eigenvectors: [][]f64,
}
JacobiResult :: struct {
	eigenvalues:  []f64,
	eigenvectors: [][]f64, // columns are eigenvectors
}


pca_dataframe :: proc(df: ^w.DataFrame, cols: []string, allocator: mem.Allocator) -> PCAResult {

	data := extract_numeric_matrix(df, cols, allocator)
	cov := covariance_matrix(data, allocator)
	return pca_from_cov(cov, allocator)
}

rolling_pca :: proc(
	df: ^w.DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> []PCAResult {

	cov_df := rolling_cov_matrix(df, cols, window, min_periods, allocator)
	defer w.destroy_dataframe(&cov_df)
	results := make([]PCAResult, cov_df.rows, allocator)

	for r in 0 ..< cov_df.rows {
		cov := extract_cov_matrix_row(&cov_df, cols, r, allocator)
		results[r] = pca_from_cov(cov, allocator)
	}

	return results
}


argsort_descending :: proc(values: []f64, allocator: mem.Allocator = context.allocator) -> []int {
	n := len(values)
	idx := make([]int, n, allocator)
	for i in 0 ..< n do idx[i] = i

	// simple selection sort (n is small for PCA)
	for i in 0 ..< n {
		max_i := i
		for j in i + 1 ..< n {
			if values[idx[j]] > values[idx[max_i]] {
				max_i = j
			}
		}
		idx[i], idx[max_i] = idx[max_i], idx[i]
	}

	return idx
}

extract_numeric_matrix :: proc(
	df: ^w.DataFrame,
	cols: []string,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	rows := df.rows
	cols_n := len(cols)

	out := make([][]f64, rows, allocator)
	for r in 0 ..< rows {
		out[r] = make([]f64, cols_n, allocator)
	}

	for col_name, c_idx in cols {
		col := w.column(df, col_name)

		for r in 0 ..< rows {
			v, is_null := w.column_at_float(col, r)
			if is_null {
				out[r][c_idx] = 0 // or NaN, but 0 is fine for covariance
			} else {
				out[r][c_idx] = v
			}
		}
	}

	return out
}
covariance_matrix :: proc(data: [][]f64, allocator: mem.Allocator = context.allocator) -> [][]f64 {
	rows := len(data)
	if rows == 0 {
		return [][]f64{}
	}
	cols := len(data[0])
	if cols == 0 {
		return [][]f64{}
	}

	// Step 1: Center the data (subtract column means)
	// We do this in the new linalg format for efficiency
	X := _to_linalg_matrix(data, context.temp_allocator)
	defer l.matrix_free(&X)

	// Compute column means
	means := make([]f64, cols, context.temp_allocator)
	for j in 0 ..< cols {
		sum := 0.0
		for i in 0 ..< rows {
			sum += X.data[i * cols + j]
		}
		means[j] = sum / f64(rows)
	}

	// Center X in-place: X[i,j] -= means[j]
	for j in 0 ..< cols {
		for i in 0 ..< rows {
			X.data[i * cols + j] -= means[j]
		}
	}

	// Step 2: Compute covariance = (Xᵀ X) / (n - 1) using optimized xtx_simd
	cov_linalg := l.xtx_simd(&X, allocator) // This is the SIMD-optimized version!
	defer l.matrix_free(&cov_linalg)

	// Scale by 1/(n-1)
	scale := 1.0 / f64(rows - 1)
	for i in 0 ..< cols * cols {
		cov_linalg.data[i] *= scale
	}

	// Step 3: Convert back to [][]f64 for API compatibility
	return _from_linalg_matrix(&cov_linalg, allocator)
}
extract_cov_matrix_row :: proc(
	cov_df: ^w.DataFrame,
	cols: []string,
	r: int,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {

	n := len(cols)
	out := make([][]f64, n, allocator)
	for i in 0 ..< n {
		out[i] = make([]f64, n, allocator)
	}

	col_row := w.column(cov_df, "row")
	col_i := w.column(cov_df, "col_i")
	col_j := w.column(cov_df, "col_j")
	col_cov := w.column(cov_df, "cov")

	for idx in 0 ..< cov_df.rows {
		row_val, _ := w.column_at_int(col_row, idx)
		if row_val != r {
			continue
		}

		i_name, _ := w.column_at_string(col_i, idx)
		j_name, _ := w.column_at_string(col_j, idx)

		// find indices
		i_idx := index_of_string(cols, i_name)
		j_idx := index_of_string(cols, j_name)

		v, is_null := w.column_at_float(col_cov, idx)
		if is_null {
			out[i_idx][j_idx] = 0
		} else {
			out[i_idx][j_idx] = v
		}
	}

	return out
}

pca_explained_variance_ratio :: proc(pca: PCAResult) -> []f64 {
	total := 0.0
	for v in pca.eigenvalues do total += v

	out := make([]f64, len(pca.eigenvalues))
	for v, i in pca.eigenvalues {
		out[i] = v / total
	}
	return out
}
pca_transform :: proc(data: [][]f64, pca: PCAResult) -> [][]f64 {
	rows := len(data)
	if rows == 0 {
		return [][]f64{}
	}
	comps := len(pca.eigenvectors)
	if comps == 0 {
		return [][]f64{}
	}
	dims := len(data[0])
	if dims == 0 {
		return [][]f64{}
	}

	// Convert data to linalg format for optimized matvec
	data_linalg := _to_linalg_matrix(data, context.temp_allocator)
	defer l.matrix_free(&data_linalg)

	// Convert eigenvectors to linalg format (comps × dims)
	// Note: pca.eigenvectors is [][]f64 where each row is a component
	evecs_linalg := l.matrix_new(f64, comps, dims, context.temp_allocator)
	defer l.matrix_free(&evecs_linalg)
	for i in 0 ..< comps {
		for j in 0 ..< dims {
			evecs_linalg.data[i * dims + j] = pca.eigenvectors[i][j]
		}
	}

	// Compute scores = data @ eigenvectorsᵀ
	// Since eigenvectors are rows in pca.eigenvectors, we need to transpose
	// scores[i,k] = sum_j data[i,j] * evecs[k,j] = (data @ evecsᵀ)[i,k]

	// Option 1: Use matmul_dyn_simd (data: rows×dims) × (evecsᵀ: dims×comps)
	// But we need to transpose evecs_linalg first

	// Option 2 (simpler): Loop over components and use matvec_dyn_simd
	scores := make([][]f64, rows, context.allocator)
	for r in 0 ..< rows {
		scores[r] = make([]f64, comps, context.allocator)
	}

	// For each component (row in eigenvectors), compute data @ componentᵀ
	for c in 0 ..< comps {
		// Extract component c as a vector (length dims)
		component := make([]f64, dims, context.temp_allocator)
		for j in 0 ..< dims {
			component[j] = pca.eigenvectors[c][j]
		}

		// Compute scores[:,c] = data @ component
		// This is a matrix-vector product: (rows×dims) × (dims) → (rows)
		for r in 0 ..< rows {
			row := data_linalg.data[r * dims:r * dims + dims]
			scores[r][c] = l.dot_simd(row, component)
		}
		delete(component, context.temp_allocator)
	}

	return scores
}
pca_inverse_transform :: proc(scores: [][]f64, pca: PCAResult) -> [][]f64 {
	rows := len(scores)
	if rows == 0 {
		return [][]f64{}
	}
	comps := len(pca.eigenvectors)
	if comps == 0 {
		return [][]f64{}
	}
	dims := len(pca.eigenvectors[0])
	if dims == 0 {
		return [][]f64{}
	}

	// Convert scores to linalg format for optimized matvec
	scores_linalg := _to_linalg_matrix(scores, context.temp_allocator)
	defer l.matrix_free(&scores_linalg)

	// Convert eigenvectors to linalg format (comps × dims)
	evecs_linalg := l.matrix_new(f64, comps, dims, context.temp_allocator)
	defer l.matrix_free(&evecs_linalg)
	for i in 0 ..< comps {
		for j in 0 ..< dims {
			evecs_linalg.data[i * dims + j] = pca.eigenvectors[i][j]
		}
	}

	// Compute reconstruction = scores @ eigenvectors
	// reconstruction[i,d] = sum_c scores[i,c] * evecs[c,d]

	out := make([][]f64, rows, context.allocator)
	for r in 0 ..< rows {
		out[r] = make([]f64, dims, context.allocator)
	}

	// For each original dimension, compute scores @ eigenvector_column
	for d in 0 ..< dims {
		// Extract column d of eigenvectors as a vector (length comps)
		evec_col := make([]f64, comps, context.temp_allocator)
		for c in 0 ..< comps {
			evec_col[c] = pca.eigenvectors[c][d]
		}

		// Compute out[:,d] = scores @ evec_col
		for r in 0 ..< rows {
			row := scores_linalg.data[r * comps:r * comps + comps]
			out[r][d] = l.dot_simd(row, evec_col)
		}
		delete(evec_col, context.temp_allocator)
	}

	return out
}


//a little convience helper

print_matrix :: proc(mat: [][]f64) {
	rows := len(mat)
	if rows == 0 {
		fmt.println("[]")
		return
	}

	cols := len(mat[0])

	for r in 0 ..< rows {
		fmt.print("[ ")
		for c in 0 ..< cols {
			fmt.printf("%8.4f ", mat[r][c])
		}
		fmt.println("]")
	}
}
index_of_string :: proc(list: []string, s: string) -> int {
	for v, i in list {
		if v == s {
			return i
		}
	}
	return -1
}

// pca_from_cov :: proc(cov: [][]f64, allocator: mem.Allocator = context.allocator) -> PCAResult {
// 	jr := jacobi_eigen_symmetric(cov, allocator)
// 	// defer destroy_matrix(jr.eigenvectors)
// 	// defer delete(jr.eigenvalues)
// 	idx := argsort_descending(jr.eigenvalues)
// 	n := len(idx)
// 	defer delete(idx)
// 	sorted_vals := make([]f64, n)
// 	sorted_vecs := make([][]f64, n)

// 	for i, j in idx {
// 		sorted_vals[i] = jr.eigenvalues[j]
// 		// sorted_vecs[i] = jr.eigenvectors[j][:]
// 		sorted_vecs[i] = make([]f64, len(jr.eigenvectors[j]))
// 		copy(sorted_vecs[i], jr.eigenvectors[j])
// 	}

// 	return PCAResult{eigenvalues = sorted_vals, eigenvectors = sorted_vecs}
// }
pca_from_cov :: proc(cov: [][]f64, allocator: mem.Allocator = context.allocator) -> PCAResult {
	// Convert covariance to linalg format for optimized eigensolver
	cov_linalg := _to_linalg_matrix(cov, context.temp_allocator)
	defer l.matrix_free(&cov_linalg)

	// FIX: Symmetrize covariance to ensure exact symmetry for eigh
	// cov = (cov + covᵀ) / 2
	cols := cov_linalg.cols
	for i in 0 ..< cols {
		for j in i + 1 ..< cols {
			avg := (cov_linalg.data[i * cols + j] + cov_linalg.data[j * cols + i]) * 0.5
			cov_linalg.data[i * cols + j] = avg
			cov_linalg.data[j * cols + i] = avg
		}
	}

	// Use optimized symmetric eigensolver (SIMD-accelerated Jacobi)
	// Returns eigenvalues (sorted ascending by default) and eigenvectors (columns)
	eigenvalues, eigenvectors_linalg := l.eigh(&cov_linalg, .Ascending, allocator)
	defer {
		if len(eigenvalues) > 0 {delete(eigenvalues, allocator)}
		if eigenvectors_linalg.data != nil {l.matrix_free(&eigenvectors_linalg)}
	}

	n := len(eigenvalues)
	if n == 0 {
		return PCAResult{}
	}

	// Sort eigenvalues descending (and reorder eigenvectors accordingly)
	// Note: l.eigh returns ascending, so we reverse the order
	sorted_vals := make([]f64, n, allocator)
	sorted_vecs := make([][]f64, n, allocator)

	for i in 0 ..< n {
		// Reverse index: ascending → descending
		src_idx := n - 1 - i
		sorted_vals[i] = eigenvalues[src_idx]

		// Extract eigenvector column src_idx from eigenvectors_linalg
		// and store as row i in sorted_vecs (API compatibility)
		sorted_vecs[i] = make([]f64, n, allocator)
		for j in 0 ..< n {
			// eigenvectors_linalg is column-major: col=src_idx, row=j
			sorted_vecs[i][j] = eigenvectors_linalg.data[j * n + src_idx]
		}
	}

	return PCAResult{eigenvalues = sorted_vals, eigenvectors = sorted_vecs}
}
// Example of how to properly delete your [][]f64 "matrices"
destroy_matrix :: proc(mat: [][]f64) {
	for row in mat {
		delete(row)
	}
	delete(mat)
}
destroy_pca_result :: proc(res: PCAResult) {
	// 1. Delete each individual eigenvector row
	for vec in res.eigenvectors {
		delete(vec)
	}

	// 2. Delete the outer slice holding the eigenvector pointers
	delete(res.eigenvectors)

	// 3. Delete the eigenvalue slice
	delete(res.eigenvalues)
}
