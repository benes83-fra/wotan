package core

import "core:fmt"
import "core:math"
import "core:mem"

PCAResult :: struct {
	eigenvalues:  []f64,
	eigenvectors: [][]f64,
}
JacobiResult :: struct {
	eigenvalues:  []f64,
	eigenvectors: [][]f64, // columns are eigenvectors
}

// Jacobi eigenvalue algorithm for real symmetric A (n x n)
// Returns eigenvalues (length n) and eigenvectors (n x n, column-major in V)
jacobi_eigen_symmetric :: proc(
	A_in: [][]f64,
	allocator: mem.Allocator = context.allocator,
) -> JacobiResult {
	n := len(A_in)
	if n == 0 {
		return JacobiResult{}
	}

	// Copy A_in into working matrix A
	A := make([][]f64, n, allocator)
	for i in 0 ..< n {
		A[i] = make([]f64, n, allocator)
		for j in 0 ..< n {
			A[i][j] = A_in[i][j]
		}
	}

	// Initialize V as identity
	V := make([][]f64, n, allocator)
	for i in 0 ..< n {
		V[i] = make([]f64, n, allocator)
		for j in 0 ..< n {
			V[i][j] = 0.0
		}
		V[i][i] = 1.0
	}

	max_iter := 50
	eps := 1e-12

	for iter in 0 ..< max_iter {
		// Find largest off-diagonal element
		max_val := 0.0
		p := 0
		q := 1
		for i in 0 ..< n {
			for j in i + 1 ..< n {
				aij := math.abs(A[i][j])
				if aij > max_val {
					max_val = aij
					p = i
					q = j
				}
			}
		}

		if max_val < eps {
			break // converged
		}

		app := A[p][p]
		aqq := A[q][q]
		apq := A[p][q]

		// Compute rotation angle
		tau := (aqq - app) / (2.0 * apq)
		t := 0.0
		if tau >= 0 {
			t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
		} else {
			t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
		}
		c := 1.0 / math.sqrt(1.0 + t * t)
		s := t * c

		// Update A
		// Diagonal entries
		app_new := app - t * apq
		aqq_new := aqq + t * apq
		A[p][p] = app_new
		A[q][q] = aqq_new
		A[p][q] = 0.0
		A[q][p] = 0.0

		// Off-diagonal rows/cols
		for k in 0 ..< n {
			if k == p || k == q {
				continue
			}
			akp := A[k][p]
			akq := A[k][q]
			A[k][p] = c * akp - s * akq
			A[p][k] = A[k][p]
			A[k][q] = s * akp + c * akq
			A[q][k] = A[k][q]
		}

		// Update eigenvector matrix V
		for k in 0 ..< n {
			vkp := V[k][p]
			vkq := V[k][q]
			V[k][p] = c * vkp - s * vkq
			V[k][q] = s * vkp + c * vkq
		}
	}

	// Extract eigenvalues from diagonal of A
	eigenvalues := make([]f64, n, allocator)
	for i in 0 ..< n {
		eigenvalues[i] = A[i][i]
	}

	// We currently have eigenvectors as columns of V.
	// Your PCA code expects eigenvectors as rows (each component = one row).
	// So we’ll transpose into that layout.
	eigenvectors := make([][]f64, n, allocator)
	for i in 0 ..< n {
		eigenvectors[i] = make([]f64, n, allocator)
		for j in 0 ..< n {
			eigenvectors[i][j] = V[j][i]
		}
	}

	return JacobiResult{eigenvalues = eigenvalues, eigenvectors = eigenvectors}
}
// Computes SVD of a symmetric matrix A (NxN)
// A = V * diag(S) * Vᵀ
svd_symmetric :: proc(A: [][]f64) -> (S: []f64, V: [][]f64) {
	n := len(A)
	if n == 0 {
		return {}, {}
	}

	// Copy A into V (will become eigenvectors)
	V = make([][]f64, n)
	for i in 0 ..< n {
		V[i] = make([]f64, n)
		for j in 0 ..< n {
			V[i][j] = A[i][j]
		}
	}

	// Initialize S as diagonal of A
	S = make([]f64, n)
	for i in 0 ..< n {
		S[i] = A[i][i]
	}

	// Jacobi sweeps
	max_iter := 50
	for iter in 0 ..< max_iter {
		changed := false

		for p in 0 ..< n {
			for q in p + 1 ..< n {
				if math.abs(V[p][q]) < 1e-12 {
					continue
				}

				changed = true

				phi := 0.5 * math.atan2(2 * V[p][q], V[q][q] - V[p][p])
				c := math.cos(phi)
				s := math.sin(phi)

				// Rotate rows/cols p and q
				for k in 0 ..< n {
					vpk := V[p][k]
					vqk := V[q][k]
					V[p][k] = c * vpk - s * vqk
					V[q][k] = s * vpk + c * vqk
				}

				for k in 0 ..< n {
					vkp := V[k][p]
					vkq := V[k][q]
					V[k][p] = c * vkp - s * vkq
					V[k][q] = s * vkp + c * vkq
				}
			}
		}

		if !changed {
			break
		}
	}

	// Extract singular values from diagonal
	for i in 0 ..< n {
		S[i] = V[i][i]
	}

	return S, V
}


pca_dataframe :: proc(df: ^DataFrame, cols: []string, allocator: mem.Allocator) -> PCAResult {

	data := extract_numeric_matrix(df, cols, allocator)
	cov := covariance_matrix(data, allocator)
	return pca_from_cov(cov, allocator)
}

rolling_pca :: proc(
	df: ^DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> []PCAResult {

	cov_df := rolling_cov_matrix(df, cols, window, min_periods, allocator)
	defer destroy_dataframe(&cov_df)
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
	df: ^DataFrame,
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
		col := column(df, col_name)

		for r in 0 ..< rows {
			v, is_null := column_at_float(col, r)
			if is_null {
				out[r][c_idx] = 0 // or NaN, but 0 is fine for covariance
			} else {
				out[r][c_idx] = v
			}
		}
	}

	return out
}
covariance_matrix :: proc(data: [][]f64, allocator: mem.Allocator) -> [][]f64 {
	rows := len(data)
	if rows == 0 {
		return [][]f64{}
	}

	cols := len(data[0])
	cov := make([][]f64, cols, allocator)
	for i in 0 ..< cols {
		cov[i] = make([]f64, cols, allocator)
	}

	// compute means
	means := make([]f64, cols, allocator)
	for r in 0 ..< rows {
		for c in 0 ..< cols {
			means[c] += data[r][c]
		}
	}
	for c in 0 ..< cols {
		means[c] /= f64(rows)
	}

	// compute covariance
	for i in 0 ..< cols {
		for j in i ..< cols {
			sum: f64 = 0
			for r in 0 ..< rows {
				sum += (data[r][i] - means[i]) * (data[r][j] - means[j])
			}
			v := sum / f64(rows - 1)
			cov[i][j] = v
			cov[j][i] = v
		}
	}

	return cov
}
extract_cov_matrix_row :: proc(
	cov_df: ^DataFrame,
	cols: []string,
	r: int,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {

	n := len(cols)
	out := make([][]f64, n, allocator)
	for i in 0 ..< n {
		out[i] = make([]f64, n, allocator)
	}

	col_row := column(cov_df, "row")
	col_i := column(cov_df, "col_i")
	col_j := column(cov_df, "col_j")
	col_cov := column(cov_df, "cov")

	for idx in 0 ..< cov_df.rows {
		row_val, _ := column_at_int(col_row, idx)
		if row_val != r {
			continue
		}

		i_name, _ := column_at_string(col_i, idx)
		j_name, _ := column_at_string(col_j, idx)

		// find indices
		i_idx := index_of_string(cols, i_name)
		j_idx := index_of_string(cols, j_name)

		v, is_null := column_at_float(col_cov, idx)
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
	comps := len(pca.eigenvectors)

	out := make([][]f64, rows)
	for r in 0 ..< rows {
		out[r] = make([]f64, comps)
		for c in 0 ..< comps {
			sum := 0.0
			for k in 0 ..< len(data[r]) {
				sum += data[r][k] * pca.eigenvectors[c][k]
			}
			out[r][c] = sum
		}
	}
	return out
}
pca_inverse_transform :: proc(scores: [][]f64, pca: PCAResult) -> [][]f64 {
	rows := len(scores)
	dims := len(pca.eigenvectors[0])

	out := make([][]f64, rows)
	for r in 0 ..< rows {
		out[r] = make([]f64, dims)
		for d in 0 ..< dims {
			sum := 0.0
			for c in 0 ..< len(scores[r]) {
				sum += scores[r][c] * pca.eigenvectors[c][d]
			}
			out[r][d] = sum
		}
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
	jr := jacobi_eigen_symmetric(cov, allocator)

	idx := argsort_descending(jr.eigenvalues, allocator) // Use temp for sorting indices


	n := len(idx)
	sorted_vals := make([]f64, n, allocator)
	sorted_vecs := make([][]f64, n, allocator)

	for new_i, old_i in idx {
		sorted_vals[new_i] = jr.eigenvalues[old_i]
		// Transfer ownership of the row instead of 'make' + 'copy'
		sorted_vecs[new_i] = jr.eigenvectors[old_i]
		// Null out the old reference so we don't accidentally use it
		jr.eigenvectors[old_i] = nil
	}

	// Now free the intermediate slices/containers from Jacobi

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
