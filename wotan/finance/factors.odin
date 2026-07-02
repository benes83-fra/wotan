package finance

import ana "../analytics"
import w "../core"
import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Factor Analysis Result
// ============================================================================
FactorAnalysisResult :: struct {
	loadings:      [][]f64, // [n_factors × n_variables] factor loadings
	communalities: []f64, // [n_variables] variance explained by factors
	uniqueness:    []f64, // [n_variables] unique variance (1 - communality)
	eigenvalues:   []f64, // [n_factors] eigenvalues of reduced correlation
	n_factors:     int,
	converged:     bool,
	n_iterations:  int,
}

RotationMethod :: enum {
	None,
	Varimax,
}

// ============================================================================
// Factor Analysis via Principal Axis Method
// ============================================================================
factor_analysis :: proc(
	data: [][]f64,
	n_factors: int,
	max_iter: int = 50,
	tol: f64 = 1e-6,
	allocator: mem.Allocator = context.allocator,
) -> FactorAnalysisResult {
	n_factors := n_factors
	n_obs := len(data)
	if n_obs == 0 {
		return FactorAnalysisResult{}
	}
	n_vars := len(data[0])
	if n_vars == 0 {
		return FactorAnalysisResult{}
	}
	if n_factors > n_vars {
		n_factors = n_vars
	}

	// Step 1: Compute correlation matrix (standardized)
	corr := _correlation_matrix(data, allocator)
	defer _destroy_matrix(corr, allocator) // ✅ FIXED: Pass allocator

	// Step 2: Initial communalities (squared multiple correlations)
	pca := ana.pca_from_cov(corr, allocator)
	defer _destroy_pca_result(pca, allocator) // ✅ FIXED: Use custom destroy with allocator

	communalities := make([]f64, n_vars, allocator)
	for i in 0 ..< n_vars {
		if i < len(pca.eigenvalues) {
			communalities[i] = min(0.99, pca.eigenvalues[0] / f64(n_vars))
		} else {
			communalities[i] = 0.5
		}
	}

	// Step 3: Iterative refinement
	converged := false
	n_iterations := 0
	loadings: [][]f64

	for iter in 0 ..< max_iter {
		n_iterations = iter + 1

		// Replace diagonal with communalities
		reduced_corr := _copy_matrix(corr, allocator)
		for i in 0 ..< n_vars {
			reduced_corr[i][i] = communalities[i]
		}

		// Extract factors via PCA on reduced correlation
		factor_pca := ana.pca_from_cov(reduced_corr, allocator)
		_destroy_matrix(reduced_corr, allocator) // ✅ FIXED: Pass allocator

		// Extract first n_factors eigenvectors as loadings
		loadings = make([][]f64, n_factors, allocator)
		for f in 0 ..< n_factors {
			loadings[f] = make([]f64, n_vars, allocator)
			if f < len(factor_pca.eigenvectors) {
				for v in 0 ..< n_vars {
					scale := math.sqrt(max(0.0, factor_pca.eigenvalues[f]))
					loadings[f][v] = factor_pca.eigenvectors[f][v] * scale
				}
			}
		}
		_destroy_pca_result(factor_pca, allocator) // ✅ FIXED: Use custom destroy with allocator

		// Update communalities: sum of squared loadings for each variable
		new_communalities := make([]f64, n_vars, allocator)
		for v in 0 ..< n_vars {
			sum_sq := 0.0
			for f in 0 ..< n_factors {
				sum_sq += loadings[f][v] * loadings[f][v]
			}
			new_communalities[v] = min(1.0, max(0.0, sum_sq))
		}

		// Check convergence
		max_change := 0.0
		for v in 0 ..< n_vars {
			change := math.abs(new_communalities[v] - communalities[v])
			if change > max_change {
				max_change = change
			}
		}

		// Clean up old communalities
		delete(communalities, allocator)
		communalities = new_communalities

		if max_change < tol {
			converged = true
			break
		}
	}

	// Compute uniqueness
	uniqueness := make([]f64, n_vars, allocator)
	for v in 0 ..< n_vars {
		uniqueness[v] = 1.0 - communalities[v]
	}

	// Extract eigenvalues from final loadings
	eigenvalues := make([]f64, n_factors, allocator)
	for f in 0 ..< n_factors {
		sum_sq := 0.0
		for v in 0 ..< n_vars {
			sum_sq += loadings[f][v] * loadings[f][v]
		}
		eigenvalues[f] = sum_sq
	}

	return FactorAnalysisResult {
		loadings = loadings,
		communalities = communalities,
		uniqueness = uniqueness,
		eigenvalues = eigenvalues,
		n_factors = n_factors,
		converged = converged,
		n_iterations = n_iterations,
	}
}

// ============================================================================
// Varimax Rotation
// ============================================================================
varimax_rotation :: proc(
	loadings: [][]f64,
	max_iter: int = 100,
	tol: f64 = 1e-6,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	n_factors := len(loadings)
	if n_factors < 2 {
		rotated := make([][]f64, n_factors, allocator)
		for f in 0 ..< n_factors {
			rotated[f] = make([]f64, len(loadings[f]), allocator)
			copy(rotated[f], loadings[f])
		}
		return rotated
	}

	n_vars := len(loadings[0])

	A := make([][]f64, n_vars, allocator)
	for v in 0 ..< n_vars {
		A[v] = make([]f64, n_factors, allocator)
		for f in 0 ..< n_factors {
			A[v][f] = loadings[f][v]
		}
	}

	for iter in 0 ..< max_iter {
		delta := 0.0

		for i in 0 ..< n_factors - 1 {
			for j in i + 1 ..< n_factors {
				u := make([]f64, n_vars, allocator)
				v := make([]f64, n_vars, allocator)

				for k in 0 ..< n_vars {
					u[k] = A[k][i] * A[k][i] - A[k][j] * A[k][j]
					v[k] = 2.0 * A[k][i] * A[k][j]
				}

				sum_u := 0.0
				sum_v := 0.0
				for k in 0 ..< n_vars {
					sum_u += u[k]
					sum_v += v[k]
				}

				A_val := 0.0
				B_val := 0.0
				for k in 0 ..< n_vars {
					A_val += u[k] - sum_u / f64(n_vars)
					B_val += v[k] - sum_v / f64(n_vars)
				}

				if math.abs(A_val) < 1e-10 && math.abs(B_val) < 1e-10 {
					delete(u, allocator)
					delete(v, allocator)
					continue
				}

				phi := math.atan2(B_val, A_val) / 4.0
				cos_phi := math.cos(phi)
				sin_phi := math.sin(phi)

				for k in 0 ..< n_vars {
					old_i := A[k][i]
					old_j := A[k][j]
					A[k][i] = old_i * cos_phi - old_j * sin_phi
					A[k][j] = old_i * sin_phi + old_j * cos_phi
				}

				delta += math.abs(sin_phi) + math.abs(1.0 - cos_phi)

				delete(u, allocator)
				delete(v, allocator)
			}
		}

		if delta < tol {
			break
		}
	}

	rotated := make([][]f64, n_factors, allocator)
	for f in 0 ..< n_factors {
		rotated[f] = make([]f64, n_vars, allocator)
		for v in 0 ..< n_vars {
			rotated[f][v] = A[v][f]
		}
	}

	for v in 0 ..< n_vars {
		delete(A[v], allocator)
	}
	delete(A, allocator)

	return rotated
}

// ============================================================================
// Factor Scoring
// ============================================================================
compute_factor_scores :: proc(
	data: [][]f64,
	fa: ^FactorAnalysisResult,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	n_obs := len(data)
	if n_obs == 0 {
		return [][]f64{}
	}
	n_vars := len(data[0])
	n_factors := fa.n_factors

	std_data := _standardize_data(data, allocator)
	defer _destroy_matrix(std_data, allocator) // ✅ FIXED: Pass allocator

	scores := make([][]f64, n_obs, allocator)
	for obs in 0 ..< n_obs {
		scores[obs] = make([]f64, n_factors, allocator)
		for f in 0 ..< n_factors {
			sum := 0.0
			for v in 0 ..< n_vars {
				sum += std_data[obs][v] * fa.loadings[f][v]
			}
			var_sum := 0.0
			for v in 0 ..< n_vars {
				var_sum += fa.loadings[f][v] * fa.loadings[f][v]
			}
			if var_sum > 1e-10 {
				scores[obs][f] = sum / var_sum
			}
		}
	}

	return scores
}

// ============================================================================
// Finance-Specific: Market Factor Extraction
// ============================================================================
extract_market_factor :: proc(
	returns_data: [][]f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	[]f64,
	f64,
) {
	corr := _correlation_matrix(returns_data, allocator)
	defer _destroy_matrix(corr, allocator) // ✅ FIXED: Pass allocator

	pca := ana.pca_from_cov(corr, allocator)
	defer _destroy_pca_result(pca, allocator) // ✅ FIXED: Use custom destroy with allocator

	if len(pca.eigenvalues) == 0 {
		return []f64{}, 0.0
	}

	n_obs := len(returns_data)
	n_vars := len(returns_data[0])

	std_data := _standardize_data(returns_data, allocator)
	defer _destroy_matrix(std_data, allocator) // ✅ FIXED: Pass allocator

	market_scores := make([]f64, n_obs, allocator)
	for obs in 0 ..< n_obs {
		sum := 0.0
		for v in 0 ..< n_vars {
			sum += std_data[obs][v] * pca.eigenvectors[0][v]
		}
		market_scores[obs] = sum
	}

	total_var := 0.0
	for ev in pca.eigenvalues {
		total_var += ev
	}
	variance_explained := 0.0
	if total_var > 1e-10 {
		variance_explained = pca.eigenvalues[0] / total_var
	}

	return market_scores, variance_explained
}

// ============================================================================
// Risk Decomposition
// ============================================================================
RiskDecomposition :: struct {
	factor_variance:   []f64,
	idiosyncratic_var: []f64,
	total_variance:    []f64,
	factor_exposure:   [][]f64,
}

decompose_risk :: proc(
	returns_data: [][]f64,
	n_factors: int,
	allocator: mem.Allocator = context.allocator,
) -> RiskDecomposition {
	fa := factor_analysis(returns_data, n_factors, 50, 1e-6, allocator)

	n_assets := len(returns_data[0])

	total_variance := make([]f64, n_assets, allocator)
	for v in 0 ..< n_assets {
		total_variance[v] = fa.communalities[v] + fa.uniqueness[v]
	}

	return RiskDecomposition {
		factor_variance = fa.communalities,
		idiosyncratic_var = fa.uniqueness,
		total_variance = total_variance,
		factor_exposure = fa.loadings,
	}
}

// ============================================================================
// Helper Functions
// ============================================================================
_correlation_matrix :: proc(
	data: [][]f64,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	n_obs := len(data)
	if n_obs == 0 {
		return [][]f64{}
	}
	n_vars := len(data[0])

	std_data := _standardize_data(data, allocator)

	corr := make([][]f64, n_vars, allocator)
	for i in 0 ..< n_vars {
		corr[i] = make([]f64, n_vars, allocator)
		for j in 0 ..< n_vars {
			sum := 0.0
			for obs in 0 ..< n_obs {
				sum += std_data[obs][i] * std_data[obs][j]
			}
			corr[i][j] = sum / f64(n_obs - 1)
		}
	}

	_destroy_matrix(std_data, allocator) // ✅ FIXED: Pass allocator
	return corr
}

_standardize_data :: proc(data: [][]f64, allocator: mem.Allocator = context.allocator) -> [][]f64 {
	n_obs := len(data)
	if n_obs == 0 {
		return [][]f64{}
	}
	n_vars := len(data[0])

	means := make([]f64, n_vars, allocator)
	stds := make([]f64, n_vars, allocator)

	for v in 0 ..< n_vars {
		sum := 0.0
		for obs in 0 ..< n_obs {
			sum += data[obs][v]
		}
		means[v] = sum / f64(n_obs)
	}

	for v in 0 ..< n_vars {
		sum_sq := 0.0
		for obs in 0 ..< n_obs {
			diff := data[obs][v] - means[v]
			sum_sq += diff * diff
		}
		stds[v] = math.sqrt(sum_sq / f64(n_obs - 1))
		if stds[v] < 1e-10 {
			stds[v] = 1.0
		}
	}

	std_data := make([][]f64, n_obs, allocator)
	for obs in 0 ..< n_obs {
		std_data[obs] = make([]f64, n_vars, allocator)
		for v in 0 ..< n_vars {
			std_data[obs][v] = (data[obs][v] - means[v]) / stds[v]
		}
	}

	delete(means, allocator)
	delete(stds, allocator)

	return std_data
}

_copy_matrix :: proc(mat: [][]f64, allocator: mem.Allocator = context.allocator) -> [][]f64 {
	rows := len(mat)
	if rows == 0 {
		return [][]f64{}
	}
	cols := len(mat[0])

	out := make([][]f64, rows, allocator)
	for r in 0 ..< rows {
		out[r] = make([]f64, cols, allocator)
		copy(out[r], mat[r])
	}
	return out
}

// ✅ FIXED: Added allocator parameter
_destroy_matrix :: proc(mat: [][]f64, allocator: mem.Allocator) {
	for row in mat {
		delete(row, allocator)
	}
	delete(mat, allocator)
}

// ✅ NEW: Custom destroy function that respects the allocator
_destroy_pca_result :: proc(res: ana.PCAResult, allocator: mem.Allocator) {
	for vec in res.eigenvectors {
		delete(vec, allocator)
	}
	delete(res.eigenvectors, allocator)
	delete(res.eigenvalues, allocator)
}

// ============================================================================
// DataFrame Integration
// ============================================================================
factor_analysis_df :: proc(
	df: ^w.DataFrame,
	cols: []string,
	n_factors: int,
	allocator: mem.Allocator = context.allocator,
) -> FactorAnalysisResult {
	data := ana.extract_numeric_matrix(df, cols, allocator)
	defer ana.destroy_matrix(data)
	return factor_analysis(data, n_factors, 50, 1e-6, allocator)
}

compute_factor_scores_df :: proc(
	df: ^w.DataFrame,
	cols: []string,
	fa: ^FactorAnalysisResult,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	data := ana.extract_numeric_matrix(df, cols, allocator)
	defer ana.destroy_matrix(data)
	return compute_factor_scores(data, fa, allocator)
}
