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
	loadings:      [][]f64,
	communalities: []f64,
	uniqueness:    []f64,
	eigenvalues:   []f64,
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
// Principal axis factoring of the correlation matrix (psych::fa(fm = "pa")):
// start from the squared multiple correlations 1 - 1/(R⁻¹)_ii, replace the
// diagonal of R by the communalities, take the top n_factors eigenpairs
// (loadings = eigenvector·sqrt(eigenvalue)) and iterate until the communalities
// change by less than tol. Communalities are clamped to [0, 1] (Heywood cases).
// loadings[f][v]; eigenvalues are the sums of squared loadings per factor
factor_analysis :: proc(
	data: [][]f64,
	n_factors: int,
	max_iter: int = 1000,
	tol: f64 = 1e-9,
	allocator: mem.Allocator = context.allocator,
) -> FactorAnalysisResult {
	n_factors := n_factors
	n_obs := len(data)
	if n_obs == 0 {return FactorAnalysisResult{}}
	n_vars := len(data[0])
	if n_vars == 0 {return FactorAnalysisResult{}}
	if n_factors > n_vars {n_factors = n_vars}

	// Step 1: Convert to linalg Matrix and compute correlation matrix
	data_flat := make([]f64, n_obs * n_vars, allocator)
	idx := 0
	for r in 0 ..< n_obs {
		for c in 0 ..< n_vars {
			data_flat[idx] = data[r][c]
			idx += 1
		}
	}
	data_mat := l.matrix_from_flat(data_flat, n_obs, n_vars, allocator)
	defer l.matrix_free(&data_mat)
	delete(data_flat, allocator)

	corr_mat := l.correlation(&data_mat, allocator)
	defer l.matrix_free(&corr_mat)

	// Convert correlation matrix back to [][]f64 for PCA
	corr := _matrix_to_slice(&corr_mat, allocator)
	defer _destroy_matrix(corr, allocator)

	// Step 2: Initial communalities: squared multiple correlations
	communalities := make([]f64, n_vars, allocator)
	{
		R := l.matrix_new(f64, n_vars, n_vars, context.temp_allocator)
		defer l.matrix_free(&R)
		copy(R.data, corr_mat.data)
		if l.cholesky_try(&R) {
			e := make([]f64, n_vars, context.temp_allocator)
			defer delete(e, context.temp_allocator)
			for i in 0 ..< n_vars {
				// (R⁻¹)_ii = ‖L⁻¹ e_i‖²
				for k in 0 ..< n_vars {e[k] = 0}
				e[i] = 1
				z := l.forward_substitute_simd(&R, e, context.temp_allocator)
				communalities[i] = math.clamp(1.0 - 1.0 / l.dot_simd(z, z), 0.0, 1.0)
				delete(z, context.temp_allocator)
			}
		} else {
			// singular R: fall back to the largest absolute correlation
			for i in 0 ..< n_vars {
				for j in 0 ..< n_vars {
					if i != j {communalities[i] = max(communalities[i], math.abs(corr[i][j]))}
				}
			}
		}
	}

	// Step 3: Iterative refinement
	converged := false
	n_iterations := 0
	loadings: [][]f64

	for iter in 0 ..< max_iter {
		n_iterations = iter + 1

		reduced_corr := _copy_matrix(corr, allocator)
		for i in 0 ..< n_vars {
			reduced_corr[i][i] = communalities[i]
		}

		factor_pca := ana.pca_from_cov(reduced_corr, allocator)
		_destroy_matrix(reduced_corr, allocator)
		if iter > 0 {
			_destroy_matrix(loadings, allocator)
		}
		loadings = make([][]f64, n_factors, allocator)
		for f in 0 ..< n_factors {
			loadings[f] = make([]f64, n_vars, allocator)
			if f < len(factor_pca.eigenvectors) {
				scale := math.sqrt(max(0.0, factor_pca.eigenvalues[f]))
				for v in 0 ..< n_vars {
					loadings[f][v] = factor_pca.eigenvectors[f][v] * scale
				}
			}
		}
		_destroy_pca_result(factor_pca, allocator)

		new_communalities := make([]f64, n_vars, allocator)
		for v in 0 ..< n_vars {
			sum_sq := 0.0
			for f in 0 ..< n_factors {
				sum_sq += loadings[f][v] * loadings[f][v]
			}
			new_communalities[v] = min(1.0, max(0.0, sum_sq))
		}

		max_change := 0.0
		for v in 0 ..< n_vars {
			change := math.abs(new_communalities[v] - communalities[v])
			if change > max_change {max_change = change}
		}

		delete(communalities, allocator)
		communalities = new_communalities

		if max_change < tol {
			converged = true
			break
		}
	}

	uniqueness := make([]f64, n_vars, allocator)
	for v in 0 ..< n_vars {uniqueness[v] = 1.0 - communalities[v]}

	eigenvalues := make([]f64, n_factors, allocator)
	for f in 0 ..< n_factors {
		eigenvalues[f] = l.dot_simd(loadings[f], loadings[f])
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
// Varimax Rotation (SIMD OPTIMIZED)
// ============================================================================
// Kaiser's varimax by pairwise planar rotations: each pair of factors is turned
// by the angle that maximises the varimax criterion for that pair, sweeping
// until the rotations fall below tol. With normalize (Kaiser normalisation, the
// default of R's varimax) each variable's loadings are scaled to unit length
// during the rotation. Reaches the same maximum as R's SVD algorithm
varimax_rotation :: proc(
	loadings: [][]f64,
	max_iter: int = 100,
	tol: f64 = 1e-6,
	allocator: mem.Allocator = context.allocator,
	normalize: bool = true,
) -> [][]f64 {
	n_factors := len(loadings)
	if n_factors < 2 {return _copy_matrix(loadings, allocator)}
	n_vars := len(loadings[0])

	L := _copy_matrix(loadings, allocator)
	row_norm := make([]f64, n_vars, context.temp_allocator)
	defer delete(row_norm, context.temp_allocator)
	for r in 0 ..< n_vars {
		row_norm[r] = 1.0
		if normalize {
			ss := 0.0
			for f in 0 ..< n_factors {ss += L[f][r] * L[f][r]}
			if ss > 0 {row_norm[r] = math.sqrt(ss)}
			for f in 0 ..< n_factors {L[f][r] /= row_norm[r]}
		}
	}

	u := make([]f64, n_vars, allocator)
	v := make([]f64, n_vars, allocator)
	Li_sq := make([]f64, n_vars, allocator)
	Lj_sq := make([]f64, n_vars, allocator)
	Li_Lj := make([]f64, n_vars, allocator)
	defer {
		delete(u, allocator)
		delete(v, allocator)
		delete(Li_sq, allocator)
		delete(Lj_sq, allocator)
		delete(Li_Lj, allocator)
	}

	for iter in 0 ..< max_iter {
		delta := 0.0

		for i in 0 ..< n_factors - 1 {
			for j in i + 1 ..< n_factors {
				Li := L[i]
				Lj := L[j]

				l.vec_mul_simd(Li, Li, Li_sq)
				l.vec_mul_simd(Lj, Lj, Lj_sq)
				l.vec_mul_simd(Li, Lj, Li_Lj)

				l.vec_sub_simd(Li_sq, Lj_sq, u)
				l.vec_scale_simd(Li_Lj, 2.0, v)

				X := l.sum_simd(u)
				Y := l.sum_simd(v)

				sum_u2 := l.dot_simd(u, u)
				sum_v2 := l.dot_simd(v, v)
				sum_uv := l.dot_simd(u, v)

				n := f64(n_vars)
				A_val := (sum_u2 - sum_v2) - (X * X - Y * Y) / n
				B_val := 2.0 * sum_uv - (2.0 * X * Y) / n

				if math.abs(A_val) < 1e-10 && math.abs(B_val) < 1e-10 {
					continue
				}

				phi := math.atan2(B_val, A_val) / 4.0
				cos_phi := math.cos(phi)
				sin_phi := math.sin(phi)

				// Li' = c Li + s Lj, Lj' = -s Li + c Lj (rotate_pair_simd turns by -s)
				l.rotate_pair_simd(cos_phi, -sin_phi, Li, Lj, n_vars)

				delta += math.abs(sin_phi) + math.abs(1.0 - cos_phi)
			}
		}

		if delta < tol {break}
	}

	if normalize {
		for f in 0 ..< n_factors {
			for r in 0 ..< n_vars {L[f][r] *= row_norm[r]}
		}
	}
	return L
}

// ============================================================================
// Factor Scoring (SIMD Optimized)
// ============================================================================
// Regression (Thurstone) factor scores F = Z R⁻¹ Λ, with Z the data standardised
// by the sample (n - 1) standard deviations and R their correlation matrix (the
// default of psych::factor.scores and factor_analyzer). Also valid for rotated
// loadings (put them in fa.loadings). scores[obs][f]
compute_factor_scores :: proc(
	data: [][]f64,
	fa: ^FactorAnalysisResult,
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	n_obs := len(data)
	if n_obs == 0 {return [][]f64{}}
	n_vars := len(data[0])
	n_factors := fa.n_factors

	std_data := _standardize_data(data, allocator)
	defer _destroy_matrix(std_data, allocator)

	// R = ZᵀZ / (n - 1), factored once; W = R⁻¹ Λ one factor at a time
	R := l.matrix_new(f64, n_vars, n_vars, context.temp_allocator)
	defer l.matrix_free(&R)
	for row in std_data {
		for i in 0 ..< n_vars {
			for j in 0 ..= i {R.data[i * n_vars + j] += row[i] * row[j]}
		}
	}
	for i in 0 ..< n_vars {
		for j in 0 ..= i {
			v := R.data[i * n_vars + j] / f64(n_obs - 1)
			R.data[i * n_vars + j] = v
			R.data[j * n_vars + i] = v
		}
	}
	if !l.cholesky_try(&R) {
		panic("compute_factor_scores: correlation matrix is singular")
	}
	W := make([][]f64, n_factors, context.temp_allocator)
	defer {
		for col in W {delete(col, context.temp_allocator)}
		delete(W, context.temp_allocator)
	}
	for f in 0 ..< n_factors {
		z := l.forward_substitute_simd(&R, fa.loadings[f], context.temp_allocator)
		W[f] = l.backward_substitute_simd(&R, z, context.temp_allocator)
		delete(z, context.temp_allocator)
	}

	scores := make([][]f64, n_obs, allocator)
	for obs in 0 ..< n_obs {
		scores[obs] = make([]f64, n_factors, allocator)
		for f in 0 ..< n_factors {
			scores[obs][f] = l.dot_simd(std_data[obs], W[f])
		}
	}
	return scores
}

// ============================================================================
// Market Factor Extraction (SIMD Optimized)
// ============================================================================
// First principal component of the correlation matrix: scores of the
// standardised returns on it, signed so that it loads positively on the average
// asset, and the fraction of the total variance it explains
extract_market_factor :: proc(
	returns_data: [][]f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	[]f64,
	f64,
) {
	if len(returns_data) == 0 || len(returns_data[0]) == 0 {return []f64{}, 0.0}
	data_flat := make([]f64, len(returns_data) * len(returns_data[0]), allocator)
	idx := 0
	for r in 0 ..< len(returns_data) {
		for c in 0 ..< len(returns_data[0]) {
			data_flat[idx] = returns_data[r][c]
			idx += 1
		}
	}
	data_mat := l.matrix_from_flat(data_flat, len(returns_data), len(returns_data[0]), allocator)
	defer l.matrix_free(&data_mat)
	delete(data_flat, allocator)

	corr_mat := l.correlation(&data_mat, allocator)
	defer l.matrix_free(&corr_mat)

	corr := _matrix_to_slice(&corr_mat, allocator)
	defer _destroy_matrix(corr, allocator)

	pca := ana.pca_from_cov(corr, allocator)
	defer _destroy_pca_result(pca, allocator)

	if len(pca.eigenvalues) == 0 {return []f64{}, 0.0}

	n_obs := len(returns_data)
	std_data := _standardize_data(returns_data, allocator)
	defer _destroy_matrix(std_data, allocator)

	market_scores := make([]f64, n_obs, allocator)
	ev0 := pca.eigenvectors[0]
	sign := 1.0 if math.sum(ev0) >= 0 else -1.0
	for obs in 0 ..< n_obs {
		market_scores[obs] = sign * l.dot_simd(std_data[obs], ev0)
	}

	total_var := 0.0
	for ev in pca.eigenvalues {total_var += ev}
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

// Factor decomposition of each asset's STANDARDISED variance (principal axis on
// the correlation matrix): factor_variance = communality, idiosyncratic_var =
// uniqueness, total_variance = 1, factor_exposure = loadings. Multiply by an
// asset's variance for return-variance units
decompose_risk :: proc(
	returns_data: [][]f64,
	n_factors: int,
	allocator: mem.Allocator = context.allocator,
) -> RiskDecomposition {
	fa := factor_analysis(returns_data, n_factors, allocator = allocator)
	n_assets := len(returns_data[0])

	total_variance := make([]f64, n_assets, allocator)
	for v in 0 ..< n_assets {
		total_variance[v] = fa.communalities[v] + fa.uniqueness[v]
	}
	delete(fa.eigenvalues, allocator)
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

_matrix_to_slice :: proc(m: ^l.Matrix(f64), allocator: mem.Allocator) -> [][]f64 {
	rows := m.rows
	cols := m.cols
	out := make([][]f64, rows, allocator)
	for i in 0 ..< rows {
		out[i] = make([]f64, cols, allocator)
		copy(out[i], m.data[i * cols:(i + 1) * cols])
	}
	return out
}

_standardize_data :: proc(data: [][]f64, allocator: mem.Allocator = context.allocator) -> [][]f64 {
	n_obs := len(data)
	if n_obs == 0 {return [][]f64{}}
	n_vars := len(data[0])

	means := make([]f64, n_vars, allocator)
	stds := make([]f64, n_vars, allocator)

	for v in 0 ..< n_vars {
		sum := 0.0
		for obs in 0 ..< n_obs {sum += data[obs][v]}
		means[v] = sum / f64(n_obs)
	}

	for v in 0 ..< n_vars {
		sum_sq := 0.0
		for obs in 0 ..< n_obs {
			diff := data[obs][v] - means[v]
			sum_sq += diff * diff
		}
		stds[v] = math.sqrt(sum_sq / f64(n_obs - 1))
		if stds[v] < 1e-10 {stds[v] = 1.0}
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
	if rows == 0 {return [][]f64{}}
	cols := len(mat[0])
	out := make([][]f64, rows, allocator)
	for r in 0 ..< rows {
		out[r] = make([]f64, cols, allocator)
		copy(out[r], mat[r])
	}
	return out
}

_destroy_matrix :: proc(mat: [][]f64, allocator: mem.Allocator) {
	for row in mat {delete(row, allocator)}
	delete(mat, allocator)
}

_destroy_pca_result :: proc(res: ana.PCAResult, allocator: mem.Allocator) {
	for vec in res.eigenvectors {delete(vec, allocator)}
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
	return factor_analysis(data, n_factors, allocator = allocator)
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

// ============================================================================
// Factor Model Covariance (Fama-French Style)
// ============================================================================

FactorModelCovariance :: struct {
	factor_cov:        [][]f64,
	betas:             [][]f64,
	idiosyncratic_var: []f64,
	reconstructed_cov: [][]f64,
	sample_cov:        [][]f64,
}

// Time-series factor model: per-asset OLS betas β = Σ_f⁻¹ cov(f, r), residual
// variance var(r) - βᵀcov(f, r) with the n - 1 denominator (so the diagonal of
// reconstructed_cov equals the sample variances), and Σ = B Σ_f Bᵀ + D
factor_model_covariance :: proc(
	assets_data: [][]f64,
	factors_data: [][]f64,
	allocator: mem.Allocator = context.allocator,
) -> FactorModelCovariance {
	n_obs := len(assets_data)
	if n_obs == 0 {return FactorModelCovariance{}}
	n_assets := len(assets_data[0])
	n_factors := len(factors_data[0])

	// 1. Compute sample covariance of assets using linalg
	assets_flat := make([]f64, n_obs * n_assets, allocator)
	idx := 0
	for r in 0 ..< n_obs {
		for c in 0 ..< n_assets {
			assets_flat[idx] = assets_data[r][c]
			idx += 1
		}
	}
	assets_mat := l.matrix_from_flat(assets_flat, n_obs, n_assets, allocator)
	defer l.matrix_free(&assets_mat)
	delete(assets_flat, allocator)

	sample_cov_mat := l.covariance(&assets_mat, allocator)
	sample_cov := _matrix_to_slice(&sample_cov_mat, allocator)
	l.matrix_free(&sample_cov_mat)

	// 2. Compute covariance of factors using linalg
	factors_flat := make([]f64, n_obs * n_factors, allocator)
	idx = 0
	for r in 0 ..< n_obs {
		for c in 0 ..< n_factors {
			factors_flat[idx] = factors_data[r][c]
			idx += 1
		}
	}
	factors_mat := l.matrix_from_flat(factors_flat, n_obs, n_factors, allocator)
	defer l.matrix_free(&factors_mat)
	delete(factors_flat, allocator)

	factor_cov_mat := l.covariance(&factors_mat, allocator)
	factor_cov := _matrix_to_slice(&factor_cov_mat, allocator)
	// FIX: Do NOT free factor_cov_mat here! We still need it for matrix_inverse.

	// 3. Compute betas (factor loadings) and idiosyncratic variance
	betas := make([][]f64, n_assets, allocator)
	idiosyncratic_var := make([]f64, n_assets, allocator)

	factor_means := make([]f64, n_factors, allocator)
	for f in 0 ..< n_factors {
		for i in 0 ..< n_obs {factor_means[f] += factors_data[i][f]}
		factor_means[f] /= f64(n_obs)
	}

	// Now it's safe to invert because factor_cov_mat is still valid
	inv_fcov_mat := l.matrix_inverse(&factor_cov_mat, allocator)
	defer l.matrix_free(&inv_fcov_mat)

	for a in 0 ..< n_assets {
		betas[a] = make([]f64, n_factors, allocator)

		asset_mean := 0.0
		for i in 0 ..< n_obs {asset_mean += assets_data[i][a]}
		asset_mean /= f64(n_obs)

		asset_var := 0.0
		cov_f_asset := make([]f64, n_factors, allocator)

		for i in 0 ..< n_obs {
			asset_diff := assets_data[i][a] - asset_mean
			asset_var += asset_diff * asset_diff

			for f in 0 ..< n_factors {
				cov_f_asset[f] += (factors_data[i][f] - factor_means[f]) * asset_diff
			}
		}
		asset_var /= f64(n_obs - 1)
		for f in 0 ..< n_factors {
			cov_f_asset[f] /= f64(n_obs - 1)
		}

		cov_f_asset_mat := l.matrix_from_flat(cov_f_asset, n_factors, 1, allocator)
		beta_mat := l.matmul_dyn(&inv_fcov_mat, &cov_f_asset_mat, allocator)

		for f in 0 ..< n_factors {
			betas[a][f] = beta_mat.data[f]
		}

		beta_dot_cov := 0.0
		for f in 0 ..< n_factors {
			beta_dot_cov += betas[a][f] * cov_f_asset[f]
		}

		idiosyncratic_var[a] = math.max(0.0, asset_var - beta_dot_cov)

		// EXPLICIT CLEANUP
		l.matrix_free(&cov_f_asset_mat)
		l.matrix_free(&beta_mat)
		delete(cov_f_asset, allocator)
	}

	delete(factor_means, allocator)

	// FIX: NOW we can safely free factor_cov_mat since we are done with it
	l.matrix_free(&factor_cov_mat)

	// 4. Reconstruct covariance matrix: Σ = B * Σ_f * B^T + D
	reconstructed_cov := make([][]f64, n_assets, allocator)
	for i in 0 ..< n_assets {
		reconstructed_cov[i] = make([]f64, n_assets, allocator)
		for j in 0 ..< n_assets {
			sys_var := 0.0
			for f1 in 0 ..< n_factors {
				for f2 in 0 ..< n_factors {
					sys_var += betas[i][f1] * factor_cov[f1][f2] * betas[j][f2]
				}
			}

			if i == j {
				sys_var += idiosyncratic_var[i]
			}

			reconstructed_cov[i][j] = sys_var
		}
	}

	return FactorModelCovariance {
		factor_cov = factor_cov,
		betas = betas,
		idiosyncratic_var = idiosyncratic_var,
		reconstructed_cov = reconstructed_cov,
		sample_cov = sample_cov,
	}
}

destroy_factor_model_covariance :: proc(fm: ^FactorModelCovariance, allocator: mem.Allocator) {
	for row in fm.factor_cov {delete(row, allocator)}
	delete(fm.factor_cov, allocator)

	for row in fm.betas {delete(row, allocator)}
	delete(fm.betas, allocator)

	delete(fm.idiosyncratic_var, allocator)

	for row in fm.reconstructed_cov {delete(row, allocator)}
	delete(fm.reconstructed_cov, allocator)

	for row in fm.sample_cov {delete(row, allocator)}
	delete(fm.sample_cov, allocator)
}
