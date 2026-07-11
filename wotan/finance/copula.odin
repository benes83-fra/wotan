package finance

import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Copula Models for Multivariate Dependence
// ============================================================================

CopulaType :: enum {
	Gaussian,
	StudentT,
	Clayton,
	Gumbel,
	Frank,
}

CopulaResult :: struct {
	copula_type:    CopulaType,
	parameters:     []f64, // Copula parameters (e.g., correlation, df)
	log_likelihood: f64,
	aic:            f64,
	bic:            f64,
	converged:      bool,
	n_obs:          int,
}

// ============================================================================
// Probability Integral Transform (PIT)
// ============================================================================

// Transform data to uniform [0,1] using empirical CDF
// This is the first step in copula modeling
pit_empirical :: proc(data: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(data)
	u := make([]f64, n, allocator)

	// Sort indices
	indices := make([]int, n, context.temp_allocator)
	defer delete(indices, context.temp_allocator)

	for i in 0 ..< n {
		indices[i] = i
	}

	// Simple sort by data values
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if data[indices[j]] > data[indices[j + 1]] {
				indices[j], indices[j + 1] = indices[j + 1], indices[j]
			}
		}
	}

	// Assign ranks
	for i in 0 ..< n {
		u[indices[i]] = f64(i + 1) / f64(n + 1)
	}

	return u
}

// Transform using normal CDF (if data is already standardized)
pit_normal :: proc(data: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(data)
	u := make([]f64, n, allocator)

	for i in 0 ..< n {
		// Standard normal CDF
		u[i] = 0.5 * (1.0 + math.erf_f64(data[i] / math.sqrt_f64(2.0)))
	}

	return u
}

// ============================================================================
// Gaussian Copula
// ============================================================================

// Bivariate Gaussian copula density
// c(u1, u2) = (1/sqrt(1-ρ²)) * exp(-ρ²(x1² + x2² - 2ρx1x2) / (2(1-ρ²)))
// where x1 = Φ⁻¹(u1), x2 = Φ⁻¹(u2)
gaussian_copula_pdf :: proc(u1: f64, u2: f64, rho: f64) -> f64 {
	if math.abs(rho) >= 1.0 {return 0.0}
	if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {return 0.0}

	// Transform to normal
	x1 := norm_inv(u1)
	x2 := norm_inv(u2)

	rho_sq := rho * rho
	denom := 1.0 - rho_sq

	// Copula density
	exponent := -(rho_sq * (x1 * x1 + x2 * x2) - 2.0 * rho * x1 * x2) / (2.0 * denom)

	return (1.0 / math.sqrt_f64(denom)) * math.exp_f64(exponent)
}

// Gaussian copula log-likelihood
gaussian_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64) -> f64 {
	n := min(len(u1), len(u2))
	loglik := 0.0

	for i in 0 ..< n {
		pdf := gaussian_copula_pdf(u1[i], u2[i], rho)
		if pdf > 0.0 {
			loglik += math.ln_f64(pdf)
		}
	}

	return loglik
}

// ============================================================================
// Student-t Copula
// ============================================================================

// Bivariate Student-t copula density
// More complex - involves bivariate t-distribution
student_t_copula_pdf :: proc(u1: f64, u2: f64, rho: f64, nu: f64) -> f64 {
	if math.abs(rho) >= 1.0 || nu <= 2.0 {return 0.0}
	if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {return 0.0}

	// Transform to t-distribution quantiles
	x1 := t_quantile(u1, nu)
	x2 := t_quantile(u2, nu)

	rho_sq := rho * rho
	denom := 1.0 - rho_sq

	// Bivariate t density
	quad_form := (x1 * x1 + x2 * x2 - 2.0 * rho * x1 * x2) / denom

	// t-distribution normalization
	lg1, _ := math.lgamma((nu + 1.0) / 2.0)
	lg2, _ := math.lgamma(nu / 2.0)

	log_coef := lg1 - lg2 - 0.5 * math.ln_f64(nu * math.PI)

	// Copula density (ratio of bivariate t to product of univariate t)
	log_pdf :=
		log_coef -
		0.5 * math.ln_f64(denom) -
		((nu + 1.0) / 2.0) * math.ln_f64(1.0 + quad_form / nu)

	// Subtract marginal densities
	log_marginal := log_coef - ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + x1 * x1 / nu)
	log_marginal += log_coef - ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + x2 * x2 / nu)

	return math.exp_f64(log_pdf - log_marginal)
}

// Student-t copula log-likelihood
student_t_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64, nu: f64) -> f64 {
	n := min(len(u1), len(u2))
	loglik := 0.0

	for i in 0 ..< n {
		pdf := student_t_copula_pdf(u1[i], u2[i], rho, nu)
		if pdf > 0.0 {
			loglik += math.ln_f64(pdf)
		}
	}

	return loglik
}

// ============================================================================
// Copula Fitting
// ============================================================================

// Fit Gaussian copula using MLE
fit_gaussian_copula :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .Gaussian
	result.n_obs = min(len(u1), len(u2))

	// Grid search for optimal rho
	best_rho := 0.0
	best_loglik := -math.INF_F64

	for rho_i in -99 ..< 100 {
		rho := f64(rho_i) / 100.0
		loglik := gaussian_copula_loglik(u1, u2, rho)

		if loglik > best_loglik {
			best_loglik = loglik
			best_rho = rho
		}
	}

	// Fine-tune around best
	for rho_i in -100 ..< 101 {
		rho := best_rho + f64(rho_i) / 10000.0
		if math.abs(rho) < 1.0 {
			loglik := gaussian_copula_loglik(u1, u2, rho)
			if loglik > best_loglik {
				best_loglik = loglik
				best_rho = rho
			}
		}
	}

	result.parameters = make([]f64, 1, allocator)
	result.parameters[0] = best_rho
	result.log_likelihood = best_loglik

	k := 1.0 // 1 parameter
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * best_loglik
	result.bic = k * math.ln_f64(n) - 2.0 * best_loglik
	result.converged = true

	return result
}

// Fit Student-t copula using MLE
fit_student_t_copula :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .StudentT
	result.n_obs = min(len(u1), len(u2))

	// Grid search for optimal (rho, nu)
	best_rho := 0.0
	best_nu := 5.0
	best_loglik := -math.INF_F64

	for rho_i in -90 ..< 91 {
		rho := f64(rho_i) / 100.0
		for nu_i in 30 ..< 200 {
			nu := f64(nu_i) / 10.0
			loglik := student_t_copula_loglik(u1, u2, rho, nu)

			if loglik > best_loglik {
				best_loglik = loglik
				best_rho = rho
				best_nu = nu
			}
		}
	}

	result.parameters = make([]f64, 2, allocator)
	result.parameters[0] = best_rho
	result.parameters[1] = best_nu
	result.log_likelihood = best_loglik

	k := 2.0 // 2 parameters
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * best_loglik
	result.bic = k * math.ln_f64(n) - 2.0 * best_loglik
	result.converged = true

	return result
}

// ============================================================================
// Empirical Dependence Measures
// ============================================================================

// Compute Kendall's tau (rank correlation)
kendall_tau :: proc(x: []f64, y: []f64) -> f64 {
	n := min(len(x), len(y))
	if n < 2 {return 0.0}

	concordant := 0
	discordant := 0

	for i in 0 ..< n - 1 {
		for j in i + 1 ..< n {
			dx := x[i] - x[j]
			dy := y[i] - y[j]

			if dx * dy > 0 {
				concordant += 1
			} else if dx * dy < 0 {
				discordant += 1
			}
		}
	}

	total := f64(n * (n - 1) / 2)
	return f64(concordant - discordant) / total
}

// Compute Spearman's rho (rank correlation)
spearman_rho :: proc(x: []f64, y: []f64, allocator: mem.Allocator) -> f64 {
	n := min(len(x), len(y))
	if n < 2 {return 0.0}

	// Transform to ranks
	rank_x := pit_empirical(x, allocator)
	rank_y := pit_empirical(y, allocator)
	defer {
		delete(rank_x, allocator)
		delete(rank_y, allocator)
	}

	// Pearson correlation of ranks
	mean_x := 0.0
	mean_y := 0.0
	for i in 0 ..< n {
		mean_x += rank_x[i]
		mean_y += rank_y[i]
	}
	mean_x /= f64(n)
	mean_y /= f64(n)

	cov := 0.0
	var_x := 0.0
	var_y := 0.0

	for i in 0 ..< n {
		dx := rank_x[i] - mean_x
		dy := rank_y[i] - mean_y
		cov += dx * dy
		var_x += dx * dx
		var_y += dy * dy
	}

	if var_x == 0.0 || var_y == 0.0 {return 0.0}

	return cov / math.sqrt_f64(var_x * var_y)
}

// ============================================================================
// Tail Dependence
// ============================================================================

// Compute lower tail dependence (probability of joint extreme losses)
lower_tail_dependence :: proc(u1: []f64, u2: []f64, threshold: f64 = 0.05) -> f64 {
	n := min(len(u1), len(u2))
	if n == 0 {return 0.0}

	count_u1_low := 0
	count_both_low := 0

	for i in 0 ..< n {
		if u1[i] < threshold {
			count_u1_low += 1
			if u2[i] < threshold {
				count_both_low += 1
			}
		}
	}

	if count_u1_low == 0 {return 0.0}

	return f64(count_both_low) / f64(count_u1_low)
}

// Compute upper tail dependence
upper_tail_dependence :: proc(u1: []f64, u2: []f64, threshold: f64 = 0.95) -> f64 {
	n := min(len(u1), len(u2))
	if n == 0 {return 0.0}

	count_u1_high := 0
	count_both_high := 0

	for i in 0 ..< n {
		if u1[i] > threshold {
			count_u1_high += 1
			if u2[i] > threshold {
				count_both_high += 1
			}
		}
	}

	if count_u1_high == 0 {return 0.0}

	return f64(count_both_high) / f64(count_u1_high)
}
