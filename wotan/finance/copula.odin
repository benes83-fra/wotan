package finance

import "core:math"
import "core:mem"
import "core:slice"
import "runtime:intrinsics"

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
// Bivariate Student-t copula log-density (Mathematically Exact)
student_t_copula_log_pdf :: proc(u1: f64, u2: f64, rho: f64, nu: f64) -> f64 {
	if math.abs(rho) >= 1.0 || nu <= 2.0 {return -math.INF_F64}
	if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {return -math.INF_F64}

	x1 := t_quantile(u1, nu)
	x2 := t_quantile(u2, nu)

	rho_sq := rho * rho
	denom := 1.0 - rho_sq
	quad_form := (x1 * x1 + x2 * x2 - 2.0 * rho * x1 * x2) / denom

	// Gamma terms for exact normalization
	lg_nu_plus_2, _ := math.lgamma((nu + 2.0) / 2.0)
	lg_nu_plus_1, _ := math.lgamma((nu + 1.0) / 2.0)
	lg_nu, _ := math.lgamma(nu / 2.0)

	// log(c(u1, u2)) = log(f_bivariate) - log(f_marginal1) - log(f_marginal2)
	term1 := lg_nu_plus_2 - 2.0 * lg_nu_plus_1 + lg_nu
	term2 := -0.5 * math.ln_f64(denom)
	term3 := -((nu + 2.0) / 2.0) * math.ln_f64(1.0 + quad_form / nu)
	term4 := ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + x1 * x1 / nu)
	term5 := ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + x2 * x2 / nu)

	return term1 + term2 + term3 + term4 + term5
}

// Student-t copula log-likelihood
student_t_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64, nu: f64) -> f64 {
	n := min(len(u1), len(u2))
	loglik := 0.0

	for i in 0 ..< n {
		log_pdf := student_t_copula_log_pdf(u1[i], u2[i], rho, nu)
		if log_pdf > -math.INF_F64 {
			loglik += log_pdf
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

// SIMD-optimized Gaussian copula log-likelihood
gaussian_copula_loglik_simd :: proc(u1: []f64, u2: []f64, rho: f64) -> f64 {
	n := min(len(u1), len(u2))
	if n == 0 {return 0.0}

	rho_sq := rho * rho
	denom := 1.0 - rho_sq
	sqrt_denom := math.sqrt_f64(denom)
	log_coef := -0.5 * math.ln_f64(denom)

	// AVX: process 8 observations at once
	if intrinsics.has_target_feature("avx") {
		acc8 := simd.f64x8{0, 0, 0, 0, 0, 0, 0, 0}
		i := 0

		for ; i + 8 <= n; i += 8 {
			// Load 8 u1 values
			vu1 := simd.f64x8 {
				u1[i],
				u1[i + 1],
				u1[i + 2],
				u1[i + 3],
				u1[i + 4],
				u1[i + 5],
				u1[i + 6],
				u1[i + 7],
			}
			// Load 8 u2 values
			vu2 := simd.f64x8 {
				u2[i],
				u2[i + 1],
				u2[i + 2],
				u2[i + 3],
				u2[i + 4],
				u2[i + 5],
				u2[i + 6],
				u2[i + 7],
			}

			// Transform to normal (need scalar norm_inv for now)
			// This is the bottleneck - norm_inv is complex
			x1_arr: [8]f64
			x2_arr: [8]f64
			for j := 0; j < 8; j += 1 {
				x1_arr[j] = norm_inv(vu1[j])
				x2_arr[j] = norm_inv(vu2[j])
			}

			vx1 := transmute(simd.f64x8)x1_arr
			vx2 := transmute(simd.f64x8)x2_arr

			// Compute exponent: -(ρ²(x1² + x2²) - 2ρx1x2) / (2(1-ρ²))
			vx1_sq := intrinsics.simd_mul(vx1, vx1)
			vx2_sq := intrinsics.simd_mul(vx2, vx2)
			vx1x2 := intrinsics.simd_mul(vx1, vx2)

			rho_sq_vec := simd.f64x8 {
				rho_sq,
				rho_sq,
				rho_sq,
				rho_sq,
				rho_sq,
				rho_sq,
				rho_sq,
				rho_sq,
			}
			two_rho_vec := simd.f64x8 {
				2.0 * rho,
				2.0 * rho,
				2.0 * rho,
				2.0 * rho,
				2.0 * rho,
				2.0 * rho,
				2.0 * rho,
				2.0 * rho,
			}
			two_denom_vec := simd.f64x8 {
				2.0 * denom,
				2.0 * denom,
				2.0 * denom,
				2.0 * denom,
				2.0 * denom,
				2.0 * denom,
				2.0 * denom,
				2.0 * denom,
			}

			sum_sq := intrinsics.simd_add(vx1_sq, vx2_sq)
			numerator := intrinsics.simd_sub(
				intrinsics.simd_mul(rho_sq_vec, sum_sq),
				intrinsics.simd_mul(two_rho_vec, vx1x2),
			)
			exponent := intrinsics.simd_div(numerator, two_denom_vec)

			// exp(exponent)
			exp_arr := transmute([8]f64)exponent
			exp_vals: [8]f64
			for j := 0; j < 8; j += 1 {
				exp_vals[j] = math.exp(exp_arr[j])
			}
			v_exp := transmute(simd.f64x8)exp_vals

			// log(pdf) = log_coef + exponent
			log_coef_vec := simd.f64x8 {
				log_coef,
				log_coef,
				log_coef,
				log_coef,
				log_coef,
				log_coef,
				log_coef,
				log_coef,
			}
			log_pdf := intrinsics.simd_add(log_coef_vec, exponent)

			acc8 = intrinsics.simd_add(acc8, log_pdf)
		}

		loglik := intrinsics.simd_reduce_add_pairs(acc8)

		// Tail
		for ; i < n; i += 1 {
			pdf := gaussian_copula_pdf(u1[i], u2[i], rho)
			if pdf > 0.0 {
				loglik += math.ln_f64(pdf)
			}
		}

		return loglik
	}

	// Fallback to scalar
	return gaussian_copula_loglik(u1, u2, rho)
}
// SIMD-optimized Kendall's Tau
kendall_tau_simd :: proc(x: []f64, y: []f64) -> f64 {
	n := min(len(x), len(y))
	if n < 2 {return 0.0}

	concordant := 0
	discordant := 0

	// AVX: vectorize inner loop
	if intrinsics.has_target_feature("avx") {
		for i := 0; i < n - 1; i += 1 {
			dx_base := x[i]
			dy_base := y[i]

			j := i + 1
			// Process 8 comparisons at once
			for ; j + 8 <= n; j += 8 {
				vx := simd.f64x8 {
					x[j],
					x[j + 1],
					x[j + 2],
					x[j + 3],
					x[j + 4],
					x[j + 5],
					x[j + 6],
					x[j + 7],
				}
				vy := simd.f64x8 {
					y[j],
					y[j + 1],
					y[j + 2],
					y[j + 3],
					y[j + 4],
					y[j + 5],
					y[j + 6],
					y[j + 7],
				}

				dx_base_vec := simd.f64x8 {
					dx_base,
					dx_base,
					dx_base,
					dx_base,
					dx_base,
					dx_base,
					dx_base,
					dx_base,
				}
				dy_base_vec := simd.f64x8 {
					dy_base,
					dy_base,
					dy_base,
					dy_base,
					dy_base,
					dy_base,
					dy_base,
					dy_base,
				}

				vdx := intrinsics.simd_sub(dx_base_vec, vx)
				vdy := intrinsics.simd_sub(dy_base_vec, vy)

				// Product of differences
				vprod := intrinsics.simd_mul(vdx, vdy)

				// Count positive (concordant) and negative (discordant)
				zero_vec := simd.f64x8{0, 0, 0, 0, 0, 0, 0, 0}
				mask_pos := intrinsics.simd_lanes_gt(vprod, zero_vec)
				mask_neg := intrinsics.simd_lanes_lt(vprod, zero_vec)

				// Convert masks to counts
				// This is tricky - need to sum the mask bits
				// For now, fall back to scalar for counting
				prod_arr := transmute([8]f64)vprod
				for k := 0; k < 8; k += 1 {
					if prod_arr[k] > 0.0 {
						concordant += 1
					} else if prod_arr[k] < 0.0 {
						discordant += 1
					}
				}
			}

			// Tail
			for ; j < n; j += 1 {
				dx := x[i] - x[j]
				dy := y[i] - y[j]
				if dx * dy > 0.0 {
					concordant += 1
				} else if dx * dy < 0.0 {
					discordant += 1
				}
			}
		}
	} else {
		// Scalar fallback
		for i := 0; i < n - 1; i += 1 {
			for j := i + 1; j < n; j += 1 {
				dx := x[i] - x[j]
				dy := y[i] - y[j]
				if dx * dy > 0.0 {
					concordant += 1
				} else if dx * dy < 0.0 {
					discordant += 1
				}
			}
		}
	}

	total := f64(n * (n - 1) / 2)
	return f64(concordant - discordant) / total
}
// SIMD-optimized backtesting
backtest_var_simd :: proc(
	returns: []f64,
	var_series: []f64,
	confidence: f64 = 0.95,
) -> VaR_BacktestResult {
	n := min(len(returns), len(var_series))
	if n == 0 {
		return VaR_BacktestResult{}
	}

	n_breaches := 0

	// AVX: process 8 observations at once
	if intrinsics.has_target_feature("avx") {
		i := 0
		for ; i + 8 <= n; i += 8 {
			vret := simd.f64x8 {
				returns[i],
				returns[i + 1],
				returns[i + 2],
				returns[i + 3],
				returns[i + 4],
				returns[i + 5],
				returns[i + 6],
				returns[i + 7],
			}
			vvar := simd.f64x8 {
				var_series[i],
				var_series[i + 1],
				var_series[i + 2],
				var_series[i + 3],
				var_series[i + 4],
				var_series[i + 5],
				var_series[i + 6],
				var_series[i + 7],
			}

			// Negate var_series
			zero_vec := simd.f64x8{0, 0, 0, 0, 0, 0, 0, 0}
			neg_var := intrinsics.simd_sub(zero_vec, vvar)

			// Check if returns < -var_series
			mask := intrinsics.simd_lanes_lt(vret, neg_var)

			// Count breaches (need to sum mask bits)
			// For simplicity, use scalar counting
			ret_arr := transmute([8]f64)vret
			var_arr := transmute([8]f64)vvar
			for j := 0; j < 8; j += 1 {
				if ret_arr[j] < -var_arr[j] {
					n_breaches += 1
				}
			}
		}

		// Tail
		for ; i < n; i += 1 {
			if returns[i] < -var_series[i] {
				n_breaches += 1
			}
		}
	} else {
		// Scalar fallback
		for i := 0; i < n; i += 1 {
			if returns[i] < -var_series[i] {
				n_breaches += 1
			}
		}
	}

	expected := f64(n) * (1.0 - confidence)
	breach_rate := f64(n_breaches) / f64(n)
	kupiec_stat := _kupiec_pof_test(n, n_breaches, 1.0 - confidence)
	p_value := _chi_squared_pvalue_1df(kupiec_stat)

	return VaR_BacktestResult {
		n_obs = n,
		n_breaches = n_breaches,
		expected_breaches = expected,
		breach_rate = breach_rate,
		kupiec_stat = kupiec_stat,
		kupiec_pvalue = p_value,
		passes_test = p_value > 0.05,
	}
}
