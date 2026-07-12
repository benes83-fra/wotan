package finance

import l "../linalg" // Import your existing SIMD library
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
	parameters:     []f64,
	log_likelihood: f64,
	aic:            f64,
	bic:            f64,
	converged:      bool,
	n_obs:          int,
}

// ============================================================================
// Probability Integral Transform (PIT)
// ============================================================================

pit_empirical :: proc(data: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(data)
	u := make([]f64, n, allocator)

	indices := make([]int, n, context.temp_allocator)
	defer delete(indices, context.temp_allocator)

	for i in 0 ..< n {
		indices[i] = i
	}

	// Bubble sort by data values
	for i in 0 ..< n - 1 {
		for j in 0 ..< n - i - 1 {
			if data[indices[j]] > data[indices[j + 1]] {
				indices[j], indices[j + 1] = indices[j + 1], indices[j]
			}
		}
	}

	for i in 0 ..< n {
		u[indices[i]] = f64(i + 1) / f64(n + 1)
	}

	return u
}

pit_normal :: proc(data: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(data)
	u := make([]f64, n, allocator)

	for i in 0 ..< n {
		u[i] = 0.5 * (1.0 + math.erf_f64(data[i] / math.sqrt_f64(2.0)))
	}

	return u
}

// ============================================================================
// Gaussian Copula - Using Existing SIMD Infrastructure
// ============================================================================

gaussian_copula_pdf :: proc(u1: f64, u2: f64, rho: f64) -> f64 {
	if math.abs(rho) >= 1.0 {return 0.0}
	if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {return 0.0}

	x1 := norm_inv(u1)
	x2 := norm_inv(u2)

	rho_sq := rho * rho
	denom := 1.0 - rho_sq
	exponent := -(rho_sq * (x1 * x1 + x2 * x2) - 2.0 * rho * x1 * x2) / (2.0 * denom)

	return (1.0 / math.sqrt_f64(denom)) * math.exp_f64(exponent)
}

// PUBLIC API: Uses existing SIMD primitives from wotan_linalg
gaussian_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64) -> f64 {
	n := min(len(u1), len(u2))
	if n == 0 {return 0.0}

	rho_sq := rho * rho
	denom := 1.0 - rho_sq
	log_coef := -0.5 * math.ln_f64(denom)

	// Allocate working arrays
	x1 := make([]f64, n, context.temp_allocator)
	x2 := make([]f64, n, context.temp_allocator)
	x1_sq := make([]f64, n, context.temp_allocator)
	x2_sq := make([]f64, n, context.temp_allocator)
	x1x2 := make([]f64, n, context.temp_allocator)
	temp1 := make([]f64, n, context.temp_allocator)
	temp2 := make([]f64, n, context.temp_allocator)
	log_pdf := make([]f64, n, context.temp_allocator)
	defer {
		delete(x1, context.temp_allocator)
		delete(x2, context.temp_allocator)
		delete(x1_sq, context.temp_allocator)
		delete(x2_sq, context.temp_allocator)
		delete(x1x2, context.temp_allocator)
		delete(temp1, context.temp_allocator)
		delete(temp2, context.temp_allocator)
		delete(log_pdf, context.temp_allocator)
	}

	// Transform to normal (scalar bottleneck - unavoidable)
	for i in 0 ..< n {
		x1[i] = norm_inv(u1[i])
		x2[i] = norm_inv(u2[i])
	}

	// Vectorized operations using existing SIMD primitives
	l.vec_mul_simd(x1, x1, x1_sq) // x1²
	l.vec_mul_simd(x2, x2, x2_sq) // x2²
	l.vec_mul_simd(x1, x2, x1x2) // x1*x2

	// x1² + x2²
	l.vec_add_simd(x1_sq, x2_sq, temp1)

	// rho² * (x1² + x2²)
	l.vec_scale_simd(temp1, rho_sq, temp2)

	// 2*rho*x1x2
	l.vec_scale_simd(x1x2, 2.0 * rho, temp1)

	// rho² * (x1² + x2²) - 2*rho*x1x2
	l.vec_sub_simd(temp2, temp1, x1x2)

	// Divide by 2*(1-rho²)
	l.vec_scale_simd(x1x2, -1.0 / (2.0 * denom), temp1)

	// Add log_coef
	l.vec_broadcast_add_simd(log_coef, temp1)

	// Sum all log-likelihoods
	return l.sum_simd(temp1)
}

// ============================================================================
// Student-t Copula - Using Existing SIMD Infrastructure
// ============================================================================

student_t_copula_log_pdf :: proc(u1: f64, u2: f64, rho: f64, nu: f64) -> f64 {
	if math.abs(rho) >= 1.0 || nu <= 2.0 {return -math.INF_F64}
	if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {return -math.INF_F64}

	x1 := t_quantile(u1, nu)
	x2 := t_quantile(u2, nu)

	rho_sq := rho * rho
	denom := 1.0 - rho_sq
	quad_form := (x1 * x1 + x2 * x2 - 2.0 * rho * x1 * x2) / denom

	lg_nu_plus_2, _ := math.lgamma((nu + 2.0) / 2.0)
	lg_nu_plus_1, _ := math.lgamma((nu + 1.0) / 2.0)
	lg_nu, _ := math.lgamma(nu / 2.0)

	term1 := lg_nu_plus_2 - 2.0 * lg_nu_plus_1 + lg_nu
	term2 := -0.5 * math.ln_f64(denom)
	term3 := -((nu + 2.0) / 2.0) * math.ln_f64(1.0 + quad_form / nu)
	term4 := ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + x1 * x1 / nu)
	term5 := ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + x2 * x2 / nu)

	return term1 + term2 + term3 + term4 + term5
}

// PUBLIC API: Uses existing SIMD primitives
student_t_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64, nu: f64) -> f64 {
	n := min(len(u1), len(u2))
	if n == 0 {return 0.0}

	// Pre-compute constants
	rho_sq := rho * rho
	denom := 1.0 - rho_sq

	lg_nu_plus_2, _ := math.lgamma((nu + 2.0) / 2.0)
	lg_nu_plus_1, _ := math.lgamma((nu + 1.0) / 2.0)
	lg_nu, _ := math.lgamma(nu / 2.0)

	term1_const := lg_nu_plus_2 - 2.0 * lg_nu_plus_1 + lg_nu
	term2_const := -0.5 * math.ln_f64(denom)

	nu_plus_2_half := (nu + 2.0) / 2.0
	nu_plus_1_half := (nu + 1.0) / 2.0
	nu_inv := 1.0 / nu

	// Allocate working arrays
	x1 := make([]f64, n, context.temp_allocator)
	x2 := make([]f64, n, context.temp_allocator)
	x1_sq := make([]f64, n, context.temp_allocator)
	x2_sq := make([]f64, n, context.temp_allocator)
	x1x2 := make([]f64, n, context.temp_allocator)
	quad_form := make([]f64, n, context.temp_allocator)
	temp1 := make([]f64, n, context.temp_allocator)
	temp2 := make([]f64, n, context.temp_allocator)
	term3 := make([]f64, n, context.temp_allocator)
	term4 := make([]f64, n, context.temp_allocator)
	term5 := make([]f64, n, context.temp_allocator)
	log_pdf := make([]f64, n, context.temp_allocator)
	defer {
		delete(x1, context.temp_allocator)
		delete(x2, context.temp_allocator)
		delete(x1_sq, context.temp_allocator)
		delete(x2_sq, context.temp_allocator)
		delete(x1x2, context.temp_allocator)
		delete(quad_form, context.temp_allocator)
		delete(temp1, context.temp_allocator)
		delete(temp2, context.temp_allocator)
		delete(term3, context.temp_allocator)
		delete(term4, context.temp_allocator)
		delete(term5, context.temp_allocator)
		delete(log_pdf, context.temp_allocator)
	}

	// Transform to t-distribution (scalar bottleneck)
	for i in 0 ..< n {
		x1[i] = t_quantile(u1[i], nu)
		x2[i] = t_quantile(u2[i], nu)
	}

	// Vectorized operations
	l.vec_mul_simd(x1, x1, x1_sq)
	l.vec_mul_simd(x2, x2, x2_sq)
	l.vec_mul_simd(x1, x2, x1x2)

	// x1² + x2²
	l.vec_add_simd(x1_sq, x2_sq, temp1)

	// 2*rho*x1x2
	l.vec_scale_simd(x1x2, 2.0 * rho, temp2)

	// (x1² + x2²) - 2*rho*x1x2
	l.vec_sub_simd(temp1, temp2, quad_form)

	// Divide by (1-rho²)
	l.vec_scale_simd(quad_form, 1.0 / denom, temp1)

	// 1 + quad_form/nu
	l.vec_scale_simd(temp1, nu_inv, temp2)
	l.vec_broadcast_add_simd(1.0, temp2)

	// ln(1 + quad_form/nu)
	for i in 0 ..< n {
		term3[i] = -nu_plus_2_half * math.ln_f64(temp2[i])
	}

	// 1 + x1²/nu
	l.vec_scale_simd(x1_sq, nu_inv, temp1)
	l.vec_broadcast_add_simd(1.0, temp1)

	// ln(1 + x1²/nu)
	for i in 0 ..< n {
		term4[i] = nu_plus_1_half * math.ln_f64(temp1[i])
	}

	// 1 + x2²/nu
	l.vec_scale_simd(x2_sq, nu_inv, temp1)
	l.vec_broadcast_add_simd(1.0, temp1)

	// ln(1 + x2²/nu)
	for i in 0 ..< n {
		term5[i] = nu_plus_1_half * math.ln_f64(temp1[i])
	}

	// Combine: term1 + term2 + term3 + term4 + term5
	const_val := term1_const + term2_const
	l.vec_broadcast_add_simd(const_val, term3)
	l.vec_add_simd(term3, term4, temp1)
	l.vec_add_simd(temp1, term5, log_pdf)

	// Sum all log-likelihoods
	return l.sum_simd(log_pdf)
}

// ============================================================================
// Copula Fitting
// ============================================================================

fit_gaussian_copula :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .Gaussian
	result.n_obs = min(len(u1), len(u2))

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

	k := 1.0
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * best_loglik
	result.bic = k * math.ln_f64(n) - 2.0 * best_loglik
	result.converged = true

	return result
}

fit_student_t_copula :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .StudentT
	result.n_obs = min(len(u1), len(u2))

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

	k := 2.0
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * best_loglik
	result.bic = k * math.ln_f64(n) - 2.0 * best_loglik
	result.converged = true

	return result
}

// ============================================================================
// Empirical Dependence Measures
// ============================================================================

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

spearman_rho :: proc(x: []f64, y: []f64, allocator: mem.Allocator) -> f64 {
	n := min(len(x), len(y))
	if n < 2 {return 0.0}

	rank_x := pit_empirical(x, allocator)
	rank_y := pit_empirical(y, allocator)
	defer {
		delete(rank_x, allocator)
		delete(rank_y, allocator)
	}

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

// ============================================================================
// Clayton Copula (Lower Tail Dependence)
// ============================================================================

clayton_copula_loglik :: proc(u1: []f64, u2: []f64, theta: f64) -> f64 {
	n := min(len(u1), len(u2))
	loglik := 0.0

	// Precompute constants
	theta_plus_1 := theta + 1.0
	inv_theta := 1.0 / theta
	two_plus_inv := 2.0 + inv_theta

	for i in 0 ..< n {
		// Guard against domain errors
		if u1[i] <= 0.0 || u1[i] >= 1.0 || u2[i] <= 0.0 || u2[i] >= 1.0 {continue}

		// u^(-theta)
		u1_neg := math.pow(u1[i], -theta)
		u2_neg := math.pow(u2[i], -theta)
		sum_neg := u1_neg + u2_neg - 1.0

		if sum_neg <= 0.0 {continue}

		// Log-density: ln(1+theta) - (theta+1)(ln u1 + ln u2) - (2 + 1/theta) ln(sum)
		loglik +=
			math.ln_f64(theta_plus_1) -
			theta_plus_1 * (math.ln_f64(u1[i]) + math.ln_f64(u2[i])) -
			two_plus_inv * math.ln_f64(sum_neg)
	}

	return loglik
}

fit_clayton :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .Clayton
	result.n_obs = min(len(u1), len(u2))

	// Clayton requires positive dependence (tau > 0)
	tau := kendall_tau(u1, u2)
	if tau <= 0.0 {
		result.converged = false
		return result
	}

	// Theta estimation from Kendall's tau: theta = 2*tau / (1-tau)
	theta := 2.0 * tau / (1.0 - tau)

	result.parameters = make([]f64, 1, allocator)
	result.parameters[0] = theta
	result.log_likelihood = clayton_copula_loglik(u1, u2, theta)

	k := 1.0
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * result.log_likelihood
	result.bic = k * math.ln_f64(n) - 2.0 * result.log_likelihood
	result.converged = true

	return result
}

// ============================================================================
// Gumbel Copula (Upper Tail Dependence)
// ============================================================================

gumbel_copula_loglik :: proc(u1: []f64, u2: []f64, theta: f64) -> f64 {
	n := min(len(u1), len(u2))
	loglik := 0.0

	inv_theta := 1.0 / theta
	theta_minus_1 := theta - 1.0

	for i in 0 ..< n {
		if u1[i] <= 0.0 || u1[i] >= 1.0 || u2[i] <= 0.0 || u2[i] >= 1.0 {continue}

		a := -math.ln_f64(u1[i])
		b := -math.ln_f64(u2[i])
		K := math.pow(a, theta) + math.pow(b, theta)

		if K <= 0.0 {continue}

		// Log-density formula for Gumbel
		// ln c = (1/theta - 2) ln K + ln(theta - 1 + K^(1/theta)) - a - b + (1-theta)(ln a + ln b)
		K_inv := math.pow(K, inv_theta)
		term := theta_minus_1 + K_inv

		if term <= 0.0 {continue}

		loglik +=
			(inv_theta - 2.0) * math.ln_f64(K) +
			math.ln_f64(term) -
			a -
			b +
			(1.0 - theta) * (math.ln_f64(a) + math.ln_f64(b))
	}

	return loglik
}

fit_gumbel :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .Gumbel
	result.n_obs = min(len(u1), len(u2))

	// Gumbel requires positive dependence (tau > 0)
	tau := kendall_tau(u1, u2)
	if tau <= 0.0 {
		result.converged = false
		return result
	}

	// Theta estimation: theta = 1 / (1 - tau)
	theta := 1.0 / (1.0 - tau)

	// Gumbel requires theta >= 1
	if theta < 1.0 {theta = 1.0}

	result.parameters = make([]f64, 1, allocator)
	result.parameters[0] = theta
	result.log_likelihood = gumbel_copula_loglik(u1, u2, theta)

	k := 1.0
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * result.log_likelihood
	result.bic = k * math.ln_f64(n) - 2.0 * result.log_likelihood
	result.converged = true

	return result
}

// ============================================================================
// Frank Copula (Symmetric, No Tail Dependence)
// ============================================================================

frank_copula_loglik :: proc(u1: []f64, u2: []f64, theta: f64) -> f64 {
	n := min(len(u1), len(u2))
	loglik := 0.0

	// Precompute constants
	e_minus_theta := math.exp(-theta)
	e_minus_theta_minus_1 := e_minus_theta - 1.0
	num_const := -theta * e_minus_theta_minus_1

	for i in 0 ..< n {
		if u1[i] <= 0.0 || u1[i] >= 1.0 || u2[i] <= 0.0 || u2[i] >= 1.0 {continue}

		// A = e^(-theta*u1), B = e^(-theta*u2)
		A := math.exp(-theta * u1[i])
		B := math.exp(-theta * u2[i])

		// Denominator: (e^-theta - 1) + (A-1)(B-1)
		denom := e_minus_theta_minus_1 + (A - 1.0) * (B - 1.0)

		if denom <= 0.0 {continue}

		// Log-density: ln(-theta(e^-theta-1)AB) - 2 ln(denom)
		// = ln(|num_const|) + ln(A) + ln(B) - 2 ln(denom)
		loglik +=
			math.ln_f64(math.abs(num_const)) +
			math.ln_f64(A) +
			math.ln_f64(B) -
			2.0 * math.ln_f64(denom)
	}

	return loglik
}

// Helper: Debye function D_1(x) for Frank copula tau-theta relationship
_debye_1 :: proc(x: f64) -> f64 {
	if x == 0.0 {return 1.0}

	// Numerical integration of t/(e^t - 1) from 0 to x
	n := 100
	h := x / f64(n)
	sum := 0.0

	for i in 0 ..< n {
		t := f64(i) * h
		if t == 0.0 {
			sum += 0.5 * 1.0 // Limit t->0 is 1
		} else {
			sum += 0.5 * (t / (math.exp(t) - 1.0))
		}

		t_mid := (f64(i) + 0.5) * h
		sum += t_mid / (math.exp(t_mid) - 1.0)
	}

	return sum * h / x
}

fit_frank :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result: CopulaResult
	result.copula_type = .Frank
	result.n_obs = min(len(u1), len(u2))

	tau := kendall_tau(u1, u2)

	// Frank can handle negative dependence, so tau can be negative
	// We use numerical root finding to solve tau(theta) = empirical_tau
	// tau(theta) = 1 - 4/theta * (1 - D_1(theta)/theta)

	// Bisection search for theta in [-50, 50]
	low := -50.0
	high := 50.0
	best_theta := 0.0
	best_ll := -math.INF_F64

	// Quick check for independence
	if math.abs(tau) < 0.01 {
		best_theta = 0.001 // Near independence
	} else {
		for iter in 0 ..< 50 {
			mid := (low + high) / 2.0
			if math.abs(mid) < 1e-6 {mid = 1e-6}

			d1 := _debye_1(mid)
			tau_model := 1.0 - 4.0 / mid * (1.0 - d1 / mid)

			if tau_model > tau {
				low = mid
			} else {
				high = mid
			}

			if math.abs(tau_model - tau) < 1e-4 {
				best_theta = mid
				break
			}
			best_theta = mid
		}
	}

	result.parameters = make([]f64, 1, allocator)
	result.parameters[0] = best_theta
	result.log_likelihood = frank_copula_loglik(u1, u2, best_theta)

	k := 1.0
	n := f64(result.n_obs)
	result.aic = 2.0 * k - 2.0 * result.log_likelihood
	result.bic = k * math.ln_f64(n) - 2.0 * result.log_likelihood
	result.converged = true

	return result
}
