package finance

import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// CREDIT DEFAULT SWAP (CDS) - Constant Hazard Rate Model
// ============================================================================

CDS_Params :: struct {
	lambda:   f64, // Hazard rate (default intensity)
	recovery: f64, // Recovery rate (e.g., 0.40)
	r:        f64, // Risk-free rate
}

// Analytical CDS Pricing: Returns PV to the protection buyer
price_cds :: proc(
	notional: f64,
	spread: f64, // Running spread (e.g., 0.0100 for 100bps)
	maturity: f64,
	params: CDS_Params,
	payment_freq: f64, // e.g., 0.25 for quarterly
) -> f64 {
	lambda := params.lambda
	R := params.recovery
	r := params.r

	// 1. PV of Premium Leg (Annuity)
	premium_pv := 0.0
	t := payment_freq
	for t <= maturity {
		survival_prob := math.exp_f64(-lambda * t)
		disc_factor := math.exp_f64(-r * t)
		premium_pv += payment_freq * survival_prob * disc_factor
		t += payment_freq
	}
	premium_pv *= notional * spread

	// 2. PV of Protection Leg (Expected Loss)
	// Analytical solution for constant hazard rate
	prot_pv :=
		notional *
		(1.0 - R) *
		(lambda / (lambda + r)) *
		(1.0 - math.exp_f64(-(lambda + r) * maturity))

	return prot_pv - premium_pv
}

// ============================================================================
// COLLATERALIZED DEBT OBLIGATION (CDO) - Gaussian Copula Model (2008 Era)
// ============================================================================

CDO_Tranche :: struct {
	name:   string,
	attach: f64, // Attachment point (e.g., 0.00 for Equity, 0.03 for Mezz)
	detach: f64, // Detachment point (e.g., 0.03 for Equity, 0.07 for Mezz)
}

CDO_Params :: struct {
	N:        int, // Number of names in portfolio
	lambda:   f64, // Homogeneous hazard rate for all names
	recovery: f64, // Homogeneous recovery rate
	r:        f64, // Risk-free rate
	rho:      f64, // Gaussian Copula correlation (the "magic" number)
	maturity: f64, // CDO maturity
}

// Single-Factor Gaussian Copula Monte Carlo for CDO Tranche Pricing
price_cdo_tranche_mc :: proc(
	tranche: CDO_Tranche,
	params: CDO_Params,
	spread: f64, // Running spread paid by the tranche (e.g., 0.0150 for 150bps)
	n_paths: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	tranche_pv: f64,
	expected_loss_pct: f64,
) {

	N := params.N
	lambda := params.lambda
	R := params.recovery
	r := params.r
	rho := params.rho
	T := params.maturity

	sqrt_rho := math.sqrt_f64(rho)
	sqrt_1_minus_rho := math.sqrt_f64(1.0 - rho)

	// 1 market factor + N idiosyncratic factors per path
	rand_count := n_paths * (1 + N)
	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	total_tranche_pv := 0.0
	total_expected_loss := 0.0
	rand_idx := 0

	for path in 0 ..< n_paths {
		Z_market := norm_data[rand_idx]
		rand_idx += 1

		defaults_count := 0
		for i in 0 ..< N {
			Z_i := norm_data[rand_idx]
			rand_idx += 1

			// Single-factor Gaussian Copula
			Z_total := sqrt_rho * Z_market + sqrt_1_minus_rho * Z_i

			// Map to uniform [0, 1] via standard normal CDF
			U := 0.5 * (1.0 + math.erf(Z_total / math.sqrt_f64(2.0)))

			// Map to default time: tau = -ln(1 - U) / lambda
			tau := -math.ln_f64(1.0 - U) / lambda

			if tau <= T {
				defaults_count += 1
			}
		}

		// Portfolio Loss at maturity (simplified to maturity for PV approximation)
		portfolio_loss_pct := f64(defaults_count) / f64(N) * (1.0 - R)

		// Tranche Loss calculation
		loss_min_det := math.min(portfolio_loss_pct, tranche.detach)
		loss_min_att := math.min(portfolio_loss_pct, tranche.attach)

		tranche_loss_pct := (loss_min_det - loss_min_att) / (tranche.detach - tranche.attach)
		tranche_loss_pct = math.max(0.0, math.min(1.0, tranche_loss_pct)) // Clamp to [0, 1]

		total_expected_loss += tranche_loss_pct

		// Simplified Tranche PV: PV(Protection) - PV(Premium)
		// Annuity factor approximation
		annuity_approx := (1.0 - math.exp_f64(-(r + lambda) * T)) / (r + lambda)

		// Tranched survival probability approximation
		running_avg_loss := total_expected_loss / f64(path + 1)
		tranche_survival := 1.0 - running_avg_loss

		premium_pv := spread * annuity_approx * math.max(0.1, tranche_survival)
		prot_pv := tranche_loss_pct * math.exp_f64(-r * T * 0.5) // Mid-point discounting

		total_tranche_pv += prot_pv - premium_pv
	}

	return (total_tranche_pv / f64(n_paths)), (total_expected_loss / f64(n_paths))
}
// ============================================================================
// THE "CORRECT" MODEL: STUDENT-T COPULA CDO PRICER
// ============================================================================
// Fast, numerically stable Student-t CDF using Simpson's Rule and math.lgamma
student_t_cdf :: proc(x: f64, nu: f64) -> f64 {
	if x == 0.0 {return 0.5}
	// For large nu, it converges to the Normal distribution
	if nu > 50.0 {return 0.5 * (1.0 + math.erf(x / math.sqrt_f64(2.0)))}

	abs_x := math.abs(x)

	// ✅ FIXED: math.lgamma returns (value, sign). We only need the value.
	lg1, _ := math.lgamma((nu + 1.0) / 2.0)
	lg2, _ := math.lgamma(nu / 2.0)

	// Precompute the log-constant of the t-PDF for speed
	log_const := lg1 - lg2 - 0.5 * math.ln_f64(nu * math.PI)

	// 50 steps of Simpson's rule is highly accurate and very fast in Odin
	n := 50
	h := abs_x / f64(n)
	sum := 0.0

	for i in 0 ..= n {
		t_val := f64(i) * h
		// PDF at t_val
		pdf_val := math.exp_f64(
			log_const - ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + (t_val * t_val) / nu),
		)

		if i == 0 || i == n {
			sum += pdf_val
		} else if i % 2 == 1 {
			sum += 4.0 * pdf_val
		} else {
			sum += 2.0 * pdf_val
		}
	}

	integral := sum * h / 3.0

	if x < 0.0 {
		return 0.5 - integral
	} else {
		return 0.5 + integral
	}
}

// Student-t Copula Monte Carlo for CDO Tranche Pricing
price_cdo_tranche_t_copula_mc :: proc(
	tranche: CDO_Tranche,
	params: CDO_Params,
	nu: f64, // Degrees of freedom (e.g., 3.0 to 5.0 for heavy tails)
	spread: f64,
	n_paths: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	tranche_pv: f64,
	expected_loss_pct: f64,
) {

	N := params.N
	lambda := params.lambda
	R := params.recovery
	r := params.r
	rho := params.rho
	T := params.maturity

	// Round nu to nearest integer for Chi-Squared generation
	nu_int := int(math.round_f64(nu))
	if nu_int < 1 {nu_int = 1}

	// Normals needed per path: 1 (market) + N (idiosyncratic) + nu_int (shared chi-squared)
	norms_per_path := 1 + N + nu_int
	rand_count := n_paths * norms_per_path

	norm_data := make([]f64, rand_count, allocator)
	defer delete(norm_data, allocator)
	for i in 0 ..< rand_count {
		norm_data[i] = rand.float64_normal(0.0, 1.0)
	}

	total_tranche_pv := 0.0
	total_expected_loss := 0.0
	rand_idx := 0

	sqrt_rho := math.sqrt_f64(rho)
	sqrt_1_minus_rho := math.sqrt_f64(1.0 - rho)

	for path in 0 ..< n_paths {
		Z_market := norm_data[rand_idx]
		rand_idx += 1

		// 🔑 THE FIX: A SINGLE shared Chi-Squared variable for the entire path.
		// This is what creates tail dependence. When V is small, ALL T_i scale up.
		V := 0.0
		for i in 0 ..< nu_int {
			z_chi := norm_data[rand_idx]
			rand_idx += 1
			V += z_chi * z_chi
		}
		sqrt_V_over_nu := math.sqrt_f64(V / f64(nu_int))

		T_market := Z_market / sqrt_V_over_nu

		defaults_count := 0
		for i in 0 ..< N {
			Z_i := norm_data[rand_idx]
			rand_idx += 1

			T_i := Z_i / sqrt_V_over_nu

			// Single-factor Student-t Copula
			T_total := sqrt_rho * T_market + sqrt_1_minus_rho * T_i

			// Map to uniform [0, 1] via Student-t CDF
			U := student_t_cdf(T_total, nu)

			// Map to default time
			tau := -math.ln_f64(1.0 - U) / lambda

			if tau <= T {
				defaults_count += 1
			}
		}

		portfolio_loss_pct := f64(defaults_count) / f64(N) * (1.0 - R)

		// Tranche Loss calculation
		loss_min_det := math.min(portfolio_loss_pct, tranche.detach)
		loss_min_att := math.min(portfolio_loss_pct, tranche.attach)

		tranche_loss_pct := (loss_min_det - loss_min_att) / (tranche.detach - tranche.attach)
		tranche_loss_pct = math.max(0.0, math.min(1.0, tranche_loss_pct))

		total_expected_loss += tranche_loss_pct

		// Simplified Tranche PV
		annuity_approx := (1.0 - math.exp_f64(-(r + lambda) * T)) / (r + lambda)
		running_avg_loss := total_expected_loss / f64(path + 1)
		tranche_survival := 1.0 - running_avg_loss

		premium_pv := spread * annuity_approx * math.max(0.1, tranche_survival)
		prot_pv := tranche_loss_pct * math.exp_f64(-r * T * 0.5)

		total_tranche_pv += prot_pv - premium_pv
	}

	return (total_tranche_pv / f64(n_paths)), (total_expected_loss / f64(n_paths))
}
