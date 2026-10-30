package finance

import ml "../analytics/ML"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// CDS LEG ENGINE (piecewise-constant hazard, flat continuous rate)
// ============================================================================
// Coupon dates run backwards from maturity every payment_freq years (a short
// first period if maturity is not a multiple). Per unit notional:
//   premium annuity = Σ Δ_k S(t_k) D(t_k) + ∫ (u − t_{k−1}) dF(u) D(u)
//                     (coupons plus the premium accrued up to default),
//   protection      = (1 − R) ∫_0^T D(u) dF(u)    (paid at default),
// with S(t) = e^{−∫λ}, D(t) = e^{−r t}; both integrals are exact on each piece
// where λ is constant (coupon periods are split at the hazard knots).

CDS_Params :: struct {
	lambda:   f64, // Hazard rate (default intensity)
	recovery: f64, // Recovery rate (e.g., 0.40)
	r:        f64, // Risk-free rate
}

@(private = "file")
_cds_n_periods :: proc(maturity, freq: f64) -> int {
	return max(1, int(math.ceil(maturity / freq - 1e-9)))
}

// (1 − e^{−kL}) / k and (1 − e^{−kL}(1 + kL)) / k², stable as k → 0
@(private = "file")
_cds_e1_e2 :: proc(k, L: f64) -> (e1, e2: f64) {
	x := k * L
	if abs(x) < 1e-6 {
		return L * (1.0 - 0.5 * x), 0.5 * L * L * (1.0 - 2.0 * x / 3.0)
	}
	E := math.exp_f64(-x)
	return (1.0 - E) / k, (1.0 - E * (1.0 + x)) / (k * k)
}

// Annuity (per unit spread, including accrual on default) and protection leg
// (per unit notional) to `maturity` on the curve's hazards
@(private = "file")
_cds_legs :: proc(
	tenors, hazards: []f64,
	recovery, r, maturity, freq: f64,
) -> (
	annuity, protection: f64,
) {
	if maturity <= 0.0 || len(hazards) == 0 {return 0.0, 0.0}
	n := _cds_n_periods(maturity, freq)
	last := len(hazards) - 1
	S := 1.0 // survival at the running time
	j := 0 // hazard piece of the running time
	prev := 0.0
	for k in 1 ..= n {
		tk := maturity - f64(n - k) * freq
		a := prev
		for a < tk {
			for j < last && tenors[j] <= a {j += 1}
			b := tk
			if j < last && tenors[j] < b {b = tenors[j]}
			h := hazards[j]
			L := b - a
			D := math.exp_f64(-r * a)
			e1, e2 := _cds_e1_e2(h + r, L)
			protection += (1.0 - recovery) * h * S * D * e1
			annuity += h * S * D * ((a - prev) * e1 + e2)
			S *= math.exp_f64(-h * L)
			a = b
		}
		annuity += (tk - prev) * S * math.exp_f64(-r * tk)
		prev = tk
	}
	return
}

// Analytical CDS Pricing: PV to the protection buyer, notional × (protection −
// spread × annuity), constant hazard and rate (premium accrued on default included)
price_cds :: proc(
	notional: f64,
	spread: f64, // Running spread (e.g., 0.0100 for 100bps)
	maturity: f64,
	params: CDS_Params,
	payment_freq: f64, // e.g., 0.25 for quarterly
) -> f64 {
	tenors := []f64{maturity}
	hazards := []f64{params.lambda}
	annuity, protection := _cds_legs(
		tenors,
		hazards,
		params.recovery,
		params.r,
		maturity,
		payment_freq,
	)
	return notional * (protection - spread * annuity)
}

// ============================================================================
// COLLATERALIZED DEBT OBLIGATION (CDO) - One-Factor Copulas
// ============================================================================
// Homogeneous pool of N names with default times τ_i = −ln(1 − U_i)/λ, where the
// U_i come from a one-factor Gaussian or Student-t copula. On each path the
// tranche loss TL(t) = (min(L, d) − min(L, a)) / (d − a) of the portfolio loss
// L(t) = (1 − R) · #defaults(t) / N is tracked at every coupon date, and per
// unit tranche notional
//   protection = Σ_k (TL_k − TL_{k−1}) D((t_{k−1} + t_k)/2)   (mid-period settlement)
//   premium    = spread Σ_k Δ_k D(t_k) (1 − (TL_{k−1} + TL_k)/2)
// tranche_pv = E[protection − premium] (to the protection buyer),
// expected_loss_pct = E[TL(T)].

CDO_Tranche :: struct {
	name:   string,
	attach: f64, // Attachment point (e.g., 0.00 for Equity, 0.03 for Mezz)
	detach: f64, // Detachment point (e.g., 0.03 for Equity, 0.07 for Mezz)
}

CDO_Params :: struct {
	N:            int, // Number of names in portfolio
	lambda:       f64, // Homogeneous hazard rate for all names
	recovery:     f64, // Homogeneous recovery rate
	r:            f64, // Risk-free rate
	rho:          f64, // Copula correlation (factor loading²)
	maturity:     f64, // CDO maturity
	payment_freq: f64, // Coupon period in years (0 = quarterly)
}

// Gamma(shape, 1) by Marsaglia & Tsang (2000); shape < 1 via the boost U^{1/shape}
@(private = "file")
_cdo_gamma :: proc(shape: f64) -> f64 {
	if shape < 1.0 {
		return _cdo_gamma(shape + 1.0) * math.pow(rand.float64(), 1.0 / shape)
	}
	d := shape - 1.0 / 3.0
	c := 1.0 / math.sqrt_f64(9.0 * d)
	for {
		x := rand.float64_normal(0.0, 1.0)
		v := 1.0 + c * x
		if v <= 0.0 {continue}
		v = v * v * v
		u := rand.float64()
		if math.ln(u) < 0.5 * x * x + d - d * v + d * math.ln(v) {return d * v}
	}
}

@(private = "file")
_cdo_tranche_mc :: proc(
	tranche: CDO_Tranche,
	params: CDO_Params,
	nu: f64, // ≤ 0: Gaussian copula
	spread: f64,
	n_paths: int,
	allocator: mem.Allocator,
) -> (
	tranche_pv: f64,
	expected_loss_pct: f64,
) {
	N, T := params.N, params.maturity
	if N <= 0 || n_paths <= 0 || T <= 0.0 || tranche.detach <= tranche.attach {return 0.0, 0.0}
	freq := params.payment_freq if params.payment_freq > 0.0 else 0.25
	n := _cds_n_periods(T, freq)
	times := make([]f64, n + 1, allocator)
	counts := make([]int, n + 1, allocator)
	defer {delete(times, allocator); delete(counts, allocator)}
	for k in 1 ..= n {times[k] = T - f64(n - k) * freq}

	// A name has defaulted by t_k iff its latent X ≤ c_k = F⁻¹(1 − e^{−λ t_k}),
	// F the copula's marginal (Φ or t_ν); c_k increases with k
	thr := make([]f64, n + 1, allocator)
	defer delete(thr, allocator)
	for k in 1 ..= n {
		pd := 1.0 - math.exp_f64(-params.lambda * times[k])
		thr[k] = ml.student_t_ppf(pd, nu) if nu > 0.0 else _N_inv(pd)
	}

	sr, s1r := math.sqrt_f64(params.rho), math.sqrt_f64(1.0 - params.rho)
	lgd := (1.0 - params.recovery) / f64(N)
	width := tranche.detach - tranche.attach
	tl :: #force_inline proc(L, a, d, w: f64) -> f64 {return (min(L, d) - min(L, a)) / w}

	sum_pv, sum_el := 0.0, 0.0
	for _ in 0 ..< n_paths {
		zm := rand.float64_normal(0.0, 1.0)
		scale := 1.0
		if nu > 0.0 {scale = 1.0 / math.sqrt_f64(2.0 * _cdo_gamma(0.5 * nu) / nu)} 	// 1/√(χ²_ν/ν)
		for k in 0 ..= n {counts[k] = 0}
		for _ in 0 ..< N {
			x := (sr * zm + s1r * rand.float64_normal(0.0, 1.0)) * scale
			if !(x <= thr[n]) {continue} 	// survives to T
			lo, hi := 1, n // first k with x ≤ thr[k]
			for lo < hi {
				mid := (lo + hi) / 2
				if x <= thr[mid] {hi = mid} else {lo = mid + 1}
			}
			counts[lo] += 1
		}
		prot, prem := 0.0, 0.0
		defaults := 0
		tl_prev := 0.0
		for k in 1 ..= n {
			defaults += counts[k]
			tlk := tl(lgd * f64(defaults), tranche.attach, tranche.detach, width)
			prot += (tlk - tl_prev) * math.exp_f64(-params.r * 0.5 * (times[k - 1] + times[k]))
			prem +=
				(times[k] - times[k - 1]) *
				math.exp_f64(-params.r * times[k]) *
				(1.0 - 0.5 * (tl_prev + tlk))
			tl_prev = tlk
		}
		sum_pv += prot - spread * prem
		sum_el += tl_prev
	}
	return sum_pv / f64(n_paths), sum_el / f64(n_paths)
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
	return _cdo_tranche_mc(tranche, params, 0.0, spread, n_paths, allocator)
}

// ============================================================================
// STUDENT-T COPULA CDO PRICER
// ============================================================================

// Student-t CDF (exact, regularized incomplete beta)
student_t_cdf :: proc(x: f64, nu: f64) -> f64 {
	return ml.student_t_cdf(x, nu)
}

// Student-t Copula Monte Carlo for CDO Tranche Pricing: X_i = (√ρ Z + √(1−ρ) ε_i)/√(W/ν)
// with one W ~ χ²_ν per path (any real ν > 0), U_i = t_ν(X_i)
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
	return _cdo_tranche_mc(tranche, params, max(nu, 1e-6), spread, n_paths, allocator)
}

// ============================================================================
// CDS CURVE BOOTSTRAPPING & PRICING
// ============================================================================

CDS_Curve :: struct {
	tenors:   []f64, // e.g., [1.0, 3.0, 5.0, 7.0, 10.0]
	hazards:  []f64, // piecewise constant hazard rates; hazards[i] applies up to tenors[i]
	recovery: f64,
	r:        f64,
}

CDS_Price_Result :: struct {
	upfront_pct:    f64, // Upfront to the protection buyer per unit notional (0.01 = 1%)
	rpv01:          f64, // PV of 1bp running (incl. accrual on default) per $1M notional
	premium_leg_pv: f64, // coupon × annuity × notional
	protection_pv:  f64,
}

// Survival probability e^{−∫_0^t λ} on the piecewise-constant curve (the last
// hazard extends beyond the last tenor)
get_survival_prob :: proc(t: f64, curve: CDS_Curve) -> f64 {
	if t <= 0.0 {return 1.0}

	// Find which tenor bucket 't' falls into
	bucket := 0
	for i in 0 ..< len(curve.tenors) {
		if t <= curve.tenors[i] {
			bucket = i
			break
		}
		bucket = i
	}

	// Calculate cumulative hazard up to time t
	cum_hazard := 0.0
	prev_T := 0.0
	for i in 0 ..= bucket {
		end_T := curve.tenors[i]
		if t < end_T || i == bucket {end_T = t}

		cum_hazard += curve.hazards[i] * (end_T - prev_T)
		prev_T = curve.tenors[i]
	}

	return math.exp_f64(-cum_hazard)
}

// Bootstrap piecewise constant hazard rates so that each par CDS (running
// spread = market spread) has zero PV under _cds_legs; tenors ascending
bootstrap_cds_curve :: proc(
	tenors: []f64,
	market_spreads: []f64, // in decimal (e.g., 0.0100 for 100bps)
	recovery: f64,
	r: f64,
	payment_freq: f64, // e.g., 0.25 for quarterly
	allocator: mem.Allocator = context.allocator,
) -> CDS_Curve {
	n := min(len(tenors), len(market_spreads))

	curve := CDS_Curve {
		tenors   = make([]f64, n, allocator),
		hazards  = make([]f64, n, allocator),
		recovery = recovery,
		r        = r,
	}
	copy(curve.tenors, tenors[:n])

	for i in 0 ..< n {
		s := market_spreads[i]
		pv :: proc(c: ^CDS_Curve, i: int, h, s, freq: f64) -> f64 {
			c.hazards[i] = h
			ann, prot := _cds_legs(
				c.tenors[:i + 1],
				c.hazards[:i + 1],
				c.recovery,
				c.r,
				c.tenors[i],
				freq,
			)
			return prot - s * ann // increasing in h
		}
		lo, hi := 0.0, 1.0
		for pv(&curve, i, hi, s, payment_freq) < 0.0 && hi < 1e6 {hi *= 2.0}
		for _ in 0 ..< 200 {
			mid := 0.5 * (lo + hi)
			if mid == lo || mid == hi {break}
			if pv(&curve, i, mid, s, payment_freq) < 0.0 {lo = mid} else {hi = mid}
		}
		curve.hazards[i] = 0.5 * (lo + hi)
	}

	return curve
}

// Price a CDS paying fixed_coupon on a bootstrapped curve
price_cds_full :: proc(
	notional: f64,
	fixed_coupon: f64, // Standardized coupon (e.g., 0.0100 for 100bps or 0.0500 for 500bps)
	maturity: f64,
	curve: CDS_Curve,
	payment_freq: f64,
) -> CDS_Price_Result {
	annuity, protection := _cds_legs(
		curve.tenors,
		curve.hazards,
		curve.recovery,
		curve.r,
		maturity,
		payment_freq,
	)
	return CDS_Price_Result {
		upfront_pct = protection - fixed_coupon * annuity,
		rpv01 = annuity * 1e-4 * 1_000_000.0,
		premium_leg_pv = fixed_coupon * annuity * notional,
		protection_pv = protection * notional,
	}
}
