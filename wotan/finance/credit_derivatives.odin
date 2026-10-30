package finance

import ml "../analytics/ML"
import w "../core"
import "core:math"
import rand "core:math/rand"
import "core:mem"
import "core:slice"

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
	tenors:         []f64, // e.g., [1.0, 3.0, 5.0, 7.0, 10.0]
	hazards:        []f64, // piecewise constant hazard rates; hazards[i] applies up to tenors[i]
	recovery:       f64,
	r:              f64, // flat rate of the year-fraction pricers (dated pricers use a YieldCurve)
	reference_date: w.Date, // dated curves: tenors are ACT/365F years from this date
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

// ============================================================================
// STANDARD CDS ON DATES (ISDA STANDARD MODEL)
// ============================================================================
// Post-2009 standard contract: quarterly coupons on the IMM 20th (CDS / CDS2015
// schedule, weekends-only calendar, following), ACT/360 accrual with the last
// period including its end date, full first coupon with the accrued part
// rebated to the buyer at cash settlement, protection from the step-in date
// (trade + 1). Valued as QuantLib's IsdaCdsEngine (ISDA CDS Standard Model:
// flat forwards and hazards between the nodes of both curves, Taylor fix,
// half-day accrual bias, piecewise forwards within coupon periods).

CDSSide :: enum u8 {
	Buyer, // buys protection, pays the coupons
	Seller,
}

DatedCDS :: struct {
	side:             CDSSide,
	notional:         f64,
	coupon:           f64, // running coupon, e.g. 0.01 (100bp)
	upfront:          f64, // paid by the buyer on upfront_date, fraction of notional
	trade_date:       w.Date,
	protection_start: w.Date,
	upfront_date:     w.Date, // cash settlement: trade date + 3 business days
	schedule:         Schedule,
}

DatedCDSResult :: struct {
	npv:            f64, // to the side
	protection_leg: f64, // PV of (1 − R) × notional paid at default (positive)
	premium_leg:    f64, // PV of the coupons plus the premium accrued at default (positive)
	accrual_rebate: f64, // PV of the accrued coupon returned to the buyer (positive)
	upfront_pv:     f64, // PV of the upfront payment (positive for a positive upfront)
	fair_spread:    f64, // running coupon with zero value and no upfront
	fair_upfront:   f64, // upfront (fraction of notional) with zero value at this coupon; NaN once the upfront date has passed
	risky_annuity:  f64, // premium leg per unit notional and unit coupon
}

// Standard CDS traded on trade_date for `tenor` (maturity cds_maturity_date);
// free with free_dated_cds
make_standard_cds :: proc(
	trade_date: w.Date,
	tenor: Period,
	coupon: f64,
	notional: f64 = 1.0,
	side: CDSSide = .Buyer,
	upfront: f64 = 0.0,
	rule: DateGenerationRule = .CDS2015,
	cash_settlement_days: int = 3,
	allocator := context.allocator,
) -> DatedCDS {
	cal := Calendar{.Weekends}
	end := cds_maturity_date(trade_date, tenor, rule)
	return DatedCDS {
		side = side,
		notional = notional,
		coupon = coupon,
		upfront = upfront,
		trade_date = trade_date,
		protection_start = trade_date,
		upfront_date = calendar_advance(cal, trade_date, cash_settlement_days, .Days),
		schedule = make_schedule(
			trade_date,
			end,
			{3, .Months},
			cal,
			.Following,
			.Unadjusted,
			rule,
			allocator = allocator,
		),
	}
}

free_dated_cds :: proc(cds: ^DatedCDS, allocator := context.allocator) {
	free_schedule(&cds.schedule, allocator)
}

// Coupon k: accrual dates, payment date and amount (the last period counts its
// end date too)
@(private = "file")
_cds_coupon :: proc(cds: DatedCDS, k: int) -> (start, end, pay: i64, amount: f64) {
	s := cds.schedule
	start, end = _dn(s.dates[k]), _dn(s.dates[k + 1])
	pay = _dn(calendar_adjust(s.calendar, s.dates[k + 1], .Following))
	days := end - start + (1 if k == len(s.dates) - 2 else 0)
	amount = cds.notional * cds.coupon * f64(days) / 360.0
	return
}

// Node dates (day numbers) of both curves, sorted and unique
@(private = "file")
_cds_nodes :: proc(hazard: ^CDS_Curve, discount: ^YieldCurve, allocator: mem.Allocator) -> []i64 {
	nodes := make([dynamic]i64, 0, len(hazard.tenors) + len(discount.tenors) + 2, allocator)
	add :: proc(nodes: ^[dynamic]i64, ref: w.Date, t: f64) {
		append(nodes, _dn(ref) + i64(math.round(t * 365.0)))
	}
	add(&nodes, discount.reference_date, 0)
	for t in discount.tenors {add(&nodes, discount.reference_date, t)}
	add(&nodes, hazard.reference_date, 0)
	for t in hazard.tenors {add(&nodes, hazard.reference_date, t)}
	slice.sort(nodes[:])
	k := 0
	for n, i in nodes {
		if i == 0 || n != nodes[k - 1] {nodes[k] = n; k += 1}
	}
	resize(&nodes, k)
	return nodes[:]
}

// ∫ over (t0, t1] of D(u) dF(u) with flat forward f and hazard h, given the
// values at both ends (ISDA's closed form, Taylor-expanded when f + h is small)
@(private = "file")
_isda_protection_piece :: proc(P0, Q0, P1, Q1: f64) -> f64 {
	fhat := math.ln(P0) - math.ln(P1)
	hhat := math.ln(Q0) - math.ln(Q1)
	x := fhat + hhat
	if x < 1e-4 {
		x2 := x * x
		return P0 * Q0 * hhat * (1.0 - 0.5 * x + x2 / 6.0 - x2 * x / 24.0 + x2 * x2 / 120.0)
	}
	return hhat / x * (P0 * Q0 - P1 * Q1)
}

// ∫ over (t0, t1] of (u − t_start) D(u) dF(u), the accrued premium at default
@(private = "file")
_isda_accrual_piece :: proc(t0, t1, tstart, P0, Q0, P1, Q1: f64) -> f64 {
	fhat := math.ln(P0) - math.ln(P1)
	hhat := math.ln(Q0) - math.ln(Q1)
	x := fhat + hhat
	if x < 1e-4 {
		x2 := x * x
		return(
			hhat *
			P0 *
			Q0 *
			((t0 - tstart) * (1.0 - 0.5 * x + x2 / 6.0 - x2 * x / 24.0) +
					(t1 - t0) * (0.5 - x / 3.0 + x2 / 8.0 - x2 * x / 30.0)) \
		)
	}
	return(
		(hhat / x) *
		((t1 - t0) * ((P0 * Q0 - P1 * Q1) / x - P1 * Q1) + (t0 - tstart) * (P0 * Q0 - P1 * Q1)) \
	)
}

// Values the CDS on a hazard curve and a discount curve sharing the valuation
// date as reference_date
price_dated_cds :: proc(
	cds: DatedCDS,
	hazard: ^CDS_Curve,
	discount: ^YieldCurve,
) -> DatedCDSResult {
	ref := _dn(discount.reference_date)
	href := _dn(hazard.reference_date)
	P :: proc(c: ^YieldCurve, ref, n: i64) -> f64 {return yield_curve_discount_factor(
			c,
			f64(n - ref) / 365.0,
		)}
	Q :: proc(c: ^CDS_Curve, ref, n: i64) -> f64 {return get_survival_prob(
			f64(n - ref) / 365.0,
			c^,
		)}
	T :: proc(ref, n: i64) -> f64 {return f64(n - ref) / 365.0}

	nodes := _cds_nodes(hazard, discount, context.temp_allocator)
	defer delete(nodes, context.temp_allocator)
	eps := max(_dn(cds.protection_start), ref + 1) // step-in date
	sd := cds.schedule.dates
	maturity := _dn(sd[len(sd) - 1])

	// protection leg from step-in to maturity
	protection := 0.0
	d0 := eps - 1
	P0, Q0 := P(discount, ref, d0), Q(hazard, href, d0)
	for n in nodes {
		if n <= eps {continue}
		d1 := min(n, maturity)
		P1, Q1 := P(discount, ref, d1), Q(hazard, href, d1)
		protection += _isda_protection_piece(P0, Q0, P1, Q1)
		d0, P0, Q0 = d1, P1, Q1
		if d1 == maturity {break}
	}
	if d0 < maturity { 	// curves ending before maturity: last piece to maturity
		protection += _isda_protection_piece(
			P0,
			Q0,
			P(discount, ref, maturity),
			Q(hazard, href, maturity),
		)
	}
	protection *= cds.notional * (1.0 - hazard.recovery)

	// premium leg: coupons (survival to the day before payment) and the premium
	// accrued at default within each period
	coupons, accrual := 0.0, 0.0
	for k in 0 ..< len(sd) - 1 {
		start, end, pay, amount := _cds_coupon(cds, k)
		if pay >= eps {
			coupons += amount * P(discount, ref, pay) * Q(hazard, href, pay - 1)
		}
		if end <= eps {continue}
		a := max(start, eps) - 1
		b := pay - 1
		tstart := T(ref, start - 1) - 1.0 / 730.0 // half-day bias
		t0, Pa, Qa := T(ref, a), P(discount, ref, a), Q(hazard, href, a)
		piece := 0.0
		for n in nodes {
			if n <= a || n >= b {continue}
			t1, Pn, Qn := T(ref, n), P(discount, ref, n), Q(hazard, href, n)
			piece += _isda_accrual_piece(t0, t1, tstart, Pa, Qa, Pn, Qn)
			t0, Pa, Qa = t1, Pn, Qn
		}
		piece += _isda_accrual_piece(
			t0,
			T(ref, b),
			tstart,
			Pa,
			Qa,
			P(discount, ref, b),
			Q(hazard, href, b),
		)
		accrual += piece * cds.notional * cds.coupon * 365.0 / 360.0
	}
	premium := coupons + accrual

	// accrued coupon from the start of the current period to step-in (trade + 1),
	// rebated with the upfront at cash settlement
	up := _dn(cds.upfront_date)
	rebate, upfront_pv := 0.0, 0.0
	trade := _dn(cds.trade_date)
	if up >= ref {
		df_up := P(discount, ref, up)
		upfront_pv = cds.upfront * cds.notional * df_up
		if trade >= _dn(sd[0]) {
			step_in := trade + 1
			for k in 0 ..< len(sd) - 1 {
				start, end, pay, _ := _cds_coupon(cds, k)
				if step_in > pay {continue}
				if step_in < pay && step_in > start {
					days := min(step_in, end) - start + (1 if k == len(sd) - 2 else 0)
					rebate = cds.notional * cds.coupon * f64(days) / 360.0 * df_up
				}
				break
			}
		}
	}

	buyer := protection - premium - upfront_pv + rebate
	risky := premium / (cds.notional * cds.coupon) if cds.coupon != 0.0 else 0.0
	df_up := P(discount, ref, up)
	return DatedCDSResult {
		npv = buyer if cds.side == .Buyer else -buyer,
		protection_leg = protection,
		premium_leg = premium,
		accrual_rebate = rebate,
		upfront_pv = upfront_pv,
		fair_spread = protection * cds.coupon / (premium - rebate) if premium - rebate != 0.0 else 0.0,
		fair_upfront = (protection - premium + rebate) / (cds.notional * df_up) if up >= ref else math.QNAN_F64,
		risky_annuity = risky,
	}
}

// Hazard bootstrap shared by the spread and upfront quotes: pillar i is the
// payment date of the i-th maturity + 1 day, and its hazard is solved so that
// the CDS reprices the quote (fair spread, or fair upfront at `coupon`)
@(private = "file")
_bootstrap_cds_dated :: proc(
	tenors: []Period,
	quotes: []f64,
	coupon: f64, // running coupon of upfront quotes
	by_upfront: bool,
	recovery: f64,
	discount: ^YieldCurve,
	rule: DateGenerationRule,
	upfront_settlement_days: int,
	allocator: mem.Allocator,
) -> CDS_Curve {
	n := min(len(tenors), len(quotes))
	ref := discount.reference_date
	curve := CDS_Curve {
		tenors         = make([]f64, n, allocator),
		hazards        = make([]f64, n, allocator),
		recovery       = recovery,
		reference_date = ref,
	}
	Quote :: struct {
		cds:        DatedCDS,
		value:      f64,
		by_upfront: bool,
		discount:   ^YieldCurve,
	}
	for i in 0 ..< n {
		q := Quote {
			make_standard_cds(
				ref,
				tenors[i],
				coupon if by_upfront else quotes[i],
				rule = rule,
				cash_settlement_days = upfront_settlement_days,
				allocator = context.temp_allocator,
			),
			quotes[i],
			by_upfront,
			discount,
		}
		defer free_dated_cds(&q.cds, context.temp_allocator)
		mat := q.cds.schedule.dates[len(q.cds.schedule.dates) - 1]
		pillar := _dn(calendar_adjust(q.cds.schedule.calendar, mat, .Following)) + 1
		curve.tenors[i] = f64(pillar - _dn(ref)) / 365.0
		// implied quote − quote, increasing in the hazard
		f :: proc(c: ^CDS_Curve, i: int, h: f64, q: ^Quote) -> f64 {
			c.hazards[i] = h
			sub := CDS_Curve {
				c.tenors[:i + 1],
				c.hazards[:i + 1],
				c.recovery,
				c.r,
				c.reference_date,
			}
			r := price_dated_cds(q.cds, &sub, q.discount)
			return (r.fair_upfront if q.by_upfront else r.fair_spread) - q.value
		}
		lo, hi := 0.0, 1.0
		for f(&curve, i, hi, &q) < 0.0 && hi < 1e3 {hi *= 2.0}
		for _ in 0 ..< 200 {
			mid := 0.5 * (lo + hi)
			if mid == lo || mid == hi {break}
			if f(&curve, i, mid, &q) < 0.0 {lo = mid} else {hi = mid}
		}
		curve.hazards[i] = 0.5 * (lo + hi)
	}
	return curve
}

// Bootstraps piecewise-flat hazard rates from par spreads of standard CDS
// traded on the discount curve's reference date (QuantLib SpreadCdsHelper with
// the ISDA model: each pillar is the maturity's payment date + 1 day, hazards
// flat in ACT/365F time and solved so that every quote is the fair spread).
// Tenors must be increasing.
bootstrap_cds_curve_dated :: proc(
	tenors: []Period,
	spreads: []f64,
	recovery: f64,
	discount: ^YieldCurve,
	rule: DateGenerationRule = .CDS2015,
	allocator: mem.Allocator = context.allocator,
) -> CDS_Curve {
	return _bootstrap_cds_dated(
		tenors,
		spreads,
		0.0,
		false,
		recovery,
		discount,
		rule,
		3,
		allocator,
	)
}

// As bootstrap_cds_curve_dated from upfront quotes (fraction of notional paid
// by the protection buyer, settled upfront_settlement_days business days after
// trade) at a common running coupon (QuantLib UpfrontCdsHelper, ISDA model)
bootstrap_cds_curve_dated_upfront :: proc(
	tenors: []Period,
	upfronts: []f64,
	coupon: f64,
	recovery: f64,
	discount: ^YieldCurve,
	rule: DateGenerationRule = .CDS2015,
	upfront_settlement_days: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> CDS_Curve {
	return _bootstrap_cds_dated(
		tenors,
		upfronts,
		coupon,
		true,
		recovery,
		discount,
		rule,
		upfront_settlement_days,
		allocator,
	)
}

// ============================================================================
// QUOTED SPREAD <-> UPFRONT AND CDS INDICES
// ============================================================================

// Points upfront of a standard CDS with running `coupon` quoted at a (flat)
// spread: the ISDA convention values it on the flat hazard curve that makes
// the quoted spread the par spread of the contract's maturity. Trade date =
// the discount curve's reference date.
cds_upfront_from_quoted_spread :: proc(
	tenor: Period,
	coupon, quoted_spread, recovery: f64,
	discount: ^YieldCurve,
	rule: DateGenerationRule = .CDS2015,
) -> f64 {
	tenors := [1]Period{tenor}
	spreads := [1]f64{quoted_spread}
	flat := bootstrap_cds_curve_dated(
		tenors[:],
		spreads[:],
		recovery,
		discount,
		rule,
		context.temp_allocator,
	)
	defer {delete(flat.tenors, context.temp_allocator)
		delete(flat.hazards, context.temp_allocator)}
	cds := make_standard_cds(
		discount.reference_date,
		tenor,
		coupon,
		rule = rule,
		allocator = context.temp_allocator,
	)
	defer free_dated_cds(&cds, context.temp_allocator)
	return price_dated_cds(cds, &flat, discount).fair_upfront
}

// Inverse of cds_upfront_from_quoted_spread: the par spread of the flat hazard
// curve that reprices the upfront at the running coupon
cds_quoted_spread_from_upfront :: proc(
	tenor: Period,
	coupon, upfront, recovery: f64,
	discount: ^YieldCurve,
	rule: DateGenerationRule = .CDS2015,
) -> f64 {
	tenors := [1]Period{tenor}
	upfronts := [1]f64{upfront}
	flat := bootstrap_cds_curve_dated_upfront(
		tenors[:],
		upfronts[:],
		coupon,
		recovery,
		discount,
		rule,
		allocator = context.temp_allocator,
	)
	defer {delete(flat.tenors, context.temp_allocator)
		delete(flat.hazards, context.temp_allocator)}
	cds := make_standard_cds(
		discount.reference_date,
		tenor,
		coupon,
		rule = rule,
		allocator = context.temp_allocator,
	)
	defer free_dated_cds(&cds, context.temp_allocator)
	return price_dated_cds(cds, &flat, discount).fair_spread
}

// Intrinsic value of a CDS index position (CDX / iTraxx): the index contract
// (common coupon and maturity) valued name by name on each constituent's
// hazard curve with notional × weight and summed. Defaulted names are left out,
// so the weights of the remaining names add up to the index factor; the
// upfront is paid on the remaining notional. Fair spread, fair upfront and
// risky annuity refer to the remaining notional.
price_cds_index_intrinsic :: proc(
	index: DatedCDS,
	constituents: []^CDS_Curve,
	weights: []f64,
	discount: ^YieldCurve,
) -> DatedCDSResult {
	res: DatedCDSResult
	factor := 0.0
	for curve, i in constituents {
		name := index
		name.notional = index.notional * weights[i]
		r := price_dated_cds(name, curve, discount)
		res.npv += r.npv
		res.protection_leg += r.protection_leg
		res.premium_leg += r.premium_leg
		res.accrual_rebate += r.accrual_rebate
		res.upfront_pv += r.upfront_pv
		factor += weights[i]
	}
	live := index.notional * factor
	res.fair_spread = res.protection_leg * index.coupon / (res.premium_leg - res.accrual_rebate)
	res.risky_annuity = res.premium_leg / (live * index.coupon) if index.coupon != 0.0 else 0.0
	ref := discount.reference_date
	if w.date_less(index.upfront_date, ref) {
		res.fair_upfront = math.QNAN_F64
	} else {
		df_up := yield_curve_discount_factor_date(discount, index.upfront_date)
		res.fair_upfront =
			(res.protection_leg - res.premium_leg + res.accrual_rebate) / (live * df_up)
	}
	return res
}
