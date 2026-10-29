package finance

import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Copula Models for Multivariate Dependence
// ============================================================================
// Bivariate copulas on pseudo-observations u1, u2 ∈ (0, 1). Log-likelihoods skip
// observations outside the open unit square. All fits are maximum likelihood,
// so log_likelihood / AIC / BIC are comparable across families.
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

// Parameter ranges searched by the fits
@(private = "file")
T_COPULA_NU_MIN :: f64(1.0)
@(private = "file")
T_COPULA_NU_MAX :: f64(200.0) // beyond this the t copula is practically Gaussian
@(private = "file")
ARCH_THETA_MAX :: f64(200.0)

// ============================================================================
// Probability Integral Transform (PIT)
// ============================================================================

// Pseudo-observations rank/(n+1); tied values get their average rank
// (scipy.stats.rankdata(method="average") / (n+1)). O(n log n).
pit_empirical :: proc(data: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(data)
	u := make([]f64, n, allocator)
	if n == 0 {return u}

	order := _sorted_order(data)
	defer delete(order, context.temp_allocator)

	i := 0
	for i < n {
		j := i + 1
		for j < n && data[order[j]] == data[order[i]] {j += 1}
		// ranks i+1 ..= j share their mean (i + 1 + j)/2
		r := 0.5 * f64(i + 1 + j) / f64(n + 1)
		for k in i ..< j {u[order[k]] = r}
		i = j
	}
	return u
}

// Φ(x): for data that is already standard normal
pit_normal :: proc(data: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	u := make([]f64, len(data), allocator)
	for x, i in data {u[i] = _bs_cdf(x)}
	return u
}

// Indices that sort `data` ascending (temp allocator)
@(private = "file")
_sorted_order :: proc(data: []f64) -> []int {
	Item :: struct {
		v: f64,
		i: int,
	}
	n := len(data)
	items := make([]Item, n, context.temp_allocator)
	defer delete(items, context.temp_allocator)
	for v, i in data {items[i] = {v, i}}
	slice.sort_by(items, proc(a, b: Item) -> bool {return a.v < b.v})
	order := make([]int, n, context.temp_allocator)
	for it, k in items {order[k] = it.i}
	return order
}

@(private = "file")
_in_unit :: #force_inline proc(u1, u2: f64) -> bool {
	return u1 > 0.0 && u1 < 1.0 && u2 > 0.0 && u2 < 1.0
}

// ============================================================================
// Gaussian Copula
// ============================================================================
// ln c = -½ln(1-ρ²) - (ρ²(x1²+x2²) - 2ρx1x2) / (2(1-ρ²)),  x = Φ⁻¹(u)
// The log-likelihood depends on the data only through n, S = Σ(x1²+x2²) and
// P = Σx1x2, so the fit needs one pass of quantiles.

gaussian_copula_pdf :: proc(u1: f64, u2: f64, rho: f64) -> f64 {
	if !(math.abs(rho) < 1.0) || !_in_unit(u1, u2) {return 0.0}
	x1, x2 := _bs_ppf(u1), _bs_ppf(u2)
	return math.exp(_gauss_ll(1, x1 * x1 + x2 * x2, x1 * x2, rho))
}

gaussian_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64) -> f64 {
	if !(math.abs(rho) < 1.0) {return -math.INF_F64}
	n, S, P := _gauss_stats(u1, u2)
	return _gauss_ll(n, S, P, rho)
}

@(private = "file")
_gauss_stats :: proc(u1, u2: []f64) -> (n: f64, S: f64, P: f64) {
	for i in 0 ..< min(len(u1), len(u2)) {
		if !_in_unit(u1[i], u2[i]) {continue}
		x1, x2 := _bs_ppf(u1[i]), _bs_ppf(u2[i])
		n += 1
		S += x1 * x1 + x2 * x2
		P += x1 * x2
	}
	return
}

@(private = "file")
_gauss_ll :: proc(n, S, P, rho: f64) -> f64 {
	d := 1.0 - rho * rho
	return -0.5 * n * math.ln(d) - (rho * rho * S - 2.0 * rho * P) / (2.0 * d)
}

// ============================================================================
// Student-t Copula
// ============================================================================
// ln c = lnΓ((ν+2)/2) + lnΓ(ν/2) - 2lnΓ((ν+1)/2) - ½ln(1-ρ²)
//        - (ν+2)/2 · ln(1 + Q/ν) + (ν+1)/2 · Σ_k ln(1 + x_k²/ν),
// Q = (x1² + x2² - 2ρx1x2)/(1-ρ²), x = t_ν⁻¹(u). Valid for any ν > 0.

student_t_copula_log_pdf :: proc(u1: f64, u2: f64, rho: f64, nu: f64) -> f64 {
	if !(math.abs(rho) < 1.0) || !(nu > 0.0) || !_in_unit(u1, u2) {return -math.INF_F64}
	x1 := []f64{t_quantile(u1, nu)}
	x2 := []f64{t_quantile(u2, nu)}
	return _t_ll(x1, x2, rho, nu)
}

student_t_copula_loglik :: proc(u1: []f64, u2: []f64, rho: f64, nu: f64) -> f64 {
	if !(math.abs(rho) < 1.0) || !(nu > 0.0) {return -math.INF_F64}
	x1, x2 := _t_scores(u1, u2, nu)
	defer {
		delete(x1, context.temp_allocator)
		delete(x2, context.temp_allocator)
	}
	return _t_ll(x1, x2, rho, nu)
}

// t quantiles of the valid observations (temp allocator)
@(private = "file")
_t_scores :: proc(u1, u2: []f64, nu: f64) -> (x1, x2: []f64) {
	n := min(len(u1), len(u2))
	x1 = make([]f64, n, context.temp_allocator)
	x2 = make([]f64, n, context.temp_allocator)
	m := 0
	for i in 0 ..< n {
		if !_in_unit(u1[i], u2[i]) {continue}
		x1[m], x2[m] = t_quantile(u1[i], nu), t_quantile(u2[i], nu)
		m += 1
	}
	return x1[:m], x2[:m]
}

@(private = "file")
_t_const :: proc(nu: f64) -> f64 {
	lg2, _ := math.lgamma(0.5 * (nu + 2.0))
	lg1, _ := math.lgamma(0.5 * (nu + 1.0))
	lg0, _ := math.lgamma(0.5 * nu)
	return lg2 + lg0 - 2.0 * lg1
}

@(private = "file")
_t_ll :: proc(x1, x2: []f64, rho, nu: f64) -> f64 {
	d := 1.0 - rho * rho
	ll := f64(len(x1)) * (_t_const(nu) - 0.5 * math.ln(d))
	for i in 0 ..< len(x1) {
		a, b := x1[i], x2[i]
		Q := (a * a + b * b - 2.0 * rho * a * b) / d
		ll +=
			-0.5 * (nu + 2.0) * math.log1p(Q / nu) +
			0.5 * (nu + 1.0) * (math.log1p(a * a / nu) + math.log1p(b * b / nu))
	}
	return ll
}

// ============================================================================
// Copula Fitting (maximum likelihood)
// ============================================================================

// ρ̂ maximizes ℓ(ρ) = -n/2·ln(1-ρ²) - (ρ²S - 2ρP)/(2(1-ρ²)); ℓ'(ρ) has the sign of
// the cubic g(ρ) = -nρ³ + Pρ² + (n - S)ρ + P, with g(-1) = Σ(x1+x2)² ≥ 0 and
// g(1) = -Σ(x1-x2)² ≤ 0. Every + → - sign change of g on (-1, 1) is a local
// maximum; they are bracketed on a grid, bisected, and the best is taken.
fit_gaussian_copula :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result := CopulaResult {
		copula_type = .Gaussian,
		n_obs       = min(len(u1), len(u2)),
	}
	n, S, P := _gauss_stats(u1, u2)
	g :: proc(n, S, P, r: f64) -> f64 {return ((-n * r + P) * r + (n - S)) * r + P}

	best_rho, best_ll := 0.0, _gauss_ll(n, S, P, 0.0)
	LIM :: 1.0 - 1e-12
	N_GRID :: 400
	lo, g_lo := -LIM, g(n, S, P, -LIM)
	for k in 1 ..= N_GRID {
		hi := -LIM + 2.0 * LIM * f64(k) / N_GRID
		g_hi := g(n, S, P, hi)
		if g_lo > 0.0 && g_hi <= 0.0 {
			a, b := lo, hi
			for _ in 0 ..< 100 {
				mid := 0.5 * (a + b)
				if mid == a || mid == b {break}
				if g(n, S, P, mid) > 0.0 {a = mid} else {b = mid}
			}
			r := 0.5 * (a + b)
			if ll := _gauss_ll(n, S, P, r); ll > best_ll {best_rho, best_ll = r, ll}
		}
		lo, g_lo = hi, g_hi
	}

	result.parameters = make([]f64, 1, allocator)
	result.parameters[0] = best_rho
	_set_info(&result, best_ll, 1)
	return result
}

// Profile likelihood: for each ν the scores x = t_ν⁻¹(u) are computed once and
// ρ̂(ν) is found by a 1-D search on atanh ρ (each step O(n), no quantiles); the
// outer 1-D search is on ln ν over [1, 200].
fit_student_t_copula :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	result := CopulaResult {
		copula_type = .StudentT,
		n_obs       = min(len(u1), len(u2)),
	}

	Ctx :: struct {
		u1, u2, x1, x2:  []f64,
		nu, rho:         f64,
	}
	ctx := Ctx {
		u1 = u1,
		u2 = u2,
	}

	inner :: proc(z: f64, data: rawptr) -> f64 {
		c := (^Ctx)(data)
		return _t_ll(c.x1, c.x2, math.tanh(z), c.nu)
	}
	outer :: proc(s: f64, data: rawptr) -> f64 {
		c := (^Ctx)(data)
		c.nu = math.exp(s)
		c.x1, c.x2 = _t_scores(c.u1, c.u2, c.nu)
		defer {
			delete(c.x1, context.temp_allocator)
			delete(c.x2, context.temp_allocator)
		}
		z, ll := _maximize_1d(inner, c, -5.0, 5.0, 20, 1e-10)
		c.rho = math.tanh(z)
		return ll
	}

	s, best_ll := _maximize_1d(outer, &ctx, math.ln(T_COPULA_NU_MIN), math.ln(T_COPULA_NU_MAX), 10, 1e-8)
	outer(s, &ctx) // ρ̂ at the chosen ν

	result.parameters = make([]f64, 2, allocator)
	result.parameters[0] = ctx.rho
	result.parameters[1] = math.exp(s)
	_set_info(&result, best_ll, 2)
	return result
}

@(private = "file")
_set_info :: proc(r: ^CopulaResult, ll: f64, k: f64) {
	r.log_likelihood = ll
	r.aic = 2.0 * k - 2.0 * ll
	r.bic = k * math.ln(f64(r.n_obs)) - 2.0 * ll
	r.converged = !math.is_nan(ll) && !math.is_inf(ll)
}

// Maximizes f on [lo, hi]: grid scan, then Brent's method (parabolic steps with
// golden-section fallback) on the bracket around the best grid point down to `tol`.
@(private = "file")
_maximize_1d :: proc(
	f: proc(x: f64, data: rawptr) -> f64,
	data: rawptr,
	lo, hi: f64,
	n_grid: int,
	tol: f64,
) -> (
	x_best: f64,
	f_best: f64,
) {
	h := (hi - lo) / f64(n_grid)
	i_best := 0
	f_best = -math.INF_F64
	for i in 0 ..= n_grid {
		if fx := f(lo + f64(i) * h, data); fx > f_best {f_best, i_best = fx, i}
	}
	x_best = lo + f64(i_best) * h
	if math.is_inf(f_best) || math.is_nan(f_best) {return}
	a := max(lo, x_best - h)
	b := min(hi, x_best + h)

	// Brent on g = -f (Numerical Recipes `brent`)
	CGOLD :: 0.3819660112501051
	x, w, v := x_best, x_best, x_best
	fx, fw, fv := -f_best, -f_best, -f_best
	d, e := 0.0, 0.0
	for _ in 0 ..< 200 {
		xm := 0.5 * (a + b)
		tol2 := 2.0 * tol
		if math.abs(x - xm) <= tol2 - 0.5 * (b - a) {break}
		golden := true
		if math.abs(e) > tol && !math.is_inf(fv) && !math.is_inf(fw) {
			r := (x - w) * (fx - fv)
			q := (x - v) * (fx - fw)
			p := (x - v) * q - (x - w) * r
			q = 2.0 * (q - r)
			if q > 0.0 {p = -p}
			q = math.abs(q)
			e_prev := e
			e = d
			if math.abs(p) < math.abs(0.5 * q * e_prev) && p > q * (a - x) && p < q * (b - x) {
				d = p / q
				golden = false
				if u := x + d; u - a < tol2 || b - u < tol2 {d = tol if xm >= x else -tol}
			}
		}
		if golden {
			e = a - x if x >= xm else b - x
			d = CGOLD * e
		}
		u := x + d if math.abs(d) >= tol else x + (tol if d >= 0 else -tol)
		fu := -f(u, data)
		if fu <= fx {
			if u >= x {a = x} else {b = x}
			v, w, x = w, x, u
			fv, fw, fx = fw, fx, fu
		} else {
			if u < x {a = u} else {b = u}
			if fu <= fw || w == x {
				v, w = w, u
				fv, fw = fw, fu
			} else if fu <= fv || v == x || v == w {
				v, fv = u, fu
			}
		}
	}
	return x, -fx
}

// ============================================================================
// Empirical Dependence Measures
// ============================================================================

// Kendall's τ_b (tie-corrected, as scipy.stats.kendalltau), Knight's
// O(n log n) algorithm: sort by (x, y), count discordant pairs as inversions of
// y with a merge sort.
kendall_tau :: proc(x: []f64, y: []f64) -> f64 {
	n := min(len(x), len(y))
	if n < 2 {return 0.0}

	Pair :: struct {
		x, y: f64,
	}
	pairs := make([]Pair, n, context.temp_allocator)
	defer delete(pairs, context.temp_allocator)
	for i in 0 ..< n {pairs[i] = {x[i], y[i]}}
	slice.sort_by(pairs, proc(a, b: Pair) -> bool {return a.x < b.x || (a.x == b.x && a.y < b.y)})

	// Pairs tied in x (n1) and in both (n3)
	n1, n3: i64
	i := 0
	for i < n {
		j := i + 1
		for j < n && pairs[j].x == pairs[i].x {j += 1}
		t := i64(j - i)
		n1 += t * (t - 1) / 2
		k := i
		for k < j {
			m := k + 1
			for m < j && pairs[m].y == pairs[k].y {m += 1}
			s := i64(m - k)
			n3 += s * (s - 1) / 2
			k = m
		}
		i = j
	}

	ys := make([]f64, n, context.temp_allocator)
	buf := make([]f64, n, context.temp_allocator)
	defer {
		delete(ys, context.temp_allocator)
		delete(buf, context.temp_allocator)
	}
	for p, k in pairs {ys[k] = p.y}
	discordant := _count_inversions(ys, buf)

	// Pairs tied in y (ys is now sorted)
	n2: i64
	i = 0
	for i < n {
		j := i + 1
		for j < n && ys[j] == ys[i] {j += 1}
		t := i64(j - i)
		n2 += t * (t - 1) / 2
		i = j
	}

	n0 := i64(n) * i64(n - 1) / 2
	den := math.sqrt(f64(n0 - n1) * f64(n0 - n2))
	if den == 0.0 {return 0.0}
	return f64(n0 - n1 - n2 + n3 - 2 * discordant) / den
}

// Sorts `a` (bottom-up merge sort) and returns the number of pairs i < j with
// a[i] > a[j]
@(private = "file")
_count_inversions :: proc(a, buf: []f64) -> (inv: i64) {
	n := len(a)
	for width := 1; width < n; width *= 2 {
		for lo := 0; lo < n - width; lo += 2 * width {
			mid, hi := lo + width, min(lo + 2 * width, n)
			i, j, k := lo, mid, lo
			for i < mid && j < hi {
				if a[j] < a[i] {
					buf[k] = a[j]
					j += 1
					inv += i64(mid - i)
				} else {
					buf[k] = a[i]
					i += 1
				}
				k += 1
			}
			for ; i < mid; i += 1 {buf[k] = a[i]; k += 1}
			for ; j < hi; j += 1 {buf[k] = a[j]; k += 1}
			copy(a[lo:hi], buf[lo:hi])
		}
	}
	return
}

// Spearman's ρ: Pearson correlation of the (average) ranks
spearman_rho :: proc(x: []f64, y: []f64, allocator: mem.Allocator) -> f64 {
	n := min(len(x), len(y))
	if n < 2 {return 0.0}

	rank_x := pit_empirical(x[:n], allocator)
	rank_y := pit_empirical(y[:n], allocator)
	defer {
		delete(rank_x, allocator)
		delete(rank_y, allocator)
	}

	mean_x, mean_y := 0.0, 0.0
	for i in 0 ..< n {
		mean_x += rank_x[i]
		mean_y += rank_y[i]
	}
	mean_x /= f64(n)
	mean_y /= f64(n)

	cov, var_x, var_y := 0.0, 0.0, 0.0
	for i in 0 ..< n {
		dx := rank_x[i] - mean_x
		dy := rank_y[i] - mean_y
		cov += dx * dy
		var_x += dx * dx
		var_y += dy * dy
	}
	if var_x == 0.0 || var_y == 0.0 {return 0.0}
	return cov / math.sqrt(var_x * var_y)
}

// ============================================================================
// Tail Dependence (empirical, at a finite threshold)
// ============================================================================

// P(U2 < t | U1 < t)
lower_tail_dependence :: proc(u1: []f64, u2: []f64, threshold: f64 = 0.05) -> f64 {
	count_u1_low, count_both_low := 0, 0
	for i in 0 ..< min(len(u1), len(u2)) {
		if u1[i] < threshold {
			count_u1_low += 1
			if u2[i] < threshold {count_both_low += 1}
		}
	}
	if count_u1_low == 0 {return 0.0}
	return f64(count_both_low) / f64(count_u1_low)
}

// P(U2 > t | U1 > t)
upper_tail_dependence :: proc(u1: []f64, u2: []f64, threshold: f64 = 0.95) -> f64 {
	count_u1_high, count_both_high := 0, 0
	for i in 0 ..< min(len(u1), len(u2)) {
		if u1[i] > threshold {
			count_u1_high += 1
			if u2[i] > threshold {count_both_high += 1}
		}
	}
	if count_u1_high == 0 {return 0.0}
	return f64(count_both_high) / f64(count_u1_high)
}

// ============================================================================
// Clayton Copula (lower tail dependence), θ ∈ (-1, ∞), θ = 0 independence
// ============================================================================
// ln c = ln(1+θ) - (1+θ)(ln u1 + ln u2) - (2 + 1/θ)·ln(u1^-θ + u2^-θ - 1)

clayton_copula_loglik :: proc(u1: []f64, u2: []f64, theta: f64) -> f64 {
	if !(theta > -1.0) {return -math.INF_F64}
	if theta == 0.0 {return 0.0}
	ll := 0.0
	for i in 0 ..< min(len(u1), len(u2)) {
		if !_in_unit(u1[i], u2[i]) {continue}
		l1, l2 := math.ln(u1[i]), math.ln(u2[i])
		// ln(e^{a} + e^{b} - 1) with a = -θ ln u1, b = -θ ln u2; for θ > 0 both
		// are ≥ 0 and the shifted form cannot overflow
		a, b := -theta * l1, -theta * l2
		log_s: f64
		if theta > 0.0 {
			m := max(a, b)
			log_s = m + math.ln(math.exp(a - m) + math.exp(b - m) - math.exp(-m))
		} else {
			s := math.exp(a) + math.exp(b) - 1.0
			if s <= 0.0 {return -math.INF_F64} // outside the support
			log_s = math.ln(s)
		}
		ll += math.log1p(theta) - (1.0 + theta) * (l1 + l2) - (2.0 + 1.0 / theta) * log_s
	}
	return ll
}

// MLE over θ ∈ (0, 200]; requires positive dependence (Kendall τ > 0)
fit_clayton :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	return _fit_archimedean(.Clayton, u1, u2, allocator)
}

// ============================================================================
// Gumbel Copula (upper tail dependence), θ ≥ 1, θ = 1 independence
// ============================================================================
// With a = -ln u1, b = -ln u2, K = a^θ + b^θ:
// ln c = -K^{1/θ} + a + b + (θ-1)(ln a + ln b) + (1/θ - 2)·ln K + ln(θ - 1 + K^{1/θ})

gumbel_copula_loglik :: proc(u1: []f64, u2: []f64, theta: f64) -> f64 {
	if !(theta >= 1.0) {return -math.INF_F64}
	ll := 0.0
	for i in 0 ..< min(len(u1), len(u2)) {
		if !_in_unit(u1[i], u2[i]) {continue}
		a, b := -math.ln(u1[i]), -math.ln(u2[i])
		la, lb := math.ln(a), math.ln(b)
		// ln K without overflow: θ·ln(max) + ln(1 + (min/max)^θ)
		hi, lo := max(la, lb), min(la, lb)
		log_K := theta * hi + math.log1p(math.exp(theta * (lo - hi)))
		K_inv := math.exp(log_K / theta)
		ll +=
			-K_inv +
			a +
			b +
			(theta - 1.0) * (la + lb) +
			(1.0 / theta - 2.0) * log_K +
			math.ln(theta - 1.0 + K_inv)
	}
	return ll
}

// MLE over θ ∈ [1, 200]; requires positive dependence (Kendall τ > 0)
fit_gumbel :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	return _fit_archimedean(.Gumbel, u1, u2, allocator)
}

// ============================================================================
// Frank Copula (no tail dependence), θ ∈ ℝ, θ = 0 independence
// ============================================================================
// c = θ(1-e^{-θ}) e^{-θ(u1+u2)} / D²,  D = (1-e^{-θ}) - (1-e^{-θu1})(1-e^{-θu2})
//   = -e^{-θu1}·expm1(-θu2) - e^{-θu2}·expm1(-θ(1-u2)),
// a sum of two terms of the same sign, so D has no cancellation for any θ.

frank_copula_loglik :: proc(u1: []f64, u2: []f64, theta: f64) -> f64 {
	if math.is_nan(theta) {return math.nan_f64()}
	if theta == 0.0 {return 0.0}
	log_num := math.ln(-theta * _expm1(-theta))
	ll := 0.0
	for i in 0 ..< min(len(u1), len(u2)) {
		u, v := u1[i], u2[i]
		if !_in_unit(u, v) {continue}
		D := -math.exp(-theta * u) * _expm1(-theta * v) - math.exp(-theta * v) * _expm1(-theta * (1.0 - v))
		ll += log_num - theta * (u + v) - 2.0 * math.ln(math.abs(D))
	}
	return ll
}

// MLE over θ ∈ [-200, 200]
fit_frank :: proc(
	u1: []f64,
	u2: []f64,
	allocator: mem.Allocator = context.allocator,
) -> CopulaResult {
	return _fit_archimedean(.Frank, u1, u2, allocator)
}

// One-parameter MLE on a transformed scale: ln θ (Clayton), ln(θ-1) (Gumbel),
// asinh θ (Frank)
@(private = "file")
_fit_archimedean :: proc(
	family: CopulaType,
	u1, u2: []f64,
	allocator: mem.Allocator,
) -> CopulaResult {
	result := CopulaResult {
		copula_type = family,
		n_obs       = min(len(u1), len(u2)),
	}
	if (family == .Clayton || family == .Gumbel) && kendall_tau(u1, u2) <= 0.0 {
		return result // no positive dependence: not converged, no parameters
	}

	Ctx :: struct {
		u1, u2: []f64,
		family: CopulaType,
	}
	to_theta :: proc(family: CopulaType, s: f64) -> f64 {
		#partial switch family {
		case .Clayton:
			return math.exp(s)
		case .Gumbel:
			return 1.0 + math.exp(s)
		}
		return math.sinh(s)
	}
	f :: proc(s: f64, data: rawptr) -> f64 {
		c := (^Ctx)(data)
		theta := to_theta(c.family, s)
		#partial switch c.family {
		case .Clayton:
			return clayton_copula_loglik(c.u1, c.u2, theta)
		case .Gumbel:
			return gumbel_copula_loglik(c.u1, c.u2, theta)
		}
		return frank_copula_loglik(c.u1, c.u2, theta)
	}

	ctx := Ctx{u1, u2, family}
	lo, hi, n_grid: f64 = -math.asinh(ARCH_THETA_MAX), math.asinh(ARCH_THETA_MAX), 24
	#partial switch family {
	case .Clayton:
		lo, hi, n_grid = math.ln(f64(1e-6)), math.ln(ARCH_THETA_MAX), 24
	case .Gumbel:
		lo, hi, n_grid = math.ln(f64(1e-6)), math.ln(ARCH_THETA_MAX - 1.0), 24
	}
	s, ll := _maximize_1d(f, &ctx, lo, hi, int(n_grid), 1e-10)

	result.parameters = make([]f64, 1, allocator)
	result.parameters[0] = to_theta(family, s)
	_set_info(&result, ll, 1)
	return result
}

// e^x - 1 without cancellation for small |x| (Kahan): a few ulps
@(private)
_expm1 :: proc(x: f64) -> f64 {
	u := math.exp(x)
	if u == 1.0 {return x}
	um1 := u - 1.0
	if um1 == -1.0 {return -1.0}
	if math.is_inf(u) {return u}
	return um1 * x / math.ln(u)
}
