package analytics

import l "../linalg"
import ml "./ML"
import "core:math"
import "core:math/rand"
import "core:slice"

// ============================================================================
// Phillips-Perron unit-root test (as arch.unitroot.PhillipsPerron)
// ============================================================================
// Regression y_t = [deterministic terms] + ρ y_{t-1} + u_t, then a non-parametric
// (Newey-West, Bartlett) correction for serial correlation in u:
//   γ₀ = mean(u²), λ² = γ₀ + 2 Σ_{j=1}^{L} (1 - j/(L+1)) γ_j,  s² = SSR / (n - k)
//   Z_τ = sqrt(γ₀/λ²) · (ρ̂ - 1)/se(ρ̂) - ½ (λ² - γ₀)/λ · n se(ρ̂) / s
//   Z_ρ = n (ρ̂ - 1) - ½ n² se(ρ̂)² / s² · (λ² - γ₀)
// H0: unit root. Z_τ uses the ADF (MacKinnon) distribution: p-value and critical
// values as adf_test; Z_ρ is returned without p-value (it needs separate tables).
// lags < 0: ⌈12 (n/100)^¼⌉ (arch's default).

PPTestType :: enum {
	Tau, // Z_τ (default, p-value available)
	Rho, // Z_ρ (statistic only)
}

PPResult :: struct {
	statistic: f64,
	p_value:   f64, // MacKinnon (Z_τ only; NaN for Z_ρ)
	crit:      [3]f64, // 1 %, 5 %, 10 % (Z_τ only)
	lags:      int, // Newey-West lags
	nobs:      int, // regression observations (n - 1)
	rho:       f64, // ρ̂, coefficient of y_{t-1}
}

phillips_perron_test :: proc(
	y: []f64,
	lags: int = -1,
	trend: CointTrend = .Constant,
	test_type: PPTestType = .Tau,
) -> PPResult {
	N := len(y)
	if trend == .ConstantTrendSq do panic("phillips_perron_test: trend must be None, Constant or ConstantTrend")
	L := lags if lags >= 0 else int(math.ceil(12 * math.pow(f64(N) / 100, 0.25)))
	n := N - 1
	kdet := int(trend)
	k := 1 + kdet
	if n <= k + 2 do panic("phillips_perron_test: series too short")
	X := l.matrix_new(f64, n, k, context.temp_allocator)
	defer l.matrix_free(&X)
	yy := y[1:]
	for t in 0 ..< n {
		X.data[t * k] = y[t] // y_{t-1}
		if kdet >= 1 do X.data[t * k + 1] = 1
		if kdet >= 2 do X.data[t * k + 2] = f64(t + 1)
	}
	fit := ml.ols_fit(&X, yy, allocator = context.temp_allocator)
	defer ml._ols_result_free(&fit, context.temp_allocator)
	u := fit.residuals
	nf := f64(n)
	gamma0, ssr := 0.0, 0.0
	for v in u {ssr += v * v}
	gamma0 = ssr / nf
	lam2 := gamma0
	for j in 1 ..= min(L, n - 1) {
		g := 0.0
		for t in j ..< n {g += u[t] * u[t - j]}
		lam2 += 2 * (1 - f64(j) / f64(L + 1)) * g / nf
	}
	s2 := ssr / f64(n - k)
	rho := fit.beta[0]
	se := fit.stderr[0]
	res := PPResult {
		lags = L,
		nobs = n,
		rho  = rho,
	}
	if test_type == .Tau {
		lam := math.sqrt(lam2)
		res.statistic = math.sqrt(gamma0 / lam2) * (rho - 1) / se - 0.5 * ((lam2 - gamma0) / lam) * (nf * se / math.sqrt(s2))
		res.p_value = mackinnon_pvalue(res.statistic, trend, 1)
		res.crit = mackinnon_critical_values(1, trend, n)
	} else {
		res.statistic = nf * (rho - 1) - 0.5 * (nf * nf * se * se / s2) * (lam2 - gamma0)
		res.p_value = math.nan_f64()
		nan := math.nan_f64()
		res.crit = {nan, nan, nan}
	}
	return res
}

// ============================================================================
// Bootstrap confidence intervals (as scipy.stats.bootstrap)
// ============================================================================

BootstrapMethod :: enum {
	BCa, // bias-corrected and accelerated (scipy's default)
	Percentile,
	Basic, // "reverse percentile": 2θ̂ - quantiles
}

BootstrapResample :: enum {
	IID, // resample observations with replacement
	MovingBlock, // Künsch moving-block bootstrap (blocks of block_size consecutive values)
	Stationary, // Politis-Romano stationary bootstrap (geometric blocks, mean block_size)
}

BootstrapResult :: struct {
	low, high:      f64, // confidence interval
	standard_error: f64, // std of the bootstrap distribution (ddof 1)
	theta_hat:      f64, // statistic on the data
}

BootstrapStatistic :: proc(x: []f64, user_data: rawptr) -> f64

// Bootstrap interval of statistic(data): n_resamples resamples (iid or blocks
// for dependent data), then the interval from the bootstrap distribution with
// bootstrap_interval. BCa needs the jackknife (n extra evaluations) and is only
// valid for iid resampling (blocks: use Percentile / Basic). seed = 0: the
// context random generator.
bootstrap_ci :: proc(
	data: []f64,
	statistic: BootstrapStatistic,
	user_data: rawptr = nil,
	n_resamples: int = 9999,
	confidence: f64 = 0.95,
	method: BootstrapMethod = .BCa,
	resample: BootstrapResample = .IID,
	block_size: int = 1,
	seed: u64 = 0,
) -> BootstrapResult {
	n := len(data)
	if n < 2 do panic("bootstrap_ci: need at least 2 observations")
	if method == .BCa && resample != .IID do panic("bootstrap_ci: BCa needs iid resampling (use Percentile or Basic with blocks)")
	if resample != .IID && block_size < 1 do panic("bootstrap_ci: block_size must be >= 1")
	state := rand.create_u64(seed)
	gen := context.random_generator
	if seed != 0 do gen = rand.default_random_generator(&state)
	context.random_generator = gen

	theta_hat := statistic(data, user_data)
	boot := make([]f64, n_resamples, context.temp_allocator)
	sample := make([]f64, n, context.temp_allocator)
	defer {
		delete(boot, context.temp_allocator)
		delete(sample, context.temp_allocator)
	}
	for b in 0 ..< n_resamples {
		switch resample {
		case .IID:
			for i in 0 ..< n {sample[i] = data[rand.int_max(n)]}
		case .MovingBlock:
			i := 0
			for i < n {
				start := rand.int_max(n - block_size + 1)
				for j in 0 ..< block_size {
					if i >= n do break
					sample[i] = data[start + j]
					i += 1
				}
			}
		case .Stationary:
			// new block with probability 1/block_size, else continue circularly
			p := 1 / f64(block_size)
			pos := rand.int_max(n)
			for i in 0 ..< n {
				if i > 0 {
					if rand.float64() < p {pos = rand.int_max(n)} else {pos = (pos + 1) % n}
				}
				sample[i] = data[pos]
			}
		}
		boot[b] = statistic(sample, user_data)
	}
	jack: []f64
	if method == .BCa {
		jack = make([]f64, n, context.temp_allocator)
		loo := make([]f64, n - 1, context.temp_allocator)
		defer delete(loo, context.temp_allocator)
		for i in 0 ..< n {
			copy(loo[:i], data[:i])
			copy(loo[i:], data[i + 1:])
			jack[i] = statistic(loo, user_data)
		}
	}
	defer delete(jack, context.temp_allocator)
	low, high := bootstrap_interval(theta_hat, boot, jack, confidence, method)
	mean := 0.0
	for v in boot {mean += v}
	mean /= f64(n_resamples)
	ss := 0.0
	for v in boot {ss += (v - mean) * (v - mean)}
	return {low = low, high = high, standard_error = math.sqrt(ss / f64(n_resamples - 1)), theta_hat = theta_hat}
}

// Interval from a bootstrap distribution (scipy.stats.bootstrap's formulas):
//   Percentile: [q(α/2), q(1 - α/2)];  Basic: [2θ̂ - q(1 - α/2), 2θ̂ - q(α/2)]
//   BCa: q(Φ(z₀ + (z₀ + z_a)/(1 - â (z₀ + z_a)))) for z_a = Φ⁻¹(α/2), Φ⁻¹(1 - α/2),
//        z₀ = Φ⁻¹(mid-rank fraction of θ̂ in boot), â from the jackknife values
// q = sample quantile with linear interpolation (numpy percentile). boot is not modified.
bootstrap_interval :: proc(theta_hat: f64, boot: []f64, jack: []f64, confidence: f64, method: BootstrapMethod) -> (low, high: f64) {
	B := len(boot)
	if B < 2 do panic("bootstrap_interval: need at least 2 bootstrap values")
	if !(confidence > 0 && confidence < 1) do panic("bootstrap_interval: confidence must be in (0, 1)")
	srt := slice.clone(boot, context.temp_allocator)
	defer delete(srt, context.temp_allocator)
	slice.sort(srt)
	alpha := (1 - confidence) / 2
	switch method {
	case .Percentile:
		return _percentile_sorted(srt, 100 * alpha), _percentile_sorted(srt, 100 * (1 - alpha))
	case .Basic:
		return 2 * theta_hat - _percentile_sorted(srt, 100 * (1 - alpha)), 2 * theta_hat - _percentile_sorted(srt, 100 * alpha)
	case .BCa:
		if len(jack) < 2 do panic("bootstrap_interval: BCa needs the jackknife values")
		below, at_or_below := 0, 0
		for v in boot {
			if v < theta_hat do below += 1
			if v <= theta_hat do at_or_below += 1
		}
		z0 := _arima_norm_ppf(f64(below + at_or_below) / (2 * f64(B)))
		jm := 0.0
		for v in jack {jm += v}
		jm /= f64(len(jack))
		num, den := 0.0, 0.0
		for v in jack {
			d := jm - v
			num += d * d * d
			den += d * d
		}
		a_hat := num / (6 * math.pow(den, 1.5))
		za := _arima_norm_ppf(alpha)
		n1 := z0 + za
		n2 := z0 - za
		a1 := 0.5 * math.erfc(-(z0 + n1 / (1 - a_hat * n1)) / math.SQRT_TWO)
		a2 := 0.5 * math.erfc(-(z0 + n2 / (1 - a_hat * n2)) / math.SQRT_TWO)
		return _percentile_sorted(srt, 100 * a1), _percentile_sorted(srt, 100 * a2)
	}
	return math.nan_f64(), math.nan_f64()
}

// q-th percentile (0 … 100) of sorted values, linear interpolation (numpy "linear")
_percentile_sorted :: proc(v: []f64, q: f64) -> f64 {
	h := f64(len(v) - 1) * q / 100
	lo := int(math.floor(h))
	if lo < 0 do return v[0]
	if lo >= len(v) - 1 do return v[len(v) - 1]
	return v[lo] + (h - f64(lo)) * (v[lo + 1] - v[lo])
}
