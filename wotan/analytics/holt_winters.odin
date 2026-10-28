package analytics

import l "../linalg"
import ml "./ML"
import "core:math"
import "core:mem"

// ============================================================================
// Holt-Winters exponential smoothing (statsmodels ExponentialSmoothing)
// ============================================================================
// Recursions (statsmodels holtwinters, ŝ = l_{t-1} + φ b_{t-1}):
//   level     l_t = α (y_t - s_{t-m}) + (1-α) ŝ          (additive season)
//             l_t = α y_t / s_{t-m} + (1-α) ŝ            (multiplicative)
//   trend     b_t = β (l_t - l_{t-1}) + (1-β) φ b_{t-1}
//   season    s_t = γ (y_t - ŝ) + (1-γ) s_{t-m}  |  γ y_t / ŝ + (1-γ) s_{t-m}
//   fitted    ŷ_t = ŝ + s_{t-m}  |  ŝ · s_{t-m}  (one-step-ahead)
//   forecast  ŷ_{T+h} = l_T + (φ + … + φ^h) b_T  (+|×) seasonal state of that phase
// Fitting minimises the SSE over α, β ≤ α, γ ≤ 1 - α, φ ∈ [0.8, 0.995] and the
// initial states (statsmodels' default initialization_method="estimated").

HWTrend :: enum {
	None,
	Additive,
}

HWSeasonal :: enum {
	None,
	Additive,
	Multiplicative,
}

HoltWintersSpec :: struct {
	trend:    HWTrend,
	damped:   bool, // damped trend (needs trend = .Additive)
	seasonal: HWSeasonal,
	period:   int, // season length m (needs seasonal != .None)
}

HoltWintersParams :: struct {
	alpha, beta, gamma, phi: f64, // smoothing level / trend / seasonal, damping
	level0, trend0:          f64, // initial level / trend
	season0:                 []f64, // initial seasonal states s_{1-m} … s_0 (m values)
}

HoltWintersResult :: struct {
	spec:      HoltWintersSpec,
	params:    HoltWintersParams,
	fitted:    []f64, // one-step-ahead predictions ŷ_t
	resid:     []f64, // y_t - ŷ_t
	level:     []f64, // l_t after observation t
	trend:     []f64, // b_t after observation t (0 without trend)
	season:    []f64, // the seasonal state array (n + m values, statsmodels layout)
	sse:       f64,
	aic, aicc, bic: f64, // statsmodels: n ln(SSE/n) + 2k etc., k counts initial states
	allocator: mem.Allocator,
}

// Run the recursions with given parameters (statsmodels fit(optimized=False)
// with initialization_method="known")
holt_winters_filter :: proc(
	y: []f64,
	spec: HoltWintersSpec,
	params: HoltWintersParams,
	allocator := context.allocator,
) -> HoltWintersResult {
	_hw_check(y, spec)
	if spec.seasonal != .None && len(params.season0) != spec.period do panic("holt_winters_filter: season0 needs `period` values")
	n := len(y)
	m := _hw_m(spec)
	res := HoltWintersResult {
		spec      = spec,
		allocator = allocator,
	}
	res.params = params
	res.params.season0 = make([]f64, m, allocator)
	if m > 0 && spec.seasonal != .None do copy(res.params.season0, params.season0)
	res.fitted = make([]f64, n, allocator)
	res.resid = make([]f64, n, allocator)
	res.level = make([]f64, n + 1, allocator)
	res.trend = make([]f64, n + 1, allocator)
	res.season = make([]f64, n + max(m, 1), allocator)
	res.sse = _hw_run(y, spec, &res.params, res.fitted, res.level, res.trend, res.season)
	for t in 0 ..< n {res.resid[t] = y[t] - res.fitted[t]}
	_hw_info(&res, n)
	return res
}

// Least-squares fit of the smoothing parameters and initial states
holt_winters_fit :: proc(y: []f64, spec: HoltWintersSpec, allocator := context.allocator) -> HoltWintersResult {
	_hw_check(y, spec)
	n := len(y)
	m := _hw_m(spec)
	np := _hw_n_smooth(spec)

	ctx := _HWCtx {
		y    = y,
		spec = spec,
	}
	ctx.fit_init = spec.seasonal == .Multiplicative
	ctx.n_init = 1 + (1 if spec.trend == .Additive else 0) + (m if spec.seasonal != .None else 0)
	n_x := np + (ctx.n_init if ctx.fit_init else 0)

	// heuristic initial states (statsmodels _initialization_simple)
	init0 := make([]f64, ctx.n_init, context.temp_allocator)
	defer delete(init0, context.temp_allocator)
	_hw_simple_init(y, spec, init0)

	best := make([]f64, n_x, context.temp_allocator)
	defer delete(best, context.temp_allocator)
	best_f := math.INF_F64
	x0 := make([]f64, n_x, context.temp_allocator)
	defer delete(x0, context.temp_allocator)
	// grid of starts (statsmodels use_brute) for the smoothing parameters, in [0, 1] coordinates
	grid := []f64{0.1, 0.5, 0.9}
	n_grid := 1
	for _ in 0 ..< np {n_grid *= len(grid)}
	for g in 0 ..< n_grid {
		code := g
		for i in 0 ..< np {
			u := grid[code % len(grid)]
			code /= len(grid)
			x0[i] = math.ln(u / (1 - u)) // logit
		}
		for i in 0 ..< ctx.n_init {if ctx.fit_init do x0[np + i] = init0[i]}
		f := _hw_objective(x0, &ctx)
		if f < best_f {best_f = f; copy(best, x0)}
	}
	for _ in 0 ..< 6 {
		xs, fs, _ := nelder_mead(_hw_objective, &ctx, best, 4000 * max(n_x / 2, 1), 1e-14, context.temp_allocator)
		gain := best_f - fs
		if fs < best_f {best_f = fs; copy(best, xs)}
		delete(xs, context.temp_allocator)
		if gain < 1e-12 * max(1, best_f) do break
	}

	params := _hw_params_from(best, &ctx, allocator)
	defer delete(params.season0, allocator)
	return holt_winters_filter(y, spec, params, allocator)
}

// h-step-ahead forecasts from the end of the fitted sample
holt_winters_forecast :: proc(res: ^HoltWintersResult, h: int, allocator := context.allocator) -> []f64 {
	out := make([]f64, max(h, 0), allocator)
	n := len(res.fitted)
	m := _hw_m(res.spec)
	p := &res.params
	phi := p.phi if res.spec.damped else 1
	phi_sum := 0.0
	for k in 1 ..= h {
		phi_sum += math.pow(phi, f64(k))
		tr := res.level[n]
		if res.spec.trend == .Additive do tr += phi_sum * res.trend[n]
		switch res.spec.seasonal {
		case .None:
			out[k - 1] = tr
		case .Additive, .Multiplicative:
			t := n + k - 1
			// statsmodels: from phase n+m-1 on the states s[n-1 …] are repeated
			si := t if t < n + m - 1 else (n - 1) + (t - (n + m - 1)) % m
			s := res.season[si]
			out[k - 1] = tr + s if res.spec.seasonal == .Additive else tr * s
		}
	}
	return out
}

holt_winters_free :: proc(res: ^HoltWintersResult) {
	a := res.allocator
	delete(res.params.season0, a)
	delete(res.fitted, a)
	delete(res.resid, a)
	delete(res.level, a)
	delete(res.trend, a)
	delete(res.season, a)
}

// ----------------------------------------------------------------------------
// Internal
// ----------------------------------------------------------------------------

_hw_check :: proc(y: []f64, spec: HoltWintersSpec) {
	if spec.damped && spec.trend == .None do panic("holt_winters: damped needs an additive trend")
	if spec.seasonal != .None && spec.period < 2 do panic("holt_winters: seasonal models need period >= 2")
	m := _hw_m(spec)
	if len(y) < max(2 * m, 4) do panic("holt_winters: series too short (needs two full seasons)")
	if spec.seasonal == .Multiplicative {
		for v in y {if !(v > 0) do panic("holt_winters: multiplicative seasonality needs positive data")}
	}
}

_hw_m :: proc(spec: HoltWintersSpec) -> int {return spec.period if spec.seasonal != .None else 0}

_hw_n_smooth :: proc(spec: HoltWintersSpec) -> int {
	n := 1
	if spec.trend == .Additive do n += 1
	if spec.seasonal != .None do n += 1
	if spec.damped do n += 1
	return n
}

// One pass of the recursions; returns the SSE. level / trend: n + 1, season: n + m
_hw_run :: proc(y: []f64, spec: HoltWintersSpec, p: ^HoltWintersParams, fitted, lvl, b, s: []f64) -> f64 {
	n := len(y)
	m := _hw_m(spec)
	has_t := spec.trend == .Additive
	phi := p.phi if spec.damped else 1
	alpha, beta, gamma := p.alpha, p.beta, p.gamma
	for &v in lvl {v = 0}
	for &v in b {v = 0}
	for &v in s {v = 0}
	lvl[0] = p.level0
	if has_t do b[0] = p.trend0
	if m > 0 do copy(s[:m], p.season0)
	sse := 0.0
	for i in 1 ..= n {
		yi := y[i - 1]
		base := lvl[i - 1] + (phi * b[i - 1] if has_t else 0)
		switch spec.seasonal {
		case .None:
			lvl[i] = alpha * yi + (1 - alpha) * base
		case .Additive:
			lvl[i] = alpha * yi - alpha * s[i - 1] + (1 - alpha) * base
			s[i + m - 1] = gamma * yi - gamma * base + (1 - gamma) * s[i - 1]
		case .Multiplicative:
			lvl[i] = alpha * yi / s[i - 1] + (1 - alpha) * base
			s[i + m - 1] = gamma * yi / base + (1 - gamma) * s[i - 1]
		}
		if has_t do b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * phi * b[i - 1]
		// one-step prediction made at i - 1
		f := base
		switch spec.seasonal {
		case .None:
		case .Additive:
			f += s[i - 1]
		case .Multiplicative:
			f *= s[i - 1]
		}
		if fitted != nil do fitted[i - 1] = f
		e := yi - f
		sse += e * e
	}
	return sse
}

_hw_info :: proc(res: ^HoltWintersResult, n: int) {
	spec := res.spec
	k := f64((spec.period if spec.seasonal != .None else 0) + (2 if spec.trend == .Additive else 0) + 2 + (1 if spec.damped else 0))
	nf := f64(n)
	ll := nf * math.ln(res.sse / nf)
	res.aic = ll + 2 * k
	res.aicc = res.aic + (2 * (k + 2) * (k + 3) / (nf - k - 3) if nf - k - 3 > 0 else math.inf_f64(1))
	res.bic = ll + k * math.ln(nf)
}

_HWCtx :: struct {
	y:        []f64,
	spec:     HoltWintersSpec,
	fit_init: bool, // initial states are optimiser variables (multiplicative); else least squares
	n_init:   int,
}

// statsmodels' restricted parameters: x (logits) → α ∈ [ε, 1-ε], β ∈ [0, α], γ ∈ [0, 1-α], φ ∈ [0.8, 0.995]
_hw_smoothing :: proc(x: []f64, spec: HoltWintersSpec) -> (alpha, beta, gamma, phi: f64) {
	sig :: proc(v: f64) -> f64 {return 1 / (1 + math.exp(-v))}
	LB :: 1.4901161193847656e-08 // sqrt(machine eps), statsmodels LOWER_BOUND
	i := 0
	alpha = LB + sig(x[i]) * (1 - 2 * LB)
	i += 1
	if spec.trend == .Additive {beta = sig(x[i]) * alpha; i += 1}
	if spec.seasonal != .None {gamma = sig(x[i]) * (1 - alpha); i += 1}
	phi = 1
	if spec.damped {phi = 0.8 + sig(x[i]) * (0.995 - 0.8); i += 1}
	return
}

// SSE at the optimiser point x; additive models: the initial states are solved
// by least squares (the fitted values are affine in them)
_hw_objective :: proc(x: []f64, ctx_ptr: rawptr) -> f64 {
	ctx := (^_HWCtx)(ctx_ptr)
	p := _hw_params_from(x, ctx, context.temp_allocator)
	defer delete(p.season0, context.temp_allocator)
	n := len(ctx.y)
	m := _hw_m(ctx.spec)
	lvl := make([]f64, n + 1, context.temp_allocator)
	b := make([]f64, n + 1, context.temp_allocator)
	s := make([]f64, n + max(m, 1), context.temp_allocator)
	defer {
		delete(lvl, context.temp_allocator)
		delete(b, context.temp_allocator)
		delete(s, context.temp_allocator)
	}
	sse := _hw_run(ctx.y, ctx.spec, &p, nil, lvl, b, s)
	return sse if !math.is_nan(sse) else math.inf_f64(1)
}

// Parameters at optimiser point x (smoothing logits, then the initial states if fit_init)
_hw_params_from :: proc(x: []f64, ctx: ^_HWCtx, allocator: mem.Allocator) -> HoltWintersParams {
	spec := ctx.spec
	m := _hw_m(spec)
	p: HoltWintersParams
	p.alpha, p.beta, p.gamma, p.phi = _hw_smoothing(x, spec)
	p.season0 = make([]f64, m, allocator)
	np := _hw_n_smooth(spec)
	init := make([]f64, ctx.n_init, context.temp_allocator)
	defer delete(init, context.temp_allocator)
	if ctx.fit_init {
		copy(init, x[np:np + ctx.n_init])
	} else {
		_hw_ls_init(ctx, p, init)
	}
	_hw_unpack_init(spec, init, &p)
	return p
}

_hw_unpack_init :: proc(spec: HoltWintersSpec, init: []f64, p: ^HoltWintersParams) {
	i := 0
	p.level0 = init[i]
	i += 1
	if spec.trend == .Additive {p.trend0 = init[i]; i += 1}
	if spec.seasonal != .None {copy(p.season0, init[i:i + spec.period])}
}

// Least-squares initial states for an additive model at given smoothing parameters:
// fitted(init; y) = fitted(0; y) + fitted(init; 0), linear in init
_hw_ls_init :: proc(ctx: ^_HWCtx, smooth: HoltWintersParams, init_out: []f64) {
	y := ctx.y
	n := len(y)
	m := _hw_m(ctx.spec)
	k := ctx.n_init
	lvl := make([]f64, n + 1, context.temp_allocator)
	b := make([]f64, n + 1, context.temp_allocator)
	s := make([]f64, n + max(m, 1), context.temp_allocator)
	f0 := make([]f64, n, context.temp_allocator)
	zero := make([]f64, n, context.temp_allocator)
	J := l.matrix_new(f64, n, k, context.temp_allocator)
	basis := make([]f64, k, context.temp_allocator)
	fc := make([]f64, n, context.temp_allocator)
	defer {
		delete(lvl, context.temp_allocator)
		delete(b, context.temp_allocator)
		delete(s, context.temp_allocator)
		delete(f0, context.temp_allocator)
		delete(zero, context.temp_allocator)
		l.matrix_free(&J)
		delete(basis, context.temp_allocator)
		delete(fc, context.temp_allocator)
	}
	p := smooth
	p.season0 = make([]f64, m, context.temp_allocator)
	defer delete(p.season0, context.temp_allocator)
	// fitted with zero initial states
	_hw_unpack_init(ctx.spec, basis, &p)
	_hw_run(y, ctx.spec, &p, f0, lvl, b, s)
	// response of the fitted values to each unit initial state (data = 0)
	for c in 0 ..< k {
		for &v in basis {v = 0}
		basis[c] = 1
		_hw_unpack_init(ctx.spec, basis, &p)
		_hw_run(zero, ctx.spec, &p, fc, lvl, b, s)
		for t in 0 ..< n {J.data[t * k + c] = fc[t]}
	}
	// init = argmin || (y - f0) - J init ||²
	r := make([]f64, n, context.temp_allocator)
	defer delete(r, context.temp_allocator)
	for t in 0 ..< n {r[t] = y[t] - f0[t]}
	G := l.matrix_new(f64, k, k, context.temp_allocator)
	Gi := l.matrix_new(f64, k, k, context.temp_allocator)
	g := make([]f64, k, context.temp_allocator)
	defer {
		l.matrix_free(&G)
		l.matrix_free(&Gi)
		delete(g, context.temp_allocator)
	}
	for t in 0 ..< n {
		for a in 0 ..< k {
			g[a] += J.data[t * k + a] * r[t]
			for c in 0 ..< k {G.data[a * k + c] += J.data[t * k + a] * J.data[t * k + c]}
		}
	}
	ml._ols_inverse(&G, &Gi) // level and seasonal states are collinear: pseudo-inverse
	for a in 0 ..< k {
		v := 0.0
		for c in 0 ..< k {v += Gi.data[a * k + c] * g[c]}
		init_out[a] = v
	}
}

// statsmodels _initialization_simple: level = mean of the first season, trend =
// mean difference between the first two seasons / m, seasonal = first season - level (or / level)
_hw_simple_init :: proc(y: []f64, spec: HoltWintersSpec, init: []f64) {
	m := _hw_m(spec)
	mm := max(m, 1)
	i := 0
	lvl := 0.0
	if m > 0 {
		for t in 0 ..< m {lvl += y[t]}
		lvl /= f64(m)
	} else {
		lvl = y[0]
	}
	init[i] = lvl
	i += 1
	if spec.trend == .Additive {
		if m > 0 {
			a, c := 0.0, 0.0
			for t in 0 ..< m {a += y[t]; c += y[m + t]}
			init[i] = (c - a) / f64(m * mm)
		} else {
			init[i] = y[1] - y[0]
		}
		i += 1
	}
	if spec.seasonal == .Additive {for t in 0 ..< m {init[i + t] = y[t] - lvl}}
	if spec.seasonal == .Multiplicative {for t in 0 ..< m {init[i + t] = y[t] / lvl}}
}
