package stats

import w "../core"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"
import "core:strings"

// Generalized linear models fitted by iteratively reweighted least squares
// (statsmodels' GLM, R's glm and MASS::glm.nb).

// The distribution of the response.
Family :: enum {
	Gaussian, // any real y; identity link
	Binomial, // y in [0, 1] (0/1 or proportions), or success counts with `trials`; logit link
	Poisson, // counts y >= 0; log link
	Gamma, // y > 0; inverse link
	Inverse_Gaussian, // y > 0; 1/mu² link
	Negative_Binomial, // overdispersed counts y >= 0, variance mu + alpha mu²; log link
}

// How the mean depends on the linear predictor eta = X beta + offset.
Link :: enum {
	Default, // the family's canonical link (log for the negative binomial)
	Identity, // mu = eta
	Log, // mu = exp(eta)
	Logit, // mu = 1 / (1 + exp(-eta))
	Probit, // mu = Phi(eta)
	Cloglog, // mu = 1 - exp(-exp(eta))
	Inverse, // mu = 1 / eta
	Inverse_Squared, // mu = 1 / sqrt(eta)
	Sqrt, // mu = eta²
}

// Options of glm and glm_from_df. The zero value is the usual model: the
// canonical link, an intercept, z statistics, 95% intervals and, for the
// negative binomial, alpha estimated by maximum likelihood.
Glm_Options :: struct {
	link:         Link,
	offset:       []f64, // added to the linear predictor, e.g. log exposure for rates (nil = none)
	trials:       []f64, // Binomial: trials per observation, y then counts the successes (nil = y in [0, 1])
	alpha:        f64, // Negative_Binomial: fixed alpha > 0, or 0 to estimate it (MASS::glm.nb)
	no_intercept: bool, // leave out the constant column
	use_t:        bool, // t statistics with df_resid degrees of freedom (R for Gaussian, Gamma, Inverse_Gaussian)
	confidence:   f64, // level of the intervals (0 = 0.95)
	max_iter:     int, // IRLS iterations (0 = 100)
}

Glm_Result :: struct {
	family:         Family,
	link:           Link, // the link used (never .Default)
	intercept:      bool,
	use_t:          bool,
	names:          []string, // "Intercept" first, then the predictors
	params:         []f64,
	std_errors:     []f64,
	statistics:     []f64, // params / std_errors: z, or t with use_t
	p_values:       []f64, // two-sided
	ci_low:         []f64,
	ci_high:        []f64,
	cov:            []f64, // covariance of the parameters, p x p row-major, scaled
	fitted:         []f64, // mu of every observation (proportions for Binomial)
	n:              int,
	df_model:       f64, // parameters besides the intercept
	df_resid:       f64, // n - parameters
	deviance:       f64,
	null_deviance:  f64, // deviance of the intercept-only model (with the offset)
	pearson_chi2:   f64,
	scale:          f64, // dispersion: pearson_chi2 / df_resid for Gaussian, Gamma, Inverse_Gaussian; 1 otherwise
	log_likelihood: f64,
	aic:            f64, // -2 llf + 2 k, k the parameters (+1 for an estimated scale or alpha, as R)
	bic:            f64, // -2 llf + k ln n
	alpha:          f64, // Negative_Binomial's alpha (estimated or given); 0 otherwise
	iterations:     int,
	converged:      bool,
}

// Fits a GLM of y on the columns x[j] (one slice per predictor, all of
// length len(y)) named names[j]. Returns ok = false for invalid input:
// lengths that differ, non-finite values, y outside the family's support,
// collinear columns, more parameters than observations. A fit that does
// not converge (complete separation in a logistic model, say) returns
// ok = true with converged = false.
//
//     res, _ := stats.glm(deaths, {age, smoker}, {"age", "smoker"}, .Poisson,
//                         {offset = log_person_years})
//     defer stats.glm_destroy(&res)
//     fmt.println(stats.glm_summary(&res, context.temp_allocator))
glm :: proc(
	y: []f64,
	x: [][]f64,
	names: []string,
	family: Family,
	options := Glm_Options{},
	allocator := context.allocator,
) -> (
	res: Glm_Result,
	ok: bool,
) {
	n := len(y)
	if len(names) != len(x) || !_all_finite(y) {return {}, false}
	for c in x {if len(c) != n || !_all_finite(c) {return {}, false}}
	if options.offset != nil &&
	   (len(options.offset) != n || !_all_finite(options.offset)) {return {}, false}
	if options.trials != nil &&
	   (family != .Binomial || len(options.trials) != n) {return {}, false}
	if options.alpha < 0 || math.is_nan(options.alpha) {return {}, false}
	conf := options.confidence if options.confidence != 0 else 0.95
	if !_valid_confidence(conf) {return {}, false}
	link := options.link if options.link != .Default else _canonical_link(family)
	intercept := !options.no_intercept
	p := len(x) + (1 if intercept else 0)
	if p == 0 || n <= p {return {}, false}
	w.TEMP_GUARD(ignore = _is_temp(allocator))

	m := _Glm_Model {
		family    = family,
		link      = link,
		max_iter  = options.max_iter if options.max_iter > 0 else 100,
		offset    = options.offset,
		prior     = make([]f64, n, context.temp_allocator),
		y         = make([]f64, n, context.temp_allocator),
		yc        = make([]f64, n, context.temp_allocator),
		counts    = options.trials != nil,
		cols      = make([][]f64, p, context.temp_allocator),
		intercept = intercept,
	}
	// response on the family's scale: proportions with the trials as prior weights
	for v, i in y {
		m.prior[i] = 1
		m.y[i] = v
		m.yc[i] = 1 - v
		if options.trials != nil {
			t := options.trials[i]
			if !(t > 0) || math.is_inf(t) || v < 0 || v > t {return {}, false}
			m.prior[i] = t
			m.y[i] = v / t
			m.yc[i] = (t - v) / t
		}
		if !_glm_y_ok(family, m.y[i]) {return {}, false}
	}
	if intercept {
		m.cols[0] = make([]f64, n, context.temp_allocator)
		for &v in m.cols[0] {v = 1}
	}
	for c, j in x {m.cols[j + (1 if intercept else 0)] = c}

	alpha := options.alpha
	estimate_alpha := family == .Negative_Binomial && alpha == 0
	beta: []f64
	if estimate_alpha {
		// glm.nb: Poisson start, then alternate the GLM fit and the ML alpha
		m.family = .Poisson
		fit := _glm_irls(&m, nil) or_return
		m.family = .Negative_Binomial
		theta := _nb_theta_ml(m.y, fit.mu, math.nan_f64())
		for _ in 0 ..< 100 {
			m.alpha = 1 / theta
			fit = _glm_irls(&m, fit.beta) or_return
			t2 := _nb_theta_ml(m.y, fit.mu, theta)
			done := abs(t2 - theta) <= 1e-13 * theta
			theta = t2
			if done || math.is_inf(theta) || math.is_nan(theta) {break}
		}
		alpha = 1 / theta
		m.alpha = alpha
		fit = _glm_irls(&m, fit.beta) or_return
		beta = fit.beta
		res.iterations, res.converged = fit.iterations, fit.converged
	} else {
		m.alpha = alpha
		fit := _glm_irls(&m, nil) or_return
		beta = fit.beta
		res.iterations, res.converged = fit.iterations, fit.converged
	}

	// statistics at the solution
	eta := make([]f64, n, context.temp_allocator)
	mu := make([]f64, n, context.temp_allocator)
	muc := make([]f64, n, context.temp_allocator)
	dmu := make([]f64, n, context.temp_allocator)
	_glm_eta(&m, beta, eta)
	for i in 0 ..< n {mu[i], muc[i], dmu[i] = _linkinv(link, eta[i])}
	m.cur = beta
	cov_unscaled, cov_ok := _glm_cov(&m, mu, muc, dmu)
	if !cov_ok {
		// only after a fit that ran into the boundary; no standard errors then
		cov_unscaled = make([]f64, p * p, context.temp_allocator)
		for &v in cov_unscaled {v = math.nan_f64()}
	}

	res.family, res.link, res.intercept, res.use_t = family, link, intercept, options.use_t
	res.n = n
	res.alpha = alpha if family == .Negative_Binomial else 0
	res.df_resid = f64(n - p)
	res.df_model = f64(p - (1 if intercept else 0))
	res.deviance = _glm_deviance(&m, mu, muc)
	pearson: _Sum
	for i in 0 ..< n {
		d := _resid(&m, mu, i)
		_sum_add(&pearson, m.prior[i] * d * d / _variance(family, mu[i], muc[i], alpha))
	}
	res.pearson_chi2 = _sum(pearson)
	estimated_scale := family == .Gaussian || family == .Gamma || family == .Inverse_Gaussian
	res.scale = res.pearson_chi2 / res.df_resid if estimated_scale else 1
	res.log_likelihood = _glm_llf(&m, mu, muc, res.scale, res.deviance)
	k := f64(p) + (1 if estimated_scale || estimate_alpha else 0)
	res.aic = -2 * res.log_likelihood + 2 * k
	res.bic = -2 * res.log_likelihood + k * math.ln(f64(n))

	// null model: intercept only, with the offset
	{
		nm := m
		nm.cols = m.cols[:1]
		if !intercept {
			nm.cols = make([][]f64, 1, context.temp_allocator)
			nm.cols[0] = make([]f64, n, context.temp_allocator)
			for &v in nm.cols[0] {v = 1}
		}
		nfit, nok := _glm_irls(&nm, nil)
		res.null_deviance = math.nan_f64()
		if nok {
			for i in 0 ..< n {mu[i], muc[i], _ = _linkinv(link, nfit.beta[0] + (m.offset[i] if m.offset != nil else 0))}
			nm.cur = nfit.beta
			res.null_deviance = _glm_deviance(&nm, mu, muc)
		}
	}

	res.names = make([]string, p, allocator)
	if intercept {res.names[0] = strings.clone("Intercept", allocator)}
	for s, j in names {res.names[j + (1 if intercept else 0)] = strings.clone(s, allocator)}
	res.params = slice.clone(beta, allocator)
	res.std_errors = make([]f64, p, allocator)
	res.statistics = make([]f64, p, allocator)
	res.p_values = make([]f64, p, allocator)
	res.ci_low = make([]f64, p, allocator)
	res.ci_high = make([]f64, p, allocator)
	res.cov = make([]f64, p * p, allocator)
	for v, i in cov_unscaled {res.cov[i] = v * res.scale}
	crit :=
		student_t_isf(Student_T{res.df_resid}, (1 - conf) / 2) if options.use_t else normal_isf(STANDARD_NORMAL, (1 - conf) / 2)
	for j in 0 ..< p {
		se := math.sqrt(res.cov[j * p + j])
		res.std_errors[j] = se
		res.statistics[j] = beta[j] / se
		res.p_values[j] =
			_p_t(res.statistics[j], res.df_resid, .Two_Sided) if options.use_t else _p_normal(res.statistics[j], .Two_Sided)
		res.ci_low[j], res.ci_high[j] = beta[j] - crit * se, beta[j] + crit * se
	}
	res.fitted = make([]f64, n, allocator)
	_glm_eta(&m, beta, eta)
	for i in 0 ..< n {res.fitted[i], _, _ = _linkinv(link, eta[i])}
	return res, true
}

// glm on DataFrame columns. Int, Float and Bool predictors enter as numbers;
// a String predictor with L labels enters as L - 1 indicator columns named
// "name[T.label]" against its first label in sorted order (patsy's
// treatment coding). `trials` and `offset` name optional columns for
// options.trials and options.offset (any slices in options are ignored).
// Rows with a null in any column used are left out.
glm_from_df :: proc(
	df: ^w.DataFrame,
	response: string,
	predictors: []string,
	family: Family,
	options := Glm_Options{},
	trials := "",
	offset := "",
	allocator := context.allocator,
) -> (
	res: Glm_Result,
	ok: bool,
) {
	w.TEMP_GUARD(ignore = _is_temp(allocator))
	ta := context.temp_allocator
	yc := _numeric_column(df, response) or_return
	nrow := yc.len
	keep := make([]bool, nrow, ta)
	for i in 0 ..< nrow {_, null := w.column_at_numeric(yc, i); keep[i] = !null}
	extra := [2]string{trials, offset}
	extra_cols: [2]^w.Column
	for name, e in extra {
		if name == "" {continue}
		c := _numeric_column(df, name) or_return
		extra_cols[e] = c
		for i in 0 ..< nrow {_, null := w.column_at_numeric(c, i); if null {keep[i] = false}}
	}
	Pred :: struct {
		num:    ^w.Column,
		codes:  []int,
		labels: []string,
	}
	preds := make([]Pred, len(predictors), ta)
	for name, j in predictors {
		idx, found := df.name_to_index[name]
		if !found {return {}, false}
		c := &df.columns[idx]
		if c.type == .String {
			codes, labels := _group_codes(df, name, ta) or_return
			preds[j] = {nil, codes, labels}
			for i in 0 ..< nrow {if codes[i] < 0 {keep[i] = false}}
		} else {
			num := _numeric_column(df, name) or_return
			preds[j] = {num, nil, nil}
			for i in 0 ..< nrow {_, null := w.column_at_numeric(num, i); if null {keep[i] = false}}
		}
	}
	rows := make([dynamic]int, 0, ta)
	for k, i in keep {if k {append(&rows, i)}}
	nr := len(rows)
	y := make([]f64, nr, ta)
	for r, i in rows {y[i], _ = w.column_at_numeric(yc, r)}
	cols := make([dynamic][]f64, 0, ta)
	names := make([dynamic]string, 0, ta)
	for pr, j in preds {
		if pr.num != nil {
			c := make([]f64, nr, ta)
			for r, i in rows {c[i], _ = w.column_at_numeric(pr.num, r)}
			append(&cols, c)
			append(&names, predictors[j])
			continue
		}
		for l in 1 ..< len(pr.labels) {
			c := make([]f64, nr, ta)
			for r, i in rows {c[i] = 1 if pr.codes[r] == l else 0}
			append(&cols, c)
			append(&names, fmt.tprintf("%s[T.%s]", predictors[j], pr.labels[l]))
		}
	}
	opts := options
	opts.trials, opts.offset = nil, nil
	for e in 0 ..< 2 {
		if extra_cols[e] == nil {continue}
		v := make([]f64, nr, ta)
		for r, i in rows {v[i], _ = w.column_at_numeric(extra_cols[e], r)}
		if e == 0 {opts.trials = v} else {opts.offset = v}
	}
	return glm(y, cols[:], names[:], family, opts, allocator)
}

glm_destroy :: proc(r: ^Glm_Result, allocator := context.allocator) {
	for s in r.names {delete(s, allocator)}
	delete(r.names, allocator)
	delete(r.params, allocator)
	delete(r.std_errors, allocator)
	delete(r.statistics, allocator)
	delete(r.p_values, allocator)
	delete(r.ci_low, allocator)
	delete(r.ci_high, allocator)
	delete(r.cov, allocator)
	delete(r.fitted, allocator)
	r^ = {}
}

// Mean response for new observations: x[j] holds predictor j as in the fit
// (without the intercept column), offset is optional.
glm_predict :: proc(
	r: ^Glm_Result,
	x: [][]f64,
	offset: []f64 = nil,
	allocator := context.allocator,
) -> (
	mu: []f64,
	ok: bool,
) {
	k := 1 if r.intercept else 0
	if len(x) + k != len(r.params) {return nil, false}
	n := len(x[0]) if len(x) > 0 else len(offset)
	for c in x {if len(c) != n {return nil, false}}
	if offset != nil && len(offset) != n {return nil, false}
	mu = make([]f64, n, allocator)
	for i in 0 ..< n {
		eta := r.params[0] if r.intercept else 0
		for c, j in x {eta += r.params[j + k] * c[i]}
		if offset != nil {eta += offset[i]}
		mu[i], _, _ = _linkinv(r.link, eta)
	}
	return mu, true
}

// Likelihood-ratio test of a model against a larger one that contains it:
// 2 (llf_full - llf_restricted), chi-squared with the difference in the
// number of parameters (statsmodels' compare_lr_test, R's anova(test =
// "LRT") for a known scale). Both fits need the same family and
// observations.
glm_lr_test :: proc(restricted, full: ^Glm_Result) -> (res: Test_Result, ok: bool) {
	df := len(full.params) - len(restricted.params)
	if df <= 0 || full.n != restricted.n || full.family != restricted.family {return {}, false}
	res.statistic = 2 * (full.log_likelihood - restricted.log_likelihood)
	res.df = f64(df)
	res.p_value = chi_squared_sf(Chi_Squared{res.df}, res.statistic)
	return res, true
}

// A summary table in the style of statsmodels' GLMResults.summary().
glm_summary :: proc(r: ^Glm_Result, allocator := context.allocator) -> string {
	b := strings.builder_make(allocator)
	fam := [Family]string {
		.Gaussian          = "Gaussian",
		.Binomial          = "Binomial",
		.Poisson           = "Poisson",
		.Gamma             = "Gamma",
		.Inverse_Gaussian  = "Inverse Gaussian",
		.Negative_Binomial = "Negative binomial",
	}
	lk := [Link]string {
		.Default         = "",
		.Identity        = "identity",
		.Log             = "log",
		.Logit           = "logit",
		.Probit          = "probit",
		.Cloglog         = "cloglog",
		.Inverse         = "inverse",
		.Inverse_Squared = "inverse squared",
		.Sqrt            = "sqrt",
	}
	fmt.sbprintf(&b, "Generalized linear model: %s family, %s link", fam[r.family], lk[r.link])
	if r.family == .Negative_Binomial {fmt.sbprintf(&b, ", alpha = %.6g", r.alpha)}
	fmt.sbprintf(
		&b,
		"\nObservations: %d    Df model: %v    Df residuals: %v\n",
		r.n,
		r.df_model,
		r.df_resid,
	)
	fmt.sbprintf(
		&b,
		"Deviance: %.6g    Null deviance: %.6g    Pearson chi2: %.6g    Scale: %.6g\n",
		r.deviance,
		r.null_deviance,
		r.pearson_chi2,
		r.scale,
	)
	fmt.sbprintf(
		&b,
		"Log-likelihood: %.6g    AIC: %.6g    BIC: %.6g\n",
		r.log_likelihood,
		r.aic,
		r.bic,
	)
	fmt.sbprintf(&b, "Iterations: %d%s\n", r.iterations, "" if r.converged else " (NOT CONVERGED)")
	wn := 9
	for s in r.names {wn = max(wn, len(s))}
	stat := "t" if r.use_t else "z"
	fmt.sbprintf(
		&b,
		"%*s %11s %11s %9s %9s %11s %11s\n",
		wn,
		"",
		"coef",
		"std err",
		stat,
		fmt.tprintf("P>|%s|", stat),
		"[lower",
		"upper]",
	)
	for s, j in r.names {
		fmt.sbprintf(
			&b,
			"%-*s %11.4g %11.4g %9.3f %9.3g %11.4g %11.4g\n",
			wn,
			s,
			r.params[j],
			r.std_errors[j],
			r.statistics[j],
			r.p_values[j],
			r.ci_low[j],
			r.ci_high[j],
		)
	}
	return strings.to_string(b)
}

// ---------------------------------------------------------------------------

@(private)
_Glm_Model :: struct {
	family:    Family,
	link:      Link,
	alpha:     f64,
	max_iter:  int,
	y, yc:     []f64, // response (proportions for Binomial) and 1 - y
	prior:     []f64, // prior weights (trials)
	counts:    bool,
	offset:    []f64,
	cols:      [][]f64,
	intercept: bool,
	cur:       []f64, // the current beta, for exact residuals with the identity link
}

// y - mu; with the identity link summed term by term from beta, so that a
// large intercept cancels against y before the rounding of mu
@(private)
_resid :: #force_inline proc(m: ^_Glm_Model, mu: []f64, i: int) -> f64 {
	if m.link != .Identity || m.cur == nil {return m.y[i] - mu[i]}
	r := m.y[i]
	if m.offset != nil {r -= m.offset[i]}
	for c, j in m.cols {r -= m.cur[j] * c[i]}
	return r
}

@(private)
_Glm_Fit :: struct {
	beta:       []f64,
	mu:         []f64,
	iterations: int,
	converged:  bool,
}

@(private)
_canonical_link :: proc(f: Family) -> Link {
	switch f {
	case .Gaussian:
		return .Identity
	case .Binomial:
		return .Logit
	case .Poisson, .Negative_Binomial:
		return .Log
	case .Gamma:
		return .Inverse
	case .Inverse_Gaussian:
		return .Inverse_Squared
	}
	return .Identity
}

@(private)
_glm_y_ok :: proc(f: Family, y: f64) -> bool {
	switch f {
	case .Gaussian:
		return true
	case .Binomial:
		return y >= 0 && y <= 1
	case .Poisson, .Negative_Binomial:
		return y >= 0
	case .Gamma, .Inverse_Gaussian:
		return y > 0
	}
	return false
}

@(private)
_mu_ok :: proc(f: Family, mu, muc: f64) -> bool {
	if math.is_nan(mu) || math.is_inf(mu) {return false}
	switch f {
	case .Gaussian:
		return true
	case .Binomial:
		return mu > 0 && muc > 0
	case .Poisson, .Negative_Binomial, .Gamma, .Inverse_Gaussian:
		return mu > 0
	}
	return false
}

@(private)
_link_fn :: proc(l: Link, mu: f64) -> f64 {
	switch l {
	case .Default, .Identity:
		return mu
	case .Log:
		return math.ln(mu)
	case .Logit:
		return math.ln(mu / (1 - mu))
	case .Probit:
		return normal_ppf(STANDARD_NORMAL, mu)
	case .Cloglog:
		return math.ln(-math.log1p(-mu))
	case .Inverse:
		return 1 / mu
	case .Inverse_Squared:
		return 1 / (mu * mu)
	case .Sqrt:
		return math.sqrt(mu)
	}
	return mu
}

// mu, 1 - mu (accurate where the link allows) and dmu/deta
@(private)
_linkinv :: proc(l: Link, eta: f64) -> (mu, muc, dmu: f64) {
	switch l {
	case .Default, .Identity:
		return eta, 1 - eta, 1
	case .Log:
		e := math.exp(eta)
		return e, 1 - e, e
	case .Logit:
		e := math.exp(-abs(eta))
		a, b := 1 / (1 + e), e / (1 + e) // a >= 1/2
		d := a * b
		if eta >= 0 {return a, b, d}
		return b, a, d
	case .Probit:
		return normal_cdf(STANDARD_NORMAL, eta),
			normal_sf(STANDARD_NORMAL, eta),
			math.exp(-0.5 * eta * eta) / math.sqrt(2 * math.PI)
	case .Cloglog:
		e := math.exp(eta)
		return -expm1(-e), math.exp(-e), math.exp(eta - e)
	case .Inverse:
		return 1 / eta, 1 - 1 / eta, -1 / (eta * eta)
	case .Inverse_Squared:
		m := 1 / math.sqrt(eta)
		return m, 1 - m, -0.5 * m * m * m
	case .Sqrt:
		return eta * eta, 1 - eta * eta, 2 * eta
	}
	return eta, 1 - eta, 1
}

@(private)
_variance :: proc(f: Family, mu, muc, alpha: f64) -> f64 {
	switch f {
	case .Gaussian:
		return 1
	case .Binomial:
		return mu * muc
	case .Poisson:
		return mu
	case .Gamma:
		return mu * mu
	case .Inverse_Gaussian:
		return mu * mu * mu
	case .Negative_Binomial:
		return mu + alpha * mu * mu
	}
	return 1
}

// x ln(x / m), 0 at x = 0
@(private)
_xlogx :: #force_inline proc(x, m: f64) -> f64 {return 0 if x == 0 else x * math.ln(x / m)}

@(private)
_glm_deviance :: proc(m: ^_Glm_Model, mu, muc: []f64) -> f64 {
	s: _Sum
	for i in 0 ..< len(mu) {
		y, u := m.y[i], mu[i]
		e := _resid(m, mu, i)
		d: f64
		switch m.family {
		case .Gaussian:
			d = e * e
		case .Binomial:
			d = 2 * (_xlogx(y, u) + _xlogx(m.yc[i], muc[i]))
		case .Poisson:
			d = 2 * (_xlogx(y, u) - e)
		case .Gamma:
			r := e / u
			d = 2 * (r - math.log1p(r))
		case .Inverse_Gaussian:
			d = e * e / (y * u * u)
		case .Negative_Binomial:
			a := m.alpha
			d = 2 * (_xlogx(y, u) - (y + 1 / a) * math.log1p(a * (y - u) / (1 + a * u)))
		}
		_sum_add(&s, m.prior[i] * d)
	}
	return _sum(s)
}

@(private)
_glm_llf :: proc(m: ^_Glm_Model, mu, muc: []f64, scale, deviance: f64) -> f64 {
	n := len(mu)
	s: _Sum
	if m.family == .Gaussian && m.link == .Identity {
		// concentrated: scale = RSS / n
		nf := f64(n)
		return -0.5 * nf * (math.ln(2 * math.PI * deviance / nf) + 1)
	}
	for i in 0 ..< n {
		y, u := m.y[i], mu[i]
		l: f64
		switch m.family {
		case .Gaussian:
			l = -0.5 * ((y - u) * (y - u) / scale + math.ln(2 * math.PI * scale))
		case .Binomial:
			t := m.prior[i]
			k := y * t
			if m.counts {
				l = log_gamma(t + 1) - log_gamma(k + 1) - log_gamma(t - k + 1)
			}
			if k > 0 {l += k * math.ln(u)}
			if t - k > 0 {l += (t - k) * math.ln(muc[i])}
		case .Poisson:
			l = -u - log_gamma(y + 1)
			if y > 0 {l += y * math.ln(u)}
		case .Gamma:
			v := 1 / scale
			l = v * math.ln(v * y / u) - v * y / u - log_gamma(v) - math.ln(y)
		case .Inverse_Gaussian:
			l =
				-0.5 *
				((y - u) * (y - u) / (y * u * u * scale) +
						math.ln(2 * math.PI * scale * y * y * y))
		case .Negative_Binomial:
			th := 1 / m.alpha
			l = log_gamma(y + th) - log_gamma(y + 1) - log_gamma(th) - th * math.log1p(m.alpha * u)
			if y > 0 {l += y * math.ln(m.alpha * u / (1 + m.alpha * u))}
		}
		_sum_add(&s, l)
	}
	return _sum(s)
}

@(private)
_glm_eta :: proc(m: ^_Glm_Model, beta: []f64, eta: []f64) {
	for i in 0 ..< len(eta) {
		s := m.offset[i] if m.offset != nil else 0
		for c, j in m.cols {s += beta[j] * c[i]}
		eta[i] = s
	}
}

// IRLS (Fisher scoring) from beta0 (nil: start from mu = y moved inside the
// support), with step halving when a step leaves the support
@(private)
_glm_irls :: proc(m: ^_Glm_Model, beta0: []f64) -> (fit: _Glm_Fit, ok: bool) {
	n, p := len(m.y), len(m.cols)
	ta := context.temp_allocator
	eta := make([]f64, n, ta)
	mu := make([]f64, n, ta)
	muc := make([]f64, n, ta)
	dmu := make([]f64, n, ta)
	beta := make([]f64, p, ta)
	z := make([]f64, n, ta)
	wt := make([]f64, n, ta)
	if beta0 != nil {
		copy(beta, beta0)
		_glm_eta(m, beta, eta)
	} else {
		ybar, wsum := 0.0, 0.0
		for v, i in m.y {ybar += m.prior[i] * v; wsum += m.prior[i]}
		ybar /= wsum
		for v, i in m.y {
			u: f64
			switch m.family {
			case .Binomial:
				u = (m.prior[i] * v + 0.5) / (m.prior[i] + 1)
			case .Gaussian, .Poisson, .Negative_Binomial, .Gamma, .Inverse_Gaussian:
				u = 0.5 * (v + ybar)
			}
			eta[i] = _link_fn(m.link, u) // a start for eta itself; the first solve absorbs any offset
		}
	}
	for i in 0 ..< n {mu[i], muc[i], dmu[i] = _linkinv(m.link, eta[i])}
	for i in 0 ..< n {if !_mu_ok(m.family, mu[i], muc[i]) {return {}, false}}
	first := beta0 == nil
	m.cur = nil if first else beta
	defer m.cur = nil
	old := make([]f64, p, ta)
	old_eta := make([]f64, n, ta)
	fit.converged = false
	it := 0
	for it < m.max_iter {
		it += 1
		for i in 0 ..< n {
			v := _variance(m.family, mu[i], muc[i], m.alpha)
			off := m.offset[i] if m.offset != nil else 0
			// working response; once beta exists, only its residual part, and the
			// solve gives the step (data far from 0 keep their digits)
			z[i] = _resid(m, mu, i) / dmu[i]
			if first {z[i] += eta[i] - off}
			wt[i] = m.prior[i] * dmu[i] * dmu[i] / v
		}
		copy(old, beta)
		copy(old_eta, eta)
		nb, rank_ok := _wls(m.cols, wt, z)
		if !rank_ok {
			// weights underflowed on the way to a boundary (separation): keep the last iterate
			if !first {break}
			return {}, false
		}
		if first {copy(beta, nb)} else {for j in 0 ..< p {beta[j] += nb[j]}}
		m.cur = beta
		// step halving while the step leaves the support or the deviance is not finite
		for half in 0 ..= 40 {
			_glm_eta(m, beta, eta)
			good := true
			for i in 0 ..< n {
				mu[i], muc[i], dmu[i] = _linkinv(m.link, eta[i])
				if !_mu_ok(m.family, mu[i], muc[i]) {good = false; break}
			}
			if good {
				new_dev := _glm_deviance(m, mu, muc)
				if !math.is_nan(new_dev) && !math.is_inf(new_dev) {break}
			}
			if first || half == 40 {return {}, false}
			for j in 0 ..< p {beta[j] = 0.5 * (beta[j] + old[j])}
		}
		// converged when the linear predictor stops moving
		change := 0.0
		if !first {
			for i in 0 ..< n {change = max(change, abs(eta[i] - old_eta[i]) / (1 + abs(eta[i])))}
		} else {
			change = math.inf_f64(1)
		}
		first = false
		if change <= 1e-14 {
			fit.converged = true
			break
		}
	}
	fit.beta = beta
	fit.mu = slice.clone(mu, ta)
	fit.iterations = it
	return fit, true
}

// weighted least squares of z on the columns by Householder QR; rank_ok
// false when a column is (nearly) a combination of the others
@(private)
_wls :: proc(cols: [][]f64, wt, z: []f64) -> (beta: []f64, rank_ok: bool) {
	r, qz := _wqr(cols, wt, z) or_return
	p := len(cols)
	beta = make([]f64, p, context.temp_allocator)
	for j := p - 1; j >= 0; j -= 1 {
		s := qz[j]
		for k in j + 1 ..< p {s -= r[j * p + k] * beta[k]}
		beta[j] = s / r[j * p + j]
	}
	return beta, true
}

// R (p x p upper, row-major) of sqrt(w) X = QR, and Qᵀ sqrt(w) z
@(private)
_wqr :: proc(cols: [][]f64, wt, z: []f64) -> (r: []f64, qz: []f64, ok: bool) {
	n, p := len(wt), len(cols)
	ta := context.temp_allocator
	a := make([][]f64, p, ta)
	sw := make([]f64, n, ta)
	for v, i in wt {sw[i] = math.sqrt(v)}
	for c, j in cols {
		a[j] = make([]f64, n, ta)
		for i in 0 ..< n {a[j][i] = sw[i] * c[i]}
	}
	qz = make([]f64, n, ta)
	if z != nil {for i in 0 ..< n {qz[i] = sw[i] * z[i]}}
	r = make([]f64, p * p, ta)
	for j in 0 ..< p {
		col := a[j]
		orig := 0.0
		for i in 0 ..< n {orig += cols[j][i] * cols[j][i] * wt[i]}
		nrm := 0.0
		for i in j ..< n {nrm += col[i] * col[i]}
		if !(nrm > 1e-24 * orig) {return nil, nil, false}
		alpha := -math.copy_sign(math.sqrt(nrm), col[j])
		col[j] -= alpha
		vtv := 0.0
		for i in j ..< n {vtv += col[i] * col[i]}
		for k in j + 1 ..< p {
			b := a[k]
			d := 0.0
			for i in j ..< n {d += col[i] * b[i]}
			f := 2 * d / vtv
			for i in j ..< n {b[i] -= f * col[i]}
			r[j * p + k] = b[j]
		}
		d := 0.0
		for i in j ..< n {d += col[i] * qz[i]}
		f := 2 * d / vtv
		for i in j ..< n {qz[i] -= f * col[i]}
		r[j * p + j] = alpha
	}
	return r, qz[:p], true
}

// (Xᵀ W X)⁻¹ at mu, from R⁻¹ R⁻ᵀ
@(private)
_glm_cov :: proc(m: ^_Glm_Model, mu, muc, dmu: []f64) -> (cov: []f64, ok: bool) {
	n, p := len(mu), len(m.cols)
	wt := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {wt[i] = m.prior[i] * dmu[i] * dmu[i] / _variance(m.family, mu[i], muc[i], m.alpha)}
	r, _ := _wqr(m.cols, wt, nil) or_return
	ri := make([]f64, p * p, context.temp_allocator) // R⁻¹, upper
	for j in 0 ..< p {
		ri[j * p + j] = 1 / r[j * p + j]
		for i := j - 1; i >= 0; i -= 1 {
			s := 0.0
			for k in i + 1 ..= j {s += r[i * p + k] * ri[k * p + j]}
			ri[i * p + j] = -s / r[i * p + i]
		}
	}
	cov = make([]f64, p * p, context.temp_allocator)
	for i in 0 ..< p {
		for j in 0 ..< p {
			s := 0.0
			for k in max(i, j) ..< p {s += ri[i * p + k] * ri[j * p + k]}
			cov[i * p + j] = s
		}
	}
	return cov, true
}

// maximum-likelihood theta = 1 / alpha of a negative binomial with means mu
// (MASS's theta.ml): Newton steps on the score, from the moment estimate
@(private)
_nb_theta_ml :: proc(y, mu: []f64, start: f64) -> f64 {
	n := f64(len(y))
	th := start
	if math.is_nan(th) {
		s := 0.0
		for v, i in y {s += (v / mu[i] - 1) * (v / mu[i] - 1)}
		th = n / s
		if !(th > 0) || math.is_inf(th) {th = 1}
	}
	for _ in 0 ..< 200 {
		score, info: _Sum
		for v, i in y {
			u := mu[i]
			_sum_add(&score, _digamma_diff(th, v) + math.ln(th / (th + u)) + (u - v) / (u + th))
			_sum_add(
				&info,
				_trigamma_diff(th, v) - 1 / th + 1 / (u + th) + (u - v) / ((u + th) * (u + th)),
			)
		}
		sc, inf := _sum(score), _sum(info)
		step := sc / inf
		nt := th + step
		for !(nt > 0) {step *= 0.5; nt = th + step}
		done := abs(step) <= 1e-15 * th
		th = nt
		if done || th > 1e12 {break}
	}
	return th
}

// ψ(t + y) - ψ(t): a direct sum for small integer y, otherwise digammas
@(private)
_digamma_diff :: proc(t, y: f64) -> f64 {
	if y == 0 {return 0}
	if y == math.floor(y) && y <= 64 {
		s := 0.0
		for j in 0 ..< int(y) {s += 1 / (t + f64(j))}
		return s
	}
	return _digamma(t + y) - _digamma(t)
}

// ψ'(t) - ψ'(t + y)
@(private)
_trigamma_diff :: proc(t, y: f64) -> f64 {
	if y == 0 {return 0}
	if y == math.floor(y) && y <= 64 {
		s := 0.0
		for j in 0 ..< int(y) {s += 1 / ((t + f64(j)) * (t + f64(j)))}
		return s
	}
	return _trigamma(t) - _trigamma(t + y)
}

@(private)
_digamma :: proc(x: f64) -> f64 {
	x := x
	r := 0.0
	for x < 10 {
		r -= 1 / x
		x += 1
	}
	f := 1 / (x * x)
	s := f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f * (1.0 / 132)))))
	return r + math.ln(x) - 0.5 / x - s
}

@(private)
_trigamma :: proc(x: f64) -> f64 {
	x := x
	r := 0.0
	for x < 10 {
		r += 1 / (x * x)
		x += 1
	}
	f := 1 / (x * x)
	s := 1 / x + f / 2 + f / x * (1.0 / 6 - f * (1.0 / 30 - f * (1.0 / 42 - f * (1.0 / 30))))
	return r + s
}
