package ML

import l "../../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Robust standard errors for OLS (statsmodels OLS(...).fit(cov_type=…))
// ============================================================================
// Sandwich covariance V = (XᵀX)⁻¹ M (XᵀX)⁻¹ with
//   HC0: M = Σ e_i² x_i x_iᵀ                     (White)
//   HC1: HC0 · n / (n - k)
//   HC2: e_i² / (1 - h_i),  HC3: e_i² / (1 - h_i)²   (h_i = leverage)
//   HAC: Newey-West, M = Γ_0 + Σ_{j=1}^{L} (1 - j/(L+1)) (Γ_j + Γ_jᵀ),
//        Γ_j = Σ_t (x_t e_t)(x_{t-j} e_{t-j})ᵀ; optional · n/(n - k)
//   Cluster: M = Σ_g (Σ_{i∈g} x_i e_i)(…)ᵀ · G/(G-1) · (n-1)/(n-k)
// Inference as statsmodels: z-tests (normal) unless use_t, then t with
// df = n - k (clusters: G - 1); F = Wald test that all slopes (all coefficients
// except a constant column) are 0, divided by their number, ~ F(q, df).

CovType :: enum {
	NonRobust, // classical σ² (XᵀX)⁻¹ (ols_fit)
	HC0,
	HC1,
	HC2,
	HC3,
	HAC, // Newey-West
	Cluster,
}

RobustCovSpec :: struct {
	type:                  CovType,
	maxlags:               int, // HAC lags L (< 0: ⌊4 (n/100)^(2/9)⌋, statsmodels' default rule)
	hac_correction:        bool, // HAC: multiply by n/(n-k) (statsmodels use_correction, default off)
	groups:                []int, // Cluster: group label of every observation
	cluster_no_correction: bool, // Cluster: skip G/(G-1)·(n-1)/(n-k) (statsmodels use_correction=False)
	use_t:                 bool, // t / F with finite df instead of normal (statsmodels use_t)
}

// OLS fit with robust standard errors
ols_fit_robust :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	spec: RobustCovSpec,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	res := ols_fit(X, y, method, allocator)
	ols_set_robust_cov(&res, X, spec)
	return res
}

// Replace vcov / stderr / tvalues / pvalues / CIs / F of an OLS result (fitted
// on X) by the robust versions. β, residuals, R² and σ² are unchanged.
ols_set_robust_cov :: proc(res: ^OLSResult, X: ^l.Matrix(f64), spec: RobustCovSpec) {
	n, k := X.rows, X.cols
	if len(res.beta) != k || len(res.residuals) != n do panic("ols_set_robust_cov: res was not fitted on this X")
	if spec.type == .NonRobust do panic("ols_set_robust_cov: use ols_fit for the classical covariance")
	e := res.residuals

	// bread: (XᵀX)⁻¹ (pseudo-inverse if rank deficient, like statsmodels)
	XtX := l.xtx_simd(X, context.temp_allocator)
	defer l.matrix_free(&XtX)
	B := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&B)
	_ols_inverse(&XtX, &B)

	// x_i e_i (scaled for HC2 / HC3)
	xu := make([]f64, n * k, context.temp_allocator)
	defer delete(xu, context.temp_allocator)
	for i in 0 ..< n {
		w := e[i]
		if spec.type == .HC2 || spec.type == .HC3 {
			xi := X.data[i * k:(i + 1) * k]
			h := 0.0
			for a in 0 ..< k {
				s := 0.0
				for b in 0 ..< k {s += B.data[a * k + b] * xi[b]}
				h += xi[a] * s
			}
			w = e[i] / math.sqrt(1 - h) if spec.type == .HC2 else e[i] / (1 - h)
		}
		for a in 0 ..< k {xu[i * k + a] = X.data[i * k + a] * w}
	}

	// meat
	M := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&M)
	add_outer :: proc(M: ^l.Matrix(f64), u, v: []f64, wgt: f64) {
		k := M.rows
		for a in 0 ..< k {
			for b in 0 ..< k {M.data[a * k + b] += wgt * u[a] * v[b]}
		}
	}
	df_inf := f64(res.df_resid)
	scale := 1.0
	switch spec.type {
	case .NonRobust:
	case .HC0, .HC1, .HC2, .HC3:
		for i in 0 ..< n {add_outer(&M, xu[i * k:(i + 1) * k], xu[i * k:(i + 1) * k], 1)}
		if spec.type == .HC1 do scale = f64(n) / f64(n - k)
	case .HAC:
		L := spec.maxlags
		if L < 0 do L = int(math.floor(4 * math.pow(f64(n) / 100, 2.0 / 9.0)))
		for i in 0 ..< n {add_outer(&M, xu[i * k:(i + 1) * k], xu[i * k:(i + 1) * k], 1)}
		for j in 1 ..= min(L, n - 1) {
			wj := 1 - f64(j) / f64(L + 1)
			for t in j ..< n {
				add_outer(&M, xu[t * k:(t + 1) * k], xu[(t - j) * k:(t - j + 1) * k], wj)
				add_outer(&M, xu[(t - j) * k:(t - j + 1) * k], xu[t * k:(t + 1) * k], wj)
			}
		}
		if spec.hac_correction do scale = f64(n) / f64(n - k)
	case .Cluster:
		if len(spec.groups) != n do panic("ols_set_robust_cov: need one group label per observation")
		labels := slice.clone(spec.groups, context.temp_allocator)
		defer delete(labels, context.temp_allocator)
		slice.sort(labels)
		uniq := slice.unique(labels)
		G := len(uniq)
		if G < 2 do panic("ols_set_robust_cov: need at least 2 clusters")
		sums := make([]f64, G * k, context.temp_allocator)
		defer delete(sums, context.temp_allocator)
		for i in 0 ..< n {
			g, _ := slice.binary_search(uniq, spec.groups[i])
			for a in 0 ..< k {sums[g * k + a] += xu[i * k + a]}
		}
		for g in 0 ..< G {add_outer(&M, sums[g * k:(g + 1) * k], sums[g * k:(g + 1) * k], 1)}
		if !spec.cluster_no_correction do scale = f64(G) / f64(G - 1) * f64(n - 1) / f64(n - k)
		df_inf = f64(G - 1) // statsmodels adjust_df
	}

	// V = B M B · scale
	tmp := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&tmp)
	for a in 0 ..< k {
		for b in 0 ..< k {
			s := 0.0
			for c in 0 ..< k {s += B.data[a * k + c] * M.data[c * k + b]}
			tmp.data[a * k + b] = s
		}
	}
	for a in 0 ..< k {
		for b in 0 ..< k {
			s := 0.0
			for c in 0 ..< k {s += tmp.data[a * k + c] * B.data[c * k + b]}
			res.vcov.data[a * k + b] = scale * s
		}
	}

	// coefficient inference
	crit := student_t_ppf(0.975, df_inf) if spec.use_t else _NORM_975
	for j in 0 ..< k {
		se := math.sqrt(res.vcov.data[j * k + j])
		res.stderr[j] = se
		t := res.beta[j] / se if se > 0 else math.nan_f64()
		res.tvalues[j] = t
		res.pvalues[j] = student_t_sf2(t, df_inf) if spec.use_t else math.erfc(math.abs(t) / math.SQRT_TWO)
		res.ci_low[j] = res.beta[j] - crit * se
		res.ci_high[j] = res.beta[j] + crit * se
	}

	// Wald F for all slopes (statsmodels fvalue with a robust cov_type)
	const_idx := _ols_constant_index(X)
	idx := make([dynamic]int, 0, k, context.temp_allocator)
	defer delete(idx)
	for j in 0 ..< k {if j != const_idx do append(&idx, j)}
	q := len(idx)
	res.fstat, res.f_pvalue = math.nan_f64(), math.nan_f64()
	if q > 0 {
		R := l.matrix_new(f64, q, q, context.temp_allocator)
		Ri := l.matrix_new(f64, q, q, context.temp_allocator)
		defer {
			l.matrix_free(&R)
			l.matrix_free(&Ri)
		}
		for a in 0 ..< q {
			for b in 0 ..< q {R.data[a * q + b] = res.vcov.data[idx[a] * k + idx[b]]}
		}
		_ols_inverse(&R, &Ri)
		w := 0.0
		for a in 0 ..< q {
			for b in 0 ..< q {w += res.beta[idx[a]] * Ri.data[a * q + b] * res.beta[idx[b]]}
		}
		res.fstat = w / f64(q)
		res.f_pvalue = f_dist_sf(res.fstat, f64(q), df_inf)
	}
	res.cov_type = spec.type
}

@(private = "file")
_NORM_975 :: 1.959963984540054 // standard normal 97.5 % quantile

// Index of the first constant (non-zero, identical in every row) column, -1 if none
_ols_constant_index :: proc(X: ^l.Matrix(f64)) -> int {
	if X.rows == 0 do return -1
	for j in 0 ..< X.cols {
		v0 := X.data[j]
		if v0 == 0.0 do continue
		same := true
		for i in 1 ..< X.rows {
			if X.data[i * X.cols + j] != v0 {same = false; break}
		}
		if same do return j
	}
	return -1
}
