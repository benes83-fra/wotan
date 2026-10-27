package ML

import w "../../core"
import l "../../linalg"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:simd"


OLSResult :: struct {
	beta:      []f64,
	vcov:      l.Matrix(f64),
	stderr:    []f64,
	tvalues:   []f64,
	pvalues:   []f64, // two-sided, t(df_resid)
	sigma2:    f64,
	residuals: []f64,
	fitted:    []f64,
	r2:        f64,
	r2_adj:    f64,
	fstat:     f64,
	ci_low:    []f64,
	ci_high:   []f64, // 95% t-interval
	df_resid:  int, // n - rank
	rank:      int,
}
OLSMethod :: enum {
	Cholesky,
	QR,
}

// X: n x p (rows = observations, cols = regressors)
ols_xtx :: proc(X: ^l.Matrix(f64), allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	return l.xtx_simd(X, allocator)
}

ols_xty :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	return l.xty(X, y, allocator)
}

ols_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	return ols_fit_full(X, y, method, allocator)
}


ols_from_df :: proc(
	df: ^w.DataFrame,
	y_name: string,
	x_names: []string,
	add_intercept: bool = true,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	// Build X
	X := l.matrix_from_df(df, x_names, allocator)
	defer l.matrix_free(&X)

	if add_intercept {
		// extend X with leading column of ones
		n := X.rows
		p := X.cols
		X2 := l.matrix_new(f64, n, p + 1, allocator)
		for i := 0; i < n; i += 1 {
			X2.data[i * X2.cols + 0] = 1.0
			for j := 0; j < p; j += 1 {
				X2.data[i * X2.cols + (j + 1)] = X.data[i * X.cols + j]
			}
		}
		l.matrix_free(&X)
		X = X2
	}

	y := l.vector_from_df(df, y_name, allocator)
	defer delete(y, allocator)

	return ols_fit(&X, y, method, allocator)
}
ols_fit_full :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	if X.rows != len(y) do panic("ols_fit: dimension mismatch")
	return _ols_core(X, y, X, y, nil, method, allocator)
}

// Shared least-squares core for OLS / WLS / GLS (statsmodels conventions).
//   Xw, yw  : whitened design and response (= X, y for OLS)
//   X, y    : original data; residuals and fitted values are reported on this scale
//   iota_w  : whitened column of ones (nil = all ones, i.e. no whitening)
// A constant column in X selects centered R² / F (df_model = rank - 1); without
// one they are uncentered, as in statsmodels. Rank-deficient X falls back to the
// pseudo-inverse (statsmodels' default pinv) with df_resid = n - rank.
_ols_core :: proc(
	Xw: ^l.Matrix(f64),
	yw: []f64,
	X: ^l.Matrix(f64),
	y: []f64,
	iota_w: []f64,
	method: OLSMethod,
	allocator: mem.Allocator,
) -> OLSResult {
	n := X.rows
	p := X.cols
	res := OLSResult{}

	// =====================================================================
	// 1. (XᵀX)⁻¹ (Cholesky, or pseudo-inverse if X is rank deficient) and β
	// =====================================================================
	XtX := l.xtx_simd(Xw, context.temp_allocator)
	defer l.matrix_free(&XtX)
	Xty := l.xty(Xw, yw, context.temp_allocator)
	defer delete(Xty, context.temp_allocator)

	XtX_inv := l.matrix_new(f64, p, p, context.temp_allocator)
	defer l.matrix_free(&XtX_inv)
	rank := _ols_inverse(&XtX, &XtX_inv)

	beta := make([]f64, p, allocator)
	if rank == p && method == .QR && n > 0 {
		lstsq_res := l.lstsq(Xw, yw, .QR, context.temp_allocator)
		copy(beta, lstsq_res.beta)
		l.lstsq_result_free(&lstsq_res, context.temp_allocator)
	} else {
		for i in 0 ..< p {
			s := 0.0
			for j in 0 ..< p {s += XtX_inv.data[i * p + j] * Xty[j]}
			beta[i] = s
		}
	}

	// =====================================================================
	// 2. Fitted values and residuals (original scale); SSR (whitened)
	// =====================================================================
	fitted := make([]f64, n, allocator)
	residuals := make([]f64, n, allocator)
	for i in 0 ..< n {
		s := 0.0
		for j in 0 ..< p {s += X.data[i * X.cols + j] * beta[j]}
		fitted[i] = s
		residuals[i] = y[i] - s
	}
	ssr := 0.0
	for i in 0 ..< n {
		s := 0.0
		for j in 0 ..< p {s += Xw.data[i * Xw.cols + j] * beta[j]}
		e := yw[i] - s
		ssr += e * e
	}

	// =====================================================================
	// 3. σ² = SSR / (n - rank), vcov = σ² (XᵀX)⁻¹
	// =====================================================================
	df_resid := n - rank
	sigma2 := ssr / f64(df_resid) if df_resid > 0 else math.nan_f64()

	vcov := l.matrix_new(f64, p, p, allocator)
	for i in 0 ..< p * p {vcov.data[i] = sigma2 * XtX_inv.data[i]}

	stderr := make([]f64, p, allocator)
	tvalues := make([]f64, p, allocator)
	pvalues := make([]f64, p, allocator)
	ci_low := make([]f64, p, allocator)
	ci_high := make([]f64, p, allocator)
	tcrit := student_t_ppf(0.975, f64(df_resid)) if df_resid > 0 else math.nan_f64()
	for j in 0 ..< p {
		stderr[j] = math.sqrt(vcov.data[j * p + j])
		tvalues[j] = beta[j] / stderr[j] if stderr[j] > 0.0 else math.nan_f64()
		pvalues[j] = student_t_sf2(tvalues[j], f64(df_resid)) if df_resid > 0 else math.nan_f64()
		ci_low[j] = beta[j] - tcrit * stderr[j]
		ci_high[j] = beta[j] + tcrit * stderr[j]
	}

	// =====================================================================
	// 4. R², adjusted R², F (centered iff X has a constant column)
	// =====================================================================
	has_const := _ols_has_constant(X)
	tss := 0.0
	if has_const {
		// weighted mean of y: (ywᵀ ιw) / (ιwᵀ ιw)
		num, den := 0.0, 0.0
		for i in 0 ..< n {
			iw := iota_w[i] if iota_w != nil else 1.0
			num += yw[i] * iw
			den += iw * iw
		}
		mean := num / den
		for i in 0 ..< n {
			iw := iota_w[i] if iota_w != nil else 1.0
			e := yw[i] - mean * iw
			tss += e * e
		}
	} else {
		for i in 0 ..< n {tss += yw[i] * yw[i]}
	}
	k_const := 1 if has_const else 0
	df_model := rank - k_const
	r2 := 1.0 - ssr / tss
	r2_adj := 1.0 - f64(n - k_const) / f64(df_resid) * (1.0 - r2)
	fstat := math.nan_f64()
	if df_model > 0 && df_resid > 0 {
		fstat = ((tss - ssr) / f64(df_model)) / (ssr / f64(df_resid))
	}

	res.beta = beta
	res.vcov = vcov
	res.stderr = stderr
	res.tvalues = tvalues
	res.pvalues = pvalues
	res.sigma2 = sigma2
	res.residuals = residuals
	res.fitted = fitted
	res.r2 = r2
	res.r2_adj = r2_adj
	res.fstat = fstat
	res.ci_low = ci_low
	res.ci_high = ci_high
	res.df_resid = df_resid
	res.rank = rank
	return res
}

// Summary statistics for penalised fits (ridge / lasso) given β, the
// effective model degrees of freedom (incl. intercept) and the coefficient
// covariance (nil = not available → NaN standard errors / p-values / CIs).
// R² and F follow the same centered/uncentered rule as _ols_core.
_penalized_summary :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	beta: []f64,
	df_fit: f64,
	vcov_unscaled: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> OLSResult {
	n, p := X.rows, X.cols
	res := OLSResult{}
	fitted := make([]f64, n, allocator)
	residuals := make([]f64, n, allocator)
	ssr := 0.0
	for i in 0 ..< n {
		s := 0.0
		for j in 0 ..< p {s += X.data[i * X.cols + j] * beta[j]}
		fitted[i] = s
		residuals[i] = y[i] - s
		ssr += residuals[i] * residuals[i]
	}
	df_resid := f64(n) - df_fit
	sigma2 := ssr / df_resid if df_resid > 0 else math.nan_f64()

	vcov := l.matrix_new(f64, p, p, allocator)
	stderr := make([]f64, p, allocator)
	tvalues := make([]f64, p, allocator)
	pvalues := make([]f64, p, allocator)
	ci_low := make([]f64, p, allocator)
	ci_high := make([]f64, p, allocator)
	tcrit := student_t_ppf(0.975, df_resid) if df_resid > 0 else math.nan_f64()
	for j in 0 ..< p {
		if vcov_unscaled != nil {
			for k in 0 ..< p {vcov.data[j * p + k] = sigma2 * vcov_unscaled.data[j * p + k]}
			stderr[j] = math.sqrt(vcov.data[j * p + j])
		} else {
			for k in 0 ..< p {vcov.data[j * p + k] = math.nan_f64()}
			stderr[j] = math.nan_f64()
		}
		tvalues[j] = beta[j] / stderr[j] if stderr[j] > 0.0 else math.nan_f64()
		pvalues[j] = student_t_sf2(tvalues[j], df_resid) if df_resid > 0 else math.nan_f64()
		ci_low[j] = beta[j] - tcrit * stderr[j]
		ci_high[j] = beta[j] + tcrit * stderr[j]
	}

	has_const := _ols_has_constant(X)
	tss := 0.0
	if has_const {
		mean := 0.0
		for v in y do mean += v
		mean /= f64(n)
		for v in y do tss += (v - mean) * (v - mean)
	} else {
		for v in y do tss += v * v
	}
	k_const := 1.0 if has_const else 0.0
	df_model := df_fit - k_const
	r2 := 1.0 - ssr / tss
	r2_adj := 1.0 - (f64(n) - k_const) / df_resid * (1.0 - r2)
	fstat := math.nan_f64()
	if df_model > 0 && df_resid > 0 {
		fstat = ((tss - ssr) / df_model) / (ssr / df_resid)
	}

	res.beta = beta
	res.vcov = vcov
	res.stderr = stderr
	res.tvalues = tvalues
	res.pvalues = pvalues
	res.sigma2 = sigma2
	res.residuals = residuals
	res.fitted = fitted
	res.r2 = r2
	res.r2_adj = r2_adj
	res.fstat = fstat
	res.ci_low = ci_low
	res.ci_high = ci_high
	res.df_resid = int(math.round(df_resid))
	res.rank = int(math.round(df_fit))
	return res
}

// Penalty weight per column: 0 for a constant (intercept) column, 1 otherwise,
// so ridge/lasso on [1, X] match sklearn's fit_intercept=True.
_penalty_mask :: proc(X: ^l.Matrix(f64), allocator: mem.Allocator) -> []f64 {
	mask := make([]f64, X.cols, allocator)
	for j in 0 ..< X.cols {
		mask[j] = 1.0
		if X.rows == 0 do continue
		v0 := X.data[j]
		if v0 == 0.0 do continue
		same := true
		for i in 1 ..< X.rows {
			if X.data[i * X.cols + j] != v0 {same = false; break}
		}
		if same do mask[j] = 0.0
	}
	return mask
}

// out = A⁻¹ for SPD A via Cholesky; if A is singular, out = pinv(A) from its
// eigen-decomposition (eigenvalues below p·eps·max are dropped). Returns the rank.
_ols_inverse :: proc(A: ^l.Matrix(f64), out: ^l.Matrix(f64)) -> int {
	p := A.rows
	if p == 0 do return 0
	L := l.matrix_new(f64, p, p, context.temp_allocator)
	defer l.matrix_free(&L)
	copy(L.data, A.data)
	// A pivot that collapsed to rounding level (L_jj² ≤ 1e-12 A_jj) means A is
	// singular to working precision: use the pseudo-inverse instead
	chol_ok := l.cholesky_try(&L)
	for j in 0 ..< p {
		if !chol_ok do break
		if !(L.data[j * p + j] * L.data[j * p + j] > 1e-12 * A.data[j * p + j]) do chol_ok = false
	}
	if chol_ok {
		e := make([]f64, p, context.temp_allocator)
		defer delete(e, context.temp_allocator)
		for k in 0 ..< p {
			for i in 0 ..< p {e[i] = 0.0}
			e[k] = 1.0
			// (XᵀX)⁻¹ e_k via L Lᵀ: L z = e_k, then Lᵀ x = z
			z := l.forward_substitute_simd(&L, e, context.temp_allocator)
			x := l.backward_substitute_simd(&L, z, context.temp_allocator)
			for i in 0 ..< p {out.data[i * p + k] = x[i]}
			delete(z, context.temp_allocator)
			delete(x, context.temp_allocator)
		}
		return p
	}

	Acopy := l.matrix_new(f64, p, p, context.temp_allocator)
	defer l.matrix_free(&Acopy)
	copy(Acopy.data, A.data)
	vals, V := l.eigh(&Acopy, .Ascending, context.temp_allocator)
	defer {
		delete(vals, context.temp_allocator)
		l.matrix_free(&V)
	}
	vmax := 0.0
	for v in vals do vmax = max(vmax, math.abs(v))
	tol := vmax * f64(p) * 2.220446049250313e-16
	rank := 0
	for i in 0 ..< p * p {out.data[i] = 0.0}
	for k in 0 ..< p {
		if vals[k] <= tol do continue
		rank += 1
		inv := 1.0 / vals[k]
		for i in 0 ..< p {
			for j in 0 ..< p {
				out.data[i * p + j] += inv * V.data[i * p + k] * V.data[j * p + k]
			}
		}
	}
	return rank
}

// True if some column of X is constant and non-zero (statsmodels' k_constant)
_ols_has_constant :: proc(X: ^l.Matrix(f64)) -> bool {
	if X.rows == 0 do return false
	for j in 0 ..< X.cols {
		v0 := X.data[j]
		if v0 == 0.0 do continue
		same := true
		for i in 1 ..< X.rows {
			if X.data[i * X.cols + j] != v0 {same = false; break}
		}
		if same do return true
	}
	return false
}


// Free OLSResult fields (call when done)
_ols_result_free :: proc(res: ^OLSResult, allocator: mem.Allocator) {
	if len(res.beta) > 0 {
		defer delete(res.beta, allocator)
		// mem.free(transmute(rawptr) &res.beta[0], allocator)
	}
	if len(res.residuals) > 0 {
		defer delete(res.residuals, allocator)
		// mem.free(transmute(rawptr) &res.residuals[0], allocator)
	}
	if len(res.stderr) > 0 {
		defer delete(res.stderr, allocator)
		// mem.free(transmute(rawptr) &res.stderr[0], allocator)
	}
	if len(res.pvalues) > 0 {
		defer delete(res.pvalues, allocator)
	}
	if len(res.tvalues) > 0 {
		defer delete(res.tvalues, allocator)
		// mem.free(transmute(rawptr) &res.tvalues[0], allocator)
	}
	if len(res.fitted) > 0 {
		defer delete(res.fitted, allocator)
		// mem.free(transmute(rawptr) &res.fitted[0], allocator)
	}
	if len(res.ci_low) > 0 {
		defer delete(res.ci_low, allocator)
		// mem.free(transmute(rawptr) &res.ci_low[0], allocator)
	}
	if len(res.ci_high) > 0 {
		defer delete(res.ci_high, allocator)
		// mem.free(transmute(rawptr) &res.ci_high[0], allocator)
	}
	if res.vcov.data != nil {
		l.matrix_free(&res.vcov)
	}
}
