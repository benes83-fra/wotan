package ml_finance

import l "../linalg"
import "core:mem"

// ============================================================================
// INSTRUMENTED PCA (Kelly, Pruitt & Su 2019, "Characteristics are covariances")
// ============================================================================
// Returns of asset i over t -> t+1 with characteristics z_{i,t} (L of them, known at t):
//   r_{i,t+1} = z_{i,t}' Gamma_alpha + z_{i,t}' Gamma_beta f_{t+1} + e
// (Gamma_alpha only with intercept). Fitted by alternating least squares on the
// characteristic-managed portfolios Q_t = Z_t' r_{t+1} / N_t and W_t = Z_t' Z_t / N_t,
// as the authors' package (ipca 0.6.7, portfolio ALS):
//   start   Gamma = first K (+1) left singular vectors of Q [L, T]
//   factors f_t = (Gb' W_t Gb)^-1 Gb' (Q_t - W_t Ga)
//   Gamma   vec(Gamma) = (sum_t N_t W_t (x) g_t g_t')^-1 sum_t N_t Q_t (x) g_t, g_t = (f_t; 1)
//   normalise Gb' Gb = I, F F' diagonal with decreasing entries, every factor's mean
//           positive; Ga projected orthogonal to Gb
//   stop    when no entry of Gamma (and, without intercept, of F) moves more than tol
// The panel may be unbalanced: observation k is asset row X[k], return y[k], date
// times[k] in 0 ..< T.

IPCAParams :: struct {
	n_factors: int,
	intercept: bool, // Gamma_alpha: characteristic-driven alphas
	max_iter:  int, // 0: 10000
	tol:       f64, // 0: 1e-5 (the package's iter_tol)
}

IPCA :: struct {
	gamma:     l.Matrix(f64), // [L, K (+1)]: Gamma_beta, then Gamma_alpha
	factors:   l.Matrix(f64), // [K, T]
	n_factors: int,
	intercept: bool,
	n_iter:    int,
	converged: bool,
	allocator: mem.Allocator,
}

ipca_free :: proc(m: ^IPCA) {
	l.matrix_free(&m.gamma)
	l.matrix_free(&m.factors)
}

// Characteristic-managed portfolios: Q [T, L] (row t = Z_t' r_{t+1} / N_t; nil without
// y), W [T, L, L] (Z_t' Z_t / N_t), N [T] observations per date
IPCAPortfolios :: struct {
	Q, W, N:   []f64,
	L, T:      int,
	allocator: mem.Allocator,
}

ipca_portfolios_free :: proc(p: ^IPCAPortfolios) {
	delete(p.Q, p.allocator)
	delete(p.W, p.allocator)
	delete(p.N, p.allocator)
}

ipca_portfolios :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	times: []int,
	allocator := context.allocator,
) -> IPCAPortfolios {
	assert(len(times) == X.rows, "ipca_portfolios: one date per observation")
	assert(y == nil || len(y) == X.rows, "ipca_portfolios: one return per observation")
	L := X.cols
	T := 0
	for t in times {assert(t >= 0, "ipca_portfolios: negative date"); T = max(T, t + 1)}
	p := IPCAPortfolios {
		W         = make([]f64, T * L * L, allocator),
		N         = make([]f64, T, allocator),
		L         = L,
		T         = T,
		allocator = allocator,
	}
	if y != nil {p.Q = make([]f64, T * L, allocator)}
	for k in 0 ..< X.rows {
		t := times[k]
		z := X.data[k * L:][:L]
		p.N[t] += 1
		W := p.W[t * L * L:][:L * L]
		for a in 0 ..< L {
			for b in 0 ..< L {W[a * L + b] += z[a] * z[b]}
			if y != nil {p.Q[t * L + a] += z[a] * y[k]}
		}
	}
	for t in 0 ..< T {
		assert(p.N[t] > 0, "ipca_portfolios: a date without observations")
		for &v in p.W[t * L * L:][:L * L] {v /= p.N[t]}
		if y != nil {for &v in p.Q[t * L:][:L] {v /= p.N[t]}}
	}
	return p
}

// Factor at one date from Gamma: (Gb' W Gb)^-1 Gb' (q - W Ga); f [K]
@(private = "file")
_ipca_factor :: proc(gamma: ^l.Matrix(f64), K: int, intercept: bool, W, q: []f64, f: []f64) {
	L, Kt := gamma.rows, gamma.cols
	G := gamma.data
	m1 := l.matrix_new(f64, K, K, context.temp_allocator)
	defer l.matrix_free(&m1)
	m2 := make([]f64, K, context.temp_allocator)
	defer delete(m2, context.temp_allocator)
	WG := make([]f64, L * Kt, context.temp_allocator) // W Gamma
	defer delete(WG, context.temp_allocator)
	for a in 0 ..< L {
		for k in 0 ..< Kt {
			s := 0.0
			for b in 0 ..< L {s += W[a * L + b] * G[b * Kt + k]}
			WG[a * Kt + k] = s
		}
	}
	for i in 0 ..< K {
		for j in 0 ..< K {
			s := 0.0
			for a in 0 ..< L {s += G[a * Kt + i] * WG[a * Kt + j]}
			m1.data[i * K + j] = s
		}
		s := 0.0
		for a in 0 ..< L {
			s += G[a * Kt + i] * q[a]
			if intercept {s -= G[a * Kt + i] * WG[a * Kt + K]}
		}
		m2[i] = s
	}
	x := l.solve_spd_cholesky(&m1, m2, .Unblocked, context.temp_allocator)
	copy(f, x)
	delete(x, context.temp_allocator)
}

ipca_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	times: []int,
	params: IPCAParams,
	allocator := context.allocator,
) -> IPCA {
	p := ipca_portfolios(X, y, times, context.temp_allocator)
	defer ipca_portfolios_free(&p)
	return ipca_fit_portfolios(&p, params, allocator)
}

ipca_fit_portfolios :: proc(
	p: ^IPCAPortfolios,
	params: IPCAParams,
	allocator := context.allocator,
) -> IPCA {
	L, T := p.L, p.T
	K := params.n_factors
	Kt := K + (1 if params.intercept else 0)
	assert(K >= 1 && Kt <= L, "ipca_fit: need 1 <= n_factors (+ intercept) <= n_characteristics")
	assert(p.Q != nil, "ipca_fit: portfolios without returns")
	max_iter := params.max_iter if params.max_iter > 0 else 10000
	tol := params.tol if params.tol > 0 else 1e-5

	// start: left singular vectors of Q [L, T] = eigenvectors of Q Q'; F = U' Q
	G := l.matrix_new(f64, L, Kt, allocator)
	F := l.matrix_new(f64, K, T, allocator)
	{
		QQ := l.matrix_new(f64, L, L, context.temp_allocator)
		defer l.matrix_free(&QQ)
		for a in 0 ..< L {
			for b in 0 ..< L {
				s := 0.0
				for t in 0 ..< T {s += p.Q[t * L + a] * p.Q[t * L + b]}
				QQ.data[a * L + b] = s
			}
		}
		ev, V := l.eigh(&QQ, .Descending, context.temp_allocator)
		defer {delete(ev, context.temp_allocator); l.matrix_free(&V)}
		for a in 0 ..< L {for k in 0 ..< Kt {G.data[a * Kt + k] = V.data[a * L + k]}}
		for k in 0 ..< K {
			for t in 0 ..< T {
				s := 0.0
				for a in 0 ..< L {s += V.data[a * L + k] * p.Q[t * L + a]}
				F.data[k * T + t] = s
			}
		}
	}

	model := IPCA {
		n_factors = K,
		intercept = params.intercept,
		allocator = allocator,
	}
	Gn := l.matrix_new(f64, L, Kt, context.temp_allocator)
	defer l.matrix_free(&Gn)
	Fn := l.matrix_new(f64, K, T, context.temp_allocator)
	defer l.matrix_free(&Fn)
	g := make([]f64, Kt, context.temp_allocator)
	defer delete(g, context.temp_allocator)
	LK := L * Kt
	num := make([]f64, LK, context.temp_allocator)
	defer delete(num, context.temp_allocator)
	den := l.matrix_new(f64, LK, LK, context.temp_allocator)
	defer l.matrix_free(&den)
	f := make([]f64, K, context.temp_allocator)
	defer delete(f, context.temp_allocator)

	for iter := 0; iter <= max_iter; iter += 1 {
		// 1. factors
		for t in 0 ..< T {
			_ipca_factor(&G, K, params.intercept, p.W[t * L * L:][:L * L], p.Q[t * L:][:L], f)
			for k in 0 ..< K {Fn.data[k * T + t] = f[k]}
		}
		// 2. Gamma
		for &v in num {v = 0}
		for &v in den.data {v = 0}
		for t in 0 ..< T {
			for k in 0 ..< K {g[k] = Fn.data[k * T + t]}
			if params.intercept {g[K] = 1}
			n := p.N[t]
			W := p.W[t * L * L:][:L * L]
			for a in 0 ..< L {
				for k in 0 ..< Kt {
					num[a * Kt + k] += p.Q[t * L + a] * g[k] * n
					for b in 0 ..< L {
						wab := W[a * L + b] * n
						for j in 0 ..< Kt {den.data[(a * Kt + k) * LK + b * Kt + j] += wab * g[k] * g[j]}
					}
				}
			}
		}
		sol := l.solve_spd_cholesky(&den, num, .Unblocked, context.temp_allocator)
		copy(Gn.data, sol)
		delete(sol, context.temp_allocator)
		// 3. normalisation
		_ipca_normalise(&Gn, &Fn, K, params.intercept)
		// 4. convergence
		change := 0.0
		for v, i in Gn.data {change = max(change, abs(v - G.data[i]))}
		if !params.intercept {
			for v, i in Fn.data {change = max(change, abs(v - F.data[i]))}
		}
		copy(G.data, Gn.data)
		copy(F.data, Fn.data)
		model.n_iter = iter + 1
		if !(change > tol) {model.converged = true; break}
	}
	model.gamma, model.factors = G, F
	return model
}

// Gb' Gb = I and F F' diagonal (decreasing) via R1 = chol(Gb' Gb)' and the eigenvectors
// R2 of R1 F F' R1': Gb <- Gb R1^-1 R2, F <- R2' R1 F; signs so every factor's mean is
// positive; with intercept Ga <- (I - Gb Gb') Ga, then F_t += Gb' Ga and the signs again
// (as the package: Gb' Ga is taken after the projection, so it is ~0)
@(private = "file")
_ipca_normalise :: proc(G: ^l.Matrix(f64), F: ^l.Matrix(f64), K: int, intercept: bool) {
	L, Kt, T := G.rows, G.cols, F.cols
	A := l.matrix_new(f64, K, K, context.temp_allocator) // Gb' Gb, then its Cholesky factor
	defer l.matrix_free(&A)
	for i in 0 ..< K {
		for j in 0 ..< K {
			s := 0.0
			for a in 0 ..< L {s += G.data[a * Kt + i] * G.data[a * Kt + j]}
			A.data[i * K + j] = s
		}
	}
	l.cholesky_decompose(&A) // lower C, R1 = C'
	// R1 F
	RF := make([]f64, K * T, context.temp_allocator)
	defer delete(RF, context.temp_allocator)
	for i in 0 ..< K {
		for t in 0 ..< T {
			s := 0.0
			for j in i ..< K {s += A.data[j * K + i] * F.data[j * T + t]}
			RF[i * T + t] = s
		}
	}
	M := l.matrix_new(f64, K, K, context.temp_allocator)
	defer l.matrix_free(&M)
	for i in 0 ..< K {
		for j in 0 ..< K {
			s := 0.0
			for t in 0 ..< T {s += RF[i * T + t] * RF[j * T + t]}
			M.data[i * K + j] = s
		}
	}
	ev, R2 := l.eigh(&M, .Descending, context.temp_allocator)
	defer {delete(ev, context.temp_allocator); l.matrix_free(&R2)}
	// Gb R1^-1: row a solves x R1 = Gb[a] (x C' = g: forward through the columns of C')
	tmp := make([]f64, K, context.temp_allocator)
	defer delete(tmp, context.temp_allocator)
	for a in 0 ..< L {
		for j in 0 ..< K {
			s := G.data[a * Kt + j]
			for i in 0 ..< j {s -= tmp[i] * A.data[j * K + i]}
			tmp[j] = s / A.data[j * K + j]
		}
		for k in 0 ..< K {
			s := 0.0
			for j in 0 ..< K {s += tmp[j] * R2.data[j * K + k]}
			G.data[a * Kt + k] = s
		}
	}
	for k in 0 ..< K {
		for t in 0 ..< T {
			s := 0.0
			for j in 0 ..< K {s += R2.data[j * K + k] * RF[j * T + t]}
			F.data[k * T + t] = s
		}
	}
	_ipca_signs(G, F, K)
	if intercept {
		// Ga <- Ga - Gb (Gb' Ga)
		c := make([]f64, K, context.temp_allocator)
		defer delete(c, context.temp_allocator)
		for k in 0 ..< K {
			s := 0.0
			for a in 0 ..< L {s += G.data[a * Kt + k] * G.data[a * Kt + K]}
			c[k] = s
		}
		for a in 0 ..< L {
			s := 0.0
			for k in 0 ..< K {s += G.data[a * Kt + k] * c[k]}
			G.data[a * Kt + K] -= s
		}
		// F_t += Gb' Ga with the projected Ga
		for k in 0 ..< K {
			s := 0.0
			for a in 0 ..< L {s += G.data[a * Kt + k] * G.data[a * Kt + K]}
			for t in 0 ..< T {F.data[k * T + t] += s}
		}
		_ipca_signs(G, F, K)
	}
}

@(private = "file")
_ipca_signs :: proc(G: ^l.Matrix(f64), F: ^l.Matrix(f64), K: int) {
	L, Kt, T := G.rows, G.cols, F.cols
	for k in 0 ..< K {
		m := 0.0
		for t in 0 ..< T {m += F.data[k * T + t]}
		if m < 0 {
			for t in 0 ..< T {F.data[k * T + t] = -F.data[k * T + t]}
			for a in 0 ..< L {G.data[a * Kt + k] = -G.data[a * Kt + k]}
		}
	}
}

// Fitted returns z' Gamma g_t with g_t = (f_t; 1); with mean_factor the time-series
// mean of the factors (the predictive fit)
ipca_predict :: proc(
	m: ^IPCA,
	X: ^l.Matrix(f64),
	times: []int,
	mean_factor := false,
	allocator := context.allocator,
) -> []f64 {
	L, Kt, K, T := m.gamma.rows, m.gamma.cols, m.n_factors, m.factors.cols
	assert(X.cols == L, "ipca_predict: characteristics differ in number")
	lam := make([]f64, K, context.temp_allocator)
	defer delete(lam, context.temp_allocator)
	for k in 0 ..< K {for t in 0 ..< T {lam[k] += m.factors.data[k * T + t] / f64(T)}}
	out := make([]f64, X.rows, allocator)
	for r in 0 ..< X.rows {
		t := times[r]
		s := 0.0
		for a in 0 ..< L {
			z := X.data[r * L + a]
			for k in 0 ..< K {
				fk := lam[k] if mean_factor else m.factors.data[k * T + t]
				s += z * m.gamma.data[a * Kt + k] * fk
			}
			if m.intercept {s += z * m.gamma.data[a * Kt + K]}
		}
		out[r] = s
	}
	return out
}

// Kelly-Pruitt-Su R²s (not demeaned): total 1 - sum (r - z' Gamma g_t)² / sum r²,
// predictive with the mean factor in place of f_t
ipca_r2 :: proc(m: ^IPCA, X: ^l.Matrix(f64), y: []f64, times: []int) -> (total, predictive: f64) {
	a := ipca_predict(m, X, times, false, context.temp_allocator)
	defer delete(a, context.temp_allocator)
	b := ipca_predict(m, X, times, true, context.temp_allocator)
	defer delete(b, context.temp_allocator)
	ss, e1, e2 := 0.0, 0.0, 0.0
	for v, i in y {
		ss += v * v
		e1 += (v - a[i]) * (v - a[i])
		e2 += (v - b[i]) * (v - b[i])
	}
	return 1 - e1 / ss, 1 - e2 / ss
}

// Out of sample: the factor realisation at a date from a fitted Gamma, given that date's
// characteristics Z [N, L] and returns (the cross-sectional regression on Z Gamma_beta
// net of the alphas); f [K]
ipca_factor_at :: proc(m: ^IPCA, Z: ^l.Matrix(f64), r: []f64, f: []f64) {
	L := Z.cols
	assert(
		L == m.gamma.rows && len(r) == Z.rows && len(f) == m.n_factors,
		"ipca_factor_at: dimensions",
	)
	times := make([]int, Z.rows, context.temp_allocator)
	defer delete(times, context.temp_allocator)
	p := ipca_portfolios(Z, r, times, context.temp_allocator)
	defer ipca_portfolios_free(&p)
	_ipca_factor(&m.gamma, m.n_factors, m.intercept, p.W, p.Q, f)
}


// Out of sample: total R² with each date's factor estimated from the fitted Gamma and
// that date's returns (ipca_factor_at), predictive R² with the fitted model's mean
// factor; dates of `times` are the new panel's own (0 ..< T')
ipca_r2_out_of_sample :: proc(
	m: ^IPCA,
	X: ^l.Matrix(f64),
	y: []f64,
	times: []int,
) -> (
	total, predictive: f64,
) {
	L, Kt, K, Tm := m.gamma.rows, m.gamma.cols, m.n_factors, m.factors.cols
	assert(X.cols == L, "ipca_r2_out_of_sample: characteristics differ in number")
	p := ipca_portfolios(X, y, times, context.temp_allocator)
	defer ipca_portfolios_free(&p)
	F := make([]f64, p.T * K, context.temp_allocator)
	defer delete(F, context.temp_allocator)
	for d in 0 ..< p.T {
		_ipca_factor(
			&m.gamma,
			K,
			m.intercept,
			p.W[d * L * L:][:L * L],
			p.Q[d * L:][:L],
			F[d * K:][:K],
		)
	}
	lam := make([]f64, K, context.temp_allocator)
	defer delete(lam, context.temp_allocator)
	for k in 0 ..< K {for d in 0 ..< Tm {lam[k] += m.factors.data[k * Tm + d] / f64(Tm)}}
	ss, e1, e2 := 0.0, 0.0, 0.0
	for r in 0 ..< X.rows {
		a, b := 0.0, 0.0
		if m.intercept {for c in 0 ..< L {a += X.data[r * L + c] * m.gamma.data[c * Kt + K]}}
		b = a
		for k in 0 ..< K {
			zg := 0.0
			for c in 0 ..< L {zg += X.data[r * L + c] * m.gamma.data[c * Kt + k]}
			a += zg * F[times[r] * K + k]
			b += zg * lam[k]
		}
		ss += y[r] * y[r]
		e1 += (y[r] - a) * (y[r] - a)
		e2 += (y[r] - b) * (y[r] - b)
	}
	return 1 - e1 / ss, 1 - e2 / ss
}
