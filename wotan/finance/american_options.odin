package finance

import l "../linalg"
import "base:runtime"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// AMERICAN OPTIONS
// ============================================================================
// Trees (CRR binomial, trinomial), finite differences (Crank-Nicolson European,
// American with Brennan-Schwartz projection) and Longstaff-Schwartz Monte Carlo
// (vanilla, Asian, antithetic + control variate).
//
// Greek conventions throughout: delta = ∂V/∂S, gamma = ∂²V/∂S², vega = ∂V/∂σ
// (per unit of volatility), theta = ∂V/∂t (per year, time passing: negative for
// most long options).

AmericanOptionResult :: struct {
	price:                  f64,
	delta:                  f64,
	gamma:                  f64,
	theta:                  f64,
	early_exercise_premium: f64,
}

@(private = "file")
_am_payoff :: #force_inline proc(S, K: f64, opt: OptionType) -> f64 {
	return max(S - K, 0) if opt == .Call else max(K - S, 0)
}

@(private = "file")
_am_expired :: proc(S, K: f64, opt: OptionType) -> AmericanOptionResult {
	return AmericanOptionResult{price = _am_payoff(S, K, opt)}
}

// ============================================================================
// CRR Binomial Tree for American Options
// ============================================================================
// u = e^{σ√Δt}, d = 1/u, p = (e^{(r−q)Δt} − d)/(u − d). The node prices of
// every level lie on the lattice S·u^k, k = −n..n, so the exercise values are
// computed once; backward induction works in place on one level (O(n) memory).
// Delta, gamma and theta come from the nodes of levels 1 and 2 (Hull).
american_option_binomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64,
	opt: OptionType,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	if T <= 0.0 {return _am_expired(S, K, opt)}

	n := max(n_steps, 2) // the Greeks need levels 1 and 2
	dt := T / f64(n)
	sdt := sigma * math.sqrt(dt)
	u := math.exp(sdt)
	d := 1.0 / u
	p := (math.exp((r - q) * dt) - d) / (u - d)
	disc := math.exp(-r * dt)

	// Exercise value at S·u^k is ex[k + n]; node (i, j) (j up-moves) is at k = 2j − i
	ex := make([]f64, 2 * n + 1, allocator)
	am := make([]f64, n + 1, allocator)
	eu := make([]f64, n + 1, allocator)
	defer {
		delete(ex, allocator)
		delete(am, allocator)
		delete(eu, allocator)
	}
	for k in -n ..= n {ex[k + n] = _am_payoff(S * math.exp(f64(k) * sdt), K, opt)}
	for j in 0 ..= n {
		am[j] = ex[2 * j]
		eu[j] = ex[2 * j]
	}

	f1, f2: [3]f64 // values at levels 1 and 2, j = 0.. (down to up)
	for i := n - 1; i >= 0; i -= 1 {
		for j in 0 ..= i {
			am[j] = max(disc * (p * am[j + 1] + (1.0 - p) * am[j]), ex[2 * j - i + n])
			eu[j] = disc * (p * eu[j + 1] + (1.0 - p) * eu[j])
		}
		if i == 2 {f2 = {am[0], am[1], am[2]}}
		if i == 1 {f1 = {am[0], am[1], 0}}
	}

	delta := (f1[1] - f1[0]) / (S * u - S * d)
	S_uu, S_dd := S * u * u, S * d * d
	gamma := ((f2[2] - f2[1]) / (S_uu - S) - (f2[1] - f2[0]) / (S - S_dd)) / (0.5 * (S_uu - S_dd))
	theta := (f2[1] - am[0]) / (2.0 * dt) // node (2, 1) has price S

	return AmericanOptionResult {
		price = am[0],
		delta = delta,
		gamma = gamma,
		theta = theta,
		early_exercise_premium = am[0] - eu[0],
	}
}

american_call_binomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_binomial(S, K, T, r, sigma, q, .Call, n_steps, allocator)
}

american_put_binomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_binomial(S, K, T, r, sigma, q, .Put, n_steps, allocator)
}

// ============================================================================
// TRINOMIAL TREE FOR AMERICAN OPTIONS
// ============================================================================
// Log-price steps of Δx = σ√(3Δt) with probabilities matching the mean νΔt and
// second moment σ²Δt + ν²Δt² of a step (ν = r − q − σ²/2):
//   p_u = ½(b + a), p_d = ½(b − a), p_m = 1 − b,
//   a = νΔt/Δx, b = (σ²Δt + ν²Δt²)/Δx²  (≈ 1/3, so p_m ≈ 2/3)
// Node j of a level has price S·e^{jΔx}; two rolling levels (O(n) memory).
// Gamma uses the non-uniform spacing of S·e^{−Δx}, S, S·e^{Δx}.
american_option_trinomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64,
	opt: OptionType,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	if T <= 0.0 {return _am_expired(S, K, opt)}

	n := max(n_steps, 2)
	dt := T / f64(n)
	dx := sigma * math.sqrt(3.0 * dt)
	nu := r - q - 0.5 * sigma * sigma
	a := nu * dt / dx
	b := (sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx)
	p_u, p_d, p_m := 0.5 * (b + a), 0.5 * (b - a), 1.0 - b
	disc := math.exp(-r * dt)

	// Index j + n for node j ∈ [−n, n]
	w := 2 * n + 1
	ex := make([]f64, w, allocator)
	am := make([]f64, w, allocator)
	am_next := make([]f64, w, allocator)
	eu := make([]f64, w, allocator)
	eu_next := make([]f64, w, allocator)
	defer {
		delete(ex, allocator)
		delete(am, allocator)
		delete(am_next, allocator)
		delete(eu, allocator)
		delete(eu_next, allocator)
	}
	for j in -n ..= n {ex[j + n] = _am_payoff(S * math.exp(f64(j) * dx), K, opt)}
	copy(am_next, ex)
	copy(eu_next, ex)

	f1: [3]f64 // level 1 at j = −1, 0, 1
	f2_mid: f64
	for i := n - 1; i >= 0; i -= 1 {
		for j in -i ..= i {
			c := j + n
			am[c] = max(
				disc * (p_u * am_next[c + 1] + p_m * am_next[c] + p_d * am_next[c - 1]),
				ex[c],
			)
			eu[c] = disc * (p_u * eu_next[c + 1] + p_m * eu_next[c] + p_d * eu_next[c - 1])
		}
		if i == 2 {f2_mid = am[n]}
		if i == 1 {f1 = {am[n - 1], am[n], am[n + 1]}}
		am, am_next = am_next, am
		eu, eu_next = eu_next, eu
	}
	price, euro := am_next[n], eu_next[n]

	S_up, S_dn := S * math.exp(dx), S * math.exp(-dx)
	delta := (f1[2] - f1[0]) / (S_up - S_dn)
	gamma := 2.0 * ((f1[2] - f1[1]) / (S_up - S) - (f1[1] - f1[0]) / (S - S_dn)) / (S_up - S_dn)
	theta := (f2_mid - price) / (2.0 * dt)

	return AmericanOptionResult {
		price = price,
		delta = delta,
		gamma = gamma,
		theta = theta,
		early_exercise_premium = price - euro,
	}
}

american_call_trinomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_trinomial(S, K, T, r, sigma, q, .Call, n_steps, allocator)
}

american_put_trinomial :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	q: f64 = 0.0,
	n_steps: int = 1000,
	allocator: mem.Allocator = context.allocator,
) -> AmericanOptionResult {
	return american_option_trinomial(S, K, T, r, sigma, q, .Put, n_steps, allocator)
}

// ============================================================================
// FINITE DIFFERENCES
// ============================================================================

FiniteDifferenceResult :: struct {
	price: f64,
	delta: f64,
	gamma: f64,
	theta: f64,
}

// ============================================================================
// Thomas Algorithm for Tridiagonal Systems
// ============================================================================
// Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
// Returns solution in d (overwrites input)
_thomas_algorithm :: proc(
	a: []f64, // Lower diagonal (length n-1)
	b: []f64, // Main diagonal (length n)
	c: []f64, // Upper diagonal (length n-1)
	d: []f64, // Right-hand side (length n, solution overwrites this)
	n: int,
) {
	// Forward elimination
	c_prime := make([]f64, n - 1, context.temp_allocator)
	defer delete(c_prime, context.temp_allocator)

	c_prime[0] = c[0] / b[0]
	d[0] = d[0] / b[0]

	for i in 1 ..< n {
		m := b[i] - a[i - 1] * c_prime[i - 1]
		if i < n - 1 {
			c_prime[i] = c[i] / m
		}
		d[i] = (d[i] - a[i - 1] * d[i - 1]) / m
	}

	// Back substitution
	for i := n - 2; i >= 0; i -= 1 {
		d[i] = d[i] - c_prime[i] * d[i + 1]
	}
}

// Black-Scholes PDE in x = ln S and time to expiry τ:
//   V_τ = ½σ²·V_xx + (r − σ²/2)·V_x − r·V
// on a uniform x grid of N (even) intervals with S at the centre node, so the
// price needs no interpolation. The half-width is 6σ√T plus the log-distance to
// the strike. Crank-Nicolson time steps, the first one replaced by two fully
// implicit half steps (Rannacher), which damps the oscillations CN produces
// from the payoff kink. American options: Brennan-Schwartz, i.e. the early
// exercise constraint is applied inside the tridiagonal solve, substituting
// from the exercise side (low S for puts, high S for calls). Greeks from the
// grid: Δ = V_x/S, Γ = (V_xx − V_x)/S² at the centre node, θ = ∂V/∂t from the
// last time step.
@(private = "file")
_fd_black_scholes :: proc(
	S, K, T, r, sigma: f64,
	opt: OptionType,
	american: bool,
	n_space, n_time: int,
	allocator: mem.Allocator,
) -> FiniteDifferenceResult {
	if T <= 0.0 {return {price = _am_payoff(S, K, opt)}}

	N := max(n_space - 1, 4)
	if N % 2 == 1 {N += 1}
	half := N / 2
	L := 6.0 * sigma * math.sqrt(T) + abs(math.ln(K / S))
	dx := L / f64(half)
	x0 := math.ln(S) - L
	n_t := max(n_time, 1)
	dt := T / f64(n_t)

	V := make([]f64, N + 1, allocator)
	payoff := make([]f64, N + 1, allocator)
	rhs := make([]f64, N + 1, allocator)
	work := make([]f64, N + 1, allocator)
	defer {
		delete(V, allocator)
		delete(payoff, allocator)
		delete(rhs, allocator)
		delete(work, allocator)
	}
	S_min, S_max := math.exp(x0), math.exp(x0 + f64(N) * dx)
	for i in 0 ..= N {
		payoff[i] = _am_payoff(math.exp(x0 + f64(i) * dx), K, opt)
		V[i] = payoff[i]
	}

	// L V = lo·V[i−1] + di·V[i] + up·V[i+1]
	alpha := 0.5 * sigma * sigma
	beta := r - 0.5 * sigma * sigma
	lo := alpha / (dx * dx) - beta / (2.0 * dx)
	di := -2.0 * alpha / (dx * dx) - r
	up := alpha / (dx * dx) + beta / (2.0 * dx)

	// One θ-scheme step of size h ending at time to expiry tau:
	// (I − θhL)V_new = (I + (1−θ)hL)V_old
	step :: proc(
		V, payoff, rhs, work: []f64,
		lo, di, up, h, theta, tau, K, r, S_min, S_max: f64,
		opt: OptionType,
		american: bool,
	) {
		N := len(V) - 1
		e := (1.0 - theta) * h
		for i in 1 ..< N {
			rhs[i] = V[i] + e * (lo * V[i - 1] + di * V[i] + up * V[i + 1])
		}
		// Boundary values at the new time level
		if opt == .Call {
			V[0] = 0
			V[N] = S_max - K * math.exp(-r * tau)
		} else {
			V[0] = (K if american else K * math.exp(-r * tau)) - S_min
			V[N] = 0
		}
		a, b, c := -theta * h * lo, 1.0 - theta * h * di, -theta * h * up
		rhs[1] -= a * V[0]
		rhs[N - 1] -= c * V[N]

		if opt == .Put {
			// Eliminate from the top (high S) down, substitute upwards from low S,
			// where the exercise region of a put lies
			work[N - 1] = b
			for i := N - 2; i >= 1; i -= 1 {
				work[i] = b - c * a / work[i + 1]
				rhs[i] -= c * rhs[i + 1] / work[i + 1]
			}
			prev := V[0]
			for i in 1 ..< N {
				v := (rhs[i] - a * prev) / work[i]
				if american {v = max(v, payoff[i])}
				V[i] = v
				prev = v
			}
		} else {
			// Eliminate upwards from low S, substitute downwards from high S
			work[1] = b
			for i in 2 ..< N {
				work[i] = b - a * c / work[i - 1]
				rhs[i] -= a * rhs[i - 1] / work[i - 1]
			}
			next := V[N]
			for i := N - 1; i >= 1; i -= 1 {
				v := (rhs[i] - c * next) / work[i]
				if american {v = max(v, payoff[i])}
				V[i] = v
				next = v
			}
		}
	}

	// Rannacher start: two fully implicit half steps, then Crank-Nicolson
	step(
		V,
		payoff,
		rhs,
		work,
		lo,
		di,
		up,
		0.5 * dt,
		1.0,
		0.5 * dt,
		K,
		r,
		S_min,
		S_max,
		opt,
		american,
	)
	before_last, last_h := V[half], 0.5 * dt
	step(V, payoff, rhs, work, lo, di, up, 0.5 * dt, 1.0, dt, K, r, S_min, S_max, opt, american)
	for n in 1 ..< n_t {
		before_last, last_h = V[half], dt
		step(
			V,
			payoff,
			rhs,
			work,
			lo,
			di,
			up,
			dt,
			0.5,
			f64(n + 1) * dt,
			K,
			r,
			S_min,
			S_max,
			opt,
			american,
		)
	}

	v_x := (V[half + 1] - V[half - 1]) / (2.0 * dx)
	v_xx := (V[half + 1] - 2.0 * V[half] + V[half - 1]) / (dx * dx)
	return FiniteDifferenceResult {
		price = V[half],
		delta = v_x / S,
		gamma = (v_xx - v_x) / (S * S),
		theta = (before_last - V[half]) / last_h,
	}
}

// European option by Crank-Nicolson (with Rannacher start-up)
crank_nicolson_european :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return _fd_black_scholes(S, K, T, r, sigma, opt, false, n_space, n_time, allocator)
}

// ============================================================================
// Convenience Wrappers
// ============================================================================
crank_nicolson_call :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return crank_nicolson_european(S, K, T, r, sigma, .Call, n_space, n_time, allocator)
}

crank_nicolson_put :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return crank_nicolson_european(S, K, T, r, sigma, .Put, n_space, n_time, allocator)
}

// American option by Crank-Nicolson with Rannacher start-up and the
// Brennan-Schwartz early-exercise projection
finite_difference_american :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return _fd_black_scholes(S, K, T, r, sigma, opt, true, n_space, n_time, allocator)
}

// Convenience Wrappers
fd_american_call :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return finite_difference_american(S, K, T, r, sigma, .Call, n_space, n_time, allocator)
}

fd_american_put :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> FiniteDifferenceResult {
	return finite_difference_american(S, K, T, r, sigma, .Put, n_space, n_time, allocator)
}


// ============================================================================
// LONGSTAFF-SCHWARTZ MONTE CARLO FOR AMERICAN OPTIONS
// ============================================================================
// Regression estimates of the continuation value decide exercise at
// n_exercise_dates equally spaced dates t_k = T·k/n (k = 1..n, the last one is
// expiry; date k falls on simulation step round(k·n_steps/n)), plus exercise at
// t = 0 if the intrinsic value exceeds the estimate. With few dates this is a
// Bermudan option.
//
// The regression uses the in-the-money paths, a polynomial of the given degree
// in s = S/K − 1 (Asian: all monomials s^i·a^j, i + j ≤ degree, with
// a = A/K − 1, as the continuation value depends on the spot and the running
// average), solved by QR (linalg.lstsq).
//
// Greeks are central differences on the same normals with the exercise rule of
// the base run frozen. By the envelope theorem the first derivatives of the
// optimally exercised value equal those under its fixed optimal rule, and the
// frozen rule avoids the noise of re-regressing on every bumped run.

LSM_Result :: struct {
	price: f64,
	delta: f64,
	gamma: f64,
	vega:  f64,
	theta: f64,
}

// ============================================================================
// Linear Regression via Least Squares (QR)
// ============================================================================
// Least-squares coefficients for the row-major design matrix X
// (n_samples × n_features) and targets y; allocated with `allocator`.
_least_squares_regression :: proc(
	X: []f64, // Flattened design matrix (n_samples * n_features)
	y: []f64, // Target vector (n_samples)
	n_samples: int,
	n_features: int,
	allocator: mem.Allocator,
) -> []f64 {
	Xm := l.Matrix(f64) {
		rows      = n_samples,
		cols      = n_features,
		data      = X[:n_samples * n_features],
		allocator = allocator,
		stride    = n_features,
	}
	sol := l.lstsq(&Xm, y[:n_samples], .QR, allocator)
	delete(sol.residuals, allocator)
	return sol.beta
}

@(private = "file")
_LSM :: struct {
	S_0, K, T, r, sigma:   f64,
	opt:                   OptionType,
	asian:                 bool, // exercise value on the running average (S_0 included)
	n_steps, n_ex, degree: int,
	antithetic:            bool, // paths n_base..2n_base−1 mirror the normals of 0..n_base−1
	control:               bool, // European control variate (vanilla only)
	n_base:                int, // independent normal paths
}

@(private = "file")
_LSM_Out :: struct {
	price:     f64, // with t = 0 exercise and (if enabled) the control variate
	var_ratio: f64, // variance of plain MC over that of this estimator, same path count
}

@(private = "file")
_lsm_n_basis :: proc(c: ^_LSM) -> int {
	d := c.degree
	return (d + 1) * (d + 2) / 2 if c.asian else d + 1
}

@(private = "file")
_lsm_basis :: proc(c: ^_LSM, S, A: f64, out: []f64) {
	s := S / c.K - 1.0
	if !c.asian {
		v := 1.0
		for k in 0 ..= c.degree {
			out[k] = v
			v *= s
		}
		return
	}
	a := A / c.K - 1.0
	idx := 0
	for total in 0 ..= c.degree {
		for i in 0 ..= total {
			v := 1.0
			for _ in 0 ..< i {v *= s}
			for _ in 0 ..< total - i {v *= a}
			out[idx] = v
			idx += 1
		}
	}
}

@(private = "file")
_lsm_exercise_value :: #force_inline proc(c: ^_LSM, S, A: f64) -> f64 {
	return _am_payoff(A if c.asian else S, c.K, c.opt)
}

// One LSM valuation on the normals z (n_base × n_steps). policy holds n_ex − 1
// rows of regression coefficients (dates 1..n_ex−1; a NaN first coefficient means
// no exercise at that date): regress = true fits and stores them, false uses
// them as given.
@(private = "file")
_lsm_run :: proc(
	c: ^_LSM,
	z: []f64,
	policy: []f64,
	regress: bool,
	allocator: mem.Allocator,
) -> _LSM_Out {
	n_paths := c.n_base * (2 if c.antithetic else 1)
	n_ex, nb := c.n_ex, _lsm_n_basis(c)
	dt := c.T / f64(c.n_steps)
	drift := (c.r - 0.5 * c.sigma * c.sigma) * dt
	vol := c.sigma * math.sqrt(dt)

	ex_step := make([]int, n_ex, allocator)
	S_ex := make([]f64, n_paths * n_ex, allocator)
	A_ex := make([]f64, n_paths * n_ex if c.asian else 0, allocator)
	cash := make([]f64, n_paths, allocator)
	defer {
		delete(ex_step, allocator)
		delete(S_ex, allocator)
		delete(A_ex, allocator)
		delete(cash, allocator)
	}
	for k in 0 ..< n_ex {ex_step[k] = int(math.round(f64(k + 1) * f64(c.n_steps) / f64(n_ex)))}

	for p in 0 ..< n_paths {
		zp := z[(p % c.n_base) * c.n_steps:][:c.n_steps]
		sign := -1.0 if p >= c.n_base else 1.0
		S, sum := c.S_0, c.S_0
		k := 0
		for step in 1 ..= c.n_steps {
			S *= math.exp(drift + vol * sign * zp[step - 1])
			sum += S
			if step == ex_step[k] {
				S_ex[p * n_ex + k] = S
				if c.asian {A_ex[p * n_ex + k] = sum / f64(step + 1)}
				k += 1
			}
		}
	}

	A_at :: #force_inline proc(c: ^_LSM, A_ex: []f64, i: int) -> f64 {return(
			A_ex[i] if c.asian else 0 \
		)}

	disc_T := math.exp(-c.r * c.T)
	for p in 0 ..< n_paths {
		i := p * n_ex + n_ex - 1
		cash[p] = _lsm_exercise_value(c, S_ex[i], A_at(c, A_ex, i)) * disc_T
	}

	basis := make([]f64, nb, allocator)
	defer delete(basis, allocator)
	for k := n_ex - 2; k >= 0; k -= 1 {
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		t_k := f64(ex_step[k]) * dt
		coef := policy[k * nb:][:nb]
		if regress {
			m := 0
			for p in 0 ..< n_paths {
				i := p * n_ex + k
				if _lsm_exercise_value(c, S_ex[i], A_at(c, A_ex, i)) > 0 {m += 1}
			}
			if m <= nb {
				coef[0] = math.nan_f64()
				continue
			}
			X := l.matrix_new(f64, m, nb, context.temp_allocator)
			y := make([]f64, m, context.temp_allocator)
			row := 0
			grow := math.exp(c.r * t_k)
			for p in 0 ..< n_paths {
				i := p * n_ex + k
				if _lsm_exercise_value(c, S_ex[i], A_at(c, A_ex, i)) <= 0 {continue}
				_lsm_basis(c, S_ex[i], A_at(c, A_ex, i), X.data[row * nb:][:nb])
				y[row] = cash[p] * grow
				row += 1
			}
			sol := l.lstsq(&X, y, .QR, context.temp_allocator)
			copy(coef, sol.beta)
		}
		if math.is_nan(coef[0]) {continue}

		disc_k := math.exp(-c.r * t_k)
		for p in 0 ..< n_paths {
			i := p * n_ex + k
			ev := _lsm_exercise_value(c, S_ex[i], A_at(c, A_ex, i))
			if ev <= 0 {continue}
			_lsm_basis(c, S_ex[i], A_at(c, A_ex, i), basis)
			cont := 0.0
			for b, j in basis {cont += b * coef[j]}
			if ev > cont {cash[p] = ev * disc_k}
		}
	}

	// Estimator per independent sample (antithetic pairs averaged), optionally
	// with the European payoff as control variate
	n_s := c.n_base
	mean_y, mean_c := 0.0, 0.0
	euro_exact := compute_greeks(c.S_0, c.K, c.T, c.r, c.sigma, c.opt).price
	sample :: proc(c: ^_LSM, cash, S_ex: []f64, s: int, disc_T: f64) -> (y, ctrl: f64) {
		n_ex := c.n_ex
		y = cash[s]
		ctrl = _am_payoff(S_ex[s * n_ex + n_ex - 1], c.K, c.opt) * disc_T
		if c.antithetic {
			s2 := s + c.n_base
			y = 0.5 * (y + cash[s2])
			ctrl = 0.5 * (ctrl + _am_payoff(S_ex[s2 * n_ex + n_ex - 1], c.K, c.opt) * disc_T)
		}
		return
	}
	for s in 0 ..< n_s {
		y, ctrl := sample(c, cash, S_ex, s, disc_T)
		mean_y += y
		mean_c += ctrl
	}
	mean_y /= f64(n_s)
	mean_c /= f64(n_s)
	cov, var_c, var_y := 0.0, 0.0, 0.0
	for s in 0 ..< n_s {
		y, ctrl := sample(c, cash, S_ex, s, disc_T)
		cov += (y - mean_y) * (ctrl - mean_c)
		var_c += (ctrl - mean_c) * (ctrl - mean_c)
		var_y += (y - mean_y) * (y - mean_y)
	}
	beta := cov / var_c if c.control && var_c > 0 else 0
	estimate := mean_y - beta * (mean_c - euro_exact)

	// Variance of the estimator vs plain MC with the same number of paths
	mean_p, var_p := 0.0, 0.0
	for v in cash {mean_p += v}
	mean_p /= f64(n_paths)
	for v in cash {var_p += (v - mean_p) * (v - mean_p)}
	var_est := (var_y - beta * beta * var_c) / f64(n_s) // Var(y − βc) with the optimal β
	var_plain := var_p / f64(n_paths) / f64(n_paths)
	var_ratio := var_plain / (var_est / f64(n_s)) if var_est > 0 else 1

	intrinsic := _am_payoff(c.S_0, c.K, c.opt) // also the Asian value at t = 0 (A_0 = S_0)
	return {price = max(estimate, intrinsic), var_ratio = var_ratio}
}

// Price and Greeks: base run with regression, bumped runs on the same normals
// under the base run's exercise rule
@(private = "file")
_lsm_price_greeks :: proc(
	c: ^_LSM,
	allocator: mem.Allocator,
) -> (
	res: LSM_Result,
	var_ratio: f64,
) {
	if c.T <= 0 {return {price = _am_payoff(c.S_0, c.K, c.opt)}, 1}
	c.n_steps = max(c.n_steps, 1)
	c.n_ex = clamp(c.n_ex, 1, c.n_steps)
	c.n_base = max(c.n_base, 1)

	z := make([]f64, c.n_base * c.n_steps, allocator)
	policy := make([]f64, max(c.n_ex - 1, 1) * _lsm_n_basis(c), allocator)
	defer {
		delete(z, allocator)
		delete(policy, allocator)
	}
	for &v in z {v = rand.float64_normal(0.0, 1.0)}

	base := _lsm_run(c, z, policy, true, allocator)
	bumped :: proc(
		c: ^_LSM,
		z, policy: []f64,
		allocator: mem.Allocator,
		dS, dsig, dT: f64,
	) -> f64 {
		b := c^
		b.S_0 += dS
		b.sigma += dsig
		b.T += dT
		return _lsm_run(&b, z, policy, false, allocator).price
	}

	// Delta from ±1% of spot; gamma from ±5%: second differences of Monte Carlo
	// prices are dominated by paths crossing the strike or exercise boundary
	// between the bumps, so a wider bump trades a small O(h²) bias for much
	// less noise
	h_S, h_G := 0.01 * c.S_0, 0.05 * c.S_0
	up := bumped(c, z, policy, allocator, h_S, 0, 0)
	dn := bumped(c, z, policy, allocator, -h_S, 0, 0)
	up_G := bumped(c, z, policy, allocator, h_G, 0, 0)
	dn_G := bumped(c, z, policy, allocator, -h_G, 0, 0)
	h_sig := max(0.01 * c.sigma, 0.001)
	h_T := min(0.01, 0.25 * c.T)
	res = LSM_Result {
		price = base.price,
		delta = (up - dn) / (2.0 * h_S),
		gamma = (up_G - 2.0 * base.price + dn_G) / (h_G * h_G),
		vega  = (bumped(c, z, policy, allocator, 0, h_sig, 0) -
			bumped(c, z, policy, allocator, 0, -h_sig, 0)) / (2.0 * h_sig),
		theta = -(bumped(c, z, policy, allocator, 0, 0, h_T) -
			bumped(c, z, policy, allocator, 0, 0, -h_T)) / (2.0 * h_T),
	}
	return res, base.var_ratio
}

// ============================================================================
// LSM: American Vanilla Option
// ============================================================================
lsm_american_vanilla :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	n_paths: int = 10000,
	n_steps: int = 50,
	n_exercise_dates: int = 10,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> LSM_Result {
	c := _LSM {
		S_0     = S_0,
		K       = K,
		T       = T,
		r       = r,
		sigma   = sigma,
		opt     = opt,
		n_steps = n_steps,
		n_ex    = n_exercise_dates,
		degree  = poly_degree,
		n_base  = n_paths,
	}
	res, _ := _lsm_price_greeks(&c, allocator)
	return res
}

// Convenience wrappers
lsm_american_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int = 10000,
	n_steps: int = 50,
	n_exercise_dates: int = 10,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> LSM_Result {
	return lsm_american_vanilla(
		S_0,
		K,
		T,
		r,
		sigma,
		.Call,
		n_paths,
		n_steps,
		n_exercise_dates,
		poly_degree,
		allocator,
	)
}

lsm_american_put :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int = 10000,
	n_steps: int = 50,
	n_exercise_dates: int = 10,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> LSM_Result {
	return lsm_american_vanilla(
		S_0,
		K,
		T,
		r,
		sigma,
		.Put,
		n_paths,
		n_steps,
		n_exercise_dates,
		poly_degree,
		allocator,
	)
}

// ============================================================================
// LONGSTAFF-SCHWARTZ: AMERICAN ASIAN CALL OPTION
// ============================================================================
// Exercise pays the running arithmetic average (S_0 included) minus K.
lsm_american_asian_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	n_paths: int = 10000,
	n_steps: int = 50,
	n_exercise_dates: int = 10,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> LSM_Result {
	c := _LSM {
		S_0     = S_0,
		K       = K,
		T       = T,
		r       = r,
		sigma   = sigma,
		opt     = .Call,
		asian   = true,
		n_steps = n_steps,
		n_ex    = n_exercise_dates,
		degree  = poly_degree,
		n_base  = n_paths,
	}
	res, _ := _lsm_price_greeks(&c, allocator)
	return res
}

// ============================================================================
// VARIANCE-REDUCED LONGSTAFF-SCHWARTZ MONTE CARLO
// ============================================================================
// Antithetic variates (⌈n_paths/2⌉ pairs) plus the European option on the same
// paths as control variate, priced exactly by Black-Scholes.

VR_LSM_Result :: struct {
	price:              f64,
	delta:              f64,
	gamma:              f64,
	vega:               f64,
	theta:              f64,
	// Variance of plain Monte Carlo with the same number of paths over the
	// variance of this estimator (antithetic pairs and control variate together)
	variance_reduction: f64,
}

lsm_american_vanilla_reduced :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	sigma: f64,
	opt: OptionType,
	n_paths: int = 10000,
	n_steps: int = 50,
	n_exercise_dates: int = 10,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> VR_LSM_Result {
	c := _LSM {
		S_0        = S_0,
		K          = K,
		T          = T,
		r          = r,
		sigma      = sigma,
		opt        = opt,
		n_steps    = n_steps,
		n_ex       = n_exercise_dates,
		degree     = poly_degree,
		antithetic = true,
		control    = true,
		n_base     = (n_paths + 1) / 2,
	}
	res, ratio := _lsm_price_greeks(&c, allocator)
	return VR_LSM_Result {
		price = res.price,
		delta = res.delta,
		gamma = res.gamma,
		vega = res.vega,
		theta = res.theta,
		variance_reduction = ratio,
	}
}
