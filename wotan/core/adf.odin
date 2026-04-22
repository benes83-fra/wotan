package core

import "core:math"
import "core:mem"

// ------------------------------------------------------------
// Augmented Dickey–Fuller (ADF) Test
// ------------------------------------------------------------
adf_test :: proc(
	y: []f64,
	max_lags: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	adf_stat: f64,
	p_value: f64,
	lags_used: int,
	n_obs: int,
	crit_1pct: f64,
	crit_5pct: f64,
	crit_10pct: f64,
) {

	n := len(y)
	if n < 10 {
		return 0, 1, 0, n, 0, 0, 0
	}

	// Δy
	dy := make([]f64, n - 1, allocator)
	for i in 1 ..< n {
		dy[i - 1] = y[i] - y[i - 1]
	}

	// y_{t-1}
	y_lag := make([]f64, n - 1, allocator)
	for i in 1 ..< n {
		y_lag[i - 1] = y[i - 1]
	}

	p := max_lags
	lags_used = p

	T := (n - 1) - p
	n_obs = T
	if T <= 5 {
		return 0, 1, p, T, 0, 0, 0
	}

	k := 2 + p
	X := make([]f64, T * k, allocator)
	Y := make([]f64, T, allocator)

	for t in 0 ..< T {
		row := t * k

		Y[t] = dy[t + p]
		X[row + 0] = 1.0
		X[row + 1] = y_lag[t + p - 1]

		for j in 0 ..< p {
			X[row + 2 + j] = dy[t + p - 1 - j]
		}
	}

	// X'X and X'Y
	XtX := make([]f64, k * k, allocator)
	XtY := make([]f64, k, allocator)

	for t in 0 ..< T {
		row := t * k
		for i in 0 ..< k {
			xi := X[row + i]
			XtY[i] += xi * Y[t]
			for j in 0 ..< k {
				XtX[i * k + j] += xi * X[row + j]
			}
		}
	}

	// invert X'X
	XtX_inv := matrix_inverse(XtX, k, allocator)

	// β = (X'X)^(-1) X'Y
	beta := make([]f64, k, allocator)
	for i in 0 ..< k {
		s := 0.0
		for j in 0 ..< k {
			s += XtX_inv[i * k + j] * XtY[j]
		}
		beta[i] = s
	}

	// residual variance
	rss := 0.0
	for t in 0 ..< T {
		row := t * k
		pred := 0.0
		for j in 0 ..< k {
			pred += X[row + j] * beta[j]
		}
		e := Y[t] - pred
		rss += e * e
	}

	sigma2 := rss / f64(T - k)

	// standard errors
	se := make([]f64, k, allocator)
	for i in 0 ..< k {
		se[i] = math.sqrt_f64(sigma2 * XtX_inv[i * k + i])
	}

	gamma_hat := beta[1]
	gamma_se := se[1]
	adf_stat = gamma_hat / gamma_se

	// critical values (constant only)
	crit_1pct = -3.43
	crit_5pct = -2.86
	crit_10pct = -2.57

	if adf_stat < crit_1pct {
		p_value = 0.01
	} else if adf_stat < crit_5pct {
		p_value = 0.05
	} else if adf_stat < crit_10pct {
		p_value = 0.10
	} else {
		p_value = 0.50
	}

	return
}

// ------------------------------------------------------------
// DataFrame wrapper
// ------------------------------------------------------------
df_adf :: proc(
	y: []f64,
	max_lags: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	stat, p, lags, n_obs, c1, c5, c10 := adf_test(y, max_lags, allocator)

	out := dataframe_new()
	add_column(&out, column_from_floats("adf_stat", []f64{stat}))
	add_column(&out, column_from_floats("p_value", []f64{p}))
	add_column(&out, column_from_ints("lags_used", []int{lags}))
	add_column(&out, column_from_ints("n_obs", []int{n_obs}))
	add_column(&out, column_from_floats("crit_1pct", []f64{c1}))
	add_column(&out, column_from_floats("crit_5pct", []f64{c5}))
	add_column(&out, column_from_floats("crit_10pct", []f64{c10}))
	out.rows = 1
	return out
}

// ------------------------------------------------------------
// Dynamic Gauss–Jordan inverse
// ------------------------------------------------------------
matrix_inverse :: proc(A: []f64, k: int, allocator: mem.Allocator = context.allocator) -> []f64 {

	aug := make([]f64, k * k * 2, allocator)

	for i in 0 ..< k {
		for j in 0 ..< k {
			aug[i * (2 * k) + j] = A[i * k + j]
		}
		aug[i * (2 * k) + (k + i)] = 1.0
	}

	for col in 0 ..< k {
		pivot := aug[col * (2 * k) + col]
		if abs(pivot) < 1e-12 {
			for r in col + 1 ..< k {
				if abs(aug[r * (2 * k) + col]) > 1e-12 {
					for c in 0 ..< 2 * k {
						aug[col * (2 * k) + c], aug[r * (2 * k) + c] =
							aug[r * (2 * k) + c], aug[col * (2 * k) + c]
					}
					pivot = aug[col * (2 * k) + col]
					break
				}
			}
		}

		inv_pivot := 1.0 / pivot
		for c in 0 ..< 2 * k {
			aug[col * (2 * k) + c] *= inv_pivot
		}

		for r in 0 ..< k {
			if r == col {continue}
			factor := aug[r * (2 * k) + col]
			if factor != 0 {
				for c in 0 ..< 2 * k {
					aug[r * (2 * k) + c] -= factor * aug[col * (2 * k) + c]
				}
			}
		}
	}

	inv := make([]f64, k * k, allocator)
	for i in 0 ..< k {
		for j in 0 ..< k {
			inv[i * k + j] = aug[i * (2 * k) + (k + j)]
		}
	}

	return inv
}
