package ml_finance

import ml "../analytics/ML"
import w "../core"
import l "../linalg" // Import your existing ML module
import "core:math"
import "core:mem"

// ============================================================================
// Pairs Trading Result Structure
// ============================================================================
PairsTradingResult :: struct {
	returns: []f64,
	trades:  []int,
	betas:   []f64,
	spreads: []f64,
}

// Helper to extract float column from DataFrame
extract_float_col :: proc(df: ^w.DataFrame, col_name: string, allocator: mem.Allocator) -> []f64 {
	n := df.rows
	out := make([]f64, n, allocator)
	col := w.column(df, col_name)
	for i in 0 ..< n {
		v, is_null := w.column_at_float(col, i)
		if !is_null {
			out[i] = v
		} else {
			out[i] = 0.0
		}
	}
	return out
}
// ============================================================================
// Kalman Filter Pairs Trading Strategy (Log-Price Version)
// ============================================================================
// ============================================================================
// Kalman Filter Pairs Trading Strategy (Log-Price Version)
// ============================================================================
kalman_pairs_strategy :: proc(
	df: ^w.DataFrame,
	col_x: string,
	col_y: string,
	window: int,
	initial_hedge_ratio: f64,
	allocator: mem.Allocator = context.allocator,
) -> PairsTradingResult {
	x_data := extract_float_col(df, col_x, allocator)
	y_data := extract_float_col(df, col_y, allocator)
	defer delete(x_data, allocator)
	defer delete(y_data, allocator)

	n := len(x_data)

	result_returns := make([dynamic]f64, 0, allocator)
	result_trades := make([dynamic]int, 0, allocator)
	result_betas := make([dynamic]f64, 0, allocator)
	result_spreads := make([dynamic]f64, 0, allocator)

	if n < 2 {
		return PairsTradingResult {
			returns = result_returns[:],
			trades = result_trades[:],
			betas = result_betas[:],
			spreads = result_spreads[:],
		}
	}

	// Kalman Filter State: [intercept, slope]
	state := make([]f64, 2, allocator)
	state[0] = 0.0
	state[1] = initial_hedge_ratio

	// Covariance Matrix [P00, P01, P10, P11]
	cov := make([]f64, 4, allocator)
	cov[0] = 1.0; cov[1] = 0.0
	cov[2] = 0.0; cov[3] = 1.0

	// =========================================================
	// USE OLS TO CALIBRATE THE KALMAN FILTER SCALES
	// =========================================================
	init_len := window
	if init_len >= n {
		init_len = n - 1
	}

	spread_variance := 1e-4 // Fallback default

	if init_len >= 2 {
		X_mat := l.matrix_new(f64, init_len, 2, allocator)
		y_vec := make([]f64, init_len, allocator)

		for i in 0 ..< init_len {
			X_mat.data[i * 2 + 0] = 1.0
			X_mat.data[i * 2 + 1] = math.ln_f64(x_data[i]) // Log price
			y_vec[i] = math.ln_f64(y_data[i]) // Log price
		}

		ols_res := ml.ols_fit(&X_mat, y_vec, .Cholesky, allocator)

		state[0] = ols_res.beta[0]
		state[1] = ols_res.beta[1]

		// CRITICAL FIX: Use the actual residual variance from OLS to scale the filter
		spread_variance = math.max(ols_res.sigma2, 1e-6)

		// Initialize P to the OLS covariance matrix (scaled)
		if ols_res.vcov.data != nil && ols_res.vcov.rows == 2 && ols_res.vcov.cols == 2 {
			cov[0] = ols_res.vcov.data[0]
			cov[1] = ols_res.vcov.data[1]
			cov[2] = ols_res.vcov.data[2]
			cov[3] = ols_res.vcov.data[3]
		} else {
			// Fallback to diagonal if vcov is unavailable
			cov[0] = spread_variance * 10.0
			cov[1] = 0.0
			cov[2] = 0.0
			cov[3] = spread_variance * 10.0
		}

		l.matrix_free(&X_mat)
		delete(y_vec, allocator)
	}

	// Process noise: small fraction of the spread variance
	Q := spread_variance * 1e-3
	// Measurement noise: the variance of the spread itself
	R := spread_variance

	position: i32 = 0
	position_beta := 0.0 // Hedge ratio locked at the time of entry

	prev_ln_px := math.ln_f64(x_data[0])
	prev_ln_py := math.ln_f64(y_data[0])

	// Industry-standard thresholds
	entry_threshold := 2.0
	exit_threshold := 0.5 // Half-mean reversion to avoid whipsaws
	stop_loss_threshold := 3.5 // Cut losses if the spread diverges further

	for i in 0 ..< n {
		px := x_data[i]
		py := y_data[i]

		ln_px := math.ln_f64(px)
		ln_py := math.ln_f64(py)

		// --- Kalman Filter Update ---
		pred_cov_00 := cov[0] + Q
		pred_cov_11 := cov[3] + Q

		spread_pred := state[0] + state[1] * ln_px
		innovation := ln_py - spread_pred

		s := pred_cov_00 + 2.0 * ln_px * cov[1] + ln_px * ln_px * pred_cov_11 + R

		k1 := (cov[0] + ln_px * cov[1]) / s
		k2 := (cov[1] + ln_px * cov[3]) / s

		state[0] = state[0] + k1 * innovation
		state[1] = state[1] + k2 * innovation

		new_cov_00 := (1.0 - k1) * pred_cov_00 - k1 * ln_px * cov[1]
		new_cov_01 := (1.0 - k1) * cov[1] - k1 * ln_px * pred_cov_11
		new_cov_10 := -k2 * pred_cov_00 + (1.0 - k2 * ln_px) * cov[1]
		new_cov_11 := -k2 * cov[1] + (1.0 - k2 * ln_px) * pred_cov_11

		cov[0] = new_cov_00
		cov[1] = new_cov_01
		cov[2] = new_cov_10
		cov[3] = new_cov_11

		current_spread := ln_py - (state[0] + state[1] * ln_px)

		z_score := 0.0
		if s > 1e-12 {
			z_score = innovation / math.sqrt(s)
		}

		// --- Trading Logic ---
		signal: i32 = position

		if position == 0 {
			if z_score > entry_threshold {
				signal = -1 // Short Spread (Bet on spread decreasing)
			} else if z_score < -entry_threshold {
				signal = 1 // Long Spread (Bet on spread increasing)
			}
		} else {
			// ✅ CRITICAL FIX: Corrected exit logic for mean reversion
			if position == 1 {
				// Entered at z < -2.0. Exit when z reverts up to -0.5,
				// OR stop loss if it crashes further to -3.5
				if z_score > -exit_threshold || z_score < -stop_loss_threshold {
					signal = 0
				}
			} else if position == -1 {
				// Entered at z > 2.0. Exit when z reverts down to 0.5,
				// OR stop loss if it explodes further to 3.5
				if z_score < exit_threshold || z_score > stop_loss_threshold {
					signal = 0
				}
			}
		}

		// 1. Calculate daily return based on the position HELD during this day
		daily_ret := 0.0
		if i > 0 {
			ret_x := ln_px - prev_ln_px
			ret_y := ln_py - prev_ln_py

			if position == 1 {
				// Long Spread: Long Y, Short X
				daily_ret = ret_y - position_beta * ret_x
			} else if position == -1 {
				// Short Spread: Short Y, Long X
				daily_ret = -ret_y + position_beta * ret_x
			}
		}

		append(&result_returns, daily_ret)
		append(&result_betas, state[1])
		append(&result_spreads, current_spread)

		// 2. Update position FOR THE NEXT DAY
		if signal != position {
			append(&result_trades, i)
			position = signal
			if position != 0 {
				position_beta = state[1] // Lock in hedge ratio at entry
			} else {
				position_beta = 0.0
			}
		}

		prev_ln_px = ln_px
		prev_ln_py = ln_py
	}

	return PairsTradingResult {
		returns = result_returns[:],
		trades = result_trades[:],
		betas = result_betas[:],
		spreads = result_spreads[:],
	}
}
