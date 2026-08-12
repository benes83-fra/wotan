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
// Kalman Filter Pairs Trading Strategy
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
	state[1] = 1.0 // Default fallback

	// Covariance Matrix [P00, P01, P10, P11]
	cov := make([]f64, 4, allocator)
	cov[0] = 1.0; cov[1] = 0.0
	cov[2] = 0.0; cov[3] = 1.0

	Q := 1e-4 // Process noise
	R := 1.0 // Measurement noise

	// =========================================================
	// USE OLS FROM analytics/ML TO INITIALIZE THE KALMAN FILTER
	// We use the first `window` observations to estimate the
	// initial intercept and hedge ratio (beta).
	// =========================================================
	init_len := window
	if init_len >= n {
		init_len = n - 1 // Ensure we have at least 1 observation left for the filter
	}

	if init_len >= 2 {
		X_mat := l.matrix_new(f64, init_len, 2, allocator)
		y_vec := make([]f64, init_len, allocator)

		for i in 0 ..< init_len {
			X_mat.data[i * 2 + 0] = 1.0 // Intercept column
			X_mat.data[i * 2 + 1] = x_data[i] // X variable
			y_vec[i] = y_data[i] // Y variable
		}

		// Call your existing OLS from analytics/ML
		ols_res := ml.ols_fit(&X_mat, y_vec, .Cholesky, allocator)

		state[0] = ols_res.beta[0] // Initial Intercept
		state[1] = ols_res.beta[1] // Initial Hedge Ratio (Slope)

		// Shrink initial covariance since we have a solid OLS estimate
		cov[0] = 0.01; cov[1] = 0.0
		cov[2] = 0.0; cov[3] = 0.01

		l.matrix_free(&X_mat)
		delete(y_vec, allocator)

		// Note: If your ml.ols_fit requires explicit cleanup (e.g. _ols_result_free),
		// you would call it here.
	}
	prev_x := x_data[0]
	prev_y := y_data[0]


	position: i32 = 0
	position_beta := 0.0 // Hedge ratio locked at the time of entry

	for i in 0 ..< n {
		px := x_data[i]
		py := y_data[i]

		// --- Kalman Filter Update ---
		pred_cov_00 := cov[0] + Q
		pred_cov_11 := cov[3] + Q

		spread_pred := state[0] + state[1] * px
		innovation := py - spread_pred

		s := pred_cov_00 + 2.0 * px * cov[1] + px * px * pred_cov_11 + R

		k1 := (cov[0] + px * cov[1]) / s
		k2 := (cov[1] + px * cov[3]) / s

		state[0] = state[0] + k1 * innovation
		state[1] = state[1] + k2 * innovation

		new_cov_00 := (1.0 - k1) * pred_cov_00 - k1 * px * cov[1]
		new_cov_01 := (1.0 - k1) * cov[1] - k1 * px * pred_cov_11
		new_cov_10 := -k2 * pred_cov_00 + (1.0 - k2 * px) * cov[1]
		new_cov_11 := -k2 * cov[1] + (1.0 - k2 * px) * pred_cov_11

		cov[0] = new_cov_00
		cov[1] = new_cov_01
		cov[2] = new_cov_10
		cov[3] = new_cov_11

		current_spread := py - (state[0] + state[1] * px)

		z_score := 0.0
		if s > 1e-12 {
			z_score = innovation / math.sqrt(s)
		}

		// --- Trading Logic ---
		entry_threshold := 1.5
		exit_threshold := 0.5

		signal: i32 = position

		if position == 0 {
			if z_score > entry_threshold {
				signal = -1 // Short Spread
			} else if z_score < -entry_threshold {
				signal = 1 // Long Spread
			}
		} else {
			if (position == 1 && z_score > -exit_threshold) ||
			   (position == -1 && z_score < exit_threshold) {
				signal = 0 // Exit
			}
		}

		// 1. Calculate daily return based on the position HELD during this day (the OLD position)
		daily_ret := 0.0
		if i > 0 && prev_x != 0.0 && prev_y != 0.0 {
			ret_x := (px / prev_x) - 1.0
			ret_y := (py / prev_y) - 1.0

			if position == 1 {
				daily_ret = ret_y - position_beta * ret_x
			} else if position == -1 {
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

		prev_x = px
		prev_y = py
	}

	return PairsTradingResult {
		returns = result_returns[:],
		trades = result_trades[:],
		betas = result_betas[:],
		spreads = result_spreads[:],
	}
}
