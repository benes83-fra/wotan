package ml_finance

import analytic "../analytics"
import ml "../analytics/ML"
import w "../core"
import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Configuration Structures
// ============================================================================
KalmanPairsConfig :: struct {
	// Kalman Filter Parameters
	process_noise:       f64, // Q matrix diagonal (default: 1e-5)
	measurement_noise:   f64, // R matrix initial value (default: 1e-4)
	initial_hedge_ratio: f64, // Initial beta estimate (default: 1.0)

	// Trading Parameters
	entry_threshold:     f64, // Z-score threshold to enter (default: 2.0)
	exit_threshold:      f64, // Z-score threshold to exit (default: 0.5)
	stop_loss_threshold: f64, // Z-score threshold for stop loss (default: 3.5)
	min_hold_days:       int, // Minimum days to hold position (default: 3)
	warmup_window:       int, // OLS warmup period (default: 60)
}

// Default configuration
kalman_pairs_default_config :: proc() -> KalmanPairsConfig {
	return KalmanPairsConfig {
		process_noise = 1e-5,
		measurement_noise = 1e-4,
		initial_hedge_ratio = 1.0,
		entry_threshold = 2.0,
		exit_threshold = 0.5,
		stop_loss_threshold = 3.5,
		min_hold_days = 3,
		warmup_window = 60,
	}
}

// ============================================================================
// Pairs Trading Result Structure
// ============================================================================
PairsTradingResult :: struct {
	returns:  []f64,
	trades:   []int,
	betas:    []f64,
	spreads:  []f64,
	z_scores: []f64,
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
// Kalman Filter Pairs Trading Strategy (Configurable)
// ============================================================================
kalman_pairs_strategy :: proc(
	df: ^w.DataFrame,
	col_x: string,
	col_y: string,
	config: KalmanPairsConfig,
	allocator: mem.Allocator = context.allocator,
) -> PairsTradingResult {
	x_data := extract_float_col(df, col_x, allocator)
	y_data := extract_float_col(df, col_y, allocator)
	defer delete(x_data, allocator)
	defer delete(y_data, allocator)

	n := len(x_data)
	if n == 0 {
		return PairsTradingResult{}
	}

	// Kalman Filter parameters (N=2, M=1)
	kf: analytic.KalmanFilter(2, 1)

	// Initial state
	kf.x[0] = 0.0
	kf.x[1] = config.initial_hedge_ratio

	// Initial covariance
	kf.P[0, 0] = 1.0
	kf.P[0, 1] = 0.0
	kf.P[1, 0] = 0.0
	kf.P[1, 1] = 1.0

	// F matrix (Random Walk)
	kf.F[0, 0] = 1.0
	kf.F[0, 1] = 0.0
	kf.F[1, 0] = 0.0
	kf.F[1, 1] = 1.0

	// Process noise (Q matrix) - configurable
	kf.Q[0, 0] = config.process_noise
	kf.Q[0, 1] = 0.0
	kf.Q[1, 0] = 0.0
	kf.Q[1, 1] = config.process_noise

	// Measurement noise (R matrix) - will be overwritten by OLS if warmup >= 2
	kf.R[0, 0] = config.measurement_noise

	// Initialize P, R, and state from OLS warmup
	if n >= config.warmup_window {
		X_mat := l.matrix_new(f64, config.warmup_window, 2, allocator)
		y_vec := make([]f64, config.warmup_window, allocator)
		for i in 0 ..< config.warmup_window {
			X_mat.data[i * 2 + 0] = 1.0
			X_mat.data[i * 2 + 1] = math.ln_f64(x_data[i])
			y_vec[i] = math.ln_f64(y_data[i])
		}
		ols_res := ml.ols_fit(&X_mat, y_vec, .Cholesky, allocator)

		// Update initial state from OLS
		kf.x[0] = ols_res.beta[0]
		kf.x[1] = ols_res.beta[1]

		if ols_res.vcov.data != nil && ols_res.vcov.rows == 2 && ols_res.vcov.cols == 2 {
			kf.P[0, 0] = ols_res.vcov.data[0]
			kf.P[0, 1] = ols_res.vcov.data[1]
			kf.P[1, 0] = ols_res.vcov.data[2]
			kf.P[1, 1] = ols_res.vcov.data[3]
		}

		// Update R from OLS residual variance
		kf.R[0, 0] = math.max(ols_res.sigma2, 1e-6)

		l.matrix_free(&X_mat)
		delete(y_vec, allocator)
	}

	result_returns := make([dynamic]f64, 0, allocator)
	result_trades := make([dynamic]int, 0, allocator)
	result_betas := make([dynamic]f64, 0, allocator)
	result_spreads := make([dynamic]f64, 0, allocator)
	result_z_scores := make([dynamic]f64, 0, allocator)

	position: i32 = 0
	position_beta := 0.0
	days_in_position := 0

	for i in 0 ..< n {
		ln_px := math.ln_f64(x_data[i])
		ln_py := math.ln_f64(y_data[i])

		// Update H matrix
		kf.H[0, 0] = 1.0
		kf.H[0, 1] = ln_px

		// Predict
		analytic.kalman_predict(&kf)

		// Innovation
		innovation := ln_py - (kf.H[0, 0] * kf.x[0] + kf.H[0, 1] * kf.x[1])

		// Innovation variance S
		H00 := kf.H[0, 0]
		H01 := kf.H[0, 1]
		P00 := kf.P[0, 0]
		P01 := kf.P[0, 1]
		P11 := kf.P[1, 1]
		R00 := kf.R[0, 0]

		S := H00 * P00 * H00 + 2.0 * H00 * P01 * H01 + H01 * P11 * H01 + R00

		// Calculate z_score
		z_score := innovation / math.sqrt_f64(math.max(S, 1e-12))

		// Update
		z: [1]f64
		z[0] = ln_py
		analytic.kalman_update(&kf, z)

		current_spread := ln_py - (kf.x[0] + kf.x[1] * ln_px)

		// --- Trading Logic with Configurable Thresholds ---
		signal: i32 = position

		// Track holding period
		if position != 0 {
			days_in_position += 1
		} else {
			days_in_position = 0
		}

		if position == 0 {
			// Entry signals
			if z_score > config.entry_threshold {
				signal = -1 // Short Spread
			} else if z_score < -config.entry_threshold {
				signal = 1 // Long Spread
			}
		} else {
			// Hard stop loss ALWAYS applies, even during min hold
			if (position == 1 && z_score < -config.stop_loss_threshold) ||
			   (position == -1 && z_score > config.stop_loss_threshold) {
				signal = 0
			} else if days_in_position >= config.min_hold_days {
				// Normal mean-reversion exit
				if (position == 1 && z_score > config.exit_threshold) ||
				   (position == -1 && z_score < -config.exit_threshold) {
					signal = 0
				}
			}
		}

		// Calculate daily return
		daily_ret := 0.0
		if i > 0 {
			prev_ln_px := math.ln_f64(x_data[i - 1])
			prev_ln_py := math.ln_f64(y_data[i - 1])
			ret_x := ln_px - prev_ln_px
			ret_y := ln_py - prev_ln_py

			if position == 1 {
				daily_ret = ret_y - position_beta * ret_x
			} else if position == -1 {
				daily_ret = -ret_y + position_beta * ret_x
			}
		}

		append(&result_returns, daily_ret)
		append(&result_betas, kf.x[1])
		append(&result_spreads, current_spread)
		append(&result_z_scores, z_score)

		// Update position FOR THE NEXT DAY
		if signal != position {
			append(&result_trades, i)
			position = signal
			if position != 0 {
				position_beta = kf.x[1] // Lock in hedge ratio at entry
			} else {
				position_beta = 0.0
			}
		}
	}

	return PairsTradingResult {
		returns = result_returns[:],
		trades = result_trades[:],
		betas = result_betas[:],
		spreads = result_spreads[:],
		z_scores = result_z_scores[:],
	}
}
