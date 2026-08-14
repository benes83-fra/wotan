package ml_finance

import analytic "../analytics"
import ml "../analytics/ML"
import w "../core"
import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Configuration & Helpers
// ============================================================================
KalmanPairsConfig :: struct {
	// Kalman Filter Parameters
	process_noise:       f64, // Q matrix diagonal (default: 1e-5)
	measurement_noise:   f64, // R matrix initial value (default: 1e-4)
	initial_hedge_ratio: f64, // Initial beta estimate (default: 1.0)
	transaction_cost:    f64, // Transaction cost per trade (default: 0.001)

	// Trading Parameters
	entry_threshold:     f64, // Z-score threshold to enter (default: 2.0)
	exit_threshold:      f64, // Z-score threshold to exit (default: 0.5)
	stop_loss_threshold: f64, // Z-score threshold for stop loss (default: 3.5)
	min_hold_days:       int, // Minimum days to hold position (default: 3)
	warmup_window:       int, // OLS warmup period (default: 60)
}

DEFAULT_KALMAN_PAIRS_CONFIG :: KalmanPairsConfig {
	entry_threshold  = 2.0,
	exit_threshold   = 0.5,
	process_noise    = 1e-5,
	transaction_cost = 0.001,
}

// Helper to calculate annualized Sharpe ratio from a returns slice
calculate_sharpe :: proc(returns: []f64, risk_free_rate: f64 = 0.0) -> f64 {
	n := len(returns)
	if n < 2 {return 0.0}

	mean_ret := 0.0
	for r in returns {mean_ret += r}
	mean_ret /= f64(n)

	var_ret := 0.0
	for r in returns {
		diff := r - mean_ret
		var_ret += diff * diff
	}
	var_ret /= f64(n - 1)
	std_dev := math.sqrt(var_ret)

	if std_dev < 1e-10 {return 0.0}

	// Annualize: (mean / std) * sqrt(252)
	return (mean_ret / std_dev) * math.sqrt_f64(252.0)
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
// Kalman Filter Pairs Trading Strategy
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
	kf.x[1] = 1.0

	// Initial covariance
	kf.P[0, 0] = 1.0
	kf.P[0, 1] = 0.0
	kf.P[1, 0] = 0.0
	kf.P[1, 1] = 1.0

	// Process noise (Q) - using config
	kf.Q[0, 0] = config.process_noise
	kf.Q[0, 1] = 0.0
	kf.Q[1, 0] = 0.0
	kf.Q[1, 1] = config.process_noise

	// Measurement noise (R)
	kf.R[0, 0] = 1e-4

	// Initialize P and R from OLS
	if n >= 60 {
		X_mat := l.matrix_new(f64, 60, 2, allocator)
		y_vec := make([]f64, 60, allocator)
		for i in 0 ..< 60 {
			X_mat.data[i * 2 + 0] = 1.0
			X_mat.data[i * 2 + 1] = x_data[i]
			y_vec[i] = y_data[i]
		}
		ols_res := ml.ols_fit(&X_mat, y_vec, .Cholesky, allocator)
		kf.x[0] = ols_res.beta[0]
		kf.x[1] = ols_res.beta[1]
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

	for i in 0 ..< n {
		px := x_data[i]
		py := y_data[i]

		// Update H matrix
		kf.H[0, 0] = 1.0
		kf.H[0, 1] = px

		// Predict
		analytic.kalman_predict(&kf)

		// Innovation
		innovation := py - (kf.H[0, 0] * kf.x[0] + kf.H[0, 1] * kf.x[1])

		// Innovation variance S
		H00 := kf.H[0, 0]
		H01 := kf.H[0, 1]
		P00 := kf.P[0, 0]
		P01 := kf.P[0, 1]
		P11 := kf.P[1, 1]
		R00 := kf.R[0, 0]

		S := H00 * P00 * H00 + 2.0 * H00 * P01 * H01 + H01 * P11 * H01 + R00
		z_score := innovation / math.sqrt_f64(math.max(S, 1e-12))

		// Update
		z: [1]f64
		z[0] = py
		analytic.kalman_update(&kf, z)

		current_spread := py - (kf.x[0] + kf.x[1] * px)

		// --- Trading Logic ---
		signal: i32 = position
		if position == 0 {
			if z_score > config.entry_threshold {
				signal = -1
			} else if z_score < -config.entry_threshold {
				signal = 1
			}
		} else {
			if position == 1 {
				if z_score > -config.exit_threshold || z_score < -3.0 {signal = 0}
			} else if position == -1 {
				if z_score < config.exit_threshold || z_score > 3.0 {signal = 0}
			}
		}

		// Calculate daily return
		daily_ret := 0.0
		if i > 0 {
			prev_px := x_data[i - 1]
			prev_py := y_data[i - 1]
			if prev_px != 0.0 && prev_py != 0.0 {
				ret_x := (px / prev_px) - 1.0
				ret_y := (py / prev_py) - 1.0
				if position == 1 {
					daily_ret = ret_y - position_beta * ret_x
				} else if position == -1 {
					daily_ret = -ret_y + position_beta * ret_x
				}
			}
		}

		// Apply Transaction Costs
		if signal != position && config.transaction_cost > 0.0 {
			daily_ret -= config.transaction_cost
		}

		append(&result_returns, daily_ret)
		append(&result_betas, kf.x[1])
		append(&result_spreads, current_spread)
		append(&result_z_scores, z_score)

		// Update position
		if signal != position {
			append(&result_trades, i)
			position = signal
			if position != 0 {
				position_beta = kf.x[1]
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

// ============================================================================
// Grid Search
// ============================================================================
kalman_pairs_grid_search :: proc(
	df: ^w.DataFrame,
	col_x: string,
	col_y: string,
	entry_thresholds: []f64,
	exit_thresholds: []f64,
	process_noises: []f64,
	transaction_cost: f64 = 0.001,
	allocator: mem.Allocator = context.allocator,
) -> (
	best_config: KalmanPairsConfig,
	best_sharpe: f64,
) {
	best_sharpe = -math.F64_MAX
	best_config = DEFAULT_KALMAN_PAIRS_CONFIG

	for entry in entry_thresholds {
		for exit in exit_thresholds {
			for noise in process_noises {
				config := KalmanPairsConfig {
					entry_threshold  = entry,
					exit_threshold   = exit,
					process_noise    = noise,
					transaction_cost = transaction_cost,
				}

				// Run strategy using temp_allocator for intermediate results
				result := kalman_pairs_strategy(df, col_x, col_y, config, context.temp_allocator)
				sharpe := calculate_sharpe(result.returns)

				if sharpe > best_sharpe {
					best_sharpe = sharpe
					best_config = config
				}
			}
		}
	}
	return best_config, best_sharpe
}
