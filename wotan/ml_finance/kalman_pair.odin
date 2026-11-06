package ml_finance

import analytic "../analytics"
import ml "../analytics/ML"
import w "../core"
import fin "../finance"
import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Configuration
// ============================================================================
KalmanPairsConfig :: struct {
	// Kalman Filter Parameters
	process_noise:       f64, // Q = process_noise * I, random-walk drift of (alpha, beta) per row
	measurement_noise:   f64, // R; replaced by the OLS residual variance when the warmup is fitted
	initial_hedge_ratio: f64, // beta_0 when there are fewer valid rows than warmup_window
	transaction_cost:    f64, // charged on each position change, fraction of gross notional

	// Trading Parameters
	entry_threshold:     f64, // enter when |z| > entry
	exit_threshold:      f64, // exit when z is back inside +-exit (after min_hold_days)
	stop_loss_threshold: f64, // exit (and cool down) when |z| > stop, also inside the minimum hold
	min_hold_days:       int, // minimum holding period for a normal exit
	warmup_window:       int, // OLS warmup rows (not traded); <= 0 means 60
	cooldown_days:       int, // no new entry for this many rows after a stop loss
}

DEFAULT_KALMAN_PAIRS_CONFIG :: KalmanPairsConfig {
	process_noise       = 1e-5,
	measurement_noise   = 1e-4,
	initial_hedge_ratio = 1.0,
	transaction_cost    = 0.001,
	entry_threshold     = 2.0,
	exit_threshold      = 0.5,
	stop_loss_threshold = 3.5,
	min_hold_days       = 3,
	warmup_window       = 60,
	cooldown_days       = 5,
}

kalman_pairs_default_config :: proc() -> KalmanPairsConfig {
	return DEFAULT_KALMAN_PAIRS_CONFIG
}

// ============================================================================
// Pairs Trading Result Structure
// ============================================================================
PairsTradingResult :: struct {
	returns:  []f64, // position * (dy - beta dx) / (|y| + |beta x|) of the previous row, less costs
	trades:   []int, // rows where the position changed
	betas:    []f64, // filtered hedge ratio
	spreads:  []f64, // y - alpha - beta x with the filtered state (NaN on skipped rows)
	z_scores: []f64, // one-step-ahead innovation / sqrt(S), the trading signal (0 when not filtered)
}

pairs_trading_result_free :: proc(
	res: ^PairsTradingResult,
	allocator: mem.Allocator = context.allocator,
) {
	delete(res.returns, allocator)
	delete(res.trades, allocator)
	delete(res.betas, allocator)
	delete(res.spreads, allocator)
	delete(res.z_scores, allocator)
	res^ = {}
}

// Float column of a DataFrame; nulls become NaN
extract_float_col :: proc(df: ^w.DataFrame, col_name: string, allocator: mem.Allocator) -> []f64 {
	n := df.rows
	out := make([]f64, n, allocator)
	col := w.column(df, col_name)
	for i in 0 ..< n {
		v, is_null := w.column_at_float(col, i)
		out[i] = is_null ? math.nan_f64() : v
	}
	return out
}

// ============================================================================
// Kalman Filter Pairs Trading Strategy
// ============================================================================
// Time-varying regression y_t = alpha_t + beta_t x_t + e_t with (alpha, beta) a random walk
// (F = I, Q = process_noise I), filtered online. With enough rows, the first warmup_window
// valid rows give the start by OLS (state = coefficients, P = their covariance, R = residual
// variance) and are not traded; the filter runs from the next row on, so no signal uses
// data from its future. The signal z_t is the innovation of y_t against the prediction
// from rows < t, over sqrt(S). The position decided on row t (+1 long y / short beta x,
// -1 the reverse, beta fixed at entry) earns from row t to the next valid row. Rows with a
// missing or non-positive price are skipped: no update, no trade, return 0.
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
	valid :: proc(v: f64) -> bool {return v > 0.0 && !math.is_inf(v)} 	// false for NaN

	kf: analytic.KalmanFilter(2, 1)
	kf.F = 1 // identity: the coefficients follow a random walk
	kf.x = {0.0, config.initial_hedge_ratio}
	kf.P = 1
	kf.Q = config.process_noise
	kf.R[0, 0] = config.measurement_noise

	warmup := config.warmup_window
	if warmup <= 0 {warmup = 60}
	start := 0 // first row the filter runs on
	{
		rows := make([dynamic]int, 0, warmup, allocator)
		defer delete(rows)
		for i in 0 ..< n {
			if len(rows) == warmup {break}
			if valid(x_data[i]) && valid(y_data[i]) {append(&rows, i)}
		}
		if len(rows) == warmup && warmup > 2 {
			X_mat := l.matrix_new(f64, warmup, 2, allocator)
			y_vec := make([]f64, warmup, allocator)
			for r, j in rows {
				X_mat.data[j * 2 + 0] = 1.0
				X_mat.data[j * 2 + 1] = x_data[r]
				y_vec[j] = y_data[r]
			}
			ols := ml.ols_fit(&X_mat, y_vec, .Cholesky, allocator)
			if len(ols.beta) == 2 && !math.is_nan(ols.beta[1]) && ols.vcov.rows == 2 {
				kf.x = {ols.beta[0], ols.beta[1]}
				for a in 0 ..< 2 {for b in 0 ..< 2 {kf.P[a, b] = ols.vcov.data[a * 2 + b]}}
				kf.R[0, 0] = math.max(ols.sigma2, 1e-12)
				start = rows[warmup - 1] + 1
			}
			ml._ols_result_free(&ols, allocator)
			l.matrix_free(&X_mat)
			delete(y_vec, allocator)
		}
	}

	result_returns := make([dynamic]f64, 0, n, allocator)
	result_trades := make([dynamic]int, 0, allocator)
	result_betas := make([dynamic]f64, 0, n, allocator)
	result_spreads := make([dynamic]f64, 0, n, allocator)
	result_z_scores := make([dynamic]f64, 0, n, allocator)

	position: i32 = 0
	position_beta := 0.0
	days_in_position := 0
	last_stop_loss_day := min(int) / 2
	prev_px, prev_py := math.nan_f64(), math.nan_f64() // last valid prices

	for i in 0 ..< n {
		px := x_data[i]
		py := y_data[i]
		ok := valid(px) && valid(py)

		// Return of the position held into this row
		daily_ret := 0.0
		if ok && position != 0 {
			gross := prev_py + math.abs(position_beta) * prev_px
			daily_ret = f64(position) * ((py - prev_py) - position_beta * (px - prev_px)) / gross
		}

		if !ok || i < start {
			// warmup or skipped row: no filtering, no trading
			spread := ok ? py - (kf.x[0] + kf.x[1] * px) : math.nan_f64()
			append(&result_returns, daily_ret)
			append(&result_betas, kf.x[1])
			append(&result_spreads, spread)
			append(&result_z_scores, 0.0)
			if ok {prev_px, prev_py = px, py}
			continue
		}

		kf.H[0, 0] = 1.0
		kf.H[0, 1] = px
		analytic.kalman_predict(&kf)
		innovation := py - (kf.x[0] + kf.x[1] * px)
		S := kf.P[0, 0] + 2.0 * px * kf.P[0, 1] + px * px * kf.P[1, 1] + kf.R[0, 0]
		z_score := innovation / math.sqrt(math.max(S, 1e-300))
		analytic.kalman_update(&kf, [1]f64{py})
		current_spread := py - (kf.x[0] + kf.x[1] * px)

		// --- Trading logic on this row's signal ---
		signal := position
		if position != 0 {days_in_position += 1} else {days_in_position = 0}

		if position == 0 {
			if i - last_stop_loss_day < config.cooldown_days {
				signal = 0
			} else if z_score > config.entry_threshold {
				signal = -1
			} else if z_score < -config.entry_threshold {
				signal = 1
			}
		} else {
			stop :=
				(position == 1 && z_score < -config.stop_loss_threshold) ||
				(position == -1 && z_score > config.stop_loss_threshold)
			back :=
				(position == 1 && z_score > -config.exit_threshold) ||
				(position == -1 && z_score < config.exit_threshold)
			if stop || (back && days_in_position >= config.min_hold_days) {
				signal = 0
				if stop {last_stop_loss_day = i}
			}
		}

		if signal != position && config.transaction_cost > 0.0 {
			daily_ret -= config.transaction_cost
		}

		append(&result_returns, daily_ret)
		append(&result_betas, kf.x[1])
		append(&result_spreads, current_spread)
		append(&result_z_scores, z_score)

		if signal != position {
			append(&result_trades, i)
			position = signal
			position_beta = position != 0 ? kf.x[1] : 0.0
		}
		prev_px, prev_py = px, py
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
// In-sample: the grid point with the highest annualised Sharpe ratio
// (fin.sharpe_ratio_from_returns); the other fields come from DEFAULT_KALMAN_PAIRS_CONFIG.
// The best of many in-sample Sharpe ratios is biased upwards: pass `trials` to get every
// grid point's returns as the columns of a [T, N] matrix (allocated with `allocator`, the
// grid in loop order: entry, exit, noise, min hold, cooldown, the last fastest) for
// deflated_sharpe_trials / backtest_overfitting / print_selection_stats.
kalman_pairs_grid_search :: proc(
	df: ^w.DataFrame,
	col_x: string,
	col_y: string,
	entry_thresholds: []f64,
	exit_thresholds: []f64,
	process_noises: []f64,
	min_hold_days_list: []int,
	cooldown_days_list: []int,
	transaction_cost: f64 = 0.001,
	allocator: mem.Allocator = context.allocator,
	trials: ^l.Matrix(f64) = nil,
) -> (
	best_config: KalmanPairsConfig,
	best_sharpe: f64,
) {
	best_sharpe = -math.F64_MAX
	best_config = DEFAULT_KALMAN_PAIRS_CONFIG
	N :=
		len(entry_thresholds) *
		len(exit_thresholds) *
		len(process_noises) *
		len(min_hold_days_list) *
		len(cooldown_days_list)
	j := 0

	for entry in entry_thresholds {
		for exit in exit_thresholds {
			for noise in process_noises {
				for min_hold in min_hold_days_list {
					for cooldown in cooldown_days_list {
						config := DEFAULT_KALMAN_PAIRS_CONFIG
						config.entry_threshold = entry
						config.exit_threshold = exit
						config.process_noise = noise
						config.transaction_cost = transaction_cost
						config.min_hold_days = min_hold
						config.cooldown_days = cooldown

						result := kalman_pairs_strategy(df, col_x, col_y, config, allocator)
						sharpe := fin.sharpe_ratio_from_returns(result.returns)
						if trials != nil {
							T := len(result.returns)
							if j == 0 {trials^ = l.matrix_new(f64, T, N, allocator)}
							assert(
								T == trials.rows,
								"kalman_pairs_grid_search: return series of different lengths",
							)
							for r, t in result.returns {trials.data[t * N + j] = r}
						}
						j += 1
						pairs_trading_result_free(&result, allocator)

						if sharpe > best_sharpe {
							best_sharpe = sharpe
							best_config = config
						}
					}
				}
			}
		}
	}
	return best_config, best_sharpe
}
