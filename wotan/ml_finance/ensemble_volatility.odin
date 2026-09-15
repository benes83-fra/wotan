package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

// EnsembleVolatilityForecaster combines GARCH and LSTM forecasts
// using learned inverse-MSE weights.
EnsembleVolatilityForecaster :: struct {
	lstm:            LSTMVolatilityForecaster, // ✅ Stored by value
	garch_omega:     f64,
	garch_alpha:     f64,
	garch_beta:      f64,
	ensemble_weight: f64,
	allocator:       mem.Allocator,
}

ensemble_volatility_new :: proc(
	input_size: int,
	hidden_size: int,
	seq_len: int,
	allocator: mem.Allocator = context.allocator,
) -> EnsembleVolatilityForecaster {
	e: EnsembleVolatilityForecaster
	e.allocator = allocator
	e.lstm = lstm_volatility_forecaster_new(input_size, hidden_size, seq_len, allocator)
	e.ensemble_weight = 0.5 // Start with equal weight
	return e
}

ensemble_volatility_free :: proc(e: ^EnsembleVolatilityForecaster) {
	lstm_volatility_forecaster_free(&e.lstm)
}

ensemble_add_to_optimizer :: proc(e: ^EnsembleVolatilityForecaster, opt: ^nn.Adam) {
	nn.adam_add_param(opt, e.lstm.lstm.w_ih)
	nn.adam_add_param(opt, e.lstm.lstm.w_hh)
	nn.adam_add_param(opt, e.lstm.lstm.bias)
	nn.adam_add_param(opt, e.lstm.fc1.weights)
	nn.adam_add_param(opt, e.lstm.fc1.bias)
	nn.adam_add_param(opt, e.lstm.fc2.weights)
	nn.adam_add_param(opt, e.lstm.fc2.bias)
}

garch_recursive_forecast :: proc(
	returns: []f64,
	omega, alpha, beta: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	vol_series := make([]f64, n, allocator)

	unconditional_var := omega / (1.0 - alpha - beta)
	if unconditional_var <= 0.0 || unconditional_var > 1.0 {
		unconditional_var = 0.0001 // Fallback
	}

	var_t := unconditional_var
	for i in 0 ..< n {
		vol_series[i] = math.sqrt_f64(var_t)
		var_t = omega + alpha * returns[i] * returns[i] + beta * var_t
		if var_t < 0.0 {var_t = unconditional_var}
	}
	return vol_series
}

ensemble_predict :: proc(e: ^EnsembleVolatilityForecaster, garch_vol, lstm_vol: f64) -> f64 {
	w := e.ensemble_weight
	if w < 0.0 {w = 0.0}
	if w > 1.0 {w = 1.0}
	return w * garch_vol + (1.0 - w) * lstm_vol
}

compute_optimal_weight :: proc(
	garch_forecasts: []f64,
	lstm_forecasts: []f64,
	actual_vols: []f64,
) -> f64 {
	n := len(actual_vols)
	if n == 0 || len(garch_forecasts) != n || len(lstm_forecasts) != n {
		return 0.5
	}

	best_weight := 0.5
	best_mse := math.F64_MAX

	for step in 0 ..= 100 {
		w := f64(step) / 100.0
		mse := 0.0
		for i in 0 ..< n {
			pred := w * garch_forecasts[i] + (1.0 - w) * lstm_forecasts[i]
			diff := pred - actual_vols[i]
			mse += diff * diff
		}
		mse /= f64(n)
		if mse < best_mse {
			best_mse = mse
			best_weight = w
		}
	}
	return best_weight
}
