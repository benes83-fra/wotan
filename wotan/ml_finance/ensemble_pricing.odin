package ml_finance

import fin "../finance"
import "core:math"
import "core:mem"

// ensemble_term_structure_vol blends the LSTM's short-term regime detection
// with GARCH's long-term mean reversion to create a hybrid volatility curve.
//
// lstm_rv_dec: The Ensemble's 20-day forward RV forecast (annualized decimal, e.g., 0.15)
// omega, alpha, beta: GARCH(1,1) parameters
// current_var: Today's conditional daily variance
// horizon_days: The option's time to expiration in days
ensemble_term_structure_vol :: proc(
	lstm_rv_dec: f64,
	omega, alpha, beta, current_var: f64,
	horizon_days: int,
) -> f64 {
	if horizon_days <= 0 {return math.sqrt(current_var) * math.sqrt_f64(252.0)}

	persistence := alpha + beta
	if persistence >= 1.0 {persistence = 0.999}
	long_run_var := omega / (1.0 - persistence)

	// Convert annualized LSTM forecast to daily variance
	lstm_daily_var := (lstm_rv_dec * lstm_rv_dec) / 252.0

	sum_var := 0.0
	tau := 20.0 // LSTM's effective memory horizon

	for h in 1 ..< horizon_days + 1 {
		// GARCH expected variance at step h
		garch_var :=
			long_run_var + math.pow(persistence, f64(h - 1)) * (current_var - long_run_var)

		// Exponential decay blend: LSTM dominates short-term, GARCH dominates long-term
		w_lstm := math.exp(-f64(h) / tau)
		w_garch := 1.0 - w_lstm

		blended_var := w_lstm * lstm_daily_var + w_garch * garch_var
		sum_var += blended_var
	}

	// Return annualized decimal volatility
	avg_daily_var := sum_var / f64(horizon_days)
	return math.sqrt(avg_daily_var * 252.0)
}

// ensemble_price_and_greeks prices an option using the hybrid term structure
// and computes exact Greeks via your autograd Black-Scholes engine.
ensemble_price_and_greeks :: proc(
	S, K, T_years, r: f64,
	lstm_rv_dec: f64,
	omega, alpha, beta, current_var: f64,
	opt: fin.OptionType,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	greeks: fin.Greeks,
) {
	horizon_days := int(math.max(1.0, T_years * 252.0))

	sigma_ensemble := ensemble_term_structure_vol(
		lstm_rv_dec,
		omega,
		alpha,
		beta,
		current_var,
		horizon_days,
	)

	return fin.price_and_greeks(S, K, T_years, r, sigma_ensemble, opt, allocator)
}
