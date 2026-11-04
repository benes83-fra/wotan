package ml_finance

import fin "../finance"
import "core:math"
import "core:mem"

// ensemble_term_structure_vol: annualised volatility over the next horizon_days days from
// a blend of the sequence model's forecast and the GARCH(1,1) variance path:
//   v_1 = current_var, v_{h+1} = omega + (alpha + beta) v_h   (E[sigma^2_{t+h}], as arch's
//   analytic forecast; exact for any persistence, incl. IGARCH where v_h = v_1 + (h - 1) omega)
//   blend_h = w_h lstm_var + (1 - w_h) v_h,  w_h = exp(-h / 20)   (the model dominates
//   the first weeks, GARCH mean reversion the long end)
//   sigma = sqrt(252 mean_h blend_h)
// lstm_rv_dec: the model's annualised vol forecast (decimal, e.g. 0.15)
// current_var: the one-step-ahead daily variance sigma^2_{t+1} (GARCH's next-day forecast)
// horizon_days <= 0: sqrt(252 current_var)
ensemble_term_structure_vol :: proc(
	lstm_rv_dec: f64,
	omega, alpha, beta, current_var: f64,
	horizon_days: int,
) -> f64 {
	if horizon_days <= 0 {return math.sqrt(current_var * 252.0)}
	persistence := alpha + beta
	lstm_daily_var := lstm_rv_dec * lstm_rv_dec / 252.0
	TAU :: 20.0 // the sequence model's effective horizon in days

	sum_var := 0.0
	garch_var := current_var
	for h in 1 ..= horizon_days {
		w_lstm := math.exp(-f64(h) / TAU)
		sum_var += w_lstm * lstm_daily_var + (1.0 - w_lstm) * garch_var
		garch_var = omega + persistence * garch_var
	}
	return math.sqrt(sum_var / f64(horizon_days) * 252.0)
}

// ensemble_price_and_greeks: Black-Scholes price and Greeks (fin.price_and_greeks, by
// autograd) at the term-structure volatility over the option's life.
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
	horizon_days := max(1, int(math.round(T_years * 252.0))) // trading days, rounded

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
