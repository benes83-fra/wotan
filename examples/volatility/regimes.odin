package main

// Stage 3: market regimes. A two-state Gaussian hidden Markov model of SPY's
// daily returns, fitted by Baum-Welch on the days before each fold. The
// filtered probability of the turbulent state on day d uses the returns up to
// day d only (hmm_filter), so it is known at that close and can be traded on.

import mlf "../../wotan/ml_finance"
import "core:math"

Regime_Summary :: struct {
	calm_vol, turbulent_vol:   f64, // annualised %
	stay_calm, stay_turbulent: f64, // probability of staying in the state the next day
}

// regime_fold fits the model to the returns before day s and writes
// P(turbulent | returns up to d) for days s ..< e to f.p_turbulent
regime_fold :: proc(m: ^Market, s, e: int, f: ^Forecasts) -> (r: Regime_Summary) {
	obs := make([]f64, e - 1, context.temp_allocator) // returns 1 ..< e, in %
	for &v, i in obs {v = 100 * m.ret[i + 1]}
	h := mlf.hmm_new(2, context.temp_allocator)
	defer mlf.hmm_free(&h)
	train := obs[:s - 1]
	mlf.hmm_init_simple(&h, train)
	mlf.hmm_fit(&h, train)
	turb := 1 if h.emission_var[1] > h.emission_var[0] else 0
	calm := 1 - turb
	// filtering the whole prefix with the fitted parameters: row d - 1 is day d
	filt := mlf.hmm_filter(&h, obs, context.temp_allocator)
	for d in s ..< e {f.p_turbulent[d] = filt[(d - 1) * 2 + turb]}
	r.calm_vol = math.sqrt(f64(TRADING_DAYS) * h.emission_var[calm])
	r.turbulent_vol = math.sqrt(f64(TRADING_DAYS) * h.emission_var[turb])
	r.stay_calm = h.transition[calm * 2 + calm]
	r.stay_turbulent = h.transition[turb * 2 + turb]
	return
}
