package ml_finance

import "core:math"
import "core:mem"

// compute_trading_features: 4 features per day, row-major [n_days * 4], each from data up
// to and including that day:
// 0. log return ln(p_t / p_{t-1})                                  (0 on day 0)
// 1. (RSI_14 - 50) / 50, Wilder RSI on price changes as TA-Lib RSI(14): seeded with the
//    mean gain and loss of the first 14 changes, then avg = (13 avg + x) / 14. 0 (RSI 50)
//    until day 14 and for a flat price, where TA-Lib reports 0.
// 2. clip(z, -3, 3) / 3 with z = (v_t - mean) / std over the 20 previous days (population
//    std, the current day excluded); 0 for the first 20 days and when std = 0
// 3. clip(v_t / mean - 1, -2, 2) / 2 with the same 20-day mean; 0 when the mean is 0
// Non-positive prices give a 0 return and no price change for RSI.
compute_trading_features :: proc(
	prices: []f64,
	volumes: []f64,
	allocator: mem.Allocator,
) -> (
	indicators: []f64,
	n_indicators: int,
) {
	assert(
		len(prices) == len(volumes),
		"compute_trading_features: prices and volumes differ in length",
	)
	n_days := len(prices)
	n_indicators = 4
	indicators = make([]f64, n_days * n_indicators, allocator)

	RSI_PERIOD :: 14
	VOL_WINDOW :: 20
	rsi :: proc(gain, loss: f64) -> f64 {
		if gain + loss == 0 {return 50.0}
		return 100.0 * gain / (gain + loss) // = 100 - 100 / (1 + gain / loss)
	}

	avg_gain, avg_loss := 0.0, 0.0
	for i in 0 ..< n_days {
		base := i * n_indicators
		ok := i > 0 && prices[i - 1] > 0 && prices[i] > 0

		// 0. log return
		if ok {indicators[base + 0] = math.ln(prices[i] / prices[i - 1])}

		// 1. RSI
		change := ok ? prices[i] - prices[i - 1] : 0.0
		gain, loss := math.max(change, 0.0), math.max(-change, 0.0)
		rsi_value := 50.0
		if i >= 1 && i <= RSI_PERIOD {
			avg_gain += gain / RSI_PERIOD
			avg_loss += loss / RSI_PERIOD
			if i == RSI_PERIOD {rsi_value = rsi(avg_gain, avg_loss)}
		} else if i > RSI_PERIOD {
			avg_gain = (avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
			avg_loss = (avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD
			rsi_value = rsi(avg_gain, avg_loss)
		}
		indicators[base + 1] = (rsi_value - 50.0) / 50.0

		// 2-3. volume vs the previous 20 days
		if i >= VOL_WINDOW {
			mean := 0.0
			for j in i - VOL_WINDOW ..< i {mean += volumes[j]}
			mean /= VOL_WINDOW
			ss := 0.0
			for j in i - VOL_WINDOW ..< i {ss += (volumes[j] - mean) * (volumes[j] - mean)}
			std := math.sqrt(ss / VOL_WINDOW)
			if std > 0 {
				z := (volumes[i] - mean) / std
				indicators[base + 2] = math.clamp(z, -3.0, 3.0) / 3.0
			}
			if mean > 0 {
				indicators[base + 3] = math.clamp(volumes[i] / mean - 1.0, -2.0, 2.0) / 2.0
			}
		}
	}
	return indicators, n_indicators
}
