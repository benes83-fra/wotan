// ===== ./wotan/ml_finance/features.odin =====
package ml_finance

import "core:math"
import "core:mem"

// compute_trading_features generates standard RL trading features:
// 1. Log Returns
// 2. RSI (14-period, Wilder smoothed)
// 3. Volume Z-Score (20-period rolling)
// 4. Volume Change Ratio
compute_trading_features :: proc(
	prices: []f64,
	volumes: []f64,
	allocator: mem.Allocator,
) -> (
	indicators: []f64,
	n_indicators: int,
) {
	n_days := len(prices)
	n_indicators = 4
	indicators = make([]f64, n_days * n_indicators, allocator)

	// 1. Log Returns
	log_returns := make([]f64, n_days, allocator)
	defer delete(log_returns, allocator)
	for i in 1 ..< n_days {
		if prices[i - 1] > 0 {
			log_returns[i] = math.ln_f64(prices[i] / prices[i - 1])
		}
	}

	// 2. RSI(14)
	rsi_period := 14
	rsi_values := make([]f64, n_days, allocator)
	defer delete(rsi_values, allocator)
	if n_days > rsi_period {
		avg_gain := 0.0
		avg_loss := 0.0
		for i in 1 ..< rsi_period + 1 {
			change := log_returns[i]
			if change > 0 {avg_gain += change} else {avg_loss += math.abs(change)}
		}
		avg_gain /= f64(rsi_period)
		avg_loss /= f64(rsi_period)
		if avg_loss > 1e-10 {
			rs := avg_gain / avg_loss
			rsi_values[rsi_period] = 100.0 - 100.0 / (1.0 + rs)
		} else {
			rsi_values[rsi_period] = 100.0
		}
		for i in rsi_period + 1 ..< n_days {
			change := log_returns[i]
			gain := 0.0; loss := 0.0
			if change > 0 {gain = change} else {loss = math.abs(change)}
			avg_gain = (avg_gain * f64(rsi_period - 1) + gain) / f64(rsi_period)
			avg_loss = (avg_loss * f64(rsi_period - 1) + loss) / f64(rsi_period)
			if avg_loss > 1e-10 {
				rs := avg_gain / avg_loss
				rsi_values[i] = 100.0 - 100.0 / (1.0 + rs)
			} else {
				rsi_values[i] = 100.0
			}
		}
	}

	// 3. Rolling Volume Mean & Std
	vol_window := 20
	vol_mean := make([]f64, n_days, allocator)
	vol_std := make([]f64, n_days, allocator)
	defer delete(vol_mean, allocator)
	defer delete(vol_std, allocator)
	for i in vol_window ..< n_days {
		sum := 0.0
		for j in (i - vol_window) ..< i {sum += volumes[j]}
		vol_mean[i] = sum / f64(vol_window)

		sum_sq := 0.0
		for j in (i - vol_window) ..< i {
			d := volumes[j] - vol_mean[i]
			sum_sq += d * d
		}
		vol_std[i] = math.sqrt(sum_sq / f64(vol_window))
		if vol_std[i] < 1.0 {vol_std[i] = 1.0}
	}

	// 4. Pack into indicators array
	for i in 0 ..< n_days {
		base := i * n_indicators
		indicators[base + 0] = log_returns[i]
		indicators[base + 1] = (rsi_values[i] - 50.0) / 50.0
		if i >= vol_window && vol_std[i] > 1.0 {
			z := (volumes[i] - vol_mean[i]) / vol_std[i]
			indicators[base + 2] = math.max(-3.0, math.min(3.0, z)) / 3.0
		}
		if i >= vol_window && vol_mean[i] > 1.0 {
			indicators[base + 3] =
				math.max(-2.0, math.min(2.0, volumes[i] / vol_mean[i] - 1.0)) / 2.0
		}
	}
	return indicators, n_indicators
}
