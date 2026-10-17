package behavioral_finance

import w "../core"
import net "../net"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Anchoring Signal (George & Hwang, 2004)
// ============================================================================

// Computes proximity to 52-week high for each asset at each time step.
// Returns a matrix [time x assets] where each value is P_t / max(P_{t-252:t})
compute_anchoring_signal :: proc(
	prices: [][]f64, // [assets][time]
	lookback: int = 252, // 52 weeks
	allocator: mem.Allocator = context.allocator,
) -> [][]f64 {
	n_assets := len(prices)
	n_time := len(prices[0])

	signal := make([][]f64, n_assets, allocator)
	for a in 0 ..< n_assets {
		signal[a] = make([]f64, n_time, allocator)
		for t in 0 ..< n_time {
			if t < lookback {
				signal[a][t] = 0.0 // Not enough history
				continue
			}

			// Find 52-week high
			max_price := -math.F64_MAX
			for i in t - lookback ..< t {
				if prices[a][i] > max_price {
					max_price = prices[a][i]
				}
			}

			// Proximity ratio
			if max_price > 0 {
				signal[a][t] = prices[a][t] / max_price
			} else {
				signal[a][t] = 0.0
			}
		}
	}

	return signal
}

// ============================================================================
// Behavioral Kalman Filter (Anchoring as Sticky Prior)
// ============================================================================

// Standard Kalman filter with behavioral anchoring modification.
// When new observation conflicts with anchor, reduce Kalman gain by alpha.
// This models slow belief updating when price approaches 52-week high.
behavioral_kalman_filter :: proc(
	observations: []f64,
	anchor: f64,
	alpha: f64 = 0.5, // Anchoring strength (0 = no anchor, 1 = full anchor)
	process_var: f64 = 0.01,
	measurement_var: f64 = 0.1,
	allocator: mem.Allocator = context.allocator,
) -> (
	[]f64,
	[]f64,
) { 	// Returns (filtered_estimates, kalman_gains)
	n := len(observations)
	estimates := make([]f64, n, allocator)
	gains := make([]f64, n, allocator)

	// Initialize
	x := observations[0]
	P := 1.0

	for t in 0 ..< n {
		// Predict
		x_pred := x
		P_pred := P + process_var

		// Update
		K := P_pred / (P_pred + measurement_var)

		// Behavioral modification: reduce K when observation conflicts with anchor
		innovation := observations[t] - x_pred
		if innovation > 0 && x_pred < anchor {
			// Observation is pushing estimate up toward anchor
			// Reduce Kalman gain to model slow updating
			K = K * (1.0 - alpha)
		}

		x = x_pred + K * innovation
		P = (1.0 - K) * P_pred

		estimates[t] = x
		gains[t] = K
	}

	return estimates, gains
}

// ============================================================================
// Portfolio Construction
// ============================================================================

// Generate long/short portfolio weights based on anchoring signal.
// Long: assets with proximity > long_threshold (near 52-week high)
// Short: assets with proximity < short_threshold (far from 52-week high)
generate_anchoring_portfolio :: proc(
	signal: []f64, // Signal at current time for all assets
	long_threshold: f64 = 0.95,
	short_threshold: f64 = 0.5,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_assets := len(signal)
	weights := make([]f64, n_assets, allocator)

	long_count := 0
	short_count := 0

	// Count how many in each bucket
	for a in 0 ..< n_assets {
		if signal[a] > long_threshold {
			long_count += 1
		} else if signal[a] < short_threshold {
			short_count += 1
		}
	}

	// Assign equal weights within each bucket
	for a in 0 ..< n_assets {
		if signal[a] > long_threshold && long_count > 0 {
			weights[a] = 1.0 / f64(long_count) // Long position
		} else if signal[a] < short_threshold && short_count > 0 {
			weights[a] = -1.0 / f64(short_count) // Short position
		} else {
			weights[a] = 0.0 // Flat
		}
	}

	return weights
}
// Cross-sectional Long/Short based on percentiles (Market Neutral)
generate_cross_sectional_portfolio :: proc(
	signal: []f64,
	n_long: int,
	n_short: int,
	allocator: mem.Allocator,
) -> []f64 {
	n_assets := len(signal)
	weights := make([]f64, n_assets, allocator)

	// Create indices and sort them based on signal (ascending order)
	indices := make([]int, n_assets, allocator)
	for i in 0 ..< n_assets {indices[i] = i}

	// Simple insertion sort for indices based on signal value
	for i in 1 ..< n_assets {
		key := indices[i]
		j := i - 1
		for j >= 0 && signal[indices[j]] > signal[key] {
			indices[j + 1] = indices[j]
			j -= 1
		}
		indices[j + 1] = key
	}

	// Short the bottom n_short assets (furthest from 52-week high)
	for i in 0 ..< n_short {
		weights[indices[i]] = -1.0 / f64(n_short)
	}

	// Long the top n_long assets (closest to 52-week high)
	for i in n_assets - n_long ..< n_assets {
		weights[indices[i]] = 1.0 / f64(n_long)
	}

	delete(indices, allocator)
	return weights
}
// Long-Only Breakout Portfolio (Cash when no breakouts)
generate_long_only_portfolio :: proc(
	signal: []f64,
	threshold: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n_assets := len(signal)
	weights := make([]f64, n_assets, allocator)

	count := 0
	for a in 0 ..< n_assets {
		if signal[a] >= threshold {
			count += 1
		}
	}

	// If nothing is breaking out, hold 100% cash (all weights remain 0.0)
	if count == 0 {
		return weights
	}

	// Equal weight among the breakouts
	weight := 1.0 / f64(count)
	for a in 0 ..< n_assets {
		if signal[a] >= threshold {
			weights[a] = weight
		}
	}

	return weights
}
