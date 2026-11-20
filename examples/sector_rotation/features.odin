package main

// Stage 2: features and targets. Every feature of day t uses data up to and
// including day t only. Each feature is z-scored across the eleven sectors
// every day, so the models see how a sector stands against the others, not
// where the market as a whole went. The target is a sector's log return over
// the next HORIZON days minus the average of all sectors (in %).

import mlf "../../wotan/ml_finance"
import "core:fmt"
import "core:math"

WINDOW :: 20 // days of features the sequence model sees
HORIZON :: 5 // forecast and holding period, in trading days
N_FEATURES :: 8
FEATURE_NAMES := [N_FEATURES]string {
	"log return",
	"RSI(14)",
	"volume z-score",
	"volume ratio",
	"momentum 20d",
	"momentum 60d",
	"volatility 20d",
	"frac. diff. price",
}

// Dataset: day-major arrays
Dataset :: struct {
	x:         []f64, // [day][asset][feature], z-scored across assets
	target:    []f64, // [day][asset], NaN where the next HORIZON days are not known
	first_day: int, // the first day on which every feature is defined
	frac_d:    f64, // the fractional differencing order
}

// build_dataset computes the features of every asset and day. The fractional
// differencing order d is the smallest that makes every sector's log price
// stationary (ADF at 5%) over the first `fit_days` days, so later data does
// not choose it.
build_dataset :: proc(m: ^Market, fit_days: int) -> (ds: Dataset) {
	T := m.n_days
	raw := make([]f64, T * N_ASSETS * N_FEATURES)
	ds.target = make([]f64, T * N_ASSETS)

	// The differencing order, from the first fit_days days
	log_price := make([][]f64, N_ASSETS)
	for a in 0 ..< N_ASSETS {
		log_price[a] = asset_series(m.close, a)
		for &v in log_price[a] {v = math.ln(v)}
		r := mlf.frac_diff_min_d(log_price[a][:fit_days], threshold = 1e-3)
		ds.frac_d = max(ds.frac_d, r.d)
	}

	first := 0
	for a in 0 ..< N_ASSETS {
		close := asset_series(m.close, a, context.temp_allocator)
		volume := asset_series(m.volume, a, context.temp_allocator)
		// log return, RSI, volume z-score and volume ratio
		trading, n_tf := mlf.compute_trading_features(close, volume, context.temp_allocator)
		fd := mlf.frac_diff_ffd(log_price[a], ds.frac_d, 1e-3, context.temp_allocator)
		for d in 0 ..< T {
			f := raw[(d * N_ASSETS + a) * N_FEATURES:][:N_FEATURES]
			for k in 0 ..< n_tf {f[k] = trading[d * n_tf + k]}
			f[4], f[5], f[6] = math.nan_f64(), math.nan_f64(), math.nan_f64()
			if d >= 60 {f[5] = log_price[a][d] - log_price[a][d - 60]}
			if d >= 20 {
				f[4] = log_price[a][d] - log_price[a][d - 20]
				// standard deviation of the last 20 daily log returns
				s, s2 := 0.0, 0.0
				for k in d - 19 ..= d {
					r := log_price[a][k] - log_price[a][k - 1]
					s += r
					s2 += r * r
				}
				f[6] = math.sqrt(max(s2 / 20 - (s / 20) * (s / 20), 0))
			}
			f[7] = fd[d]
			for v in f {if math.is_nan(v) {first = max(first, d + 1)}}
		}
		free_all(context.temp_allocator)
	}
	// the volume features start after 20 days, RSI after 14
	ds.first_day = max(first, 21)

	// z-score across the sectors, day by day
	ds.x = make([]f64, T * N_ASSETS * N_FEATURES)
	for d in ds.first_day ..< T {
		for k in 0 ..< N_FEATURES {
			mean, sq := 0.0, 0.0
			for a in 0 ..< N_ASSETS {mean += raw[(d * N_ASSETS + a) * N_FEATURES + k] / N_ASSETS}
			for a in 0 ..< N_ASSETS {
				v := raw[(d * N_ASSETS + a) * N_FEATURES + k] - mean
				sq += v * v / N_ASSETS
			}
			sd := math.sqrt(sq)
			for a in 0 ..< N_ASSETS {
				i := (d * N_ASSETS + a) * N_FEATURES + k
				ds.x[i] = (raw[i] - mean) / sd if sd > 0 else 0
			}
		}
	}

	// targets: next-HORIZON-day log return relative to the sector average, in %
	for d in 0 ..< T {
		if d + HORIZON >= T {
			for a in 0 ..< N_ASSETS {ds.target[d * N_ASSETS + a] = math.nan_f64()}
			continue
		}
		mean := 0.0
		for a in 0 ..< N_ASSETS {mean += (log_price[a][d + HORIZON] - log_price[a][d]) / N_ASSETS}
		for a in 0 ..< N_ASSETS {
			ds.target[d * N_ASSETS + a] =
				100 * (log_price[a][d + HORIZON] - log_price[a][d] - mean)
		}
	}
	return
}

// window_into copies the WINDOW days of features ending at day d for asset a
// into dst [WINDOW · N_FEATURES] (oldest day first)
window_into :: proc(ds: ^Dataset, d, a: int, dst: []f64) {
	for k in 0 ..< WINDOW {
		day := d - WINDOW + 1 + k
		copy(
			dst[k * N_FEATURES:][:N_FEATURES],
			ds.x[(day * N_ASSETS + a) * N_FEATURES:][:N_FEATURES],
		)
	}
}

print_dataset :: proc(m: ^Market, ds: ^Dataset) {
	fmt.printfln(
		"  %d features per sector and day, from %s on (day %d); fractional differencing d = %.2f",
		N_FEATURES,
		date_string(m.dates[ds.first_day]),
		ds.first_day,
		ds.frac_d,
	)
}
