package main

// Stage 4: walk-forward forecasts and their evaluation. Every RETRAIN_EVERY
// days every model is fitted again on all the days before, and forecasts the
// next RETRAIN_EVERY days, which it has never seen. The forecasts are
// compared with the volatility that followed: by RMSE, by QLIKE (the loss
// that ranks variance forecasts correctly even though realised volatility is
// only a noisy measure of the true one, Patton 2011), and by Diebold-Mariano
// tests against GARCH.

import ml "../../wotan/analytics/ML"
import l "../../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:time"

TRAIN_MIN_DAYS :: 2520 // ten years of history before the first forecast
RETRAIN_EVERY :: 504 // new models every two years

Model :: enum {
	VIX,
	GARCH,
	HAR,
	LSTM,
	Mamba,
}
MODEL_NAMES := [Model]string {
	.VIX   = "VIX",
	.GARCH = "GJR-GARCH",
	.HAR   = "HAR + VIX",
	.LSTM  = "LSTM",
	.Mamba = "Mamba",
}

Forecasts :: struct {
	vol:              [Model][]f64, // annualised % over the next HORIZON days; NaN before test_lo
	var99:            []f64, // GARCH's 99% VaR of the next day's return (a loss fraction)
	p_turbulent:      []f64, // filtered probability of the turbulent regime
	vrp_mean, vrp_sd: []f64, // VIX minus realised volatility on the training days of the fold
	test_lo:          int, // the first forecast day
	seconds:          [Model]f64, // fitting and forecasting, all folds
}

walk_forward :: proc(m: ^Market, on_gpu: bool) -> (f: Forecasts) {
	T := m.n_days
	nan_array :: proc(n: int) -> []f64 {
		a := make([]f64, n)
		for &v in a {v = math.nan_f64()}
		return a
	}
	for &v in f.vol {v = nan_array(T)}
	f.var99 = nan_array(T)
	f.p_turbulent = nan_array(T)
	f.vrp_mean = nan_array(T)
	f.vrp_sd = nan_array(T)
	f.test_lo = TRAIN_MIN_DAYS
	fold := 0
	for s := f.test_lo; s < T; s += RETRAIN_EVERY {
		e := min(s + RETRAIN_EVERY, T)
		for d in s ..< e {f.vol[.VIX][d] = m.vix[d]}

		tick := time.tick_now()
		g := garch_fold(m, s, e, &f)
		f.seconds[.GARCH] += time.duration_seconds(time.tick_lap_time(&tick))
		har_fold(m, s, e, &f)
		f.seconds[.HAR] += time.duration_seconds(time.tick_lap_time(&tick))
		lstm_loss := net_fold(m, .LSTM, s, e, on_gpu, u64(100 + fold), f.vol[.LSTM])
		f.seconds[.LSTM] += time.duration_seconds(time.tick_lap_time(&tick))
		mamba_loss := net_fold(m, .Mamba, s, e, on_gpu, u64(200 + fold), f.vol[.Mamba])
		f.seconds[.Mamba] += time.duration_seconds(time.tick_lap_time(&tick))
		r := regime_fold(m, s, e, &f)

		// the volatility risk premium on the training days, for the option signal
		mean, sq, n := 0.0, 0.0, 0
		for d in FIRST_FEATURE_DAY ..= s - HORIZON - 1 {
			v := m.vix[d] - m.rv_fwd[d]
			mean += v
			sq += v * v
			n += 1
		}
		mean /= f64(n)
		sd := math.sqrt(sq / f64(n) - mean * mean)
		for d in s ..< e {f.vrp_mean[d], f.vrp_sd[d] = mean, sd}

		fmt.printfln(
			"  fold %2d: %s .. %s | GARCH a %.3f b %.3f g %.3f nu %4.1f | regimes %4.1f%% / %4.1f%% | loss LSTM %.2f Mamba %.2f",
			fold + 1,
			date_string(m.dates[s]),
			date_string(m.dates[e - 1]),
			g.alpha,
			g.beta,
			g.gamma,
			g.nu,
			r.calm_vol,
			r.turbulent_vol,
			lstm_loss,
			mamba_loss,
		)
		free_all(context.temp_allocator)
		fold += 1
	}
	return
}

Accuracy :: struct {
	n:                    int, // forecast days evaluated
	rmse, bias:           f64, // volatility points
	qlike:                f64,
	rmse_calm, rmse_turb: f64, // RMSE on the days the filtered regime was calm / turbulent
	n_turb:               int,
	dm_t, dm_p:           f64, // Diebold-Mariano on QLIKE against GARCH (t < 0: better than GARCH)
}

// qlike: the loss of a variance forecast f² for the realised variance rv²
qlike :: proc(rv, f: f64) -> f64 {
	ratio := (rv * rv) / (f * f)
	return ratio - math.ln(ratio) - 1
}

// evaluate: every forecast day whose realised volatility is known
evaluate :: proc(m: ^Market, f: ^Forecasts) -> (acc: [Model]Accuracy) {
	lo, hi := f.test_lo, m.n_days - HORIZON // days lo ..< hi
	n := hi - lo
	for model in Model {
		a := &acc[model]
		se_calm, se_turb := 0.0, 0.0
		for d in lo ..< hi {
			rv, fc := m.rv_fwd[d], f.vol[model][d]
			err := fc - rv
			a.rmse += err * err
			a.bias += err
			a.qlike += qlike(rv, fc)
			if f.p_turbulent[d] > 0.5 {
				se_turb += err * err
				a.n_turb += 1
			} else {
				se_calm += err * err
			}
		}
		a.n = n
		a.rmse = math.sqrt(a.rmse / f64(n))
		a.bias /= f64(n)
		a.qlike /= f64(n)
		a.rmse_calm = math.sqrt(se_calm / f64(n - a.n_turb))
		a.rmse_turb = math.sqrt(se_turb / f64(a.n_turb))
		a.dm_t, a.dm_p = math.nan_f64(), math.nan_f64()
	}
	// Diebold-Mariano: regress the loss difference on a constant. The forecasts
	// overlap (each covers the next HORIZON days), so the differences are
	// autocorrelated: Newey-West standard errors with HORIZON - 1 lags.
	ones := l.matrix_new(f64, n, 1, context.temp_allocator)
	for &v in ones.data {v = 1}
	diff := make([]f64, n, context.temp_allocator)
	for model in Model {
		if model == .GARCH {continue}
		for d in lo ..< hi {
			rv := m.rv_fwd[d]
			diff[d - lo] = qlike(rv, f.vol[model][d]) - qlike(rv, f.vol[.GARCH][d])
		}
		res := ml.ols_fit_robust(
			&ones,
			diff,
			{type = .HAC, maxlags = HORIZON - 1},
			allocator = context.temp_allocator,
		)
		acc[model].dm_t, acc[model].dm_p = res.tvalues[0], res.pvalues[0]
		ml.ols_result_free(&res, context.temp_allocator)
	}
	return
}

print_accuracy :: proc(acc: [Model]Accuracy, f: ^Forecasts) {
	fmt.printfln(
		"  %-10s %6s %6s %7s %10s %10s %8s %8s %8s",
		"",
		"RMSE",
		"bias",
		"QLIKE",
		"RMSE calm",
		"RMSE turb.",
		"DM t",
		"DM p",
		"time",
	)
	for model in Model {
		a := acc[model]
		fmt.printf(
			"  %-10s %6.2f %+6.2f %7.4f %10.2f %10.2f",
			MODEL_NAMES[model],
			a.rmse,
			a.bias,
			a.qlike,
			a.rmse_calm,
			a.rmse_turb,
		)
		if math.is_nan(a.dm_t) {
			fmt.printf(" %8s %8s", "-", "-")
		} else {
			fmt.printf(" %8.2f %8.3f", a.dm_t, a.dm_p)
		}
		if model == .VIX {
			fmt.println("        -")
		} else {
			fmt.printfln(" %7.1fs", f.seconds[model])
		}
	}
}
