package main

// Stage 5b: value at risk. The one-day 99% VaR of a long SPY position, from
// three sources, each known at the previous close:
// - GJR-GARCH with Student-t errors (the walk-forward fits of stage 2);
// - the VIX: a normal quantile times the implied daily volatility;
// - historical simulation: the 1% quantile of the last 250 returns.
// A breach is a day whose loss exceeds the VaR. With 99% there should be one
// breach in a hundred days; Kupiec's test asks whether the count is consistent
// with that.

import fin "../../wotan/finance"
import "core:fmt"
import "core:math"

VaR_Model :: enum {
	GARCH,
	VIX,
	Historical,
}
VAR_NAMES := [VaR_Model]string {
	.GARCH      = "GJR-GARCH-t",
	.VIX        = "VIX (normal)",
	.Historical = "Historical 250d",
}

VaR_Result :: struct {
	series:     [VaR_Model][]f64, // VaR of the return of day d (a loss fraction), test days
	backtest:   [VaR_Model]fin.VaR_BacktestResult,
	turb_share: [VaR_Model]f64, // share of the breaches on days the regime model called turbulent
	returns:    []f64, // the test days' returns
	lo:         int, // the first test day
}

value_at_risk :: proc(m: ^Market, f: ^Forecasts) -> (v: VaR_Result) {
	v.lo = f.test_lo + 1 // the first return after the first forecast
	n := m.n_days - v.lo
	v.returns = m.ret[v.lo:]
	hist := fin.historical_var_rolling(m.ret, 0.99, 250) // hist[d]: from the 250 returns before d
	z := 2.3263478740408408 // the normal 99% quantile
	for model in VaR_Model {v.series[model] = make([]f64, n)}
	for i in 0 ..< n {
		d := v.lo + i
		v.series[.GARCH][i] = f.var99[d - 1]
		v.series[.VIX][i] = z * m.vix[d - 1] / 100 / math.sqrt(f64(TRADING_DAYS))
		v.series[.Historical][i] = hist[d]
	}
	for model in VaR_Model {
		v.backtest[model] = fin.backtest_var(v.returns, v.series[model], 0.99)
		turb := 0
		for r, i in v.returns {
			if r < -v.series[model][i] && f.p_turbulent[v.lo + i - 1] > 0.5 {turb += 1}
		}
		v.turb_share[model] = f64(turb) / f64(max(v.backtest[model].n_breaches, 1))
	}
	return
}

print_var :: proc(v: ^VaR_Result) {
	fmt.printfln(
		"  %-16s %9s %9s %8s %9s %18s",
		"",
		"breaches",
		"expected",
		"rate",
		"Kupiec p",
		"in turbulent regime",
	)
	for model in VaR_Model {
		b := v.backtest[model]
		fmt.printfln(
			"  %-16s %9d %9.1f %7.2f%% %9.3f %17.0f%%",
			VAR_NAMES[model],
			b.n_breaches,
			b.expected_breaches,
			100 * b.breach_rate,
			b.kupiec_pvalue,
			100 * v.turb_share[model],
		)
	}
}
