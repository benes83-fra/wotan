package analytics


import w "../core"
import "core:mem"

StationarityDecision :: enum {
	Stationary,
	DifferenceStationary,
	TrendStationary,
	Inconclusive,
}

stationarity_test :: proc(
	y: []f64,
	adf_lags: int = 10,
	adf_reg: RegressionType = .Constant,
	adf_lag_sel: LagSelection = .AIC,
	kpss_type: KPSS_Type = .Level,
	allocator: mem.Allocator = context.allocator,
) -> (
	decision: StationarityDecision,
	adf_stat: f64,
	adf_p: f64,
	kpss_stat: f64,
	kpss_p: f64,
) {

	// --- Run ADF ---
	adf_stat, adf_p, _, _, _, _, _ = adf_test(y, adf_lags, adf_reg, adf_lag_sel, allocator)

	// --- Run KPSS ---
	kpss_stat, kpss_p, _, _, _, _ = kpss_test(y, kpss_type, -1, allocator)

	// --- Decision matrix ---
	adf_reject := adf_p < 0.05
	kpss_reject := kpss_p < 0.05

	if adf_reject && !kpss_reject {
		decision = .Stationary
	} else if !adf_reject && kpss_reject {
		decision = .DifferenceStationary
	} else if adf_reject && kpss_reject {
		decision = .TrendStationary
	} else {
		decision = .Inconclusive
	}

	return
}


df_stationarity :: proc(
	y: []f64,
	adf_lags: int = 10,
	adf_reg: RegressionType = .Constant,
	adf_lag_sel: LagSelection = .AIC,
	kpss_type: KPSS_Type = .Level,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	decision, adf_stat, adf_p, kpss_stat, kpss_p := stationarity_test(
		y,
		adf_lags,
		adf_reg,
		adf_lag_sel,
		kpss_type,
		allocator,
	)

	df := w.dataframe_new()

	// Compute decision string BEFORE array literal
	decision_str := ""
	switch decision {
	case .Stationary:
		decision_str = "Stationary"
	case .DifferenceStationary:
		decision_str = "Difference-Stationary"
	case .TrendStationary:
		decision_str = "Trend-Stationary"
	case .Inconclusive:
		decision_str = "Inconclusive"
	}

	w.add_column(&df, w.column_from_strings("decision", []string{decision_str}))
	w.add_column(&df, w.column_from_floats("adf_stat", []f64{adf_stat}))
	w.add_column(&df, w.column_from_floats("adf_p", []f64{adf_p}))
	w.add_column(&df, w.column_from_floats("kpss_stat", []f64{kpss_stat}))
	w.add_column(&df, w.column_from_floats("kpss_p", []f64{kpss_p}))

	df.rows = 1
	return df
}
