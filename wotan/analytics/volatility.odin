package analytics

import w "../core"
import "core:fmt"
import "core:mem"

// ============================================================================
// Unified Volatility Modeling Interface
// ============================================================================

VolatilityModel_Type :: enum {
	Rolling, // Simple rolling window variance
	EWMA, // Exponentially weighted moving average variance
	GARCH, // Parametric GARCH model
}

VolatilityModel :: struct {
	model_type:      VolatilityModel_Type,
	// For Rolling/EWMA
	window:          int,
	alpha:           f64,
	// For GARCH - store by value, not pointer
	garch_result:    GARCH_Result,
	// Common
	conditional_var: []f64,
}

// Create rolling variance model
create_rolling_volatility :: proc(
	df: ^w.DataFrame,
	column: string,
	window: int,
	allocator: mem.Allocator = context.allocator,
) -> VolatilityModel {
	rw := rolling_window(df, column, window, window)
	agg := w.make_var("var", column)
	var_col := rolling_apply(rw, agg, allocator)

	// Extract variance values
	n := var_col.len
	cond_var := make([]f64, n, allocator)
	for i in 0 ..< n {
		v, is_null := w.column_at_float(&var_col, i)
		if !is_null {
			cond_var[i] = v
		}
	}

	w.destroy_column(&var_col)

	return VolatilityModel{model_type = .Rolling, window = window, conditional_var = cond_var}
}

// Create EWMA variance model
create_ewma_volatility :: proc(
	df: ^w.DataFrame,
	column: string,
	alpha: f64,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> VolatilityModel {
	rw := rolling_window(df, column, 0, min_periods)
	agg := make_ewm_var("ewm_var", column, alpha, false)
	var_col := rolling_apply_float_ewm_var(rw, agg, allocator)

	n := var_col.len
	cond_var := make([]f64, n, allocator)
	for i in 0 ..< n {
		v, is_null := w.column_at_float(&var_col, i)
		if !is_null {
			cond_var[i] = v
		}
	}

	w.destroy_column(&var_col)

	return VolatilityModel{model_type = .EWMA, alpha = alpha, conditional_var = cond_var}
}

// Create GARCH model - returns by value
create_garch_volatility :: proc(
	series: []f64,
	p: int = 1,
	q: int = 1,
	allocator: mem.Allocator = context.allocator,
) -> VolatilityModel {
	residuals := extract_residuals(series, allocator)
	defer delete(residuals, allocator)

	// Fit GARCH and return result directly (by value)
	result := garch_fit(residuals, .GARCH, p, q, 1000, 1e-6, allocator)

	return VolatilityModel {
		model_type = .GARCH,
		garch_result = result,
		conditional_var = result.conditional_var,
	}
}

// Cleanup function for GARCH volatility model
destroy_garch_volatility :: proc(
	model: ^VolatilityModel,
	allocator: mem.Allocator = context.allocator,
) {
	if model.model_type == .GARCH {
		// Free GARCH result internals
		delete(model.garch_result.params.alpha, allocator)
		delete(model.garch_result.params.beta, allocator)
		delete(model.garch_result.conditional_var, allocator)
		delete(model.garch_result.standardized_resid, allocator)
	}
	if model.conditional_var != nil {
		delete(model.conditional_var, allocator)
	}
}

// Compare volatility models
compare_volatility_models :: proc(rolling_var: []f64, ewma_var: []f64, garch_var: []f64) {
	n := min(len(rolling_var), min(len(ewma_var), len(garch_var)))

	fmt.println("\n=== Volatility Model Comparison (Last 10 observations) ===")
	fmt.printf("%-10s %-15s %-15s %-15s\n", "Time", "Rolling", "EWMA", "GARCH")

	for i in max(0, n - 10) ..< n {
		fmt.printf("%-10d %-15.6f %-15.6f %-15.6f\n", i, rolling_var[i], ewma_var[i], garch_var[i])
	}
}
