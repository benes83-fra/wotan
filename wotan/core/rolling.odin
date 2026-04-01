package core


import "core:math"
import "core:mem"
import "core:slice"

RollingWindow :: struct {
	df:          ^DataFrame,
	column:      string,
	window:      int,
	min_periods: int,
	centered:    bool,
}


rolling_window :: proc(
	df: ^DataFrame,
	column: string,
	window: int,
	min_periods: int,
) -> RollingWindow {
	return RollingWindow{df = df, column = column, window = window, min_periods = min_periods}
}


rolling_apply :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> Column {
	src := column(r.df, r.column)
	switch src.type {
	case .Int:
		return rolling_apply_int(r, agg, allocator)
	case .Float:
		return rolling_apply_float(r, agg, allocator)
	case .Bool:
		return rolling_apply_bool(r, agg, allocator)
	case .Date:
		return rolling_apply_date(r, agg, allocator)
	case .Time:
		return rolling_apply_time(r, agg, allocator)
	case .Datetime:
		return rolling_apply_datetime(r, agg, allocator)
	case .String:
		return rolling_apply_string(r, agg, allocator)
	case .Invalid:
		panic("Invalid column type for rolling apply")
	}
	return column_new("invalid", .Invalid, 0)
}


rolling_apply_int :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]int, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_int(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_int_into(&out, values[:], agg)
	}
	if agg.kind == .EWM_Mean {
		return rolling_apply_int_ewm(r, agg, allocator)
	}
	if agg.kind == .EWM_Var {
		return rolling_apply_int_ewm_var(r, agg, allocator)
	}
	return out
}

rolling_apply_float :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]f64, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_float(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.window {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_float_into(&out, values[:], agg)
	}
	if agg.kind == .EWM_Mean {
		return rolling_apply_float_ewm(r, agg, allocator)
	}
	if agg.kind == .EWM_Var {
		return rolling_apply_float_ewm_var(r, agg, allocator)
	}
	return out
}


rolling_apply_bool :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]bool, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_bool(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_bool_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_date :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]Date, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_date(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_date_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_time :: proc(r: RollingWindow, agg: Aggregator, allocator: mem.Allocator) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]Time, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_time(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_time_into(&out, values[:], agg)
	}
	return out
}
rolling_apply_datetime :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]Datetime, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_datetime(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_datetime_into(&out, values[:], agg)
	}
	return out
}
rolling_apply_string :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, src.type, 0)

	values := make([dynamic]string, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		// Expand window
		v, is_null := column_at_string(src, i)
		if !is_null do append(&values, v)

		// Shrink window if too large
		if len(values) > r.min_periods {
			ordered_remove(&values, 0)
		}
		// Compute aggregation
		rolling_agg_string_into(&out, values[:], agg)
	}
	return out
}


rolling_apply_many :: proc(
	r: RollingWindow,
	aggs: []Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	out := dataframe_new()

	for agg in aggs {
		col := rolling_apply(r, agg, allocator)
		add_column(&out, col)
	}

	out.rows = out.columns[0].len
	return out
}

rolling_agg_int_into :: proc(out: ^Column, values: []int, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Sum:
		total := 0
		for v in values do total += v
		append_int(out, total)

	case .Mean:
		total := 0
		for v in values do total += v
		append_int(out, total / len(values))

	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		append_int(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		append_int(out, m)

	case .Count:
		append_int(out, len(values))

	case .Median, .Quantile:
		tmp := make([dynamic]int, len(values))
		copy(tmp[:], values)
		slice.sort(tmp[:])

		q := agg.quantile
		if kind == .Median do q = 0.5

		idx := int(q * f64(len(tmp) - 1))
		append_int(out, tmp[idx])
		delete(tmp)
	case .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
	// Does not apply to in
	}
}


rolling_agg_float_into :: proc(out: ^Column, values: []f64, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Sum:
		total: f64 = 0
		for v in values do total += v
		append_float(out, total)

	case .Mean:
		total: f64 = 0
		for v in values do total += v
		append_float(out, total / f64(len(values)))

	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		append_float(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		append_float(out, m)

	case .Count:
		append_float(out, f64(len(values)))

	case .Median, .Quantile:
		tmp := make([dynamic]f64, len(values))
		copy(tmp[:], values)
		slice.sort(tmp[:])

		q := agg.quantile
		if kind == .Median do q = 0.5

		idx := int(q * f64(len(tmp) - 1))
		append_float(out, tmp[idx])
		delete(tmp)
	case .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
	//Handled by rolling_apply_float_ewn
	}
}


rolling_agg_bool_into :: proc(out: ^Column, values: []bool, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	count_true := 0
	for v in values do if v do count_true += 1

	switch kind {
	case .Sum:
		// Sum = any true?
		append_bool(out, count_true > 0)

	case .Mean:
		// Mean = ratio of true
		append_float(out, f64(count_true) / f64(len(values)))

	case .Min:
		// Min = false if any false exists
		m := true
		for v in values do if !v do m = false
		append_bool(out, m)

	case .Max:
		// Max = true if any true exists
		m := false
		for v in values do if v do m = true
		append_bool(out, m)

	case .Count:
		append_int(out, len(values))

	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
		append_null(out)
	}
}

rolling_agg_string_into :: proc(out: ^Column, values: []string, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		append_string(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		append_string(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
		append_null(out)
	}
}

rolling_agg_date_into :: proc(out: ^Column, values: []Date, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if date_less(v, m) do m = v
		append_date(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if date_less(m, v) do m = v
		append_date(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
		append_null(out)
	}
}

rolling_agg_time_into :: proc(out: ^Column, values: []Time, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if time_less(v, m) do m = v
		append_time(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if time_less(m, v) do m = v
		append_time(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
		append_null(out)
	}
}

rolling_agg_datetime_into :: proc(out: ^Column, values: []Datetime, agg: Aggregator) {
	if len(values) == 0 {
		append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if datetime_less(v, m) do m = v
		append_datetime(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if datetime_less(m, v) do m = v
		append_datetime(out, m)

	case .Count:
		append_int(out, len(values))

	case .Sum, .Mean, .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov:
		append_null(out)
	}
}


make_ewm_mean :: proc(name, column: string, alpha: f64) -> Aggregator {
	return Aggregator{name = name, column = column, kind = .EWM_Mean, alpha = alpha}
}
make_ewm_var :: proc(name, column: string, alpha: f64) -> Aggregator {
	return Aggregator{name = name, column = column, kind = .EWM_Var, alpha = alpha}
}


rolling_apply_float_ewm :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	prev: f64
	has_prev := false

	for i in 0 ..< r.df.rows {
		v, is_null := column_at_float(src, i)
		if is_null {
			append_null(&out)
			continue
		}

		if !has_prev {
			prev = v
			has_prev = true
		} else {
			prev = alpha * v + (1 - alpha) * prev
		}

		append_float(&out, prev)
	}

	return out
}

rolling_apply_int_ewm :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	prev: f64
	has_prev := false

	for i in 0 ..< r.df.rows {
		v, is_null := column_at_int(src, i)
		if is_null {
			append_null(&out)
			continue
		}

		fv := f64(v)

		if !has_prev {
			prev = fv
			has_prev = true
		} else {
			prev = alpha * fv + (1 - alpha) * prev
		}

		append_float(&out, prev)
	}

	return out
}


rolling_apply_float_ewm_var :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := alpha // adjust = False
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int

	initialized := false

	for i in 0 ..< r.df.rows {
		v, is_null := column_at_float(src, i)
		if is_null {
			append_null(&out)
			continue
		}

		if !initialized {
			// first observation
			mean = v
			cov = 0.0
			sum_wt = 1.0
			sum_wt2 = 1.0
			old_wt = 1.0
			nobs = 1
			initialized = true

			// pandas: first var is NaN for bias=False
			if nobs >= minp {
				append_null(&out)
			} else {
				append_null(&out)
			}
			continue
		}

		nobs += 1

		// decay previous weights
		sum_wt *= old_wt_factor
		sum_wt2 *= (old_wt_factor * old_wt_factor)
		old_wt *= old_wt_factor

		old_mean := mean

		// avoid numerical errors on constant series
		if mean != v {
			mean = (old_wt * old_mean + new_wt * v) / (old_wt + new_wt)
		}

		// update covariance (here: variance, x == y)
		cov =
			(old_wt * (cov + (old_mean - mean) * (old_mean - mean)) +
				new_wt * (v - mean) * (v - mean)) /
			(old_wt + new_wt)

		// update weights with new observation
		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		// adjust = False branch: renormalize
		sum_wt /= old_wt
		sum_wt2 /= (old_wt * old_wt)
		old_wt = 1.0

		if nobs >= minp {
			// bias = False: apply debiasing factor
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, (num / den) * cov)
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}


rolling_apply_int_ewm_var :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := alpha // adjust = False
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int

	initialized := false

	for i in 0 ..< r.df.rows {
		v, is_null := column_at_int(src, i)
		if is_null {
			append_null(&out)
			continue
		}
		fv := f64(v)

		if !initialized {
			mean = fv
			cov = 0.0
			sum_wt = 1.0
			sum_wt2 = 1.0
			old_wt = 1.0
			nobs = 1
			initialized = true

			if nobs >= minp {
				append_null(&out)
			} else {
				append_null(&out)
			}
			continue
		}

		nobs += 1

		sum_wt *= old_wt_factor
		sum_wt2 *= (old_wt_factor * old_wt_factor)
		old_wt *= old_wt_factor

		old_mean := mean

		if mean != fv {
			mean = (old_wt * old_mean + new_wt * fv) / (old_wt + new_wt)
		}

		cov =
			(old_wt * (cov + (old_mean - mean) * (old_mean - mean)) +
				new_wt * (fv - mean) * (fv - mean)) /
			(old_wt + new_wt)

		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		sum_wt /= old_wt
		sum_wt2 /= (old_wt * old_wt)
		old_wt = 1.0

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, (num / den) * cov)
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}

rolling_apply_float_ewm_std :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	// compute variance first
	var_col := rolling_apply_float_ewm_var(r, agg, allocator)
	out := column_new(agg.name, .Float, 0)

	for i in 0 ..< var_col.len {
		v, is_null := column_at_float(&var_col, i)
		if is_null {
			append_null(&out)
		} else {
			append_float(&out, math.sqrt(v))
		}
	}

	destroy_column(&var_col)
	return out
}

rolling_apply_int_ewm_std :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	var_col := rolling_apply_int_ewm_var(r, agg, allocator)
	out := column_new(agg.name, .Float, 0)

	for i in 0 ..< var_col.len {
		v, is_null := column_at_float(&var_col, i)
		if is_null {
			append_null(&out)
		} else {
			append_float(&out, math.sqrt(v))
		}
	}

	destroy_column(&var_col)
	return out
}

rolling_apply_float_ewm_cov :: proc(
	r: RollingWindow,
	other: ^Column,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := alpha
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean_x: f64
	mean_y: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int
	initialized := false

	for i in 0 ..< r.df.rows {
		x, nx := column_at_float(src, i)
		y, ny := column_at_float(other, i)

		if nx || ny {
			append_null(&out)
			continue
		}

		if !initialized {
			mean_x = x
			mean_y = y
			cov = 0
			sum_wt = 1
			sum_wt2 = 1
			old_wt = 1
			nobs = 1
			initialized = true

			append_null(&out)
			continue
		}

		nobs += 1

		sum_wt *= old_wt_factor
		sum_wt2 *= old_wt_factor * old_wt_factor
		old_wt *= old_wt_factor

		old_mean_x := mean_x
		old_mean_y := mean_y

		if mean_x != x {
			mean_x = (old_wt * old_mean_x + new_wt * x) / (old_wt + new_wt)
		}
		if mean_y != y {
			mean_y = (old_wt * old_mean_y + new_wt * y) / (old_wt + new_wt)
		}

		cov =
			(old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y)) +
				new_wt * (x - mean_x) * (y - mean_y)) /
			(old_wt + new_wt)

		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		sum_wt /= old_wt
		sum_wt2 /= old_wt * old_wt
		old_wt = 1

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, cov * (num / den))
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}


rolling_apply_int_ewm_cov :: proc(
	r: RollingWindow,
	other: ^Column,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := alpha
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean_x: f64
	mean_y: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int
	initialized := false

	for i in 0 ..< r.df.rows {
		xi, nx := column_at_int(src, i)
		yi, ny := column_at_int(other, i)

		if nx || ny {
			append_null(&out)
			continue
		}

		x := f64(xi)
		y := f64(yi)

		if !initialized {
			mean_x = x
			mean_y = y
			cov = 0
			sum_wt = 1
			sum_wt2 = 1
			old_wt = 1
			nobs = 1
			initialized = true

			append_null(&out)
			continue
		}

		nobs += 1

		// decay weights
		sum_wt *= old_wt_factor
		sum_wt2 *= old_wt_factor * old_wt_factor
		old_wt *= old_wt_factor

		old_mean_x := mean_x
		old_mean_y := mean_y

		// update means
		if mean_x != x {
			mean_x = (old_wt * old_mean_x + new_wt * x) / (old_wt + new_wt)
		}
		if mean_y != y {
			mean_y = (old_wt * old_mean_y + new_wt * y) / (old_wt + new_wt)
		}

		// update covariance
		cov =
			(old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y)) +
				new_wt * (x - mean_x) * (y - mean_y)) /
			(old_wt + new_wt)

		// update weights
		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		// adjust=False renormalization
		sum_wt /= old_wt
		sum_wt2 /= (old_wt * old_wt)
		old_wt = 1

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, cov * (num / den))
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}


rolling_apply_float_ewm_var_adjust_true :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := 1.0 // adjust=True
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int
	initialized := false

	for i in 0 ..< r.df.rows {
		v, is_null := column_at_float(src, i)
		if is_null {
			append_null(&out)
			continue
		}

		if !initialized {
			mean = v
			cov = 0
			sum_wt = 1
			sum_wt2 = 1
			old_wt = 1
			nobs = 1
			initialized = true

			append_null(&out)
			continue
		}

		nobs += 1

		// decay previous weights
		sum_wt *= old_wt_factor
		sum_wt2 *= old_wt_factor * old_wt_factor
		old_wt *= old_wt_factor

		old_mean := mean

		// update mean
		if mean != v {
			mean = (old_wt * old_mean + new_wt * v) / (old_wt + new_wt)
		}

		// update covariance
		cov =
			(old_wt * (cov + (old_mean - mean) * (old_mean - mean)) +
				new_wt * (v - mean) * (v - mean)) /
			(old_wt + new_wt)

		// update weights
		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, cov * (num / den))
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}

rolling_apply_int_ewm_var_adjust_true :: proc(
	r: RollingWindow,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := 1.0 // adjust=True
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int
	initialized := false

	for i in 0 ..< r.df.rows {
		vi, is_null := column_at_int(src, i)
		if is_null {
			append_null(&out)
			continue
		}
		v := f64(vi)

		if !initialized {
			mean = v
			cov = 0
			sum_wt = 1
			sum_wt2 = 1
			old_wt = 1
			nobs = 1
			initialized = true

			append_null(&out)
			continue
		}

		nobs += 1

		// decay previous weights
		sum_wt *= old_wt_factor
		sum_wt2 *= old_wt_factor * old_wt_factor
		old_wt *= old_wt_factor

		old_mean := mean

		// update mean
		if mean != v {
			mean = (old_wt * old_mean + new_wt * v) / (old_wt + new_wt)
		}

		// update covariance
		cov =
			(old_wt * (cov + (old_mean - mean) * (old_mean - mean)) +
				new_wt * (v - mean) * (v - mean)) /
			(old_wt + new_wt)

		// update weights
		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, cov * (num / den))
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}


rolling_apply_float_ewm_cov_adjust_false :: proc(
	r: RollingWindow,
	other: ^Column,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := alpha
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean_x: f64
	mean_y: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int
	initialized := false

	for i in 0 ..< r.df.rows {
		x, nx := column_at_float(src, i)
		y, ny := column_at_float(other, i)

		if nx || ny {
			append_null(&out)
			continue
		}

		if !initialized {
			mean_x = x
			mean_y = y
			cov = 0
			sum_wt = 1
			sum_wt2 = 1
			old_wt = 1
			nobs = 1
			initialized = true

			append_null(&out)
			continue
		}

		nobs += 1

		// decay weights
		sum_wt *= old_wt_factor
		sum_wt2 *= old_wt_factor * old_wt_factor
		old_wt *= old_wt_factor

		old_mean_x := mean_x
		old_mean_y := mean_y

		// update means
		if mean_x != x {
			mean_x = (old_wt * old_mean_x + new_wt * x) / (old_wt + new_wt)
		}
		if mean_y != y {
			mean_y = (old_wt * old_mean_y + new_wt * y) / (old_wt + new_wt)
		}

		// update covariance
		cov =
			(old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y)) +
				new_wt * (x - mean_x) * (y - mean_y)) /
			(old_wt + new_wt)

		// update weights
		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		// adjust=False renormalization
		sum_wt /= old_wt
		sum_wt2 /= (old_wt * old_wt)
		old_wt = 1

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, cov * (num / den))
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}
rolling_apply_int_ewm_cov_adjust_false :: proc(
	r: RollingWindow,
	other: ^Column,
	agg: Aggregator,
	allocator: mem.Allocator,
) -> Column {
	src := column(r.df, r.column)
	out := column_new(agg.name, .Float, 0)

	alpha := agg.alpha
	if alpha <= 0 || alpha > 1 {
		panic("EWMA alpha must be in (0,1]")
	}

	old_wt_factor := 1.0 - alpha
	new_wt := alpha
	minp := r.min_periods
	if minp < 1 {
		minp = 1
	}

	mean_x: f64
	mean_y: f64
	cov: f64
	sum_wt: f64
	sum_wt2: f64
	old_wt: f64
	nobs: int
	initialized := false

	for i in 0 ..< r.df.rows {
		xi, nx := column_at_int(src, i)
		yi, ny := column_at_int(other, i)

		if nx || ny {
			append_null(&out)
			continue
		}

		x := f64(xi)
		y := f64(yi)

		if !initialized {
			mean_x = x
			mean_y = y
			cov = 0
			sum_wt = 1
			sum_wt2 = 1
			old_wt = 1
			nobs = 1
			initialized = true

			append_null(&out)
			continue
		}

		nobs += 1

		// decay previous weights
		sum_wt *= old_wt_factor
		sum_wt2 *= old_wt_factor * old_wt_factor
		old_wt *= old_wt_factor

		old_mean_x := mean_x
		old_mean_y := mean_y

		// update means
		if mean_x != x {
			mean_x = (old_wt * old_mean_x + new_wt * x) / (old_wt + new_wt)
		}
		if mean_y != y {
			mean_y = (old_wt * old_mean_y + new_wt * y) / (old_wt + new_wt)
		}

		// update covariance
		cov =
			(old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y)) +
				new_wt * (x - mean_x) * (y - mean_y)) /
			(old_wt + new_wt)

		// update weights
		sum_wt += new_wt
		sum_wt2 += new_wt * new_wt
		old_wt += new_wt

		// adjust=False renormalization
		sum_wt /= old_wt
		sum_wt2 /= (old_wt * old_wt)
		old_wt = 1

		if nobs >= minp {
			num := sum_wt * sum_wt
			den := num - sum_wt2
			if den > 0 {
				append_float(&out, cov * (num / den))
			} else {
				append_null(&out)
			}
		} else {
			append_null(&out)
		}
	}

	return out
}
