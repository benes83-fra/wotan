package analytics


import w "../core"
import "core:math"
import "core:mem"
import "core:slice"

// Rolling and exponentially weighted window statistics, matching pandas 1.5
// (Series.rolling / Series.ewm) except where noted:
//
// - The window counts ROWS (nulls included), like pandas: the window for row i
//   is rows [i - window + 1, i], or, when `centered`, rows
//   [i - window/2, i + (window - 1)/2] (pandas center=True). Nulls and NaN
//   floats inside the window are skipped.
// - A row is null when its window holds fewer than max(min_periods, 1)
//   non-null values (Count: fewer than that many rows, as in pandas). (pandas lets min_periods = 0 return 0 for an empty Sum;
//   here an empty window is always null.)
// - Mean / Median / Quantile of an Int column return a Float column. Sum, Min,
//   Max and Count of an Int column stay Int (exact); Count of a Float column
//   is Float. pandas returns float for all of them.
// - Quantile uses linear interpolation between the two nearest ranks
//   (pandas' default interpolation="linear").
// - Var / Cov use the sample estimator (ddof = 1, pandas' default); a window
//   with fewer than 2 complete pairs is null. Cov / Corr use only the rows
//   where both columns are non-null (pandas' pairwise alignment), and that
//   pair count is what min_periods is compared against.
// - Bool columns: Sum = number of trues (Int), Mean = fraction of trues,
//   Min / Max = all / any.
// - EWM: see ewm_mean_column / ewm_cov_column below. Through rolling_apply
//   (the Aggregator has no `adjust` field) all EWM kinds use adjust=False,
//   i.e. pandas ewm(alpha=..., adjust=False); call ewm_*_column for adjust=True.
//   EWM ignores `window` and `centered`.
//
// Note: rolling_apply reads the column named by RollingWindow.column;
// Aggregator.column is ignored (Aggregator.other names the second column of
// Cov / Corr / EWM_Cov / EWM_Corr).

RollingWindow :: struct {
	df:          ^w.DataFrame,
	column:      string,
	window:      int,
	min_periods: int,
	centered:    bool,
}


rolling_window :: proc(
	df: ^w.DataFrame,
	column: string,
	window: int,
	min_periods: int,
) -> RollingWindow {
	return RollingWindow{df = df, column = column, window = window, min_periods = min_periods}
}

// Rows [start, end) of the window labelled at row i (pandas FixedWindowIndexer)
rolling_bounds :: proc(r: RollingWindow, i: int) -> (start, end: int) {
	offset := (r.window - 1) / 2 if r.centered && r.window > 0 else 0
	last := i + 1 + offset
	start = clamp(last - r.window, 0, r.df.rows)
	end = clamp(last, 0, r.df.rows)
	return
}

// Minimum number of non-null values a window needs for a non-null result
@(private = "file")
_rolling_min_obs :: proc(r: RollingWindow) -> int {
	return max(r.min_periods, 1)
}

// Whether a window with n non-null values out of `rows` rows gets a result.
// pandas' rolling count compares min_periods with the number of ROWS in the
// window (it sums a not-null indicator), every other statistic with n.
@(private = "file")
_rolling_enough :: proc(r: RollingWindow, agg: w.Aggregator, n, rows: int) -> bool {
	return (rows if agg.kind == .Count else n) >= _rolling_min_obs(r)
}

// Numeric cell as f64; null for nulls, NaN floats and non-numeric columns
@(private = "file")
_numeric_at :: proc(c: ^w.Column, i: int) -> (f64, bool) {
	#partial switch c.type {
	case .Float:
		v, is_null := w.column_at_float(c, i)
		return v, is_null || math.is_nan(v)
	case .Int:
		v, is_null := w.column_at_int(c, i)
		return f64(v), is_null
	case .Bool:
		v, is_null := w.column_at_bool(c, i)
		return (1 if v else 0), is_null
	}
	return 0, true
}

// Linear interpolation between the two nearest ranks (pandas' default)
@(private = "file")
_quantile_linear :: proc(sorted: []f64, q: f64) -> f64 {
	q := q
	if math.is_nan(q) do q = 0.5
	q = clamp(q, 0, 1)
	pos := q * f64(len(sorted) - 1)
	lo := int(math.floor(pos))
	hi := min(lo + 1, len(sorted) - 1)
	return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - f64(lo))
}


rolling_apply :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	if r.window < 0 do panic("rolling_apply: window must be >= 0")
	src := w.column(r.df, r.column)
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
	return w.column_new("invalid", .Invalid, 0)
}


rolling_apply_int :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	#partial switch agg.kind {
	case .EWM_Mean:
		return rolling_apply_int_ewm(r, agg, allocator)
	case .EWM_Var:
		return rolling_apply_int_ewm_var(r, agg, allocator)
	case .EWM_Std:
		return rolling_apply_int_ewm_std(r, agg, allocator)
	case .EWM_Cov:
		other := w.column(r.df, agg.other)
		return rolling_apply_int_ewm_cov(r, other, agg, allocator)
	case .EWM_Corr:
		return rolling_apply_int_ewm_corr(r, agg, allocator)
	case .Corr:
		return rolling_apply_int_corr(r, agg, allocator)
	case .Var:
		return rolling_apply_int_var(r, agg, allocator)
	case .Cov:
		other := w.column(r.df, agg.other)
		return rolling_apply_int_cov(r, other, agg, allocator)
	}

	src := w.column(r.df, r.column)
	out_type := src.type
	#partial switch agg.kind {
	case .Mean, .Median, .Quantile:
		out_type = .Float
	}
	out := w.column_new(agg.name, out_type, 0)

	values := make([dynamic]int, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := w.column_at_int(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_int_into(&out, values[:], agg)
	}

	return out
}

rolling_apply_float :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	#partial switch agg.kind {
	case .EWM_Mean:
		return rolling_apply_float_ewm(r, agg, allocator)
	case .EWM_Var:
		return rolling_apply_float_ewm_var(r, agg, allocator)
	case .EWM_Std:
		return rolling_apply_float_ewm_std(r, agg, allocator)
	case .EWM_Cov:
		other := w.column(r.df, agg.other)
		return rolling_apply_float_ewm_cov(r, other, agg, allocator)
	case .EWM_Corr:
		return rolling_apply_float_ewm_corr(r, agg, allocator)
	case .Corr:
		return rolling_apply_float_corr(r, agg, allocator)
	case .Var:
		return rolling_apply_float_var(r, agg, allocator)
	case .Cov:
		other := w.column(r.df, agg.other)
		return rolling_apply_float_cov(r, other, agg, allocator)
	}

	src := w.column(r.df, r.column)
	out := w.column_new(agg.name, src.type, 0)

	values := make([dynamic]f64, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := _numeric_at(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_float_into(&out, values[:], agg)
	}

	return out
}


rolling_apply_bool :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	out_type := src.type
	if agg.kind == .Count || agg.kind == .Sum do out_type = .Int
	if agg.kind == .Mean do out_type = .Float // ratio of true
	out := w.column_new(agg.name, out_type, 0)

	values := make([dynamic]bool, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := w.column_at_bool(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_bool_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_date :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	out := w.column_new(agg.name, .Int if agg.kind == .Count else src.type, 0)

	values := make([dynamic]w.Date, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := w.column_at_date(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_date_into(&out, values[:], agg)
	}
	return out
}

rolling_apply_time :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	out := w.column_new(agg.name, .Int if agg.kind == .Count else src.type, 0)

	values := make([dynamic]w.Time, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := w.column_at_time(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_time_into(&out, values[:], agg)
	}
	return out
}
rolling_apply_datetime :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	out := w.column_new(agg.name, .Int if agg.kind == .Count else src.type, 0)

	values := make([dynamic]w.Datetime, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := w.column_at_datetime(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_datetime_into(&out, values[:], agg)
	}
	return out
}
rolling_apply_string :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	out := w.column_new(agg.name, .Int if agg.kind == .Count else src.type, 0)

	values := make([dynamic]string, 0, r.window, allocator)
	defer delete(values)

	for i in 0 ..< r.df.rows {
		clear(&values)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			v, is_null := w.column_at_string(src, j)
			if !is_null do append(&values, v)
		}
		if !_rolling_enough(r, agg, len(values), end - start) {
			w.append_null(&out)
			continue
		}
		rolling_agg_string_into(&out, values[:], agg)
	}
	return out
}


rolling_apply_many :: proc(
	r: RollingWindow,
	aggs: []w.Aggregator,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	out := w.dataframe_new()

	for agg in aggs {
		col := rolling_apply(r, agg, allocator)
		w.add_column(&out, col)
	}

	out.rows = out.columns[0].len if len(out.columns) > 0 else 0
	return out
}

// Mean / Median / Quantile append Float (out must be a Float column), the rest Int
rolling_agg_int_into :: proc(out: ^w.Column, values: []int, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Sum:
		total := 0
		for v in values do total += v
		w.append_int(out, total)

	case .Mean:
		total := 0
		for v in values do total += v
		w.append_float(out, f64(total) / f64(len(values)))

	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		w.append_int(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		w.append_int(out, m)

	case .Count:
		w.append_int(out, len(values))

	case .Median, .Quantile:
		tmp := make([]f64, len(values))
		defer delete(tmp)
		for v, i in values do tmp[i] = f64(v)
		slice.sort(tmp)
		w.append_float(out, _quantile_linear(tmp, 0.5 if kind == .Median else agg.quantile))
	case .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr, .Corr, .Var, .Cov:
	// Does not apply to in
	}
}


rolling_agg_float_into :: proc(out: ^w.Column, values: []f64, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Sum:
		total: f64 = 0
		for v in values do total += v
		w.append_float(out, total)

	case .Mean:
		total: f64 = 0
		for v in values do total += v
		w.append_float(out, total / f64(len(values)))

	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		w.append_float(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		w.append_float(out, m)

	case .Count:
		w.append_float(out, f64(len(values)))

	case .Median, .Quantile:
		tmp := make([]f64, len(values))
		defer delete(tmp)
		copy(tmp, values)
		slice.sort(tmp)
		w.append_float(out, _quantile_linear(tmp, 0.5 if kind == .Median else agg.quantile))
	case .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr, .Corr, .Var, .Cov:
	//Handled by rolling_apply_float_ewn
	}
}


// Sum and Count append Int, Mean appends Float, Min / Max append Bool
rolling_agg_bool_into :: proc(out: ^w.Column, values: []bool, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	count_true := 0
	for v in values do if v do count_true += 1

	switch kind {
	case .Sum:
		// Sum = number of trues (pandas / groupby2)
		w.append_int(out, count_true)

	case .Mean:
		// Mean = ratio of true
		w.append_float(out, f64(count_true) / f64(len(values)))

	case .Min:
		// Min = false if any false exists
		m := true
		for v in values do if !v do m = false
		w.append_bool(out, m)

	case .Max:
		// Max = true if any true exists
		m := false
		for v in values do if v do m = true
		w.append_bool(out, m)

	case .Count:
		w.append_int(out, len(values))

	case .Median, .Quantile, .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr, .Corr, .Var, .Cov:
		w.append_null(out)
	}
}

rolling_agg_string_into :: proc(out: ^w.Column, values: []string, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if v < m do m = v
		w.append_string(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if v > m do m = v
		w.append_string(out, m)

	case .Count:
		w.append_int(out, len(values))

	case .Sum,
	     .Mean,
	     .Median,
	     .Quantile,
	     .EWM_Mean,
	     .EWM_Var,
	     .EWM_Std,
	     .EWM_Cov,
	     .EWM_Corr,
	     .Corr,
	     .Var,
	     .Cov:
		w.append_null(out)
	}
}

rolling_agg_date_into :: proc(out: ^w.Column, values: []w.Date, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if w.date_less(v, m) do m = v
		w.append_date(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if w.date_less(m, v) do m = v
		w.append_date(out, m)

	case .Count:
		w.append_int(out, len(values))

	case .Sum,
	     .Mean,
	     .Median,
	     .Quantile,
	     .EWM_Mean,
	     .EWM_Var,
	     .EWM_Std,
	     .EWM_Cov,
	     .EWM_Corr,
	     .Corr,
	     .Var,
	     .Cov:
		w.append_null(out)
	}
}

rolling_agg_time_into :: proc(out: ^w.Column, values: []w.Time, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if w.time_less(v, m) do m = v
		w.append_time(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if w.time_less(m, v) do m = v
		w.append_time(out, m)

	case .Count:
		w.append_int(out, len(values))

	case .Sum,
	     .Mean,
	     .Median,
	     .Quantile,
	     .EWM_Mean,
	     .EWM_Var,
	     .EWM_Std,
	     .EWM_Cov,
	     .EWM_Corr,
	     .Corr,
	     .Var,
	     .Cov:
		w.append_null(out)
	}
}

rolling_agg_datetime_into :: proc(out: ^w.Column, values: []w.Datetime, agg: w.Aggregator) {
	if len(values) == 0 && agg.kind != .Count {
		w.append_null(out)
		return
	}

	kind := agg.kind

	switch kind {
	case .Min:
		m := values[0]
		for v in values[1:] do if w.datetime_less(v, m) do m = v
		w.append_datetime(out, m)

	case .Max:
		m := values[0]
		for v in values[1:] do if w.datetime_less(m, v) do m = v
		w.append_datetime(out, m)

	case .Count:
		w.append_int(out, len(values))

	case .Sum,
	     .Mean,
	     .Median,
	     .Quantile,
	     .EWM_Mean,
	     .EWM_Var,
	     .EWM_Std,
	     .EWM_Cov,
	     .EWM_Corr,
	     .Corr,
	     .Var,
	     .Cov:
		w.append_null(out)
	}
}


make_ewm_mean :: proc(name, column: string, alpha: f64) -> w.Aggregator {
	return w.Aggregator{name = name, column = column, kind = .EWM_Mean, alpha = alpha}
}
make_ewm_var :: proc(name, column: string, alpha: f64, bias: bool = false) -> w.Aggregator {
	return w.Aggregator{name = name, column = column, kind = .EWM_Var, alpha = alpha, bias = bias}
}
make_ewm_cov :: proc(name, column: string, alpha: f64, bias := false) -> w.Aggregator {
	return w.Aggregator{name = name, column = column, kind = .EWM_Cov, alpha = alpha, bias = bias}
}

make_ewm_corr :: proc(name, col, other: string, alpha: f64, bias := false) -> w.Aggregator {
	return w.Aggregator {
		name = name,
		column = col,
		other = other,
		kind = .EWM_Corr,
		alpha = alpha,
		bias = bias,
	}
}


// ============================================================================
// Exponentially weighted moments
// ============================================================================

@(private = "file")
_EWM_Moment :: enum {
	Mean,
	Cov,
	Corr,
}

// EWM mean / covariance / correlation of x and y, a transcription of pandas
// 1.5's `ewm` and `ewmcov` kernels (pandas/_libs/window/aggregations.pyx):
// - adjust=True weights the k-th most recent observation (1 - alpha)^k and
//   divides by the weight sum; adjust=False is the recursion
//   m_t = (1 - alpha) m_{t-1} + alpha x_t.
// - ignore_na=False (pandas' default) keeps decaying the old weights across
//   null rows (absolute positions); ignore_na=True skips null rows entirely.
// - bias=False rescales the weighted covariance by sum_wt^2 / (sum_wt^2 - sum_wt2)
//   (null while that is undefined, e.g. at the first observation).
// - Output is null until max(min_periods, 1) observations (complete x/y pairs)
//   have been seen; after that, null rows carry the current value, like pandas.
// - Corr uses the same pairwise-complete rows for cov(x, y), var(x), var(y).
@(private = "file")
_ewm_core :: proc(
	r: RollingWindow,
	x, y: ^w.Column,
	name: string,
	alpha: f64,
	adjust, bias, ignore_na: bool,
	moment: _EWM_Moment,
) -> w.Column {
	if !(alpha > 0 && alpha <= 1) {
		panic("EWMA alpha must be in (0,1]")
	}
	out := w.column_new(name, .Float, 0)

	minp := _rolling_min_obs(r)
	old_wt_factor := 1.0 - alpha
	new_wt := 1.0 if adjust else alpha

	mean_x, mean_y := math.nan_f64(), math.nan_f64()
	cov_xy, var_x, var_y := 0.0, 0.0, 0.0
	sum_wt, sum_wt2, old_wt := 1.0, 1.0, 1.0
	nobs := 0

	for i in 0 ..< r.df.rows {
		cur_x, nx := _numeric_at(x, i)
		cur_y, ny := _numeric_at(y, i)
		is_observation := !nx && !ny
		if is_observation do nobs += 1

		if !math.is_nan(mean_x) {
			if is_observation || !ignore_na {
				sum_wt *= old_wt_factor
				sum_wt2 *= old_wt_factor * old_wt_factor
				old_wt *= old_wt_factor
				if is_observation {
					old_mean_x, old_mean_y := mean_x, mean_y
					// avoid numerical errors on constant series
					if mean_x != cur_x {
						mean_x = (old_wt * old_mean_x + new_wt * cur_x) / (old_wt + new_wt)
					}
					if mean_y != cur_y {
						mean_y = (old_wt * old_mean_y + new_wt * cur_y) / (old_wt + new_wt)
					}
					dmx, dmy := old_mean_x - mean_x, old_mean_y - mean_y
					ex, ey := cur_x - mean_x, cur_y - mean_y
					cov_xy = (old_wt * (cov_xy + dmx * dmy) + new_wt * ex * ey) / (old_wt + new_wt)
					var_x = (old_wt * (var_x + dmx * dmx) + new_wt * ex * ex) / (old_wt + new_wt)
					var_y = (old_wt * (var_y + dmy * dmy) + new_wt * ey * ey) / (old_wt + new_wt)

					sum_wt += new_wt
					sum_wt2 += new_wt * new_wt
					old_wt += new_wt
					if !adjust {
						sum_wt /= old_wt
						sum_wt2 /= old_wt * old_wt
						old_wt = 1
					}
				}
			}
		} else if is_observation {
			mean_x, mean_y = cur_x, cur_y
		}

		if nobs < minp {
			w.append_null(&out)
			continue
		}
		switch moment {
		case .Mean:
			w.append_float(&out, mean_x)
		case .Cov:
			if bias {
				w.append_float(&out, cov_xy)
			} else {
				num := sum_wt * sum_wt
				den := num - sum_wt2
				if den > 0 {
					w.append_float(&out, (num / den) * cov_xy)
				} else {
					w.append_null(&out)
				}
			}
		case .Corr:
			// the bias correction cancels in the ratio
			d := var_x * var_y
			if d > 0 {
				w.append_float(&out, cov_xy / math.sqrt(d))
			} else {
				w.append_null(&out)
			}
		}
	}
	return out
}

// pandas: df[col].ewm(alpha=alpha, adjust=adjust, ignore_na=ignore_na,
// min_periods=r.min_periods).mean(). Int / Float / Bool columns.
ewm_mean_column :: proc(
	r: RollingWindow,
	alpha: f64,
	adjust: bool,
	ignore_na := false,
	name := "ewm_mean",
) -> w.Column {
	src := w.column(r.df, r.column)
	return _ewm_core(r, src, src, name, alpha, adjust, false, ignore_na, .Mean)
}

// pandas: df[col].ewm(...).cov(other, bias=bias); other == the source column
// gives ewm(...).var(bias=bias)
ewm_cov_column :: proc(
	r: RollingWindow,
	other: ^w.Column,
	alpha: f64,
	adjust: bool,
	bias: bool,
	ignore_na := false,
	name := "ewm_cov",
) -> w.Column {
	src := w.column(r.df, r.column)
	return _ewm_core(r, src, other, name, alpha, adjust, bias, ignore_na, .Cov)
}

// pandas: df[col].ewm(...).var(bias=bias)
ewm_var_column :: proc(
	r: RollingWindow,
	alpha: f64,
	adjust: bool,
	bias: bool,
	ignore_na := false,
	name := "ewm_var",
) -> w.Column {
	src := w.column(r.df, r.column)
	return _ewm_core(r, src, src, name, alpha, adjust, bias, ignore_na, .Cov)
}

// pandas: df[col].ewm(...).std(bias=bias) = sqrt(var)
ewm_std_column :: proc(
	r: RollingWindow,
	alpha: f64,
	adjust: bool,
	bias: bool,
	ignore_na := false,
	name := "ewm_std",
) -> w.Column {
	var_col := ewm_var_column(r, alpha, adjust, bias, ignore_na, name)
	defer w.destroy_column(&var_col)
	return _sqrt_column(&var_col, name)
}

// pandas: df[col].ewm(...).corr(other)
ewm_corr_column :: proc(
	r: RollingWindow,
	other: ^w.Column,
	alpha: f64,
	adjust: bool,
	ignore_na := false,
	name := "ewm_corr",
) -> w.Column {
	src := w.column(r.df, r.column)
	return _ewm_core(r, src, other, name, alpha, adjust, false, ignore_na, .Corr)
}

@(private = "file")
_sqrt_column :: proc(c: ^w.Column, name: string) -> w.Column {
	out := w.column_new(name, .Float, 0)
	for i in 0 ..< c.len {
		v, is_null := w.column_at_float(c, i)
		if is_null {
			w.append_null(&out)
		} else {
			w.append_float(&out, math.sqrt(v))
		}
	}
	return out
}

@(private = "file")
_ewm_other :: proc(r: RollingWindow, other: ^w.Column) -> ^w.Column {
	src := w.column(r.df, r.column)
	if other.type != src.type do panic("rolling ewm cov: `other` column must have the same type as the source column")
	return other
}

// The procs below are the Aggregator-driven entry points used by rolling_apply.
// Unless the name says adjust_true they are adjust=False (pandas
// ewm(alpha=agg.alpha, adjust=False)); all use ignore_na=False.

rolling_apply_float_ewm :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_mean_column(r, agg.alpha, false, false, agg.name)
}

rolling_apply_int_ewm :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_mean_column(r, agg.alpha, false, false, agg.name)
}


rolling_apply_float_ewm_var :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_var_column(r, agg.alpha, false, agg.bias, false, agg.name)
}


rolling_apply_int_ewm_var :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_var_column(r, agg.alpha, false, agg.bias, false, agg.name)
}

rolling_apply_float_ewm_std :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_std_column(r, agg.alpha, false, agg.bias, false, agg.name)
}

rolling_apply_int_ewm_std :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_std_column(r, agg.alpha, false, agg.bias, false, agg.name)
}

rolling_apply_float_ewm_cov :: proc(
	r: RollingWindow,
	other: ^w.Column,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_cov_column(r, _ewm_other(r, other), agg.alpha, false, agg.bias, false, agg.name)
}


rolling_apply_int_ewm_cov :: proc(
	r: RollingWindow,
	other: ^w.Column,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_cov_column(r, _ewm_other(r, other), agg.alpha, false, agg.bias, false, agg.name)
}


rolling_apply_float_ewm_var_adjust_true :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_var_column(r, agg.alpha, true, agg.bias, false, agg.name)
}

rolling_apply_int_ewm_var_adjust_true :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_var_column(r, agg.alpha, true, agg.bias, false, agg.name)
}


// Same as rolling_apply_float_ewm_cov (kept for API compatibility)
rolling_apply_float_ewm_cov_adjust_false :: proc(
	r: RollingWindow,
	other: ^w.Column,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_cov_column(r, _ewm_other(r, other), agg.alpha, false, agg.bias, false, agg.name)
}
// Same as rolling_apply_int_ewm_cov (kept for API compatibility)
rolling_apply_int_ewm_cov_adjust_false :: proc(
	r: RollingWindow,
	other: ^w.Column,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_cov_column(r, _ewm_other(r, other), agg.alpha, false, agg.bias, false, agg.name)
}
rolling_apply_float_ewm_corr :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_corr_column(r, w.column(r.df, agg.other), agg.alpha, false, false, agg.name)
}
rolling_apply_int_ewm_corr :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return ewm_corr_column(r, w.column(r.df, agg.other), agg.alpha, false, false, agg.name)
}


// ============================================================================
// Rolling (co)variance and correlation
// ============================================================================

@(private = "file")
_Rolling_Moment :: enum {
	Var,
	Cov,
	Corr,
}

// Sample (ddof = 1) var / cov and Pearson corr over the rows of each window
// where both x and y are non-null (pandas' pairwise alignment)
@(private = "file")
_rolling_moment :: proc(
	r: RollingWindow,
	x, y: ^w.Column,
	name: string,
	moment: _Rolling_Moment,
	allocator: mem.Allocator,
) -> w.Column {
	out := w.column_new(name, .Float, 0)

	xs := make([dynamic]f64, 0, max(r.window, 0), allocator)
	ys := make([dynamic]f64, 0, max(r.window, 0), allocator)
	defer delete(xs)
	defer delete(ys)

	for i in 0 ..< r.df.rows {
		clear(&xs)
		clear(&ys)
		start, end := rolling_bounds(r, i)
		for j in start ..< end {
			vx, nx := _numeric_at(x, j)
			vy, ny := _numeric_at(y, j)
			if !nx && !ny {
				append(&xs, vx)
				append(&ys, vy)
			}
		}

		n := len(xs)
		if n < _rolling_min_obs(r) || n < 2 {
			w.append_null(&out)
			continue
		}

		mean_x, mean_y := 0.0, 0.0
		for k in 0 ..< n {
			mean_x += xs[k]
			mean_y += ys[k]
		}
		mean_x /= f64(n)
		mean_y /= f64(n)

		sxy, sxx, syy := 0.0, 0.0, 0.0
		for k in 0 ..< n {
			dx := xs[k] - mean_x
			dy := ys[k] - mean_y
			sxy += dx * dy
			sxx += dx * dx
			syy += dy * dy
		}

		switch moment {
		case .Var:
			w.append_float(&out, sxx / f64(n - 1))
		case .Cov:
			w.append_float(&out, sxy / f64(n - 1))
		case .Corr:
			if sxx > 0 && syy > 0 {
				w.append_float(&out, sxy / math.sqrt(sxx * syy))
			} else {
				w.append_null(&out)
			}
		}
	}

	return out
}

rolling_apply_float_corr :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return _rolling_moment(r, w.column(r.df, r.column), w.column(r.df, agg.other), agg.name, .Corr, allocator)
}


rolling_apply_int_corr :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	return _rolling_moment(r, w.column(r.df, r.column), w.column(r.df, agg.other), agg.name, .Corr, allocator)
}

// Sample variance (ddof = 1, pandas' default)
rolling_apply_float_var :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	return _rolling_moment(r, src, src, agg.name, .Var, allocator)
}


rolling_apply_int_var :: proc(
	r: RollingWindow,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	return _rolling_moment(r, src, src, agg.name, .Var, allocator)
}

// Sample covariance (ddof = 1, pandas' default)
rolling_apply_float_cov :: proc(
	r: RollingWindow,
	other: ^w.Column,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	if other.type != src.type do panic("rolling cov: `other` column must have the same type as the source column")
	return _rolling_moment(r, src, other, agg.name, .Cov, allocator)
}
rolling_apply_int_cov :: proc(
	r: RollingWindow,
	other: ^w.Column,
	agg: w.Aggregator,
	allocator: mem.Allocator,
) -> w.Column {
	src := w.column(r.df, r.column)
	if other.type != src.type do panic("rolling cov: `other` column must have the same type as the source column")
	return _rolling_moment(r, src, other, agg.name, .Cov, allocator)
}
