package analytics

import w "../core"
import "core:mem"
acf :: proc(y: []f64, max_lag: int, allocator := context.allocator) -> []f64 {
	max_lag := max(max_lag, 0)

	n := len(y)
	if n == 0 {
		// Always return max_lag+1 entries so callers can index every lag.
		return make([]f64, max_lag + 1, allocator)
	}

	mean := 0.0
	for v in y {
		mean += v
	}
	mean /= f64(n)

	// variance γ(0)
	var0 := 0.0
	for v in y {
		diff := v - mean
		var0 += diff * diff
	}
	if var0 == 0 {
		return make([]f64, max_lag + 1, allocator)
	}

	out := make([]f64, max_lag + 1, allocator)

	for k in 0 ..= max_lag {
		num := 0.0
		for t in k ..< n {
			num += (y[t] - mean) * (y[t - k] - mean)
		}
		out[k] = num / var0
	}

	return out
}


pacf :: proc(y: []f64, max_lag: int, allocator := context.allocator) -> []f64 {

	max_lag := max(max_lag, 0)
	ac := acf(y, max_lag, allocator)
	pac := make([]f64, max_lag + 1, allocator)

	// φ_{1,1} = acf(1)
	pac[0] = 1.0
	if max_lag >= 1 {
		pac[1] = ac[1]
	}

	// workspace
	phi := make([][]f64, max_lag + 1, allocator)
	for k in 0 ..= max_lag {
		phi[k] = make([]f64, k + 1, allocator)
	}

	if max_lag >= 1 {
		phi[1][1] = ac[1]
	}

	for k in 2 ..= max_lag {
		// compute φ_{k,k}
		num := ac[k]
		for j in 1 ..< k {
			num -= phi[k - 1][j] * ac[k - j]
		}
		denom := 1.0
		for j in 1 ..< k {
			denom -= phi[k - 1][j] * ac[j]
		}
		phi[k][k] = num / denom

		// update φ_{k,j} for j < k
		for j in 1 ..< k {
			phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
		}

		pac[k] = phi[k][k]
	}

	return pac
}
// Non-null values of an Int or Float column as f64 (column_at_float does not
// check the type, so reading other column types would run past the buffer).
_numeric_column_values :: proc(c: ^w.Column, allocator: mem.Allocator) -> [dynamic]f64 {
	y := make([dynamic]f64, 0, allocator)
	#partial switch c.type {
	case .Float:
		for i in 0 ..< c.len {
			v, is_null := w.column_at_float(c, i)
			if !is_null do append(&y, v)
		}
	case .Int:
		for i in 0 ..< c.len {
			v, is_null := w.column_at_int(c, i)
			if !is_null do append(&y, f64(v))
		}
	case:
		panic("acf/pacf: column must be Int or Float")
	}
	return y
}

df_acf :: proc(
	df: ^w.DataFrame,
	col: string,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {

	max_lag := max(max_lag, 0)
	c := w.column(df, col)
	y := _numeric_column_values(c, allocator)

	// Compute ACF
	ac := acf(y[:], max_lag, allocator)

	// Return as a Column
	out := w.column_new("acf", .Float, max_lag + 1)
	for i in 0 ..= max_lag {
		w.append_float(&out, ac[i])
	}

	return out
}

df_pacf :: proc(
	df: ^w.DataFrame,
	col: string,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {

	max_lag := max(max_lag, 0)
	c := w.column(df, col)
	y := _numeric_column_values(c, allocator)

	// Compute PACF
	pc := pacf(y[:], max_lag, allocator)

	// Return as a Column
	out := w.column_new("pacf", .Float, max_lag + 1)
	for i in 0 ..= max_lag {
		w.append_float(&out, pc[i])
	}

	return out
}
