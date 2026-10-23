package core

import "core:fmt"
import "core:strconv"
import "core:strings"
// --- Select Expression Types ------------------------------------------------
//

Select_Kind :: enum {
	Column, // copy an existing column
	Mask, // convert []bool mask to Bool column
	IntAdd, // col + int
	IntSub, // col - int
	IntMult,
	IntDiv,
	FloatAdd, // col + float
	FloatSub, // col - float
	FloatMult,
	FloatDiv,
	ApplyInt,
	ApplyFloat,
	ApplyString,
	ApplyBool,
	ApplyDate,
	ApplyTime,
	ApplyDatetime,
	ConvIntFloat,
	ConvFloatInt,
	Conv,
}

Select_Expr :: struct {
	name:         string,
	kind:         Select_Kind,
	col:          ^Column, // for column-based expressions
	mask:         []bool, // for mask-based expressions
	int_value:    int,
	float_value:  f64,
	string_value: string,
	conv:         string,
	fn_int:       proc(x: int) -> int,
	fn_float:     proc(x: f64) -> f64,
	fn_string:    proc(x: string) -> string,
	fn_bool:      proc(x: bool) -> bool,
	fn_date:      proc(x: Date) -> Date,
	fn_time:      proc(x: Time) -> Time,
	fn_datetime:  proc(x: Datetime) -> Datetime,
}

//
// --- Convenience Constructors -----------------------------------------------
//

col_expr :: proc(name: string, col: ^Column) -> Select_Expr {
	return Select_Expr{name = name, kind = .Column, col = col}
}

mask_expr :: proc(name: string, mask: []bool) -> Select_Expr {
	return Select_Expr{name = name, kind = .Mask, mask = mask}
}

add_expr :: proc {
	add_int_expr,
	add_float_expr,
}

sub_expr :: proc {
	sub_int_expr,
	sub_float_expr,
}


mult_expr :: proc {
	mult_int_expr,
	mult_float_expr,
}

div_expr :: proc {
	div_int_expr,
	div_float_expr,
}

add_int_expr :: proc(name: string, col: ^Column, v: int) -> Select_Expr {
	return Select_Expr{name = name, kind = .IntAdd, col = col, int_value = v}
}

sub_int_expr :: proc(name: string, col: ^Column, v: int) -> Select_Expr {
	return Select_Expr{name = name, kind = .IntSub, col = col, int_value = v}
}

add_float_expr :: proc(name: string, col: ^Column, v: f64) -> Select_Expr {
	return Select_Expr{name = name, kind = .FloatAdd, col = col, float_value = v}
}

sub_float_expr :: proc(name: string, col: ^Column, v: f64) -> Select_Expr {
	return Select_Expr{name = name, kind = .FloatSub, col = col, float_value = v}
}


mult_int_expr :: proc(name: string, col: ^Column, v: int) -> Select_Expr {
	return Select_Expr{name = name, kind = .IntMult, col = col, int_value = v}
}

div_int_expr :: proc(name: string, col: ^Column, v: int) -> Select_Expr {
	return Select_Expr{name = name, kind = .IntDiv, col = col, int_value = v}

}

mult_float_expr :: proc(name: string, col: ^Column, v: f64) -> Select_Expr {
	return Select_Expr{name = name, kind = .FloatMult, col = col, float_value = v}
}
div_float_expr :: proc(name: string, col: ^Column, v: f64) -> Select_Expr {
	assert(col.type == .Float)
	return Select_Expr{name = name, kind = .FloatDiv, col = col, float_value = v}
}


apply_expr :: proc {
	apply_int_expr,
	apply_float_expr,
	apply_string_expr,
	apply_bool_expr,
	apply_date_expr,
	apply_time_expr,
	apply_datetime_expr,
}

apply_int_expr :: proc(name: string, col: ^Column, fn: proc(x: int) -> int) -> Select_Expr {

	return Select_Expr{name = name, kind = .ApplyInt, col = col, fn_int = fn}
}

apply_float_expr :: proc(name: string, col: ^Column, fn: proc(x: f64) -> f64) -> Select_Expr {
	return Select_Expr{name = name, kind = .ApplyFloat, col = col, fn_float = fn}
}

apply_string_expr :: proc(
	name: string,
	col: ^Column,
	fn: proc(x: string) -> string,
) -> Select_Expr {

	return Select_Expr{name = name, kind = .ApplyString, col = col, fn_string = fn}
}

apply_bool_expr :: proc(name: string, col: ^Column, fn: proc(x: bool) -> bool) -> Select_Expr {
	return Select_Expr{name = name, kind = .ApplyBool, col = col, fn_bool = fn}
}

apply_date_expr :: proc(name: string, col: ^Column, fn: proc(x: Date) -> Date) -> Select_Expr {
	return Select_Expr{name = name, kind = .ApplyDate, col = col, fn_date = fn}
}

apply_time_expr :: proc(name: string, col: ^Column, fn: proc(x: Time) -> Time) -> Select_Expr {
	return Select_Expr{name = name, kind = .ApplyTime, col = col, fn_time = fn}
}

apply_datetime_expr :: proc(
	name: string,
	col: ^Column,
	fn: proc(x: Datetime) -> Datetime,
) -> Select_Expr {
	return Select_Expr{name = name, kind = .ApplyDatetime, col = col, fn_datetime = fn}
}

// Int (or Bool) column as Float; a Float column is passed through
conv_int_to_f64_expr :: proc(name: string, col: ^Column) -> Select_Expr {
	assert(col.type == .Int || col.type == .Float || col.type == .Bool, "conv_int_to_f64_expr: column must be numeric")
	return Select_Expr{name = name, kind = .ConvIntFloat, col = col}
}

conv_f64_to_int_expr :: proc(name: string, col: ^Column) -> Select_Expr {
	return Select_Expr{name = name, kind = .ConvFloatInt, col = col}
}

conv_expr :: proc(name: string, col: ^Column, conv: string) -> Select_Expr {
	return Select_Expr{name = name, kind = .Conv, col = col, conv = conv}
}
//
// --- Select Overload Group --------------------------------------------------
//

select :: proc {
	select_columns,
	select_exprs,
}

//
// --- Column-only Select -----------------------------------------------------
//

select_columns :: proc(df: ^DataFrame, names: []string) -> DataFrame {
	return dataframe_select_columns(df, names, false)
}

//
// --- Expression-based Select ------------------------------------------------
//

select_exprs :: proc(df: ^DataFrame, exprs: []Select_Expr) -> DataFrame {
	out := dataframe_new()

	for expr in exprs {
		#partial switch expr.kind {

		// ---------------------------------------------------------------------
		case .Column:
			orig := expr.col
			new_col := column_new(expr.name, orig.type, orig.len)

			for i in 0 ..< orig.len {
				#partial switch orig.type {
				case .Int:
					v, n := column_at_int(orig, i)
					if n {append_null(&new_col)} else {append_int(&new_col, v)}

				case .Float:
					v, n := column_at_float(orig, i)
					if n {append_null(&new_col)} else {append_float(&new_col, v)}

				case .Bool:
					v, n := column_at_bool(orig, i)
					if n {append_null(&new_col)} else {append_bool(&new_col, v)}

				case .String:
					v, n := column_at_string(orig, i)
					if n {append_null(&new_col)} else {append_string(&new_col, v)}

				case .Date:
					v, n := column_at_date(orig, i)
					if n {append_null(&new_col)} else {append_date(&new_col, v)}
				case .Time:
					v, n := column_at_time(orig, i)
					if n {append_null(&new_col)} else {append_time(&new_col, v)}
				case .Datetime:
					v, n := column_at_datetime(orig, i)
					if n {append_null(&new_col)} else {append_datetime(&new_col, v)}
				}
			}

			add_column(&out, new_col)

		// ---------------------------------------------------------------------
		case .Mask:
			m := expr.mask
			new_col := column_new(expr.name, .Bool, len(m))

			for i in 0 ..< len(m) {
				append_bool(&new_col, m[i])
			}

			add_column(&out, new_col)

		// ---------------------------------------------------------------------
		// Arithmetic / apply on numeric columns: values are read with numeric
		// coercion (Int, Float, Bool); integer arithmetic on a Float column is
		// done in floating point and gives a Float column.
		case .IntAdd, .IntSub, .IntMult, .IntDiv:
			orig := expr.col
			_require_numeric(orig, expr.name)
			as_float := orig.type == .Float
			new_col := column_new(expr.name, .Float if as_float else .Int, orig.len)
			if expr.kind == .IntDiv && expr.int_value == 0 {
				panic("select: Division by zero")
			}
			for i in 0 ..< orig.len {
				x, xi, n := _num_at(orig, i)
				if n {
					append_null(&new_col)
					continue
				}
				if as_float {
					d := f64(expr.int_value)
					#partial switch expr.kind {
					case .IntAdd:
						append_float(&new_col, x + d)
					case .IntSub:
						append_float(&new_col, x - d)
					case .IntMult:
						append_float(&new_col, x * d)
					case:
						append_float(&new_col, x / d)
					}
				} else {
					#partial switch expr.kind {
					case .IntAdd:
						append_int(&new_col, xi + expr.int_value)
					case .IntSub:
						append_int(&new_col, xi - expr.int_value)
					case .IntMult:
						append_int(&new_col, xi * expr.int_value)
					case:
						append_int(&new_col, xi / expr.int_value)
					}
				}
			}
			add_column(&out, new_col)

		case .FloatAdd, .FloatSub, .FloatMult, .FloatDiv:
			orig := expr.col
			_require_numeric(orig, expr.name)
			new_col := column_new(expr.name, .Float, orig.len)
			if expr.kind == .FloatDiv && expr.float_value == 0 {
				panic("select: Division by zero")
			}
			for i in 0 ..< orig.len {
				x, _, n := _num_at(orig, i)
				if n {
					append_null(&new_col)
					continue
				}
				#partial switch expr.kind {
				case .FloatAdd:
					append_float(&new_col, x + expr.float_value)
				case .FloatSub:
					append_float(&new_col, x - expr.float_value)
				case .FloatMult:
					append_float(&new_col, x * expr.float_value)
				case:
					append_float(&new_col, x / expr.float_value)
				}
			}
			add_column(&out, new_col)

		case .ApplyInt:
			orig := expr.col
			_require_numeric(orig, expr.name)
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				_, xi, n := _num_at(orig, i)
				if n {append_null(&new_col)} else {append_int(&new_col, expr.fn_int(xi))}
			}
			add_column(&out, new_col)

		case .ApplyFloat:
			orig := expr.col
			_require_numeric(orig, expr.name)
			new_col := column_new(expr.name, .Float, orig.len)
			for i in 0 ..< orig.len {
				x, _, n := _num_at(orig, i)
				if n {append_null(&new_col)} else {append_float(&new_col, expr.fn_float(x))}
			}
			add_column(&out, new_col)

		case .ApplyString:
			orig := expr.col
			_require_type(orig, .String, expr.name)
			new_col := column_new(expr.name, .String, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_string(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_string(&new_col, expr.fn_string(v))
				}
			}
			add_column(&out, new_col)

		case .ApplyBool:
			orig := expr.col
			_require_numeric(orig, expr.name) // numbers count as true when non-zero
			new_col := column_new(expr.name, .Bool, orig.len)
			for i in 0 ..< orig.len {
				x, _, n := _num_at(orig, i)
				if n {append_null(&new_col)} else {append_bool(&new_col, expr.fn_bool(x != 0))}
			}
			add_column(&out, new_col)

		case .ApplyDate:
			orig := expr.col
			_require_type(orig, .Date, expr.name)
			new_col := column_new(expr.name, .Date, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_date(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_date(&new_col, expr.fn_date(v))
				}
			}
			add_column(&out, new_col)
		case .ApplyTime:
			orig := expr.col
			_require_type(orig, .Time, expr.name)
			new_col := column_new(expr.name, .Time, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_time(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_time(&new_col, expr.fn_time(v))
				}
			}
			add_column(&out, new_col)
		case .ApplyDatetime:
			orig := expr.col
			_require_type(orig, .Datetime, expr.name)
			new_col := column_new(expr.name, .Datetime, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_datetime(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_datetime(&new_col, expr.fn_datetime(v))
				}
			}
			add_column(&out, new_col)
		case .ConvFloatInt:
			orig := expr.col
			_require_numeric(orig, expr.name)
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				_, xi, n := _num_at(orig, i)
				if n {append_null(&new_col)} else {append_int(&new_col, xi)}
			}
			add_column(&out, new_col)
		case .ConvIntFloat:
			orig := expr.col
			_require_numeric(orig, expr.name)
			new_col := column_new(expr.name, .Float, orig.len)
			for i in 0 ..< orig.len {
				x, _, n := _num_at(orig, i)
				if n {append_null(&new_col)} else {append_float(&new_col, x)}
			}
			add_column(&out, new_col)
		case .Conv:
			if expr.col == nil {
				panic(fmt.tprintf("Conv: expr.col is nil for %s", expr.name))
			}

			orig := expr.col
			target_type := get_column_type_by_type(expr.conv)
			if target_type == .Invalid {
				panic(fmt.tprintf("Conv: unknown target type %q for %s", expr.conv, expr.name))
			}
			new_col := column_new(expr.name, target_type, orig.len)
			for i in 0 ..< orig.len {
				if is_null(orig, i) {
					append_null(&new_col)
				} else {
					_conv_append(&new_col, orig, i, expr.name)
				}
			}
			add_column(&out, new_col)


		}


	}

	return out
}


// Convert cell i of src (not null) to dst.type and append it. Supported:
// any type -> String (canonical text) and -> itself; Int/Float/Bool between
// each other (non-zero = true); Int <-> Date (YYYYMMDD), Float <-> Date (Julian
// day); Int/Float <-> Time (seconds of day); Int/Float <-> Datetime (Unix
// seconds); Datetime -> Date / Time; Date -> Datetime (midnight); String ->
// any type (parsed). Other pairs panic instead of producing a short column.
@(private = "file")
_conv_append :: proc(dst: ^Column, src: ^Column, i: int, name: string) {
	target := dst.type
	if target == .String {
		append_string(dst, value_as_string(src, i, context.temp_allocator))
		return
	}
	unsupported :: proc(name: string, from, to: ColumnType) -> ! {
		panic(fmt.tprintf("Conv %s: no conversion from %v to %v", name, from, to))
	}
	bad_text :: proc(name: string, v: string, to: ColumnType) -> ! {
		panic(fmt.tprintf("Conv %s: cannot parse %q as %v", name, v, to))
	}

	switch src.type {
	case .Int, .Float, .Bool:
		x: f64
		iv: int
		switch src.type {
		case .Int:
			iv, _ = column_at_int(src, i)
			x = f64(iv)
		case .Float:
			x, _ = column_at_float(src, i)
			iv = int(x)
		case .Bool:
			bv, _ := column_at_bool(src, i)
			iv = 1 if bv else 0
			x = f64(iv)
		case .String, .Date, .Time, .Datetime, .Invalid:
		}
		#partial switch target {
		case .Int:
			append_int(dst, iv)
		case .Float:
			append_float(dst, x)
		case .Bool:
			append_bool(dst, x != 0)
		case .Date:
			if src.type == .Int {
				append_date(dst, int_to_date(i32(iv)))
			} else if src.type == .Float {
				append_date(dst, f64_to_date(x))
			} else {
				unsupported(name, src.type, target)
			}
		case .Time:
			if src.type == .Bool {unsupported(name, src.type, target)}
			append_time(dst, int_to_time(i32(iv)))
		case .Datetime:
			if src.type == .Bool {unsupported(name, src.type, target)}
			append_datetime(dst, datetime_from_unix(i64(x)))
		case:
			unsupported(name, src.type, target)
		}

	case .String:
		v, _ := column_at_string(src, i)
		t := strings.trim_space(v)
		#partial switch target {
		case .Int:
			x, ok := strconv.parse_int(t)
			if !ok {bad_text(name, v, target)}
			append_int(dst, x)
		case .Float:
			x, ok := strconv.parse_f64(t)
			if !ok {bad_text(name, v, target)}
			append_float(dst, x)
		case .Bool:
			append_bool(dst, strings.equal_fold(t, "true") || t == "1")
		case .Date:
			x, ok := parse_date(t)
			if !ok {bad_text(name, v, target)}
			append_date(dst, x)
		case .Time:
			x, ok := parse_time(t)
			if !ok {bad_text(name, v, target)}
			append_time(dst, x)
		case .Datetime:
			x, ok := parse_datetime(t)
			if !ok {bad_text(name, v, target)}
			append_datetime(dst, x)
		case:
			unsupported(name, src.type, target)
		}

	case .Date:
		v, _ := column_at_date(src, i)
		#partial switch target {
		case .Int:
			append_int(dst, int(date_to_int(v)))
		case .Float:
			append_float(dst, date_to_f64(v))
		case .Date:
			append_date(dst, v)
		case .Datetime:
			append_datetime(dst, Datetime{v.year, v.month, v.day, 0, 0, 0})
		case:
			unsupported(name, src.type, target)
		}

	case .Time:
		v, _ := column_at_time(src, i)
		#partial switch target {
		case .Int:
			append_int(dst, int(time_to_int(v)))
		case .Float:
			append_float(dst, f64(time_to_int(v)))
		case .Time:
			append_time(dst, v)
		case:
			unsupported(name, src.type, target)
		}

	case .Datetime:
		v, _ := column_at_datetime(src, i)
		#partial switch target {
		case .Int:
			append_int(dst, int(datetime_to_unix(v)))
		case .Float:
			append_float(dst, f64(datetime_to_unix(v)))
		case .Date:
			append_date(dst, Date{v.year, v.month, v.day})
		case .Time:
			append_time(dst, Time{v.hour, v.minute, v.second})
		case .Datetime:
			append_datetime(dst, v)
		case:
			unsupported(name, src.type, target)
		}

	case .Invalid:
		unsupported(name, src.type, target)
	}
}


// Numeric cell with coercion: (as float, as int (truncated), is_null)
@(private = "file")
_num_at :: proc(col: ^Column, i: int) -> (f64, int, bool) {
	#partial switch col.type {
	case .Int:
		v, n := column_at_int(col, i)
		return f64(v), v, n
	case .Float:
		v, n := column_at_float(col, i)
		return v, int(v), n
	case .Bool:
		v, n := column_at_bool(col, i)
		return 1 if v else 0, 1 if v else 0, n
	}
	return 0, 0, true
}

@(private = "file")
_require_numeric :: proc(col: ^Column, name: string) {
	if col.type != .Int && col.type != .Float && col.type != .Bool {
		panic(fmt.tprintf("select %s: column %q is %v, expected a numeric column", name, col.name, col.type))
	}
}

@(private = "file")
_require_type :: proc(col: ^Column, t: ColumnType, name: string) {
	if col.type != t {
		panic(fmt.tprintf("select %s: column %q is %v, expected %v", name, col.name, col.type, t))
	}
}


free_select_exprs :: proc(exprs: []Select_Expr) {
	for expr in exprs {
		if expr.kind == .Mask {
			delete(expr.mask)
		}

	}
}
