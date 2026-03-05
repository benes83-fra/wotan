package wotan

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
  fn_time:      proc(x: Time) ->Time,
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

apply_datetime_expr :: proc(name: string, col: ^Column, fn: proc(x: Datetime) -> Datetime) -> Select_Expr {
	return Select_Expr{name = name, kind = .ApplyDatetime, col = col, fn_datetime = fn}
}

conv_int_to_f64_expr :: proc(name: string, col: ^Column) -> Select_Expr {
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
		case .IntAdd:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)

			for i in 0 ..< orig.len {
				v, n := column_at_int(orig, i)
				if n {append_null(&new_col)} else {append_int(&new_col, v + expr.int_value)}
			}

			add_column(&out, new_col)

		// ---------------------------------------------------------------------
		case .IntSub:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)

			for i in 0 ..< orig.len {
				v, n := column_at_int(orig, i)
				if n {append_null(&new_col)} else {append_int(&new_col, v - expr.int_value)}
			}

			add_column(&out, new_col)

		// ---------------------------------------------------------------------
		case .FloatAdd:
			orig := expr.col
			new_col := column_new(expr.name, .Float, orig.len)

			for i in 0 ..< orig.len {
				v, n := column_at_float(orig, i)
				if n {append_null(&new_col)} else {append_float(&new_col, v + expr.float_value)}
			}

			add_column(&out, new_col)

		// ---------------------------------------------------------------------
		case .FloatSub:
			orig := expr.col
			new_col := column_new(expr.name, .Float, orig.len)

			for i in 0 ..< orig.len {
				v, n := column_at_float(orig, i)
				if n {append_null(&new_col)} else {append_float(&new_col, v - expr.float_value)}
			}

			add_column(&out, new_col)
		case .IntMult:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_int(orig, i)
				if n {append_null(&new_col)} else {append_int(&new_col, v * expr.int_value)}

			}
			add_column(&out, new_col)
		case .IntDiv:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_int(orig, i)
				if n {append_null(&new_col)} else {
					div := expr.int_value
					if div != 0 {
						append_int(&new_col, v / div)
					} else {
						panic("select: Division by zero")
					}
				}

			}
			add_column(&out, new_col)
		case .FloatMult:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_float(orig, i)
				if n {append_null(&new_col)} else {append_float(&new_col, v * expr.float_value)}

			}
			add_column(&out, new_col)
		case .FloatDiv:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_float(orig, i)
				if n {append_null(&new_col)} else {
					div := expr.float_value
					if div != 0 {
						append_float(&new_col, v / div)
					} else {
						panic("select: Division by zero")
					}
				}

			}
			add_column(&out, new_col)
		case .ApplyInt:
			orig := expr.col
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_int(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_int(&new_col, expr.fn_int(v))
				}
			}
			add_column(&out, new_col)

		case .ApplyFloat:
			orig := expr.col
			new_col := column_new(expr.name, .Float, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_float(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_float(&new_col, expr.fn_float(v))
				}
			}
			add_column(&out, new_col)

		case .ApplyString:
			orig := expr.col
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
			new_col := column_new(expr.name, .Bool, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_bool(orig, i)
				if n {
					append_null(&new_col)
				} else {
					append_bool(&new_col, expr.fn_bool(v))
				}
			}
			add_column(&out, new_col)

		case .ApplyDate:
			orig := expr.col
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
			new_col := column_new(expr.name, .Int, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_float(orig, i)
				if n {
					append_null(&new_col)
				} else {
					w: int = int(v)
					append_int(&new_col, w)
				}
			}
			add_column(&out, new_col)
		case .ConvIntFloat:
			orig := expr.col
			new_col := column_new(expr.name, .Float, orig.len)
			for i in 0 ..< orig.len {
				v, n := column_at_int(orig, i)
				if n {
					append_null(&new_col)
				} else {
					w: f64 = f64(v)
					append_float(&new_col, w)
				}
			}
			add_column(&out, new_col)
		case .Conv:
			if expr.col == nil {
				panic("Conv: expr.col is nil for")
			}

			orig := expr.col
			target_type := get_column_type_by_type(expr.conv)
			new_col := column_new(expr.name, target_type, orig.len)

			for i in 0 ..< orig.len {
				// read value based on *source* type
				#partial switch orig.type {
				case .Int:
					v, n := column_at_int(orig, i)
					if n {
						append_null(&new_col)
					} else {
						#partial switch target_type {
						case .Int:
							append_int(&new_col, v)
						case .Float:
							append_float(&new_col, f64(v))
						case .String:
							buf: [8]u8
							res := strconv.write_int(buf[:], i64(v), 10)
							append_string(&new_col, res)
						case .Bool:
							append_bool(&new_col, v != 0)
						case .Date:
							date := int_to_date(i32(v))
							append_date(&new_col, date)
						}
					}

				case .Float:
					v, n := column_at_float(orig, i)
					if n {
						append_null(&new_col)
					} else {
						#partial switch target_type {
						case .Int:
							append_int(&new_col, int(v))
						case .Float:
							append_float(&new_col, v)
						case .String:
							buf: [8]u8
							res := strconv.write_float(buf[:], v, 'f', 4, 64)
							append_string(&new_col, res)
						case .Bool:
							append_bool(&new_col, v != 0.0)
						case .Date:
							date := f64_to_date(v)
							append_date(&new_col, date)
						}
					}
				case .String:
					v, n := column_at_string(orig, i)
					if n {
						append_null(&new_col)
					} else {
						#partial switch target_type {
						case .Int:
							integer, ok := strconv.parse_int(v)
							if !ok {
								panic("parse int: could not parse Integer")
							}
							append_int(&new_col, integer)
						case .Float:
							float, ok := strconv.parse_f64(v)
							if !ok {
								panic("parse float: could not parse Float")
							}
							append_float(&new_col, float)
						case .String:
							append_string(&new_col, v)
						case .Bool:
							b: bool = false
							if strings.to_upper(v) == "TRUE" {
								b = true
							}
							append_bool(&new_col, b)
						case .Date:
							date, ok := parse_date(v)
							if !ok {
								panic(
									"parse date: could not parse Date, string formated incorrectly",
								)
							}
							append_date(&new_col, date)
						}
					}
				case .Date:
					v, n := column_at_date(orig, i)
					if n {
						append_null(&new_col)
					} else {
						#partial switch target_type {
						case .Int:
							integer := date_to_int(v)

							append_int(&new_col, int(integer))
						case .Float:
							float := date_to_f64(v)

							append_float(&new_col, float)
						case .String:
							str := date_to_string(v)
							append_string(&new_col, str)
						case .Bool:
							b: bool = false

							append_bool(&new_col, b)
						case .Date:
							append_date(&new_col, v)
						}
					}

				// add .String, .Bool, .Date similarly
				}
			}

			add_column(&out, new_col)


		}


	}

	return out
}


free_select_exprs :: proc(exprs: []Select_Expr) {
	for expr in exprs {
		if expr.kind == .Mask {
			delete(expr.mask)
		}

	}
}
