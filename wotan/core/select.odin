package wotan

import "core:fmt"
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


conv_int_to_f64_expr :: proc(name: string, col: ^Column) -> Select_Expr {
	return Select_Expr{name = name, kind = .ConvIntFloat, col = col}
}

conv_f64_to_int_expr :: proc(name: string, col: ^Column) -> Select_Expr {
	return Select_Expr{name = name, kind = .ConvFloatInt, col = col}
}

conv_expr :: proc(name: string, col: ^Column, conv: string) -> Select_Expr {
	return Select_Expr{name = name, kind = .Conv, conv = conv}
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
			//
			orig := expr.col
			new_col: Column
			col_type := get_column_type_by_type(expr.conv)
			fmt.println(col_type)
			conv := expr.conv
      new_col = column_new(expr.name, col_type, orig.len)
			fmt.println("Survived Col creation")
			for i in 0 ..< orig.len {
				v, n := column_at_ptr(orig, i)
				fmt.println(v)
				if n {
					append_null(&new_col)
				} else {
					append_null(&new_col)

					if conv == "int" {
						append_int(&new_col, (cast(^int)v)^)
					} else if conv == "float" {
						append_float(&new_col, (cast(^f64)v)^)
					} else if conv == "string" {
						append_string(&new_col, (cast(^string)v)^)
					} else if conv == "date" {
						append_date(&new_col, (cast(^Date)v)^)
					} else if conv == "bool" {
						append_bool(&new_col, (cast(^bool)v)^)
					} else {
						append_null(&new_col)
					}
				}
				add_column(&out, new_col)

			}
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
