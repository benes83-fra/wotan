package core

// --- WHERE API -------------------------------------------------------------
// wobei is a wotan way of doing whereish things on dataframes.
wobei :: proc {
	where_mask,
	where_column,
	where_column_name,
}

// where(mask)
where_mask :: proc(df: ^DataFrame, mask: []bool) -> DataFrame {
	return filter(df, mask)
}

// where(column)
where_column :: proc(df: ^DataFrame, col: ^Column) -> DataFrame {
	if col.type != .Bool {
		panic("where_column: column is not Bool")
	}
	mask := column_mask(col)
	out := filter(df, mask)
	delete(mask)

	return out
}

// where("colname")
where_column_name :: proc(df: ^DataFrame, name: string) -> DataFrame {
	col := column(df, name)
	if col.type != .Bool {
		panic("where_column_name: column is not Bool")
	}
	mask := column_mask(col)
	out := filter(df, mask)
	delete(mask)
	return out
}
