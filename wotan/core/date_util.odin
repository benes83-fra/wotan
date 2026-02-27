package wotan


date_compare :: proc(a, b: Date) -> i32 {
	if a.year != b.year {return a.year - b.year}
	if a.month != b.month {return a.month - b.month}
	return a.day - b.day

}
