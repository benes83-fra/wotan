package main

import "core:fmt"

import w "./wotan/core"


groupby2_test :: proc() {
	fmt.println("=== Testing GroupBy2 ===")

	df_gb := w.dataframe_new()
	w.add_column(&df_gb, w.column_from_strings("key", []string{"A", "A", "B", "B", "B"}))
	w.add_column(&df_gb, w.column_from_ints("value", []int{1, 2, 3, 4, 5}))
	df_gb.rows = 5

	fmt.println("Input DF for GroupBy2:")
	w.dataframe_pretty_print(&df_gb, 20)

	gb2 := w.groupby2(&df_gb, []string{"key"}, allocator = context.temp_allocator)

	aggs2 := []w.Aggregator {
		{name = "sum_value", column = "value", kind = w.AggregationKind.Sum},
		{name = "count_value", column = "value", kind = w.AggregationKind.Count},
		{name = "min_value", column = "value", kind = w.AggregationKind.Min},
		{name = "max_value", column = "value", kind = w.AggregationKind.Max},
		{name = "median_value", column = "value", kind = w.AggregationKind.Median, quantile = 0.2},
	}

	out2 := w.groupby2_agg(&gb2, aggs2, allocator = context.temp_allocator)

	fmt.println("GroupBy2 Output:")
	w.dataframe_pretty_print(&out2, 20)

	fmt.println("=== Testing GroupBy2 (Multi-Key) ===")

	df_mk := w.dataframe_new()
	w.add_column(&df_mk, w.column_from_strings("A", []string{"x", "x", "x", "y", "y"}))
	w.add_column(&df_mk, w.column_from_ints("B", []int{1, 1, 2, 1, 1}))
	w.add_column(&df_mk, w.column_from_ints("V", []int{10, 20, 30, 40, 50}))
	df_mk.rows = 5

	fmt.println("Input DF for Multi-Key GroupBy2:")
	w.dataframe_pretty_print(&df_mk, 20)

	gb_mk := w.groupby2(&df_mk, []string{"A", "B"}, allocator = context.temp_allocator)

	aggs_mk := []w.Aggregator {
		{name = "sumV", column = "V", kind = w.AggregationKind.Sum},
		{name = "countV", column = "V", kind = w.AggregationKind.Count},
	}

	out_mk := w.groupby2_agg(&gb_mk, aggs_mk, allocator = context.temp_allocator)

	fmt.println("GroupBy2 Multi-Key Output:")
	w.dataframe_pretty_print(&out_mk, 20)

	w.destroy_dataframe(&df_mk)
	w.destroy_dataframe(&out_mk)


	w.destroy_dataframe(&df_gb)
	w.destroy_dataframe(&out2)


}
