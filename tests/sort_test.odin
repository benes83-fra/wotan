package tests


import w "../wotan/core"
import "core:fmt"

sort_test :: proc() {
	fmt.println("=== SORT TEST ===")

	df := w.dataframe_new()

	// Create columns
	col_age := w.column_new("age", .Int, 5)
	col_name := w.column_new("name", .String, 5)

	// Unsorted data
	ages := []int{40, 10, 30, 20, 50}
	names := []string{"Julia", "Hubert", "Markus", "Anna", "Zelda"}

	for i in 0 ..< 5 {
		w.append_int(&col_age, ages[i])
		w.append_string(&col_name, names[i])
	}

	w.add_column(&df, col_age)
	w.add_column(&df, col_name)

	fmt.println("Original:")
	w.dataframe_pretty_print(&df)

	// Ascending sort
	df_sorted := w.dataframe_sort(&df, "age", false)
	fmt.println("\nSorted ascending by age:")
	w.dataframe_pretty_print(&df_sorted, max_rows = 20)

	// Descending sort
	df_sorted_desc := w.dataframe_sort(&df, "age", true)
	fmt.println("\nSorted descending by age:")
	w.dataframe_pretty_print(&df_sorted_desc, max_rows = 20)

	w.destroy_dataframe(&df)
	w.destroy_dataframe(&df_sorted)
	w.destroy_dataframe(&df_sorted_desc)

	fmt.println("=== END SORT TEST ===")
}
