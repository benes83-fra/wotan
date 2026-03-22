package main


import w "./wotan/core"
import "core:fmt"


join_test :: proc() {

	fmt.println("TEST 1: Inner Join - Perfect Match")

	left3 := w.dataframe_new()
	w.add_column(&left3, w.column_from_ints("id", []int{1, 2, 3}))
	w.add_column(&left3, w.column_from_ints("dept", []int{10, 20, 30}))
	w.add_column(&left3, w.column_from_strings("name", []string{"A", "B", "C"}))
	left3.rows = 3

	right3 := w.dataframe_new()
	w.add_column(&right3, w.column_from_ints("id", []int{1, 2, 3}))
	w.add_column(&right3, w.column_from_ints("dept", []int{10, 20, 30}))
	w.add_column(&right3, w.column_from_ints("salary", []int{100, 200, 300}))
	right3.rows = 3

	out2 := w.join(&left3, &right3, []string{"id", "dept"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out2, 20)


	fmt.println("TEST 2: Inner Join - No Matches")

	left4 := w.dataframe_new()
	w.add_column(&left4, w.column_from_ints("id", []int{1, 2}))
	left4.rows = 2

	right4 := w.dataframe_new()
	w.add_column(&right4, w.column_from_ints("id", []int{3, 4}))
	right4.rows = 2

	out3 := w.join(&left4, &right4, []string{"id"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out3, 20)


	fmt.println("TEST 3: Left Join - Missing Right Rows")

	left5 := w.dataframe_new()
	w.add_column(&left5, w.column_from_ints("id", []int{1, 2, 3}))
	left5.rows = 3

	right5 := w.dataframe_new()
	w.add_column(&right5, w.column_from_ints("id", []int{1, 3}))
	w.add_column(&right5, w.column_from_ints("salary", []int{100, 300}))
	right5.rows = 2

	out4 := w.join(&left5, &right5, []string{"id"}, .Left, context.temp_allocator)
	w.dataframe_pretty_print(&out4, 20)


	fmt.println("TEST 4: Right Join - Missing Left Rows")

	left6 := w.dataframe_new()
	w.add_column(&left6, w.column_from_ints("id", []int{1, 3}))
	left6.rows = 2

	right6 := w.dataframe_new()
	w.add_column(&right6, w.column_from_ints("id", []int{1, 2, 3}))
	w.add_column(&right6, w.column_from_ints("salary", []int{100, 200, 300}))
	right6.rows = 3

	out5 := w.join(&left6, &right6, []string{"id"}, .Right, context.temp_allocator)
	w.dataframe_pretty_print(&out5, 20)


	fmt.println("TEST 5: Outer Join - Symmetric Missing Rows")

	left7 := w.dataframe_new()
	w.add_column(&left7, w.column_from_ints("id", []int{1, 2}))
	left7.rows = 2

	right7 := w.dataframe_new()
	w.add_column(&right7, w.column_from_ints("id", []int{2, 3}))
	right7.rows = 2

	out6 := w.join(&left7, &right7, []string{"id"}, .Outer, context.temp_allocator)
	w.dataframe_pretty_print(&out6, 20)


	fmt.println("TEST 6: Multi - Key Strictness")

	left8 := w.dataframe_new()
	w.add_column(&left8, w.column_from_ints("id", []int{1}))
	w.add_column(&left8, w.column_from_ints("dept", []int{10}))
	left8.rows = 1

	right8 := w.dataframe_new()
	w.add_column(&right8, w.column_from_ints("id", []int{1}))
	w.add_column(&right8, w.column_from_ints("dept", []int{20}))
	right8.rows = 1

	out7 := w.join(&left8, &right8, []string{"id", "dept"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out7, 20)

	fmt.println("TEST 7: 1-to-Many (Right duplicates)")

	left9 := w.dataframe_new()
	w.add_column(&left9, w.column_from_ints("id", []int{1}))
	left9.rows = 1

	right9 := w.dataframe_new()
	w.add_column(&right9, w.column_from_ints("id", []int{1, 1, 1}))
	w.add_column(&right9, w.column_from_ints("salary", []int{10, 20, 30}))
	right9.rows = 3

	out8 := w.join(&left9, &right9, []string{"id"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out8, 20)

	fmt.println("TEST 8: 1-to-Many (Left duplicates)")

	left10 := w.dataframe_new()
	w.add_column(&left10, w.column_from_ints("id", []int{1, 1}))
	left10.rows = 2

	right10 := w.dataframe_new()
	w.add_column(&right10, w.column_from_ints("id", []int{1}))
	w.add_column(&right10, w.column_from_ints("salary", []int{999}))
	right10.rows = 1

	out9 := w.join(&left10, &right10, []string{"id"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out9, 20)

	fmt.println("TEST 9: Many-to-Many")

	left11 := w.dataframe_new()
	w.add_column(&left11, w.column_from_ints("id", []int{1, 1}))
	left11.rows = 2

	right11 := w.dataframe_new()
	w.add_column(&right11, w.column_from_ints("id", []int{1, 1}))
	w.add_column(&right11, w.column_from_ints("salary", []int{10, 20}))
	right11.rows = 2

	out10 := w.join(&left11, &right11, []string{"id"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out10, 20)

	fmt.println("TEST 10: Multi-Key Many-to-Many")

	left12 := w.dataframe_new()
	w.add_column(&left12, w.column_from_ints("id", []int{1, 1}))
	w.add_column(&left12, w.column_from_ints("dept", []int{10, 10}))
	left12.rows = 2

	right12 := w.dataframe_new()
	w.add_column(&right12, w.column_from_ints("id", []int{1, 1}))
	w.add_column(&right12, w.column_from_ints("dept", []int{10, 10}))
	w.add_column(&right12, w.column_from_ints("salary", []int{10, 20}))
	right12.rows = 2

	out11 := w.join(&left12, &right12, []string{"id", "dept"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out11, 20)


	fmt.println("TEST 11: Mixed-Type Composite Keys")

	left13 := w.dataframe_new()
	w.add_column(&left13, w.column_from_ints("id", []int{1, 1}))
	w.add_column(&left13, w.column_from_strings("country", []string{"DE", "US"}))
	left13.rows = 2

	right13 := w.dataframe_new()
	w.add_column(&right13, w.column_from_ints("id", []int{1, 1}))
	w.add_column(&right13, w.column_from_strings("country", []string{"DE", "DE"}))
	w.add_column(&right13, w.column_from_ints("salary", []int{100, 200}))
	right13.rows = 2

	out12 := w.join(&left13, &right13, []string{"id", "country"}, .Inner, context.temp_allocator)
	w.dataframe_pretty_print(&out12, 20)


	fmt.println("TEST 12: Outer Join - NULL Propagation")

	left14 := w.dataframe_new()
	w.add_column(&left14, w.column_from_ints("id", []int{1, 2}))
	left14.rows = 2

	right14 := w.dataframe_new()
	w.add_column(&right14, w.column_from_ints("id", []int{2, 3}))
	w.add_column(&right14, w.column_from_ints("salary", []int{200, 300}))
	right14.rows = 2

	out13 := w.join(&left14, &right14, []string{"id"}, .Outer, context.temp_allocator)
	w.dataframe_pretty_print(&out13, 20)

	w.destroy_dataframe(&left14)
	w.destroy_dataframe(&right14)
	w.destroy_dataframe(&out13)


	w.destroy_dataframe(&left13)
	w.destroy_dataframe(&right13)
	w.destroy_dataframe(&out12)


	w.destroy_dataframe(&left12)
	w.destroy_dataframe(&right12)
	w.destroy_dataframe(&out11)


	w.destroy_dataframe(&left11)
	w.destroy_dataframe(&right11)
	w.destroy_dataframe(&out10)

	w.destroy_dataframe(&left10)
	w.destroy_dataframe(&right10)
	w.destroy_dataframe(&out9)


	w.destroy_dataframe(&left9)
	w.destroy_dataframe(&right9)
	w.destroy_dataframe(&out8)


	w.destroy_dataframe(&left8)
	w.destroy_dataframe(&right8)
	w.destroy_dataframe(&out7)

	w.destroy_dataframe(&left7)
	w.destroy_dataframe(&right7)
	w.destroy_dataframe(&out6)


	w.destroy_dataframe(&left6)
	w.destroy_dataframe(&right6)
	w.destroy_dataframe(&out5)


	w.destroy_dataframe(&left5)
	w.destroy_dataframe(&right5)
	w.destroy_dataframe(&out4)


	w.destroy_dataframe(&left4)
	w.destroy_dataframe(&right4)
	w.destroy_dataframe(&out3)


	w.destroy_dataframe(&left3)
	w.destroy_dataframe(&right3)
	w.destroy_dataframe(&out2)


}
