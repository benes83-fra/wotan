package tests

import w "../wotan/core"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

plot_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Pure PNG Plotter ===")

	// 1. Create a DataFrame with some numeric data
	df := w.dataframe_new(allocator)
	defer w.destroy_dataframe(&df)

	n := 100
	x_col := w.column_new("x", .Float, n)
	y_col := w.column_new("y", .Float, n)

	for i in 0 ..< n {
		x_val := f64(i) / 10.0
		y_val := math.sin(x_val) + rand.float64_normal(0, 0.2)
		w.append_float(&x_col, x_val)
		w.append_float(&y_col, y_val)
	}

	w.add_column(&df, x_col)
	w.add_column(&df, y_col)

	// 2. Generate Scatter Plot
	ok1 := plot.scatter_png(&df, "x", "y", "test_scatter.png", allocator)
	fmt.printf("Scatter Plot Saved: %v\n", ok1)

	// 3. Generate Histogram
	ok2 := plot.histogram_png(&df, "y", "test_hist.png", 15, allocator)
	fmt.printf("Histogram Saved: %v\n", ok2)
}
