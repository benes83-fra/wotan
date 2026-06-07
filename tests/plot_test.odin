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
	// Example of using the new configurability
	my_config := plot.DEFAULT_PLOT_CONFIG
	my_config.bg_color = plot.Color{20, 20, 20, 255}
	my_config.axis_color = plot.WHITE
	my_config.point_color = plot.Color{0, 255, 0, 255}
	my_config.font_scale = 3
	my_config.title = "Sine Wave with Noise"
	my_config.x_label = "Time"
	my_config.y_label = "Amplitude"

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
	ok1 := plot.scatter_png(&df, "x", "y", "test_scatter.png", my_config, allocator)
	fmt.printf("Scatter Plot Saved: %v\n", ok1)

	// 3. Generate Histogram
	ok2 := plot.histogram_png(&df, "y", "test_hist.png", 15, my_config, allocator)
	fmt.printf("Histogram Saved: %v\n", ok2)
}
line_plot_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Line Plot ===")

	// Test 1: Simple sine wave
	n := 100
	xs := make([]f64, n, allocator)
	ys := make([]f64, n, allocator)
	defer delete(xs, allocator)
	defer delete(ys, allocator)

	for i in 0 ..< n {
		x := f64(i) / 10.0
		xs[i] = x
		ys[i] = math.sin(x)
	}

	// Solid line
	config1 := plot.DEFAULT_PLOT_CONFIG
	config1.title = "Sine Wave - Solid Line"
	config1.x_label = "Time"
	config1.y_label = "Amplitude"
	config1.point_color = plot.Color{0, 100, 255, 255} // Blue
	config1.bg_color = plot.Color{20, 20, 20, 255}
	config1.axis_color = plot.WHITE
	config1.line_style = .Solid

	ok1 := plot.line_png(xs, ys, "line_solid.png", config1, allocator)
	fmt.printf("Solid line plot: %v\n", ok1)

	// Test 2: Dashed line
	config2 := config1
	config2.title = "Sine Wave - Dashed Line"
	config2.line_style = .Dashed

	ok2 := plot.line_png(xs, ys, "line_dashed.png", config2, allocator)
	fmt.printf("Dashed line plot: %v\n", ok2)

	// Test 3: Dotted line (exponential decay)
	config3 := config1
	config3.title = "Exponential Decay - Dotted"
	config3.line_style = .Dotted
	config3.point_color = plot.Color{0, 255, 100, 255} // Green

	for i in 0 ..< n {
		xs[i] = f64(i) / 10.0
		ys[i] = math.exp(-f64(i) / 20.0)
	}

	ok3 := plot.line_png(xs, ys, "line_dotted.png", config3, allocator)
	fmt.printf("Dotted line plot: %v\n", ok3)

	// Test 4: Training curve example (multiple lines would need separate calls)
	epochs := 50
	train_xs := make([]f64, epochs, allocator)
	train_ys := make([]f64, epochs, allocator)
	val_ys := make([]f64, epochs, allocator)
	defer delete(train_xs, allocator)
	defer delete(train_ys, allocator)
	defer delete(val_ys, allocator)

	for i in 0 ..< epochs {
		train_xs[i] = f64(i)
		train_ys[i] = 1.0 / f64(i + 1) + rand.float64_normal(0, 0.01)
		val_ys[i] = 1.0 / f64(i + 1) + 0.1 + rand.float64_normal(0, 0.01)
	}

	config4 := plot.DEFAULT_PLOT_CONFIG
	config4.title = "Training Loss"
	config4.x_label = "Epoch"
	config4.y_label = "Loss"
	config4.point_color = plot.Color{255, 100, 0, 255} // Orange
	config4.bg_color = plot.Color{20, 20, 20, 255}
	config4.axis_color = plot.WHITE
	config4.line_style = .Solid

	ok4 := plot.line_png(train_xs, train_ys, "training_curve.png", config4, allocator)
	fmt.printf("Training curve: %v\n", ok4)
}


bar_chart_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Bar Chart ===")

	// Test 1: Simple categorical data
	labels1 := []string{"A", "B", "C", "D", "E", "F"}
	values1 := []f64{23.5, 45.2, 12.8, 67.3, 34.1, 55.0}

	config1 := plot.DEFAULT_PLOT_CONFIG
	config1.title = "Category Values"
	config1.y_label = "Count"
	config1.bar_color = plot.Color{0, 150, 255, 255} // Light blue
	config1.bg_color = plot.Color{20, 20, 20, 255}
	config1.axis_color = plot.WHITE

	ok1 := plot.bar_png(labels1, values1, "bar_simple.png", config1, allocator)
	fmt.printf("Simple bar chart: %v\n", ok1)

	// Test 2: Model coefficients (feature importance)
	labels2 := []string{"Age", "Income", "Score", "Experience", "Education"}
	values2 := []f64{0.45, 0.72, 0.38, 0.91, 0.55}

	config2 := config1
	config2.title = "Feature Importance"
	config2.y_label = "Coefficient"
	config2.bar_color = plot.Color{0, 255, 150, 255} // Green

	ok2 := plot.bar_png(labels2, values2, "bar_coefficients.png", config2, allocator)
	fmt.printf("Coefficients bar chart: %v\n", ok2)

	// Test 3: Confusion matrix style (multiple categories)
	labels3 := []string{"Cat", "Dog", "Bird", "Fish", "Rabbit"}
	values3 := []f64{15.0, 28.0, 12.0, 8.0, 22.0}

	config3 := config1
	config3.title = "Pet Distribution"
	config3.y_label = "Count"
	config3.bar_color = plot.Color{255, 150, 0, 255} // Orange

	ok3 := plot.bar_png(labels3, values3, "bar_pets.png", config3, allocator)
	fmt.printf("Pet distribution: %v\n", ok3)

	// Test 4: Performance comparison
	labels4 := []string{"Model A", "Model B", "Model C", "Model D"}
	values4 := []f64{0.85, 0.92, 0.78, 0.88}

	config4 := config1
	config4.title = "Model Accuracy Comparison"
	config4.y_label = "Accuracy"
	config4.bar_color = plot.Color{200, 0, 255, 255} // Purple

	ok4 := plot.bar_png(labels4, values4, "bar_accuracy.png", config4, allocator)
	fmt.printf("Model comparison: %v\n", ok4)
}
