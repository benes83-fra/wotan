package tests

import plot "../wotan/plot"
import "core:fmt"
import "core:mem"
import "core:os"

// Regression tests for wotan/plot issues found while writing the plot guide:
// multi_line_png with a line whose xs and ys differ in length (skipped when
// drawing) indexed past ys while placing the legend (bounds-check panic);
// tick labels of a step below 1e-6 got 0 decimals (1 .. 1 + 2e-12: every
// label "1"); scientific labels took their exponent from log10, which falls
// just short of some powers of ten (±1e300 was labelled "10e299").

plot_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== plot regression tests ===")
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	{ 	// fine steps: distinct labels
		ticks, step := plot.nice_ticks(1, 1 + 2e-12, 6, allocator)
		labels := plot.format_ticks(ticks, step, allocator)
		all_differ := len(labels) >= 3
		for i in 1 ..< len(labels) {all_differ = all_differ && labels[i] != labels[i - 1]}
		report(
			fmt.tprintf("ticks of 1 .. 1 + 2e-12 all differ: %v", labels),
			all_differ && labels[0] == "1.0000000000000",
			&passed,
			&total,
		)
	}
	{ 	// exponent at a power of ten
		ticks, step := plot.nice_ticks(-1e300, 1e300, 6, allocator)
		labels := plot.format_ticks(ticks, step, allocator)
		report(
			fmt.tprintf("ticks of ±1e300: %v", labels),
			len(labels) > 0 && labels[len(labels) - 1] == "1.0e300" && labels[0] == "−1.0e300",
			&passed,
			&total,
		)
	}
	{ 	// a line with fewer ys than xs is skipped, also by the legend (last: it used to panic)
		xs := []f64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
		ys := []f64{0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
		lines := []plot.LineData {
			{xs = xs, ys = ys[:4], label = "short"},
			{xs = xs, ys = ys, label = "ok"},
		}
		path := "plots/plot_fixes_mismatch.png"
		os.make_directory("plots")
		ok := plot.multi_line_png(lines, path, plot.DEFAULT_PLOT_CONFIG, allocator)
		report("multi_line_png with a mismatched line: drawn, no panic", ok, &passed, &total)
		_ = os.remove(path)
	}

	fmt.printf("plot regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
