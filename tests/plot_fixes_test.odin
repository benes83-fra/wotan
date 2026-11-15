package tests

import w "../wotan/core"
import plot "../wotan/plot"
import "base:runtime"
import "core:fmt"
import "core:mem"
import "core:os"

// Regression tests for wotan/plot issues found while writing the plot guide:
// multi_line_png with a line whose xs and ys differ in length (skipped when
// drawing) indexed past ys while placing the legend (bounds-check panic);
// tick labels of a step below 1e-6 got 0 decimals (1 .. 1 + 2e-12: every
// label "1"); scientific labels took their exponent from log10, which falls
// just short of some powers of ten (±1e300 was labelled "10e299");
// scatter_png / histogram_png panicked on a missing column name; every chart
// left its ticks, labels and PNG bytes on context.temp_allocator; a
// PlotConfig literal drew transparent black backgrounds, black text and the
// smallest font, because its unnamed fields were zero.

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

	{ 	// scatter_png / histogram_png: a missing column returns false like a non-Float one
		df := w.df_from(
			w.column_from_floats("a", []f64{1, 2, 3}),
			w.column_from_floats("b", []f64{2, 1, 3}),
			w.column_from_ints("k", []int{1, 2, 3}),
		)
		defer w.destroy_dataframe(&df)
		os.make_directory("plots")
		path := "plots/plot_fixes_missing.png"
		s_missing := plot.scatter_png(&df, "a", "nope", path, plot.DEFAULT_PLOT_CONFIG, allocator)
		s_missing_x := plot.scatter_png(
			&df,
			"nope",
			"b",
			path,
			plot.DEFAULT_PLOT_CONFIG,
			allocator,
		)
		s_int := plot.scatter_png(&df, "a", "k", path, plot.DEFAULT_PLOT_CONFIG, allocator)
		h_missing := plot.histogram_png(&df, "nope", path, 10, plot.DEFAULT_PLOT_CONFIG, allocator)
		h_int := plot.histogram_png(&df, "k", path, 10, plot.DEFAULT_PLOT_CONFIG, allocator)
		s_ok := plot.scatter_png(&df, "a", "b", path, plot.DEFAULT_PLOT_CONFIG, allocator)
		h_ok := plot.histogram_png(&df, "a", path, 10, plot.DEFAULT_PLOT_CONFIG, allocator)
		report(
			"scatter_png / histogram_png: missing or non-Float column -> false, valid -> true",
			!s_missing && !s_missing_x && !s_int && !h_missing && !h_int && s_ok && h_ok,
			&passed,
			&total,
		)
		_ = os.remove(path)
	}
	{ 	// a PlotConfig literal: unnamed (zero) fields take the defaults
		report(
			"config_resolve(PlotConfig{}) == DEFAULT_PLOT_CONFIG",
			plot.config_resolve(plot.PlotConfig{}) == plot.DEFAULT_PLOT_CONFIG,
			&passed,
			&total,
		)
		dark := plot.PlotConfig {
			bg_color   = plot.BLACK,
			axis_color = plot.WHITE,
			grid_color = plot.GREEN,
		}
		report(
			"set fields are kept: explicit grid colour, dark theme",
			plot.config_resolve(dark).bg_color == plot.BLACK &&
			plot.grid_color_of(dark) == plot.GREEN &&
			plot.grid_color_of(plot.PlotConfig{bg_color = plot.BLACK}) == plot.Color{5, 5, 5, 255},
			&passed,
			&total,
		)

		lit := plot.PlotConfig {
			width     = 400,
			height    = 300,
			title     = "literal",
			show_grid = true,
		}
		ref := plot.DEFAULT_PLOT_CONFIG
		ref.width, ref.height, ref.title, ref.show_grid = 400, 300, "literal", true
		df := w.df_from(
			w.column_from_floats("a", []f64{1, 2, 3, 4, 5}),
			w.column_from_floats("b", []f64{2, 1, 3, 5, 4}),
		)
		defer w.destroy_dataframe(&df)
		xs := []f64{0, 1, 2, 3, 4}
		ys := []f64{1, 3, 2, 5, 4}
		os.make_directory("plots")
		p_lit, p_ref := "plots/plot_fixes_lit.png", "plots/plot_fixes_ref.png"
		same := 0
		for i in 0 ..< 6 {
			for k in 0 ..< 2 {
				cfg := lit if k == 0 else ref
				path := p_lit if k == 0 else p_ref
				switch i {
				case 0:
					plot.line_png(xs, ys, path, cfg)
				case 1:
					plot.multi_line_png({{xs = xs, ys = ys, label = "a"}}, path, cfg)
				case 2:
					plot.scatter_png(&df, "a", "b", path, cfg)
				case 3:
					plot.histogram_png(&df, "a", path, 5, cfg)
				case 4:
					plot.bar_png({"a", "b", "c"}, {1, -2, 3}, path, cfg)
				case 5:
					plot.heatmap_png(ys[:4], 2, 2, path, {"r1", "r2"}, {"c1", "c2"}, cfg)
				}
			}
			a, a_err := os.read_entire_file(p_lit, allocator)
			b, b_err := os.read_entire_file(p_ref, allocator)
			if a_err == nil && b_err == nil && len(a) > 0 && string(a) == string(b) {same += 1}
			delete(a, allocator)
			delete(b, allocator)
		}
		report(
			fmt.tprintf(
				"charts from a literal match DEFAULT_PLOT_CONFIG: %d/6 identical PNGs",
				same,
			),
			same == 6,
			&passed,
			&total,
		)
		_ = os.remove(p_lit)
		_ = os.remove(p_ref)
	}
	{ 	// 50 charts in a loop: the temp allocator does not grow (scratch released per chart)
		if context.temp_allocator.data != &runtime.global_default_temp_allocator_data {
			report(
				"temp allocator steady over 50 PNGs: skipped (non-default temp allocator)",
				true,
				&passed,
				&total,
			)
		} else {
			arena := &runtime.global_default_temp_allocator_data.arena
			_ = make([]u8, 16, context.temp_allocator) // the arena has a block before the loop
			df := w.df_from(
				w.column_from_floats("a", []f64{1, 2, 3, 4, 5}),
				w.column_from_floats("b", []f64{2, 1, 3, 5, 4}),
			)
			defer w.destroy_dataframe(&df)
			xs := []f64{0, 1, 2, 3, 4}
			ys := []f64{1, 3, 2, 5, 4}
			os.make_directory("plots")
			path := "plots/plot_fixes_loop.png"
			cfg := plot.DEFAULT_PLOT_CONFIG
			cfg.width, cfg.height = 400, 300
			cfg.title = "loop"
			before := arena.total_used
			all_ok := true
			for i in 0 ..< 50 {
				ok: bool
				switch i % 6 {
				case 0:
					ok = plot.line_png(xs, ys, path, cfg)
				case 1:
					ok = plot.multi_line_png({{xs = xs, ys = ys, label = "a"}}, path, cfg)
				case 2:
					ok = plot.scatter_png(&df, "a", "b", path, cfg)
				case 3:
					ok = plot.histogram_png(&df, "a", path, 5, cfg)
				case 4:
					ok = plot.bar_png({"a", "b", "c"}, {1, -2, 3}, path, cfg)
				case 5:
					ok = plot.heatmap_png(ys[:4], 2, 2, path, {"r1", "r2"}, {"c1", "c2"}, cfg)
				}
				all_ok = all_ok && ok
			}
			after := arena.total_used
			report(
				fmt.tprintf(
					"temp allocator steady over 50 PNGs: %d -> %d bytes used",
					before,
					after,
				),
				all_ok && after == before,
				&passed,
				&total,
			)
			_ = os.remove(path)
		}
	}

	fmt.printf("plot regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
