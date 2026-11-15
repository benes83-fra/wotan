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
// smallest font, because its unnamed fields were zero; tick labels of very
// narrow ranges had no offset (1.0000000000005); heatmap_png ignored x_label
// and y_label and drew cell values with a hyphen as minus.

// matplotlib 3.7.0 ScalarFormatter offsets (threshold 4): {first, step, n, offset},
// ticks[i] = (first + i) * step
@(private = "file")
PF_MPL_OFFSETS := [?][4]f64 {
	{2000000000000.0, 5e-13, 5, 1.000000000001},
	{10001.0, 0.1, 5, 1000.0},
	{-10005.0, 0.1, 5, -1000.0},
	{2000.0, 2, 6, 0.0},
	{400.0, 5, 5, 0.0},
	{100000001.0, 0.01, 5, 1000000.0},
	{99.0, 1, 5, 0.0},
	{-2.0, 0.0001, 6, 0.0},
	{61728390.0, 0.2, 6, 12345670.0},
	{19999999.0, 5e-08, 4, 1.0},
	{-1000003.0, 1, 6, -1000000.0},
	{499998.0, 0.2, 5, 100000.0},
	{50000.0, 1e-11, 5, 5e-07},
	{89.0, 20000, 6, 0.0},
	{-53387110733.0, 0.002, 7, -106774221.4},
	{-66426857.0, 5, 5, -332134200.0},
	{-27382237749.0, 5e-12, 4, -0.1369111887},
	{-588.0, 0.001, 4, 0.0},
	{1219993757379.0, 5.0000000000000005e-17, 7, 6.099968786e-05},
	{387.0, 5e-09, 3, 0.0},
	{-405308044780.0, 2e-05, 5, -8106160.8950000005},
	{-7782856.0, 2e-09, 4, -0.0155657},
	{-180954224644.0, 1e-15, 5, -0.00018095422464},
	{507728547150.0, 2e-14, 6, 0.01015457094},
	{-2236.0, 1e-06, 4, 0.0},
	{-7871630383488.0, 5e-10, 6, -3935.81519174},
	{5765196599.0, 0.005, 5, 28825983.0},
	{41412782387.0, 2e-10, 6, 8.28255647},
	{-514229714.0, 10, 6, -5142297000.0},
	{48440562578463.0, 5.000000000000001e-15, 7, 0.2422028128923},
	{-75.0, 5000, 6, 0.0},
	{-77.0, 50000, 5, 0.0},
	{30328960153.0, 5e-09, 4, 151.6448007},
	{6812.0, 1e-09, 6, 0.0},
	{4341596795560.0, 1e-19, 6, 4.3415967955000006e-07},
	{727143994511.0, 1e-10, 5, 72.714399451},
	{5103902601.0, 0.5, 4, 2551951300.0},
	{531297.0, 5e-08, 3, 0.026563999999999997},
	{10712372484483.0, 0.002, 4, 21424744968.9},
	{2866753500.0, 1e-16, 6, 2.8667535000000004e-07},
	{7635444015816.0, 1e-12, 4, 7.635444015809999},
	{-107857773003.0, 0.0001, 4, -10785777.3},
	{-1799897478472.0, 1e-08, 4, -17998.974783999998},
	{395654841.0, 2e-10, 7, 0.07913096},
	{-1437.0, 0.01, 7, 0.0},
	{6896672802.0, 2, 6, 13793345600.0},
	{802366806509.0, 1e-14, 5, 0.008023668065},
	{362535882476.0, 0.0001, 7, 36253588.24},
	{21872152.0, 5, 3, 109360700.0},
	{2230925018.0, 5e-16, 6, 1.1154625e-06},
	{-100692.0, 20, 6, -2013000.0},
	{34.0, 1000000, 6, 0.0},
	{340019925982.0, 0.02, 5, 6800398519.0},
	{-5490357800.0, 5e-09, 6, -27.451788999999998},
	{359437.0, 5e-05, 4, 17.97},
	{24844.0, 4.9999999999999996e-06, 4, 0.1242},
	{855.0, 0.0002, 7, 0.0},
	{17857951.0, 2000, 6, 35715900000.0},
	{-803.0, 2e-07, 4, 0.0},
	{41726.0, 5, 5, 208600.0},
	{-23845513.0, 0.0005, 7, -11922.75},
	{836.0, 2e-05, 6, 0.0},
	{579232912596.0, 2e-07, 6, 115846.5825},
	{22471.0, 0.2, 6, 0.0},
	{-62439540.0, 5e-05, 5, -3121.9700000000003},
	{-1737632368093.0, 5.0000000000000005e-17, 4, -8.6881618404e-05},
	{-289285083.0, 1e-07, 4, -28.928500000000003},
	{2278624.0, 2e-07, 5, 0.45572},
	{-1833739589.0, 0.0002, 5, -366747.917},
	{111870.0, 2000, 4, 223740000.0},
	{6793834091.0, 2, 4, 13587668180.0},
	{2641785.0, 0.0001, 5, 264.178},
	{2581.0, 10000, 4, 0.0},
}

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

	{ 	// tick offsets = matplotlib's ScalarFormatter (threshold 4) on 73 tick sets
		agree, with_offset := 0, 0
		worst := ""
		for c in PF_MPL_OFFSETS {
			n := int(c[2])
			ticks := make([]f64, n, allocator)
			for i in 0 ..< n {ticks[i] = (c[0] + f64(i)) * c[1]}
			_, off, text := plot.format_ticks_offset(ticks, c[1], allocator)
			ok := abs(off - c[3]) <= 1e-15 * abs(c[3]) && (text != "") == (c[3] != 0)
			if ok {agree += 1} else if worst == "" {worst = fmt.tprintf(" first miss %v: %v", c, off)}
			if c[3] != 0 {with_offset += 1}
		}
		report(
			fmt.tprintf(
				"tick offsets equal matplotlib's: %d/%d (%d with an offset)%s",
				agree,
				len(PF_MPL_OFFSETS),
				with_offset,
				worst,
			),
			agree == len(PF_MPL_OFFSETS),
			&passed,
			&total,
		)
	}
	{ 	// labels relative to the offset, and the offset text
		Case :: struct {
			lo, hi: f64,
			labels: []string,
			text:   string,
		}
		cases := []Case {
			// matplotlib: offset 1.000000000001 (it straddles 1 by a tiny span)
			{
				1,
				1 + 2e-12,
				{"−1.0e−12", "−0.5e−12", "0", "0.5e−12", "1.0e−12"},
				"+1.000000000001",
			},
			{1000.1, 1000.5, {"0.1", "0.2", "0.3", "0.4", "0.5"}, "+1000"},
			{-1000.5, -1000.1, {"−0.5", "−0.4", "−0.3", "−0.2", "−0.1"}, "−1000"},
			{12345678, 12345679, {"8.0", "8.2", "8.4", "8.6", "8.8", "9.0"}, "+1.234567e7"},
			{2000, 2020, {"2000", "2005", "2010", "2015", "2020"}, ""},
		}
		bad := ""
		for c in cases {
			ticks, step := plot.nice_ticks(c.lo, c.hi, 6, allocator)
			labels, _, text := plot.format_ticks_offset(ticks, step, allocator)
			same := len(labels) == len(c.labels) && text == c.text
			for s, i in c.labels {same = same && labels[i] == s}
			if !same {bad = fmt.tprintf("%s [%v, %v]: %v %q;", bad, c.lo, c.hi, labels, text)}
		}
		report(
			fmt.tprintf(
				"offset labels and texts (1 + 2e-12 → +1.000000000001, 1000.x → +1000, −1000.x → −1000, 12345678 → +1.234567e7, years: none)%s",
				bad,
			),
			bad == "",
			&passed,
			&total,
		)
	}
	{ 	// axes_setup: an axis with an offset gets a row for it (y above the box,
		// x below its labels)
		img := plot.image_new(640, 480, allocator)
		defer plot.image_free(&img)
		cfg := plot.DEFAULT_PLOT_CONFIG
		plain := plot.axes_setup(&img, cfg, 0, 10, 0, 10)
		fine := plot.axes_setup(&img, cfg, 1000, 1000.5, 1, 1 + 1e-12)
		lh := plot.line_height(fine.font)
		// the x offset row (below the tick labels) fits in the image
		x_row_end := fine.bottom + fine.tick_len + fine.pad / 2 + 2 * lh + fine.pad / 2
		report(
			fmt.tprintf(
				"axes offsets: x %q, y %q; plot box top %d → %d (one line for the y offset), bottom %d → %d (x offset row ends at %d of 480)",
				fine.xoffset,
				fine.yoffset,
				plain.top,
				fine.top,
				plain.bottom,
				fine.bottom,
				x_row_end,
			),
			plain.xoffset == "" &&
			plain.yoffset == "" &&
			fine.xoffset == "+1000" &&
			fine.yoffset != "" &&
			fine.top == plain.top + lh &&
			fine.bottom < plain.bottom &&
			x_row_end <= 480,
			&passed,
			&total,
		)
	}
	{ 	// heatmap_png draws x_label and y_label
		os.make_directory("plots")
		data := []f64{1, -0.35, -0.35, 1}
		paths := [3]string {
			"plots/plot_fixes_hm0.png",
			"plots/plot_fixes_hm1.png",
			"plots/plot_fixes_hm2.png",
		}
		cfg := plot.DEFAULT_PLOT_CONFIG
		cfg.width, cfg.height = 400, 300
		ok := plot.heatmap_png(data, 2, 2, paths[0], {"a", "b"}, {"a", "b"}, cfg)
		cfg.x_label = "column factor"
		ok &&= plot.heatmap_png(data, 2, 2, paths[1], {"a", "b"}, {"a", "b"}, cfg)
		cfg.x_label, cfg.y_label = "", "row factor"
		ok &&= plot.heatmap_png(data, 2, 2, paths[2], {"a", "b"}, {"a", "b"}, cfg)
		pngs: [3][]u8
		for p, i in paths {pngs[i], _ = os.read_entire_file(p, allocator)}
		differ :=
			len(pngs[0]) > 0 &&
			string(pngs[0]) != string(pngs[1]) &&
			string(pngs[0]) != string(pngs[2])
		for p, i in paths {
			delete(pngs[i], allocator)
			_ = os.remove(p)
		}
		report(
			fmt.tprintf("heatmap_png draws x_label and y_label: %v", ok && differ),
			ok && differ,
			&passed,
			&total,
		)
	}

	fmt.printf("plot regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
