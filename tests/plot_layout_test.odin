package tests

import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

// Plot layout and text contracts:
// - nice ticks are 1-2-5 multiples inside the range; labels share one precision,
//   use a real minus sign and switch to scientific for tiny / huge axes
// - the embedded font covers ASCII, punctuation and the Greek letters used in titles
// - margins grow to fit the tick labels: nothing is clipped at the image edges
// - the legend goes to the plot corner with the fewest data points

plot_layout_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Plot Layout Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// 1. nice ticks
	ticks, step := plot.nice_ticks(0.37, 62.2, 6, allocator)
	defer delete(ticks, allocator)
	expect(&passed, &total, step == 10 && len(ticks) == 6 && ticks[0] == 10 && ticks[5] == 60,
		fmt.tprintf("nice ticks for [0.37, 62.2]: step %v, %v", step, ticks))

	// 2. labels: shared precision, minus sign, scientific
	l1 := plot.format_ticks([]f64{-0.5, 0, 0.5, 1.0}, 0.5, allocator)
	l2 := plot.format_ticks([]f64{-2e-4, 0, 2e-4}, 2e-4, allocator)
	defer {delete(l1, allocator); delete(l2, allocator)}
	expect(&passed, &total, l1[0] == "−0.5" && l1[1] == "0.0" && l1[3] == "1.0" && l2[0] == "−2e−4" && l2[1] == "0" && l2[2] == "2e−4",
		fmt.tprintf("tick labels %v / %v", l1, l2))

	// 3. font coverage
	f := plot.font_get(13)
	_, ok_sigma := plot.font_glyph(f, 'σ')
	_, ok_lambda := plot.font_glyph(f, 'λ')
	_, ok_paren := plot.font_glyph(f, '(')
	_, ok_pct := plot.font_glyph(f, '%')
	expect(&passed, &total, f.size == 13 && ok_sigma && ok_lambda && ok_paren && ok_pct &&
		plot.text_width(f, "WWW") > plot.text_width(f, "iii"),
		fmt.tprintf("font: 13 px, Greek and punctuation glyphs, proportional widths"))

	// 4. margins fit the labels: wide y labels, last x label inside the image
	img := plot.image_new(800, 600, allocator)
	defer plot.image_free(&img)
	cfg := plot.DEFAULT_PLOT_CONFIG
	cfg.title, cfg.x_label, cfg.y_label = "Wide labels", "x", "y"
	ax := plot.axes_setup(&img, cfg, 0, 987654, -123456, 4567890)
	widest := 0.0
	for s in ax.ylabels {widest = max(widest, plot.text_width(ax.font, s))}
	last_x := plot.axes_px(&ax, ax.xticks[len(ax.xticks) - 1]) + plot.text_width(ax.font, ax.xlabels[len(ax.xlabels) - 1]) / 2
	expect(&passed, &total, f64(ax.left) > widest + f64(plot.line_height(ax.font)) && last_x < 800 && ax.top > plot.line_height(ax.title_font),
		fmt.tprintf("margins fit labels (left %d > widest y label %.0f + y title; last x label ends at %.0f < 800)", ax.left, widest, last_x))

	// 5. legend avoids the data: points only in the top-right quadrant
	xs := make([]f64, 50)
	ys := make([]f64, 50)
	defer {delete(xs); delete(ys)}
	for i in 0 ..< 50 {xs[i], ys[i] = 0.8 + 0.004 * f64(i), 0.8 + 0.004 * f64(i)}
	anchor_x := []f64{0, 1}
	anchor_y := []f64{0, 1}
	img2 := plot.image_new(400, 300, allocator)
	defer plot.image_free(&img2)
	plot.image_clear(&img2, 255, 255, 255, 255)
	ax2 := plot.axes_setup(&img2, plot.DEFAULT_PLOT_CONFIG, 0, 1, 0, 1)
	entries := []plot.Legend_Entry{{"series", plot.BLUE, .Solid, false}}
	pts_x := [][]f64{xs, anchor_x}
	pts_y := [][]f64{ys, anchor_y}
	plot.axes_draw_legend(&ax2, entries, pts_x, pts_y, allocator)
	// count drawn (non-white) pixels in the top-left / top-right quarters of the plot box
	drawn :: proc(img: ^plot.Image, x0, x1, y0, y1: int) -> int {
		n := 0
		for y in y0 ..< y1 {
			for x in x0 ..< x1 {
				if img.pixels[(y * img.width + x) * 4] != 255 {n += 1}
			}
		}
		return n
	}
	mid_x, mid_y := (ax2.left + ax2.right) / 2, (ax2.top + ax2.bottom) / 2
	n_tl := drawn(&img2, ax2.left + 1, mid_x, ax2.top + 1, mid_y)
	n_tr := drawn(&img2, mid_x, ax2.right, ax2.top + 1, mid_y)
	expect(&passed, &total, n_tl > 0 && n_tr == 0,
		fmt.tprintf("legend placed in the empty top-left corner, not over the data (%d / %d drawn px)", n_tl, n_tr))

	// 6. anti-aliased line: partial coverage at the edges, dashes leave gaps
	cv := plot.coverage_new(&ax2, allocator)
	defer plot.coverage_free(&cv, allocator)
	lx := []f64{f64(ax2.left + 10), f64(ax2.right - 10)}
	ly := []f64{f64(ax2.top + 20), f64(ax2.top + 20)}
	plot.cov_polyline(&cv, lx, ly, 0.75, .Dashed)
	row := (ax2.top + 20 - cv.y0) * cv.w
	edge := (ax2.top + 21 - cv.y0) * cv.w
	full, empty, partial := 0, 0, 0
	for x in 0 ..< cv.w {
		if cv.a[row + x] >= 0.999 {full += 1} else if cv.a[row + x] < 1e-6 {empty += 1}
		if cv.a[edge + x] > 0.01 && cv.a[edge + x] < 0.99 {partial += 1}
	}
	expect(&passed, &total, full > 50 && empty > 50 && partial > 50,
		fmt.tprintf("dashed AA line: %d covered / %d gap pixels on its row, %d soft edge pixels", full, empty, partial))

	fmt.printf("Plot layout summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
