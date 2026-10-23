package tests

import plot "../wotan/plot"
import "core:bytes"
import "core:compress/zlib"
import "core:fmt"
import "core:image/png"
import "core:math"
import "core:math/rand"
import "core:mem"

// PNG compression contracts:
// - zlib_compress output inflates (core:compress/zlib) back to the input for
//   empty, tiny, incompressible, run-heavy and multi-block data
// - encoded PNGs decode (core:image/png) to exactly the source pixels, on the
//   opaque RGB path and on the RGBA path
// - a rendered plot compresses far below its raw size

png_compress_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== PNG Compression Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	roundtrip :: proc(data: []u8, allocator: mem.Allocator) -> (ok: bool, ratio: f64) {
		z := plot.zlib_compress(data, allocator)
		defer delete(z, allocator)
		buf: bytes.Buffer
		bytes.buffer_init_allocator(&buf, 0, len(data) + 16, allocator)
		defer bytes.buffer_destroy(&buf)
		if err := zlib.inflate(z, &buf); err != nil {return false, 0}
		out := bytes.buffer_to_bytes(&buf)
		if len(out) != len(data) {return false, 0}
		for v, i in data {if out[i] != v {return false, 0}}
		return true, f64(len(z)) / max(f64(len(data)), 1)
	}

	// 1. zlib round trips
	cases: [6][]u8
	cases[0] = nil
	cases[1] = make([]u8, 1)
	cases[1][0] = 42
	cases[2] = make([]u8, 100_000) // incompressible
	for &v in cases[2] {v = u8(rand.uint32())}
	cases[3] = make([]u8, 300_000) // long runs
	for &v, i in cases[3] {v = u8((i / 7000) % 3)}
	cases[4] = make([]u8, 200_000) // text-like: small alphabet, repeats at many distances
	words := []string{"the ", "volatility ", "surface ", "of ", "GARCH(1,1) ", "and ", "λ=0.94 "}
	for i := 0; i < len(cases[4]); {
		w := words[rand.int_max(len(words))]
		for b in transmute([]u8)w {if i < len(cases[4]) {cases[4][i] = b; i += 1}}
	}
	cases[5] = make([]u8, 1_000_000) // > 64K tokens: several blocks, mixed content
	for &v, i in cases[5] {v = u8(i * 7 % 251) if (i / 50_000) % 2 == 0 else u8(rand.uint32() & 3)}
	defer for c in cases {delete(c)}
	names := []string{"empty", "1 byte", "100 KB random", "300 KB runs", "200 KB text", "1 MB multi-block"}
	all_ok := true
	report := ""
	for c, i in cases {
		ok, ratio := roundtrip(c, allocator)
		all_ok &&= ok
		report = fmt.tprintf("%s%s %.3f%s; ", report, names[i], ratio, "" if ok else " FAILED")
	}
	expect(&passed, &total, all_ok, fmt.tprintf("zlib round trips (compressed/raw: %s)", report))

	// 2. PNG decode equals the source, opaque (RGB) and translucent (RGBA)
	check_png :: proc(img: ^plot.Image, allocator: mem.Allocator) -> (ok: bool, channels: int, size: int) {
		data := plot.image_encode_png(img, allocator)
		defer delete(data, allocator)
		dec, err := png.load_from_bytes(data, {}, allocator)
		if err != nil {return false, 0, 0}
		defer png.destroy(dec)
		px := bytes.buffer_to_bytes(&dec.pixels)
		if dec.width != img.width || dec.height != img.height || dec.depth != 8 {return false, 0, 0}
		ch := dec.channels
		for i in 0 ..< img.width * img.height {
			for k in 0 ..< ch {
				if px[i * ch + k] != img.pixels[i * 4 + k] {return false, ch, len(data)}
			}
		}
		return true, ch, len(data)
	}
	img := plot.image_new(320, 200, allocator)
	defer plot.image_free(&img)
	for y in 0 ..< img.height {
		for x in 0 ..< img.width {
			i := (y * img.width + x) * 4
			img.pixels[i] = u8(x)
			img.pixels[i + 1] = u8(y * 3)
			img.pixels[i + 2] = u8((x ~ y) & 0xFF)
			img.pixels[i + 3] = 255
		}
	}
	ok_rgb, ch_rgb, _ := check_png(&img, allocator)
	for i in 0 ..< img.width * img.height {img.pixels[i * 4 + 3] = u8(i % 256)}
	ok_rgba, ch_rgba, _ := check_png(&img, allocator)
	expect(&passed, &total, ok_rgb && ch_rgb == 3 && ok_rgba && ch_rgba == 4,
		fmt.tprintf("PNG decodes to the source pixels (opaque -> %d channels, alpha -> %d channels)", ch_rgb, ch_rgba))

	// 3. a rendered plot: exact pixels and a small file
	n := 500
	xs := make([]f64, n)
	ys := make([]f64, n)
	defer {delete(xs); delete(ys)}
	for i in 0 ..< n {xs[i], ys[i] = f64(i), math.sin(f64(i) / 40) + 0.1 * rand.norm_float64()}
	canvas := plot.image_new(800, 600, allocator)
	defer plot.image_free(&canvas)
	plot.image_clear(&canvas, 255, 255, 255, 255)
	cfg := plot.DEFAULT_PLOT_CONFIG
	cfg.title, cfg.x_label, cfg.y_label, cfg.show_grid = "Compression check", "x", "y", true
	ax := plot.axes_setup(&canvas, cfg, 0, f64(n - 1), -1.5, 1.5)
	plot.axes_draw_grid(&ax)
	cv := plot.coverage_new(&ax, allocator)
	defer plot.coverage_free(&cv, allocator)
	px := make([]f64, n)
	py := make([]f64, n)
	defer {delete(px); delete(py)}
	for i in 0 ..< n {px[i], py[i] = plot.axes_px(&ax, xs[i]), plot.axes_py(&ax, ys[i])}
	plot.cov_polyline(&cv, px, py, 0.75, .Solid)
	plot.cov_flush(&canvas, &cv, plot.BLUE)
	plot.axes_draw_frame(&ax)
	ok_plot, _, size := check_png(&canvas, allocator)
	raw := 800 * 600 * 4
	expect(&passed, &total, ok_plot && size < raw / 20,
		fmt.tprintf("rendered 800x600 plot: %d bytes vs %d raw (%.1f%%), decodes exactly", size, raw, 100 * f64(size) / f64(raw)))

	fmt.printf("PNG compression summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
