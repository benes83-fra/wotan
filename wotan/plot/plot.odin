package plot

import w "../core"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:sync"

LineStyle :: enum {
	Solid,
	Dashed,
	Dotted,
}

PlotConfig :: struct {
	width:         int,
	height:        int,
	margin_left:   int, // minimum margins; grown to fit tick / axis labels
	margin_right:  int,
	margin_top:    int,
	margin_bottom: int,
	point_color:   Color, // Color for scatter points / single line plots
	bar_color:     Color, // Color for bars and histogram bars
	axis_color:    Color, // Color for axes and text
	bg_color:      Color, // Background color
	font_scale:    int, // 1 small .. 5 large (used when font_size == 0)
	title:         string,
	x_label:       string,
	y_label:       string,
	line_style:    LineStyle, // Line style for line plots
	show_grid:     bool, // Show grid lines at the ticks
	grid_color:    Color, // Color for grid lines
	font_size:     int, // body text size in pixels (0 = from font_scale)
	line_width:    f64, // data line width in pixels (0 = 1.5)
}

DEFAULT_PLOT_CONFIG :: PlotConfig {
	width         = 800,
	height        = 600,
	margin_left   = 60,
	margin_right  = 20,
	margin_top    = 20,
	margin_bottom = 40,
	point_color   = RED,
	bar_color     = BLUE,
	axis_color    = Color{40, 40, 40, 255},
	bg_color      = WHITE,
	font_scale    = 2,
	title         = "",
	x_label       = "",
	y_label       = "",
	line_style    = .Solid,
	show_grid     = false,
	grid_color    = Color{225, 225, 225, 255},
	font_size     = 0,
	line_width    = 1.5,
}
// ============================================================================
// 1. Software Rasterizer & Image Buffer
// ============================================================================

LineData :: struct {
	xs:    []f64,
	ys:    []f64,
	color: Color, // zero value = next palette colour
	style: LineStyle,
	label: string, // Used for the legend
}
Color :: struct {
	r, g, b, a: u8,
}
WHITE :: Color{255, 255, 255, 255}
BLACK :: Color{0, 0, 0, 255}
// Named colours follow the matplotlib "tab10" palette (readable on white)
RED :: Color{214, 39, 40, 255}
BLUE :: Color{31, 119, 180, 255}
GREEN :: Color{44, 160, 44, 255}
ORANGE :: Color{255, 127, 14, 255}
PURPLE :: Color{148, 103, 189, 255}
BROWN :: Color{140, 86, 75, 255}
PINK :: Color{227, 119, 194, 255}
GRAY :: Color{127, 127, 127, 255}
OLIVE :: Color{188, 189, 34, 255}
CYAN :: Color{23, 190, 207, 255}
PALETTE :: [10]Color{BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GRAY, OLIVE, CYAN}

Image :: struct {
	width:     int,
	height:    int,
	pixels:    []u8, // RGBA format
	allocator: mem.Allocator,
}

image_new :: proc(w, h: int, allocator: mem.Allocator = context.allocator) -> Image {
	return Image {
		width = w,
		height = h,
		pixels = make([]u8, w * h * 4, allocator),
		allocator = allocator,
	}
}

image_free :: proc(img: ^Image) {
	if img.pixels != nil {
		delete(img.pixels, img.allocator)
		img.pixels = nil
	}
}

image_clear :: proc(img: ^Image, r, g, b, a: u8) {
	for i in 0 ..< len(img.pixels) / 4 {
		img.pixels[i * 4] = r
		img.pixels[i * 4 + 1] = g
		img.pixels[i * 4 + 2] = b
		img.pixels[i * 4 + 3] = a
	}
}

draw_pixel :: proc(img: ^Image, x, y: int, c: Color) {
	if x < 0 || x >= img.width || y < 0 || y >= img.height {return}
	idx := (y * img.width + x) * 4
	img.pixels[idx] = c.r
	img.pixels[idx + 1] = c.g
	img.pixels[idx + 2] = c.b
	img.pixels[idx + 3] = c.a
}

// Bresenham's Line Algorithm (crisp 1-px lines for frames and grids)
draw_line :: proc(img: ^Image, x0, y0, x1, y1: int, c: Color) {
	x0 := x0
	y0 := y0
	dx := math.abs(x1 - x0)
	dy := -math.abs(y1 - y0)

	sx: int
	if x0 < x1 {sx = 1} else {sx = -1}

	sy: int
	if y0 < y1 {sy = 1} else {sy = -1}

	err := dx + dy

	for {
		draw_pixel(img, x0, y0, c)
		if x0 == x1 && y0 == y1 {break}
		e2 := 2 * err
		if e2 >= dy {
			if x0 == x1 {break}
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			if y0 == y1 {break}
			err += dx
			y0 += sy
		}
	}
}

// Text at (x, y) = top-left of the line box, in the font size of `scale`
draw_text :: proc(img: ^Image, x, y: int, text: string, scale: int, c: Color) {
	draw_string(img, font_get(font_size_for_scale(scale)), f64(x), f64(y), text, c)
}

draw_rect :: proc(img: ^Image, x, y, w, h: int, c: Color) {
	for i in 0 ..< w {
		for j in 0 ..< h {
			draw_pixel(img, x + i, y + j, c)
		}
	}
}

draw_char :: proc(img: ^Image, x, y: int, ch: u8, scale: int, c: Color) {
	buf := [1]u8{ch}
	draw_text(img, x, y, string(buf[:]), scale, c)
}

draw_number :: proc(img: ^Image, x, y: int, val: f64, scale: int, c: Color) {
	draw_text(img, x, y, fmt.tprintf("%.2f", val), scale, c)
}

// Anti-aliased line with the given style (single segment)
draw_line_segment :: proc(img: ^Image, x0, y0, x1, y1: int, c: Color, style: LineStyle) {
	ax := Axes {
		img    = img,
		left   = -1,
		top    = -1,
		right  = img.width,
		bottom = img.height,
	}
	cv := coverage_new(&ax, context.temp_allocator)
	defer coverage_free(&cv, context.temp_allocator)
	xs := [2]f64{f64(x0), f64(x1)}
	ys := [2]f64{f64(y0), f64(y1)}
	cov_polyline(&cv, xs[:], ys[:], 0.75, style)
	cov_flush(img, &cv, c)
}

// ============================================================================
// 3. Pure Odin PNG Encoder (with built-in CRC32 and Adler32)
// ============================================================================

@(private = "file")
_crc_table: [256]u32
@(private = "file")
_crc_once: sync.Once

@(private = "file")
_build_crc_table :: proc() {
	for i in 0 ..< 256 {
		crc := u32(i)
		for _ in 0 ..< 8 {
			crc = (crc >> 1) ~ 0xEDB88320 if (crc & 1) != 0 else crc >> 1
		}
		_crc_table[i] = crc
	}
}

// Running CRC-32 (start with crc = 0, feed chunks in order)
crc32_update :: proc(crc: u32, data: []u8) -> u32 {
	sync.once_do(&_crc_once, _build_crc_table)
	c := ~crc
	for b in data {c = (c >> 8) ~ _crc_table[(c ~ u32(b)) & 0xFF]}
	return ~c
}

crc32_checksum :: proc(data: []u8) -> u32 {return crc32_update(0, data)}

_adler32_checksum :: proc(data: []u8) -> u32 {
	a: u32 = 1
	b: u32 = 0
	for i := 0; i < len(data); {
		// 5552 bytes keep the sums below 2^32 before the modulo (zlib's NMAX)
		end := min(i + 5552, len(data))
		for v in data[i:end] {
			a += u32(v)
			b += a
		}
		a %= 65521
		b %= 65521
		i = end
	}
	return (b << 16) | a
}

png_write_chunk :: proc(f: ^os.File, chunk_type: string, data: []u8) {
	data_len := u32(len(data))
	b_len := [4]u8{u8(data_len >> 24), u8(data_len >> 16), u8(data_len >> 8), u8(data_len)}
	_, _ = os.write(f, b_len[:])

	type_bytes := transmute([]u8)chunk_type
	_, _ = os.write(f, type_bytes)
	if data_len > 0 {
		_, _ = os.write(f, data)
	}

	crc := crc32_update(crc32_update(0, type_bytes), data)
	b_crc := [4]u8{u8(crc >> 24), u8(crc >> 16), u8(crc >> 8), u8(crc)}
	_, _ = os.write(f, b_crc[:])
}

// Encode the image as PNG bytes: RGB when fully opaque (else RGBA), adaptive
// per-row filters, DEFLATE-compressed IDAT. Result allocated with `allocator`.
image_encode_png :: proc(img: ^Image, allocator := context.allocator) -> []u8 {
	n_px := img.width * img.height
	opaque := true
	for i in 0 ..< n_px {
		if img.pixels[i * 4 + 3] != 255 {opaque = false; break}
	}
	bpp := 3 if opaque else 4
	src := img.pixels
	if opaque {
		rgb := make([]u8, n_px * 3, context.temp_allocator)
		for i in 0 ..< n_px {
			rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2] =
				img.pixels[i * 4], img.pixels[i * 4 + 1], img.pixels[i * 4 + 2]
		}
		src = rgb
	}
	filtered := png_filter_rows(src, img.width, img.height, bpp, context.temp_allocator)
	idat := zlib_compress(filtered, context.temp_allocator)
	defer {
		if opaque {delete(src, context.temp_allocator)}
		delete(filtered, context.temp_allocator)
		delete(idat, context.temp_allocator)
	}

	out := make([dynamic]u8, 0, len(idat) + 64, allocator)
	chunk :: proc(out: ^[dynamic]u8, kind: string, data: []u8) {
		n := u32(len(data))
		append(out, u8(n >> 24), u8(n >> 16), u8(n >> 8), u8(n))
		append(out, ..transmute([]u8)kind)
		append(out, ..data)
		crc := crc32_update(crc32_update(0, transmute([]u8)kind), data)
		append(out, u8(crc >> 24), u8(crc >> 16), u8(crc >> 8), u8(crc))
	}
	append(&out, 137, 80, 78, 71, 13, 10, 26, 10)
	ihdr := [13]u8 {
		u8(img.width >> 24),
		u8(img.width >> 16),
		u8(img.width >> 8),
		u8(img.width),
		u8(img.height >> 24),
		u8(img.height >> 16),
		u8(img.height >> 8),
		u8(img.height),
		8,
		2 if opaque else 6,
		0,
		0,
		0, // 8-bit, RGB / RGBA, deflate, adaptive filters, no interlace
	}
	chunk(&out, "IHDR", ihdr[:])
	chunk(&out, "IDAT", idat)
	chunk(&out, "IEND", nil)
	return out[:]
}

image_save_png :: proc(img: ^Image, path: string) -> bool {
	png := image_encode_png(img, context.temp_allocator)
	defer delete(png, context.temp_allocator)
	return os.write_entire_file(path, png) == nil
}


// Maps a value between 0.0 and 1.0 to a color (perceptually ordered,
// viridis-like: dark blue -> teal -> green -> yellow)
get_heat_color :: proc(val: f64) -> Color {
	stops := [5][3]f64{{68, 1, 84}, {59, 82, 139}, {33, 145, 140}, {94, 201, 98}, {253, 231, 37}}
	v := clamp(val, 0, 1) if !math.is_nan(val) else 0
	pos := v * 4
	i := min(int(pos), 3)
	t := pos - f64(i)
	a, b := stops[i], stops[i + 1]
	return Color {
		u8(a[0] + (b[0] - a[0]) * t),
		u8(a[1] + (b[1] - a[1]) * t),
		u8(a[2] + (b[2] - a[2]) * t),
		255,
	}
}


// ============================================================================
// 4. Public Plotting API
// ============================================================================

@(private = "file")
_begin :: proc(cfg: PlotConfig, allocator: mem.Allocator) -> Image {
	img := image_new(cfg.width, cfg.height, allocator)
	image_clear(&img, cfg.bg_color.r, cfg.bg_color.g, cfg.bg_color.b, cfg.bg_color.a)
	return img
}

// Map data to pixel coordinates (NaN stays NaN and breaks lines)
@(private = "file")
_to_pixels :: proc(ax: ^Axes, xs, ys: []f64, allocator: mem.Allocator) -> (px, py: []f64) {
	n := min(len(xs), len(ys))
	px = make([]f64, n, allocator)
	py = make([]f64, n, allocator)
	for i in 0 ..< n {
		px[i] = axes_px(ax, xs[i])
		py[i] = axes_py(ax, ys[i])
	}
	return
}

scatter_png :: proc(
	df: ^w.DataFrame,
	x_col: string,
	y_col: string,
	path: string,
	config: PlotConfig = DEFAULT_PLOT_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	col_x := w.column(df, x_col)
	col_y := w.column(df, y_col)
	if col_x.type != .Float || col_y.type != .Float {return false}

	n := df.rows
	xs := make([]f64, n, allocator)
	ys := make([]f64, n, allocator)
	defer delete(xs, allocator)
	defer delete(ys, allocator)
	for i in 0 ..< n {
		xs[i], _ = w.column_at_float(col_x, i)
		ys[i], _ = w.column_at_float(col_y, i)
	}
	min_x, max_x := data_range(xs)
	min_y, max_y := data_range(ys)

	img := _begin(config, allocator)
	defer image_free(&img)
	ax := axes_setup(&img, config, min_x, max_x, min_y, max_y, {pad_frac = 0.05})
	axes_draw_grid(&ax)

	cv := coverage_new(&ax, allocator)
	defer coverage_free(&cv, allocator)
	radius := max(2.5, 1.6 * line_width_of(config))
	for i in 0 ..< n {
		px, py := axes_px(&ax, xs[i]), axes_py(&ax, ys[i])
		if !math.is_nan(px) && !math.is_nan(py) {cov_disc(&cv, px, py, radius)}
	}
	marker := config.point_color
	marker.a = u8(f64(marker.a) * 0.85)
	cov_flush(&img, &cv, marker)

	axes_draw_frame(&ax)
	return image_save_png(&img, path)
}

histogram_png :: proc(
	df: ^w.DataFrame,
	col_name: string,
	path: string,
	bins: int = 20,
	config: PlotConfig = DEFAULT_PLOT_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	col := w.column(df, col_name)
	if col.type != .Float || bins <= 0 {return false}

	n := df.rows
	vals := make([]f64, n, allocator)
	defer delete(vals, allocator)
	for i in 0 ..< n {vals[i], _ = w.column_at_float(col, i)}
	min_v, max_v := data_range(vals)
	if !(min_v <= max_v) {min_v, max_v = 0, 1}
	if min_v == max_v {min_v, max_v = min_v - 0.5, max_v + 0.5}
	val_range := max_v - min_v

	counts := make([]int, bins, allocator)
	defer delete(counts, allocator)
	for v in vals {
		if math.is_nan(v) || math.is_inf(v) {continue}
		idx := clamp(int((v - min_v) / val_range * f64(bins)), 0, bins - 1)
		counts[idx] += 1
	}
	max_count := 1
	for c in counts {max_count = max(max_count, c)}

	img := _begin(config, allocator)
	defer image_free(&img)
	ax := axes_setup(
		&img,
		config,
		min_v,
		max_v,
		0,
		f64(max_count),
		{y_from_zero = true, x_exact = true, pad_frac = 0.05},
	)
	axes_draw_grid(&ax)

	gap := 1.0 if (f64(ax.right - ax.left) / f64(bins)) > 4 else 0.0
	for i in 0 ..< bins {
		if counts[i] == 0 {continue}
		x0 := axes_px(&ax, min_v + val_range * f64(i) / f64(bins))
		x1 := axes_px(&ax, min_v + val_range * f64(i + 1) / f64(bins))
		fill_rect_clipped(
			&ax,
			x0,
			axes_py(&ax, f64(counts[i])),
			x1 - gap,
			axes_py(&ax, 0),
			config.bar_color,
		)
	}

	axes_draw_frame(&ax)
	return image_save_png(&img, path)
}

line_png :: proc(
	xs: []f64,
	ys: []f64,
	path: string,
	config: PlotConfig = DEFAULT_PLOT_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(xs) != len(ys) || len(xs) == 0 {return false}
	min_x, max_x := data_range(xs)
	min_y, max_y := data_range(ys)

	img := _begin(config, allocator)
	defer image_free(&img)
	ax := axes_setup(&img, config, min_x, max_x, min_y, max_y)
	axes_draw_grid(&ax)

	cv := coverage_new(&ax, allocator)
	defer coverage_free(&cv, allocator)
	px, py := _to_pixels(&ax, xs, ys, allocator)
	defer {delete(px, allocator); delete(py, allocator)}
	cov_polyline(&cv, px, py, 0.5 * line_width_of(config), config.line_style)
	cov_flush(&img, &cv, config.point_color)

	axes_draw_frame(&ax)
	return image_save_png(&img, path)
}

bar_png :: proc(
	labels: []string,
	values: []f64,
	path: string,
	config: PlotConfig = DEFAULT_PLOT_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(labels) != len(values) || len(labels) == 0 {return false}
	n := len(labels)
	min_val, max_val := data_range(values)

	img := _begin(config, allocator)
	defer image_free(&img)
	ax := axes_setup(
		&img,
		config,
		0,
		f64(n - 1),
		min(min_val, 0),
		max(max_val, 0),
		{y_from_zero = true, x_categories = labels, pad_frac = 0.05},
	)
	axes_draw_grid(&ax)

	slot := f64(ax.right - ax.left) / f64(n)
	half := 0.4 * slot
	zero := axes_py(&ax, 0)
	for v, i in values {
		if math.is_nan(v) {continue}
		cx := axes_px(&ax, f64(i))
		fill_rect_clipped(&ax, cx - half, axes_py(&ax, v), cx + half, zero, config.bar_color)
	}
	if ax.ymin < 0 && ax.ymax > 0 {
		zy := int(math.round(zero))
		draw_line(&img, ax.left, zy, ax.right, zy, config.axis_color)
	}

	axes_draw_frame(&ax)
	return image_save_png(&img, path)
}

multi_line_png :: proc(
	lines: []LineData,
	path: string,
	config: PlotConfig = DEFAULT_PLOT_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(lines) == 0 {return false}

	// 1. Global data range across all lines
	min_x, max_x := math.inf_f64(1), math.inf_f64(-1)
	min_y, max_y := math.inf_f64(1), math.inf_f64(-1)
	for line in lines {
		if len(line.xs) != len(line.ys) {continue}
		lo, hi := data_range(line.xs)
		min_x, max_x = min(min_x, lo), max(max_x, hi)
		lo, hi = data_range(line.ys)
		min_y, max_y = min(min_y, lo), max(max_y, hi)
	}

	img := _begin(config, allocator)
	defer image_free(&img)
	ax := axes_setup(&img, config, min_x, max_x, min_y, max_y)
	axes_draw_grid(&ax)

	// 2. Lines (zero colour = next palette colour)
	palette := PALETTE
	colors := make([]Color, len(lines), allocator)
	defer delete(colors, allocator)
	next := 0
	for line, i in lines {
		colors[i] = line.color
		if line.color == (Color{}) {
			colors[i] = palette[next % len(palette)]
			next += 1
		}
	}
	cv := coverage_new(&ax, allocator)
	defer coverage_free(&cv, allocator)
	hw := 0.5 * line_width_of(config)
	for line, i in lines {
		if len(line.xs) == 0 || len(line.xs) != len(line.ys) {continue}
		px, py := _to_pixels(&ax, line.xs, line.ys, allocator)
		cov_polyline(&cv, px, py, hw, line.style)
		cov_flush(&img, &cv, colors[i])
		delete(px, allocator)
		delete(py, allocator)
	}

	// 3. Legend for labelled lines, in the emptiest corner
	entries := make([dynamic]Legend_Entry, 0, len(lines), allocator)
	defer delete(entries)
	pts_x := make([][]f64, len(lines), allocator)
	pts_y := make([][]f64, len(lines), allocator)
	defer {delete(pts_x, allocator); delete(pts_y, allocator)}
	for line, i in lines {
		pts_x[i], pts_y[i] = line.xs, line.ys
		if line.label !=
		   "" {append(&entries, Legend_Entry{line.label, colors[i], line.style, false})}
	}

	axes_draw_frame(&ax)
	axes_draw_legend(&ax, entries[:], pts_x, pts_y, allocator)
	return image_save_png(&img, path)
}

heatmap_png :: proc(
	data: []f64,
	rows: int,
	cols: int,
	path: string,
	row_labels: []string, // Optional: pass nil if not needed
	col_labels: []string, // Optional: pass nil if not needed
	config: PlotConfig = DEFAULT_PLOT_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(data) != rows * cols || rows == 0 || cols == 0 {return false}

	min_val, max_val := data_range(data)
	if !(min_val <= max_val) {min_val, max_val = 0, 1}
	if min_val == max_val {max_val = min_val + 1}
	val_range := max_val - min_val

	img := _begin(config, allocator)
	defer image_free(&img)
	size := config.font_size if config.font_size > 0 else font_size_for_scale(config.font_scale)
	f := font_get(size)
	tf := font_get(int(math.round(f64(size) * 1.25)), true)
	pad := max(6, size / 2)
	lh := line_height(f)

	// Margins from the labels and a colour bar on the right
	widest_row := 0.0
	for s in row_labels {widest_row = max(widest_row, text_width(f, s))}
	widest_col := 0.0
	for s in col_labels {widest_col = max(widest_col, text_width(f, s))}
	cb_w := max(12, size)
	cb_ticks, cb_step := nice_ticks(min_val, max_val, 6)
	cb_labels := format_ticks(cb_ticks, cb_step)
	widest_cb := 0.0
	for s in cb_labels {widest_cb = max(widest_cb, text_width(f, s))}

	left := max(config.margin_left / 2, pad + int(math.ceil(widest_row)) + pad)
	right := config.width - (pad + cb_w + pad / 2 + int(math.ceil(widest_cb)) + 2 * pad)
	top := pad + (line_height(tf) + pad if config.title != "" else 0)
	cell_w := f64(right - left - 2 * pad) / f64(cols)
	vertical_cols := widest_col > cell_w - f64(pad)
	col_h := (int(math.ceil(widest_col)) if vertical_cols else lh) if len(col_labels) > 0 else 0
	bottom := config.height - (pad + col_h + pad)
	right -= 2 * pad
	cell_h := f64(bottom - top) / f64(rows)

	if config.title != "" {
		draw_string(
			&img,
			tf,
			f64(left + right) / 2,
			f64(pad),
			config.title,
			config.axis_color,
			.Center,
			.Top,
		)
	}

	// Cells with values (text colour from cell luminance)
	show_values := cell_w > 4 * f64(size) && cell_h > 1.4 * f64(lh)
	for r in 0 ..< rows {
		for c in 0 ..< cols {
			val := data[r * cols + c]
			color := get_heat_color((val - min_val) / val_range)
			x0 := int(math.round(f64(left) + f64(c) * cell_w))
			x1 := int(math.round(f64(left) + f64(c + 1) * cell_w))
			y0 := int(math.round(f64(top) + f64(r) * cell_h))
			y1 := int(math.round(f64(top) + f64(r + 1) * cell_h))
			draw_rect(&img, x0, y0, x1 - x0, y1 - y0, color)
			if show_values {
				text :=
					fmt.tprintf("%d", int(val)) if val == math.floor(val) && abs(val) < 1e9 else fmt.tprintf("%.2f", val)
				lum := 0.299 * f64(color.r) + 0.587 * f64(color.g) + 0.114 * f64(color.b)
				tc := BLACK if lum > 140 else WHITE
				draw_string(
					&img,
					f,
					f64(x0 + x1) / 2,
					f64(y0 + y1) / 2,
					text,
					tc,
					.Center,
					.Middle,
				)
			}
		}
	}

	for r in 0 ..< min(rows, len(row_labels)) {
		cy := f64(top) + (f64(r) + 0.5) * cell_h
		draw_string(
			&img,
			f,
			f64(left - pad / 2),
			cy,
			row_labels[r],
			config.axis_color,
			.Right,
			.Middle,
		)
	}
	for c in 0 ..< min(cols, len(col_labels)) {
		cx := f64(left) + (f64(c) + 0.5) * cell_w
		if vertical_cols {
			draw_string_vertical(
				&img,
				f,
				cx,
				f64(bottom + pad / 2) + text_width(f, col_labels[c]) / 2,
				col_labels[c],
				config.axis_color,
			)
		} else {
			draw_string(
				&img,
				f,
				cx,
				f64(bottom + pad / 2),
				col_labels[c],
				config.axis_color,
				.Center,
				.Top,
			)
		}
	}

	// Colour bar
	cb_x := right + 2 * pad
	for y in top ..< bottom {
		t := 1 - f64(y - top) / f64(max(bottom - top - 1, 1))
		color := get_heat_color(t)
		for x in cb_x ..< cb_x + cb_w {draw_pixel(&img, x, y, color)}
	}
	for t, i in cb_ticks {
		py := int(math.round(f64(bottom - 1) - (t - min_val) / val_range * f64(bottom - top - 1)))
		draw_line(&img, cb_x + cb_w, py, cb_x + cb_w + 3, py, config.axis_color)
		draw_string(
			&img,
			f,
			f64(cb_x + cb_w + 3 + pad / 2),
			f64(py),
			cb_labels[i],
			config.axis_color,
			.Left,
			.Middle,
		)
	}

	return image_save_png(&img, path)
}
