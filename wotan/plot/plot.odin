package plot

import w "../core"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

LineStyle :: enum {
	Solid,
	Dashed,
	Dotted,
}

PlotConfig :: struct {
	width:         int,
	height:        int,
	margin_left:   int,
	margin_right:  int,
	margin_top:    int,
	margin_bottom: int,
	point_color:   Color, // Color for scatter points
	bar_color:     Color, // Color for histogram bars
	axis_color:    Color, // Color for axes and text
	bg_color:      Color, // Background color
	font_scale:    int,
	title:         string,
	x_label:       string,
	y_label:       string,
	line_style:    LineStyle, // Line style for line plots
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
	axis_color    = BLACK,
	bg_color      = WHITE,
	font_scale    = 2,
	title         = "",
	x_label       = "",
	y_label       = "",
	line_style    = .Solid,
}
// ============================================================================
// 1. Software Rasterizer & Image Buffer
// ============================================================================


Color :: struct {
	r, g, b, a: u8,
}
WHITE :: Color{255, 255, 255, 255}
BLACK :: Color{0, 0, 0, 255}
RED :: Color{255, 0, 0, 255}
BLUE :: Color{0, 0, 255, 255}

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

// Bresenham's Line Algorithm
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
draw_text :: proc(img: ^Image, x, y: int, text: string, scale: int, c: Color) {
	cx := x
	for b in text {
		if b == ' ' {
			cx += 4 * scale // Space character width
		} else {
			draw_char(img, cx, y, u8(b), scale, c)
			cx += 4 * scale
		}
	}
}
draw_rect :: proc(img: ^Image, x, y, w, h: int, c: Color) {
	for i in 0 ..< w {
		for j in 0 ..< h {
			draw_pixel(img, x + i, y + j, c)
		}
	}
}

// ============================================================================
// 2. Minimal 3x5 Bitmap Font (Zero Dependencies)
// ============================================================================

FONT_3X5 :: [36][5]u8 {
	// Digits 0-9
	{0b111, 0b101, 0b101, 0b101, 0b111}, // 0
	{0b010, 0b110, 0b010, 0b010, 0b111}, // 1
	{0b111, 0b001, 0b111, 0b100, 0b111}, // 2
	{0b111, 0b001, 0b111, 0b001, 0b111}, // 3
	{0b101, 0b101, 0b111, 0b001, 0b001}, // 4
	{0b111, 0b100, 0b111, 0b001, 0b111}, // 5
	{0b111, 0b100, 0b111, 0b101, 0b111}, // 6
	{0b111, 0b001, 0b010, 0b010, 0b010}, // 7
	{0b111, 0b101, 0b111, 0b101, 0b111}, // 8
	{0b111, 0b101, 0b111, 0b001, 0b111}, // 9
	// Letters A-Z
	{0b111, 0b101, 0b111, 0b101, 0b101}, // A
	{0b110, 0b101, 0b110, 0b101, 0b110}, // B
	{0b111, 0b100, 0b100, 0b100, 0b111}, // C
	{0b110, 0b101, 0b101, 0b101, 0b110}, // D
	{0b111, 0b100, 0b111, 0b100, 0b111}, // E
	{0b111, 0b100, 0b111, 0b100, 0b100}, // F
	{0b111, 0b100, 0b101, 0b101, 0b111}, // G
	{0b101, 0b101, 0b111, 0b101, 0b101}, // H
	{0b111, 0b010, 0b010, 0b010, 0b111}, // I
	{0b011, 0b001, 0b001, 0b101, 0b110}, // J
	{0b101, 0b110, 0b100, 0b110, 0b101}, // K
	{0b100, 0b100, 0b100, 0b100, 0b111}, // L
	{0b101, 0b111, 0b101, 0b101, 0b101}, // M
	{0b101, 0b111, 0b101, 0b101, 0b101}, // N (simplified)
	{0b111, 0b101, 0b101, 0b101, 0b111}, // O
	{0b111, 0b101, 0b111, 0b100, 0b100}, // P
	{0b111, 0b101, 0b101, 0b111, 0b101}, // Q
	{0b111, 0b101, 0b111, 0b110, 0b101}, // R
	{0b111, 0b100, 0b111, 0b001, 0b111}, // S
	{0b010, 0b111, 0b010, 0b010, 0b010}, // T
	{0b101, 0b101, 0b101, 0b101, 0b111}, // U
	{0b101, 0b101, 0b101, 0b101, 0b010}, // V
	{0b101, 0b101, 0b101, 0b111, 0b101}, // W
	{0b101, 0b101, 0b010, 0b101, 0b101}, // X
	{0b101, 0b101, 0b010, 0b010, 0b010}, // Y
	{0b111, 0b001, 0b010, 0b100, 0b111}, // Z
}

draw_char :: proc(img: ^Image, x, y: int, ch: u8, scale: int, c: Color) {
	idx: int = -1

	if ch >= '0' && ch <= '9' {
		idx = int(ch - '0')
	} else if ch >= 'A' && ch <= 'Z' {
		idx = int(ch - 'A' + 10)
	} else if ch >= 'a' && ch <= 'z' {
		idx = int(ch - 'a' + 10)
	}

	if idx < 0 || idx >= len(FONT_3X5) {return}
	font := FONT_3X5
	glyph := font[idx]
	for row in 0 ..< 5 {
		for col in 0 ..< 3 {
			if (glyph[row] & (1 << u32(2 - col))) != 0 {
				for sx in 0 ..< scale {
					for sy in 0 ..< scale {
						draw_pixel(img, x + col * scale + sx, y + row * scale + sy, c)
					}
				}
			}
		}
	}
}
draw_number :: proc(img: ^Image, x, y: int, val: f64, scale: int, c: Color) {
	str := fmt.tprintf("%.2f", val)
	cx := x
	for b in str {
		if b >= '0' && b <= '9' {
			draw_char(img, cx, y, u8(b), scale, c)
			cx += 4 * scale
		} else if b == '.' {
			for sx in 0 ..< scale {
				for sy in 0 ..< scale {
					draw_pixel(img, cx + sx, y + 4 * scale + sy, c)
				}
			}
			cx += 2 * scale
		} else if b == '-' {
			for sx in 0 ..< 3 * scale {
				for sy in 0 ..< scale {
					draw_pixel(img, cx + sx, y + 2 * scale + sy, c)
				}
			}
			cx += 4 * scale
		}
	}
}

draw_line_segment :: proc(img: ^Image, x0, y0, x1, y1: int, c: Color, style: LineStyle) {
	switch style {
	case .Solid:
		draw_line(img, x0, y0, x1, y1, c)
	case .Dashed:
		// Draw dashed line (6 pixels on, 4 pixels off)
		dx := math.abs(x1 - x0)
		dy := math.abs(y1 - y0)
		steps := dx
		if dy > dx {steps = dy}

		if steps == 0 {return}

		x_inc := f64(x1 - x0) / f64(steps)
		y_inc := f64(y1 - y0) / f64(steps)

		for i in 0 ..< steps {
			if (i % 10) < 6 { 	// 6 on, 4 off
				x := x0 + int(x_inc * f64(i))
				y := y0 + int(y_inc * f64(i))
				draw_pixel(img, x, y, c)
			}
		}
	case .Dotted:
		// Draw dotted line (2 pixels on, 3 pixels off)
		dx := math.abs(x1 - x0)
		dy := math.abs(y1 - y0)
		steps := dx
		if dy > dx {steps = dy}

		if steps == 0 {return}

		x_inc := f64(x1 - x0) / f64(steps)
		y_inc := f64(y1 - y0) / f64(steps)

		for i in 0 ..< steps {
			if (i % 5) < 2 { 	// 2 on, 3 off
				x := x0 + int(x_inc * f64(i))
				y := y0 + int(y_inc * f64(i))
				draw_pixel(img, x, y, c)
			}
		}
	}
}

// ============================================================================
// 3. Pure Odin PNG Encoder (with built-in CRC32 and Adler32)
// ============================================================================

_crc32_table :: proc() -> [256]u32 {
	table: [256]u32
	for i in 0 ..< 256 {
		crc := u32(i)
		for _ in 0 ..< 8 {
			if (crc & 1) != 0 {
				crc = (crc >> 1) ~ 0xEDB88320
			} else {
				crc = crc >> 1
			}
		}
		table[i] = crc
	}
	return table
}

crc32_checksum :: proc(data: []u8) -> u32 {
	table := _crc32_table()
	crc := u32(0xFFFFFFFF)
	for b in data {
		idx := (crc ~ u32(b)) & 0xFF
		crc = (crc >> 8) ~ table[idx]
	}
	return crc ~ 0xFFFFFFFF
}

_adler32_checksum :: proc(data: []u8) -> u32 {
	a: u32 = 1
	b: u32 = 0
	for v in data {
		a = (a + u32(v)) % 65521
		b = (b + a) % 65521
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

	crc_data := make([]u8, 4 + len(data))
	copy(crc_data, type_bytes)
	if data_len > 0 {
		copy(crc_data[4:], data)
	}

	crc := crc32_checksum(crc_data)
	b_crc := [4]u8{u8(crc >> 24), u8(crc >> 16), u8(crc >> 8), u8(crc)}
	_, _ = os.write(f, b_crc[:])
	delete(crc_data)
}

image_save_png :: proc(img: ^Image, path: string) -> bool {
	f, err := os.create(path)
	if err != nil {return false}
	defer os.close(f)

	sig := [8]u8{137, 80, 78, 71, 13, 10, 26, 10}
	_, _ = os.write(f, sig[:])

	ihdr := make([]u8, 13)
	ihdr[0] = u8(img.width >> 24); ihdr[1] = u8(img.width >> 16)
	ihdr[2] = u8(img.width >> 8); ihdr[3] = u8(img.width)
	ihdr[4] = u8(img.height >> 24); ihdr[5] = u8(img.height >> 16)
	ihdr[6] = u8(img.height >> 8); ihdr[7] = u8(img.height)
	ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0
	png_write_chunk(f, "IHDR", ihdr)
	delete(ihdr)

	// Prepare raw data with filter byte 0 at the start of each row
	raw_len := img.width * img.height * 4 + img.height
	raw := make([]u8, raw_len)
	for y in 0 ..< img.height {
		raw[y * (img.width * 4 + 1)] = 0
		copy(
			raw[y * (img.width * 4 + 1) + 1:],
			img.pixels[y * img.width * 4:(y + 1) * img.width * 4],
		)
	}

	// Compute Adler32 checksum of the raw data BEFORE wrapping it
	adler := _adler32_checksum(raw)

	// ✅ MASTERPIECE: Build a valid Zlib stream using Deflate "Stored" blocks!
	// This bypasses the need for core:compress/flate entirely.
	n_blocks := (raw_len + 65534) / 65535
	idat_len := 2 + 4 + n_blocks * 5 + raw_len

	idat := make([]u8, idat_len)
	idat[0] = 0x78 // Zlib CMF (Deflate, 32K window)
	idat[1] = 0x01 // Zlib FLG (Check bits)

	offset := 2
	remaining := raw_len
	raw_offset := 0

	for remaining > 0 {
		block_len := remaining
		if block_len > 65535 {block_len = 65535}

		is_final := (remaining == block_len)
		if is_final {
			idat[offset] = 0x01 // BFINAL=1, BTYPE=00 (Stored)
		} else {
			idat[offset] = 0x00 // BFINAL=0, BTYPE=00 (Stored)
		}
		offset += 1

		// LEN (little-endian)
		idat[offset] = u8(block_len)
		idat[offset + 1] = u8(block_len >> 8)
		offset += 2

		// NLEN (little-endian, one's complement)
		nlen := ~u16(block_len) & 0xFFFF
		idat[offset] = u8(nlen)
		idat[offset + 1] = u8(nlen >> 8)
		offset += 2

		// Raw data
		copy(idat[offset:], raw[raw_offset:raw_offset + block_len])
		offset += block_len
		raw_offset += block_len
		remaining -= block_len
	}

	delete(raw) // Free the temporary raw buffer

	// Adler32 checksum (big-endian)
	idat[offset] = u8(adler >> 24)
	idat[offset + 1] = u8(adler >> 16)
	idat[offset + 2] = u8(adler >> 8)
	idat[offset + 3] = u8(adler)

	png_write_chunk(f, "IDAT", idat)
	delete(idat)

	png_write_chunk(f, "IEND", nil)
	return true
}

// ============================================================================
// 4. Public Plotting API
// ============================================================================

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

	min_x, max_x := math.F64_MAX, -math.F64_MAX
	min_y, max_y := math.F64_MAX, -math.F64_MAX

	for i in 0 ..< n {
		vx, _ := w.column_at_float(col_x, i)
		vy, _ := w.column_at_float(col_y, i)
		xs[i] = vx
		ys[i] = vy
		if vx < min_x {min_x = vx}
		if vx > max_x {max_x = vx}
		if vy < min_y {min_y = vy}
		if vy > max_y {max_y = vy}
	}

	W, H := config.width, config.height
	img := image_new(W, H, allocator)
	defer image_free(&img)
	image_clear(&img, config.bg_color.r, config.bg_color.g, config.bg_color.b, config.bg_color.a)

	margin_l, margin_r, margin_t, margin_b :=
		config.margin_left, config.margin_right, config.margin_top, config.margin_bottom
	plot_w := W - margin_l - margin_r
	plot_h := H - margin_t - margin_b

	draw_line(&img, margin_l, margin_t, margin_l, H - margin_b, config.axis_color)
	draw_line(&img, margin_l, H - margin_b, W - margin_r, H - margin_b, config.axis_color)
	// Draw title
	if config.title != "" {
		title_x := (W - len(config.title) * 4 * config.font_scale) / 2
		draw_text(&img, title_x, 10, config.title, config.font_scale, config.axis_color)
	}

	// Draw axis labels
	if config.x_label != "" {
		label_x := margin_l + (plot_w - len(config.x_label) * 4 * config.font_scale) / 2
		draw_text(
			&img,
			label_x,
			H - margin_b + 10,
			config.x_label,
			config.font_scale,
			config.axis_color,
		)
	}

	if config.y_label != "" {
		// For y-label, we'll just draw it horizontally at the top left for now
		draw_text(&img, 10, margin_t, config.y_label, config.font_scale, config.axis_color)
	}
	for i in 0 ..< 5 {
		px := margin_l + (plot_w * i) / 4
		val := min_x + (max_x - min_x) * f64(i) / 4.0
		draw_line(&img, px, H - margin_b, px, H - margin_b + 5, config.axis_color)
		draw_number(&img, px - 15, H - margin_b + 10, val, config.font_scale, config.axis_color)
	}
	for i in 0 ..< 5 {
		py := H - margin_b - (plot_h * i) / 4
		val := min_y + (max_y - min_y) * f64(i) / 4.0
		draw_line(&img, margin_l - 5, py, margin_l, py, config.axis_color)
		draw_number(&img, 5, py - 5, val, config.font_scale, config.axis_color)
	}

	range_x := max_x - min_x
	range_y := max_y - min_y
	if range_x == 0 {range_x = 1}
	if range_y == 0 {range_y = 1}

	for i in 0 ..< n {
		px := margin_l + int((xs[i] - min_x) / range_x * f64(plot_w))
		py := H - margin_b - int((ys[i] - min_y) / range_y * f64(plot_h))

		draw_line(&img, px - 2, py, px + 2, py, config.point_color)
		draw_line(&img, px, py - 2, px, py + 2, config.point_color)
	}

	return image_save_png(&img, path)
}

histogram_png :: proc(
	df: ^w.DataFrame,
	col_name: string,
	path: string,
	bins: int = 20,
	config: PlotConfig = DEFAULT_PLOT_CONFIG, // <--- ADD THIS LINE
	allocator: mem.Allocator = context.allocator,
) -> bool {
	col := w.column(df, col_name)
	if col.type != .Float {return false}

	n := df.rows
	vals := make([]f64, n, allocator)
	defer delete(vals, allocator)

	min_v, max_v := math.F64_MAX, -math.F64_MAX
	for i in 0 ..< n {
		v, _ := w.column_at_float(col, i)
		vals[i] = v
		if v < min_v {min_v = v}
		if v > max_v {max_v = v}
	}

	val_range := max_v - min_v
	if val_range == 0 {val_range = 1}

	counts := make([]int, bins, allocator)
	defer delete(counts, allocator)

	for v in vals {
		idx := int((v - min_v) / val_range * f64(bins))
		if idx >= bins {idx = bins - 1}
		counts[idx] += 1
	}

	max_count := 0
	for c in counts {
		if c > max_count {max_count = c}
	}
	if max_count == 0 {max_count = 1}

	W, H := config.width, config.height
	img := image_new(W, H, allocator)
	defer image_free(&img)
	image_clear(&img, config.bg_color.r, config.bg_color.g, config.bg_color.b, config.bg_color.a)

	margin_l, margin_r, margin_t, margin_b :=
		config.margin_left, config.margin_right, config.margin_top, config.margin_bottom
	plot_w := W - margin_l - margin_r
	plot_h := H - margin_t - margin_b

	draw_line(&img, margin_l, margin_t, margin_l, H - margin_b, config.axis_color)
	draw_line(&img, margin_l, H - margin_b, W - margin_r, H - margin_b, config.axis_color)
	// Draw title
	if config.title != "" {
		title_x := (W - len(config.title) * 4 * config.font_scale) / 2
		draw_text(&img, title_x, 10, config.title, config.font_scale, config.axis_color)
	}

	// Draw axis labels
	if config.x_label != "" {
		label_x := margin_l + (plot_w - len(config.x_label) * 4 * config.font_scale) / 2
		draw_text(
			&img,
			label_x,
			H - margin_b + 10,
			config.x_label,
			config.font_scale,
			config.axis_color,
		)
	}

	if config.y_label != "" {
		draw_text(&img, 10, margin_t, config.y_label, config.font_scale, config.axis_color)
	}
	bar_w := plot_w / bins
	for i in 0 ..< bins {
		h := int(f64(counts[i]) / f64(max_count) * f64(plot_h))
		x := margin_l + i * bar_w
		y := H - margin_b - h
		draw_rect(&img, x, y, bar_w - 1, h, config.point_color)
	}

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
	n := len(xs)

	min_x, max_x := math.F64_MAX, -math.F64_MAX
	min_y, max_y := math.F64_MAX, -math.F64_MAX

	for i in 0 ..< n {
		if xs[i] < min_x {min_x = xs[i]}
		if xs[i] > max_x {max_x = xs[i]}
		if ys[i] < min_y {min_y = ys[i]}
		if ys[i] > max_y {max_y = ys[i]}
	}

	W, H := config.width, config.height
	img := image_new(W, H, allocator)
	defer image_free(&img)
	image_clear(&img, config.bg_color.r, config.bg_color.g, config.bg_color.b, config.bg_color.a)

	margin_l, margin_r, margin_t, margin_b :=
		config.margin_left, config.margin_right, config.margin_top, config.margin_bottom
	plot_w := W - margin_l - margin_r
	plot_h := H - margin_t - margin_b

	// Draw axes
	draw_line(&img, margin_l, margin_t, margin_l, H - margin_b, config.axis_color)
	draw_line(&img, margin_l, H - margin_b, W - margin_r, H - margin_b, config.axis_color)

	// Draw title
	if config.title != "" {
		title_x := (W - len(config.title) * 4 * config.font_scale) / 2
		draw_text(&img, title_x, 10, config.title, config.font_scale, config.axis_color)
	}

	// Draw axis labels
	if config.x_label != "" {
		label_x := margin_l + (plot_w - len(config.x_label) * 4 * config.font_scale) / 2
		draw_text(
			&img,
			label_x,
			H - margin_b + 10,
			config.x_label,
			config.font_scale,
			config.axis_color,
		)
	}

	if config.y_label != "" {
		draw_text(&img, 10, margin_t, config.y_label, config.font_scale, config.axis_color)
	}

	// Draw tick marks and labels
	for i in 0 ..< 5 {
		px := margin_l + (plot_w * i) / 4
		val := min_x + (max_x - min_x) * f64(i) / 4.0
		draw_line(&img, px, H - margin_b, px, H - margin_b + 5, config.axis_color)
		draw_number(&img, px - 15, H - margin_b + 10, val, config.font_scale, config.axis_color)
	}
	for i in 0 ..< 5 {
		py := H - margin_b - (plot_h * i) / 4
		val := min_y + (max_y - min_y) * f64(i) / 4.0
		draw_line(&img, margin_l - 5, py, margin_l, py, config.axis_color)
		draw_number(&img, 5, py - 5, val, config.font_scale, config.axis_color)
	}

	// Plot the line
	range_x := max_x - min_x
	range_y := max_y - min_y
	if range_x == 0 {range_x = 1}
	if range_y == 0 {range_y = 1}

	// Convert data points to pixel coordinates
	prev_px, prev_py := -1, -1
	for i in 0 ..< n {
		px := margin_l + int((xs[i] - min_x) / range_x * f64(plot_w))
		py := H - margin_b - int((ys[i] - min_y) / range_y * f64(plot_h))

		if prev_px >= 0 {
			draw_line_segment(
				&img,
				prev_px,
				prev_py,
				px,
				py,
				config.point_color,
				config.line_style,
			)
		}
		prev_px = px
		prev_py = py
	}

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

	// Find max value for scaling
	max_val := 0.0
	for v in values {
		if v > max_val {max_val = v}
		if v < 0 {
			// Handle negative values
			if -v > max_val {max_val = -v}
		}
	}
	if max_val == 0 {max_val = 1}

	W, H := config.width, config.height
	img := image_new(W, H, allocator)
	defer image_free(&img)
	image_clear(&img, config.bg_color.r, config.bg_color.g, config.bg_color.b, config.bg_color.a)

	margin_l, margin_r, margin_t, margin_b :=
		config.margin_left, config.margin_right, config.margin_top, config.margin_bottom
	plot_w := W - margin_l - margin_r
	plot_h := H - margin_t - margin_b

	// Draw axes
	draw_line(&img, margin_l, margin_t, margin_l, H - margin_b, config.axis_color)
	draw_line(&img, margin_l, H - margin_b, W - margin_r, H - margin_b, config.axis_color)

	// Draw title
	if config.title != "" {
		title_x := (W - len(config.title) * 4 * config.font_scale) / 2
		draw_text(&img, title_x, 10, config.title, config.font_scale, config.axis_color)
	}

	// Draw axis labels
	if config.y_label != "" {
		draw_text(&img, 10, margin_t, config.y_label, config.font_scale, config.axis_color)
	}

	// Draw y-axis tick marks and labels
	for i in 0 ..< 5 {
		py := H - margin_b - (plot_h * i) / 4
		val := max_val * f64(i) / 4.0
		draw_line(&img, margin_l - 5, py, margin_l, py, config.axis_color)
		draw_number(&img, 5, py - 5, val, config.font_scale, config.axis_color)
	}

	// Calculate bar width and gap
	bar_width := plot_w / n
	bar_gap := bar_width / 5 // 20% gap between bars
	actual_bar_width := bar_width - bar_gap

	// Draw bars and labels
	for i in 0 ..< n {
		// Calculate bar height (proportional to value)
		bar_height := int(values[i] / max_val * f64(plot_h))

		// Calculate bar position
		x := margin_l + i * bar_width + bar_gap / 2
		y := H - margin_b - bar_height

		// Draw the bar
		draw_rect(&img, x, y, actual_bar_width, bar_height, config.bar_color)

		// Draw label (centered under the bar)
		label := labels[i]
		label_width := len(label) * 4 * config.font_scale
		label_x := x + (actual_bar_width - label_width) / 2
		draw_text(&img, label_x, H - margin_b + 10, label, config.font_scale, config.axis_color)
	}

	return image_save_png(&img, path)
}
