package plot

import "core:c/libc"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"

// ============================================================================
// Axis layout: nice ticks, label formatting, margins sized from the text
// ============================================================================

// 1-2-5 rounding of x (Heckbert, "Nice numbers for graph labels")
@(private = "file")
_nice_num :: proc(x: f64, round: bool) -> f64 {
	e := math.floor(math.log10(x))
	f := x / math.pow(10, e)
	nf: f64
	if round {
		nf = 1 if f < 1.5 else (2 if f < 3 else (5 if f < 7 else 10))
	} else {
		nf = 1 if f <= 1 else (2 if f <= 2 else (5 if f <= 5 else 10))
	}
	return nf * math.pow(10, e)
}

// Ticks at multiples of a nice (1-2-5) step inside [lo, hi], aiming for ~max_ticks
nice_ticks :: proc(
	lo, hi: f64,
	max_ticks: int,
	allocator := context.temp_allocator,
) -> (
	ticks: []f64,
	step: f64,
) {
	n := max(max_ticks, 2)
	step = _nice_num((hi - lo) / f64(n - 1), true)
	first := math.ceil(lo / step - 1e-9)
	last := math.floor(hi / step + 1e-9)
	count := max(int(last - first) + 1, 0)
	ticks = make([]f64, count, allocator)
	for i in 0 ..< count {
		v := (first + f64(i)) * step
		if abs(v) < step * 1e-9 {v = 0} 	// no "-0"
		ticks[i] = v
	}
	return
}

// Fewest decimals that represent multiples of `step` exactly
@(private = "file")
_decimals :: proc(step: f64) -> int {
	for d in 0 ..< 17 {
		s := step * math.pow(10, f64(d))
		// a whole number of units (≥ 1: a step below 1e-6 is not "0 decimals")
		if r := math.round(s); r >= 1 && abs(s - r) < 1e-6 * s {return d}
	}
	return 17
}

// Tick labels with consistent precision; scientific for very large/small axes
format_ticks :: proc(ticks: []f64, step: f64, allocator := context.temp_allocator) -> []string {
	labels := make([]string, len(ticks), allocator)
	abs_max := 0.0
	for v in ticks {abs_max = max(abs_max, abs(v))}
	if abs_max >= 1e6 || (abs_max > 0 && abs_max < 1e-3) {
		e := math.floor(math.log10(abs_max))
		// log10 can fall just short of a power of ten (log10(1e300) < 300)
		if abs_max / math.pow(10, e) >= 10 * (1 - 1e-12) {e += 1}
		scale := math.pow(10, e)
		d := _decimals(step / scale)
		for v, i in ticks {
			labels[i] =
				v == 0 ? "0" : _minus(fmt.aprintf("%.*fe%d", d, v / scale, int(e), allocator = allocator), allocator)
		}
		return labels
	}
	d := _decimals(step)
	for v, i in ticks {
		labels[i] = _minus(fmt.aprintf("%.*f", d, v, allocator = allocator), allocator)
	}
	return labels
}

// Tick labels relative to an offset when the ticks share many leading digits:
// a tick's value is its label plus the offset. The offset follows matplotlib's
// ScalarFormatter (offset threshold 4): with at least two ticks of one sign,
// it is their common leading part, a multiple of a power of ten, used when it
// saves at least 4 digits. offset_text shows it with its sign ("+1000",
// "−1.234567e7"). Otherwise offset is 0, offset_text "" and the labels are
// format_ticks'.
format_ticks_offset :: proc(
	ticks: []f64,
	step: f64,
	allocator := context.temp_allocator,
) -> (
	labels: []string,
	offset: f64,
	offset_text: string,
) {
	oom: int
	offset, oom = _tick_offset(ticks)
	if offset == 0 {return format_ticks(ticks, step, allocator), 0, ""}
	shifted := make([]f64, len(ticks), context.temp_allocator)
	for v, i in ticks {
		// a multiple of the step (the offset is one): no rounding residue
		shifted[i] = math.round((v - offset) / step) * step + 0
	}
	labels = format_ticks(shifted, step, allocator)
	sign := "−" if offset < 0 else "+"
	offset_text = strings.concatenate({sign, _offset_digits(abs(offset), oom)}, allocator)
	return
}

// matplotlib's ScalarFormatter._compute_offset (offset threshold 4): the
// offset and the power of ten it is a multiple of
@(private = "file")
_tick_offset :: proc(ticks: []f64) -> (offset: f64, oom: int) {
	if len(ticks) == 0 {return}
	lmin, lmax := math.inf_f64(1), math.inf_f64(-1)
	for v in ticks {lmin, lmax = min(lmin, v), max(lmax, v)}
	if !_finite(lmin) || !_finite(lmax) || lmin == lmax || (lmin <= 0 && 0 <= lmax) {return}
	abs_min, abs_max := min(abs(lmin), abs(lmax)), max(abs(lmin), abs(lmax))
	sign := -1.0 if lmin < 0 else 1.0
	fdiv :: proc(a: f64, o: int) -> f64 {return _py_floordiv(a, _pow10(o))}
	oom_max := int(math.ceil(math.log10(abs_max)))
	// the smallest power of ten to which abs_min and abs_max agree
	o := oom_max
	for fdiv(abs_min, o) == fdiv(abs_max, o) && o > -330 {o -= 1}
	oom = o + 1
	if (abs_max - abs_min) / _pow10(oom) <= 1e-2 {
		// straddling a multiple of a large power of ten (relative to the span):
		// the smallest power of ten at which they are no more than 1 apart
		o = oom_max
		for !(fdiv(abs_max, o) - fdiv(abs_min, o) > 1) && o > -330 {o -= 1}
		oom = o + 1
	}
	if fdiv(abs_max, oom) < 1000 {return 0, 0}
	offset = sign * fdiv(abs_max, oom) * _pow10(oom)
	return
}

// 10^o correctly rounded for |o| ≤ 22 (as Python's 10.0 ** o)
@(private = "file")
_pow10 :: proc(o: int) -> f64 {
	if o >= 0 && o <= 22 {
		p := 1.0
		for _ in 0 ..< o {p *= 10} 	// exact up to 1e22
		return p
	}
	if o < 0 && o >= -22 {return 1 / _pow10(-o)} 	// one rounding
	return math.pow(10, f64(o))
}

// Python's float floor division a // b (CPython float_floor_div): the floor of
// the exact quotient, which differs from floor(a / b) when a / b rounds up to
// a whole number (1.0 // 1e-10 = 9999999999)
@(private = "file")
_py_floordiv :: proc(a, b: f64) -> f64 {
	m := libc.fmod(a, b) // exact (C fmod), sign of a
	div := (a - m) / b
	if m != 0 && (b < 0) != (m < 0) {div -= 1}
	if div == 0 {return 0}
	q := math.floor(div)
	if div - q > 0.5 {q += 1}
	return q
}

// |offset| = a multiple of 10^k, without trailing zeros; scientific
// ("1.0000001e6") from 10⁶ and below 10⁻³
@(private = "file")
_offset_digits :: proc(a: f64, k: int) -> string {
	trim :: proc(s: string) -> string {
		if !strings.contains_rune(s, '.') {return s}
		s := strings.trim_right(s, "0")
		return strings.trim_right(s, ".")
	}
	if a >= 1e6 || a < 1e-3 {
		e := math.floor(math.log10(a))
		if a / math.pow(10, e) >= 10 * (1 - 1e-12) {e += 1}
		digits := max(0, int(e) - k)
		mant := trim(fmt.tprintf("%.*f", digits, a / math.pow(10, e)))
		return _minus(fmt.tprintf("%se%d", mant, int(e)), context.temp_allocator)
	}
	return trim(fmt.tprintf("%.*f", max(0, -k), a))
}

// Typographic minus sign instead of the hyphen (also in exponents)
@(private)
_minus :: proc(s: string, allocator: mem.Allocator) -> string {
	if !strings.contains_rune(s, '-') {return s}
	out, _ := strings.replace_all(s, "-", "−", allocator)
	return out
}

Axes_Options :: struct {
	pad_frac:     f64, // data padding as a fraction of the range (default 0.03)
	y_from_zero:  bool, // bars / histograms: include 0, no padding on the zero side
	x_exact:      bool, // no x padding (histogram edges)
	x_categories: []string, // categorical x axis at 0..n-1 (bar charts)
}

Axes :: struct {
	img:          ^Image,
	cfg:          PlotConfig,
	font:         ^Font,
	title_font:   ^Font,
	pad:          int,
	tick_len:     int,
	left, top:    int, // plot box (frame) in pixels
	right:        int,
	bottom:       int,
	xmin, xmax:   f64,
	ymin, ymax:   f64,
	xticks:       []f64,
	yticks:       []f64,
	xlabels:      []string,
	ylabels:      []string,
	xoffset:      string, // offset of the x labels ("" = none), see format_ticks_offset
	yoffset:      string,
	categories:   []string,
	vertical_cat: bool, // category labels rotated
}

@(private = "file")
_finite :: proc(v: f64) -> bool {return !math.is_nan(v) && !math.is_inf(v)}

@(private = "file")
_fix_range :: proc(lo, hi: f64) -> (f64, f64) {
	if !_finite(lo) || !_finite(hi) || lo > hi {return 0, 1}
	if lo == hi {
		d := abs(lo) * 0.1 if lo != 0 else 0.5
		return lo - d, hi + d
	}
	return lo, hi
}

// Data range of the finite values of xs
data_range :: proc(xs: []f64) -> (lo, hi: f64) {
	lo, hi = math.inf_f64(1), math.inf_f64(-1)
	for v in xs {
		if _finite(v) {lo, hi = min(lo, v), max(hi, v)}
	}
	return
}

axes_setup :: proc(
	img: ^Image,
	cfg: PlotConfig,
	xmin, xmax, ymin, ymax: f64,
	opts := Axes_Options{},
) -> Axes {
	cfg := config_resolve(cfg)
	ax := Axes {
		img = img,
		cfg = cfg,
	}
	size := cfg.font_size if cfg.font_size > 0 else font_size_for_scale(cfg.font_scale)
	ax.font = font_get(size)
	ax.title_font = font_get(int(math.round(f64(size) * 1.25)), true)
	ax.pad = max(6, size / 2)
	ax.tick_len = max(4, size / 3)
	pad := ax.pad
	lh := line_height(ax.font)
	W, H := img.width, img.height
	pad_frac := opts.pad_frac if opts.pad_frac > 0 else 0.03

	// ---- data limits
	x0, x1 := _fix_range(xmin, xmax)
	y0, y1 := _fix_range(ymin, ymax)
	if opts.y_from_zero {
		y0, y1 = min(y0, 0), max(y1, 0)
		span := y1 - y0
		if y1 > 0 {y1 += pad_frac * span}
		if y0 < 0 {y0 -= pad_frac * span}
	} else {
		span := y1 - y0
		y0, y1 = y0 - pad_frac * span, y1 + pad_frac * span
	}
	if len(opts.x_categories) > 0 {
		x0, x1 = -0.5, f64(len(opts.x_categories)) - 0.5
		ax.categories = opts.x_categories
	} else if !opts.x_exact {
		span := x1 - x0
		x0, x1 = x0 - pad_frac * span, x1 + pad_frac * span
	}
	ax.xmin, ax.xmax, ax.ymin, ax.ymax = x0, x1, y0, y1

	// The layout is computed again (at most twice more) when an axis needs a
	// row for its offset text: above the plot box (y) or below the x labels
	extra_top, extra_bottom := 0, 0
	for _ in 0 ..< 3 {
		_axes_layout(&ax, cfg, extra_top, extra_bottom)
		need_top := lh if ax.yoffset != "" else 0
		need_bottom := pad / 2 + lh if ax.xoffset != "" else 0
		if need_top <= extra_top && need_bottom <= extra_bottom {break}
		extra_top, extra_bottom = max(extra_top, need_top), max(extra_bottom, need_bottom)
	}
	return ax
}

// Plot box, ticks and labels for the data limits in ax, with extra room above
// the box and below the x labels
@(private = "file")
_axes_layout :: proc(ax: ^Axes, cfg: PlotConfig, extra_top, extra_bottom: int) {
	img, pad := ax.img, ax.pad
	lh := line_height(ax.font)
	W, H := img.width, img.height
	x0, x1, y0, y1 := ax.xmin, ax.xmax, ax.ymin, ax.ymax

	// ---- vertical layout (independent of tick values)
	ax.top =
		max(cfg.margin_top, pad + (line_height(ax.title_font) + pad if cfg.title != "" else pad)) +
		extra_top
	cat_extra := 0
	if len(ax.categories) > 0 {
		widest := 0.0
		for c in ax.categories {widest = max(widest, text_width(ax.font, c))}
		slot := f64(W - 2 * cfg.margin_left) / f64(len(ax.categories))
		if widest > slot - f64(pad) {
			ax.vertical_cat = true
			cat_extra = int(widest) - lh + ax.font.cap_bot - ax.font.cap_top
		}
	}
	bottom_margin :=
		ax.tick_len +
		pad / 2 +
		lh +
		cat_extra +
		extra_bottom +
		(pad / 2 + lh if cfg.x_label != "" else 0) +
		pad
	ax.bottom = H - max(cfg.margin_bottom, bottom_margin)

	// ---- y ticks -> left margin
	plot_h := ax.bottom - ax.top
	ny := clamp(plot_h / (3 * lh), 3, 10)
	yticks, ystep := nice_ticks(y0, y1, ny)
	ax.yticks = yticks
	ax.ylabels, _, ax.yoffset = format_ticks_offset(ax.yticks, ystep)
	widest_y := 0.0
	for s in ax.ylabels {widest_y = max(widest_y, text_width(ax.font, s))}
	left_margin :=
		pad +
		(lh + pad / 2 if cfg.y_label != "" else 0) +
		int(math.ceil(widest_y)) +
		pad / 2 +
		ax.tick_len
	ax.left = max(cfg.margin_left, left_margin)

	// ---- x ticks -> right margin (last label must not be clipped)
	right := W - max(cfg.margin_right, 2 * pad)
	if len(ax.categories) == 0 {
		nx := clamp((right - ax.left) / (5 * lh), 3, 12)
		for _ in 0 ..< 6 {
			step: f64
			ax.xticks, step = nice_ticks(x0, x1, nx)
			ax.xlabels, _, ax.xoffset = format_ticks_offset(ax.xticks, step)
			widest := 0.0
			for s in ax.xlabels {widest = max(widest, text_width(ax.font, s))}
			if f64(len(ax.xticks)) * (widest + f64(2 * pad)) <= f64(right - ax.left) ||
			   nx <= 2 {break}
			nx -= 1
		}
		if len(ax.xticks) > 0 {
			last := ax.xticks[len(ax.xticks) - 1]
			half := text_width(ax.font, ax.xlabels[len(ax.xlabels) - 1]) / 2
			px := f64(ax.left) + (last - x0) / (x1 - x0) * f64(right - ax.left)
			overflow := int(math.ceil(px + half)) - (W - pad / 2)
			if overflow > 0 {right -= overflow}
		}
	}
	ax.right = right
}

axes_px :: proc(ax: ^Axes, x: f64) -> f64 {
	return f64(ax.left) + (x - ax.xmin) / (ax.xmax - ax.xmin) * f64(ax.right - ax.left)
}

axes_py :: proc(ax: ^Axes, y: f64) -> f64 {
	return f64(ax.bottom) - (y - ax.ymin) / (ax.ymax - ax.ymin) * f64(ax.bottom - ax.top)
}

// Background grid at the ticks (drawn before the data)
// Default grid colour follows the theme: a light tint of the axis colour over
// the background (subtle on both light and dark backgrounds)
grid_color_of :: proc(cfg: PlotConfig) -> Color {
	if cfg.grid_color != (Color{}) {return cfg.grid_color}
	cfg := config_resolve(cfg)
	mix :: proc(a, b: u8, t: f64) -> u8 {return u8(f64(a) + (f64(b) - f64(a)) * t + 0.5)}
	bg, fg := cfg.bg_color, cfg.axis_color
	return Color{mix(bg.r, fg.r, 0.13), mix(bg.g, fg.g, 0.13), mix(bg.b, fg.b, 0.13), 255}
}

axes_draw_grid :: proc(ax: ^Axes) {
	if !ax.cfg.show_grid {return}
	c := grid_color_of(ax.cfg)
	for t in ax.yticks {
		py := int(math.round(axes_py(ax, t)))
		if py > ax.top && py < ax.bottom {draw_line(ax.img, ax.left + 1, py, ax.right - 1, py, c)}
	}
	for t in ax.xticks {
		px := int(math.round(axes_px(ax, t)))
		if px > ax.left && px < ax.right {draw_line(ax.img, px, ax.top + 1, px, ax.bottom - 1, c)}
	}
}

// Frame, ticks, tick labels, axis labels and title (drawn after the data)
axes_draw_frame :: proc(ax: ^Axes) {
	img, cfg, f := ax.img, ax.cfg, ax.font
	c := cfg.axis_color
	draw_line(img, ax.left, ax.top, ax.right, ax.top, c)
	draw_line(img, ax.left, ax.bottom, ax.right, ax.bottom, c)
	draw_line(img, ax.left, ax.top, ax.left, ax.bottom, c)
	draw_line(img, ax.right, ax.top, ax.right, ax.bottom, c)
	pad, tl := ax.pad, ax.tick_len

	for t, i in ax.yticks {
		py := int(math.round(axes_py(ax, t)))
		draw_line(img, ax.left - tl, py, ax.left, py, c)
		draw_string(
			img,
			f,
			f64(ax.left - tl - pad / 2),
			f64(py),
			ax.ylabels[i],
			c,
			.Right,
			.Middle,
		)
	}
	label_top := f64(ax.bottom + tl + pad / 2)
	if len(ax.categories) > 0 {
		for name, i in ax.categories {
			px := int(math.round(axes_px(ax, f64(i))))
			draw_line(img, px, ax.bottom, px, ax.bottom + tl, c)
			if ax.vertical_cat {
				w := text_width(f, name)
				draw_string_vertical(img, f, f64(px), label_top + w / 2, name, c)
			} else {
				draw_string(img, f, f64(px), label_top, name, c, .Center, .Top)
			}
		}
	} else {
		for t, i in ax.xticks {
			px := int(math.round(axes_px(ax, t)))
			draw_line(img, px, ax.bottom, px, ax.bottom + tl, c)
			draw_string(img, f, f64(px), label_top, ax.xlabels[i], c, .Center, .Top)
		}
	}
	// offsets: x below the right end of its labels, y above the plot box
	if ax.xoffset != "" {
		y := label_top + f64(line_height(f) + pad / 2)
		draw_string(img, f, f64(ax.right), y, ax.xoffset, c, .Right, .Top)
	}
	if ax.yoffset != "" {
		draw_string(img, f, f64(ax.left), f64(ax.top - pad / 4), ax.yoffset, c, .Left, .Bottom)
	}

	cx := f64(ax.left + ax.right) / 2
	if cfg.x_label != "" {
		draw_string(img, f, cx, f64(img.height - pad), cfg.x_label, c, .Center, .Bottom)
	}
	if cfg.y_label != "" {
		draw_string_vertical(
			img,
			f,
			f64(pad + line_height(f) / 2),
			f64(ax.top + ax.bottom) / 2,
			cfg.y_label,
			c,
		)
	}
	if cfg.title != "" {
		draw_string(img, ax.title_font, cx, f64(pad), cfg.title, c, .Center, .Top)
	}
}

// ============================================================================
// Anti-aliased rasterisation through a coverage buffer over the plot box:
// shapes accumulate max-coverage, then one composite per colour (clean joints,
// no double-blending where segments of the same series overlap)
// ============================================================================

Coverage :: struct {
	x0, y0: int,
	w, h:   int,
	a:      []f32,
}

coverage_new :: proc(ax: ^Axes, allocator: mem.Allocator) -> Coverage {
	w, h := ax.right - ax.left - 1, ax.bottom - ax.top - 1
	w, h = max(w, 1), max(h, 1)
	return Coverage{ax.left + 1, ax.top + 1, w, h, make([]f32, w * h, allocator)}
}

coverage_free :: proc(cv: ^Coverage, allocator: mem.Allocator) {
	delete(cv.a, allocator)
}

// Capsule of half-width hw around segment (x0,y0)-(x1,y1), in image pixels
cov_segment :: proc(cv: ^Coverage, x0, y0, x1, y1, hw: f64) {
	r := hw + 1
	ix0 := max(int(math.floor(min(x0, x1) - r)), cv.x0)
	ix1 := min(int(math.ceil(max(x0, x1) + r)), cv.x0 + cv.w - 1)
	iy0 := max(int(math.floor(min(y0, y1) - r)), cv.y0)
	iy1 := min(int(math.ceil(max(y0, y1) + r)), cv.y0 + cv.h - 1)
	dx, dy := x1 - x0, y1 - y0
	len2 := dx * dx + dy * dy
	for py in iy0 ..= iy1 {
		row := (py - cv.y0) * cv.w - cv.x0
		for px in ix0 ..= ix1 {
			qx, qy := f64(px) - x0, f64(py) - y0
			t := 0.0
			if len2 > 0 {t = clamp((qx * dx + qy * dy) / len2, 0, 1)}
			ex, ey := qx - t * dx, qy - t * dy
			a := f32(clamp(hw + 0.5 - math.sqrt(ex * ex + ey * ey), 0, 1))
			if a > cv.a[row + px] {cv.a[row + px] = a}
		}
	}
}

cov_disc :: proc(cv: ^Coverage, cx, cy, radius: f64) {
	cov_segment(cv, cx, cy, cx, cy, radius)
}

// Polyline in pixel coordinates; NaN points break the line. The dash pattern
// runs continuously along the arc length (not restarted per segment)
cov_polyline :: proc(cv: ^Coverage, px, py: []f64, hw: f64, style: LineStyle) {
	on, off: f64
	switch style {
	case .Solid:
		on, off = math.inf_f64(1), 0
	case .Dashed:
		on, off = 4 + 8 * hw, 3 + 4 * hw
	case .Dotted:
		on, off = 0.01, 2 + 4 * hw
	}
	period := on + off
	phase := 0.0 // position within the dash period
	for i in 1 ..< len(px) {
		ax, ay, bx, by := px[i - 1], py[i - 1], px[i], py[i]
		if !_finite(ax) || !_finite(ay) || !_finite(bx) || !_finite(by) {continue}
		if style == .Solid {
			cov_segment(cv, ax, ay, bx, by, hw)
			continue
		}
		seg := math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
		pos := 0.0
		for pos < seg {
			if phase < on {
				run := min(on - phase, seg - pos)
				t0, t1 := pos / max(seg, 1e-12), (pos + run) / max(seg, 1e-12)
				cov_segment(
					cv,
					ax + t0 * (bx - ax),
					ay + t0 * (by - ay),
					ax + t1 * (bx - ax),
					ay + t1 * (by - ay),
					hw,
				)
				pos += run
				phase += run
			} else {
				run := min(period - phase, seg - pos)
				pos += run
				phase += run
			}
			if phase >= period {phase -= period}
		}
	}
}

// Composite the accumulated coverage in colour c, then clear the buffer
cov_flush :: proc(img: ^Image, cv: ^Coverage, c: Color) {
	for y in 0 ..< cv.h {
		for x in 0 ..< cv.w {
			a := cv.a[y * cv.w + x]
			if a > 0 {
				blend_pixel(img, cv.x0 + x, cv.y0 + y, c, a)
				cv.a[y * cv.w + x] = 0
			}
		}
	}
}

// Filled rectangle in image pixels, clipped to the plot box (crisp edges)
fill_rect_clipped :: proc(ax: ^Axes, x0, y0, x1, y1: f64, c: Color) {
	ix0 := max(int(math.round(min(x0, x1))), ax.left + 1)
	ix1 := min(int(math.round(max(x0, x1))), ax.right)
	iy0 := max(int(math.round(min(y0, y1))), ax.top + 1)
	iy1 := min(int(math.round(max(y0, y1))), ax.bottom)
	for y in iy0 ..< iy1 {
		for x in ix0 ..< ix1 {blend_pixel(ax.img, x, y, c, 1)}
	}
}

// ============================================================================
// Legend: placed in the plot corner that covers the fewest data points
// ============================================================================

Legend_Entry :: struct {
	label:  string,
	color:  Color,
	style:  LineStyle,
	marker: bool, // scatter-style entry (dot instead of line sample)
}

axes_draw_legend :: proc(
	ax: ^Axes,
	entries: []Legend_Entry,
	pts_x, pts_y: [][]f64,
	allocator: mem.Allocator,
) {
	if len(entries) == 0 {return}
	f, pad := ax.font, ax.pad
	lh := line_height(f)
	sample := 2 * lh
	widest := 0.0
	for e in entries {widest = max(widest, text_width(f, e.label))}
	row := lh + 2
	w := pad + sample + pad + int(math.ceil(widest)) + pad
	h := pad / 2 + len(entries) * row + pad / 2
	inset := pad

	// candidate corners: TR, TL, BR, BL; pick the one hiding the fewest points
	cands := [4][2]int {
		{ax.right - inset - w, ax.top + inset},
		{ax.left + inset, ax.top + inset},
		{ax.right - inset - w, ax.bottom - inset - h},
		{ax.left + inset, ax.bottom - inset - h},
	}
	best, best_n := 0, max(int)
	for cand, ci in cands {
		n := 0
		for xs, s in pts_x {
			ys := pts_y[s]
			for i in 0 ..< min(len(xs), len(ys)) {
				px, py := axes_px(ax, xs[i]), axes_py(ax, ys[i])
				if px >= f64(cand[0]) &&
				   px <= f64(cand[0] + w) &&
				   py >= f64(cand[1]) &&
				   py <= f64(cand[1] + h) {n += 1}
			}
		}
		if n < best_n {best, best_n = ci, n}
	}
	lx, ly := cands[best][0], cands[best][1]

	bg := ax.cfg.bg_color
	for y in ly ..< ly + h {
		for x in lx ..< lx + w {blend_pixel(ax.img, x, y, bg, 0.85)}
	}
	border := Color{ax.cfg.axis_color.r, ax.cfg.axis_color.g, ax.cfg.axis_color.b, 90}
	for x in lx ..< lx + w {blend_pixel(ax.img, x, ly, border, 1); blend_pixel(ax.img, x, ly + h - 1, border, 1)}
	for y in ly + 1 ..< ly + h - 1 {blend_pixel(ax.img, lx, y, border, 1); blend_pixel(ax.img, lx + w - 1, y, border, 1)}

	cv := coverage_new(ax, allocator)
	defer coverage_free(&cv, allocator)
	hw := 0.5 * line_width_of(ax.cfg)
	for e, i in entries {
		cy := f64(ly + pad / 2 + i * row + 1) + f64(f.cap_top + f.cap_bot) / 2
		x0 := f64(lx + pad)
		if e.marker {
			cov_disc(&cv, x0 + f64(sample) / 2, cy, 2.5 * max(hw, 0.75))
		} else {
			xs := [2]f64{x0, x0 + f64(sample)}
			ys := [2]f64{cy, cy}
			cov_polyline(&cv, xs[:], ys[:], hw, e.style)
		}
		cov_flush(ax.img, &cv, e.color)
		draw_string(
			ax.img,
			f,
			x0 + f64(sample + pad),
			cy,
			e.label,
			ax.cfg.axis_color,
			.Left,
			.Middle,
		)
	}
}

line_width_of :: proc(cfg: PlotConfig) -> f64 {
	return cfg.line_width if cfg.line_width > 0 else 1.5
}
