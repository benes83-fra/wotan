package nn

import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Augmentation Configuration
// ============================================================================

AugmentationConfig :: struct {
	rotation_range:    f64, // ±degrees (e.g., 15.0 for ±15°)
	translation_range: int, // ±pixels (e.g., 2 for ±2 pixels)
	scale_range:       f64, // ±percentage (e.g., 0.1 for ±10%)
	apply:             bool, // Enable/disable augmentation
}

// ============================================================================
// Affine Transformation
// ============================================================================

// Rotates one [height, width] image by angle_deg (counter-clockwise), scales
// it by `scale` about its center and shifts it by (tx, ty) pixels, as
// torchvision.transforms.functional.affine: every output pixel samples the
// input bilinearly at its inverse-mapped position, with zeros outside the
// image (torch grid_sample, padding "zeros", align_corners = False).
affine_transform_image :: proc(
	input: []f64,
	output: []f64,
	width: int,
	height: int,
	angle_deg: f64,
	tx: int,
	ty: int,
	scale: f64,
) {
	cx := f64(width - 1) / 2.0
	cy := f64(height - 1) / 2.0
	angle := angle_deg * math.PI / 180.0
	cos_a := math.cos(angle) / scale
	sin_a := math.sin(angle) / scale

	pixel :: proc(img: []f64, width, height, x, y: int) -> f64 {
		return img[y * width + x] if x >= 0 && x < width && y >= 0 && y < height else 0.0
	}

	for y in 0 ..< height {
		for x in 0 ..< width {
			// Position relative to the center before the shift, then the
			// inverse rotation and scaling
			dx := f64(x) - cx - f64(tx)
			dy := f64(y) - cy - f64(ty)
			x_src := cos_a * dx - sin_a * dy + cx
			y_src := sin_a * dx + cos_a * dy + cy

			x0 := int(math.floor(x_src))
			y0 := int(math.floor(y_src))
			fx := x_src - f64(x0)
			fy := y_src - f64(y0)
			v0 :=
				pixel(input, width, height, x0, y0) * (1.0 - fx) +
				pixel(input, width, height, x0 + 1, y0) * fx
			v1 :=
				pixel(input, width, height, x0, y0 + 1) * (1.0 - fx) +
				pixel(input, width, height, x0 + 1, y0 + 1) * fx
			output[y * width + x] = v0 * (1.0 - fy) + v1 * fy
		}
	}
}

// ============================================================================
// Batch Augmentation
// ============================================================================

// A random affine transformation per image of an [N, C, H, W] batch (the same
// for all its channels): angle uniform in ±rotation_range, integer shifts
// uniform in ±translation_range, scale uniform in 1 ± scale_range. Returns the
// batch itself if !config.apply.
augment_batch :: proc(
	batch: ^t.Tensor,
	config: AugmentationConfig,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	if !config.apply {
		return batch // No augmentation, return original
	}

	N := batch.shape[0]
	C := batch.shape[1]
	H := batch.shape[2]
	W := batch.shape[3]

	output := t.tensor_new_4d(N, C, H, W, false, allocator)
	shift :: proc(r: int) -> int {
		return int(math.floor(rand.float64() * f64(2 * r + 1))) - r
	}

	for n in 0 ..< N {
		angle_deg := (rand.float64() * 2.0 - 1.0) * config.rotation_range
		tx := shift(config.translation_range)
		ty := shift(config.translation_range)
		scale := 1.0 + (rand.float64() * 2.0 - 1.0) * config.scale_range

		for c in 0 ..< C {
			off := (n * C + c) * H * W
			affine_transform_image(
				batch.data.data[off:off + H * W],
				output.data.data[off:off + H * W],
				W,
				H,
				angle_deg,
				tx,
				ty,
				scale,
			)
		}
	}

	return output
}
