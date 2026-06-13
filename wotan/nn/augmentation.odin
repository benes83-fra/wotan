package nn

import l "../linalg"
import t "../tensor"
import "core:fmt"
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
// Affine Transformation Helpers
// ============================================================================

// _apply_affine_transform applies rotation, translation, and scaling to a single image
// Input: (1, 28, 28) image as flat array
// Output: transformed image as flat array
_apply_affine_transform :: proc(
	input: []f64,
	output: []f64,
	width: int,
	height: int,
	angle_rad: f64,
	tx: int,
	ty: int,
	scale: f64,
) {
	center_x := f64(width) / 2.0
	center_y := f64(height) / 2.0

	cos_a := math.cos(angle_rad)
	sin_a := math.sin(angle_rad)

	// Apply transformation to each output pixel
	for y in 0 ..< height {
		for x in 0 ..< width {
			// Translate to center
			x_centered := f64(x) - center_x - f64(tx)
			y_centered := f64(y) - center_y - f64(ty)

			// Apply inverse rotation and scaling
			x_rot := (x_centered * cos_a + y_centered * sin_a) / scale
			y_rot := (-x_centered * sin_a + y_centered * cos_a) / scale

			// Translate back
			x_src := x_rot + center_x
			y_src := y_rot + center_y

			// Bilinear interpolation
			out_idx := y * width + x
			if x_src >= 0 && x_src < f64(width - 1) && y_src >= 0 && y_src < f64(height - 1) {
				x0 := int(x_src)
				y0 := int(y_src)
				x1 := x0 + 1
				y1 := y0 + 1

				fx := x_src - f64(x0)
				fy := y_src - f64(y0)

				v00 := input[y0 * width + x0]
				v01 := input[y0 * width + x1]
				v10 := input[y1 * width + x0]
				v11 := input[y1 * width + x1]

				// Bilinear interpolation
				v0 := v00 * (1.0 - fx) + v01 * fx
				v1 := v10 * (1.0 - fx) + v11 * fx
				output[out_idx] = v0 * (1.0 - fy) + v1 * fy
			} else {
				output[out_idx] = 0.0 // Zero padding
			}
		}
	}
}

// ============================================================================
// Batch Augmentation
// ============================================================================

// augment_batch applies random augmentations to a batch of images
// Input: (N, 1, 28, 28) tensor
// Output: augmented (N, 1, 28, 28) tensor
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

	// Create output tensor
	output := t.tensor_new_4d(N, C, H, W, false, allocator)

	for n in 0 ..< N {
		// Generate random transformation parameters
		angle_deg := (rand.float64() * 2.0 - 1.0) * config.rotation_range
		angle_rad := angle_deg * math.PI / 180.0

		tx := int((rand.float64() * 2.0 - 1.0) * f64(config.translation_range))
		ty := int((rand.float64() * 2.0 - 1.0) * f64(config.translation_range))

		scale := 1.0 + (rand.float64() * 2.0 - 1.0) * config.scale_range

		// Get input and output slices for this image
		img_offset := n * C * H * W
		input_slice := batch.data.data[img_offset:img_offset + C * H * W]
		output_slice := output.data.data[img_offset:img_offset + C * H * W]

		// Apply transformation
		_apply_affine_transform(input_slice, output_slice, W, H, angle_rad, tx, ty, scale)
	}

	return output
}
