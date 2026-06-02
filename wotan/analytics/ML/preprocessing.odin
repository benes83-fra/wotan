package ML

import l "../../linalg"
import "core:mem"
import "core:slice"
import "core:sort"
import "core:strings"

// ============================================================================
// One-Hot Encoder
// Transforms []string into a flat []f64 of binary indicators.
// ============================================================================

OneHotEncoder :: struct {
	categories: []string, // Sorted unique categories learned during fit
	drop_first: bool, // Drop first category to avoid multicollinearity
	allocator:  mem.Allocator,
}

// Fit: Learns the unique categories from the training data
ohe_fit :: proc(
	data: []string,
	drop_first: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> OneHotEncoder {
	enc: OneHotEncoder
	enc.drop_first = drop_first
	enc.allocator = allocator

	if len(data) == 0 {
		return enc
	}

	// 1. Copy data
	sorted_data := make([]string, len(data), allocator)
	copy(sorted_data, data)

	// ✅ FIX 1: Use sort.sort_slice. Strings are natively comparable in Odin.
	slice.sort(sorted_data)

	// 2. Count unique values
	unique_count := 1
	for i in 1 ..< len(sorted_data) {
		if sorted_data[i] != sorted_data[i - 1] {
			unique_count += 1
		}
	}

	// 3. Extract unique values
	enc.categories = make([]string, unique_count, allocator)
	enc.categories[0] = sorted_data[0]
	idx := 1
	for i in 1 ..< len(sorted_data) {
		if sorted_data[i] != sorted_data[i - 1] {
			enc.categories[idx] = sorted_data[i]
			idx += 1
		}
	}

	delete(sorted_data, allocator)
	return enc
}

// Transform: Converts []string to flat []f64
ohe_transform :: proc(
	enc: ^OneHotEncoder,
	data: []string,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	num_categories := len(enc.categories)
	if enc.drop_first {
		num_categories -= 1
	}

	out := make([]f64, len(data) * num_categories, allocator)

	// ✅ FIX 2: Odin's loop syntax is `for value, index in slice` (value first!)
	for val, i in data {
		idx := _binary_search(enc.categories, val)

		if idx >= 0 {
			if enc.drop_first {
				if idx > 0 {
					out[i * num_categories + (idx - 1)] = 1.0
				}
			} else {
				out[i * num_categories + idx] = 1.0
			}
		}
	}
	return out
}

// Binary search for sorted string array
_binary_search :: proc(arr: []string, target: string) -> int {
	left := 0
	right := len(arr) - 1
	for left <= right {
		mid := left + (right - left) / 2
		cmp := strings.compare(arr[mid], target)
		if cmp == 0 {
			return mid
		} else if cmp < 0 {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

ohe_free :: proc(enc: ^OneHotEncoder) {
	if len(enc.categories) > 0 {
		delete(enc.categories, enc.allocator)
	}
}


// ============================================================================
// Label Encoder
// Transforms []string into []f64 (0.0, 1.0, 2.0, ...)
// ============================================================================

LabelEncoder :: struct {
	classes:   []string,
	allocator: mem.Allocator,
}

le_fit :: proc(data: []string, allocator: mem.Allocator = context.allocator) -> LabelEncoder {
	enc := ohe_fit(data, false, allocator)

	le: LabelEncoder
	le.classes = enc.categories
	le.allocator = allocator

	return le
}

le_transform :: proc(
	le: ^LabelEncoder,
	data: []string,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	out := make([]f64, len(data), allocator)

	// ✅ FIX 2: Odin's loop syntax is `for value, index in slice`
	for val, i in data {
		idx := _binary_search(le.classes, val)
		if idx >= 0 {
			out[i] = f64(idx)
		} else {
			out[i] = -1.0
		}
	}
	return out
}

le_free :: proc(le: ^LabelEncoder) {
	if len(le.classes) > 0 {
		delete(le.classes, le.allocator)
	}
}
