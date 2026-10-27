package ML

import "core:math"
import "core:mem"

// ============================================================================
// Sample / Class Weights
// ============================================================================
// Fit procedures that support weights take an optional `sample_weight: []f64`
// (one non-negative weight per row, nil = all 1; pass it by name) and, for
// classifiers, a `class_weight` setting in their params. As in sklearn the
// effective weight of sample i is sample_weight[i] · class_weight[y_i]; a sample
// with weight w counts like w copies of it in the loss / impurity / C.

ClassWeight :: enum {
	None, // every class weight 1
	Balanced, // class c gets n_samples / (n_classes · n_c) (sklearn "balanced")
}

// Effective per-sample weights, or nil if all of them are 1 (the unweighted
// code paths stay bit-identical). Panics on a length mismatch, negative or
// non-finite weights, or an all-zero weight vector.
_effective_weights :: proc(
	y: []f64,
	sample_weight: []f64,
	class_weight: ClassWeight,
	name: string,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(y)
	if sample_weight == nil && class_weight == .None do return nil
	if sample_weight != nil && len(sample_weight) != n do panic("sample_weight: length does not match y")

	w := make([]f64, n, allocator)
	for i in 0 ..< n {w[i] = 1}
	if sample_weight != nil do copy(w, sample_weight)

	if class_weight == .Balanced {
		// classes and their counts (unweighted, as sklearn compute_class_weight)
		classes := make([dynamic]f64, 0, 8, context.temp_allocator)
		counts := make([dynamic]int, 0, 8, context.temp_allocator)
		defer {
			delete(classes)
			delete(counts)
		}
		for v in y {
			found := false
			for c, k in classes {if c == v {counts[k] += 1; found = true; break}}
			if !found {append(&classes, v); append(&counts, 1)}
		}
		K := f64(len(classes))
		for v, i in y {
			for c, k in classes {
				if c == v {w[i] *= f64(n) / (K * f64(counts[k])); break}
			}
		}
	}

	total := 0.0
	for v in w {
		if !(v >= 0) || math.is_inf(v) do panic("sample_weight: weights must be finite and >= 0")
		total += v
	}
	if !(total > 0) do panic("sample_weight: the weights sum to 0")
	_ = name
	return w
}
