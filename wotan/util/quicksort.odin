package util
// ✅ FIX: Custom Quicksort for f64 to avoid core:sort/core:slice API quirks
quicksort_f64 :: proc(arr: []f64, low, high: int) {
	if low < high {
		pi := partition_f64(arr, low, high)
		quicksort_f64(arr, low, pi - 1)
		quicksort_f64(arr, pi + 1, high)
	}
}

partition_f64 :: proc(arr: []f64, low, high: int) -> int {
	pivot := arr[high]
	i := low - 1
	for j in low ..< high {
		if arr[j] <= pivot {
			i += 1
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i + 1], arr[high] = arr[high], arr[i + 1]
	return i + 1
}
