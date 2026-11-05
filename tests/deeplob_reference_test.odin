package tests

import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:mem"

// DeepLOB data pipeline against numpy on a 9-event, 2-level LOBSTER book (layout per level:
// ask price, ask size, bid price, bid size): the windows' mid / spread / imbalance
// channels, the labels from the mean mid of the next events, the future returns, and the
// normaliser fitted over all windows. Changing an event after a window leaves its inputs
// unchanged (they hold no future data); the old pipeline labelled each window with a
// price move inside it.

@(private = "file")
LOB_X_REF := [?]f64 {
		100.0, 100.0, 100.02, 100.02,
		100.02, 100.02, 0.020000000000010232, 0.040000000000020464,
		0.020000000000010232, 0.040000000000020464, 0.020000000000010232, 0.040000000000020464,
		0.3333333333333333, 0.23529411764705882, 0.26666666666666666, 0.17647058823529413,
		0.2, 0.11764705882352941, 100.02, 100.02,
		100.02, 100.02, 100.06, 100.06,
		0.020000000000010232, 0.040000000000020464, 0.020000000000010232, 0.040000000000020464,
		0.020000000000010232, 0.040000000000020464, 0.26666666666666666, 0.17647058823529413,
		0.2, 0.11764705882352941, 0.13333333333333333, 0.058823529411764705,
		100.02, 100.02, 100.06, 100.06,
		100.04, 100.04, 0.020000000000010232, 0.040000000000020464,
		0.020000000000010232, 0.040000000000020464, 0.020000000000010232, 0.040000000000020464,
		0.2, 0.11764705882352941, 0.13333333333333333, 0.058823529411764705,
		0.06666666666666667, 0.0, 100.06, 100.06,
		100.04, 100.04, 100.04, 100.04,
		0.020000000000010232, 0.040000000000020464, 0.020000000000010232, 0.040000000000020464,
		0.020000000000010232, 0.040000000000020464, 0.13333333333333333, 0.058823529411764705,
		0.06666666666666667, 0.0, 0.0, -0.058823529411764705,
		100.04, 100.04, 100.04, 100.04,
		100.1, 100.1, 0.020000000000010232, 0.040000000000020464,
		0.020000000000010232, 0.040000000000020464, 0.020000000000010232, 0.040000000000020464,
		0.06666666666666667, 0.0, 0.0, -0.058823529411764705,
		-0.06666666666666667, -0.11764705882352941,
	}
@(private = "file")
LOB_Z_REF := [?]f64 {
		-1.6365646791493413, -1.6365646791493413, -0.7900657071748803, -0.7900657071748803,
		-0.7900657071748803, -0.7900657071748803, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		1.8371173070873834, 1.8371173070873836, 1.224744871391589, 1.2247448713915894,
		0.6123724356957946, 0.6123724356957946, -0.7900657071748803, -0.7900657071748803,
		-0.7900657071748803, -0.7900657071748803, 0.902932236774643, 0.902932236774643,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.224744871391589, 1.2247448713915894,
		0.6123724356957946, 0.6123724356957946, 0.0, 0.0,
		-0.7900657071748803, -0.7900657071748803, 0.902932236774643, 0.902932236774643,
		0.056433264800182015, 0.056433264800182015, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.6123724356957946, 0.6123724356957946, 0.0, 0.0,
		-0.6123724356957945, -0.6123724356957946, 0.902932236774643, 0.902932236774643,
		0.056433264800182015, 0.056433264800182015, 0.056433264800182015, 0.056433264800182015,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		-0.6123724356957945, -0.6123724356957946, -1.224744871391589, -1.2247448713915892,
		0.056433264800182015, 0.056433264800182015, 0.056433264800182015, 0.056433264800182015,
		2.5959301807235646, 2.5959301807235646, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		-0.6123724356957945, -0.6123724356957946, -1.224744871391589, -1.2247448713915892,
		-1.8371173070873836, -1.8371173070873836,
	}
@(private = "file")
LOB_FUT_REF := [?]f64 {
		0.00019996000799848979, -0.000199880071956815, 0.0005997600959615212, 0.0003998400639744215,
		0.00019980019980025965,
	}

@(private = "file")
_lob_book :: proc() -> []f64 {
	steps := [9]f64{0, 1, 1, 3, 2, 2, 5, 4, 6}
	L :: 2
	book := make([]f64, 9 * 4 * L)
	for r in 0 ..< 9 {
		m := 100 + 0.02 * steps[r]
		for lv in 0 ..< L {
			row := book[r * 4 * L + 4 * lv:][:4]
			row[0] = m + 0.01 + 0.01 * f64(lv)
			row[1] = 10 + f64(r) + 3 * f64(lv)
			row[2] = m - 0.01 - 0.01 * f64(lv)
			row[3] = 20 - f64(r) + f64(lv)
		}
	}
	return book
}

deeplob_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DeepLOB Pipeline Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	book := _lob_book()
	x, labels, fut := ml_fin.deeplob_lobster_windows(book, 9, 2, 3, 2, 1e-4)
	want_labels := []int{2, 0, 2, 2, 1}
	err_x, err_f := 0.0, 0.0
	for v, i in LOB_X_REF {err_x = max(err_x, abs(x.data.data[i] - v))}
	for v, i in LOB_FUT_REF {err_f = max(err_f, abs(fut[i] - v))}
	same_labels := len(labels) == len(want_labels)
	for v, i in want_labels {same_labels &&= labels[i] == v}
	shape_ok := x.shape == [4]int{5, 3, 3, 2}
	expect(
		&passed,
		&total,
		shape_ok && err_x < 1e-12 && err_f < 1e-15 && same_labels,
		fmt.tprintf(
			"5 windows of 3 events (horizon 2) vs numpy: channels max err %.1e, labels %v (%v), future returns max err %.1e",
			err_x,
			labels,
			want_labels,
			err_f,
		),
	)

	nm := ml_fin.deeplob_norm_fit(x)
	ml_fin.deeplob_norm_apply(x, nm)
	err_z := 0.0
	for v, i in LOB_Z_REF {err_z = max(err_z, abs(x.data.data[i] - v))}
	expect(
		&passed,
		&total,
		err_z < 1e-9,
		fmt.tprintf(
			"normaliser fitted over windows and time per channel and level vs numpy (constant spread only centred): max err %.1e",
			err_z,
		),
	)

	// no future data in the inputs: change the events after window 0 (rows 3.., its label
	// and return use rows 3 and 4) - window 0's inputs stay, its label follows
	book2 := _lob_book()
	for r in 3 ..< 9 {for j in 0 ..< 8 {if j % 4 == 0 || j % 4 == 2 {book2[r * 8 + j] -= 1}}}
	x2, labels2, _ := ml_fin.deeplob_lobster_windows(book2, 9, 2, 3, 2, 1e-4)
	x1, _, _ := ml_fin.deeplob_lobster_windows(book, 9, 2, 3, 2, 1e-4)
	same := true
	for i in 0 ..< 3 * 3 * 2 {same &&= x1.data.data[i] == x2.data.data[i]}
	expect(
		&passed,
		&total,
		same && labels2[0] == 0 && labels[0] == 2,
		fmt.tprintf(
			"inputs of a window hold no later event: prices after window 0 lowered by 1 -> its inputs unchanged %v, label up -> down (%d -> %d)",
			same,
			labels[0],
			labels2[0],
		),
	)
	fmt.printf("DeepLOB pipeline reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
