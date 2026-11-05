package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

// DeepLOB on the LOBSTER TSLA sample (2015-01-07, 10 levels). The file is a LOBSTER order
// book without header: per level ask price, ask size, bid price, bid size (prices x 10000).
// It holds only 50 events, so these runs exercise the pipeline (windows of 20 events,
// labels from the mean mid of the next 5, in-sample); they say nothing about
// predictability. (They used to read the columns as 10 bid prices, 10 bid sizes, ...,
// mixing prices and sizes, and to label each window with a move inside it.)

@(private = "file")
LOBSTER_URL :: "https://raw.githubusercontent.com/thertrader/Using-random-forest-to-model-limit-order-book-dynamic/master/TSLA_2015-01-07_34200000_57600000_orderbook_10_SAMPLE.csv"
@(private = "file")
LOB_LEVELS, LOB_T, LOB_H :: 10, 20, 5

// The book [rows, 40] (nil when the download fails), prices in dollars
@(private = "file")
_load_lobster :: proc(allocator: mem.Allocator) -> (book: []f64, rows: int) {
	text, ok := net.http_get(LOBSTER_URL, allocator)
	if !ok {return}
	out := make([dynamic]f64, allocator)
	for line in strings.split_lines_iterator(&text) {
		if len(strings.trim_space(line)) == 0 {continue}
		n := 0
		rest := line
		for field in strings.split_iterator(&rest, ",") {
			v, _ := strconv.parse_f64(strings.trim_space(field))
			append(&out, v / 10000 if n % 2 == 0 else v) // even columns are prices
			n += 1
		}
		if n != 4 * LOB_LEVELS {
			fmt.printf("Unexpected row with %d fields\n", n)
			return nil, 0
		}
		rows += 1
	}
	return out[:], rows
}

@(private = "file")
_lobster_windows :: proc(allocator: mem.Allocator) -> (x: ^t.Tensor, labels: []int, fut: []f64, ok: bool) {
	book, rows := _load_lobster(allocator)
	if rows == 0 {
		fmt.println("No LOBSTER data (offline?): skipped")
		return
	}
	x, labels, fut = ml_fin.deeplob_lobster_windows(book, rows, LOB_LEVELS, LOB_T, LOB_H, 1e-5, allocator)
	if x == nil {return}
	nm := ml_fin.deeplob_norm_fit(x, allocator)
	ml_fin.deeplob_norm_apply(x, nm)
	counts: [3]int
	for v in labels {counts[v] += 1}
	fmt.printf(
		"%d events -> %d windows of %d events, labels down/flat/up %v (mid of the next %d events)\n",
		rows,
		x.shape[0],
		LOB_T,
		counts,
		LOB_H,
	)
	return x, labels, fut, true
}

deeplob_real_data_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== DeepLOB on the LOBSTER TSLA sample (cross-entropy) ===")
	x, labels, _, ok := _lobster_windows(allocator)
	if !ok {return}
	n := x.shape[0]
	model := ml_fin.deeplob_new(
		{time_steps = LOB_T, price_levels = LOB_LEVELS, num_classes = 3, hidden_dim = 32, seq_backend = .Mamba},
	)
	defer ml_fin.deeplob_free(model)
	opt := nn.adam_new(1e-3)
	defer nn.adam_free(&opt)
	ml_fin.deeplob_add_to_adam(model, &opt)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	fmt.println("Epoch | loss    | in-sample accuracy")
	for epoch in 0 ..< 15 {
		t.graph_begin(&scope)
		nn.adam_zero_grad(&opt)
		logits := ml_fin.deeplob_forward(model, x)
		loss := t.tensor_cross_entropy_loss(logits, labels)
		t.tensor_backward(loss)
		nn.adam_step(&opt)
		hit := 0
		for v, i in labels {
			row := logits.data.data[i * 3:][:3]
			best := 0
			for c in 1 ..< 3 {if row[c] > row[best] {best = c}}
			if best == v {hit += 1}
		}
		if epoch % 5 == 0 || epoch == 14 {
			fmt.printf(" %3d  | %.5f | %5.1f%%\n", epoch + 1, loss.data.data[0], 100 * f64(hit) / f64(n))
		}
		t.graph_end(&scope)
	}
}

// Trained on the Sharpe ratio of the position E[position] = p_up - p_down times the
// mid return over the next 5 events
deeplob_real_data_sharpe_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== DeepLOB on the LOBSTER TSLA sample (Sharpe loss) ===")
	x, _, fut, ok := _lobster_windows(allocator)
	if !ok {return}
	n := x.shape[0]
	model := ml_fin.deeplob_new(
		{time_steps = LOB_T, price_levels = LOB_LEVELS, num_classes = 3, hidden_dim = 32, seq_backend = .Mamba},
	)
	defer ml_fin.deeplob_free(model)
	opt := nn.adam_new(1e-3)
	defer nn.adam_free(&opt)
	ml_fin.deeplob_add_to_adam(model, &opt)

	ret := t.tensor_new(l.matrix_new(f64, n, 1))
	copy(ret.data.data, fut)
	pos_w := t.tensor_new(l.matrix_new(f64, 3, 1)) // position per class: down -1, flat 0, up +1
	pos_w.data.data[0], pos_w.data.data[2] = -1, 1
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	fmt.println("Epoch | -Sharpe  | mean strategy return (bp)")
	for epoch in 0 ..< 15 {
		t.graph_begin(&scope)
		nn.adam_zero_grad(&opt)
		position := t.tensor_matmul(t.tensor_softmax(ml_fin.deeplob_forward(model, x)), pos_w)
		strategy := t.tensor_mul(position, ret)
		loss := t.tensor_sharpe_loss(strategy, 0.0)
		t.tensor_backward(loss)
		nn.adam_step(&opt)
		mean := 0.0
		for v in strategy.data.data {mean += v / f64(n)}
		if epoch % 5 == 0 || epoch == 14 {
			fmt.printf(" %3d  | %8.4f | %.3f\n", epoch + 1, loss.data.data[0], 1e4 * mean)
		}
		t.graph_end(&scope)
	}
	fmt.println("(in sample, 26 overlapping windows: no evidence of an edge)")
}
