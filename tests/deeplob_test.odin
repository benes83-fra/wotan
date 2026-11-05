package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

deeplob_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== DeepLOB Training Step Test ===")

	config := ml_fin.DeepLOBConfig {
		time_steps   = 100,
		price_levels = 10,
		num_classes  = 3,
		hidden_dim   = 64,
		seq_backend  = .Mamba,
	}

	model := ml_fin.deeplob_new(config, allocator)
	defer ml_fin.deeplob_free(model)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)

	ml_fin.deeplob_add_to_adam(model, &opt)

	// ✅ FIX: Dummy input now has 3 channels (Mid-Price, Spread, OBI)
	// Shape: [Batch=8, Channels=3, Time=100, Levels=10]
	input_data := l.matrix_new(f64, 8, 3 * 100 * 10, allocator)
	input_tensor := t.tensor_new(input_data, false, allocator)
	input_tensor.shape = [4]int{8, 3, 100, 10}

	// Forward pass
	logits := ml_fin.deeplob_forward(model, input_tensor)

	// Dummy targets for Cross Entropy Loss
	targets := []int{0, 1, 2, 0, 1, 2, 0, 1}
	loss := t.tensor_cross_entropy_loss(logits, targets)

	fmt.printf("Initial Loss: %.4f\n", loss.data.data[0])

	// Backward pass & Optimizer Step
	nn.adam_zero_grad(&opt)
	t.tensor_backward(loss)
	nn.adam_step(&opt)

	// Cleanup the ENTIRE computation graph safely
	t.tensor_free_graph(loss)
	t.tensor_free(input_tensor)

	fmt.println("✓ DeepLOB training step complete!")
}
// DeepLOB on a synthetic order book with a signal: the level-1 volume imbalance I (a
// persistent AR(1)) tilts the next mid moves, up with probability 0.25 + 0.2 I and down
// with 0.25 - 0.2 I (one tick each, else unchanged). Windows of 20 events, labels from
// the mean mid of the next 10 events; trained on the first 70% of the windows, scored on
// the last 30% (after a gap, so no event is in both) against always predicting the most
// frequent class of the training labels (4 epochs: more overfit the 4000 windows).
deeplob_talkative_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DeepLOB on a synthetic order book with an imbalance signal ===")
	context.allocator = allocator
	N_EVENTS, L, T, H :: 6000, 4, 20, 10
	TICK :: 0.01
	book := make([]f64, N_EVENTS * 4 * L)
	mid, imb := 100.0, 0.0
	for r in 0 ..< N_EVENTS {
		for lv in 0 ..< L {
			row := book[(r * L + lv) * 4:][:4]
			row[0] = mid + TICK / 2 + f64(lv) * TICK // ask price
			row[2] = mid - TICK / 2 - f64(lv) * TICK // bid price
			if lv == 0 {
				row[1], row[3] = 100 * (1 - imb), 100 * (1 + imb)
			} else {
				row[1], row[3] = rand.float64_range(50, 150), rand.float64_range(50, 150)
			}
		}
		u := rand.float64()
		if u < 0.25 + 0.2 * imb {mid += TICK} else if u > 0.75 + 0.2 * imb {mid -= TICK}
		imb = clamp(0.95 * imb + 0.3 * rand.float64_normal(0, 1), -0.9, 0.9)
	}
	x, labels, _ := ml_fin.deeplob_lobster_windows(book, N_EVENTS, L, T, H, 2e-5)
	n := x.shape[0]
	n_train := n * 7 / 10
	test_start := n_train + T + H // no event shared with the training windows
	W := 3 * T * L
	split :: proc(x: ^t.Tensor, lo, hi, W: int) -> ^t.Tensor {
		m := l.matrix_new(f64, hi - lo, W)
		copy(m.data, x.data.data[lo * W:hi * W])
		out := t.tensor_new(m)
		out.shape = {hi - lo, x.shape[1], x.shape[2], x.shape[3]}
		return out
	}
	x_train, x_test := split(x, 0, n_train, W), split(x, test_start, n, W)
	nm := ml_fin.deeplob_norm_fit(x_train) // training windows only
	ml_fin.deeplob_norm_apply(x_train, nm)
	ml_fin.deeplob_norm_apply(x_test, nm)
	y_train, y_test := labels[:n_train], labels[test_start:]
	// the imbalance rule: up when the last level-1 imbalance is positive, else down
	rule := 0
	for v, i in y_test {
		imb_last := x.data.data[(test_start + i) * W + 2 * T * L + (T - 1) * L]
		if (2 if imb_last > 0 else 0) == v {rule += 1}
	}
	counts: [3]int
	for v in y_train {counts[v] += 1}
	majority := 0
	for c in 1 ..< 3 {if counts[c] > counts[majority] {majority = c}}
	base := 0
	for v in y_test {if v == majority {base += 1}}
	fmt.printf(
		"%d windows (train %d, test %d); training labels down/flat/up %v\n",
		n,
		n_train,
		len(y_test),
		counts,
	)

	model := ml_fin.deeplob_new(
		{time_steps = T, price_levels = L, num_classes = 3, hidden_dim = 16, seq_backend = .LSTM},
	)
	defer ml_fin.deeplob_free(model)
	opt := nn.adam_new(1e-3)
	defer nn.adam_free(&opt)
	ml_fin.deeplob_add_to_adam(model, &opt)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	EPOCHS, BATCH :: 4, 64
	order := make([]int, n_train)
	for &v, i in order {v = i}
	accuracy :: proc(model: ^ml_fin.DeepLOB, x: ^t.Tensor, y: []int, scope: ^t.Graph_Scope) -> f64 {
		ml_fin.deeplob_eval(model)
		defer ml_fin.deeplob_train(model)
		t.graph_begin(scope, no_grad = true)
		defer t.graph_end(scope)
		logits := ml_fin.deeplob_forward(model, x)
		hit := 0
		for v, i in y {
			row := logits.data.data[i * 3:][:3]
			best := 0
			for c in 1 ..< 3 {if row[c] > row[best] {best = c}}
			if best == v {hit += 1}
		}
		return f64(hit) / f64(len(y))
	}
	fmt.println("Epoch | train loss | test accuracy")
	for epoch in 0 ..< EPOCHS {
		rand.shuffle(order)
		loss_sum := 0.0
		for b in 0 ..< n_train / BATCH {
			t.graph_begin(&scope)
			alloc := t._scope_or(allocator)
			xb := l.matrix_new(f64, BATCH, W, alloc)
			yb := make([]int, BATCH, alloc)
			for k in 0 ..< BATCH {
				i := order[b * BATCH + k]
				copy(xb.data[k * W:][:W], x_train.data.data[i * W:][:W])
				yb[k] = y_train[i]
			}
			xt := t.tensor_new(xb, false, alloc)
			xt.shape = {BATCH, 3, T, L}
			nn.adam_zero_grad(&opt)
			loss := t.tensor_cross_entropy_loss(ml_fin.deeplob_forward(model, xt), yb)
			t.tensor_backward(loss)
			nn.adam_step(&opt)
			loss_sum += loss.data.data[0]
			t.graph_end(&scope)
		}
		fmt.printf(
			" %3d  | %.4f     | %.1f%%\n",
			epoch + 1,
			loss_sum / f64(n_train / BATCH),
			100 * accuracy(model, x_test, y_test, &scope),
		)
	}
	acc := accuracy(model, x_test, y_test, &scope)
	base_acc := f64(base) / f64(len(y_test))
	fmt.printf(
		"(the imbalance rule, up if the last level-1 imbalance > 0 else down: %.1f%%)\n",
		100 * f64(rule) / f64(len(y_test)),
	)
	ok := acc > base_acc + 0.10
	fmt.printf(
		"%s held-out accuracy %.1f%% vs %.1f%% for always predicting the most frequent class (at least 10 points above)\n",
		"OK  " if ok else "FAIL",
		100 * acc,
		100 * base_acc,
	)
	record_failures(int(!ok))
}

// DeepLOB gradients against central finite differences, for every sequence
// backend: every parameter (CNN, inception block, sequence layer, head) gets a
// gradient, and it matches. The last time step used to be copied out of the
// sequence layer without a graph link, so only the head was trained.
deeplob_gradient_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DeepLOB gradients vs finite differences ===")
	context.allocator = allocator
	passed, total := 0, 0

	@(static) R: []f64 // weights of the scalar loss Σ R ⊙ logits
	loss :: proc(model: ^ml_fin.DeepLOB, x: ^t.Tensor) -> ^t.Tensor {
		logits := ml_fin.deeplob_forward(model, x)
		alloc := t._scope_or(context.allocator) // the active scope
		w := t.tensor_new(
			l.matrix_new(f64, logits.data.rows, logits.data.cols, alloc),
			false,
			alloc,
		)
		w.shape = logits.shape
		copy(w.data.data, R)
		return t.tensor_sum(t.tensor_mul(logits, w))
	}

	for backend in nn.SeqBackend {
		config := ml_fin.DeepLOBConfig {
			time_steps   = 8,
			price_levels = 4,
			num_classes  = 3,
			hidden_dim   = 4,
			seq_backend  = backend,
		}
		model := ml_fin.deeplob_new(config)
		defer ml_fin.deeplob_free(model)
		ml_fin.deeplob_eval(model) // no dropout
		opt := nn.adam_new(0.001)
		defer nn.adam_free(&opt)
		ml_fin.deeplob_add_to_adam(model, &opt)

		B := 2
		xd := l.matrix_new(f64, B, 3 * 8 * 4)
		for &v in xd.data {v = rand.float64_range(-1, 1)}
		x := t.tensor_new(xd, false)
		x.shape = {B, 3, 8, 4}
		defer t.tensor_free(x)
		R = make([]f64, B * 3)
		defer delete(R)
		for &v in R {v = rand.float64_range(-1, 1)}

		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		value :: proc(model: ^ml_fin.DeepLOB, x: ^t.Tensor, scope: ^t.Graph_Scope) -> f64 {
			t.graph_begin(scope, no_grad = true)
			defer t.graph_end(scope)
			return loss(model, x).data.data[0]
		}

		nn.adam_zero_grad(&opt)
		t.graph_begin(&scope)
		t.tensor_backward(loss(model, x))
		t.graph_end(&scope)

		with_grad, max_err := 0, 0.0
		h := 1e-6
		for p in opt.parameters {
			nonzero := false
			for g in p.grad.data {nonzero ||= g != 0}
			if nonzero {with_grad += 1}
			n := len(p.data.data)
			for i in ([]int{0, n / 2, n - 1}) {
				x0 := p.data.data[i]
				p.data.data[i] = x0 + h
				up := value(model, x, &scope)
				p.data.data[i] = x0 - h
				down := value(model, x, &scope)
				p.data.data[i] = x0
				fd := (up - down) / (2 * h)
				max_err = max(max_err, abs(fd - p.grad.data[i]) / max(1, abs(fd)))
			}
		}
		n_params := len(opt.parameters)
		ok := with_grad == n_params && max_err < 1e-6
		total += 1
		if ok {passed += 1}
		fmt.printf(
			"%s DeepLOB with %v: %d/%d parameters get a gradient, max error vs finite differences %.1e\n",
			"OK  " if ok else "FAIL",
			backend,
			with_grad,
			n_params,
			max_err,
		)
	}
	fmt.printf("DeepLOB gradients: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
