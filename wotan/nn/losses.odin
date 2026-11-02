package nn

import t "../tensor"

// ============================================================================
// Probabilistic losses
// ============================================================================
// Built from differentiable tensor ops, so gradients reach every input that
// requires them. Targets are data (no gradient).

// Gaussian negative log-likelihood of target under N(mu, var), as torch's
// gaussian_nll_loss (full = false, reduction = mean):
//   mean(0.5 · (log v + (target - mu)² / v)),  v = max(var, eps)
// mu, var, target: [N, 1] (or any equal shape). The constant 0.5·log(2π) is
// left out. Unlike torch, the clamp at eps also stops the gradient of var.
gaussian_nll_loss :: proc(mu, var, target: ^t.Tensor, eps := 1e-6) -> ^t.Tensor {
	v := t.tensor_clamp(var, eps, max(f64))
	diff := t.tensor_sub(target, mu)
	terms := t.tensor_add(t.tensor_log(v), t.tensor_div(t.tensor_mul(diff, diff), v))
	return t.tensor_scale(t.tensor_mean(terms), 0.5)
}

// Pinball (quantile) loss of predicted quantiles pred [N, Q] (column j the
// quantile at level quantiles[j]) for targets [N]: the mean over all N·Q of
//   ρ_q(u) = q·u for u ≥ 0, (q - 1)·u for u < 0,  u = target - prediction
// Minimizing it per column gives the conditional q-quantile.
quantile_loss :: proc(pred: ^t.Tensor, target: []f64, quantiles: []f64) -> ^t.Tensor {
	N, Q := len(target), len(quantiles)
	assert(len(pred.data.data) == N * Q, "quantile_loss: pred must be [N, len(quantiles)]")
	alloc := t._scope_or(pred.allocator)
	// targets repeated over the quantiles, and q / 1 - q per column
	y := _constant(N * Q, {N, Q, 1, 1}, alloc)
	w_pos := _constant(N * Q, {N, Q, 1, 1}, alloc)
	w_neg := _constant(N * Q, {N, Q, 1, 1}, alloc)
	for c in ([]^t.Tensor{y, w_pos, w_neg}) {c.data.rows, c.data.cols = pred.data.rows, pred.data.cols}
	for i in 0 ..< N {
		for j in 0 ..< Q {
			assert(quantiles[j] > 0 && quantiles[j] < 1, "quantile_loss: levels must be in (0, 1)")
			y.data.data[i * Q + j] = target[i]
			w_pos.data.data[i * Q + j] = quantiles[j]
			w_neg.data.data[i * Q + j] = 1 - quantiles[j]
		}
	}
	u := t.tensor_sub(y, pred)
	terms := t.tensor_add(
		t.tensor_mul(w_pos, t.tensor_relu(u)),
		t.tensor_mul(w_neg, t.tensor_relu(t.tensor_neg(u))),
	)
	loss := t.tensor_mean(terms)
	_constant_done(y, loss)
	_constant_done(w_pos, loss)
	_constant_done(w_neg, loss)
	return loss
}
