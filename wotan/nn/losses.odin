package nn

import t "../tensor"
import "core:math"

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

// Student-t negative log-likelihood of target under the location-scale t
// distribution with location mu, scale sigma > 0 and ν = nu > 0 degrees of
// freedom, as -torch.distributions.StudentT(nu, mu, sigma).log_prob(target)
// .mean() (with all constants):
//   mean(log Γ(ν/2) - log Γ((ν+1)/2) + ½·log(νπ) + log σ + (ν+1)/2 · log(1 + z²/ν)),
//   z = (target - mu) / sigma
// mu, sigma, nu, target: equal shapes; nu may be learned (ψ via tensor_lgamma).
student_t_nll_loss :: proc(mu, sigma, nu, target: ^t.Tensor) -> ^t.Tensor {
	alloc := t._scope_or(mu.allocator)
	ones := _constant(len(mu.data.data), mu.shape, alloc)
	ones.data.rows, ones.data.cols = mu.data.rows, mu.data.cols
	for &v in ones.data.data {v = 1}
	z := t.tensor_div(t.tensor_sub(target, mu), sigma)
	half_nu1 := t.tensor_scale(t.tensor_add(nu, ones), 0.5)
	log_kernel := t.tensor_log(t.tensor_add(ones, t.tensor_div(t.tensor_mul(z, z), nu)))
	terms := t.tensor_add(
		t.tensor_sub(t.tensor_lgamma(t.tensor_scale(nu, 0.5)), t.tensor_lgamma(half_nu1)),
		t.tensor_add(
			t.tensor_add(t.tensor_scale(t.tensor_log(nu), 0.5), t.tensor_log(sigma)),
			t.tensor_mul(half_nu1, log_kernel),
		),
	)
	half_log_pi := _constant(1, {1, 1, 1, 1}, alloc, {0.5 * math.ln(f64(math.PI))})
	loss := t.tensor_add_bias(t.tensor_mean(terms), half_log_pi)
	_constant_done(ones, loss)
	_constant_done(half_log_pi, loss)
	return loss
}

// Pinball (quantile) loss of predicted quantiles pred [N, Q] (column j the
// quantile at level quantiles[j]) for targets [N]: the mean over all N·Q of
//   ρ_q(u) = q·u for u ≥ 0, (q - 1)·u for u < 0,  u = target - prediction
// Minimizing it per column gives the conditional q-quantile.
// Several targets per row (e.g. horizons): pred [N, H·Q] with column h·Q + j
// for target[n·H + h], targets [N·H] row-major.
quantile_loss :: proc(pred: ^t.Tensor, target: []f64, quantiles: []f64) -> ^t.Tensor {
	N, Q := len(target), len(quantiles)
	assert(
		len(pred.data.data) == N * Q,
		"quantile_loss: pred must have len(quantiles) columns per target",
	)
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
