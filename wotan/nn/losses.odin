package nn

import t "../tensor"
import "core:math"
import "core:slice"

// ============================================================================
// Probabilistic losses
// ============================================================================
// Built from differentiable tensor ops, so gradients reach every input that
// requires them. Targets are data (no gradient). Inputs may also be data
// (e.g. a fixed variance or ν): the intermediates computed from data alone
// are freed with the loss's graph (or right away) as well.

// Settles the intermediates of a loss: an op output without gradient keeps no
// inputs, so tensor_free_graph(loss) would not reach intermediates computed
// from data alone. Those that an op of the loss's graph reads (its backward
// may) are handed to the graph; the others are freed now.
@(private = "file")
_settle :: proc(loss: ^t.Tensor, nodes: ..^t.Tensor) {
	reached := make([dynamic]^t.Tensor, context.temp_allocator)
	stack := make([dynamic]^t.Tensor, context.temp_allocator)
	defer {delete(reached); delete(stack)}
	append(&stack, loss)
	for len(stack) > 0 {
		n := pop(&stack)
		for x in n.inputs {
			if slice.contains(nodes, x) && !slice.contains(reached[:], x) {
				append(&reached, x)
				append(&stack, x)
			}
		}
	}
	for x in nodes {
		if x == loss || x.requires_grad {continue} 	// the loss, or a node of its graph
		if slice.contains(reached[:], x) {x.owned_by_graph = true} else {t.tensor_free(x)}
	}
}

// Gaussian negative log-likelihood of target under N(mu, var), as torch's
// gaussian_nll_loss (full = false, reduction = mean):
//   mean(0.5 · (log v + (target - mu)² / v)),  v = max(var, eps)
// mu, var, target: [N, 1] (or any equal shape). The constant 0.5·log(2π) is
// left out. Unlike torch, the clamp at eps also stops the gradient of var.
gaussian_nll_loss :: proc(mu, var, target: ^t.Tensor, eps := 1e-6) -> ^t.Tensor {
	v := t.tensor_clamp(var, eps, max(f64))
	diff := t.tensor_sub(target, mu)
	log_v := t.tensor_log(v)
	sq := t.tensor_mul(diff, diff)
	ratio := t.tensor_div(sq, v)
	terms := t.tensor_add(log_v, ratio)
	mean := t.tensor_mean(terms)
	loss := t.tensor_scale(mean, 0.5)
	_settle(loss, v, diff, log_v, sq, ratio, terms, mean)
	return loss
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
	resid := t.tensor_sub(target, mu)
	z := t.tensor_div(resid, sigma)
	nu1 := t.tensor_add(nu, ones)
	half_nu1 := t.tensor_scale(nu1, 0.5)
	z2 := t.tensor_mul(z, z)
	z2_nu := t.tensor_div(z2, nu)
	kernel := t.tensor_add(ones, z2_nu)
	log_kernel := t.tensor_log(kernel)
	half_nu := t.tensor_scale(nu, 0.5)
	lg_half_nu := t.tensor_lgamma(half_nu)
	lg_half_nu1 := t.tensor_lgamma(half_nu1)
	lg_diff := t.tensor_sub(lg_half_nu, lg_half_nu1)
	log_nu := t.tensor_log(nu)
	half_log_nu := t.tensor_scale(log_nu, 0.5)
	log_sigma := t.tensor_log(sigma)
	scale_terms := t.tensor_add(half_log_nu, log_sigma)
	tail := t.tensor_mul(half_nu1, log_kernel)
	rest := t.tensor_add(scale_terms, tail)
	terms := t.tensor_add(lg_diff, rest)
	mean := t.tensor_mean(terms)
	half_log_pi := _constant(1, {1, 1, 1, 1}, alloc, {0.5 * math.ln(f64(math.PI))})
	loss := t.tensor_add_bias(mean, half_log_pi)
	_settle(
		loss,
		ones,
		resid,
		z,
		nu1,
		half_nu1,
		z2,
		z2_nu,
		kernel,
		log_kernel,
		half_nu,
		lg_half_nu,
		lg_half_nu1,
		lg_diff,
		log_nu,
		half_log_nu,
		log_sigma,
		scale_terms,
		tail,
		rest,
		terms,
		mean,
		half_log_pi,
	)
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
	pos := t.tensor_relu(u)
	neg_u := t.tensor_neg(u)
	neg := t.tensor_relu(neg_u)
	pos_w := t.tensor_mul(w_pos, pos)
	neg_w := t.tensor_mul(w_neg, neg)
	terms := t.tensor_add(pos_w, neg_w)
	loss := t.tensor_mean(terms)
	_settle(loss, y, w_pos, w_neg, u, pos, neg_u, neg, pos_w, neg_w, terms)
	return loss
}
