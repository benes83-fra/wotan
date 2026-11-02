package ml_finance

import fin "../finance"
import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Probabilistic forecasting (DeepAR-style, one step ahead)
// ============================================================================
// A sequence encoder (any nn.SeqBackend: LSTM, GRU, Mamba, TCN) reads a window
// x [batch, T, input_size]; its last step feeds a linear head that predicts
// the distribution of the next value instead of a point:
//   .Gaussian  mean μ and standard deviation σ = softplus(raw) + min_sigma,
//              trained with the Gaussian negative log-likelihood
//   .Quantile  the quantiles at the configured levels, trained with the
//              pinball loss (distribution-free)
// prob_forecast_* turn the predictions into quantiles, intervals, VaR / ES,
// and score them (coverage, pinball loss, NLL, CRPS).

Prob_Output :: enum {
	Gaussian,
	Quantile,
}

ProbForecasterConfig :: struct {
	input_size:  int,
	hidden_size: int,
	seq_backend: nn.SeqBackend,
	output:      Prob_Output,
	quantiles:   []f64, // .Quantile: levels in (0, 1), ascending (copied)
	min_sigma:   f64, // .Gaussian: floor of σ (0: 1e-4)
}

ProbForecaster :: struct {
	encoder:   ^nn.Sequential,
	head:      nn.LinearLayer, // hidden → 2 (μ, raw σ) or len(quantiles)
	config:    ProbForecasterConfig,
	allocator: mem.Allocator,
}

// Predicted distributions of a batch: mu, sigma [batch, 1] (.Gaussian) or
// quantiles [batch, Q] (.Quantile); graph tensors of the forward pass
Prob_Forecast :: struct {
	mu, sigma: ^t.Tensor,
	quantiles: ^t.Tensor,
}

prob_forecaster_new :: proc(
	config: ProbForecasterConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^ProbForecaster {
	model := new(ProbForecaster, allocator)
	model.config = config
	model.allocator = allocator
	if config.min_sigma == 0 {model.config.min_sigma = 1e-4}
	n_out := 2
	if config.output == .Quantile {
		assert(len(config.quantiles) > 0, "prob_forecaster_new: no quantile levels")
		for q, i in config.quantiles {
			assert(
				q > 0 && q < 1 && (i == 0 || q > config.quantiles[i - 1]),
				"prob_forecaster_new: levels must be ascending in (0, 1)",
			)
		}
		model.config.quantiles = slice.clone(config.quantiles, allocator)
		n_out = len(config.quantiles)
	}
	model.encoder = nn.sequential_new(allocator)
	nn.sequential_add_seq_block(
		model.encoder,
		config.seq_backend,
		config.input_size,
		config.hidden_size,
		16,
		allocator,
	)
	model.head = nn.linear_layer_new(config.hidden_size, n_out, allocator)
	return model
}

prob_forecaster_free :: proc(model: ^ProbForecaster) {
	nn.sequential_free(model.encoder)
	nn.linear_layer_free(&model.head)
	if model.config.output == .Quantile {delete(model.config.quantiles, model.allocator)}
	free(model, model.allocator)
}

// All trainable parameters (encoder, then head)
prob_forecaster_parameters :: proc(
	model: ^ProbForecaster,
	allocator := context.allocator,
) -> [dynamic]^t.Tensor {
	ps := nn.sequential_parameters(model.encoder, allocator)
	append(&ps, model.head.weights, model.head.bias)
	return ps
}

prob_forecaster_add_to_adam :: proc(model: ^ProbForecaster, opt: ^nn.Adam) {
	ps := prob_forecaster_parameters(model, context.temp_allocator)
	defer delete(ps)
	for p in ps {nn.adam_add_param(opt, p)}
}

// Train / eval mode of the encoder (dropout of a TCN, Sequential dropout layers)
prob_forecaster_train :: proc(model: ^ProbForecaster, training: bool) {
	model.encoder.training = training
}

prob_forecaster_forward :: proc(model: ^ProbForecaster, x: ^t.Tensor) -> (f: Prob_Forecast) {
	batch, seq_len := x.shape[0], x.shape[1]
	alloc := t._scope_or(x.allocator)
	h := nn.sequential_forward(model.encoder, x)
	last := nn.as_matrix(
		nn.sequence_step(h, seq_len - 1, alloc),
		batch,
		model.config.hidden_size,
		alloc,
	)
	out := nn.linear_forward(&model.head, last) // [batch, n_out]
	if model.config.output == .Quantile {
		f.quantiles = out
		return
	}
	f.mu = t.tensor_slice_gate(out, 0)
	// σ = softplus(raw) + min_sigma
	floor := t.tensor_new(l.matrix_new(f64, 1, 1, alloc), false, alloc)
	floor.data.data[0] = model.config.min_sigma
	f.sigma = t.tensor_add_bias(t.tensor_softplus(t.tensor_slice_gate(out, 1)), floor)
	if f.sigma.requires_grad {floor.owned_by_graph = true} else {t.tensor_free(floor)}
	return
}

// Training loss of a batch: Gaussian NLL or pinball loss of the next values y
prob_forecaster_loss :: proc(model: ^ProbForecaster, x: ^t.Tensor, y: []f64) -> ^t.Tensor {
	f := prob_forecaster_forward(model, x)
	if model.config.output == .Quantile {
		return nn.quantile_loss(f.quantiles, y, model.config.quantiles)
	}
	alloc := t._scope_or(x.allocator)
	target := t.tensor_new(l.matrix_new(f64, len(y), 1, alloc), false, alloc)
	copy(target.data.data, y)
	loss := nn.gaussian_nll_loss(f.mu, t.tensor_mul(f.sigma, f.sigma), target)
	if loss.requires_grad {target.owned_by_graph = true} else {t.tensor_free(target)}
	return loss
}

// ============================================================================
// Using and scoring the predictions
// ============================================================================

@(private = "file")
_norm_pdf :: proc(z: f64) -> f64 {return math.exp(-0.5 * z * z) / math.sqrt(f64(2 * math.PI))}

@(private = "file")
_norm_cdf :: proc(z: f64) -> f64 {return 0.5 * (1 + math.erf(z / math.SQRT_TWO))}

// q-quantile of N(mu, sigma²)
prob_forecast_gaussian_quantile :: proc(mu, sigma, q: f64) -> f64 {
	return mu + sigma * fin.norm_inv(q)
}

// Value at risk and expected shortfall at tail level alpha (e.g. 0.05) of a
// return ~ N(mu, sigma²), as positive losses:
//   VaR = -(mu + sigma·z_alpha),  ES = -(mu - sigma·φ(z_alpha) / alpha)
prob_forecast_gaussian_var_es :: proc(mu, sigma, alpha: f64) -> (var, es: f64) {
	z := fin.norm_inv(alpha)
	return -(mu + sigma * z), -(mu - sigma * _norm_pdf(z) / alpha)
}

// Continuous ranked probability score of N(mu, sigma²) for an outcome y
// (closed form; lower is better, in the units of y)
prob_forecast_gaussian_crps :: proc(mu, sigma, y: f64) -> f64 {
	z := (y - mu) / sigma
	return sigma * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z) - 1 / math.sqrt(f64(math.PI)))
}

// Gaussian negative log-likelihood of y (with the constant 0.5·log 2π)
prob_forecast_gaussian_nll :: proc(mu, sigma, y: f64) -> f64 {
	z := (y - mu) / sigma
	return 0.5 * z * z + math.ln(sigma) + 0.5 * math.ln(f64(2 * math.PI))
}

// Mean pinball loss of quantile predictions pred [N·Q] (row-major, levels
// quantiles) for outcomes y [N]
prob_forecast_pinball :: proc(pred: []f64, y: []f64, quantiles: []f64) -> f64 {
	Q := len(quantiles)
	s := 0.0
	for yi, i in y {
		for q, j in quantiles {
			u := yi - pred[i * Q + j]
			s += q * u if u >= 0 else (q - 1) * u
		}
	}
	return s / f64(len(y) * Q)
}

// Share of outcomes y[i] ≤ pred[i·Q + j] per level j (calibrated: ≈ quantiles[j])
prob_forecast_coverage :: proc(
	pred: []f64,
	y: []f64,
	Q: int,
	allocator := context.allocator,
) -> []f64 {
	cov := make([]f64, Q, allocator)
	for yi, i in y {
		for j in 0 ..< Q {if yi <= pred[i * Q + j] {cov[j] += 1}}
	}
	for &c in cov {c /= f64(len(y))}
	return cov
}

// Sorts each row of quantile predictions pred [N·Q] (monotone rearrangement,
// Chernozhukov et al. 2010): removes crossing quantiles and never increases
// the pinball loss. Returns how many rows crossed.
prob_forecast_sort_quantiles :: proc(pred: []f64, Q: int) -> (crossed: int) {
	for i in 0 ..< len(pred) / Q {
		row := pred[i * Q:(i + 1) * Q]
		if !slice.is_sorted(row) {
			crossed += 1
			slice.sort(row)
		}
	}
	return
}
