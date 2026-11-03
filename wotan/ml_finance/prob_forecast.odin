package ml_finance

import ml "../analytics/ML"
import fin "../finance"
import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Probabilistic forecasting (DeepAR-style, direct multi-horizon)
// ============================================================================
// A sequence encoder (any nn.SeqBackend: LSTM, GRU, Mamba, TCN) reads a window
// x [batch, T, input_size]; its last step feeds a linear head that predicts
// the distribution of each of the next `horizon` values instead of a point:
//   .Gaussian  mean μ and standard deviation σ = softplus(raw) + min_sigma,
//              trained with the Gaussian negative log-likelihood
//   .StudentT  location μ, scale σ (as above) and degrees of freedom
//              ν = 2 + exp(raw) (finite variance; starts at 10), trained
//              with the Student-t negative log-likelihood: fat tails
//   .Quantile  the quantiles at the configured levels, trained with the
//              pinball loss (distribution-free)
// Each horizon h = 1..H gets its own distribution from the same encoding
// (direct forecasting: no feedback of predictions, no error build-up).
// prob_forecast_* turn the predictions into quantiles, intervals, VaR / ES,
// and score them (coverage, pinball loss, NLL, CRPS).

Prob_Output :: enum {
	Gaussian,
	Quantile,
	StudentT,
}

ProbForecasterConfig :: struct {
	input_size:  int,
	hidden_size: int,
	seq_backend: nn.SeqBackend,
	output:      Prob_Output,
	quantiles:   []f64, // .Quantile: levels in (0, 1), ascending (copied)
	min_sigma:   f64, // .Gaussian / .StudentT: floor of σ (0: 1e-4)
	horizon:     int, // number of future steps predicted (0: 1)
}

ProbForecaster :: struct {
	encoder:   ^nn.Sequential,
	head:      nn.LinearLayer, // hidden → horizon · (2, 3 or len(quantiles))
	config:    ProbForecasterConfig,
	allocator: mem.Allocator,
}

// Predicted distributions of a batch, graph tensors of the forward pass:
// mu, sigma (and nu for .StudentT) [batch, horizon], column h for step h + 1;
// quantiles [batch, horizon · Q], column h·Q + j the level quantiles[j] of step h + 1
Prob_Forecast :: struct {
	mu, sigma, nu: ^t.Tensor,
	quantiles:     ^t.Tensor,
}

PROB_MIN_NU :: 2.0 // ν = PROB_MIN_NU + exp(raw)

prob_forecaster_new :: proc(
	config: ProbForecasterConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^ProbForecaster {
	model := new(ProbForecaster, allocator)
	model.config = config
	model.allocator = allocator
	if config.min_sigma == 0 {model.config.min_sigma = 1e-4}
	if config.horizon == 0 {model.config.horizon = 1}
	H := model.config.horizon
	assert(H > 0, "prob_forecaster_new: horizon must be positive")
	n_out := (3 if config.output == .StudentT else 2) * H
	if config.output == .Quantile {
		assert(len(config.quantiles) > 0, "prob_forecaster_new: no quantile levels")
		for q, i in config.quantiles {
			assert(
				q > 0 && q < 1 && (i == 0 || q > config.quantiles[i - 1]),
				"prob_forecaster_new: levels must be ascending in (0, 1)",
			)
		}
		model.config.quantiles = slice.clone(config.quantiles, allocator)
		n_out = len(config.quantiles) * H
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
	if config.output == .StudentT {
		// ν starts at 10 (raw = log 8): moderately fat tails. On the log scale
		// ν can still reach 3 or 100 within a few hundred Adam steps (with
		// softplus, raw ≈ ν - 2 and ν stayed near its start value)
		for &b in model.head.bias.data.data[2 * H:] {b = math.ln(f64(8))}
	}
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

// x + c for a [N, H] tensor x and a scalar c (c is a graph constant)
@(private = "file")
_add_scalar :: proc(x: ^t.Tensor, c: f64, alloc: mem.Allocator) -> ^t.Tensor {
	row := t.tensor_new(l.matrix_new(f64, 1, x.data.cols, alloc), false, alloc)
	for &v in row.data.data {v = c}
	out := t.tensor_add_bias(x, row)
	if out.requires_grad {row.owned_by_graph = true} else {t.tensor_free(row)}
	return out
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
	// columns: μ for each horizon, then raw σ, then raw ν
	H := model.config.horizon
	f.mu = t.tensor_slice_cols(out, 0, H)
	f.sigma = _add_scalar(
		t.tensor_softplus(t.tensor_slice_cols(out, H, H)),
		model.config.min_sigma,
		alloc,
	)
	if model.config.output == .StudentT {
		f.nu = _add_scalar(t.tensor_exp(t.tensor_slice_cols(out, 2 * H, H)), PROB_MIN_NU, alloc)
	}
	return
}

// Training loss of a batch for the next values y [batch · horizon] (row-major:
// y[b·H + h] is step h + 1 after window b): Gaussian or Student-t NLL, or the
// pinball loss, averaged over the batch and the horizons
prob_forecaster_loss :: proc(model: ^ProbForecaster, x: ^t.Tensor, y: []f64) -> ^t.Tensor {
	H := model.config.horizon
	assert(len(y) == x.shape[0] * H, "prob_forecaster_loss: y must be [batch · horizon]")
	f := prob_forecaster_forward(model, x)
	if model.config.output == .Quantile {
		return nn.quantile_loss(f.quantiles, y, model.config.quantiles)
	}
	alloc := t._scope_or(x.allocator)
	target := t.tensor_new(l.matrix_new(f64, len(y) / H, H, alloc), false, alloc)
	copy(target.data.data, y)
	loss: ^t.Tensor
	if model.config.output == .StudentT {
		loss = nn.student_t_nll_loss(f.mu, f.sigma, f.nu, target)
	} else {
		loss = nn.gaussian_nll_loss(f.mu, t.tensor_mul(f.sigma, f.sigma), target)
	}
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

// Student-t (location mu, scale sigma, ν = nu degrees of freedom) negative
// log-likelihood of y (with all constants)
prob_forecast_student_t_nll :: proc(mu, sigma, nu, y: f64) -> f64 {
	z := (y - mu) / sigma
	lg_half, _ := math.lgamma(0.5 * nu)
	lg_half1, _ := math.lgamma(0.5 * (nu + 1))
	log_norm := lg_half - lg_half1 + 0.5 * math.ln(nu * math.PI) + math.ln(sigma)
	return log_norm + 0.5 * (nu + 1) * math.ln(1 + z * z / nu)
}

// Density of the standard t(ν) at z
@(private = "file")
_t_pdf :: proc(z, nu: f64) -> f64 {return math.exp(-prob_forecast_student_t_nll(0, 1, nu, z))}

// q-quantile of the location-scale t distribution
prob_forecast_student_t_quantile :: proc(mu, sigma, nu, q: f64) -> f64 {
	return mu + sigma * ml.student_t_ppf(q, nu)
}

// Value at risk and expected shortfall at tail level alpha of a return with a
// location-scale t distribution (ν > 1), as positive losses:
//   VaR = -(mu + sigma·t_alpha),
//   ES  = -(mu - sigma·f_ν(t_alpha)·(ν + t_alpha²) / ((ν - 1)·alpha))
prob_forecast_student_t_var_es :: proc(mu, sigma, nu, alpha: f64) -> (var, es: f64) {
	assert(nu > 1, "prob_forecast_student_t_var_es: ES needs ν > 1")
	q := ml.student_t_ppf(alpha, nu)
	return -(mu + sigma * q), -(mu - sigma * _t_pdf(q, nu) * (nu + q * q) / ((nu - 1) * alpha))
}

// CRPS of the location-scale t distribution (ν > 1) for an outcome y
// (closed form, Jordan, Krüger & Lerch 2019):
//   σ·[z(2F_ν(z) - 1) + 2f_ν(z)(ν + z²)/(ν - 1) - 2√ν·B(½, ν - ½) / ((ν - 1)·B(½, ν/2)²)]
prob_forecast_student_t_crps :: proc(mu, sigma, nu, y: f64) -> f64 {
	assert(nu > 1, "prob_forecast_student_t_crps: needs ν > 1")
	log_beta :: proc(a, b: f64) -> f64 {
		la, _ := math.lgamma(a)
		lb, _ := math.lgamma(b)
		lab, _ := math.lgamma(a + b)
		return la + lb - lab
	}
	z := (y - mu) / sigma
	spread :=
		2 *
		math.sqrt(nu) /
		(nu - 1) *
		math.exp(log_beta(0.5, nu - 0.5) - 2 * log_beta(0.5, 0.5 * nu))
	fit := z * (2 * ml.student_t_cdf(z, nu) - 1) + 2 * _t_pdf(z, nu) * (nu + z * z) / (nu - 1)
	return sigma * (fit - spread)
}

// Mean pinball loss of quantile predictions pred [N·Q] (row-major, levels
// quantiles) for outcomes y [N]. Several horizons: N = batch · horizon, the
// layout of Prob_Forecast.quantiles.
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
