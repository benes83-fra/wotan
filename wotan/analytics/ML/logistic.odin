package ML

import l "../../linalg"
import optim "../../optimize"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Logistic Regression Structures
// ============================================================================

LogisticRegression :: struct {
	weights:   []f64, // Feature weights
	bias:      f64, // Intercept
	n_iter:    int,
	converged: bool,
	allocator: mem.Allocator,
}

LogisticParams :: struct {
	C:              f64, // Inverse of regularization strength, as in sklearn (math.INF_F64 = no penalty)
	max_iter:       int,
	tol:            f64, // Convergence tolerance (gradient norm)
	learning_rate:  f64, // Initial learning rate / max step size
	fit_intercept:  bool,
	optimizer_type: optim.OptimizerType, // .SGD, .Adam, .RMSProp, or .LBFGS
	class_weight:   ClassWeight, // weight of class c in the loss (sklearn class_weight)
}

// ============================================================================
// Internal: Context & Callbacks (No closures!)
// ============================================================================

_LogisticContext :: struct {
	X:          ^l.Matrix(f64),
	y:          []f64, // Expected to be 0.0 or 1.0
	w:          []f64, // per-sample weights (nil = 1)
	C:          f64,
	fit_bias:   bool,
	n_samples:  int,
	n_features: int,
}

// Numerically stable sigmoid
_sigmoid :: proc(z: f64) -> f64 {
	if z >= 0.0 {
		return 1.0 / (1.0 + math.exp(-z))
	}
	exp_z := math.exp(z)
	return exp_z / (1.0 + exp_z)
}

// Numerically stable log
_safe_log :: proc(x: f64) -> f64 {
	return math.ln_f64(math.max(x, 1e-15))
}

// Objective (sklearn LogisticRegression, L2 penalty), divided by n·C:
//   (1/n) Σ s_i log-loss_i + 1/(2nC) ||w||²   (bias not penalised, s_i = sample weight)
// The log-loss is evaluated as softplus(z) - y·z, exact for any z.
_logistic_loss :: proc(params_vec: []f64, user_data: rawptr) -> f64 {
	ctx := cast(^_LogisticContext)(user_data)
	w := params_vec[0:ctx.n_features]
	b := params_vec[ctx.n_features]

	loss := 0.0
	reg_term := 0.0

	for f in 0 ..< ctx.n_features {
		reg_term += w[f] * w[f]
	}

	for i in 0 ..< ctx.n_samples {
		x_i := ctx.X.data[i * ctx.n_features:i * ctx.n_features + ctx.n_features]
		z := b + l.dot_simd(w, x_i)

		// Binary cross-entropy: log(1 + e^z) - y z
		li := math.max(z, 0.0) + math.ln_f64(1.0 + math.exp(-math.abs(z))) - ctx.y[i] * z
		if ctx.w != nil do li *= ctx.w[i]
		loss += li
	}

	lambda := 1.0 / (f64(ctx.n_samples) * ctx.C)
	return (loss / f64(ctx.n_samples)) + 0.5 * lambda * reg_term
}

_logistic_gradient :: proc(params_vec: []f64, user_data: rawptr, grad: []f64) {
	ctx := cast(^_LogisticContext)(user_data)
	w := params_vec[0:ctx.n_features]
	b := params_vec[ctx.n_features]

	// Initialize gradient to zero
	for i in 0 ..< len(grad) {
		grad[i] = 0.0
	}

	lambda := 1.0 / (f64(ctx.n_samples) * ctx.C)

	for i in 0 ..< ctx.n_samples {
		x_i := ctx.X.data[i * ctx.n_features:i * ctx.n_features + ctx.n_features]
		z := b + l.dot_simd(w, x_i)
		y_hat := _sigmoid(z)
		err := y_hat - ctx.y[i]
		if ctx.w != nil do err *= ctx.w[i]

		// Accumulate gradients
		for f in 0 ..< ctx.n_features {
			grad[f] += err * x_i[f]
		}
		grad[ctx.n_features] += err
	}

	// Average and add regularization to weights (not bias)
	for f in 0 ..< ctx.n_features {
		grad[f] = (grad[f] / f64(ctx.n_samples)) + lambda * w[f]
	}
	grad[ctx.n_features] /= f64(ctx.n_samples)
	if !ctx.fit_bias do grad[ctx.n_features] = 0.0
}

// ============================================================================
// Public API: Fit Logistic Regression
// ============================================================================

logistic_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64, // Labels: 0.0 or 1.0
	params: LogisticParams,
	allocator: mem.Allocator = context.allocator,
	sample_weight: []f64 = nil, // optional, one weight >= 0 per row (sklearn sample_weight)
) -> LogisticRegression {
	n_features := X.cols
	n_params := n_features + 1
	if len(y) != X.rows do panic("logistic_fit: y length mismatch")
	if X.rows == 0 do panic("logistic_fit: no samples")
	if !(params.C > 0.0) do panic("logistic_fit: C must be > 0 (use math.INF_F64 for no penalty)")
	for v in y {
		if v != 0.0 && v != 1.0 do panic("logistic_fit: labels must be 0 or 1")
	}

	params_vec := make([]f64, n_params, allocator)
	defer delete(params_vec, allocator)

	grad := make([]f64, n_params, allocator)
	defer delete(grad, allocator)

	w := _effective_weights(y, sample_weight, params.class_weight, "logistic_fit", allocator)
	defer delete(w, allocator)

	ctx := _LogisticContext {
		X          = X,
		y          = y,
		w          = w,
		C          = params.C,
		fit_bias   = params.fit_intercept,
		n_samples  = X.rows,
		n_features = n_features,
	}

	// Setup unified optimizer
	opt_config := optim.optimizer_default_config(params.optimizer_type)
	opt_config.learning_rate = params.learning_rate

	opt := optim.optimizer_init(opt_config, n_params, allocator)
	defer optim.optimizer_free(&opt)

	converged := false
	n_iter := 0

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1

		// 1. Compute gradient and loss
		_logistic_gradient(params_vec, &ctx, grad)
		loss := _logistic_loss(params_vec, &ctx)

		// 2. Unified optimizer step (L-BFGS will use loss & obj_fn; others ignore them)
		optim.optimizer_step(&opt, params_vec, grad, loss, _logistic_loss, &ctx)

		// 3. Convergence check: gradient norm squared < tol^2
		grad_norm_sq := l.dot_simd(grad, grad)
		if grad_norm_sq < params.tol * params.tol {
			converged = true
			break
		}
	}

	// Extract final weights and bias
	coef := make([]f64, n_features, allocator)
	copy(coef, params_vec[0:n_features])
	b := params_vec[n_features]

	return LogisticRegression {
		weights = coef,
		bias = b,
		n_iter = n_iter,
		converged = converged,
		allocator = allocator,
	}
}

// ============================================================================
// Public API: Predict Probabilities
// ============================================================================

logistic_predict_proba :: proc(
	model: ^LogisticRegression,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	n_features := len(model.weights)
	if X.cols != n_features do panic("logistic_predict_proba: X has the wrong number of columns")
	probs := make([]f64, n, allocator)

	for i in 0 ..< n {
		x_row := X.data[i * n_features:i * n_features + n_features]
		z := model.bias + l.dot_simd(model.weights, x_row)
		probs[i] = _sigmoid(z)
	}

	return probs
}

// ============================================================================
// Public API: Predict Class Labels
// ============================================================================

logistic_predict :: proc(
	model: ^LogisticRegression,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	n_features := len(model.weights)
	if X.cols != n_features do panic("logistic_predict: X has the wrong number of columns")
	preds := make([]f64, n, allocator)

	for i in 0 ..< n {
		x_row := X.data[i * n_features:i * n_features + n_features]
		z := model.bias + l.dot_simd(model.weights, x_row)
		// Threshold at 0.5
		if _sigmoid(z) >= 0.5 {
			preds[i] = 1.0
		} else {
			preds[i] = 0.0
		}
	}

	return preds
}

// ============================================================================
// Public API: Free Resources
// ============================================================================

logistic_free :: proc(model: ^LogisticRegression) {
	if len(model.weights) > 0 {
		delete(model.weights, model.allocator)
	}
}
