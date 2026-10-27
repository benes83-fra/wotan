package ML

import l "../../linalg"
import optim "../../optimize"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "base:runtime"
import "core:mem"

// ============================================================================
// MLP Structures
// ============================================================================

Activation :: enum {
	ReLU,
	Sigmoid,
	Tanh,
	Linear,
}

MLPTask :: enum {
	Regression,
	BinaryClassification,
}

MLPParams :: struct {
	hidden_layers:     []int,
	activation:        Activation,
	output_activation: Activation,
	task:              MLPTask,
	learning_rate:     f64,
	max_iter:          int,
	batch_size:        int,
	optimizer_type:      optim.OptimizerType, // .Adam, anything else = plain SGD
	verbose:             bool, // print the epoch loss every 100 iterations
	// Early stopping (sklearn early_stopping=True): hold out validation_fraction of
	// the rows (stratified for BinaryClassification, holdout_split with
	// random_state), score them after every epoch (R² / accuracy), stop when the
	// score has not improved by tol for more than n_iter_no_change epochs and
	// return the weights of the best epoch.
	early_stopping:      bool,
	validation_fraction: f64, // ≤ 0: 0.1
	n_iter_no_change:    int, // ≤ 0: 10
	tol:                 f64, // ≤ 0: 1e-4
	random_state:        int, // seed of the validation split
}

MLP :: struct {
	weights:               []l.Matrix(f64),
	biases:                [][]f64,
	activations:           []Activation,
	n_layers:              int,
	n_iter:                int, // epochs run
	best_validation_score: f64, // early stopping: validation score of the returned weights (else NaN)
	allocator:             mem.Allocator,
}

// ============================================================================
// Internal: Activation Functions & Derivatives
// ============================================================================

_apply_activation :: proc(x: []f64, act: Activation) {
	switch act {
	case .ReLU:
		for i in 0 ..< len(x) {if x[i] < 0.0 {x[i] = 0.0}}
	case .Sigmoid:
		for i in 0 ..< len(x) {x[i] = 1.0 / (1.0 + math.exp(-x[i]))}
	case .Tanh:
		for i in 0 ..< len(x) {x[i] = math.tanh(x[i])}
	case .Linear:
	}
}

_apply_activation_derivative :: proc(z: []f64, act: Activation, out: []f64) {
	switch act {
	case .ReLU:
		for i in 0 ..< len(z) {
			tmp: f64
			if z[i] > 0.0 {
				tmp = 1.0
			} else {
				tmp = 0.0
			}
			out[i] = tmp
		}
	case .Sigmoid:
		for i in 0 ..< len(z) {
			s := 1.0 / (1.0 + math.exp(-z[i]))
			out[i] = s * (1.0 - s)
		}
	case .Tanh:
		for i in 0 ..< len(z) {
			t := math.tanh(z[i])
			out[i] = 1.0 - t * t
		}
	case .Linear:
		for i in 0 ..< len(z) {out[i] = 1.0}
	}
}

// ============================================================================
// Internal: Forward + Backward Pass
// Loss (consistent with the gradients):
//   BinaryClassification with Sigmoid output: mean BCE = mean(softplus(z) - y z)
//   otherwise:                                 mean ½(a - y)²
// Returns the loss on (Xb, yb) and fills dW / db with its exact gradients.
// All scratch memory comes from the temp allocator.
// ============================================================================

_mlp_forward_backward :: proc(
	weights: []l.Matrix(f64),
	biases: [][]f64,
	activations: []Activation,
	task: MLPTask,
	output_activation: Activation,
	Xb: []f64,
	yb: []f64,
	dW: []l.Matrix(f64),
	db: [][]f64,
) -> f64 {
	n_layers := len(weights)
	B := len(yb)

	// --- Forward ---
	A := make([][]f64, n_layers + 1, context.temp_allocator)
	Z := make([][]f64, n_layers, context.temp_allocator)
	A[0] = Xb
	for i in 0 ..< n_layers {
		fan_in, fan_out := weights[i].rows, weights[i].cols
		A_prev := l.Matrix(f64) {
			rows = B,
			cols = fan_in,
			data = A[i],
		}
		Zm := l.matmul_dyn_simd(&A_prev, &weights[i], context.temp_allocator)
		for r in 0 ..< B {
			for c in 0 ..< fan_out {Zm.data[r * fan_out + c] += biases[i][c]}
		}
		Z[i] = Zm.data
		a := make([]f64, len(Zm.data), context.temp_allocator)
		copy(a, Zm.data)
		_apply_activation(a, activations[i])
		A[i + 1] = a
	}

	// --- Loss and output gradient dL/da (without the 1/B, applied below) ---
	bce := task == .BinaryClassification && output_activation == .Sigmoid
	out := A[n_layers]
	z_out := Z[n_layers - 1]
	loss := 0.0
	dA := make([]f64, len(out), context.temp_allocator)
	for s in 0 ..< B {
		if bce {
			z := z_out[s]
			loss += math.max(z, 0.0) + math.ln_f64(1.0 + math.exp(-math.abs(z))) - yb[s] * z
		} else {
			d := out[s] - yb[s]
			loss += 0.5 * d * d
		}
		dA[s] = out[s] - yb[s]
	}
	loss /= f64(B)

	// --- Backward (propagate through the weights *before* they are updated) ---
	for i := n_layers - 1; i >= 0; i -= 1 {
		fan_in, fan_out := weights[i].rows, weights[i].cols
		dZ := make([]f64, len(Z[i]), context.temp_allocator)
		if i == n_layers - 1 && bce {
			copy(dZ, dA) // d BCE / dz = σ(z) - y
		} else {
			_apply_activation_derivative(Z[i], activations[i], dZ)
			for j in 0 ..< len(dZ) {dZ[j] *= dA[j]}
		}

		A_prev := A[i]
		for r in 0 ..< fan_in {
			for c in 0 ..< fan_out {
				sum := 0.0
				for s in 0 ..< B {sum += A_prev[s * fan_in + r] * dZ[s * fan_out + c]}
				dW[i].data[r * fan_out + c] = sum / f64(B)
			}
		}
		for c in 0 ..< fan_out {
			sum := 0.0
			for s in 0 ..< B {sum += dZ[s * fan_out + c]}
			db[i][c] = sum / f64(B)
		}

		if i > 0 {
			dA = make([]f64, len(A_prev), context.temp_allocator)
			W := weights[i]
			for s in 0 ..< B {
				for r in 0 ..< fan_in {
					sum := 0.0
					for c in 0 ..< fan_out {sum += dZ[s * fan_out + c] * W.data[r * fan_out + c]}
					dA[s * fan_in + r] = sum
				}
			}
		}
	}
	return loss
}

// ============================================================================
// Public API: Fit MLP
// One iteration = one epoch over the data in mini-batches of batch_size
// (≤ 0 or ≥ n: full batch), reshuffled every epoch; Adam (bias-corrected,
// β = 0.9 / 0.999, ε = 1e-8, step count per batch) or plain SGD for every
// other optimizer_type. Weights: He-normal for ReLU layers, 1/fan_in otherwise.
// ============================================================================

mlp_fit :: proc(
	X_in: ^l.Matrix(f64),
	y_in: []f64,
	params: MLPParams,
	allocator: mem.Allocator = context.allocator,
) -> MLP {
	if len(y_in) != X_in.rows do panic("mlp_fit: y length mismatch")
	if X_in.rows == 0 do panic("mlp_fit: no samples")

	// Early stopping: train on the training part, score the validation part
	X, y := X_in, y_in
	X_tr, X_val: l.Matrix(f64)
	y_tr, y_val: []f64
	if params.early_stopping {
		frac := params.validation_fraction if params.validation_fraction > 0 else 0.1
		tr, va := holdout_split(X_in.rows, y_in, frac, params.task == .BinaryClassification, params.random_state, context.temp_allocator)
		X_tr = subset_matrix(X_in, tr, allocator)
		y_tr = subset_slice(y_in, tr, allocator)
		X_val = subset_matrix(X_in, va, allocator)
		y_val = subset_slice(y_in, va, allocator)
		delete(tr, context.temp_allocator)
		delete(va, context.temp_allocator)
		X, y = &X_tr, y_tr
	}
	defer if params.early_stopping {
		l.matrix_free(&X_tr)
		l.matrix_free(&X_val)
		delete(y_tr, allocator)
		delete(y_val, allocator)
	}

	n_samples := X.rows
	n_features := X.cols
	n_outputs := 1
	for h in params.hidden_layers {
		if h < 1 do panic("mlp_fit: hidden layer sizes must be >= 1")
	}

	layer_sizes := make([]int, len(params.hidden_layers) + 2, context.temp_allocator)
	defer delete(layer_sizes, context.temp_allocator)
	layer_sizes[0] = n_features
	for h, i in params.hidden_layers {layer_sizes[i + 1] = h}
	layer_sizes[len(layer_sizes) - 1] = n_outputs

	n_layers := len(layer_sizes) - 1

	weights := make([]l.Matrix(f64), n_layers, allocator)
	biases := make([][]f64, n_layers, allocator)
	activations := make([]Activation, n_layers, allocator)

	for i in 0 ..< n_layers {
		fan_in := layer_sizes[i]
		fan_out := layer_sizes[i + 1]

		weights[i] = l.matrix_new(f64, fan_in, fan_out, allocator)
		biases[i] = make([]f64, fan_out, allocator)
		act := params.activation if i < n_layers - 1 else params.output_activation
		activations[i] = act

		scale := math.sqrt(2.0 / f64(fan_in)) if act == .ReLU else math.sqrt(1.0 / f64(fan_in))
		for r in 0 ..< fan_in {
			for c in 0 ..< fan_out {
				weights[i].data[r * fan_out + c] = rand.float64_normal(0, scale)
			}
		}
	}

	batch_size := params.batch_size
	if batch_size <= 0 || batch_size > n_samples {batch_size = n_samples}

	// Gradient buffers and Adam moments
	dW := make([]l.Matrix(f64), n_layers, context.temp_allocator)
	db := make([][]f64, n_layers, context.temp_allocator)
	m_w := make([]l.Matrix(f64), n_layers, context.temp_allocator)
	v_w := make([]l.Matrix(f64), n_layers, context.temp_allocator)
	m_b := make([][]f64, n_layers, context.temp_allocator)
	v_b := make([][]f64, n_layers, context.temp_allocator)
	for i in 0 ..< n_layers {
		dW[i] = l.matrix_new(f64, layer_sizes[i], layer_sizes[i + 1], allocator)
		m_w[i] = l.matrix_new(f64, layer_sizes[i], layer_sizes[i + 1], allocator)
		v_w[i] = l.matrix_new(f64, layer_sizes[i], layer_sizes[i + 1], allocator)
		db[i] = make([]f64, layer_sizes[i + 1], allocator)
		m_b[i] = make([]f64, layer_sizes[i + 1], allocator)
		v_b[i] = make([]f64, layer_sizes[i + 1], allocator)
	}
	defer {
		for i in 0 ..< n_layers {
			l.matrix_free(&dW[i]); l.matrix_free(&m_w[i]); l.matrix_free(&v_w[i])
			delete(db[i], allocator); delete(m_b[i], allocator); delete(v_b[i], allocator)
		}
		delete(dW, context.temp_allocator); delete(db, context.temp_allocator)
		delete(m_w, context.temp_allocator); delete(v_w, context.temp_allocator)
		delete(m_b, context.temp_allocator); delete(v_b, context.temp_allocator)
	}

	perm := make([]int, n_samples, allocator)
	defer delete(perm, allocator)
	for i in 0 ..< n_samples {perm[i] = i}
	Xb := make([]f64, batch_size * n_features, allocator)
	defer delete(Xb, allocator)
	yb := make([]f64, batch_size, allocator)
	defer delete(yb, allocator)

	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-8
	t := 0
	lr := params.learning_rate

	// early-stopping state: best score / weights, epochs without improvement
	n_no_change := params.n_iter_no_change if params.n_iter_no_change > 0 else 10
	es_tol := params.tol if params.tol > 0 else 1e-4
	best_score := -math.INF_F64
	no_improve := 0
	best_w := make([]l.Matrix(f64), n_layers if params.early_stopping else 0, allocator)
	best_b := make([][]f64, n_layers if params.early_stopping else 0, allocator)
	for i in 0 ..< len(best_w) {
		best_w[i] = l.matrix_new(f64, layer_sizes[i], layer_sizes[i + 1], allocator)
		best_b[i] = make([]f64, layer_sizes[i + 1], allocator)
	}
	defer {
		for i in 0 ..< len(best_w) {
			l.matrix_free(&best_w[i])
			delete(best_b[i], allocator)
		}
		delete(best_w, allocator)
		delete(best_b, allocator)
	}
	n_iter := 0

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1
		if batch_size < n_samples do rand.shuffle(perm)
		epoch_loss := 0.0

		for start := 0; start < n_samples; start += batch_size {
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
			bs := min(batch_size, n_samples - start)
			for k in 0 ..< bs {
				src := perm[start + k]
				copy(Xb[k * n_features:(k + 1) * n_features], X.data[src * n_features:(src + 1) * n_features])
				yb[k] = y[src]
			}
			loss := _mlp_forward_backward(
				weights,
				biases,
				activations,
				params.task,
				params.output_activation,
				Xb[:bs * n_features],
				yb[:bs],
				dW,
				db,
			)
			epoch_loss += loss * f64(bs)

			t += 1
			bc1 := 1.0 - math.pow(beta1, f64(t))
			bc2 := 1.0 - math.pow(beta2, f64(t))
			for i in 0 ..< n_layers {
				for idx in 0 ..< len(weights[i].data) {
					g := dW[i].data[idx]
					if params.optimizer_type == .Adam {
						m_w[i].data[idx] = beta1 * m_w[i].data[idx] + (1.0 - beta1) * g
						v_w[i].data[idx] = beta2 * v_w[i].data[idx] + (1.0 - beta2) * g * g
						weights[i].data[idx] -= lr * (m_w[i].data[idx] / bc1) / (math.sqrt(v_w[i].data[idx] / bc2) + epsilon)
					} else {
						weights[i].data[idx] -= lr * g
					}
				}
				for c in 0 ..< len(biases[i]) {
					g := db[i][c]
					if params.optimizer_type == .Adam {
						m_b[i][c] = beta1 * m_b[i][c] + (1.0 - beta1) * g
						v_b[i][c] = beta2 * v_b[i][c] + (1.0 - beta2) * g * g
						biases[i][c] -= lr * (m_b[i][c] / bc1) / (math.sqrt(v_b[i][c] / bc2) + epsilon)
					} else {
						biases[i][c] -= lr * g
					}
				}
			}
		}

		if params.verbose && iter % 100 == 0 {
			fmt.printf("[Iter %v] Loss: %.6f\n", iter, epoch_loss / f64(n_samples))
		}

		if params.early_stopping {
			// sklearn _update_no_improvement_count (early_stopping=True)
			view := MLP {
				weights     = weights,
				biases      = biases,
				activations = activations,
				n_layers    = n_layers,
			}
			score := _mlp_score(&view, &X_val, y_val, params.task)
			if score < best_score + es_tol {no_improve += 1} else {no_improve = 0}
			if score > best_score {
				best_score = score
				for i in 0 ..< n_layers {
					copy(best_w[i].data, weights[i].data)
					copy(best_b[i], biases[i])
				}
			}
			if no_improve > n_no_change do break
		}
	}

	if params.early_stopping {
		for i in 0 ..< n_layers {
			copy(weights[i].data, best_w[i].data)
			copy(biases[i], best_b[i])
		}
	}

	return MLP {
		weights = weights,
		biases = biases,
		activations = activations,
		n_layers = n_layers,
		n_iter = n_iter,
		best_validation_score = best_score if params.early_stopping else math.nan_f64(),
		allocator = allocator,
	}
}

// Validation score: R² (regression) or accuracy of output ≥ 0.5 (binary)
_mlp_score :: proc(model: ^MLP, X: ^l.Matrix(f64), y: []f64, task: MLPTask) -> f64 {
	pred := mlp_predict(model, X, context.temp_allocator)
	defer delete(pred, context.temp_allocator)
	if task == .Regression do return metrics_r2(y, pred)
	for &p in pred {p = 1 if p >= 0.5 else 0}
	return metrics_accuracy(y, pred)
}

// ============================================================================
// Public API: Predict
// ============================================================================

mlp_predict :: proc(
	model: ^MLP,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_samples := X.rows
	if model.n_layers > 0 && X.cols != model.weights[0].rows do panic("mlp_predict: X has the wrong number of columns")
	// Result first: `allocator` may be the temp allocator the guard below rewinds
	out := make([]f64, n_samples, allocator)
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		current_A := X.data

		for i in 0 ..< model.n_layers {
			fan_in := model.weights[i].rows
			fan_out := model.weights[i].cols

			A_prev_mat := l.Matrix(f64) {
				rows = n_samples,
				cols = fan_in,
				data = current_A,
			}
			Z_mat := l.matmul_dyn_simd(&A_prev_mat, &model.weights[i], context.temp_allocator)
			for r in 0 ..< n_samples {
				for c in 0 ..< fan_out {
					Z_mat.data[r * fan_out + c] += model.biases[i][c]
				}
			}
			_apply_activation(Z_mat.data, model.activations[i])
			current_A = Z_mat.data
		}
		copy(out, current_A[:n_samples])
	}
	return out
}

// ============================================================================
// Public API: Free
// ============================================================================

mlp_free :: proc(model: ^MLP) {
	for i in 0 ..< model.n_layers {
		l.matrix_free(&model.weights[i])
		delete(model.biases[i], model.allocator)
	}
	delete(model.weights, model.allocator)
	delete(model.biases, model.allocator)
	delete(model.activations, model.allocator)
}
