package ML

import l "../../linalg"
import optim "../../optimize"
import "core:fmt"
import "core:math"
import "core:math/rand"
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
	optimizer_type:    optim.OptimizerType,
}

MLP :: struct {
	weights:     []l.Matrix(f64),
	biases:      [][]f64,
	activations: []Activation,
	n_layers:    int,
	allocator:   mem.Allocator,
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
// Public API: Fit MLP
// ============================================================================

mlp_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: MLPParams,
	allocator: mem.Allocator = context.allocator,
) -> MLP {
	n_samples := X.rows
	n_features := X.cols
	n_outputs := 1

	layer_sizes := make([]int, len(params.hidden_layers) + 2, allocator)
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

		tmp: Activation
		if i < n_layers - 1 {
			tmp = params.activation
		} else {
			tmp = params.output_activation
		}
		act := tmp
		activations[i] = act

		scale := 1.0
		if act == .ReLU {
			scale = math.sqrt(2.0 / f64(fan_in))
		} else {
			scale = math.sqrt(1.0 / f64(fan_in))
		}

		for r in 0 ..< fan_in {
			for c in 0 ..< fan_out {
				weights[i].data[r * fan_out + c] = rand.float64_normal(0, scale)
			}
		}
		for c in 0 ..< fan_out {biases[i][c] = 0.0}
	}

	batch_size := params.batch_size
	if batch_size <= 0 {batch_size = n_samples}

	m_w := make([]l.Matrix(f64), n_layers, allocator)
	v_w := make([]l.Matrix(f64), n_layers, allocator)
	m_b := make([][]f64, n_layers, allocator)
	v_b := make([][]f64, n_layers, allocator)

	for i in 0 ..< n_layers {
		m_w[i] = l.matrix_new(f64, layer_sizes[i], layer_sizes[i + 1], allocator)
		v_w[i] = l.matrix_new(f64, layer_sizes[i], layer_sizes[i + 1], allocator)
		m_b[i] = make([]f64, layer_sizes[i + 1], allocator)
		v_b[i] = make([]f64, layer_sizes[i + 1], allocator)
	}

	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-8
	t := 0

	for iter in 0 ..< params.max_iter {
		t += 1
		lr := params.learning_rate

		// --- Forward Pass ---
		activations_fwd := make([][]f64, n_layers + 1, context.temp_allocator)
		z_vals := make([][]f64, n_layers, context.temp_allocator)

		activations_fwd[0] = X.data

		for i in 0 ..< n_layers {
			A_prev := activations_fwd[i]
			W := weights[i]
			b := biases[i]
			fan_out := layer_sizes[i + 1]

			Z := make([]f64, n_samples * fan_out, context.temp_allocator)
			z_vals[i] = Z

			A_prev_mat := l.Matrix(f64) {
				rows = n_samples,
				cols = layer_sizes[i],
				data = A_prev,
			}
			Z_mat := l.matmul_dyn_simd(&A_prev_mat, &W, context.temp_allocator)

			// ✅ FIX 1: Removed 'defer'. Free explicitly to prevent temp_allocator exhaustion!
			for r in 0 ..< n_samples {
				for c in 0 ..< fan_out {
					idx := r * fan_out + c
					Z[idx] = Z_mat.data[idx] + b[c]
				}
			}
			l.matrix_free(&Z_mat) // <--- Explicit free here!

			A_curr := make([]f64, len(Z), context.temp_allocator)
			copy(A_curr, Z)
			_apply_activation(A_curr, activations[i])
			activations_fwd[i + 1] = A_curr
		}

		// --- Backward Pass ---
		dA := make([]f64, len(activations_fwd[n_layers]), context.temp_allocator)

		A_out := activations_fwd[n_layers]
		for i in 0 ..< n_samples {
			dA[i] = A_out[i] - y[i]
		}

		// ✅ DEBUG PRINTS: See exactly what the network is doing every 100 epochs
		if iter % 100 == 0 {
			loss := 0.0
			out_mean := 0.0
			for i in 0 ..< n_samples {
				diff := A_out[i] - y[i]
				loss += diff * diff
				out_mean += A_out[i]
			}
			loss /= f64(n_samples)
			out_mean /= f64(n_samples)
			fmt.printf("[Iter %v] Loss: %.4f | Output Mean: %.4f\n", iter, loss, out_mean)
		}

		for i := n_layers - 1; i >= 0; i -= 1 {
			Z_prev := z_vals[i]
			A_prev := activations_fwd[i]
			W := weights[i]
			b := biases[i]
			fan_in := layer_sizes[i]
			fan_out := layer_sizes[i + 1]

			dZ := make([]f64, len(Z_prev), context.temp_allocator)

			// ✅ FIX 2: BCE Gradient Shortcut.
			// If this is the output layer + Sigmoid + Binary Classification,
			// the derivative of BCE w.r.t Z is simply (A - y).
			// We SKIP the sigmoid derivative to prevent vanishing gradients!
			if i == n_layers - 1 &&
			   params.task == .BinaryClassification &&
			   params.output_activation == .Sigmoid {
				copy(dZ, dA)
			} else {
				_apply_activation_derivative(Z_prev, activations[i], dZ)
				for j in 0 ..< len(dZ) {dZ[j] *= dA[j]}
			}

			dW := make([]f64, fan_in * fan_out, context.temp_allocator)
			db := make([]f64, fan_out, context.temp_allocator)

			for r in 0 ..< fan_in {
				for c in 0 ..< fan_out {
					sum := 0.0
					for s in 0 ..< n_samples {
						sum += A_prev[s * fan_in + r] * dZ[s * fan_out + c]
					}
					dW[r * fan_out + c] = sum / f64(n_samples)
				}
			}
			for c in 0 ..< fan_out {
				sum := 0.0
				for s in 0 ..< n_samples {
					sum += dZ[s * fan_out + c]
				}
				db[c] = sum / f64(n_samples)
			}

			for r in 0 ..< fan_in {
				for c in 0 ..< fan_out {
					idx := r * fan_out + c
					grad := dW[idx]

					if params.optimizer_type == .Adam {
						m_w[i].data[idx] = beta1 * m_w[i].data[idx] + (1.0 - beta1) * grad
						v_w[i].data[idx] = beta2 * v_w[i].data[idx] + (1.0 - beta2) * grad * grad

						m_hat := m_w[i].data[idx] / (1.0 - math.pow(beta1, f64(t)))
						v_hat := v_w[i].data[idx] / (1.0 - math.pow(beta2, f64(t)))

						W.data[idx] -= lr * m_hat / (math.sqrt(v_hat) + epsilon)
					} else {
						W.data[idx] -= lr * grad
					}
				}
			}
			for c in 0 ..< fan_out {
				grad := db[c]
				if params.optimizer_type == .Adam {
					m_b[i][c] = beta1 * m_b[i][c] + (1.0 - beta1) * grad
					v_b[i][c] = beta2 * v_b[i][c] + (1.0 - beta2) * grad * grad

					m_hat := m_b[i][c] / (1.0 - math.pow(beta1, f64(t)))
					v_hat := v_b[i][c] / (1.0 - math.pow(beta2, f64(t)))

					b[c] -= lr * m_hat / (math.sqrt(v_hat) + epsilon)
				} else {
					b[c] -= lr * grad
				}
			}

			if i > 0 {
				dA = make([]f64, len(A_prev), context.temp_allocator)
				for r in 0 ..< fan_in {
					for s in 0 ..< n_samples {
						sum := 0.0
						for c in 0 ..< fan_out {
							sum += dZ[s * fan_out + c] * W.data[r * fan_out + c]
						}
						dA[s * fan_in + r] = sum
					}
				}
			}
		}
	}

	fmt.println("[Training Complete]")

	return MLP {
		weights = weights,
		biases = biases,
		activations = activations,
		n_layers = n_layers,
		allocator = allocator,
	}
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
	current_A := X.data

	for i in 0 ..< model.n_layers {
		fan_in := model.weights[i].rows
		fan_out := model.weights[i].cols

		Z := make([]f64, n_samples * fan_out, context.temp_allocator)

		A_prev_mat := l.Matrix(f64) {
			rows = n_samples,
			cols = fan_in,
			data = current_A,
		}
		Z_mat := l.matmul_dyn_simd(&A_prev_mat, &model.weights[i], context.temp_allocator)
		defer l.matrix_free(&Z_mat)
		// ✅ FIX 1: Removed 'defer'. Free explicitly!
		for r in 0 ..< n_samples {
			for c in 0 ..< fan_out {
				idx := r * fan_out + c
				Z[idx] = Z_mat.data[idx] + model.biases[i][c]
			}
		}
		l.matrix_free(&Z_mat) // <--- Explicit free here!

		current_A = make([]f64, len(Z), context.temp_allocator)
		copy(current_A, Z)
		_apply_activation(current_A, model.activations[i])
	}

	out := make([]f64, n_samples, allocator)
	for i in 0 ..< n_samples {
		out[i] = current_A[i]
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
