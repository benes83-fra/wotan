package ML

import l "../../linalg"
import optim "../../optimize"
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
	hidden_layers:     []int, // e.g., [64, 32]
	activation:        Activation, // Hidden layer activation
	output_activation: Activation, // Output layer activation
	task:              MLPTask,
	learning_rate:     f64,
	max_iter:          int,
	batch_size:        int, // 0 for full-batch
	optimizer_type:    optim.OptimizerType, // .SGD or .Adam
}

MLP :: struct {
	weights:     []l.Matrix(f64), // weights[i] connects layer i to i+1
	biases:      [][]f64, // biases[i] for layer i+1
	activations: []Activation, // activation for each layer (including output)
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
	// Do nothing
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
	n_outputs := 1 // Assuming single output for now (Regression/Binary)

	// 1. Determine layer sizes
	layer_sizes := make([]int, len(params.hidden_layers) + 2, allocator)
	layer_sizes[0] = n_features
	for h, i in params.hidden_layers {layer_sizes[i + 1] = h}
	layer_sizes[len(layer_sizes) - 1] = n_outputs

	n_layers := len(layer_sizes) - 1

	// 2. Initialize Weights and Biases (Xavier/He Initialization)
	weights := make([]l.Matrix(f64), n_layers, allocator)
	biases := make([][]f64, n_layers, allocator)
	activations := make([]Activation, n_layers, allocator)

	for i in 0 ..< n_layers {
		fan_in := layer_sizes[i]
		fan_out := layer_sizes[i + 1]

		weights[i] = l.matrix_new(f64, fan_in, fan_out, allocator)
		biases[i] = make([]f64, fan_out, allocator)

		// He initialization for ReLU, Xavier for others
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

		// Initialize weights
		for r in 0 ..< fan_in {
			for c in 0 ..< fan_out {
				weights[i].data[r * fan_out + c] = rand.float64_normal(0, scale)
			}
		}
		// Initialize biases to 0
		for c in 0 ..< fan_out {biases[i][c] = 0.0}
	}

	// 3. Training Loop (Mini-batch Adam/SGD)
	// For simplicity in this implementation, we use Full-Batch to perfectly integrate
	// with the existing pipeline structure without complex index shuffling.
	// (Mini-batch can be added later by shuffling indices and slicing X/y).

	batch_size := params.batch_size
	if batch_size <= 0 {batch_size = n_samples}

	// Pre-allocate Adam moments if needed
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
		// We use temp_allocator for intermediate activations to guarantee zero heap allocs in loop
		activations_fwd := make([][]f64, n_layers + 1, context.temp_allocator)
		z_vals := make([][]f64, n_layers, context.temp_allocator)

		activations_fwd[0] = X.data // Input layer

		for i in 0 ..< n_layers {
			// Z = A_prev * W + b
			// We do this row by row to avoid allocating a full matrix for Z if we don't have to,
			// but for SIMD matmul, we need matrices.
			A_prev := activations_fwd[i]
			W := weights[i]
			b := biases[i]
			fan_out := layer_sizes[i + 1]

			Z := make([]f64, n_samples * fan_out, context.temp_allocator)
			z_vals[i] = Z

			// Matrix multiply A_prev (n_samples x fan_in) * W (fan_in x fan_out)
			// Since we don't have a matmul that outputs to a pre-allocated slice easily without linalg changes,
			// we will use l.matmul_dyn_simd and copy, OR we can just do it manually for small batches.
			// Let's use l.matmul_dyn_simd for maximum SIMD performance.
			A_prev_mat := l.Matrix(f64) {
				rows = n_samples,
				cols = layer_sizes[i],
				data = A_prev,
			}
			Z_mat := l.matmul_dyn_simd(&A_prev_mat, &W, context.temp_allocator)
			defer l.matrix_free(&Z_mat)

			// Add bias and apply activation
			for r in 0 ..< n_samples {
				for c in 0 ..< fan_out {
					idx := r * fan_out + c
					Z[idx] = Z_mat.data[idx] + b[c]
				}
			}

			A_curr := make([]f64, len(Z), context.temp_allocator)
			copy(A_curr, Z)
			_apply_activation(A_curr, activations[i])
			activations_fwd[i + 1] = A_curr
		}

		// --- Backward Pass ---
		dA := make([]f64, len(activations_fwd[n_layers]), context.temp_allocator)

		// Output layer error
		A_out := activations_fwd[n_layers]
		for i in 0 ..< n_samples {
			if params.task == .Regression || params.output_activation == .Sigmoid {
				dA[i] = A_out[i] - y[i] // Derivative of MSE or BCE
			} else {
				dA[i] = A_out[i] - y[i]
			}
		}

		for i := n_layers - 1; i >= 0; i -= 1 {
			Z_prev := z_vals[i]
			A_prev := activations_fwd[i]
			W := weights[i]
			b := biases[i]
			fan_in := layer_sizes[i]
			fan_out := layer_sizes[i + 1]

			// dZ = dA * activation'(Z)
			dZ := make([]f64, len(Z_prev), context.temp_allocator)
			_apply_activation_derivative(Z_prev, activations[i], dZ)
			for j in 0 ..< len(dZ) {dZ[j] *= dA[j]} 	// Element-wise multiply

			// Gradients for W and b
			// dW = A_prev^T * dZ  (fan_in x n_samples) * (n_samples x fan_out) -> (fan_in x fan_out)
			dW := make([]f64, fan_in * fan_out, context.temp_allocator)
			db := make([]f64, fan_out, context.temp_allocator)

			// Compute dW and db manually for simplicity and zero-alloc
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

			// Update Weights and Biases (Adam/SGD)
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

			// Propagate error to previous layer: dA_prev = dZ * W^T
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

		for r in 0 ..< n_samples {
			for c in 0 ..< fan_out {
				idx := r * fan_out + c
				Z[idx] = Z_mat.data[idx] + model.biases[i][c]
			}
		}

		current_A = make([]f64, len(Z), context.temp_allocator)
		copy(current_A, Z)
		_apply_activation(current_A, model.activations[i])
	}

	// Extract the single output column
	out := make([]f64, n_samples, allocator)
	for i in 0 ..< n_samples {
		out[i] = current_A[i] // Assuming 1 output neuron
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
