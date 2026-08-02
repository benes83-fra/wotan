package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Deep Hedging Configuration
// ============================================================================

RiskMeasure :: enum {
	Variance, // Minimizes variance of the terminal PnL
	CVaR, // Conditional Value at Risk (Expected Shortfall)
}

DeepHedgerConfig :: struct {
	state_size:   int, // Number of features in the state (e.g., 3: spot, time, vol)
	hidden_size:  int, // Hidden layer size for the MLP
	num_layers:   int, // Number of hidden layers
	risk_measure: RiskMeasure,
	cvar_alpha:   f64, // Alpha for CVaR (e.g., 0.05 for 95% CVaR)
}

DeepHedger :: struct {
	network:   ^nn.Sequential,
	config:    DeepHedgerConfig,
	allocator: mem.Allocator,
}

// ============================================================================
// Initialization
// ============================================================================

deep_hedger_new :: proc(
	config: DeepHedgerConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^DeepHedger {
	hedger := new(DeepHedger, allocator)
	hedger.config = config
	hedger.allocator = allocator

	// Build a simple MLP: input -> hidden -> ... -> 1 (delta)
	seq := nn.sequential_new(allocator)

	// First layer
	nn.sequential_add(seq, nn.linear_layer_new(config.state_size, config.hidden_size))
	nn.sequential_add(seq, nn.Activation.ReLU)

	// Hidden layers
	for _ in 1 ..< config.num_layers {
		nn.sequential_add(seq, nn.linear_layer_new(config.hidden_size, config.hidden_size))
		nn.sequential_add(seq, nn.Activation.ReLU)
	}

	// Output layer (predicts delta, no activation)
	nn.sequential_add(seq, nn.linear_layer_new(config.hidden_size, 1))

	hedger.network = seq
	return hedger
}

deep_hedger_free :: proc(hedger: ^DeepHedger) {
	if hedger.network != nil {
		nn.sequential_free(hedger.network)
	}
	free(hedger, hedger.allocator)
}

// ============================================================================
// Training Step
// ============================================================================

// paths: [batch_size, seq_len, state_size] (e.g., spot, time, vol)
// payoffs: [batch_size, 1] (option payoff at maturity, e.g., max(S_T - K, 0))
// Returns the loss (risk measure of the PnL)
deep_hedger_train_step :: proc(
	hedger: ^DeepHedger,
	paths: ^t.Tensor,
	payoffs: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	batch_size := paths.shape[0]
	seq_len := paths.shape[1]
	state_size := paths.shape[2]
	alloc := hedger.allocator

	// 1. Initialize V_0 (learnable initial hedging price)
	v0_data := l.matrix_new(f64, 1, 1, alloc)
	v0_data.data[0] = 0.0
	v0 := t.tensor_new(v0_data, true, alloc)

	// Note: Ensure V_0 is registered with the optimizer externally!

	// 2. Initialize PnL accumulator (zeros)
	pnl_data := l.matrix_new(f64, batch_size, 1, alloc)
	for i in 0 ..< batch_size {pnl_data.data[i] = 0.0}
	pnl := t.tensor_new(pnl_data, false, alloc)
	pnl.shape = [4]int{batch_size, 1, 1, 1}
	// Broadcast V_0 to batch_size and add to PnL
	v0_broadcast_data := l.matrix_new(f64, batch_size, 1, alloc)
	for i in 0 ..< batch_size {v0_broadcast_data.data[i] = v0.data.data[0]}
	v0_broadcast := t.tensor_new(v0_broadcast_data, false, alloc)
	pnl = t.tensor_add(pnl, v0_broadcast)

	// 3. Pre-allocate reusable buffers for the time loop (prevents memory leaks)
	state_data := l.matrix_new(f64, batch_size, state_size, alloc)
	state := t.tensor_new(state_data, false, alloc)

	s_t_data := l.matrix_new(f64, batch_size, 1, alloc)
	s_t := t.tensor_new(s_t_data, false, alloc)

	s_prev_data := l.matrix_new(f64, batch_size, 1, alloc)
	s_prev := t.tensor_new(s_prev_data, false, alloc)

	prev_delta: ^t.Tensor = nil

	// 4. Loop over time steps
	for t_step in 0 ..< seq_len {
		// Fill state buffer
		for b in 0 ..< batch_size {
			for s in 0 ..< state_size {
				src_idx := b * (seq_len * state_size) + t_step * state_size + s
				dst_idx := b * state_size + s
				state_data.data[dst_idx] = paths.data.data[src_idx]
			}
		}

		// Forward pass through network
		delta := nn.sequential_forward(hedger.network, state)

		// Compute PnL increment if not the first step
		if t_step > 0 {
			// Fill S_t and S_{t-1} buffers (assuming spot is feature index 0)
			for b in 0 ..< batch_size {
				s_t_data.data[b] =
					paths.data.data[b * (seq_len * state_size) + t_step * state_size + 0]
				s_prev_data.data[b] =
					paths.data.data[b * (seq_len * state_size) + (t_step - 1) * state_size + 0]
			}

			ds := t.tensor_sub(s_t, s_prev)
			pnl_increment := t.tensor_mul(prev_delta, ds)
			pnl = t.tensor_add(pnl, pnl_increment)
		}

		prev_delta = delta
	}

	// 5. Subtract the payoff at maturity
	// PnL = V_0 + sum(delta * dS) - payoff
	pnl = t.tensor_sub(pnl, payoffs)

	// 6. Compute Loss (Risk Measure)
	loss: ^t.Tensor

	if hedger.config.risk_measure == .Variance {
		// Variance proxy: Mean(PnL^2)
		pnl_sq := t.tensor_mul(pnl, pnl)
		loss = t.tensor_mean(pnl_sq)
	} else if hedger.config.risk_measure == .CVaR {
		// CVaR (Expected Shortfall) of the loss L = -PnL
		// Uses Rockafellar-Uryasev formulation with empirical VaR threshold
		n := len(pnl.data.data)
		alpha := hedger.config.cvar_alpha // e.g., 0.05 for 95% CVaR

		// 1. Extract PnL data to compute empirical VaR (non-differentiable step)
		losses := make([]f64, n, hedger.allocator)
		defer delete(losses, hedger.allocator)
		for i in 0 ..< n {
			losses[i] = -pnl.data.data[i] // Convert Profit to Loss
		}

		// Sort losses to find VaR threshold
		sorted_losses := make([]f64, n, hedger.allocator)
		defer delete(sorted_losses, hedger.allocator)
		copy(sorted_losses, losses)
		slice.sort(sorted_losses)

		// VaR is the (1-alpha) quantile of the loss distribution
		cutoff_idx := int((1.0 - alpha) * f64(n))
		if cutoff_idx >= n {cutoff_idx = n - 1}
		if cutoff_idx < 0 {cutoff_idx = 0}

		var_val := sorted_losses[cutoff_idx]

		// 2. Compute CVaR using Rockafellar-Uryasev formulation (differentiable step)
		// Formula: CVaR = VaR + 1/alpha * mean( max(L - VaR, 0) )

		// Create a BROADCASTED constant tensor for the VaR threshold
		// Must match pnl's shape: [batch_size, 1, 1, 1]
		batch_size := pnl.shape[0]
		var_data := l.matrix_new(f64, batch_size, 1, hedger.allocator)
		for i in 0 ..< batch_size {
			var_data.data[i] = var_val
		}
		var_tensor := t.tensor_new(var_data, false, hedger.allocator)
		var_tensor.shape = [4]int{batch_size, 1, 1, 1}
		var_tensor.owned_by_graph = true

		// L = -PnL (Differentiable)
		L := t.tensor_neg(pnl)

		// excess = max(L - VaR, 0) (Differentiable via ReLU)
		diff := t.tensor_sub(L, var_tensor)
		excess := t.tensor_relu(diff)

		// mean_excess = mean(excess)
		mean_excess := t.tensor_mean(excess)

		// CVaR = VaR + mean_excess / alpha
		scaled_excess := t.tensor_scale(mean_excess, 1.0 / alpha)

		// For the final loss, we need a scalar. Use the mean of var_tensor.
		var_mean := t.tensor_mean(var_tensor)

		// Add VaR to get final CVaR loss
		loss = t.tensor_add(var_mean, scaled_excess)
	} else {
		// Fallback to Variance
		pnl_sq := t.tensor_mul(pnl, pnl)
		loss = t.tensor_mean(pnl_sq)
	}

	// 7. Backward pass
	t.tensor_backward(loss)

	// 8. Optimizer step
	nn.adam_step(opt)

	// 9. Get loss value
	loss_val := loss.data.data[0]

	// 10. Cleanup
	// Free the computation graph rooted at `loss`.
	t.tensor_free_graph(loss)

	// Manually free the reusable buffers and V_0
	l.matrix_free(&state_data)
	t.tensor_free(state)

	l.matrix_free(&s_t_data)
	t.tensor_free(s_t)

	l.matrix_free(&s_prev_data)
	t.tensor_free(s_prev)

	l.matrix_free(&v0_data)
	t.tensor_free(v0)

	l.matrix_free(&v0_broadcast_data)
	t.tensor_free(v0_broadcast)

	return loss_val
}
// ============================================================================
// Model Persistence (Save/Load)
// ============================================================================

// deep_hedger_save serializes the trained MLP weights to a binary file.
deep_hedger_save :: proc(hedger: ^DeepHedger, path: string) -> bool {
	// nn.save_checkpoint requires an Adam optimizer to satisfy its signature.
	// We create a dummy one just to register the network's parameters for saving.
	dummy_opt := nn.adam_new(0.001, allocator = hedger.allocator)
	defer nn.adam_free(&dummy_opt)

	// Register the network parameters with the dummy optimizer
	nn.sequential_add_to_adam(hedger.network, &dummy_opt)

	return nn.save_checkpoint(hedger.network, &dummy_opt, path, 0, hedger.allocator)
}

// deep_hedger_load deserializes the MLP weights from a binary file into a new DeepHedger.
deep_hedger_load :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^DeepHedger,
	bool,
) {
	// load_checkpoint returns the model, optimizer state, epoch, and success flag
	loaded_model, loaded_opt, _, ok := nn.load_checkpoint(path, allocator)
	if !ok {
		return nil, false
	}

	// Free the loaded optimizer state since we only need the model weights for inference
	if loaded_opt != nil {
		nn.adam_free(loaded_opt)
	}

	// Wrap the loaded Sequential model in a DeepHedger struct
	hedger := new(DeepHedger, allocator)
	hedger.network = loaded_model
	hedger.allocator = allocator

	return hedger, true
}
