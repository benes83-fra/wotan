package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

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
	} else {
		// Fallback to Variance for CVaR if not implemented
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
