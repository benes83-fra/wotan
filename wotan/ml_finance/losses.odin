package ml_finance

import t "../tensor"
import "core:mem"

// ============================================================================
// Financial Loss Wrappers
// ============================================================================

// sharpe_loss optimizes the model to maximize risk-adjusted returns.
// returns: Tensor of shape [Batch, Time] or [Batch] containing periodic returns.
sharpe_loss :: proc(
	returns: ^t.Tensor,
	risk_free_rate: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> ^t.Tensor {
	return t.tensor_sharpe_loss(returns, risk_free_rate, allocator)
}