package nn

import t "../tensor"
import "core:mem"

// SeqBackend selects the engine that processes the time dimension.
// .LSTM is the default everywhere, so existing behavior never changes.
SeqBackend :: enum {
	LSTM,
	GRU,
	Mamba,
	TCN, // 3 causal TCN blocks, kernel 3, dilations 1, 2, 4 (receptive field 29 steps)
}

// sequential_add_seq_block appends a sequence block whose OUTPUT feature
// dimension is `hidden`, regardless of backend.
// LSTM/GRU map in_size -> hidden natively.
// Mamba preserves dimension (d_model -> d_model), so we prepend a linear
// adapter in_size -> hidden to keep all downstream shapes identical.
// TCN maps in_size -> hidden in its first block.
sequential_add_seq_block :: proc(
	s: ^Sequential,
	backend: SeqBackend,
	in_size: int,
	hidden: int,
	d_state: int = 16,
	allocator: mem.Allocator = context.allocator,
) {
	switch backend {
	case .LSTM:
		sequential_add(s, lstm_layer_new(in_size, hidden, allocator))
	case .GRU:
		sequential_add(s, gru_layer_new(in_size, hidden, allocator))
	case .Mamba:
		sequential_add(s, linear_layer_new(in_size, hidden, allocator))
		sequential_add(s, mamba_layer_new(hidden, d_state, allocator))
	case .TCN:
		sequential_add_tcn(s, in_size, hidden, allocator = allocator)
	}
}
