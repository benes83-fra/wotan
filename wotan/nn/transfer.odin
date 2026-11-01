package nn

import t "../tensor"

// ============================================================================
// Freeze/Unfreeze Utilities
// ============================================================================
// A frozen parameter has requires_grad = false: it gets no gradient and the
// optimizers leave it unchanged. Unfreezing restores its gradient buffer.

tensor_set_requires_grad :: proc(tensor: ^t.Tensor, requires_grad: bool) {
	if tensor == nil {return}
	tensor.requires_grad = requires_grad
	if requires_grad {t.tensor_ensure_grad(tensor)}
}

@(private = "file")
_set_requires_grad :: proc(params: []^t.Tensor, requires_grad: bool) {
	for p in params {tensor_set_requires_grad(p, requires_grad)}
}

// Layers [start, end) of a Sequential model (clamped to the model)
@(private = "file")
_sequential_set :: proc(seq: ^Sequential, start, end: int, requires_grad: bool) {
	ps := make([dynamic]^t.Tensor, context.temp_allocator)
	defer delete(ps)
	n := len(seq.layers)
	for i in clamp(start, 0, n) ..< clamp(end, 0, n) {layer_parameters(&seq.layers[i], &ps, all = true)}
	_set_requires_grad(ps[:], requires_grad)
}

sequential_freeze_all :: proc(seq: ^Sequential) {
	_sequential_set(seq, 0, len(seq.layers), false)
}

sequential_unfreeze_all :: proc(seq: ^Sequential) {
	_sequential_set(seq, 0, len(seq.layers), true)
}

// Freezes layers [start, end)
sequential_freeze_range :: proc(seq: ^Sequential, start: int, end: int) {
	_sequential_set(seq, start, end, false)
}

// GPT model freeze/unfreeze
gpt_freeze_all :: proc(model: ^GPTModel) {
	ps := gpt_model_parameters(model, context.temp_allocator, all = true)
	defer delete(ps)
	_set_requires_grad(ps[:], false)
}

gpt_unfreeze_all :: proc(model: ^GPTModel) {
	ps := gpt_model_parameters(model, context.temp_allocator, all = true)
	defer delete(ps)
	_set_requires_grad(ps[:], true)
}

// Freezes blocks [start, end)
gpt_freeze_blocks :: proc(model: ^GPTModel, start: int, end: int) {
	ps := make([dynamic]^t.Tensor, context.temp_allocator)
	defer delete(ps)
	n := len(model.blocks)
	for i in clamp(start, 0, n) ..< clamp(end, 0, n) {gpt_block_parameters(&model.blocks[i], &ps, all = true)}
	_set_requires_grad(ps[:], false)
}

// BERT model freeze/unfreeze
bert_freeze_all :: proc(model: ^BERTModel) {
	ps := bert_model_parameters(model, context.temp_allocator, all = true)
	defer delete(ps)
	_set_requires_grad(ps[:], false)
}

bert_unfreeze_all :: proc(model: ^BERTModel) {
	ps := bert_model_parameters(model, context.temp_allocator, all = true)
	defer delete(ps)
	_set_requires_grad(ps[:], true)
}

// Freezes encoder blocks [start, end)
bert_freeze_blocks :: proc(model: ^BERTModel, start: int, end: int) {
	ps := make([dynamic]^t.Tensor, context.temp_allocator)
	defer delete(ps)
	n := len(model.encoder_blocks)
	for i in clamp(start, 0, n) ..< clamp(end, 0, n) {bert_block_parameters(&model.encoder_blocks[i], &ps, all = true)}
	_set_requires_grad(ps[:], false)
}
