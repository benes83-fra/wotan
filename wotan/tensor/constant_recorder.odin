package tensor

import "base:runtime"
import "core:slice"

// ============================================================================
// Constant recorders: freeing the constant intermediates of a forward pass
// ============================================================================
//
// An op whose inputs all lack gradients records no graph: its output keeps no
// inputs, so tensor_free_graph never reaches it. A model whose first layers
// are frozen, run on data, produces such constant intermediates in every
// forward pass. A recording finds them:
//
//     t.constant_record_begin()
//     out := ... // the ops of a forward pass
//     t.constant_record_end(out)
//
// While a recording is active on a thread, every op output made on that
// thread is noted: outputs without gradient as constants, outputs with
// gradient as possible consumers. constant_record_end settles every constant
// except the ones passed to it (the results the caller hands on):
// - one that an op with gradient reads is marked owned_by_graph, so the
//   tensor_free_graph of that graph frees it;
// - any other is freed at once, even if it is marked owned_by_graph: no graph
//   reads it, so no graph would free it.
// Freeing a
// noted tensor (tensor_free, tensor_free_graph) during the recording removes
// it from the recording. Recordings nest: an op output is noted by every
// active recording of the thread, and constant_record_end ends the innermost.
//
// Record only code whose intermediates do not escape: a tensor that the code
// keeps elsewhere (a cache, a struct field) must be passed to
// constant_record_end. Inside a graph scope there is nothing to do: graph_end
// frees everything.

@(private)
_Recording :: struct {
	constants: map[^Tensor]struct{},
	consumers: map[^Tensor]struct{},
}

// The active recordings of the thread, innermost last (heap memory, released
// when the last recording ends)
@(private, thread_local)
_recordings: [dynamic]_Recording

constant_record_begin :: proc() {
	if _recordings == nil {_recordings = make([dynamic]_Recording, runtime.heap_allocator())}
	append(
		&_recordings,
		_Recording {
			constants = make(map[^Tensor]struct{}, runtime.heap_allocator()),
			consumers = make(map[^Tensor]struct{}, runtime.heap_allocator()),
		},
	)
}

constant_record_end :: proc(keep: ..^Tensor, loc := #caller_location) {
	assert(len(_recordings) > 0, "constant_record_end without constant_record_begin", loc)
	rec := pop(&_recordings)
	if len(_recordings) == 0 {
		delete(_recordings)
		_recordings = nil
	}

	reached := make(map[^Tensor]struct{}, runtime.heap_allocator())
	defer delete(reached)
	mark :: proc(n: ^Tensor, constants: map[^Tensor]struct{}, reached: ^map[^Tensor]struct{}) {
		for x in n.inputs {
			if x in constants {reached[x] = {}}
		}
	}
	for n in rec.consumers {mark(n, rec.constants, &reached)}
	// an output that got its gradient flag after creation is a consumer too
	for c in rec.constants {if c.requires_grad {mark(c, rec.constants, &reached)}}

	to_free := make([dynamic]^Tensor, runtime.heap_allocator())
	defer delete(to_free)
	for c in rec.constants {
		if c.requires_grad || slice.contains(keep, c) {continue}
		if c in reached {
			c.owned_by_graph = true
		} else {
			append(&to_free, c)
		}
	}
	delete(rec.constants)
	delete(rec.consumers)
	// tensor_free removes each tensor from the outer recordings as well
	for c in to_free {tensor_free(c)}
}

// Notes a new op output in every active recording
@(private)
_record_new :: proc(t: ^Tensor) {
	for &r in _recordings {
		if t.requires_grad {r.consumers[t] = {}} else {r.constants[t] = {}}
	}
}

// Removes a tensor that is being freed from every active recording
@(private)
_record_forget :: proc(t: ^Tensor) {
	for &r in _recordings {
		delete_key(&r.constants, t)
		delete_key(&r.consumers, t)
	}
}
