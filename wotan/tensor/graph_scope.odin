package tensor

import "base:runtime"
import "core:mem"
import vmem "core:mem/virtual"
import "core:sync"

// ============================================================================
// Graph scopes: one arena per computation graph
// ============================================================================
//
// While a scope is active on a thread, every op output created on that thread
// (data, gradient buffer, saved state, inputs list) and the scratch memory of
// tensor_backward come from the scope's arena. graph_end releases all of it at
// once and keeps the memory for the next graph, so a training loop allocates
// fresh pages only in its first iteration.
//
//     scope: t.Graph_Scope
//     defer t.graph_scope_destroy(&scope)
//     for step in 0 ..< steps {
//         t.graph_begin(&scope)
//         loss := model_loss(...)
//         t.tensor_backward(loss)
//         loss_value := loss.data.data[0] // copy out what must outlive the graph
//         t.graph_end(&scope)             // every node of the graph is gone
//         adam_step(&opt)                 // parameters and their gradients live on
//     }
//
// Rules:
// - Tensors made with tensor_new keep the allocator they are given (parameters,
//   inputs, targets); only op outputs move into the scope.
// - After graph_end, nodes created in the scope must not be used or freed.
//   tensor_free / tensor_free_graph on them inside the scope are allowed and
//   cost nothing (frees are no-ops in the arena).
// - Without an active scope everything behaves exactly as before.
// - Scopes nest: graph_end releases only the innermost scope.

// Arena chunks are reserved and committed in units of at least this size
GRAPH_CHUNK_SIZE :: 64 * mem.Megabyte

@(private = "file")
_Chunk :: struct {
	base:       [^]u8,
	size:       int,
	high_water: int, // bytes ever handed out; memory beyond it is still zero
}

Graph_Scope :: struct {
	chunks:  [dynamic]_Chunk,
	cur:     int, // chunk in use
	offset:  int, // bump offset in chunks[cur]
	last:    rawptr, // most recent allocation, may grow in place
	prev:    ^Graph_Scope,
	mutex:   sync.Mutex,
	active:  bool,
	created: int, // tensors created in the scope since graph_begin
	freed:   int, // ... and explicitly freed (counted again at graph_end)
}

@(private = "file", thread_local)
_active_scope: ^Graph_Scope

// graph_begin makes `scope` the active scope of the calling thread
graph_begin :: proc(scope: ^Graph_Scope, loc := #caller_location) {
	assert(!scope.active, "graph_begin: scope is already active", loc)
	scope.prev = _active_scope
	scope.active = true
	_active_scope = scope
}

// graph_end deactivates `scope` and releases every allocation made in it.
// The arena keeps its memory for the next graph_begin.
graph_end :: proc(scope: ^Graph_Scope, loc := #caller_location) {
	assert(_active_scope == scope, "graph_end: scope is not the innermost active scope", loc)
	_active_scope = scope.prev
	scope.prev = nil
	scope.active = false

	// Tensors released by the reset count as freed in the global counters
	global_tensors_freed += scope.created - scope.freed
	scope.created, scope.freed = 0, 0

	scope.cur, scope.offset, scope.last = 0, 0, nil
}

// graph_scope_destroy returns the scope's memory to the OS
graph_scope_destroy :: proc(scope: ^Graph_Scope, loc := #caller_location) {
	assert(!scope.active, "graph_scope_destroy: scope is still active", loc)
	for c in scope.chunks {vmem.release(c.base, uint(c.size))}
	delete(scope.chunks)
	scope^ = {}
}

// graph_allocator returns the allocator of a scope (valid until graph_end)
graph_allocator :: proc(scope: ^Graph_Scope) -> mem.Allocator {
	return mem.Allocator{procedure = _graph_arena_proc, data = scope}
}

// graph_active_scope returns the innermost active scope of the calling thread, or nil
graph_active_scope :: proc() -> ^Graph_Scope {
	return _active_scope
}

// graph_scope_bytes reports the bytes in use and reserved by a scope
graph_scope_bytes :: proc(scope: ^Graph_Scope) -> (used, reserved: int) {
	for c, i in scope.chunks {
		reserved += c.size
		if i < scope.cur {used += c.high_water} else if i == scope.cur {used += scope.offset}
	}
	return
}

// _op_alloc: allocator for the output of an op whose primary input is `t`
_op_alloc :: #force_inline proc(t: ^Tensor) -> mem.Allocator {
	if _active_scope != nil {return graph_allocator(_active_scope)}
	return t.allocator
}

// _scope_or: the active scope's arena, else `fallback` (op outputs given an
// explicit allocator, and temporary buffers)
_scope_or :: #force_inline proc(fallback: mem.Allocator) -> mem.Allocator {
	if _active_scope != nil {return graph_allocator(_active_scope)}
	return fallback
}

// _scope_of returns the scope owning `allocator`, or nil for other allocators
_scope_of :: #force_inline proc(allocator: mem.Allocator) -> ^Graph_Scope {
	if allocator.procedure == _graph_arena_proc {return (^Graph_Scope)(allocator.data)}
	return nil
}

_scope_note_new :: #force_inline proc(allocator: mem.Allocator) {
	if s := _scope_of(allocator); s != nil {s.created += 1}
}

_scope_note_free :: #force_inline proc(allocator: mem.Allocator) {
	if s := _scope_of(allocator); s != nil {s.freed += 1}
}

// _arena_alloc bumps `size` bytes; zeroes them unless non_zeroed. Holds scope.mutex.
@(private = "file")
_arena_alloc :: proc(
	s: ^Graph_Scope,
	size, alignment: int,
	non_zeroed: bool,
) -> (
	[]byte,
	mem.Allocator_Error,
) {
	align := max(alignment, 16)
	for {
		if s.cur < len(s.chunks) {
			c := &s.chunks[s.cur]
			start := mem.align_forward_int(s.offset, align)
			if start + size <= c.size {
				p := c.base[start:start + size]
				if !non_zeroed {
					// Only memory below the high-water mark can be dirty
					dirty := min(start + size, c.high_water) - start
					if dirty > 0 {mem.zero(raw_data(p), dirty)}
				}
				s.offset = start + size
				c.high_water = max(c.high_water, s.offset)
				s.last = raw_data(p)
				return p, nil
			}
			// Does not fit: move on to the next chunk that is large enough
			if s.cur + 1 < len(s.chunks) && s.chunks[s.cur + 1].size >= size + align {
				s.cur += 1
				s.offset = 0
				continue
			}
		}
		// Need a new chunk right after the current one
		chunk_size := mem.align_forward_int(max(GRAPH_CHUNK_SIZE, size + align), 1 << 16)
		data, err := vmem.reserve_and_commit(uint(chunk_size))
		if err != nil {return nil, .Out_Of_Memory}
		if s.chunks == nil {s.chunks = make([dynamic]_Chunk, runtime.heap_allocator())}
		at := s.cur + 1 if s.cur < len(s.chunks) else len(s.chunks)
		inject_at(&s.chunks, at, _Chunk{base = raw_data(data), size = chunk_size})
		s.cur = at
		s.offset = 0
	}
}

_graph_arena_proc :: proc(
	allocator_data: rawptr,
	mode: mem.Allocator_Mode,
	size, alignment: int,
	old_memory: rawptr,
	old_size: int,
	loc := #caller_location,
) -> (
	[]byte,
	mem.Allocator_Error,
) {
	s := (^Graph_Scope)(allocator_data)
	switch mode {
	case .Alloc, .Alloc_Non_Zeroed:
		sync.mutex_guard(&s.mutex)
		return _arena_alloc(s, size, alignment, mode == .Alloc_Non_Zeroed)

	case .Free:
		return nil, nil // released by graph_end

	case .Free_All:
		return nil, .Mode_Not_Implemented // use graph_end

	case .Resize, .Resize_Non_Zeroed:
		sync.mutex_guard(&s.mutex)
		if old_memory == nil {
			return _arena_alloc(s, size, alignment, mode == .Resize_Non_Zeroed)
		}
		if size <= old_size {
			return ([^]u8)(old_memory)[:size], nil
		}
		// Grow the most recent allocation in place when it still fits
		if old_memory == s.last && s.cur < len(s.chunks) {
			c := &s.chunks[s.cur]
			start := int(uintptr(old_memory) - uintptr(c.base))
			if start >= 0 && start + size <= c.size {
				if mode == .Resize {
					dirty := min(start + size, c.high_water) - (start + old_size)
					if dirty > 0 {mem.zero(rawptr(uintptr(old_memory) + uintptr(old_size)), dirty)}
				}
				s.offset = start + size
				c.high_water = max(c.high_water, s.offset)
				return ([^]u8)(old_memory)[:size], nil
			}
		}
		p, err := _arena_alloc(s, size, alignment, true)
		if err != nil {return nil, err}
		mem.copy(raw_data(p), old_memory, old_size)
		if mode == .Resize {mem.zero(raw_data(p[old_size:]), size - old_size)}
		return p, nil

	case .Query_Features:
		set := (^mem.Allocator_Mode_Set)(old_memory)
		if set != nil {
			set^ = {.Alloc, .Alloc_Non_Zeroed, .Free, .Resize, .Resize_Non_Zeroed, .Query_Features}
		}
		return nil, nil

	case .Query_Info:
		return nil, .Mode_Not_Implemented
	}
	return nil, nil
}
