package core

import "base:runtime"
import "core:mem"
import "core:mem/virtual"

// ============================================================================
// Temporary-allocator guard
// ============================================================================
// TEMP_GUARD() marks the current position of context.temp_allocator and, when the
// enclosing scope ends, releases everything allocated on it after the mark:
//
//	draw_chart :: proc() {
//		w.TEMP_GUARD() // scratch put on context.temp_allocator below is freed on return
//		...
//	}
//
// It works for every temporary allocator that can be rewound:
//   - runtime.Default_Temp_Allocator (Odin's default, or one of your own);
//   - runtime.Arena (runtime.arena_allocator);
//   - mem.Arena (core:mem);
//   - virtual.Arena (core:mem/virtual).
// Any other allocator (a tracking allocator, a scratch or dynamic arena, ...)
// cannot be rewound to a mark: there the guard does nothing and the memory stays
// until free_all(context.temp_allocator). runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD
// handles only Odin's own global default.
// ignore = true makes the guard a no-op, e.g. when a result is being allocated on
// the temporary allocator and must outlive the scope.

Temp_Mark :: struct {
	kind:    enum {
		None,
		Runtime_Arena,
		Mem_Arena,
		Virtual_Arena,
	},
	runtime: runtime.Arena_Temp,
	mem:     mem.Arena_Temp_Memory,
	virtual: virtual.Arena_Temp,
}

// Marks the current position of context.temp_allocator (see TEMP_GUARD)
temp_begin :: proc(ignore := false, loc := #caller_location) -> (m: Temp_Mark) {
	if ignore {return}
	a := context.temp_allocator
	if a.data == nil {return}
	switch a.procedure {
	case runtime.default_temp_allocator_proc:
		when !runtime.NO_DEFAULT_TEMP_ALLOCATOR {
			m.kind = .Runtime_Arena
			m.runtime = runtime.arena_temp_begin(
				&(^runtime.Default_Temp_Allocator)(a.data).arena,
				loc,
			)
		}
	case runtime.arena_allocator_proc:
		m.kind = .Runtime_Arena
		m.runtime = runtime.arena_temp_begin((^runtime.Arena)(a.data), loc)
	case mem.arena_allocator_proc:
		m.kind = .Mem_Arena
		m.mem = mem.begin_arena_temp_memory((^mem.Arena)(a.data))
	case virtual.arena_allocator_proc:
		m.kind = .Virtual_Arena
		m.virtual = virtual.arena_temp_begin((^virtual.Arena)(a.data), loc)
	}
	return
}

// Releases everything allocated after the mark
temp_end :: proc(m: Temp_Mark, loc := #caller_location) {
	switch m.kind {
	case .None:
	case .Runtime_Arena:
		runtime.arena_temp_end(m.runtime, loc)
	case .Mem_Arena:
		mem.end_arena_temp_memory(m.mem)
	case .Virtual_Arena:
		virtual.arena_temp_end(m.virtual, loc)
	}
}

@(deferred_out = _temp_guard_end)
TEMP_GUARD :: #force_inline proc(
	ignore := false,
	loc := #caller_location,
) -> (
	Temp_Mark,
	runtime.Source_Code_Location,
) {
	return temp_begin(ignore, loc), loc
}

@(private = "file")
_temp_guard_end :: proc(m: Temp_Mark, loc: runtime.Source_Code_Location) {
	temp_end(m, loc)
}
