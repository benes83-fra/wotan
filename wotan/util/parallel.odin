package util

import "base:runtime"
import "core:os"
import "core:sync"
import "core:thread"

// ============================================================================
// Shared worker pool for data-parallel loops
// ============================================================================
// Threads are started once (lazily) and sleep on a semaphore between jobs, so a
// parallel loop only pays for a wake-up. Calls made from inside a parallel loop
// (e.g. a threaded matmul inside a parallel calibration probe) run serially on
// the calling thread instead of oversubscribing or deadlocking the pool.

// parallel_max_threads caps the threads used by parallel_for
// (0 = one per logical processor, 1 = always serial).
parallel_max_threads := 0

// Body of a parallel loop: processes the index range [begin, end).
Parallel_Body :: proc(data: rawptr, begin, end: int)

// parallel_thread_count returns how many threads parallel_for would use.
parallel_thread_count :: proc() -> int {
	n := parallel_max_threads
	if n <= 0 {n = os.get_processor_core_count()}
	return clamp(n, 1, MAX_WORKERS + 1)
}

// parallel_for runs body over [0, n), split into `tasks` contiguous ranges of
// near-equal size (0 = one per thread). The calling thread processes the first
// range itself. Each index is handled by exactly one task, so results do not
// depend on the number of threads as long as body writes disjoint outputs.
parallel_for :: proc(n: int, data: rawptr, body: Parallel_Body, tasks := 0) {
	if n <= 0 {return}
	t := tasks if tasks > 0 else parallel_thread_count()
	t = min(t, n, parallel_thread_count())
	if t <= 1 || _in_parallel_region {
		body(data, 0, n)
		return
	}

	chunk := (n + t - 1) / t
	t = (n + chunk - 1) / chunk // drop empty trailing ranges

	sync.mutex_lock(&_pool.mutex)
	defer sync.mutex_unlock(&_pool.mutex)
	_pool_ensure_workers(t - 1)

	sync.wait_group_add(&_pool.done, t - 1)
	for k in 1 ..< t {
		_pool.jobs[k - 1] = _Pool_Job{body, data, k * chunk, min(n, (k + 1) * chunk)}
		sync.sema_post(&_pool.wake[k - 1])
	}

	// The calling thread takes the first range; mark it as inside a parallel
	// region so nested parallel_for calls run serially instead of deadlocking.
	_in_parallel_region = true
	body(data, 0, min(n, chunk))
	_in_parallel_region = false

	sync.wait_group_wait(&_pool.done)
}

MAX_WORKERS :: 63

@(private = "file")
_Pool_Job :: struct {
	body:       Parallel_Body,
	data:       rawptr,
	begin, end: int,
}

@(private = "file")
_Worker_Pool :: struct {
	mutex:   sync.Mutex, // one parallel dispatch at a time
	done:    sync.Wait_Group,
	wake:    [MAX_WORKERS]sync.Sema,
	jobs:    [MAX_WORKERS]_Pool_Job,
	started: int,
}

@(private = "file")
_pool: _Worker_Pool

@(private = "file", thread_local)
_in_parallel_region: bool

@(private = "file")
_pool_ensure_workers :: proc(n: int) {
	// The workers live for the whole process; allocate them outside any tracking
	// allocator so debug builds do not report them as leaks at exit.
	context.allocator = runtime.heap_allocator()
	for _pool.started < n {
		thread.create_and_start_with_poly_data(_pool.started, proc(slot: int) {
			_in_parallel_region = true // nested parallel_for calls run serially
			for {
				sync.sema_wait(&_pool.wake[slot])
				j := _pool.jobs[slot]
				j.body(j.data, j.begin, j.end)
				sync.wait_group_done(&_pool.done)
			}
		})
		_pool.started += 1
	}
}
