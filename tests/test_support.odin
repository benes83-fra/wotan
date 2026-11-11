package tests

import "core:fmt"
import "core:hash"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:mem/virtual"
import "core:os"
import "core:slice"

// Shared test infrastructure: reproducible seeding, runs over several seeds and
// a global count of failed correctness checks (used to set the process exit code).

check_failures: int

// seed_test resets the global random generator to a seed derived from the test's
// name. Tests call it as their first statement, seed_test(#procedure), so every
// run is reproducible and independent of which tests ran before.
seed_test :: proc(name: string) {
	rand.reset(_test_seed(name))
}

@(private = "file")
_test_seed :: proc(name: string) -> u64 {
	return hash.fnv64a(transmute([]u8)name)
}

// record_failures adds a suite's failed checks to the global count.
record_failures :: proc(failed: int) {
	check_failures += failed
}

// fail_missing_data reports that a test could not get the real data it needs
// (a loader returned ok = false) and counts it as one failed check. Real data is
// never replaced by synthetic data: the test returns after this call and the
// rest of the suite still runs.
fail_missing_data :: proc(test: string, what: string) {
	fmt.printf("FAIL %s: %s unavailable\n", test, what)
	record_failures(1)
}

// exit_on_failures exits with code 1 if any correctness check failed.
// main defers it first, so it runs after all other cleanup and the leak report.
exit_on_failures :: proc() {
	if check_failures > 0 {
		fmt.eprintf("\n%d correctness check(s) FAILED\n", check_failures)
		os.exit(1)
	}
}

// One metric of a seed_sweep: its value per seed and their summary (std is
// the sample standard deviation, sem = std / √n the standard error of the mean).
Seed_Stats :: struct {
	name:                     string,
	values:                   []f64,
	mean, std, sem, min, max: f64,
}

// seed_sweep runs `run` with n_seeds seeds and summarizes the metrics it returns
// (one value per name in `metrics`), to tell a real change of a training result
// from seed noise. Run i starts from seed_test's seed for `name` plus i, so run
// 0 is the regular test run. Each run allocates from its own arena (also as
// the temp allocator), released after its metrics are copied.
seed_sweep :: proc(
	name: string,
	n_seeds: int,
	metrics: []string,
	run: proc(allocator: mem.Allocator) -> []f64,
	allocator := context.allocator,
) -> []Seed_Stats {
	stats := make([]Seed_Stats, len(metrics), allocator)
	for &s, m in stats {s = {
			name   = metrics[m],
			values = make([]f64, n_seeds, allocator),
		}}
	for i in 0 ..< n_seeds {
		arena: virtual.Arena
		if err := virtual.arena_init_growing(&arena); err != nil {
			panic("seed_sweep: no memory for the run's arena")
		}
		{
			context.allocator = virtual.arena_allocator(&arena)
			context.temp_allocator = context.allocator
			rand.reset(_test_seed(name) + u64(i))
			values := run(context.allocator)
			assert(
				len(values) == len(metrics),
				"seed_sweep: run returned another number of metrics",
			)
			for &s, m in stats {s.values[i] = values[m]}
		}
		virtual.arena_destroy(&arena)
	}
	for &s in stats {
		n := f64(n_seeds)
		s.mean = math.sum(s.values) / n
		ss := 0.0
		for v in s.values {ss += (v - s.mean) * (v - s.mean)}
		s.std = math.sqrt(ss / (n - 1)) if n_seeds > 1 else 0
		s.sem = s.std / math.sqrt(n)
		s.min, s.max = slice.min(s.values), slice.max(s.values)
	}
	return stats
}

// Prints mean ± std, the standard error, the range and the values per seed
seed_sweep_print :: proc(stats: []Seed_Stats) {
	for s in stats {
		fmt.printf(
			"   %-24s %.4f ± %.4f (sem %.4f, range %.4f .. %.4f) per seed %.4f\n",
			s.name,
			s.mean,
			s.std,
			s.sem,
			s.min,
			s.max,
			s.values,
		)
	}
}
