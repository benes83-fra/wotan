package tests

import "core:fmt"
import "core:hash"
import "core:math/rand"
import "core:os"

// Shared test infrastructure: reproducible seeding and a global count of failed
// correctness checks (used to set the process exit code).

check_failures: int

// seed_test resets the global random generator to a seed derived from the test's
// name. Tests call it as their first statement, seed_test(#procedure), so every
// run is reproducible and independent of which tests ran before.
seed_test :: proc(name: string) {
	rand.reset(hash.fnv64a(transmute([]u8)name))
}

// record_failures adds a suite's failed checks to the global count.
record_failures :: proc(failed: int) {
	check_failures += failed
}

// exit_on_failures exits with code 1 if any correctness check failed.
// main defers it first, so it runs after all other cleanup and the leak report.
exit_on_failures :: proc() {
	if check_failures > 0 {
		fmt.eprintf("\n%d correctness check(s) FAILED\n", check_failures)
		os.exit(1)
	}
}
