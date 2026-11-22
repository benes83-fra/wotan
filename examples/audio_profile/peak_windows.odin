package audio_profile

import win "core:sys/windows"

// the process's peak working set in bytes
_peak_working_set :: proc() -> uint {
	pmc: win.PROCESS_MEMORY_COUNTERS
	if win.GetProcessMemoryInfo(win.GetCurrentProcess(), &pmc, size_of(pmc)) {
		return pmc.PeakWorkingSetSize
	}
	return 0
}
