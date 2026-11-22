#+build !windows
package audio_profile

// the peak working set is reported on Windows only
_peak_working_set :: proc() -> uint {
	return 0
}
