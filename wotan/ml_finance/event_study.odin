package ml_finance

import "core:math"
import "core:mem"

// ============================================================================
// Event Study & Cumulative Abnormal Returns (CAR)
// ============================================================================
// Constant-mean-return model (MacKinlay 1997, "Event studies in economics and finance"):
// for an event on row e the normal return is the mean over the estimation window
// rows [e + est_start, e + est_end) (end exclusive), and
//   CAR = sum over the event window rows [e + evt_start, e + evt_end] (inclusive) of r - mean.
// An event is used only if both windows lie inside the data, the estimation window has at
// least 2 rows and every return in them is finite; others are not counted. The test is the
// cross-sectional t = mean(CAR) / (sd(CAR) / sqrt(N)), sd with N - 1 (as
// scipy.stats.ttest_1samp(CAR, 0)); 0 for fewer than 2 events or no dispersion.

EventStudyResult :: struct {
	event_window_start: int,
	event_window_end:   int,
	dates:              []string, // matched event dates (borrowed from event_dates)
	car_values:         []f64, // their CARs, len = num_events
	mean_car:           f64,
	t_statistic:        f64,
	num_events:         int,
}

compute_car :: proc(
	all_dates: []string,
	returns: []f64,
	event_dates: []string,
	est_start: int,
	est_end: int,
	evt_start: int,
	evt_end: int,
	allocator: mem.Allocator,
) -> EventStudyResult {
	assert(len(all_dates) == len(returns), "compute_car: all_dates and returns differ in length")
	num_rows := len(all_dates)
	inside :: proc(lo, hi, n: int) -> bool {return lo >= 0 && hi <= n && lo < hi} 	// [lo, hi)
	finite :: proc(r: []f64) -> bool {
		for v in r {if math.is_nan(v) || math.is_inf(v) {return false}}
		return true
	}

	dates := make([dynamic]string, 0, len(event_dates), context.temp_allocator)
	cars := make([dynamic]f64, 0, len(event_dates), context.temp_allocator)
	defer {delete(dates); delete(cars)}

	for target in event_dates {
		e := -1
		for d, i in all_dates {
			if d == target {e = i; break}
		}
		if e < 0 {continue}

		est_lo, est_hi := e + est_start, e + est_end
		evt_lo, evt_hi := e + evt_start, e + evt_end + 1
		if !inside(est_lo, est_hi, num_rows) ||
		   est_hi - est_lo < 2 ||
		   !inside(evt_lo, evt_hi, num_rows) {
			continue
		}
		est, evt := returns[est_lo:est_hi], returns[evt_lo:evt_hi]
		if !finite(est) || !finite(evt) {continue}

		normal := 0.0
		for r in est {normal += r}
		normal /= f64(len(est))
		car := 0.0
		for r in evt {car += r - normal}
		append(&dates, target)
		append(&cars, car)
	}

	n := len(cars)
	res := EventStudyResult {
		event_window_start = evt_start,
		event_window_end   = evt_end,
		dates              = make([]string, n, allocator),
		car_values         = make([]f64, n, allocator),
		num_events         = n,
	}
	copy(res.dates, dates[:])
	copy(res.car_values, cars[:])
	if n == 0 {return res}

	for c in cars {res.mean_car += c}
	res.mean_car /= f64(n)
	if n > 1 {
		ss := 0.0
		for c in cars {ss += (c - res.mean_car) * (c - res.mean_car)}
		sd := math.sqrt(ss / f64(n - 1))
		if sd > 0 {res.t_statistic = res.mean_car / sd * math.sqrt(f64(n))}
	}
	return res
}

event_study_result_free :: proc(res: ^EventStudyResult, allocator: mem.Allocator) {
	delete(res.dates, allocator)
	delete(res.car_values, allocator)
}
