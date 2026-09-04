package ml_finance

import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Event Study & Cumulative Abnormal Returns (CAR)
// ============================================================================

EventStudyResult :: struct {
	event_window_start: int,
	event_window_end:   int,
	dates:              []string,
	car_values:         []f64,
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

	max_events := len(event_dates)
	num_rows := len(all_dates)

	res_dates := make([]string, max_events, allocator)
	res_cars := make([]f64, max_events, allocator)
	found_count := 0

	fmt.printf("  [compute_car] num_rows=%d, max_events=%d\n", num_rows, max_events)


	for ev in 0 ..< max_events {
		target := event_dates[ev]

		event_idx := -1
		for i in 0 ..< num_rows {
			// Explicit length check first, then equality
			if len(all_dates[i]) == len(target) && all_dates[i] == target {
				event_idx = i
				break
			}
		}

		if event_idx < 0 {
			fmt.printf("  [compute_car] Date '%s' NOT found in dataset.\n", target)
			continue
		}

		fmt.printf("  [compute_car] Date '%s' found at index %d.\n", target, event_idx)

		// 1. Expected return from estimation window
		est_sum := 0.0
		est_count := 0
		est_lo := event_idx + est_start
		est_hi := event_idx + est_end
		for i in est_lo ..< est_hi {
			if i >= 0 && i < num_rows {
				est_sum += returns[i]
				est_count += 1
			}
		}

		expected_return := 0.0
		if est_count > 0 {
			expected_return = est_sum / f64(est_count)
		}

		// 2. CAR from event window
		car := 0.0
		evt_lo := event_idx + evt_start
		evt_hi := event_idx + evt_end + 1
		for i in evt_lo ..< evt_hi {
			if i >= 0 && i < num_rows {
				car += returns[i] - expected_return
			}
		}

		res_dates[found_count] = target
		res_cars[found_count] = car
		found_count += 1
	}

	fmt.printf("  [compute_car] Matched %d events.\n", found_count)

	// 3. Aggregate statistics
	mean_car := 0.0
	if found_count > 0 {
		for i in 0 ..< found_count {
			mean_car += res_cars[i]
		}
		mean_car /= f64(found_count)
	}

	variance := 0.0
	if found_count > 1 {
		for i in 0 ..< found_count {
			diff := res_cars[i] - mean_car
			variance += diff * diff
		}
		variance /= f64(found_count - 1)
	}

	std_dev := math.sqrt(variance)
	t_stat := 0.0
	if std_dev > 1e-12 && found_count > 0 {
		t_stat = (mean_car / std_dev) * math.sqrt(f64(found_count))
	}

	return EventStudyResult {
		event_window_start = evt_start,
		event_window_end = evt_end,
		dates = res_dates,
		car_values = res_cars,
		mean_car = mean_car,
		t_statistic = t_stat,
		num_events = found_count,
	}
}

event_study_result_free :: proc(res: ^EventStudyResult, allocator: mem.Allocator) {
	delete(res.dates, allocator)
	delete(res.car_values, allocator)
}
