package ml_finance

import "core:math"
import "core:mem"
import "core:strings" // ✅ Added for robust string comparison

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

	// ✅ FIX: Trim whitespace to prevent parser formatting mismatches
	find_date_idx :: proc(dates: []string, date_str: string) -> int {
		target := strings.trim_space(date_str)
		for d, i in dates {
			if strings.trim_space(d) == target {
				return i
			}
		}
		return -1
	}

	car_values := make([dynamic]f64, allocator)
	valid_dates := make([dynamic]string, allocator)

	num_rows := len(all_dates)

	for event_date in event_dates {
		event_idx := find_date_idx(all_dates, event_date)
		if event_idx < 0 {
			// Optional: uncomment to debug missing dates
			// fmt.printf("Warning: Date %s not found in dataset.\n", event_date)
			continue
		}

		// 1. Calculate Expected Return (Mean of estimation window)
		est_sum := 0.0
		est_count := 0
		for i in event_idx + est_start ..< event_idx + est_end {
			if i >= 0 && i < num_rows {
				est_sum += returns[i]
				est_count += 1
			}
		}

		expected_return := 0.0
		if est_count > 0 {expected_return = est_sum / f64(est_count)}

		// 2. Calculate Abnormal Returns (AR) and Cumulative Abnormal Return (CAR)
		car := 0.0
		for i in event_idx + evt_start ..< event_idx + evt_end + 1 {
			if i >= 0 && i < num_rows {
				abnormal_return := returns[i] - expected_return
				car += abnormal_return
			}
		}

		append(&car_values, car)
		append(&valid_dates, event_date)
	}

	// 3. Aggregate Statistics
	mean_car := 0.0
	n := len(car_values)
	if n > 0 {
		for car in car_values {mean_car += car}
		mean_car /= f64(n)
	}

	variance := 0.0
	if n > 1 {
		for car in car_values {
			diff := car - mean_car
			variance += diff * diff
		}
		variance /= f64(n - 1)
	}

	std_dev := math.sqrt(variance)
	t_stat := 0.0
	if std_dev > 1e-12 && n > 0 {
		t_stat = (mean_car / std_dev) * math.sqrt(f64(n))
	}

	res_dates := make([]string, n, allocator)
	res_cars := make([]f64, n, allocator)
	for i in 0 ..< n {
		res_dates[i] = valid_dates[i]
		res_cars[i] = car_values[i]
	}

	delete(car_values)
	delete(valid_dates)

	return EventStudyResult {
		event_window_start = evt_start,
		event_window_end = evt_end,
		dates = res_dates,
		car_values = res_cars,
		mean_car = mean_car,
		t_statistic = t_stat,
	}
}

event_study_result_free :: proc(res: ^EventStudyResult, allocator: mem.Allocator) {
	delete(res.dates, allocator)
	delete(res.car_values, allocator)
}
