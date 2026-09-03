package tests

import w "../wotan/core"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

event_study_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== FinBERT Event Study & CAR Pipeline Test ===")

	symbol := "AAPL"
	fmt.printf("Fetching daily data for %s to establish baseline returns...\n", symbol)

	df := net.read_yahoo(symbol, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Failed to fetch data. Aborting.")
		return
	}

	close_col := &df.columns[5] // AdjClose
	date_col := &df.columns[0] // Date

	all_dates := make([]string, df.rows, allocator)
	returns := make([]f64, df.rows, allocator)
	defer {
		delete(all_dates, allocator)
		delete(returns, allocator)
	}

	for i in 0 ..< df.rows {
		date_str, _ := w.column_at_string(date_col, i)
		all_dates[i] = date_str

		if i > 0 {
			prev_close, _ := w.column_at_float(close_col, i - 1)
			curr_close, _ := w.column_at_float(close_col, i)
			if prev_close > 0.0 {
				returns[i] = (curr_close - prev_close) / prev_close
			}
		}
	}

	fmt.printf("Successfully loaded %d days of data.\n", df.rows)

	fmt.println("\nSimulating NLP Sentiment Analysis on Earnings Dates...")

	// ✅ FIX: Updated to 2025/2026 dates to ensure they fall within the .TwoYears fetch window from Sept 2026
	event_dates := []string {
		"2025-11-03", // Q4 2025
		"2026-02-02", // Q1 2026
		"2026-05-04", // Q2 2026
		"2026-08-03", // Q3 2026
	}

	for date_str in event_dates {
		sentiment := "Positive"
		if date_str == "2026-08-03" {
			sentiment = "Negative" // Mocking a mixed reaction
		}
		fmt.printf("  [%s] Detected Sentiment: %s\n", date_str, sentiment)
	}

	fmt.println("\nRunning Event Study (Cumulative Abnormal Returns)...")
	fmt.println("Estimation Window: -120 to -10 days relative to event")
	fmt.println("Event Window:      -5 to +5 days relative to event")

	car_result := ml_fin.compute_car(
		all_dates,
		returns,
		event_dates,
		-120, // est_start
		-10, // est_end
		-5, // evt_start
		5, // evt_end
		allocator,
	)
	defer ml_fin.event_study_result_free(&car_result, allocator)

	fmt.println("\n=== Event Study Results ===")
	fmt.printf("Events Analyzed: %d\n", len(car_result.dates))

	if len(car_result.dates) > 0 {
		fmt.printf("Mean CAR (-5 to +5 days): %+.4f%%\n", car_result.mean_car * 100.0)
		fmt.printf("T-Statistic:              %.4f\n", car_result.t_statistic)

		fmt.println("\nPer-Event CAR:")
		for i in 0 ..< len(car_result.dates) {
			fmt.printf("  %s : %+.4f%%\n", car_result.dates[i], car_result.car_values[i] * 100.0)
		}

		fmt.println("\n--- Interpretation ---")
		if math.abs(car_result.t_statistic) > 1.96 {
			fmt.println(
				"✓ STATISTICALLY SIGNIFICANT: The NLP-detected events have a measurable impact on price.",
			)
		} else {
			fmt.println(
				"~ NOT SIGNIFICANT: The average CAR is not statistically different from zero (efficient market).",
			)
		}
	} else {
		fmt.println(
			"⚠ WARNING: No events matched the dataset. Check date formats or fetch window.",
		)
	}

	fmt.println("\n✓ Event Study Pipeline Test Complete!")
}
