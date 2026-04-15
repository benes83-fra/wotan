package main


import w "./wotan/core"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
arima_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== ARIMA TEST ===")

	// Synthetic ARMA(1,1) data
	// y_t = 0.7 y_{t-1} + e_t + 0.4 e_{t-1}
	phi := []f64{0.7}
	theta := []f64{0.4}
	sigma2 := 0.1
	d := 0

	// Generate synthetic data
	T := 200
	y := make([]f64, T, allocator)
	e_prev := 0.0
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64() * math.sqrt_f64(sigma2)
		y[t] = 0.7 * y_prev + e + 0.4 * e_prev
		y_prev = y[t]
		e_prev = e
	}

	// Compute log-likelihood
	ll := w.arima_loglik(y, phi, theta, d, sigma2, allocator)

	fmt.printf("Log-likelihood = %f\n", ll)
	fmt.println("=== END ARIMA TEST ===")
}
arima_fit_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== ARIMA(p,d,q) FIT TEST ===")

	phi_true := []f64{0.7}
	theta_true := []f64{0.4}
	sigma2_true := 0.1
	p, d, q := 1, 0, 1

	T := 400
	y := make([]f64, T, allocator)
	e_prev := 0.0
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, math.sqrt_f64(sigma2_true))
		y[t] = phi_true[0] * y_prev + e + theta_true[0] * e_prev
		y_prev = y[t]
		e_prev = e
	}

	fit := w.arima_fit(y, p, d, q, allocator)

	fmt.printf("True phi=%.3f, theta=%.3f, sigma2=%.3f\n", phi_true[0], theta_true[0], sigma2_true)
	fmt.printf(
		"Fit  phi=%.3f, theta=%.3f, sigma2=%.3f, loglik=%.3f\n",
		fit.phi[0],
		fit.theta[0],
		fit.sigma2,
		fit.loglik,
	)
}
arima_dataframe_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== ARIMA DataFrame Test ===")

	// --- 1. Generate synthetic ARMA(1,1) data ---
	phi_true := 0.7
	theta_true := 0.4
	sigma2_true := 0.1

	T := 300
	y := make([]f64, T, allocator)

	e_prev := 0.0
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, math.sqrt_f64(sigma2_true))
		y[t] = phi_true * y_prev + e + theta_true * e_prev
		y_prev = y[t]
		e_prev = e
	}

	// --- 2. Build Wotan DataFrame ---
	df := w.dataframe_new()
	col := w.column_from_floats("y", y)
	w.add_column(&df, col)
	df.rows = T

	fmt.println("Original data:")
	w.df_head(&df, 10)

	// --- 3. Fit ARIMA(1,0,1) ---
	fit := w.arima_fit(y, 1, 0, 1, allocator)
	fmt.printf(
		"Fitted ARIMA(1,0,1): phi=%.3f theta=%.3f sigma2=%.3f loglik=%.3f\n",
		fit.phi[0],
		fit.theta[0],
		fit.sigma2,
		fit.loglik,
	)

	// --- 4. Forecast 20 steps ahead ---
	h := 20
	fc := w.arima_forecast(y, fit, 1, 0, 1, h, 0.05, allocator)

	// --- 5. Add forecast results to DataFrame ---
	// w.add_column(&df, w.column_from_floats("forecast_mean", fc.mean))
	// w.add_column(&df, w.column_from_floats("forecast_lower", fc.lower))
	// w.add_column(&df, w.column_from_floats("forecast_upper", fc.upper))
	// --- 5. Add forecast results to DataFrame ---
	// We need 300 rows of NULLs followed by 20 rows of forecast
	total_rows := T + h

	// Note: You would likely want to extend the 'y' column to total_rows as well
	// But to just fix the crash, let's create padded columns for the forecast:

	mean_col := w.column_new("forecast_mean", .Float, T + h)
	lower_col := w.column_new("forecast_lower", .Float, T + h)
	upper_col := w.column_new("forecast_upper", .Float, T + h)
	defer w.destroy_column(&mean_col)
	defer w.destroy_column(&lower_col)
	defer w.destroy_column(&upper_col)

	// Fill historical period with NULL
	for i in 0 ..< T {
		w.append_null(&mean_col)
		w.append_null(&lower_col)
		w.append_null(&upper_col)
	}

	// Fill forecast period with actual values
	for i in 0 ..< h {
		w.append_float(&mean_col, fc.mean[i])
		w.append_float(&lower_col, fc.lower[i])
		w.append_float(&upper_col, fc.upper[i])
	}
	fmt.println("Forecast preview:")
	for i in 0 ..< 5 {
		fmt.printf("t+%v: mean=%.3f  [%.3f, %.3f]\n", i + 1, fc.mean[i], fc.lower[i], fc.upper[i])
	}

	fmt.println("=== END ARIMA DataFrame Test ===")

	w.destroy_dataframe(&df)
}
