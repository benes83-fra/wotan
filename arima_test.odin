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
mc_arima_arma11 :: proc(allocator: mem.Allocator, n_sims: int, T: int) {
	fmt.printf("=== ARIMA(1,0,1) Monte Carlo: n_sims=%v, T=%v ===\n", n_sims, T)

	// true parameters
	phi_true := 0.7
	theta_true := 0.4
	sigma2_true := 0.1
	sigma_true := math.sqrt_f64(sigma2_true)

	sum_phi := 0.0
	sum_theta := 0.0
	sum_sig2 := 0.0

	sum_phi2 := 0.0
	sum_theta2 := 0.0
	sum_sig22 := 0.0

	n_converged := 0

	for s in 0 ..< n_sims {
		// --- simulate ARMA(1,1) ---
		y := make([]f64, T, allocator)
		e_prev := 0.0
		y_prev := 0.0

		for t in 0 ..< T {
			e := rand.float64_normal(0.0, sigma_true)
			y[t] = phi_true * y_prev + e + theta_true * e_prev
			y_prev = y[t]
			e_prev = e
		}

		fit := w.arima_fit(y, 1, 0, 1, allocator)

		if !fit.converged {
			continue
		}

		n_converged += 1

		phi_hat := fit.phi[0]
		theta_hat := fit.theta[0]
		sig2_hat := fit.sigma2

		sum_phi += phi_hat
		sum_theta += theta_hat
		sum_sig2 += sig2_hat

		sum_phi2 += phi_hat * phi_hat
		sum_theta2 += theta_hat * theta_hat
		sum_sig22 += sig2_hat * sig2_hat
	}

	if n_converged == 0 {
		fmt.println("No converged fits.")
		return
	}

	n := f64(n_converged)

	mean_phi := sum_phi / n
	mean_theta := sum_theta / n
	mean_sig2 := sum_sig2 / n

	var_phi := sum_phi2 / n - mean_phi * mean_phi
	var_theta := sum_theta2 / n - mean_theta * mean_theta
	var_sig2 := sum_sig22 / n - mean_sig2 * mean_sig2

	sd_phi := math.sqrt_f64(max(var_phi, 0.0))
	sd_theta := math.sqrt_f64(max(var_theta, 0.0))
	sd_sig2 := math.sqrt_f64(max(var_sig2, 0.0))

	fmt.printf("Converged: %v / %v\n", n_converged, n_sims)
	fmt.printf("True   phi=%.3f theta=%.3f sigma2=%.3f\n", phi_true, theta_true, sigma2_true)
	fmt.printf("Mean   phi=%.3f (sd=%.3f)\n", mean_phi, sd_phi)
	fmt.printf("Mean theta=%.3f (sd=%.3f)\n", mean_theta, sd_theta)
	fmt.printf("Mean sigma2=%.3f (sd=%.3f)\n", mean_sig2, sd_sig2)
	fmt.printf(
		"Bias   phi=%.3f, theta=%.3f, sigma2=%.3f\n",
		mean_phi - phi_true,
		mean_theta - theta_true,
		mean_sig2 - sigma2_true,
	)
	fmt.println("=== END ARIMA(1,0,1) Monte Carlo ===")
}
arima_fit_arma22_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== ARIMA(p,d,q) FIT TEST ARMA(2,2) ===")

	phi_true := []f64{0.5, -0.2}
	theta_true := []f64{0.4, 0.3}
	sigma2_true := 0.1
	p, d, q := 2, 0, 2

	T := 600
	y := w.arma22_simulate(phi_true, theta_true, sigma2_true, T, allocator)

	fit := w.arima_fit(y, p, d, q, allocator)

	fmt.printf(
		"True phi=[%.3f, %.3f], theta=[%.3f, %.3f], sigma2=%.3f\n",
		phi_true[0],
		phi_true[1],
		theta_true[0],
		theta_true[1],
		sigma2_true,
	)
	fmt.printf(
		"Fit  phi=[%.3f, %.3f], theta=[%.3f, %.3f], sigma2=%.3f, loglik=%.3f\n",
		fit.phi[0],
		fit.phi[1],
		fit.theta[0],
		fit.theta[1],
		fit.sigma2,
		fit.loglik,
	)
	fmt.println("=== END ARMA(2,2) FIT TEST ===")
}


mc_arima_arma22 :: proc(allocator: mem.Allocator, n_sims: int, T: int) {
	fmt.printf("=== ARIMA(2,0,2) Monte Carlo: n_sims=%v, T=%v ===\n", n_sims, T)

	phi_true := []f64{0.5, -0.2}
	theta_true := []f64{0.4, 0.3}
	sigma2_true := 0.1

	sum_phi := make([]f64, 2, allocator)
	sum_theta := make([]f64, 2, allocator)
	sum_sig2 := 0.0

	sum_phi2 := make([]f64, 2, allocator)
	sum_theta2 := make([]f64, 2, allocator)
	sum_sig22 := 0.0

	n_converged := 0

	for s in 0 ..< n_sims {
		y := w.arma22_simulate(phi_true, theta_true, sigma2_true, T, allocator)
		fit := w.arima_fit(y, 2, 0, 2, allocator)
		if !fit.converged {
			continue
		}
		n_converged += 1

		for i in 0 ..< 2 {
			sum_phi[i] += fit.phi[i]
			sum_phi2[i] += fit.phi[i] * fit.phi[i]
			sum_theta[i] += fit.theta[i]
			sum_theta2[i] += fit.theta[i] * fit.theta[i]
		}
		sum_sig2 += fit.sigma2
		sum_sig22 += fit.sigma2 * fit.sigma2
	}

	if n_converged == 0 {
		fmt.println("No converged fits.")
		return
	}

	n := f64(n_converged)

	mean_phi := make([]f64, 2, allocator)
	mean_theta := make([]f64, 2, allocator)
	sd_phi := make([]f64, 2, allocator)
	sd_theta := make([]f64, 2, allocator)

	for i in 0 ..< 2 {
		mean_phi[i] = sum_phi[i] / n
		mean_theta[i] = sum_theta[i] / n
		var_phi := sum_phi2[i] / n - mean_phi[i] * mean_phi[i]
		var_theta := sum_theta2[i] / n - mean_theta[i] * mean_theta[i]
		sd_phi[i] = math.sqrt_f64(max(var_phi, 0.0))
		sd_theta[i] = math.sqrt_f64(max(var_theta, 0.0))
	}

	mean_sig2 := sum_sig2 / n
	var_sig2 := sum_sig22 / n - mean_sig2 * mean_sig2
	sd_sig2 := math.sqrt_f64(max(var_sig2, 0.0))

	fmt.printf("Converged: %v / %v\n", n_converged, n_sims)
	fmt.printf(
		"True   phi=[%.3f, %.3f], theta=[%.3f, %.3f], sigma2=%.3f\n",
		phi_true[0],
		phi_true[1],
		theta_true[0],
		theta_true[1],
		sigma2_true,
	)
	fmt.printf(
		"Mean   phi=[%.3f (sd=%.3f), %.3f (sd=%.3f)]\n",
		mean_phi[0],
		sd_phi[0],
		mean_phi[1],
		sd_phi[1],
	)
	fmt.printf(
		"Mean theta=[%.3f (sd=%.3f), %.3f (sd=%.3f)]\n",
		mean_theta[0],
		sd_theta[0],
		mean_theta[1],
		sd_theta[1],
	)
	fmt.printf("Mean sigma2=%.3f (sd=%.3f)\n", mean_sig2, sd_sig2)
	fmt.printf(
		"Bias   phi=[%.3f, %.3f], theta=[%.3f, %.3f], sigma2=%.3f\n",
		mean_phi[0] - phi_true[0],
		mean_phi[1] - phi_true[1],
		mean_theta[0] - theta_true[0],
		mean_theta[1] - theta_true[1],
		mean_sig2 - sigma2_true,
	)
	fmt.println("=== END ARIMA(2,0,2) Monte Carlo ===")
}
mc_arima_arma_pq :: proc(
	phi_true: []f64,
	theta_true: []f64,
	sigma2_true: f64,
	n_sims: int,
	T: int,
	allocator: mem.Allocator,
) {
	p := len(phi_true)
	q := len(theta_true)

	fmt.printf("=== ARIMA(%v,0,%v) Monte Carlo: n_sims=%v, T=%v ===\n", p, q, n_sims, T)

	sigma_true := math.sqrt_f64(sigma2_true)

	// accumulators
	sum_phi := make([]f64, p, allocator)
	sum_theta := make([]f64, q, allocator)
	sum_sig2 := 0.0

	sum_phi2 := make([]f64, p, allocator)
	sum_theta2 := make([]f64, q, allocator)
	sum_sig22 := 0.0

	n_converged := 0

	for s in 0 ..< n_sims {
		// simulate ARMA(p,q)
		y := w.arma22_simulate(phi_true, theta_true, sigma2_true, T, allocator)

		fit := w.arima_fit(y, p, 0, q, allocator)
		if !fit.converged {
			continue
		}

		n_converged += 1

		// accumulate
		for i in 0 ..< p {
			sum_phi[i] += fit.phi[i]
			sum_phi2[i] += fit.phi[i] * fit.phi[i]
		}
		for j in 0 ..< q {
			sum_theta[j] += fit.theta[j]
			sum_theta2[j] += fit.theta[j] * fit.theta[j]
		}

		sum_sig2 += fit.sigma2
		sum_sig22 += fit.sigma2 * fit.sigma2
	}

	if n_converged == 0 {
		fmt.println("No converged fits.")
		return
	}

	n := f64(n_converged)

	// compute means and SDs
	mean_phi := make([]f64, p, allocator)
	mean_theta := make([]f64, q, allocator)
	sd_phi := make([]f64, p, allocator)
	sd_theta := make([]f64, q, allocator)

	for i in 0 ..< p {
		mean_phi[i] = sum_phi[i] / n
		var_phi := sum_phi2[i] / n - mean_phi[i] * mean_phi[i]
		sd_phi[i] = math.sqrt_f64(max(var_phi, 0.0))
	}
	for j in 0 ..< q {
		mean_theta[j] = sum_theta[j] / n
		var_theta := sum_theta2[j] / n - mean_theta[j] * mean_theta[j]
		sd_theta[j] = math.sqrt_f64(max(var_theta, 0.0))
	}

	mean_sig2 := sum_sig2 / n
	var_sig2 := sum_sig22 / n - mean_sig2 * mean_sig2
	sd_sig2 := math.sqrt_f64(max(var_sig2, 0.0))

	// print results
	fmt.printf("Converged: %v / %v\n", n_converged, n_sims)

	fmt.printf("True phi = [")
	for i in 0 ..< p {
		fmt.printf("%.3f", phi_true[i])
		if i < p - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("True theta = [")
	for j in 0 ..< q {
		fmt.printf("%.3f", theta_true[j])
		if j < q - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("True sigma2 = %.3f\n", sigma2_true)

	fmt.printf("Mean phi = [")
	for i in 0 ..< p {
		fmt.printf("%.3f (sd=%.3f)", mean_phi[i], sd_phi[i])
		if i < p - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Mean theta = [")
	for j in 0 ..< q {
		fmt.printf("%.3f (sd=%.3f)", mean_theta[j], sd_theta[j])
		if j < q - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Mean sigma2 = %.3f (sd=%.3f)\n", mean_sig2, sd_sig2)

	fmt.printf("Bias phi = [")
	for i in 0 ..< p {
		fmt.printf("%.3f", mean_phi[i] - phi_true[i])
		if i < p - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Bias theta = [")
	for j in 0 ..< q {
		fmt.printf("%.3f", mean_theta[j] - theta_true[j])
		if j < q - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Bias sigma2 = %.3f\n", mean_sig2 - sigma2_true)

	fmt.println("=== END ARIMA Monte Carlo ===")
}


mc_arima_pdq :: proc(
	n_sims: int,
	T: int,
	phi_true: []f64,
	d: int,
	theta_true: []f64,
	sigma2_true: f64,
	allocator: mem.Allocator,
) {
	p := len(phi_true)
	q := len(theta_true)

	fmt.printf("=== ARIMA(%v,%v,%v) Monte Carlo: n_sims=%v, T=%v ===\n", p, d, q, n_sims, T)

	// accumulators
	sum_phi := make([]f64, p, allocator)
	sum_theta := make([]f64, q, allocator)
	sum_phi2 := make([]f64, p, allocator)
	sum_theta2 := make([]f64, q, allocator)

	sum_sig2 := 0.0
	sum_sig22 := 0.0

	n_converged := 0

	for s in 0 ..< n_sims {
		// simulate ARIMA(p,d,q)
		y := w.simulate_arima_pdq(phi_true, d, theta_true, sigma2_true, T, allocator)

		fit := w.arima_fit(y, p, d, q, allocator)
		if !fit.converged {
			continue
		}

		n_converged += 1

		for i in 0 ..< p {
			sum_phi[i] += fit.phi[i]
			sum_phi2[i] += fit.phi[i] * fit.phi[i]
		}
		for j in 0 ..< q {
			sum_theta[j] += fit.theta[j]
			sum_theta2[j] += fit.theta[j] * fit.theta[j]
		}

		sum_sig2 += fit.sigma2
		sum_sig22 += fit.sigma2 * fit.sigma2
	}

	if n_converged == 0 {
		fmt.println("No converged fits.")
		return
	}

	n := f64(n_converged)

	// compute means and SDs
	mean_phi := make([]f64, p, allocator)
	mean_theta := make([]f64, q, allocator)
	sd_phi := make([]f64, p, allocator)
	sd_theta := make([]f64, q, allocator)

	for i in 0 ..< p {
		mean_phi[i] = sum_phi[i] / n
		var_phi := sum_phi2[i] / n - mean_phi[i] * mean_phi[i]
		sd_phi[i] = math.sqrt_f64(max(var_phi, 0.0))
	}
	for j in 0 ..< q {
		mean_theta[j] = sum_theta[j] / n
		var_theta := sum_theta2[j] / n - mean_theta[j] * mean_theta[j]
		sd_theta[j] = math.sqrt_f64(max(var_theta, 0.0))
	}

	mean_sig2 := sum_sig2 / n
	var_sig2 := sum_sig22 / n - mean_sig2 * mean_sig2
	sd_sig2 := math.sqrt_f64(max(var_sig2, 0.0))

	// print results
	fmt.printf("Converged: %v / %v\n", n_converged, n_sims)

	fmt.printf("True phi = [")
	for i in 0 ..< p {
		fmt.printf("%.3f", phi_true[i])
		if i < p - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("True theta = [")
	for j in 0 ..< q {
		fmt.printf("%.3f", theta_true[j])
		if j < q - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("True sigma2 = %.3f\n", sigma2_true)

	fmt.printf("Mean phi = [")
	for i in 0 ..< p {
		fmt.printf("%.3f (sd=%.3f)", mean_phi[i], sd_phi[i])
		if i < p - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Mean theta = [")
	for j in 0 ..< q {
		fmt.printf("%.3f (sd=%.3f)", mean_theta[j], sd_theta[j])
		if j < q - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Mean sigma2 = %.3f (sd=%.3f)\n", mean_sig2, sd_sig2)

	fmt.printf("Bias phi = [")
	for i in 0 ..< p {
		fmt.printf("%.3f", mean_phi[i] - phi_true[i])
		if i < p - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Bias theta = [")
	for j in 0 ..< q {
		fmt.printf("%.3f", mean_theta[j] - theta_true[j])
		if j < q - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.printf("Bias sigma2 = %.3f\n", mean_sig2 - sigma2_true)

	fmt.println("=== END ARIMA Monte Carlo ===")
}


mc_arima_pdq_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== ARIMA(p,d,q) Monte Carlo Test ===")

	// Choose a nice example model
	phi_true := []f64{0.5, -0.2} // AR(2)
	theta_true := []f64{0.4, 0.3} // MA(2)
	d := 1 // one differencing
	sigma2_true := 0.1

	// Simulation settings
	n_sims := 200
	T := 300

	// Call the general MC engine
	mc_arima_pdq(n_sims, T, phi_true, d, theta_true, sigma2_true, allocator)

	fmt.println("=== END ARIMA(p,d,q) Monte Carlo Test ===")
}
arima_auto_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== Automatic ARIMA Selection Test ===")

	// simulate ARIMA(2,1,2)
	phi := []f64{0.5, -0.2}
	theta := []f64{0.4, 0.3}
	d := 1
	sigma2 := 0.1
	T := 300

	y := w.simulate_arima_pdq(phi, d, theta, sigma2, T, allocator)

	result := w.arima_auto(y, 3, 2, 3, "aic", allocator)

	fmt.printf("Selected model: ARIMA(%v,%v,%v)\n", result.p, result.d, result.q)
	fmt.printf("AIC = %.3f, BIC = %.3f\n", result.fit.aic, result.fit.bic)

	fmt.println("=== END Automatic ARIMA Selection Test ===")
}


autocorrelation_test :: proc(allocator: mem.Allocator) {
	ddf := w.dataframe_new()
	defer w.destroy_dataframe(&ddf)
	col := w.column_from_floats("y", []f64{1, 2, 3, 4, 5, 4, 3, 2, 1})
	//defer w.destroy_column(&col)
	w.add_column(&ddf, col)
	ddf.rows = 9

	ac := w.df_acf(&ddf, "y", 10, allocator)
	pc := w.df_pacf(&ddf, "y", 10, allocator)
	dac := w.dataframe_new(allocator)
	dpc := w.dataframe_new(allocator)
	defer w.destroy_column(&ac)
	defer w.destroy_column(&pc)
	w.add_column(&dac, ac)
	w.add_column(&dpc, pc)
	w.dataframe_pretty_print(&dac, 10)
	w.dataframe_pretty_print(&dpc, 20)
}
ljung_box_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== LJUNG-BOX TEST ===")

	// ------------------------------------------------------------
	// 1) White noise residuals (should NOT reject)
	// ------------------------------------------------------------
	T := 300
	sigma := 1.0

	wn := make([]f64, T, allocator)
	for i in 0 ..< T {
		wn[i] = rand.float64_normal(0.0, sigma)
	}

	Q1, df1, p1 := w.ljung_box(wn, 10, 0, allocator)
	fmt.printf("White noise test:\n")
	fmt.printf("  Q = %.4f, df = %v, p = %.4f\n", Q1, df1, p1)

	// ------------------------------------------------------------
	// 2) Autocorrelated residuals (AR(1) with phi=0.7)
	// ------------------------------------------------------------
	ac := make([]f64, T, allocator)
	ac_prev := 0.0
	for i in 0 ..< T {
		e := rand.float64_normal(0.0, sigma)
		ac[i] = 0.7 * ac_prev + e
		ac_prev = ac[i]
	}

	Q2, df2, p2 := w.ljung_box(ac, 10, 0, allocator)
	fmt.printf("Autocorrelated test:\n")
	fmt.printf("  Q = %.4f, df = %v, p = %.4f\n", Q2, df2, p2)

	fmt.println("=== END LJUNG-BOX TEST ===")
}
jarque_bera_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== JARQUE-BERA TEST ===")

	T := 500

	// 1) Normal data (should NOT reject)
	normal := make([]f64, T, allocator)
	for i in 0 ..< T {
		normal[i] = rand.float64_normal(0.0, 1.0)
	}
	JB1, p1 := w.jarque_bera(normal, allocator)
	fmt.printf("Normal data: JB=%.4f, p=%.4f\n", JB1, p1)

	// 2) Heavy-tailed data (should reject)
	heavy := make([]f64, T, allocator)
	for i in 0 ..< T {
		// t-distribution with df=3 (very heavy tails)
		heavy[i] = rand.float64_normal(0.0, 1.0) / math.sqrt_f64(rand.float64())
	}
	JB2, p2 := w.jarque_bera(heavy, allocator)
	fmt.printf("Heavy-tailed data: JB=%.4f, p=%.4f\n", JB2, p2)

	fmt.println("=== END JARQUE-BERA TEST ===")
}
residual_diagnostics_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== RESIDUAL DIAGNOSTICS TEST ===")

	// --- 1) Simulate ARMA(1,1) data ---
	phi_true := []f64{0.7}
	theta_true := []f64{0.4}
	sigma2_true := 0.1
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

	// --- 2) Fit ARIMA(1,0,1) ---
	fit := w.arima_fit(y, 1, 0, 1, allocator)
	fmt.printf(
		"Fitted ARIMA(1,0,1): phi=%.3f theta=%.3f sigma2=%.3f\n",
		fit.phi[0],
		fit.theta[0],
		fit.sigma2,
	)

	// --- 3) Compute residuals via Kalman filter ---
	F, Q, P0, H, R, x0, N := w.arima_state_space(fit.phi, fit.theta, 0, fit.sigma2, allocator)
	v, S := w.kalman_filter_residuals(y, F, Q, P0, H, R, x0, N, allocator)
	_ = S // not used here, but available

	// --- 4) Diagnostics ---
	diag := w.df_residual_diagnostics(v, 20, 1 + 1, allocator) // dof_adj = p+q = 2
	fmt.println("Residual diagnostics:")
	w.dataframe_pretty_print(&diag, 20)
	defer w.destroy_dataframe(&diag)
	// --- 5) ACF and PACF ---
	acf_df := w.df_residual_acf(v, 20, allocator)
	defer w.destroy_dataframe(&acf_df)
	pacf_df := w.df_residual_pacf(v, 20, allocator)
	defer w.destroy_dataframe(&pacf_df)
	fmt.println("\nResidual ACF:")
	w.dataframe_pretty_print(&acf_df, 20)

	fmt.println("\nResidual PACF:")
	w.dataframe_pretty_print(&pacf_df, 20)

	fmt.println("=== END RESIDUAL DIAGNOSTICS TEST ===")
}
residuals_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== RESIDUALS TEST ===")

	// Simulate ARMA(1,1)
	phi := []f64{0.7}
	theta := []f64{0.4}
	sigma2 := 0.1
	T := 300

	y := make([]f64, T, allocator)
	e_prev := 0.0
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, math.sqrt_f64(sigma2))
		y[t] = phi[0] * y_prev + e + theta[0] * e_prev
		y_prev = y[t]
		e_prev = e
	}

	// Fit ARIMA(1,0,1)
	fit := w.arima_fit(y, 1, 0, 1, allocator)

	// Extract residuals
	df_res := w.df_residuals(y, fit, 1, 0, 1, allocator)
	defer w.destroy_dataframe(&df_res)
	fmt.println("Residuals head:")
	w.df_head(&df_res, 10)

	// Convert residual column to []f64 without adding new helper functions
	col_res := w.column(&df_res, "residual")
	residuals := make([dynamic]f64, 0, col_res.len, allocator)
	for i in 0 ..< col_res.len {
		v, is_null := w.column_at_float(col_res, i)
		if !is_null {
			append(&residuals, v)
		}
	}

	// Diagnostics
	diag := w.df_residual_diagnostics(
		residuals[:],
		10,
		1 + 1, // p+q
		allocator,
	)
	defer w.destroy_dataframe(&diag)
	fmt.println("Diagnostics:")
	w.dataframe_pretty_print(&diag, 20)

	fmt.println("=== END RESIDUALS TEST ===")
}
adf_test_block :: proc(allocator: mem.Allocator) {
	fmt.println("=== ADF TEST BLOCK ===")

	// Simulate AR(1) with phi = 0.7 (stationary)
	T := 300
	phi := 0.7
	sigma := 1.0

	y := make([]f64, T, allocator)
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, sigma)
		y[t] = phi * y_prev + e
		y_prev = y[t]
	}

	// updated signature: (y, max_lags, reg_type, lag_sel, allocator)
	df := w.df_adf(y, 5, .Constant, .Fixed, allocator)
	w.dataframe_pretty_print(&df, 20)
	defer w.destroy_dataframe(&df)

	fmt.println("=== END ADF TEST BLOCK ===")
}
kpss_test_block :: proc(allocator: mem.Allocator) {
	fmt.println("=== KPSS TEST BLOCK ===")

	T := 300

	// -------------------------------
	// Case A: Stationary AR(1), phi=0.7
	// -------------------------------
	y1 := make([]f64, T, allocator)
	phi := 0.7
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, 1.0)
		y1[t] = phi * y_prev + e
		y_prev = y1[t]
	}

	df1 := w.df_kpss(y1, .Level, -1, allocator)
	fmt.println("Stationary AR(1) KPSS (should NOT reject):")
	w.dataframe_pretty_print(&df1, 20)
	defer w.destroy_dataframe(&df1)

	// -------------------------------
	// Case B: Random Walk (unit root)
	// -------------------------------
	y2 := make([]f64, T, allocator)
	y_prev = 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, 1.0)
		y2[t] = y_prev + e
		y_prev = y2[t]
	}

	df2 := w.df_kpss(y2, .Level, -1, allocator)
	fmt.println("\nRandom Walk KPSS (should REJECT):")
	w.dataframe_pretty_print(&df2, 20)
	defer w.destroy_dataframe(&df2)

	fmt.println("=== END KPSS TEST BLOCK ===")
}

stationarity_test_block :: proc(allocator: mem.Allocator) {
	fmt.println("=== STATIONARITY TEST BLOCK ===")

	T := 300

	// ------------------------------------------------------------
	// Case A: Stationary AR(1), phi = 0.7
	// ------------------------------------------------------------
	y1 := make([]f64, T, allocator)
	phi := 0.7
	y_prev := 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, 1.0)
		y1[t] = phi * y_prev + e
		y_prev = y1[t]
	}

	df1 := w.df_stationarity(y1, 10, .Constant, .AIC, .Level, allocator)
	fmt.println("Stationary AR(1):")
	w.dataframe_pretty_print(&df1, 20)
	defer w.destroy_dataframe(&df1)

	// ------------------------------------------------------------
	// Case B: Random Walk (unit root)
	// ------------------------------------------------------------
	y2 := make([]f64, T, allocator)
	y_prev = 0.0

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, 1.0)
		y2[t] = y_prev + e
		y_prev = y2[t]
	}

	df2 := w.df_stationarity(y2, 10, .Constant, .AIC, .Level, allocator)
	fmt.println("\nRandom Walk:")
	w.dataframe_pretty_print(&df2, 20)
	defer w.destroy_dataframe(&df2)

	// ------------------------------------------------------------
	// Case C: Trend-Stationary (deterministic trend + noise)
	// ------------------------------------------------------------
	y3 := make([]f64, T, allocator)
	for t in 0 ..< T {
		trend := 0.05 * f64(t)
		noise := rand.float64_normal(0.0, 1.0)
		y3[t] = trend + noise
	}

	df3 := w.df_stationarity(y3, 10, .ConstantTrend, .AIC, .Trend, allocator)
	fmt.println("\nTrend-Stationary Series:")
	w.dataframe_pretty_print(&df3, 20)
	defer w.destroy_dataframe(&df3)

	fmt.println("=== END STATIONARITY TEST BLOCK ===")
}
auto_arima_stationarity_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== AUTO-ARIMA WITH STATIONARITY TEST ===")

	T := 400

	// ------------------------------------------------------------
	// Case A: Stationary ARMA(1,1)
	// ------------------------------------------------------------
	phi := []f64{0.7}
	theta := []f64{0.4}
	sigma2 := 0.1

	y1 := make([]f64, T, allocator)
	e_prev := 0.0
	y_prev := 0.0
	for t in 0 ..< T {
		e := rand.float64_normal(0.0, math.sqrt_f64(sigma2))
		y1[t] = phi[0] * y_prev + e + theta[0] * e_prev
		y_prev = y1[t]
		e_prev = e
	}

	d1 := w.auto_arima_d_from_tests(y1, allocator)
	fmt.printf("Case A (Stationary ARMA(1,1)): suggested d = %v\n", d1)

	result1 := w.arima_auto(y1, 3, 2, 3, "aic", allocator)
	fmt.printf(
		"Selected ARIMA(%v,%v,%v), AIC=%.3f\n",
		result1.p,
		result1.d,
		result1.q,
		result1.fit.aic,
	)


	// ------------------------------------------------------------
	// Case B: Random Walk (unit root)
	// ------------------------------------------------------------
	y2 := make([]f64, T, allocator)
	y_prev = 0.0
	for t in 0 ..< T {
		e := rand.float64_normal(0.0, 1.0)
		y2[t] = y_prev + e
		y_prev = y2[t]
	}

	d2 := w.auto_arima_d_from_tests(y2, allocator)
	fmt.printf("\nCase B (Random Walk): suggested d = %v\n", d2)

	result2 := w.arima_auto(y2, 3, 2, 3, "aic", allocator)
	fmt.printf(
		"Selected ARIMA(%v,%v,%v), AIC=%.3f\n",
		result2.p,
		result2.d,
		result2.q,
		result2.fit.aic,
	)


	// ------------------------------------------------------------
	// Case C: Trend-Stationary
	// ------------------------------------------------------------
	y3 := make([]f64, T, allocator)
	for t in 0 ..< T {
		trend := 0.05 * f64(t)
		noise := rand.float64_normal(0.0, 1.0)
		y3[t] = trend + noise
	}

	d3 := w.auto_arima_d_from_tests(y3, allocator)
	fmt.printf("\nCase C (Trend-Stationary): suggested d = %v\n", d3)

	result3 := w.arima_auto(y3, 3, 2, 3, "aic", allocator)
	fmt.printf(
		"Selected ARIMA(%v,%v,%v), AIC=%.3f\n",
		result3.p,
		result3.d,
		result3.q,
		result3.fit.aic,
	)

	fmt.println("=== END AUTO-ARIMA WITH STATIONARITY TEST ===")
}


mc_sarima_forecast :: proc(allocator: mem.Allocator, n_sims: int) {
	fmt.printf("=== SARIMA Monte Carlo forecast test: n_sims=%v ===\n", n_sims)

	// True model: SARIMA(1,1,1)(1,1,1)_12 for example
	phi := []f64{0.5}
	theta := []f64{0.4}
	Phi := []f64{0.3}
	Theta := []f64{0.2}
	d := 1
	D := 1
	s := 12
	sigma2 := 0.1

	T := 200 // history length
	h := 12 // forecast horizon

	sum_mse := 0.0
	sum_cov := 0.0 // coverage count over all horizons

	total_points := n_sims * h

	for sim in 0 ..< n_sims {
		// simulate T + h observations
		y_full := w.simulate_sarima_pdqPDQ(
			phi,
			d,
			theta,
			Phi,
			D,
			Theta,
			s,
			sigma2,
			T + h,
			allocator,
		)

		y_hist := y_full[0:T]
		y_future := y_full[T:T + h]

		fc := w.sarima_forecast(
			y_hist,
			phi,
			theta,
			Phi,
			Theta,
			d,
			D,
			s,
			h,
			sigma2,
			0.05,
			allocator,
		)

		// MSE and interval coverage
		for k in 0 ..< h {
			err := fc.mean[k] - y_future[k]
			sum_mse += err * err

			if y_future[k] >= fc.lower[k] && y_future[k] <= fc.upper[k] {
				sum_cov += 1.0
			}
		}
	}

	mse := sum_mse / f64(total_points)
	coverage := sum_cov / f64(total_points)

	fmt.printf("Forecast MSE (all horizons): %.6f\n", mse)
	fmt.printf("Empirical 95%% PI coverage:  %.3f\n", coverage)
	fmt.println("=== END SARIMA Monte Carlo forecast test ===")
}


sarima_mc_test :: proc(
	n_sims: int,
	T: int,
	phi_true: []f64,
	d: int,
	theta_true: []f64,
	Phi_true: []f64,
	D: int,
	Theta_true: []f64,
	s: int,
	sigma2_true: f64,
	allocator: mem.Allocator,
) {
	p := len(phi_true)
	q := len(theta_true)
	P := len(Phi_true)
	Q := len(Theta_true)

	fmt.printf(
		"=== SARIMA(%v,%v,%v)(%v,%v,%v)_%v Monte Carlo: n_sims=%v, T=%v ===\n",
		p,
		d,
		q,
		P,
		D,
		Q,
		s,
		n_sims,
		T,
	)

	// accumulators
	sum_phi := make([]f64, p, allocator)
	sum_theta := make([]f64, q, allocator)
	sum_Phi := make([]f64, P, allocator)
	sum_Theta := make([]f64, Q, allocator)
	sum_sig2 := 0.0

	sum_phi2 := make([]f64, p, allocator)
	sum_theta2 := make([]f64, q, allocator)
	sum_Phi2 := make([]f64, P, allocator)
	sum_Theta2 := make([]f64, Q, allocator)
	sum_sig22 := 0.0

	n_converged := 0

	for sim in 0 ..< n_sims {
		// --- simulate SARIMA ---
		y := w.simulate_sarima_pdqPDQ(
			phi_true,
			d,
			theta_true,
			Phi_true,
			D,
			Theta_true,
			s,
			sigma2_true,
			T,
			allocator,
		)

		// --- fit SARIMA ---
		fit := w.sarima_fit(y, p, d, q, P, D, Q, s, allocator)
		if !fit.converged {
			continue
		}

		n_converged += 1

		// accumulate
		for i in 0 ..< p {
			sum_phi[i] += fit.phi[i]
			sum_phi2[i] += fit.phi[i] * fit.phi[i]
		}
		for i in 0 ..< q {
			sum_theta[i] += fit.theta[i]
			sum_theta2[i] += fit.theta[i] * fit.theta[i]
		}
		for i in 0 ..< P {
			sum_Phi[i] += fit.Phi[i]
			sum_Phi2[i] += fit.Phi[i] * fit.Phi[i]
		}
		for i in 0 ..< Q {
			sum_Theta[i] += fit.Theta[i]
			sum_Theta2[i] += fit.Theta[i] * fit.Theta[i]
		}

		sum_sig2 += fit.sigma2
		sum_sig22 += fit.sigma2 * fit.sigma2
	}

	if n_converged == 0 {
		fmt.println("No converged fits.")
		return
	}

	n := f64(n_converged)

	// compute means + SDs
	mean_phi := make([]f64, p, allocator)
	mean_theta := make([]f64, q, allocator)
	mean_Phi := make([]f64, P, allocator)
	mean_Theta := make([]f64, Q, allocator)

	sd_phi := make([]f64, p, allocator)
	sd_theta := make([]f64, q, allocator)
	sd_Phi := make([]f64, P, allocator)
	sd_Theta := make([]f64, Q, allocator)

	for i in 0 ..< p {
		mean_phi[i] = sum_phi[i] / n
		var := sum_phi2[i] / n - mean_phi[i] * mean_phi[i]
		sd_phi[i] = math.sqrt_f64(max(var, 0.0))
	}
	for i in 0 ..< q {
		mean_theta[i] = sum_theta[i] / n
		var := sum_theta2[i] / n - mean_theta[i] * mean_theta[i]
		sd_theta[i] = math.sqrt_f64(max(var, 0.0))
	}
	for i in 0 ..< P {
		mean_Phi[i] = sum_Phi[i] / n
		var := sum_Phi2[i] / n - mean_Phi[i] * mean_Phi[i]
		sd_Phi[i] = math.sqrt_f64(max(var, 0.0))
	}
	for i in 0 ..< Q {
		mean_Theta[i] = sum_Theta[i] / n
		var := sum_Theta2[i] / n - mean_Theta[i] * mean_Theta[i]
		sd_Theta[i] = math.sqrt_f64(max(var, 0.0))
	}

	mean_sig2 := sum_sig2 / n
	var_sig2 := sum_sig22 / n - mean_sig2 * mean_sig2
	sd_sig2 := math.sqrt_f64(max(var_sig2, 0.0))

	// --- print results ---
	fmt.printf("Converged: %v / %v\n", n_converged, n_sims)

	fmt.printf("True phi   = %v\n", phi_true)
	fmt.printf("True theta = %v\n", theta_true)
	fmt.printf("True Phi   = %v\n", Phi_true)
	fmt.printf("True Theta = %v\n", Theta_true)
	fmt.printf("True sigma2 = %.4f\n", sigma2_true)

	fmt.printf("Mean phi   = %v\n", mean_phi)
	fmt.printf("Mean theta = %v\n", mean_theta)
	fmt.printf("Mean Phi   = %v\n", mean_Phi)
	fmt.printf("Mean Theta = %v\n", mean_Theta)
	fmt.printf("Mean sigma2 = %.4f\n", mean_sig2)

	fmt.printf("SD phi     = %v\n", sd_phi)
	fmt.printf("SD theta   = %v\n", sd_theta)
	fmt.printf("SD Phi     = %v\n", sd_Phi)
	fmt.printf("SD Theta   = %v\n", sd_Theta)
	fmt.printf("SD sigma2  = %.4f\n", sd_sig2)

	fmt.printf("Bias phi   = %v\n", vec_sub(mean_phi, phi_true, allocator))
	fmt.printf("Bias theta = %v\n", vec_sub(mean_theta, theta_true, allocator))
	fmt.printf("Bias Phi   = %v\n", vec_sub(mean_Phi, Phi_true, allocator))
	fmt.printf("Bias Theta = %v\n", vec_sub(mean_Theta, Theta_true, allocator))
	fmt.printf("Bias sigma2 = %.4f\n", mean_sig2 - sigma2_true)

	fmt.println("=== END SARIMA Monte Carlo ===")
}
vec_sub :: proc(a, b: []f64, allocator: mem.Allocator) -> []f64 {
	out := make([]f64, len(a), allocator)
	for i in 0 ..< len(a) {
		out[i] = a[i] - b[i]
	}
	return out
}
