package tests

import ts "../wotan/analytics"
import "core:fmt"
import "core:math"
import "core:mem"

garch_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GARCH Model Test ===\n")

	// Use context.allocator for long-lived allocations to avoid temp_allocator panics
	main_alloc := context.allocator

	fmt.println("--- Generating Synthetic Returns ---")
	n := 500
	returns := make([]f64, n, main_alloc)
	defer delete(returns, main_alloc)

	true_omega := 0.00001
	true_alpha := 0.1
	true_beta := 0.85

	cond_var := make([]f64, n, context.temp_allocator)
	defer delete(cond_var, context.temp_allocator)

	cond_var[0] = 0.0001
	for i in 1 ..< n {
		cond_var[i] =
			true_omega + true_alpha * returns[i - 1] * returns[i - 1] + true_beta * cond_var[i - 1]
		std_dev := math.sqrt_f64(cond_var[i])

		// Box-Muller transform for normal distribution
		u1 := f64(i) / f64(n) * 0.5 + 0.25
		u2 := f64(i) / f64(n) * 0.3 + 0.1
		// Avoid log(0)
		if u1 < 1e-10 {u1 = 1e-10}
		z := math.sqrt_f64(-2.0 * math.ln_f64(u1)) * math.cos_f64(2.0 * math.PI * u2)
		returns[i] = z * std_dev
	}

	fmt.printf("Generated %d returns\n", n)

	residuals := ts.extract_residuals(returns, main_alloc)
	defer delete(residuals, main_alloc)

	fmt.println("\n--- Fitting GARCH(1,1) Model ---")
	// Pass main_alloc to garch_fit
	result := ts.garch_fit(residuals, .GARCH, 1, 1, 500, 1e-6, main_alloc)
	defer {
		delete(result.params.alpha, main_alloc)
		delete(result.params.beta, main_alloc)
		delete(result.conditional_var, main_alloc)
		delete(result.standardized_resid, main_alloc)
	}

	fmt.printf("\nGARCH(1,1) Results:\n")
	fmt.printf("  Omega: %.6f (true: %.6f)\n", result.params.omega, true_omega)
	fmt.printf("  Alpha: %.6f (true: %.6f)\n", result.params.alpha[0], true_alpha)
	fmt.printf("  Beta:  %.6f (true: %.6f)\n", result.params.beta[0], true_beta)
	fmt.printf("  Log-Likelihood: %.2f\n", result.log_likelihood)
	fmt.printf("  AIC: %.2f\n", result.aic)
	fmt.printf("  BIC: %.2f\n", result.bic)
	fmt.printf("  Converged: %v\n", result.converged)

	fmt.println("\n--- Volatility Forecast ---")
	forecast_1 := ts.garch_forecast(&result, residuals, 1, main_alloc)
	forecast_5 := ts.garch_forecast(&result, residuals, 5, main_alloc)
	forecast_10 := ts.garch_forecast(&result, residuals, 10, main_alloc)

	fmt.printf("1-day ahead forecast:\n")
	fmt.printf("  Variance: %.6f\n", forecast_1.variance_forecast)
	fmt.printf("  Std Dev:  %.6f\n", forecast_1.std_forecast)
	fmt.printf(
		"  95%% CI:   [%.6f, %.6f]\n",
		forecast_1.confidence_95[0],
		forecast_1.confidence_95[1],
	)

	fmt.printf("\n5-day ahead forecast:\n")
	fmt.printf("  Variance: %.6f\n", forecast_5.variance_forecast)
	fmt.printf("  Std Dev:  %.6f\n", forecast_5.std_forecast)

	fmt.printf("\n10-day ahead forecast:\n")
	fmt.printf("  Variance: %.6f\n", forecast_10.variance_forecast)
	fmt.printf("  Std Dev:  %.6f\n", forecast_10.std_forecast)

	fmt.println("\n✓ GARCH test completed!")
}
