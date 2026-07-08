package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

garch_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GARCH Model Test ===\n")

	main_alloc := context.allocator

	// Generate proper GARCH(1,1) data
	fmt.println("--- Generating Synthetic GARCH(1,1) Returns ---")
	n := 1000 // More data for better estimation
	returns := make([]f64, n, main_alloc)
	defer delete(returns, main_alloc)

	true_omega := 0.00001
	true_alpha := 0.1
	true_beta := 0.85

	cond_var := make([]f64, n, context.temp_allocator)
	defer delete(cond_var, context.temp_allocator)

	// Initialize
	cond_var[0] = true_omega / (1.0 - true_alpha - true_beta)
	returns[0] = 0.0

	// Generate returns with proper Box-Muller
	for i in 1 ..< n {
		cond_var[i] =
			true_omega + true_alpha * returns[i - 1] * returns[i - 1] + true_beta * cond_var[i - 1]
		std_dev := math.sqrt_f64(cond_var[i])

		// Box-Muller transform
		u1 := rand.float64()
		u2 := rand.float64()
		if u1 < 1e-10 {u1 = 1e-10}
		z := math.sqrt_f64(-2.0 * math.ln_f64(u1)) * math.cos_f64(2.0 * math.PI * u2)
		returns[i] = z * std_dev
	}

	fmt.printf("Generated %d returns\n", n)
	fmt.printf(
		"True parameters: ω=%.6f, α=%.4f, β=%.4f, α+β=%.4f\n",
		true_omega,
		true_alpha,
		true_beta,
		true_alpha + true_beta,
	)

	// Extract residuals
	residuals := ts.extract_residuals(returns, main_alloc)
	defer delete(residuals, main_alloc)

	// Fit GARCH(1,1)
	fmt.println("\n--- Fitting GARCH(1,1) Model ---")
	result := ts.garch_fit(residuals, .GARCH, 1, 1, 1000, 1e-6, main_alloc)
	defer {
		delete(result.params.alpha, main_alloc)
		delete(result.params.beta, main_alloc)
		delete(result.conditional_var, main_alloc)
		delete(result.standardized_resid, main_alloc)
	}

	fmt.printf("\n=== GARCH(1,1) Results ===\n")
	fmt.printf("  Omega: %.6f (true: %.6f)\n", result.params.omega, true_omega)
	fmt.printf("  Alpha: %.6f (true: %.6f)\n", result.params.alpha[0], true_alpha)
	fmt.printf("  Beta:  %.6f (true: %.6f)\n", result.params.beta[0], true_beta)
	fmt.printf(
		"  Persistence (α+β): %.4f (true: %.4f)\n",
		result.persistence,
		true_alpha + true_beta,
	)
	fmt.printf("  Log-Likelihood: %.2f\n", result.log_likelihood)
	fmt.printf("  AIC: %.2f\n", result.aic)
	fmt.printf("  BIC: %.2f\n", result.bic)
	fmt.printf("  Converged: %v (iterations: %d)\n", result.converged, result.n_iterations)

	// Forecast
	fmt.println("\n--- Volatility Forecast ---")
	forecast_1 := ts.garch_forecast(&result, residuals, 1, main_alloc)
	forecast_5 := ts.garch_forecast(&result, residuals, 5, main_alloc)
	forecast_10 := ts.garch_forecast(&result, residuals, 10, main_alloc)

	fmt.printf(
		"1-day ahead:  σ²=%.6f, σ=%.6f\n",
		forecast_1.variance_forecast,
		forecast_1.std_forecast,
	)
	fmt.printf(
		"5-day ahead:  σ²=%.6f, σ=%.6f\n",
		forecast_5.variance_forecast,
		forecast_5.std_forecast,
	)
	fmt.printf(
		"10-day ahead: σ²=%.6f, σ=%.6f\n",
		forecast_10.variance_forecast,
		forecast_10.std_forecast,
	)

	fmt.println("\n✓ GARCH test completed!")
}
