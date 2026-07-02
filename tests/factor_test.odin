package tests

import w "../wotan/core"
import fin "../wotan/finance"
import "core:fmt"
import "core:math/rand"
import "core:mem"

factor_analysis_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Factor Analysis Test ===\n")

	// Generate synthetic data with 3 latent factors
	n_obs := 200
	n_vars := 6

	// Create data with known factor structure
	data := make([][]f64, n_obs, allocator)
	defer {
		for row in data {
			delete(row, allocator)
		}
		delete(data, allocator)
	}

	for obs in 0 ..< n_obs {
		data[obs] = make([]f64, n_vars, allocator)

		// Generate 3 latent factors
		f1 := rand.float64_normal(0, 1)
		f2 := rand.float64_normal(0, 1)
		f3 := rand.float64_normal(0, 1)

		// Observed variables = linear combination of factors + noise
		data[obs][0] = 0.8 * f1 + 0.1 * rand.float64_normal(0, 1)
		data[obs][1] = 0.7 * f1 + 0.2 * rand.float64_normal(0, 1)
		data[obs][2] = 0.6 * f2 + 0.3 * rand.float64_normal(0, 1)
		data[obs][3] = 0.8 * f2 + 0.1 * rand.float64_normal(0, 1)
		data[obs][4] = 0.9 * f3 + 0.1 * rand.float64_normal(0, 1)
		data[obs][5] = 0.7 * f3 + 0.2 * rand.float64_normal(0, 1)
	}

	// Perform factor analysis
	fmt.println("--- Factor Analysis (3 factors) ---")
	fa := fin.factor_analysis(data, 3, 50, 1e-6, allocator)
	defer {
		for row in fa.loadings {
			delete(row, allocator)
		}
		delete(fa.loadings, allocator)
		delete(fa.communalities, allocator)
		delete(fa.uniqueness, allocator)
		delete(fa.eigenvalues, allocator)
	}

	fmt.printf("Converged: %v in %d iterations\n", fa.converged, fa.n_iterations)

	fmt.println("\nFactor Loadings:")
	for f in 0 ..< fa.n_factors {
		fmt.printf("  Factor %d: [", f + 1)
		for v in 0 ..< n_vars {
			if v > 0 {fmt.print(", ")}
			fmt.printf("%.3f", fa.loadings[f][v])
		}
		fmt.println("]")
	}

	fmt.println("\nCommunalities (variance explained by factors):")
	for v in 0 ..< n_vars {
		fmt.printf("  Variable %d: %.3f\n", v + 1, fa.communalities[v])
	}

	fmt.println("\nUniqueness (unique variance):")
	for v in 0 ..< n_vars {
		fmt.printf("  Variable %d: %.3f\n", v + 1, fa.uniqueness[v])
	}

	fmt.println("\nEigenvalues:")
	for f in 0 ..< fa.n_factors {
		fmt.printf("  Factor %d: %.3f\n", f + 1, fa.eigenvalues[f])
	}

	// Varimax rotation
	fmt.println("\n--- Varimax Rotation ---")
	rotated := fin.varimax_rotation(fa.loadings, 100, 1e-6, allocator)
	defer {
		for row in rotated {
			delete(row, allocator)
		}
		delete(rotated, allocator)
	}

	fmt.println("Rotated Loadings:")
	for f in 0 ..< fa.n_factors {
		fmt.printf("  Factor %d: [", f + 1)
		for v in 0 ..< n_vars {
			if v > 0 {fmt.print(", ")}
			fmt.printf("%.3f", rotated[f][v])
		}
		fmt.println("]")
	}

	// Factor scores
	fmt.println("\n--- Factor Scores (first 5 observations) ---")
	scores := fin.compute_factor_scores(data, &fa, allocator)
	defer {
		for row in scores {
			delete(row, allocator)
		}
		delete(scores, allocator)
	}

	for obs in 0 ..< min(5, n_obs) {
		fmt.printf("  Obs %d: [", obs + 1)
		for f in 0 ..< fa.n_factors {
			if f > 0 {fmt.print(", ")}
			fmt.printf("%.3f", scores[obs][f])
		}
		fmt.println("]")
	}

	// Market factor extraction
	fmt.println("\n--- Market Factor Extraction ---")
	market_scores, var_explained := fin.extract_market_factor(data, allocator)
	defer delete(market_scores, allocator)

	fmt.printf("Variance explained by market factor: %.1f%%\n", var_explained * 100)
	fmt.printf("Market factor scores (first 5): [")
	for i in 0 ..< min(5, n_obs) {
		if i > 0 {fmt.print(", ")}
		fmt.printf("%.3f", market_scores[i])
	}
	fmt.println("]")

	// Risk decomposition
	fmt.println("\n--- Risk Decomposition ---")
	decomp := fin.decompose_risk(data, 3, allocator)
	defer {
		delete(decomp.factor_variance, allocator)
		delete(decomp.idiosyncratic_var, allocator)
		delete(decomp.total_variance, allocator)
		for row in decomp.factor_exposure {
			delete(row, allocator)
		}
		delete(decomp.factor_exposure, allocator)
	}

	fmt.println("Variance decomposition per variable:")
	for v in 0 ..< n_vars {
		factor_pct := 0.0
		if decomp.total_variance[v] > 1e-10 {
			factor_pct = decomp.factor_variance[v] / decomp.total_variance[v] * 100
		}
		fmt.printf(
			"  Variable %d: Total=%.3f, Factor=%.3f (%.1f%%), Idio=%.3f\n",
			v + 1,
			decomp.total_variance[v],
			decomp.factor_variance[v],
			factor_pct,
			decomp.idiosyncratic_var[v],
		)
	}

	fmt.println("\n✓ Factor analysis test completed!")
}
