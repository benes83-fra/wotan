package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


logistic_test :: proc(allocator: mem.Allocator) {
	// Generate linearly separable 2D data (Labels: 0.0 or 1.0)
	n := 200
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X.data[i * 2 + 0] = x0
		X.data[i * 2 + 1] = x1
		// Label: 1.0 if x0 + x1 > 0, else 0.0
		if x0 + x1 > 0 {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
	}

	// Try L-BFGS (It should converge in ~10-20 iterations!)
	params := ml.LogisticParams {
		C              = 1.0,
		max_iter       = 100, // L-BFGS needs very few iterations
		tol            = 1e-5,
		learning_rate  = 1.0, // L-BFGS default step size
		fit_intercept  = true,
		optimizer_type = .LBFGS, // ← Try .SGD or .Adam here too!
	}

	model := ml.logistic_fit(&X, y, params, allocator)
	defer ml.logistic_free(&model)

	fmt.printf(
		"Logistic Regression converged: %v after %v iterations\n",
		model.converged,
		model.n_iter,
	)
	fmt.printf(
		"Weights: [%.3f, %.3f], Bias: %.3f\n",
		model.weights[0],
		model.weights[1],
		model.bias,
	)

	// Predict and compute accuracy
	preds := ml.logistic_predict(&model, &X, allocator)
	defer delete(preds, allocator)

	correct := 0
	for i in 0 ..< n {
		if preds[i] == y[i] {
			correct += 1
		}
	}
	accuracy := f64(correct) / f64(n)
	fmt.printf("Training accuracy: %.2f%%\n", accuracy * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}


real_world_logistic_test :: proc(allocator: mem.Allocator) {
	// Raw nominal data (e.g., "Weather" and "Wind")
	weather_data := []string{"Sunny", "Rainy", "Sunny", "Overcast", "Rainy", "Sunny"}
	wind_data := []string{"Weak", "Strong", "Weak", "Strong", "Weak", "Strong"}

	// Target variable (Nominal)
	play_tennis := []string{"Yes", "No", "Yes", "Yes", "No", "Yes"}

	// 1. Encode Features (One-Hot)
	// drop_first=true is CRITICAL for Logistic Regression to avoid perfect multicollinearity
	weather_enc := ml.ohe_fit(weather_data, true, allocator)
	defer ml.ohe_free(&weather_enc)
	weather_encoded := ml.ohe_transform(&weather_enc, weather_data, allocator)
	defer delete(weather_encoded, allocator)

	wind_enc := ml.ohe_fit(wind_data, true, allocator)
	defer ml.ohe_free(&wind_enc)
	wind_encoded := ml.ohe_transform(&wind_enc, wind_data, allocator)
	defer delete(wind_encoded, allocator)

	// 2. Encode Target (Label Encoding to 0.0 / 1.0)
	target_enc := ml.le_fit(play_tennis, allocator)
	defer ml.le_free(&target_enc)
	y := ml.le_transform(&target_enc, play_tennis, allocator)
	defer delete(y, allocator)

	// 3. Combine encoded features into a single Matrix
	// weather_encoded has 2 cols (3 categories - 1), wind_encoded has 1 col (2 categories - 1)
	// Total features = 3
	n_samples := len(weather_data)
	n_features := (len(weather_enc.categories) - 1) + (len(wind_enc.categories) - 1)

	X := l.matrix_new(f64, n_samples, n_features, allocator)
	defer l.matrix_free(&X)

	for i in 0 ..< n_samples {
		// Copy weather (cols 0 to 1)
		X.data[i * n_features + 0] = weather_encoded[i * 2 + 0]
		X.data[i * n_features + 1] = weather_encoded[i * 2 + 1]
		// Copy wind (col 2)
		X.data[i * n_features + 2] = wind_encoded[i * 1 + 0]
	}

	// 4. Train Logistic Regression
	params := ml.LogisticParams {
		C              = 1.0,
		max_iter       = 100,
		tol            = 1e-5,
		learning_rate  = 1.0,
		fit_intercept  = true,
		optimizer_type = .LBFGS, // L-BFGS will crush this small dataset
	}

	model := ml.logistic_fit(&X, y, params, allocator)
	defer ml.logistic_free(&model)

	fmt.printf("Converged: %v in %v iterations\n", model.converged, model.n_iter)

	// 5. Predict on new raw data
	new_weather := []string{"Overcast"}
	new_wind := []string{"Weak"}

	new_weather_enc := ml.ohe_transform(&weather_enc, new_weather, allocator)
	new_wind_enc := ml.ohe_transform(&wind_enc, new_wind, allocator)

	X_new := l.matrix_new(f64, 1, n_features, allocator)
	X_new.data[0] = new_weather_enc[0]
	X_new.data[1] = new_weather_enc[1]
	X_new.data[2] = new_wind_enc[0]

	pred_proba := ml.logistic_predict_proba(&model, &X_new, allocator)
	fmt.printf("Probability of 'Yes': %.2f%%\n", pred_proba[0] * 100)

	delete(new_weather_enc, allocator)
	delete(new_wind_enc, allocator)
	l.matrix_free(&X_new)
}
