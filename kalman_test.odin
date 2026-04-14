package main

import w "./wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

kalman_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== KALMAN FILTER TEST ===")

	x0: [2]f64 = {0.0, 1.0}

	// Matrices are FLAT and COLUMN-MAJOR.
	// For P0 (Identity), it's {col1_row1, col1_row2, col2_row1, col2_row2}
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	// F = {{1, 1}, {0, 1}} -> Column-major: {1.0, 0.0, 1.0, 1.0}
	F := matrix[2, 2]f64{
		1.0, 0.0,
		1.0, 1.0,
	}

	// H = {{1, 0}} (1 row, 2 cols) -> {1.0, 0.0}
	H := matrix[1, 2]f64{
		1.0, 0.0,
	}

	Q := matrix[2, 2]f64{
		0.01, 0.0,
		0.0, 0.01,
	}
	R := matrix[1, 1]f64{
		1.0,
	}

	// REMOVED "2, 1". The compiler infers N and M from the arguments.
	kf := w.kalman_init(x0, P0, F, H, Q, R)

	measurements := [6]f64{1.2, 2.1, 2.9, 4.2, 5.1, 6.05}

	for i in 0 ..< len(measurements) {
		fmt.printf("\n--- Step %d ---\n", i)

		z: [1]f64 = {measurements[i]}

		fmt.println("Predicting...")
		// REMOVED "2, 1"
		w.kalman_predict(&kf)
		fmt.println("Predicted state:", kf.x)

		fmt.println("Updating with measurement:", z)
		// REMOVED "2, 1"
		w.kalman_update(&kf, z)
		fmt.println("Updated state:", kf.x)
	}


	T := len(measurements)

	xf := make([]w.KalmanState(2), T, allocator)
	xp := make([]w.KalmanState(2), T, allocator)

	// forward pass
	for t in 0 ..< T {
		xp[t].x = kf.x
		xp[t].P = kf.P

		w.kalman_predict(&kf)

		z: [1]f64 = {measurements[t]}
		w.kalman_update(&kf, z)

		xf[t].x = kf.x
		xf[t].P = kf.P
	}


	// smoothing
	smoothed := w.rts_smooth(F, xf[:], xp[:])
	fmt.println("\n--- FILTERED vs SMOOTHED ---")
	for i in 0 ..< T {
		fmt.printf(
			"t=%d  filtered=[%f, %f]  smoothed=[%f, %f]\n",
			i,
			xf[i].x[0],
			xf[i].x[1],
			smoothed[i].x[0],
			smoothed[i].x[1],
		)
	}
	defer delete(smoothed)


	fmt.println("\n=== END KALMAN FILTER TEST ===")
}


kalman_control_test :: proc(allocator: mem.Allocator) {


	fmt.println("=== KALMAN CONTROL FILTER TEST ===")

	x0: [2]f64 = {0.0, 1.0}

	// Matrices are FLAT and COLUMN-MAJOR.
	// For P0 (Identity), it's {col1_row1, col1_row2, col2_row1, col2_row2}
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	// F = {{1, 1}, {0, 1}} -> Column-major: {1.0, 0.0, 1.0, 1.0}
	F := matrix[2, 2]f64{
		1.0, 0.0,
		1.0, 1.0,
	}

	// H = {{1, 0}} (1 row, 2 cols) -> {1.0, 0.0}
	H := matrix[1, 2]f64{
		1.0, 0.0,
	}

	Q := matrix[2, 2]f64{
		0.01, 0.0,
		0.0, 0.01,
	}
	R := matrix[1, 1]f64{
		1.0,
	}

	// REMOVED "2, 1". The compiler infers N and M from the arguments.
	kf := w.kalman_init(x0, P0, F, H, Q, R)

	measurements := [6]f64{1.2, 2.1, 2.9, 4.2, 5.1, 6.05}


	// Control matrix B (N=2, U=1)


	T := len(measurements)
	B := matrix[2, 1]f64{
		0.5,
		1.0,
	}
	u := make([]([1]f64), T, allocator)
	for i in 0 ..< T {
		u[i] = [1]f64{1.0}
	}

	xf := make([]w.KalmanState(2), T, allocator)
	xp := make([]w.KalmanState(2), T, allocator)

	for i in 0 ..< len(measurements) {
		fmt.printf("\n--- Step %d ---\n", i)

		z: [1]f64 = {measurements[i]}

		fmt.println("Predicting...")
		// REMOVED "2, 1"
		w.kalman_predict(&kf)
		fmt.println("Predicted state:", kf.x)

		fmt.println("Updating with measurement:", z)
		// REMOVED "2, 1"
		w.kalman_update(&kf, z)
		fmt.println("Updated state:", kf.x)
	}
	// forward pass
	for t in 0 ..< T {
		xp[t].x = kf.x
		xp[t].P = kf.P

		w.kalman_predict_control(&kf, B, u[t])


		z: [1]f64 = {measurements[t]}
		w.kalman_update(&kf, z)

		xf[t].x = kf.x
		xf[t].P = kf.P
	}


	// smoothing
	smoothed := w.rts_smooth_control(F, B, u[:], xf[:], xp[:])
	fmt.println("\n--- FILTERED vs SMOOTHED ---")
	for i in 0 ..< T {
		fmt.printf(
			"t=%d  filtered=[%f, %f]  smoothed=[%f, %f]\n",
			i,
			xf[i].x[0],
			xf[i].x[1],
			smoothed[i].x[0],
			smoothed[i].x[1],
		)
	}
	defer delete(smoothed)


	fmt.println("\n=== END Control KALMAN FILTER TEST ===")
}
kalman_tv_control_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== KALMAN TV CONTROL FILTER + SMOOTHER TEST ===")

	// State: [position, velocity]
	x0: [2]f64 = {0.0, 0.0}
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	T := 6

	// Time-varying F (here: constant, but as a sequence)
	F_seq := make([]matrix[2, 2]f64, T, allocator)
	for i in 0 ..< T {
		F_seq[i] = matrix[2, 2]f64{
			1.0, 0.0,
			1.0, 1.0,
		}
	}

	// Time-varying H (measure position only)
	H_seq := make([]matrix[1, 2]f64, T, allocator)
	for i in 0 ..< T {
		H_seq[i] = matrix[1, 2]f64{
			1.0, 0.0,
		}
	}

	// Time-varying B (control affects acceleration -> velocity & position)
	B_seq := make([]matrix[2, 1]f64, T, allocator)
	for i in 0 ..< T {
		B_seq[i] = matrix[2, 1]f64{
			0.5,
			1.0,
		}
	}

	// Time-varying Q, R (kept constant but as sequences)
	Q_seq := make([]matrix[2, 2]f64, T, allocator)
	R_seq := make([]matrix[1, 1]f64, T, allocator)
	for i in 0 ..< T {
		Q_seq[i] = matrix[2, 2]f64{
			0.01, 0.0,
			0.0, 0.01,
		}
		R_seq[i] = matrix[1, 1]f64{
			1.0,
		}
	}

	// Controls: constant acceleration u_t = 1.0
	u_seq := make([]([1]f64), T, allocator)
	for i in 0 ..< T {
		u_seq[i] = [1]f64{1.0}
	}

	// Measurements: noisy positions along roughly quadratic path
	z_seq := make([]([1]f64), T, allocator)
	measurements := [6]f64{0.2, 1.1, 2.9, 5.8, 9.9, 15.2}
	for i in 0 ..< T {
		z_seq[i] = [1]f64{measurements[i]}
	}

	xf, xp := w.kalman_forward_tv_control(
		x0,
		P0,
		F_seq,
		H_seq,
		B_seq,
		Q_seq,
		R_seq,
		u_seq,
		z_seq,
		allocator,
	)

	smoothed := w.rts_smooth_tv_control(F_seq, B_seq, u_seq, xf, xp)

	fmt.println("\n--- FILTERED vs SMOOTHED (TV + CONTROL) ---")
	for t in 0 ..< T {
		fmt.printf(
			"t=%d  z=%f  filtered=[%f, %f]  smoothed=[%f, %f]\n",
			t,
			measurements[t],
			xf[t].x[0],
			xf[t].x[1],
			smoothed[t].x[0],
			smoothed[t].x[1],
		)
	}


	defer delete(smoothed)

	fmt.println("\n=== END KALMAN TV CONTROL FILTER + SMOOTHER TEST ===")
}


ekf_tiny_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== EKF TINY NONLINEAR TEST ===")

	// Initial state
	x: [2]f64 = {0.1, 1.0} // pos=0.1, vel=1.0
	P := matrix[2, 2]f64{
		0.1, 0.0,
		0.0, 0.1,
	}

	Q := matrix[2, 2]f64{
		0.001, 0.0,
		0.0, 0.001,
	}

	R := matrix[1, 1]f64{
		0.05,
	}

	// Nonlinear state transition
	f := proc(x: [2]f64) -> [2]f64 {
		pos := x[0]
		vel := x[1]
		return [2]f64{pos + vel * vel, vel}
	}

	// Jacobian of f
	F_jac := proc(x: [2]f64) -> matrix[2, 2]f64 {
		vel := x[1]
		return matrix[2, 2]f64{
			1.0, 0.0,
			2.0 * vel, 1.0,
		}
	}

	// Nonlinear measurement
	h := proc(x: [2]f64) -> [1]f64 {
		return [1]f64{math.sin_f64(x[0])}
	}

	// Jacobian of h
	H_jac := proc(x: [2]f64) -> matrix[1, 2]f64 {
		return matrix[1, 2]f64{
			math.cos(x[0]), 0.0,
		}
	}

	// Fake measurements: sin_f64()(true_position)
	z_seq := [5]f64 {
		math.sin_f64(0.1),
		math.sin_f64(1.1),
		math.sin_f64(2.1),
		math.sin_f64(3.1),
		math.sin_f64(4.1),
	}

	for t in 0 ..< len(z_seq) {
		fmt.printf("\n--- Step %d ---\n", t)

		// Predict
		x_pred, P_pred := w.ekf_predict(x, P, f, F_jac, Q)

		fmt.println("Predicted x:", x_pred)

		// Update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := w.ekf_update(x_pred, P_pred, z, h, H_jac, R)

		fmt.println("Updated x:", x_upd)

		x = x_upd
		P = P_upd
	}

	fmt.println("\n=== END EKF TINY NONLINEAR TEST ===")
}


ekf_tiny_rts_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== EKF TINY NONLINEAR + RTS TEST ===")

	x: [2]f64 = {0.1, 1.0}
	P := matrix[2, 2]f64{
		0.1, 0.0,
		0.0, 0.1,
	}

	Q := matrix[2, 2]f64{
		0.001, 0.0,
		0.0, 0.001,
	}

	R := matrix[1, 1]f64{
		0.05,
	}

	f := proc(x: [2]f64) -> [2]f64 {
		pos := x[0]
		vel := x[1]
		return [2]f64{pos + vel * vel, vel}
	}

	F_jac := proc(x: [2]f64) -> matrix[2, 2]f64 {
		vel := x[1]
		return matrix[2, 2]f64{
			1.0, 0.0,
			2.0 * vel, 1.0,
		}
	}

	h := proc(x: [2]f64) -> [1]f64 {
		return [1]f64{math.sin_f64(x[0])}
	}

	H_jac := proc(x: [2]f64) -> matrix[1, 2]f64 {
		return matrix[1, 2]f64{
			math.cos_f64(x[0]), 0.0,
		}
	}

	z_seq := [5]f64 {
		math.sin_f64(0.1),
		math.sin_f64(1.1),
		math.sin_f64(2.1),
		math.sin_f64(3.1),
		math.sin_f64(4.1),
	}

	T := len(z_seq)

	xf := make([]w.KalmanState(2), T, allocator)
	xp := make([]w.KalmanState(2), T, allocator)
	F_seq := make([]matrix[2, 2]f64, T, allocator)

	for t in 0 ..< T {
		// store predicted state
		xp[t].x = x
		xp[t].P = P

		// EKF predict
		x_pred, P_pred := w.ekf_predict(x, P, f, F_jac, Q)
		F_seq[t] = F_jac(x) // Jacobian at previous state (or at x_pred if you prefer)

		// EKF update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := w.ekf_update(x_pred, P_pred, z, h, H_jac, R)

		xf[t].x = x_upd
		xf[t].P = P_upd

		x = x_upd
		P = P_upd
	}

	smoothed := w.ekf_rts_smooth(F_seq, xf, xp)

	fmt.println("\n--- FILTERED vs SMOOTHED (EKF) ---")
	for t in 0 ..< T {
		fmt.printf(
			"t=%d  z=%f  filtered=[%f, %f]  smoothed=[%f, %f]\n",
			t,
			z_seq[t],
			xf[t].x[0],
			xf[t].x[1],
			smoothed[t].x[0],
			smoothed[t].x[1],
		)
	}

	defer delete(smoothed)
	fmt.println("\n=== END EKF TINY NONLINEAR + RTS TEST ===")
}
ukf_tiny_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== UKF TINY NONLINEAR TEST ===")

	// Initial state
	x: [2]f64 = {0.1, 1.0} // pos=0.1, vel=1.0
	P := matrix[2, 2]f64{
		0.1, 0.0,
		0.0, 0.1,
	}

	Q := matrix[2, 2]f64{
		0.001, 0.0,
		0.0, 0.001,
	}

	R := matrix[1, 1]f64{
		0.05,
	}

	// UKF parameters (standard choice)
	params := w.UKF_Params {
		alpha = 1e-3,
		beta  = 2.0,
		kappa = 0.0,
	}

	// Nonlinear state transition: x' = f(x)
	f := proc(x: [2]f64) -> [2]f64 {
		pos := x[0]
		vel := x[1]
		return [2]f64{pos + vel * vel, vel}
	}

	// Nonlinear measurement: z = h(x) = sin(pos)
	h := proc(x: [2]f64) -> [1]f64 {
		return [1]f64{math.sin_f64(x[0])}
	}

	// Fake measurements: sin(true_position)
	z_seq := [5]f64 {
		math.sin_f64(0.1),
		math.sin_f64(1.1),
		math.sin_f64(2.1),
		math.sin_f64(3.1),
		math.sin_f64(4.1),
	}

	for t in 0 ..< len(z_seq) {
		fmt.printf("\n--- UKF Step %d ---\n", t)

		// Predict
		x_pred, P_pred := w.ukf_predict(x, P, f, Q, params, allocator)
		fmt.println("Predicted x:", x_pred)

		// Update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := w.ukf_update(x_pred, P_pred, z, h, R, params, allocator)
		fmt.println("Updated x:", x_upd)

		x = x_upd
		P = P_upd
	}

	fmt.println("\n=== END UKF TINY NONLINEAR TEST ===")
}
ukf_tiny_rts_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== UKF TINY NONLINEAR + RTS TEST ===")

	// Initial state
	x: [2]f64 = {0.1, 1.0}
	P := matrix[2, 2]f64{
		0.1, 0.0,
		0.0, 0.1,
	}

	Q := matrix[2, 2]f64{
		0.001, 0.0,
		0.0, 0.001,
	}

	R := matrix[1, 1]f64{
		0.05,
	}

	params := w.UKF_Params {
		alpha = 1e-3,
		beta  = 2.0,
		kappa = 0.0,
	}

	// Nonlinear state transition
	f := proc(x: [2]f64) -> [2]f64 {
		pos := x[0]
		vel := x[1]
		return [2]f64{pos + vel * vel, vel}
	}

	// Nonlinear measurement
	h := proc(x: [2]f64) -> [1]f64 {
		return [1]f64{math.sin_f64(x[0])}
	}

	// Fake measurements
	z_seq := [5]f64 {
		math.sin_f64(0.1),
		math.sin_f64(1.1),
		math.sin_f64(2.1),
		math.sin_f64(3.1),
		math.sin_f64(4.1),
	}

	T := len(z_seq)

	// Storage for forward pass
	xf := make([]w.KalmanState(2), T, allocator)
	xp := make([]w.KalmanState(2), T, allocator)

	Xi_pred := make([]([]([2]f64)), T, allocator)
	Xi_filt := make([]([]([2]f64)), T, allocator)
	Wc_seq := make([]([]f64), T, allocator)

	for t in 0 ..< T {
		// Store predicted state BEFORE prediction
		xp[t].x = x
		xp[t].P = P

		// Generate sigma points BEFORE prediction
		Xi_p, Wm, Wc := w.ukf_sigma_points(x, P, params, allocator)
		Xi_pred[t] = Xi_p
		Wc_seq[t] = Wc

		// Predict
		x_pred, P_pred := w.ukf_predict(x, P, f, Q, params, allocator)

		// Generate sigma points AFTER prediction (for smoothing)
		Xi_f, _, _ := w.ukf_sigma_points(x_pred, P_pred, params, allocator)
		Xi_filt[t] = Xi_f

		// Update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := w.ukf_update(x_pred, P_pred, z, h, R, params, allocator)

		xf[t].x = x_upd
		xf[t].P = P_upd

		x = x_upd
		P = P_upd
	}

	// RTS smoothing
	smoothed := w.ukf_rts_smooth(Xi_pred, Xi_filt, Wc_seq, xf, xp)

	fmt.println("\n--- FILTERED vs SMOOTHED (UKF) ---")
	for t in 0 ..< T {
		fmt.printf(
			"t=%d  z=%f  filtered=[%f, %f]  smoothed=[%f, %f]\n",
			t,
			z_seq[t],
			xf[t].x[0],
			xf[t].x[1],
			smoothed[t].x[0],
			smoothed[t].x[1],
		)
	}

	defer delete(smoothed)
	fmt.println("\n=== END UKF TINY NONLINEAR + RTS TEST ===")
}
ukf_tiny_control_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== UKF TINY NONLINEAR + CONTROL TEST ===")

	// Initial state
	x: [2]f64 = {0.1, 1.0}
	P := matrix[2, 2]f64{
		0.1, 0.0,
		0.0, 0.1,
	}

	Q := matrix[2, 2]f64{
		0.001, 0.0,
		0.0, 0.001,
	}

	R := matrix[1, 1]f64{
		0.05,
	}

	params := w.UKF_Params {
		alpha = 1e-3,
		beta  = 2.0,
		kappa = 0.0,
	}

	// control: scalar acceleration
	u: [1]f64 = {1.0}

	// Nonlinear state transition with control, e.g. pos += vel^2 + u[0]
	f := proc(x: [2]f64, u: [1]f64) -> [2]f64 {
		pos := x[0]
		vel := x[1]
		return [2]f64{pos + vel * vel + u[0], vel}
	}

	// Measurement independent of control (reuse h from UKF tiny test)
	h := proc(x: [2]f64) -> [1]f64 {
		return [1]f64{math.sin_f64(x[0])}
	}

	// Fake measurements: sin(true_position) as before
	z_seq := [5]f64 {
		math.sin_f64(0.1),
		math.sin_f64(1.1),
		math.sin_f64(2.1),
		math.sin_f64(3.1),
		math.sin_f64(4.1),
	}

	for t in 0 ..< len(z_seq) {
		fmt.printf("\n--- UKF+CTRL Step %d ---\n", t)

		// Predict with control
		x_pred, P_pred := w.ukf_predict_control(x, P, u, f, Q, params, allocator)
		fmt.println("Predicted x:", x_pred)

		// Update (no control in measurement here)
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := w.ukf_update(x_pred, P_pred, z, h, R, params, allocator)
		fmt.println("Updated x:", x_upd)

		x = x_upd
		P = P_upd
	}

	fmt.println("\n=== END UKF TINY NONLINEAR + CONTROL TEST ===")
}
