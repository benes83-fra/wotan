package tests

import analytic "../wotan/analytics"
import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

// Kalman / EKF / UKF filters and RTS smoothers.
//
// Conventions checked here (see wotan/analytics/kalman.odin):
// - Odin matrix literals are written ROW-MAJOR: matrix[2, 2]f64{a, b, c, d} = [[a, b], [c, d]].
// - xp[t] is the PREDICTED state x_{t|t-1}, P_{t|t-1}, stored after predict and before update.
// - u[t] / F_seq[t] / B_seq[t] are the ones used in the predict step that produced xp[t].
// - UKF RTS: Xi_filt[t] = sigma points of xf[t], Xi_pred[t] = those of xf[t-1] pushed through f.
// Reference values: straightforward numpy implementations (Joseph-form covariance update,
// Wan & van der Merwe sigma point weights, Särkkä's unscented RTS smoother).

// max |got - want| / max(1, |want|)
kt_err :: proc(got, want: []f64) -> f64 {
	assert(len(got) == len(want))
	e := 0.0
	for i in 0 ..< len(got) {
		e = max(e, abs(got[i] - want[i]) / max(1.0, abs(want[i])))
	}
	return e
}

kt_check :: proc(passed, total: ^int, got, want: []f64, tol: f64, name: string) {
	e := kt_err(got, want)
	ok := e < tol // NaN fails
	total^ += 1
	if ok {passed^ += 1}
	fmt.printf("%s %s (max rel err %.2e)\n", "OK  " if ok else "FAIL", name, e)
}

// row-major flattening of a 2x2 matrix
kt_flat2 :: proc(m: matrix[2, 2]f64) -> [4]f64 {
	return {m[0, 0], m[0, 1], m[1, 0], m[1, 1]}
}

kt_check_state :: proc(passed, total: ^int, s: analytic.KalmanState(2), x_ref, P_ref: []f64, tol: f64, name: string) {
	x := s.x
	P := kt_flat2(s.P)
	kt_check(passed, total, x[:], x_ref, tol, fmt.tprintf("%s mean", name))
	kt_check(passed, total, P[:], P_ref, tol, fmt.tprintf("%s covariance", name))
}

kalman_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== KALMAN FILTER TEST ===")
	passed, total := 0, 0

	x0: [2]f64 = {0.0, 1.0}

	// Matrix literals are row-major.
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	// Constant velocity: F = [[1, 1], [0, 1]]
	F := matrix[2, 2]f64{
		1.0, 1.0,
		0.0, 1.0,
	}

	// H = [[1, 0]]: position is observed
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

	kf := analytic.kalman_init(x0, P0, F, H, Q, R)

	measurements := [6]f64{1.2, 2.1, 2.9, 4.2, 5.1, 6.05}

	for i in 0 ..< len(measurements) {
		fmt.printf("\n--- Step %d ---\n", i)

		z: [1]f64 = {measurements[i]}

		fmt.println("Predicting...")
		analytic.kalman_predict(&kf)
		fmt.println("Predicted state:", kf.x)

		fmt.println("Updating with measurement:", z)
		analytic.kalman_update(&kf, z)
		fmt.println("Updated state:", kf.x)
	}


	T := len(measurements)

	xf := make([]analytic.KalmanState(2), T, allocator)
	xp := make([]analytic.KalmanState(2), T, allocator)

	// forward pass (restart from the prior)
	kf = analytic.kalman_init(x0, P0, F, H, Q, R)
	for t in 0 ..< T {
		analytic.kalman_predict(&kf)
		xp[t].x = kf.x // predicted x_{t|t-1}
		xp[t].P = kf.P

		z: [1]f64 = {measurements[t]}
		analytic.kalman_update(&kf, z)

		xf[t].x = kf.x
		xf[t].P = kf.P
	}


	// smoothing
	smoothed := analytic.rts_smooth(F, xf[:], xp[:])
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

	// numpy reference
	kt_check_state(&passed, &total, xf[T - 1],
		{6.0831720159952569, 1.0012155502585141},
		{0.47318033389019487, 0.118576103737942, 0.11857610373794197, 0.062651900045759171},
		1e-12, "KF filtered state t=5")
	kt_check_state(&passed, &total, smoothed[0],
		{1.0759213090169446, 1.0016751633576928},
		{0.28573612185621222, -0.069673343056310122, -0.069673343056310122, 0.043954756656376381},
		1e-12, "RTS smoothed state t=0")

	fmt.printf("Kalman filter summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END KALMAN FILTER TEST ===")
}


kalman_control_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)


	fmt.println("=== KALMAN CONTROL FILTER TEST ===")
	passed, total := 0, 0

	x0: [2]f64 = {0.0, 1.0}

	// Matrix literals are row-major.
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	// Constant velocity: F = [[1, 1], [0, 1]]
	F := matrix[2, 2]f64{
		1.0, 1.0,
		0.0, 1.0,
	}

	// H = [[1, 0]]
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

	kf := analytic.kalman_init(x0, P0, F, H, Q, R)

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

	xf := make([]analytic.KalmanState(2), T, allocator)
	xp := make([]analytic.KalmanState(2), T, allocator)

	for i in 0 ..< len(measurements) {
		fmt.printf("\n--- Step %d ---\n", i)

		z: [1]f64 = {measurements[i]}

		fmt.println("Predicting...")
		analytic.kalman_predict(&kf)
		fmt.println("Predicted state:", kf.x)

		fmt.println("Updating with measurement:", z)
		analytic.kalman_update(&kf, z)
		fmt.println("Updated state:", kf.x)
	}
	// forward pass with control (restart from the prior)
	kf = analytic.kalman_init(x0, P0, F, H, Q, R)
	for t in 0 ..< T {
		analytic.kalman_predict_control(&kf, B, u[t])
		xp[t].x = kf.x // predicted x_{t|t-1} = F x_{t-1|t-1} + B u[t]
		xp[t].P = kf.P

		z: [1]f64 = {measurements[t]}
		analytic.kalman_update(&kf, z)

		xf[t].x = kf.x
		xf[t].P = kf.P
	}


	// smoothing
	smoothed := analytic.rts_smooth_control(F, B, u[:], xf[:], xp[:])
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

	// numpy reference
	kt_check_state(&passed, &total, xf[T - 1],
		{8.7166511800250603, 3.9832761346560841},
		{0.47318033389019487, 0.118576103737942, 0.11857610373794197, 0.062651900045759171},
		1e-12, "KF+control filtered state t=5")
	kt_check_state(&passed, &total, smoothed[0],
		{0.87307214347987627, -0.79982987538662087},
		{0.28573612185621222, -0.069673343056310122, -0.069673343056310122, 0.043954756656376381},
		1e-12, "RTS+control smoothed state t=0")

	fmt.printf("Kalman control summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END Control KALMAN FILTER TEST ===")
}
kalman_tv_control_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== KALMAN TV CONTROL FILTER + SMOOTHER TEST ===")
	passed, total := 0, 0

	// State: [position, velocity]
	x0: [2]f64 = {0.0, 0.0}
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	T := 6

	// Time-varying F (here: constant velocity, but as a sequence)
	F_seq := make([]matrix[2, 2]f64, T, allocator)
	for i in 0 ..< T {
		F_seq[i] = matrix[2, 2]f64{
			1.0, 1.0,
			0.0, 1.0,
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

	xf, xp := analytic.kalman_forward_tv_control(
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

	smoothed := analytic.rts_smooth_tv_control(F_seq, B_seq, u_seq, xf, xp)

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

	// numpy reference; xp[0] must be the prediction F x0 + B u0, F P0 Fᵀ + Q
	kt_check_state(&passed, &total, xp[0], {0.5, 1}, {2.0099999999999998, 1, 1, 1.01}, 1e-14,
		"TV forward: xp[0] is the predicted state")
	kt_check_state(&passed, &total, xf[T - 1],
		{15.036932961484824, 5.5095138821166243},
		{0.47318033389019487, 0.118576103737942, 0.11857610373794197, 0.062651900045759171},
		1e-12, "TV filtered state t=5")
	kt_check_state(&passed, &total, smoothed[0],
		{0.0043426644404910286, 0.50841500706997211},
		{0.28573612185621222, -0.069673343056310122, -0.069673343056310122, 0.043954756656376381},
		1e-12, "TV RTS smoothed state t=0")

	fmt.printf("Kalman TV control summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END KALMAN TV CONTROL FILTER + SMOOTHER TEST ===")
}


ekf_tiny_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== EKF TINY NONLINEAR TEST ===")
	passed, total := 0, 0

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

	// Jacobian of f: [[1, 2 vel], [0, 1]] (row-major literal)
	F_jac := proc(x: [2]f64) -> matrix[2, 2]f64 {
		vel := x[1]
		return matrix[2, 2]f64{
			1.0, 2.0 * vel,
			0.0, 1.0,
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
		x_pred, P_pred := analytic.ekf_predict(x, P, f, F_jac, Q)

		fmt.println("Predicted x:", x_pred)

		// Update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := analytic.ekf_update(x_pred, P_pred, z, h, H_jac, R)

		fmt.println("Updated x:", x_upd)

		x = x_upd
		P = P_upd
	}

	// numpy reference
	kt_check_state(&passed, &total, analytic.KalmanState(2){x, P},
		{3.9196252112559429, 0.98790059391239182},
		{0.047833894092969141, 0.0083676778080124595, 0.0083676778080124525, 0.0041504338541931366},
		1e-10, "EKF filtered state t=4")

	fmt.printf("EKF summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END EKF TINY NONLINEAR TEST ===")
}


ekf_tiny_rts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== EKF TINY NONLINEAR + RTS TEST ===")
	passed, total := 0, 0

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

	// Jacobian of f: [[1, 2 vel], [0, 1]] (row-major literal)
	F_jac := proc(x: [2]f64) -> matrix[2, 2]f64 {
		vel := x[1]
		return matrix[2, 2]f64{
			1.0, 2.0 * vel,
			0.0, 1.0,
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

	xf := make([]analytic.KalmanState(2), T, allocator)
	xp := make([]analytic.KalmanState(2), T, allocator)
	F_seq := make([]matrix[2, 2]f64, T, allocator)

	for t in 0 ..< T {
		// EKF predict
		x_pred, P_pred := analytic.ekf_predict(x, P, f, F_jac, Q)
		F_seq[t] = F_jac(x) // Jacobian used for this predict step (at xf[t-1])

		// store predicted state x_{t|t-1}
		xp[t].x = x_pred
		xp[t].P = P_pred

		// EKF update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := analytic.ekf_update(x_pred, P_pred, z, h, H_jac, R)

		xf[t].x = x_upd
		xf[t].P = P_upd

		x = x_upd
		P = P_upd
	}

	smoothed := analytic.ekf_rts_smooth(F_seq, xf, xp)

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

	// numpy reference
	kt_check_state(&passed, &total, smoothed[0],
		{0.41143498901478154, 0.96435920192257496},
		{0.029168235823495176, -0.004400404179866868, -0.0044004041798668958, 0.0029979616192346709},
		1e-10, "EKF RTS smoothed state t=0")

	fmt.printf("EKF RTS summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END EKF TINY NONLINEAR + RTS TEST ===")
}
ukf_tiny_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== UKF TINY NONLINEAR TEST ===")
	passed, total := 0, 0

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
	params := analytic.UKF_Params {
		alpha = 1e-3,
		beta  = 2.0,
		kappa = 0.0,
	}

	// Sigma point weights (Wan & van der Merwe): lambda = alpha^2 (n + kappa) - n,
	// Wm0 = lambda / (n + lambda), Wc0 = Wm0 + 1 - alpha^2 + beta, Wi = 1 / (2 (n + lambda))
	{
		_, Wm, Wc := analytic.ukf_sigma_points([2]f64{}, P, params, allocator)
		kt_check(&passed, &total, Wm,
			{-999998.99997124437, 249999.99999281109, 249999.99999281109, 249999.99999281109, 249999.99999281109},
			1e-12, "UKF mean weights Wm")
		kt_check(&passed, &total, Wc,
			{-999995.99997224438, 249999.99999281109, 249999.99999281109, 249999.99999281109, 249999.99999281109},
			1e-12, "UKF covariance weights Wc")
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
		x_pred, P_pred := analytic.ukf_predict(x, P, f, Q, params, allocator)
		fmt.println("Predicted x:", x_pred)

		// Update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := analytic.ukf_update(x_pred, P_pred, z, h, R, params, allocator)
		fmt.println("Updated x:", x_upd)

		x = x_upd
		P = P_upd
	}

	// numpy reference (alpha = 1e-3 weights of size 1e6 cost ~6 digits to cancellation)
	kt_check_state(&passed, &total, analytic.KalmanState(2){x, P},
		{3.6590370305522133, 0.82707378145810639},
		{0.16731180054029426, 0.021013275384935221, 0.021013275384935193, 0.0071498494238763609},
		1e-6, "UKF filtered state t=4")

	fmt.printf("UKF summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END UKF TINY NONLINEAR TEST ===")
}
ukf_tiny_rts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== UKF TINY NONLINEAR + RTS TEST ===")
	passed, total := 0, 0

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

	params := analytic.UKF_Params {
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
	xf := make([]analytic.KalmanState(2), T, allocator)
	xp := make([]analytic.KalmanState(2), T, allocator)

	Xi_pred := make([]([]([2]f64)), T, allocator)
	Xi_filt := make([]([]([2]f64)), T, allocator)
	Wc_seq := make([]([]f64), T, allocator)

	for t in 0 ..< T {
		// Propagated sigma points f(X) of the previous filtered state (they produce xp[t])
		Xi, _, _ := analytic.ukf_sigma_points(x, P, params, allocator)
		for i in 0 ..< len(Xi) {Xi[i] = f(Xi[i])}
		Xi_pred[t] = Xi

		// Predict
		x_pred, P_pred := analytic.ukf_predict(x, P, f, Q, params, allocator)
		xp[t].x = x_pred
		xp[t].P = P_pred

		// Update
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := analytic.ukf_update(x_pred, P_pred, z, h, R, params, allocator)

		xf[t].x = x_upd
		xf[t].P = P_upd

		// Sigma points of the filtered state (and their covariance weights)
		Xi_filt[t], _, Wc_seq[t] = analytic.ukf_sigma_points(x_upd, P_upd, params, allocator)

		x = x_upd
		P = P_upd
	}

	// RTS smoothing
	smoothed := analytic.ukf_rts_smooth(Xi_pred, Xi_filt, Wc_seq, xf, xp)

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

	// numpy reference (Särkkä URTS)
	kt_check_state(&passed, &total, smoothed[0],
		{0.71961208999182902, 0.82465685883201212},
		{0.078705912095713093, -0.0041485552655528557, -0.0041485552655528835, 0.0051936079793146739},
		1e-6, "UKF RTS smoothed state t=0")

	fmt.printf("UKF RTS summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END UKF TINY NONLINEAR + RTS TEST ===")
}
ukf_tiny_control_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== UKF TINY NONLINEAR + CONTROL TEST ===")
	passed, total := 0, 0

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

	params := analytic.UKF_Params {
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
		x_pred, P_pred := analytic.ukf_predict_control(x, P, u, f, Q, params, allocator)
		fmt.println("Predicted x:", x_pred)

		// Update (no control in measurement here)
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := analytic.ukf_update(x_pred, P_pred, z, h, R, params, allocator)
		fmt.println("Updated x:", x_upd)

		x = x_upd
		P = P_upd
	}

	// numpy reference
	kt_check_state(&passed, &total, analytic.KalmanState(2){x, P},
		{12.354585046504935, 1.1671702666995269},
		{0.41679768120001426, 0.039558418466256545, 0.039558418466256545, 0.006468139854150989},
		1e-6, "UKF+control filtered state t=4")

	fmt.printf("UKF control summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END UKF TINY NONLINEAR + CONTROL TEST ===")
}
ukf_tiny_control_rts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("=== UKF TINY NONLINEAR + CONTROL + RTS TEST ===")
	passed, total := 0, 0

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

	// UKF parameters
	params := analytic.UKF_Params {
		alpha = 1e-3,
		beta  = 2.0,
		kappa = 0.0,
	}

	// Nonlinear state transition with control
	f := proc(x: [2]f64, u: [1]f64) -> [2]f64 {
		pos := x[0]
		vel := x[1]
		acc := u[0]
		return [2]f64{pos + vel * vel + 0.5 * acc, vel + acc}
	}

	// Nonlinear measurement
	h := proc(x: [2]f64, u: [1]f64) -> [1]f64 {
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

	// Constant control input u = 1.0
	u: [1]f64 = {1.0}

	T := len(z_seq)

	xf := make([]analytic.KalmanState(2), T, allocator)
	xp := make([]analytic.KalmanState(2), T, allocator)

	Xi_pred := make([]([]([2]f64)), T, allocator)
	Xi_filt := make([]([]([2]f64)), T, allocator)
	Wc_seq := make([]([]f64), T, allocator)

	for t in 0 ..< T {
		// Propagated sigma points f(X, u_t) of the previous filtered state
		Xi, _, _ := analytic.ukf_sigma_points(x, P, params, allocator)
		for i in 0 ..< len(Xi) {Xi[i] = f(Xi[i], u)}
		Xi_pred[t] = Xi

		// UKF predict with control
		x_pred, P_pred := analytic.ukf_predict_control(x, P, u, f, Q, params, allocator)
		xp[t].x = x_pred
		xp[t].P = P_pred

		// UKF update with control
		z: [1]f64 = {z_seq[t]}
		x_upd, P_upd := analytic.ukf_update_control(x_pred, P_pred, z, u, h, R, params, allocator)
		Xi_filt[t], _, Wc_seq[t] = analytic.ukf_sigma_points(x_upd, P_upd, params, allocator)

		xf[t].x = x_upd
		xf[t].P = P_upd

		x = x_upd
		P = P_upd
	}

	// RTS smoothing
	smoothed := analytic.ukf_rts_smooth_control(Xi_pred, Xi_filt, Wc_seq, xf, xp)

	fmt.println("\n--- FILTERED vs SMOOTHED (UKF + CONTROL) ---")
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

	// numpy reference (Särkkä URTS)
	kt_check_state(&passed, &total, xf[T - 1],
		{66.763892110401116, 6.302639398508858},
		{0.048396513836652555, 0.0028893095315669186, 0.0028893095315669186, 0.001568944420328133},
		1e-6, "UKF+control filtered state t=4")
	kt_check_state(&passed, &total, smoothed[0],
		{2.2344445261079215, 2.274282942027714},
		{0.092359991811301723, -0.0031209151392220524, -0.0031209151392221079, 0.0011264668981293829},
		1e-6, "UKF+control RTS smoothed state t=0")

	delete(smoothed)
	fmt.printf("UKF control RTS summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
	fmt.println("\n=== END UKF TINY NONLINEAR + CONTROL + RTS TEST ===")
}
