package core

import linalg "core:math/linalg"

KalmanFilter :: struct($N: int, $M: int) {
	x: [N]f64,
	P: matrix[N, N]f64,
	F: matrix[N, N]f64,
	H: matrix[M, N]f64,
	Q: matrix[N, N]f64,
	R: matrix[M, M]f64,
}

KalmanState :: struct($N: int) {
	x: [N]f64,
	P: matrix[N, N]f64,
}
kalman_init :: proc(
	x0: [$N]f64,
	P0: matrix[N, N]f64,
	F: matrix[N, N]f64,
	H: matrix[$M, N]f64,
	Q: matrix[N, N]f64,
	R: matrix[M, M]f64,
) -> KalmanFilter(N, M) {
	return KalmanFilter(N, M){x = x0, P = P0, F = F, H = H, Q = Q, R = R}
}

kalman_predict :: proc(kf: ^KalmanFilter($N, $M)) {
	// Cast x to a column matrix for consistent multiplication
	x_mat := transmute(matrix[N, 1]f64)kf.x
	new_x := kf.F * x_mat
	kf.x = transmute([N]f64)new_x

	kf.P = (kf.F * kf.P * linalg.transpose(kf.F)) + kf.Q
}


kalman_predict_control :: proc(kf: ^KalmanFilter($N, $M), B: matrix[N, $U]f64, u: [U]f64) {
	// x and u as column vectors
	x_mat := transmute(matrix[N, 1]f64)kf.x
	u_mat := transmute(matrix[U, 1]f64)u

	// x' = F x + B u
	new_x := kf.F * x_mat + B * u_mat
	kf.x = transmute([N]f64)new_x

	// P' = F P Fᵀ + Q   (control assumed deterministic)
	kf.P = (kf.F * kf.P * linalg.transpose(kf.F)) + kf.Q
}


kalman_update :: proc(kf: ^KalmanFilter($N, $M), z: [M]f64) {
	// 1. Cast arrays to matrices to avoid linalg.mul ambiguity
	x_mat := transmute(matrix[N, 1]f64)kf.x
	z_mat := transmute(matrix[M, 1]f64)z

	// 2. Innovation: y = z - H*x
	y_mat := z_mat - (kf.H * x_mat)

	// 3. S = H*P*Ht + R
	Ht := linalg.transpose(kf.H)
	S := (kf.H * kf.P * Ht) + kf.R

	// 4. K = P*Ht*inv(S)
	S_inv := linalg.inverse(S)
	K := (kf.P * Ht) * S_inv

	// 5. Update state: x = x + K*y
	new_x_mat := x_mat + (K * y_mat)
	kf.x = transmute([N]f64)new_x_mat

	// 6. Update covariance: P = (I - K*H) * P
	I := linalg.identity(matrix[N, N]f64)
	kf.P = (I - (K * kf.H)) * kf.P
}
rts_smooth :: proc(
	F: matrix[$N, N]f64,
	xf: []KalmanState(N), // filtered
	xp: []KalmanState(N), // predicted
) -> []KalmanState(N) {
	assert(len(xf) == len(xp))
	T := len(xf)

	smoothed := make([]KalmanState(N), T)
	if T == 0 {
		return smoothed
	}

	smoothed[T - 1] = xf[T - 1]

	Ft := linalg.transpose(F)

	for k := T - 2; k >= 0; k -= 1 {
		Pf_k := xf[k].P
		Pp_k1 := xp[k + 1].P

		Pp_inv := linalg.inverse(Pp_k1)
		Ck := Pf_k * Ft * Pp_inv

		diff_x := smoothed[k + 1].x - xp[k + 1].x
		corr_x := linalg.mul(Ck, diff_x)
		smoothed[k].x = xf[k].x + corr_x

		diff_P := smoothed[k + 1].P - Pp_k1
		Ck_t := linalg.transpose(Ck)
		smoothed[k].P = Pf_k + Ck * diff_P * Ck_t
	}

	return smoothed
}

rts_smooth_control :: proc(
	F: matrix[$N, N]f64,
	B: matrix[N, $U]f64,
	u: []([U]f64), // control inputs for each step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states (must include control)
) -> []KalmanState(N) {

	assert(len(xf) == len(xp))
	assert(len(u) == len(xf)) // one control per step

	T := len(xf)
	smoothed := make([]KalmanState(N), T)

	if T == 0 {
		return smoothed
	}

	smoothed[T - 1] = xf[T - 1]

	Ft := linalg.transpose(F)

	for k := T - 2; k >= 0; k -= 1 {
		Pf_k := xf[k].P
		Pp_k1 := xp[k + 1].P

		// Ck = Pf_k * Fᵀ * inv(Pp_{k+1})
		Pp_inv := linalg.inverse(Pp_k1)
		Ck := Pf_k * Ft * Pp_inv

		// Compute controlled prediction: x_pred = F * xf[k] + B * u[k]
		xk_mat := transmute(matrix[N, 1]f64)xf[k].x
		uk_mat := transmute(matrix[U, 1]f64)u[k]

		x_pred := F * xk_mat + B * uk_mat
		x_pred_vec := transmute([N]f64)x_pred

		// x_s[k] = x_f[k] + Ck * (x_s[k+1] - x_pred)
		diff_x := smoothed[k + 1].x - x_pred_vec
		corr_x := linalg.mul(Ck, diff_x)
		smoothed[k].x = xf[k].x + corr_x

		// P_s[k] = P_f[k] + Ck * (P_s[k+1] - P_p[k+1]) * Ckᵀ
		diff_P := smoothed[k + 1].P - Pp_k1
		Ck_t := linalg.transpose(Ck)
		smoothed[k].P = Pf_k + Ck * diff_P * Ck_t
	}

	return smoothed
}
