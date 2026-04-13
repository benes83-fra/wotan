package core

import linalg "core:math/linalg"
import "core:mem"

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
kalman_forward_tv_control :: proc(
	x0: [$N]f64,
	P0: matrix[N, N]f64,
	F_seq: []matrix[N, N]f64,
	H_seq: []matrix[$M, N]f64,
	B_seq: []matrix[N, $U]f64,
	Q_seq: []matrix[N, N]f64,
	R_seq: []matrix[M, M]f64,
	u_seq: []([U]f64),
	z_seq: []([M]f64),
	allocator: mem.Allocator,
) -> (
	xf: []KalmanState(N),
	xp: []KalmanState(N),
) {
	T := len(z_seq)
	assert(T == len(F_seq))
	assert(T == len(H_seq))
	assert(T == len(B_seq))
	assert(T == len(Q_seq))
	assert(T == len(R_seq))
	assert(T == len(u_seq))

	xf = make([]KalmanState(N), T, allocator)
	xp = make([]KalmanState(N), T, allocator)

	x := x0
	P := P0

	for t in 0 ..< T {
		xp[t].x = x
		xp[t].P = P

		F := F_seq[t]
		B := B_seq[t]
		Q := Q_seq[t]

		x_mat := transmute(matrix[N, 1]f64)x
		u_mat := transmute(matrix[U, 1]f64)u_seq[t]

		x_pred := F * x_mat + B * u_mat
		x = transmute([N]f64)x_pred

		P = (F * P * linalg.transpose(F)) + Q

		H := H_seq[t]
		R := R_seq[t]

		x_mat = transmute(matrix[N, 1]f64)x
		z_mat := transmute(matrix[M, 1]f64)z_seq[t]

		y := z_mat - (H * x_mat)

		Ht := linalg.transpose(H)
		S := (H * P * Ht) + R
		S_inv := linalg.inverse(S)

		K := (P * Ht) * S_inv

		x_new := x_mat + (K * y)
		x = transmute([N]f64)x_new

		I := linalg.identity(matrix[N, N]f64)
		P = (I - (K * H)) * P

		xf[t].x = x
		xf[t].P = P
	}

	return
}
rts_smooth_tv_control :: proc(
	F_seq: []matrix[$N, N]f64,
	B_seq: []matrix[N, $U]f64,
	u: []([U]f64), // control inputs per step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states (with control)
) -> []KalmanState(N) {
	assert(len(xf) == len(xp))
	assert(len(F_seq) == len(xf))
	assert(len(B_seq) == len(xf))
	assert(len(u) == len(xf))

	T := len(xf)
	smoothed := make([]KalmanState(N), T)
	if T == 0 {
		return smoothed
	}

	// last smoothed = last filtered
	smoothed[T - 1] = xf[T - 1]

	for k := T - 2; k >= 0; k -= 1 {
		F := F_seq[k]
		B := B_seq[k]
		Ft := linalg.transpose(F)

		Pf_k := xf[k].P
		Pp_k1 := xp[k + 1].P

		// Ck = Pf_k * Fᵀ * inv(Pp_{k+1})
		Pp_inv := linalg.inverse(Pp_k1)
		Ck := Pf_k * Ft * Pp_inv

		// controlled prediction: x_pred = F * xf[k] + B * u[k]
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
// =========================
// Extended Kalman Filter
// =========================

// Nonlinear predict: x' = f(x), P' = F(x) P F(x)ᵀ + Q
ekf_predict :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	f: proc(x: [N]f64) -> [N]f64,
	F_jac: proc(x: [N]f64) -> matrix[N, N]f64,
	Q: matrix[N, N]f64,
) -> (
	x_pred: [N]f64,
	P_pred: matrix[N, N]f64,
) {
	// propagate state nonlinearly
	x_pred = f(x)

	// Jacobian at current state
	F := F_jac(x)

	// propagate covariance
	P_pred = (F * P * linalg.transpose(F)) + Q

	return
}

// Nonlinear predict with control: x' = f(x, u), P' = F(x,u) P F(x,u)ᵀ + Q
ekf_predict_control :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	u: [$U]f64,
	f: proc(x: [N]f64, u: [U]f64) -> [N]f64,
	F_jac: proc(x: [N]f64, u: [U]f64) -> matrix[N, N]f64,
	Q: matrix[N, N]f64,
) -> (
	x_pred: [N]f64,
	P_pred: matrix[N, N]f64,
) {
	x_pred = f(x, u)
	F := F_jac(x, u)
	P_pred = (F * P * linalg.transpose(F)) + Q
	return
}

// Nonlinear update: z = h(x), with Jacobian H(x)
ekf_update :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	z: [$M]f64,
	h: proc(x: [N]f64) -> [M]f64,
	H_jac: proc(x: [N]f64) -> matrix[M, N]f64,
	R: matrix[M, M]f64,
) -> (
	x_upd: [N]f64,
	P_upd: matrix[N, N]f64,
) {
	// predicted measurement
	z_pred := h(x)

	// innovation: y = z - z_pred
	z_mat := transmute(matrix[M, 1]f64)z
	zp_mat := transmute(matrix[M, 1]f64)z_pred
	y_mat := z_mat - zp_mat

	// Jacobian at current state
	H := H_jac(x)
	Ht := linalg.transpose(H)

	// S = H P Hᵀ + R
	S := (H * P * Ht) + R
	S_inv := linalg.inverse(S)

	// K = P Hᵀ S⁻¹
	K := (P * Ht) * S_inv

	// x_new = x + K y
	x_mat := transmute(matrix[N, 1]f64)x
	x_new_mat := x_mat + (K * y_mat)
	x_upd = transmute([N]f64)x_new_mat

	// P_new = (I - K H) P
	I := linalg.identity(matrix[N, N]f64)
	P_upd = (I - (K * H)) * P

	return
}

// Nonlinear update with control-dependent measurement: z = h(x, u)
ekf_update_control :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	z: [$M]f64,
	u: [$U]f64,
	h: proc(x: [N]f64, u: [U]f64) -> [M]f64,
	H_jac: proc(x: [N]f64, u: [U]f64) -> matrix[M, N]f64,
	R: matrix[M, M]f64,
) -> (
	x_upd: [N]f64,
	P_upd: matrix[N, N]f64,
) {
	z_pred := h(x, u)

	z_mat := transmute(matrix[M, 1]f64)z
	zp_mat := transmute(matrix[M, 1]f64)z_pred
	y_mat := z_mat - zp_mat

	H := H_jac(x, u)
	Ht := linalg.transpose(H)

	S := (H * P * Ht) + R
	S_inv := linalg.inverse(S)

	K := (P * Ht) * S_inv

	x_mat := transmute(matrix[N, 1]f64)x
	x_new_mat := x_mat + (K * y_mat)
	x_upd = transmute([N]f64)x_new_mat

	I := linalg.identity(matrix[N, N]f64)
	P_upd = (I - (K * H)) * P

	return
}


// EKF RTS smoother (time-varying, no explicit control)
// Uses the linearized F_k (Jacobians) from the EKF forward pass.
ekf_rts_smooth :: proc(
	F_seq: []matrix[$N, N]f64, // Jacobians F_k at each step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states
) -> []KalmanState(N) {
	assert(len(xf) == len(xp))
	assert(len(F_seq) == len(xf))

	T := len(xf)
	smoothed := make([]KalmanState(N), T)
	if T == 0 {
		return smoothed
	}

	// last smoothed = last filtered
	smoothed[T - 1] = xf[T - 1]

	for k := T - 2; k >= 0; k -= 1 {
		F := F_seq[k]
		Ft := linalg.transpose(F)

		Pf_k := xf[k].P
		Pp_k1 := xp[k + 1].P

		// Ck = Pf_k * Fᵀ * inv(Pp_{k+1})
		Pp_inv := linalg.inverse(Pp_k1)
		Ck := Pf_k * Ft * Pp_inv

		// x_s[k] = x_f[k] + Ck * (x_s[k+1] - x_p[k+1])
		diff_x := smoothed[k + 1].x - xp[k + 1].x
		corr_x := linalg.mul(Ck, diff_x)
		smoothed[k].x = xf[k].x + corr_x

		// P_s[k] = P_f[k] + Ck * (P_s[k+1] - P_p[k+1]) * Ckᵀ
		diff_P := smoothed[k + 1].P - Pp_k1
		Ck_t := linalg.transpose(Ck)
		smoothed[k].P = Pf_k + Ck * diff_P * Ck_t
	}

	return smoothed
}
