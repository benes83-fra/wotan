package core

import "base:intrinsics"
import "core:math"
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

UKF_Params :: struct {
	alpha: f64,
	beta:  f64,
	kappa: f64,
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


// Generate sigma points and weights for UKF
ukf_sigma_points :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	params: UKF_Params,
	allocator: mem.Allocator,
) -> (
	Xi: []([N]f64),
	Wm: []f64,
	Wc: []f64, // sigma points// mean weights// covariance weights
) {
	n := N
	lambda := params.alpha * params.alpha * (f64(n) + params.kappa) - f64(n)
	c := f64(n) + lambda

	L := 2 * n + 1
	Xi = make([]([N]f64), L, allocator)
	Wm = make([]f64, L, allocator)
	Wc = make([]f64, L, allocator)

	// Weights
	Wm[0] = lambda / c
	Wc[0] = lambda / c + (1.0 - params.alpha * params.alpha + params.beta)
	for i in 1 ..< L {
		w := 0.5 / c
		Wm[i] = w
		Wc[i] = w
	}

	// Cholesky of scaled covariance
	P_scaled := matrix_scalar_mul(P, c)
	S := cholesky(P_scaled) // assumes linalg.cholesky returns upper or lower; we just use columns

	// Center sigma point
	Xi[0] = x

	// +/- columns of S
	for i in 0 ..< n {
		col_i := [N]f64{}
		for r in 0 ..< n {
			col_i[r] = S[r, i]
		}

		// x + col_i
		x_plus := [N]f64{}
		x_minus := [N]f64{}
		for r in 0 ..< n {
			x_plus[r] = x[r] + col_i[r]
			x_minus[r] = x[r] - col_i[r]
		}

		Xi[1 + i] = x_plus
		Xi[1 + i + n] = x_minus
	}

	return
}

// UKF predict: x' = f(x), P' = cov(f(Xi)) + Q
ukf_predict :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	f: proc(x: [N]f64) -> [N]f64,
	Q: matrix[N, N]f64,
	params: UKF_Params,
	allocator: mem.Allocator,
) -> (
	x_pred: [N]f64,
	P_pred: matrix[N, N]f64,
) {
	Xi, Wm, Wc := ukf_sigma_points(x, P, params, allocator)

	L := len(Xi)

	// propagate sigma points through f
	Y := make([]([N]f64), L, allocator)
	for i in 0 ..< L {
		Y[i] = f(Xi[i])
	}

	// mean
	for j in 0 ..< N {
		sum := 0.0
		for i in 0 ..< L {
			sum += Wm[i] * Y[i][j]
		}
		x_pred[j] = sum
	}

	// covariance
	P_pred = Q
	for i in 0 ..< L {
		dx := [N]f64{}
		for j in 0 ..< N {
			dx[j] = Y[i][j] - x_pred[j]
		}
		dx_mat := transmute(matrix[N, 1]f64)dx
		dx_t := linalg.transpose(dx_mat)
		P_pred += Wc[i] * (dx_mat * dx_t)
	}

	return
}

// UKF update: z = h(x)
ukf_update :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	z: [$M]f64,
	h: proc(x: [N]f64) -> [M]f64,
	R: matrix[M, M]f64,
	params: UKF_Params,
	allocator: mem.Allocator,
) -> (
	x_upd: [N]f64,
	P_upd: matrix[N, N]f64,
) {
	// sigma points from prior
	Xi, Wm, Wc := ukf_sigma_points(x, P, params, allocator)
	L := len(Xi)

	// propagate to measurement space
	Zi := make([]([M]f64), L, allocator)
	for i in 0 ..< L {
		Zi[i] = h(Xi[i])
	}

	// predicted measurement mean
	z_pred := [M]f64{}
	for j in 0 ..< M {
		sum := 0.0
		for i in 0 ..< L {
			sum += Wm[i] * Zi[i][j]
		}
		z_pred[j] = sum
	}

	// innovation covariance S and cross-covariance Pxz
	S := R
	Pxz := matrix[N, M]f64{} // initialized to zero

	for i in 0 ..< L {
		// state deviation
		dx := [N]f64{}
		for j in 0 ..< N {
			dx[j] = Xi[i][j] - x[j]
		}
		dx_mat := transmute(matrix[N, 1]f64)dx

		// measurement deviation
		dz := [M]f64{}
		for j in 0 ..< M {
			dz[j] = Zi[i][j] - z_pred[j]
		}
		dz_mat := transmute(matrix[M, 1]f64)dz
		dz_t := linalg.transpose(dz_mat)

		S += Wc[i] * (dz_mat * dz_t)
		Pxz += Wc[i] * (dx_mat * dz_t)
	}

	// Kalman gain K = Pxz * S⁻¹
	S_inv := linalg.inverse(S)
	K := Pxz * S_inv

	// update state
	z_mat := transmute(matrix[M, 1]f64)z
	zp_mat := transmute(matrix[M, 1]f64)z_pred
	y_mat := z_mat - zp_mat
	x_mat := transmute(matrix[N, 1]f64)x
	x_new := x_mat + K * y_mat
	x_upd = transmute([N]f64)x_new

	// update covariance
	K_t := linalg.transpose(K)
	P_upd = P - K * S * K_t

	return
}


@(require_results)
matrix_scalar_mul :: proc "contextless" (
	a: $A/matrix[$N, $M]$T,
	s: T,
) -> (
	c: A,
) where intrinsics.type_is_numeric(T) #no_bounds_check {
	for j in 0 ..< M {
		for i in 0 ..< N {
			c[i, j] = a[i, j] * s
		}
	}
	return
}


mul :: proc {
	linalg.matrix_mul,
	linalg.matrix_mul_differ,
	linalg.matrix_mul_vector,
	linalg.quaternion64_mul_vector3,
	linalg.quaternion128_mul_vector3,
	linalg.quaternion256_mul_vector3,
	linalg.quaternion_mul_quaternion,
	matrix_scalar_mul, // ← ADD THIS
}


@(require_results)
cholesky :: proc "contextless" (A: $M/matrix[$N, N]$T) -> (L: M) where intrinsics.type_is_float(T),
	N > 0 #no_bounds_check {
	// Zero initialize
	for j in 0 ..< N {
		for i in 0 ..< N {
			L[i, j] = 0
		}
	}

	for j in 0 ..< N {
		// diagonal
		sum := A[j, j]
		for k in 0 ..< j {
			sum -= L[j, k] * L[j, k]
		}
		L[j, j] = math.sqrt_f64(sum)

		// off-diagonal
		for i in j + 1 ..< N {
			sum = A[i, j]
			for k in 0 ..< j {
				sum -= L[i, k] * L[j, k]
			}
			L[i, j] = sum / L[j, j]
		}
	}

	return
}
ukf_rts_smooth :: proc(
	Xi_pred: []([]([$N]f64)), // sigma points BEFORE update (prediction)
	Xi_filt: []([]([N]f64)), // sigma points AFTER update (filtered)
	Wc_seq: []([]f64), // covariance weights per step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states
) -> []KalmanState(N) {

	T := len(xf)
	assert(T == len(xp))
	assert(T == len(Xi_pred))
	assert(T == len(Xi_filt))
	assert(T == len(Wc_seq))

	smoothed := make([]KalmanState(N), T)
	if T == 0 {
		return smoothed
	}

	// last smoothed = last filtered
	smoothed[T - 1] = xf[T - 1]

	for k := T - 2; k >= 0; k -= 1 {
		// predicted and filtered sigma points
		Xi_p := Xi_pred[k]
		Xi_f := Xi_filt[k]

		Wc := Wc_seq[k]

		// predicted mean and covariance
		x_pred := xp[k].x
		P_pred := xp[k].P

		// filtered mean and covariance
		x_filt := xf[k].x
		P_filt := xf[k].P

		// cross-covariance Pxf
		Pxf := matrix[N, N]f64{}

		L := len(Xi_p)
		for i in 0 ..< L {
			dx_p := [N]f64{}
			dx_f := [N]f64{}

			for j in 0 ..< N {
				dx_p[j] = Xi_p[i][j] - x_pred[j]
				dx_f[j] = Xi_f[i][j] - x_filt[j]
			}

			dxp_mat := transmute(matrix[N, 1]f64)dx_p
			dxf_t := linalg.transpose(transmute(matrix[N, 1]f64)dx_f)

			Pxf += Wc[i] * (dxp_mat * dxf_t)
		}

		// smoother gain Gk = Pxf * inv(P_filt)
		P_filt_inv := linalg.inverse(P_filt)
		Gk := Pxf * P_filt_inv

		// smoothed state
		diff_x := smoothed[k + 1].x - x_filt
		dx_mat := transmute(matrix[N, 1]f64)diff_x
		corr := Gk * dx_mat
		smoothed[k].x = x_pred + transmute([N]f64)corr

		// smoothed covariance
		diff_P := smoothed[k + 1].P - P_filt
		smoothed[k].P = P_pred + Gk * diff_P * linalg.transpose(Gk)
	}

	return smoothed
}
// UKF predict with control: x' = f(x, u)
ukf_predict_control :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	u: [$U]f64,
	f: proc(x: [N]f64, u: [U]f64) -> [N]f64,
	Q: matrix[N, N]f64,
	params: UKF_Params,
	allocator: mem.Allocator,
) -> (
	x_pred: [N]f64,
	P_pred: matrix[N, N]f64,
) {
	Xi, Wm, Wc := ukf_sigma_points(x, P, params, allocator)
	L := len(Xi)

	// propagate sigma points through f(x,u)
	Y := make([]([N]f64), L, allocator)
	for i in 0 ..< L {
		Y[i] = f(Xi[i], u)
	}

	// mean
	for j in 0 ..< N {
		sum := 0.0
		for i in 0 ..< L {
			sum += Wm[i] * Y[i][j]
		}
		x_pred[j] = sum
	}

	// covariance
	P_pred = Q
	for i in 0 ..< L {
		dx := [N]f64{}
		for j in 0 ..< N {
			dx[j] = Y[i][j] - x_pred[j]
		}
		dx_mat := transmute(matrix[N, 1]f64)dx
		dx_t := linalg.transpose(dx_mat)
		P_pred += Wc[i] * (dx_mat * dx_t)
	}

	return
}

// UKF update with control-dependent measurement: z = h(x, u)
ukf_update_control :: proc(
	x: [$N]f64,
	P: matrix[N, N]f64,
	z: [$M]f64,
	u: [$U]f64,
	h: proc(x: [N]f64, u: [U]f64) -> [M]f64,
	R: matrix[M, M]f64,
	params: UKF_Params,
	allocator: mem.Allocator,
) -> (
	x_upd: [N]f64,
	P_upd: matrix[N, N]f64,
) {
	// sigma points from prior
	Xi, Wm, Wc := ukf_sigma_points(x, P, params, allocator)
	L := len(Xi)

	// propagate to measurement space: h(x,u)
	Zi := make([]([M]f64), L, allocator)
	for i in 0 ..< L {
		Zi[i] = h(Xi[i], u)
	}

	// predicted measurement mean
	z_pred := [M]f64{}
	for j in 0 ..< M {
		sum := 0.0
		for i in 0 ..< L {
			sum += Wm[i] * Zi[i][j]
		}
		z_pred[j] = sum
	}

	// innovation covariance S and cross-covariance Pxz
	S := R
	Pxz := matrix[N, M]f64{} // zero

	for i in 0 ..< L {
		dx := [N]f64{}
		for j in 0 ..< N {
			dx[j] = Xi[i][j] - x[j]
		}
		dx_mat := transmute(matrix[N, 1]f64)dx

		dz := [M]f64{}
		for j in 0 ..< M {
			dz[j] = Zi[i][j] - z_pred[j]
		}
		dz_mat := transmute(matrix[M, 1]f64)dz
		dz_t := linalg.transpose(dz_mat)

		S += Wc[i] * (dz_mat * dz_t)
		Pxz += Wc[i] * (dx_mat * dz_t)
	}

	// Kalman gain
	S_inv := linalg.inverse(S)
	K := Pxz * S_inv

	// update state
	z_mat := transmute(matrix[M, 1]f64)z
	zp_mat := transmute(matrix[M, 1]f64)z_pred
	y_mat := z_mat - zp_mat
	x_mat := transmute(matrix[N, 1]f64)x
	x_new := x_mat + K * y_mat
	x_upd = transmute([N]f64)x_new

	// update covariance
	K_t := linalg.transpose(K)
	P_upd = P - K * S * K_t

	return
}

ukf_rts_smooth_control :: proc(
	Xi_pred: []([]([$N]f64)), // sigma points BEFORE update (prediction, with control)
	Xi_filt: []([]([N]f64)), // sigma points AFTER update (filtered)
	Wc_seq: []([]f64), // covariance weights per step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states (from ukf_predict_control)
) -> []KalmanState(N) {

	T := len(xf)
	assert(T == len(xp))
	assert(T == len(Xi_pred))
	assert(T == len(Xi_filt))
	assert(T == len(Wc_seq))

	smoothed := make([]KalmanState(N), T)
	if T == 0 {
		return smoothed
	}

	// last smoothed = last filtered
	smoothed[T - 1] = xf[T - 1]

	for k := T - 2; k >= 0; k -= 1 {
		Xi_p := Xi_pred[k]
		Xi_f := Xi_filt[k]
		Wc := Wc_seq[k]

		x_pred := xp[k].x
		P_pred := xp[k].P

		x_filt := xf[k].x
		P_filt := xf[k].P

		Pxf := matrix[N, N]f64{}

		L := len(Xi_p)
		for i in 0 ..< L {
			dx_p := [N]f64{}
			dx_f := [N]f64{}

			for j in 0 ..< N {
				dx_p[j] = Xi_p[i][j] - x_pred[j]
				dx_f[j] = Xi_f[i][j] - x_filt[j]
			}

			dxp_mat := transmute(matrix[N, 1]f64)dx_p
			dxf_t := linalg.transpose(transmute(matrix[N, 1]f64)dx_f)

			Pxf += Wc[i] * (dxp_mat * dxf_t)
		}

		// smoother gain
		P_filt_inv := linalg.inverse(P_filt)
		Gk := Pxf * P_filt_inv

		// smoothed state
		diff_x := smoothed[k + 1].x - x_filt
		dx_mat := transmute(matrix[N, 1]f64)diff_x
		corr := Gk * dx_mat
		smoothed[k].x = x_pred + transmute([N]f64)corr

		// smoothed covariance
		diff_P := smoothed[k + 1].P - P_filt
		smoothed[k].P = P_pred + Gk * diff_P * linalg.transpose(Gk)
	}

	return smoothed
}
kalman_loglik_scalar :: proc(
	y: []f64, // observations
	F, Q, P0: []f64, // N×N, flat, row-major: [row*N + col]
	H: []f64, // 1×N, flat
	R: []f64, // 1×1
	x0: []f64, // N
	N: int,
) -> f64 {
	T := len(y)
	if T == 0 {
		return 0.0
	}

	// State mean and covariance
	x := make([]f64, N)
	P := make([]f64, N * N)
	defer delete(x)
	defer delete(P)
	// init
	for i in 0 ..< N {
		x[i] = x0[i]
	}
	for i in 0 ..< N * N {
		P[i] = P0[i]
	}

	loglik := 0.0
	two_pi := 2.0 * math.PI

	for t in 0 ..< T {
		// --- Predict ---
		// x_pred = F * x
		x_pred := make([]f64, N)
		defer delete(x_pred)
		for i in 0 ..< N {
			sum := 0.0
			for j in 0 ..< N {
				sum += F[i * N + j] * x[j]
			}
			x_pred[i] = sum
		}

		// P_pred = F P Fᵀ + Q
		P_pred := make([]f64, N * N)
		defer delete(P_pred)
		// temp = F * P
		temp := make([]f64, N * N)
		defer delete(temp)
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {
					s += F[i * N + k] * P[k * N + j]
				}
				temp[i * N + j] = s
			}
		}
		// P_pred = temp * Fᵀ + Q
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {
					s += temp[i * N + k] * F[j * N + k] // Fᵀ
				}
				P_pred[i * N + j] = s + Q[i * N + j]
			}
		}

		// --- Innovation ---
		// y_pred = H * x_pred (scalar)
		y_pred := 0.0
		for j in 0 ..< N {
			y_pred += H[j] * x_pred[j]
		}

		v := y[t] - y_pred

		// S = H P_pred Hᵀ + R (scalar)
		S := 0.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				S += H[i] * P_pred[i * N + j] * H[j]
			}
		}
		S += R[0]

		// log-likelihood contribution
		loglik += -0.5 * (math.ln(two_pi) + math.ln(S) + (v * v) / S)

		// --- Update ---
		// K = P_pred Hᵀ / S  (N×1)
		K := make([]f64, N)
		defer delete(K)
		for i in 0 ..< N {
			s := 0.0
			for j in 0 ..< N {
				s += P_pred[i * N + j] * H[j]
			}
			K[i] = s / S
		}

		// x = x_pred + K * v
		for i in 0 ..< N {
			x[i] = x_pred[i] + K[i] * v
		}

		// P = P_pred - K S Kᵀ
		for i in 0 ..< N {
			for j in 0 ..< N {
				P[i * N + j] = P_pred[i * N + j] - K[i] * S * K[j]
			}
		}
	}

	return loglik
}
