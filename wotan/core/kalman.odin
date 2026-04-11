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
