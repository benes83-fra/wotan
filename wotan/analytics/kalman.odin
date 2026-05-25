package analytics

import l "../linalg"
import "base:intrinsics"
import "core:math"
import linalg "core:math/linalg"
import "core:mem"


// ============================================================================
// Dispatch helper: choose between built-in and optimized linalg based on size
// ============================================================================
_kalman_use_optimized :: proc(N: int) -> bool {
	// Built-in Odin matrix ops are optimized for N ≤ 8
	// Use wotan_linalg for larger matrices where blocking helps
	return N > 8
}

// ============================================================================
// Helper: Convert flat row-major []f64 → l.Matrix(f64)
// ============================================================================
_matrix_from_flat :: proc(
	data: []f64,
	rows, cols: int,
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	m := l.matrix_new(f64, rows, cols, allocator)
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			m.data[i * cols + j] = data[i * cols + j]
		}
	}
	return m
}

// ============================================================================
// Helper: Convert l.Matrix(f64) → flat row-major []f64
// ============================================================================
_matrix_to_flat :: proc(m: ^l.Matrix(f64), allocator: mem.Allocator = context.allocator) -> []f64 {
	rows, cols := m.rows, m.cols
	out := make([]f64, rows * cols, allocator)
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			out[i * cols + j] = m.data[i * cols + j]
		}
	}
	return out
}
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

// ============================================================================
// Optimized predict for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for F * x and F * P * Fᵀ
// ============================================================================
kalman_predict_large :: proc(kf: ^KalmanFilter($N, $M)) {
	// Convert fixed-size arrays to flat arrays, then to slices for dynamic linalg
	// [N]f64 -> [N]f64 (identity) -> slice
	x_array := kf.x // Already [N]f64
	x_slice := x_array[:]

	// matrix[N,N]f64 -> [N*N]f64 -> slice
	F_array := transmute([N * N]f64)kf.F
	F_slice := F_array[:]

	P_array := transmute([N * N]f64)kf.P
	P_slice := P_array[:]

	Q_array := transmute([N * N]f64)kf.Q
	Q_slice := Q_array[:]

	// Convert to dynamic matrices for optimized linalg
	x_mat := _matrix_from_flat(x_slice, N, 1, context.temp_allocator)
	defer l.matrix_free(&x_mat)

	F_mat := _matrix_from_flat(F_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&F_mat)

	P_mat := _matrix_from_flat(P_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&P_mat)

	Q_mat := _matrix_from_flat(Q_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&Q_mat)

	// x_pred = F * x
	x_pred_vec := l.matvec_dyn_simd(&F_mat, x_slice, context.temp_allocator)
	defer delete(x_pred_vec, context.temp_allocator)

	// P_pred = F * P * Fᵀ + Q
	// Since matmul_dyn_simd computes A * Bᵀ:
	// Step 1: FP = F * P (P is symmetric, so Pᵀ = P)
	FP := l.matmul_dyn_simd(&F_mat, &P_mat, context.temp_allocator)
	defer l.matrix_free(&FP)

	// Step 2: P_pred = FP * Fᵀ (matmul_dyn_simd computes FP * Fᵀ directly)
	P_pred := l.matmul_dyn_simd(&FP, &F_mat, context.temp_allocator)
	defer l.matrix_free(&P_pred)

	// Add Q
	for i in 0 ..< N * N {P_pred.data[i] += Q_mat.data[i]}

	// Convert results back to fixed-size arrays
	for i in 0 ..< N {kf.x[i] = x_pred_vec[i]}
	for i in 0 ..< N * N {kf.P[i / N, i % N] = P_pred.data[i]} 	// row-major to [N,N] layout
}

kalman_predict :: proc(kf: ^KalmanFilter($N, $M)) {
	if _kalman_use_optimized(N) {
		kalman_predict_large(kf)
		return
	}

	// Built-in Odin ops for small N (≤8)
	x_mat := transmute(matrix[N, 1]f64)kf.x
	new_x := kf.F * x_mat
	kf.x = transmute([N]f64)new_x

	kf.P = (kf.F * kf.P * linalg.transpose(kf.F)) + kf.Q
}
// ============================================================================
// Optimized predict with control for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for F*x, B*u, and F*P*Fᵀ
// ============================================================================
kalman_predict_control_large :: proc(kf: ^KalmanFilter($N, $M), B: matrix[N, $U]f64, u: [U]f64) {
	// Convert fixed-size arrays to slices for dynamic linalg
	x_array := kf.x
	x_slice := x_array[:]

	F_array := transmute([N * N]f64)kf.F
	F_slice := F_array[:]

	P_array := transmute([N * N]f64)kf.P
	P_slice := P_array[:]

	Q_array := transmute([N * N]f64)kf.Q
	Q_slice := Q_array[:]

	B_array := transmute([N * U]f64)B
	B_slice := B_array[:]

	u_copy := u
	u_slice := u_copy[:]

	// Convert to dynamic matrices for optimized linalg
	x_mat := _matrix_from_flat(x_slice, N, 1, context.temp_allocator)
	defer l.matrix_free(&x_mat)

	F_mat := _matrix_from_flat(F_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&F_mat)

	P_mat := _matrix_from_flat(P_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&P_mat)

	Q_mat := _matrix_from_flat(Q_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&Q_mat)

	B_mat := _matrix_from_flat(B_slice, N, U, context.temp_allocator)
	defer l.matrix_free(&B_mat)

	// x' = F*x + B*u
	Fx := l.matmul_dyn_simd(&F_mat, &x_mat, context.temp_allocator)
	defer l.matrix_free(&Fx)

	u_mat_temp := _matrix_from_flat(u_slice, U, 1, context.temp_allocator)
	defer l.matrix_free(&u_mat_temp)

	Bu := l.matmul_dyn_simd(&B_mat, &u_mat_temp, context.temp_allocator)
	defer l.matrix_free(&Bu)

	// Add Fx + Bu
	for i in 0 ..< N {x_slice[i] = Fx.data[i] + Bu.data[i]}

	// P' = F*P*Fᵀ + Q
	// Since matmul_dyn_simd computes A * Bᵀ:
	// Step 1: FP = F * P (P is symmetric, so Pᵀ = P)
	FP := l.matmul_dyn_simd(&F_mat, &P_mat, context.temp_allocator)
	defer l.matrix_free(&FP)

	// Step 2: P_pred = FP * Fᵀ (matmul_dyn_simd computes FP * Fᵀ directly)
	P_pred := l.matmul_dyn_simd(&FP, &F_mat, context.temp_allocator)
	defer l.matrix_free(&P_pred)

	// Add Q
	for i in 0 ..< N * N {P_pred.data[i] += Q_mat.data[i]}

	// Convert results back to fixed-size arrays
	for i in 0 ..< N {kf.P[i / N, i % N] = P_pred.data[i]}
}

kalman_predict_control :: proc(kf: ^KalmanFilter($N, $M), B: matrix[N, $U]f64, u: [U]f64) {
	if _kalman_use_optimized(N) {
		kalman_predict_control_large(kf, B, u)
		return
	}

	// Built-in Odin ops for small N (≤8)
	// x and u as column vectors
	x_mat := transmute(matrix[N, 1]f64)kf.x
	u_mat := transmute(matrix[U, 1]f64)u

	// x' = F x + B u
	new_x := kf.F * x_mat + B * u_mat
	kf.x = transmute([N]f64)new_x

	// P' = F P Fᵀ + Q   (control assumed deterministic)
	kf.P = (kf.F * kf.P * linalg.transpose(kf.F)) + kf.Q
}
// ============================================================================
// Optimized update for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for H*P*Hᵀ, P*Hᵀ, and matrix updates
// ============================================================================
kalman_update_large :: proc(kf: ^KalmanFilter($N, $M), z: [M]f64) {
	// Convert fixed-size arrays to slices for dynamic linalg
	x_array := kf.x
	x_slice := x_array[:]

	P_array := transmute([N * N]f64)kf.P
	P_slice := P_array[:]

	H_array := transmute([M * N]f64)kf.H
	H_slice := H_array[:]

	R_array := transmute([M * M]f64)kf.R
	R_slice := R_array[:]

	// z is [M]f64 parameter — copy to slice
	z_copy := z // Already [M]f64
	z_slice := z_copy[:]

	// Convert to dynamic matrices for optimized linalg
	x_mat := _matrix_from_flat(x_slice, N, 1, context.temp_allocator)
	defer l.matrix_free(&x_mat)

	P_mat := _matrix_from_flat(P_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&P_mat)

	H_mat := _matrix_from_flat(H_slice, M, N, context.temp_allocator)
	defer l.matrix_free(&H_mat)

	R_mat := _matrix_from_flat(R_slice, M, M, context.temp_allocator)
	defer l.matrix_free(&R_mat)

	// 1. Innovation: y = z - H*x
	Hx := l.matmul_dyn_simd(&H_mat, &x_mat, context.temp_allocator)
	defer l.matrix_free(&Hx)

	y := make([]f64, M, context.temp_allocator)
	for i in 0 ..< M {y[i] = z_slice[i] - Hx.data[i]}
	defer delete(y, context.temp_allocator)

	// 2. S = H*P*Hᵀ + R
	// Since matmul_dyn_simd computes A * Bᵀ:
	// Step 1: HP = H * P (P is symmetric, so Pᵀ = P)
	HP := l.matmul_dyn_simd(&H_mat, &P_mat, context.temp_allocator)
	defer l.matrix_free(&HP)

	// Step 2: S = HP * Hᵀ + R (matmul_dyn_simd computes HP * Hᵀ directly)
	S := l.matmul_dyn_simd(&HP, &H_mat, context.temp_allocator)
	defer l.matrix_free(&S)

	// Add R
	for i in 0 ..< M * M {S.data[i] += R_mat.data[i]}

	// 3. K = P*Hᵀ * inv(S)
	// Since matmul_dyn_simd computes A * Bᵀ:
	// P*Hᵀ = matmul_dyn_simd(&P_mat, &H_mat) (because it computes P * Hᵀ)
	PHt := l.matmul_dyn_simd(&P_mat, &H_mat, context.temp_allocator)
	defer l.matrix_free(&PHt)

	// Solve K * S = PHt for K (K = PHt * S⁻¹)
	// Use LU decomposition since we don't have cholesky_solve
	K := l.matrix_new(f64, N, M, context.temp_allocator)
	defer l.matrix_free(&K)

	// Solve column by column: S * k_col = ph_col
	for j in 0 ..< M {
		// Extract column j of PHt
		ph_col := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {ph_col[i] = PHt.data[i * M + j]}

		// Solve S * k_col = ph_col using LU decomposition
		// First, make a copy of S for LU decomposition
		S_lu := l.matrix_new(f64, M, M, context.temp_allocator)
		copy(S_lu.data, S.data)
		defer l.matrix_free(&S_lu)

		// LU decompose S_lu
		_, piv, _, ok := l.lu_decompose(&S_lu, context.temp_allocator)
		if !ok {
			// Fallback: use naive solve (shouldn't happen for SPD S)
			for i in 0 ..< N {K.data[i * M + j] = 0.0}
			delete(ph_col, context.temp_allocator)
			continue
		}

		// Solve for k_col
		k_col := l.lu_solve_simd(&S_lu, piv, ph_col, context.temp_allocator)
		defer delete(k_col, context.temp_allocator)

		// Store in K
		for i in 0 ..< N {K.data[i * M + j] = k_col[i]}
		delete(ph_col, context.temp_allocator)
	}

	// 4. Update state: x = x + K*y
	y_mat := _matrix_from_flat(y, M, 1, context.temp_allocator)
	defer l.matrix_free(&y_mat)

	Ky := l.matmul_dyn_simd(&K, &y_mat, context.temp_allocator)
	defer l.matrix_free(&Ky)

	for i in 0 ..< N {x_slice[i] += Ky.data[i]}

	// 5. Update covariance: P = (I - K*H) * P
	// Compute KH = K * H
	KH := l.matmul_dyn_simd(&K, &H_mat, context.temp_allocator)
	defer l.matrix_free(&KH)

	// Compute I - KH
	IKH := l.matrix_new(f64, N, N, context.temp_allocator)
	defer l.matrix_free(&IKH)
	for i in 0 ..< N * N {
		IKH.data[i] = -KH.data[i]
		if i / N == i % N {IKH.data[i] += 1.0} 	// Add identity
	}

	// P_new = (I - KH) * P
	P_new := l.matmul_dyn_simd(&IKH, &P_mat, context.temp_allocator)
	defer l.matrix_free(&P_new)

	// Convert results back to fixed-size arrays
	for i in 0 ..< N {kf.x[i] = x_slice[i]}
	for i in 0 ..< N * N {kf.P[i / N, i % N] = P_new.data[i]}
}

kalman_update :: proc(kf: ^KalmanFilter($N, $M), z: [M]f64) {
	if _kalman_use_optimized(N) {
		kalman_update_large(kf, z)
		return
	}

	// Built-in Odin ops for small N (≤8)
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

// ============================================================================
// Optimized RTS smoother for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for matrix inverses and multiplications
// ============================================================================
rts_smooth_large :: proc(
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

	// Convert F to dynamic matrix for optimized linalg
	F_array := transmute([N * N]f64)F
	F_slice := F_array[:]
	F_mat := _matrix_from_flat(F_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&F_mat)

	// Precompute Fᵀ once
	Ft := l.matrix_transpose(&F_mat, context.temp_allocator)
	defer l.matrix_free(&Ft)

	for k := T - 2; k >= 0; k -= 1 {
		// Convert Pf_k and Pp_k1 to dynamic matrices
		Pf_array := transmute([N * N]f64)xf[k].P
		Pf_slice := Pf_array[:]
		Pf_mat := _matrix_from_flat(Pf_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pf_mat)

		Pp_array := transmute([N * N]f64)xp[k + 1].P
		Pp_slice := Pp_array[:]
		Pp_mat := _matrix_from_flat(Pp_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_mat)

		// Pp_inv = inv(Pp_k1) using LU decomposition
		Pp_inv := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_inv)

		// Make a copy for LU decomposition
		Pp_lu := l.matrix_new(f64, N, N, context.temp_allocator)
		copy(Pp_lu.data, Pp_mat.data)
		defer l.matrix_free(&Pp_lu)

		_, piv, _, ok := l.lu_decompose(&Pp_lu, context.temp_allocator)
		if ok {
			// Solve Pp_inv * Pp = I for Pp_inv (column by column)
			for j in 0 ..< N {
				e_col := make([]f64, N, context.temp_allocator)
				for i in 0 ..< N {e_col[i] = 0.0}
				e_col[j] = 1.0

				col := l.lu_solve_simd(&Pp_lu, piv, e_col, context.temp_allocator)
				defer delete(col, context.temp_allocator)

				for i in 0 ..< N {Pp_inv.data[i * N + j] = col[i]}
				delete(e_col, context.temp_allocator)
			}
		} else {
			// Fallback: use identity (shouldn't happen for valid covariance)
			for i in 0 ..< N * N {Pp_inv.data[i] = 0.0}
			for i in 0 ..< N {Pp_inv.data[i * N + i] = 1.0}
		}

		// Ck = Pf_k * Fᵀ * Pp_inv
		// Step 1: PfFt = Pf_k * Fᵀ (matmul_dyn_simd computes A * Bᵀ)
		PfFt := l.matmul_dyn_simd(&Pf_mat, &F_mat, context.temp_allocator)
		defer l.matrix_free(&PfFt)

		// Step 2: Ck = PfFt * Pp_inv
		Ck := l.matmul_dyn_simd(&PfFt, &Pp_inv, context.temp_allocator)
		defer l.matrix_free(&Ck)

		// diff_x = smoothed[k+1].x - xp[k+1].x
		diff_x := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {diff_x[i] = smoothed[k + 1].x[i] - xp[k + 1].x[i]}

		// corr_x = Ck * diff_x
		diff_x_mat := _matrix_from_flat(diff_x, N, 1, context.temp_allocator)
		defer l.matrix_free(&diff_x_mat)

		corr_x_mat := l.matmul_dyn_simd(&Ck, &diff_x_mat, context.temp_allocator)
		defer l.matrix_free(&corr_x_mat)

		// smoothed[k].x = xf[k].x + corr_x
		for i in 0 ..< N {smoothed[k].x[i] = xf[k].x[i] + corr_x_mat.data[i]}
		delete(diff_x, context.temp_allocator)

		// diff_P = smoothed[k+1].P - Pp_k1
		diff_P := make([]f64, N * N, context.temp_allocator)

		// Convert smoothed[k+1].P to flat slice first
		smoothed_P_array := transmute([N * N]f64)smoothed[k + 1].P
		smoothed_P_slice := smoothed_P_array[:]

		for i in 0 ..< N * N {
			diff_P[i] = smoothed_P_slice[i] - Pp_slice[i]
		}

		// Ck_t = Ckᵀ
		Ck_t := l.matrix_transpose(&Ck, context.temp_allocator)
		defer l.matrix_free(&Ck_t)

		// temp = Ck * diff_P
		diff_P_mat := _matrix_from_flat(diff_P, N, N, context.temp_allocator)
		defer l.matrix_free(&diff_P_mat)

		temp := l.matmul_dyn_simd(&Ck, &diff_P_mat, context.temp_allocator)
		defer l.matrix_free(&temp)

		// P_new = Pf_k + temp * Ck_t
		temp_Ckt := l.matmul_dyn_simd(&temp, &Ck_t, context.temp_allocator)
		defer l.matrix_free(&temp_Ckt)

		P_new := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&P_new)
		for i in 0 ..< N * N {P_new.data[i] = Pf_slice[i] + temp_Ckt.data[i]}

		// Convert result back to fixed-size array
		for i in 0 ..< N * N {smoothed[k].P[i / N, i % N] = P_new.data[i]}
		delete(diff_P, context.temp_allocator)
	}

	return smoothed
}


rts_smooth :: proc(
	F: matrix[$N, N]f64,
	xf: []KalmanState(N), // filtered
	xp: []KalmanState(N), // predicted
) -> []KalmanState(N) {
	if _kalman_use_optimized(N) {
		return rts_smooth_large(F, xf, xp)
	}

	// Built-in Odin ops for small N (≤8)
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

// ============================================================================
// Optimized RTS smoother with control for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for matrix inverses and multiplications
// ============================================================================
rts_smooth_control_large :: proc(
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

	// Convert F and B to dynamic matrices for optimized linalg
	F_array := transmute([N * N]f64)F
	F_slice := F_array[:]
	F_mat := _matrix_from_flat(F_slice, N, N, context.temp_allocator)
	defer l.matrix_free(&F_mat)

	B_array := transmute([N * U]f64)B
	B_slice := B_array[:]
	B_mat := _matrix_from_flat(B_slice, N, U, context.temp_allocator)
	defer l.matrix_free(&B_mat)

	// Precompute Fᵀ once
	Ft := l.matrix_transpose(&F_mat, context.temp_allocator)
	defer l.matrix_free(&Ft)

	for k := T - 2; k >= 0; k -= 1 {
		// Convert Pf_k and Pp_k1 to dynamic matrices
		Pf_array := transmute([N * N]f64)xf[k].P
		Pf_slice := Pf_array[:]
		Pf_mat := _matrix_from_flat(Pf_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pf_mat)

		Pp_array := transmute([N * N]f64)xp[k + 1].P
		Pp_slice := Pp_array[:]
		Pp_mat := _matrix_from_flat(Pp_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_mat)

		// Pp_inv = inv(Pp_k1) using LU decomposition
		Pp_inv := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_inv)

		// Make a copy for LU decomposition
		Pp_lu := l.matrix_new(f64, N, N, context.temp_allocator)
		copy(Pp_lu.data, Pp_mat.data)
		defer l.matrix_free(&Pp_lu)

		_, piv, _, ok := l.lu_decompose(&Pp_lu, context.temp_allocator)
		if ok {
			// Solve Pp_inv * Pp = I for Pp_inv (column by column)
			for j in 0 ..< N {
				e_col := make([]f64, N, context.temp_allocator)
				for i in 0 ..< N {e_col[i] = 0.0}
				e_col[j] = 1.0

				col := l.lu_solve_simd(&Pp_lu, piv, e_col, context.temp_allocator)
				defer delete(col, context.temp_allocator)

				for i in 0 ..< N {Pp_inv.data[i * N + j] = col[i]}
				delete(e_col, context.temp_allocator)
			}
		} else {
			// Fallback: use identity (shouldn't happen for valid covariance)
			for i in 0 ..< N * N {Pp_inv.data[i] = 0.0}
			for i in 0 ..< N {Pp_inv.data[i * N + i] = 1.0}
		}

		// Ck = Pf_k * Fᵀ * Pp_inv
		// Step 1: PfFt = Pf_k * Fᵀ (matmul_dyn_simd computes A * Bᵀ)
		PfFt := l.matmul_dyn_simd(&Pf_mat, &F_mat, context.temp_allocator)
		defer l.matrix_free(&PfFt)

		// Step 2: Ck = PfFt * Pp_inv
		Ck := l.matmul_dyn_simd(&PfFt, &Pp_inv, context.temp_allocator)
		defer l.matrix_free(&Ck)

		// Compute controlled prediction: x_pred = F * xf[k] + B * u[k]
		xf_k_array := transmute([N]f64)xf[k].x
		xf_k_slice := xf_k_array[:]
		xf_k_mat := _matrix_from_flat(xf_k_slice, N, 1, context.temp_allocator)
		defer l.matrix_free(&xf_k_mat)

		u_k_copy := u[k]
		u_k_slice := u_k_copy[:]
		u_k_mat := _matrix_from_flat(u_k_slice, U, 1, context.temp_allocator)
		defer l.matrix_free(&u_k_mat)

		Fxf := l.matmul_dyn_simd(&F_mat, &xf_k_mat, context.temp_allocator)
		defer l.matrix_free(&Fxf)

		Bu := l.matmul_dyn_simd(&B_mat, &u_k_mat, context.temp_allocator)
		defer l.matrix_free(&Bu)

		x_pred := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {x_pred[i] = Fxf.data[i] + Bu.data[i]}
		defer delete(x_pred, context.temp_allocator)

		// x_s[k] = x_f[k] + Ck * (x_s[k+1] - x_pred)
		diff_x := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {diff_x[i] = smoothed[k + 1].x[i] - x_pred[i]}

		diff_x_mat := _matrix_from_flat(diff_x, N, 1, context.temp_allocator)
		defer l.matrix_free(&diff_x_mat)

		corr_x_mat := l.matmul_dyn_simd(&Ck, &diff_x_mat, context.temp_allocator)
		defer l.matrix_free(&corr_x_mat)

		for i in 0 ..< N {smoothed[k].x[i] = xf[k].x[i] + corr_x_mat.data[i]}
		delete(diff_x, context.temp_allocator)

		// P_s[k] = P_f[k] + Ck * (P_s[k+1] - P_p[k+1]) * Ckᵀ
		// diff_P = smoothed[k+1].P - Pp_k1
		diff_P := make([]f64, N * N, context.temp_allocator)

		// Convert smoothed[k+1].P to flat slice first
		smoothed_P_array := transmute([N * N]f64)smoothed[k + 1].P
		smoothed_P_slice := smoothed_P_array[:]

		for i in 0 ..< N * N {
			diff_P[i] = smoothed_P_slice[i] - Pp_slice[i]
		}

		// Ck_t = Ckᵀ
		Ck_t := l.matrix_transpose(&Ck, context.temp_allocator)
		defer l.matrix_free(&Ck_t)

		// temp = Ck * diff_P
		diff_P_mat := _matrix_from_flat(diff_P, N, N, context.temp_allocator)
		defer l.matrix_free(&diff_P_mat)

		temp := l.matmul_dyn_simd(&Ck, &diff_P_mat, context.temp_allocator)
		defer l.matrix_free(&temp)

		// P_new = Pf_k + temp * Ck_t
		temp_Ckt := l.matmul_dyn_simd(&temp, &Ck_t, context.temp_allocator)
		defer l.matrix_free(&temp_Ckt)

		P_new := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&P_new)
		for i in 0 ..< N * N {P_new.data[i] = Pf_slice[i] + temp_Ckt.data[i]}

		// Convert result back to fixed-size array
		for i in 0 ..< N * N {smoothed[k].P[i / N, i % N] = P_new.data[i]}
		delete(diff_P, context.temp_allocator)
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
	if _kalman_use_optimized(N) {
		return rts_smooth_control_large(F, B, u, xf, xp)
	}

	// Built-in Odin ops for small N (≤8)
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
// ============================================================================
// Optimized RTS smoother with time-varying control for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for matrix inverses and multiplications
// ============================================================================
rts_smooth_tv_control_large :: proc(
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
		// Convert F_seq[k] and B_seq[k] to dynamic matrices for optimized linalg
		F_array := transmute([N * N]f64)F_seq[k]
		F_slice := F_array[:]
		F_mat := _matrix_from_flat(F_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&F_mat)

		B_array := transmute([N * U]f64)B_seq[k]
		B_slice := B_array[:]
		B_mat := _matrix_from_flat(B_slice, N, U, context.temp_allocator)
		defer l.matrix_free(&B_mat)

		// Precompute Fᵀ once per iteration
		Ft := l.matrix_transpose(&F_mat, context.temp_allocator)
		defer l.matrix_free(&Ft)

		// Convert Pf_k and Pp_k1 to dynamic matrices
		Pf_array := transmute([N * N]f64)xf[k].P
		Pf_slice := Pf_array[:]
		Pf_mat := _matrix_from_flat(Pf_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pf_mat)

		Pp_array := transmute([N * N]f64)xp[k + 1].P
		Pp_slice := Pp_array[:]
		Pp_mat := _matrix_from_flat(Pp_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_mat)

		// Pp_inv = inv(Pp_k1) using LU decomposition
		Pp_inv := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_inv)

		// Make a copy for LU decomposition
		Pp_lu := l.matrix_new(f64, N, N, context.temp_allocator)
		copy(Pp_lu.data, Pp_mat.data)
		defer l.matrix_free(&Pp_lu)

		_, piv, _, ok := l.lu_decompose(&Pp_lu, context.temp_allocator)
		if ok {
			// Solve Pp_inv * Pp = I for Pp_inv (column by column)
			for j in 0 ..< N {
				e_col := make([]f64, N, context.temp_allocator)
				for i in 0 ..< N {e_col[i] = 0.0}
				e_col[j] = 1.0

				col := l.lu_solve_simd(&Pp_lu, piv, e_col, context.temp_allocator)
				defer delete(col, context.temp_allocator)

				for i in 0 ..< N {Pp_inv.data[i * N + j] = col[i]}
				delete(e_col, context.temp_allocator)
			}
		} else {
			// Fallback: use identity (shouldn't happen for valid covariance)
			for i in 0 ..< N * N {Pp_inv.data[i] = 0.0}
			for i in 0 ..< N {Pp_inv.data[i * N + i] = 1.0}
		}

		// Ck = Pf_k * Fᵀ * Pp_inv
		// Step 1: PfFt = Pf_k * Fᵀ (matmul_dyn_simd computes A * Bᵀ)
		PfFt := l.matmul_dyn_simd(&Pf_mat, &F_mat, context.temp_allocator)
		defer l.matrix_free(&PfFt)

		// Step 2: Ck = PfFt * Pp_inv
		Ck := l.matmul_dyn_simd(&PfFt, &Pp_inv, context.temp_allocator)
		defer l.matrix_free(&Ck)

		// Compute controlled prediction: x_pred = F * xf[k] + B * u[k]
		xf_k_array := transmute([N]f64)xf[k].x
		xf_k_slice := xf_k_array[:]
		xf_k_mat := _matrix_from_flat(xf_k_slice, N, 1, context.temp_allocator)
		defer l.matrix_free(&xf_k_mat)

		u_k_copy := u[k]
		u_k_slice := u_k_copy[:]
		u_k_mat := _matrix_from_flat(u_k_slice, U, 1, context.temp_allocator)
		defer l.matrix_free(&u_k_mat)

		Fxf := l.matmul_dyn_simd(&F_mat, &xf_k_mat, context.temp_allocator)
		defer l.matrix_free(&Fxf)

		Bu := l.matmul_dyn_simd(&B_mat, &u_k_mat, context.temp_allocator)
		defer l.matrix_free(&Bu)

		x_pred := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {x_pred[i] = Fxf.data[i] + Bu.data[i]}
		defer delete(x_pred, context.temp_allocator)

		// x_s[k] = x_f[k] + Ck * (x_s[k+1] - x_pred)
		diff_x := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {diff_x[i] = smoothed[k + 1].x[i] - x_pred[i]}

		diff_x_mat := _matrix_from_flat(diff_x, N, 1, context.temp_allocator)
		defer l.matrix_free(&diff_x_mat)

		corr_x_mat := l.matmul_dyn_simd(&Ck, &diff_x_mat, context.temp_allocator)
		defer l.matrix_free(&corr_x_mat)

		for i in 0 ..< N {smoothed[k].x[i] = xf[k].x[i] + corr_x_mat.data[i]}
		delete(diff_x, context.temp_allocator)

		// P_s[k] = P_f[k] + Ck * (P_s[k+1] - P_p[k+1]) * Ckᵀ
		// diff_P = smoothed[k+1].P - Pp_k1
		diff_P := make([]f64, N * N, context.temp_allocator)

		// Convert smoothed[k+1].P to flat slice first
		smoothed_P_array := transmute([N * N]f64)smoothed[k + 1].P
		smoothed_P_slice := smoothed_P_array[:]

		for i in 0 ..< N * N {
			diff_P[i] = smoothed_P_slice[i] - Pp_slice[i]
		}

		// Ck_t = Ckᵀ
		Ck_t := l.matrix_transpose(&Ck, context.temp_allocator)
		defer l.matrix_free(&Ck_t)

		// temp = Ck * diff_P
		diff_P_mat := _matrix_from_flat(diff_P, N, N, context.temp_allocator)
		defer l.matrix_free(&diff_P_mat)

		temp := l.matmul_dyn_simd(&Ck, &diff_P_mat, context.temp_allocator)
		defer l.matrix_free(&temp)

		// P_new = Pf_k + temp * Ck_t
		temp_Ckt := l.matmul_dyn_simd(&temp, &Ck_t, context.temp_allocator)
		defer l.matrix_free(&temp_Ckt)

		P_new := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&P_new)
		for i in 0 ..< N * N {P_new.data[i] = Pf_slice[i] + temp_Ckt.data[i]}

		// Convert result back to fixed-size array
		for i in 0 ..< N * N {smoothed[k].P[i / N, i % N] = P_new.data[i]}
		delete(diff_P, context.temp_allocator)
	}

	return smoothed
}
rts_smooth_tv_control :: proc(
	F_seq: []matrix[$N, N]f64,
	B_seq: []matrix[N, $U]f64,
	u: []([U]f64), // control inputs per step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states (with control)
) -> []KalmanState(N) {
	if _kalman_use_optimized(N) {
		return rts_smooth_tv_control_large(F_seq, B_seq, u, xf, xp)
	}

	// Built-in Odin ops for small N (≤8)
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
// ============================================================================
// Optimized EKF RTS smoother for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for matrix inverses and multiplications
// ============================================================================
ekf_rts_smooth_large :: proc(
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
		// Convert F_seq[k] to dynamic matrix for optimized linalg
		F_array := transmute([N * N]f64)F_seq[k]
		F_slice := F_array[:]
		F_mat := _matrix_from_flat(F_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&F_mat)

		// Precompute Fᵀ once per iteration
		Ft := l.matrix_transpose(&F_mat, context.temp_allocator)
		defer l.matrix_free(&Ft)

		// Convert Pf_k and Pp_k1 to dynamic matrices
		Pf_array := transmute([N * N]f64)xf[k].P
		Pf_slice := Pf_array[:]
		Pf_mat := _matrix_from_flat(Pf_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pf_mat)

		Pp_array := transmute([N * N]f64)xp[k + 1].P
		Pp_slice := Pp_array[:]
		Pp_mat := _matrix_from_flat(Pp_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_mat)

		// Pp_inv = inv(Pp_k1) using LU decomposition
		Pp_inv := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&Pp_inv)

		// Make a copy for LU decomposition
		Pp_lu := l.matrix_new(f64, N, N, context.temp_allocator)
		copy(Pp_lu.data, Pp_mat.data)
		defer l.matrix_free(&Pp_lu)

		_, piv, _, ok := l.lu_decompose(&Pp_lu, context.temp_allocator)
		if ok {
			// Solve Pp_inv * Pp = I for Pp_inv (column by column)
			for j in 0 ..< N {
				e_col := make([]f64, N, context.temp_allocator)
				for i in 0 ..< N {e_col[i] = 0.0}
				e_col[j] = 1.0

				col := l.lu_solve_simd(&Pp_lu, piv, e_col, context.temp_allocator)
				defer delete(col, context.temp_allocator)

				for i in 0 ..< N {Pp_inv.data[i * N + j] = col[i]}
				delete(e_col, context.temp_allocator)
			}
		} else {
			// Fallback: use identity (shouldn't happen for valid covariance)
			for i in 0 ..< N * N {Pp_inv.data[i] = 0.0}
			for i in 0 ..< N {Pp_inv.data[i * N + i] = 1.0}
		}

		// Ck = Pf_k * Fᵀ * Pp_inv
		// Step 1: PfFt = Pf_k * Fᵀ (matmul_dyn_simd computes A * Bᵀ)
		PfFt := l.matmul_dyn_simd(&Pf_mat, &F_mat, context.temp_allocator)
		defer l.matrix_free(&PfFt)

		// Step 2: Ck = PfFt * Pp_inv
		Ck := l.matmul_dyn_simd(&PfFt, &Pp_inv, context.temp_allocator)
		defer l.matrix_free(&Ck)

		// x_s[k] = x_f[k] + Ck * (x_s[k+1] - x_p[k+1])
		diff_x := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {diff_x[i] = smoothed[k + 1].x[i] - xp[k + 1].x[i]}

		diff_x_mat := _matrix_from_flat(diff_x, N, 1, context.temp_allocator)
		defer l.matrix_free(&diff_x_mat)

		corr_x_mat := l.matmul_dyn_simd(&Ck, &diff_x_mat, context.temp_allocator)
		defer l.matrix_free(&corr_x_mat)

		for i in 0 ..< N {smoothed[k].x[i] = xf[k].x[i] + corr_x_mat.data[i]}
		delete(diff_x, context.temp_allocator)

		// P_s[k] = P_f[k] + Ck * (P_s[k+1] - P_p[k+1]) * Ckᵀ
		// diff_P = smoothed[k+1].P - Pp_k1
		diff_P := make([]f64, N * N, context.temp_allocator)

		// Convert smoothed[k+1].P to flat slice first
		smoothed_P_array := transmute([N * N]f64)smoothed[k + 1].P
		smoothed_P_slice := smoothed_P_array[:]

		for i in 0 ..< N * N {
			diff_P[i] = smoothed_P_slice[i] - Pp_slice[i]
		}

		// Ck_t = Ckᵀ
		Ck_t := l.matrix_transpose(&Ck, context.temp_allocator)
		defer l.matrix_free(&Ck_t)

		// temp = Ck * diff_P
		diff_P_mat := _matrix_from_flat(diff_P, N, N, context.temp_allocator)
		defer l.matrix_free(&diff_P_mat)

		temp := l.matmul_dyn_simd(&Ck, &diff_P_mat, context.temp_allocator)
		defer l.matrix_free(&temp)

		// P_new = Pf_k + temp * Ck_t
		temp_Ckt := l.matmul_dyn_simd(&temp, &Ck_t, context.temp_allocator)
		defer l.matrix_free(&temp_Ckt)

		P_new := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&P_new)
		for i in 0 ..< N * N {P_new.data[i] = Pf_slice[i] + temp_Ckt.data[i]}

		// Convert result back to fixed-size array
		for i in 0 ..< N * N {smoothed[k].P[i / N, i % N] = P_new.data[i]}
		delete(diff_P, context.temp_allocator)
	}

	return smoothed
}

// EKF RTS smoother (time-varying, no explicit control)
// Uses the linearized F_k (Jacobians) from the EKF forward pass.
ekf_rts_smooth :: proc(
	F_seq: []matrix[$N, N]f64, // Jacobians F_k at each step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states
) -> []KalmanState(N) {
	if _kalman_use_optimized(N) {
		return ekf_rts_smooth_large(F_seq, xf, xp)
	}

	// Built-in Odin ops for small N (≤8)
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
// ============================================================================
// Optimized UKF RTS smoother for large state dimensions (N > 8)
// Uses wotan_linalg SIMD operations for matrix inverses and multiplications
// ============================================================================
ukf_rts_smooth_large :: proc(
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

			// Use SIMD dot product for outer product accumulation
			for ii in 0 ..< N {
				for jj in 0 ..< N {
					Pxf[ii, jj] += Wc[i] * dx_p[ii] * dx_f[jj]
				}
			}
		}

		// Convert Pxf and P_filt to dynamic matrices for optimized linalg
		Pxf_array := transmute([N * N]f64)Pxf
		Pxf_slice := Pxf_array[:]
		Pxf_mat := _matrix_from_flat(Pxf_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&Pxf_mat)

		P_filt_array := transmute([N * N]f64)P_filt
		P_filt_slice := P_filt_array[:]
		P_filt_mat := _matrix_from_flat(P_filt_slice, N, N, context.temp_allocator)
		defer l.matrix_free(&P_filt_mat)

		// P_filt_inv = inv(P_filt) using LU decomposition
		P_filt_inv := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&P_filt_inv)

		// Make a copy for LU decomposition
		P_filt_lu := l.matrix_new(f64, N, N, context.temp_allocator)
		copy(P_filt_lu.data, P_filt_mat.data)
		defer l.matrix_free(&P_filt_lu)

		_, piv, _, ok := l.lu_decompose(&P_filt_lu, context.temp_allocator)
		if ok {
			// Solve P_filt_inv * P_filt = I for P_filt_inv (column by column)
			for j in 0 ..< N {
				e_col := make([]f64, N, context.temp_allocator)
				for i in 0 ..< N {e_col[i] = 0.0}
				e_col[j] = 1.0

				col := l.lu_solve_simd(&P_filt_lu, piv, e_col, context.temp_allocator)
				defer delete(col, context.temp_allocator)

				for i in 0 ..< N {P_filt_inv.data[i * N + j] = col[i]}
				delete(e_col, context.temp_allocator)
			}
		} else {
			// Fallback: use identity (shouldn't happen for valid covariance)
			for i in 0 ..< N * N {P_filt_inv.data[i] = 0.0}
			for i in 0 ..< N {P_filt_inv.data[i * N + i] = 1.0}
		}

		// Gk = Pxf * P_filt_inv
		Gk := l.matmul_dyn_simd(&Pxf_mat, &P_filt_inv, context.temp_allocator)
		defer l.matrix_free(&Gk)

		// smoothed state: x_s[k] = x_pred + Gk * (x_s[k+1] - x_filt)
		diff_x := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {diff_x[i] = smoothed[k + 1].x[i] - x_filt[i]}

		diff_x_mat := _matrix_from_flat(diff_x, N, 1, context.temp_allocator)
		defer l.matrix_free(&diff_x_mat)

		corr_x_mat := l.matmul_dyn_simd(&Gk, &diff_x_mat, context.temp_allocator)
		defer l.matrix_free(&corr_x_mat)

		for i in 0 ..< N {smoothed[k].x[i] = x_pred[i] + corr_x_mat.data[i]}
		delete(diff_x, context.temp_allocator)

		// smoothed covariance: P_s[k] = P_pred + Gk * (P_s[k+1] - P_filt) * Gkᵀ
		// diff_P = smoothed[k+1].P - P_filt
		diff_P := make([]f64, N * N, context.temp_allocator)

		// Convert smoothed[k+1].P to flat slice first
		smoothed_P_array := transmute([N * N]f64)smoothed[k + 1].P
		smoothed_P_slice := smoothed_P_array[:]

		for i in 0 ..< N * N {
			diff_P[i] = smoothed_P_slice[i] - P_filt_slice[i]
		}

		// Gk_t = Gkᵀ
		Gk_t := l.matrix_transpose(&Gk, context.temp_allocator)
		defer l.matrix_free(&Gk_t)

		// temp = Gk * diff_P
		diff_P_mat := _matrix_from_flat(diff_P, N, N, context.temp_allocator)
		defer l.matrix_free(&diff_P_mat)

		temp := l.matmul_dyn_simd(&Gk, &diff_P_mat, context.temp_allocator)
		defer l.matrix_free(&temp)

		temp_Gkt := l.matmul_dyn_simd(&temp, &Gk_t, context.temp_allocator)
		defer l.matrix_free(&temp_Gkt)

		P_new := l.matrix_new(f64, N, N, context.temp_allocator)
		defer l.matrix_free(&P_new)

		// Convert P_pred to flat slice first
		P_pred_array := transmute([N * N]f64)P_pred
		P_pred_slice := P_pred_array[:]

		for i in 0 ..< N * N {
			P_new.data[i] = P_pred_slice[i] + temp_Gkt.data[i]
		}

		// Convert result back to fixed-size array
		for i in 0 ..< N * N {smoothed[k].P[i / N, i % N] = P_new.data[i]}
		delete(diff_P, context.temp_allocator)
	}

	return smoothed
}

ukf_rts_smooth :: proc(
	Xi_pred: []([]([$N]f64)), // sigma points BEFORE update (prediction)
	Xi_filt: []([]([N]f64)), // sigma points AFTER update (filtered)
	Wc_seq: []([]f64), // covariance weights per step
	xf: []KalmanState(N), // filtered states
	xp: []KalmanState(N), // predicted states
) -> []KalmanState(N) {
	if _kalman_use_optimized(N) {
		return ukf_rts_smooth_large(Xi_pred, Xi_filt, Wc_seq, xf, xp)
	}

	// Built-in Odin ops for small N (≤8)
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
	y: []f64,
	F, Q, P0: []f64, // N×N, flat, row-major
	H: []f64, // 1×N
	R: []f64, // 1×1
	x0: []f64, // N
	N: int,
) -> f64 {
	T := len(y)
	if T == 0 {return 0.0}

	burn_in := 20
	if burn_in >= T {burn_in = 0}

	// Convert flat arrays to dynamic matrices for optimized linalg
	F_mat := _matrix_from_flat(F, N, N, context.temp_allocator)
	Q_mat := _matrix_from_flat(Q, N, N, context.temp_allocator)
	P_mat := _matrix_from_flat(P0, N, N, context.temp_allocator)
	H_vec := make([]f64, N, context.temp_allocator)
	copy(H_vec, H)
	R_scalar := R[0]
	x_vec := make([]f64, N, context.temp_allocator)
	copy(x_vec, x0)

	defer {
		l.matrix_free(&F_mat)
		l.matrix_free(&Q_mat)
		l.matrix_free(&P_mat)
		delete(H_vec, context.temp_allocator)
		delete(x_vec, context.temp_allocator)
	}

	loglik := 0.0
	two_pi := 2.0 * math.PI

	for t in 0 ..< T {
		// Predict: x_pred = F * x
		x_pred := l.matvec_dyn_simd(&F_mat, x_vec, context.temp_allocator)
		defer delete(x_pred, context.temp_allocator)

		// P_pred = F * P * Fᵀ + Q
		// Step 1: FP = F * P
		// P_pred = F * P * Fᵀ + Q
		FP := l.matmul_dyn_simd(&F_mat, &P_mat, context.temp_allocator)
		defer l.matrix_free(&FP)

		// Use new transpose helper
		Ft := l.matrix_transpose(&F_mat, context.temp_allocator)
		defer l.matrix_free(&Ft)

		P_pred := l.matmul_dyn_simd(&FP, &Ft, context.temp_allocator)
		defer l.matrix_free(&P_pred)

		// Add Q
		for i in 0 ..< N * N {P_pred.data[i] += Q_mat.data[i]}

		// Innovation: y_pred = H * x_pred
		y_pred := l.dot_simd(H_vec, x_pred)
		v := y[t] - y_pred

		// S = H * P_pred * Hᵀ + R
		// H is 1×N, P_pred is N×N, Hᵀ is N×1 → result is scalar
		HP := make([]f64, N, context.temp_allocator)
		for j in 0 ..< N {
			sum := 0.0
			for i in 0 ..< N {
				sum += H_vec[i] * P_pred.data[i * N + j]
			}
			HP[j] = sum
		}
		S := l.dot_simd(HP, H_vec) + R_scalar
		delete(HP, context.temp_allocator)

		if t >= burn_in {
			loglik += -0.5 * (math.ln(two_pi) + math.ln(S) + (v * v) / S)
		}

		// K = P_pred * Hᵀ / S
		K := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {
			sum := 0.0
			for j in 0 ..< N {
				sum += P_pred.data[i * N + j] * H_vec[j]
			}
			K[i] = sum / S
		}

		// Update: x = x_pred + K * v
		for i in 0 ..< N {x_vec[i] = x_pred[i] + K[i] * v}

		// P = P_pred - K * S * Kᵀ
		for i in 0 ..< N {
			for j in 0 ..< N {
				P_mat.data[i * N + j] = P_pred.data[i * N + j] - K[i] * S * K[j]
			}
		}
		delete(K, context.temp_allocator)
	}

	return loglik
}


// ============================================================================
// Kalman filter to get last state (safe manual loops + SIMD dot products)
// ============================================================================
kalman_filter_last_scalar :: proc(
	y: []f64,
	F, Q, P0: []f64,
	H: []f64,
	R: []f64,
	x0: []f64,
	N: int,
) -> (
	xT: []f64,
	PT: []f64,
) {
	T := len(y)
	xT = make([]f64, N)
	defer delete(xT)
	PT = make([]f64, N * N)
	defer delete(PT)

	if T == 0 {
		for i in 0 ..< N {xT[i] = x0[i]}
		for i in 0 ..< N * N {PT[i] = P0[i]}
		return
	}

	// Use manual loops (known working) with SIMD dot products for speed
	x_vec := make([]f64, N, context.allocator)
	P_vec := make([]f64, N * N, context.allocator)
	copy(x_vec, x0)
	copy(P_vec, P0)

	for t in 0 ..< T {
		// --- Predict ---
		// x_pred = F * x (manual loop, but could use matvec_dyn_simd later)
		x_pred := make([]f64, N, context.allocator)
		for i in 0 ..< N {
			s := 0.0
			for j in 0 ..< N {
				s += F[i * N + j] * x_vec[j]
			}
			x_pred[i] = s
		}

		// P_pred = F * P * Fᵀ + Q (manual loops)
		P_pred := make([]f64, N * N, context.allocator)
		temp := make([]f64, N * N, context.allocator)
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {
					s += F[i * N + k] * P_vec[k * N + j]
				}
				temp[i * N + j] = s
			}
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {
					s += temp[i * N + k] * F[j * N + k]
				}
				P_pred[i * N + j] = s + Q[i * N + j]
			}
		}

		// --- Innovation ---
		// y_pred = H * x_pred using SIMD dot product
		y_pred := l.dot_simd(H, x_pred)
		v := y[t] - y_pred

		// S = H * P_pred * Hᵀ + R using SIMD for the quadratic form
		HP := make([]f64, N, context.allocator)
		for j in 0 ..< N {
			sum := 0.0
			for i in 0 ..< N {
				sum += H[i] * P_pred[i * N + j]
			}
			HP[j] = sum
		}
		S := l.dot_simd(HP, H) + R[0]
		delete(HP, context.allocator)

		// --- Update ---
		// K = P_pred * Hᵀ / S
		K := make([]f64, N, context.allocator)
		for i in 0 ..< N {
			sum := 0.0
			for j in 0 ..< N {
				sum += P_pred[i * N + j] * H[j]
			}
			K[i] = sum / S
		}

		// x = x_pred + K * v
		for i in 0 ..< N {x_vec[i] = x_pred[i] + K[i] * v}

		// P = P_pred - K * S * Kᵀ
		for i in 0 ..< N {
			for j in 0 ..< N {
				P_vec[i * N + j] = P_pred[i * N + j] - K[i] * S * K[j]
			}
		}

		// Clean up
		delete(x_pred, context.allocator)
		delete(P_pred, context.allocator)
		delete(temp, context.allocator)
		delete(K, context.allocator)
	}

	// Copy results back to output arrays
	for i in 0 ..< N {xT[i] = x_vec[i]}
	for i in 0 ..< N * N {PT[i] = P_vec[i]}

	delete(x_vec, context.allocator)
	delete(P_vec, context.allocator)

	return
}
// ============================================================================
// Optimized version using wotan_linalg SIMD operations
// ============================================================================
kalman_filter_residuals :: proc(
	y: []f64, // observations
	F, Q, P0: []f64, // N×N, flat, row-major
	H: []f64, // 1×N
	R: []f64, // 1×1
	x0: []f64, // N
	N: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	v: []f64,
	S: []f64, // innovations / innovation variances
) {
	T := len(y)
	v = make([]f64, T, allocator)
	S = make([]f64, T, allocator)

	if T == 0 {
		return
	}

	// Convert flat arrays to dynamic matrices for optimized linalg
	F_mat := _matrix_from_flat(F, N, N, context.temp_allocator)
	Q_mat := _matrix_from_flat(Q, N, N, context.temp_allocator)
	P_mat := _matrix_from_flat(P0, N, N, context.temp_allocator)
	H_vec := make([]f64, N, context.temp_allocator)
	copy(H_vec, H)
	R_scalar := R[0]
	x_vec := make([]f64, N, allocator)
	copy(x_vec, x0)

	defer {
		l.matrix_free(&F_mat)
		l.matrix_free(&Q_mat)
		l.matrix_free(&P_mat)
		delete(H_vec, context.temp_allocator)
		// x_vec and P_mat are returned/used, so don't free here
	}

	for t in 0 ..< T {
		// --- Predict ---
		// x_pred = F * x
		x_pred := l.matvec_dyn_simd(&F_mat, x_vec, context.temp_allocator)
		defer delete(x_pred, context.temp_allocator)

		// P_pred = F * P * Fᵀ + Q
		FP := l.matmul_dyn_simd(&F_mat, &P_mat, context.temp_allocator)
		defer l.matrix_free(&FP)

		Ft := l.matrix_transpose(&F_mat, context.temp_allocator)
		defer l.matrix_free(&Ft)

		P_pred := l.matmul_dyn_simd(&FP, &Ft, context.temp_allocator)
		defer l.matrix_free(&P_pred)

		// Add Q
		for i in 0 ..< N * N {P_pred.data[i] += Q_mat.data[i]}

		// --- Innovation ---
		y_pred := l.dot_simd(H_vec, x_pred)
		vt := y[t] - y_pred

		// S = H * P_pred * Hᵀ + R
		HP := make([]f64, N, context.temp_allocator)
		for j in 0 ..< N {
			sum := 0.0
			for i in 0 ..< N {
				sum += H_vec[i] * P_pred.data[i * N + j]
			}
			HP[j] = sum
		}
		St := l.dot_simd(HP, H_vec) + R_scalar
		delete(HP, context.temp_allocator)

		v[t] = vt
		S[t] = St

		// --- Update ---
		// K = P_pred * Hᵀ / S
		K := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {
			sum := 0.0
			for j in 0 ..< N {
				sum += P_pred.data[i * N + j] * H_vec[j]
			}
			K[i] = sum / St
		}

		// x = x_pred + K * v
		for i in 0 ..< N {x_vec[i] = x_pred[i] + K[i] * vt}

		// P = P_pred - K * S * Kᵀ
		for i in 0 ..< N {
			for j in 0 ..< N {
				P_mat.data[i * N + j] = P_pred.data[i * N + j] - K[i] * St * K[j]
			}
		}
		delete(K, context.temp_allocator)
	}

	return
}
