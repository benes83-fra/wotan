package gpu

import "core:fmt"
import "core:strings"

// ============================================================================
// WGSL compute kernels (f32)
// ============================================================================
//
// Every kernel has entry point `main`, the parameter block at binding 0 and its
// storage buffers at bindings 1, 2, ... in the order the dispatching code binds
// them. 1D kernels run 256 threads per workgroup over a 2D grid (idx1).
// Each kernel mirrors the CPU op of the same name in wotan/tensor; where the CPU
// code has a numerically careful form (sigmoid, tanh, softplus), so does the kernel.
// Gradient kernels written as `dx += ...` overwrite instead when their overwrite
// flag is set (u7 unless noted; see keep): the first write to a fresh gradient
// then needs no clear before it.

@(private)
Kernel :: enum {
	Unary, // y = f(x)
	Unary_Bwd, // dx += dy * f'(x, y)
	Binary, // out = a (+ - * /) b
	Binary_Bwd, // d(a or b) += dy * ∂(a ∘ b)
	Binary_Bwd_Both, // da and db of a ∘ b in one launch
	Matmul, // C (+)= op(A) @ op(B), 16x16 tiles, one output per thread
	Matmul_Tiled, // the same, 64x64 tiles, 4x4 outputs per thread (register tiling)
	Matmul_Heads, // Matmul and Matmul_Tiled with operands in head layout (attention;
	Matmul_Tiled_Heads, // generated source, see _kernel_source)
	Matmul_Group, // Matmul and Matmul_Tiled for up to LINEAR_GROUP products in one
	Matmul_Tiled_Group, // launch (generated source, see _matmul_group_source)
	Reduce, // out[k] = scale * Σ x (or Σ x²), per workgroup
	Bcast_Add, // dx += scale * s[k]
	Mse_Bwd, // dx += scale * dy[0] * (a - b), or sign(a - b) (L1 loss)
	Mask_Bcast_Add, // dx += scale * dy[0] * m[i / inner] (masked sum / mean)
	Bias_Add, // out = a + bias[i % D]
	Col_Sum, // db[j] += Σ_i dy[i, j]
	Col_Sum_Group, // Col_Sum of up to LINEAR_GROUP gradients in one launch
	Lstm_Fwd, // the LSTM recurrence over all steps (one workgroup per sequence)
	Lstm_Bwd, // its backpropagation through time
	Gru_Fwd, // the GRU recurrence (as Lstm_Fwd)
	Gru_Bwd,
	Rnn_Fwd, // the tanh RNN recurrence (as Lstm_Fwd)
	Rnn_Bwd,
	Ssm_Fwd, // Mamba's selective scan, one thread per (sequence, channel)
	Ssm_Bwd,
	Ssm_Reduce, // its gradients that sum over channels or steps
	Row_Sum, // out[b] = Σ_j x[b, j]
	Row_Bcast_Add, // dx[b, j] += dy[b]
	Transpose, // y (+)= xᵀ
	Strided_Copy, // row blocks from one layout to another (slice, concat)
	Scale_Inplace, // y *= f
	Adam,
	Adam_2, // Adam on 2, 3, 4 parameters (generated source, see _adam_source)
	Adam_3,
	Adam_4,
	Grad_Sq_Sum, // Σ g² over up to 7 gradients, per workgroup (generated, see _grad_source)
	Grad_Scale, // g *= f on up to 8 gradients (generated)
	Sgd,
	Softmax_Rows, // per row: softmax (scaled, masked), logsumexp or cross-entropy loss
	Softmax_Bwd, // per row: dx (+)= f * p * (g - Σ p g)
	Row_Prob_Grad, // per row: dx += coef * (softmax(x) - onehot) (logsumexp, cross-entropy)
	Norm_Fwd, // per row: layer norm / RMS norm
	Norm_Stats, // per row: (mean, 1 / std) or (0, 1 / rms)
	Norm_Bwd_X, // per row: input gradient of layer norm / RMS norm
	Norm_Bwd_Param, // per column: gamma (and beta) gradients
	Gather_Rows, // embedding lookup
	Scatter_Add_Rows, // embedding backward (atomic adds)
	Permute_Mha, // [B, S, H·hd] <-> [B·H, S, hd]
	Bce, // binary cross-entropy: per-element loss or gradients
	Row_Scale, // dst[i] (+)= src[i] * g[i / D] (gate_mul)
	Row_Dot, // o[r] += Σ_j a[r, j] b[r, j]
	// phase 3: convolutions, pooling, batch norm
	Im2col_2d, // conv2d input patches as columns [C·kH·kW, N·L]
	Col2im_2d, // dx += the column gradients summed back onto the image
	Im2col_1d, // conv1d input patches as rows [B·T_out, C_in·K]
	Col2im_1d, // dx += the row gradients summed back onto the sequence
	Conv1d_Grouped, // grouped conv1d (groups > 1) computed directly
	Conv1d_Grouped_Dx, // its input gradient
	Conv1d_Grouped_Dw, // its weight gradient (one workgroup per weight)
	Chan_Bias, // dst[i] += b[(i / L) % C]
	Chan_Sum, // per channel: db[c] += Σ_{n, l} dy[n, c, l]
	Pool, // max / average pooling
	Pool_Bwd, // pooling backward, gathered per input element
	Bn_Stats, // per channel: batch or running statistics, running-stat update
	Bn_Apply, // y = (x - mean) · inv_std · γ + β
	Bn_Bwd_Sums, // per channel: Σ dy, Σ dy·x̂ (and dγ, dβ)
	Bn_Bwd_X, // input gradient of batch norm
	// phase 4: finance and VAE ops
	Loss_Terms, // per-element terms of KL divergence, entropy, prospect utility
	Loss_Grad, // dx += g[0] · f0 · ∂term (scalar losses)
	Sharpe, // negative Sharpe ratio (one workgroup), mean and std kept
	Norm_Time, // z-score over the time axis of [N, C, T, L], forward and backward
	Top_K, // per row: 1 at the k largest entries
	Reparam, // z = mu + exp(log_var / 2) · eps, and its log_var gradient
}

@(private)
KERNEL_PRELUDE :: `
struct P {
	u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, u6: u32, u7: u32,
	f0: f32, f1: f32, f2: f32, f3: f32, f4: f32, f5: f32, f6: f32, f7: f32,
}
@group(0) @binding(0) var<uniform> p: P;

fn idx1(g: vec3<u32>, nw: vec3<u32>) -> u32 {
	return g.y * nw.x * 256u + g.x;
}
// The old value of an accumulated gradient, or 0 when the kernel overwrites
// it (overwrite = 1: the buffer holds no gradient yet and was not cleared;
// gradient kernels take the flag in u7)
fn keep(old: f32, overwrite: u32) -> f32 {
	return select(old, 0.0, overwrite == 1u);
}
// 1 / (1 + e^-x) without overflow for large |x|
fn sigm(x: f32) -> f32 {
	if (x >= 0.0) { return 1.0 / (1.0 + exp(-x)); }
	let e = exp(x);
	return e / (1.0 + e);
}
// tanh through e^-2|x| (no inf / inf for large |x|)
fn tanh_s(x: f32) -> f32 {
	let e = exp(-2.0 * abs(x));
	let r = (1.0 - e) / (1.0 + e);
	return select(-r, r, x >= 0.0);
}
// erfc with fractional error < 1.2e-7 everywhere (Numerical Recipes erfcc), so
// the normal CDF keeps its relative accuracy deep in the lower tail
fn erfc_s(x: f32) -> f32 {
	let z = abs(x);
	let t = 1.0 / (1.0 + 0.5 * z);
	let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
		t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
		t * (-0.82215223 + t * 0.17087277)))))))));
	return select(2.0 - r, r, x >= 0.0);
}
const GELU_C: f32 = 0.7978845608028654;
const PI_F: f32 = 3.14159265358979;
// log|Γ(x)| for x > 0: shift to x ≥ 8, then Stirling's series
fn lgamma_pos(x0: f32) -> f32 {
	var x = x0;
	var prod = 1.0;
	while (x < 8.0) { prod = prod * x; x = x + 1.0; }
	let r = 1.0 / (x * x);
	let s = (x - 0.5) * log(x) - x + 0.91893853320467274 + (1.0 / 12.0 - r * (1.0 / 360.0 - r / 1260.0)) / x;
	return s - log(prod);
}
// log|Γ(x)|, reflection below 0.5
fn lgamma_s(x: f32) -> f32 {
	if (x < 0.5) { return log(PI_F / abs(sin(PI_F * x))) - lgamma_pos(1.0 - x); }
	return lgamma_pos(x);
}
// ψ(x) for x > 0: recurrence up to x ≥ 10, then the asymptotic series (as the CPU digamma)
fn digamma_pos(x0: f32) -> f32 {
	var x = x0;
	var acc = 0.0;
	while (x < 10.0) { acc = acc - 1.0 / x; x = x + 1.0; }
	let r = 1.0 / (x * x);
	let series = r * (1.0 / 12.0 - r * (1.0 / 120.0 - r * (1.0 / 252.0 - r * (1.0 / 240.0 - r / 132.0))));
	return acc + log(x) - 0.5 / x - series;
}
// ψ(x): NaN at the poles (0, -1, -2, ...), reflection for x < 0
fn digamma_s(x: f32) -> f32 {
	if (x <= 0.0 && x == floor(x)) { return bitcast<f32>(0x7fc00000u); }
	if (x < 0.0) { return digamma_pos(1.0 - x) - PI_F / tan(PI_F * x); }
	return digamma_pos(x);
}
// prospect value v(x) = x^α (x ≥ 0), -λ(-x)^β (x < 0)
fn prospect_v(x: f32, al: f32, be: f32, la: f32) -> f32 {
	if (x > 0.0) { return pow(x, al); }
	if (x < 0.0) { return -la * pow(-x, be); }
	return 0.0;
}
// dv/dx, 0 within 1e-10 of 0 (as the CPU)
fn prospect_dv(x: f32, al: f32, be: f32, la: f32) -> f32 {
	if (x > 1e-10) { return al * pow(x, al - 1.0); }
	if (x < -1e-10) { return la * be * pow(-x, be - 1.0); }
	return 0.0;
}
// one workgroup per row over a 2D grid of workgroups
fn row_of(wid: vec3<u32>, nw: vec3<u32>) -> u32 {
	return wid.y * nw.x + wid.x;
}
`

// Unary op codes (shared by Unary and Unary_Bwd)
@(private)
U_COPY :: 0 // backward: dx += f0 * dy (axpy)
@(private)
U_NEG :: 1
@(private)
U_SCALE :: 2
@(private)
U_RELU :: 3
@(private)
U_LEAKY :: 4
@(private)
U_SIGMOID :: 5
@(private)
U_TANH :: 6
@(private)
U_EXP :: 7
@(private)
U_LOG :: 8
@(private)
U_SQRT :: 9
@(private)
U_CLAMP :: 10
@(private)
U_SOFTPLUS :: 11
@(private)
U_GELU_TANH :: 12
@(private)
U_NORM_CDF :: 13
@(private)
U_LGAMMA :: 14
@(private)
U_PROSPECT :: 15 // f0 α, f1 β, f2 λ
@(private)
U_PROB_WEIGHT :: 16 // f0 γ
@(private)
U_GELU :: 17 // x · Φ(x) (tensor_gelu)
@(private)
U_ABS :: 18
@(private)
U_SIN :: 19
@(private)
U_COS :: 20
@(private)
U_SIGN :: 21 // gradient 0

@(private)
KERNEL_SOURCES := [Kernel]string {
	.Unary              = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let v = x[i];
	var r = v;
	switch p.u1 {
		case 1u: { r = -v; }
		case 2u: { r = p.f0 * v; }
		case 3u: { r = select(0.0, v, v > 0.0); }
		case 4u: { r = select(p.f0 * v, v, v > 0.0); }
		case 5u: { r = sigm(v); }
		case 6u: { r = tanh_s(v); }
		case 7u: { r = exp(v); }
		case 8u: { r = log(v); }
		case 9u: { r = sqrt(v); }
		case 10u: { r = v; if (r < p.f0) { r = p.f0; } if (r > p.f1) { r = p.f1; } }
		case 11u: {
			if (v > 20.0) { r = v; } else if (v < -20.0) { r = exp(v); } else { r = log(1.0 + exp(v)); }
		}
		case 12u: { r = 0.5 * v * (1.0 + tanh_s(GELU_C * (v + 0.044715 * v * v * v))); }
		case 13u: { r = 0.5 * erfc_s(-v * 0.70710678118654752); }
		case 14u: { r = lgamma_s(v); }
		case 17u: { r = v * (0.5 * erfc_s(-v * 0.70710678118654752)); }
		case 15u: { r = prospect_v(v, p.f0, p.f1, p.f2); }
		case 16u: {
			let q = clamp(v, 1e-7, 1.0 - 1e-7);
			let pg = pow(q, p.f0);
			let qg = pow(1.0 - q, p.f0);
			r = pg / pow(pg + qg, 1.0 / p.f0);
		}
		case 18u: { r = abs(v); }
		case 19u: { r = sin(v); }
		case 20u: { r = cos(v); }
		case 21u: { r = sign(v); }
		default: {}
	}
	y[i] = r;
}
`,
	.Unary_Bwd          = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let gy = dy[i];
	let v = x[i];
	var d = 0.0;
	switch p.u1 {
		case 0u: { d = p.f0 * gy; }
		case 3u: { d = select(0.0, gy, v > 0.0); }
		case 4u: { d = select(gy * p.f0, gy, v > 0.0); }
		case 5u: { let s = sigm(v); d = gy * s * (1.0 - s); }
		case 6u: { let th = tanh_s(v); d = gy * (1.0 - th * th); }
		case 7u: { d = gy * y[i]; }
		case 8u: { d = gy / v; }
		case 9u: { d = gy / (2.0 * y[i]); }
		case 10u: { d = select(0.0, gy, v >= p.f0 && v <= p.f1); }
		case 11u: { d = gy / (1.0 + exp(-v)); }
		case 12u: {
			let th = tanh_s(GELU_C * (v + 0.044715 * v * v * v));
			let du = GELU_C * (1.0 + 3.0 * 0.044715 * v * v);
			d = gy * (0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * du);
		}
		case 13u: { d = gy * exp(-0.5 * v * v) * 0.3989422804014327; }
		case 14u: { d = gy * digamma_s(v); }
		case 17u: {
			let cdf = 0.5 * erfc_s(-v * 0.70710678118654752);
			d = gy * cdf + gy * v * exp(-0.5 * v * v) * 0.3989422804014327;
		}
		case 15u: { d = gy * prospect_dv(v, p.f0, p.f1, p.f2); }
		case 16u: {
			let q = clamp(v, 1e-7, 1.0 - 1e-7);
			let pg = pow(q, p.f0);
			let qg = pow(1.0 - q, p.f0);
			let t2 = (pow(q, p.f0 - 1.0) - pow(1.0 - q, p.f0 - 1.0)) / (pg + qg);
			d = gy * y[i] * (p.f0 / q - t2);
		}
		case 18u: { d = gy * sign(v); }
		case 19u: { d = gy * cos(v); }
		case 20u: { d = -gy * sin(v); }
		default: {}
	}
	dx[i] = keep(dx[i], p.u7) + d;
}
`,
	.Binary             = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let x = a[i];
	let z = b[i];
	var r = x + z;
	switch p.u1 {
		case 1u: { r = x - z; }
		case 2u: { r = x * z; }
		case 3u: { r = x / z; }
		default: {}
	}
	o[i] = r;
}
`,
	// u1: 2 mul, 3 div; u2: 0 gradient of a, 1 gradient of b
	.Binary_Bwd         = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let gy = dy[i];
	var d = 0.0;
	if (p.u1 == 2u) {
		d = select(gy * b[i], gy * a[i], p.u2 == 1u);
	} else {
		let z = b[i];
		d = select(gy / z, -gy * a[i] / (z * z), p.u2 == 1u);
	}
	dst[i] = keep(dst[i], p.u7) + d;
}
`,
	// both gradients of a + b, a - b, a * b, a / b (u1 0..3, same shapes);
	// u6 / u7 overwrite da / db (as keep)
	.Binary_Bwd_Both    = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> da: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let gy = dy[i];
	var d_a = gy;
	var d_b = gy;
	if (p.u1 == 1u) {
		d_b = -gy;
	} else if (p.u1 == 2u) {
		d_a = gy * b[i];
		d_b = gy * a[i];
	} else if (p.u1 == 3u) {
		let z = b[i];
		d_a = gy / z;
		d_b = -gy * a[i] / (z * z);
	}
	da[i] = keep(da[i], p.u6) + d_a;
	db[i] = keep(db[i], p.u7) + d_b;
}
`,
	// C[M, N] (+)= op(A)[M, K] @ op(B)[K, N]; u0 M, u1 N, u2 K, u3 bit 0 transA,
	// u4 bit 0 transB, u5 bit 0 accumulate, u6 first row of this dispatch, u7 bit 0:
	// add bias[column] (tensor_linear; the dummy buffer is bound otherwise).
	// Workgroup z is the batch: batch z uses the z-th A, B and C blocks (base and
	// stride: MATMUL_CONTIGUOUS, or MATMUL_HEADS in the _Heads kernels). 16x16
	// tiles in workgroup memory.
	.Matmul             = `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let M = p.u0; let N = p.u1; let K = p.u2;
	let ta = (p.u3 & 1u) == 1u;
	let tb = (p.u4 & 1u) == 1u;
	let a0 = select(base(g.z, M, K, p.u3), base(g.z, K, M, p.u3), ta);
	let b0 = select(base(g.z, K, N, p.u4), base(g.z, N, K, p.u4), tb);
	let c0 = base(g.z, M, N, p.u5);
	let row = g.y + p.u6;
	let col = g.x;
	var acc = 0.0;
	let tiles = (K + 15u) / 16u;
	for (var tt = 0u; tt < tiles; tt++) {
		let ak = tt * 16u + lid.x;
		var av = 0.0;
		if (row < M && ak < K) {
			if (!ta) { av = A[a0 + row * stride(K, p.u3) + ak]; } else { av = A[a0 + ak * stride(M, p.u3) + row]; }
		}
		As[lid.y][lid.x] = av;
		let bk = tt * 16u + lid.y;
		var bv = 0.0;
		if (bk < K && col < N) {
			if (!tb) { bv = B[b0 + bk * stride(N, p.u4) + col]; } else { bv = B[b0 + col * stride(K, p.u4) + bk]; }
		}
		Bs[lid.y][lid.x] = bv;
		workgroupBarrier();
		for (var k = 0u; k < 16u; k++) { acc = acc + As[lid.y][k] * Bs[k][lid.x]; }
		workgroupBarrier();
	}
	if (row < M && col < N) {
		let o = c0 + row * stride(N, p.u5) + col;
		var v = acc;
		if ((p.u7 & 1u) == 1u) { v = v + bias[col]; }
		if ((p.u5 & 1u) == 1u) { C[o] = C[o] + v; } else { C[o] = v; }
	}
}
`,
	// Register-tiled matmul, parameters as Matmul. A workgroup of 16x16 threads
	// computes a 64x64 tile of C; thread (tx, ty) owns rows ty + 16i and columns
	// tx + 16j (i, j < 4), so neighbouring threads touch neighbouring columns
	// (coalesced global writes, no shared-memory bank conflicts). Each step of
	// 16 along K stages a 64x16 slice of op(A) and a 16x64 slice of op(B) in
	// workgroup memory; the 4x4 accumulators stay in registers.
	.Matmul_Tiled       = `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
var<workgroup> As: array<array<f32, 64>, 16>;
var<workgroup> Bs: array<array<f32, 64>, 16>;
// element (r, k) of op(A) and (k, c) of op(B); 0 outside the matrix
fn load_a(a0: u32, r: u32, k: u32) -> f32 {
	if (r >= p.u0 || k >= p.u2) { return 0.0; }
	if ((p.u3 & 1u) == 0u) { return A[a0 + r * stride(p.u2, p.u3) + k]; }
	return A[a0 + k * stride(p.u0, p.u3) + r];
}
fn load_b(b0: u32, k: u32, c: u32) -> f32 {
	if (k >= p.u2 || c >= p.u1) { return 0.0; }
	if ((p.u4 & 1u) == 0u) { return B[b0 + k * stride(p.u1, p.u4) + c]; }
	return B[b0 + c * stride(p.u2, p.u4) + k];
}
fn store(c0: u32, r: u32, col: u32, v: f32) {
	if (r >= p.u0 || col >= p.u1) { return; }
	let o = c0 + r * stride(p.u1, p.u5) + col;
	var w = v;
	if ((p.u7 & 1u) == 1u) { w = w + bias[col]; }
	if ((p.u5 & 1u) == 1u) { C[o] = C[o] + w; } else { C[o] = w; }
}
fn store_row(c0: u32, r: u32, col: u32, v: vec4<f32>) {
	store(c0, r, col, v.x);
	store(c0, r, col + 16u, v.y);
	store(c0, r, col + 32u, v.z);
	store(c0, r, col + 48u, v.w);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let M = p.u0; let N = p.u1; let K = p.u2;
	let ta = (p.u3 & 1u) == 1u;
	let tb = (p.u4 & 1u) == 1u;
	let a0 = select(base(wid.z, M, K, p.u3), base(wid.z, K, M, p.u3), ta);
	let b0 = select(base(wid.z, K, N, p.u4), base(wid.z, N, K, p.u4), tb);
	let c0 = base(wid.z, M, N, p.u5);
	let row0 = wid.y * 64u + p.u6;
	let col0 = wid.x * 64u;
	let tx = lid.x; let ty = lid.y;
	let tid = ty * 16u + tx;
	// staging coordinates: (ar, ak) walks a 16-wide index fastest, (bk, bc) a
	// 64-wide one
	let ak = tid % 16u; let ar = tid / 16u;
	let bc = tid % 64u; let bk = tid / 64u;
	var c0v = vec4<f32>(0.0); var c1v = vec4<f32>(0.0);
	var c2v = vec4<f32>(0.0); var c3v = vec4<f32>(0.0);
	for (var kt = 0u; kt < K; kt += 16u) {
		// neighbouring threads read neighbouring addresses: along k for A, along
		// the rows for Aᵀ; along the columns for B, along k for Bᵀ
		if (!ta) {
			As[ak][ar] = load_a(a0, row0 + ar, kt + ak);
			As[ak][ar + 16u] = load_a(a0, row0 + ar + 16u, kt + ak);
			As[ak][ar + 32u] = load_a(a0, row0 + ar + 32u, kt + ak);
			As[ak][ar + 48u] = load_a(a0, row0 + ar + 48u, kt + ak);
		} else {
			As[bk][bc] = load_a(a0, row0 + bc, kt + bk);
			As[bk + 4u][bc] = load_a(a0, row0 + bc, kt + bk + 4u);
			As[bk + 8u][bc] = load_a(a0, row0 + bc, kt + bk + 8u);
			As[bk + 12u][bc] = load_a(a0, row0 + bc, kt + bk + 12u);
		}
		if (!tb) {
			Bs[bk][bc] = load_b(b0, kt + bk, col0 + bc);
			Bs[bk + 4u][bc] = load_b(b0, kt + bk + 4u, col0 + bc);
			Bs[bk + 8u][bc] = load_b(b0, kt + bk + 8u, col0 + bc);
			Bs[bk + 12u][bc] = load_b(b0, kt + bk + 12u, col0 + bc);
		} else {
			Bs[ak][ar] = load_b(b0, kt + ak, col0 + ar);
			Bs[ak][ar + 16u] = load_b(b0, kt + ak, col0 + ar + 16u);
			Bs[ak][ar + 32u] = load_b(b0, kt + ak, col0 + ar + 32u);
			Bs[ak][ar + 48u] = load_b(b0, kt + ak, col0 + ar + 48u);
		}
		workgroupBarrier();
		for (var k = 0u; k < 16u; k++) {
			let bv = vec4<f32>(Bs[k][tx], Bs[k][tx + 16u], Bs[k][tx + 32u], Bs[k][tx + 48u]);
			c0v = fma(vec4<f32>(As[k][ty]), bv, c0v);
			c1v = fma(vec4<f32>(As[k][ty + 16u]), bv, c1v);
			c2v = fma(vec4<f32>(As[k][ty + 32u]), bv, c2v);
			c3v = fma(vec4<f32>(As[k][ty + 48u]), bv, c3v);
		}
		workgroupBarrier();
	}
	store_row(c0, row0 + ty, col0 + tx, c0v);
	store_row(c0, row0 + ty + 16u, col0 + tx, c1v);
	store_row(c0, row0 + ty + 32u, col0 + tx, c2v);
	store_row(c0, row0 + ty + 48u, col0 + tx, c3v);
}
`,
	.Matmul_Heads       = "", // generated (_kernel_source)
	.Matmul_Tiled_Heads = "",
	.Matmul_Group       = "", // generated (_matmul_group_source)
	.Matmul_Tiled_Group = "",
	// u0 n, u1 0 sum / 1 sum of squares / 2 sum of absolute values, u2 output
	// index, f0 scale. Each workgroup writes the scaled sum of its grid-stride
	// share to out[u2 + wg].
	.Reduce             = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	var s = 0.0;
	var i = wid.x * 256u + lid.x;
	let stride = nw.x * 256u;
	loop {
		if (i >= p.u0) { break; }
		let v = x[i];
		s = s + select(select(v, v * v, p.u1 == 1u), abs(v), p.u1 == 2u);
		i = i + stride;
	}
	sh[lid.x] = s;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid.x < w) { sh[lid.x] = sh[lid.x] + sh[lid.x + w]; }
		workgroupBarrier();
	}
	if (lid.x == 0u) { o[p.u2 + wid.x] = sh[0] * p.f0; }
}
`,
	// dx[i] += f0 * s[u2]
	.Bcast_Add          = `
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dx[i] = keep(dx[i], p.u7) + p.f0 * s[p.u2];
}
`,
	// dx[i] += f0 * dy[0] * (a[i] - b[i]), or with u1 = 1 (L1 loss)
	// f0 * dy[0] * sign(a[i] - b[i])
	.Mse_Bwd            = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let d = a[i] - b[i];
	dx[i] = keep(dx[i], p.u7) + p.f0 * dy[0] * select(d, sign(d), p.u1 == 1u);
}
`,
	// dx[i] += f0 * dy[0] * m[i / u1] (masked sum / mean); u0 n
	.Mask_Bcast_Add     = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dx[i] = keep(dx[i], p.u7) + p.f0 * dy[0] * m[i / p.u1];
}
`,
	// out[i] = a[i] + bias[i % D]; u0 n, u1 D
	.Bias_Add           = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	o[i] = a[i] + bias[i % p.u1];
}
`,
	// db[j] += Σ_i dy[i * D + c0 + j]; u0 N rows, u1 D, u2 first column c0,
	// u3 columns summed (0: D)
	.Col_Sum            = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let j = idx1(g, nw);
	if (j >= select(p.u3, p.u1, p.u3 == 0u)) { return; }
	var s = 0.0;
	for (var i = 0u; i < p.u0; i++) { s = s + dy[i * p.u1 + p.u2 + j]; }
	db[j] = keep(db[j], p.u7) + s;
}
`,
	// Col_Sum for products m < LINEAR_GROUP (workgroup z): db_m[j] += Σ_i
	// dy_m[i * D + j]; u0 N rows, u1 D, u7 bit m: overwrite db_m (as keep)
	.Col_Sum_Group      = `
@group(0) @binding(1) var<storage, read> dy0: array<f32>;
@group(0) @binding(2) var<storage, read> dy1: array<f32>;
@group(0) @binding(3) var<storage, read> dy2: array<f32>;
@group(0) @binding(4) var<storage, read_write> db0: array<f32>;
@group(0) @binding(5) var<storage, read_write> db1: array<f32>;
@group(0) @binding(6) var<storage, read_write> db2: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let j = idx1(g, nw);
	if (j >= p.u1) { return; }
	let m = g.z;
	let o = (p.u7 >> m) & 1u;
	var s = 0.0;
	switch m {
		case 1u: {
			for (var i = 0u; i < p.u0; i++) { s = s + dy1[i * p.u1 + j]; }
			db1[j] = keep(db1[j], o) + s;
		}
		case 2u: {
			for (var i = 0u; i < p.u0; i++) { s = s + dy2[i * p.u1 + j]; }
			db2[j] = keep(db2[j], o) + s;
		}
		default: {
			for (var i = 0u; i < p.u0; i++) { s = s + dy0[i * p.u1 + j]; }
			db0[j] = keep(db0[j], o) + s;
		}
	}
}
`,
	// LSTM over all steps, gates [i, f, g, o]; u0 batch B, u1 steps S, u2
	// hidden H. proj [B·S, 4H] = x @ W_ih (row b·S + s). One workgroup per
	// sequence b runs the steps in order; its threads share the units of h.
	// saved (the CPU's layout): hs, cs [(S + 1)·B·H] (row s = state entering
	// step s, index b·H + unit), then i, f, g, o, tanh(c) [S·B·H] each.
	.Lstm_Fwd           = `
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> w_hh: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> h0: array<f32>;
@group(0) @binding(5) var<storage, read> c0: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<storage, read_write> saved: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let H = p.u2;
	let b = wid.y * nw.x + wid.x;
	if (b >= B) { return; }
	let BH = B * H; let G = 4u * H; let SBH = S * BH;
	let cb = (S + 1u) * BH;
	let gb = 2u * cb;
	for (var u = lid.x; u < H; u += 256u) {
		saved[b * H + u] = h0[b * H + u];
		saved[cb + b * H + u] = c0[b * H + u];
	}
	storageBarrier();
	for (var s = 0u; s < S; s++) {
		let hp = s * BH + b * H; // h and c entering the step
		let pr = (b * S + s) * G;
		for (var u = lid.x; u < H; u += 256u) {
			var ai = 0.0; var af = 0.0; var ag = 0.0; var ao = 0.0;
			for (var k = 0u; k < H; k++) {
				let h = saved[hp + k];
				let w = k * G + u;
				ai = ai + h * w_hh[w];
				af = af + h * w_hh[w + H];
				ag = ag + h * w_hh[w + 2u * H];
				ao = ao + h * w_hh[w + 3u * H];
			}
			let iv = sigm((proj[pr + u] + bias[u]) + ai);
			let fv = sigm((proj[pr + H + u] + bias[H + u]) + af);
			let gv = tanh_s((proj[pr + 2u * H + u] + bias[2u * H + u]) + ag);
			let ov = sigm((proj[pr + 3u * H + u] + bias[3u * H + u]) + ao);
			let c = fv * saved[cb + hp + u] + iv * gv;
			let tc = tanh_s(c);
			let h = ov * tc;
			saved[hp + BH + u] = h;
			saved[cb + hp + BH + u] = c;
			let gi = gb + hp;
			saved[gi + u] = iv;
			saved[gi + SBH + u] = fv;
			saved[gi + 2u * SBH + u] = gv;
			saved[gi + 3u * SBH + u] = ov;
			saved[gi + 4u * SBH + u] = tc;
			o[(b * S + s) * H + u] = h;
		}
		storageBarrier();
	}
}
`,
	// LSTM backpropagation through time (layout as Lstm_Fwd), one workgroup per
	// sequence. Writes the pre-activation gate gradients dG [B·S, 4H] (row
	// b·S + s) and the states entering each step hb [B·S, H] for the weight
	// products; carry [2·B·H] holds dh and dc of the next step. At the end
	// dh0 / dc0 += dL/dh_0, dL/dc_0 (u3 bit 0 / 1: wanted; u4 bit 0 / 1:
	// overwrite, as keep). w_hh_t is W_hhᵀ [4H, H], so that neighbouring threads
	// read neighbouring weights.
	.Lstm_Bwd           = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> saved: array<f32>;
@group(0) @binding(3) var<storage, read> w_hh_t: array<f32>;
@group(0) @binding(4) var<storage, read_write> dG: array<f32>;
@group(0) @binding(5) var<storage, read_write> hb: array<f32>;
@group(0) @binding(6) var<storage, read_write> carry: array<f32>;
@group(0) @binding(7) var<storage, read_write> dh0: array<f32>;
@group(0) @binding(8) var<storage, read_write> dc0: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let H = p.u2;
	let b = wid.y * nw.x + wid.x;
	if (b >= B) { return; }
	let BH = B * H; let G = 4u * H; let SBH = S * BH;
	let cb = (S + 1u) * BH;
	let gb = 2u * cb;
	for (var u = lid.x; u < H; u += 256u) {
		carry[b * H + u] = 0.0;
		carry[BH + b * H + u] = 0.0;
	}
	storageBarrier();
	for (var q = 0u; q < S; q++) {
		let s = S - 1u - q;
		let hp = s * BH + b * H;
		let r = (b * S + s) * G;
		for (var u = lid.x; u < H; u += 256u) {
			let gi = gb + hp + u;
			let ig = saved[gi]; let fg = saved[gi + SBH]; let gg = saved[gi + 2u * SBH];
			let og = saved[gi + 3u * SBH]; let tc = saved[gi + 4u * SBH];
			let dh = dy[(b * S + s) * H + u] + carry[b * H + u];
			let d_o = dh * tc;
			let dc = dh * og * (1.0 - tc * tc) + carry[BH + b * H + u];
			dG[r + u] = dc * gg * ig * (1.0 - ig);
			dG[r + H + u] = dc * saved[cb + hp + u] * fg * (1.0 - fg);
			dG[r + 2u * H + u] = dc * ig * (1.0 - gg * gg);
			dG[r + 3u * H + u] = d_o * og * (1.0 - og);
			carry[BH + b * H + u] = dc * fg;
			hb[(b * S + s) * H + u] = saved[hp + u];
		}
		storageBarrier();
		// dh of the previous step: dG row @ W_hhᵀ
		for (var k = lid.x; k < H; k += 256u) {
			var acc = 0.0;
			for (var j = 0u; j < G; j++) { acc = acc + dG[r + j] * w_hh_t[j * H + k]; }
			carry[b * H + k] = acc;
		}
		storageBarrier();
	}
	for (var u = lid.x; u < H; u += 256u) {
		let i = b * H + u;
		if ((p.u3 & 1u) != 0u) { dh0[i] = keep(dh0[i], p.u4 & 1u) + carry[i]; }
		if ((p.u3 & 2u) != 0u) { dc0[i] = keep(dc0[i], (p.u4 >> 1u) & 1u) + carry[BH + i]; }
	}
}
`,
	// GRU over all steps (PyTorch's gates [r, z, n]); parameters, grid and
	// proj as Lstm_Fwd, proj [B·S, 3H]. u3 bit 0: bias_hn given. saved (the
	// CPU's layout): hs [(S + 1)·B·H], then r, z, n and hh_n = h W_hn + b_hn
	// [S·B·H] each.
	.Gru_Fwd            = `
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> w_hh: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> bias_hn: array<f32>;
@group(0) @binding(5) var<storage, read> h0: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<storage, read_write> saved: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let H = p.u2;
	let b = wid.y * nw.x + wid.x;
	if (b >= B) { return; }
	let BH = B * H; let G = 3u * H; let SBH = S * BH;
	let gb = (S + 1u) * BH;
	for (var u = lid.x; u < H; u += 256u) { saved[b * H + u] = h0[b * H + u]; }
	storageBarrier();
	for (var s = 0u; s < S; s++) {
		let hp = s * BH + b * H;
		let pr = (b * S + s) * G;
		for (var u = lid.x; u < H; u += 256u) {
			var ar = 0.0; var az = 0.0; var an = 0.0;
			for (var k = 0u; k < H; k++) {
				let h = saved[hp + k];
				let w = k * G + u;
				ar = ar + h * w_hh[w];
				az = az + h * w_hh[w + H];
				an = an + h * w_hh[w + 2u * H];
			}
			let r = sigm((proj[pr + u] + bias[u]) + ar);
			let z = sigm((proj[pr + H + u] + bias[H + u]) + az);
			var hn = an;
			if ((p.u3 & 1u) != 0u) { hn = an + bias_hn[u]; }
			let n = tanh_s((proj[pr + 2u * H + u] + bias[2u * H + u]) + r * hn);
			let h = (1.0 - z) * n + z * saved[hp + u];
			saved[hp + BH + u] = h;
			let gi = gb + hp + u;
			saved[gi] = r;
			saved[gi + SBH] = z;
			saved[gi + 2u * SBH] = n;
			saved[gi + 3u * SBH] = hn;
			o[(b * S + s) * H + u] = h;
		}
		storageBarrier();
	}
}
`,
	// GRU backpropagation through time (layout as Gru_Fwd). Writes the gate
	// gradients on the input path d_ih and on the hidden path d_hh [B·S, 3H]
	// (they differ in the n gate: d_hh_n = d_ih_n · r), and the entering
	// states hb [B·S, H]. carry [2·B·H]: dh of the next step, then dh · z.
	// u3 bit 0: dh0 += dL/dh_0 (u4 bit 0: overwrite, as keep). w_hh_t: W_hhᵀ.
	.Gru_Bwd            = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> saved: array<f32>;
@group(0) @binding(3) var<storage, read> w_hh_t: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_ih: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_hh: array<f32>;
@group(0) @binding(6) var<storage, read_write> hb: array<f32>;
@group(0) @binding(7) var<storage, read_write> carry: array<f32>;
@group(0) @binding(8) var<storage, read_write> dh0: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let H = p.u2;
	let b = wid.y * nw.x + wid.x;
	if (b >= B) { return; }
	let BH = B * H; let G = 3u * H; let SBH = S * BH;
	let gb = (S + 1u) * BH;
	for (var u = lid.x; u < H; u += 256u) { carry[b * H + u] = 0.0; }
	storageBarrier();
	for (var q = 0u; q < S; q++) {
		let s = S - 1u - q;
		let hp = s * BH + b * H;
		let r0 = (b * S + s) * G;
		for (var u = lid.x; u < H; u += 256u) {
			let gi = gb + hp + u;
			let r = saved[gi]; let z = saved[gi + SBH];
			let n = saved[gi + 2u * SBH]; let hn = saved[gi + 3u * SBH];
			let h_prev = saved[hp + u];
			let dh = dy[(b * S + s) * H + u] + carry[b * H + u];
			let dz = dh * (h_prev - n);
			let dn_pre = dh * (1.0 - z) * (1.0 - n * n);
			let dr = dn_pre * hn;
			let dr_pre = dr * r * (1.0 - r);
			let dz_pre = dz * z * (1.0 - z);
			d_ih[r0 + u] = dr_pre;
			d_ih[r0 + H + u] = dz_pre;
			d_ih[r0 + 2u * H + u] = dn_pre;
			d_hh[r0 + u] = dr_pre;
			d_hh[r0 + H + u] = dz_pre;
			d_hh[r0 + 2u * H + u] = dn_pre * r;
			carry[BH + b * H + u] = dh * z;
			hb[(b * S + s) * H + u] = h_prev;
		}
		storageBarrier();
		// dh of the previous step: dh · z + d_hh row @ W_hhᵀ
		for (var k = lid.x; k < H; k += 256u) {
			var acc = 0.0;
			for (var j = 0u; j < G; j++) { acc = acc + d_hh[r0 + j] * w_hh_t[j * H + k]; }
			carry[b * H + k] = carry[BH + b * H + k] + acc;
		}
		storageBarrier();
	}
	if ((p.u3 & 1u) != 0u) {
		for (var u = lid.x; u < H; u += 256u) {
			let i = b * H + u;
			dh0[i] = keep(dh0[i], p.u4 & 1u) + carry[i];
		}
	}
}
`,
	// tanh RNN over all steps: h = tanh(x W_ih + h W_hh + b); parameters, grid
	// and proj [B·S, H] as Lstm_Fwd. saved: hs [(S + 1)·B·H].
	.Rnn_Fwd            = `
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> w_hh: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> h0: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<storage, read_write> saved: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let H = p.u2;
	let b = wid.y * nw.x + wid.x;
	if (b >= B) { return; }
	let BH = B * H;
	for (var u = lid.x; u < H; u += 256u) { saved[b * H + u] = h0[b * H + u]; }
	storageBarrier();
	for (var s = 0u; s < S; s++) {
		let hp = s * BH + b * H;
		let pr = (b * S + s) * H;
		for (var u = lid.x; u < H; u += 256u) {
			var acc = 0.0;
			for (var k = 0u; k < H; k++) { acc = acc + saved[hp + k] * w_hh[k * H + u]; }
			let h = tanh_s((proj[pr + u] + acc) + bias[u]);
			saved[hp + BH + u] = h;
			o[(b * S + s) * H + u] = h;
		}
		storageBarrier();
	}
}
`,
	// tanh RNN backpropagation through time: dG [B·S, H] (pre-activation
	// gradients), hb [B·S, H] the entering states, carry [B·H] dh of the next
	// step; u3 / u4 and w_hh_t as Gru_Bwd.
	.Rnn_Bwd            = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> saved: array<f32>;
@group(0) @binding(3) var<storage, read> w_hh_t: array<f32>;
@group(0) @binding(4) var<storage, read_write> dG: array<f32>;
@group(0) @binding(5) var<storage, read_write> hb: array<f32>;
@group(0) @binding(6) var<storage, read_write> carry: array<f32>;
@group(0) @binding(7) var<storage, read_write> dh0: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let H = p.u2;
	let b = wid.y * nw.x + wid.x;
	if (b >= B) { return; }
	let BH = B * H;
	for (var u = lid.x; u < H; u += 256u) { carry[b * H + u] = 0.0; }
	storageBarrier();
	for (var q = 0u; q < S; q++) {
		let s = S - 1u - q;
		let hp = s * BH + b * H;
		let r0 = (b * S + s) * H;
		for (var u = lid.x; u < H; u += 256u) {
			let h = saved[hp + BH + u];
			let dh = dy[r0 + u] + carry[b * H + u];
			dG[r0 + u] = dh * (1.0 - h * h);
			hb[r0 + u] = saved[hp + u];
		}
		storageBarrier();
		for (var k = lid.x; k < H; k += 256u) {
			var acc = 0.0;
			for (var j = 0u; j < H; j++) { acc = acc + dG[r0 + j] * w_hh_t[j * H + k]; }
			carry[b * H + k] = acc;
		}
		storageBarrier();
	}
	if ((p.u3 & 1u) != 0u) {
		for (var u = lid.x; u < H; u += 256u) {
			let i = b * H + u;
			dh0[i] = keep(dh0[i], p.u4 & 1u) + carry[i];
		}
	}
}
`,
	// Mamba's selective scan, one thread per (batch row b, channel d, state n):
	//   h_n = exp(Δ·A_dn)·h_n + (Δ·B_n)·x,  y = x·D_d + Σ_n C_n·h_n
	// u0 batch, u1 steps S, u2 d_model, u3 d_state N (≤ 256). A workgroup
	// holds 256 / N channels; each step the thread of state 0 sums the C·h of
	// its channel in workgroup memory, in the order of n. aux: exp(Δ·A) of
	// every step [B·S·DN] (the CPU's saved layout), then the states entering
	// every step [B·S·DN] (for the backward pass).
	.Ssm_Fwd            = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> h0: array<f32>;
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read> Bm: array<f32>;
@group(0) @binding(5) var<storage, read> C: array<f32>;
@group(0) @binding(6) var<storage, read> Dl: array<f32>;
@group(0) @binding(7) var<storage, read> D: array<f32>;
@group(0) @binding(8) var<storage, read_write> o: array<f32>;
@group(0) @binding(9) var<storage, read_write> aux: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let Dm = p.u2; let N = p.u3;
	let per = 256u / N; // channels per workgroup
	let pair = (wid.y * nw.x + wid.x) * per + lid.x / N;
	let n = lid.x % N;
	let ok = lid.x < per * N && pair < B * Dm;
	let b = pair / Dm; let d = pair % Dm;
	let DN = Dm * N;
	let HS = B * S * DN;
	let e = d * N + n;
	var h = 0.0;
	if (ok) { h = h0[b * DN + e]; }
	for (var s = 0u; s < S; s++) {
		let row_x = (b * S + s) * Dm;
		let row_bc = (b * S + s) * N;
		let st = (b * S + s) * DN;
		if (ok) {
			let x_val = x[row_x + d];
			let delta = Dl[row_x + d];
			aux[HS + st + e] = h;
			let ab = exp(delta * A[e]);
			aux[st + e] = ab;
			h = ab * h + (delta * Bm[row_bc + n]) * x_val;
			sh[lid.x] = C[row_bc + n] * h;
		}
		workgroupBarrier();
		if (ok && n == 0u) {
			var y = x[row_x + d] * D[d];
			for (var k = 0u; k < N; k++) { y = y + sh[lid.x + k]; }
			o[row_x + d] = y;
		}
		workgroupBarrier();
	}
}
`,
	// Backpropagation through the scan (layout and threads as Ssm_Fwd); dh
	// stays in a register. dx and dΔ sum over n in workgroup memory (u4 bit 0
	// / 1: wanted; u5 bit 0 / 1: overwrite, as keep); dh0 += dL/dh_0 at the
	// end (bit 2). The terms of the gradients that sum over channels or steps
	// go to terms: dA terms [B·S·DN], then dB terms, then dC terms (reduced by
	// Ssm_Reduce and Col_Sum).
	.Ssm_Bwd            = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> Bm: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
@group(0) @binding(5) var<storage, read> Dl: array<f32>;
@group(0) @binding(6) var<storage, read> D: array<f32>;
@group(0) @binding(7) var<storage, read> dy: array<f32>;
@group(0) @binding(8) var<storage, read> aux: array<f32>;
@group(0) @binding(9) var<storage, read_write> terms: array<f32>;
@group(0) @binding(10) var<storage, read_write> dx: array<f32>;
@group(0) @binding(11) var<storage, read_write> dDl: array<f32>;
@group(0) @binding(12) var<storage, read_write> dh0: array<f32>;
var<workgroup> shx: array<f32, 256>;
var<workgroup> shd: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let B = p.u0; let S = p.u1; let Dm = p.u2; let N = p.u3;
	let per = 256u / N;
	let pair = (wid.y * nw.x + wid.x) * per + lid.x / N;
	let n = lid.x % N;
	let ok = lid.x < per * N && pair < B * Dm;
	let b = pair / Dm; let d = pair % Dm;
	let DN = Dm * N;
	let TS = B * S * DN;
	let e = d * N + n;
	var dh = 0.0;
	for (var q = 0u; q < S; q++) {
		let s = S - 1u - q;
		let row_x = (b * S + s) * Dm;
		let row_bc = (b * S + s) * N;
		let st = (b * S + s) * DN;
		if (ok) {
			let x_val = x[row_x + d];
			let delta = Dl[row_x + d];
			let dy_val = dy[row_x + d];
			let A_val = A[e];
			let ab = aux[st + e];
			let B_val = Bm[row_bc + n];
			let B_bar = delta * B_val;
			let hp = aux[TS + st + e];
			let dh_n = dh + dy_val * C[row_bc + n];
			let h_n = ab * hp + B_bar * x_val;
			terms[st + e] = dh_n * hp * delta * ab;
			terms[TS + st + e] = dh_n * x_val * delta;
			terms[2u * TS + st + e] = dy_val * h_n;
			dh = dh_n * ab;
			shx[lid.x] = dh_n * B_bar;
			shd[lid.x] = dh_n * (hp * A_val * ab + x_val * B_val);
		}
		workgroupBarrier();
		if (ok && n == 0u) {
			var dx_acc = dy[row_x + d] * D[d];
			var dd_acc = 0.0;
			for (var k = 0u; k < N; k++) {
				dx_acc = dx_acc + shx[lid.x + k];
				dd_acc = dd_acc + shd[lid.x + k];
			}
			if ((p.u4 & 1u) != 0u) { dx[row_x + d] = keep(dx[row_x + d], p.u5 & 1u) + dx_acc; }
			if ((p.u4 & 2u) != 0u) { dDl[row_x + d] = keep(dDl[row_x + d], (p.u5 >> 1u) & 1u) + dd_acc; }
		}
		workgroupBarrier();
	}
	if (ok && (p.u4 & 4u) != 0u) {
		let i = b * DN + e;
		dh0[i] = keep(dh0[i], (p.u5 >> 2u) & 1u) + dh;
	}
}
`,
	// The scan gradients that sum over channels or steps (parameters as
	// Ssm_Fwd): dB[b, s, n] and dC[b, s, n] sum the terms of Ssm_Bwd over the
	// channels d; dD[d] = Σ dy · x over steps and batch rows. u4 bits 0, 1, 2:
	// dB, dC, dD wanted; u5 the same bits: overwrite (as keep). One thread per
	// (b, s, n), the first d_model of them also compute dD.
	.Ssm_Reduce         = `
@group(0) @binding(1) var<storage, read> terms: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> dB: array<f32>;
@group(0) @binding(5) var<storage, read_write> dC: array<f32>;
@group(0) @binding(6) var<storage, read_write> dD: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let B = p.u0; let S = p.u1; let Dm = p.u2; let N = p.u3;
	let i = idx1(g, nw);
	let DN = Dm * N;
	let TS = B * S * DN;
	if (i < B * S * N) {
		let bs = i / N; let n = i % N;
		var sb = 0.0; var sc = 0.0;
		for (var d = 0u; d < Dm; d++) {
			let e = bs * DN + d * N + n;
			sb = sb + terms[TS + e];
			sc = sc + terms[2u * TS + e];
		}
		if ((p.u4 & 1u) != 0u) { dB[i] = keep(dB[i], p.u5 & 1u) + sb; }
		if ((p.u4 & 2u) != 0u) { dC[i] = keep(dC[i], (p.u5 >> 1u) & 1u) + sc; }
	}
	if (i < Dm && (p.u4 & 4u) != 0u) {
		var sd = 0.0;
		for (var q = 0u; q < S; q++) {
			let s = S - 1u - q;
			for (var b = 0u; b < B; b++) {
				let r = (b * S + s) * Dm + i;
				sd = sd + dy[r] * x[r];
			}
		}
		dD[i] = keep(dD[i], (p.u5 >> 2u) & 1u) + sd;
	}
}
`,
	// out[b] = Σ_j x[b * W + j]; u0 rows, u1 W
	.Row_Sum            = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let r = idx1(g, nw);
	if (r >= p.u0) { return; }
	var s = 0.0;
	for (var j = 0u; j < p.u1; j++) { s = s + x[r * p.u1 + j]; }
	o[r] = s;
}
`,
	// dx[i] += dy[i / W]; u0 n, u1 W
	.Row_Bcast_Add      = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dx[i] = keep(dx[i], p.u7) + dy[i / p.u1];
}
`,
	// x [rows, cols] -> y [cols, rows]; u0 rows, u1 cols, u2 accumulate
	.Transpose          = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0 * p.u1) { return; }
	let r = i / p.u1;
	let c = i % p.u1;
	let o = c * p.u0 + r;
	if (p.u2 == 1u) { y[o] = y[o] + x[i]; } else { y[o] = x[i]; }
}
`,
	// u0 rows, u1 len, u2 src stride, u3 src offset, u4 dst stride, u5 dst offset, u6 accumulate
	.Strided_Copy       = `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0 * p.u1) { return; }
	let r = i / p.u1;
	let c = i % p.u1;
	let v = src[r * p.u2 + p.u3 + c];
	let o = r * p.u4 + p.u5 + c;
	if (p.u6 == 1u) { dst[o] = dst[o] + v; } else { dst[o] = v; }
}
`,
	.Scale_Inplace      = `
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	y[i] = y[i] * p.f0;
}
`,
	// optim.adam_update: f0 beta1, f1 beta2, f2 epsilon, f3 lr / (1 - beta1^t),
	// f4 sqrt(1 - beta2^t), f5 1 - lr * weight_decay
	.Adam               = `
@group(0) @binding(1) var<storage, read> gr: array<f32>;
@group(0) @binding(2) var<storage, read_write> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let gi = gr[i];
	let mi = p.f0 * m[i] + (1.0 - p.f0) * gi;
	let vi = p.f1 * v[i] + (1.0 - p.f1) * gi * gi;
	m[i] = mi;
	v[i] = vi;
	w[i] = w[i] * p.f5 - p.f3 * mi / (sqrt(vi) / p.f4 + p.f2);
}
`,
	.Adam_2             = "",
	.Adam_3             = "",
	.Adam_4             = "",
	.Grad_Sq_Sum        = "",
	.Grad_Scale         = "",
	// optim.sgd_update: u1 momentum on, u2 nesterov; f0 lr, f1 momentum, f2 weight decay
	.Sgd                = `
@group(0) @binding(1) var<storage, read> gr: array<f32>;
@group(0) @binding(2) var<storage, read_write> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> vel: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	var d = gr[i] + p.f2 * w[i];
	if (p.u1 == 1u) {
		let vi = p.f1 * vel[i] + d;
		vel[i] = vi;
		d = select(vi, d + p.f1 * vi, p.u2 == 1u);
	}
	w[i] = w[i] - p.f0 * d;
}
`,
	// Rows of width C; u0 rows, u1 C, u2 mode (0 softmax into y[r, j], 1 logsumexp
	// into y[r], 2 cross-entropy loss lse - x[r, target] into y[r] with the targets
	// in aux), f0 scale applied to x first, u3 mask length (0: none; the additive
	// mask aux[(r·C + j) % u3]), u4 causal (1: entries j > r % u5 masked out), u5 seq_q
	.Softmax_Rows       = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> aux: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> sh: array<f32, 256>;
// workgroup-wide max and sum (all 256 threads must call them)
fn wg_max(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = max(sh[lid], sh[lid + w]); }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn logit(r: u32, j: u32) -> f32 {
	let C = p.u1;
	var v = x[r * C + j] * p.f0;
	if (p.u3 > 0u) { v = v + aux[(r * C + j) % p.u3]; }
	return v;
}
fn masked(r: u32, j: u32) -> bool {
	return p.u4 == 1u && j > r % p.u5;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let r = row_of(wid, nw);
	let live = r < p.u0;
	let C = p.u1;
	var m = -3.402823e38;
	if (live) {
		for (var j = lid; j < C; j += 256u) {
			if (!masked(r, j)) { m = max(m, logit(r, j)); }
		}
	}
	m = wg_max(m, lid);
	var s = 0.0;
	if (live) {
		for (var j = lid; j < C; j += 256u) {
			if (!masked(r, j)) { s = s + exp(logit(r, j) - m); }
		}
	}
	s = wg_sum(s, lid);
	if (!live) { return; }
	if (p.u2 == 0u) {
		let inv = 1.0 / s;
		for (var j = lid; j < C; j += 256u) {
			var v = 0.0;
			if (!masked(r, j)) { v = exp(logit(r, j) - m) * inv; }
			y[r * C + j] = v;
		}
	} else if (lid == 0u) {
		let lse = m + log(s);
		if (p.u2 == 1u) { y[r] = lse; } else { y[r] = lse - x[r * C + u32(aux[r])]; }
	}
}
`,
	// dx (+)= f0 * p * (g - Σ_j p g) per row; u0 rows, u1 C, u2 accumulate
	.Softmax_Bwd        = `
@group(0) @binding(1) var<storage, read> pr: array<f32>;
@group(0) @binding(2) var<storage, read> gr: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
var<workgroup> sh: array<f32, 256>;
// workgroup-wide max and sum (all 256 threads must call them)
fn wg_max(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = max(sh[lid], sh[lid + w]); }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let r = row_of(wid, nw);
	let live = r < p.u0;
	let C = p.u1;
	var d = 0.0;
	if (live) {
		for (var j = lid; j < C; j += 256u) { d = d + pr[r * C + j] * gr[r * C + j]; }
	}
	d = wg_sum(d, lid);
	if (!live) { return; }
	for (var j = lid; j < C; j += 256u) {
		let i = r * C + j;
		let v = p.f0 * pr[i] * (gr[i] - d);
		if (p.u2 == 1u) { dx[i] = dx[i] + v; } else { dx[i] = v; }
	}
}
`,
	// dx[r, j] += coef_r * (softmax(x_r)[j] - [j == target_r]); u0 rows, u1 C,
	// u2 mode (0 logsumexp: coef_r = g[r], no target; 1 cross-entropy:
	// coef = g[0] * f0, targets in t)
	.Row_Prob_Grad      = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
var<workgroup> sh: array<f32, 256>;
// workgroup-wide max and sum (all 256 threads must call them)
fn wg_max(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = max(sh[lid], sh[lid + w]); }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let r = row_of(wid, nw);
	let live = r < p.u0;
	let C = p.u1;
	var m = -3.402823e38;
	if (live) {
		for (var j = lid; j < C; j += 256u) { m = max(m, x[r * C + j]); }
	}
	m = wg_max(m, lid);
	var s = 0.0;
	if (live) {
		for (var j = lid; j < C; j += 256u) { s = s + exp(x[r * C + j] - m); }
	}
	s = wg_sum(s, lid);
	if (!live) { return; }
	var coef = 0.0;
	var tgt = 0xffffffffu;
	if (p.u2 == 1u) {
		coef = g[0] * p.f0;
		tgt = u32(t[r]);
	} else {
		coef = g[r];
	}
	for (var j = lid; j < C; j += 256u) {
		let i = r * C + j;
		let prob = exp(x[i] - m) / s;
		dx[i] = keep(dx[i], p.u7) + coef * (prob - select(0.0, 1.0, j == tgt));
	}
}
`,
	// u0 rows, u1 D, u2 1 = RMS norm (no centering, no beta), f0 eps; u3 = 1:
	// also keep the row statistics as Norm_Stats writes them (for the backward)
	.Norm_Fwd           = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;
var<workgroup> sh: array<f32, 256>;
// workgroup-wide max and sum (all 256 threads must call them)
fn wg_max(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = max(sh[lid], sh[lid + w]); }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let r = row_of(wid, nw);
	let live = r < p.u0;
	let D = p.u1;
	let rms = p.u2 == 1u;
	var s = 0.0;
	if (live && !rms) {
		for (var j = lid; j < D; j += 256u) { s = s + x[r * D + j]; }
	}
	let mean = wg_sum(s, lid) / f32(D);
	var q = 0.0;
	if (live) {
		for (var j = lid; j < D; j += 256u) { let c = x[r * D + j] - mean; q = q + c * c; }
	}
	let inv = 1.0 / sqrt(wg_sum(q, lid) / f32(D) + p.f0);
	if (!live) { return; }
	if (p.u3 == 1u && lid == 0u) {
		stats[2u * r] = mean;
		stats[2u * r + 1u] = inv;
	}
	for (var j = lid; j < D; j += 256u) {
		let i = r * D + j;
		if (rms) { y[i] = x[i] * inv * gamma[j]; } else { y[i] = (x[i] - mean) * inv * gamma[j] + beta[j]; }
	}
}
`,
	// stats[2r] = mean (0 for RMS), stats[2r + 1] = 1 / std (1 / rms)
	.Norm_Stats         = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
var<workgroup> sh: array<f32, 256>;
// workgroup-wide max and sum (all 256 threads must call them)
fn wg_max(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = max(sh[lid], sh[lid + w]); }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let r = row_of(wid, nw);
	let live = r < p.u0;
	let D = p.u1;
	let rms = p.u2 == 1u;
	var s = 0.0;
	if (live && !rms) {
		for (var j = lid; j < D; j += 256u) { s = s + x[r * D + j]; }
	}
	let mean = wg_sum(s, lid) / f32(D);
	var q = 0.0;
	if (live) {
		for (var j = lid; j < D; j += 256u) { let c = x[r * D + j] - mean; q = q + c * c; }
	}
	let inv = 1.0 / sqrt(wg_sum(q, lid) / f32(D) + p.f0);
	if (live && lid == 0u) {
		stats[2u * r] = mean;
		stats[2u * r + 1u] = inv;
	}
}
`,
	// Layer norm: dx += (D·a - Σa - x̂·Σ(a x̂)) · inv / D with a = dy·γ;
	// RMS norm: dx += r·a - x·r³·Σ(a x)/D
	.Norm_Bwd_X         = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
var<workgroup> sh: array<f32, 256>;
// workgroup-wide max and sum (all 256 threads must call them)
fn wg_max(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = max(sh[lid], sh[lid + w]); }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let r = row_of(wid, nw);
	let live = r < p.u0;
	let D = p.u1;
	let rms = p.u2 == 1u;
	var mean = 0.0;
	var inv = 0.0;
	if (live) { mean = stats[2u * r]; inv = stats[2u * r + 1u]; }
	var sa = 0.0;
	var sax = 0.0;
	if (live) {
		for (var j = lid; j < D; j += 256u) {
			let i = r * D + j;
			let a = dy[i] * gamma[j];
			sa = sa + a;
			sax = sax + a * (x[i] - mean) * select(inv, 1.0, rms);
		}
	}
	sa = wg_sum(sa, lid);
	sax = wg_sum(sax, lid);
	if (!live) { return; }
	let Df = f32(D);
	for (var j = lid; j < D; j += 256u) {
		let i = r * D + j;
		let a = dy[i] * gamma[j];
		if (rms) {
			dx[i] = keep(dx[i], p.u7) + inv * a - x[i] * inv * inv * inv * sax / Df;
		} else {
			let xh = (x[i] - mean) * inv;
			dx[i] = keep(dx[i], p.u7) + (Df * a - sa - xh * sax) * inv / Df;
		}
	}
}
`,
	// per column j: u4 = 1: dgamma[j] += Σ_r dy·x̂; u3 = 1: dbeta[j] += Σ_r dy;
	// u6, u7: overwrite dgamma, dbeta
	.Norm_Bwd_Param     = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbeta: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let j = idx1(g, nw);
	let D = p.u1;
	if (j >= D) { return; }
	var sg = 0.0;
	var sb = 0.0;
	for (var r = 0u; r < p.u0; r++) {
		let i = r * D + j;
		sg = sg + dy[i] * (x[i] - stats[2u * r]) * stats[2u * r + 1u];
		sb = sb + dy[i];
	}
	if (p.u4 == 1u) { dgamma[j] = keep(dgamma[j], p.u6) + sg; }
	if (p.u3 == 1u) { dbeta[j] = keep(dbeta[j], p.u7) + sb; }
}
`,
	// out[r, e] = W[ids[r], e] (0 for an id outside [0, u2)); u0 rows, u1 E, u2 vocab
	.Gather_Rows        = `
@group(0) @binding(1) var<storage, read> ids: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0 * p.u1) { return; }
	let r = i / p.u1;
	let e = i % p.u1;
	let id = i32(ids[r]);
	var v = 0.0;
	if (id >= 0 && u32(id) < p.u2) { v = W[u32(id) * p.u1 + e]; }
	o[i] = v;
}
`,
	// dW[ids[r], e] += dy[r, e]; rows sharing an id add atomically (compare-and-swap)
	.Scatter_Add_Rows   = `
@group(0) @binding(1) var<storage, read> ids: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dW: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0 * p.u1) { return; }
	let r = i / p.u1;
	let e = i % p.u1;
	let id = i32(ids[r]);
	if (id < 0 || u32(id) >= p.u2) { return; }
	let a = u32(id) * p.u1 + e;
	let v = dy[i];
	var old = atomicLoad(&dW[a]);
	loop {
		let res = atomicCompareExchangeWeak(&dW[a], old, bitcast<u32>(bitcast<f32>(old) + v));
		if (res.exchanged) { break; }
		old = res.old_value;
	}
}
`,
	// element (b, h, s, d): sequence layout b·S·H·hd + s·H·hd + h·hd + d, head
	// layout (b·H + h)·S·hd + s·hd + d; u0 B, u1 S, u2 H, u3 hd, u4 0: head <-
	// sequence, 1: sequence <- head; u5 accumulate
	.Permute_Mha        = `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let B = p.u0; let S = p.u1; let H = p.u2; let hd = p.u3;
	if (i >= B * S * H * hd) { return; }
	let d = i % hd;
	let s = (i / hd) % S;
	let bh = i / (hd * S);
	let h = bh % H;
	let b = bh / H;
	let seq = b * S * H * hd + s * H * hd + h * hd + d;
	var si = seq;
	var di = i;
	if (p.u4 == 1u) { si = i; di = seq; }
	if (p.u5 == 1u) { dst[di] = dst[di] + src[si]; } else { dst[di] = src[si]; }
}
`,
	// p clamped to [1e-7, 1 - 1e-7]; u0 n, u1 mode (0 loss per element into dst,
	// 1 dpred += f0 g[0] (-(t/p) + (1-t)/(1-p)), 2 dtarget += f0 g[0] (ln(1-p) - ln p))
	.Bce                = `
@group(0) @binding(1) var<storage, read> pr: array<f32>;
@group(0) @binding(2) var<storage, read> tg: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(gi, nw);
	if (i >= p.u0) { return; }
	let q = clamp(pr[i], 1e-7, 1.0 - 1e-7);
	let t = tg[i];
	if (p.u1 == 0u) {
		dst[i] = -(t * log(q) + (1.0 - t) * log(1.0 - q));
	} else if (p.u1 == 1u) {
		dst[i] = dst[i] + p.f0 * g[0] * (-t / q + (1.0 - t) / (1.0 - q));
	} else {
		dst[i] = dst[i] + p.f0 * g[0] * (log(1.0 - q) - log(q));
	}
}
`,
	// u0 n, u1 D, u2 accumulate
	.Row_Scale          = `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(gi, nw);
	if (i >= p.u0) { return; }
	let v = src[i] * g[i / p.u1];
	if (p.u2 == 1u) { dst[i] = dst[i] + v; } else { dst[i] = v; }
}
`,
	// u0 rows, u1 W
	.Row_Dot            = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let r = idx1(gi, nw);
	if (r >= p.u0) { return; }
	var s = 0.0;
	for (var j = 0u; j < p.u1; j++) { s = s + a[r * p.u1 + j] * b[r * p.u1 + j]; }
	o[r] = keep(o[r], p.u7) + s;
}
`,
	// ---- phase 3 ----
	// col[k, n·L + l] = x[n, c, oh·s + kh - pad, ow·s + kw - pad] (0 in the padding),
	// k = (c·kH + kh)·kW + kw, l = oh·out_w + ow (the CPU _im2col order per sample).
	// u0 N, u1 C, u2 H, u3 W, u4 kH, u5 kW, u6 stride, u7 pad
	.Im2col_2d          = CONV2D_DIMS + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let d = conv2d_dims();
	let NL = p.u0 * d.L;
	if (i >= p.u1 * p.u4 * p.u5 * NL) { return; }
	let k = i / NL;
	let j = i % NL;
	let n = j / d.L;
	let l = j % d.L;
	let c = k / (p.u4 * p.u5);
	let kh = (k / p.u5) % p.u4;
	let kw = k % p.u5;
	let ih = i32((l / d.ow) * p.u6 + kh) - i32(p.u7);
	let iw = i32((l % d.ow) * p.u6 + kw) - i32(p.u7);
	var v = 0.0;
	if (ih >= 0 && ih < i32(p.u2) && iw >= 0 && iw < i32(p.u3)) {
		v = x[((n * p.u1 + c) * p.u2 + u32(ih)) * p.u3 + u32(iw)];
	}
	col[i] = v;
}
`,
	// dx[n, c, ih, iw] += Σ_{kh, kw} dcol[k, n·L + l] over the outputs that read
	// (ih, iw); parameters as Im2col_2d
	.Col2im_2d          = CONV2D_DIMS + `
@group(0) @binding(1) var<storage, read> dcol: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let N = p.u0; let C = p.u1; let H = p.u2; let W = p.u3;
	if (i >= N * C * H * W) { return; }
	let d = conv2d_dims();
	let iw = i32(i % W);
	let ih = i32((i / W) % H);
	let c = (i / (W * H)) % C;
	let n = i / (W * H * C);
	let s = i32(p.u6);
	var acc = 0.0;
	for (var kh = 0u; kh < p.u4; kh++) {
		let th = ih + i32(p.u7) - i32(kh);
		if (th < 0 || th % s != 0 || th / s >= i32(d.oh)) { continue; }
		for (var kw = 0u; kw < p.u5; kw++) {
			let tw = iw + i32(p.u7) - i32(kw);
			if (tw < 0 || tw % s != 0 || tw / s >= i32(d.ow)) { continue; }
			let k = (c * p.u4 + kh) * p.u5 + kw;
			let l = u32(th / s) * d.ow + u32(tw / s);
			acc = acc + dcol[k * N * d.L + n * d.L + l];
		}
	}
	dx[i] = dx[i] + acc;
}
`,
	// col[b·T_out + t, c·K + k] = x[b, t + k·dil - pad_left, c] (0 outside);
	// u0 B, u1 T, u2 C_in, u3 K, u4 dilation, u5 pad_left, u6 T_out
	.Im2col_1d          = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let CK = p.u2 * p.u3;
	if (i >= p.u0 * p.u6 * CK) { return; }
	let r = i / CK;
	let c = (i % CK) / p.u3;
	let k = i % p.u3;
	let b = r / p.u6;
	let src = i32((r % p.u6) * p.u7 + k * p.u4) - i32(p.u5);
	var v = 0.0;
	if (src >= 0 && src < i32(p.u1)) { v = x[(b * p.u1 + u32(src)) * p.u2 + c]; }
	col[i] = v;
}
`,
	// dx[b, s, c] += Σ_k dcol[b·T_out + t, c·K + k] with t = s - k·dil + pad_left;
	// parameters as Im2col_1d
	.Col2im_1d          = `
@group(0) @binding(1) var<storage, read> dcol: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let C = p.u2; let K = p.u3;
	if (i >= p.u0 * p.u1 * C) { return; }
	let c = i % C;
	let s = (i / C) % p.u1;
	let b = i / (C * p.u1);
	var acc = 0.0;
	for (var k = 0u; k < K; k++) {
		let num = i32(s + p.u5) - i32(k * p.u4);
		if (num < 0 || u32(num) % p.u7 != 0u) { continue; }
		let t = u32(num) / p.u7;
		if (t >= p.u6) { continue; }
		acc = acc + dcol[(b * p.u6 + t) * C * K + c * K + k];
	}
	dx[i] = dx[i] + acc;
}
`,
	// Grouped conv1d (groups > 1), computed directly. u0 B, u1 T, u2 C_in, u3 K,
	// u4 dilation, u5 pad_left, u6 T_out, u7 stride | groups << 16, f0 C_out,
	// f1 1: overwrite the gradient (see keep). One thread per output element.
	.Conv1d_Grouped     = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let st = p.u7 & 0xFFFFu; let G = p.u7 >> 16u; let Co = u32(p.f0);
	if (i >= p.u0 * p.u6 * Co) { return; }
	let K = p.u3;
	let o = i % Co;
	let t = (i / Co) % p.u6;
	let b = i / (Co * p.u6);
	let Cig = p.u2 / G;
	let g0 = (o / (Co / G)) * Cig;
	var acc = 0.0;
	for (var k = 0u; k < K; k++) {
		let src = i32(t * st + k * p.u4) - i32(p.u5);
		if (src < 0 || src >= i32(p.u1)) { continue; }
		let xb = (b * p.u1 + u32(src)) * p.u2 + g0;
		let wb = o * Cig * K + k;
		for (var c = 0u; c < Cig; c++) { acc = acc + w[wb + c * K] * x[xb + c]; }
	}
	y[i] = acc;
}
`,
	// its input gradient: one thread per element of x
	.Conv1d_Grouped_Dx  = `
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let st = p.u7 & 0xFFFFu; let G = p.u7 >> 16u; let Co = u32(p.f0);
	let Ci = p.u2; let K = p.u3;
	if (i >= p.u0 * p.u1 * Ci) { return; }
	let c = i % Ci;
	let s = (i / Ci) % p.u1;
	let b = i / (Ci * p.u1);
	let Cig = Ci / G; let Cog = Co / G;
	let cc = c % Cig;
	let o0 = (c / Cig) * Cog;
	var acc = 0.0;
	for (var k = 0u; k < K; k++) {
		let num = i32(s + p.u5) - i32(k * p.u4);
		if (num < 0 || u32(num) % st != 0u) { continue; }
		let t = u32(num) / st;
		if (t >= p.u6) { continue; }
		let db = (b * p.u6 + t) * Co;
		for (var q = 0u; q < Cog; q++) {
			let o = o0 + q;
			acc = acc + w[(o * Cig + cc) * K + k] * dy[db + o];
		}
	}
	dx[i] = keep(dx[i], u32(p.f1)) + acc;
}
`,
	// its weight gradient: one workgroup per weight element, reducing over the
	// batch and the output steps
	.Conv1d_Grouped_Dw  = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = row_of(wid, nw);
	let st = p.u7 & 0xFFFFu; let G = p.u7 >> 16u; let Co = u32(p.f0);
	let Ci = p.u2; let K = p.u3;
	let Cig = Ci / G;
	if (i >= Co * Cig * K) { return; }
	let k = i % K;
	let cc = (i / K) % Cig;
	let o = i / (K * Cig);
	let xc = (o / (Co / G)) * Cig + cc;
	var acc = 0.0;
	let R = p.u0 * p.u6;
	for (var r = lid.x; r < R; r = r + 256u) {
		let t = r % p.u6;
		let b = r / p.u6;
		let src = i32(t * st + k * p.u4) - i32(p.u5);
		if (src < 0 || src >= i32(p.u1)) { continue; }
		acc = acc + x[(b * p.u1 + u32(src)) * Ci + xc] * dy[r * Co + o];
	}
	sh[lid.x] = acc;
	workgroupBarrier();
	for (var h = 128u; h > 0u; h = h >> 1u) {
		if (lid.x < h) { sh[lid.x] = sh[lid.x] + sh[lid.x + h]; }
		workgroupBarrier();
	}
	if (lid.x == 0u) { dw[i] = keep(dw[i], u32(p.f1)) + sh[0]; }
}
`,
	// dst[i] += b[(i / u1) % u2]; u0 n, u1 L (elements per channel block), u2 C
	.Chan_Bias          = `
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dst[i] = dst[i] + b[(i / p.u1) % p.u2];
}
`,
	// one workgroup per channel c: db[c] += Σ_{n, l} dy[(n·C + c)·L + l];
	// u0 C, u1 L, u2 N
	.Chan_Sum           = WG_SUM + `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let c = row_of(wid, nw);
	let live = c < p.u0;
	let L = p.u1;
	var s = 0.0;
	if (live) {
		for (var j = lid; j < p.u2 * L; j += 256u) { s = s + dy[((j / L) * p.u0 + c) * L + j % L]; }
	}
	s = wg_sum(s, lid);
	if (live && lid == 0u) { db[c] = keep(db[c], p.u7) + s; }
}
`,
	// planes [u0 = N·C, H = u1, W = u2], window u3 x u4, stride u5; u6 0 max, 1 average.
	// Max takes the first largest value of the window in row-major order (as the CPU).
	.Pool               = POOL_DIMS + `
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let d = pool_dims();
	if (i >= p.u0 * d.oh * d.ow) { return; }
	let plane = i / (d.oh * d.ow);
	let oh = (i / d.ow) % d.oh;
	let ow = i % d.ow;
	if (p.u6 == 0u) {
		y[i] = x[window_argmax(plane, oh, ow)];
	} else {
		var s = 0.0;
		for (var kh = 0u; kh < p.u3; kh++) {
			for (var kw = 0u; kw < p.u4; kw++) {
				s = s + x[(plane * p.u1 + oh * p.u5 + kh) * p.u2 + ow * p.u5 + kw];
			}
		}
		y[i] = s / f32(p.u3 * p.u4);
	}
}
`,
	// dx[e] += Σ over the windows containing e: dy (max: only where e is the
	// window's argmax; average: dy / (kH·kW)); parameters as Pool
	.Pool_Bwd           = POOL_DIMS + `
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	let H = p.u1; let W = p.u2; let s = p.u5;
	if (i >= p.u0 * H * W) { return; }
	let d = pool_dims();
	let plane = i / (H * W);
	let ih = (i / W) % H;
	let iw = i % W;
	var acc = 0.0;
	var oh = i32(min(ih / s, d.oh - 1u));
	loop {
		if (oh < 0 || u32(oh) * s + p.u3 <= ih) { break; }
		var ow = i32(min(iw / s, d.ow - 1u));
		loop {
			if (ow < 0 || u32(ow) * s + p.u4 <= iw) { break; }
			let o = (plane * d.oh + u32(oh)) * d.ow + u32(ow);
			if (p.u6 == 1u) {
				acc = acc + dy[o] / f32(p.u3 * p.u4);
			} else if (window_argmax(plane, u32(oh), u32(ow)) == i) {
				acc = acc + dy[o];
			}
			ow = ow - 1;
		}
		oh = oh - 1;
	}
	dx[i] = keep(dx[i], p.u7) + acc;
}
`,
	// one workgroup per channel c of x [N, C, HW]. Training (u3 = 1): batch mean
	// and biased variance; the running statistics are updated with momentum f0
	// (unbiased variance, as torch). Evaluation: the running statistics.
	// stats[2c] = mean, stats[2c + 1] = 1 / sqrt(var + eps); u0 C, u1 HW, u2 N, f1 eps
	.Bn_Stats           = WG_SUM + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> rm: array<f32>;
@group(0) @binding(3) var<storage, read_write> rv: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let c = row_of(wid, nw);
	let live = c < p.u0;
	let HW = p.u1;
	let M = p.u2 * HW;
	if (p.u3 == 0u) {
		if (live && lid == 0u) {
			stats[c] = rm[c];
			stats[p.u0 + c] = 1.0 / sqrt(rv[c] + p.f1);
		}
		return;
	}
	var s = 0.0;
	if (live) {
		for (var j = lid; j < M; j += 256u) { s = s + x[((j / HW) * p.u0 + c) * HW + j % HW]; }
	}
	let mean = wg_sum(s, lid) / f32(M);
	var q = 0.0;
	if (live) {
		for (var j = lid; j < M; j += 256u) {
			let e = x[((j / HW) * p.u0 + c) * HW + j % HW] - mean;
			q = q + e * e;
		}
	}
	let v = wg_sum(q, lid) / f32(M);
	if (live && lid == 0u) {
		stats[c] = mean;
		stats[p.u0 + c] = 1.0 / sqrt(v + p.f1);
		var unbiased = v;
		if (M > 1u) { unbiased = v * f32(M) / f32(M - 1u); }
		rm[c] = (1.0 - p.f0) * rm[c] + p.f0 * mean;
		rv[c] = (1.0 - p.f0) * rv[c] + p.f0 * unbiased;
	}
}
`,
	// stats [means of the C channels, then 1 / std]; u0 n, u1 HW, u2 C
	.Bn_Apply           = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let c = (i / p.u1) % p.u2;
	y[i] = (x[i] - stats[c]) * stats[p.u2 + c] * gamma[c] + beta[c];
}
`,
	// one workgroup per channel: sums[2c] = Σ dy, sums[2c + 1] = Σ dy·x̂;
	// u3 = 1: dgamma[c] += Σ dy·x̂, u4 = 1: dbeta[c] += Σ dy; u0 C, u1 HW, u2 N;
	// u5, u6: overwrite dgamma, dbeta
	.Bn_Bwd_Sums        = WG_SUM + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> sums: array<f32>;
@group(0) @binding(5) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(6) var<storage, read_write> dbeta: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let c = row_of(wid, nw);
	let live = c < p.u0;
	let HW = p.u1;
	var mean = 0.0;
	var inv = 0.0;
	if (live) { mean = stats[c]; inv = stats[p.u0 + c]; }
	var sd = 0.0;
	var sdx = 0.0;
	if (live) {
		for (var j = lid; j < p.u2 * HW; j += 256u) {
			let e = ((j / HW) * p.u0 + c) * HW + j % HW;
			sd = sd + dy[e];
			sdx = sdx + dy[e] * (x[e] - mean) * inv;
		}
	}
	sd = wg_sum(sd, lid);
	sdx = wg_sum(sdx, lid);
	if (live && lid == 0u) {
		sums[2u * c] = sd;
		sums[2u * c + 1u] = sdx;
		if (p.u3 == 1u) { dgamma[c] = keep(dgamma[c], p.u5) + sdx; }
		if (p.u4 == 1u) { dbeta[c] = keep(dbeta[c], p.u6) + sd; }
	}
}
`,
	// training: dx += γ·inv / M · (M·dy - Σdy - x̂·Σ(dy·x̂)); evaluation (the
	// statistics are constants): dx += γ·inv·dy. u0 n, u1 HW, u2 C, u3 training, u4 M
	.Bn_Bwd_X           = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read> sums: array<f32>;
@group(0) @binding(6) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	let c = (i / p.u1) % p.u2;
	let inv = stats[p.u2 + c];
	let gi = gamma[c] * inv;
	if (p.u3 == 0u) {
		dx[i] = keep(dx[i], p.u7) + dy[i] * gi;
		return;
	}
	let M = f32(p.u4);
	let xh = (x[i] - stats[c]) * inv;
	dx[i] = keep(dx[i], p.u7) + gi / M * (M * dy[i] - sums[2u * c] - xh * sums[2u * c + 1u]);
}
`,
	// ---- phase 4 ----
	// dst[i] = term of a scalar loss; u0 n, u1 mode: 0 KL (a mu, b log_var)
	// 1 + b - a² - min(e^b, 1e10); 1 entropy -a·ln a (0 for a ≤ 1e-10); 2 prospect
	// value v(a) with f1 α, f2 β, f3 λ
	.Loss_Terms         = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(gi, nw);
	if (i >= p.u0) { return; }
	let v = a[i];
	var r = 0.0;
	if (p.u1 == 0u) {
		r = 1.0 + b[i] - v * v - min(exp(b[i]), 1e10);
	} else if (p.u1 == 1u) {
		if (v > 1e-10) { r = -v * log(v); }
	} else {
		r = prospect_v(v, p.f1, p.f2, p.f3);
	}
	dst[i] = r;
}
`,
	// dx[i] += g[0] · f0 · d; u0 n, u1 mode: 0 KL ∂mu (d = a), 1 KL ∂log_var
	// (d = e^b - 1, e^b taken as 0 above 1e10), 2 entropy (d = -(1 + ln a), 0 for
	// a ≤ 1e-10), 3 prospect utility (d = -v'(a), f1 α, f2 β, f3 λ), 4 Sharpe
	// (aux = [mean, std], f1 risk-free rate): d = ((mean - rf)(a - mean) - std²) / (n std³)
	.Loss_Grad          = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> aux: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(gi, nw);
	if (i >= p.u0) { return; }
	let v = a[i];
	var d = 0.0;
	switch p.u1 {
		case 0u: { d = v; }
		case 1u: { var e = exp(b[i]); if (e > 1e10) { e = 0.0; } d = e - 1.0; }
		case 2u: { if (v > 1e-10) { d = -(1.0 + log(v)); } }
		case 3u: { d = -prospect_dv(v, p.f1, p.f2, p.f3); }
		case 4u: {
			let mean = aux[0]; let sd = aux[1];
			d = ((mean - p.f1) * (v - mean) - sd * sd) / (f32(p.u0) * sd * sd * sd);
		}
		default: {}
	}
	dx[i] = dx[i] + g[0] * p.f0 * d;
}
`,
	// one workgroup: o[0] = -(mean - f0) / sqrt(var + 1e-8), aux = [mean, std]; u0 n
	.Sharpe             = WG_SUM + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<storage, read_write> aux: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>) {
	let lid = l3.x;
	let n = p.u0;
	var s = 0.0;
	for (var j = lid; j < n; j += 256u) { s = s + x[j]; }
	let mean = wg_sum(s, lid) / f32(n);
	var q = 0.0;
	for (var j = lid; j < n; j += 256u) { let e = x[j] - mean; q = q + e * e; }
	let sd = sqrt(wg_sum(q, lid) / f32(n) + 1e-8);
	if (lid == 0u) {
		o[0] = -(mean - p.f0) / sd;
		aux[0] = mean;
		aux[1] = sd;
	}
}
`,
	// one workgroup per (n, c, l) slice of x [N, C, T, L], elements L apart.
	// u4 = 0: dst = (x - mean) / sqrt(var + eps); u4 = 1: dst += (dy - mean(dy) -
	// y·Σ(dy·y)/T) / sqrt(var + eps) with y the forward output.
	// u0 slices N·C·L, u1 C, u2 T, u3 L, f0 eps
	.Norm_Time          = WG_SUM + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let lid = l3.x;
	let sl = row_of(wid, nw);
	let live = sl < p.u0;
	let C = p.u1; let T = p.u2; let L = p.u3;
	let n = sl / (C * L);
	let c = (sl / L) % C;
	let el = sl % L;
	let base = (n * C + c) * T * L + el;
	var s = 0.0;
	if (live) { for (var t = lid; t < T; t += 256u) { s = s + x[base + t * L]; } }
	let mean = wg_sum(s, lid) / f32(T);
	var q = 0.0;
	if (live) { for (var t = lid; t < T; t += 256u) { let e = x[base + t * L] - mean; q = q + e * e; } }
	let inv = 1.0 / sqrt(wg_sum(q, lid) / f32(T) + p.f0);
	if (p.u4 == 0u) {
		if (live) { for (var t = lid; t < T; t += 256u) { dst[base + t * L] = (x[base + t * L] - mean) * inv; } }
		return;
	}
	var sg = 0.0;
	var sgy = 0.0;
	if (live) {
		for (var t = lid; t < T; t += 256u) {
			let k = base + t * L;
			sg = sg + dy[k];
			sgy = sgy + dy[k] * y[k];
		}
	}
	let mg = wg_sum(sg, lid) / f32(T);
	let dgy = wg_sum(sgy, lid) / f32(T);
	if (live) {
		for (var t = lid; t < T; t += 256u) {
			let k = base + t * L;
			dst[k] = dst[k] + (dy[k] - mg - y[k] * dgy) * inv;
		}
	}
}
`,
	// per row r of probs [N, E]: mask 1 at the k largest entries (ties: the lower
	// index), 0 elsewhere; u0 N, u1 E, u2 k
	.Top_K              = `
@group(0) @binding(1) var<storage, read> pr: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let r = idx1(gi, nw);
	if (r >= p.u0) { return; }
	let E = p.u1;
	let o = r * E;
	for (var e = 0u; e < E; e++) { mask[o + e] = 0.0; }
	for (var k = 0u; k < p.u2; k++) {
		var best = -1;
		for (var e = 0u; e < E; e++) {
			if (mask[o + e] == 0.0 && (best < 0 || pr[o + e] > pr[o + u32(best)])) { best = i32(e); }
		}
		mask[o + u32(best)] = 1.0;
	}
}
`,
	// u1 = 0: dst = a + exp(b / 2)·c (a mu, b log_var, c eps);
	// u1 = 1: dst += f0·c·(a - b) (a z, b mu, c dz: the log_var gradient with f0 = 1/2)
	.Reparam            = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(gi, nw);
	if (i >= p.u0) { return; }
	if (p.u1 == 0u) {
		dst[i] = a[i] + exp(0.5 * b[i]) * c[i];
	} else {
		dst[i] = dst[i] + p.f0 * c[i] * (a[i] - b[i]);
	}
}
`,
}

// workgroup-wide sum for the one-workgroup-per-row kernels (all 256 threads
// must call it)
@(private)
WG_SUM :: `
var<workgroup> sh: array<f32, 256>;
fn wg_sum(v: f32, lid: u32) -> f32 {
	sh[lid] = v;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid < w) { sh[lid] = sh[lid] + sh[lid + w]; }
		workgroupBarrier();
	}
	let r = sh[0];
	workgroupBarrier();
	return r;
}
`

// conv2d output size from the Im2col_2d parameters
@(private)
CONV2D_DIMS :: `
struct Conv_Dims { oh: u32, ow: u32, L: u32, }
fn conv2d_dims() -> Conv_Dims {
	let oh = (p.u2 + 2u * p.u7 - p.u4) / p.u6 + 1u;
	let ow = (p.u3 + 2u * p.u7 - p.u5) / p.u6 + 1u;
	return Conv_Dims(oh, ow, oh * ow);
}
`

// pooling input (binding 1), output size and the argmax of a window (Pool parameters)
@(private)
POOL_DIMS :: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
struct Pool_Dims { oh: u32, ow: u32, }
fn pool_dims() -> Pool_Dims {
	return Pool_Dims((p.u1 - p.u3) / p.u5 + 1u, (p.u2 - p.u4) / p.u5 + 1u);
}
fn window_argmax(plane: u32, oh: u32, ow: u32) -> u32 {
	var best = -3.402823e38;
	var arg = 0u;
	for (var kh = 0u; kh < p.u3; kh++) {
		for (var kw = 0u; kw < p.u4; kw++) {
			let e = (plane * p.u1 + oh * p.u5 + kh) * p.u2 + ow * p.u5 + kw;
			if (x[e] > best) { best = x[e]; arg = e; }
		}
	}
	return arg;
}
`

// parameters per launch of the multi-parameter Adam kernels
@(private)
ADAM_GROUP :: 4

@(private)
_adam_kernel :: proc(count: int) -> Kernel {
	switch count {
	case 2:
		return .Adam_2
	case 3:
		return .Adam_3
	case 4:
		return .Adam_4
	}
	return .Adam
}

// Where matrix z of a batched matmul operand stored as [rows, cols] starts, and
// its row stride (flags: u3 for A, u4 for B, u5 for C). The contiguous form is
// the z-th block; the address arithmetic inlines to that of a plain kernel.
@(private)
MATMUL_CONTIGUOUS :: `
fn base(z: u32, rows: u32, cols: u32, flags: u32) -> u32 { return z * rows * cols; }
fn stride(cols: u32, flags: u32) -> u32 { return cols; }
`

// With flag bit 1 an operand is in head layout: the matrices of z = b·H + h
// interleave as [B, rows, H, cols] (H = u7 >> 1), so attention reads and writes
// [B, S, H·hd] directly. These are separate kernels so that plain products keep
// their address arithmetic (one kernel for both forms made large register-tiled
// products about 20% slower).
@(private)
MATMUL_HEADS :: `
fn base(z: u32, rows: u32, cols: u32, flags: u32) -> u32 {
	if ((flags & 2u) == 0u) { return z * rows * cols; }
	let H = p.u7 >> 1u;
	return (z / H) * rows * H * cols + (z % H) * cols;
}
fn stride(cols: u32, flags: u32) -> u32 {
	return select(cols, (p.u7 >> 1u) * cols, (flags & 2u) != 0u);
}
`

// _kernel_source: the WGSL body of a kernel (generated for the matmul address
// forms and the multi-parameter Adam)
@(private)
_kernel_source :: proc(k: Kernel, allocator := context.temp_allocator) -> string {
	#partial switch k {
	case .Matmul, .Matmul_Tiled:
		return strings.concatenate({MATMUL_CONTIGUOUS, KERNEL_SOURCES[k]}, allocator)
	case .Matmul_Heads:
		return strings.concatenate({MATMUL_HEADS, KERNEL_SOURCES[.Matmul]}, allocator)
	case .Matmul_Tiled_Heads:
		return strings.concatenate({MATMUL_HEADS, KERNEL_SOURCES[.Matmul_Tiled]}, allocator)
	case .Matmul_Group:
		return _matmul_group_source(false, allocator)
	case .Matmul_Tiled_Group:
		return _matmul_group_source(true, allocator)
	case .Adam_2:
		return _adam_source(2, allocator)
	case .Adam_3:
		return _adam_source(3, allocator)
	case .Adam_4:
		return _adam_source(4, allocator)
	case .Grad_Sq_Sum:
		return _grad_source(false, allocator)
	case .Grad_Scale:
		return _grad_source(true, allocator)
	}
	return KERNEL_SOURCES[k]
}

// Grouped products: up to LINEAR_GROUP products m in one launch, with A, B, C
// and the bias of product m at bindings 1 + m, 4 + m, 7 + m and 10 + m
// (unused ones bound to A0 or C0). Parameters as Matmul, except:
//   u5 bit m: product m accumulates into C (else overwrites it)
//   u7 bit 0: add bias[column]; bit 1: sum mode; bit 2: A per product (else
//   A0 for all); bits 4..: number of products
// Separate mode: product m = workgroup z writes C_m. Sum mode (one z): every
// workgroup runs the products one after the other and each one adds into C0,
// the same values as one launch per product. The source is that of Matmul or
// Matmul_Tiled with the operand reads and the stores replaced, so the plain
// kernels keep their code.
@(private)
MATMUL_GROUP :: `
@group(0) @binding(1) var<storage, read> A0: array<f32>;
@group(0) @binding(2) var<storage, read> A1: array<f32>;
@group(0) @binding(3) var<storage, read> A2: array<f32>;
@group(0) @binding(4) var<storage, read> B0: array<f32>;
@group(0) @binding(5) var<storage, read> B1: array<f32>;
@group(0) @binding(6) var<storage, read> B2: array<f32>;
@group(0) @binding(7) var<storage, read_write> C0: array<f32>;
@group(0) @binding(8) var<storage, read_write> C1: array<f32>;
@group(0) @binding(9) var<storage, read_write> C2: array<f32>;
@group(0) @binding(10) var<storage, read> bias0: array<f32>;
@group(0) @binding(11) var<storage, read> bias1: array<f32>;
@group(0) @binding(12) var<storage, read> bias2: array<f32>;
// the product being computed (set by main; the loop runs on a local copy so
// that the barriers stay in uniform control flow)
var<private> gm: u32;
fn base(z: u32, rows: u32, cols: u32, flags: u32) -> u32 { return 0u; }
fn stride(cols: u32, flags: u32) -> u32 { return cols; }
fn group_sum() -> bool { return (p.u7 & 2u) != 0u; }
fn group_first(z: u32) -> u32 { return select(z, 0u, group_sum()); }
fn group_end(z: u32) -> u32 { return select(z + 1u, p.u7 >> 4u, group_sum()); }
fn get_a(i: u32) -> f32 {
	switch select(0u, gm, (p.u7 & 4u) != 0u) {
		case 1u: { return A1[i]; }
		case 2u: { return A2[i]; }
		default: { return A0[i]; }
	}
}
fn get_b(i: u32) -> f32 {
	switch gm {
		case 1u: { return B1[i]; }
		case 2u: { return B2[i]; }
		default: { return B0[i]; }
	}
}
fn get_bias(i: u32) -> f32 {
	switch gm {
		case 1u: { return bias1[i]; }
		case 2u: { return bias2[i]; }
		default: { return bias0[i]; }
	}
}
fn put_c(o: u32, v: f32) {
	let acc = ((p.u5 >> gm) & 1u) == 1u;
	switch select(gm, 0u, group_sum()) {
		case 1u: { if (acc) { C1[o] = C1[o] + v; } else { C1[o] = v; } }
		case 2u: { if (acc) { C2[o] = C2[o] + v; } else { C2[o] = v; } }
		default: { if (acc) { C0[o] = C0[o] + v; } else { C0[o] = v; } }
	}
}
`

// _matmul_group_source: Matmul (tiled = false) or Matmul_Tiled rewritten for
// grouped products (see MATMUL_GROUP)
@(private)
_matmul_group_source :: proc(tiled: bool, allocator := context.temp_allocator) -> string {
	src := KERNEL_SOURCES[.Matmul_Tiled if tiled else .Matmul]
	rep :: proc(s: string, old, new: string) -> string {
		assert(strings.count(s, old) == 1, "gpu: grouped matmul source out of date")
		r, _ := strings.replace(s, old, new, 1, context.temp_allocator)
		return r
	}
	// A[...] and B[...] become get_a(...) and get_b(...) (no nested brackets)
	wrap :: proc(s: string, name, fn: string) -> string {
		b := strings.builder_make(context.temp_allocator)
		rest := s
		for {
			at := strings.index(rest, name)
			if at < 0 {break}
			end := strings.index_byte(rest[at:], ']')
			strings.write_string(&b, rest[:at])
			strings.write_string(&b, fn)
			strings.write_byte(&b, '(')
			strings.write_string(&b, rest[at + len(name):at + end])
			strings.write_byte(&b, ')')
			rest = rest[at + end + 1:]
		}
		strings.write_string(&b, rest)
		return strings.to_string(b)
	}
	src = rep(
		src,
		`@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
`,
		"",
	)
	src = wrap(src, " A[", " get_a")
	src = wrap(src, " B[", " get_b")
	src = rep(src, "bias[col]", "get_bias(col)")
	// one product per pass of the loop over products (closed before the end
	// of main, the last line of the source)
	if tiled {
		src = rep(
			src,
			"if ((p.u5 & 1u) == 1u) { C[o] = C[o] + w; } else { C[o] = w; }",
			"put_c(o, w);",
		)
		src = rep(
			src,
			"\tvar c0v = vec4<f32>(0.0);",
			"\tfor (var m = group_first(wid.z); m < group_end(wid.z); m++) {\n\tgm = m;\n\tvar c0v = vec4<f32>(0.0);",
		)
	} else {
		src = rep(
			src,
			"if ((p.u5 & 1u) == 1u) { C[o] = C[o] + v; } else { C[o] = v; }",
			"put_c(o, v);",
		)
		src = rep(
			src,
			"@builtin(local_invocation_id) lid: vec3<u32>) {",
			"@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {",
		)
		src = rep(
			src,
			"\tvar acc = 0.0;",
			"\tfor (var m = group_first(wid.z); m < group_end(wid.z); m++) {\n\tgm = m;\n\tvar acc = 0.0;",
		)
	}
	src = strings.trim_right(src, "\n")
	assert(strings.has_suffix(src, "\n}"), "gpu: grouped matmul source out of date")
	return strings.concatenate({MATMUL_GROUP, src[:len(src) - 1], "\t}\n}\n"}, allocator)
}

// The Adam kernel for `count` parameters: parameter k binds its gradient,
// weights and moments at bindings 4k + 1 .. 4k + 4 and has u_k elements; the
// threads run over the parameters one after the other. The update is the one
// of the Adam kernel (the same coefficients for all parameters).
@(private)
_adam_source :: proc(count: int, allocator := context.temp_allocator) -> string {
	b := strings.builder_make(allocator)
	for k in 0 ..< count {
		fmt.sbprintf(
			&b,
			"@group(0) @binding(%d) var<storage, read> gr%d: array<f32>;\n",
			4 * k + 1,
			k,
		)
		fmt.sbprintf(
			&b,
			"@group(0) @binding(%d) var<storage, read_write> w%d: array<f32>;\n",
			4 * k + 2,
			k,
		)
		fmt.sbprintf(
			&b,
			"@group(0) @binding(%d) var<storage, read_write> m%d: array<f32>;\n",
			4 * k + 3,
			k,
		)
		fmt.sbprintf(
			&b,
			"@group(0) @binding(%d) var<storage, read_write> v%d: array<f32>;\n",
			4 * k + 4,
			k,
		)
	}
	strings.write_string(&b, "@compute @workgroup_size(256)\n")
	strings.write_string(
		&b,
		"fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {\n",
	)
	strings.write_string(&b, "\tvar i = idx1(g, nw);\n")
	for k in 0 ..< count {
		fmt.sbprintf(&b, "\tif (i < p.u%d) ", k)
		strings.write_string(&b, "{\n") // (a brace in a format string is a format directive)
		fmt.sbprintf(&b, "\t\tlet gi = gr%d[i];\n", k)
		fmt.sbprintf(&b, "\t\tlet mi = p.f0 * m%d[i] + (1.0 - p.f0) * gi;\n", k)
		fmt.sbprintf(&b, "\t\tlet vi = p.f1 * v%d[i] + (1.0 - p.f1) * gi * gi;\n", k)
		fmt.sbprintf(&b, "\t\tm%d[i] = mi;\n\t\tv%d[i] = vi;\n", k, k)
		fmt.sbprintf(
			&b,
			"\t\tw%d[i] = w%d[i] * p.f5 - p.f3 * mi / (sqrt(vi) / p.f4 + p.f2);\n",
			k,
			k,
		)
		strings.write_string(&b, "\t\treturn;\n\t}\n")
		fmt.sbprintf(&b, "\ti = i - p.u%d;\n", k)
	}
	strings.write_string(&b, "}\n")
	return strings.to_string(b)
}

// gradients per launch of Grad_Sq_Sum (the 8th storage binding is the output)
// and Grad_Scale; 8 is WebGPU's minimum of storage buffers per kernel
@(private)
GRAD_SUM_GROUP :: 7
@(private)
GRAD_SCALE_GROUP :: 8

// The kernels of clip_grad_norm. Gradient k binds at binding k + 1 and has u_k
// elements (0 for an unused slot); the elements of all gradients are numbered
// one after the other.
// Grad_Sq_Sum: each workgroup adds the squares of its grid-stride share and
// writes the sum to o[u7 + workgroup] (binding 8).
// Grad_Scale: one thread per element, g *= f0.
@(private)
_grad_source :: proc(scale: bool, allocator := context.temp_allocator) -> string {
	count := GRAD_SCALE_GROUP if scale else GRAD_SUM_GROUP
	access := "read_write" if scale else "read"
	b := strings.builder_make(allocator)
	for k in 0 ..< count {
		fmt.sbprintf(
			&b,
			"@group(0) @binding(%d) var<storage, %s> g%d: array<f32>;\n",
			k + 1,
			access,
			k,
		)
	}
	if scale {
		strings.write_string(&b, "@compute @workgroup_size(256)\n")
		strings.write_string(
			&b,
			"fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {\n",
		)
		strings.write_string(&b, "\tvar i = idx1(gid, nw);\n")
		for k in 0 ..< count {
			fmt.sbprintf(&b, "\tif (i < p.u%d) ", k)
			strings.write_string(&b, "{ ") // (a brace in a format string is a format directive)
			fmt.sbprintf(&b, "g%d[i] = g%d[i] * p.f0; return; ", k, k)
			strings.write_string(&b, "}\n")
			fmt.sbprintf(&b, "\ti = i - p.u%d;\n", k)
		}
		strings.write_string(&b, "}\n")
		return strings.to_string(b)
	}
	strings.write_string(&b, "@group(0) @binding(8) var<storage, read_write> o: array<f32>;\n")
	strings.write_string(&b, "var<workgroup> sh: array<f32, 256>;\n")
	strings.write_string(&b, "fn sq(j: u32) -> f32 {\n\tvar i = j;\n\tvar v = 0.0;\n")
	for k in 0 ..< count {
		fmt.sbprintf(&b, "\tif (i < p.u%d) ", k)
		strings.write_string(&b, "{ ")
		fmt.sbprintf(&b, "v = g%d[i]; return v * v; ", k)
		strings.write_string(&b, "}\n")
		fmt.sbprintf(&b, "\ti = i - p.u%d;\n", k)
	}
	strings.write_string(&b, "\treturn 0.0;\n}\n")
	strings.write_string(
		&b,
		`@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let n = p.u0 + p.u1 + p.u2 + p.u3 + p.u4 + p.u5 + p.u6;
	var s = 0.0;
	var i = wid.x * 256u + lid.x;
	let stride = nw.x * 256u;
	loop {
		if (i >= n) { break; }
		s = s + sq(i);
		i = i + stride;
	}
	sh[lid.x] = s;
	workgroupBarrier();
	for (var w = 128u; w > 0u; w = w >> 1u) {
		if (lid.x < w) { sh[lid.x] = sh[lid.x] + sh[lid.x + w]; }
		workgroupBarrier();
	}
	if (lid.x == 0u) { o[p.u7 + wid.x] = sh[0]; }
}
`,
	)
	return strings.to_string(b)
}
