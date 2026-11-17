package gpu

// ============================================================================
// WGSL compute kernels (f32)
// ============================================================================
//
// Every kernel has entry point `main`, the parameter block at binding 0 and its
// storage buffers at bindings 1, 2, ... in the order the dispatching code binds
// them. 1D kernels run 256 threads per workgroup over a 2D grid (idx1).
// Each kernel mirrors the CPU op of the same name in wotan/tensor; where the CPU
// code has a numerically careful form (sigmoid, tanh, softplus), so does the kernel.

@(private)
Kernel :: enum {
	Unary, // y = f(x)
	Unary_Bwd, // dx += dy * f'(x, y)
	Binary, // out = a (+ - * /) b
	Binary_Bwd, // d(a or b) += dy * ∂(a ∘ b)
	Matmul, // C (+)= op(A) @ op(B)
	Reduce, // out[k] = scale * Σ x (or Σ x²), per workgroup
	Bcast_Add, // dx += scale * s[k]
	Mse_Bwd, // dx += scale * dy[0] * (a - b)
	Bias_Add, // out = a + bias[i % D]
	Col_Sum, // db[j] += Σ_i dy[i, j]
	Row_Sum, // out[b] = Σ_j x[b, j]
	Row_Bcast_Add, // dx[b, j] += dy[b]
	Transpose, // y (+)= xᵀ
	Strided_Copy, // row blocks from one layout to another (slice, concat)
	Scale_Inplace, // y *= f
	Adam,
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
KERNEL_SOURCES := [Kernel]string {
	.Unary            = `
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
		default: {}
	}
	y[i] = r;
}
`,
	.Unary_Bwd        = `
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
		default: {}
	}
	dx[i] = dx[i] + d;
}
`,
	.Binary           = `
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
	.Binary_Bwd       = `
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
	dst[i] = dst[i] + d;
}
`,
	// C[M, N] (+)= op(A)[M, K] @ op(B)[K, N]; u0 M, u1 N, u2 K, u3 transA, u4 transB,
	// u5 accumulate, u6 first row of this dispatch. Workgroup z is the batch:
	// batch z uses the z-th contiguous A, B and C blocks. 16x16 tiles in
	// workgroup memory.
	.Matmul           = `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let M = p.u0; let N = p.u1; let K = p.u2;
	let a0 = g.z * M * K;
	let b0 = g.z * K * N;
	let c0 = g.z * M * N;
	let row = g.y + p.u6;
	let col = g.x;
	var acc = 0.0;
	let tiles = (K + 15u) / 16u;
	for (var tt = 0u; tt < tiles; tt++) {
		let ak = tt * 16u + lid.x;
		var av = 0.0;
		if (row < M && ak < K) {
			if (p.u3 == 0u) { av = A[a0 + row * K + ak]; } else { av = A[a0 + ak * M + row]; }
		}
		As[lid.y][lid.x] = av;
		let bk = tt * 16u + lid.y;
		var bv = 0.0;
		if (bk < K && col < N) {
			if (p.u4 == 0u) { bv = B[b0 + bk * N + col]; } else { bv = B[b0 + col * K + bk]; }
		}
		Bs[lid.y][lid.x] = bv;
		workgroupBarrier();
		for (var k = 0u; k < 16u; k++) { acc = acc + As[lid.y][k] * Bs[k][lid.x]; }
		workgroupBarrier();
	}
	if (row < M && col < N) {
		let o = c0 + row * N + col;
		if (p.u5 == 1u) { C[o] = C[o] + acc; } else { C[o] = acc; }
	}
}
`,
	// u0 n, u1 0 sum / 1 sum of squares, u2 output index, f0 scale. Each
	// workgroup writes the scaled sum of its grid-stride share to out[u2 + wg].
	.Reduce           = `
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
		s = s + select(v, v * v, p.u1 == 1u);
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
	.Bcast_Add        = `
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dx[i] = dx[i] + p.f0 * s[p.u2];
}
`,
	// dx[i] += f0 * dy[0] * (a[i] - b[i])
	.Mse_Bwd          = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dx[i] = dx[i] + p.f0 * dy[0] * (a[i] - b[i]);
}
`,
	// out[i] = a[i] + bias[i % D]; u0 n, u1 D
	.Bias_Add         = `
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
	// db[j] += Σ_i dy[i * D + j]; u0 N rows, u1 D
	.Col_Sum          = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let j = idx1(g, nw);
	if (j >= p.u1) { return; }
	var s = 0.0;
	for (var i = 0u; i < p.u0; i++) { s = s + dy[i * p.u1 + j]; }
	db[j] = db[j] + s;
}
`,
	// out[b] = Σ_j x[b * W + j]; u0 rows, u1 W
	.Row_Sum          = `
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
	.Row_Bcast_Add    = `
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let i = idx1(g, nw);
	if (i >= p.u0) { return; }
	dx[i] = dx[i] + dy[i / p.u1];
}
`,
	// x [rows, cols] -> y [cols, rows]; u0 rows, u1 cols, u2 accumulate
	.Transpose        = `
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
	.Strided_Copy     = `
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
	.Scale_Inplace    = `
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
	.Adam             = `
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
	// optim.sgd_update: u1 momentum on, u2 nesterov; f0 lr, f1 momentum, f2 weight decay
	.Sgd              = `
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
	.Softmax_Rows     = `
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
	.Softmax_Bwd      = `
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
	.Row_Prob_Grad    = `
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
		dx[i] = dx[i] + coef * (prob - select(0.0, 1.0, j == tgt));
	}
}
`,
	// u0 rows, u1 D, u2 1 = RMS norm (no centering, no beta), f0 eps
	.Norm_Fwd         = `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
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
	for (var j = lid; j < D; j += 256u) {
		let i = r * D + j;
		if (rms) { y[i] = x[i] * inv * gamma[j]; } else { y[i] = (x[i] - mean) * inv * gamma[j] + beta[j]; }
	}
}
`,
	// stats[2r] = mean (0 for RMS), stats[2r + 1] = 1 / std (1 / rms)
	.Norm_Stats       = `
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
	.Norm_Bwd_X       = `
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
			dx[i] = dx[i] + inv * a - x[i] * inv * inv * inv * sax / Df;
		} else {
			let xh = (x[i] - mean) * inv;
			dx[i] = dx[i] + (Df * a - sa - xh * sax) * inv / Df;
		}
	}
}
`,
	// per column j: u4 = 1: dgamma[j] += Σ_r dy·x̂; u3 = 1: dbeta[j] += Σ_r dy
	.Norm_Bwd_Param   = `
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
	if (p.u4 == 1u) { dgamma[j] = dgamma[j] + sg; }
	if (p.u3 == 1u) { dbeta[j] = dbeta[j] + sb; }
}
`,
	// out[r, e] = W[ids[r], e] (0 for an id outside [0, u2)); u0 rows, u1 E, u2 vocab
	.Gather_Rows      = `
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
	.Scatter_Add_Rows = `
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
	.Permute_Mha      = `
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
	.Bce              = `
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
	.Row_Scale        = `
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
	.Row_Dot          = `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
	let r = idx1(gi, nw);
	if (r >= p.u0) { return; }
	var s = 0.0;
	for (var j = 0u; j < p.u1; j++) { s = s + a[r * p.u1 + j] * b[r * p.u1 + j]; }
	o[r] = o[r] + s;
}
`,
}
