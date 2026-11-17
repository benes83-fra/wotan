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
KERNEL_SOURCES := [Kernel]string {
	.Unary         = `
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
		default: {}
	}
	y[i] = r;
}
`,
	.Unary_Bwd     = `
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
		default: {}
	}
	dx[i] = dx[i] + d;
}
`,
	.Binary        = `
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
	.Binary_Bwd    = `
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
	// u5 accumulate, u6 first row of this dispatch. 16x16 tiles in workgroup memory.
	.Matmul        = `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
	let M = p.u0; let N = p.u1; let K = p.u2;
	let row = g.y + p.u6;
	let col = g.x;
	var acc = 0.0;
	let tiles = (K + 15u) / 16u;
	for (var tt = 0u; tt < tiles; tt++) {
		let ak = tt * 16u + lid.x;
		var av = 0.0;
		if (row < M && ak < K) {
			if (p.u3 == 0u) { av = A[row * K + ak]; } else { av = A[ak * M + row]; }
		}
		As[lid.y][lid.x] = av;
		let bk = tt * 16u + lid.y;
		var bv = 0.0;
		if (bk < K && col < N) {
			if (p.u4 == 0u) { bv = B[bk * N + col]; } else { bv = B[col * K + bk]; }
		}
		Bs[lid.y][lid.x] = bv;
		workgroupBarrier();
		for (var k = 0u; k < 16u; k++) { acc = acc + As[lid.y][k] * Bs[k][lid.x]; }
		workgroupBarrier();
	}
	if (row < M && col < N) {
		let o = row * N + col;
		if (p.u5 == 1u) { C[o] = C[o] + acc; } else { C[o] = acc; }
	}
}
`,
	// u0 n, u1 0 sum / 1 sum of squares, u2 output index, f0 scale. Each
	// workgroup writes the scaled sum of its grid-stride share to out[u2 + wg].
	.Reduce        = `
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
	.Bcast_Add     = `
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
	.Mse_Bwd       = `
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
	.Bias_Add      = `
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
	.Col_Sum       = `
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
	.Row_Sum       = `
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
	.Row_Bcast_Add = `
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
	.Transpose     = `
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
	.Strided_Copy  = `
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
	.Scale_Inplace = `
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
	.Adam          = `
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
	.Sgd           = `
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
}
