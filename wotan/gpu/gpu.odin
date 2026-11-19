/*
Opt-in GPU backend for wotan/tensor and wotan/nn (WebGPU compute shaders through
wgpu-native: Vulkan, DX12 or Metal).

Without a call to gpu.init nothing changes: tensors stay on the CPU. Only
programs that import this package need the wgpu-native library (see the
`vendor:wgpu` docs for where it goes).

	ok := gpu.init()                 // false: no GPU, keep using the CPU
	defer gpu.shutdown()
	gpu.to_device(params[:])         // parameters (and inputs) move to the GPU
	loss := ...                      // the same tensor / nn calls as on the CPU
	t.tensor_backward(loss)
	nn.adam_step(&opt)               // runs on the GPU for device parameters
	fmt.println(t.tensor_item(loss)) // read results through the sync accessors

The GPU computes in f32; the host API stays f64 and values are converted at
upload and download. Ops without a GPU kernel run on the CPU automatically
(see stats().forward_fallbacks).
*/
package gpu

import t "../tensor"
import "core:time"

Backend :: enum {
	Auto, // best available: Vulkan / Metal, then DX12
	Vulkan,
	DX12,
	Metal,
	OpenGL, // only when asked for explicitly
}

Adapter_Preference :: enum {
	Discrete, // a discrete GPU when there is one
	Integrated,
}

Options :: struct {
	backend:        Backend,
	prefer:         Adapter_Preference,
	// also accept a software adapter (e.g. Microsoft Basic Render Driver,
	// lavapipe): slow, but runs without a GPU
	allow_software: bool,
	// panic when a host tensor is uploaded automatically because a device op
	// uses it (finds forgotten to_device calls)
	strict:         bool,
}

Device_Info :: struct {
	name:         string,
	vendor:       string,
	backend:      Backend,
	adapter_type: string,
}

Stats :: struct {
	dispatches:         int, // kernel launches
	bind_group_hits:    int, // launches whose bind group came from the cache
	flushes:            int, // command buffer submissions
	uploads:            int,
	downloads:          int, // host syncs that waited for the GPU
	bytes_up:           u64,
	bytes_down:         u64,
	pool_bytes:         u64, // device memory held by the buffer pool
	auto_uploads:       int, // host tensors uploaded because a device op used them
	forward_fallbacks:  [t.Op]int, // ops run on the CPU although an input was on the GPU
	backward_fallbacks: [t.Op]int,
}

// init opens the GPU and enables the device paths of wotan/tensor and wotan/nn.
// Returns false (and leaves everything on the CPU) if no suitable adapter exists;
// init_error() then says why.
init :: proc(opts := Options{}, allocator := context.allocator) -> bool {
	if G.active {return true}
	G.allocator = allocator
	ok, reason := _open(opts)
	if !ok {
		_close()
		_init_error = reason
		return false
	}
	G.active = true
	t._device_hooks = {
		alloc         = _hook_alloc,
		release       = _hook_release,
		upload        = _hook_upload,
		download      = _hook_download,
		zero          = _hook_zero,
		copy          = _hook_copy,
		copy_rows     = _hook_copy_rows,
		has_forward   = _hook_has_forward,
		forward       = _hook_forward,
		backward      = _hook_backward,
		adam          = _hook_adam,
		sgd           = _hook_sgd,
		optim_sync    = _hook_optim_sync,
		optim_release = _hook_optim_release,
		grad_sq_sum   = _hook_grad_sq_sum,
		grad_scale    = _hook_grad_scale,
		strict        = opts.strict,
	}
	t.device_stats = {}
	return true
}

@(private)
_init_error: string

// init_error: why the last init failed
init_error :: proc() -> string {
	return _init_error
}

// shutdown waits for the GPU and releases it. Device tensors still alive lose
// their device data: move what you keep with to_host first.
shutdown :: proc() {
	if !G.active {return}
	t._device_hooks = {}
	_close()
}

// is_active: whether init succeeded and shutdown was not called yet
is_active :: proc() -> bool {
	return G.active
}

device_info :: proc() -> Device_Info {
	return G.info
}

// synchronize waits until the GPU has finished all recorded work
synchronize :: proc() {
	if G.active {_wait()}
}

stats :: proc() -> Stats {
	ds := t.device_stats
	return Stats {
		dispatches = G.counters.dispatches,
		bind_group_hits = G.counters.bind_group_hits,
		flushes = G.counters.flushes,
		uploads = G.counters.uploads,
		downloads = G.counters.downloads,
		bytes_up = G.counters.bytes_up,
		bytes_down = G.counters.bytes_down,
		pool_bytes = G.pool_bytes,
		auto_uploads = ds.auto_uploads,
		forward_fallbacks = ds.forward_fallbacks,
		backward_fallbacks = ds.backward_fallbacks,
	}
}

reset_stats :: proc() {
	G.counters = {}
	t.device_stats = {}
}

// to_device moves tensors to the GPU (data, and the gradient if there is one).
// Their host slices are parked: read them with t.tensor_values afterwards.
to_device_one :: proc(x: ^t.Tensor) -> ^t.Tensor {
	assert(G.active, "gpu.to_device: call gpu.init first")
	if x == nil {return nil}
	t._device_storage(x)
	t._data_to_device(x)
	t._data_written_on_device(x)
	if t.tensor_has_grad(x) {
		t._grad_to_device(x)
		t._grad_written_on_device(x)
	}
	return x
}

to_device_many :: proc(xs: []^t.Tensor) {
	for x in xs {to_device_one(x)}
}

to_device :: proc {
	to_device_one,
	to_device_many,
}

// to_host moves tensors back to the CPU: their data and gradients are
// downloaded and their device buffers released
to_host_one :: proc(x: ^t.Tensor) -> ^t.Tensor {
	if x == nil || x.device == nil {return x}
	t.tensor_sync_host(x)
	t._device_release(x)
	return x
}

to_host_many :: proc(xs: []^t.Tensor) {
	for x in xs {to_host_one(x)}
}

to_host :: proc {
	to_host_one,
	to_host_many,
}

// set_kernel_enabled turns the GPU kernels of one op off (it then runs through
// the CPU fallback) or back on. For testing and for comparing the two paths.
set_kernel_enabled :: proc(op: t.Op, enabled: bool) {
	G.disabled[op] = !enabled
}

// has_kernel: whether `op` has a GPU kernel (enabled or not)
has_kernel :: proc(op: t.Op) -> bool {
	return op in SUPPORTED
}

// Matmul_Kernel: which kernel computes matrix products
Matmul_Kernel :: enum {
	Auto, // register-tiled for products of at least MATMUL_TILED_MIN rows and columns
	Tiled, // always the register-tiled kernel (64x64 tiles, 4x4 outputs per thread)
	Simple, // always the 16x16-tile kernel (one output per thread)
}

// MATMUL_TILED_MIN: the smallest row and column count for which .Auto uses the
// register-tiled kernel
MATMUL_TILED_MIN :: 32

// set_matmul_kernel chooses the matmul kernel (default .Auto). For comparisons
// and tests; both kernels compute the same products.
set_matmul_kernel :: proc(k: Matmul_Kernel) {
	G.matmul_kernel = k
}

// set_linear_grouping turns the grouping of linear layers on one input off or
// back on (default on): attention's query, key and value projections run as
// one launch per kernel instead of three (see linear_group.odin). For
// comparisons and tests; the values are the same.
set_linear_grouping :: proc(enabled: bool) {
	_held_flush()
	G.linear_ungrouped = !enabled
}

// benchmark_matmul: the mean time in milliseconds of one f32 product
// C[M, N] = op(A) @ op(B) (op: transposed if asked; A stored [M, K] or [K, M],
// B [K, N] or [N, K]) on the GPU with the current kernel choice. Fixed inputs;
// `reps` products after one warm-up, timed until the GPU has finished.
benchmark_matmul :: proc(M, N, K: int, reps := 10, trans_a := false, trans_b := false) -> f64 {
	assert(G.active, "gpu.benchmark_matmul: call gpu.init first")
	vals := make([]f64, max(M * K, K * N), context.temp_allocator)
	for &v, i in vals {v = f64((i * 7919) % 1000) / 1000 - 0.5}
	A, B, C := _acquire(M * K), _acquire(K * N), _acquire(M * N)
	defer {_release_buf(A); _release_buf(B); _release_buf(C)}
	_upload(A, vals[:M * K])
	_upload(B, vals[:K * N])
	a, b, c := Bind{A, M * K}, Bind{B, K * N}, Bind{C, M * N}
	_matmul(a, b, c, M, N, K, trans_a, trans_b, false)
	_wait()
	start := time.tick_now()
	for _ in 0 ..< reps {_matmul(a, b, c, M, N, K, trans_a, trans_b, false)}
	_wait()
	return time.duration_milliseconds(time.tick_since(start)) / f64(reps)
}
