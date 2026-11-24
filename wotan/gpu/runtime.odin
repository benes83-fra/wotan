package gpu

import t "../tensor"
import "base:runtime"
import "core:fmt"
import "core:mem"
import "core:strings"
import wgpu "vendor:wgpu"

// ============================================================================
// Runtime: device, buffer pool, command stream, transfers, pipelines
// ============================================================================
//
// Execution model: every op records its dispatches into one pending command
// encoder and returns at once; nothing waits between ops. The pending work is
// submitted (flushed) when the host needs data (a download), when the
// per-batch uniform space is full, and every FLUSH_EVERY dispatches so the GPU
// starts early on long graphs.
//
// Ordering: all device-side writes to pool buffers (kernels, clears, copies) are
// recorded in the encoder, in program order. A host upload with nothing pending
// writes the buffer directly (QueueWriteBuffer). With work pending, a direct
// write would land before that work runs, so the upload goes through a staging
// buffer instead: the data is written there and a copy into the target is
// recorded in the encoder, in program order. Staging buffers have their own
// pool and return to it when the batch is submitted (later writes to them are
// queued after that submission).
//
// Per-dispatch cost: the parameter blocks of a batch are collected on the host
// and written to the uniform buffer in one call when the batch is submitted.
// The parameter block is bound with a dynamic offset, so a bind group depends
// only on the kernel and its storage buffers, and bind groups are cached (pool
// buffers live until shutdown, so a cached bind group never refers to a
// destroyed buffer; a buffer the pool gives back clears the cache). A training
// loop with a graph scope acquires the same pool buffers in the same order every
// step, so after the first step its bind groups come from the cache.

@(private)
FLUSH_EVERY :: 128

@(private)
PARAMS_SIZE :: 64

// Per-dispatch parameters (one uniform block per dispatch)
@(private)
Params :: struct #align (16) {
	u: [8]u32,
	f: [8]f32,
}
#assert(size_of(Params) == PARAMS_SIZE)

@(private)
Buf :: struct {
	buffer: wgpu.Buffer, // nil once the pool gave the buffer back (the record is reused)
	bytes:  u64,
	stamp:  u64, // release order: the pool gives back the longest-unused buffers first
	gen:    u32,
	free:   bool,
}

@(private)
Pipeline :: struct {
	pipeline:        wgpu.ComputePipeline,
	layout:          wgpu.BindGroupLayout,
	pipeline_layout: wgpu.PipelineLayout,
}

@(private)
Optim_State :: struct {
	a, b: ^Buf, // Adam: moment 1, moment 2; SGD: velocity, nil
	n:    int,
}

@(private)
State :: struct {
	active:           bool,
	instance:         wgpu.Instance,
	adapter:          wgpu.Adapter,
	device:           wgpu.Device,
	queue:            wgpu.Queue,
	limits:           wgpu.Limits,
	info:             Device_Info,
	// command stream
	encoder:          wgpu.CommandEncoder,
	pass:             wgpu.ComputePassEncoder,
	pending:          int,
	// uniform parameters of the pending batch
	uniform:          wgpu.Buffer,
	uniform_off:      u64,
	uniform_cap:      u64,
	uniform_align:    u64,
	uniform_host:     []u8, // the batch's parameter blocks, written at submit
	// bind group cache
	bind_groups:      map[Bind_Key]wgpu.BindGroup,
	// readback staging
	readback:         wgpu.Buffer,
	readback_bytes:   u64,
	// buffer pool: free buffers by size class
	pool:             map[u64][dynamic]^Buf,
	bufs:             [dynamic]^Buf, // every buffer record, given-back ones included
	spare:            [dynamic]^Buf, // records of buffers given back, for reuse
	pool_bytes:       u64, // device memory of all pool buffers
	live_bytes:       u64, // of the buffers in use
	peak_live_bytes:  u64,
	release_stamp:    u64,
	staging_free:     map[u64][dynamic]wgpu.Buffer, // upload staging buffers by size
	staging_used:     [dynamic]Staging, // staging buffers of the pending batch
	dummy:            ^Buf, // bound where a kernel needs no buffer
	pipelines:        [Kernel]Pipeline,
	optim:            map[rawptr]Optim_State,
	disabled:         [t.Op]bool,
	matmul_kernel:    Matmul_Kernel,
	linear_ungrouped: bool, // set_linear_grouping(false)
	rec_steps:        int, // set_recurrent_steps (0: from the work)
	counters:         Counters,
	allocator:        mem.Allocator,
	error_message:    string,
}

@(private)
Counters :: struct {
	dispatches, flushes, downloads, uploads: int,
	bytes_up, bytes_down:                    u64,
	bind_group_hits:                         int,
	given_back:                              int, // pool buffers released to the device
}

@(private)
Staging :: struct {
	buffer: wgpu.Buffer,
	bytes:  u64,
}

// storage buffers per kernel at most (the multi-parameter Adam binds 16)
@(private)
MAX_STORAGE_BINDS :: 16

// a bind group: the kernel and the bound storage buffers with sizes
@(private)
Bind_Key :: struct {
	kernel: Kernel,
	n:      int,
	bufs:   [MAX_STORAGE_BINDS]^Buf,
	sizes:  [MAX_STORAGE_BINDS]u64,
}

// cached bind groups beyond this count are released (a loop whose buffers change
// every step would otherwise grow the cache without bound)
@(private)
BIND_CACHE_MAX :: 1 << 14

@(private)
G: State

// ----------------------------------------------------------------------------
// Device selection
// ----------------------------------------------------------------------------

@(private)
_adapter_rank :: proc(info: wgpu.AdapterInfo, opts: Options) -> int {
	if opts.backend != .Auto && _backend_of(info.backendType) != opts.backend {return -1}
	type_rank := 0
	#partial switch info.adapterType {
	case .DiscreteGPU:
		type_rank = 4 if opts.prefer != .Integrated else 3
	case .IntegratedGPU:
		type_rank = 3 if opts.prefer != .Integrated else 4
	case .CPU:
		if !opts.allow_software {return -1}
		type_rank = 1
	case:
		type_rank = 2
	}
	backend_rank := 0
	#partial switch info.backendType {
	case .Vulkan, .Metal:
		backend_rank = 3
	case .D3D12:
		backend_rank = 2
	case .OpenGL, .OpenGLES:
		if opts.backend != .OpenGL {return -1} 	// compute on GL is limited: explicit only
		backend_rank = 1
	case:
		return -1
	}
	return type_rank * 10 + backend_rank
}

@(private)
_backend_of :: proc(b: wgpu.BackendType) -> Backend {
	#partial switch b {
	case .Vulkan:
		return .Vulkan
	case .D3D12:
		return .DX12
	case .Metal:
		return .Metal
	case .OpenGL, .OpenGLES:
		return .OpenGL
	}
	return .Auto
}

@(private)
_open :: proc(opts: Options) -> (ok: bool, reason: string) {
	G.instance = wgpu.CreateInstance(nil)
	if G.instance == nil {return false, "no WebGPU instance"}
	adapters := wgpu.InstanceEnumerateAdapters(G.instance, nil, context.temp_allocator)
	best, best_rank := wgpu.Adapter(nil), -1
	for a in adapters {
		info, status := wgpu.AdapterGetInfo(a)
		if status != .Success {continue}
		if r := _adapter_rank(info, opts); r > best_rank {best, best_rank = a, r}
		wgpu.AdapterInfoFreeMembers(info)
	}
	for a in adapters {if a != best {wgpu.AdapterRelease(a)}}
	if best == nil {return false, "no suitable GPU adapter"}
	G.adapter = best

	info, _ := wgpu.AdapterGetInfo(best)
	G.info = Device_Info {
		name         = strings.clone(info.device, G.allocator),
		vendor       = strings.clone(info.vendor, G.allocator),
		backend      = _backend_of(info.backendType),
		adapter_type = fmt.aprint(info.adapterType, allocator = G.allocator),
	}
	wgpu.AdapterInfoFreeMembers(info)

	// Request the adapter's own limits (the WebGPU defaults cap storage
	// bindings at 128 MB)
	limits, _ := wgpu.AdapterGetLimits(best)
	G.limits = limits
	Wait :: struct {
		device: wgpu.Device,
		done:   bool,
		msg:    string,
	}
	w: Wait
	wgpu.AdapterRequestDevice(
		best,
		&{
			requiredLimits = &limits,
			uncapturedErrorCallbackInfo = {callback = _on_error},
			deviceLostCallbackInfo = {mode = .AllowProcessEvents, callback = _on_lost},
		},
		{
			mode = .AllowProcessEvents,
			callback = proc "c" (
				s: wgpu.RequestDeviceStatus,
				d: wgpu.Device,
				m: wgpu.StringView,
				u1, u2: rawptr,
			) {
				context = runtime.default_context()
				w := (^Wait)(u1)
				if s == .Success {w.device = d} else {w.msg = strings.clone(m)}
				w.done = true
			},
			userdata1 = &w,
		},
	)
	for !w.done {wgpu.InstanceProcessEvents(G.instance)}
	if w.device == nil {return false, w.msg if w.msg != "" else "device request failed"}
	G.device = w.device
	G.queue = wgpu.DeviceGetQueue(G.device)

	G.uniform_align = u64(max(limits.minUniformBufferOffsetAlignment, PARAMS_SIZE))
	G.uniform_cap = 4096 * G.uniform_align
	G.uniform = wgpu.DeviceCreateBuffer(
		G.device,
		&{usage = {.Uniform, .CopyDst}, size = G.uniform_cap},
	)
	G.uniform_host = make([]u8, G.uniform_cap, G.allocator)
	G.bind_groups = make(map[Bind_Key]wgpu.BindGroup, G.allocator)
	G.staging_free = make(map[u64][dynamic]wgpu.Buffer, G.allocator)
	G.staging_used = make([dynamic]Staging, G.allocator)
	G.pool = make(map[u64][dynamic]^Buf, G.allocator)
	G.bufs = make([dynamic]^Buf, G.allocator)
	G.spare = make([dynamic]^Buf, G.allocator)
	G.optim = make(map[rawptr]Optim_State, G.allocator)
	G.dummy = _acquire(1)
	return true, ""
}

@(private)
_on_error :: proc "c" (d: ^wgpu.Device, type: wgpu.ErrorType, m: wgpu.StringView, u1, u2: rawptr) {
	context = runtime.default_context()
	panic(fmt.tprintf("gpu: WebGPU %v error: %s", type, m))
}

@(private)
_on_lost :: proc "c" (
	d: ^wgpu.Device,
	reason: wgpu.DeviceLostReason,
	m: wgpu.StringView,
	u1, u2: rawptr,
) {
	context = runtime.default_context()
	if reason == .Destroyed || reason == .CallbackCancelled {return} 	// shutdown
	panic(fmt.tprintf("gpu: device lost (%v): %s", reason, m))
}

@(private)
_close :: proc() {
	if G.device != nil {
		_flush()
		wgpu.DevicePoll(G.device, true)
	}
	for b in G.bufs {
		if b.buffer != nil {
			wgpu.BufferDestroy(b.buffer)
			wgpu.BufferRelease(b.buffer)
		}
		free(b, G.allocator)
	}
	delete(G.bufs)
	delete(G.spare)
	for _, &list in G.pool {delete(list)}
	delete(G.pool)
	delete(G.optim)
	for _, &list in G.staging_free {
		for b in list {wgpu.BufferDestroy(b); wgpu.BufferRelease(b)}
		delete(list)
	}
	delete(G.staging_free)
	for st in G.staging_used {wgpu.BufferDestroy(st.buffer); wgpu.BufferRelease(st.buffer)}
	delete(G.staging_used)
	_clear_bind_groups()
	delete(G.bind_groups)
	delete(G.uniform_host, G.allocator)
	for p in G.pipelines {
		if p.layout != nil {wgpu.BindGroupLayoutRelease(p.layout)}
		if p.pipeline_layout != nil {wgpu.PipelineLayoutRelease(p.pipeline_layout)}
		if p.pipeline != nil {wgpu.ComputePipelineRelease(p.pipeline)}
	}
	if G.uniform != nil {wgpu.BufferRelease(G.uniform)}
	if G.readback != nil {wgpu.BufferRelease(G.readback)}
	if G.queue != nil {wgpu.QueueRelease(G.queue)}
	if G.device != nil {
		wgpu.DeviceDestroy(G.device)
		wgpu.DeviceRelease(G.device)
	}
	if G.adapter != nil {wgpu.AdapterRelease(G.adapter)}
	if G.instance != nil {wgpu.InstanceRelease(G.instance)}
	delete(G.info.name, G.allocator)
	delete(G.info.vendor, G.allocator)
	delete(G.info.adapter_type, G.allocator)
	allocator := G.allocator
	G = {}
	G.allocator = allocator
}

// ----------------------------------------------------------------------------
// Buffer pool (f32 elements)
// ----------------------------------------------------------------------------
//
// Released buffers go back to the pool and are reused by requests of the same
// size class: powers of two up to 4 KB, above that four classes per doubling
// (sizes are rounded up by at most a quarter). A buffer serves only its own
// class, so a training loop with fixed shapes reuses the same buffers every
// step.
//
// The pool gives memory back when shapes change from step to step (sequences
// of different lengths): before it creates a buffer that would take the pool
// beyond POOL_SLACK times the peak of the memory in use, it gives back the
// free buffers that have been unused the longest. If the device has no memory
// left for a new buffer, it gives back all free buffers and tries once more.

// the pool may hold this much more than the peak of the buffers in use
@(private)
POOL_SLACK :: 1.25

// size classes per doubling above 4 KB
@(private)
POOL_CLASSES :: 4

@(private)
_bucket :: proc(numel: int) -> u64 {
	bytes := u64(max(numel, 1) * 4)
	b := u64(256)
	for b < bytes {b <<= 1}
	if b <= 4096 {return b}
	step := (b >> 1) / POOL_CLASSES
	return (bytes + step - 1) / step * step
}

@(private)
_acquire :: proc(numel: int) -> ^Buf {
	bytes := _bucket(numel)
	if bytes > G.limits.maxStorageBufferBindingSize || bytes > G.limits.maxBufferSize {
		panic(
			fmt.tprintf(
				"gpu: a tensor of %d elements exceeds the device's buffer limit (%d MB)",
				numel,
				G.limits.maxStorageBufferBindingSize >> 20,
			),
		)
	}
	b: ^Buf
	if list, ok := &G.pool[bytes]; ok && len(list) > 0 {
		b = pop(list)
	} else {
		// a new buffer: first give back long-unused ones if the pool would grow
		// beyond POOL_SLACK times the peak in use
		limit := u64(POOL_SLACK * f64(max(G.peak_live_bytes, G.live_bytes + bytes)))
		if G.pool_bytes + bytes > limit {_give_back(G.pool_bytes + bytes - limit)}
		buffer := _new_buffer(bytes, {.Storage, .CopySrc, .CopyDst})
		if len(G.spare) > 0 {
			b = pop(&G.spare)
		} else {
			b = new(Buf, G.allocator)
			append(&G.bufs, b)
		}
		b.buffer, b.bytes = buffer, bytes
		G.pool_bytes += bytes
	}
	b.free = false
	b.gen += 1
	G.live_bytes += b.bytes
	G.peak_live_bytes = max(G.peak_live_bytes, G.live_bytes)
	return b
}

// _new_buffer creates a device buffer (pool, staging or readback). If the
// device has no memory left, it gives back every free pool and staging buffer,
// waits until the device has released them, and tries once more; then it
// panics with the size.
@(private)
_new_buffer :: proc(bytes: u64, usage: wgpu.BufferUsageFlags) -> wgpu.Buffer {
	if buffer, ok := _create_buffer(bytes, usage); ok {return buffer}
	_give_back(max(u64))
	for _, &list in G.staging_free {
		for s in list {wgpu.BufferRelease(s)}
		clear(&list)
	}
	_wait()
	if buffer, ok := _create_buffer(bytes, usage); ok {return buffer}
	panic(
		fmt.tprintf(
			"gpu: out of device memory for a buffer of %.1f MB (%.2f GB in use by tensors and optimizer state); use a smaller batch",
			f64(bytes) / 1e6,
			f64(G.live_bytes) / 1e9,
		),
	)
}

// _create_buffer: a new buffer, or false if the device has no memory left
@(private)
_create_buffer :: proc(bytes: u64, usage: wgpu.BufferUsageFlags) -> (wgpu.Buffer, bool) {
	wgpu.DevicePushErrorScope(G.device, .OutOfMemory)
	wgpu.DevicePushErrorScope(G.device, .Validation) // wgpu reports some allocation failures as validation errors
	buffer := wgpu.DeviceCreateBuffer(G.device, &{usage = usage, size = bytes})
	ok := _pop_error_scope()
	ok = _pop_error_scope() && ok
	if !ok {
		wgpu.BufferRelease(buffer)
		return nil, false
	}
	return buffer, true
}

// _pop_error_scope: false if the scope caught an error
@(private)
_pop_error_scope :: proc() -> bool {
	Wait :: struct {
		done, failed: bool,
	}
	w: Wait
	wgpu.DevicePopErrorScope(G.device, {
		mode = .AllowProcessEvents,
		callback = proc "c" (
			s: wgpu.PopErrorScopeStatus,
			type: wgpu.ErrorType,
			m: wgpu.StringView,
			u1, u2: rawptr,
		) {
			w := (^Wait)(u1)
			w.failed = s != .Success || type != .NoError
			w.done = true
		},
		userdata1 = &w,
	})
	for !w.done {wgpu.InstanceProcessEvents(G.instance)}
	return !w.failed
}

// _give_back releases free pool buffers, the longest-unused first, until at
// least `need` bytes are given back or no free buffer is left. wgpu frees the
// memory once recorded or submitted work that uses a buffer has finished.
// Cached bind groups may refer to these buffers, so the cache is cleared.
@(private)
_give_back :: proc(need: u64) {
	if need == 0 {return}
	given := u64(0)
	for given < need {
		oldest: ^Buf
		for b in G.bufs {
			if b.free && b.buffer != nil && (oldest == nil || b.stamp < oldest.stamp) {oldest = b}
		}
		if oldest == nil {break}
		list := &G.pool[oldest.bytes]
		for x, i in list {if x == oldest {unordered_remove(list, i); break}}
		wgpu.BufferRelease(oldest.buffer)
		oldest.buffer = nil
		given += oldest.bytes
		G.pool_bytes -= oldest.bytes
		oldest.bytes = 0
		append(&G.spare, oldest)
		G.counters.given_back += 1
	}
	if given > 0 {_clear_bind_groups()}
}

@(private)
_release_buf :: proc(b: ^Buf) {
	if b == nil || b.free {return}
	b.free = true
	b.gen += 1
	G.live_bytes -= b.bytes
	G.release_stamp += 1
	b.stamp = G.release_stamp
	list, ok := &G.pool[b.bytes]
	if !ok {
		G.pool[b.bytes] = make([dynamic]^Buf, G.allocator)
		list = &G.pool[b.bytes]
	}
	append(list, b)
}

@(private)
_handle :: proc(b: ^Buf) -> t.Device_Buffer {
	return {ptr = b, gen = b.gen}
}

@(private)
_buf :: #force_inline proc(h: t.Device_Buffer) -> ^Buf {
	b := (^Buf)(h.ptr)
	assert(b != nil && b.gen == h.gen && !b.free, "gpu: use of a released device buffer")
	return b
}

// tensor hooks
@(private)
_hook_alloc :: proc(numel: int) -> t.Device_Buffer {
	return _handle(_acquire(numel))
}

@(private)
_hook_release :: proc(h: t.Device_Buffer) {
	b := (^Buf)(h.ptr)
	if b == nil || !G.active || b.gen != h.gen || b.free {return} 	// stale handle: already reused
	_release_buf(b)
}

// ----------------------------------------------------------------------------
// Command stream
// ----------------------------------------------------------------------------

@(private)
_encoder :: proc() -> wgpu.CommandEncoder {
	if G.encoder == nil {G.encoder = wgpu.DeviceCreateCommandEncoder(G.device)}
	return G.encoder
}

@(private)
_compute_pass :: proc() -> wgpu.ComputePassEncoder {
	if G.pass == nil {G.pass = wgpu.CommandEncoderBeginComputePass(_encoder())}
	return G.pass
}

@(private)
_end_pass :: proc() {
	if G.pass != nil {
		wgpu.ComputePassEncoderEnd(G.pass)
		wgpu.ComputePassEncoderRelease(G.pass)
		G.pass = nil
	}
}

// _flush submits the pending work (no wait)
@(private)
_flush :: proc() {
	_held_flush()
	_end_pass()
	if G.encoder != nil {
		if G.uniform_off > 0 {
			wgpu.QueueWriteBuffer(
				G.queue,
				G.uniform,
				0,
				raw_data(G.uniform_host),
				uint(G.uniform_off),
			)
		}
		cb := wgpu.CommandEncoderFinish(G.encoder)
		wgpu.QueueSubmit(G.queue, {cb})
		wgpu.CommandBufferRelease(cb)
		wgpu.CommandEncoderRelease(G.encoder)
		G.encoder = nil
		G.counters.flushes += 1
	}
	// the batch's staging buffers may be written again (queued after this submit)
	for st in G.staging_used {
		list, ok := &G.staging_free[st.bytes]
		if !ok {
			G.staging_free[st.bytes] = make([dynamic]wgpu.Buffer, G.allocator)
			list = &G.staging_free[st.bytes]
		}
		append(list, st.buffer)
	}
	clear(&G.staging_used)
	G.pending = 0
	G.uniform_off = 0
}

// _wait submits and blocks until the GPU finished all work
@(private)
_wait :: proc() {
	_flush()
	wgpu.DevicePoll(G.device, true)
}

@(private)
Bind :: struct {
	buf:   ^Buf,
	numel: int,
}

@(private)
_bind :: proc(h: t.Device_Buffer, numel: int) -> Bind {
	return {_buf(h), numel}
}

// _dispatch records one kernel launch over a [gx, gy] workgroup grid
@(private)
_dispatch :: proc(k: Kernel, p: Params, binds: []Bind, gx, gy: u32, gz: u32 = 1) {
	_held_flush() // held linear layers come first (linear_group.odin)
	if gx == 0 || gy == 0 || gz == 0 {return}
	if G.uniform_off + G.uniform_align > G.uniform_cap {_flush()}
	off := G.uniform_off
	G.uniform_off += G.uniform_align
	(^Params)(&G.uniform_host[off])^ = p

	pl := _pipeline(k)
	bg := _bind_group(k, pl, binds)
	pass := _compute_pass()
	wgpu.ComputePassEncoderSetPipeline(pass, pl.pipeline)
	dyn := [1]u32{u32(off)}
	wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, dyn[:])
	wgpu.ComputePassEncoderDispatchWorkgroups(pass, gx, gy, gz)
	G.counters.dispatches += 1
	G.pending += 1
	if G.pending >= FLUSH_EVERY {_flush()}
}

// _bind_group: the cached bind group of these bindings, created on first use
@(private)
_bind_group :: proc(k: Kernel, pl: Pipeline, binds: []Bind) -> wgpu.BindGroup {
	key := Bind_Key {
		kernel = k,
		n      = len(binds),
	}
	for b, i in binds {
		key.bufs[i] = b.buf
		key.sizes[i] = u64(max(b.numel, 1) * 4)
	}
	if bg, ok := G.bind_groups[key]; ok {
		G.counters.bind_group_hits += 1
		return bg
	}
	if len(G.bind_groups) >= BIND_CACHE_MAX {_clear_bind_groups()}
	entries: [MAX_STORAGE_BINDS + 1]wgpu.BindGroupEntry
	entries[0] = {
		binding = 0,
		buffer  = G.uniform,
		size    = PARAMS_SIZE,
	}
	for b, i in binds {
		entries[i + 1] = {
			binding = u32(i + 1),
			buffer  = b.buf.buffer,
			size    = key.sizes[i],
		}
	}
	bg := wgpu.DeviceCreateBindGroup(
		G.device,
		&{layout = pl.layout, entryCount = uint(len(binds) + 1), entries = &entries[0]},
	)
	G.bind_groups[key] = bg
	return bg
}

// _clear_bind_groups releases the cached bind groups (wgpu keeps the ones that
// recorded, unsubmitted commands still use alive until they finish)
@(private)
_clear_bind_groups :: proc() {
	for _, bg in G.bind_groups {wgpu.BindGroupRelease(bg)}
	clear(&G.bind_groups)
}

// _dispatch_1d: one thread per element, 256 per workgroup (kernels index with idx1)
@(private)
_dispatch_1d :: proc(k: Kernel, p: Params, binds: []Bind, n: int) {
	if n <= 0 {return}
	groups := (n + 255) / 256
	gx := min(groups, 65535)
	gy := (groups + gx - 1) / gx
	_dispatch(k, p, binds, u32(gx), u32(gy))
}

// ----------------------------------------------------------------------------
// Transfers (f64 on the host, f32 on the device)
// ----------------------------------------------------------------------------

@(private)
_upload :: proc(b: ^Buf, src: []f64) {
	if len(src) == 0 {return}
	_held_flush()
	tmp := make([]f32, len(src), runtime.heap_allocator())
	defer delete(tmp, runtime.heap_allocator())
	for v, i in src {tmp[i] = f32(v)}
	bytes := u64(len(tmp) * 4)
	if G.pending > 0 || G.encoder != nil {
		// recorded work may still read `b`: stage, and copy in program order
		// staging sizes: powers of two (few sizes; staging buffers are kept)
		size := u64(256)
		for size < u64(len(tmp) * 4) {size <<= 1}
		st := Staging {
			bytes = size,
		}
		if list, ok := &G.staging_free[size]; ok && len(list) > 0 {
			st.buffer = pop(list)
		} else {
			st.buffer = _new_buffer(size, {.CopySrc, .CopyDst})
		}
		append(&G.staging_used, st)
		wgpu.QueueWriteBuffer(G.queue, st.buffer, 0, raw_data(tmp), uint(bytes))
		_end_pass()
		wgpu.CommandEncoderCopyBufferToBuffer(_encoder(), st.buffer, 0, b.buffer, 0, bytes)
		G.pending += 1
	} else {
		wgpu.QueueWriteBuffer(G.queue, b.buffer, 0, raw_data(tmp), uint(bytes))
	}
	G.counters.uploads += 1
	G.counters.bytes_up += bytes
}

@(private)
_download :: proc(b: ^Buf, dst: []f64) {
	if len(dst) == 0 {return}
	_held_flush()
	bytes := u64(len(dst) * 4)
	if G.readback == nil || G.readback_bytes < bytes {
		if G.readback != nil {wgpu.BufferRelease(G.readback)}
		G.readback_bytes = max(bytes, 1 << 20)
		G.readback = _new_buffer(G.readback_bytes, {.MapRead, .CopyDst})
	}
	_end_pass()
	wgpu.CommandEncoderCopyBufferToBuffer(_encoder(), b.buffer, 0, G.readback, 0, bytes)
	_flush()
	done := false
	status: wgpu.MapAsyncStatus
	Map_Wait :: struct {
		done:   ^bool,
		status: ^wgpu.MapAsyncStatus,
	}
	mw := Map_Wait{&done, &status}
	wgpu.BufferMapAsync(G.readback, {.Read}, 0, uint(bytes), {
		mode = .AllowProcessEvents,
		callback = proc "c" (s: wgpu.MapAsyncStatus, m: wgpu.StringView, u1, u2: rawptr) {
			mw := (^Map_Wait)(u1)
			mw.status^ = s
			mw.done^ = true
		},
		userdata1 = &mw,
	})
	for !done {
		wgpu.DevicePoll(G.device, true)
		wgpu.InstanceProcessEvents(G.instance)
	}
	if status != .Success {panic(fmt.tprintf("gpu: reading a buffer back failed (%v)", status))}
	src := wgpu.BufferGetConstMappedRangeSlice(G.readback, 0, uint(len(dst)), f32)
	for v, i in src {dst[i] = f64(v)}
	wgpu.BufferUnmap(G.readback)
	G.counters.downloads += 1
	G.counters.bytes_down += bytes
}

@(private)
_zero :: proc(b: ^Buf, numel: int) {
	if numel <= 0 {return}
	_held_flush()
	_end_pass()
	wgpu.CommandEncoderClearBuffer(_encoder(), b.buffer, 0, u64(numel * 4))
	G.pending += 1
}

@(private)
_copy :: proc(dst, src: ^Buf, numel: int) {
	if numel <= 0 {return}
	_held_flush()
	_end_pass()
	wgpu.CommandEncoderCopyBufferToBuffer(_encoder(), src.buffer, 0, dst.buffer, 0, u64(numel * 4))
	G.pending += 1
}

@(private)
_hook_upload :: proc(h: t.Device_Buffer, src: []f64) {_upload(_buf(h), src)}
@(private)
_hook_download :: proc(h: t.Device_Buffer, dst: []f64) {_download(_buf(h), dst)}
@(private)
_hook_zero :: proc(h: t.Device_Buffer, numel: int) {_zero(_buf(h), numel)}
@(private)
_hook_copy :: proc(dst, src: t.Device_Buffer, numel: int) {_copy(_buf(dst), _buf(src), numel)}
@(private)
_hook_copy_rows :: proc(dst, src: t.Device_Buffer, rows, length, dst_stride, dst_off: int) {
	_strided(
		_bind(src, rows * length),
		_bind(dst, rows * dst_stride),
		rows,
		length,
		length,
		0,
		dst_stride,
		dst_off,
		false,
	)
}

// ----------------------------------------------------------------------------
// Pipelines (created on first use)
// ----------------------------------------------------------------------------

// The bind group layout is explicit: binding 0 is the parameter block with a
// dynamic offset; the storage bindings 1, 2, ... are read from the kernel's
// declarations (`var<storage, read>` or `var<storage, read_write>`).
@(private)
_pipeline :: proc(k: Kernel) -> Pipeline {
	if G.pipelines[k].pipeline != nil {return G.pipelines[k]}
	src := strings.concatenate({KERNEL_PRELUDE, _kernel_source(k)}, context.temp_allocator)
	module := wgpu.DeviceCreateShaderModule(
		G.device,
		&{
			nextInChain = &wgpu.ShaderSourceWGSL{sType = .ShaderSourceWGSL, code = src},
			label = fmt.tprint(k),
		},
	)
	entries: [MAX_STORAGE_BINDS + 1]wgpu.BindGroupLayoutEntry
	entries[0] = {
		binding = 0,
		visibility = {.Compute},
		buffer = {type = .Uniform, hasDynamicOffset = true, minBindingSize = PARAMS_SIZE},
	}
	n := 1
	for binding in 1 ..= MAX_STORAGE_BINDS {
		decl := fmt.tprintf("@binding(%d) var<storage, ", binding)
		at := strings.index(src, decl)
		if at < 0 {break}
		rest := src[at + len(decl):]
		type: wgpu.BufferBindingType =
			.Storage if strings.has_prefix(rest, "read_write") else .ReadOnlyStorage
		entries[n] = {
			binding = u32(binding),
			visibility = {.Compute},
			buffer = {type = type},
		}
		n += 1
	}
	layout := wgpu.DeviceCreateBindGroupLayout(
		G.device,
		&{entryCount = uint(n), entries = &entries[0], label = fmt.tprint(k)},
	)
	pl_layout := wgpu.DeviceCreatePipelineLayout(
		G.device,
		&{bindGroupLayoutCount = 1, bindGroupLayouts = &layout},
	)
	pipe := wgpu.DeviceCreateComputePipeline(
		G.device,
		&{
			layout = pl_layout,
			compute = {module = module, entryPoint = "main"},
			label = fmt.tprint(k),
		},
	)
	wgpu.ShaderModuleRelease(module)
	G.pipelines[k] = {pipe, layout, pl_layout}
	return G.pipelines[k]
}
