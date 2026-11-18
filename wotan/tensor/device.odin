package tensor

import "base:runtime"
import "core:fmt"
import "core:mem"

// ============================================================================
// Device residency (opt-in GPU backend)
// ============================================================================
//
// A tensor is on the CPU while `device` is nil: everything in this package then
// works exactly as without a GPU. The GPU backend (package wotan/gpu, the only
// package that links WebGPU) fills in `_device_hooks`; until it does, none of
// the code below runs.
//
// A device tensor keeps its data and its gradient in device buffers. Each of the
// two is current on the host, on the device, or on both (Residency). While only
// the device copy is current, the host slice is parked in the storage and
// `t.data.data` (or `t.grad.data`) is nil: code that reads it without syncing
// fails instead of computing on stale numbers. rows, cols and shape stay valid.
//
// Read device data with tensor_values / tensor_grad_values / tensor_item (they
// sync). After writing a synced host copy in place, call tensor_mark_host_dirty
// so the next device use uploads it.

Residency :: enum u8 {
	Host, // only the host copy is current (or there is no data yet)
	Device, // only the device copy is current; the host slice is parked
	Both,
}

// Device_Buffer is an opaque handle to a buffer of the GPU backend. The
// generation makes releasing a stale handle (a buffer already reused) a no-op.
Device_Buffer :: struct {
	ptr: rawptr,
	gen: u32,
}

Device_Storage :: struct {
	data_buf:          Device_Buffer,
	grad_buf:          Device_Buffer,
	data_where:        Residency,
	grad_where:        Residency,
	host_data:         []f64, // host data slice, parked while data_where == .Device
	host_grad:         []f64, // host gradient slice, parked while grad_where == .Device
	// op state the backward kernel needs (dropout mask, attention mask)
	aux_buf:           Device_Buffer,
	// The device gradient is all zeros, but its buffer has not been cleared:
	// the first backward kernel that writes the whole gradient overwrites it
	// instead of adding to it, and every other use clears it first
	// (_device_grad_materialize). This saves one clear command per gradient.
	grad_zero_pending: bool,
}

// Device_Hooks: the operations the GPU backend provides. All nil without a GPU.
Device_Hooks :: struct {
	alloc:         proc(numel: int) -> Device_Buffer,
	release:       proc(b: Device_Buffer),
	upload:        proc(b: Device_Buffer, src: []f64),
	download:      proc(b: Device_Buffer, dst: []f64),
	zero:          proc(b: Device_Buffer, numel: int),
	copy:          proc(dst, src: Device_Buffer, numel: int),
	// has_forward reports whether a kernel computes `op` on the device
	has_forward:   proc(op: Op) -> bool,
	// forward computes `out` (device data allocated) from inputs on the device
	forward:       proc(op: Op, out: ^Tensor, inputs: []^Tensor, ints: []int, floats: []f64),
	// backward runs a node's backward kernel; false: no kernel, use the CPU fallback
	backward:      proc(node: ^Tensor, allocator: mem.Allocator) -> bool,
	// optimizer kernels (wotan/nn calls these for device parameters); adam
	// takes all device parameters of one step, so it can update several
	// parameters per kernel launch
	adam:          proc(updates: []Device_Adam_Update, beta_1, beta_2, epsilon, weight_decay: f64),
	sgd:           proc(
		param: ^Tensor,
		key: rawptr,
		velocity: []f64,
		lr, momentum, weight_decay: f64,
		nesterov: bool,
	),
	// optim_sync copies the device optimizer state of `key` into the host slices
	// (and frees it if `release`); false if there is none
	optim_sync:    proc(key: rawptr, a, b: []f64, release: bool) -> bool,
	optim_release: proc(key: rawptr),
	grad_sq_sum:   proc(params: []^Tensor) -> f64,
	grad_scale:    proc(t: ^Tensor, coef: f64),
	strict:        bool, // panic when a host tensor would be uploaded automatically
}

_device_hooks: Device_Hooks

// Device_Adam_Update: one parameter's part of an Adam step on the device
// (the arguments of optimize.adam_update; key identifies the optimizer state)
Device_Adam_Update :: struct {
	param: ^Tensor,
	key:   rawptr,
	m, v:  []f64,
	lr:    f64,
	step:  int,
}

// Device_Stats counts what ran where since the last reset (see gpu.stats)
Device_Stats :: struct {
	forward_fallbacks:  [Op]int, // ops computed on the CPU although an input was on the device
	backward_fallbacks: [Op]int,
	auto_uploads:       int, // host tensors uploaded because a device op used them
	host_syncs:         int, // device copies brought to the host
}

device_stats: Device_Stats

// tensor_is_on_device: whether the tensor has device storage
tensor_is_on_device :: proc(t: ^Tensor) -> bool {
	return t != nil && t.device != nil
}

// tensor_numel: number of elements (valid for host and device tensors)
tensor_numel :: proc(t: ^Tensor) -> int {
	return t.data.rows * t.data.cols
}

// tensor_values returns the tensor's data on the host, downloading it first if
// only the device copy is current. The slice stays valid until the next device
// op writes the tensor.
tensor_values :: proc(t: ^Tensor) -> []f64 {
	_data_to_host(t)
	return t.data.data
}

// tensor_grad_values returns the gradient on the host (nil if there is none)
tensor_grad_values :: proc(t: ^Tensor) -> []f64 {
	_grad_to_host(t)
	return t.grad.data
}

// tensor_item: the value of a one-element tensor (e.g. a loss)
tensor_item :: proc(t: ^Tensor, loc := #caller_location) -> f64 {
	fmt.assertf(
		tensor_numel(t) == 1,
		"tensor_item: tensor has %d elements",
		tensor_numel(t),
		loc = loc,
	)
	return tensor_values(t)[0]
}

// tensor_sync_host makes the host copies of data and gradient current
tensor_sync_host :: proc(t: ^Tensor) {
	_data_to_host(t)
	_grad_to_host(t)
}

// tensor_mark_host_dirty: the host data of a device tensor was written in place
// (after tensor_values / tensor_sync_host); the next device use uploads it
tensor_mark_host_dirty :: proc(t: ^Tensor) {
	if t.device != nil && t.data.data != nil {t.device.data_where = .Host}
}

// tensor_mark_grad_host_dirty: as tensor_mark_host_dirty, for the gradient
tensor_mark_grad_host_dirty :: proc(t: ^Tensor) {
	if t.device != nil && t.grad.data != nil {t.device.grad_where = .Host}
}

// ----------------------------------------------------------------------------
// Residency transitions (internal; the GPU backend uses them too)
// ----------------------------------------------------------------------------

_device_on :: #force_inline proc() -> bool {
	return _device_hooks.alloc != nil
}

// _device_storage returns the tensor's storage, creating it (host-current) if needed
_device_storage :: proc(t: ^Tensor) -> ^Device_Storage {
	if t.device == nil {
		t.device = new(Device_Storage, t.allocator)
	}
	return t.device
}

// _device_alloc gets a buffer for `t`; buffers of tensors living in a graph
// scope are returned to the pool by graph_end
_device_alloc :: proc(t: ^Tensor, numel: int) -> Device_Buffer {
	b := _device_hooks.alloc(numel)
	if s := _scope_of(t.allocator); s != nil {
		if s.device_buffers.allocator.procedure == nil {
			s.device_buffers = make([dynamic]Device_Buffer, runtime.heap_allocator())
		}
		append(&s.device_buffers, b)
	}
	return b
}

// _data_to_device makes the device copy of the data current (uploading it)
_data_to_device :: proc(t: ^Tensor) {
	if t.device == nil {
		if _device_hooks.strict {
			panic(
				"gpu strict mode: a host tensor was used by a device op (move it with gpu.to_device)",
			)
		}
		device_stats.auto_uploads += 1
	}
	s := _device_storage(t)
	if s.data_where != .Host {return}
	n := tensor_numel(t)
	if s.data_buf.ptr == nil {s.data_buf = _device_alloc(t, n)}
	if n > 0 {_device_hooks.upload(s.data_buf, t.data.data)}
	s.data_where = .Both
}

// _data_written_on_device: a device op wrote the data; park the host slice
_data_written_on_device :: proc(t: ^Tensor) {
	s := t.device
	if s.data_where != .Device {
		if t.data.data != nil {s.host_data = t.data.data}
		t.data.data = nil
		s.data_where = .Device
	}
}

// _data_to_host makes the host copy of the data current (downloading it)
_data_to_host :: proc(t: ^Tensor) {
	if t == nil || t.device == nil || t.device.data_where != .Device {return}
	s := t.device
	n := tensor_numel(t)
	if s.host_data == nil {s.host_data = make([]f64, n, t.data.allocator)}
	if n > 0 && _device_hooks.download != nil {_device_hooks.download(s.data_buf, s.host_data)}
	t.data.data = s.host_data
	s.host_data = nil
	s.data_where = .Both
	device_stats.host_syncs += 1
}

// _grad_exists: the tensor holds a gradient buffer (on the host or the device)
_grad_exists :: #force_inline proc(t: ^Tensor) -> bool {
	return t.grad.data != nil || (t.device != nil && t.device.grad_where == .Device)
}

// _grad_to_device makes the device copy of the gradient current; a tensor
// without a gradient gets a zeroed device gradient
_grad_to_device :: proc(t: ^Tensor) {
	s := _device_storage(t)
	n := tensor_numel(t)
	if !_grad_exists(t) {
		if s.grad_buf.ptr == nil {s.grad_buf = _device_alloc(t, n)}
		s.grad_zero_pending = true
		t.grad.rows, t.grad.cols = t.data.rows, t.data.cols
		if t.grad.allocator.procedure == nil {t.grad.allocator = t.data.allocator}
		s.grad_where = .Device
		t.grad_dropped = false
		return
	}
	if s.grad_where != .Host {return}
	if s.grad_buf.ptr == nil {s.grad_buf = _device_alloc(t, n)}
	if n > 0 {_device_hooks.upload(s.grad_buf, t.grad.data)}
	s.grad_zero_pending = false
	s.grad_where = .Both
}

// _grad_written_on_device: a device kernel wrote the gradient; park the host slice
_grad_written_on_device :: proc(t: ^Tensor) {
	s := t.device
	if s.grad_where != .Device {
		if t.grad.data != nil {s.host_grad = t.grad.data}
		t.grad.data = nil
		s.grad_where = .Device
	}
}

// _grad_to_host makes the host copy of the gradient current
_grad_to_host :: proc(t: ^Tensor) {
	if t == nil || t.device == nil || t.device.grad_where != .Device {return}
	s := t.device
	n := tensor_numel(t)
	if s.host_grad == nil {s.host_grad = make([]f64, n, t.grad.allocator)}
	if s.grad_zero_pending {
		// the device buffer was never written (its clear stays pending)
		for &v in s.host_grad {v = 0}
	} else if n > 0 && _device_hooks.download != nil {
		_device_hooks.download(s.grad_buf, s.host_grad)
	}
	t.grad.data = s.host_grad
	s.host_grad = nil
	s.grad_where = .Both
	device_stats.host_syncs += 1
}

// _grad_host_for_overwrite: a host gradient slice whose old values do not
// matter (it is about to be filled completely); no download
_grad_host_for_overwrite :: proc(t: ^Tensor) -> []f64 {
	s := t.device
	if s.grad_where == .Device {
		if s.host_grad == nil {s.host_grad = make([]f64, tensor_numel(t), t.grad.allocator)}
		t.grad.data = s.host_grad
		s.host_grad = nil
		s.grad_where = .Both
	}
	return t.grad.data
}

// _grad_written_on_host: CPU code changed the host gradient of a device tensor
_grad_written_on_host :: proc(t: ^Tensor) {
	if t.device != nil && t.grad.data != nil {t.device.grad_where = .Host}
}

// _device_zero_grad: the gradient becomes a zeroed device buffer (the clear
// itself stays pending, see Device_Storage.grad_zero_pending)
_device_zero_grad :: proc(t: ^Tensor) {
	s := t.device
	n := tensor_numel(t)
	if s.grad_buf.ptr == nil {s.grad_buf = _device_alloc(t, n)}
	s.grad_zero_pending = true
	if t.grad.allocator.procedure == nil {t.grad.allocator = t.data.allocator}
	t.grad.rows, t.grad.cols = t.data.rows, t.data.cols
	if s.grad_where != .Device {
		if t.grad.data != nil {s.host_grad = t.grad.data}
		t.grad.data = nil
		s.grad_where = .Device
	}
	t.grad_dropped = false
}

// _device_grad_materialize records the pending clear of the device gradient,
// before a use that reads it or writes only part of it
_device_grad_materialize :: proc(t: ^Tensor) {
	s := t.device
	if s == nil || !s.grad_zero_pending {return}
	_device_hooks.zero(s.grad_buf, tensor_numel(t))
	s.grad_zero_pending = false
}

// _device_grad_take_pending: for a kernel that writes the whole device
// gradient. True if the clear was pending: the kernel then overwrites the
// buffer instead of adding to it, and the clear is dropped.
_device_grad_take_pending :: proc(t: ^Tensor) -> bool {
	s := t.device
	if s == nil || !s.grad_zero_pending {return false}
	s.grad_zero_pending = false
	return true
}

// _device_release returns the tensor's device buffers to the pool and puts
// parked host slices back, so the normal frees release them
_device_release :: proc(t: ^Tensor) {
	s := t.device
	if s == nil {return}
	if _device_hooks.release != nil {
		if s.data_buf.ptr != nil {_device_hooks.release(s.data_buf)}
		if s.grad_buf.ptr != nil {_device_hooks.release(s.grad_buf)}
		if s.aux_buf.ptr != nil {_device_hooks.release(s.aux_buf)}
	}
	if t.data.data == nil {t.data.data = s.host_data}
	if t.grad.data == nil {t.grad.data = s.host_grad}
	free(s, t.allocator)
	t.device = nil
}

// ----------------------------------------------------------------------------
// Op dispatch helpers
// ----------------------------------------------------------------------------

// _device_use: run `op` on the device? True when the backend is active, an
// input is on the device and a kernel exists for the op.
_device_use :: proc(op: Op, inputs: ..^Tensor) -> bool {
	if _device_hooks.forward == nil {return false}
	on := false
	for t in inputs {
		if t != nil && t.device != nil {on = true; break}
	}
	return on && _device_hooks.has_forward(op)
}

// _host_inputs: CPU fallback of an op without a device kernel. Brings the
// inputs' data to the host; the op then runs on the CPU and its output is a
// host tensor (uploaded automatically by the next device op that uses it).
_host_inputs :: proc(op: Op, inputs: ..^Tensor) {
	if _device_hooks.alloc == nil {return}
	counted := false
	for t in inputs {
		if t == nil || t.device == nil {continue}
		if !counted {device_stats.forward_fallbacks[op] += 1; counted = true}
		_data_to_host(t)
	}
}

// _device_new_op creates the output node of a device op: same contract as
// _tensor_new_op, with its data in a device buffer (no host data)
_device_new_op :: proc(
	rows, cols: int,
	shape: [4]int,
	requires_grad: bool,
	allocator: mem.Allocator,
	record := true, // false: a leaf the caller owns (constant recordings skip it)
	loc := #caller_location,
) -> ^Tensor {
	numel := shape[0] * shape[1] * shape[2] * shape[3]
	if numel != rows * cols {
		panic(
			fmt.tprintf(
				"tensor shape %v (%d elements) does not match its data (%d x %d)",
				shape,
				numel,
				rows,
				cols,
			),
			loc,
		)
	}
	t := new(Tensor, allocator)
	t.data = {
		rows      = rows,
		cols      = cols,
		allocator = allocator,
	}
	t.shape = shape
	t.requires_grad = requires_grad && grad_enabled()
	t.op = .None
	t.allocator = allocator
	t.inputs = make([dynamic]^Tensor, 0, allocator)
	t.int_metadata = make([dynamic]int, 0, allocator)
	t.f64_metadata = make([dynamic]f64, 0, allocator)
	_scope_note_new(allocator)
	global_tensors_created += 1
	if record && len(_recordings) > 0 {_record_new(t)}
	s := _device_storage(t)
	s.data_buf = _device_alloc(t, numel)
	s.data_where = .Device
	return t
}

// _device_forward computes a device op: inputs to the device, then the kernel
_device_forward :: proc(
	op: Op,
	out: ^Tensor,
	inputs: []^Tensor,
	ints: []int = nil,
	floats: []f64 = nil,
) {
	for t in inputs {_data_to_device(t)}
	_device_hooks.forward(op, out, inputs, ints, floats)
}

// _node_on_device: whether a backward node or one of its inputs is on the device
_node_on_device :: proc(node: ^Tensor) -> bool {
	if node.device != nil {return true}
	for t in node.inputs {
		if t != nil && t.device != nil {return true}
	}
	return false
}

// _node_to_host brings a node and its inputs (data and gradients) to the host
// for the CPU backward case
_node_to_host :: proc(node: ^Tensor) {
	device_stats.backward_fallbacks[node.op] += 1
	_data_to_host(node)
	_grad_to_host(node)
	for t in node.inputs {
		if t == nil {continue}
		_data_to_host(t)
		if t.requires_grad {_grad_to_host(t)}
	}
}

// _node_grads_written_on_host: after a CPU backward case, the inputs' host
// gradients are the current ones
_node_grads_written_on_host :: proc(node: ^Tensor) {
	for t in node.inputs {
		if t != nil && t.requires_grad {_grad_written_on_host(t)}
	}
}

// _device_elementwise: device version of the element-wise ops (output shaped
// like `a`). `floats` are the op's parameters, recorded in f64_metadata exactly
// as the CPU op records them.
_device_elementwise :: proc(op: Op, a, b: ^Tensor, floats: ..f64) -> ^Tensor {
	rg := a.requires_grad || (b != nil && b.requires_grad)
	out := _device_new_op(a.data.rows, a.data.cols, a.shape, rg, _op_alloc(a))
	inputs := [2]^Tensor{a, b}
	n := 2 if b != nil else 1
	_device_forward(op, out, inputs[:n], nil, floats)
	if out.requires_grad {
		out.op = op
		append(&out.inputs, ..inputs[:n])
		append(&out.f64_metadata, ..floats)
	}
	return out
}

// _device_scalar_op: device Sum / Mean / MSELoss (scalar output)
_device_scalar_op :: proc(op: Op, a, b: ^Tensor) -> ^Tensor {
	rg := a.requires_grad || (b != nil && b.requires_grad)
	out := _device_new_op(1, 1, SCALAR_SHAPE, rg, _op_alloc(a))
	inputs := [2]^Tensor{a, b}
	n := 2 if b != nil else 1
	_device_forward(op, out, inputs[:n])
	if out.requires_grad {
		out.op = op
		append(&out.inputs, ..inputs[:n])
	}
	return out
}

// _device_matmul: device tensor_matmul, both paths (2D and the flattened
// [1, N*in] @ [in, out]), recording the same metadata as the CPU op
_device_matmul :: proc(a, b: ^Tensor) -> ^Tensor {
	rg := a.requires_grad || b.requires_grad
	if a.data.rows == 1 && a.data.cols != b.data.rows && a.data.cols % b.data.rows == 0 {
		N := a.data.cols / b.data.rows
		in_features, out_features := b.data.rows, b.data.cols
		out := _device_new_op(
			1,
			N * out_features,
			_flattened_matmul_shape(a.shape, N, in_features, out_features),
			rg,
			_op_alloc(a),
		)
		_device_forward(.MatMul, out, {a, b}, {N, in_features, out_features})
		if out.requires_grad {
			out.op = .MatMul
			append(&out.inputs, a, b)
			append(&out.int_metadata, N, in_features, out_features)
		}
		return out
	}
	if a.data.cols != b.data.rows {panic("tensor_matmul: dimension mismatch")}
	out := _device_new_op(
		a.data.rows,
		b.data.cols,
		{a.data.rows, b.data.cols, 1, 1},
		rg,
		_op_alloc(a),
	)
	_device_forward(.MatMul, out, {a, b}, {a.data.rows, a.data.cols, b.data.cols})
	if out.requires_grad {
		out.op = .MatMul
		append(&out.inputs, a, b)
	}
	return out
}

// _device_linear: device tensor_linear (bias given), both paths
_device_linear :: proc(x, w, b: ^Tensor) -> ^Tensor {
	N, in_f, out_f, flattened := _linear_dims(x, w)
	rg := x.requires_grad || w.requires_grad || b.requires_grad
	shape :=
		_flattened_matmul_shape(x.shape, N, in_f, out_f) if flattened else [4]int{N, out_f, 1, 1}
	rows, cols := N, out_f
	if flattened {rows, cols = 1, N * out_f}
	out := _device_new_op(rows, cols, shape, rg, _op_alloc(x))
	_device_forward(.Linear, out, {x, w, b}, {N, in_f, out_f})
	if out.requires_grad {
		out.op = .Linear
		append(&out.inputs, x, w, b)
		append(&out.int_metadata, N, in_f, out_f)
	}
	return out
}

// _device_add_bias: device tensor_add_bias, both paths
_device_add_bias :: proc(a, bias: ^Tensor) -> ^Tensor {
	D := bias.data.cols
	if bias.data.rows != 1 {panic("tensor_add_bias: bias must be a 1xD row vector")}
	rg := a.requires_grad || bias.requires_grad
	flattened := a.data.rows == 1 && a.data.cols != D && a.data.cols % D == 0
	if !flattened && a.data.cols != D {panic("tensor_add_bias: column mismatch")}
	N := a.data.cols / D if flattened else a.data.rows
	out := _device_new_op(a.data.rows, a.data.cols, a.shape, rg, _op_alloc(a))
	_device_forward(.AddBias, out, {a, bias}, {N, D})
	if out.requires_grad {
		out.op = .AddBias
		append(&out.inputs, a, bias)
		if flattened {append(&out.int_metadata, N, D)}
	}
	return out
}

// _device_attention: device version of the three attention ops (plain, with an
// additive mask, flash with an optional causal mask), recording the metadata
// each CPU op records
_device_attention :: proc(op: Op, Q, K, V: ^Tensor, mask: []f64, causal: bool) -> ^Tensor {
	_check_shape(Q)
	_check_shape(K)
	_check_shape(V)
	batch, seq_q, d_v := Q.shape[0], Q.shape[1], V.shape[2]
	rg := Q.requires_grad || K.requires_grad || V.requires_grad
	out := _device_new_op(1, batch * seq_q * d_v, {batch, seq_q, d_v, 1}, rg, _op_alloc(Q))
	if mask != nil {
		out.device.aux_buf = _device_alloc(out, len(mask))
		_device_hooks.upload(out.device.aux_buf, mask)
	}
	_device_forward(op, out, {Q, K, V}, {len(mask), int(causal)})
	if out.requires_grad {
		out.op = op
		append(&out.inputs, Q, K, V)
		if op == .MaskedScaledDotProductAttention {append(&out.f64_metadata, ..mask)}
		if op == .FlashAttention {append(&out.int_metadata, int(causal))}
	}
	return out
}

// _device_multi_head_attention: device tensor_multi_head_attention as one node.
// int_metadata: [batch, seq_q, seq_k, num_heads, head_dim, kind, causal] with
// kind 0 plain, 1 masked (the mask in f64_metadata and the aux buffer), 2 flash;
// the forward also gets the mask length.
_device_multi_head_attention :: proc(
	q, k, v: ^Tensor,
	dims: [5]int, // batch, seq_q, seq_k, num_heads, head_dim
	mask: []f64,
	flash, causal: bool,
) -> ^Tensor {
	batch, seq_q, d := dims[0], dims[1], dims[3] * dims[4]
	rg := q.requires_grad || k.requires_grad || v.requires_grad
	out := _device_new_op(1, batch * seq_q * d, {batch, seq_q, d, 1}, rg, _op_alloc(q))
	if mask != nil {
		out.device.aux_buf = _device_alloc(out, len(mask))
		_device_hooks.upload(out.device.aux_buf, mask)
	}
	kind := 1 if mask != nil else (2 if flash else 0)
	meta := [8]int{dims[0], dims[1], dims[2], dims[3], dims[4], kind, int(causal), len(mask)}
	_device_forward(.MultiHeadAttention, out, {q, k, v}, meta[:])
	if out.requires_grad {
		out.op = .MultiHeadAttention
		append(&out.inputs, q, k, v)
		append(&out.int_metadata, ..meta[:7])
		append(&out.f64_metadata, ..mask)
	}
	return out
}

// _device_pool: device max / average pooling. The max version records no
// argmax indices: its backward finds each window's maximum again.
_device_pool :: proc(op: Op, input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	_check_shape(input)
	N, C, H, W := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	out_h, out_w := (H - kH) / stride + 1, (W - kW) / stride + 1
	out := _device_new_op(
		1,
		N * C * out_h * out_w,
		{N, C, out_h, out_w},
		input.requires_grad,
		_op_alloc(input),
	)
	_device_forward(op, out, {input}, {kH, kW, stride})
	if out.requires_grad {
		out.op = op
		append(&out.inputs, input)
		out.pool_params = PoolParams{kH, kW, stride, 0}
	}
	return out
}

// _device_conv2d: device tensor_conv2d (same output shape and recorded parameters)
_device_conv2d :: proc(input, weight, bias: ^Tensor, stride, pad: int) -> ^Tensor {
	_check_shape(input)
	_check_shape(weight)
	N, H, W := input.shape[0], input.shape[2], input.shape[3]
	C_out, kH, kW := weight.shape[0], weight.shape[2], weight.shape[3]
	out_h := (H + 2 * pad - kH) / stride + 1
	out_w := (W + 2 * pad - kW) / stride + 1
	rg := input.requires_grad || weight.requires_grad || (bias != nil && bias.requires_grad)
	out := _device_new_op(
		1,
		N * C_out * out_h * out_w,
		{N, C_out, out_h, out_w},
		rg,
		_op_alloc(input),
	)
	inputs := [3]^Tensor{input, weight, bias}
	ni := 3 if bias != nil else 2
	_device_forward(.Conv2d, out, inputs[:ni], {stride, pad})
	if out.requires_grad {
		out.op = .Conv2d
		append(&out.inputs, ..inputs[:ni])
		out.conv_params = ConvParams{kH, kW, stride, pad}
	}
	return out
}

// _device_batch_norm_2d: device tensor_batch_norm_2d. The statistics used per
// channel ([means, then 1 / std]) stay on the device in the output's aux
// buffer, so f64_metadata holds only [training]; training updates the running
// statistics on the device.
_device_batch_norm_2d :: proc(
	input, weight, bias, running_mean, running_var: ^Tensor,
	training: bool,
	momentum, eps: f64,
) -> ^Tensor {
	_check_shape(input)
	C := input.shape[1]
	rg := input.requires_grad || weight.requires_grad || bias.requires_grad
	out := _device_new_op(1, tensor_numel(input), input.shape, rg, _op_alloc(input))
	out.device.aux_buf = _device_alloc(out, 2 * C)
	flag := 1.0 if training else 0.0
	_device_forward(
		.BatchNorm2d,
		out,
		{input, weight, bias, running_mean, running_var},
		nil,
		{flag, momentum, eps},
	)
	if out.requires_grad {
		out.op = .BatchNorm2d
		append(&out.inputs, input, weight, bias)
		append(&out.f64_metadata, flag)
	}
	return out
}

// _device_loss: a device op with a scalar output that records its inputs and
// `floats` as f64_metadata (KL divergence, entropy, prospect utility)
_device_loss :: proc(op: Op, inputs: []^Tensor, floats: []f64) -> ^Tensor {
	rg := false
	for x in inputs {rg ||= x.requires_grad}
	out := _device_new_op(1, 1, SCALAR_SHAPE, rg, _op_alloc(inputs[0]))
	_device_forward(op, out, inputs, nil, floats)
	if out.requires_grad {
		out.op = op
		append(&out.inputs, ..inputs)
		append(&out.f64_metadata, ..floats)
	}
	return out
}

// _device_reparameterize: z = mu + exp(log_var / 2) · eps on the device, with
// eps drawn on the host (as the CPU draws it) and kept in the aux buffer
_device_reparameterize :: proc(
	mu, log_var: ^Tensor,
	eps: []f64,
	allocator: mem.Allocator,
) -> ^Tensor {
	rg := mu.requires_grad || log_var.requires_grad
	out := _device_new_op(mu.data.rows, mu.data.cols, mu.shape, rg, allocator)
	out.device.aux_buf = _device_alloc(out, len(eps))
	_device_hooks.upload(out.device.aux_buf, eps)
	_device_forward(.Reparameterize, out, {mu, log_var})
	if out.requires_grad {
		out.op = .Reparameterize
		append(&out.inputs, mu, log_var)
	}
	return out
}
