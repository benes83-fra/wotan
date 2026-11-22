// audio_profile: memory and throughput of audio-sized training steps.
//
// Times training steps of models with the sizes of a FastSpeech 2 acoustic
// model and a Vocos vocoder (24 kHz, 100 mel bins, n_fft 1024, hop 256) on the
// CPU (f64) or the GPU (f32), and reports per step: wall time, kernel launches,
// CPU fallbacks, and the memory they need: the process's peak working set, the
// graph arena and, on the GPU, the buffer pool. It also checks for leaks: live
// tensors, heap bytes and pool bytes must not grow from one step to the next.
// The models are size proxies with the published layer shapes, built from
// wotan layers and ops; the trainable models themselves belong to skald.
//
// Run one configuration per process (the peak working set is per process):
//
//     odin run examples/audio_profile -o:speed -- <config> <cpu|gpu> [steps] [batch]
//
// Configurations:
//     fs2        FastSpeech 2, batch 16, 10-s clips (938 mel frames, 120 phonemes)
//     vocos      Vocos generator + mel L1 loss, batch 16, 10-s clips
//     vocos_gan  Vocos GAN step (discriminator update, then generator update with
//                multi-period and multi-resolution discriminators, least-squares
//                GAN and feature-matching losses), batch 16, 1-s segments
package audio_profile

import gpu "../../wotan/gpu"
import nn "../../wotan/nn"
import t "../../wotan/tensor"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:time"

// Model: what a configuration provides to the harness
Model :: struct {
	name:   string,
	params: [dynamic]^t.Tensor, // every trainable tensor (for counting and the device)
	opts:   [dynamic]^nn.Adam, // optimizers (the GAN has two)
	inputs: [dynamic]^t.Tensor, // inputs, targets and constants (moved to the device)
	step:   proc(m: ^Model, scope: ^t.Graph_Scope) -> f64, // one training step, returns the loss
	data:   rawptr, // the configuration's own state
}

// the batch size of every configuration (16 unless given)
batch := 16

Step_Stats :: struct {
	ms:           f64,
	launches:     int,
	fallbacks:    int,
	live_tensors: int,
	heap_bytes:   i64,
	pool_bytes:   u64,
	arena_used:   int,
	loss:         f64,
}

main :: proc() {
	args := os.args
	if len(args) < 3 {
		fmt.eprintln("usage: audio_profile <fs2|vocos|vocos_gan> <cpu|gpu> [steps] [batch]")
		os.exit(2)
	}
	config, device := args[1], args[2]
	steps := 5
	if len(args) > 3 {
		if n, ok := strconv.parse_int(args[3]); ok {steps = max(n, 4)}
	}
	if len(args) > 4 {
		if n, ok := strconv.parse_int(args[4]); ok {batch = max(n, 1)}
	}

	// every heap allocation goes through a tracking allocator (leak check)
	tracking: mem.Tracking_Allocator
	mem.tracking_allocator_init(&tracking, context.allocator)
	context.allocator = mem.tracking_allocator(&tracking)

	on_gpu := device == "gpu"
	if on_gpu && !gpu.init() {
		fmt.eprintln("no usable GPU:", gpu.init_error())
		os.exit(1)
	}

	m: Model
	switch config {
	case "fs2":
		m = fs2_model()
	case "vocos":
		m = vocos_model(gan = false)
	case "vocos_gan":
		m = vocos_model(gan = true)
	case:
		fmt.eprintln("unknown configuration", config)
		os.exit(2)
	}
	n_params := 0
	for p in m.params {n_params += t.tensor_numel(p)}
	if on_gpu {
		gpu.to_device(m.params[:])
		gpu.to_device(m.inputs[:])
		info := gpu.device_info()
		fmt.printfln("GPU: %s", info.name)
	}
	fmt.printfln(
		"%s, batch %d, on the %s: %.2f M parameters",
		m.name,
		batch,
		device,
		f64(n_params) / 1e6,
	)

	scope: t.Graph_Scope
	stats := make([]Step_Stats, steps)
	for s in 0 ..< steps {
		if on_gpu {gpu.reset_stats()}
		start := time.tick_now()
		loss := m.step(&m, &scope)
		if on_gpu {gpu.synchronize()}
		free_all(context.temp_allocator) // as any training loop
		st := &stats[s]
		st.ms = time.duration_milliseconds(time.tick_since(start))
		st.loss = loss
		st.live_tensors = t.global_tensors_created - t.global_tensors_freed
		st.heap_bytes = tracking.current_memory_allocated
		_, st.arena_used = t.graph_scope_bytes(&scope) // reserved: the arena's peak
		if on_gpu {
			gs := gpu.stats()
			st.launches = gs.dispatches
			st.pool_bytes = gs.pool_bytes
			for op in t.Op {st.fallbacks += gs.forward_fallbacks[op] + gs.backward_fallbacks[op]}
		}
		fmt.printfln(
			"  step %d: %9.1f ms  loss %.4f  launches %d  fallbacks %d  live tensors %d  heap %.1f MB  arena %.1f MB  pool %.1f MB",
			s + 1,
			st.ms,
			st.loss,
			st.launches,
			st.fallbacks,
			st.live_tensors,
			f64(st.heap_bytes) / 1e6,
			f64(st.arena_used) / 1e6,
			f64(st.pool_bytes) / 1e6,
		)
	}

	// steady state: the steps after the first (which allocates arena chunks,
	// pool buffers and the optimizer state). Caches fill up to the second
	// step, so the leak check compares the third step with the last.
	mean_ms := 0.0
	for st in stats[1:] {mean_ms += st.ms}
	mean_ms /= f64(steps - 1)
	last, second := stats[steps - 1], stats[2]
	leak_free :=
		last.live_tensors == second.live_tensors &&
		last.heap_bytes == second.heap_bytes &&
		last.pool_bytes == second.pool_bytes &&
		last.arena_used == second.arena_used
	fmt.printfln(
		"RESULT %s %s batch %d: step %.1f ms (mean of steps 2-%d), peak working set %.2f GB, graph arena %.2f GB, GPU pool %.2f GB, launches/step %d, fallbacks/step %d, leak-free %v",
		config,
		device,
		batch,
		mean_ms,
		steps,
		f64(_peak_working_set()) / 1e9,
		f64(last.arena_used) / 1e9,
		f64(last.pool_bytes) / 1e9,
		last.launches,
		last.fallbacks,
		leak_free,
	)
	if !leak_free {
		fmt.printfln(
			"LEAK: from step 3 to %d: live tensors %d -> %d, heap %d -> %d bytes, pool %d -> %d bytes",
			steps,
			second.live_tensors,
			last.live_tensors,
			second.heap_bytes,
			last.heap_bytes,
			second.pool_bytes,
			last.pool_bytes,
		)
	}
	t.graph_scope_destroy(&scope)
	if on_gpu {gpu.shutdown()}
}

// _to_device: a constant made after setup goes to the device too
_to_device :: proc(x: ^t.Tensor) {
	gpu.to_device(x)
}
