package main

// GPU test suite (separate entry point: only this program links wgpu-native).
//
//     odin run gpu_tests -collection:root=. -debug
//
// Every GPU check runs on the device and compares against the CPU engine.
// Without a usable GPU adapter the suite prints SKIP and exits with code 0.
// Exit code 1 if any check failed.

import gpu "../wotan/gpu"
import "core:fmt"
import "core:mem"
import "core:os"

failures: int

check :: proc(ok: bool, format: string, args: ..any) {
	if !ok {
		failures += 1
		fmt.printf("    FAIL ")
		fmt.printfln(format, ..args)
	}
}

main :: proc() {
	exit_code := 0
	defer os.exit(exit_code)
	when ODIN_DEBUG {
		tracking: mem.Tracking_Allocator
		mem.tracking_allocator_init(&tracking, context.allocator)
		context.allocator = mem.tracking_allocator(&tracking)
		defer {
			for _, entry in tracking.allocation_map {
				fmt.eprintf("%v leaked %v bytes\n", entry.location, entry.size)
			}
			for entry in tracking.bad_free_array {
				fmt.eprintf("%v bad free\n", entry.location)
			}
			mem.tracking_allocator_destroy(&tracking)
		}
	}
	defer free_all(context.temp_allocator)

	if !gpu.init() {
		fmt.println("SKIP gpu_tests: no usable GPU adapter:", gpu.init_error())
		return
	}
	info := gpu.device_info()
	fmt.printfln("GPU: %s (%v, %s)", info.name, info.backend, info.adapter_type)

	gpu_residency_test()
	gpu_op_parity_test()
	gpu_fallback_op_test()
	gpu_matmul_kernel_test()
	gpu_linear_group_test()
	gpu_mlp_training_test()
	gpu_all_fallback_training_test()
	gpu_sgd_clip_test()
	gpu_clip_grad_norm_test()
	gpu_grad_accumulation_adam_test()
	gpu_gpt_training_test()
	gpu_bert_training_test()
	gpu_device_forward_cpu_backward_test()
	gpu_batch_norm_running_stats_test()
	gpu_cnn_training_test()
	gpu_tcn_training_test()
	gpu_ml_finance_test()
	gpu_scope_memory_test()
	gpu_checkpoint_test()
	gpu_speed_report()
	gpu_matmul_benchmark()

	gpu.shutdown()
	if failures > 0 {
		fmt.eprintf("\n%d GPU check(s) FAILED\n", failures)
		exit_code = 1
	} else {
		fmt.println("\nAll GPU checks passed")
	}
}
