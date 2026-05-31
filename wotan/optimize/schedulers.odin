package optimize

import "core:math"

// ============================================================================
// Linear Decay Scheduler
// ============================================================================

LinearDecayScheduler :: struct {
	initial_lr:  f64,
	current_lr:  f64,
	total_steps: int,
	step:        int,
}

// Contract: start_lr > 0, total_steps >= 0
scheduler_linear_decay_init :: proc(start_lr: f64, total_steps: int) -> LinearDecayScheduler {
	return LinearDecayScheduler {
		initial_lr = start_lr,
		current_lr = start_lr,
		total_steps = total_steps,
		step = 0,
	}
}

// Contract: sched initialized via scheduler_linear_decay_init
// Updates sched.current_lr = initial_lr * (1 - step/total_steps)
scheduler_linear_decay_step :: proc(sched: ^LinearDecayScheduler) {
	sched.step += 1
	if sched.total_steps > 0 {
		progress := f64(sched.step) / f64(sched.total_steps)
		sched.current_lr = sched.initial_lr * (1.0 - progress)
	}
}

// ============================================================================
// Cosine Annealing Scheduler
// ============================================================================

CosineAnnealingScheduler :: struct {
	initial_lr:  f64,
	min_lr:      f64,
	total_steps: int,
	step:        int,
	current_lr:  f64,
}

// Contract: start_lr >= min_lr >= 0, total_steps >= 0
scheduler_cosine_annealing_init :: proc(
	start_lr: f64,
	min_lr: f64,
	total_steps: int,
) -> CosineAnnealingScheduler {
	return CosineAnnealingScheduler {
		initial_lr = start_lr,
		min_lr = min_lr,
		total_steps = total_steps,
		step = 0,
		current_lr = start_lr,
	}
}

// Contract: sched initialized via scheduler_cosine_annealing_init
// Updates sched.current_lr using cosine annealing formula
scheduler_cosine_annealing_step :: proc(sched: ^CosineAnnealingScheduler) {
	sched.step += 1
	if sched.total_steps > 0 {
		progress := f64(sched.step) / f64(sched.total_steps)
		// lr = min_lr + 0.5*(max-min)*(1 + cos(π*progress))
		sched.current_lr =
			sched.min_lr +
			0.5 * (sched.initial_lr - sched.min_lr) * (1.0 + math.cos(math.PI * progress))
	}
}

// ============================================================================
// Step Decay Scheduler
// ============================================================================

StepDecayScheduler :: struct {
	initial_lr:     f64,
	drop_factor:    f64, // e.g., 0.1 for 10x drop
	steps_per_drop: int,
	step:           int,
	current_lr:     f64,
}

// Contract: start_lr > 0, drop_factor in (0,1], steps_per_drop >= 0
scheduler_step_decay_init :: proc(
	start_lr: f64,
	drop_factor: f64,
	steps_per_drop: int,
) -> StepDecayScheduler {
	return StepDecayScheduler {
		initial_lr = start_lr,
		drop_factor = drop_factor,
		steps_per_drop = steps_per_drop,
		step = 0,
		current_lr = start_lr,
	}
}

// Contract: sched initialized via scheduler_step_decay_init
// Drops LR by drop_factor every steps_per_drop steps
scheduler_step_decay_step :: proc(sched: ^StepDecayScheduler) {
	sched.step += 1
	if sched.steps_per_drop > 0 && sched.step % sched.steps_per_drop == 0 {
		sched.current_lr *= sched.drop_factor
	}
}
