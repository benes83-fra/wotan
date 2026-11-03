package optimize

import "core:math"

// ============================================================================
// Learning-rate schedules
// ============================================================================
// The learning rate of step 0, 1, 2, … as a closed form (so training can resume
// at any step): an optional linear warmup from warmup_start·base_lr to base_lr
// over warmup_steps, then
//   Constant: base_lr
//   Step:     base_lr·gamma^⌊t / step_size⌋
//   Linear:   base_lr → final_lr over the remaining steps up to total_steps
//   Cosine:   base_lr → final_lr along a half cosine up to total_steps
// where t counts the steps after the warmup; Linear and Cosine stay at
// final_lr after total_steps. As torch.optim.lr_scheduler: SequentialLR of
// LinearLR(warmup_start, 1, warmup_steps) and ConstantLR(1) / StepLR /
// LinearLR / CosineAnnealingLR(eta_min = final_lr). Set the optimizer's
// learning rate before each step:
//   for it in 0 ..< n {opt.learning_rate = lr_schedule_at(s, it); …; adam_step(&opt)}
// With per-parameter rates use nn.adam_step_scheduled(&opt, s, it): each one
// is scheduled as its own torch param group.

LR_Decay :: enum {
	Constant,
	Step,
	Linear,
	Cosine,
}

LR_Schedule :: struct {
	decay:        LR_Decay,
	base_lr:      f64,
	final_lr:     f64, // Linear, Cosine
	total_steps:  int, // Linear, Cosine: including the warmup
	step_size:    int, // Step
	gamma:        f64, // Step
	warmup_steps: int,
	warmup_start: f64, // fraction of base_lr at step 0 of the warmup
}

lr_constant :: proc(base_lr: f64, warmup_steps := 0, warmup_start := 0.1) -> LR_Schedule {
	return {.Constant, base_lr, base_lr, 0, 0, 1, warmup_steps, warmup_start}
}

// Multiplies the learning rate by gamma every step_size steps
lr_step_decay :: proc(
	base_lr: f64,
	step_size: int,
	gamma: f64,
	warmup_steps := 0,
	warmup_start := 0.1,
) -> LR_Schedule {
	assert(step_size > 0, "lr_step_decay: step_size must be positive")
	return {.Step, base_lr, 0, 0, step_size, gamma, warmup_steps, warmup_start}
}

lr_linear_decay :: proc(
	base_lr: f64,
	total_steps: int,
	final_lr := 0.0,
	warmup_steps := 0,
	warmup_start := 0.1,
) -> LR_Schedule {
	assert(total_steps > warmup_steps, "lr_linear_decay: total_steps must exceed warmup_steps")
	return {.Linear, base_lr, final_lr, total_steps, 0, 1, warmup_steps, warmup_start}
}

lr_cosine :: proc(
	base_lr: f64,
	total_steps: int,
	final_lr := 0.0,
	warmup_steps := 0,
	warmup_start := 0.1,
) -> LR_Schedule {
	assert(total_steps > warmup_steps, "lr_cosine: total_steps must exceed warmup_steps")
	return {.Cosine, base_lr, final_lr, total_steps, 0, 1, warmup_steps, warmup_start}
}

// The learning rate of step `step` (0-based).
// base_lr > 0: the rate of a parameter group with its own base rate, as torch
// schedules each param group: warmup, Constant, Step and Linear scale it like
// s.base_lr (LinearLR / StepLR are factors), Cosine runs from it to the same
// absolute final_lr (CosineAnnealingLR's eta_min).
lr_schedule_at :: proc(s: LR_Schedule, step: int, base_lr := 0.0) -> f64 {
	base := base_lr if base_lr > 0 else s.base_lr
	if step < s.warmup_steps {
		frac := f64(step) / f64(s.warmup_steps)
		return base * (s.warmup_start + (1.0 - s.warmup_start) * frac)
	}
	t := step - s.warmup_steps
	switch s.decay {
	case .Constant:
		return base
	case .Step:
		return base * math.pow(s.gamma, f64(t / s.step_size))
	case .Linear:
		final := s.final_lr if base == s.base_lr else s.final_lr * base / s.base_lr
		frac := f64(t) / f64(s.total_steps - s.warmup_steps)
		if frac >= 1.0 {return final}
		return base + (final - base) * frac
	case .Cosine:
		frac := f64(t) / f64(s.total_steps - s.warmup_steps)
		if frac >= 1.0 {return s.final_lr}
		return s.final_lr + 0.5 * (base - s.final_lr) * (1.0 + math.cos(math.PI * frac))
	}
	return base
}
