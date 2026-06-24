"""K1 distillation algorithm: rsl-rl Distillation + the two DAgger-lite ingredients its
stock version lacks (and that the project's other distillation pipeline relies on).

1. Teacher-action mixing in the rollout (beta: 1.0 -> 0.0). The untrained student would
   otherwise immediately drive the robot into states the teacher never saw, where the teacher
   emits extreme/inconsistent targets and BC diverges. Starting beta=1.0 collects teacher-driven
   trajectories, then anneals toward pure student rollout (true DAgger) as the student improves.

2. Clipping the teacher TARGET to the env's clip_actions. The K1 teacher's raw output is large
   (mean |a| ~ 40, max ~ 360) and the env clips it to +-clip_actions before execution. Regressing
   the clipped value = what the robot actually does = bounded, lower-variance targets.

Only the executed action and the BC target change; the MSE behavior-cloning update is unchanged.
"""

from __future__ import annotations

import torch

from rsl_rl.algorithms import Distillation


class K1Distillation(Distillation):
    def __init__(
        self,
        *args,
        beta_decay_iters: int = 300,
        action_clip: float | None = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta_decay_iters = max(1, int(beta_decay_iters))
        self.action_clip = action_clip
        print(
            f"[K1Distillation] DAgger beta 1.0->0.0 over {self.beta_decay_iters} iters; "
            f"teacher target clipped to +-{action_clip}"
        )

    @property
    def beta(self) -> float:
        # num_updates increments exactly once per training iteration (in Distillation.update),
        # so it doubles as the iteration counter for the linear beta anneal.
        return max(0.0, 1.0 - self.num_updates / self.beta_decay_iters)

    def act(self, obs):
        # Roll out the student DETERMINISTICALLY (execute the mean). The BC MSE loss only trains
        # the mean, so the Gaussian std stays pinned at init_std (1.0 raw = 0.25 rad after
        # action_scale ≈ 14°/step of white noise); sampling that during rollout topples the robot
        # as DAgger's beta hands control from teacher to student, collapsing episode length with no
        # recovery. We still do a STOCHASTIC forward so the distribution's internal Normal is built
        # (the runner logs get_policy().output_std → distribution.std, which is None until update()
        # runs) — then take output_mean as the executed action. State coverage comes from the
        # teacher beta-mix below.
        self.student(obs, stochastic_output=True)
        student_a = self.student.output_mean.detach()
        teacher_a = self.teacher(obs).detach()

        # BC target = the action the env actually executes (clipped), not the raw teacher output.
        target = teacher_a if self.action_clip is None else teacher_a.clamp(-self.action_clip, self.action_clip)
        self.transition.privileged_actions = target
        self.transition.observations = obs

        # Executed action: per-env, use the teacher with probability beta, else the student.
        beta = self.beta
        if beta > 0.0:
            use_teacher = torch.rand(student_a.shape[0], 1, device=student_a.device) < beta
            exec_a = torch.where(use_teacher, teacher_a, student_a)
        else:
            exec_a = student_a
        self.transition.actions = exec_a
        return exec_a
