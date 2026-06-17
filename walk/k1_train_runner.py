"""OnPolicyRunner with periodic checkpoint videos for wandb."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from typing import Any

import imageio.v2 as imageio
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from k1_reward_log import build_step_reward_row, episode_summary, reward_names
from k1_video_overlay import render_annotated_frame


class K1TrainRunner(OnPolicyRunner):
    """Saves checkpoints every save_interval and records rollout videos for wandb."""

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str,
        device: str,
        *,
        video_interval: int = 250,
        video_steps: int = 300,
        video_fps: int = 50,
        video_env_cfgs: tuple[dict, dict, dict, dict] | None = None,
        enable_video: bool = True,
        on_iteration_end: Callable[[int, Any], None] | None = None,
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.video_interval = video_interval
        self.video_steps = video_steps
        self.video_fps = video_fps
        self.video_env_cfgs = video_env_cfgs
        self.enable_video = enable_video
        self.on_iteration_end = on_iteration_end
        self._video_env = None

    def _log_video_rollout(
        self,
        it: int,
        step_rows: list[dict[str, float]],
        summary: dict[str, float],
        video_path: str,
    ) -> None:
        """Video-Rollout getrennt vom PPO-Logging — ein einziger wandb.log @ step=it."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        payload: dict = {
            "video_rollout/train_iteration": float(it),
            "video_rollout/video": wandb.Video(video_path, fps=self.video_fps, format="mp4"),
        }
        for k, v in summary.items():
            payload[k] = v
        if step_rows:
            columns = ["frame", *step_rows[0].keys()]
            table = wandb.Table(columns=columns)
            for t, row in enumerate(step_rows):
                table.add_data(t, *[row[c] for c in step_rows[0].keys()])
            payload["video_rollout/step_rewards"] = table
        wandb.log(payload, step=it)

    def _get_video_env(self):
        if self._video_env is None:
            if self.video_env_cfgs is None:
                raise RuntimeError("video_env_cfgs required for video recording")
            from K1_env import K1Env

            env_cfg, obs_cfg, reward_cfg, command_cfg = self.video_env_cfgs
            self._video_env = K1Env(
                num_envs=1,
                env_cfg=copy.deepcopy(env_cfg),
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=False,
                record_camera=True,
            )
        return self._video_env

    def _record_video(self, it: int) -> str:
        video_env = self._get_video_env()
        assert video_env.cam is not None

        video_dir = os.path.join(self.logger.log_dir, "videos")  # type: ignore
        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"train_{it}.mp4")

        policy = self.get_inference_policy(device=self.device)
        frames: list = []
        step_rows: list[dict[str, float]] = []
        names = reward_names(video_env)
        episode_sums = {n: 0.0 for n in names}
        raw_sums = {n: 0.0 for n in names}
        duration_s = self.video_steps / self.video_fps

        with torch.inference_mode():
            obs = video_env.reset()
            for frame_idx in range(self.video_steps):
                actions = policy(obs)
                obs, _, _, _ = video_env.step(actions)

                row = build_step_reward_row(video_env, names)
                step_rows.append(row)
                for n in names:
                    raw_sums[n] += row[f"reward_raw/{n}"]
                    if n in video_env.reward_scales:
                        episode_sums[n] += row.get(f"reward_step/{n}", 0.0)

                if video_env.hero_ghost is not None:
                    video_env.hero_ghost.set_frame(frame_idx, video_env)
                if video_env.cam._followed_entity is not None:
                    video_env.cam.update_following()
                frames.append(render_annotated_frame(video_env))

        imageio.mimsave(path, frames, fps=self.video_fps)
        summary = episode_summary(raw_sums, episode_sums, self.video_steps, duration_s)

        self._pending_video_log = (it, step_rows, summary, path)
        print(f"Recorded rollout video: {path}")
        return path

    def _flush_pending_video_log(self) -> None:
        pending = getattr(self, "_pending_video_log", None)
        if pending is None:
            return
        it, step_rows, summary, path = pending
        self._pending_video_log = None
        self._log_video_rollout(it, step_rows, summary, path)
        print(f"  → video_rollout/* logged @ train step {it}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()
        self._pending_video_log = None

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            record_video = (
                self.enable_video
                and self.logger.writer is not None
                and it % self.video_interval == 0
            )
            if record_video:
                self._record_video(it)

            # Trainings-Metriken ZUERST (globaler Step = it) — unverändert wie vorher.
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Video-Rollout danach: eigene frame-Achse + Summary/Video @ step=it.
            if record_video:
                self._flush_pending_video_log()

            if self.on_iteration_end is not None:
                self.on_iteration_end(it, self)

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
