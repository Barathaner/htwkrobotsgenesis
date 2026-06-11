"""OnPolicyRunner with periodic checkpoint videos for wandb."""

from __future__ import annotations

import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


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
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.video_interval = video_interval
        self.video_steps = video_steps
        self.video_fps = video_fps
        self.video_env_cfgs = video_env_cfgs
        self._video_env = None

    def _get_video_env(self):
        if self._video_env is None:
            if self.video_env_cfgs is None:
                raise RuntimeError("video_env_cfgs required for video recording")
            from K1_env import K1Env

            env_cfg, obs_cfg, reward_cfg, command_cfg = self.video_env_cfgs
            self._video_env = K1Env(
                num_envs=1,
                env_cfg=env_cfg,
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
        video_env.cam.start_recording()

        with torch.inference_mode():
            obs = video_env.reset()
            for _ in range(self.video_steps):
                actions = policy(obs)
                obs, _, _, _ = video_env.step(actions)
                if video_env.cam._followed_entity is not None:
                    video_env.cam.update_following()  # Kamera dem Roboter nachführen
                video_env.cam.render()

        video_env.cam.stop_recording(save_to_filename=path, fps=self.video_fps)
        print(f"Recorded rollout video: {path}")
        return path

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

            if self.logger.writer is not None and it % self.video_interval == 0:
                self._record_video(it)

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

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
