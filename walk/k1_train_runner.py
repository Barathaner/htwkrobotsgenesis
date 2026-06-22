"""OnPolicyRunner with AMP_PPO (amp-rsl-rl) and periodic checkpoint videos for wandb."""

from __future__ import annotations

import collections
import copy
import os
import time
from collections.abc import Callable
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from k1_video_overlay import render_annotated_frame, update_camera_centroid


class K1TrainRunner(OnPolicyRunner):
    """OnPolicyRunner extended with AMP_PPO (amp-rsl-rl), video recording and curriculum.

    After OnPolicyRunner.__init__ constructs the PPO actor/critic, __init__ replaces
    self.alg with AMP_PPO reusing the same actor and critic, then initialises the
    amp-rsl-rl Discriminator and K1AMPLoader expert-data loader.
    """

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

        # ── Swap standard PPO for AMP_PPO ────────────────────────────────────
        self.style_weight: float = self.cfg.get("style_weight", 0.5)
        amp_cfg = self.cfg.get("amp", {})
        self._setup_amp_ppo(amp_cfg)

        self.video_interval = video_interval
        self.video_steps = video_steps
        self.video_fps = video_fps
        self.video_env_cfgs = video_env_cfgs
        self.enable_video = enable_video
        self.on_iteration_end = on_iteration_end
        self._video_env = None
        self._setup_conditioned_amp_data()

    def _setup_amp_ppo(self, amp_cfg: dict) -> None:
        """Replace self.alg (PPO) with AMP_PPO, reusing the actor/critic already built."""
        import inspect
        from amp_rsl_rl.algorithms import AMP_PPO
        from amp_rsl_rl.networks import Discriminator
        from k1_amp_loader import K1AMPLoader

        obs = self.env.get_observations()
        if "amp" not in obs.keys():
            raise RuntimeError("K1Env must include 'amp' key in get_observations() for AMP training.")
        amp_obs_dim = obs["amp"].shape[-1]

        discriminator = Discriminator(
            input_dim=amp_obs_dim * 2,
            hidden_layer_sizes=amp_cfg.get("hidden_dims", [256, 128]),
            reward_scale=amp_cfg.get("reward_scale", 1.0),
            loss_type=amp_cfg.get("loss_type", "BCEWithLogits"),
            empirical_normalization=False,  # K1AMPLoader pre-normalizes identically to _build_amp_obs
            device=self.device,
        ).to(self.device)

        # Include jog NPZ alongside slow NPZ when jogging curriculum is enabled so the
        # single discriminator training pass (via alg.update) covers both motion styles.
        cmd_cfg = self.env.command_cfg
        jog_cfg = self.env.env_cfg.get("jogging_curriculum", {})
        jog_path = jog_cfg.get("style_motion_file") if jog_cfg.get("enabled", False) else None
        default_slow = self.env.reward_cfg["style_motion_file"]
        motion_paths = amp_cfg.get("motion_paths",
                                   [default_slow] + ([jog_path] if jog_path else []))

        # Per-file command ranges: expert transitions get commands sampled from the
        # appropriate speed range so the discriminator learns to condition on velocity.
        yaw = cmd_cfg["ang_vel_range"]
        slow_lo = np.array([cmd_cfg["lin_vel_x_range"][0], cmd_cfg["lin_vel_y_range"][0], yaw[0]], dtype=np.float32)
        slow_hi = np.array([cmd_cfg["lin_vel_x_range"][1], cmd_cfg["lin_vel_y_range"][1], yaw[1]], dtype=np.float32)
        cmd_ranges = [(slow_lo, slow_hi)]
        if jog_path:
            jog_lo = np.array([jog_cfg["target_lin_vel_x_range"][0], jog_cfg["target_lin_vel_y_range"][0], yaw[0]], dtype=np.float32)
            jog_hi = np.array([jog_cfg["target_lin_vel_x_range"][1], jog_cfg["target_lin_vel_y_range"][1], yaw[1]], dtype=np.float32)
            cmd_ranges.append((jog_lo, jog_hi))

        joint_names = self.env.env_cfg["joint_names"]
        amp_data = K1AMPLoader(motion_paths, joint_names, device=self.device, cmd_ranges=cmd_ranges)

        # Push combined motion normalization stats to the env so _build_amp_obs() uses the
        # same feature space as the expert data (commands are appended raw in both).
        self.env._amp_obs_mean   = amp_data.mean
        self.env._amp_obs_std    = amp_data.std
        self.env._amp_obs_sqrt_w = amp_data.sqrt_w

        ppo = self.alg
        # Filter cfg["algorithm"] to only keys AMP_PPO accepts
        amp_ppo_params = set(inspect.signature(AMP_PPO.__init__).parameters)
        alg_kwargs = {k: v for k, v in self.cfg["algorithm"].items() if k in amp_ppo_params}
        # amp_replay_buffer_size lives at train_cfg level (not under algorithm)
        if "amp_replay_buffer_size" in self.cfg:
            alg_kwargs["amp_replay_buffer_size"] = self.cfg["amp_replay_buffer_size"]

        self.alg = AMP_PPO(
            actor=ppo.actor,
            critic=ppo.critic,
            discriminator=discriminator,
            amp_data=amp_data,
            device=self.device,
            **alg_kwargs,
        )
        self.alg.init_storage(
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs.clone().detach().to(self.device),
            (self.env.num_actions,),
        )
        self.discriminator = discriminator
        print(f"[K1TrainRunner] AMP_PPO active  amp_obs_dim={amp_obs_dim}  "
              f"style_weight={self.style_weight}")

    def _setup_conditioned_amp_data(self) -> None:
        # Both NPZs are now loaded in _setup_amp_ppo (combined into amp_data) so the
        # single alg.update() training pass covers both motion styles without a second
        # optimizer on the same network.  This method is kept as a no-op so external
        # call sites don't need to change.
        self.amp_slow = None
        self.amp_jog = None
        self.opt_disc_cond = None

    def get_inference_policy(self, device=None):
        """Return a callable policy for inference (eval mode, deterministic)."""
        self.alg.test_mode()
        actor = self.alg.actor
        if device is not None:
            actor = actor.to(device)
        return lambda obs: actor(obs, stochastic_output=False)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save actor, critic, discriminator and optimizer state."""
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict | None:
        """Load actor, critic, discriminator and optimizer state."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location or self.device)
        load_cfg = load_cfg or {}
        if load_cfg.get("actor", True) and "actor_state_dict" in loaded_dict:
            self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic", True) and "critic_state_dict" in loaded_dict:
            self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("discriminator", True) and "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=False)
        if load_cfg.get("optimizer", True) and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def _log_video_rollout(self, it: int, video_path: str) -> None:
        """Log video to wandb @ step=it."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.log({"video_rollout/video": wandb.Video(video_path, fps=self.video_fps, format="mp4")}, step=it)

    def _get_video_env(self):
        if self._video_env is None:
            if self.video_env_cfgs is None:
                raise RuntimeError("video_env_cfgs required for video recording")
            from K1_env import K1Env

            env_cfg, obs_cfg, reward_cfg, command_cfg = self.video_env_cfgs
            video_num_envs = env_cfg.get("video", {}).get("num_envs", 1)
            self._video_env = K1Env(
                num_envs=video_num_envs,
                env_cfg=copy.deepcopy(env_cfg),
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=False,
                record_camera=True,
            )
            # If the video env is created after the jogging curriculum already
            # triggered (lazy init past start_iter), apply it immediately.
            self._video_env.sync_jogging_curriculum(self.current_learning_iteration)
        return self._video_env

    def _record_video(self, it: int) -> str:
        video_env = self._get_video_env()
        assert video_env.cam is not None

        video_dir = os.path.join(self.logger.log_dir, "videos")  # type: ignore
        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"train_{it}.mp4")

        policy = self.get_inference_policy(device=self.device)
        frames: list = []

        _TERM_REASONS = ("timeout", "pitch", "roll", "height", "sim_error")
        video_term: dict[str, int] = {r: 0 for r in _TERM_REASONS}
        video_term_total = 0

        with torch.inference_mode():
            obs = video_env.reset()
            for _ in range(self.video_steps):
                actions = policy(obs)
                obs, _, dones, _ = video_env.step(actions)

                done_mask = dones.bool()
                if done_mask.any():
                    video_term_total += int(done_mask.sum().item())
                    for reason in _TERM_REASONS:
                        rm = getattr(video_env, f"_term_{reason}", None)
                        if rm is not None:
                            video_term[reason] += int((rm & done_mask).sum().item())

                if video_env.hero_ghost is not None:
                    video_env.hero_ghost.set_frame(video_env)
                if video_env.color_shadow is not None:
                    video_env.color_shadow.update(video_env)
                if video_env.cam._followed_entity is not None:
                    video_env.cam.update_following()
                update_camera_centroid(video_env)
                frames.append(render_annotated_frame(video_env))

        imageio.mimsave(
            path,
            frames,
            fps=self.video_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "18"],
        )
        self._pending_video_log = (it, path)

        # Print + wandb-log termination breakdown for this video rollout
        if video_term_total > 0:
            parts = [f"{r}={video_term[r]}" for r in _TERM_REASONS if video_term[r] > 0]
            print(f"Recorded rollout video: {path}  |  resets={video_term_total} ({', '.join(parts)})")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {f"video_termination/{r}": video_term[r] for r in _TERM_REASONS},
                        step=it,
                    )
            except ImportError:
                pass
        else:
            print(f"Recorded rollout video: {path}  |  no resets in {self.video_steps} steps")
        return path

    def _flush_pending_video_log(self) -> None:
        pending = getattr(self, "_pending_video_log", None)
        if pending is None:
            return
        it, path = pending
        self._pending_video_log = None
        self._log_video_rollout(it, path)
        print(f"  → video_rollout logged @ train step {it}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = obs["amp"].clone()
        self.alg.train_mode()

        # Per-env accumulators + deques for exact Mean reward breakdown logged to wandb
        _task_ep_buf = torch.zeros(self.env.num_envs, device=self.device)
        _style_ep_buf = torch.zeros(self.env.num_envs, device=self.device)
        _task_ep_deque: collections.deque[float] = collections.deque(maxlen=200)
        _style_ep_deque: collections.deque[float] = collections.deque(maxlen=200)

        # Per-reason termination counters (reset each iteration)
        _TERM_REASONS = ("timeout", "pitch", "roll", "height", "sim_error")
        _term_counts: dict[str, int] = {r: 0 for r in _TERM_REASONS}
        _term_total = 0

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()
        self._pending_video_log = None

        # ── Style weight linear warmup ────────────────────────────────────────
        _sw_target = self.cfg.get("style_weight", 0.5)
        _sw_start = self.cfg.get("style_warmup_start", 0)
        _sw_duration = max(1, self.cfg.get("style_warmup_duration", 1))

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            # Linear style weight warmup: 0 before _sw_start, ramp to target, then flat
            if it < _sw_start:
                self.style_weight = 0.0
            elif it < _sw_start + _sw_duration:
                self.style_weight = _sw_target * (it - _sw_start) / _sw_duration
            else:
                self.style_weight = _sw_target

            # ── Jogging curriculum ────────────────────────────────────────────
            # Hard-switch command ranges at start_iter (velocity range only; style
            # conditioning is handled continuously by _setup_conditioned_amp_data).
            self.env.update_jogging_curriculum(it)
            if self._video_env is not None:
                self._video_env.update_jogging_curriculum(it)

            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    self.alg.act_amp(amp_obs)

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    next_amp_obs = obs["amp"].clone()

                    # Style reward: multiplicative gating — style is a bonus multiplier on task.
                    # r_total = task * (1 + style_weight * style_disc)
                    # When task → 0 (robot fails velocity command) the style bonus collapses too,
                    # so the agent cannot earn reward by looking stylish while standing still.
                    task_rewards = rewards.clone()
                    style_reward = self.discriminator.predict_reward(amp_obs, next_amp_obs)
                    rewards = task_rewards * (1.0 + self.style_weight * style_reward)

                    # Log actual contributions so rew_ep_task + rew_ep_style ≈ mean_reward.
                    style_contribution = (task_rewards * self.style_weight * style_reward).detach()
                    _task_ep_buf += task_rewards.detach()
                    _style_ep_buf += style_contribution

                    done_mask = dones.bool()
                    if done_mask.any():
                        for v in _task_ep_buf[done_mask].tolist():
                            _task_ep_deque.append(v)
                        for v in _style_ep_buf[done_mask].tolist():
                            _style_ep_deque.append(v)
                        _task_ep_buf[done_mask] = 0.0
                        _style_ep_buf[done_mask] = 0.0

                        # Per-reason termination counts (env stores _term_* from last _termination_mask)
                        n = done_mask.sum().item()
                        _term_total += n
                        for reason in _TERM_REASONS:
                            reason_mask = getattr(self.env, f"_term_{reason}", None)
                            if reason_mask is not None:
                                _term_counts[reason] += int((reason_mask & done_mask).sum().item())

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.alg.process_amp_step(next_amp_obs)
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards=None)
                    amp_obs = next_amp_obs

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_kl_divergence,
                _mean_symmetry_loss,
            ) = self.alg.update()

            loss_dict = {
                "value_function": mean_value_loss,
                "surrogate": mean_surrogate_loss,
                "amp_loss": mean_amp_loss,
                "grad_pen": mean_grad_pen_loss,
                "disc_policy_pred": mean_policy_pred,
                "disc_expert_pred": mean_expert_pred,
                "disc_accuracy_policy": mean_accuracy_policy,
                "disc_accuracy_expert": mean_accuracy_expert,
                "kl_divergence": mean_kl_divergence,
            }

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # ── Reward breakdown: task vs AMP style (exact ingredients of Mean reward) ──
            if _task_ep_deque:
                loss_dict["reward_task"] = sum(_task_ep_deque) / len(_task_ep_deque)
                loss_dict["reward_style"] = sum(_style_ep_deque) / len(_style_ep_deque)

            loss_dict["style_weight"] = self.style_weight

            # ── Termination reason breakdown ──────────────────────────────────
            if _term_total > 0:
                for reason in _TERM_REASONS:
                    loss_dict[f"termination/{reason}_pct"] = 100.0 * _term_counts[reason] / _term_total
            _term_counts = {r: 0 for r in _TERM_REASONS}
            _term_total = 0

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
                action_std=self.alg.actor.output_std,
                rnd_weight=None,
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
