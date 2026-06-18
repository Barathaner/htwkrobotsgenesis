"""Optuna Bayesian tuning of reward_scales + AMP params for K1 locomotion.

Requires: pip install optuna
          wandb login   # once, before first run

Usage (from repo root):
  python walk/k1_tune_rewards.py --n_trials 50
  python walk/k1_tune_rewards.py --study_name k1-reward-tune --resume

Each trial logs to wandb (unique run name) and records rollout videos.
Best params are written to logs/optuna/<study_name>_best.yaml for manual merge into k1_env.yaml.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
OPTUNA_CONFIG_PATH = os.path.join(WALK_DIR, "config", "optuna_reward_tune.yaml")


def load_optuna_cfg(path: str = OPTUNA_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_reward_scales(
    trial,
    baseline_scales: dict[str, float],
    search_space: dict[str, dict],
    fixed_scales: list[str],
) -> dict[str, float]:
    """Sample absolute reward_scales from per-term low/high in search_space."""
    scales: dict[str, float] = {}
    for name, base in baseline_scales.items():
        if name in fixed_scales:
            scales[name] = base
            continue
        bounds = search_space.get(name)
        if bounds is None:
            scales[name] = base
            continue
        low = float(bounds["low"])
        high = float(bounds["high"])
        use_log = bool(bounds.get("log", low > 0 and high > 0))
        scales[name] = trial.suggest_float(f"scale/{name}", low, high, log=use_log)
    return scales


def sample_amp_params(
    trial,
    amp_search_space: dict[str, dict],
) -> dict[str, float]:
    """Sample AMP/train_cfg params (style_weight, amp_reward_scale) from amp_search_space."""
    sampled: dict[str, float] = {}
    for name, bounds in amp_search_space.items():
        low = float(bounds["low"])
        high = float(bounds["high"])
        use_log = bool(bounds.get("log", low > 0 and high > 0))
        sampled[name] = trial.suggest_float(f"amp/{name}", low, high, log=use_log)
    return sampled


def apply_amp_params(train_cfg: dict, amp_params: dict[str, float]) -> dict:
    """Inject sampled AMP params into a copy of train_cfg."""
    cfg = copy.deepcopy(train_cfg)
    if "style_weight" in amp_params:
        cfg["style_weight"] = amp_params["style_weight"]
    if "amp_reward_scale" in amp_params:
        cfg.setdefault("amp", {})["reward_scale"] = amp_params["amp_reward_scale"]
    return cfg


def export_best_params(
    study,
    baseline_scales: dict[str, float],
    amp_search_space: dict[str, dict],
    out_path: str,
) -> dict:
    best = study.best_trial
    scales = sample_reward_scales_from_params(best.params, baseline_scales)
    amp_params = {name: best.params[f"amp/{name}"] for name in amp_search_space if f"amp/{name}" in best.params}
    payload = {
        "best_value": best.value,
        "best_trial": best.number,
        "reward_scales": scales,
        "amp_params": amp_params,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    return payload


def sample_reward_scales_from_params(
    params: dict[str, float],
    baseline_scales: dict[str, float],
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name, base in baseline_scales.items():
        key = f"scale/{name}"
        scales[name] = params[key] if key in params else base
    return scales


def print_top_trials(
    study,
    baseline_scales: dict[str, float],
    amp_search_space: dict[str, dict],
    n: int = 5,
) -> None:
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    print(f"\nTop {n} trials:")
    for t in trials[:n]:
        if t.value is None:
            continue
        scales = sample_reward_scales_from_params(t.params, baseline_scales)
        amp_params = {name: t.params[f"amp/{name}"] for name in amp_search_space if f"amp/{name}" in t.params}
        print(f"  trial {t.number:3d}  score={t.value:.4f}")
        for k, v in sorted(scales.items()):
            print(f"    scale/{k:20s} {v:+.4f}")
        for k, v in sorted(amp_params.items()):
            print(f"    amp/{k:20s} {v:+.4f}")


def trial_run_name(study_name: str, trial_number: int) -> str:
    return f"{study_name}-trial-{trial_number:04d}"


def main() -> None:
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Please install optuna: pip install optuna") from e

    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)

    from k1_train_session import (
        apply_reward_scales,
        build_train_cfg,
        create_k1_env,
        create_train_runner,
        init_genesis,
        load_all_cfgs,
        run_training,
        save_session_cfgs,
    )
    from k1_tune_objective import (
        eval_raw_means,
        hero_composite_score,
        intermediate_training_metric,
    )

    optuna_cfg = load_optuna_cfg()
    study_cfg = optuna_cfg["study"]
    trial_cfg = optuna_cfg["trial"]
    search_space = optuna_cfg["search_space"]
    amp_search_space = optuna_cfg.get("amp_search_space", {})
    objective_weights = optuna_cfg["objective"]["weights"]
    fixed_scales = optuna_cfg.get("fixed_scales", [])

    parser = argparse.ArgumentParser(description="Optuna reward-scale + AMP tuning for K1")
    parser.add_argument("--n_trials", type=int, default=study_cfg.get("n_trials", 50))
    parser.add_argument("--study_name", type=str, default=study_cfg.get("study_name", "k1-reward-tune"))
    parser.add_argument("--storage", type=str, default=study_cfg.get("storage", "logs/optuna/k1_reward_tune.db"))
    parser.add_argument("--config", type=str, default=OPTUNA_CONFIG_PATH)
    parser.add_argument("--resume", action="store_true", help="Resume existing study in storage")
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    if args.config != OPTUNA_CONFIG_PATH:
        optuna_cfg = load_optuna_cfg(args.config)
        study_cfg = optuna_cfg["study"]
        trial_cfg = optuna_cfg["trial"]
        search_space = optuna_cfg["search_space"]
        amp_search_space = optuna_cfg.get("amp_search_space", {})
        objective_weights = optuna_cfg["objective"]["weights"]
        fixed_scales = optuna_cfg.get("fixed_scales", [])

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_yaml = load_all_cfgs()
    baseline_scales = dict(reward_cfg["reward_scales"])
    wandb_project = (
        args.wandb_project
        or trial_cfg.get("wandb_project")
        or train_cfg_yaml.get("wandb_project", "k1-locomotion")
    )

    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.resolve()}"

    sampler_name = study_cfg.get("sampler", "TPE").upper()
    seed = trial_cfg.get("seed", 1)
    if sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "CMA":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler_name == "RANDOM":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}'. Supported: TPE, CMA, RANDOM")

    pruner = None
    if study_cfg.get("pruner", "").lower() == "median":
        pruner = optuna.pruners.MedianPruner()

    if args.resume:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage_url,
            sampler=sampler,
        )
    else:
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=storage_url,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=False,
            )
        except optuna.exceptions.DuplicatedStudyError:
            raise SystemExit(
                f"Study '{args.study_name}' already exists in {storage_path}.\n"
                f"  • Resume it:      --resume\n"
                f"  • Start fresh:    delete {storage_path} or use a different --study_name"
            )

    max_iterations = trial_cfg["max_iterations"]
    num_envs = trial_cfg["num_envs"]
    eval_steps = trial_cfg["eval_steps"]
    report_interval = trial_cfg.get("report_interval", 100)
    base_seed = trial_cfg.get("seed", 1)

    init_genesis(base_seed)

    def objective(trial: optuna.Trial) -> float:
        scales = sample_reward_scales(trial, baseline_scales, search_space, fixed_scales)
        amp_params = sample_amp_params(trial, amp_search_space)
        trial_reward_cfg = apply_reward_scales(reward_cfg, scales)
        run_name = trial_run_name(args.study_name, trial.number)

        log_dir = os.path.join("logs", "optuna", args.study_name, f"trial_{trial.number:04d}")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        train_cfg, video_opts = build_train_cfg(
            train_cfg_yaml,
            run_name,
            wandb_project=wandb_project,
            logger_class="WandbLogWriter",
        )
        train_cfg = apply_amp_params(train_cfg, amp_params)
        train_cfg["save_interval"] = trial_cfg.get("save_interval", 10000)
        if "video_interval" in trial_cfg:
            video_opts["video_interval"] = int(trial_cfg["video_interval"])
        enable_video = bool(trial_cfg.get("enable_video", False))

        save_session_cfgs(
            log_dir, env_cfg, obs_cfg, trial_reward_cfg, command_cfg, train_cfg, video_opts
        )

        env = create_k1_env(
            env_cfg,
            obs_cfg,
            trial_reward_cfg,
            command_cfg,
            num_envs=num_envs,
        )

        optuna_trial = trial

        def on_iteration_end(it: int, runner) -> None:
            if (it + 1) % report_interval != 0:
                return
            metric = intermediate_training_metric(runner)
            optuna_trial.report(metric, it)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        runner = create_train_runner(
            env,
            train_cfg,
            log_dir,
            (env_cfg, obs_cfg, trial_reward_cfg, command_cfg),
            video_opts,
            enable_video=enable_video,
            on_iteration_end=on_iteration_end,
        )

        try:
            print(
                f"\n[optuna] trial {trial.number}: style_weight={amp_params.get('style_weight', '?'):.3f}  "
                f"amp_reward_scale={amp_params.get('amp_reward_scale', '?'):.3f}  "
                f"wandb='{run_name}'"
            )
            try:
                run_training(runner, max_iterations)
            except optuna.TrialPruned:
                try:
                    if runner.logger.writer is not None:
                        runner.logger.stop_logging_writer()
                except Exception:
                    pass
                raise
            policy = runner.get_inference_policy(device=runner.device)
            raw_means = eval_raw_means(env, policy, n_steps=eval_steps)
            score = hero_composite_score(raw_means, objective_weights)
            trial.set_user_attr("raw_means", raw_means)
            trial.set_user_attr("reward_scales", scales)
            trial.set_user_attr("amp_params", amp_params)
            trial.set_user_attr("wandb_run_name", run_name)
            return score
        finally:
            del runner, env
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"Optuna study '{args.study_name}': {args.n_trials} trials, "
        f"{max_iterations} iters, {num_envs} envs, wandb={wandb_project}, storage={args.storage}"
    )
    if amp_search_space:
        print(f"AMP params in search space: {list(amp_search_space.keys())}")
    study.optimize(objective, n_trials=args.n_trials)

    out_path = os.path.join("logs", "optuna", f"{args.study_name}_best.yaml")
    best = export_best_params(study, baseline_scales, amp_search_space, out_path)
    print(f"\nBest trial {study.best_trial.number}: score={study.best_value:.4f}")
    print(f"Best params written to {out_path}")
    for k, v in sorted(best.get("reward_scales", {}).items()):
        print(f"  scale/{k:20s} {v:+.4f}")
    for k, v in sorted(best.get("amp_params", {}).items()):
        print(f"  amp/{k:20s} {v:+.4f}")
    print_top_trials(study, baseline_scales, amp_search_space)


if __name__ == "__main__":
    main()
