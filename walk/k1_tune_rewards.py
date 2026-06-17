"""Optuna Bayesian tuning of reward_scales for K1 locomotion.

Requires: pip install optuna
          wandb login   # once, before first run

Usage (from repo root):
  # run 50 trials
  python walk/k1_tune_rewards.py --n_trials 50

  # resume an existing study
  python walk/k1_tune_rewards.py --study_name k1-reward-tune --resume

  # quick single trial (no Optuna, just train once with baseline scales)
  python walk/k1_tune_rewards.py --single_run --max_iterations 500 --num_envs 2048

  # after a study, write best scales back into k1_env.yaml
  python walk/k1_tune_rewards.py --resume --n_trials 0 --apply_best

Outputs:
  logs/optuna/<study_name>/trial_NNNN/   checkpoints + cfgs per trial
  logs/optuna/<study_name>_best.yaml     best scales (human-readable diff)
  k1_env.yaml                            patched in-place when --apply_best
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WALK_DIR, "config", "k1_env.yaml")
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


def sample_reward_scales_from_params(
    params: dict[str, float],
    baseline_scales: dict[str, float],
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name, base in baseline_scales.items():
        key = f"scale/{name}"
        scales[name] = params[key] if key in params else base
    return scales


def export_best_scales(study, baseline_scales: dict[str, float], out_path: str) -> dict[str, float]:
    best = study.best_trial
    scales = sample_reward_scales_from_params(best.params, baseline_scales)
    payload = {
        "best_value": best.value,
        "best_trial": best.number,
        "reward_scales": scales,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    return scales


def apply_best_to_config(best_scales: dict[str, float], config_path: str) -> None:
    """Patch reward_scales in k1_env.yaml in-place. Creates a .bak backup first."""
    backup = config_path + ".bak"
    shutil.copy2(config_path, backup)
    print(f"[apply_best] backed up {config_path} → {backup}")

    with open(config_path, "r") as f:
        all_cfg = yaml.safe_load(f)

    current = all_cfg["reward_cfg"]["reward_scales"]
    print("\n[apply_best] scale changes:")
    for k, v in sorted(best_scales.items()):
        old = current.get(k, "—")
        marker = "  (unchanged)" if isinstance(old, float) and abs(old - v) < 1e-6 else ""
        print(f"  {k:22s} {str(old):>8} → {v:+.4f}{marker}")

    all_cfg["reward_cfg"]["reward_scales"] = best_scales
    with open(config_path, "w") as f:
        yaml.safe_dump(all_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[apply_best] wrote updated config to {config_path}")
    print("  Note: YAML comments were stripped — restore from .bak if needed.")


def print_top_trials(study, baseline_scales: dict[str, float], n: int = 5) -> None:
    trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )
    print(f"\nTop {n} trials:")
    for t in trials[:n]:
        if t.value is None:
            continue
        scales = sample_reward_scales_from_params(t.params, baseline_scales)
        print(f"  trial {t.number:3d}  score={t.value:.4f}")
        for k, v in sorted(scales.items()):
            base = baseline_scales.get(k, 0.0)
            delta = v - base
            print(f"    {k:22s} {v:+.4f}  (Δ{delta:+.4f} from baseline)")


def trial_run_name(study_name: str, trial_number: int) -> str:
    return f"{study_name}-trial-{trial_number:04d}"


def main() -> None:
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Please install optuna: pip install optuna optuna-dashboard") from e

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
    objective_weights = optuna_cfg["objective"]["weights"]
    fixed_scales = optuna_cfg.get("fixed_scales", [])

    parser = argparse.ArgumentParser(description="Optuna reward-scale tuning for K1")
    parser.add_argument("--n_trials", type=int, default=study_cfg.get("n_trials", 50))
    parser.add_argument("--study_name", type=str, default=study_cfg.get("study_name", "k1-reward-tune"))
    parser.add_argument("--storage", type=str, default=study_cfg.get("storage", "logs/optuna/k1_reward_tune.db"))
    parser.add_argument("--config", type=str, default=OPTUNA_CONFIG_PATH)
    parser.add_argument("--resume", action="store_true", help="Resume existing study from storage")
    parser.add_argument("--apply_best", action="store_true", help="After study, patch best scales into k1_env.yaml")
    parser.add_argument("--single_run", action="store_true", help="Train once with baseline scales (no Optuna)")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override trial.max_iterations")
    parser.add_argument("--num_envs", type=int, default=None, help="Override trial.num_envs")
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    if args.config != OPTUNA_CONFIG_PATH:
        optuna_cfg = load_optuna_cfg(args.config)
        study_cfg = optuna_cfg["study"]
        trial_cfg = optuna_cfg["trial"]
        search_space = optuna_cfg["search_space"]
        objective_weights = optuna_cfg["objective"]["weights"]
        fixed_scales = optuna_cfg.get("fixed_scales", [])

    max_iterations = args.max_iterations or trial_cfg["max_iterations"]
    num_envs = args.num_envs or trial_cfg["num_envs"]
    eval_steps = trial_cfg["eval_steps"]
    report_interval = trial_cfg.get("report_interval", 100)
    base_seed = trial_cfg.get("seed", 1)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_yaml = load_all_cfgs()
    baseline_scales = dict(reward_cfg["reward_scales"])
    wandb_project = (
        args.wandb_project
        or trial_cfg.get("wandb_project")
        or train_cfg_yaml.get("wandb_project", "k1-locomotion")
    )

    init_genesis(base_seed)

    # ── single_run: train once with baseline scales, no Optuna ───────────────
    if args.single_run:
        run_name = f"{args.study_name}-single"
        log_dir = os.path.join("logs", "optuna", args.study_name, "single_run")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        train_cfg, video_opts = build_train_cfg(
            train_cfg_yaml, run_name, wandb_project=wandb_project
        )
        save_session_cfgs(log_dir, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, video_opts)
        env = create_k1_env(env_cfg, obs_cfg, reward_cfg, command_cfg, num_envs=num_envs)
        runner = create_train_runner(
            env, train_cfg, log_dir, (env_cfg, obs_cfg, reward_cfg, command_cfg), video_opts, enable_video=True
        )
        print(f"[single_run] {max_iterations} iters, {num_envs} envs, wandb={wandb_project}")
        run_training(runner, max_iterations)
        return

    # ── Optuna study ──────────────────────────────────────────────────────────
    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.resolve()}"

    sampler_name = study_cfg.get("sampler", "TPE").upper()
    sampler = optuna.samplers.TPESampler(seed=base_seed)
    if sampler_name == "CMA":
        sampler = optuna.samplers.CmaEsSampler(seed=base_seed)

    pruner = None
    if study_cfg.get("pruner", "").lower() == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=report_interval * 2)

    if args.resume:
        study = optuna.load_study(study_name=args.study_name, storage=storage_url)
        print(f"[optuna] resumed study '{args.study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_url,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False,
        )

    tuned = [k for k in baseline_scales if k not in fixed_scales and k in search_space]
    print(
        f"\n[optuna] study='{args.study_name}'  trials={args.n_trials}  "
        f"iters/trial={max_iterations}  envs={num_envs}  wandb={wandb_project}\n"
        f"  tuning {len(tuned)} scales: {', '.join(sorted(tuned))}\n"
        f"  fixed:  {', '.join(sorted(fixed_scales)) or '—'}\n"
    )

    def objective(trial) -> float:
        scales = sample_reward_scales(trial, baseline_scales, search_space, fixed_scales)
        trial_reward_cfg = apply_reward_scales(reward_cfg, scales)
        run_name = trial_run_name(args.study_name, trial.number)
        log_dir = os.path.join("logs", "optuna", args.study_name, f"trial_{trial.number:04d}")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        train_cfg, video_opts = build_train_cfg(
            train_cfg_yaml,
            run_name,
            wandb_project=wandb_project,
        )
        train_cfg["save_interval"] = trial_cfg.get("save_interval", 10000)
        if "video_interval" in trial_cfg:
            video_opts["video_interval"] = int(trial_cfg["video_interval"])

        save_session_cfgs(log_dir, env_cfg, obs_cfg, trial_reward_cfg, command_cfg, train_cfg, video_opts)
        env = create_k1_env(env_cfg, obs_cfg, trial_reward_cfg, command_cfg, num_envs=num_envs)

        def on_iteration_end(it: int, runner) -> None:
            if (it + 1) % report_interval != 0:
                return
            metric = intermediate_training_metric(runner)
            trial.report(metric, it)
            if trial.should_prune():
                raise optuna.TrialPruned()

        runner = create_train_runner(
            env,
            train_cfg,
            log_dir,
            (env_cfg, obs_cfg, trial_reward_cfg, command_cfg),
            video_opts,
            enable_video=True,
            on_iteration_end=on_iteration_end,
        )

        try:
            print(f"\n[trial {trial.number:03d}] start  run='{run_name}'")
            run_training(runner, max_iterations)
            policy = runner.get_inference_policy(device=runner.device)
            raw_means = eval_raw_means(env, policy, n_steps=eval_steps)
            score = hero_composite_score(raw_means, objective_weights)
            trial.set_user_attr("raw_means", {k: round(v, 4) for k, v in raw_means.items()})
            trial.set_user_attr("reward_scales", {k: round(v, 4) for k, v in scales.items()})
            trial.set_user_attr("wandb_run_name", run_name)
            print(f"[trial {trial.number:03d}] score={score:.4f}  " +
                  "  ".join(f"{k}={v:.3f}" for k, v in sorted(raw_means.items())
                            if k in ("tracking_lin_vel", "style", "foot_clearance", "swing_clearance")))
            return score
        finally:
            del runner, env
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.n_trials > 0:
        study.optimize(objective, n_trials=args.n_trials)

    if len(study.trials) == 0 or study.best_trial is None:
        print("No completed trials yet.")
        return

    out_path = os.path.join("logs", "optuna", f"{args.study_name}_best.yaml")
    best_scales = export_best_scales(study, baseline_scales, out_path)
    print(f"\nBest trial {study.best_trial.number}: score={study.best_value:.4f}")
    print(f"Best scales written to {out_path}")
    print_top_trials(study, baseline_scales)

    if args.apply_best:
        apply_best_to_config(best_scales, CONFIG_PATH)


if __name__ == "__main__":
    main()
