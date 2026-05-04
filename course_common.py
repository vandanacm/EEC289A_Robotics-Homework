#!/usr/bin/env python3
"""Shared utilities for the Go2 MuJoCo Playground course code.

This file keeps the training, evaluation, and benchmark scripts small.
Students should read this file once, but they do not need to modify it.
Most homework changes should live in:
- go2_pg_env/joystick.py
- go2_pg_env/randomize.py
- configs/course_config.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "course_config.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    """Convert nested config / JAX values into JSON-safe Python objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return str(value)
    return str(value)


def detect_gpu_name() -> str | None:
    """Return the first visible NVIDIA GPU name, or None on CPU runtimes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return names[0] if names else None


def set_runtime_env(*, force_cpu: bool = False) -> None:
    """Set runtime flags that make Colab runs more stable and reproducible."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        return

    extra_flag = "--xla_gpu_triton_gemm_any=True"
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if extra_flag not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " " + extra_flag).strip()


def lazy_import_stack() -> dict[str, Any]:
    """Import heavy dependencies only after runtime flags are set."""
    import jax
    import mediapy as media
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo_train_module
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    from go2_pg_env import register as register_go2_env

    register_go2_env()

    return {
        "jax": jax,
        "media": media,
        "ppo_networks": ppo_networks,
        "ppo_train": ppo_train_module.train if hasattr(ppo_train_module, "train") else ppo_train_module,
        "registry": registry,
        "wrapper": wrapper,
        "locomotion_params": locomotion_params,
    }


def get_ppo_config(locomotion_params: Any, env_name: str, impl: str) -> Any:
    """Reuse official Go1 PPO defaults as the starting point for the local Go2 env.

    The local Go2 environment matches the same task structure as Go1 joystick
    locomotion, so the official Go1 PPO recipe is a reasonable baseline.
    """
    reference_env_name = "Go1JoystickFlatTerrain" if env_name == "Go2JoystickFlatTerrain" else env_name
    try:
        return locomotion_params.brax_ppo_config(reference_env_name, impl)
    except TypeError:
        # Older Playground releases do not expose the `impl` argument.
        return locomotion_params.brax_ppo_config(reference_env_name)


def ensure_environment_available(registry: Any, env_name: str) -> None:
    try:
        registry.get_default_config(env_name)
    except Exception as exc:
        raise RuntimeError(
            f"Environment '{env_name}' is not registered in the current MuJoCo Playground install."
        ) from exc


def build_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    return {"impl": config["backend_impl"]}


def apply_stage_config(env_cfg: Any, ppo_cfg: Any, config: dict[str, Any], stage_name: str) -> None:
    """Apply course-specific stage settings to the environment and PPO config."""
    stage_cfg = config[stage_name]
    defaults = config["training_defaults"]
    runtime_overrides = config.get("runtime_overrides", {})

    if "command_range" in stage_cfg:
        env_cfg.command_config.min = list(stage_cfg["command_range"]["min"])
        env_cfg.command_config.max = list(stage_cfg["command_range"]["max"])
    else:
        # Backward compatibility for older configs that used symmetric amplitudes.
        amplitude = [float(value) for value in stage_cfg["command_amplitude"]]
        env_cfg.command_config.min = [-value for value in amplitude]
        env_cfg.command_config.max = amplitude
    if "command_keep_prob" in stage_cfg:
        env_cfg.command_config.b = list(stage_cfg["command_keep_prob"])
    env_cfg.command_config.stage_name = stage_name
    if "student_stage2_goal" in stage_cfg:
        env_cfg.command_config.student_stage2_goal_min = list(stage_cfg["student_stage2_goal"]["command_range"]["min"])
        env_cfg.command_config.student_stage2_goal_max = list(stage_cfg["student_stage2_goal"]["command_range"]["max"])
        env_cfg.command_config.student_stage2_goal_b = list(stage_cfg["student_stage2_goal"]["command_keep_prob"])
    for reward_name, reward_scale in stage_cfg.get("reward_scales", {}).items():
        if reward_name not in env_cfg.reward_config.scales:
            raise KeyError(f"Unknown reward scale '{reward_name}' in {stage_name}")
        env_cfg.reward_config.scales[reward_name] = float(reward_scale)

    stage_steps_key = f"{stage_name}_num_timesteps"
    ppo_cfg.num_timesteps = int(runtime_overrides.get(stage_steps_key, stage_cfg["num_timesteps"]))
    ppo_cfg.num_envs = int(runtime_overrides.get("num_envs", defaults["num_envs"]))
    ppo_cfg.num_eval_envs = int(runtime_overrides.get("num_eval_envs", defaults["num_eval_envs"]))
    ppo_cfg.num_evals = int(runtime_overrides.get("num_evals", defaults["num_evals"]))
    ppo_cfg.batch_size = int(runtime_overrides.get("batch_size", defaults["batch_size"]))

    ppo_cfg.network_factory.policy_hidden_layer_sizes = tuple(
        runtime_overrides.get("policy_hidden_layer_sizes", defaults["policy_hidden_layer_sizes"])
    )
    ppo_cfg.network_factory.value_hidden_layer_sizes = tuple(
        runtime_overrides.get("value_hidden_layer_sizes", defaults["value_hidden_layer_sizes"])
    )
    ppo_cfg.network_factory.policy_obs_key = config["actor_obs_key"]
    ppo_cfg.network_factory.value_obs_key = config["critic_obs_key"]

    if "episode_length" in runtime_overrides:
        env_cfg.episode_length = int(runtime_overrides["episode_length"])
        ppo_cfg.episode_length = int(runtime_overrides["episode_length"])
    if "num_minibatches" in runtime_overrides:
        ppo_cfg.num_minibatches = int(runtime_overrides["num_minibatches"])
    if "unroll_length" in runtime_overrides:
        ppo_cfg.unroll_length = int(runtime_overrides["unroll_length"])
    if "num_updates_per_batch" in runtime_overrides:
        ppo_cfg.num_updates_per_batch = int(runtime_overrides["num_updates_per_batch"])


def stage_sequence(stage_arg: str) -> list[str]:
    return ["stage_1", "stage_2"] if stage_arg == "both" else [stage_arg]


def resolve_latest_checkpoint_dir(checkpoint_root: Path) -> Path | None:
    """Return the numerically latest checkpoint directory."""
    if not checkpoint_root.exists():
        return None
    numeric_dirs = []
    for candidate in checkpoint_root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            numeric_dirs.append((int(candidate.name), candidate))
        except ValueError:
            continue
    if not numeric_dirs:
        return None
    numeric_dirs.sort(key=lambda item: item[0])
    return numeric_dirs[-1][1]


def _load_progress_records(stage_dir: Path) -> list[dict[str, Any]]:
    for name in ("progress.json", "progress_live.json"):
        path = stage_dir / name
        if path.exists():
            data = load_json(path)
            if isinstance(data, list):
                return data
    return []


def resolve_best_checkpoint_dir(stage_dir: Path) -> dict[str, Any] | None:
    """Pick the checkpoint with the highest eval reward.

    Brax saves checkpoints by step index. We match `progress.json` against the
    checkpoint directories and keep the checkpoint with the best eval reward.
    """
    checkpoint_root = stage_dir / "checkpoints"
    records = _load_progress_records(stage_dir)
    scored_candidates: list[tuple[float, int, Path]] = []

    for record in records:
        metrics = record.get("metrics", {})
        reward = metrics.get("eval/episode_reward")
        if reward is None:
            continue
        step = int(record["num_steps"])
        checkpoint_dir = checkpoint_root / f"{step:012d}"
        if checkpoint_dir.is_dir():
            scored_candidates.append((float(reward), step, checkpoint_dir))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: (item[0], item[1]))
    best_reward, best_step, best_dir = scored_candidates[-1]
    return {
        "selection_method": "best_eval_reward",
        "selected_step": best_step,
        "selected_eval_reward": best_reward,
        "selected_checkpoint_dir": best_dir,
    }


def export_selected_checkpoint(stage_dir: Path, export_dir: Path) -> dict[str, Any]:
    """Copy the best checkpoint if available, otherwise fall back to the latest."""
    checkpoint_root = stage_dir / "checkpoints"
    selected = resolve_best_checkpoint_dir(stage_dir)
    if selected is None:
        latest = resolve_latest_checkpoint_dir(checkpoint_root)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint directories found under {checkpoint_root}")
        selected = {
            "selection_method": "latest_fallback",
            "selected_step": int(latest.name),
            "selected_eval_reward": None,
            "selected_checkpoint_dir": latest,
        }

    export_dir = export_dir.resolve()
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(selected["selected_checkpoint_dir"], export_dir)

    manifest = {
        "selection_method": selected["selection_method"],
        "selected_step": selected["selected_step"],
        "selected_eval_reward": selected["selected_eval_reward"],
        "source_checkpoint_dir": str(selected["selected_checkpoint_dir"]),
        "exported_checkpoint_dir": str(export_dir),
        "exported_at_unix": time.time(),
    }
    save_json(export_dir / "manifest.json", manifest)
    return manifest
