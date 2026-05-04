#!/usr/bin/env python3
"""Custom directional/magnitude evaluation for a trained Go2 policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to a PPO checkpoint directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts" / "custom_eval_bundle" / "custom_eval.json",
        help="Where to save custom evaluation results.",
    )
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--episode-length-steps", type=int, default=1500)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def _command_cases() -> list[tuple[str, np.ndarray]]:
    return [
        ("vx_pos_0.6", np.array([0.6, 0.0, 0.0], dtype=np.float32)),
        ("vx_pos_0.8", np.array([0.8, 0.0, 0.0], dtype=np.float32)),
        ("vx_pos_1.0", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("vx_neg_0.6", np.array([-0.6, 0.0, 0.0], dtype=np.float32)),
        ("vx_neg_0.8", np.array([-0.8, 0.0, 0.0], dtype=np.float32)),
        ("vy_pos_0.2", np.array([0.0, 0.2, 0.0], dtype=np.float32)),
        ("vy_pos_0.3", np.array([0.0, 0.3, 0.0], dtype=np.float32)),
        ("vy_pos_0.4", np.array([0.0, 0.4, 0.0], dtype=np.float32)),
        ("vy_neg_0.2", np.array([0.0, -0.2, 0.0], dtype=np.float32)),
        ("vy_neg_0.3", np.array([0.0, -0.3, 0.0], dtype=np.float32)),
        ("vy_neg_0.4", np.array([0.0, -0.4, 0.0], dtype=np.float32)),
        ("yaw_pos_0.6", np.array([0.0, 0.0, 0.6], dtype=np.float32)),
        ("yaw_pos_0.8", np.array([0.0, 0.0, 0.8], dtype=np.float32)),
        ("yaw_pos_1.0", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("yaw_neg_0.6", np.array([0.0, 0.0, -0.6], dtype=np.float32)),
        ("yaw_neg_0.8", np.array([0.0, 0.0, -0.8], dtype=np.float32)),
        ("yaw_neg_1.0", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
        ("combo_pos", np.array([0.8, 0.3, 0.8], dtype=np.float32)),
        ("combo_neg", np.array([-0.6, -0.3, -0.8], dtype=np.float32)),
    ]


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)
    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)
    env_cfg.episode_length = int(args.episode_length_steps)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)
    rng = jax.random.PRNGKey(int(config["seed"]) + 31415)

    case_results: list[dict[str, Any]] = []
    for case_name, command in _command_cases():
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        state = _force_command(state, command, jax)

        measured_xy: list[np.ndarray] = []
        measured_yaw: list[float] = []
        fell = False
        realized_steps = 0
        for _ in range(int(args.episode_length_steps)):
            state = _force_command(state, command, jax)
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = _force_command(state, command, jax)

            measured_xy.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
            measured_yaw.append(float(np.asarray(env.get_gyro(state.data)[2])))
            realized_steps += 1
            if bool(np.asarray(state.done)):
                fell = True
                break

        measured_xy_arr = np.asarray(measured_xy, dtype=np.float32) if measured_xy else np.zeros((0, 2), dtype=np.float32)
        measured_yaw_arr = np.asarray(measured_yaw, dtype=np.float32) if measured_yaw else np.zeros((0,), dtype=np.float32)
        command_xy = command[:2]
        command_yaw = float(command[2])

        vel_err = float(np.linalg.norm(measured_xy_arr - command_xy, axis=-1).mean()) if measured_xy_arr.size else 0.0
        yaw_err = float(np.abs(measured_yaw_arr - command_yaw).mean()) if measured_yaw_arr.size else 0.0

        case_results.append(
            {
                "case": case_name,
                "command": [float(x) for x in command],
                "velocity_tracking_error": vel_err,
                "yaw_tracking_error": yaw_err,
                "fell": fell,
                "realized_steps": realized_steps,
            }
        )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "episode_length_steps": int(args.episode_length_steps),
        "num_cases": len(case_results),
        "fall_rate": float(np.mean([item["fell"] for item in case_results])) if case_results else 0.0,
        "mean_velocity_tracking_error": float(np.mean([item["velocity_tracking_error"] for item in case_results]))
        if case_results
        else 0.0,
        "mean_yaw_tracking_error": float(np.mean([item["yaw_tracking_error"] for item in case_results])) if case_results else 0.0,
        "cases": case_results,
    }

    save_json(args.output_json.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
