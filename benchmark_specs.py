#!/usr/bin/env python3
"""Deterministic command scripts used for demo videos and public benchmark runs."""

from __future__ import annotations

from typing import Any

import numpy as np


PUBLIC_EPISODE_LABELS = (
    "forward_only",
    "lateral_only",
    "yaw_only",
    "combined",
)


def build_demo_segments(config: dict[str, Any]) -> list[list[float]]:
    """Return an easy-to-read command script for demo videos.

    The demo is intentionally choreographed from velocity commands:
    stand -> side shuffles -> pivots -> moonwalk -> figure-eight arcs -> stand.
    """
    demo_cfg = config["demo_rollout"]
    if "segments" in demo_cfg and demo_cfg["segments"]:
        return [[float(x) for x in segment] for segment in demo_cfg["segments"]]
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.20, 0.0],
        [0.0, -0.20, 0.0],
        [0.0, 0.0, 0.60],
        [0.0, 0.0, -0.60],
        [-0.25, 0.0, 0.0],
        [0.50, 0.15, 0.40],
        [0.50, -0.15, -0.40],
        [0.0, 0.0, 0.0],
    ]


def public_command_script(safe_ranges: dict[str, list[float]], episode_idx: int) -> list[list[float]]:
    """Return the deterministic command schedule for one public benchmark episode."""
    _, vx_max = map(float, safe_ranges["vx"])
    _, vy_max = map(float, safe_ranges["vy"])
    _, yaw_max = map(float, safe_ranges["yaw"])

    scripts = [
        [
            [0.0, 0.0, 0.0],
            [0.35 * vx_max, 0.0, 0.0],
            [0.60 * vx_max, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.40 * vy_max, 0.0],
            [0.0, 0.70 * vy_max, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.40 * yaw_max],
            [0.0, 0.0, 0.70 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.35 * vx_max, 0.35 * vy_max, 0.25 * yaw_max],
            [0.60 * vx_max, 0.50 * vy_max, 0.40 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
    ]
    return scripts[episode_idx % len(scripts)]


def public_command_episode_label(episode_idx: int) -> str:
    return PUBLIC_EPISODE_LABELS[episode_idx % len(PUBLIC_EPISODE_LABELS)]


def command_for_step(segments: list[list[float]], step_idx: int, total_steps: int) -> np.ndarray:
    """Convert a segment list into one command vector for the current step."""
    segment_length = max(1, total_steps // len(segments))
    segment_idx = min(len(segments) - 1, step_idx // segment_length)
    return np.asarray(segments[segment_idx], dtype=np.float32)


def seconds_to_steps(duration_seconds: float, ctrl_dt: float) -> int:
    return max(1, int(round(float(duration_seconds) / float(ctrl_dt))))
