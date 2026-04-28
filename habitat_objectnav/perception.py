from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PerceptionFeatures:
    goal_score: float
    goal_hint: float
    center_contrast: float
    center_saturation: float
    spatial_focus: float
    close_range_support: float
    boundary_strength: float
    forward_clearance: float
    brightness: float
    depth_stats: dict[str, float]
    signature_vec: np.ndarray
    signature_key: tuple[int, ...]


class TerritorialPerception:
    """Extract region-level evidence from RGB-D observations.

    This module turns raw observations into a compact territorial signature.
    The output is intentionally higher-level than a local controller state:
    it is meant to support memory, subgoal selection, and future multi-agent
    communication.
    """

    def extract(self, observation: dict[str, Any]) -> PerceptionFeatures:
        rgb = observation.get("rgb")
        depth = observation.get("depth")

        rgb_arr = np.asarray(rgb, dtype=np.float32) if rgb is not None else None
        depth_arr = np.asarray(depth, dtype=np.float32).squeeze() if depth is not None else None

        salience = self._estimate_goal_salience(rgb_arr, depth_arr, observation.get("objectgoal"))
        depth_stats = self._depth_layout(depth_arr)
        boundary_strength = self._estimate_boundary_strength(depth_stats)
        brightness = 0.0 if rgb_arr is None else float(rgb_arr.mean() / 255.0)
        signature_vec = self._build_signature_vector(
            brightness=brightness,
            depth_stats=depth_stats,
            goal_score=salience["goal_score"],
            boundary_strength=boundary_strength,
        )
        signature_key = tuple(int(x) for x in np.round(signature_vec * 10.0))

        return PerceptionFeatures(
            goal_score=salience["goal_score"],
            goal_hint=salience["goal_hint"],
            center_contrast=salience["center_contrast"],
            center_saturation=salience["center_saturation"],
            spatial_focus=salience["spatial_focus"],
            close_range_support=salience["close_range_support"],
            boundary_strength=boundary_strength,
            forward_clearance=depth_stats["mid"],
            brightness=brightness,
            depth_stats=depth_stats,
            signature_vec=signature_vec,
            signature_key=signature_key,
        )

    def _estimate_goal_salience(self, rgb, depth, objectgoal) -> dict[str, float]:
        if rgb is None:
            return {
                "goal_score": 0.0,
                "goal_hint": 0.0,
                "center_contrast": 0.0,
                "center_saturation": 0.0,
                "spatial_focus": 0.0,
                "close_range_support": 0.0,
            }

        h, w = rgb.shape[:2]
        center = rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        outer_mask = np.ones((h, w), dtype=bool)
        outer_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = False
        outer = rgb[outer_mask]

        center_mean = float(center.mean() / 255.0)
        contrast = float(np.abs(center.mean(axis=(0, 1)) - outer.mean(axis=0)).mean() / 255.0) if outer.size else 0.0
        saturation = float((center.max(axis=2) - center.min(axis=2)).mean() / 255.0)

        depth_focus = 0.0
        close_range_support = 0.0
        if depth is not None and depth.size:
            center_depth = depth[depth.shape[0] // 4 : 3 * depth.shape[0] // 4, depth.shape[1] // 4 : 3 * depth.shape[1] // 4]
            center_depth = np.nan_to_num(center_depth, nan=0.0, posinf=0.0, neginf=0.0)
            if center_depth.size:
                mean_center_depth = float(np.nanmean(center_depth))
                depth_focus = float(np.clip(1.5 - mean_center_depth, 0.0, 1.5) / 1.5)
                close_ratio = float(np.mean(center_depth < 1.4))
                mid_ratio = float(np.mean(center_depth < 2.0))
                close_range_support = 0.65 * close_ratio + 0.35 * mid_ratio

        goal_hint = 0.0 if objectgoal is None else float(int(np.asarray(objectgoal).reshape(-1)[0]) % 5) / 25.0
        score = 0.34 * center_mean + 0.24 * contrast + 0.14 * saturation + 0.18 * depth_focus + 0.08 * close_range_support + 0.02 * goal_hint
        return {
            "goal_score": float(np.clip(score, 0.0, 1.0)),
            "goal_hint": goal_hint,
            "center_contrast": contrast,
            "center_saturation": saturation,
            "spatial_focus": depth_focus,
            "close_range_support": float(np.clip(close_range_support, 0.0, 1.0)),
        }

    def _depth_layout(self, depth) -> dict[str, float]:
        if depth is None or depth.size == 0:
            return {
                "left": 0.0,
                "mid": 0.0,
                "right": 0.0,
                "wide_mid": 0.0,
                "left_edge": 0.0,
                "right_edge": 0.0,
                "variance": 0.0,
            }

        depth_arr = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        width = depth_arr.shape[1]
        left = depth_arr[:, : width // 3]
        mid = depth_arr[:, width // 3 : 2 * width // 3]
        right = depth_arr[:, 2 * width // 3 :]
        wide_mid = depth_arr[:, width // 4 : 3 * width // 4]
        left_edge = depth_arr[:, : width // 5]
        right_edge = depth_arr[:, 4 * width // 5 :]

        return {
            "left": float(np.mean(left)),
            "mid": float(np.mean(mid)),
            "right": float(np.mean(right)),
            "wide_mid": float(np.mean(wide_mid)),
            "left_edge": float(np.mean(left_edge)),
            "right_edge": float(np.mean(right_edge)),
            "variance": float(np.var(depth_arr)),
        }

    def _estimate_boundary_strength(self, depth_stats: dict[str, float]) -> float:
        side_gap = abs(depth_stats["left"] - depth_stats["right"])
        center_pinch = max(depth_stats["wide_mid"] - depth_stats["mid"], 0.0)
        return float(np.clip(0.55 * center_pinch + 0.45 * side_gap, 0.0, 2.5) / 2.5)

    def _build_signature_vector(
        self,
        *,
        brightness: float,
        depth_stats: dict[str, float],
        goal_score: float,
        boundary_strength: float,
    ) -> np.ndarray:
        return np.asarray(
            [
                np.clip(brightness, 0.0, 1.0),
                np.clip(goal_score, 0.0, 1.0),
                np.clip(depth_stats["left"] / 5.0, 0.0, 1.0),
                np.clip(depth_stats["mid"] / 5.0, 0.0, 1.0),
                np.clip(depth_stats["right"] / 5.0, 0.0, 1.0),
                np.clip(boundary_strength, 0.0, 1.0),
                np.clip(depth_stats["variance"] / 4.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
