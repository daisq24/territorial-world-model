"""
Probing metrics for ToS trajectories.

Reproduces the four metric families from the Theory-of-Space paper §5.1 / §5.3:
  • Stability         — does the agent's belief about a thing stay constant
                         when nothing relevant should have changed it?
  • Self-tracking     — does the agent always know which room it is in?
  • Local↔Global      — do per-step local descriptions agree with the
                         consolidated global cogmap?
  • Belief Inertia    — in false-belief settings, how slowly does the belief
                         update after evidence of change?

Each metric returns a number in [0, 1] (higher = better, except inertia).
Inputs are pure data — no env required.

Usage:
    from tos_probing_metrics import compute_all_metrics
    metrics = compute_all_metrics(trajectory, gt_meta=meta_data)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def stability_metric(belief_log: List[Dict[str, Any]],
                     gt_room_by_object: Optional[Dict[str, int]] = None) -> float:
    """
    Stability across the trajectory.

    Definition:
      For each object that has been observed at least twice, look at the
      sequence of inferred room_ids the agent assigns to it across snapshots.
      If `gt_room_by_object` is provided we compare against ground truth;
      otherwise we just check temporal consistency (same answer over time).

    Returns: mean fraction of "stable" objects in [0, 1].
    """
    if not belief_log:
        return 0.0

    # Build per-object room history from snapshots.
    obj_room_history: Dict[str, List[int]] = defaultdict(list)
    for snap in belief_log:
        rid = snap.get("current_room_id")
        for n in snap.get("observed_objects", []):
            if rid is not None:
                obj_room_history[n].append(rid)

    if not obj_room_history:
        return 0.0

    stabilities: List[float] = []
    for n, hist in obj_room_history.items():
        if len(hist) < 2:
            continue
        if gt_room_by_object is not None and n in gt_room_by_object:
            gt = gt_room_by_object[n]
            stabilities.append(sum(1 for r in hist if r == gt) / len(hist))
        else:
            mode_room = Counter(hist).most_common(1)[0][1]
            stabilities.append(mode_room / len(hist))

    if not stabilities:
        return 0.0
    return sum(stabilities) / len(stabilities)


# ---------------------------------------------------------------------------
# Self-tracking
# ---------------------------------------------------------------------------

def self_tracking_metric(belief_log: List[Dict[str, Any]],
                         gt_visited_rooms: Optional[List[int]] = None) -> float:
    """
    Fraction of snapshots where the agent has a non-None current_room_id.
    If gt_visited_rooms is given we also require it to be a plausible room.
    """
    if not belief_log:
        return 0.0

    n_correct = 0
    for snap in belief_log:
        rid = snap.get("current_room_id")
        if rid is None:
            continue
        if gt_visited_rooms is None or rid in gt_visited_rooms:
            n_correct += 1
    return n_correct / len(belief_log)


# ---------------------------------------------------------------------------
# Local ↔ Global
# ---------------------------------------------------------------------------

def local_global_consistency(cogmap: Dict[str, Any],
                              observations: List[Dict[str, Any]]) -> float:
    """
    For every object that appears in a local Observe, is it also represented
    in the global cogmap (i.e. consolidated into the agent's persistent view)?

    Returns: fraction of (step, visible_object) pairs that are present in the
    cogmap snapshot at end of episode.
    """
    if not observations:
        return 0.0
    known_objects = set(cogmap.get("objects", {}).keys())
    if not known_objects:
        return 0.0

    n_pairs = 0
    n_consistent = 0
    for obs in observations:
        for n in obs.get("visible_objects", []):
            n_pairs += 1
            if n in known_objects:
                n_consistent += 1
    return n_consistent / n_pairs if n_pairs else 0.0


# ---------------------------------------------------------------------------
# Belief Inertia (false-belief setting)
# ---------------------------------------------------------------------------

def belief_inertia(belief_log_before: List[Dict[str, Any]],
                   belief_log_after: List[Dict[str, Any]],
                   changed_objects: List[str]) -> float:
    """
    How much of the *old* belief survives after the false-belief twist.
    Lower is better (faster revision).

    For each changed object, check whether the agent's last snapshot still
    matches the pre-change room assignment.
    """
    if not changed_objects:
        return 0.0
    if not belief_log_before or not belief_log_after:
        return 1.0

    # Old belief = last snapshot before the twist
    old_room: Dict[str, Optional[int]] = {}
    for snap in reversed(belief_log_before):
        rid = snap.get("current_room_id")
        if rid is not None:
            for n in snap.get("observed_objects", []):
                old_room.setdefault(n, rid)
        if all(n in old_room for n in changed_objects):
            break

    # New belief = last snapshot after the twist
    last_after = belief_log_after[-1]
    new_room_for: Dict[str, Optional[int]] = {}
    rid = last_after.get("current_room_id")
    for n in last_after.get("observed_objects", []):
        new_room_for[n] = rid

    persistent = 0
    for n in changed_objects:
        old = old_room.get(n)
        new = new_room_for.get(n)
        if old is not None and new is not None and old == new:
            persistent += 1
    return persistent / len(changed_objects)


# ---------------------------------------------------------------------------
# Coverage / sanity metrics
# ---------------------------------------------------------------------------

def coverage_metrics(cogmap: Dict[str, Any], gt_meta: Dict[str, Any]) -> Dict[str, float]:
    total_objects = len(gt_meta.get("objects", []))
    seen_objects = len(cogmap.get("objects", {}))
    visited_rooms = len(cogmap.get("room_visits", {}))
    total_rooms = len({obj["attributes"].get("room_id")
                       for obj in gt_meta.get("objects", [])
                       if obj.get("attributes", {}).get("room_id") is not None})

    return {
        "object_coverage": seen_objects / total_objects if total_objects else 0.0,
        "room_coverage": visited_rooms / total_rooms if total_rooms else 0.0,
        "n_seen_objects": seen_objects,
        "n_visited_rooms": visited_rooms,
    }


def policy_metrics(cogmap: Dict[str, Any], gt_meta: Dict[str, Any]) -> Dict[str, float]:
    """Behavioural metrics — these are what should differ across modes."""
    jump_history = cogmap.get("jump_history", [])
    if not jump_history:
        return {"door_jump_ratio": 0.0, "cross_room_jump_ratio": 0.0,
                "target_diversity": 0.0, "steps_to_full_room_coverage": -1}

    door_jumps = cogmap.get("door_jumps", 0)
    cross_room = cogmap.get("cross_room_jumps", 0)
    n_jumps = len(jump_history)
    unique_targets = len(set(jump_history))
    n_total_objects = len(gt_meta.get("objects", []))

    # Steps to full room coverage (-1 if never reached)
    belief_log = cogmap.get("belief_log", [])
    total_rooms = len({obj["attributes"].get("room_id")
                       for obj in gt_meta.get("objects", [])
                       if obj.get("attributes", {}).get("room_id") is not None})
    steps_to_full = -1
    for snap in belief_log:
        visits = snap.get("room_visits", {})
        if len(visits) >= total_rooms:
            steps_to_full = snap.get("step", -1) + 1
            break

    return {
        "door_jump_ratio":        door_jumps / max(1, n_jumps),
        "cross_room_jump_ratio":  cross_room / max(1, n_jumps),
        "target_diversity":       unique_targets / max(1, n_total_objects),
        "steps_to_full_room_coverage": steps_to_full,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(trajectory: Dict[str, Any],
                        gt_meta: Optional[Dict[str, Any]] = None,
                        false_belief: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute all probing metrics on one trajectory.

    Args:
      trajectory:  must include 'cogmap' and 'observations' (see TerritorialAgent)
      gt_meta:     parsed meta_data.json — enables ground-truth-aware variants
      false_belief: optional dict with keys 'belief_log_before', 'belief_log_after',
                   'changed_objects' for inertia computation
    """
    cogmap = trajectory.get("cogmap", {})
    belief_log = cogmap.get("belief_log", [])
    observations = trajectory.get("observations", [])

    # ground-truth tables
    gt_room_by_object: Optional[Dict[str, int]] = None
    gt_visited_rooms: Optional[List[int]] = None
    if gt_meta:
        gt_room_by_object = {
            obj["name"]: obj["attributes"].get("room_id")
            for obj in gt_meta.get("objects", [])
            if obj.get("attributes", {}).get("room_id") is not None
        }
        gt_visited_rooms = sorted({rid for rid in gt_room_by_object.values()
                                   if rid is not None})

    metrics: Dict[str, float] = {
        "stability":                 stability_metric(belief_log, gt_room_by_object),
        "self_tracking":             self_tracking_metric(belief_log, gt_visited_rooms),
        "local_global_consistency":  local_global_consistency(cogmap, observations),
    }

    if false_belief:
        metrics["belief_inertia"] = belief_inertia(
            false_belief.get("belief_log_before", []),
            false_belief.get("belief_log_after", []),
            false_belief.get("changed_objects", []),
        )

    if gt_meta:
        metrics.update(coverage_metrics(cogmap, gt_meta))
        metrics.update(policy_metrics(cogmap, gt_meta))

    metrics["total_steps"] = len(belief_log)
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True, help="path to trajectory JSON")
    p.add_argument("--meta", required=False, help="path to meta_data.json")
    args = p.parse_args()

    with open(args.trajectory) as f:
        traj = json.load(f)
    gt_meta = None
    if args.meta:
        with open(args.meta) as f:
            gt_meta = json.load(f)

    print(json.dumps(compute_all_metrics(traj, gt_meta), indent=2))
