"""
Territorial scripted agent for the Theory-of-Space (ToS) benchmark.

Implements a dual-source territorial cognitive map:
  1. Physical partition prior (room_id from meta_data.json)
  2. Online familiarity score (per-room visit counts)

Outputs ToS-format action strings that the SpatialGym env can parse:
    Actions: [JumpTo(table), Observe()]

Designed to be plugged into vagen.env.spatial.SpatialGym via:
    agent = TerritorialAgent(meta_path="room_data/3-room/run00/meta_data.json", mode="dual")
    obs, info = env.reset(seed=0)
    while not done:
        action_str = agent.act(obs, info)         # raw "Actions: [...]"
        wrapped    = f"<think>...</think><answer>{action_str}</answer>"  # if env expects it
        obs, reward, done, info = env.step(wrapped)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cognitive map
# ---------------------------------------------------------------------------

class TerritorialCognitiveMap:
    """
    Territorial representation with two information sources.

    Modes (ablation knobs):
        flat         — baseline; ignores room structure entirely
        partition    — uses ground-truth room_id only (structural prior)
        familiarity  — uses online visit counts only
        dual         — combines partition + familiarity (the proposed thing)
    """

    VALID_MODES = ("flat", "partition", "familiarity", "dual")

    def __init__(self, mode: str = "dual", familiarity_alpha: float = 0.6):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.familiarity_alpha = familiarity_alpha

        # Object DB (populated from meta_data.json)
        self.objects: Dict[str, Dict[str, Any]] = {}      # name → metadata
        self.observed_objects: set[str] = set()
        self.room_objects: Dict[int, List[str]] = defaultdict(list)

        # Online familiarity
        self.room_visits: Counter = Counter()             # room_id → visits
        self.current_room_id: Optional[int] = None
        self.position_history: List[Tuple[float, float]] = []

        # Belief snapshots (for stability / inertia metrics)
        self.belief_log: List[Dict[str, Any]] = []

    # ---- init ----------------------------------------------------------
    def init_from_meta(self, meta: Dict[str, Any]) -> None:
        """Seed the map with all objects + ground-truth room_ids."""
        for obj in meta.get("objects", []):
            name = obj["name"]
            self.objects[name] = {
                "object_id": obj["object_id"],
                "name": name,
                "label": obj.get("label", ""),
                "pos": (float(obj["pos"]["x"]), float(obj["pos"]["z"])),
                "room_id": obj["attributes"].get("room_id"),
                "orientation": obj["attributes"].get("orientation"),
                "has_orientation": obj["attributes"].get("has_orientation", False),
                "observed_count": 0,
            }
            rid = obj["attributes"].get("room_id")
            if rid is not None:
                self.room_objects[rid].append(name)

    # ---- updates -------------------------------------------------------
    def update_from_observation(self, obs_str: str, info: Dict[str, Any]) -> None:
        """Called every step to update familiarity + observed set."""
        # 1. Mark visible objects as observed
        visible = info.get("visible_objects") or self._parse_visible(obs_str)
        for n in visible:
            if n in self.objects:
                self.observed_objects.add(n)
                self.objects[n]["observed_count"] += 1

        # 2. Track current room (env may report; otherwise infer)
        rid = info.get("agent_room_id")
        if rid is None:
            rid = self._infer_room_from_visible(visible)
        if rid is not None:
            self.current_room_id = rid
            self.room_visits[rid] += 1

        # 3. Snapshot belief
        self.belief_log.append({
            "step": len(self.belief_log),
            "observed_objects": sorted(self.observed_objects),
            "current_room_id": self.current_room_id,
            "room_visits": dict(self.room_visits),
        })

    def _parse_visible(self, obs_str: str) -> List[str]:
        """Best-effort extract object names from Observe() bullets."""
        if not isinstance(obs_str, str):
            return []
        # Pattern: "• <name>: ..."
        return [m.strip() for m in re.findall(r"•\s*([a-z][a-z0-9_ -]*?):", obs_str)]

    def _infer_room_from_visible(self, visible: List[str]) -> Optional[int]:
        """Majority-vote room_id of visible objects we know."""
        rids = [self.objects[n]["room_id"] for n in visible
                if n in self.objects and self.objects[n]["room_id"] is not None]
        if not rids:
            return None
        return Counter(rids).most_common(1)[0][0]

    # ---- queries -------------------------------------------------------
    def room_familiarity(self, room_id: Optional[int]) -> float:
        if room_id is None:
            return 0.0
        total = sum(self.room_visits.values())
        if total == 0:
            return 0.0
        return self.room_visits.get(room_id, 0) / total

    def territorial_score(self, obj_name: str) -> float:
        """
        Score in [0, 1]. Higher = more "in our territory".
        Used to *deprioritise* (we want to explore unfamiliar territory).
        """
        if obj_name not in self.objects:
            return 0.0
        rid = self.objects[obj_name]["room_id"]

        if self.mode == "flat":
            return 0.0
        if self.mode == "partition":
            return 1.0 if rid is not None else 0.0
        if self.mode == "familiarity":
            return self.room_familiarity(rid)
        # dual
        partition = 1.0 if rid is not None else 0.0
        fam = self.room_familiarity(rid)
        return (1.0 - self.familiarity_alpha) * partition + self.familiarity_alpha * fam

    def unexplored_rooms(self) -> List[int]:
        all_rooms = set(self.room_objects.keys())
        visited = {rid for rid, c in self.room_visits.items() if c > 0}
        return sorted(all_rooms - visited)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class TerritorialPolicy:
    """
    Scripted exploration policy driven by the cognitive map.

    Each mode produces *materially different* target rankings:
        flat        — purely first-visible (no territorial info at all)
        partition   — uses ground-truth room_id; doors > unexplored > visited
        familiarity — uses online visit counts only; least-visited room wins
        dual        — combines both with door priority + familiarity decay

    Step strategy:
        step 1               → Observe()  (no JumpTo allowed at first step)
        step 2 .. n-2        → JumpTo(target), Observe()  with target chosen
                                by mode-specific priority
        step n-1             → Term()
    """

    def __init__(self, cogmap: TerritorialCognitiveMap, max_steps: int = 20):
        self.cogmap = cogmap
        self.max_steps = max_steps
        self.step_count = 0
        self.last_visible: List[str] = []
        self.jumped_to: List[str] = []        # ordered jump history (for analytics)
        self.door_jumps: int = 0
        self.cross_room_jumps: int = 0

    def decide(self, obs_str: str, info: Dict[str, Any]) -> str:
        self.step_count += 1
        visible = info.get("visible_objects") or self.cogmap._parse_visible(obs_str)
        self.last_visible = visible

        # Reserve last step for Term()
        if self.step_count >= self.max_steps:
            return "Actions: [Term()]"

        # First step: must Observe (no JumpTo)
        if self.step_count == 1:
            return "Actions: [Observe()]"

        target = self._pick_target(visible)
        if target is not None:
            # Track analytics
            if "door" in target.lower():
                self.door_jumps += 1
            prev_room = self.cogmap.current_room_id
            target_rid = self.cogmap.objects.get(target, {}).get("room_id")
            if (target_rid is not None and prev_room is not None
                    and target_rid != prev_room):
                self.cross_room_jumps += 1

            self.jumped_to.append(target)
            target_token = target.replace(" ", "_")
            return f"Actions: [JumpTo({target_token}), Observe()]"

        # Nothing useful visible → rotate
        return "Actions: [Rotate(90), Observe()]"

    # ---- target selection (mode-specific) -----------------------------
    def _pick_target(self, visible: List[str]) -> Optional[str]:
        """Mode-specific priority. Higher priority = pick first."""
        candidates: List[Tuple[float, int, str]] = []
        for idx, n in enumerate(visible):
            if n not in self.cogmap.objects:
                continue
            if n in self.jumped_to:
                continue
            priority = self._priority_for(n)
            # Negative idx so insertion order breaks ties stably (earlier wins)
            candidates.append((priority, -idx, n))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _priority_for(self, name: str) -> float:
        """Return a scalar priority — higher means more attractive to jump to."""
        obj = self.cogmap.objects[name]
        rid = obj.get("room_id")
        is_door = "door" in name.lower()
        already_visited_room = (rid is not None
                                and self.cogmap.room_visits.get(rid, 0) > 0)

        mode = self.cogmap.mode

        # ---- FLAT: no territorial info; everything equal ----
        if mode == "flat":
            return 0.0

        # ---- PARTITION: use ground-truth structure only ----
        # Doors are the structural seams of the world.
        # Unexplored rooms are the structural prior we want to honour.
        if mode == "partition":
            if is_door:
                return 3.0
            if not already_visited_room:
                return 2.0
            return 0.5

        # ---- FAMILIARITY: online visits only, no structural prior ----
        # We pretend we don't know room_id structure; use visit fractions.
        if mode == "familiarity":
            if is_door:
                # Doors aren't "rooms"; treat as neutral exploration target
                return 1.5
            fam = self.cogmap.room_familiarity(rid) if rid is not None else 0.5
            # Lower familiarity → higher priority
            return 1.0 + (1.0 - fam)        # range [1.0, 2.0]

        # ---- DUAL: structural prior + familiarity decay ----
        if mode == "dual":
            if is_door:
                # Top priority — doors give cross-room visibility
                return 3.5
            partition_bonus = 1.0 if not already_visited_room else 0.0
            fam = self.cogmap.room_familiarity(rid) if rid is not None else 0.5
            fam_bonus = 1.0 - fam
            # Weights: structural prior 60%, familiarity decay 40%
            return 1.0 + 0.6 * partition_bonus + 0.4 * fam_bonus

        return 0.0


# ---------------------------------------------------------------------------
# Top-level Agent
# ---------------------------------------------------------------------------

class TerritorialAgent:
    """Combines cogmap + policy. Stateful across one episode."""

    def __init__(self, meta_path: str | Path, mode: str = "dual",
                 max_steps: int = 20, familiarity_alpha: float = 0.6,
                 wrap_with_think: bool = True):
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"meta_data.json not found at {meta_path}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.mode = mode
        self.max_steps = max_steps
        self.familiarity_alpha = familiarity_alpha
        self.wrap_with_think = wrap_with_think

        self.cogmap = TerritorialCognitiveMap(mode=mode, familiarity_alpha=familiarity_alpha)
        self.cogmap.init_from_meta(self.meta)
        self.policy = TerritorialPolicy(self.cogmap, max_steps=max_steps)

    def reset_episode(self) -> None:
        self.cogmap = TerritorialCognitiveMap(mode=self.mode,
                                              familiarity_alpha=self.familiarity_alpha)
        self.cogmap.init_from_meta(self.meta)
        self.policy = TerritorialPolicy(self.cogmap, max_steps=self.max_steps)

    def act(self, obs: Dict[str, Any] | str, info: Dict[str, Any] | None = None) -> str:
        info = info or {}
        obs_str = obs.get("obs_str", "") if isinstance(obs, dict) else str(obs)
        self.cogmap.update_from_observation(obs_str, info)
        action_str = self.policy.decide(obs_str, info)
        if self.wrap_with_think:
            return (f"<think>territorial({self.mode}) step={self.policy.step_count} "
                    f"visible={len(self.policy.last_visible)} "
                    f"rooms_seen={len(self.cogmap.room_visits)}</think>"
                    f"<answer>{action_str}</answer>")
        return action_str

    def export_cogmap(self) -> Dict[str, Any]:
        """Snapshot for downstream metrics."""
        return {
            "mode": self.cogmap.mode,
            "objects": {
                n: {
                    "pos": obj["pos"],
                    "room_id": obj["room_id"],
                    "observed_count": obj["observed_count"],
                }
                for n, obj in self.cogmap.objects.items()
                if n in self.cogmap.observed_objects
            },
            "all_objects_known": {
                n: {"room_id": obj["room_id"], "pos": obj["pos"]}
                for n, obj in self.cogmap.objects.items()
            },
            "room_visits": dict(self.cogmap.room_visits),
            "rooms_known": sorted(self.cogmap.room_objects.keys()),
            "belief_log": self.cogmap.belief_log,
            "current_room_id": self.cogmap.current_room_id,
            # --- new policy analytics ---
            "jump_history": list(self.policy.jumped_to),
            "door_jumps": self.policy.door_jumps,
            "cross_room_jumps": self.policy.cross_room_jumps,
            "n_unique_targets": len(set(self.policy.jumped_to)),
        }


# ---------------------------------------------------------------------------
# CLI sanity check (runs without env)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="path to meta_data.json")
    p.add_argument("--mode", default="dual", choices=TerritorialCognitiveMap.VALID_MODES)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()

    agent = TerritorialAgent(meta_path=args.meta, mode=args.mode,
                              max_steps=args.steps, wrap_with_think=False)

    print(f"# Loaded scene: {len(agent.cogmap.objects)} objects, "
          f"{len(agent.cogmap.room_objects)} rooms")
    print(f"# Mode: {args.mode}")
    print(f"# Rooms: {sorted(agent.cogmap.room_objects.keys())}")
    for rid, names in sorted(agent.cogmap.room_objects.items()):
        print(f"#   room {rid}: {names}")

    # Simulate dummy observations to trace policy decisions
    fake_obs_template = "Visible:\n• {a}: 2m east\n• {b}: 3m north"
    names = list(agent.cogmap.objects.keys())
    for s in range(args.steps):
        a = names[s % len(names)]
        b = names[(s + 1) % len(names)]
        fake_obs = {"obs_str": fake_obs_template.format(a=a, b=b),
                    "visible_objects": [a, b]}
        action = agent.act(fake_obs, info={"visible_objects": [a, b]})
        print(f"step {s+1}: {action}")

    print("\n# Cogmap export keys:", list(agent.export_cogmap().keys()))
