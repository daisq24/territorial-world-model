"""
Explicit geometric world model for ToS scenes.

Two construction modes:
  • from_meta(meta_data.json)  — oracle: knows everything from the start
  • from_observations()        — incremental: built from Observe() feedback
                                  (TODO: hook into our TerritorialAgent)

Provides geometric primitives that the GeometricReasoner uses to answer
each ToS eval task type without language reasoning.

Coordinate convention (matches ToS):
  - 2-D floor plane (x, z)        ← we drop the y-axis (height) entirely
  - Ground-truth orientation in degrees: 0=north, 90=east, 180=south, 270=west
  - Agent.ori is a unit vector (cos_th, sin_th) where th is measured from +x.
    init_ori is always [0, 1] (= north) per ToS convention.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


CARDINAL_TO_VEC = {
    "north": np.array([0.0,  1.0]),
    "east":  np.array([1.0,  0.0]),
    "south": np.array([0.0, -1.0]),
    "west":  np.array([-1.0, 0.0]),
}
DEG_TO_CARD = {0: "north", 90: "east", 180: "south", 270: "west"}


@dataclass
class GeoObject:
    name: str
    pos: np.ndarray              # (x, z)  shape (2,)
    ori: np.ndarray              # (cos_th, sin_th) unit vector  shape (2,)
    ori_deg: int                 # original 0/90/180/270 (matches ToS rot.y)
    room_id: int
    has_orientation: bool
    object_id: int = 0
    label: str = ""

    def cardinal_facing(self) -> str:
        return DEG_TO_CARD.get(int(self.ori_deg) % 360, "unknown")


@dataclass
class GeoAgent:
    pos: np.ndarray              # (2,)
    ori: np.ndarray              # (2,) facing unit vec
    room_id: Optional[int]
    init_pos: np.ndarray         # (2,)
    init_ori: np.ndarray         # (2,) — should be [0, 1]

    def cardinal_facing(self) -> str:
        # cos_th = self.ori[0], sin_th = self.ori[1]
        # Round to cardinal
        ang = math.degrees(math.atan2(self.ori[1], self.ori[0]))  # angle from +x
        # Convert to compass: compass north = +y = atan2(1, 0) = 90°
        compass = (90 - ang) % 360
        compass = round(compass / 90.0) * 90 % 360
        return DEG_TO_CARD.get(compass, "north")


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def deg_to_vec(deg: float) -> np.ndarray:
    """Convert ToS rotation (deg, 0=north) to unit vector (x, z)."""
    th = math.radians(90 - (deg % 360))   # compass → math angle from +x
    return np.array([math.cos(th), math.sin(th)])


def vec_to_compass_deg(vec: np.ndarray) -> float:
    """Return compass-style degrees (0=north, 90=east) from (x, z) vector."""
    ang = math.degrees(math.atan2(vec[1], vec[0]))
    return (90 - ang) % 360


def project_to_cardinal(vec: np.ndarray) -> str:
    """Pick dominant cardinal direction of a (x, z) vector."""
    deg = vec_to_compass_deg(vec)
    # Snap to nearest of {0, 90, 180, 270}
    snapped = int(round(deg / 90.0) * 90) % 360
    return DEG_TO_CARD[snapped]


def relative_direction(
    from_pos: np.ndarray, to_pos: np.ndarray, frame_ori: np.ndarray
) -> str:
    """Cardinal direction of `to` from `from` in `frame` ego-frame."""
    delta = np.asarray(to_pos) - np.asarray(from_pos)
    north = np.asarray(frame_ori)
    east = np.array([north[1], -north[0]])
    n = np.dot(delta, north)
    e = np.dot(delta, east)
    if abs(n) > abs(e):
        return "north" if n > 0 else "south"
    return "east" if e > 0 else "west"


def relative_offset(
    from_pos: np.ndarray, to_pos: np.ndarray, frame_ori: np.ndarray
) -> Tuple[float, float]:
    """Return (forward, right) signed scalars in `frame`'s ego-frame."""
    delta = np.asarray(to_pos) - np.asarray(from_pos)
    north = np.asarray(frame_ori)
    east = np.array([north[1], -north[0]])
    return float(np.dot(delta, north)), float(np.dot(delta, east))


def rotate_vec(vec: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a (x, z) unit vector by `deg` (positive = clockwise in compass)."""
    th = math.radians(-deg)   # compass clockwise = math counterclockwise negated
    c, s = math.cos(th), math.sin(th)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def relative_orientation(
    obj_ori: np.ndarray, frame_ori: np.ndarray
) -> str:
    """Return cardinal compass of obj_ori relative to frame_ori."""
    # Project obj_ori into frame's ego frame, snap to cardinal.
    return project_to_cardinal(np.array([
        np.dot(obj_ori, frame_ori),                                # forward
        np.dot(obj_ori, np.array([frame_ori[1], -frame_ori[0]])),  # right
    ]))


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class GeometricWorldModel:
    """Floor-plan world model with explicit (pos, ori, room_id) per object."""

    def __init__(self):
        self.objects: Dict[str, GeoObject] = {}
        self.agent: Optional[GeoAgent] = None
        self.rooms: Dict[int, List[str]] = {}        # room_id → object names
        self.observed: set[str] = set()
        self.fov_deg: float = 90.0                    # ToS default
        self.room_size: Tuple[float, float] = (0, 0)

    # ---- construction ------------------------------------------------
    @classmethod
    def from_meta(cls, meta: Dict, agent_init_pos=(0.0, 0.0),
                  agent_init_ori=(0.0, 1.0)) -> "GeometricWorldModel":
        """Oracle mode: read every object from meta_data.json."""
        wm = cls()
        wm.room_size = tuple(meta.get("room_size", (0, 0)))
        for obj in meta.get("objects", []):
            attrs = obj.get("attributes", {})
            ori_deg = int(obj.get("rot", {}).get("y", 0)) % 360
            go = GeoObject(
                name=obj["name"],
                pos=np.array([float(obj["pos"]["x"]), float(obj["pos"]["z"])]),
                ori=deg_to_vec(ori_deg),
                ori_deg=ori_deg,
                room_id=int(attrs.get("room_id", 0)),
                has_orientation=bool(attrs.get("has_orientation", False)),
                object_id=int(obj.get("object_id", 0)),
                label=str(obj.get("label", "")),
            )
            wm.objects[go.name] = go
            wm.observed.add(go.name)
            wm.rooms.setdefault(go.room_id, []).append(go.name)

        # Agent: ToS initial pose is conventionally (0,0) facing north
        wm.agent = GeoAgent(
            pos=np.array(agent_init_pos, dtype=float),
            ori=np.array(agent_init_ori, dtype=float),
            room_id=None,
            init_pos=np.array(agent_init_pos, dtype=float),
            init_ori=np.array(agent_init_ori, dtype=float),
        )
        return wm

    # ---- observation update (for non-oracle mode) --------------------
    def mark_observed(self, name: str) -> None:
        if name in self.objects:
            self.observed.add(name)

    def set_agent(self, pos, ori, room_id=None) -> None:
        if self.agent is None:
            self.agent = GeoAgent(
                pos=np.array(pos, dtype=float),
                ori=np.array(ori, dtype=float),
                room_id=room_id,
                init_pos=np.array(pos, dtype=float),
                init_ori=np.array(ori, dtype=float),
            )
        else:
            self.agent.pos = np.array(pos, dtype=float)
            self.agent.ori = np.array(ori, dtype=float)
            self.agent.room_id = room_id

    # ---- queries -----------------------------------------------------
    def get_object(self, name: str) -> Optional[GeoObject]:
        return self.objects.get(name)

    def all_objects(self) -> List[GeoObject]:
        return list(self.objects.values())

    def visible_from(self, pos: np.ndarray, ori: np.ndarray,
                     fov_deg: Optional[float] = None,
                     same_room_only: bool = True,
                     agent_room: Optional[int] = None) -> List[GeoObject]:
        """Return objects within FOV of (pos, ori). Same-room rule mirrors ToS."""
        if fov_deg is None:
            fov_deg = self.fov_deg
        half = fov_deg / 2.0
        out = []
        for o in self.objects.values():
            if same_room_only and agent_room is not None and o.room_id != agent_room:
                # ToS allows seeing across doors; we approximate by allowing
                # objects in connected rooms when agent is at a door — caller
                # should handle that case.
                continue
            delta = o.pos - pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                continue
            unit = delta / dist
            # Angle between facing and obj direction
            cos_a = float(np.dot(ori, unit))
            cos_a = max(-1.0, min(1.0, cos_a))
            ang = math.degrees(math.acos(cos_a))
            if ang <= half:
                out.append(o)
        return out

    def cardinal_dir(self, from_name: str, to_name: str,
                     frame: str = "ego") -> str:
        """Cardinal direction of to relative to from. frame in {'ego', 'allo'}."""
        a = self.get_object(from_name) if from_name != "agent" else self.agent
        b = self.get_object(to_name) if to_name != "agent" else self.agent
        if a is None or b is None:
            return "unknown"
        if frame == "ego":
            ori = a.ori
        else:
            ori = np.array([0.0, 1.0])  # allocentric north
        return relative_direction(a.pos, b.pos, ori)

    def query_initial_frame(self, name: str) -> Tuple[Tuple[int, int], str]:
        """Returns ((dx, dy), facing) of `name` in agent's initial frame.
        Matches ToS Query() output format."""
        if self.agent is None:
            return ((0, 0), "north")
        obj = self.get_object(name)
        if obj is None:
            return ((0, 0), "north")
        forward, right = relative_offset(self.agent.init_pos, obj.pos,
                                          self.agent.init_ori)
        # ToS says: Query returns (x, y) with init_pos as origin, north as y+
        # init_ori = [0, 1] = +z is north; so forward → y, right → x
        coords = (int(round(right)), int(round(forward)))
        ori_card = obj.cardinal_facing() if obj.has_orientation else "no_orientation"
        return coords, ori_card


# ---------------------------------------------------------------------------
# CLI sanity
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    args = p.parse_args()
    meta = json.load(open(args.meta))
    wm = GeometricWorldModel.from_meta(meta)
    print(f"# Built world model")
    print(f"# Objects: {len(wm.objects)}")
    print(f"# Rooms:   {sorted(wm.rooms.keys())}")
    print(f"# Agent init pose: pos={wm.agent.pos}  ori={wm.agent.ori}")
    print()
    print("# Sample queries:")
    names = list(wm.objects.keys())[:3]
    for n in names:
        coords, facing = wm.query_initial_frame(n)
        obj = wm.get_object(n)
        ego_dir = wm.cardinal_dir("agent", n, frame="ego")
        print(f"  {n} (room {obj.room_id}): pos={tuple(obj.pos)}  Query={coords},{facing}  ego_dir={ego_dir}")
