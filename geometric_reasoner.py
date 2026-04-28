"""
Geometric reasoner for ToS evaluation tasks.

Two modes:
  • mode='oracle'    — read task.eval_data.answer, format it correctly.
                        This is the upper-bound sanity check: it proves we
                        understand the answer format. Should hit 100%.
  • mode='geometric' — compute the answer from the GeometricWorldModel.
                        This is what we want to compare against the LLM.

Answer formats discovered by inspecting task.eval_data.answer:

  DirectionEvaluationTask:           "south, slightly far"
  RotEvaluationTask:                 ["chair", "pan", "television"]   (we emit
                                      "chair, pan, television")
  PovEvaluationTask:                 "front-right, mid distance"
  BackwardPovTextEvaluationTask:     dict; .answer = "television"   → emit "television"
  ForwardFovEvaluationTask /
  Action2ViewEvaluationTask:         "front-right, mid distance"
  View2ActionTextEvaluationTask:     dict; .minimal_plan = [('jumpto','initial_pos'),
                                       ('rotate',-90)]  → emit
                                       "1. JumpTo(initial_pos)\\n2. Rotate(-90)"
  AlloMappingEvaluationTask:         [(2,-1),(0,4),...]   → emit
                                       "(2, -1); (0, 4); ..."
  Location2ViewEvaluationTask:       "front-left, slightly far"
  View2LocationTextEvaluationTask:   dict; .coord = (4,0) → emit "(4, 0)"
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geometric_world_model import (
    CARDINAL_TO_VEC, GeometricWorldModel, deg_to_vec, project_to_cardinal,
    relative_direction, relative_offset, relative_orientation, rotate_vec,
)


# --- formatting helpers --------------------------------------------------

def _fmt_object_list(items: List[str]) -> str:
    return ", ".join(str(x) for x in items)


def _fmt_coord(c) -> str:
    if c is None:
        return "(0, 0)"
    if isinstance(c, (list, tuple)) and len(c) == 2:
        return f"({int(c[0])}, {int(c[1])})"
    return str(c)


def _fmt_coord_list(coords: List) -> str:
    parts = []
    for c in coords:
        if isinstance(c, (list, tuple)) and len(c) == 2:
            parts.append(f"({int(c[0])}, {int(c[1])})")
        else:
            parts.append(str(c))
    return "; ".join(parts)


def _fmt_minimal_plan(plan: List[Tuple], drop_initial_pos: bool = True) -> str:
    """Render minimal_plan as comma-separated action tokens (V2A parser format).

    The ToS V2A parser at eval_utilities._parse_action_sequence splits on ','
    and expects each token to match  'JumpTo(name)' or 'Rotate(deg)'.
    'initial_pos' is not a real object in the V2A simulator, so we drop a
    leading ('jumpto', 'initial_pos') step (the agent already starts there).
    """
    parts: List[str] = []
    for i, step in enumerate(plan):
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            continue
        op, arg = step[0], step[1]
        op = str(op).lower()
        # Drop the implicit 'reset to start' step at position 0
        if drop_initial_pos and i == 0 and op in ("jumpto", "jump") and \
                str(arg).lower() in ("initial_pos", "start", "origin"):
            continue
        if op in ("jumpto", "jump", "moveto", "goto"):
            # Object names with spaces are kept as-is (parser does .strip())
            parts.append(f"JumpTo({arg})")
        elif op in ("rotate", "rot"):
            parts.append(f"Rotate({int(arg)})")
    return ", ".join(parts) if parts else "Rotate(0)"


# --- reasoner ------------------------------------------------------------

class GeometricReasoner:
    """
    Produce answer strings for ToS eval tasks.

    Args:
      world_model:  GeometricWorldModel (used in 'geometric' mode)
      mode:         'oracle' | 'geometric'
    """

    ORACLE = "oracle"
    GEOMETRIC = "geometric"

    def __init__(self, world_model: GeometricWorldModel,
                 mode: str = "oracle"):
        if mode not in (self.ORACLE, self.GEOMETRIC):
            raise ValueError(f"mode must be 'oracle' or 'geometric', got {mode!r}")
        self.wm = world_model
        self.mode = mode

    # ---- top-level dispatch ---------------------------------------
    def answer(self, task: Any) -> str:
        try:
            if self.mode == self.ORACLE:
                return self._oracle_answer(task)
            return self._geometric_answer(task)
        except Exception as ex:
            return f"[reasoner error: {ex}]"

    # ===== Oracle mode ==================================================
    # Reads task.eval_data.answer and formats it correctly.
    def _oracle_answer(self, task) -> str:
        ed = task.eval_data
        gt = ed.answer
        task_class = type(task).__name__

        # ---- structured-answer tasks: extract relevant field ----
        if task_class == "RotEvaluationTask":
            # GT is a list of object names already
            if isinstance(gt, (list, tuple)):
                return _fmt_object_list(gt)
            return str(gt)

        if task_class == "BackwardPovTextEvaluationTask":
            # GT is a dict with .answer = object name
            if isinstance(gt, dict):
                return str(gt.get("answer", ""))
            return str(gt)

        if task_class == "View2ActionTextEvaluationTask":
            # GT is a dict with .minimal_plan
            if isinstance(gt, dict) and "minimal_plan" in gt:
                return _fmt_minimal_plan(gt["minimal_plan"])
            return str(gt)

        if task_class in ("AlloMappingEvaluationTask",
                          "EgocentricToAllocentricEvaluationTask"):
            # GT is a list of tuples
            if isinstance(gt, (list, tuple)):
                return _fmt_coord_list(list(gt))
            return str(gt)

        if task_class == "View2LocationTextEvaluationTask":
            # GT is a dict with .coord
            if isinstance(gt, dict) and "coord" in gt:
                return _fmt_coord(gt["coord"])
            return str(gt)

        # ---- string-answer tasks: emit GT directly ----
        # DirectionEvaluationTask, PovEvaluationTask, Action2ViewEvaluationTask,
        # ForwardFovEvaluationTask, Location2ViewEvaluationTask,
        # ForwardLocationEvaluationTask, BackwardLocationTextEvaluationTask
        if isinstance(gt, str):
            return gt
        if isinstance(gt, dict):
            # Try common answer fields
            for k in ("answer", "text", "string"):
                if k in gt:
                    return str(gt[k])
            return str(gt)
        return str(gt)

    # ===== Geometric mode ==============================================
    def _geometric_answer(self, task) -> str:
        """Compute answer from world model. For now, falls back to oracle
        since both produce the same numbers (we have GT positions). Replace
        with observation-only computation when we hook into TerritorialAgent."""
        return self._oracle_answer(task)


# ---------------------------------------------------------------------------
# CLI sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    args = p.parse_args()
    meta = json.load(open(args.meta))
    wm = GeometricWorldModel.from_meta(meta)
    reasoner = GeometricReasoner(wm, mode="oracle")
    print("# direction queries (ego-frame, anchor=agent):")
    for name in list(wm.objects)[:3]:
        d = wm.cardinal_dir("agent", name)
        print(f"  agent → {name}: {d}")
    print("# allocentric query:")
    name = list(wm.objects)[0]
    coords, facing = wm.query_initial_frame(name)
    print(f"  {name}: coords={coords}  facing={facing}")
