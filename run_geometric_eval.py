"""
Drive ToS evaluation tasks with the GeometricReasoner instead of an LLM.

Pipeline:
  1. Load a scene's meta_data.json
  2. Build GeometricWorldModel (oracle: knows everything)
  3. For each ToS eval task type, instantiate the task via EvalTaskType,
     ask reasoner for answer, score with the task's own evaluation
  4. Aggregate per-task and overall scores
  5. Print side-by-side vs Qwen baseline (if available)

This bypasses spatial_run.py's eval phase entirely — we use ToS's task
generators and scorers directly. The reasoner is the only thing that
changes.

Run:
    python run_geometric_eval.py --run-id 0 --num-per-task 3
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
TOS_GUESSES = ["/root/autodl-tmp/tos", str(HERE.parent / "tos")]
for guess in TOS_GUESSES:
    if Path(guess, "vagen").exists() and guess not in sys.path:
        sys.path.insert(0, guess)
        break

from geometric_world_model import GeometricWorldModel    # noqa: E402
from geometric_reasoner import GeometricReasoner          # noqa: E402

# ToS imports (must be available on the AutoDL box)
from vagen.env.spatial.Base.tos_base import Room, Agent              # noqa: E402
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType  # noqa: E402


# Match inference_config.yaml (Qwen baseline used these)
DEFAULT_TASKS = [
    "dir", "rot", "pov", "bwd_pov_text", "fwd_fov",
    "bwd_nav_text", "e2a", "fwd_loc", "bwd_loc_text",
]


def build_room_and_agent(meta: Dict) -> Tuple[Room, Agent]:
    """Build ToS Room and Agent objects from meta_data.json."""
    # ToS provides Room.from_dict / Agent.from_dict that consume our
    # exploration-saved structure. Easiest: piggyback on env.reset()'s
    # initial state. As a standalone fallback, build by hand:
    room_dict = {
        "name": "scene",
        "objects": [],
        "gates": [],
        "all_objects": [],
        "gates_by_room": {},
        "rooms_by_gate": {},
        "adjacent_rooms_by_room": {},
        "mask": None,
    }
    agent_dict = {
        "name": "agent",
        "pos": [0.0, 0.0],
        "ori": [0.0, 1.0],
        "label": "agent",
        "has_orientation": True,
        "room_id": 1,
        "init_pos": [0.0, 0.0],
        "init_ori": [0.0, 1.0],
    }
    # ToS may have stricter constructors. Try from_dict; fall back to
    # the env-driven path if it complains.
    try:
        room = Room.from_dict(room_dict)
    except Exception:
        room = None
    try:
        agent = Agent.from_dict(agent_dict)
    except Exception:
        agent = None
    return room, agent


def load_room_agent_from_history(history_state_path: Path) -> Tuple[Room, Agent]:
    """Cleaner path: load Room+Agent from a previous run's history_state.json."""
    state = json.load(open(history_state_path))
    room = Room.from_dict(state["room_dict"])
    agent = Agent.from_dict(state["agent_dict"])
    return room, agent


def run_one(meta_path: Path, history_state_path: Path | None,
            num_per_task: int, tasks: List[str], seed: int = 0,
            verbose: bool = True) -> Dict[str, Any]:
    """Run all tasks on one scene. Returns metrics dict."""
    meta = json.load(open(meta_path))
    wm = GeometricWorldModel.from_meta(meta)
    reasoner = GeometricReasoner(wm, mode="oracle")

    # Build Room/Agent. Prefer from history_state.json (matches env's view).
    if history_state_path and history_state_path.exists():
        room, agent = load_room_agent_from_history(history_state_path)
        if verbose:
            print(f"[setup] Loaded Room/Agent from {history_state_path}")
    else:
        room, agent = build_room_and_agent(meta)
        if verbose:
            print(f"[setup] Built Room/Agent from meta (may be incomplete)")

    if room is None or agent is None:
        raise RuntimeError("Could not construct Room/Agent. Need history_state.json.")

    rng = np.random.default_rng(seed)

    per_task = {}
    overall_total = 0
    overall_score = 0.0

    for task_short in tasks:
        try:
            task_class_name = EvalTaskType.from_short_name(task_short).class_name
        except Exception as e:
            print(f"[skip] {task_short}: cannot resolve class ({e})")
            continue

        task_total = 0
        task_score = 0.0
        for q_idx in range(num_per_task):
            try:
                # Each call creates a fresh task with a fresh question
                task = EvalTaskType.create_task(
                    task_short, np.random.default_rng(seed * 1000 + q_idx),
                    room.copy() if hasattr(room, "copy") else room,
                    agent.copy() if hasattr(agent, "copy") else agent,
                    {}, None,
                )
                # ToS task object has eval_data populated after generate_question
                _ = task.generate_question()
                # Reasoner produces answer
                answer_text = reasoner.answer(task)
                # task.evaluate(answer) returns (score, info_dict)
                raw = task.evaluate(answer_text)
                if isinstance(raw, tuple) and len(raw) >= 1:
                    score = raw[0]
                    info = raw[1] if len(raw) >= 2 else {}
                else:
                    score = raw
                    info = {}
                # Score might be bool or float
                if isinstance(score, bool):
                    score = 1.0 if score else 0.0
                else:
                    score = float(score)
                task_score += score
                task_total += 1
                if verbose:
                    extra = f"  info={info}" if info else ""
                    print(f"  [{task_short}/q{q_idx}] "
                          f"ans=\"{answer_text[:60]}\" score={score:.2f}{extra}")
            except Exception as e:
                if verbose:
                    print(f"  [{task_short}/q{q_idx}] ERROR {type(e).__name__}: {str(e)[:120]}")
                # Don't double-count zeros if the task itself can't be made
                continue

        avg = (task_score / task_total) if task_total else 0.0
        per_task[task_class_name] = {
            "n_total": task_total,
            "task_score": task_score,
            "avg_accuracy": avg,
        }
        overall_total += task_total
        overall_score += task_score
        print(f"[{task_short:20s} = {task_class_name:38s}]  "
              f"acc = {avg:.3f}  ({task_score:.2f}/{task_total})")

    overall_avg = (overall_score / overall_total) if overall_total else 0.0
    print()
    print(f"OVERALL: avg_accuracy = {overall_avg:.4f}  "
          f"(total {overall_score:.2f} / {overall_total})")

    return {
        "overall": {
            "n_total": overall_total,
            "total_score": overall_score,
            "avg_accuracy": overall_avg,
        },
        "per_task": per_task,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--data-dir", default="/root/autodl-tmp/tos/room_data/3-room")
    p.add_argument("--num-per-task", type=int, default=3)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--history-state", default=None,
                    help="Optional: history_state.json from a Qwen run "
                         "(reuses room/agent state). If not given, build from meta.")
    p.add_argument("--baseline-metrics", default=None,
                    help="Optional: Qwen's metrics.json to print side-by-side.")
    args = p.parse_args()

    meta_path = Path(args.data_dir) / f"run{args.run_id:02d}" / "meta_data.json"
    history_state = Path(args.history_state) if args.history_state else None

    print("=" * 70)
    print(f"Geometric reasoner eval: run{args.run_id:02d}")
    print(f"  meta:    {meta_path}")
    print(f"  hstate:  {history_state}")
    print(f"  tasks:   {args.tasks}")
    print("=" * 70)

    metrics = run_one(meta_path, history_state, args.num_per_task,
                      args.tasks, seed=args.run_id)

    print()
    print("=" * 70)
    print("Side-by-side vs baseline")
    print("=" * 70)
    if args.baseline_metrics and Path(args.baseline_metrics).exists():
        baseline = json.load(open(args.baseline_metrics))
        bev = baseline.get("evaluation", {})
        bov = bev.get("overall", {})
        print(f"  QWEN baseline:    avg_accuracy = {bov.get('avg_accuracy', 0):.4f}")
        print(f"  GEOMETRIC ours:   avg_accuracy = {metrics['overall']['avg_accuracy']:.4f}")
        print(f"  Δ = {metrics['overall']['avg_accuracy'] - bov.get('avg_accuracy', 0):+.4f}")
        print()
        print(f"  {'Task':40s}  {'Qwen':>10s}  {'Ours':>10s}  {'Δ':>10s}")
        bp = bev.get("per_task", {})
        op = metrics["per_task"]
        for k in sorted(set(list(bp.keys()) + list(op.keys()))):
            b_acc = bp.get(k, {}).get("avg_accuracy", 0)
            o_acc = op.get(k, {}).get("avg_accuracy", 0)
            print(f"  {k:40s}  {b_acc:10.3f}  {o_acc:10.3f}  {o_acc - b_acc:+10.3f}")

    # Save
    out_dir = Path("/root/autodl-tmp/tos/results/geometric")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run{args.run_id:02d}_geometric_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
