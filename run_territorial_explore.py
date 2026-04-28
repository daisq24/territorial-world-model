"""
Run territorial agent through SpatialGym's exploration phase and save the
trajectory in the same on-disk format that spatial_run.py's --phase explore
produces.

Output files (saved to --output-dir):
  messages.json                  conversation history (system / user / assistant)
  exploration_turn_logs.json     per-turn structured log
  history_state.json             agent + room snapshot
  metrics.json                   exploration metrics (env auto-fills)

These are exactly the files that --phase eval / cogmap / aggregate consume.

Usage on the AutoDL box:
    cd /root/autodl-tmp/tos
    python run_territorial_explore.py \
        --run-id 0 --mode dual --max-steps 15 \
        --output-dir /tmp/territorial_seed0_dual

Then to run eval+cogmap+aggregate on this trajectory using Qwen as the answer
model, copy the four JSON files into spatial_run.py's expected location and
re-run with --phase eval --eval-override (see run_territorial_pipeline.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Bring tos_territorial_agent + vagen onto path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
TOS_ROOT_GUESSES = ["/root/autodl-tmp/tos", str(HERE.parent / "tos"), str(HERE)]
for guess in TOS_ROOT_GUESSES:
    if Path(guess, "vagen").exists() and guess not in sys.path:
        sys.path.insert(0, guess)
        break

from tos_territorial_agent import TerritorialAgent, TerritorialCognitiveMap   # noqa: E402


# env config requires at least one eval_task. The actual eval task list used
# by --phase eval comes from inference_config.yaml (eval_task_counts) and is
# applied per-run when spatial_run.py drives the eval phase. So during our
# explore-only run, a single placeholder is fine.
#
# Valid task_type values discovered from env_config _validate_eval_tasks:
#   dir, rot, pov, bwd_pov_text, e2a, fwd_loc, bwd_loc_text, fwd_fov,
#   bwd_nav_text, bwd_pov_vision, bwd_loc_vision, bwd_nav_vision,
#   rot_dual, bwd_nav_rev, false_belief, dir_anchor
DEFAULT_EVAL_TASKS = [
    {"task_type": "rot", "num": 1, "task_kwargs": {}},
]


def build_env(data_dir, render_mode, max_steps, run_id, output_dir, eval_tasks):
    """Construct SpatialGym, mirroring the smoke driver but writing to the
    target output_dir so the env's history_manager dumps the correct files."""
    from vagen.env import REGISTERED_ENV
    if "spatial" not in REGISTERED_ENV:
        raise RuntimeError("'spatial' env not registered in vagen.env")
    env_cls = REGISTERED_ENV["spatial"]["env_cls"]
    config_cls = REGISTERED_ENV["spatial"]["config_cls"]

    os.makedirs(output_dir, exist_ok=True)

    config = config_cls(
        name=f"territorial_run{run_id:02d}",
        render_mode=render_mode,
        exp_type="active",
        max_exp_steps=max_steps,
        data_dir=data_dir,
        eval_tasks=eval_tasks,
        calculate_information_gain=False,
        prompt_config={
            "enable_think": True,
            "enable_query": True,
            "enable_observe": True,
            "vision": (render_mode == "vision"),
        },
        kwargs={
            "model_config": {
                "model_name": "territorial-scripted",
                "model_type": "scripted",
            },
            "output_dir": output_dir,
            "all_override": True,
            "false_belief_override": True,
            "seed_start": run_id,
            "seed_end": run_id,
        },
    )
    print(f"[env] config_id = {config.config_id()}")
    return env_cls(config)


def run_explore(run_id, mode, max_steps, data_dir, output_dir, render_mode):
    run_dir = Path(data_dir) / f"run{run_id:02d}"
    meta_path = run_dir / "meta_data.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing scene meta: {meta_path}")

    env = build_env(data_dir=data_dir, render_mode=render_mode,
                    max_steps=max_steps, run_id=run_id,
                    output_dir=output_dir, eval_tasks=DEFAULT_EVAL_TASKS)

    print(f"[explore] reset(seed={run_id}) ...")
    obs, info = env.reset(seed=run_id)

    agent = TerritorialAgent(meta_path=meta_path, mode=mode,
                              max_steps=max_steps, wrap_with_think=True)
    print(f"[agent] mode={mode}; cogmap={len(agent.cogmap.objects)} objects, "
          f"{len(agent.cogmap.room_objects)} rooms")

    done = False
    step = 0
    actions = []
    while not done and step < max_steps + 5:
        action_str = agent.act(obs, info if isinstance(info, dict) else {})
        try:
            obs, reward, done, info = env.step(action_str)
        except Exception:
            print(f"[explore] env.step crashed on step {step+1}")
            traceback.print_exc()
            break
        step += 1
        actions.append(action_str)
        if step % 3 == 0 or done:
            tail = action_str.splitlines()[-1] if action_str else ""
            print(f"[step {step}] {tail[:120]}  r={reward}  done={done}")

    print(f"[explore] finished after {step} steps; done={done}")

    # Find what env actually saved
    saved = {}
    for name in ("messages.json", "exploration_turn_logs.json",
                 "history_state.json", "metrics.json"):
        # env may save under output_dir directly OR in a nested subdir
        for p in Path(output_dir).rglob(name):
            saved[name] = str(p)
            break
    print("[explore] saved files:")
    for k, v in saved.items():
        print(f"   {k}: {v}")
    if not saved:
        print(f"[explore] WARNING: no files found under {output_dir}. "
              f"Listing tree:")
        for p in Path(output_dir).rglob("*"):
            print(f"   {p}")

    # Also dump our cogmap to a separate file (for debugging / metrics)
    cogmap_path = Path(output_dir) / "territorial_cogmap.json"
    with open(cogmap_path, "w") as f:
        json.dump(agent.export_cogmap(), f, indent=2, default=str)
    print(f"[explore] cogmap saved → {cogmap_path}")

    try:
        env.close()
    except Exception:
        pass

    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--mode", default="dual",
                    choices=TerritorialCognitiveMap.VALID_MODES)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--data-dir", default="/root/autodl-tmp/tos/room_data/3-room")
    p.add_argument("--output-dir", required=True,
                    help="Where the env saves messages.json / *_logs.json")
    p.add_argument("--render-mode", default="text", choices=["text", "vision"])
    args = p.parse_args()

    run_explore(
        run_id=args.run_id,
        mode=args.mode,
        max_steps=args.max_steps,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
