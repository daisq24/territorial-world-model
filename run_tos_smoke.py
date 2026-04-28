"""
Smoke driver: run TerritorialAgent on a single ToS scene end-to-end.

What it does:
  1. Builds SpatialGymConfig pointing at /root/autodl-tmp/tos/room_data/3-room
  2. Instantiates SpatialGym (vagen) and resets with the scene seed
  3. Loads meta_data.json for that run, builds TerritorialAgent
  4. Loops env.step(agent.act(...)) until done
  5. Computes probing metrics on the resulting trajectory
  6. Saves trajectory + metrics JSON

Run on the AutoDL server (where /root/autodl-tmp/tos exists):
    cd /root/autodl-tmp/tos
    python /root/autodl-tmp/tos/scripts_user/run_tos_smoke.py --run-id 0 --mode dual
    # or with the file scp'd alongside:
    python run_tos_smoke.py --run-id 0 --mode dual

If you hit AssertionError or AttributeError on env.reset/step, the env API
diverged from what we modelled — paste the trace and we patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


# Make the local agent + metrics importable when scp'd next to this script
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Make the ToS package importable when this script is anywhere on the box
TOS_ROOT_GUESSES = [
    "/root/autodl-tmp/tos",
    str(HERE.parent / "tos"),
    str(HERE),
]
for guess in TOS_ROOT_GUESSES:
    if Path(guess, "vagen").exists() and guess not in sys.path:
        sys.path.insert(0, guess)
        break

from tos_territorial_agent import TerritorialAgent, TerritorialCognitiveMap   # noqa: E402
from tos_probing_metrics import compute_all_metrics                            # noqa: E402


def build_env(data_dir: str, render_mode: str, max_steps: int, run_name: str,
              run_id: int, output_dir: str):
    """Construct a SpatialGym instance via the registry."""
    from vagen.env import REGISTERED_ENV

    if "spatial" not in REGISTERED_ENV:
        raise RuntimeError(f"'spatial' not in REGISTERED_ENV: {list(REGISTERED_ENV.keys())}")

    env_cls = REGISTERED_ENV["spatial"]["env_cls"]
    config_cls = REGISTERED_ENV["spatial"]["config_cls"]

    # SpatialGymConfig.reset() touches:
    #   self.kwargs['model_config']           (dict, HistoryManager logs only)
    #   self.kwargs['output_dir']             (str)
    #   self.kwargs.get('all_override')       (optional bool)
    #   self.kwargs.get('false_belief_override')
    #   self.kwargs.get('seed_start')/seed_end (used by generate_seeds)
    # We pass minimal stubs — none of these trigger real LLM inference.
    sample_dir = os.path.join(output_dir, f"smoke_run{run_id:02d}")
    os.makedirs(sample_dir, exist_ok=True)

    config = config_cls(
        name=run_name,
        render_mode=render_mode,        # 'text' is cheaper for smoke
        exp_type="active",
        max_exp_steps=max_steps,
        data_dir=data_dir,
        eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
        calculate_information_gain=False,
        # prompt_config is a top-level field in SpatialGymConfig.
        # HistoryManager / prompter read these keys directly.
        prompt_config={
            "enable_think": True,
            "enable_query": True,
            "enable_observe": True,
            "vision": (render_mode == "vision"),
        },
        kwargs={
            "model_config": {
                "model_name": "scripted-territorial",
                "model_type": "scripted",
            },
            "output_dir": sample_dir,
            "all_override": True,
            "false_belief_override": True,
            "seed_start": run_id,
            "seed_end": run_id,
        },
    )
    print(f"[env] config_id = {config.config_id()}")
    env = env_cls(config)
    return env


def run_one(run_id: int, mode: str, max_steps: int, data_dir: str,
            output_dir: str, render_mode: str, verbose: bool) -> Dict[str, Any]:

    run_dir = Path(data_dir) / f"run{run_id:02d}"
    meta_path = run_dir / "meta_data.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing scene meta: {meta_path}")
    with open(meta_path) as f:
        gt_meta = json.load(f)

    env = build_env(data_dir=data_dir, render_mode=render_mode,
                    max_steps=max_steps, run_name=f"smoke_run{run_id:02d}",
                    run_id=run_id, output_dir=output_dir)

    print(f"[smoke] reset(seed={run_id}) ...")
    obs, info = env.reset(seed=run_id)

    if verbose:
        if isinstance(obs, dict):
            print("[obs] keys:", list(obs.keys()))
            print("[obs.obs_str head]:", (obs.get("obs_str") or "")[:400])
            mm = obs.get("multi_modal_data") or {}
            for k, v in mm.items():
                print(f"[obs.multi_modal_data] {k}: {len(v)} items")
        else:
            print("[obs raw]:", str(obs)[:400])
        print("[info] keys:", list(info.keys()) if isinstance(info, dict) else "n/a")

    agent = TerritorialAgent(meta_path=meta_path, mode=mode,
                              max_steps=max_steps, wrap_with_think=True)
    print(f"[agent] cogmap initialized: "
          f"{len(agent.cogmap.objects)} objects, "
          f"{len(agent.cogmap.room_objects)} rooms "
          f"(rooms = {sorted(agent.cogmap.room_objects.keys())})")

    trajectory: Dict[str, Any] = {
        "run_id": run_id, "mode": mode, "render_mode": render_mode,
        "max_steps": max_steps,
        "actions": [], "rewards": [], "observations": [],
        "infos": [], "done": False, "final_reward": None,
    }

    done = False
    step = 0
    while not done and step < max_steps + 5:
        action_str = agent.act(obs, info if isinstance(info, dict) else {})
        try:
            obs, reward, done, info = env.step(action_str)
        except Exception:
            print(f"[smoke] env.step raised on step {step+1} with action: {action_str}")
            traceback.print_exc()
            raise

        ostr = obs.get("obs_str") if isinstance(obs, dict) else str(obs)
        # env's info dict doesn't expose visible_objects directly; pull from agent's
        # parser (which reads the obs_str) so local_global_consistency can compute.
        visible = (info or {}).get("visible_objects") or list(agent.policy.last_visible)
        trajectory["actions"].append(action_str)
        trajectory["rewards"].append(float(reward) if reward is not None else 0.0)
        trajectory["observations"].append({
            "step": step + 1,
            "obs_str_head": (ostr or "")[:600],
            "visible_objects": visible,
        })
        trajectory["infos"].append({
            "metrics": (info or {}).get("metrics", {}),
            "agent_room_id": (info or {}).get("agent_room_id"),
        })

        if verbose:
            print(f"[step {step+1}] action={action_str.splitlines()[-1][:120]}  "
                  f"r={reward}  done={done}  visible={visible[:5]}")
        step += 1

    # SpatialGym does not implement compute_reward; silently skip if missing.
    final_reward = None
    if hasattr(env, "compute_reward"):
        try:
            final_reward = env.compute_reward()
        except Exception:
            traceback.print_exc()

    trajectory["done"] = bool(done)
    trajectory["final_reward"] = final_reward
    trajectory["cogmap"] = agent.export_cogmap()

    metrics = compute_all_metrics(trajectory, gt_meta=gt_meta)
    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Persist
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"run{run_id:02d}_{mode}.json"
    with open(out_path, "w") as f:
        json.dump({"trajectory": trajectory, "metrics": metrics}, f,
                  indent=2, default=str)
    print(f"\n[smoke] saved → {out_path}")

    try:
        env.close()
    except Exception:
        pass

    return {"trajectory": trajectory, "metrics": metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--mode", choices=TerritorialCognitiveMap.VALID_MODES, default="dual")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--data-dir", default="/root/autodl-tmp/tos/room_data/3-room")
    p.add_argument("--output-dir", default="/root/autodl-tmp/tos/territorial_results")
    p.add_argument("--render-mode", default="text", choices=["text", "vision"])
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()
    run_one(**vars(args))


if __name__ == "__main__":
    main()
