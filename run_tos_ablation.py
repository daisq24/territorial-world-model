"""
Ablation runner: compare 4 territorial modes on the same scenes.

Modes:
    flat        — no territory structure (baseline)
    partition   — physical room_id only
    familiarity — online visit-based only
    dual        — both (the proposed method)

Usage:
    python run_tos_ablation.py --seeds 0 1 2 3 4 --max-steps 15 --render-mode text
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_tos_smoke import run_one     # noqa: E402


MODES = ("flat", "partition", "familiarity", "dual")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--data-dir", default="/root/autodl-tmp/tos/room_data/3-room")
    p.add_argument("--output-dir", default="/root/autodl-tmp/tos/territorial_results")
    p.add_argument("--render-mode", default="text", choices=["text", "vision"])
    args = p.parse_args()

    # results[mode][seed] = metrics dict
    results: Dict[str, Dict[int, dict]] = {m: {} for m in args.modes}

    for seed in args.seeds:
        for mode in args.modes:
            print(f"\n>>> seed={seed}, mode={mode}")
            try:
                out = run_one(
                    run_id=seed, mode=mode, max_steps=args.max_steps,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    render_mode=args.render_mode,
                    verbose=False,
                )
                results[mode][seed] = out["metrics"]
            except Exception:
                traceback.print_exc()
                results[mode][seed] = {"error": traceback.format_exc(limit=2)}

    # Aggregate
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    metrics_keys = ["stability", "self_tracking",
                    "object_coverage", "room_coverage",
                    "door_jump_ratio", "cross_room_jump_ratio",
                    "target_diversity", "steps_to_full_room_coverage"]
    header = f"{'mode':<14}" + "".join(f"{k:<22}" for k in metrics_keys)
    print(header)
    print("-" * len(header))
    summary = {}
    for mode in args.modes:
        row = f"{mode:<14}"
        summary[mode] = {}
        for k in metrics_keys:
            vals = [r[k] for r in results[mode].values()
                    if isinstance(r, dict) and k in r]
            if not vals:
                row += f"{'n/a':<22}"
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row += f"{mean:.3f} ± {std:.3f}      "
            summary[mode][k] = {"mean": mean, "std": std, "n": len(vals)}
        print(row)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = Path(args.output_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"per_run": results, "summary": summary,
                   "config": vars(args)}, f, indent=2, default=str)
    print(f"\n[ablation] saved → {summary_path}")

    # Quick verdict
    print("\n=== Quick verdict ===")
    if "dual" in summary and "flat" in summary:
        # higher-better metrics
        for k in ["stability", "self_tracking", "room_coverage",
                  "door_jump_ratio", "cross_room_jump_ratio", "target_diversity"]:
            d = summary["dual"].get(k, {}).get("mean", 0)
            f_ = summary["flat"].get(k, {}).get("mean", 0)
            arrow = "✅" if d > f_ else "⚠️"
            print(f"  {arrow}  {k}: dual={d:.3f} vs flat={f_:.3f} (Δ={d - f_:+.3f})")
        # lower-better metric
        d = summary["dual"].get("steps_to_full_room_coverage", {}).get("mean", -1)
        f_ = summary["flat"].get("steps_to_full_room_coverage", {}).get("mean", -1)
        # treat -1 (never covered) as worse than any positive value
        better = (f_ == -1 and d != -1) or (d != -1 and f_ != -1 and d < f_)
        arrow = "✅" if better else "⚠️"
        print(f"  {arrow}  steps_to_full_room_coverage (lower=better): "
              f"dual={d:.1f} vs flat={f_:.1f}")


if __name__ == "__main__":
    main()
