"""Run the household ToM experiment.

3 conditions (fixer agent only — partner is scripted in HouseholdEnv):

    flat            FlatReverter
    territorial     TerritorialReverter (rooms split even/odd → B owns "odd")
    territorial_tom TerritorialToMReverter (user's proposed method)

Metrics per episode:
    final_at_home_rate    fraction of objects sitting on canonical home cell
    false_revert_count    times the fixer reverted an *intentionally* moved obj
                          (these are the "good" rearrangements partner made
                           that fixer should have accepted)
    correct_revert_count  times the fixer reverted an *accidentally* moved obj
                          (the actual win for "tidy up" behavior)
    wasted_steps          steps spent reverting+placing for false reverts

Headline number: TerritorialToMReverter should have:
    * comparable correct_revert_count to FlatReverter (still tidies up accidents)
    * substantially LOWER false_revert_count (defers to partner's intent)
    → Pareto-optimal vs Flat (tidies as well, respects partner more).

Usage:
    python run_household_experiment.py --episodes 10 --seeds 5 --steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agents import FIXER_REGISTRY  # noqa: E402
from household_env import HouseholdEnv  # noqa: E402


def run_episode(env: HouseholdEnv, fixer, max_steps: int) -> dict:
    obs = env.reset()
    fixer.reset(obs)

    # Track every revert action by the fixer to compute false / correct revert counts
    revert_log = []  # list of (obj_id, was_intentional)
    last_inventory = None

    for t in range(max_steps):
        a_B = fixer.select(obs)
        # Detect: just before fixer's action, was the obj they're about to
        # pick up an intentionally-displaced one? We log on PICKUP rather than
        # on PLACE because pickup is the moment of "decision to revert".
        if a_B == 5:  # pickup
            cur = obs['B']['pos']
            for obj in obs['global']['objects'].values():
                if obj['cell'] == list(cur) or tuple(obj['cell']) == cur:
                    if obj['holder'] is None:
                        oid = obj['obj_id']
                        was_intentional = obs['global']['intentional_displacements'].get(oid, False)
                        # Only log if the object was AWAY from canonical home
                        # (otherwise pickup-then-place at same cell is harmless)
                        home = obs['B']['canonical_homes'][oid]
                        if tuple(obj['cell']) != tuple(home):
                            revert_log.append({'obj_id': oid,
                                               'intentional': bool(was_intentional),
                                               'step': t})
                        break

        obs, info = env.step({'A': None, 'B': a_B})

    # Final metrics
    final_objects = obs['global']['objects']
    homes = obs['B']['canonical_homes']
    n = len(final_objects)
    at_home = sum(1 for o in final_objects.values()
                  if tuple(o['cell']) == tuple(homes[o['obj_id']]))

    # Build a map obj_id → intentional target (cell) for use in acceptable_rate
    intentional_targets = {}
    for d in env.disturbances:
        if d.intentional:
            intentional_targets[d.obj_id] = d.target_cell
    intentional_done = obs['global']['intentional_displacements']

    acceptable = 0
    for o in final_objects.values():
        oid = o['obj_id']
        cell = tuple(o['cell'])
        home = tuple(homes[oid])
        if cell == home:
            acceptable += 1
        elif (intentional_done.get(oid, False) and
              oid in intentional_targets and
              cell == tuple(intentional_targets[oid])):
            # at the intentional alt_good — partner's deliberate rearrangement
            acceptable += 1
    false_reverts = sum(1 for r in revert_log if r['intentional'])
    correct_reverts = sum(1 for r in revert_log if not r['intentional'])

    return {
        'final_at_home_count': int(at_home),
        'final_at_home_rate': float(at_home / max(1, n)),
        'acceptable_rate': float(acceptable / max(1, n)),
        'false_revert_count': int(false_reverts),
        'correct_revert_count': int(correct_reverts),
        'total_revert_attempts': int(len(revert_log)),
        'n_objects': int(n),
    }


def aggregate(per_episode: list[dict]) -> dict:
    keys = ['final_at_home_rate', 'acceptable_rate', 'false_revert_count',
            'correct_revert_count', 'total_revert_attempts']
    out = {}
    for k in keys:
        arr = np.array([ep[k] for ep in per_episode], dtype=float)
        out[k + '_mean'] = float(arr.mean())
        out[k + '_std'] = float(arr.std())
    out['n_episodes'] = len(per_episode)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--conditions',
                   default='flat,territorial,tom_room,tom_visit,territorial_tom')
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--out_prefix', default='',
                   help='Prefix for output files, e.g. "phase2_"')
    args = p.parse_args()

    out_dir = _HERE / args.outdir
    out_dir.mkdir(exist_ok=True, parents=True)
    cond_names = [c.strip() for c in args.conditions.split(',')]

    print(f'[hh] episodes={args.episodes} seeds={args.seeds} steps={args.steps}')
    print(f'[hh] conditions={cond_names}')

    # Per condition: list of episodes across all seeds.
    by_cond: dict[str, list[dict]] = {c: [] for c in cond_names}

    t0 = time.time()
    for seed_ix in range(args.seeds):
        for ep in range(args.episodes):
            env_seed = seed_ix * 10_000 + ep
            for cname in cond_names:
                FixerCls = FIXER_REGISTRY[cname]
                fixer = FixerCls(agent_id='B', seed=env_seed)
                env = HouseholdEnv(seed=env_seed, max_steps=args.steps,
                                   partner_schedule=[30, 80, 130])
                metrics = run_episode(env, fixer, args.steps)
                by_cond[cname].append(metrics)
                line = (f'  {cname:<18} home_rate={metrics["final_at_home_rate"]:.2f} '
                        f'false_rev={metrics["false_revert_count"]} '
                        f'correct_rev={metrics["correct_revert_count"]}')
                print(f'[hh] seed={seed_ix} ep={ep:>2d}{line}')

    elapsed = time.time() - t0

    print(f'\n[hh] done in {elapsed:.1f}s')
    print('\n=== summary ===')
    print(f'{"condition":<22} {"home_rate":>13} {"acceptable":>13} '
          f'{"false_rev":>13} {"correct_rev":>13}')
    agg_all = {}
    for c in cond_names:
        agg = aggregate(by_cond[c])
        agg_all[c] = agg
        print(
            f'{c:<22} '
            f'{agg["final_at_home_rate_mean"]:>5.2f}±{agg["final_at_home_rate_std"]:>4.2f}  '
            f'{agg["acceptable_rate_mean"]:>5.2f}±{agg["acceptable_rate_std"]:>4.2f}  '
            f'{agg["false_revert_count_mean"]:>5.2f}±{agg["false_revert_count_std"]:>4.2f}  '
            f'{agg["correct_revert_count_mean"]:>5.2f}±{agg["correct_revert_count_std"]:>4.2f}'
        )

    metrics_path = out_dir / f'{args.out_prefix}household_metrics.json'
    with metrics_path.open('w') as f:
        json.dump({'config': vars(args), 'aggregate': agg_all,
                   'per_episode': by_cond, 'wall_time_sec': elapsed},
                  f, indent=2)
    print(f'\n[hh] wrote {metrics_path}')

    # Bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        cs = cond_names
        # 5-condition palette: gray (flat) / blue (territorial) / amber (room-only)
        # / purple (visit-only) / red (full ToM, ours)
        palette = {
            'flat':            '#888888',
            'territorial':     '#3a86ff',
            'tom_room':        '#f2a900',
            'tom_visit':       '#9b5de5',
            'territorial_tom': '#e63946',
        }
        colors = [palette.get(c, '#888') for c in cs]
        for ax, key, ylabel, want in zip(
            axes,
            ['final_at_home_rate', 'false_revert_count', 'correct_revert_count'],
            ['final at-home rate', '#false reverts (lower better)',
             '#correct reverts (higher better)'],
            ['high', 'low', 'high'],
        ):
            means = [agg_all[c][key + '_mean'] for c in cs]
            stds = [agg_all[c][key + '_std'] for c in cs]
            ax.bar(cs, means, yerr=stds, capsize=5, color=colors[:len(cs)])
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
        plt.suptitle(f'Household ToM Experiment — {args.episodes * args.seeds} episodes')
        plt.tight_layout()
        comp_path = out_dir / f'{args.out_prefix}comparison.png'
        plt.savefig(comp_path, dpi=120, bbox_inches='tight')
        print(f'[hh] wrote {comp_path}')
    except Exception as e:
        print(f'(chart skipped: {e})')

    return 0


if __name__ == '__main__':
    sys.exit(main())
